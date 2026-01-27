from fastapi import APIRouter, HTTPException, Body
import os
import json
from . import embeddings
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize Gemini for RAG responses
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        rag_model = genai.GenerativeModel("gemini-2.0-flash-exp")
        logger.info("✅ Gemini RAG model initialized")
    else:
        rag_model = None
        logger.warning("⚠️ GEMINI_API_KEY not set, RAG disabled")
except Exception as e:
    rag_model = None
    logger.warning(f"⚠️ Could not initialize Gemini for RAG: {e}")


def generate_rag_answer(query: str, context_docs: list) -> dict:
    """
    Generate an answer using retrieved context and LLM (RAG pattern).
    Returns ONLY information found in the provided context.
    """
    if not rag_model:
        # Fallback: return raw excerpts if no LLM available
        return {
            "answer": "\n\n".join([doc.get("excerpt", "") for doc in context_docs]),
            "mode": "excerpts_only",
            "sources": [doc.get("source", "") for doc in context_docs]
        }
    
    # Build context from retrieved documents
    context_text = "\n\n".join([
        f"[Document {i+1}] Source: {doc.get('source', 'Unknown')}\n{doc.get('excerpt', '')}" 
        for i, doc in enumerate(context_docs)
    ])
    
    # Strict RAG prompt - prevents hallucinations
    prompt = f"""You are a helpful assistant that answers questions STRICTLY based on the provided context documents. 

CRITICAL RULES:
1. Answer ONLY using information from the documents below
2. If the answer is not in the documents, say "I don't have information about that in my knowledge base"
3. Do NOT use external knowledge or make assumptions
4. Be conversational and natural, but accurate
5. Cite which document you're using (e.g., "According to Document 1...")
6. If multiple documents have relevant info, synthesize them into a coherent answer

CONTEXT DOCUMENTS:
{context_text}

USER QUESTION: {query}

ANSWER (based only on the documents above):"""

    try:
        response = rag_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,  # Low temperature for factual responses
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 500,
            }
        )
        
        answer = response.text.strip()
        
        return {
            "answer": answer,
            "mode": "rag_generated",
            "sources": [doc.get("source", "") for doc in context_docs],
            "confidence": "high" if len(context_docs) > 0 else "low"
        }
        
    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
        # Fallback to excerpts
        return {
            "answer": "\n\n".join([doc.get("excerpt", "") for doc in context_docs]),
            "mode": "excerpts_fallback",
            "sources": [doc.get("source", "") for doc in context_docs]
        }


@router.post('/kb-query')
async def kb_query(payload: dict = Body(...)):
    """
    RAG-powered Knowledge Base query endpoint.
    Searches knowledge base and generates natural language answers.
    
    Expects JSON: {"query": "your question here", "top_k": 5, "use_rag": true}
    Returns: {"answer": "...", "sources": [...], "raw_results": [...]}
    """
    query = payload.get("query", "").strip()
    top_k = int(payload.get("top_k", 5))  # Increased default for better context
    use_rag = payload.get("use_rag", True)  # Enable RAG by default

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    logger.info(f"KB Query: '{query}' (top_k={top_k}, use_rag={use_rag})")

    # STEP 1: Try Pinecone first (if configured)
    results = None
    source_backend = None
    
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if pinecone_api_key:
        logger.info("Using Pinecone for KB query")
        try:
            results = embeddings.query_pinecone(
                query, 
                top_k=top_k, 
                index_name=os.getenv('PINECONE_INDEX', 'kb-index')
            )
            if results:
                source_backend = "pinecone"
                logger.info(f"Pinecone returned {len(results)} results")
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")

    # STEP 2: Try local Chroma DB (if Pinecone failed)
    if not results:
        logger.info("Attempting Chroma fallback")
        try:
            results = embeddings.query_chroma(
                query, 
                top_k=top_k, 
                persist_directory=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kb_chroma'))
            )
            if results:
                source_backend = "chroma"
                logger.info(f"Chroma returned {len(results)} results")
        except Exception as e:
            logger.warning(f"Chroma query failed: {e}")

    # STEP 3: Fallback to local kb_store.json with TF-IDF or substring search
    if not results:
        logger.info("Using local kb_store.json fallback")
        kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb_store.json")
        
        if not os.path.exists(kb_path):
            raise HTTPException(
                status_code=503, 
                detail="Knowledge base not available. Please contact support."
            )

        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)

        texts = [item.get("text", "") for item in kb]

        # Try TF-IDF + cosine similarity (requires scikit-learn)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer().fit(texts + [query])
            mat = vectorizer.transform(texts + [query])
            sims = cosine_similarity(mat[-1], mat[:-1])[0]
            idxs = sims.argsort()[::-1][:top_k]
            
            results = []
            for i in idxs:
                item = kb[i]
                excerpt = (item.get("text", "")[:800]).replace("\n", " ")
                results.append({
                    "id": item.get("id"), 
                    "source": item.get("source"), 
                    "score": float(sims[i]), 
                    "excerpt": excerpt
                })

            source_backend = "tfidf"
            logger.info(f"TF-IDF returned {len(results)} results")

        except ImportError:
            logger.warning("scikit-learn not available, using substring match")
            # Final fallback: simple substring match
            ql = query.lower()
            results = []
            for item in kb:
                txt = item.get("text", "")
                if ql in txt.lower():
                    results.append({
                        "id": item.get("id"), 
                        "source": item.get("source"), 
                        "score": 1.0, 
                        "excerpt": txt[:800].replace("\n", " ")
                    })
                    if len(results) >= top_k:
                        break
            
            source_backend = "substring"
            logger.info(f"Substring match returned {len(results)} results")

    # Filter out low-quality results (optional)
    if results and source_backend in ["tfidf", "pinecone", "chroma"]:
        # Keep only results with reasonable similarity scores
        results = [r for r in results if r.get("score", 0) > 0.1]
    
    if not results:
        return {
            "answer": "I don't have information about that in my knowledge base. Please try rephrasing your question or contact support.",
            "sources": [],
            "raw_results": [],
            "search_backend": source_backend,
            "mode": "no_results"
        }
    
    # STEP 4: Generate RAG answer if enabled
    if use_rag:
        rag_response = generate_rag_answer(query, results)
        return {
            "answer": rag_response["answer"],
            "sources": rag_response["sources"],
            "raw_results": results,
            "search_backend": source_backend,
            "mode": rag_response["mode"],
            "confidence": rag_response.get("confidence", "medium")
        }
    else:
        # Return raw results without RAG
        return {
            "results": results, 
            "search_backend": source_backend,
            "mode": "raw_only"
        }


@router.get('/kb-status')
async def kb_status():
    """Check which KB backend is available"""
    status = {
        "pinecone": {
            "available": bool(os.getenv('PINECONE_API_KEY')),
            "index": os.getenv('PINECONE_INDEX', 'kb-index'),
            "environment": os.getenv('PINECONE_ENVIRONMENT', 'not-set')
        },
        "chroma": {
            "available": False,
            "path": os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kb_chroma'))
        },
        "local_kb": {
            "available": False,
            "path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb_store.json")
        }
    }
    
    # Check Chroma
    chroma_path = status["chroma"]["path"]
    if os.path.exists(chroma_path):
        status["chroma"]["available"] = True
    
    # Check local KB
    kb_path = status["local_kb"]["path"]
    if os.path.exists(kb_path):
        status["local_kb"]["available"] = True
        with open(kb_path, "r") as f:
            kb_data = json.load(f)
            status["local_kb"]["documents"] = len(kb_data)
    
    return status
