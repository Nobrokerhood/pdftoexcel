from fastapi import APIRouter, HTTPException, Body
import os
import json
from . import embeddings

router = APIRouter()

@router.post('/kb-query')
async def kb_query(payload: dict = Body(...)):
    """Simple KB query endpoint. Expects JSON {"query": "...", "top_k": 1}.
    Looks for a local `kb_store.json` produced by ingestion and returns best matches.
    """
    query = payload.get("query", "").strip()
    top_k = int(payload.get("top_k", 1))

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb_store.json")
    if not os.path.exists(kb_path):
        raise HTTPException(status_code=400, detail="Knowledge base not found. Run the ingestion script first.")

    # Attempt embedding-based retrieval: Pinecone (if configured) -> Chroma
    try:
        # Pinecone preferred when API key present
        if os.getenv('PINECONE_API_KEY'):
            pine_results = embeddings.query_pinecone(query, top_k=top_k, index_name=os.getenv('PINECONE_INDEX','kb_index'))
            if pine_results:
                return {"results": pine_results}

        chroma_results = embeddings.query_chroma(query, top_k=top_k, persist_directory=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kb_chroma')))
        if chroma_results:
            return {"results": chroma_results}
    except Exception:
        # continue to fallback methods
        pass

    # Load raw KB and fallback to TF-IDF / substring
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    texts = [item.get("text", "") for item in kb]

    # Try TF-IDF + cosine similarity if sklearn is available
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
            excerpt = (item.get("text","" )[:600]).replace("\n"," ")
            results.append({"id": item.get("id"), "source": item.get("source"), "score": float(sims[i]), "excerpt": excerpt})

        return {"results": results}

    except Exception:
        # Fallback: simple substring match
        ql = query.lower()
        results = []
        for item in kb:
            txt = item.get("text","")
            if ql in txt.lower():
                results.append({"id": item.get("id"), "source": item.get("source"), "score": 1.0, "excerpt": txt[:600].replace("\n"," ")})
                if len(results) >= top_k:
                    break
        return {"results": results}
