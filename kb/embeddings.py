"""
Embeddings and Vector Database helper.
Supports both Pinecone and Chroma for vector search.

Dependencies:
- pip install pinecone sentence-transformers chromadb
"""
import os
import json
import logging

logger = logging.getLogger(__name__)


def _load_kb(kb_path):
    with open(kb_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==================== CHROMA FUNCTIONS ====================

def build_chroma_from_kb(kb_path, persist_directory='kb_chroma'):
    """Build a local Chroma vector database from kb_store.json"""
    try:
        import chromadb
    except Exception as e:
        raise RuntimeError('chromadb is required: pip install chromadb') from e

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError('sentence-transformers required: pip install sentence-transformers') from e

    client = chromadb.PersistentClient(path=persist_directory)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    kb = _load_kb(kb_path)
    texts = [item.get('text', '') for item in kb]
    ids = [item.get('id') for item in kb]
    metadatas = [{'source': item.get('source', '')} for item in kb]

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Delete old collection if exists
    try:
        client.delete_collection(name='kb_collection')
    except Exception:
        pass
    
    collection = client.create_collection(name='kb_collection')
    collection.add(
        ids=ids, 
        embeddings=embeddings.tolist(), 
        metadatas=metadatas, 
        documents=texts
    )

    logger.info(f"Built Chroma DB with {len(texts)} documents at {persist_directory}")
    return persist_directory


def query_chroma(query, top_k=3, persist_directory='kb_chroma'):
    """Query the local Chroma database"""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    if not os.path.exists(persist_directory):
        return None

    try:
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_collection(name='kb_collection')
    except Exception as e:
        logger.warning(f"Chroma collection not found: {e}")
        return None

    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb = model.encode([query], convert_to_numpy=True)[0].tolist()

    res = collection.query(
        query_embeddings=[q_emb], 
        n_results=top_k, 
        include=['metadatas', 'documents', 'ids', 'distances']
    )
    
    results = []
    for i in range(len(res['ids'][0])):
        results.append({
            'id': res['ids'][0][i],
            'source': res['metadatas'][0][i].get('source', ''),
            'score': float(res['distances'][0][i]),
            'excerpt': res['documents'][0][i][:800]
        })
    return results


# ==================== PINECONE FUNCTIONS ====================

def build_pinecone_from_kb(kb_path, index_name='kb-index'):
    """
    Build a Pinecone index from kb_store.json
    Uses Pinecone SDK v3+ (new API)
    """
    try:
        from pinecone import Pinecone, ServerlessSpec
    except Exception as e:
        raise RuntimeError('pinecone required: pip install pinecone') from e

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError('sentence-transformers required: pip install sentence-transformers') from e

    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise RuntimeError('PINECONE_API_KEY environment variable not set')

    # Initialize Pinecone (v3+ API)
    pc = Pinecone(api_key=api_key)

    kb = _load_kb(kb_path)
    texts = [item.get('text', '') for item in kb]
    ids = [str(item.get('id')) for item in kb]  # IDs must be strings
    metadatas = [{
        'source': item.get('source', ''), 
        'excerpt': (item.get('text', '')[:800])
    } for item in kb]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    # Create index if it doesn't exist (Pinecone v3+ API)
    existing_indexes = [idx['name'] for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        logger.info(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
            )
        )
    else:
        logger.info(f"Using existing Pinecone index: {index_name}")

    # Get the index
    index = pc.Index(index_name)

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_emb = embeddings[i:i+batch_size].tolist()
        batch_meta = metadatas[i:i+batch_size]
        
        vectors = [
            {
                "id": batch_ids[j], 
                "values": batch_emb[j], 
                "metadata": batch_meta[j]
            } 
            for j in range(len(batch_ids))
        ]
        
        index.upsert(vectors=vectors)
        logger.info(f"Uploaded batch {i//batch_size + 1} ({len(batch_ids)} vectors)")

    logger.info(f"✅ Built Pinecone index '{index_name}' with {len(ids)} documents")
    return index_name


def query_pinecone(query, top_k=3, index_name='kb-index'):
    """
    Query Pinecone index using SDK v3+
    """
    try:
        from pinecone import Pinecone
    except Exception:
        logger.error("Pinecone SDK not available")
        return None

    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        logger.warning("PINECONE_API_KEY not set")
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        logger.error("sentence-transformers not available")
        return None

    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        # Generate query embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        q_emb = model.encode([query], convert_to_numpy=True)[0].tolist()

        # Query the index
        res = index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True
        )

        # Parse results
        results = []
        for match in res.get('matches', []):
            meta = match.get('metadata', {})
            results.append({
                'id': match.get('id'),
                'source': meta.get('source', ''),
                'score': float(match.get('score', 0.0)),
                'excerpt': meta.get('excerpt', '')
            })

        return results

    except Exception as e:
        logger.error(f"Pinecone query failed: {e}")
        return None
