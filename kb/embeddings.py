"""
Embeddings and Chroma helper.
- build_chroma_from_kb(kb_path, persist_directory)
- query_chroma(query, top_k)

Dependencies (recommended): chromadb, sentence-transformers
pip install chromadb sentence-transformers
"""
import os
import json


def _load_kb(kb_path):
    with open(kb_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_chroma_from_kb(kb_path, persist_directory='kb_chroma'):
    try:
        import chromadb
    except Exception as e:
        raise RuntimeError('chromadb is required to build the vector DB: pip install chromadb') from e

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError('sentence-transformers is required: pip install sentence-transformers') from e

    # Use the new Chroma client
    client = chromadb.PersistentClient(path=persist_directory)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    kb = _load_kb(kb_path)
    texts = [item.get('text','') for item in kb]
    ids = [item.get('id') for item in kb]
    metadatas = [{'source': item.get('source','')} for item in kb]

    # build embeddings in batches
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # create collection
    try:
        collection = client.delete_collection(name='kb_collection')
    except Exception:
        pass
    
    collection = client.create_collection(name='kb_collection')

    collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=texts)

    return persist_directory


def query_chroma(query, top_k=3, persist_directory='kb_chroma'):
    try:
        import chromadb
    except Exception:
        return None  # chroma not available

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection(name='kb_collection')
    except Exception:
        return None

    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb = model.encode([query], convert_to_numpy=True)[0].tolist()

    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=['metadatas','documents','ids','distances'])
    results = []
    for i in range(len(res['ids'][0])):
        results.append({
            'id': res['ids'][0][i],
            'source': res['metadatas'][0][i].get('source',''),
            'score': float(res['distances'][0][i]),
            'excerpt': res['documents'][0][i][:800]
        })
    return results


def build_pinecone_from_kb(kb_path, index_name='kb_index'):
    try:
        import pinecone
    except Exception as e:
        raise RuntimeError('pinecone-client is required to build Pinecone index: pip install "pinecone-client[grpc]"') from e

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError('sentence-transformers is required: pip install sentence-transformers') from e

    api_key = os.getenv('PINECONE_API_KEY')
    env = os.getenv('PINECONE_ENV')
    if not api_key:
        raise RuntimeError('PINECONE_API_KEY environment variable not set')

    pinecone.init(api_key=api_key, environment=env)

    kb = _load_kb(kb_path)
    texts = [item.get('text','') for item in kb]
    ids = [item.get('id') for item in kb]
    metadatas = [{'source': item.get('source',''), 'excerpt': (item.get('text','')[:800])} for item in kb]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    # create index if not exists
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension, metric='cosine')

    index = pinecone.Index(index_name)

    # upsert in batches of 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_emb = embeddings[i:i+batch_size].tolist()
        batch_meta = metadatas[i:i+batch_size]
        vectors = [(batch_ids[j], batch_emb[j], batch_meta[j]) for j in range(len(batch_ids))]
        index.upsert(vectors=vectors)

    return index_name


def query_pinecone(query, top_k=3, index_name='kb_index'):
    try:
        import pinecone
    except Exception:
        return None

    api_key = os.getenv('PINECONE_API_KEY')
    env = os.getenv('PINECONE_ENV')
    if not api_key:
        return None

    pinecone.init(api_key=api_key, environment=env)
    try:
        index = pinecone.Index(index_name)
    except Exception:
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb = model.encode([query], convert_to_numpy=True)[0].tolist()

    res = index.query(queries=[q_emb], top_k=top_k, include_metadata=True, include_values=False)
    results = []

    # parse returned matches
    matches = None
    if isinstance(res, dict) and 'results' in res:
        # newer pinecone structure
        all_matches = []
        for r in res['results']:
            all_matches.extend(r.get('matches', []))
        matches = all_matches
    elif isinstance(res, dict) and 'matches' in res:
        matches = res['matches']
    elif isinstance(res, list) and len(res) > 0 and 'matches' in res[0]:
        matches = res[0]['matches']
    else:
        matches = []

    for m in matches:
        meta = m.get('metadata', {})
        results.append({
            'id': m.get('id'),
            'source': meta.get('source',''),
            'score': float(m.get('score', 0.0) if 'score' in m else m.get('distance', 0.0)),
            'excerpt': meta.get('excerpt','')
        })

    return results
