#!/usr/bin/env python3
"""
Quick Pinecone KB Setup Script
Run this script to upload your knowledge base to Pinecone
"""

import os
import sys
import json
from pathlib import Path

print("=" * 60)
print("  NoBrokerHood Knowledge Bot - Pinecone Setup")
print("=" * 60)

# Step 1: Check environment variables
print("\n[1/5] Checking environment variables...")

required_vars = {
    'PINECONE_API_KEY': 'Your Pinecone API key',
    'PINECONE_INDEX': 'Your Pinecone index name (default: kb-index)',
    'PINECONE_ENVIRONMENT': 'Your Pinecone environment (e.g., us-east-1)'
}

missing_vars = []
for var, description in required_vars.items():
    if not os.getenv(var):
        missing_vars.append(f"  ❌ {var} - {description}")
    else:
        print(f"  ✅ {var} = {os.getenv(var)[:20]}...")

if missing_vars:
    print("\n⚠️  Missing environment variables:")
    for var in missing_vars:
        print(var)
    print("\nPlease set these before continuing:")
    print("  export PINECONE_API_KEY='your-key-here'")
    print("  export PINECONE_INDEX='kb-index'")
    print("  export PINECONE_ENVIRONMENT='us-east-1'")
    sys.exit(1)

# Step 2: Check dependencies
print("\n[2/5] Checking dependencies...")

try:
    from pinecone import Pinecone, ServerlessSpec
    print("  ✅ pinecone installed")
except ImportError:
    print("  ❌ pinecone not installed")
    print("  Run: pip install pinecone")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print("  ✅ sentence-transformers installed")
except ImportError:
    print("  ❌ sentence-transformers not installed")
    print("  Run: pip install sentence-transformers")
    sys.exit(1)

# Step 3: Load KB data
print("\n[3/5] Loading knowledge base...")

kb_path = Path(__file__).parent / 'kb_store.json'
if not kb_path.exists():
    print(f"  ❌ kb_store.json not found at {kb_path}")
    print("  Please create kb_store.json with your documents")
    sys.exit(1)

with open(kb_path, 'r', encoding='utf-8') as f:
    kb_data = json.load(f)

print(f"  ✅ Loaded {len(kb_data)} documents")

# Step 4: Generate embeddings
print("\n[4/5] Generating embeddings...")

model = SentenceTransformer('all-MiniLM-L6-v2')
print("  ✅ Model loaded: all-MiniLM-L6-v2")

texts = [doc['text'] for doc in kb_data]
ids = [str(doc['id']) for doc in kb_data]
metadatas = [{'source': doc['source'], 'excerpt': doc['text'][:800]} for doc in kb_data]

print(f"  📊 Embedding {len(texts)} documents...")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
print(f"  ✅ Generated {len(embeddings)} embeddings (dimension: {embeddings.shape[1]})")

# Step 5: Upload to Pinecone
print("\n[5/5] Uploading to Pinecone...")

api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('PINECONE_INDEX', 'kb-index')
environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')

pc = Pinecone(api_key=api_key)

# Check if index exists
existing_indexes = [idx['name'] for idx in pc.list_indexes()]

if index_name in existing_indexes:
    print(f"  ℹ️  Index '{index_name}' already exists")
    response = input("  Do you want to delete and recreate it? (y/n): ")
    if response.lower() == 'y':
        print(f"  🗑️  Deleting existing index...")
        pc.delete_index(index_name)
        existing_indexes.remove(index_name)

if index_name not in existing_indexes:
    print(f"  🔨 Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=embeddings.shape[1],
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=environment)
    )
    print("  ✅ Index created")

# Get index
index = pc.Index(index_name)

# Upload in batches
print("  📤 Uploading vectors...")
batch_size = 100
for i in range(0, len(ids), batch_size):
    batch_ids = ids[i:i+batch_size]
    batch_emb = embeddings[i:i+batch_size].tolist()
    batch_meta = metadatas[i:i+batch_size]
    
    vectors = [
        {"id": batch_ids[j], "values": batch_emb[j], "metadata": batch_meta[j]}
        for j in range(len(batch_ids))
    ]
    
    index.upsert(vectors=vectors)
    print(f"  ✅ Uploaded batch {i//batch_size + 1} ({len(batch_ids)} vectors)")

print("\n" + "=" * 60)
print("  🎉 SUCCESS! Knowledge Base uploaded to Pinecone")
print("=" * 60)

# Test query
print("\n[BONUS] Testing with a sample query...")
test_query = "How do I pay maintenance charges?"
q_emb = model.encode([test_query])[0].tolist()

results = index.query(vector=q_emb, top_k=1, include_metadata=True)

if results['matches']:
    match = results['matches'][0]
    print(f"\n  Query: '{test_query}'")
    print(f"  📄 Top Result: {match['metadata']['source']}")
    print(f"  📊 Score: {match['score']:.3f}")
    print(f"  📝 Excerpt: {match['metadata']['excerpt'][:200]}...")
else:
    print("  ⚠️  No results found")

print("\n✅ Setup complete! Your KB is ready to use.")
print("\nNext steps:")
print("  1. Update your backend environment variables on Render")
print("  2. Deploy your updated code")
print("  3. Test the /kb-query endpoint")
print("  4. Open ocr.html and try the Knowledge Bot\n")
