from pinecone import Pinecone
import json
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX", "kb")

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# Load KB data
kb_path = os.path.join(os.path.dirname(__file__), 'kb_store.json')
with open(kb_path, "r", encoding="utf-8") as f:
    kb_data = json.load(f)

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare data for upload
vectors = []
for item in kb_data:
    text = item.get("text", "")
    if text:
        embedding = model.encode(text)
        vectors.append((item["id"], embedding, {"source": item.get("source", "")}))

# Upload data to Pinecone
index.upsert(vectors)
print("Data uploaded successfully!")
