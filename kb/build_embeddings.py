#!/usr/bin/env python3
import os
import sys
from kb.embeddings import build_chroma_from_kb, build_pinecone_from_kb

KB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kb_store.json'))
if not os.path.exists(KB_PATH):
    print('kb_store.json not found. Run ingest first: python3 scripts/ingest.py /path/to/docs')
    sys.exit(1)

provider = os.getenv('KB_PROVIDER','chroma').lower()
if provider == 'pinecone' or os.getenv('PINECONE_API_KEY'):
    print('Building Pinecone index from', KB_PATH)
    try:
        out = build_pinecone_from_kb(KB_PATH, index_name=os.getenv('PINECONE_INDEX','kb_index'))
        print('Built Pinecone index:', out)
    except Exception as e:
        print('Failed to build Pinecone index:', e)
        sys.exit(2)
else:
    print('Building Chroma DB from', KB_PATH)
    try:
        out = build_chroma_from_kb(KB_PATH, persist_directory=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','kb_chroma')))
        print('Built Chroma DB at', out)
    except Exception as e:
        print('Failed to build Chroma DB:', e)
        sys.exit(2)
