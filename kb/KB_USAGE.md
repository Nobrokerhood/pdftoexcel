KB Ingestion & Usage

1) Ingest files into `kb_store.json`

Install recommended packages (optional but recommended):

```bash
python3 -m pip install pdfminer.six python-pptx pandas openai whisper transformers sentence-transformers
```

Run ingestion on a folder:

```bash
python3 scripts/ingest.py /path/to/your/docs
```

This writes `kb_store.json` in the project root.

2) (Optional) Build embedding DB (Chroma)

Install chromadb and sentence-transformers:

```bash
python3 -m pip install chromadb sentence-transformers
```

Build the local Chroma DB from `kb_store.json`:

```bash
python3 kb/build_embeddings.py
```

The `/kb-query` endpoint will use the Chroma DB if present for faster, more accurate retrieval.
If you prefer Pinecone, set `PINECONE_API_KEY` and `PINECONE_ENV` and run the same builder script; it will populate Pinecone instead of Chroma.

2) Query the KB via backend

The FastAPI app exposes `/kb-query`. Example request:

```bash
curl -X POST http://localhost:8000/kb-query -H "Content-Type: application/json" -d '{"query":"billing frequency","top_k":2}'
```

Response format:
```
{ "results": [ {"id":"...","source":"file.pdf","score":0.72,"excerpt":"..."} ] }
```

3) Connect frontend

`knowledge-bot.js` includes a commented placeholder to POST the user question to `/kb-query` and display results. Replace the local-search fallback with a network call to that endpoint for production.
