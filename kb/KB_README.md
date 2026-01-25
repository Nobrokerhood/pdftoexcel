Knowledge Bot — Quick Framework

Goal
- Provide a small, extendable knowledge-based chatbot UI available from `ocr.html`.
- Use local sample KB (for demo) or a backend vector search for production.

Files added
- `knowledge-bot.js` - Frontend chat UI and local KB demo logic.
- `kb-config.json` - Example KB with inline text documents.

Recommended Ingestion Workflow (video/pdf/ppt -> KB)
1. Extract text
   - Video: transcribe with Whisper/Whisper.cpp or Google Speech-to-Text. Example:
     ```bash
     # extract audio
     ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
     # transcribe with OpenAI/whisper local
     whisper audio.wav --model small > transcript.txt
     ```
   - PDF: use `pdfminer.six` or `pdftotext`.
     ```bash
     pdftotext document.pdf - | sed -e 's/\n/ /g' > doc.txt
     ```
   - PPTX: use `python-pptx` to extract text.

2. Chunk and embed
   - Split text into ~500-token chunks, create embeddings (OpenAI/other), store in a vector DB (Chroma, Pinecone, Weaviate).

3. Query flow
   - Frontend POST `/kb-query` with `{query}`
   - Backend: vector search -> retrieve top documents -> call LLM (Gemini) with doc context and user query -> return structured answer.

4. Wiring the current scaffold
   - Frontend: `knowledge-bot.js` can POST to `/kb-query` (uncomment code block in the file).
   - Backend: implement `/kb-query` endpoint that accepts `query` and returns `{answer: "..."}`.

Next steps to production
- Implement document ingestion pipeline (scripts suggested above).
- Use a vector DB and embeddings for accurate retrieval.
- Replace local search in `knowledge-bot.js` with a call to your backend `/kb-query`.
- Add authentication/authorization checks on the query endpoint.

Embedding-based retrieval (Chroma) — optional but recommended
---------------------------------------------------------
1. Install dependencies:

```bash
python3 -m pip install chromadb sentence-transformers
```

2. Build the Chroma DB from your `kb_store.json` (created by `scripts/ingest.py`):

```bash
python3 kb/build_embeddings.py
```

3. The `/kb-query` endpoint will now use Chroma + `sentence-transformers` for fast, accurate retrieval. If Chroma is not available it falls back to TF-IDF or substring matching.
 
Pinecone option
---------------
If you prefer Pinecone as your vector database, set the following environment variables and run the builder:

```bash
export PINECONE_API_KEY="<your-key>"
export PINECONE_ENV="<your-env>"   # e.g. us-west1-gcp
export PINECONE_INDEX="kb_index"  # optional
python3 kb/build_embeddings.py
```

The build script will detect `PINECONE_API_KEY` and create/populate the Pinecone index. The `/kb-query` route will prefer Pinecone if the env var is set.

Notes
- This scaffold is intentionally small and local-first so you can test quickly.
- When ready, I can scaffold a `/kb-query` FastAPI endpoint and an ingestion script.
