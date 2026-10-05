# Production-Style RAG Evaluation System

Hybrid retrieval (BM25 + FAISS, RRF fusion) -> cross-encoder reranking -> LLM answer, with Recall@K / MRR / faithfulness evaluation across retrieval configurations.

1. Put `.txt` documents in `data/docs/` (filename stem = doc id).
2. Fill `data/benchmark.jsonl` (50+ lines): `{"question": "...", "relevant": ["doc_id", ...]}`.
3. Run:
   - Eval CLI: `python -m app.evaluation -k 5 [--faithfulness]`
   - API: `uvicorn app.main:app` -> `/query`, `/evaluate`, `/health`
   - Docker: `docker build -t rag-eval . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... -v $PWD/data:/srv/data rag-eval`

Without `ANTHROPIC_API_KEY`, generation falls back to the top retrieved chunk. Set `LLM_MODEL` to change the model.
