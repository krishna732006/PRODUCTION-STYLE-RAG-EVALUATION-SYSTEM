# Production-Style RAG Evaluation System

Hybrid retrieval (BM25 + FAISS, RRF fusion) -> cross-encoder reranking -> LLM answer, with Recall@K / MRR / faithfulness evaluation across retrieval configurations.

1. Put `.txt` documents in `data/docs/` (filename stem = doc id).
2. Fill `data/benchmark.jsonl` (50+ lines): `{"question": "...", "relevant": ["doc_id", ...]}`.
3. Run:
   - Eval CLI: `python -m app.evaluation -k 5 [--faithfulness]`
   - API: `uvicorn app.main:app` -> `/query`, `/evaluate`, `/health`
   - Docker: `docker build -t rag-eval . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... -v $PWD/data:/srv/data rag-eval`

Without `ANTHROPIC_API_KEY`, generation falls back to the top retrieved chunk. Set `LLM_MODEL` to change the model.


## Results

Evaluated on 55 questions over 30 Wikipedia ML articles (k=5, retrieval metrics).

| Setup | Recall@5 | MRR |
|---|---|---|
| BM25 | 0.909 | 0.888 |
| FAISS (dense) | 0.936 | 0.906 |
| Hybrid (BM25 + FAISS, RRF) | 0.936 | 0.911 |
| Hybrid + cross-encoder rerank | 0.936 | 0.906 |

Dense and hybrid retrieval slightly outperformed BM25. Reranking gave no gain on this benchmark, likely because the questions are relatively easy. Faithfulness was not run for these results.
