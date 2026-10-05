import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def load_chunks(folder: str, size: int = 200, overlap: int = 40) -> list[dict]:
    chunks = []
    for p in sorted(Path(folder).glob("*.txt")):
        words = p.read_text(encoding="utf-8").split()
        for i in range(0, max(len(words), 1), size - overlap):
            chunks.append({"doc_id": p.stem, "text": " ".join(words[i:i + size])})
            if i + size >= len(words):
                break
    return chunks


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.encoder = SentenceTransformer(BI_ENCODER)
        emb = self.encoder.encode(texts, normalize_embeddings=True).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.reranker = CrossEncoder(CROSS_ENCODER)

    def _bm25(self, q: str, k: int) -> list[int]:
        return [int(i) for i in np.argsort(-self.bm25.get_scores(tokenize(q)))[:k]]

    def _dense(self, q: str, k: int) -> list[int]:
        v = self.encoder.encode([q], normalize_embeddings=True).astype("float32")
        _, idx = self.index.search(v, k)
        return [int(i) for i in idx[0] if i >= 0]

    def _hybrid(self, q: str, k: int, rrf_k: int = 60) -> list[int]:
        """Reciprocal Rank Fusion of BM25 and FAISS rankings."""
        scores: dict[int, float] = {}
        for ranking in (self._bm25(q, k * 3), self._dense(q, k * 3)):
            for rank, i in enumerate(ranking):
                scores[i] = scores.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)[:k]

    def _rerank(self, q: str, ids: list[int], k: int) -> list[int]:
        s = self.reranker.predict([(q, self.chunks[i]["text"]) for i in ids])
        return [ids[j] for j in np.argsort(-s)[:k]]

    def search(self, q: str, k: int = 5, mode: str = "hybrid_rerank", pool: int = 20) -> list[dict]:
        if mode == "bm25":
            ids = self._bm25(q, k)
        elif mode == "dense":
            ids = self._dense(q, k)
        elif mode == "hybrid":
            ids = self._hybrid(q, k)
        elif mode == "hybrid_rerank":
            ids = self._rerank(q, self._hybrid(q, pool), k)
        else:
            raise ValueError(f"mode must be one of {MODES}")
        return [self.chunks[i] for i in ids]
