import json
import re
from functools import lru_cache

from sentence_transformers import CrossEncoder

from .generation import generate
from .retrieval import MODES, Retriever

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"  # labels: contradiction, entailment, neutral


def load_benchmark(path: str) -> list[dict]:
    """JSONL lines: {"question": str, "relevant": [doc_id, ...]}"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def recall_at_k(retrieved_docs: list[str], relevant: set[str]) -> float:
    return len(relevant & set(retrieved_docs)) / len(relevant)


def reciprocal_rank(retrieved_docs: list[str], relevant: set[str]) -> float:
    for rank, d in enumerate(retrieved_docs, 1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


@lru_cache(maxsize=1)
def _nli() -> CrossEncoder:
    return CrossEncoder(NLI_MODEL)


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences entailed by at least one retrieved chunk."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    if not sents or not contexts:
        return 0.0
    nli, supported = _nli(), 0
    for s in sents:
        preds = nli.predict([(c, s) for c in contexts]).argmax(axis=1)
        supported += int((preds == 1).any())
    return supported / len(sents)


def evaluate(retriever: Retriever, benchmark: list[dict], k: int = 5,
             modes: tuple[str, ...] = MODES, with_faithfulness: bool = False) -> dict:
    results = {}
    for mode in modes:
        rec, rr, faith = [], [], []
        for item in benchmark:
            hits = retriever.search(item["question"], k, mode)
            docs, relevant = [h["doc_id"] for h in hits], set(item["relevant"])
            rec.append(recall_at_k(docs, relevant))
            rr.append(reciprocal_rank(docs, relevant))
            if with_faithfulness:
                ctx = [h["text"] for h in hits]
                faith.append(faithfulness(generate(item["question"], ctx), ctx))
        n = len(benchmark)
        results[mode] = {f"recall@{k}": round(sum(rec) / n, 4), "mrr": round(sum(rr) / n, 4)}
        if with_faithfulness:
            results[mode]["faithfulness"] = round(sum(faith) / n, 4)
    return results


if __name__ == "__main__":
    import argparse

    from .retrieval import load_chunks

    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--benchmark", default="data/benchmark.jsonl")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--faithfulness", action="store_true")
    a = ap.parse_args()
    r = Retriever(load_chunks(a.docs))
    print(json.dumps(evaluate(r, load_benchmark(a.benchmark), a.k, with_faithfulness=a.faithfulness), indent=2))
