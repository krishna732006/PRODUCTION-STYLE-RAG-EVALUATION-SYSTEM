import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .evaluation import evaluate, load_benchmark
from .generation import generate
from .retrieval import MODES, Retriever, load_chunks

DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")
BENCHMARK = os.getenv("BENCHMARK_PATH", "data/benchmark.jsonl")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever(load_chunks(DOCS_DIR))
    yield


app = FastAPI(title="RAG Evaluation System", lifespan=lifespan)


class QueryIn(BaseModel):
    question: str
    k: int = 5
    mode: str = "hybrid_rerank"


class EvalIn(BaseModel):
    k: int = 5
    modes: list[str] = list(MODES)
    faithfulness: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].chunks)}


@app.post("/query")
def query(body: QueryIn):
    hits = state["retriever"].search(body.question, body.k, body.mode)
    answer = generate(body.question, [h["text"] for h in hits])
    return {"answer": answer, "sources": hits}


@app.post("/evaluate")
def run_eval(body: EvalIn):
    return evaluate(state["retriever"], load_benchmark(BENCHMARK), body.k,
                    tuple(body.modes), body.faithfulness)
