"""Basic retrieval evaluation: hit-rate@k, MRR and keyword recall over eval/qa.jsonl.

Usage: python -m scripts.evaluate --docs data/sample --qa eval/qa.jsonl --k 4
Runs fully offline with EMBEDDING_PROVIDER=hash (default here) or with the real model.
"""
import argparse
import json
import os
from pathlib import Path

from app.config import Settings
from app.core.embeddings import get_embedder
from app.core.llm import MockLLM
from app.core.loaders import SUPPORTED_EXTENSIONS
from app.core.pipeline import RAGPipeline
from app.core.vectorstore import InMemoryVectorStore


def evaluate(pipe: RAGPipeline, qa: list[dict], k: int) -> dict:
    hits, rr, kw = 0, 0.0, 0
    for item in qa:
        results = pipe.retrieve(item["question"], k)
        sources = [r.chunk.metadata.get("source") for r in results]
        if item["expected_source"] in sources:
            hits += 1
            rr += 1 / (sources.index(item["expected_source"]) + 1)
        joined = " ".join(r.chunk.text.lower() for r in results)
        kw += all(w.lower() in joined for w in item.get("expected_keywords", []))
    n = len(qa)
    return {"n": n, f"hit_rate@{k}": hits / n, "mrr": rr / n, "keyword_recall": kw / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/sample")
    ap.add_argument("--qa", default="eval/qa.jsonl")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    s = Settings(embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hash"))
    pipe = RAGPipeline(get_embedder(s), InMemoryVectorStore(), MockLLM(), s.chunk_size, s.chunk_overlap)
    for f in Path(a.docs).rglob("*"):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            pipe.ingest_file(f)
    qa = [json.loads(l) for l in Path(a.qa).read_text().splitlines() if l.strip()]
    print(json.dumps(evaluate(pipe, qa, a.k), indent=2))


if __name__ == "__main__":
    main()
