"""RAG pipeline: ingest -> chunk -> embed -> store; query -> retrieve -> prompt -> answer + citations."""
import logging
import time
from pathlib import Path

from .chunking import chunk_pages
from .loaders import load_document
from .models import RetrievedChunk

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "Cite sources inline as [1], [2] matching the context numbers. "
    "If the answer is not in the context, say you don't know."
)


def build_prompt(question: str, hits: list[RetrievedChunk]) -> str:
    ctx = "\n".join(f"[{i}] ({_label(h)}) {h.chunk.text}" for i, h in enumerate(hits, 1))
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


def _label(h: RetrievedChunk) -> str:
    m = h.chunk.metadata
    return m.get("source", "?") + (f" p.{m['page']}" if "page" in m else "")


class RAGPipeline:
    def __init__(self, embedder, store, llm, chunk_size=800, overlap=120, top_k=4, min_score=0.0):
        self.embedder, self.store, self.llm = embedder, store, llm
        self.chunk_size, self.overlap, self.top_k, self.min_score = chunk_size, overlap, top_k, min_score

    def ingest_pages(self, pages) -> int:
        chunks = chunk_pages(pages, self.chunk_size, self.overlap)
        if chunks:
            self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        log.info("Ingested %d chunks", len(chunks))
        return len(chunks)

    def ingest_file(self, path: str | Path) -> int:
        return self.ingest_pages(load_document(path))

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        qv = self.embedder.embed([question])[0]
        return [h for h in self.store.search(qv, k or self.top_k) if h.score >= self.min_score]

    def ask(self, question: str, k: int | None = None) -> dict:
        t0 = time.perf_counter()
        hits = self.retrieve(question, k)
        if not hits:
            return {"answer": "I couldn't find anything relevant in the documents.", "citations": []}
        answer = self.llm.generate(SYSTEM_PROMPT, build_prompt(question, hits))
        log.info("Answered in %.2fs using %d chunks", time.perf_counter() - t0, len(hits))
        return {
            "answer": answer,
            "citations": [
                {"ref": i, "source": h.chunk.metadata.get("source"), "page": h.chunk.metadata.get("page"),
                 "score": round(h.score, 4), "text": h.chunk.text}
                for i, h in enumerate(hits, 1)
            ],
        }
