"""FastAPI service exposing ingest / query / health endpoints."""
import logging
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import get_settings
from .core.embeddings import get_embedder
from .core.llm import get_llm
from .core.loaders import SUPPORTED_EXTENSIONS
from .core.pipeline import RAGPipeline
from .core.vectorstore import get_vector_store
from .logging_config import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
log = logging.getLogger("api")
app = FastAPI(title="RAG Docs Chat", version="1.0.0")


@lru_cache
def get_pipeline() -> RAGPipeline:
    """Built lazily so the embedding model loads once, on first use."""
    return RAGPipeline(get_embedder(settings), get_vector_store(settings), get_llm(settings),
                       settings.chunk_size, settings.chunk_overlap, settings.top_k)


class Query(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/health")
def health():
    p = get_pipeline()
    return {"status": "ok", "chunks": p.store.count(), "llm": settings.llm_provider}


@app.get("/documents")
def documents():
    return {"sources": get_pipeline().store.sources()}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    results = {}
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {f.filename}")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / Path(f.filename).name  # keep original name for citations
            path.write_bytes(await f.read())
            results[f.filename] = get_pipeline().ingest_file(path)
    return {"ingested_chunks": results}


@app.post("/query")
def query(q: Query):
    try:
        return get_pipeline().ask(q.question, q.top_k)
    except Exception as e:  # surface LLM/provider errors cleanly
        log.exception("Query failed")
        raise HTTPException(502, f"Generation failed: {e}")
