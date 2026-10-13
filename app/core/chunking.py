"""Recursive character chunking with overlap.

Strategy: try to split on the "largest" natural boundary first (paragraphs,
then lines, then sentences, then words) so chunks stay semantically coherent,
then merge pieces greedily up to `chunk_size`, carrying `overlap` characters
of context into the next chunk.
"""
import hashlib
import re

from .models import Chunk

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:  # last resort: hard cut
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    pieces: list[str] = []
    for part in text.split(sep):
        if part:
            pieces.extend(_split(part + sep, size, rest) if len(part) > size else [part + sep])
    return pieces


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    chunks, current = [], ""
    for piece in _split(text, chunk_size, SEPARATORS):
        if len(current) + len(piece) > chunk_size and current:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""  # carry context forward
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_pages(pages: list[tuple[str, dict]], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk every page and attach metadata + a deterministic id (enables idempotent re-ingest)."""
    out: list[Chunk] = []
    for text, meta in pages:
        for i, piece in enumerate(split_text(text, chunk_size, overlap)):
            key = f"{meta.get('source')}|{meta.get('page', 0)}|{i}|{piece}"
            cid = hashlib.sha1(key.encode()).hexdigest()[:16]
            out.append(Chunk(id=cid, text=piece, metadata={**meta, "chunk_index": i}))
    return out
