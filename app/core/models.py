"""Shared data types."""
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A piece of a source document plus metadata used for citations."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)  # source, page, chunk_index


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity, higher = more relevant
