"""Vector stores: an in-memory NumPy store (tests / small demos) and ChromaDB (persistent)."""
import numpy as np

from .models import Chunk, RetrievedChunk


class VectorStore:
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, query_vec: np.ndarray, k: int) -> list[RetrievedChunk]: ...
    def count(self) -> int: ...
    def sources(self) -> list[str]: ...


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine similarity. Vectors are pre-normalised, so dot product == cosine."""

    def __init__(self):
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, np.ndarray] = {}

    def add(self, chunks, vectors):
        for c, v in zip(chunks, vectors):
            self.chunks[c.id], self.vectors[c.id] = c, v  # upsert by id

    def search(self, query_vec, k):
        if not self.chunks:
            return []
        ids = list(self.chunks)
        scores = np.stack([self.vectors[i] for i in ids]) @ query_vec
        top = np.argsort(-scores)[:k]
        return [RetrievedChunk(self.chunks[ids[i]], float(scores[i])) for i in top]

    def count(self):
        return len(self.chunks)

    def sources(self):
        return sorted({c.metadata.get("source", "?") for c in self.chunks.values()})


class ChromaVectorStore(VectorStore):
    def __init__(self, path: str, collection: str):
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks, vectors):
        if chunks:
            self.col.upsert(ids=[c.id for c in chunks], documents=[c.text for c in chunks],
                            metadatas=[c.metadata for c in chunks], embeddings=vectors.tolist())

    def search(self, query_vec, k):
        r = self.col.query(query_embeddings=[query_vec.tolist()], n_results=k)
        return [
            RetrievedChunk(Chunk(i, d, m), 1.0 - dist)  # chroma returns cosine distance
            for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]

    def count(self):
        return self.col.count()

    def sources(self):
        metas = self.col.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("source", "?") for m in metas})


def get_vector_store(settings) -> VectorStore:
    if settings.vector_store.lower() == "memory":
        return InMemoryVectorStore()
    return ChromaVectorStore(settings.storage_dir, settings.collection_name)
