"""Embedding providers behind one tiny interface: embed(list[str]) -> np.ndarray."""
import hashlib
import logging
import re

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    def embed(self, texts: list[str]) -> np.ndarray:  # shape (n, dim), L2-normalised
        raise NotImplementedError


def _normalise(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


class HashEmbedder(Embedder):
    """Offline bag-of-words hashing embedder. No downloads, deterministic.
    Used in tests/CI and as a zero-dependency demo mode — not for real quality."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _normalise(out)


class SentenceTransformerEmbedder(Embedder):
    """Local default: runs on CPU, no API key needed."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI
        self.client, self.model = OpenAI(api_key=api_key), model

    def embed(self, texts):
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return _normalise(np.array([d.embedding for d in resp.data], dtype=np.float32))


def get_embedder(settings) -> Embedder:
    p = settings.embedding_provider.lower()
    if p == "hash":
        return HashEmbedder()
    if p == "openai":
        return OpenAIEmbedder(settings.openai_embedding_model, settings.openai_api_key)
    return SentenceTransformerEmbedder(settings.embedding_model)
