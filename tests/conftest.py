import os

# Force fully offline, key-free settings before the app is imported.
os.environ.update(EMBEDDING_PROVIDER="hash", VECTOR_STORE="memory", LLM_PROVIDER="mock")

import pytest

from app.core.embeddings import HashEmbedder
from app.core.llm import MockLLM
from app.core.pipeline import RAGPipeline
from app.core.vectorstore import InMemoryVectorStore


@pytest.fixture
def pipeline():
    p = RAGPipeline(HashEmbedder(), InMemoryVectorStore(), MockLLM(), chunk_size=200, overlap=30, top_k=3)
    p.ingest_pages([
        ("Python is a programming language created by Guido van Rossum.", {"source": "python.txt"}),
        ("The Eiffel Tower is located in Paris, France.", {"source": "paris.md"}),
        ("Photosynthesis converts sunlight into chemical energy in plants.", {"source": "bio.pdf", "page": 2}),
    ])
    return p
