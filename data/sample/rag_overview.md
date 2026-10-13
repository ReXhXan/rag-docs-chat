# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a retriever with a language model. The retriever finds relevant passages from a document collection and the language model writes an answer grounded in those passages.

## Chunking

Documents are split into overlapping chunks so that each piece fits the embedding model's context window. Overlap preserves context that would otherwise be cut at chunk boundaries.

## Embeddings

An embedding model maps text to dense vectors. Semantically similar texts end up close together, which is measured with cosine similarity.

## Vector databases

A vector database such as ChromaDB or FAISS stores embeddings and supports fast nearest-neighbour search.
