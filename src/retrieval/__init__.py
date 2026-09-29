"""Retrieval Package for Local Vector Search and Indexing.

Exports:
- VectorIndex: Core FAISS IndexFlatIP index with metadata resolution
- build_index_from_embeddings: Builder function linking M3 embeddings to M2 tile metadata
"""

from src.retrieval.builder import build_index_from_embeddings
from src.retrieval.index import VectorIndex

__all__ = [
    "VectorIndex",
    "build_index_from_embeddings",
]
