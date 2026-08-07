"""
rag/ — Retrieval-Augmented Generation subsystem
================================================
Owned by Person 3.

Public surface
--------------
Pipeline      – ingest & embed documents
VectorStore   – HNSW ANN index + metadata + pre-filter index
NaiveRetriever, HybridRetriever, AgenticRetriever – retrieval strategies
verify_relevance, verify_groundedness – Self-RAG verification (importable by Person 1)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support direct execution for local development.
if __package__ is None:
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir))
    __package__ = package_dir.name

from .pipeline import (
    Chunk,
    IngestConfig,
    RAGPipeline,
)
from .vector_store import (
    VectorStore,
    VectorStoreConfig,
    SearchFilter,
    SearchResult,
)
from .retrievers import (
    NaiveRetriever,
    HybridRetriever,
    AgenticRetriever,
    RetrieverConfig,
    RetrievalResult,
)
from .verification import (
    verify_relevance,
    verify_groundedness,
    RelevanceVerdict,
    GroundednessVerdict,
    VerificationConfig,
)

__all__ = [
    # Pipeline
    "Chunk",
    "IngestConfig",
    "RAGPipeline",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "SearchFilter",
    "SearchResult",
    # Retrievers
    "NaiveRetriever",
    "HybridRetriever",
    "AgenticRetriever",
    "RetrieverConfig",
    "RetrievalResult",
    # Verification (also importable by Person 1)
    "verify_relevance",
    "verify_groundedness",
    "RelevanceVerdict",
    "GroundednessVerdict",
    "VerificationConfig",
]