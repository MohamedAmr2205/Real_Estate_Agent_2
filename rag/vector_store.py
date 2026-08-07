"""
rag/vector_store.py
====================
Vector store with:
- Real cosine similarity search (not a fixed 0.85 score)
- HNSW-style ANN index via hnswlib (falls back to brute-force if not installed)
- Metadata payload store
- Metadata index for pre-search filtering
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ----------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------

@dataclass
class SearchFilter:
    """
    Metadata filter applied BEFORE similarity search.
    All keys in `must` must match the chunk's metadata exactly.

    Example:
        SearchFilter(must={"source": "policy_manual.txt", "section": "sedation"})
    """
    must: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float          # cosine similarity in [0, 1]
    source: str = ""
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int = 0


@dataclass
class VectorStoreConfig:
    dim: int = 128
    hnsw_m: int = 16          # HNSW connectivity parameter
    hnsw_ef_construction: int = 200


# ----------------------------------------------------------------
# HNSW index wrapper (hnswlib if available, else brute-force)
# ----------------------------------------------------------------

class _HNSWIndex:
    """
    Thin wrapper around hnswlib.Index.
    Falls back to exact brute-force cosine search if hnswlib
    is not installed, so the code runs everywhere.
    """

    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200) -> None:
        self.dim = dim
        self._id_to_cid: dict[int, str] = {}
        self._cid_to_id: dict[str, int] = {}
        self._next_id = 0

        try:
            import hnswlib
            self._index = hnswlib.Index(space="cosine", dim=dim)
            self._index.init_index(max_elements=100_000,
                                   M=m,
                                   ef_construction=ef_construction)
            self._index.set_ef(50)
            self._use_hnsw = True
        except ImportError:
            # brute-force fallback
            self._vectors: dict[str, list[float]] = {}
            self._use_hnsw = False

    def add(self, chunk_id: str, vector: list[float]) -> None:
        int_id = self._next_id
        self._next_id += 1
        self._id_to_cid[int_id] = chunk_id
        self._cid_to_id[chunk_id] = int_id

        if self._use_hnsw:
            import numpy as np
            self._index.add_items(
                np.array([vector], dtype="float32"), [int_id]
            )
        else:
            self._vectors[chunk_id] = vector

    def search(self, query_vector: list[float],
               top_k: int) -> list[tuple[str, float]]:
        """Returns list of (chunk_id, cosine_similarity) sorted descending."""
        if self._next_id == 0:
            return []

        if self._use_hnsw:
            import numpy as np
            k = min(top_k, self._next_id)
            labels, distances = self._index.knn_query(
                np.array([query_vector], dtype="float32"), k=k
            )
            # hnswlib cosine space returns distance = 1 - similarity
            return [
                (self._id_to_cid[int(lid)], float(1.0 - dist))
                for lid, dist in zip(labels[0], distances[0])
            ]
        else:
            return self._brute_force(query_vector, top_k)

    def _brute_force(self, query: list[float],
                     top_k: int) -> list[tuple[str, float]]:
        scores = []
        for cid, vec in self._vectors.items():
            scores.append((cid, _cosine(query, vec)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)


# ----------------------------------------------------------------
# VectorStore
# ----------------------------------------------------------------

class VectorStore:
    """
    Vector store with:
    - ANN index (HNSW via hnswlib or brute-force fallback)
    - Metadata payload store  (_metadata dict)
    - Metadata index           (_meta_index) for O(1) pre-filtering
    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        self.config = config or VectorStoreConfig()

        # ANN index
        self._ann = _HNSWIndex(
            dim=self.config.dim,
            m=self.config.hnsw_m,
            ef_construction=self.config.hnsw_ef_construction,
        )

        # Metadata payload store: chunk_id → full metadata dict
        self._metadata: dict[str, dict[str, Any]] = {}

        # Metadata index: field_name → value → set of chunk_ids
        # Enables O(1) pre-filtering without scanning all chunks
        self._meta_index: dict[str, dict[Any, set[str]]] = {}

    # ----------------------------------------------------------------
    # Write
    # ----------------------------------------------------------------

    def upsert(self, chunk, vector: list[float]) -> None:
        """Insert or update a chunk + its vector."""
        from rag.pipeline import Chunk
        cid = chunk.chunk_id

        self._ann.add(cid, vector)

        payload = {
            "text": chunk.text,
            "source": chunk.source,
            "section": chunk.section,
            **chunk.metadata,
        }
        self._metadata[cid] = payload

        # Update metadata index
        for key, val in payload.items():
            if key == "text":
                continue
            if key not in self._meta_index:
                self._meta_index[key] = {}
            if val not in self._meta_index[key]:
                self._meta_index[key][val] = set()
            self._meta_index[key][val].add(cid)

    # ----------------------------------------------------------------
    # Read
    # ----------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        search_filter: SearchFilter | None = None,
    ) -> list[SearchResult]:
        """
        1. Pre-filter by metadata (using metadata index) if filter provided
        2. Run ANN search
        3. Return top_k results with real cosine scores
        """
        # Step 1 — pre-filter: find allowed chunk_ids
        allowed: set[str] | None = None
        if search_filter and search_filter.must:
            for key, val in search_filter.must.items():
                matching = self._meta_index.get(key, {}).get(val, set())
                if allowed is None:
                    allowed = set(matching)
                else:
                    allowed &= matching

        # Step 2 — ANN search (fetch more than top_k to account for filtering)
        fetch_k = top_k * 4 if allowed is not None else top_k
        ann_results = self._ann.search(query_vector, fetch_k)

        # Step 3 — apply filter + build SearchResult list
        results: list[SearchResult] = []
        rank = 0
        for cid, score in ann_results:
            if allowed is not None and cid not in allowed:
                continue
            meta = self._metadata.get(cid, {})
            results.append(SearchResult(
                chunk_id=cid,
                text=meta.get("text", ""),
                score=round(score, 4),
                source=meta.get("source", ""),
                section=meta.get("section", ""),
                metadata={k: v for k, v in meta.items() if k != "text"},
                rank=rank,
            ))
            rank += 1
            if rank >= top_k:
                break

        return results

    def get_all_chunk_ids(self) -> list[str]:
        return list(self._metadata.keys())

    def get_chunk_text(self, chunk_id: str) -> str | None:
        meta = self._metadata.get(chunk_id)
        return meta.get("text") if meta else None