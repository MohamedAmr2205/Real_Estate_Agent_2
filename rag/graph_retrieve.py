"""
rag/graph_retriever.py
=======================
Graph-based retrieval (bonus architecture).

Builds a lightweight knowledge graph over the ingested chunks:
  - Nodes  = chunks (same chunk_ids as the vector store)
  - Edges  = shared entities between chunks — policy codes (e.g. "3.2b",
             "6.1") extracted via regex, plus a fixed domain-term
             vocabulary relevant to a real-estate brokerage (broker,
             escrow, offer, threshold, disclosure, ...).

Unlike AgenticRetriever (which does iterative LLM-guided re-querying),
GraphRetriever does a single-shot vector search to find seed chunks, then
expands outward along graph edges to pull in chunks that are *topically
or referentially linked* but might not individually score high on vector
similarity to the query — exactly the case in Q07-Q09, where the answer
is spread across several policy sections that reference the same entities
(broker approval, below-threshold offers, risk tiers) but use different
wording than the question.

This is a genuinely different retrieval mechanism from the other three:
no iterative LLM calls, no BM25 term matching. It trades a one-time
graph-build cost (cheap, pure Python) for zero query-time LLM cost, and
wins specifically when relevant chunks are entity-linked rather than
lexically or semantically similar to the query itself.
"""

# ...existing code...
from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any

# Allow running this module directly so package imports work
import sys
import pathlib
if __package__ is None:
    proj_root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(proj_root))

from rag.vector_store import SearchResult, VectorStore
from rag.retrievers import RetrieverConfig, RetrievalResult
# ...existing code...
# Domain vocabulary: real-estate/brokerage entities worth linking chunks on.
# Policy codes (e.g. "3.2b", "6.1") are extracted separately via regex, so
# any new policy number added to the knowledge base is picked up
# automatically without touching this list.
_DOMAIN_ENTITIES = [
    "broker", "escrow", "offer", "buyer", "seller", "threshold",
    "inspection", "commission", "deposit", "disclosure", "closing",
    "agency", "listing", "property", "risk", "tier", "negotiation",
    "counter", "rejection", "rejections", "deadline", "approval",
    "justification", "repair", "roof",
]

_POLICY_CODE_RE = re.compile(r"\b\d+\.\d+[a-z]?\b")
_WORD_RE = re.compile(r"[a-z]+")


def _extract_entities(text: str) -> set[str]:
    """Policy codes + domain terms present in a chunk (or a query)."""
    lower = text.lower()
    entities: set[str] = set(_POLICY_CODE_RE.findall(lower))
    words = set(_WORD_RE.findall(lower))
    entities |= {term for term in _DOMAIN_ENTITIES if term in words}
    return entities


class GraphIndex:
    """
    Builds and holds the chunk graph. Rebuilt on each retrieve() call the
    same way HybridRetriever rebuilds its BM25 index — cheap for a KB this
    size, and guarantees the graph reflects whatever is currently in the
    vector store.
    """

    def __init__(self) -> None:
        self.entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        self.chunk_entities: dict[str, set[str]] = {}
        # adjacency: chunk_id -> {neighbor_chunk_id: edge_weight}
        self.edges: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def build(self, store: VectorStore) -> None:
        self.entity_to_chunks.clear()
        self.chunk_entities.clear()
        self.edges.clear()

        for cid in store.get_all_chunk_ids():
            text = store.get_chunk_text(cid) or ""
            entities = _extract_entities(text)
            self.chunk_entities[cid] = entities
            for e in entities:
                self.entity_to_chunks[e].add(cid)

        # Chunks sharing an entity get connected. Policy-code matches are an
        # exact cross-reference (e.g. two chunks both mention "3.2b"), so
        # they're weighted heavier than a shared generic domain term, which
        # is only a topical coincidence.
        for entity, cids in self.entity_to_chunks.items():
            is_code = bool(_POLICY_CODE_RE.fullmatch(entity))
            weight = 3.0 if is_code else 1.0
            cid_list = list(cids)
            for i in range(len(cid_list)):
                for j in range(i + 1, len(cid_list)):
                    a, b = cid_list[i], cid_list[j]
                    self.edges[a][b] += weight
                    self.edges[b][a] += weight

    def neighbors(self, chunk_id: str) -> dict[str, float]:
        return self.edges.get(chunk_id, {})


class GraphRetriever:
    """
    1. Vector-search for `seed_k` seed chunks (same embedding backend as
       the other retrievers, so latency/relevance are comparable apples
       to apples in the eval table).
    2. Expand from seeds along graph edges up to `max_hops`, accumulating
       an edge-weight score per newly-reached chunk (decayed per hop so
       directly-linked chunks outrank two-hops-away ones).
    3. Rank all reached chunks by (seed similarity + graph weight) and
       return the top_k.
    """

    def __init__(self, vector_store: VectorStore, embedding_backend: Any,
                 config: RetrieverConfig | None = None,
                 max_hops: int = 2, seed_k: int = 2) -> None:
        self.store = vector_store
        self.backend = embedding_backend
        self.config = config or RetrieverConfig()
        self.max_hops = max_hops
        self.seed_k = seed_k
        self.graph = GraphIndex()
        self.graph.build(vector_store)

    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()

        # Rebuild in case the store changed since init — mirrors
        # HybridRetriever's _rebuild_bm25() pattern.
        self.graph.build(self.store)

        query_vec = self.backend.embed([query])[0]
        seeds = self.store.search(
            query_vector=query_vec,
            top_k=self.seed_k,
            search_filter=self.config.search_filter,
        )

        scores: dict[str, float] = defaultdict(float)
        for s in seeds:
            scores[s.chunk_id] += s.score * 2.0  # seed similarity, weighted up

        frontier = {s.chunk_id for s in seeds}
        visited = set(frontier)
        for hop in range(self.max_hops):
            next_frontier: set[str] = set()
            for cid in frontier:
                for neighbor, weight in self.graph.neighbors(cid).items():
                    scores[neighbor] += weight * (0.5 ** hop)  # decay per hop
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.config.top_k]

        chunks: list[SearchResult] = []
        for rank, (cid, score) in enumerate(ranked):
            text = self.store.get_chunk_text(cid)
            if not text:
                continue
            meta = self.store._metadata.get(cid, {})
            chunks.append(SearchResult(
                chunk_id=cid,
                text=text,
                score=round(score, 4),
                source=meta.get("source", ""),
                section=meta.get("section", ""),
                metadata={k: v for k, v in meta.items() if k != "text"},
                rank=rank,
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            query=query,
            chunks=chunks,
            strategy="graph",
            latency_ms=elapsed_ms,
            metadata={
                "seed_chunk_ids": [s.chunk_id for s in seeds],
                "hops": self.max_hops,
            },
        )