# rag/retrievers.py
from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag.vector_store import SearchFilter, SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    top_k: int = 5
    similarity_threshold: float = 0.0
    search_filter: SearchFilter | None = None
    bm25_weight: float = 0.4
    rrf_k: int = 60
    max_iterations: int = 3
    min_context_tokens: int = 200
    agent_model: str = "claude-sonnet-4-6"


@dataclass
class RetrievalResult:
    query: str
    chunks: list[SearchResult]
    strategy: str
    latency_ms: float
    iterations: int = 1
    intermediate_queries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {c.source} | Section: {c.section}]\n{c.text}"
            for c in self.chunks
        )


class LLMClient(Protocol):
    def complete(self, prompt: str, system: str | None = None) -> str:
        ...


class MockLLMClient:
    def complete(self, prompt: str, system: str | None = None) -> str:
        if "Determine if the gathered context" in prompt or "Sufficient" in prompt:
            return json.dumps({
                "sufficient": True,
                "reasoning": "Sufficient information collected from initial query.",
                "next_query": ""
            })
        return json.dumps({
            "sufficient": False,
            "reasoning": "Need more specific information.",
            "next_query": "refined search query"
        })


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.term_freqs: dict[str, dict[str, int]] = {}
        self.num_docs: int = 0

    def index_chunks(self, chunks: list[tuple[str, str]]) -> None:
        self.num_docs = len(chunks)
        if self.num_docs == 0:
            return

        total_len = 0
        for cid, text in chunks:
            tokens = self._tokenize(text)
            length = len(tokens)
            self.doc_len[cid] = length
            total_len += length

            tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.term_freqs[cid] = tf

            for t in set(tokens):
                self.doc_freqs[t] += 1

        self.avg_doc_len = total_len / max(self.num_docs, 1)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        tokens = self._tokenize(query)
        scores: dict[str, float] = defaultdict(float)

        for cid, tf in self.term_freqs.items():
            doc_l = self.doc_len[cid]
            score = 0.0
            for t in tokens:
                if t not in tf:
                    continue
                df = self.doc_freqs.get(t, 0)
                idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf[t] + self.k1 * (1.0 - self.b + self.b * (doc_l / max(self.avg_doc_len, 1e-6)))
                score += idf * (tf[t] * (self.k1 + 1.0)) / denom
            if score > 0:
                scores[cid] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


class NaiveRetriever:
    def __init__(self, vector_store: VectorStore, embedding_backend: Any, config: RetrieverConfig | None = None) -> None:
        self.store = vector_store
        self.backend = embedding_backend
        self.config = config or RetrieverConfig()

    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        query_vec = self.backend.embed([query])[0]
        results = self.store.search(
            query_vector=query_vec,
            top_k=self.config.top_k,
            search_filter=self.config.search_filter,
        )

        filtered = [r for r in results if r.score >= self.config.similarity_threshold]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query=query,
            chunks=filtered,
            strategy="naive",
            latency_ms=elapsed_ms,
        )


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, embedding_backend: Any, config: RetrieverConfig | None = None) -> None:
        self.store = vector_store
        self.backend = embedding_backend
        self.config = config or RetrieverConfig()
        self.bm25 = BM25Index()
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        chunk_ids = self.store.get_all_chunk_ids()
        pairs = []
        for cid in chunk_ids:
            txt = self.store.get_chunk_text(cid)
            if txt:
                pairs.append((cid, txt))
        self.bm25.index_chunks(pairs)

    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        query_vec = self.backend.embed([query])[0]
        dense_results = self.store.search(
            query_vector=query_vec,
            top_k=self.config.top_k * 2,
            search_filter=self.config.search_filter,
        )

        self._rebuild_bm25()
        bm25_scores = self.bm25.search(query, top_k=self.config.top_k * 2)

        rrf_scores: dict[str, float] = defaultdict(float)
        k = self.config.rrf_k
        w_bm25 = self.config.bm25_weight
        w_dense = 1.0 - w_bm25

        for rank, res in enumerate(dense_results):
            rrf_scores[res.chunk_id] += w_dense * (1.0 / (k + rank + 1))

        for rank, (cid, score) in enumerate(bm25_scores):
            rrf_scores[cid] += w_bm25 * (1.0 / (k + rank + 1))

        sorted_cids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:self.config.top_k]

        final_chunks: list[SearchResult] = []
        for rank, (cid, rrf_score) in enumerate(sorted_cids):
            txt = self.store.get_chunk_text(cid)
            if not txt:
                continue
            meta = self.store._metadata.get(cid, {})
            final_chunks.append(
                SearchResult(
                    chunk_id=cid,
                    text=txt,
                    score=float(rrf_score),
                    source=meta.get("source", ""),
                    section=meta.get("section", ""),
                    metadata={k: v for k, v in meta.items() if k != "text"},
                    rank=rank,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            strategy="hybrid",
            latency_ms=elapsed_ms,
        )


class AgenticRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_backend: Any,
        llm_client: LLMClient | None = None,
        config: RetrieverConfig | None = None,
    ) -> None:
        self.store = vector_store
        self.backend = embedding_backend
        self.llm = llm_client or MockLLMClient()
        self.config = config or RetrieverConfig()
        self.hybrid_fallback = HybridRetriever(vector_store, embedding_backend, self.config)

    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        accumulated_chunks: dict[str, SearchResult] = {}
        queries_run: list[str] = []
        current_query = query

        for iteration in range(1, self.config.max_iterations + 1):
            queries_run.append(current_query)

            step_result = self.hybrid_fallback.retrieve(current_query)
            for chunk in step_result.chunks:
                accumulated_chunks[chunk.chunk_id] = chunk

            total_tokens = sum(c.metadata.get("token_count", len(c.text) // 4) for c in accumulated_chunks.values())

            if total_tokens >= self.config.min_context_tokens or iteration == self.config.max_iterations:
                break

            prompt = f"""
            User Query: {query}
            Current Accumulated Context ({total_tokens} tokens):
            {'\n---\n'.join(c.text for c in accumulated_chunks.values())}

            Determine if the gathered context is sufficient to fully answer the user query.
            Return a JSON object:
            {{
              "sufficient": bool,
              "reasoning": "...",
              "next_query": "refined search query if not sufficient, else empty string"
            }}
            """
            try:
                llm_response = self.llm.complete(prompt)
                parsed = json.loads(llm_response)
                if parsed.get("sufficient", False):
                    break
                next_q = parsed.get("next_query", "").strip()
                if next_q:
                    current_query = next_q
                else:
                    break
            except Exception as e:
                logger.warning("Agent reflection failed: %s. Halting iterations.", e)
                break

        sorted_final = sorted(accumulated_chunks.values(), key=lambda x: x.score, reverse=True)[:self.config.top_k]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query=query,
            chunks=sorted_final,
            strategy="agentic",
            latency_ms=elapsed_ms,
            iterations=len(queries_run),
            intermediate_queries=queries_run,
        )