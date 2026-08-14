from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from rag.retrievers import HybridRetriever, LLMClient, MockLLMClient, RetrievalResult, RetrieverConfig
from rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT = """
You are an expert search assistant. Your task is to break down a complex user query into 2 to 4 simple, independent sub-queries for document retrieval.
Each sub-query must focus on a specific, distinct part of the main question.

User Query: {query}

Return ONLY a valid JSON object in the following format:
{{
  "sub_queries": [
    "sub-query 1",
    "sub-query 2"
  ]
}}
"""


class DecomposedSearchRetriever:
    """Retriever that decomposes a complex query into simpler sub-queries,

    executes search for each, and aggregates/deduplicates the results.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_backend: Any,
        llm_client: LLMClient | None = None,
        config: RetrieverConfig | None = None,
        base_retriever: Any | None = None,
    ) -> None:
        self.store = vector_store
        self.backend = embedding_backend
        self.llm = llm_client or MockLLMClient()
        self.config = config or RetrieverConfig()
        self.base_retriever = base_retriever or HybridRetriever(
            vector_store, embedding_backend, self.config
        )

    def decompose_query(self, query: str) -> list[str]:
        """Uses LLM to split a complex query into smaller sub-queries."""
        prompt = DECOMPOSITION_PROMPT.format(query=query)
        try:
            llm_response = self.llm.complete(prompt)

            # Safely parse JSON response even if wrapped in Markdown blocks
            json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(llm_response)

            sub_queries = parsed.get("sub_queries", [])
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                cleaned_queries = [
                    sq.strip()
                    for sq in sub_queries
                    if isinstance(sq, str) and sq.strip()
                ]
                if cleaned_queries:
                    return cleaned_queries

        except Exception as e:
            logger.warning(
                "Query decomposition failed: %s. Falling back to original query.", e
            )

        # Fallback to the original query if LLM parsing fails
        return [query]

    def retrieve(self, query: str) -> RetrievalResult:
        """Executes query decomposition, retrieves chunks for each sub-query,

        and returns deduplicated top-k chunks.
        """
        t0 = time.perf_counter()

        # 1. Break down original query into sub-queries
        sub_queries = self.decompose_query(query)
        logger.info(
            "Decomposed query '%s' into %d sub-queries: %s",
            query,
            len(sub_queries),
            sub_queries,
        )

        # 2. Retrieve results for each sub-query and accumulate unique chunks
        accumulated_chunks: dict[str, SearchResult] = {}
        for sub_q in sub_queries:
            step_result = self.base_retriever.retrieve(sub_q)
            for chunk in step_result.chunks:
                # Keep the instance with the higher score if a chunk appears in multiple sub-searches
                if (
                    chunk.chunk_id not in accumulated_chunks
                    or chunk.score > accumulated_chunks[chunk.chunk_id].score
                ):
                    accumulated_chunks[chunk.chunk_id] = chunk

        # 3. Sort accumulated chunks by score descending and keep top_k
        sorted_chunks = sorted(
            accumulated_chunks.values(), key=lambda x: x.score, reverse=True
        )[: self.config.top_k]

        # Re-assign ranks for final result
        for rank, chunk in enumerate(sorted_chunks):
            chunk.rank = rank

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query=query,
            chunks=sorted_chunks,
            strategy="decomposed",
            latency_ms=elapsed_ms,
            intermediate_queries=sub_queries,
            metadata={"sub_queries_count": len(sub_queries)},
        )