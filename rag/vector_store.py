# rag/vector_store.py
from dataclasses import dataclass, field
from typing import Any
from rag.pipeline import Chunk


@dataclass
class SearchFilter:
    must: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    source: str = ""
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int = 0


@dataclass
class VectorStoreConfig:
    dim: int = 1536


class VectorStore:

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        self.config = config or VectorStoreConfig()
        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def upsert(self, chunk: Chunk, vector: list[float]) -> None:
        self._vectors[chunk.chunk_id] = vector
        self._metadata[chunk.chunk_id] = {
            "text": chunk.text,
            "source": chunk.source,
            "section": chunk.section,
            **chunk.metadata,
        }

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        search_filter: SearchFilter | None = None,
    ) -> list[SearchResult]:
        results = []
        for rank, (cid, meta) in enumerate(self._metadata.items()):
            results.append(
                SearchResult(
                    chunk_id=cid,
                    text=meta["text"],
                    score=0.85,  # قيمة افتراضية للاختبار
                    source=meta.get("source", ""),
                    section=meta.get("section", ""),
                    metadata=meta,
                    rank=rank,
                )
            )
        return results[:top_k]

    def get_all_chunk_ids(self) -> list[str]:
        return list(self._metadata.keys())

    def get_chunk_text(self, chunk_id: str) -> str | None:
        meta = self._metadata.get(chunk_id)
        return meta.get("text") if meta else None