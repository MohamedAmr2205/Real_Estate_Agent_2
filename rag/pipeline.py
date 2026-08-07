"""
rag/pipeline.py
================
Chunking + embedding pipeline for the Meridian Realty knowledge base.
"""
from __future__ import annotations
import os
import re
import uuid
import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str = ""
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str, source: str = "", section: str = "",
                  metadata: dict[str, Any] | None = None) -> "Chunk":
        return cls(chunk_id=str(uuid.uuid4()), text=text,
                   source=source, section=section, metadata=metadata or {})


@dataclass
class IngestConfig:
    chunk_size: int = 400
    chunk_overlap: int = 80


# ----------------------------------------------------------------
# Keyword dimensions
# ----------------------------------------------------------------
_KEYWORD_DIMS: dict[str, int] = {
    "offer": 0, "submit": 1, "submission": 1,
    "documents": 2, "document": 2, "national": 2, "id": 2,
    "required": 3,
    "proof": 4, "funds": 4,
    "approval": 5, "preapproval": 5,
    "incomplete": 6, "form": 6,
    "commission": 7, "listing": 8, "agent": 9,
    "buyer": 10, "seller": 11,
    "percent": 12, "rate": 12,
    "closing": 13, "proceeds": 14,
    "response": 15, "respond": 15,
    "48": 16, "hours": 16,
    "accept": 17, "reject": 18, "counter": 19,
    "expire": 20, "deposit": 21, "returned": 22,
    "threshold": 23, "below": 24, "85": 25,
    "broker": 26, "written": 27, "forwarded": 28,
    "sign": 30, "3.2b": 31, "policy": 32,
    "escrow": 33, "clause": 34,
    "10": 35, "purchase": 36, "price": 37,
    "business": 38, "days": 39,
    "void": 40, "voids": 40,
    "forfeits": 41,
    "dual": 43, "agency": 44, "disclosure": 45,
    "prohibited": 46, "conflict": 47, "interest": 48,
    "consent": 49, "negotiation": 50,
    "4.1": 51, "section": 52,
    "smouha": 53, "villa": 54,
    "list": 56, "risk": 57, "tier": 58,
    "70": 59, "memo": 60, "justification": 61,
    "rejected": 62, "rejections": 62, "repeated": 63,
    "30": 64, "call": 66, "5.2": 67, "5.4": 68, "90": 69,
    "inspection": 70, "roof": 71, "structural": 72,
    "certified": 73, "report": 74, "repair": 75,
    "50000": 76, "renegotiated": 77,
    "6.1": 78, "6.2": 79, "6.3": 80,
    "available": 81, "alexandria": 82,
    "properties": 83, "property": 84,
    "bed": 85, "bath": 86, "sqft": 87, "asking": 88,
    "standard": 89, "timeline": 90, "residential": 91,
    "extensions": 93, "agreement": 94, "8.1": 95,
    "car": 96, "rental": 97, "vehicle": 98, "tank": 99,
    "penalty": 100, "missing": 101, "deadline": 102,
    "additional": 103, "penalties": 104,
    "sale": 105, "signed": 106, "letter": 107,
    "dated": 108, "copy": 110,
    "transaction": 111, "record": 112,
    "within": 109,
}

_DIM = 128


def _keyword_vector(text: str) -> list[float]:
    text_lower = re.sub(r"[^a-z0-9\s\.]", "", text.lower())
    words = set(text_lower.split())
    patterns = set(re.findall(r"\d+\.?\d*[a-z]?", text_lower))
    words |= patterns

    vec = [0.0] * _DIM
    for kw, dim in _KEYWORD_DIMS.items():
        if kw in words or kw in text_lower:
            vec[dim] += 1.0

    vec[127] = 0.01
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


class MockEmbeddingBackend:
    """Keyword-aware deterministic embedding backend (fallback/testing)."""
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_keyword_vector(t) for t in texts]


class GeminiEmbeddingBackend:
    """
    Real embedding backend using Google Gemini text-embedding-004.
    Uses the new `google-genai` package (not the deprecated google-generativeai).

    Install:  pip install google-genai python-dotenv
    .env:     GEMINI_API_KEY=your_key_here
    """
    def __init__(self, dim: int = 3072) -> None:
        self.dim = dim
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai is not installed.\n"
                "Run: pip install google-genai"
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment.\n"
                "Add it to your .env file: GEMINI_API_KEY=your_key_here"
            )

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = "models/gemini-embedding-001"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts (document task type)."""
        results = []
        for i, text in enumerate(texts):
            response = self._client.models.embed_content(
                model=self._model,
                contents=text,
                config=self._types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            results.append(response.embeddings[0].values)
            # Small delay to avoid rate limits on bulk ingestion
            if i < len(texts) - 1:
                time.sleep(0.05)
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (query task type)."""
        response = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )
        return response.embeddings[0].values


class RAGPipeline:
    def __init__(self, vector_store, embedding_backend=None,
                 config: IngestConfig | None = None) -> None:
        self.store = vector_store
        self.backend = embedding_backend or MockEmbeddingBackend()
        self.config = config or IngestConfig()

    def ingest_text(self, text: str, source: str = "", section: str = "",
                    extra_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        chunks = self._split(text, source=source, section=section,
                             extra_metadata=extra_metadata or {})
        self._embed_and_upsert(chunks)
        return chunks

    def ingest_documents(self, documents: list[dict[str, Any]]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self.ingest_text(
                text=doc.get("text", ""),
                source=doc.get("source", ""),
                section=doc.get("section", ""),
                extra_metadata=doc.get("metadata", {}),
            )
            all_chunks.extend(chunks)
        return all_chunks

    def _split(self, text: str, source: str, section: str,
               extra_metadata: dict[str, Any]) -> list[Chunk]:
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = max(size - overlap, 1)
        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                meta = {"chunk_index": chunk_index,
                        "token_count": len(chunk_text) // 4,
                        **extra_metadata}
                chunks.append(Chunk.from_text(chunk_text, source, section, meta))
                chunk_index += 1
            if end == len(text):
                break
            start += step
        return chunks

    def _embed_and_upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.backend.embed([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            self.store.upsert(chunk, vector)