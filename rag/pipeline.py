# rag/pipeline.py
import uuid
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
    def from_text(cls, text: str, source: str = "", section: str = "") -> "Chunk":
        return cls(
            chunk_id=str(uuid.uuid4()), text=text, source=source, section=section
        )


@dataclass
class IngestConfig:
    chunk_size: int = 512


class MockEmbeddingBackend:

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        np.random.seed(42)
        return [np.random.rand(self.dim).tolist() for _ in texts]


class RAGPipeline:
    pass