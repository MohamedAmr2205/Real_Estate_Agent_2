from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rag import (
    GroundednessVerdict,
    RelevanceVerdict,
    SearchResult,
    VerificationConfig,
    verify_groundedness,
    verify_relevance,
)


test_set_path = Path(__file__).resolve().parent / "test_set.json"


@dataclass
class EvalQuery:
    query: str
    expected_chunk_ids: list[str]
    expected_answer: str = ""
    context: str = ""
    metadata_filter: dict[str, Any] | None = None


@dataclass
class EvalMetrics:
    query: str
    precision_at_5: float
    recall_at_5: float
    relevance: RelevanceVerdict
    groundedness: GroundednessVerdict


def load_test_set(path: Path) -> list[EvalQuery]:
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    queries: list[EvalQuery] = []
    for entry in raw:
        queries.append(
            EvalQuery(
                query=entry.get("query", ""),
                expected_chunk_ids=entry.get("expected_chunk_ids", []),
                expected_answer=entry.get("expected_answer", ""),
                context=entry.get("context", ""),
                metadata_filter=entry.get("metadata_filter"),
            )
        )
    return queries


def precision_at_k(results: list[SearchResult], expected_ids: set[str], k: int = 5) -> float:
    if not results:
        return 0.0
    hits = sum(1 for result in results[:k] if result.chunk_id in expected_ids)
    return hits / min(k, len(results))


def recall_at_k(results: list[SearchResult], expected_ids: set[str], k: int = 5) -> float:
    if not expected_ids:
        return 0.0
    hits = sum(1 for result in results[:k] if result.chunk_id in expected_ids)
    return hits / len(expected_ids)


def evaluate_results(
    query: str,
    results: list[SearchResult],
    expected_ids: set[str],
    expected_answer: str,
    context: str,
    config: VerificationConfig,
) -> EvalMetrics:
    relevance = verify_relevance(query, "\n\n".join(r.text for r in results), config)
    groundedness = verify_groundedness(context, expected_answer or "", config)

    return EvalMetrics(
        query=query,
        precision_at_5=precision_at_k(results, expected_ids, k=5),
        recall_at_5=recall_at_k(results, expected_ids, k=5),
        relevance=relevance,
        groundedness=groundedness,
    )


def summarize(metrics: list[EvalMetrics]) -> dict[str, float]:
    if not metrics:
        return {
            "precision_at_5": 0.0,
            "recall_at_5": 0.0,
            "average_relevance": 0.0,
            "average_groundedness": 0.0,
        }

    return {
        "precision_at_5": sum(m.precision_at_5 for m in metrics) / len(metrics),
        "recall_at_5": sum(m.recall_at_5 for m in metrics) / len(metrics),
        "average_relevance": sum(m.relevance.score for m in metrics) / len(metrics),
        "average_groundedness": sum(m.groundedness.confidence for m in metrics) / len(metrics),
    }


def main() -> None:
    test_queries = load_test_set(test_set_path)
    if not test_queries:
        print("No evaluation queries found in retrieval_eval/test_set.json.")
        print("Create a JSON array of query objects with fields: query, expected_chunk_ids, expected_answer, context.")
        return

    config = VerificationConfig()
    print(f"Loaded {len(test_queries)} evaluation queries.")

    all_metrics: list[EvalMetrics] = []
    for entry in test_queries:
        print(f"- Query: {entry.query}")
        all_metrics.append(
            evaluate_results(
                query=entry.query,
                results=[],
                expected_ids=set(entry.expected_chunk_ids),
                expected_answer=entry.expected_answer,
                context=entry.context,
                config=config,
            )
        )

    summary = summarize(all_metrics)
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
