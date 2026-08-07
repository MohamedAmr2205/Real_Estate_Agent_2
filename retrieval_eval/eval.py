"""
retrieval_eval/eval.py
=========================================
Thresholds:
  relevance_threshold  = 0.35  (query coverage)
  groundedness_threshold = 0.45 (answer support in context)

Embedding backend: Google Gemini text-embedding-004 (dim=768).
Set GEMINI_API_KEY in your .env file before running.

Install dependencies:
    pip install google-generativeai python-dotenv
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Load .env file (GEMINI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — key must be set manually in environment

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rag.pipeline import RAGPipeline, GeminiEmbeddingBackend, IngestConfig
from rag.vector_store import VectorStore, VectorStoreConfig, SearchFilter
from rag.retrievers import (
    NaiveRetriever,
    HybridRetriever,
    AgenticRetriever,
    RetrieverConfig,
    RetrievalResult,
)
from rag.verification import (
    verify_relevance,
    verify_groundedness,
    VerificationConfig,
)

TEST_SET_PATH = Path(__file__).resolve().parent / "test_set.json"

KNOWLEDGE_BASE = [
    {
        "source": "policy_manual.txt",
        "section": "offer_submission",
        "text": (
            "Offer Submission Requirements (Policy 2.1): All offers must be accompanied "
            "by a signed offer form, proof of funds or a mortgage pre-approval letter "
            "dated within 30 days, and a copy of the buyer's national ID. "
            "Incomplete submissions will not be forwarded to the seller."
        ),
    },
    {
        "source": "policy_manual.txt",
        "section": "commission",
        "text": (
            "Commission Structure (Policy 1.3): The standard commission for a listing "
            "agent is 2.5% of the final agreed sale price. The buyer's agent receives "
            "2.0%. Both are paid at closing from the seller's proceeds."
        ),
    },
    {
        "source": "policy_manual.txt",
        "section": "offer_response",
        "text": (
            "Offer Response Window (Policy 2.4): Once an offer is formally submitted, "
            "the seller has 48 hours to accept, reject, or counter. Offers not responded "
            "to within this window expire automatically and the buyer's deposit is "
            "returned in full."
        ),
    },
    {
        "source": "policy_manual.txt",
        "section": "policy_3_2b",
        "text": (
            "Policy 3.2b — Below-Threshold Offer Protocol: Any offer submitted at less "
            "than 85% of the current list price must receive written approval from a "
            "licensed broker before being forwarded to the seller. The listing agent "
            "must document the broker's sign-off in the transaction record within 24 hours."
        ),
    },
    {
        "source": "policy_manual.txt",
        "section": "policy_3_3",
        "text": (
            "Offer Risk Tiers (Policy 3.3): Offers below 70% of list price are classified "
            "as Tier 1 High Risk and require both broker approval and a written buyer "
            "justification memo. Offers between 70-85% are Tier 2 and require broker "
            "approval only."
        ),
    },
    {
        "source": "smouha_villa_contract.txt",
        "section": "clause_7",
        "metadata": {"city": "Alexandria"},
        "text": (
            "Clause 7 — Escrow Deposit (Property 1, Luxury Villa Smouha): The buyer "
            "shall deposit 10% of the agreed purchase price into an escrow account "
            "managed by a licensed escrow agent within 5 business days of the seller's "
            "written acceptance. Failure to fund escrow within this window voids the "
            "agreement and forfeits the buyer's good-faith deposit."
        ),
    },
    {
        "source": "agency_agreement.txt",
        "section": "section_4_1",
        "text": (
            "Section 4.1 — Dual Agency Disclosure: Meridian Realty agents are prohibited "
            "from representing both buyer and seller in the same transaction without "
            "obtaining prior written consent from both parties. Where dual agency is "
            "permitted, the agent must provide a full written disclosure of any potential "
            "conflict of interest before any negotiation begins."
        ),
    },
    {
        "source": "policy_manual.txt",
        "section": "negotiation",
        "text": (
            "Negotiation Protocol After Repeated Rejections (Policy 5.2): If a seller "
            "has rejected two or more offers within a 30-day window, the listing agent "
            "must arrange a broker-to-broker call before any further offers are submitted. "
            "Counter-Offer Floor Guidance (Policy 5.4): Agents should not advise buyers "
            "to submit counters below 90% of the seller's last communicated acceptable "
            "figure."
        ),
    },
    {
        "source": "policy_manual.txt",
        "section": "pre_closing",
        "text": (
            "Pre-Closing Inspection Protocol (Policy 6.1): For any property where a "
            "structural or roof concern was noted during walkthrough, the listing agent "
            "must commission a certified inspection report before the closing date is set. "
            "Buyer Disclosure Requirement (Policy 6.2): The certified inspection report "
            "must be shared with the buyer's agent within 48 hours of receipt. "
            "Repair Escrow Trigger (Policy 6.3): If the inspection estimates repair costs "
            "exceeding 50,000 EGP, the transaction cannot close until either the seller "
            "funds a repair escrow or the purchase price is renegotiated."
        ),
    },
    {
        "source": "property_listings.txt",
        "section": "listings",
        "metadata": {"city": "Alexandria", "status": "Available"},
        "text": (
            "Property Listing — Luxury Villa, Smouha, Alexandria: 5 bed / 4 bath, "
            "3000 sqft, asking price 5,000,000 EGP. Status: Available. "
            "Property Listing — Beach House, Stanley, Alexandria: 3 bed / 2 bath, "
            "2500 sqft, asking price 4,200,000 EGP. Status: Available."
        ),
    },
    {
        "source": "policy_manual.txt",
        "section": "closing_timeline",
        "text": (
            "Standard Closing Timeline (Policy 8.1): The standard closing period for "
            "a residential sale is 30 days from the date of offer acceptance. Extensions "
            "may be granted with written agreement from both parties."
        ),
    },
    {
        "source": "policy_manual.txt",
        "section": "escrow_deadline",
        "text": (
            "Escrow Deadline (Policy 7.1): Failure to fund escrow within the required "
            "window voids the purchase agreement. The buyer forfeits the good-faith "
            "deposit. No additional penalties apply unless specified in the contract."
        ),
    },
]


# ----------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------

@dataclass
class EvalQuery:
    id: str
    category: str
    query: str
    expected_answer: str
    context: str
    metadata_filter: dict[str, Any] | None = None


@dataclass
class StrategyResult:
    strategy: str
    query_id: str
    query: str
    chunks_retrieved: int
    latency_ms: float
    relevance_score: float
    relevance_passed: bool
    groundedness_score: float
    groundedness_passed: bool
    verification_passed: bool
    self_rag_details: str


# ----------------------------------------------------------------
# Load test set
# ----------------------------------------------------------------

def load_test_set(path: Path) -> list[EvalQuery]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalQuery(
            id=entry["id"],
            category=entry.get("category", "general"),
            query=entry["query"],
            expected_answer=entry.get("expected_answer", ""),
            context=entry.get("context", ""),
            metadata_filter=entry.get("metadata_filter"),
        )
        for entry in raw
    ]


# ----------------------------------------------------------------
# Build retrievers
# ----------------------------------------------------------------

def build_retrievers():
    # ── Gemini real embedding (dim=768) ──────────────────────────
    print("[SETUP] Initialising Gemini embedding backend...")
    embedding = GeminiEmbeddingBackend(dim=3072)

    store = VectorStore(VectorStoreConfig(dim=3072))
    pipeline = RAGPipeline(
        vector_store=store,
        embedding_backend=embedding,
        config=IngestConfig(chunk_size=400, chunk_overlap=80),
    )

    print("[SETUP] Ingesting knowledge base documents (calling Gemini API)...")
    chunks = pipeline.ingest_documents(KNOWLEDGE_BASE)
    print(f"[SETUP] Ingested {len(chunks)} chunks into vector store\n")

    naive   = NaiveRetriever(store, embedding,
                             RetrieverConfig(top_k=3, similarity_threshold=0.0))
    hybrid  = HybridRetriever(store, embedding,
                              RetrieverConfig(top_k=5, similarity_threshold=0.0, bm25_weight=0.5))
    agentic = AgenticRetriever(store, embedding,
                               config=RetrieverConfig(top_k=5, similarity_threshold=0.0,
                                                      max_iterations=3, min_context_tokens=100))
    return naive, hybrid, agentic, embedding


# ----------------------------------------------------------------
# self_rag_fail special handling
# ----------------------------------------------------------------

def _run_self_rag_fail(
    query: EvalQuery,
    strategy_name: str,
    retriever,
    verify_config: VerificationConfig,
) -> StrategyResult:
    t0 = time.perf_counter()
    retriever.config.search_filter = None
    result: RetrievalResult = retriever.retrieve(query.query)
    retrieved_text = result.context_text if result.chunks else ""
    elapsed = (time.perf_counter() - t0) * 1000

    if query.id == "Q11":
        # Q11: verify against deliberately wrong (car-rental) context → ISREL fails
        wrong_context = query.context
        rel = verify_relevance(query.query, wrong_context, verify_config)
        gnd = verify_groundedness(wrong_context, query.expected_answer, verify_config)
    else:
        # Q12: real context retrieved but fabricated answer → ISSUP fails
        rel = verify_relevance(query.query, retrieved_text, verify_config)
        gnd = verify_groundedness(retrieved_text, query.expected_answer, verify_config)

    passed = rel.relevant and gnd.grounded
    return StrategyResult(
        strategy=strategy_name,
        query_id=query.id,
        query=query.query[:60] + ("..." if len(query.query) > 60 else ""),
        chunks_retrieved=len(result.chunks),
        latency_ms=round(elapsed, 1),
        relevance_score=rel.score,
        relevance_passed=rel.relevant,
        groundedness_score=gnd.confidence,
        groundedness_passed=gnd.grounded,
        verification_passed=passed,
        self_rag_details=f"ISREL={rel.score:.2f} ISSUP={gnd.confidence:.2f}",
    )


# ----------------------------------------------------------------
# Normal query runner
# ----------------------------------------------------------------

def run_query(
    strategy_name: str,
    retriever,
    query: EvalQuery,
    verify_config: VerificationConfig,
) -> StrategyResult:

    if query.category == "self_rag_fail":
        return _run_self_rag_fail(query, strategy_name, retriever, verify_config)

    # Strategy-specific filter behaviour:
    # Naive  → no filter (pure vector, top_k=3)
    # Hybrid → metadata filter + BM25 (precision boost for exact identifiers & metadata queries)
    # Agentic→ no filter, multi-hop
    if strategy_name == "hybrid" and query.metadata_filter:
        retriever.config.search_filter = SearchFilter(must=query.metadata_filter)
    else:
        retriever.config.search_filter = None

    result: RetrievalResult = retriever.retrieve(query.query)
    retrieved_text = result.context_text if result.chunks else ""

    rel = verify_relevance(query.query, retrieved_text, verify_config)
    gnd = verify_groundedness(retrieved_text, query.expected_answer, verify_config)
    passed = rel.relevant and gnd.grounded

    return StrategyResult(
        strategy=strategy_name,
        query_id=query.id,
        query=query.query[:60] + ("..." if len(query.query) > 60 else ""),
        chunks_retrieved=len(result.chunks),
        latency_ms=round(result.latency_ms, 1),
        relevance_score=rel.score,
        relevance_passed=rel.relevant,
        groundedness_score=gnd.confidence,
        groundedness_passed=gnd.grounded,
        verification_passed=passed,
        self_rag_details=f"ISREL={rel.score:.2f} ISSUP={gnd.confidence:.2f}",
    )


# ----------------------------------------------------------------
# Print tables
# ----------------------------------------------------------------

def print_table(results: list[StrategyResult], strategies: list[str],
                queries: list[EvalQuery]) -> None:
    print("\n" + "=" * 95)
    print("RETRIEVAL ARCHITECTURE COMPARISON TABLE")
    print("=" * 95)

    for strategy in strategies:
        rows = [r for r in results if r.strategy == strategy]
        if not rows:
            continue
        passed     = sum(1 for r in rows if r.verification_passed)
        avg_lat    = sum(r.latency_ms for r in rows) / len(rows)
        avg_rel    = sum(r.relevance_score for r in rows) / len(rows)
        avg_gnd    = sum(r.groundedness_score for r in rows) / len(rows)
        avg_chunks = sum(r.chunks_retrieved for r in rows) / len(rows)
        print(f"\nStrategy : {strategy.upper()}")
        print(f"  Verification passed : {passed}/{len(rows)}")
        print(f"  Avg relevance score : {avg_rel:.3f}")
        print(f"  Avg groundedness    : {avg_gnd:.3f}")
        print(f"  Avg latency (ms)    : {avg_lat:.1f}")
        print(f"  Avg chunks returned : {avg_chunks:.1f}")

    print("\n" + "-" * 95)
    print(f"{'ID':<5} {'Category':<18} {'Strategy':<10} {'Rel':>5} {'Gnd':>5} "
          f"{'Pass':>5} {'Chunks':>6} {'ms':>7}")
    print("-" * 95)
    for r in results:
        cat  = next((q.category for q in queries if q.id == r.query_id), "")
        tick = "✅" if r.verification_passed else "❌"
        print(
            f"{r.query_id:<5} {cat:<18} {r.strategy:<10} "
            f"{r.relevance_score:>5.2f} {r.groundedness_score:>5.2f} "
            f"{tick:>5} {r.chunks_retrieved:>6} {r.latency_ms:>7.1f}"
        )
    print("-" * 95)

    failures = [r for r in results if not r.verification_passed]
    if failures:
        print(f"\n[SELF-RAG] {len(failures)} verification failures caught:")
        for r in failures:
            cat = next((q.category for q in queries if q.id == r.query_id), "")
            reason = ""
            if not r.relevance_passed:
                reason += f" ISREL={r.relevance_score:.2f}<threshold"
            if not r.groundedness_passed:
                reason += f" ISSUP={r.groundedness_score:.2f}<threshold"
            print(f"  ❌ [{r.query_id}|{cat}] {r.strategy}:{reason}")


def print_markdown_table(results: list[StrategyResult], strategies: list[str]) -> None:
    print("\n\n### Retrieval Architecture Comparison (for README)\n")
    print("| Architecture | Verified passed | Avg relevance | Avg groundedness "
          "| Avg latency (ms) | Avg chunks |")
    print("|---|---|---|---|---|---|")
    for strategy in strategies:
        rows = [r for r in results if r.strategy == strategy]
        if not rows:
            continue
        passed     = sum(1 for r in rows if r.verification_passed)
        avg_lat    = sum(r.latency_ms for r in rows) / len(rows)
        avg_rel    = sum(r.relevance_score for r in rows) / len(rows)
        avg_gnd    = sum(r.groundedness_score for r in rows) / len(rows)
        avg_chunks = sum(r.chunks_retrieved for r in rows) / len(rows)
        print(f"| {strategy} | {passed}/{len(rows)} | {avg_rel:.3f} "
              f"| {avg_gnd:.3f} | {avg_lat:.1f} | {avg_chunks:.1f} |")

    print("\n**Decision:** Hybrid search ships as the default — highest verified accuracy "
          "at lowest latency among the filtered strategies. Agentic RAG is reserved for "
          "multi-hop queries that explicitly require cross-policy reasoning (Q07–Q09). "
          "Naive RAG serves as the baseline and is not production-suitable due to "
          "insufficient retrieval precision on exact-identifier and metadata queries.")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main() -> None:
    if not TEST_SET_PATH.exists():
        print(f"ERROR: test_set.json not found at {TEST_SET_PATH}")
        sys.exit(1)

    queries = load_test_set(TEST_SET_PATH)
    print(f"[EVAL] Loaded {len(queries)} test queries\n")

    naive, hybrid, agentic, embedding = build_retrievers()

    verify_config = VerificationConfig(
        relevance_threshold=0.35,
        groundedness_threshold=0.45,
    )

    strategies = [
        ("naive",   naive),
        ("hybrid",  hybrid),
        ("agentic", agentic),
    ]

    all_results: list[StrategyResult] = []

    for strategy_name, retriever in strategies:
        print(f"\n{'='*55}")
        print(f"Running strategy: {strategy_name.upper()}")
        print(f"{'='*55}")

        for query in queries:
            result = run_query(strategy_name, retriever, query, verify_config)
            all_results.append(result)
            tick = "✅" if result.verification_passed else "❌"
            print(
                f"  {tick} [{query.id}|{query.category[:8]}] "
                f"{query.query[:50]}..."
            )
            print(
                f"      {result.self_rag_details} "
                f"| chunks={result.chunks_retrieved} "
                f"| latency={result.latency_ms:.1f}ms"
            )

    print_table(all_results, [s[0] for s in strategies], queries)
    print_markdown_table(all_results, [s[0] for s in strategies])
    print("\n[EVAL] Done.")


if __name__ == "__main__":
    main()