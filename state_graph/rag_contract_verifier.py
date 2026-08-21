"""
state_graph/rag_contract_verifier.py
======================================
RAG Architecture — بيستخدم الـ rag/ pipeline الحقيقي
اللي اتبنى في الـ Memory & RAG Lab.

مش hardcoded rules — بيعمل retrieval حقيقي من الـ vector store.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.pipeline import RAGPipeline, MockEmbeddingBackend, IngestConfig
from rag.vector_store import VectorStore, VectorStoreConfig, SearchFilter
from rag.retrievers import HybridRetriever, RetrieverConfig
from rag.verification import verify_relevance, verify_groundedness, VerificationConfig

# ----------------------------------------------------------------
# Contract knowledge base — policies اللي الـ RAG بيسترجعها
# ----------------------------------------------------------------
CONTRACT_DOCUMENTS = [
    {
        "source": "contract_policy.txt",
        "section": "escrow",
        "text": (
            "Escrow Policy 7.1: Buyer must deposit 10% of agreed purchase price "
            "into a licensed escrow account within 5 business days of offer acceptance. "
            "Failure to fund escrow voids the purchase agreement."
        ),
    },
    {
        "source": "contract_policy.txt",
        "section": "broker_signoff",
        "text": (
            "Broker Signoff Policy 3.2: All deals exceeding 1,000,000 EGP require "
            "mandatory written broker approval before contract execution. "
            "The broker must sign the final contract within 48 hours."
        ),
    },
    {
        "source": "contract_policy.txt",
        "section": "closing_timeline",
        "text": (
            "Closing Timeline Policy 8.1: Standard closing period is 30 days from "
            "offer acceptance. Extensions require written consent from both parties. "
            "Late closing incurs a 0.1% daily penalty on the deal value."
        ),
    },
    {
        "source": "contract_policy.txt",
        "section": "dual_agency",
        "text": (
            "Dual Agency Policy 4.1: Agent representing both buyer and seller "
            "must obtain written consent from both parties before proceeding. "
            "Full conflict-of-interest disclosure required."
        ),
    },
    {
        "source": "contract_policy.txt",
        "section": "inspection",
        "text": (
            "Property Inspection Policy 6.1: Any structural or roof concern flagged "
            "during walkthrough requires certified inspection before closing. "
            "Repair costs exceeding 50,000 EGP trigger mandatory renegotiation."
        ),
    },
]

# ----------------------------------------------------------------
# Build RAG pipeline (shared across calls)
# ----------------------------------------------------------------

_store: VectorStore | None = None
_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _store, _retriever
    if _retriever is not None:
        return _retriever

    embedding = MockEmbeddingBackend(dim=128)
    _store = VectorStore(VectorStoreConfig(dim=128))
    pipeline = RAGPipeline(
        vector_store=_store,
        embedding_backend=embedding,
        config=IngestConfig(chunk_size=400, chunk_overlap=80),
    )

    print("[RAG] Ingesting contract policy documents...")
    chunks = pipeline.ingest_documents(CONTRACT_DOCUMENTS)
    print(f"[RAG] {len(chunks)} chunks indexed")

    cfg = RetrieverConfig(top_k=3, similarity_threshold=0.0)
    _retriever = HybridRetriever(_store, embedding, cfg)
    return _retriever


# ----------------------------------------------------------------
# RAG verification function
# ----------------------------------------------------------------

def retrieve_contract_rules(
    query: str,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Retrieve relevant contract rules for a given query.
    Runs Self-RAG verification on results.

    Returns:
        retrieved_rules: list of relevant rule texts
        verification_passed: bool
        relevance_score: float
    """
    retriever = _get_retriever()

    if metadata_filter:
        retriever.config.search_filter = SearchFilter(must=metadata_filter)
    else:
        retriever.config.search_filter = None

    result = retriever.retrieve(query)
    retrieved_text = "\n\n".join(c.text for c in result.chunks)

    # Self-RAG ISREL check
    verify_cfg = VerificationConfig(relevance_threshold=0.1)
    rel = verify_relevance(query, retrieved_text, verify_cfg)

    print(f"[RAG] Retrieved {len(result.chunks)} chunks for: '{query[:50]}'")
    print(f"[RAG] ISREL={rel.score:.3f} relevant={rel.relevant}")

    return {
        "retrieved_rules": [c.text for c in result.chunks],
        "retrieved_text": retrieved_text,
        "relevance_score": rel.score,
        "verification_passed": rel.relevant,
        "latency_ms": result.latency_ms,
    }


# ----------------------------------------------------------------
# LangGraph node wrapper
# ----------------------------------------------------------------

def rag_contract_verifier_node(state: dict) -> dict:
    """
    Drop-in replacement for the old hardcoded rag_contract_verifier_node.
    Uses real RAG retrieval instead of hardcoded rules.
    """
    deal_id = state.get("deal_id", "unknown")
    deal_value = state.get("deal_value", 0)

    # Build query from deal context
    query = f"escrow broker signoff requirements for deal {deal_id}"
    if deal_value > 1_000_000:
        query += " large deal above 1 million broker approval"

    rag_result = retrieve_contract_rules(query)

    rules = rag_result["retrieved_rules"]
    verification_passed = rag_result["verification_passed"]

    # Determine if HITL needed based on retrieved rules
    needs_broker = any("broker" in r.lower() for r in rules)
    needs_escrow = any("escrow" in r.lower() for r in rules)

    hitl_reasons = []
    if needs_broker and deal_value > 1_000_000:
        hitl_reasons.append("Broker sign-off required for deals above 1M EGP")
    if needs_escrow:
        hitl_reasons.append("Escrow verification requires admin confirmation")

    hitl_reason = " | ".join(hitl_reasons) if hitl_reasons else \
        "Contract verification requires human review"

    return {
        "retrieved_contract_rules": rules,
        "rag_verification_passed": verification_passed,
        "hitl_reason": hitl_reason,
        "pending_action": {
            "action": "BROKER_FINAL_SIGNATURE",
            "rules": rules,
            "deal_id": deal_id,
        },
    }