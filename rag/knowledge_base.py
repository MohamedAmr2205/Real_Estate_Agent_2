"""
rag/knowledge_base.py
======================
Bridges the Meridian Realty policy knowledge base into the MCP agent's
search_knowledge_base tool.

Uses the same vector store + Gemini embeddings pipeline from pipeline.py,
retrieval from retrievers.py, and Self-RAG verification from verification.py.

Called from mcp_server/server.py (SECTION 10).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------
# Support direct execution from the repository root.
# ----------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# ----------------------------------------------------------------
# Load .env so GEMINI_API_KEY is available when imported by server
# ----------------------------------------------------------------
def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

_load_env(Path(__file__).resolve().parents[1] / ".env")

from rag.pipeline import RAGPipeline, GeminiEmbeddingBackend, IngestConfig
from rag.vector_store import VectorStore, VectorStoreConfig, SearchFilter
from rag.retrievers import HybridRetriever, RetrieverConfig
from rag.verification import verify_relevance, verify_groundedness, VerificationConfig

# ----------------------------------------------------------------
# Policy knowledge base (same corpus as retrieval_eval/eval.py)
# ----------------------------------------------------------------
_KNOWLEDGE_BASE = [
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
# Singleton — build the index once at server startup
# ----------------------------------------------------------------
_store: VectorStore | None = None
_retriever: HybridRetriever | None = None
_embedding: GeminiEmbeddingBackend | None = None

_VERIFY_CONFIG = VerificationConfig(
    relevance_threshold=0.35,
    groundedness_threshold=0.45,
)


def index_property_notes() -> None:
    """
    Build the vector index from the policy knowledge base.
    Called once at server startup (mcp_server/server.py top-level).
    Falls back silently if GEMINI_API_KEY is missing — tool will
    return an informative message instead of crashing the server.
    """
    global _store, _retriever, _embedding

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[RAG] No GEMINI_API_KEY — knowledge base search disabled.")
        return

    try:
        _embedding = GeminiEmbeddingBackend(dim=3072)
        _store = VectorStore(VectorStoreConfig(dim=3072))
        pipeline = RAGPipeline(
            vector_store=_store,
            embedding_backend=_embedding,
            config=IngestConfig(chunk_size=400, chunk_overlap=80),
        )
        pipeline.ingest_documents(_KNOWLEDGE_BASE)
        _retriever = HybridRetriever(
            _store,
            _embedding,
            RetrieverConfig(top_k=5, similarity_threshold=0.0, bm25_weight=0.5),
        )
        print(f"[RAG] Knowledge base indexed — {len(_KNOWLEDGE_BASE)} documents ready.")
    except Exception as e:
        print(f"[RAG] Indexing failed: {e}")
        _store = _retriever = _embedding = None


def search_knowledge_base(
    query: str,
    property_id: int,
    caller_agent_id: int,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Search the policy knowledge base using hybrid RAG (vector + BM25).
    Applies Self-RAG verification before returning results.

    Role check: agent_id=4 is Broker and sees all results.
    Other agents see only non-sensitive policy sections.
    """
    # Sensitive sections only visible to Broker (agent_id == 4)
    BROKER_ONLY_SECTIONS = {"clause_7"}
    IS_BROKER = (caller_agent_id == 4)

    if _retriever is None:
        return {
            "results": [],
            "note": "Knowledge base unavailable (no GEMINI_API_KEY configured).",
        }

    # Retrieve
    result = _retriever.retrieve(query)
    if not result.chunks:
        return {"results": [], "note": "No results found."}

    # Self-RAG verification
    context_text = result.context_text
    rel = verify_relevance(query, context_text, _VERIFY_CONFIG)
    gnd = verify_groundedness(context_text, query, _VERIFY_CONFIG)

    if not rel.relevant:
        return {
            "results": [],
            "note": (
                f"[Self-RAG] Retrieved context not relevant to query "
                f"(ISREL={rel.score:.2f} < {_VERIFY_CONFIG.relevance_threshold}). "
                "No answer returned."
            ),
        }

    # Role filter
    visible_chunks = []
    hidden_count = 0
    for chunk in result.chunks[:top_k]:
        if chunk.section in BROKER_ONLY_SECTIONS and not IS_BROKER:
            hidden_count += 1
            continue
        visible_chunks.append(chunk.text)

    note = ""
    if hidden_count:
        note = (
            f"No relevant notes visible to your role for this property. "
            f"({hidden_count} match(es) exist but require Broker access.)"
        )

    return {
        "results": visible_chunks,
        "note": note,
        "self_rag": {
            "relevance": round(rel.score, 3),
            "groundedness": round(gnd.confidence, 3),
            "verified": rel.relevant and gnd.grounded,
        },
    }