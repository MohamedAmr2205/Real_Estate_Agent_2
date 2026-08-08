"""
memory/recall.py
=================
Self-RAG-style verification for memory recall.

Person 3 built verify_relevance / verify_groundedness in rag/verification.py.
This module imports those same functions and applies them when the agent
recalls facts from episodic or semantic memory — so the same verification
logic covers both RAG retrieval and memory recall, with no duplicate code.

Usage in agent/client.py:
    from memory.recall import recall_episodic, recall_semantic
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure rag/ is importable when running from any directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.verification import verify_relevance, verify_groundedness, VerificationConfig

try:
    from .episodic_store import EpisodicStore, Episode
    from .semantic_store import SemanticStore
except ImportError:
    from episodic_store import EpisodicStore, Episode
    from semantic_store import SemanticStore


_VERIFY_CONFIG = VerificationConfig(
    relevance_threshold=0.35,
    groundedness_threshold=0.45,
)


def recall_episodic(
    customer_id: int,
    query: str,
    episodic: EpisodicStore,
) -> dict:
    """
    Recall episodes for a customer and verify they're relevant to the query.

    Returns:
        {
            "episodes": [...],          # filtered relevant episodes
            "dropped": int,             # how many failed Self-RAG check
            "self_rag": {...}           # verification scores
        }
    """
    episodes: list[Episode] = episodic.get(customer_id)
    if not episodes:
        return {
            "episodes": [],
            "dropped": 0,
            "self_rag": {"relevance": 0.0, "groundedness": 0.0, "verified": False},
            "note": "No episodes found for this customer.",
        }

    # Build context text from episodes
    context_text = "\n".join(
        f"[{e.timestamp[:10]}] {e.role}: {e.content[:200]}"
        for e in episodes
    )

    # Self-RAG check
    rel = verify_relevance(query, context_text, _VERIFY_CONFIG)
    gnd = verify_groundedness(context_text, query, _VERIFY_CONFIG)
    verified = rel.relevant and gnd.grounded

    if not verified:
        print(
            f"[RECALL][Self-RAG] Episodic memory for customer={customer_id} "
            f"failed verification (ISREL={rel.score:.2f}, ISSUP={gnd.confidence:.2f}) "
            f"— not surfacing to agent."
        )
        return {
            "episodes": [],
            "dropped": len(episodes),
            "self_rag": {
                "relevance": round(rel.score, 3),
                "groundedness": round(gnd.confidence, 3),
                "verified": False,
            },
            "note": "[Self-RAG] Recalled episodes not relevant to current query.",
        }

    print(
        f"[RECALL][Self-RAG] Episodic memory verified "
        f"(ISREL={rel.score:.2f}, ISSUP={gnd.confidence:.2f}) "
        f"— {len(episodes)} episode(s) returned."
    )
    return {
        "episodes": episodes,
        "dropped": 0,
        "self_rag": {
            "relevance": round(rel.score, 3),
            "groundedness": round(gnd.confidence, 3),
            "verified": True,
        },
    }


def recall_semantic(
    customer_id: int,
    query: str,
    semantic: SemanticStore,
) -> dict:
    """
    Recall current semantic facts for a customer and verify relevance.

    Returns:
        {
            "facts": {...},             # key→value dict of current facts
            "dropped": int,
            "self_rag": {...}
        }
    """
    facts: dict = semantic.get_all(customer_id)
    if not facts:
        return {
            "facts": {},
            "dropped": 0,
            "self_rag": {"relevance": 0.0, "groundedness": 0.0, "verified": False},
            "note": "No semantic facts found for this customer.",
        }

    # Build context text from facts
    context_text = "\n".join(f"{k}: {v}" for k, v in facts.items())

    # Self-RAG check
    rel = verify_relevance(query, context_text, _VERIFY_CONFIG)
    gnd = verify_groundedness(context_text, query, _VERIFY_CONFIG)
    verified = rel.relevant and gnd.grounded

    if not verified:
        print(
            f"[RECALL][Self-RAG] Semantic facts for customer={customer_id} "
            f"failed verification (ISREL={rel.score:.2f}, ISSUP={gnd.confidence:.2f}) "
            f"— not surfacing to agent."
        )
        return {
            "facts": {},
            "dropped": len(facts),
            "self_rag": {
                "relevance": round(rel.score, 3),
                "groundedness": round(gnd.confidence, 3),
                "verified": False,
            },
            "note": "[Self-RAG] Recalled facts not relevant to current query.",
        }

    print(
        f"[RECALL][Self-RAG] Semantic facts verified "
        f"(ISREL={rel.score:.2f}, ISSUP={gnd.confidence:.2f}) "
        f"— {len(facts)} fact(s) returned."
    )
    return {
        "facts": facts,
        "dropped": 0,
        "self_rag": {
            "relevance": round(rel.score, 3),
            "groundedness": round(gnd.confidence, 3),
            "verified": True,
        },
    }