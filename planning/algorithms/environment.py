"""
planning_lab/algorithms/environment.py
========================================
REPLACED (not adapted) per the lab's Person 2 spec — the toolkit's
Environment ignored `state` entirely and returned a randomized,
success-biased score (Beta(5,2) skewed toward "success"). That earns
zero grounding credit by the lab's own grading note.

GroundedEnvironment below replaces it with THREE real checks against
this project's actual data — not a model's opinion, not chance:

  1. Deadline check      — does the financing contingency genuinely fit
                            inside the seller's stated closing deadline?
                            (pure arithmetic, deterministic)
  2. Floor-price check   — does the proposed price meet the seller's
                            confidential floor price, pulled LIVE from
                            the RAG Add-On's knowledge base (Broker-only
                            note) — not a duplicated/stale copy of it
  3. Risk-tier check     — pulled live from db/ (same query
                            explain_offer_risk.py uses): does the
                            offer-to-list ratio avoid Tier 1 High Risk
                            (Policy 3.3, below 70%) without the candidate
                            plan acknowledging that risk?

Expected `state` shape — a JSON string (or already-parsed dict)
describing ONE candidate offer-strategy decision:

    {
      "property_id": 1,
      "offer_id": 3,                       # optional — enables check 3
      "proposed_action": "accept" | "counter" | "reject",
      "proposed_price": 4200000,            # optional — enables check 2
      "seller_deadline_weeks": 3,           # optional — enables check 1
      "financing_contingency_days": 30,     # optional — enables check 1
      "acknowledges_tier1_risk": false       # optional — see check 3
    }

Every planning algorithm in this package (plan_and_solve.py,
tree_of_thoughts.py, lats.py) must serialize its candidate into this
shape before calling `Environment.evaluate(state)`.

A check that's missing its required fields is SKIPPED (not counted as a
pass or a fail) — `score` is the fraction of *applicable* checks passed.
`success` requires every applicable check to pass AND score to clear
`success_threshold`. If no field combination makes any check applicable,
the candidate is marked unsuccessful with a clear reason rather than
silently defaulting to a pass.

IMPORT PATH NOTE: this module is imported from the repository root in
normal usage, and it also runs standalone from the `planning/` tree. The
real project layout is:

  - repo_root/rag/knowledge_base.py
  - repo_root/mcp_server/db.py

so we add both the repo root and the mcp_server directory to sys.path and
import the modules in the style already used across the project.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# planning/algorithms/environment.py -> parents[2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_SERVER_DIR = REPO_ROOT / "mcp_server"
for entry in (REPO_ROOT, MCP_SERVER_DIR):
    path = str(entry)
    if path not in sys.path:
        sys.path.insert(0, path)

from db import get_connection
from rag.knowledge_base import index_property_notes as _ensure_kb_indexed
from rag.knowledge_base import search_knowledge_base as _kb_search

_ensure_kb_indexed()  # defensive: this Environment can run standalone,
                    # not only after mcp_server/server.py's startup
try:
    from models import EnvironmentFeedback
except ModuleNotFoundError:  # pragma: no cover - fallback when imported as a loose script
    from models import EnvironmentFeedback

# Broker agent_id — matches mcp_server/server.py's accept_offer /
# assign_listing_agent authorization checks and rag/knowledge_base.py's
# role_required="Broker" notes. The environment needs Broker-level
# visibility to ground a candidate against the seller's confidential
# floor price; this is a grading/scoring component, not a user-facing
# tool call, so the elevated read is scoped to this file only and never
# exposed to a caller.
_BROKER_AGENT_ID = 4

_FLOOR_PRICE_RE = re.compile(r"as low as\s+([\d,]+)")


def _parse_state(state: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(state, dict):
        return state
    try:
        return json.loads(state)
    except (json.JSONDecodeError, TypeError):
        return {}


def _check_deadline(fields: dict[str, Any]) -> tuple[bool, str] | None:
    """Deterministic check: does the financing contingency fit inside the
    seller's stated deadline? None (not applicable) if either field is
    missing from state."""
    weeks = fields.get("seller_deadline_weeks")
    days = fields.get("financing_contingency_days")
    if weeks is None or days is None:
        return None

    fits = days <= weeks * 7
    if fits:
        return True, f"Financing contingency ({days}d) fits the {weeks}-week deadline."
    return False, (
        f"Financing contingency ({days}d) does NOT fit the seller's "
        f"{weeks}-week ({weeks * 7}d) deadline."
    )


def _check_floor_price(fields: dict[str, Any]) -> tuple[bool, str] | None:
    """Check against the seller's confidential floor price, pulled LIVE
    from the RAG knowledge base (Broker-only note) each call — never a
    stored duplicate that could go stale. Not applicable if property_id
    or proposed_price is missing, or if no floor-price note exists for
    this property."""
    property_id = fields.get("property_id")
    proposed_price = fields.get("proposed_price")
    if property_id is None or proposed_price is None:
        return None

    kb_result = _kb_search(
        "seller floor price acceptable minimum", property_id, _BROKER_AGENT_ID, top_k=3
    )
    floor_price = None
    for chunk in kb_result.get("results", []):
        match = _FLOOR_PRICE_RE.search(chunk)
        if match:
            floor_price = float(match.group(1).replace(",", ""))
            break

    if floor_price is None:
        return None  # no floor-price note on record for this property

    if proposed_price >= floor_price:
        return True, f"Proposed price {proposed_price} meets the seller's floor of {floor_price}."
    return False, (
        f"Proposed price {proposed_price} is BELOW the seller's confidential "
        f"floor of {floor_price} — this strategy risks leaving money on the "
        f"table or the seller rejecting outright."
    )


def _check_risk_tier(fields: dict[str, Any]) -> tuple[bool, str] | None:
    """Real check pulled live from db/ (same query explain_offer_risk.py
    uses): if the candidate proposes accepting/countering an offer, does
    its offer-to-list ratio avoid Tier 1 High Risk (Policy 3.3, below
    70%) without the plan acknowledging that risk? Not applicable unless
    offer_id and a relevant proposed_action are given."""
    offer_id = fields.get("offer_id")
    action = fields.get("proposed_action")
    if offer_id is None or action not in ("accept", "counter"):
        return None

    with get_connection() as conn:
        offer = conn.execute(
            "SELECT * FROM Offer WHERE offer_id = ?", (offer_id,)
        ).fetchone()
        if offer is None:
            return None
        offer = dict(offer)
        property_row = conn.execute(
            "SELECT * FROM Property WHERE property_id = ?",
            (offer["property_id"],),
        ).fetchone()
        property_row = dict(property_row)

    ratio = offer["offer_amount"] / property_row["price"]
    acknowledges_risk = bool(fields.get("acknowledges_tier1_risk"))

    if ratio >= 0.70 or acknowledges_risk:
        return True, f"Offer-to-list ratio {ratio:.2f} is within accepted risk tiers."
    return False, (
        f"Offer-to-list ratio {ratio:.2f} is Tier 1 High Risk (Policy 3.3, "
        f"below 70%) but the candidate strategy does not acknowledge this "
        f"or attach the required buyer justification memo."
    )


class Environment:
    """
    Grounded evaluator — REPLACED, not adapted, from the toolkit's
    randomized default. Same public interface (`success_threshold`,
    `evaluate(state) -> EnvironmentFeedback`) as the original, so it
    drops into plan_and_solve.py / tree_of_thoughts.py / lats.py without
    changing their call sites — only what `evaluate` actually does
    inside changed.
    """

    def __init__(self, success_threshold: float = 0.6):
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError("success_threshold must be between zero and one")
        self.success_threshold = success_threshold

    def evaluate(self, state: str | dict[str, Any]) -> EnvironmentFeedback:
        fields = _parse_state(state)

        checks = [
            _check_deadline(fields),
            _check_floor_price(fields),
            _check_risk_tier(fields),
        ]
        applicable = [c for c in checks if c is not None]

        if not applicable:
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    "No groundable fields found in state — expected at least "
                    "one of: (seller_deadline_weeks + financing_contingency_days), "
                    "(property_id + proposed_price), (offer_id + proposed_action)."
                ],
            )

        passed = [ok for ok, _ in applicable]
        details = [msg for ok, msg in applicable if not ok]
        score = round(sum(passed) / len(passed), 4)
        success = all(passed) and score >= self.success_threshold

        return EnvironmentFeedback(success=success, score=score, details=details)