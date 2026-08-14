"""
planning/routing.py
====================
Decides which planning algorithm (PS / ToT / LATS) fits each
sub-task in the offer-strategy DAG.

Routing logic:
  - plan_and_solve   → deterministic, single-pass sub-tasks
                       (deadline math, risk-tier arithmetic)
  - tree_of_thoughts → multiple valid strategies worth comparing
                       (ranking offers, choosing negotiation approach)
  - lats             → needs grounded external validation before committing
                       (final recommendation, counter-offer proposal)

Person 2 owns this file.
"""

from __future__ import annotations
from typing import Any

from plan_and_solve import plan_and_solve
from tree_of_thoughts import tree_of_thoughts
from lats import lats

# ---------------------------------------------------------------------------
# Keyword signals for each algorithm
# ---------------------------------------------------------------------------
_PS_SIGNALS = [
    "deadline", "timeline", "days", "weeks", "calculate", "arithmetic",
    "math", "fit", "contingency period", "financing period", "does it fit",
    "how many days", "timing", "schedule",
]

_TOT_SIGNALS = [
    "rank", "compare", "which offer", "best strategy", "recommend strategy",
    "evaluate options", "consider", "weigh", "multiple", "alternatives",
    "which approach", "negotiation approach", "options",
]

_LATS_SIGNALS = [
    "final recommendation", "propose", "counter-offer", "counter offer",
    "draft response", "commit", "accept", "submit", "final decision",
    "conclude", "advise seller", "advise the seller",
]


def route(
    sub_task: str,
    context: dict[str, Any],
    environment=None,
    method_hint: str | None = None,
) -> dict[str, Any]:
    """
    Route a sub-task to the right planning algorithm and run it.

    Args:
        sub_task:    Natural-language sub-task description.
        context:     Offer/property context dict.
        environment: Grounded Environment (required for LATS).
        method_hint: Optional explicit hint from decomposition routing
                     ('plan_and_solve' | 'tree_of_thoughts' | 'lats').
                     If provided, overrides keyword detection.

    Returns:
        Result dict from the chosen algorithm, with 'algorithm' key added.
    """
    algorithm = _pick_algorithm(sub_task, method_hint, environment)
    print(f"[ROUTER] '{sub_task[:60]}...' → {algorithm}")

    if algorithm == "plan_and_solve":
        result = plan_and_solve(sub_task, context, environment=environment)

    elif algorithm == "tree_of_thoughts":
        result = tree_of_thoughts(sub_task, context, environment=environment)

    else:  # lats
        if environment is None:
            raise ValueError(
                "LATS selected but no grounded Environment provided. "
                "Pass environment=Environment() to route()."
            )
        result = lats(sub_task, context, environment=environment)

    result["algorithm"] = algorithm
    return result


def _pick_algorithm(
    sub_task: str,
    method_hint: str | None,
    environment,
) -> str:
    """
    Pick the algorithm. Priority:
      1. Explicit method_hint from decomposition DAG routing hints
      2. Keyword detection on sub_task text
      3. Default: plan_and_solve
    """
    # Explicit hint from the decomposition DAG
    if method_hint in ("plan_and_solve", "tree_of_thoughts", "lats"):
        # LATS needs a grounded environment — fall back to ToT if not provided
        if method_hint == "lats" and environment is None:
            print("[ROUTER] Warning: LATS hinted but no environment — using tree_of_thoughts")
            return "tree_of_thoughts"
        return method_hint

    lower = sub_task.lower()

    # LATS first — highest cost, most specific signals
    if any(signal in lower for signal in _LATS_SIGNALS):
        if environment is not None:
            return "lats"
        return "tree_of_thoughts"  # degrade gracefully without environment

    # ToT — multiple strategies worth comparing
    if any(signal in lower for signal in _TOT_SIGNALS):
        return "tree_of_thoughts"

    # PS — deterministic, single-pass
    if any(signal in lower for signal in _PS_SIGNALS):
        return "plan_and_solve"

    # Default
    return "plan_and_solve"