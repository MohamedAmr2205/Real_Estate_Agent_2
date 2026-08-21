"""
state_graph/graph_3_property_listing.py
=========================================
Graph 3: Property Listing
- LATS حقيقي مع grounded DB feedback للتسعير
- Constrained ReAct للنشر
- HITL لموافقة المالك
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict, List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, START, END

from state_graph.checkpoint import checkpointer
from state_graph.hitl_node import hitl_node
from state_graph.ticket_system import create_failure_ticket
from state_graph.lats_pricing import lats_pricing_evaluator_node


# ----------------------------------------------------------------
# State
# ----------------------------------------------------------------

class PropertyListingState(TypedDict):
    property_details: Dict[str, Any]
    lats_evaluated_trees: List[Dict[str, Any]]
    selected_listing_plan: Dict[str, Any]
    lats_best_score: float
    pending_action: Dict[str, Any]
    hitl_reason: str
    approved: bool
    status: str


# ----------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------

def constrained_react_listing_node(state: PropertyListingState) -> Dict[str, Any]:
    """
    Constrained ReAct:
    - فقط actions معينة مسموح بيها (whitelist)
    - دايماً بيعمل HITL لموافقة المالك قبل النشر
    """
    try:
        plan  = state.get("selected_listing_plan", {})
        score = state.get("lats_best_score", 0)
        price = plan.get("price", 0)

        # Whitelist check — فقط actions دي مسموح بيها
        ALLOWED_ACTIONS = {"PUBLISH_LISTING", "DRAFT_LISTING", "REQUEST_PHOTOS"}
        action = "PUBLISH_LISTING"

        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Action '{action}' not in whitelist: {ALLOWED_ACTIONS}")

        print(f"[ReAct] action={action} price={price:,.0f} "
              f"score={score:.3f}")

        return {
            "hitl_reason": (
                f"New property listing at {price:,.0f} EGP "
                f"(LATS score={score:.3f}) requires Owner Signoff before publishing."
            ),
            "pending_action": {
                "action": action,
                "plan": plan,
                "lats_score": score,
            },
            "status": "AWAITING_OWNER_APPROVAL",
        }

    except Exception as e:
        prop = state.get("property_details", {})
        create_failure_ticket(
            thread_id=str(prop.get("property_id", "unknown")),
            graph_name="graph_3_property_listing",
            error_msg=str(e),
            current_state=dict(state),
            node_name="constrained_react_listing_node",
            error_type="REACT_CONSTRAINT_ERROR",
        )
        raise


# ----------------------------------------------------------------
# Graph
# ----------------------------------------------------------------

workflow = StateGraph(PropertyListingState)

workflow.add_node("lats_evaluator",  lats_pricing_evaluator_node)
workflow.add_node("react_listing",   constrained_react_listing_node)
workflow.add_node("hitl",            hitl_node)

workflow.add_edge(START,             "lats_evaluator")
workflow.add_edge("lats_evaluator",  "react_listing")
workflow.add_edge("react_listing",   "hitl")
workflow.add_edge("hitl",            END)

graph_3 = workflow.compile(checkpointer=checkpointer)