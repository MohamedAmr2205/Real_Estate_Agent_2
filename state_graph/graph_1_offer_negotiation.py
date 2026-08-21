"""
state_graph/graph_1_offer_negotiation.py
=========================================
Graph 1: Offer Negotiation
- ToT (real LLM branches) للاستراتيجية
- Constrained ReAct للتنفيذ
- HITL لو الخصم > 15%
- Ticket لو في runtime error
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from state_graph.checkpoint import checkpointer
from state_graph.hitl_node import hitl_node
from state_graph.ticket_system import create_failure_ticket
from state_graph.tot_strategy import tot_strategy_node


# ----------------------------------------------------------------
# State
# ----------------------------------------------------------------

class NegotiationState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    property_id: str
    offered_price: float
    original_price: float
    evaluated_strategies: List[Dict[str, Any]]
    chosen_strategy: str
    recommended_counter: float
    tot_rationale: str
    pending_action: Dict[str, Any]
    hitl_reason: str
    approved: bool
    status: str


# ----------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------

def constrained_react_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Constrained ReAct:
    - Checks if action is within allowed bounds
    - Auto-accepts if discount <= 15%
    - Routes to HITL if discount > 15%
    - Creates ticket on unexpected error
    """
    try:
        strategy   = state.get("chosen_strategy", "Unknown")
        offered    = state.get("offered_price", 0)
        original   = state.get("original_price", 1)
        counter    = state.get("recommended_counter", offered)
        discount   = (original - offered) / original if original else 0

        print(f"[ReAct] strategy={strategy} offered={offered:,.0f} "
              f"discount={discount*100:.1f}%")

        # Constraint check
        if discount > 0.15:
            return {
                "hitl_reason": (
                    f"Discount rate ({discount*100:.1f}%) exceeds "
                    f"auto-approval limit (15%). "
                    f"Recommended counter: {counter:,.0f} EGP."
                ),
                "pending_action": {
                    "action": "APPROVE_OFFER",
                    "offered_price": offered,
                    "recommended_counter": counter,
                    "strategy": strategy,
                },
                "status": "AWAITING_HITL",
            }

        return {"status": "AUTO_ACCEPTED"}

    except Exception as e:
        ticket_id = create_failure_ticket(
            thread_id=state.get("property_id", "unknown"),
            graph_name="graph_1_offer_negotiation",
            error_msg=str(e),
            current_state=dict(state),
            node_name="constrained_react_node",
            error_type="REACT_ERROR",
        )
        raise RuntimeError(f"Ticket #{ticket_id} created for error: {e}") from e


# ----------------------------------------------------------------
# Routing
# ----------------------------------------------------------------

def route_after_react(state: NegotiationState) -> str:
    if state.get("hitl_reason"):
        return "hitl"
    return END


def route_after_hitl(state: NegotiationState) -> str:
    if state.get("approved"):
        return END
    # Rejected — re-run ToT for new strategy
    return "tot_strategy"


# ----------------------------------------------------------------
# Graph
# ----------------------------------------------------------------

workflow = StateGraph(NegotiationState)

workflow.add_node("tot_strategy",  tot_strategy_node)
workflow.add_node("react_agent",   constrained_react_node)
workflow.add_node("hitl",          hitl_node)

workflow.add_edge(START, "tot_strategy")
workflow.add_edge("tot_strategy", "react_agent")

workflow.add_conditional_edges(
    "react_agent",
    route_after_react,
    {"hitl": "hitl", END: END},
)
workflow.add_conditional_edges(
    "hitl",
    route_after_hitl,
    {END: END, "tot_strategy": "tot_strategy"},
)

graph_1 = workflow.compile(checkpointer=checkpointer)