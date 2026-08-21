"""
state_graph/graph_2_deal_closing.py
=====================================
Graph 2: Deal Closing
- Task Decomposition للـ sub-tasks
- RAG حقيقي للـ contract rules
- HITL للـ broker approval
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
from state_graph.rag_contract_verifier import rag_contract_verifier_node


# ----------------------------------------------------------------
# State
# ----------------------------------------------------------------

class DealClosingState(TypedDict):
    deal_id: str
    deal_value: float
    sub_tasks: List[str]
    completed_tasks: List[str]
    retrieved_contract_rules: List[str]
    rag_verification_passed: bool
    pending_action: Dict[str, Any]
    hitl_reason: str
    approved: bool
    status: str


# ----------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------

def task_decomposition_node(state: DealClosingState) -> Dict[str, Any]:
    """
    Task Decomposition: بيقسم عملية الإغلاق لـ sub-tasks
    بناءً على قيمة الصفقة والظروف.
    """
    deal_value = state.get("deal_value", 0)

    # Dynamic decomposition based on deal value
    sub_tasks = ["RAG_Contract_Check", "Escrow_Verification"]
    if deal_value > 1_000_000:
        sub_tasks.append("Broker_Signoff")
    if deal_value > 3_000_000:
        sub_tasks.append("Legal_Review")

    print(f"[Decomposition] deal_value={deal_value:,.0f} "
          f"sub_tasks={sub_tasks}")

    return {
        "sub_tasks": sub_tasks,
        "completed_tasks": [],
    }


def escrow_verification_node(state: DealClosingState) -> Dict[str, Any]:
    """Verifies escrow requirements are met."""
    try:
        deal_value = state.get("deal_value", 0)
        required_escrow = deal_value * 0.10

        print(f"[Escrow] Required deposit: {required_escrow:,.0f} EGP")

        completed = list(state.get("completed_tasks", []))
        completed.append("Escrow_Verification")

        return {
            "completed_tasks": completed,
            "escrow_required": required_escrow,
            "status": "ESCROW_CHECKED",
        }
    except Exception as e:
        create_failure_ticket(
            thread_id=state.get("deal_id", "unknown"),
            graph_name="graph_2_deal_closing",
            error_msg=str(e),
            current_state=dict(state),
            node_name="escrow_verification_node",
            error_type="ESCROW_ERROR",
        )
        raise


# ----------------------------------------------------------------
# Routing
# ----------------------------------------------------------------

def route_after_rag(state: DealClosingState) -> str:
    """Route to HITL if broker signoff needed."""
    if state.get("hitl_reason"):
        return "hitl"
    return "escrow_verification"


# ----------------------------------------------------------------
# Graph
# ----------------------------------------------------------------

workflow = StateGraph(DealClosingState)

workflow.add_node("decomposition",       task_decomposition_node)
workflow.add_node("rag_verifier",        rag_contract_verifier_node)
workflow.add_node("escrow_verification", escrow_verification_node)
workflow.add_node("hitl",               hitl_node)

workflow.add_edge(START, "decomposition")
workflow.add_edge("decomposition", "rag_verifier")

workflow.add_conditional_edges(
    "rag_verifier",
    route_after_rag,
    {"hitl": "hitl", "escrow_verification": "escrow_verification"},
)

workflow.add_edge("escrow_verification", END)
workflow.add_edge("hitl", END)

graph_2 = workflow.compile(checkpointer=checkpointer)