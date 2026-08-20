from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from checkpoint import checkpointer
from hitl_node import hitl_node

class DealClosingState(TypedDict):
    deal_id: str
    sub_tasks: List[str]
    retrieved_contract_rules: List[str]
    completed_tasks: List[str]
    pending_action: Dict[str, Any]
    hitl_reason: str
    approved: bool

def task_decomposition_node(state: DealClosingState) -> Dict[str, Any]:
    """Task Decomposition: Splits closing sequence into distinct sub-tasks."""
    sub_tasks = ["RAG_Contract_Check", "Escrow_Verification", "Broker_Signoff"]
    return {"sub_tasks": sub_tasks, "completed_tasks": []}

def rag_contract_verifier_node(state: DealClosingState) -> Dict[str, Any]:
    """RAG Architecture: Retrieves escrow policies and legal rules."""
    rules = ["Escrow requires 10% deposit", "Broker signoff mandatory for deals > 1M"]
    
    return {
        "retrieved_contract_rules": rules,
        "hitl_reason": "Escrow verification requires mandatory Broker approval.",
        "pending_action": {"action": "BROKER_FINAL_SIGNATURE", "rules": rules}
    }

workflow = StateGraph(DealClosingState)
workflow.add_node("decomposition", task_decomposition_node)
workflow.add_node("rag_verifier", rag_contract_verifier_node)
workflow.add_node("hitl", hitl_node)

workflow.add_edge(START, "decomposition")
workflow.add_edge("decomposition", "rag_verifier")
workflow.add_edge("rag_verifier", "hitl")
workflow.add_edge("hitl", END)

graph_2 = workflow.compile(checkpointer=checkpointer)