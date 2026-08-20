from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from checkpoint import checkpointer
from hitl_node import hitl_node

class PropertyListingState(TypedDict):
    property_details: Dict[str, Any]
    lats_evaluated_trees: List[Dict[str, Any]]
    selected_listing_plan: Dict[str, Any]
    pending_action: Dict[str, Any]
    hitl_reason: str
    approved: bool

def lats_pricing_evaluator_node(state: PropertyListingState) -> Dict[str, Any]:
    """LATS: Tree search evaluation over pricing and strategy branches."""
    plans = [
        {"price": 2500000, "marketing": "Social_Ads", "score": 0.88},
        {"price": 2800000, "marketing": "Exclusive_Listing", "score": 0.94}
    ]
    best_plan = max(plans, key=lambda x: x["score"])
    return {"lats_evaluated_trees": plans, "selected_listing_plan": best_plan}

def constrained_react_listing_node(state: PropertyListingState) -> Dict[str, Any]:
    """Constrained ReAct: Prepares listing and routes for owner signoff."""
    plan = state.get("selected_listing_plan", {})
    return {
        "hitl_reason": f"New property listing at {plan.get('price')} EGP requires Owner Signoff.",
        "pending_action": {"action": "PUBLISH_LISTING", "plan": plan}
    }

workflow = StateGraph(PropertyListingState)
workflow.add_node("lats_evaluator", lats_pricing_evaluator_node)
workflow.add_node("react_listing", constrained_react_listing_node)
workflow.add_node("hitl", hitl_node)

workflow.add_edge(START, "lats_evaluator")
workflow.add_edge("lats_evaluator", "react_listing")
workflow.add_edge("react_listing", "hitl")
workflow.add_edge("hitl", END)

graph_3 = workflow.compile(checkpointer=checkpointer)