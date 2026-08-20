from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from checkpoint import checkpointer
from hitl_node import hitl_node
from ticket_system import create_failure_ticket

class NegotiationState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    property_id: str
    offered_price: float
    original_price: float
    evaluated_strategies: List[Dict[str, Any]]
    chosen_strategy: str
    pending_action: Dict[str, Any]
    hitl_reason: str
    approved: bool

def tot_strategy_node(state: NegotiationState) -> Dict[str, Any]:
    """Tree of Thoughts (ToT): Evaluates multiple counter-offer paths."""
    offered = state.get("offered_price", 0)
    original = state.get("original_price", 1)
    
    strategies = [
        {"name": "Aggressive_Counter", "score": 0.6 if offered / original < 0.85 else 0.3},
        {"name": "Compromise_Split", "score": 0.9 if 0.85 <= offered / original <= 0.95 else 0.4},
        {"name": "Direct_Acceptance", "score": 0.95 if offered / original > 0.95 else 0.1}
    ]
    
    best_strategy = max(strategies, key=lambda x: x["score"])["name"]
    return {"evaluated_strategies": strategies, "chosen_strategy": best_strategy}

def constrained_react_node(state: NegotiationState) -> Dict[str, Any]:
    """Constrained ReAct: Executes action or triggers HITL if discount > 15%."""
    try:
        strategy = state.get("chosen_strategy")
        offered = state.get("offered_price", 0)
        original = state.get("original_price", 1)
        discount = (original - offered) / original
        
        if discount > 0.15:
            return {
                "hitl_reason": f"Discount rate ({discount * 100:.1f}%) exceeds auto-approval limit (15%).",
                "pending_action": {"action": "APPROVE_OFFER", "offered_price": offered, "strategy": strategy}
            }
        
        return {"status": "AUTO_ACCEPTED"}
    except Exception as e:
        create_failure_ticket("negotiation_thread", "graph_1_offer_negotiation", str(e), state)
        raise e

def route_after_hitl(state: NegotiationState):
    if state.get("approved"):
        return END
    return "tot_strategy"

workflow = StateGraph(NegotiationState)
workflow.add_node("tot_strategy", tot_strategy_node)
workflow.add_node("react_agent", constrained_react_node)
workflow.add_node("hitl", hitl_node)

workflow.add_edge(START, "tot_strategy")
workflow.add_edge("tot_strategy", "react_agent")
workflow.add_edge("react_agent", "hitl")
workflow.add_conditional_edges("hitl", route_after_hitl, {END: END, "tot_strategy": "tot_strategy"})

graph_1 = workflow.compile(checkpointer=checkpointer)