from typing import Dict, Any
from langgraph.types import interrupt

def hitl_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suspends execution to await explicit approval via the admin platform.
    """
    condition_reason = state.get("hitl_reason", "Action requires human authorization.")
    
    admin_response = interrupt({
        "type": "HITL_APPROVAL_REQUIRED",
        "reason": condition_reason,
        "pending_action": state.get("pending_action", {})
    })
    
    return {
        "approved": admin_response.get("approved", False),
        "admin_feedback": admin_response.get("feedback", ""),
        "status": "APPROVED" if admin_response.get("approved") else "REJECTED"
    }