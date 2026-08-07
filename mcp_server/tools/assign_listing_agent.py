"""
assign_listing_agent — WRITE tool, NOTIFICATIONS concern (Issue #4).

Only a Broker may reassign which agent lists a property (see
validation.assert_can_reassign_listing). The interesting part isn't the
UPDATE statement itself — it's what happens right after: the server must
push notifications/tools/list_changed so the newly-assigned agent's
session immediately gains tools like accept_offer for that property,
without reconnecting.

The actual notification SEND happens in server.py (it needs the MCP
`ctx`/session object) — this file holds the business logic and returns
enough information for server.py to know a notification is warranted.
"""

from db import get_connection
from validation import Agent, assert_can_reassign_listing


def load_agent(agent_id: int) -> Agent:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM Agent WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No agent with id {agent_id}")
        return Agent(agent_id=row["agent_id"], role=row["role"])


def assign_listing_agent(property_id: int, new_agent_id: int,
                          caller_agent_id: int) -> dict:
    """
    Reassigns Property.agent_id to new_agent_id. Returns the old and new
    agent ids so server.py can decide who needs a tools/list_changed
    notification (the newly assigned agent gains property-specific tools;
    the previous listing agent loses them).
    """
    caller = load_agent(caller_agent_id)
    assert_can_reassign_listing(caller)  # only a Broker may do this

    with get_connection() as conn:
        prop = conn.execute(
            "SELECT * FROM Property WHERE property_id = ?", (property_id,)
        ).fetchone()
        if prop is None:
            raise ValueError(f"No property with id {property_id}")

        old_agent_id = prop["agent_id"]

        # confirm the new agent actually exists before assigning
        new_agent = conn.execute(
            "SELECT agent_id FROM Agent WHERE agent_id = ?", (new_agent_id,)
        ).fetchone()
        if new_agent is None:
            raise ValueError(f"No agent with id {new_agent_id}")

        conn.execute(
            "UPDATE Property SET agent_id = ? WHERE property_id = ?",
            (new_agent_id, property_id),
        )
        conn.commit()

    return {
        "property_id": property_id,
        "old_agent_id": old_agent_id,
        "new_agent_id": new_agent_id,
        "notify_agent_ids": [old_agent_id, new_agent_id],
    }