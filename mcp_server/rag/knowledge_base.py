"""
search_knowledge_base — Add-On Lab, Option A (RAG over unstructured data)

DOMAIN CHOICE: internal property history notes.

Our structured tables (Property, Offer, Contract...) capture facts a
form could hold — price, status, dates. They do NOT capture the kind of
free-text observations an agent actually writes during a walkthrough or
a negotiation: "roof shows wear," "seller hinted at a lower floor
price," "buyer flagged the noise from the street." That's exactly the
document-shaped content this tool searches, instead of an agent (or a
model) reading every note on every property to find the one that
matters.

GENUINE STAKES (why this needed handler-level authorization, not just a
schema): some of these notes are commercially sensitive — a seller's
confidential floor price should be visible to a Broker, but leaking it
to a buyer's agent (or the wrong listing agent) would undermine the
seller's negotiating position. So, consistent with the rest of this
server, the role check happens in the handler, not the schema.
"""

from db import get_connection
from keyword_search import KeywordStore

knowledge_store = KeywordStore()


# ---------------------------------------------------------------------
# Seed content — in-memory is fine per the lab spec. In a real system
# these notes would come from agent write-ups after a showing, a
# walkthrough, or a negotiation call.
# ---------------------------------------------------------------------
PROPERTY_NOTES = [
    {
        "text": "Walkthrough 2026-06-02: roof shows minor wear on the south "
                "side, recommend a professional inspection before closing. "
                "No active leaks observed.",
        "property_id": 1,
        "role_required": "any",
    },
    {
        "text": "Negotiation call 2026-07-18: seller has privately indicated "
                "they would accept as low as 4,500,000 if a clean, "
                "contingency-free offer comes in. Do not disclose this "
                "figure to the buyer's agent.",
        "property_id": 1,
        "role_required": "Broker",
    },
    {
        "text": "Showing feedback 2026-07-15: buyer's agent noted street "
                "noise from the ground-floor unit facing Mostafa Kamel "
                "Street; buyer is still interested but price-sensitive.",
        "property_id": 2,
        "role_required": "any",
    },
    {
        "text": "Internal note 2026-06-20: tenant currently occupying unit "
                "2, lease ends in three months — factor into any offer "
                "timeline discussions.",
        "property_id": 2,
        "role_required": "any",
    },
    {
        "text": "Walkthrough 2026-05-10: elevator serviced last quarter, "
                "no outstanding maintenance issues in common areas.",
        "property_id": 3,
        "role_required": "any",
    },
    {
        "text": "Negotiation call 2026-07-01: seller open to covering "
                "closing costs if the deal closes before end of quarter — "
                "not yet reflected in the listed price.",
        "property_id": 3,
        "role_required": "Broker",
    },
    {
        "text": "Showing feedback 2026-07-22: buyer liked the layout but "
                "flagged the small kitchen; considering a renovation "
                "budget in any offer.",
        "property_id": 4,
        "role_required": "any",
    },
    {
        "text": "Walkthrough 2026-04-18: property withdrawn from market "
                "pending owner's decision on a full renovation before "
                "relisting; roof and foundation both in good condition.",
        "property_id": 5,
        "role_required": "any",
    },
]


def index_property_notes() -> None:
    """Load the in-memory notes into the keyword store. Called once at
    server startup (see server.py)."""
    for note in PROPERTY_NOTES:
        knowledge_store.upsert(
            payload=note["text"],
            metadata={"property_id": note["property_id"],
                      "role_required": note["role_required"]},
        )


# ---------------------------------------------------------------------
# The handler: called from server.py's @mcp.tool() wrapper, same
# pattern as tools/accept_offer.py — business logic + authorization
# live here, not inline in server.py.
# ---------------------------------------------------------------------
def load_agent_role(agent_id: int) -> str:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT role FROM Agent WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No agent with id {agent_id}")
        return row["role"]


def search_knowledge_base(query: str, property_id: int, caller_agent_id: int,
                           top_k: int = 3) -> dict:
    """
    Search property history notes scoped to one property. Notes tagged
    role_required='Broker' are filtered out in the handler for any
    caller who isn't a Broker — never trust a role passed in `query`
    or inferred from anything other than the database.
    """
    caller_role = load_agent_role(caller_agent_id)

    matches = knowledge_store.query(
        query_text=query, top_k=top_k * 2,  # over-fetch, then filter by role
        filter={"property_id": property_id},
    )

    visible = [
        m for m in matches
        if m["metadata"]["role_required"] in ("any", caller_role)
    ][:top_k]

    hidden_count = len(matches) - len(visible)

    if not visible:
        return {
            "results": [],
            "note": "No relevant notes visible to your role for this property."
                    + (" (Some matches exist but require Broker access.)"
                       if hidden_count else ""),
        }

    return {
        "results": [m["payload"] for m in visible],
        "note": f"{hidden_count} additional matching note(s) require Broker "
                f"access and were withheld." if hidden_count else "",
    }