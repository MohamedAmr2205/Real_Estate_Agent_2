"""
submit_offer — WRITE tool, ELICITATION concern.

Trigger condition (see validation.offer_is_high_risk): an offer below 85%
of list price is risky enough that a broker must explicitly sign off
before it's recorded as 'Submitted'. The actual elicitation/create call
happens in server.py (it needs the MCP `ctx` object) — this file only
holds the business logic: is this risky, and how do we persist the result.
"""

from db import get_connection
from validation import Property, validate_offer_amount, offer_is_high_risk


def load_property_for_offer(property_id: int) -> Property:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM Property WHERE property_id = ?", (property_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No property with id {property_id}")
        return Property(
            property_id=row["property_id"],
            price=row["price"],
            status=row["status"],
            agent_id=row["agent_id"],
        )


def check_offer(property_id: int, offer_amount: float) -> tuple[Property, bool]:
    """
    Runs schema-independent validation and returns whether this offer
    needs elicitation. Raises ValidationError if the offer is invalid
    outright (server.py should surface that as a tool error, not silently
    proceed).
    """
    prop = load_property_for_offer(property_id)
    validate_offer_amount(offer_amount, prop)
    is_risky = offer_is_high_risk(offer_amount, prop.price)
    return prop, is_risky


def insert_offer(property_id: int, customer_id: int, offer_amount: float,
                  status: str = "Pending") -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO Offer (property_id, customer_id, offer_amount,
                                   offer_date, status)
               VALUES (?, ?, ?, date('now'), ?)""",
            (property_id, customer_id, offer_amount, status),
        )
        conn.commit()
        return cur.lastrowid