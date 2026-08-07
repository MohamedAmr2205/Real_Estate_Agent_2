"""
Reusable prompt templates (PROMPTS concern, Issue #7).

These are discoverable via prompts/list and retrievable via prompts/get
in server.py. Each function here returns the actual templated text after
pulling the real record's data — the host surfaces this as a canned,
parameterized starting point instead of every client re-writing it.
"""

from db import get_connection


def draft_listing_description_prompt(property_id: int) -> str:
    with get_connection() as conn:
        prop = conn.execute(
            "SELECT * FROM Property WHERE property_id = ?", (property_id,)
        ).fetchone()
        if prop is None:
            raise ValueError(f"No property with id {property_id}")
        prop = dict(prop)

    return (
        f"Before writing anything, read the resource at policy://fair-housing "
        f"and follow its rules strictly.\n\n"
        f"Draft a compelling listing description for this property:\n"
        f"- Title: {prop['title']}\n"
        f"- Type: {prop['property_type']} in {prop['city']}\n"
        f"- Bedrooms: {prop['bedrooms']}, Bathrooms: {prop['bathrooms']}\n"
        f"- Area: {prop['area_sqft']} sqft\n"
        f"- Price: {prop['price']}\n\n"
        f"Describe the property itself only — never speculate about who "
        f"should live there."
    )


def draft_offer_rejection_prompt(offer_id: int) -> str:
    with get_connection() as conn:
        offer = conn.execute(
            "SELECT * FROM Offer WHERE offer_id = ?", (offer_id,)
        ).fetchone()
        if offer is None:
            raise ValueError(f"No offer with id {offer_id}")
        offer = dict(offer)

        prop = conn.execute(
            "SELECT * FROM Property WHERE property_id = ?",
            (offer["property_id"],),
        ).fetchone()
        prop = dict(prop)

    return (
        f"Draft a professional, courteous rejection message for this offer:\n"
        f"- Offer amount: {offer['offer_amount']}\n"
        f"- Property: {prop['title']} (listed at {prop['price']})\n\n"
        f"Keep it brief, thank the buyer, and leave the door open for a "
        f"revised offer without committing to any specific counter amount."
    )