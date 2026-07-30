"""
explain_offer_risk — SAMPLING concern (Issue #10).

Important: this file does NOT call any LLM itself. It only gathers the
facts an LLM would need to reason about risk, and formats them as a
prompt. The actual sampling/createMessage call — which asks the CLIENT's
model to do the reasoning, not a model owned by the server — happens in
server.py, using ctx.session.create_message(...).

Keeping the server "dumb" here is deliberate: a server that ran its own
LLM call to judge risk would defeat the purpose of sampling, which is to
borrow the human's already-trusted model instead of the server silently
having its own opinion.
"""

from db import get_connection


def build_risk_prompt(offer_id: int) -> dict:
    """
    Gathers the facts about an offer and its property, and returns a
    ready-to-send sampling prompt plus the raw facts (so server.py can
    include the facts in the tool result alongside the model's analysis).
    """
    with get_connection() as conn:
        offer = conn.execute(
            "SELECT * FROM Offer WHERE offer_id = ?", (offer_id,)
        ).fetchone()
        if offer is None:
            raise ValueError(f"No offer with id {offer_id}")
        offer = dict(offer)

        property_row = conn.execute(
            "SELECT * FROM Property WHERE property_id = ?",
            (offer["property_id"],),
        ).fetchone()
        property_row = dict(property_row)

    ratio = offer["offer_amount"] / property_row["price"]

    facts = {
        "offer_id": offer_id,
        "offer_amount": offer["offer_amount"],
        "list_price": property_row["price"],
        "offer_to_list_ratio": round(ratio, 3),
        "property_status": property_row["status"],
        "property_title": property_row["title"],
    }

    prompt_text = (
        f"An offer of {offer['offer_amount']} was made on a property "
        f"listed at {property_row['price']} "
        f"(offer is {round(ratio * 100, 1)}% of list price). "
        f"Property status: {property_row['status']}. "
        "In 2-3 sentences, explain the practical risk level of this offer "
        "to a listing agent deciding whether to recommend accepting it, "
        "countering it, or rejecting it. Be concrete, not generic."
    )

    return {"facts": facts, "prompt_text": prompt_text}