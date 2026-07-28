"""
generate_cma — PROGRESS TRACKING concern (Issue #9).

Generates a Comparative Market Analysis for a property: finds comparable
properties (same city + type, within a price band) and computes summary
stats. This genuinely takes multiple DB round-trips + computation steps,
so it reports real intermediate progress instead of leaving the client
blocked with a single response at the end.

This file is written as a GENERATOR (yields progress steps) so server.py
can drive it and forward each yielded step to ctx.report_progress(...)
without this file needing to know anything about MCP itself.
"""

from db import get_connection


def generate_cma(property_id: int):
    """
    Yields a sequence of (progress, total, message) tuples as work
    proceeds, and finally yields ("done", result_dict) as the last item.
    server.py is responsible for translating each yield into a real
    progress notification sent to the client.
    """
    total_steps = 4

    yield (1, total_steps, "Loading subject property")
    with get_connection() as conn:
        subject = conn.execute(
            "SELECT * FROM Property WHERE property_id = ?", (property_id,)
        ).fetchone()
        if subject is None:
            raise ValueError(f"No property with id {property_id}")
        subject = dict(subject)

    yield (2, total_steps, "Searching comparable properties")
    with get_connection() as conn:
        # comparables: same city + type, excluding the subject itself
        comps = conn.execute(
            """SELECT * FROM Property
               WHERE city = ? AND property_type = ? AND property_id != ?""",
            (subject["city"], subject["property_type"], property_id),
        ).fetchall()
        comps = [dict(c) for c in comps]

    yield (3, total_steps, f"Computing statistics over {len(comps)} comparables")
    if comps:
        prices = [c["price"] for c in comps]
        avg_price = sum(prices) / len(prices)
        price_per_sqft = [
            c["price"] / c["area_sqft"] for c in comps if c["area_sqft"]
        ]
        avg_price_per_sqft = (
            sum(price_per_sqft) / len(price_per_sqft) if price_per_sqft else None
        )
    else:
        avg_price = None
        avg_price_per_sqft = None

    yield (4, total_steps, "Finalizing report")
    result = {
        "subject_property_id": property_id,
        "subject_price": subject["price"],
        "comparable_count": len(comps),
        "average_comparable_price": avg_price,
        "average_price_per_sqft": avg_price_per_sqft,
        "comparables": [
            {"property_id": c["property_id"], "title": c["title"],
             "price": c["price"]} for c in comps
        ],
    }

    yield ("done", result)