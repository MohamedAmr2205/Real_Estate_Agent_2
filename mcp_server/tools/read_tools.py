"""
Read-only tools. No elicitation, no authorization beyond "is connected" —
these are safe to expose even to a client without elicitation/sampling
support (see server.py capability fallback).
"""

from db import get_connection


def search_properties(city: str | None = None,
                       property_type: str | None = None,
                       status: str | None = None,
                       max_price: float | None = None) -> list[dict]:
    query = "SELECT * FROM Property WHERE 1=1"
    params: list = []

    if city:
        query += " AND city = ?"
        params.append(city)
    if property_type:
        query += " AND property_type = ?"
        params.append(property_type)
    if status:
        query += " AND status = ?"
        params.append(status)
    if max_price is not None:
        query += " AND price <= ?"
        params.append(max_price)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_property(property_id: int) -> dict:
    with get_connection() as conn:
        prop = conn.execute(
            "SELECT * FROM Property WHERE property_id = ?", (property_id,)
        ).fetchone()
        if prop is None:
            raise ValueError(f"No property with id {property_id}")

        docs = conn.execute(
            "SELECT * FROM Property_Document WHERE property_id = ?",
            (property_id,),
        ).fetchall()

        return {
            **dict(prop),
            "documents": [dict(d) for d in docs],
        }