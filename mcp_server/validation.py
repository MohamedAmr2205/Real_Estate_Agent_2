"""
Server-side validation, independent of JSON Schema.

The JSON Schema on each tool (see tools/*.py `input_schema`) only guarantees
TYPES and SHAPE (e.g. "offer_amount is a number"). It cannot know whether
that number makes business sense, or whether the caller is actually allowed
to do this. That's what this module is for.

Used directly by: submit_offer, accept_offer, assign_listing_agent.
"""

from dataclasses import dataclass


class ValidationError(Exception):
    """Raised when a request is well-typed but still not acceptable."""


@dataclass
class Property:
    property_id: int
    price: float
    status: str
    agent_id: int


@dataclass
class Agent:
    agent_id: int
    role: str


# ---- business-rule checks (beyond "is this an int?") -----------------

def validate_offer_amount(amount: float, property_row: Property) -> None:
    if amount <= 0:
        raise ValidationError("offer_amount must be a positive number.")
    if property_row.status != "Available":
        raise ValidationError(
            f"Property {property_row.property_id} is not available for "
            f"offers (status={property_row.status})."
        )
    if amount > property_row.price * 1.5:
        # sanity ceiling — catches fat-finger / bad-data submissions
        raise ValidationError(
            "offer_amount is unrealistically high relative to list price."
        )


def offer_is_high_risk(amount: float, list_price: float) -> bool:
    """
    True when the offer needs a human (broker) sign-off before it can be
    submitted. This is the trigger condition for elicitation/create in
    tools/submit_offer.py.
    """
    RISK_THRESHOLD = 0.85  # offers below 85% of list price are "risky"
    return amount < list_price * RISK_THRESHOLD


# ---- authorization checks (independent of "is the ID an integer") ----

def assert_can_accept_offer(caller: Agent, property_row: Property) -> None:
    """
    Only the listing agent for the property, or someone with role=Broker,
    is allowed to accept an offer. This is business logic that has
    to run in the handler — a JSON Schema has no way to express
    "must equal the property's listing_agent_id at query time".
    """
    is_listing_agent = caller.agent_id == property_row.agent_id
    is_broker = caller.role == "Broker"
    if not (is_listing_agent or is_broker):
        raise ValidationError(
            f"Agent {caller.agent_id} is not authorized to accept offers "
            f"on property {property_row.property_id}: not the listing "
            f"agent and not a Broker."
        )


def assert_can_reassign_listing(caller: Agent) -> None:
    """Only a Broker may reassign which agent lists a property."""
    if caller.role != "Broker":
        raise ValidationError(
            f"Agent {caller.agent_id} is not authorized to reassign "
            f"listing agents: role={caller.role}, requires role=Broker."
        )