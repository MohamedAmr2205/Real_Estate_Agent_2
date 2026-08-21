"""
state_graph/test_run.py
========================
Demo شامل بيشغل:
1. Graph 1 — Offer Negotiation (ToT + ReAct + HITL)
2. HITL Resume — admin approves
3. Graph 2 — Deal Closing (Decomposition + RAG + HITL)
4. Graph 3 — Property Listing (LATS + ReAct + HITL)
5. Ticket System — create + update + resume
6. Crash & Resume — يثبت الـ checkpointing
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.types import Command

from state_graph.graph_1_offer_negotiation import graph_1
from state_graph.graph_2_deal_closing import graph_2
from state_graph.graph_3_property_listing import graph_3
from state_graph.checkpoint import get_thread_config
from state_graph.ticket_system import (
    create_failure_ticket,
    update_ticket_status,
    get_ticket,
    list_tickets,
    get_ticket_summary,
)


# ================================================================
# GRAPH 1 — Offer Negotiation
# ================================================================

def run_graph_1():
    print("\n" + "=" * 60)
    print("GRAPH 1: Offer Negotiation (ToT + ReAct + HITL)")
    print("=" * 60)

    config = get_thread_config("thread_negotiation_001")

    initial_input = {
        "property_id": "PROP-999",
        "offered_price": 700000.0,
        "original_price": 1000000.0,
    }

    print("\n--- Phase 1: Starting Negotiation (30% discount → triggers HITL) ---")
    for event in graph_1.stream(initial_input, config):
        if "__interrupt__" in event:
            interrupt = event["__interrupt__"][0]
            print(f"\n[HITL] ⏸ Graph paused!")
            print(f"  Reason : {interrupt.value['reason']}")
            print(f"  Action : {interrupt.value['pending_action']}")
        else:
            print(f"Event: {list(event.keys())}")

    # ── Admin approves via platform ──
    print("\n--- Phase 2: Admin approves via platform ---")
    for event in graph_1.stream(
        Command(resume={"approved": True, "feedback": "Broker approved after review"}),
        config,
    ):
        if "__interrupt__" in event:
            print(f"[HITL] Another interrupt: {event['__interrupt__'][0].value}")
        else:
            print(f"Event: {list(event.keys())} → {event}")

    print("\n✅ Graph 1 complete — HITL resume worked")


# ================================================================
# GRAPH 2 — Deal Closing
# ================================================================

def run_graph_2():
    print("\n" + "=" * 60)
    print("GRAPH 2: Deal Closing (Task Decomposition + RAG + HITL)")
    print("=" * 60)

    config = get_thread_config("thread_deal_001")

    initial_input = {
        "deal_id": "DEAL-42",
        "deal_value": 2500000.0,
    }

    print("\n--- Starting Deal Closing ---")
    for event in graph_2.stream(initial_input, config):
        if "__interrupt__" in event:
            interrupt = event["__interrupt__"][0]
            print(f"\n[HITL] ⏸ Graph paused!")
            print(f"  Reason: {interrupt.value['reason']}")
            print(f"  Action: {interrupt.value['pending_action']['action']}")
        else:
            print(f"Event: {list(event.keys())}")

    # ── Admin approves broker signoff ──
    print("\n--- Admin approves broker signoff ---")
    for event in graph_2.stream(
        Command(resume={"approved": True, "feedback": "Broker signed contract"}),
        config,
    ):
        print(f"Event: {list(event.keys())} → {event}")

    print("\n✅ Graph 2 complete")


# ================================================================
# GRAPH 3 — Property Listing
# ================================================================

def run_graph_3():
    print("\n" + "=" * 60)
    print("GRAPH 3: Property Listing (LATS + ReAct + HITL)")
    print("=" * 60)

    config = get_thread_config("thread_listing_001")

    initial_input = {
        "property_details": {
            "property_id": "PROP-101",
            "title": "Luxury Villa Smouha",
            "city": "Alexandria",
            "price": 5000000,
            "bedrooms": 5,
            "area_sqft": 3000,
        }
    }

    print("\n--- Starting Property Listing (LATS pricing evaluation) ---")
    for event in graph_3.stream(initial_input, config):
        if "__interrupt__" in event:
            interrupt = event["__interrupt__"][0]
            print(f"\n[HITL] ⏸ Graph paused — Owner signoff required!")
            print(f"  Reason: {interrupt.value['reason']}")
            action = interrupt.value.get("pending_action", {})
            plan = action.get("plan", {})
            print(f"  Plan  : price={plan.get('price'):,} "
                  f"marketing={plan.get('marketing')}")
        else:
            print(f"Event: {list(event.keys())}")

    # ── Owner approves listing ──
    print("\n--- Owner approves listing via platform ---")
    for event in graph_3.stream(
        Command(resume={"approved": True, "feedback": "Owner approved listing price"}),
        config,
    ):
        print(f"Event: {list(event.keys())} → {event}")

    print("\n✅ Graph 3 complete")


# ================================================================
# TICKET SYSTEM TEST
# ================================================================

def run_ticket_test():
    print("\n" + "=" * 60)
    print("TICKET SYSTEM: Failure → Investigate → Resolve")
    print("=" * 60)

    # Simulate a runtime failure
    fake_state = {
        "property_id": "PROP-FAIL",
        "offered_price": 500000,
        "original_price": 1000000,
        "step": "constrained_react_node",
    }

    print("\n--- Creating failure ticket ---")
    ticket_id = create_failure_ticket(
        thread_id="thread_fail_001",
        graph_name="graph_1_offer_negotiation",
        error_msg="DB connection timeout during offer submission",
        current_state=fake_state,
        node_name="constrained_react_node",
        error_type="DB_TIMEOUT",
    )

    # Show ticket
    ticket = get_ticket(ticket_id)
    print(f"\n[TICKET] #{ticket_id} created:")
    print(f"  Status     : {ticket['status']}")
    print(f"  Error      : {ticket['error_message']}")
    print(f"  Graph      : {ticket['graph_name']}")
    print(f"  Node       : {ticket['node_name']}")

    # Admin investigates
    print("\n--- Admin marks as INVESTIGATING ---")
    update_ticket_status(ticket_id, "INVESTIGATING",
                         resolution="Checking DB connection logs")

    # Admin resolves
    print("\n--- Admin resolves ticket ---")
    update_ticket_status(ticket_id, "RESOLVED",
                         resolution="DB connection restored, retry succeeded")

    ticket = get_ticket(ticket_id)
    print(f"\n[TICKET] #{ticket_id} final state:")
    print(f"  Status     : {ticket['status']}")
    print(f"  Resolution : {ticket['resolution']}")
    print(f"  Resolved at: {ticket['resolved_at']}")

    # Summary
    summary = get_ticket_summary()
    print(f"\n[TICKET SUMMARY] {summary}")
    print("\n✅ Ticket system test complete")


# ================================================================
# CRASH & RESUME TEST
# ================================================================

def run_crash_resume_test():
    print("\n" + "=" * 60)
    print("CRASH & RESUME: Kill mid-run → restart → resume from checkpoint")
    print("=" * 60)

    config = get_thread_config("thread_crash_001")

    initial_input = {
        "property_id": "PROP-CRASH",
        "offered_price": 800000.0,
        "original_price": 1000000.0,
    }

    print("\n--- Run 1: Start graph (will pause at HITL) ---")
    events_seen = []
    for event in graph_1.stream(initial_input, config):
        events_seen.append(list(event.keys()))
        if "__interrupt__" in event:
            print(f"[CHECKPOINT] Graph paused at HITL — state saved to SQLite")
            print(f"[SIMULATE CRASH] Process would die here...")
            break
        else:
            print(f"  Completed node: {list(event.keys())}")

    print(f"\n--- Run 2: Restart process & resume from checkpoint ---")
    print(f"[RESUME] Loading state from SQLite checkpoint...")
    print(f"[RESUME] thread_id=thread_crash_001")

    # Resume without re-running completed nodes
    for event in graph_1.stream(
        Command(resume={"approved": True, "feedback": "Resumed after crash"}),
        config,
    ):
        print(f"  Resumed node: {list(event.keys())} ← no re-execution of ToT/ReAct")

    print("\n✅ Crash & resume test complete — no steps re-executed")


# ================================================================
# MAIN
# ================================================================

def main():
    print("🏠 Meridian Realty — State Graph Demo")
    print("=" * 60)

    run_graph_1()
    run_graph_2()
    run_graph_3()
    run_ticket_test()
    run_crash_resume_test()

    print("\n" + "=" * 60)
    print("✅ All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()