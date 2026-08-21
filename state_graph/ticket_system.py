"""
state_graph/ticket_system.py
==============================
Failure ticket system مع:
- Status tracking: OPEN → INVESTIGATING → RESOLVED
- Checkpoint resume بعد الـ resolve
- منفصل تماماً عن الـ HITL path

الفرق بين HITL والـ Ticket:
- HITL: توقف متوقع — الـ agent مش مسموحله يقرر لوحده
- Ticket: فشل غير متوقع — tool error, schema fail, model error
"""

from __future__ import annotations

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

STATE_GRAPH_DIR = Path(__file__).resolve().parent
DB_PATH = STATE_GRAPH_DIR / "checkpoints.sqlite"


# ----------------------------------------------------------------
# DB setup
# ----------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id   TEXT NOT NULL,
            graph_name  TEXT NOT NULL,
            node_name   TEXT NOT NULL DEFAULT 'unknown',
            error_type  TEXT NOT NULL DEFAULT 'RUNTIME_ERROR',
            error_message TEXT NOT NULL,
            state_snapshot TEXT,
            status      TEXT NOT NULL DEFAULT 'OPEN',
            resolution  TEXT,
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now')),
            resolved_at TEXT
        )
    """)
    conn.commit()
    conn.close()


_ensure_table()


# ----------------------------------------------------------------
# Create ticket
# ----------------------------------------------------------------

def create_failure_ticket(
    thread_id: str,
    graph_name: str,
    error_msg: str,
    current_state: dict[str, Any],
    node_name: str = "unknown",
    error_type: str = "RUNTIME_ERROR",
) -> str:
    """
    Create a new failure ticket.
    - Saves full state snapshot for resume
    - Returns ticket_id as string
    """
    conn = _get_conn()

    state_json = json.dumps(current_state, default=str)

    cursor = conn.execute(
        """
        INSERT INTO tickets
            (thread_id, graph_name, node_name, error_type,
             error_message, state_snapshot, status)
        VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
        """,
        (thread_id, graph_name, node_name, error_type,
         str(error_msg), state_json),
    )
    conn.commit()
    ticket_id = str(cursor.lastrowid)
    conn.close()

    print(f"[TICKET] ❌ Created ticket #{ticket_id} | "
          f"graph={graph_name} node={node_name} | {error_msg[:80]}")
    return ticket_id


# ----------------------------------------------------------------
# Status transitions
# ----------------------------------------------------------------

def update_ticket_status(
    ticket_id: str,
    new_status: str,
    resolution: str | None = None,
) -> bool:
    """
    Transition ticket status.
    Valid: OPEN → INVESTIGATING → RESOLVED

    Returns True if update succeeded.
    """
    valid_transitions = {
        "OPEN": ["INVESTIGATING", "RESOLVED"],
        "INVESTIGATING": ["RESOLVED"],
        "RESOLVED": [],
    }

    conn = _get_conn()
    row = conn.execute(
        "SELECT status FROM tickets WHERE ticket_id = ?",
        (ticket_id,)
    ).fetchone()

    if not row:
        print(f"[TICKET] Ticket #{ticket_id} not found")
        conn.close()
        return False

    current = row["status"]
    if new_status not in valid_transitions.get(current, []):
        print(f"[TICKET] Invalid transition: {current} → {new_status}")
        conn.close()
        return False

    resolved_at = datetime.utcnow().isoformat() if new_status == "RESOLVED" else None

    conn.execute(
        """
        UPDATE tickets
        SET status = ?, resolution = ?, updated_at = datetime('now'),
            resolved_at = ?
        WHERE ticket_id = ?
        """,
        (new_status, resolution, resolved_at, ticket_id),
    )
    conn.commit()
    conn.close()

    print(f"[TICKET] #{ticket_id} → {new_status}"
          + (f" | {resolution}" if resolution else ""))
    return True


# ----------------------------------------------------------------
# Resume from checkpoint
# ----------------------------------------------------------------

def get_ticket(ticket_id: str) -> dict[str, Any] | None:
    """Get full ticket including state snapshot."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    result = dict(row)
    if result.get("state_snapshot"):
        try:
            result["state_snapshot"] = json.loads(result["state_snapshot"])
        except Exception:
            pass
    return result


def resume_from_ticket(
    ticket_id: str,
    graph,
    resolution_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resume a graph run from its checkpointed state after ticket resolution.

    1. Load ticket + state snapshot
    2. Mark ticket RESOLVED
    3. Resume graph from last checkpoint (LangGraph handles this)
    4. Return final state

    NOTE: LangGraph's SqliteSaver already has the checkpoint —
    we just need the thread_id to resume from it.
    """
    ticket = get_ticket(ticket_id)
    if not ticket:
        return {"error": f"Ticket #{ticket_id} not found"}

    if ticket["status"] == "RESOLVED":
        return {"error": f"Ticket #{ticket_id} already resolved"}

    # Mark resolved
    update_ticket_status(
        ticket_id,
        "RESOLVED",
        resolution=f"Manually resolved: {resolution_input or 'admin action'}",
    )

    thread_id = ticket["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}

    print(f"[TICKET] Resuming thread={thread_id} from checkpoint...")

    # Resume — LangGraph picks up from last saved checkpoint
    try:
        resume_value = resolution_input or {}
        final_state = None
        for event in graph.stream(resume_value, config):
            print(f"[RESUME] event: {list(event.keys())}")
            final_state = event

        print(f"[TICKET] ✅ Thread {thread_id} resumed successfully")
        return {"resumed": True, "thread_id": thread_id, "final_event": final_state}
    except Exception as e:
        print(f"[TICKET] ❌ Resume failed: {e}")
        return {"resumed": False, "error": str(e)}


# ----------------------------------------------------------------
# List tickets (for admin platform)
# ----------------------------------------------------------------

def list_tickets(
    status: str | None = None,
    graph_name: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List tickets for admin platform display."""
    conn = _get_conn()
    query = "SELECT * FROM tickets WHERE 1=1"
    params: list[Any] = []

    if status:
        query += " AND status = ?"
        params.append(status)
    if graph_name:
        query += " AND graph_name = ?"
        params.append(graph_name)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    tickets = []
    for row in rows:
        t = dict(row)
        if t.get("state_snapshot"):
            try:
                t["state_snapshot"] = json.loads(t["state_snapshot"])
            except Exception:
                pass
        tickets.append(t)
    return tickets


def get_ticket_summary() -> dict[str, int]:
    """Count tickets by status — for admin dashboard."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM tickets GROUP BY status"
    ).fetchall()
    conn.close()
    return {row["status"]: row["cnt"] for row in rows}