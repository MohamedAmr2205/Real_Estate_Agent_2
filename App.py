"""
platform/app.py
================
Meridian Realty Platform — Flask backend
Admin + User interface wired to live MCP server and state graphs.
"""

from __future__ import annotations

import json
import sys
import os
import sqlite3
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Load .env ──
def _load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_load_env()

from state_graph.ticket_system import (
    list_tickets, update_ticket_status, get_ticket,
    get_ticket_summary
)

app = Flask(__name__)

STATE_GRAPH_DB = Path(__file__).resolve().parent / "db" / "database.sqlite"
MCP_TOOLS_DB   = ROOT / "db" / "database.sqlite"
RAG_DOCS_FILE  = ROOT / "platform" / "rag_docs.json"

# ── RAG docs store (simple JSON file) ──
def _load_rag_docs() -> list[dict]:
    if RAG_DOCS_FILE.exists():
        return json.loads(RAG_DOCS_FILE.read_text())
    return []

def _save_rag_docs(docs: list[dict]):
    RAG_DOCS_FILE.write_text(json.dumps(docs, indent=2))

# ── MCP tools store (SQLite) ──
def _get_tools_conn():
    conn = sqlite3.connect(str(STATE_GRAPH_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_tools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            added_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn

def _get_default_tools() -> list[dict]:
    return [
        {"agent": "memory_rag_agent",    "tool": "search_properties",    "enabled": True},
        {"agent": "memory_rag_agent",    "tool": "get_property",          "enabled": True},
        {"agent": "memory_rag_agent",    "tool": "search_knowledge_base", "enabled": True},
        {"agent": "memory_rag_agent",    "tool": "generate_cma",          "enabled": True},
        {"agent": "planning_agent",      "tool": "search_properties",     "enabled": True},
        {"agent": "planning_agent",      "tool": "submit_offer",          "enabled": True},
        {"agent": "planning_agent",      "tool": "accept_offer",          "enabled": True},
        {"agent": "state_graph_agent",   "tool": "submit_offer",          "enabled": True},
        {"agent": "state_graph_agent",   "tool": "assign_listing_agent",  "enabled": True},
        {"agent": "state_graph_agent",   "tool": "explain_offer_risk",    "enabled": True},
    ]

def _ensure_tools():
    conn = _get_tools_conn()
    count = conn.execute("SELECT COUNT(*) FROM agent_tools").fetchone()[0]
    if count == 0:
        for t in _get_default_tools():
            conn.execute(
                "INSERT INTO agent_tools (agent_name, tool_name, enabled) VALUES (?,?,?)",
                (t["agent"], t["tool"], 1 if t["enabled"] else 0)
            )
        conn.commit()
    conn.close()

_ensure_tools()


# ================================================================
# ROUTES — User Side
# ================================================================

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.json
    message = data.get("message", "")
    agent   = data.get("agent", "memory_rag_agent")

    agent_prompts = {
    "memory_rag_agent":  "You are a real estate memory and RAG agent for Cornerstone Realty in Alexandria Egypt. Help users find properties, answer questions about listings, and recall past conversations.",
    "planning_agent":    "You are a real estate planning agent for Cornerstone Realty. Help users plan property purchases, decompose complex requests into steps, and coordinate deals.",
    "state_graph_agent": "You are a real estate workflow agent for Cornerstone Realty. Handle offer negotiations, deal closings, and property listings step by step.",
    }
    system_prompt = agent_prompts.get(agent, "You are a helpful real estate assistant.")

    try:
        import openai, os
        client = openai.OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": message},
            ],
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        reply = f"Agent error: {e}"

    return jsonify({"reply": reply, "agent": agent})


# ================================================================
# ROUTES — Admin Side
# ================================================================

@app.route("/admin")
def admin():
    summary = get_ticket_summary()
    tickets = list_tickets(limit=5)
    hitl    = list_tickets(status="OPEN", limit=10)
    return render_template("admin.html",
                           summary=summary,
                           tickets=tickets,
                           hitl_tasks=hitl)


# ── Tools Management ──

@app.route("/admin/tools")
def admin_tools():
    conn = _get_tools_conn()
    rows = conn.execute(
        "SELECT * FROM agent_tools ORDER BY agent_name, tool_name"
    ).fetchall()
    conn.close()
    tools = [dict(r) for r in rows]
    agents = sorted(set(t["agent_name"] for t in tools))
    return render_template("tools.html", tools=tools, agents=agents)


@app.route("/admin/tools/toggle", methods=["POST"])
def toggle_tool():
    data    = request.json
    tool_id = data.get("id")
    enabled = data.get("enabled", True)
    conn = _get_tools_conn()
    conn.execute(
        "UPDATE agent_tools SET enabled=? WHERE id=?",
        (1 if enabled else 0, tool_id)
    )
    conn.commit()
    conn.close()
    print(f"[PLATFORM] Tool #{tool_id} → {'enabled' if enabled else 'disabled'}")
    return jsonify({"success": True})


@app.route("/admin/tools/add", methods=["POST"])
def add_tool():
    data  = request.json
    agent = data.get("agent_name", "")
    tool  = data.get("tool_name", "")
    if not agent or not tool:
        return jsonify({"success": False, "error": "agent and tool required"})
    conn = _get_tools_conn()
    conn.execute(
        "INSERT INTO agent_tools (agent_name, tool_name, enabled) VALUES (?,?,1)",
        (agent, tool)
    )
    conn.commit()
    conn.close()
    print(f"[PLATFORM] Added tool '{tool}' to agent '{agent}'")
    return jsonify({"success": True})


@app.route("/admin/tools/remove", methods=["POST"])
def remove_tool():
    data    = request.json
    tool_id = data.get("id")
    conn = _get_tools_conn()
    conn.execute("DELETE FROM agent_tools WHERE id=?", (tool_id,))
    conn.commit()
    conn.close()
    print(f"[PLATFORM] Removed tool #{tool_id}")
    return jsonify({"success": True})


# ── RAG Documents ──

@app.route("/admin/rag")
def admin_rag():
    docs = _load_rag_docs()
    return render_template("rag.html", docs=docs)


@app.route("/admin/rag/add", methods=["POST"])
def add_rag_doc():
    data = request.json
    docs = _load_rag_docs()
    new_doc = {
        "id":      len(docs) + 1,
        "title":   data.get("title", ""),
        "content": data.get("content", ""),
        "source":  data.get("source", "manual"),
        "section": data.get("section", "general"),
        "added_at": __import__("datetime").datetime.utcnow().isoformat(),
    }
    docs.append(new_doc)
    _save_rag_docs(docs)
    print(f"[PLATFORM] Added RAG doc: {new_doc['title']}")
    return jsonify({"success": True, "doc": new_doc})


@app.route("/admin/rag/remove", methods=["POST"])
def remove_rag_doc():
    data   = request.json
    doc_id = data.get("id")
    docs   = _load_rag_docs()
    docs   = [d for d in docs if d["id"] != doc_id]
    _save_rag_docs(docs)
    print(f"[PLATFORM] Removed RAG doc #{doc_id}")
    return jsonify({"success": True})


# ── Tickets ──

@app.route("/admin/tickets")
def admin_tickets():
    status = request.args.get("status")
    tickets = list_tickets(status=status, limit=50)
    summary = get_ticket_summary()
    return render_template("tickets.html", tickets=tickets, summary=summary,
                           current_status=status)


@app.route("/admin/tickets/<ticket_id>")
def ticket_detail(ticket_id):
    ticket = get_ticket(ticket_id)
    return render_template("ticket_detail.html", ticket=ticket)


@app.route("/admin/tickets/<ticket_id>/update", methods=["POST"])
def update_ticket(ticket_id):
    data       = request.json
    new_status = data.get("status")
    resolution = data.get("resolution", "")
    success    = update_ticket_status(ticket_id, new_status, resolution)
    return jsonify({"success": success})


# ── HITL ──

@app.route("/admin/hitl")
def admin_hitl():
    conn = _get_tools_conn()
    hitl_conn = sqlite3.connect(str(STATE_GRAPH_DB), check_same_thread=False)
    hitl_conn.row_factory = sqlite3.Row
    try:
        rows = hitl_conn.execute(
            "SELECT * FROM hitl_tasks ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        tasks = [dict(r) for r in rows]
    except Exception:
        tasks = []
    hitl_conn.close()
    return render_template("hitl.html", tasks=tasks)


@app.route("/admin/hitl/<task_id>/approve", methods=["POST"])
def approve_hitl(task_id):
    data     = request.json
    approved = data.get("approved", True)
    feedback = data.get("feedback", "Approved by admin")
    print(f"[PLATFORM] HITL task #{task_id} → {'APPROVED' if approved else 'REJECTED'}")
    return jsonify({"success": True, "approved": approved, "feedback": feedback})


# ── API: Agent list ──

@app.route("/api/agents")
def api_agents():
    conn  = _get_tools_conn()
    rows  = conn.execute(
        "SELECT DISTINCT agent_name FROM agent_tools WHERE enabled=1"
    ).fetchall()
    conn.close()
    agents = [r["agent_name"] for r in rows]
    return jsonify({"agents": agents})


@app.route("/api/tools/<agent_name>")
def api_tools(agent_name):
    conn = _get_tools_conn()
    rows = conn.execute(
        "SELECT * FROM agent_tools WHERE agent_name=? AND enabled=1",
        (agent_name,)
    ).fetchall()
    conn.close()
    tools = [r["tool_name"] for r in rows]
    return jsonify({"agent": agent_name, "tools": tools})


# ================================================================
# Run
# ================================================================

if __name__ == "__main__":
    print("🏠 Cornerstone Realty Platform")
    print("   User  : http://127.0.0.1:5000")
    print("   Admin : http://127.0.0.1:5000/admin")
    app.run(debug=True, port=5000)