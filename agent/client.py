"""
Meridian Realty MCP Agent Client
==================================

This is the "host" side of the MCP handshake: it spawns mcp_server/server.py
over stdio, performs a real initialize/initialized exchange, and drives
every tool so each protocol concern can be demonstrated live (this is
what agent/README.md's demo transcript is generated from).

Sections:
  SECTION 1: .env loading + Groq client (for real SAMPLING, free tier)
  SECTION 2: CAPABILITY DECLARATION (client side) — elicitation + sampling
             callbacks are how THIS client opts in; if you don't pass a
             callback, that capability is not offered to the server.
  SECTION 3: NOTIFICATIONS — message handler reacting to tools/list_changed
  SECTION 4: Elicitation callback — a real human-in-the-loop pause (CLI input)
  SECTION 5: Sampling callback — the CLIENT's own model does the reasoning
  SECTION 6: main() — connects, negotiates, and exercises every tool
             (including the Add-On Lab's search_knowledge_base tool)
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure the package root is importable when running this script directly.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from memory.short_term import ShortTermMemory
from memory.router import route_overflow
from memory.episodic_store import EpisodicStore
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    ElicitRequestParams,
    ElicitResult,
    TextContent,
)


# ---------------------------------------------------------------------------
# SECTION 1 — .env loading + Groq client
# ---------------------------------------------------------------------------
# Groq is used here because it has a genuinely free tier (no credit card
# required) — this is purely a development-cost choice. The server has no
# dependency on which provider the client uses; sampling/createMessage
# just asks "the client's model", whatever that happens to be.
def _load_env_file(path: Path) -> None:
    """Minimal .env loader — avoids adding python-dotenv as a hard dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file(Path(__file__).resolve().parent.parent / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

try:
    import openai
    _groq_client = openai.OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    ) if GROQ_API_KEY else None
except ImportError:
    _groq_client = None


# ---------------------------------------------------------------------------
# CONFIG — flip this to demo the "no elicitation support" fallback path
# (Issue #3: capability negotiation). When False, this client does not
# pass an elicitation_callback at all, so the server-side ctx.elicit()
# call in submit_offer would fail for risky offers — so this client
# simply avoids calling submit_offer/accept_offer in that mode.
# ---------------------------------------------------------------------------
SUPPORTS_ELICITATION = True


# ---------------------------------------------------------------------------
# SECTION 3 — NOTIFICATIONS: react to tools/list_changed
# ---------------------------------------------------------------------------
async def message_handler(message) -> None:
    """
    Generic notification handler passed to ClientSession. When the server
    fires notifications/tools/list_changed (see assign_listing_agent in
    server.py), this prints an explicit log line so it's visible in the
    demo transcript that the client actually reacted, rather than the
    tool set silently being different next time we happened to check.
    """
    method = getattr(message.root, "method", None) if hasattr(message, "root") else None
    if method == "notifications/tools/list_changed":
        print("\n[NOTIFICATION RECEIVED] tools/list_changed — "
              "re-fetching tool list now (not polling, reacting).\n")


# ---------------------------------------------------------------------------
# SECTION 4 — ELICITATION callback: real human-in-the-loop pause
# ---------------------------------------------------------------------------
async def elicitation_callback(
    context: RequestContext, params: ElicitRequestParams
) -> ElicitResult:
    """
    Called when the server invokes elicitation/create (see submit_offer
    in server.py for a below-threshold offer). This genuinely pauses and
    waits for a human at the keyboard to type a decision — it does not
    auto-accept, which would defeat the point of the concern.
    """
    print(f"\n[ELICITATION REQUEST] {params.message}")
    answer = input("Confirm this offer should be submitted? (y/n): ").strip().lower()

    if answer == "y":
        return ElicitResult(action="accept", content={"confirm": True,
                                                        "broker_note": "Confirmed via CLI"})
    return ElicitResult(action="decline", content={"confirm": False})


# ---------------------------------------------------------------------------
# SECTION 5 — SAMPLING callback: the CLIENT's own model reasons, not the server's
# ---------------------------------------------------------------------------
async def sampling_callback(
    context: RequestContext, params: CreateMessageRequestParams
) -> CreateMessageResult:
    """
    Called when the server invokes sampling/createMessage (see
    explain_offer_risk in server.py). The server never runs its own LLM —
    this callback is where the actual reasoning happens, using the
    client's own model (Llama 3.3 70B via Groq's free API), not a model
    the server owns.
    """
    prompt_text = params.messages[0].content.text

    if _groq_client is None:
        # No API key configured — degrade honestly instead of faking a result.
        text = ("[No GROQ_API_KEY configured — set one in .env to get "
                "a real model-generated risk analysis here.]")
    else:
        response = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=params.maxTokens or 200,
            messages=[{"role": "user", "content": prompt_text}],
        )
        text = response.choices[0].message.content

    print(f"\n[SAMPLING] Client's model produced:\n{text}\n")
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=text),
        model="llama-3.3-70b-versatile",
    )


# ---------------------------------------------------------------------------
# SECTION 6 — main(): connect, negotiate, exercise every tool
# ---------------------------------------------------------------------------
async def main() -> None:
    memory = ShortTermMemory(max_turns=20)
    episodic = EpisodicStore()  
    transport_mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"

    if transport_mode == "http":
        # SECTION 10 counterpart: connects to a server already running
        # via `python server.py streamable-http` on a separate process,
        # instead of spawning it locally over stdio. This is the
        # deployment-shaped path (multi-office agents hitting one
        # running server instance).
        from mcp.client.streamable_http import streamablehttp_client
        context_manager = streamablehttp_client("http://127.0.0.1:8000/mcp")
    else:
        server_script = str(Path(__file__).resolve().parent.parent / "mcp_server" / "server.py")
        server_params = StdioServerParameters(command=sys.executable, args=[server_script])
        context_manager = stdio_client(server_params)

    async with context_manager as (read, write, *_):
        session_kwargs = {"message_handler": message_handler}
        if SUPPORTS_ELICITATION:
            session_kwargs["elicitation_callback"] = elicitation_callback
        session_kwargs["sampling_callback"] = sampling_callback  # sampling always on

        async with ClientSession(read, write, **session_kwargs) as session:

            # --- SECTION 2: CAPABILITY NEGOTIATION -------------------------
            init_result = await session.initialize()
            print("=== INITIALIZE RESULT ===")
            print(f"Server: {init_result.serverInfo.name} "
                  f"{init_result.serverInfo.version}")
            print(f"Server capabilities: {init_result.capabilities}")
            print(f"This client declared elicitation support: {SUPPORTS_ELICITATION}\n")

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"=== AVAILABLE TOOLS ({len(tool_names)}) ===\n{tool_names}\n")

            if not SUPPORTS_ELICITATION and "submit_offer" in tool_names:
                print("[CAPABILITY CHECK] This client did not declare "
                      "elicitation support, so it will NOT call submit_offer "
                      "or accept_offer even though the server lists them — "
                      "falling back to read-only tools only.\n")

            # --- SECTION 4: read-only tools ---------------------------------
            print("=== search_properties(city='Alexandria', status='Available') ===")
            result = await session.call_tool(
                "search_properties", {"city": "Alexandria", "status": "Available"}
            )
            print(result.content[0].text, "\n")

            # --- SECTION 8: PROGRESS TRACKING -------------------------------
            print("=== generate_cma(property_id=1) — watch for progress ===")

            async def on_progress(progress: float, total: float | None, message: str | None):
                print(f"  [PROGRESS] {progress}/{total} — {message}")

            cma_result = await session.call_tool(
                "generate_cma", {"property_id": 1}, progress_callback=on_progress
            )
            print(cma_result.content[0].text, "\n")

            if SUPPORTS_ELICITATION:
                # --- SECTION 5: ELICITATION (offer #3-style: below threshold) --
                print("=== submit_offer — a below-threshold offer (triggers elicitation) ===")
                memory.add_turn(customer_id=5, role="user", content={
                    "tool": "submit_offer",
                    "property_id": 1,
                    "offer_amount": 3000000
})
                offer_result = await session.call_tool(
                    "submit_offer",
                    {"property_id": 1, "customer_id": 5, "offer_amount": 3000000},
                )
                memory.add_turn(customer_id=5, role="tool", content=offer_result.content[0].text)
                print(f"[MEMORY] turns for customer 5: {len(memory.get_turns(5))}")  # ← ضيف دا عشان تشوف إن الـ memory شغالة
                
                old_turns = memory.get_turns(customer_id=5)
                promoted, dropped = route_overflow(old_turns)
                for turn in promoted:
                 episodic.add(turn, reason="promoted by router")

                 print(f"[EPISODIC] total episodes for customer 5: {len(episodic.get(5))}")

                print(offer_result.content[0].text, "\n")

                # --- SECTION 7: NOTIFICATIONS -----------------------------------
                print("=== assign_listing_agent — Broker (4) takes over property 4 ===")
                assign_result = await session.call_tool(
                    "assign_listing_agent",
                    {"property_id": 4, "new_agent_id": 4, "caller_agent_id": 4},
                )
                print(assign_result.content[0].text, "\n")

                # re-fetch tools after the notification to show the reaction
                tools_after = await session.list_tools()
                print(f"Tool list after notification: "
                      f"{[t.name for t in tools_after.tools]}\n")

                # --- SECTION 6: DEFENSIVE TOOL DESIGN ---------------------------
                print("=== accept_offer — offer 1, called by the listing agent ===")
                accept_result = await session.call_tool(
                    "accept_offer", {"offer_id": 1, "caller_agent_id": 1}
                )
                print(accept_result.content[0].text, "\n")

                print("=== accept_offer — SAME offer, called by an unrelated agent (should fail) ===")
                bad_accept = await session.call_tool(
                    "accept_offer", {"offer_id": 2, "caller_agent_id": 3}
                )
                print(bad_accept.content[0].text, "\n")

                # --- SECTION 9: SAMPLING -----------------------------------------
                print("=== explain_offer_risk(offer_id=3) ===")
                risk_result = await session.call_tool(
                    "explain_offer_risk", {"offer_id": 3}
                )
                print(risk_result.content[0].text, "\n")

                # --- ADD-ON LAB: search_knowledge_base (RAG, Option A) -----------
                print("=== search_knowledge_base — regular agent asks about seller's floor price ===")
                kb_result_1 = await session.call_tool(
                    "search_knowledge_base",
                    {"query": "seller lowest price floor accept", "property_id": 1,
                     "caller_agent_id": 1, "top_k": 3},
                )
                print(kb_result_1.content[0].text, "\n")

                print("=== search_knowledge_base — SAME question, asked by the Broker ===")
                kb_result_2 = await session.call_tool(
                    "search_knowledge_base",
                    {"query": "seller lowest price floor accept", "property_id": 1,
                     "caller_agent_id": 4, "top_k": 3},
                )
                print(kb_result_2.content[0].text, "\n")

                print("=== search_knowledge_base — a question only this tool can answer ===")
                kb_result_3 = await session.call_tool(
                    "search_knowledge_base",
                    {"query": "roof condition inspection", "property_id": 1,
                     "caller_agent_id": 1, "top_k": 3},
                )
                print(kb_result_3.content[0].text, "\n")

            print("=== Demo run complete ===")


if __name__ == "__main__":
    asyncio.run(main())