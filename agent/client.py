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
from memory.semantic_store import SemanticStore
from memory.recall import recall_episodic, recall_semantic
from memory.consolidation import consolidate
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
# CONFIG
# ---------------------------------------------------------------------------
SUPPORTS_ELICITATION = True


# ---------------------------------------------------------------------------
# SECTION 3 — NOTIFICATIONS
# ---------------------------------------------------------------------------
async def message_handler(message) -> None:
    method = getattr(message.root, "method", None) if hasattr(message, "root") else None
    if method == "notifications/tools/list_changed":
        print("\n[NOTIFICATION RECEIVED] tools/list_changed — "
              "re-fetching tool list now (not polling, reacting).\n")


# ---------------------------------------------------------------------------
# SECTION 4 — ELICITATION callback
# ---------------------------------------------------------------------------
async def elicitation_callback(
    context: RequestContext, params: ElicitRequestParams
) -> ElicitResult:
    print(f"\n[ELICITATION REQUEST] {params.message}")
    answer = input("Confirm this offer should be submitted? (y/n): ").strip().lower()

    if answer == "y":
        return ElicitResult(action="accept", content={"confirm": True,
                                                        "broker_note": "Confirmed via CLI"})
    return ElicitResult(action="decline", content={"confirm": False})


# ---------------------------------------------------------------------------
# SECTION 5 — SAMPLING callback
# ---------------------------------------------------------------------------
async def sampling_callback(
    context: RequestContext, params: CreateMessageRequestParams
) -> CreateMessageResult:
    prompt_text = params.messages[0].content.text

    if _groq_client is None:
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
# SECTION 6 — main()
# ---------------------------------------------------------------------------
async def main() -> None:

    # ----------------------------------------------------------------
    # MEMORY SETUP — إنشاء واحدة من كل نوع (مش اتنين)
    # ----------------------------------------------------------------
    memory = ShortTermMemory(max_turns=6)  # صغير عشان نوضح الـ overflow
    episodic = EpisodicStore()
    semantic = SemanticStore()

    # ----------------------------------------------------------------
    # SCRATCHPAD — منفصل عن البافر، مش بيتمسح لما الـ buffer يـ prune
    # ----------------------------------------------------------------
    memory.set_goal(5, "Find villa under 5M in Alexandria, close by August")
    print(f"[SCRATCHPAD] goal set for customer 5: {memory.get_goal(5)}\n")

    transport_mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"

    if transport_mode == "http":
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
        session_kwargs["sampling_callback"] = sampling_callback

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

                # ----------------------------------------------------------------
                # MEMORY DEMO — ضيف small talk الأول عشان الـ router يعمله drop
                # ----------------------------------------------------------------
                print("=== MEMORY DEMO: adding turns to show promote vs drop ===")
                memory.add_turn(customer_id=5, role="user", content="hello thanks for your help")
                memory.add_turn(customer_id=5, role="user", content="okay sounds good")
                memory.add_turn(customer_id=5, role="user", content="hi there")
                memory.add_turn(customer_id=5, role="user", content="great bye")

                # ضيف turns مهمة
                memory.add_turn(customer_id=5, role="user", content={
                    "tool": "submit_offer",
                    "property_id": 1,
                    "offer_amount": 3000000
                })

                # --- SECTION 5: ELICITATION -----------------------------------
                print("=== submit_offer — a below-threshold offer (triggers elicitation) ===")
                offer_result = await session.call_tool(
                    "submit_offer",
                    {"property_id": 1, "customer_id": 5, "offer_amount": 3000000},
                )
                memory.add_turn(customer_id=5, role="tool", content=offer_result.content[0].text)

                print(f"\n[MEMORY] total turns for customer 5: {len(memory.get_turns(5))}")
                print(f"[SCRATCHPAD] goal still intact after buffer fills: {memory.get_goal(5)}")

                # ----------------------------------------------------------------
                # PROMOTE-OR-DROP — يشتغل لما البافر يتملى
                # ----------------------------------------------------------------
                print(f"\n[ROUTER] buffer full ({len(memory.get_turns(5))}/{memory.max_turns}) — running promote-or-drop...")
                old_turns = memory.get_turns(customer_id=5)
                promoted, dropped = route_overflow(old_turns)

                print(f"[ROUTER] promoted={len(promoted)} dropped={len(dropped)}")

                # ابعت المهمين للـ episodic store
                for turn in promoted:
                    episodic.add(turn, reason="promoted by router")

                print(f"[EPISODIC] total episodes for customer 5: {len(episodic.get(5))}")
                print(f"[SCRATCHPAD] goal still safe after routing: {memory.get_goal(5)}\n")

                print(offer_result.content[0].text, "\n")

                # --- SECTION 7: NOTIFICATIONS -----------------------------------
                print("=== assign_listing_agent — Broker (4) takes over property 4 ===")
                assign_result = await session.call_tool(
                    "assign_listing_agent",
                    {"property_id": 4, "new_agent_id": 4, "caller_agent_id": 4},
                )
                print(assign_result.content[0].text, "\n")

                tools_after = await session.list_tools()
                print(f"Tool list after notification: "
                      f"{[t.name for t in tools_after.tools]}\n")

                # --- SECTION 6: DEFENSIVE TOOL DESIGN --------------------------
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

                # --- SECTION 9: SAMPLING ----------------------------------------
                print("=== explain_offer_risk(offer_id=3) ===")
                risk_result = await session.call_tool(
                    "explain_offer_risk", {"offer_id": 3}
                )
                print(risk_result.content[0].text, "\n")

                # --- ADD-ON LAB: search_knowledge_base --------------------------
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

            # ----------------------------------------------------------------
            # CONSOLIDATION — pass دوري منفصل عن الـ router
            # ----------------------------------------------------------------
            print("=== CONSOLIDATION PASS (periodic, separate from router) ===")
            consolidate(customer_id=5, episodic=episodic, semantic=semantic)

            print(f"[SEMANTIC] budget for customer 5: {semantic.get_fact(5, 'budget')}")
            print(f"[SEMANTIC] full history: {semantic.get_history(5, 'budget')}")

            # ----------------------------------------------------------------
            # CONFLICT DEMO — غير الـ budget وشوف الـ versioning
            # ----------------------------------------------------------------
            print("\n=== CONFLICT DEMO: customer updates budget ===")
            memory.add_turn(customer_id=5, role="user",
                            content="actually my budget is 4500000 now")
            new_turns = memory.get_turns(customer_id=5)
            promoted2, _ = route_overflow(new_turns)
            for turn in promoted2:
                episodic.add(turn, reason="budget update")

            consolidate(customer_id=5, episodic=episodic, semantic=semantic)
            print(f"[SEMANTIC] updated budget: {semantic.get_fact(5, 'budget')}")
            print(f"[SEMANTIC] full version history:")
            for f in semantic.get_history(5, 'budget'):
                print(f"  → value='{f.value}' superseded={f.superseded} ts={f.timestamp[:19]}")

            # ----------------------------------------------------------------
            # SELF-RAG MEMORY RECALL VERIFICATION
            # ----------------------------------------------------------------
            print("\n=== MEMORY RECALL — Self-RAG verification ===")
            ep_result = recall_episodic(customer_id=5, query="what is the client budget?", episodic=episodic)
            print(f"[RECALL] episodic verified={ep_result['self_rag']['verified']} episodes={len(ep_result['episodes'])}")

            sem_result = recall_semantic(customer_id=5, query="what is the client budget?", semantic=semantic)
            print(f"[RECALL] semantic verified={sem_result['self_rag']['verified']} facts={sem_result['facts']}")

            print("\n=== Demo run complete ===")


if __name__ == "__main__":
    asyncio.run(main())