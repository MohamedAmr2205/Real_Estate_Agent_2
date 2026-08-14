"""
offer_strategy_agent/agent.py — Cornerstone Realty
====================================================
Person 1's integration piece: wires the DAG executor into the
offer_strategy_agent/ entrypoint.

This agent:
  1. Connects to the SAME mcp_server/ used by the memory/RAG agent
  2. Accepts a real offer-strategy request
  3. Runs BOTH decomposition methods against it
  4. Returns the final recommendation + comparison trace

Never touches agent/ (the memory/RAG agent's code path).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Groq LLM (same pattern as the rest of the repo)
def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

_load_env_file(ROOT_DIR / ".env")

from langchain_groq import ChatGroq

def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.1,
    )

# ---------------------------------------------------------------------------
# Import Person 1's decomposition modules
# ---------------------------------------------------------------------------
from planning.algorithms.decomposition import (
    decompose_goal,
    execute_plan,
    final_output,
)
from planning.algorithms.dynamic_decomposition import dynamic_decomposition


# ---------------------------------------------------------------------------
# Main entry — runs both methods, prints comparison
# ---------------------------------------------------------------------------
async def run_offer_strategy(
    request: str,
    context: dict,
    use_dynamic: bool = True,
) -> dict:
    """
    Args:
        request:     The raw offer-strategy request string from the broker.
        context:     Dict with property_id, offer_id, city, caller_agent_id, etc.
        use_dynamic: If True, runs dynamic decomposition (default for production).
                     Set False to run decomposition-first only (for eval comparison).
    Returns:
        Dict with 'method', 'final_recommendation', 'history' (dynamic) or 'outputs' (static).
    """
    server_script = str(ROOT_DIR / "mcp_server" / "server.py")
    server_params = StdioServerParameters(
        command=sys.executable, args=[server_script]
    )

    llm = _get_llm()

    async with stdio_client(server_params) as (read, write, *_):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if use_dynamic:
                # --- Dynamic decomposition ---
                print(f"\n[OFFER STRATEGY AGENT] Running DYNAMIC decomposition...")
                history = await dynamic_decomposition(
                    goal=request,
                    llm=llm,
                    session=session,
                    context=context,
                )
                final = history[-1][1] if history else "No result."
                print(f"\n[FINAL RECOMMENDATION]\n{final}")
                return {
                    "method": "dynamic_decomposition",
                    "final_recommendation": final,
                    "history": history,
                }
            else:
                # --- Decomposition-first ---
                print(f"\n[OFFER STRATEGY AGENT] Running DECOMPOSITION-FIRST...")
                plan = decompose_goal(goal=request, llm=llm)
                print(f"[DAG] Generated {len(plan.tasks)} tasks")

                outputs = await execute_plan(
                    plan=plan,
                    llm=llm,
                    session=session,
                    context=context,
                )
                final = final_output(plan, outputs)
                print(f"\n[FINAL RECOMMENDATION]\n{final}")
                return {
                    "method": "decomposition_first",
                    "final_recommendation": final,
                    "outputs": outputs,
                }


# ---------------------------------------------------------------------------
# CLI entry — python -m offer_strategy_agent.agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example real request (the canonical divergence demo case)
    DEMO_REQUEST = (
        "A seller needs to close within 3 weeks and has received two offers: "
        "(1) a cash offer 8% below asking with no contingencies, "
        "(2) a financed offer at full asking price with a 30-day financing contingency. "
        "Recommend which offer to accept and draft the counter-offer strategy."
    )

    # Context for the divergence demo: financing pre-approval is expired
    DEMO_CONTEXT = {
        "property_id": 1,
        "offer_id": 3,
        "caller_agent_id": 1,
        "city": "Alexandria",
        "financing_pre_approval_expired": True,   # triggers the pivot
    }

    method = sys.argv[1] if len(sys.argv) > 1 else "dynamic"
    asyncio.run(
        run_offer_strategy(
            request=DEMO_REQUEST,
            context=DEMO_CONTEXT,
            use_dynamic=(method != "static"),
        )
    )