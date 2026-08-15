"""
offer_strategy_agent/agent.py — Cornerstone Realty
====================================================
Complete Offer Strategy Agent wiring all three persons' work:

  Person 1 — Decomposition (decomposition.py + dynamic_decomposition.py)
  Person 2 — Planning algorithms + routing + grounded environment
  Person 3 — Self-Refine + Reflexion

This agent:
  1. Connects to the SAME mcp_server/ used by the memory/RAG agent
  2. Never touches agent/ (the memory/RAG agent's code path)
  3. Runs BOTH decomposition methods for comparison
  4. Routes each sub-task to PS / ToT / LATS via routing.py
  5. Applies Self-Refine to the counter-offer draft
  6. Applies Reflexion if the grounded check fails
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PLANNING_DIR = ROOT_DIR / "planning"
MCP_SERVER_DIR = ROOT_DIR / "mcp_server"

for _p in (str(ROOT_DIR), str(PLANNING_DIR), str(MCP_SERVER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# LLM — Groq (same pattern as the rest of the repo)
# ---------------------------------------------------------------------------
from langchain_groq import ChatGroq

def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.1,
    )

# ---------------------------------------------------------------------------
# Person 1 — Decomposition
# ---------------------------------------------------------------------------
from planning.algorithms.decomposition import (
    decompose_goal,
    execute_plan,
    final_output,
)
from planning.algorithms.dynamic_decomposition import dynamic_decomposition

# ---------------------------------------------------------------------------
# Person 2 — Planning algorithms + routing + grounded environment
# ---------------------------------------------------------------------------
from planning.algorithms.environment import Environment
from planning.algorithms.routing import route

# ---------------------------------------------------------------------------
# Person 3 — Self-correction
# ---------------------------------------------------------------------------
from planning.algorithms.self_refine import SelfRefine
from planning.algorithms.reflexion import Reflexion

# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Rubric for counter-offer drafts (used by Self-Refine + Reflexion)
# ---------------------------------------------------------------------------
COUNTER_OFFER_RUBRIC = [
    "Must keep a professional and polite tone throughout.",
    "Must NOT reveal the seller's minimum floor price.",
    "Must propose a concrete counter-offer price.",
    "Must reference the seller's closing deadline.",
    "Must comply with Cornerstone Realty policy.",
]

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
async def run_offer_strategy(
    request: str,
    context: dict,
    use_dynamic: bool = True,
    apply_self_refine: bool = True,
    apply_reflexion: bool = True,
) -> dict:
    """
    Full offer-strategy pipeline:
      1. Decompose (static or dynamic)
      2. Route sub-tasks to PS / ToT / LATS
      3. Self-Refine the counter-offer draft
      4. Reflexion if grounded check fails

    Args:
        request:           Raw offer-strategy request from the broker.
        context:           property_id, offer_id, city, deadlines, etc.
        use_dynamic:       True = dynamic decomposition (default).
        apply_self_refine: True = run Self-Refine on the final draft.
        apply_reflexion:   True = run Reflexion if grounded check fails.
    """
    server_script = str(ROOT_DIR / "mcp_server" / "server.py")
    server_params = StdioServerParameters(
        command=sys.executable, args=[server_script]
    )

    llm = _get_llm()
    grounded_env = Environment(success_threshold=0.6)

    async with stdio_client(server_params) as (read, write, *_):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ----------------------------------------------------------------
            # STEP 1 — Decomposition (Person 1)
            # ----------------------------------------------------------------
            if use_dynamic:
                print("\n[AGENT] STEP 1 — Dynamic decomposition...")
                history = await dynamic_decomposition(
                    goal=request,
                    llm=llm,
                    session=session,
                    context=context,
                )
                recommendation = history[-1][1] if history else ""
                llm_calls_decomp = len(history) * 2
                print(f"[AGENT] Dynamic decomposition: {len(history)} steps")
                if any("pivot" in str(h).lower() for h in history):
                    print("[AGENT] *** PIVOT DETECTED — dynamic changed course ***")
            else:
                print("\n[AGENT] STEP 1 — Decomposition-first...")
                plan = decompose_goal(goal=request, llm=llm)
                print(f"[AGENT] DAG generated: {len(plan.tasks)} tasks")
                for t in plan.tasks:
                    hint = getattr(plan, "_routing_hints", {}).get(t.id, {})
                    print(f"  {t.id} → {t.instruction[:50]}... [{hint.get('method','?')}]")

                outputs = await execute_plan(
                    plan=plan, llm=llm,
                    session=session, context=context,
                )
                recommendation = final_output(plan, outputs)
                llm_calls_decomp = 1 + len(plan.tasks)

            print(f"\n[AGENT] Initial recommendation:\n{recommendation[:300]}...")

            # ----------------------------------------------------------------
            # STEP 2 — Route sub-tasks (Person 2)
            # ----------------------------------------------------------------
            print("\n[AGENT] STEP 2 — Routing sub-tasks to PS / ToT / LATS...")

            # Timeline check → Plan-and-Solve
            ps_result = route(
                sub_task=(
                    "Does the financing contingency deadline fit the seller's "
                    "closing deadline? Calculate and state clearly."
                ),
                context=context,
                environment=grounded_env,
                method_hint="plan_and_solve",
            )
            print(f"[ROUTER] plan_and_solve → {ps_result['solution'][:100]}...")

            # Strategy selection → Tree of Thoughts
            tot_result = route(
                sub_task=(
                    "Which recommendation strategy best serves the seller: "
                    "accept cash offer, counter financed offer, or request "
                    "proof of funds first?"
                ),
                context=context,
                environment=grounded_env,
                method_hint="tree_of_thoughts",
            )
            print(f"[ROUTER] tree_of_thoughts → {tot_result['best_thought'][:100]}...")

            # Final recommendation → LATS (grounded)
            lats_result = route(
                sub_task=(
                    "Propose the final offer-acceptance recommendation and "
                    "counter-offer strategy for the seller."
                ),
                context=context,
                environment=grounded_env,
                method_hint="lats",
            )
            print(f"[ROUTER] lats_grounded → score={lats_result['best_score']:.2f}")
            print(f"[ROUTER] lats best_thought → {lats_result['best_thought'][:100]}...")

            final_recommendation = lats_result["best_thought"]

            # ----------------------------------------------------------------
            # STEP 3 — Self-Refine the counter-offer draft (Person 3)
            # ----------------------------------------------------------------
            self_refine_result = None
            if apply_self_refine:
                print("\n[AGENT] STEP 3 — Self-Refine on counter-offer draft...")
                refiner = SelfRefine()
                self_refine_result = refiner.run(
                    task_description=(
                        "Draft a professional counter-offer response based on "
                        "the recommendation: " + final_recommendation[:200]
                    ),
                    context=context,
                    rubric=COUNTER_OFFER_RUBRIC,
                    environment=grounded_env,
                    max_steps=2,
                )
                print(f"[SELF-REFINE] Iterations: {self_refine_result['total_iterations']}")
                print(f"[SELF-REFINE] Grounded: {self_refine_result['grounded']}")
                print(f"[SELF-REFINE] Final draft:\n{self_refine_result['final_output'][:200]}...")

            # ----------------------------------------------------------------
            # STEP 4 — Reflexion if grounded check failed (Person 3)
            # ----------------------------------------------------------------
            reflexion_result = None
            env_check = grounded_env.evaluate(context)

            if apply_reflexion and not env_check.success:
                print("\n[AGENT] STEP 4 — Grounded check FAILED → running Reflexion...")
                print(f"[REFLEXION] Failures: {env_check.details}")
                reflector = Reflexion()
                reflexion_result = reflector.run(
                    task_description=(
                        "Draft a compliant offer-strategy recommendation that "
                        "passes all business rules: " + request[:200]
                    ),
                    context=context,
                    rubric=COUNTER_OFFER_RUBRIC,
                    environment=grounded_env,
                    max_steps=3,
                )
                print(f"[REFLEXION] Trials: {reflexion_result['total_iterations']}")
                print(f"[REFLEXION] Memory: {reflexion_result['reflections_memory']}")
                print(f"[REFLEXION] Final:\n{reflexion_result['final_output'][:200]}...")
                final_recommendation = reflexion_result["final_output"]
            else:
                print("\n[AGENT] STEP 4 — Grounded check PASSED → Reflexion not needed.")
                print(f"[GROUNDED] Score: {env_check.score:.2f}")

            # ----------------------------------------------------------------
            # Final output
            # ----------------------------------------------------------------
            print("\n" + "="*60)
            print("[FINAL RECOMMENDATION]")
            print("="*60)
            print(final_recommendation)

            return {
                "method": "dynamic" if use_dynamic else "static",
                "final_recommendation": final_recommendation,
                "ps_result": ps_result,
                "tot_result": tot_result,
                "lats_result": lats_result,
                "self_refine_result": self_refine_result,
                "reflexion_result": reflexion_result,
                "grounded_check": {
                    "success": env_check.success,
                    "score": env_check.score,
                    "details": env_check.details,
                },
            }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    DEMO_REQUEST = (
        "A seller just told us they need to close within 3 weeks and they've "
        "received two offers today: "
        "(1) a cash offer 8% below asking with no contingencies, "
        "(2) a financed offer at full asking price with a 30-day financing "
        "contingency. "
        "Help me figure out which one to recommend and draft the "
        "counter/response strategy."
    )

    # Canonical divergence demo — financing pre-approval is expired
    DEMO_CONTEXT = {
        "property_id": 1,
        "offer_id": 3,
        "caller_agent_id": 1,
        "city": "Alexandria",
        "seller_deadline_weeks": 3,
        "financing_contingency_days": 30,
        "proposed_action": "counter",
        "proposed_price": 4600000,
        "acknowledges_tier1_risk": False,
        "financing_pre_approval_expired": True,   # triggers dynamic pivot
    }

    method = sys.argv[1] if len(sys.argv) > 1 else "dynamic"

    asyncio.run(
        run_offer_strategy(
            request=DEMO_REQUEST,
            context=DEMO_CONTEXT,
            use_dynamic=(method != "static"),
            apply_self_refine=True,
            apply_reflexion=True,
        )
    )