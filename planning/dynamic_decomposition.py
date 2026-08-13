"""
dynamic_decomposition.py — Cornerstone Realty Offer Strategy Agent
====================================================================
Changes from toolkit original:
  - ADAPTIVE PLANNER prompt rewritten for real-estate offer-strategy domain
  - _execute_step() now calls REAL MCP tools via an injected async session
  - Added the "expired pre-approval" divergence scenario:
      If an early observation reveals that the financing contingency
      deadline has already passed, the planner pivots to a cash-only
      recommendation path instead of continuing to evaluate the
      financed offer — behaviour that decomposition-first cannot do.
  - artifacts/ JSON trace emitted in toolkit's own format (same as
    decomposition.py) so the two methods produce comparable evidence.

Toolkit interfaces kept intact:
  - dynamic_decomposition(goal, llm, ...) -> list[tuple[str, str]]
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Artifacts directory
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Wire schema (toolkit pattern, kept intact)
# ---------------------------------------------------------------------------
class DynamicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    next_task: str
    method: str = "plan_and_solve"      # planning algorithm hint for routing.py
    mcp_tool: str | None = None         # None = pure-reasoning step
    pivot_reason: str | None = None     # populated when planner changes course


# ---------------------------------------------------------------------------
# Constraint-violation detector
# ---------------------------------------------------------------------------
_FINANCING_EXPIRED_SIGNALS = [
    "pre-approval expired",
    "pre-approval has expired",
    "financing contingency deadline",
    "contingency period has passed",
    "approval letter expired",
    "expired pre-approval",
    "financing deadline missed",
]


def _financing_expired(observation: str) -> bool:
    """
    Returns True if the observation contains clear evidence that the
    financing pre-approval has expired. This is the canonical trigger
    for the dynamic-vs-static divergence demo.
    """
    lower = observation.lower()
    return any(signal in lower for signal in _FINANCING_EXPIRED_SIGNALS)


# ---------------------------------------------------------------------------
# MCP tool dispatcher  (same mapping as decomposition.py)
# ---------------------------------------------------------------------------
async def _execute_mcp_step(
    tool_name: str,
    task_description: str,
    context: dict[str, Any],
    session,
) -> str:
    """Dispatch one dynamic step's MCP tool call and return result as string."""
    if tool_name == "generate_cma":
        result = await session.call_tool(
            "generate_cma",
            {"property_id": context.get("property_id", 1)},
        )
    elif tool_name == "explain_offer_risk":
        offer_id = context.get("offer_id") or context.get("primary_offer_id", 1)
        result = await session.call_tool(
            "explain_offer_risk",
            {"offer_id": offer_id},
        )
    elif tool_name == "search_properties":
        result = await session.call_tool(
            "search_properties",
            {
                "city": context.get("city"),
                "status": "Available",
                "max_price": context.get("max_price"),
            },
        )
    elif tool_name == "get_property":
        result = await session.call_tool(
            "get_property",
            {"property_id": context.get("property_id", 1)},
        )
    elif tool_name == "search_knowledge_base":
        result = await session.call_tool(
            "search_knowledge_base",
            {
                "query": task_description,
                "property_id": context.get("property_id", 1),
                "caller_agent_id": context.get("caller_agent_id", 1),
                "top_k": 3,
            },
        )
    else:
        return f"[pure-reasoning step — no MCP call] {task_description}"

    return result.content[0].text if result.content else "[empty tool result]"


# ---------------------------------------------------------------------------
# Public API — dynamic_decomposition  (toolkit interface, domain-adapted)
# ---------------------------------------------------------------------------
async def dynamic_decomposition(
    goal: str,
    llm: BaseChatModel,
    session,                            # live mcp.ClientSession
    context: dict[str, Any],            # property_id, offer_id, city, …
    max_steps: int = 8,
) -> list[tuple[str, str]]:
    """
    Adaptive planner: decides the next sub-task AFTER observing the
    result of the previous one. An early surprise (e.g. expired
    financing pre-approval) reshapes what comes next.

    Returns history: list of (task_description, result) tuples —
    same shape as the toolkit original.

    KEY DIFFERENCE from decomposition_first:
      decomposition-first generates the full plan upfront then executes
      blindly. dynamic_decomposition can PIVOT when an observation
      reveals a changed constraint, e.g. financing pre-approval expired.
    """
    history: list[tuple[str, str]] = []
    trace_steps = []
    pivot_log: list[str] = []

    # Inject pre-approval expiry status into the observation stream
    # if the context carries it (set by the test harness for the divergence demo)
    if context.get("financing_pre_approval_expired"):
        history.append((
            "[context injection] financing pre-approval status check",
            "WARNING: The buyer's financing pre-approval letter has expired. "
            "The 30-day financing contingency cannot be satisfied within the "
            "seller's 3-week closing deadline.",
        ))

    for step in range(max_steps):
        observation_text = "\n".join(
            f"STEP {i+1} — {task}:\n{result}"
            for i, (task, result) in enumerate(history)
        ) or "None"

        # --- Ask the adaptive planner what to do next ---
        decision: DynamicDecision = llm.with_structured_output(
            DynamicDecision,
            method="json_schema",
        ).invoke(
            [
                ("system", _ADAPTIVE_PLANNER_SYSTEM),
                ("human", (
                    f"Goal: {goal}\n\n"
                    f"Completed steps and observations:\n{observation_text}\n\n"
                    "Decide the single best NEXT step.\n"
                    "Set done=true only when the goal is fully met.\n"
                    "When done=true, set next_task to empty string.\n"
                    "If a previous observation revealed an expired financing "
                    "pre-approval, pivot to a cash-offer-only recommendation "
                    "path and explain why in pivot_reason."
                )),
            ],
            temperature=0.1,
        )

        if decision.done:
            break

        task = decision.next_task.strip()
        if not task:
            raise ValueError(
                f"Dynamic planner omitted next_task at step {step + 1}"
            )

        # Log pivot if the planner changed course
        if decision.pivot_reason:
            pivot_log.append(f"Step {step + 1}: {decision.pivot_reason}")

        # --- Execute the step (MCP tool or LLM reasoning) ---
        start = time.time()
        if decision.mcp_tool:
            result = await _execute_mcp_step(
                decision.mcp_tool, task, context, session
            )
            step_type = "mcp"
        else:
            # Pure-reasoning step
            response = llm.invoke(
                [
                    ("system", "Execute the next adaptive sub-task using prior observations."),
                    ("human", (
                        f"Goal: {goal}\n"
                        f"Next task: {task}\n"
                        f"Prior observations:\n{observation_text}"
                    )),
                ],
                temperature=0.2,
            )
            result = response.content
            if not isinstance(result, str) or not result.strip():
                raise RuntimeError(
                    f"LLM returned empty response at dynamic step {step + 1}"
                )
            result = result.strip()
            step_type = "reasoning"

        latency = round(time.time() - start, 3)
        history.append((task, result))
        trace_steps.append({
            "step": step + 1,
            "type": step_type,
            "task": task,
            "method_hint": decision.method,
            "mcp_tool": decision.mcp_tool,
            "pivot_reason": decision.pivot_reason,
            "latency_s": latency,
            "result_preview": result[:200],
        })

        # --- Financing-expired mid-plan detection ---
        # Even without a context flag, if the MCP tool RETURNS evidence
        # of expiry, we inject a clear signal so the next decision sees it.
        if _financing_expired(result) and not context.get("financing_pre_approval_expired"):
            context["financing_pre_approval_expired"] = True
            history.append((
                "[dynamic pivot trigger] financing pre-approval check",
                "DETECTED: Financing pre-approval is expired. "
                "Dropping financed-offer evaluation path. "
                "Pivoting to cash-offer-only recommendation.",
            ))
            pivot_log.append(
                f"Step {step + 1}: Auto-detected expired pre-approval "
                "from tool output — pivoting to cash-only path."
            )

    # --- Emit toolkit-format artifact trace ---
    _save_trace(
        goal=goal,
        steps=trace_steps,
        history=history,
        pivot_log=pivot_log,
    )

    return history


# ---------------------------------------------------------------------------
# Adaptive planner system prompt
# ---------------------------------------------------------------------------
_ADAPTIVE_PLANNER_SYSTEM = """You are an adaptive real-estate offer-strategy planner.

You decide ONE next step at a time, based on what you have already observed.
You do NOT commit to a full plan upfront — you adapt.

Available MCP tools (use exact names in mcp_tool field):
  - generate_cma           : Comparative Market Analysis
  - explain_offer_risk     : risk analysis for a specific offer
  - search_properties      : find comparable listings
  - get_property           : full property details
  - search_knowledge_base  : search property notes and history

Method hints (fill in 'method' field):
  - "plan_and_solve"    → deterministic calculation (e.g. deadline math)
  - "tree_of_thoughts"  → multiple strategies worth comparing
  - "lats"              → needs grounded external validation

CRITICAL PIVOT RULE:
If any observation reveals that the buyer's financing pre-approval has
expired or the financing contingency cannot be met within the seller's
deadline, you MUST:
  1. Set pivot_reason explaining why you are changing course.
  2. Drop all remaining financed-offer evaluation steps.
  3. Pivot immediately to recommending the cash offer only.

This is the key difference from decomposition-first planning, which
would blindly continue evaluating the financed offer anyway."""


# ---------------------------------------------------------------------------
# Artifact trace writer (toolkit's own artifacts/ format)
# ---------------------------------------------------------------------------
def _save_trace(
    goal: str,
    steps: list[dict],
    history: list[tuple[str, str]],
    pivot_log: list[str],
) -> Path:
    trace = {
        "method": "dynamic_decomposition",
        "goal": goal,
        "timestamp": datetime.utcnow().isoformat(),
        "total_steps": len(steps),
        "pivot_log": pivot_log,
        "diverged_from_static": len(pivot_log) > 0,
        "steps": steps,
        "final_history": [
            {"task": t, "result_preview": r[:300]}
            for t, r in history
        ],
    }
    path = ARTIFACTS_DIR / f"dynamic_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return path