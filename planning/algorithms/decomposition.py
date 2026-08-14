"""
decomposition.py — Cornerstone Realty Offer Strategy Agent
============================================================

Changes from toolkit original:
  - PLANNER_SYSTEM rewritten for real-estate offer-strategy domain
  - execute_plan() now calls REAL MCP tools via an injected async
    session instead of asking the LLM to "execute" free-form text
  - sub-task routing hook added: each PlannedTask carries a
    "method" hint (plan_and_solve | tree_of_thoughts | lats) that
    routing.py uses to pick the right planning algorithm
  - acyclicity enforced at construction time via _assert_acyclic()
    (raises CyclicPlanError before any tool is called)
  - artifacts/ JSON trace emitted in the toolkit's own format

Toolkit interfaces kept intact:
  - decompose_goal(goal, llm) -> Plan
  - execute_plan(plan, llm, ...) -> dict[str, str]
  - final_output(plan, outputs) -> str
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
import os
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Re-use toolkit's Plan / Task domain models unchanged
# ---------------------------------------------------------------------------
try:
    from ..models import Plan
except Exception:
    import sys
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from models import Plan
    except Exception as e:
        models_path = repo_root / "models.py"
        if models_path.exists():
            spec = importlib.util.spec_from_file_location("models", str(models_path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["models"] = mod
            spec.loader.exec_module(mod)
            Plan = getattr(mod, "Plan")
        else:
            raise ModuleNotFoundError(
                "Could not import 'models'. Ensure models.py exists in the repository root or run as a package."
            ) from e

# ---------------------------------------------------------------------------
# Artifacts directory (toolkit's own trace format)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Domain-specific planner prompt
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """You are an expert real-estate offer-strategy planner.

Your job is to decompose an offer-comparison request into a small executable DAG
of 4-6 concrete sub-tasks. Each sub-task must:
  1. Map to exactly ONE of the available MCP tools or be a pure-reasoning step.
  2. Carry a "method" hint telling the executor which planning algorithm fits it:
       - "plan_and_solve"   → deterministic, single-pass (e.g. timeline math)
       - "tree_of_thoughts" → multiple valid strategies exist, worth comparing
       - "lats"             → needs external grounded validation before committing
  3. Have dependency ids that refer only to earlier tasks in the plan.

Available MCP tools (use these names exactly in instructions):
  - search_properties      : look up comparable listings
  - get_property           : fetch full property details and documents
  - generate_cma           : run a Comparative Market Analysis
  - explain_offer_risk     : risk analysis for a specific offer
  - submit_offer           : record a new offer (write — use carefully)
  - accept_offer           : accept a pending offer (write — use carefully)
  - search_knowledge_base  : search unstructured property notes and history

The plan MUST end with exactly one synthesis task (id: "t_final") that depends
on every substantive branch. That task produces the seller recommendation and
counter-offer strategy.

Enforce acyclicity: a task may only depend on tasks whose id appears earlier
in the tasks list."""


# ---------------------------------------------------------------------------
# Wire schemas (toolkit pattern, extended with "method" hint)
# ---------------------------------------------------------------------------
class PlannedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    depends_on: list[str]
    method: str = "plan_and_solve"
    mcp_tool: str | None = None


class GeneratedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask]


# ---------------------------------------------------------------------------
# Acyclicity guard
# ---------------------------------------------------------------------------
class CyclicPlanError(ValueError):
    """Raised when the generated DAG contains a cycle."""


def _assert_acyclic(tasks: list[PlannedTask]) -> None:
    ids = {t.id for t in tasks}
    for task in tasks:
        for dep in task.depends_on:
            if dep not in ids:
                raise ValueError(
                    f"Task '{task.id}' depends on unknown task '{dep}'"
                )

    in_degree = {t.id: len(t.depends_on) for t in tasks}
    dependents: dict[str, list[str]] = {t.id: [] for t in tasks}
    for task in tasks:
        for dep in task.depends_on:
            dependents[dep].append(task.id)

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for child in dependents[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if visited != len(tasks):
        cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
        raise CyclicPlanError(
            f"DAG contains a cycle involving tasks: {cycle_nodes}. "
            "Fix dependencies before executing."
        )


# ---------------------------------------------------------------------------
# Helper — strip extra fields and return (clean_payload, routing_hints)
# ---------------------------------------------------------------------------
def _strip_and_hint(generated: GeneratedPlan, goal: str) -> tuple[dict, dict]:
    """
    Toolkit's Plan model has extra='forbid', so we strip 'method' and
    'mcp_tool' before calling Plan.model_validate(). We save them first
    as routing_hints so the executor can still use them.
    """
    routing_hints = {
        t.id: {"method": t.method, "mcp_tool": t.mcp_tool}
        for t in generated.tasks
    }
    payload = generated.model_dump()
    payload["goal"] = goal
    for task in payload["tasks"]:
        task.pop("method", None)
        task.pop("mcp_tool", None)
    return payload, routing_hints


# ---------------------------------------------------------------------------
# Public API — decompose_goal  (toolkit interface, domain-adapted)
# ---------------------------------------------------------------------------
def decompose_goal(goal: str, llm: BaseChatModel | None = None) -> Plan:
    """
    Generate a static DAG for the offer-strategy goal.
    Acyclicity is enforced before returning — CyclicPlanError is raised
    if the model produced a cycle.
    """
    if llm is not None:
        generated: GeneratedPlan = llm.with_structured_output(
            GeneratedPlan,
            method="function_calling",
        ).invoke(
            [
                ("system", PLANNER_SYSTEM),
                ("human", (
                    f"Decompose this offer-strategy request into 4-6 tasks:\n\n"
                    f"{goal!r}\n\n"
                    "Use short task ids: t1, t2, … t_final.\n"
                    "Preserve the supplied goal exactly in the plan's goal field.\n"
                    "Set mcp_tool to the tool name if the task calls one, else null."
                )),
            ],
            temperature=0.1,
        )

        _assert_acyclic(generated.tasks)
        payload, routing_hints = _strip_and_hint(generated, goal)
        plan = Plan.model_validate(payload)
        plan._routing_hints = routing_hints
        return plan

    # Fallback: Google Gemini
    try:
        from google import genai as google_genai
    except Exception as e:
        raise RuntimeError("No LLM provided and google-genai is not installed") from e

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No LLM provided and GEMINI_API_KEY is not set")

    client = google_genai.Client(api_key=api_key)
    human_msg = (
        f"Decompose this offer-strategy request into 4-6 tasks:\n\n"
        f"{goal!r}\n\n"
        'Return a JSON object matching this schema: {"goal":string, "tasks": [{"id":string, "instruction":string, "depends_on": [string], "method":string, "mcp_tool": string|null}]}'
    )
    prompt = PLANNER_SYSTEM + "\n\n" + human_msg
    resp = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    raw = getattr(resp, "text", None) or str(resp)

    try:
        parsed = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Failed to parse Gemini JSON output: {e}\nRaw output:\n{raw}") from e

    generated = GeneratedPlan.model_validate(parsed)
    _assert_acyclic(generated.tasks)
    payload, routing_hints = _strip_and_hint(generated, goal)
    plan = Plan.model_validate(payload)
    plan._routing_hints = routing_hints
    return plan


# ---------------------------------------------------------------------------
# MCP tool dispatcher
# ---------------------------------------------------------------------------
async def _dispatch_mcp_tool(
    tool_name: str,
    instruction: str,
    context: dict[str, Any],
    session,
) -> str:
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
                "query": instruction,
                "property_id": context.get("property_id", 1),
                "caller_agent_id": context.get("caller_agent_id", 1),
                "top_k": 3,
            },
        )
    else:
        return f"[pure-reasoning] {instruction}"

    return result.content[0].text if result.content else "[empty tool result]"


# ---------------------------------------------------------------------------
# Public API — execute_plan
# ---------------------------------------------------------------------------
async def execute_plan(
    plan: Plan,
    llm: BaseChatModel,
    session,
    context: dict[str, Any],
    max_workers: int = 4,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    trace_nodes = []
    routing_hints = getattr(plan, "_routing_hints", {})

    for batch in plan.execution_batches():
        tasks_in_batch = [plan.task(tid) for tid in batch]

        mcp_tasks = [
            t for t in tasks_in_batch
            if routing_hints.get(t.id, {}).get("mcp_tool")
        ]
        reasoning_tasks = [
            t for t in tasks_in_batch
            if not routing_hints.get(t.id, {}).get("mcp_tool")
        ]

        for task in mcp_tasks:
            tool_name = routing_hints[task.id]["mcp_tool"]
            start = time.time()
            result = await _dispatch_mcp_tool(
                tool_name, task.instruction, context, session
            )
            outputs[task.id] = result
            trace_nodes.append({
                "id": task.id,
                "type": "mcp",
                "tool": tool_name,
                "method": routing_hints[task.id].get("method", "plan_and_solve"),
                "latency_s": round(time.time() - start, 3),
                "output_preview": result[:200],
            })

        if reasoning_tasks:
            def _run_reasoning(task):
                hint = routing_hints.get(task.id, {})
                dep_context = "\n\n".join(
                    f"OUTPUT FROM {dep}:\n{outputs[dep]}"
                    for dep in task.depends_on
                    if dep in outputs
                ) or "No prerequisite outputs yet."

                prompt = (
                    f"Overall goal: {plan.goal}\n\n"
                    f"Current sub-task: {task.instruction}\n\n"
                    f"Prerequisite outputs:\n{dep_context}\n\n"
                    "Complete ONLY the current sub-task. "
                    "Be concrete and specific to real-estate offer strategy."
                )
                start = time.time()
                response = llm.invoke(
                    [
                        ("system", "You execute one node in a validated real-estate offer-strategy DAG."),
                        ("human", prompt),
                    ],
                    temperature=0.2,
                )
                content = response.content
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(f"LLM returned empty response for task {task.id}")
                return task.id, content.strip(), round(time.time() - start, 3), hint.get("method", "plan_and_solve")

            with ThreadPoolExecutor(max_workers=min(max_workers, len(reasoning_tasks))) as pool:
                futures = {pool.submit(_run_reasoning, t): t for t in reasoning_tasks}
                for future in as_completed(futures):
                    tid, content, latency, method = future.result()
                    outputs[tid] = content
                    trace_nodes.append({
                        "id": tid,
                        "type": "reasoning",
                        "method": method,
                        "latency_s": latency,
                        "output_preview": content[:200],
                    })

    _save_trace(
        method="decomposition_first",
        goal=plan.goal,
        nodes=trace_nodes,
        outputs=outputs,
    )

    return outputs


# ---------------------------------------------------------------------------
# Public API — final_output
# ---------------------------------------------------------------------------
def final_output(plan: Plan, outputs: dict[str, str]) -> str:
    terminals = plan.terminal_tasks()
    if len(terminals) != 1:
        raise ValueError(
            f"Expected exactly one terminal synthesis task, found: {terminals}"
        )
    return outputs[terminals[0]]


# ---------------------------------------------------------------------------
# Artifact trace writer
# ---------------------------------------------------------------------------
def _save_trace(
    method: str,
    goal: str,
    nodes: list[dict],
    outputs: dict[str, str],
) -> Path:
    trace = {
        "method": method,
        "goal": goal,
        "timestamp": datetime.utcnow().isoformat(),
        "nodes": nodes,
        "outputs": {k: v[:500] for k, v in outputs.items()},
    }
    path = ARTIFACTS_DIR / f"{method}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return path