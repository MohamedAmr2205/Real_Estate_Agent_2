"""
planning/algorithms/plan_and_solve.py
=======================================
Adapted from: AmrSheta22/task_decomposition_and_planning → planning_lab/algorithms/plan_and_solve.py

Plan-and-Solve (Wang et al., ACL 2023):
  Phase 1 — PLAN:   one explicit planning call that lays out the steps
  Phase 2 — SOLVE:  execute each step in order, single pass, no branching

Fits sub-tasks that are DETERMINISTIC and need no lookahead:
  → "Does the 30-day financing contingency fit the seller's 3-week deadline?"
  → Timeline math, risk-tier arithmetic, single-pass comparisons

Changes vs. toolkit original:
  - Model provider → Groq (same openai client pattern as agent/client.py)
  - Environment.evaluate() wired in for grounded validation at the end
  - Artifact JSON saved to planning/artifacts/
  - Domain prompt rewritten for real-estate offer-strategy

Person 2 owns this file.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — same pattern used across the repo
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_SERVER_DIR = REPO_ROOT / "mcp_server"
PLANNING_DIR = REPO_ROOT / "planning"
for _p in (str(REPO_ROOT), str(MCP_SERVER_DIR), str(PLANNING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import EnvironmentFeedback, Plan

# ---------------------------------------------------------------------------
# Groq client — same pattern as agent/client.py
# ---------------------------------------------------------------------------
def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_load_env(REPO_ROOT / ".env")

import openai as _openai
_groq = _openai.OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url="https://api.groq.com/openai/v1",
)
GROQ_MODEL = "llama-3.3-70b-versatile"

ARTIFACTS_DIR = PLANNING_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_PLAN_SYSTEM = """You are an expert real-estate offer-strategy analyst.

PHASE 1 — PLANNING ONLY.
Given a sub-task from an offer-comparison request, produce a numbered
step-by-step plan to solve it. Do NOT solve it yet — only plan.

Be concrete: each step should reference specific data fields
(offer_amount, list_price, contingency_days, seller_deadline_weeks, etc.)

Return your plan as a JSON array of step strings:
["Step 1: ...", "Step 2: ...", ...]
"""

_SOLVE_SYSTEM = """You are an expert real-estate offer-strategy analyst.

PHASE 2 — SOLVING.
You are given a sub-task and a plan. Execute the plan step by step.
Show your working for each step. Be specific with numbers and deadlines.
End with a clear, actionable conclusion.
"""

# ---------------------------------------------------------------------------
# Core: plan_and_solve
# ---------------------------------------------------------------------------
def plan_and_solve(
    sub_task: str,
    context: dict[str, Any],
    environment=None,          # Environment instance for grounded validation
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Plan-and-Solve for one deterministic sub-task.

    Args:
        sub_task:     Natural-language description of the sub-task.
        context:      Dict with offer amounts, deadlines, property info, etc.
        environment:  Optional grounded Environment — if provided, the final
                      answer is validated against real DB/RAG checks.
        artifacts_dir: Where to save the JSON trace.

    Returns:
        {
          "plan": [...steps...],
          "solution": "...",
          "environment_feedback": EnvironmentFeedback | None,
          "llm_calls": int,
          "latency_s": float,
        }
    """
    run_id = str(uuid.uuid4())[:8]
    start = time.time()
    llm_calls = 0

    context_str = json.dumps(context, indent=2)

    # -----------------------------------------------------------------------
    # PHASE 1: PLAN
    # -----------------------------------------------------------------------
    plan_resp = _groq.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=600,
        temperature=0.1,
        messages=[
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": (
                f"Sub-task: {sub_task}\n\n"
                f"Available context:\n{context_str}"
            )},
        ],
    )
    llm_calls += 1
    plan_text = plan_resp.choices[0].message.content.strip()

    # Parse plan steps
    try:
        if plan_text.startswith("```"):
            plan_text = plan_text.split("```")[1]
            if plan_text.startswith("json"):
                plan_text = plan_text[4:]
        steps = json.loads(plan_text)
        if not isinstance(steps, list):
            steps = [plan_text]
    except json.JSONDecodeError:
        steps = [plan_text]

    # -----------------------------------------------------------------------
    # PHASE 2: SOLVE
    # -----------------------------------------------------------------------
    solve_resp = _groq.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=800,
        temperature=0.1,
        messages=[
            {"role": "system", "content": _SOLVE_SYSTEM},
            {"role": "user", "content": (
                f"Sub-task: {sub_task}\n\n"
                f"Context:\n{context_str}\n\n"
                f"Plan to follow:\n" +
                "\n".join(steps) +
                "\n\nNow execute the plan step by step and give your conclusion."
            )},
        ],
    )
    llm_calls += 1
    solution = solve_resp.choices[0].message.content.strip()

    # -----------------------------------------------------------------------
    # GROUNDED VALIDATION (optional)
    # -----------------------------------------------------------------------
    env_feedback = None
    if environment is not None:
        state = {
            "property_id": context.get("property_id"),
            "offer_id": context.get("offer_id"),
            "proposed_action": context.get("proposed_action", "counter"),
            "proposed_price": context.get("proposed_price"),
            "seller_deadline_weeks": context.get("seller_deadline_weeks"),
            "financing_contingency_days": context.get("financing_contingency_days"),
            "acknowledges_tier1_risk": context.get("acknowledges_tier1_risk", False),
        }
        env_feedback = environment.evaluate(state)

    elapsed = round(time.time() - start, 2)

    # -----------------------------------------------------------------------
    # Artifact
    # -----------------------------------------------------------------------
    trace = {
        "run_id": run_id,
        "algorithm": "plan_and_solve",
        "sub_task": sub_task,
        "timestamp": datetime.utcnow().isoformat(),
        "llm_calls": llm_calls,
        "latency_s": elapsed,
        "plan_steps": steps,
        "solution": solution,
        "environment_feedback": env_feedback.model_dump() if env_feedback else None,
    }
    _dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    _dir.mkdir(parents=True, exist_ok=True)
    trace_path = _dir / f"plan_and_solve_{run_id}.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    return {
        "plan": steps,
        "solution": solution,
        "environment_feedback": env_feedback,
        "llm_calls": llm_calls,
        "latency_s": elapsed,
        "trace_path": str(trace_path),
    }