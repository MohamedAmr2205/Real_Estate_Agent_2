"""
planning/algorithms/tree_of_thoughts.py
=========================================
Adapted from: AmrSheta22/task_decomposition_and_planning → planning_lab/algorithms/tree_of_thoughts.py

Tree of Thoughts (Yao et al., 2023):
  1. GENERATE: produce N candidate next-thoughts
  2. EVALUATE: self-score each candidate (0–10)
  3. SEARCH:   BFS — keep top-K at each level, prune the rest
  4. COMMIT:   return the highest-scoring leaf

Fits sub-tasks where MULTIPLE valid strategies exist and it's worth
comparing before committing:
  → "Which recommendation strategy fits this seller best?"
  → "What counter-offer terms maximize the seller's position?"

Changes vs. toolkit original:
  - Model provider → Groq
  - BFS search loop kept intact (toolkit interface preserved)
  - Domain prompts rewritten for real-estate offer-strategy
  - Artifact JSON saved to planning/artifacts/

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

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNING_DIR = REPO_ROOT / "planning"
for _p in (str(REPO_ROOT), str(PLANNING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import EnvironmentFeedback

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
GROQ_MODEL = "openai/gpt-oss-120b"

ARTIFACTS_DIR = PLANNING_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_GENERATE_SYSTEM = """You are a real-estate offer-strategy expert.

Generate {n} distinct candidate strategies for the given sub-task.
Each strategy must be meaningfully different — vary the recommendation
approach, risk tolerance, or negotiation tactic.

Return ONLY a JSON array of {n} strategy strings:
["Strategy 1: ...", "Strategy 2: ...", ...]
"""

_EVALUATE_SYSTEM = """You are a senior real-estate broker evaluating offer strategies.

Score this candidate strategy from 0 to 10 based on:
  - Feasibility given the seller's deadline and constraints (3 points)
  - Financial outcome for the seller (3 points)
  - Risk management (2 points)
  - Clarity and actionability (2 points)

Return ONLY a JSON object:
{"score": <0-10>, "reason": "<one sentence>"}
"""

# ---------------------------------------------------------------------------
# Core: tree_of_thoughts (BFS, toolkit interface preserved)
# ---------------------------------------------------------------------------
def tree_of_thoughts(
    sub_task: str,
    context: dict[str, Any],
    n_candidates: int = 3,
    beam_width: int = 2,
    max_depth: int = 3,
    environment=None,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Tree of Thoughts with BFS for offer-strategy sub-tasks.

    Args:
        sub_task:      Natural-language description of what to decide.
        context:       Offer/property context dict.
        n_candidates:  How many thoughts to generate per node.
        beam_width:    How many top thoughts to keep at each BFS level.
        max_depth:     Maximum depth of the search tree.
        environment:   Optional grounded Environment for final validation.
        artifacts_dir: Where to save the JSON trace.

    Returns:
        {
          "best_thought": str,
          "score": float,
          "all_branches": [...],
          "environment_feedback": EnvironmentFeedback | None,
          "llm_calls": int,
          "latency_s": float,
        }
    """
    run_id = str(uuid.uuid4())[:8]
    start = time.time()
    llm_calls = 0
    all_branches = []

    context_str = json.dumps(context, indent=2)

    # BFS beam — each element: (thought_text, cumulative_score, depth)
    beam: list[tuple[str, float, int]] = [("", 0.0, 0)]

    for depth in range(max_depth):
        next_beam: list[tuple[str, float, int]] = []

        for parent_thought, parent_score, _ in beam:
            # ----------------------------------------------------------------
            # GENERATE: N candidates from this parent
            # ----------------------------------------------------------------
            gen_resp = _groq.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=700,
                temperature=0.7,   # higher temp for diversity
                messages=[
                    {"role": "system",
                     "content": _GENERATE_SYSTEM.format(n=n_candidates)},
                    {"role": "user", "content": (
                        f"Sub-task: {sub_task}\n\n"
                        f"Context:\n{context_str}\n\n"
                        + (f"Parent thought to extend:\n{parent_thought}\n\n"
                           if parent_thought else "") +
                        f"Generate {n_candidates} distinct candidate strategies."
                    )},
                ],
            )
            llm_calls += 1
            gen_text = gen_resp.choices[0].message.content.strip()

            try:
                if gen_text.startswith("```"):
                    gen_text = gen_text.split("```")[1]
                    if gen_text.startswith("json"):
                        gen_text = gen_text[4:]
                candidates = json.loads(gen_text)
                if not isinstance(candidates, list):
                    candidates = [gen_text]
            except json.JSONDecodeError:
                candidates = [gen_text]

            # ----------------------------------------------------------------
            # EVALUATE: score each candidate
            # ----------------------------------------------------------------
            for candidate in candidates[:n_candidates]:
                eval_resp = _groq.chat.completions.create(
                    model=GROQ_MODEL,
                    max_tokens=200,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": _EVALUATE_SYSTEM},
                        {"role": "user", "content": (
                            f"Sub-task: {sub_task}\n\n"
                            f"Context:\n{context_str}\n\n"
                            f"Candidate strategy:\n{candidate}"
                        )},
                    ],
                )
                llm_calls += 1
                eval_text = eval_resp.choices[0].message.content.strip()

                try:
                    if eval_text.startswith("```"):
                        eval_text = eval_text.split("```")[1]
                        if eval_text.startswith("json"):
                            eval_text = eval_text[4:]
                    eval_data = json.loads(eval_text)
                    score = float(eval_data.get("score", 5))
                    reason = eval_data.get("reason", "")
                except (json.JSONDecodeError, ValueError):
                    score = 5.0
                    reason = "Could not parse evaluation"

                cumulative = parent_score + score
                next_beam.append((candidate, cumulative, depth + 1))
                all_branches.append({
                    "depth": depth + 1,
                    "thought": candidate,
                    "score": score,
                    "cumulative_score": cumulative,
                    "reason": reason,
                })

        # Keep top beam_width thoughts
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]

        if not beam:
            break

    # Best thought = highest cumulative score
    best_thought, best_score, _ = max(beam, key=lambda x: x[1]) if beam else ("", 0.0, 0)

    # -----------------------------------------------------------------------
    # GROUNDED VALIDATION
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

    # Artifact
    trace = {
        "run_id": run_id,
        "algorithm": "tree_of_thoughts",
        "sub_task": sub_task,
        "timestamp": datetime.utcnow().isoformat(),
        "config": {"n_candidates": n_candidates, "beam_width": beam_width,
                   "max_depth": max_depth},
        "llm_calls": llm_calls,
        "latency_s": elapsed,
        "all_branches": all_branches,
        "best_thought": best_thought,
        "best_score": best_score,
        "environment_feedback": env_feedback.model_dump() if env_feedback else None,
    }
    _dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    _dir.mkdir(parents=True, exist_ok=True)
    trace_path = _dir / f"tot_{run_id}.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    return {
        "best_thought": best_thought,
        "score": best_score,
        "all_branches": all_branches,
        "environment_feedback": env_feedback,
        "llm_calls": llm_calls,
        "latency_s": elapsed,
        "trace_path": str(trace_path),
    }