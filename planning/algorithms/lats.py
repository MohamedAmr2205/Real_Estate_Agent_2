"""
planning/algorithms/lats.py
=============================
Adapted from: AmrSheta22/task_decomposition_and_planning → planning_lab/algorithms/lats.py

LATS — Language Agent Tree Search (Zhou et al., 2023):
  MCTS-guided search with REAL external feedback (not model self-scoring).
  Four phases per node:
    1. SELECT:   UCT formula picks the most promising node to expand
    2. EXPAND:   generate candidate next-actions from selected node
    3. EVALUATE: call Environment.evaluate() — REAL grounded check
                 (not the toolkit's randomized default)
    4. REFLECT:  failed branches get a verbal reflection that steers
                 future search away from the same mistake
    5. BACKPROP: update visit counts and scores up the tree

Fits sub-tasks where a WRONG PLAN HAS REAL COST and we need external
validation before committing:
  → "Propose the final offer-acceptance recommendation"
  → Any sub-task where grounded.evaluate() can catch a bad plan

Changes vs. toolkit original:
  - Model provider → Groq
  - environment.evaluate() is the REAL grounded check (not random)
  - Reflection text fed back into next expansion (toolkit's own pattern)
  - MCTS visit counts and branch reflections saved to artifacts/

Person 2 owns this file.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
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
# MCTS Node
# ---------------------------------------------------------------------------
@dataclass
class MCTSNode:
    thought: str
    parent: "MCTSNode | None" = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_score: float = 0.0
    reflection: str = ""          # verbal reflection from failed evaluation
    env_feedback: EnvironmentFeedback | None = None

    @property
    def avg_score(self) -> float:
        return self.total_score / self.visits if self.visits > 0 else 0.0

    def uct(self, exploration: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return self.avg_score + exploration * math.sqrt(
            math.log(parent_visits) / self.visits
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_EXPAND_SYSTEM = """You are a senior real-estate broker proposing offer strategies.

Generate {n} distinct candidate next-actions for the given sub-task.
If there are reflections from failed attempts, use them to AVOID the
same mistakes.

Return ONLY a JSON array of {n} strategy strings:
["Strategy 1: ...", "Strategy 2: ...", ...]
"""

_REFLECT_SYSTEM = """You are a real-estate compliance officer reviewing a failed
offer-strategy proposal.

The proposal FAILED the grounded business-rule check for these reasons:
{failure_details}

Write ONE concise sentence explaining what mistake was made and what
to avoid in future attempts. This reflection will be used to guide
the next search iteration.
"""

# ---------------------------------------------------------------------------
# Core: lats (MCTS with grounded environment)
# ---------------------------------------------------------------------------
def lats(
    sub_task: str,
    context: dict[str, Any],
    environment,                   # REQUIRED — must be grounded Environment
    n_expansions: int = 3,
    n_simulations: int = 10,
    exploration: float = 1.41,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """
    LATS search for the final offer-strategy recommendation.

    Args:
        sub_task:       Natural-language description of what to recommend.
        context:        Offer/property context dict.
        environment:    Grounded Environment (REQUIRED — not optional here).
        n_expansions:   Candidates to generate per expansion.
        n_simulations:  Total MCTS iterations.
        exploration:    UCT exploration constant.
        artifacts_dir:  Where to save the JSON trace.

    Returns:
        {
          "best_thought": str,
          "best_score": float,
          "environment_feedback": EnvironmentFeedback,
          "reflections": [...],
          "mcts_visits": [...],
          "llm_calls": int,
          "latency_s": float,
        }
    """
    if environment is None:
        raise ValueError(
            "LATS requires a grounded Environment. "
            "Pass environment=Environment() — do not use None."
        )

    run_id = str(uuid.uuid4())[:8]
    start = time.time()
    llm_calls = 0
    reflections: list[str] = []
    mcts_log: list[dict] = []
    context_str = json.dumps(context, indent=2)

    root = MCTSNode(thought="ROOT")
    best_node: MCTSNode | None = None
    best_score: float = -1.0

    for sim in range(n_simulations):
        # ----------------------------------------------------------------
        # 1. SELECT — UCT walk from root
        # ----------------------------------------------------------------
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.uct(exploration))

        # ----------------------------------------------------------------
        # 2. EXPAND — generate candidates
        # ----------------------------------------------------------------
        reflection_text = "\n".join(reflections[-3:]) if reflections else "None yet."
        expand_resp = _groq.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=700,
            temperature=0.7,
            messages=[
                {"role": "system",
                 "content": _EXPAND_SYSTEM.format(n=n_expansions)},
                {"role": "user", "content": (
                    f"Sub-task: {sub_task}\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Parent thought:\n{node.thought or 'Start fresh'}\n\n"
                    f"Reflections from failed attempts:\n{reflection_text}\n\n"
                    f"Generate {n_expansions} distinct candidate strategies."
                )},
            ],
        )
        llm_calls += 1
        exp_text = expand_resp.choices[0].message.content.strip()

        try:
            if exp_text.startswith("```"):
                exp_text = exp_text.split("```")[1]
                if exp_text.startswith("json"):
                    exp_text = exp_text[4:]
            candidates = json.loads(exp_text)
            if not isinstance(candidates, list):
                candidates = [exp_text]
        except json.JSONDecodeError:
            candidates = [exp_text]

        # ----------------------------------------------------------------
        # 3. EVALUATE — REAL grounded check (not random)
        # ----------------------------------------------------------------
        for candidate in candidates[:n_expansions]:
            child = MCTSNode(thought=candidate, parent=node)
            node.children.append(child)

            state = {
                "property_id": context.get("property_id"),
                "offer_id": context.get("offer_id"),
                "proposed_action": context.get("proposed_action", "counter"),
                "proposed_price": context.get("proposed_price"),
                "seller_deadline_weeks": context.get("seller_deadline_weeks"),
                "financing_contingency_days": context.get("financing_contingency_days"),
                "acknowledges_tier1_risk": "tier 1" in candidate.lower()
                    or "high risk" in candidate.lower(),
            }
            feedback: EnvironmentFeedback = environment.evaluate(state)
            child.env_feedback = feedback
            score = feedback.score

            # ----------------------------------------------------------------
            # 4. REFLECT — verbal reflection for failed branches
            # ----------------------------------------------------------------
            if not feedback.success and feedback.details:
                failure_details = "\n".join(feedback.details)
                reflect_resp = _groq.chat.completions.create(
                    model=GROQ_MODEL,
                    max_tokens=150,
                    temperature=0.3,
                    messages=[
                        {"role": "system",
                         "content": _REFLECT_SYSTEM.format(
                             failure_details=failure_details
                         )},
                        {"role": "user", "content": (
                            f"Failed candidate:\n{candidate}"
                        )},
                    ],
                )
                llm_calls += 1
                reflection = reflect_resp.choices[0].message.content.strip()
                child.reflection = reflection
                reflections.append(reflection)

            # ----------------------------------------------------------------
            # 5. BACKPROP — update scores up the tree
            # ----------------------------------------------------------------
            current = child
            while current is not None:
                current.visits += 1
                current.total_score += score
                current = current.parent

            # Track best
            if score > best_score:
                best_score = score
                best_node = child

            mcts_log.append({
                "simulation": sim + 1,
                "candidate": candidate,
                "score": score,
                "success": feedback.success,
                "details": feedback.details,
                "reflection": child.reflection or None,
            })

    elapsed = round(time.time() - start, 2)
    best_thought = best_node.thought if best_node else ""
    best_feedback = best_node.env_feedback if best_node else None

    # Artifact
    trace = {
        "run_id": run_id,
        "algorithm": "lats",
        "sub_task": sub_task,
        "timestamp": datetime.utcnow().isoformat(),
        "config": {"n_expansions": n_expansions, "n_simulations": n_simulations,
                   "exploration": exploration},
        "llm_calls": llm_calls,
        "latency_s": elapsed,
        "best_thought": best_thought,
        "best_score": best_score,
        "reflections": reflections,
        "mcts_log": mcts_log,
        "environment_feedback": best_feedback.model_dump() if best_feedback else None,
    }
    _dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    _dir.mkdir(parents=True, exist_ok=True)
    trace_path = _dir / f"lats_{run_id}.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    return {
        "best_thought": best_thought,
        "best_score": best_score,
        "environment_feedback": best_feedback,
        "reflections": reflections,
        "mcts_visits": mcts_log,
        "llm_calls": llm_calls,
        "latency_s": elapsed,
        "trace_path": str(trace_path),
    }