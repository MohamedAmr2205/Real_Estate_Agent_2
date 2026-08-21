"""
state_graph/tot_strategy.py
============================
Tree of Thoughts — حقيقي بيستخدم LLM يولد branches ويقيمهم.
مش if/else — بيولد كل branch بـ LLM call منفصل.
"""

from __future__ import annotations
import os
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

try:
    import openai
    _client = openai.OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    ) if GROQ_API_KEY else None
except ImportError:
    _client = None

MODEL = "openai/gpt-oss-120b"

def _llm(prompt: str, max_tokens: int = 300) -> str:
    """Call LLM — falls back to heuristic if no API key."""
    if _client is None:
        return json.dumps({"fallback": True})
    resp = _client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------
# ToT core
# ----------------------------------------------------------------

def _generate_branch(offered: float, original: float,
                     branch_name: str) -> dict[str, Any]:
    """Ask LLM to evaluate ONE negotiation strategy branch."""
    ratio = offered / original if original else 0
    prompt = (
        f"Real estate negotiation analysis.\n"
        f"Property list price: {original:,.0f} EGP\n"
        f"Buyer offer: {offered:,.0f} EGP ({ratio*100:.1f}% of list price)\n\n"
        f"Evaluate the strategy: '{branch_name}'\n"
        f"Return ONLY valid JSON with keys:\n"
        f"  name (str), rationale (str, max 30 words), "
        f"score (float 0-1), recommended_counter (float)\n"
        f"No markdown, no explanation outside the JSON."
    )
    raw = _llm(prompt, max_tokens=150)
    try:
        # strip markdown fences if present
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except Exception:
        # heuristic fallback
        fallback_scores = {
            "Aggressive_Counter": 0.6 if ratio < 0.85 else 0.3,
            "Compromise_Split":   0.9 if 0.85 <= ratio <= 0.95 else 0.5,
            "Direct_Acceptance":  0.95 if ratio > 0.95 else 0.1,
        }
        return {
            "name": branch_name,
            "rationale": "Heuristic fallback (no LLM response)",
            "score": fallback_scores.get(branch_name, 0.5),
            "recommended_counter": offered * 1.05,
        }


def run_tot(offered: float, original: float,
            branches: list[str] | None = None) -> dict[str, Any]:
    """
    Tree of Thoughts over negotiation strategies.

    1. Generate candidate branches (LLM call per branch)
    2. Self-evaluate each branch (score included in generation)
    3. BFS: keep all, pick highest score
    4. Return best strategy + full tree for tracing
    """
    if branches is None:
        branches = ["Aggressive_Counter", "Compromise_Split", "Direct_Acceptance"]

    print(f"[ToT] Evaluating {len(branches)} strategy branches...")
    evaluated = []
    for branch in branches:
        result = _generate_branch(offered, original, branch)
        print(f"  [ToT] branch={result.get('name')} "
              f"score={result.get('score', 0):.2f} "
              f"counter={result.get('recommended_counter', 0):,.0f}")
        evaluated.append(result)

    # BFS pruning — keep best score
    best = max(evaluated, key=lambda x: x.get("score", 0))
    print(f"[ToT] Best branch: {best.get('name')} "
          f"(score={best.get('score', 0):.2f})")

    return {
        "evaluated_strategies": evaluated,
        "chosen_strategy": best.get("name"),
        "recommended_counter": best.get("recommended_counter", offered),
        "tot_rationale": best.get("rationale", ""),
    }


# ----------------------------------------------------------------
# LangGraph node wrapper
# ----------------------------------------------------------------

def tot_strategy_node(state: dict) -> dict:
    """
    Drop-in replacement for the old tot_strategy_node.
    Uses real LLM calls instead of hardcoded scores.
    """
    offered = state.get("offered_price", 0)
    original = state.get("original_price", 1)

    result = run_tot(offered, original)
    return result