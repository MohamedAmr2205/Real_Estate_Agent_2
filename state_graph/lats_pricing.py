"""
state_graph/lats_pricing.py
============================
LATS — Language Agent Tree Search للتقييم الحقيقي لأسعار العقارات.

الـ 4 phases الحقيقية:
  1. Select   — اختار أحسن node حسب UCB score
  2. Expand   — ولد child branches بـ LLM
  3. Evaluate — grounded feedback من الـ DB (مش random)
  4. Backpropagate — حدّث scores للـ ancestors

الـ environment feedback حقيقي — بيشوف الـ DB الفعلي.
"""

from __future__ import annotations

import math
import os
import sys
import json
import sqlite3
from dataclasses import dataclass, field
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
DB_PATH = ROOT / "db" / "database.sqlite"
try:
    import openai
    _client = openai.OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    ) if GROQ_API_KEY else None
except ImportError:
    _client = None

MODEL = "openai/gpt-oss-120b"

def _llm(prompt: str, max_tokens: int = 200) -> str:
    if _client is None:
        return "{}"
    resp = _client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------
# Grounded Environment — بيشوف DB حقيقي
# ----------------------------------------------------------------

def grounded_environment_feedback(plan: dict[str, Any]) -> dict[str, Any]:
    """
    REAL environment feedback — مش randomized.
    بيشيك على:
    1. السعر المقترح vs متوسط أسعار DB
    2. Marketing strategy feasibility
    3. هل الـ price في range معقول؟

    Returns: {score: float, feedback: str, passed: bool}
    """
    price = plan.get("price", 0)
    marketing = plan.get("marketing", "")

    # ── اسأل الـ DB عن متوسط الأسعار ──
    avg_price = _get_avg_price_from_db()
    if avg_price is None:
        avg_price = 3_000_000  # fallback لو DB مش موجود

    # ── Grounded checks ──
    ratio = price / avg_price if avg_price else 0

    feedback_parts = []
    score = 0.5

    # Check 1: Price range
    if 0.8 <= ratio <= 1.2:
        score += 0.3
        feedback_parts.append(f"Price {price:,.0f} is within market range "
                               f"(avg={avg_price:,.0f}, ratio={ratio:.2f})")
    elif ratio < 0.8:
        score -= 0.1
        feedback_parts.append(f"Price {price:,.0f} is BELOW market "
                               f"(avg={avg_price:,.0f}) — may undervalue asset")
    else:
        score -= 0.2
        feedback_parts.append(f"Price {price:,.0f} is ABOVE market "
                               f"(avg={avg_price:,.0f}) — may slow sale")

    # Check 2: Marketing strategy
    valid_strategies = ["Social_Ads", "Exclusive_Listing", "Open_House", "Agent_Network"]
    if marketing in valid_strategies:
        score += 0.2
        feedback_parts.append(f"Marketing strategy '{marketing}' is valid")
    else:
        score -= 0.1
        feedback_parts.append(f"Marketing strategy '{marketing}' is unrecognized")

    score = max(0.0, min(1.0, score))
    passed = score >= 0.6

    return {
        "score": round(score, 3),
        "feedback": " | ".join(feedback_parts),
        "passed": passed,
        "avg_market_price": avg_price,
    }


def _get_avg_price_from_db() -> float | None:
    """جيب متوسط أسعار العقارات من الـ DB الحقيقي."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(price) FROM Property WHERE status='Available'")
        result = cursor.fetchone()
        conn.close()
        return float(result[0]) if result and result[0] else None
    except Exception as e:
        print(f"[LATS] DB query failed: {e} — using fallback")
        return None


# ----------------------------------------------------------------
# MCTS Node
# ----------------------------------------------------------------

@dataclass
class MCTSNode:
    plan: dict[str, Any]
    parent: "MCTSNode | None" = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_score: float = 0.0
    reflection: str = ""

    @property
    def avg_score(self) -> float:
        return self.total_score / self.visits if self.visits else 0.0

    def ucb(self, exploration: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return self.avg_score + exploration * math.sqrt(
            math.log(parent_visits + 1) / self.visits
        )


# ----------------------------------------------------------------
# LATS core
# ----------------------------------------------------------------

def _expand_node(node: MCTSNode, property_details: dict) -> list[MCTSNode]:
    """LLM generates 2-3 child pricing plans from current plan."""
    prompt = (
        f"Real estate pricing strategy.\n"
        f"Property: {json.dumps(property_details, default=str)}\n"
        f"Current plan: {json.dumps(node.plan)}\n"
        f"Prior reflection: {node.reflection or 'None'}\n\n"
        f"Generate 3 alternative pricing and marketing plans.\n"
        f"Return ONLY a JSON array of objects, each with:\n"
        f"  price (int EGP), marketing (str), rationale (str max 20 words)\n"
        f"No markdown."
    )
    raw = _llm(prompt, max_tokens=300)
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        plans = json.loads(clean)
        if not isinstance(plans, list):
            raise ValueError("not a list")
    except Exception:
        # heuristic fallback
        base = node.plan.get("price", 2_500_000)
        plans = [
            {"price": int(base * 0.95), "marketing": "Social_Ads",
             "rationale": "Slight discount to attract more buyers"},
            {"price": int(base * 1.05), "marketing": "Exclusive_Listing",
             "rationale": "Premium price for exclusive market"},
            {"price": int(base), "marketing": "Open_House",
             "rationale": "Same price with broader exposure"},
        ]
    return [MCTSNode(plan=p, parent=node) for p in plans[:3]]


def _reflect(node: MCTSNode, feedback: dict) -> str:
    """Generate verbal reflection for failed branch."""
    prompt = (
        f"A real estate pricing plan failed evaluation.\n"
        f"Plan: {json.dumps(node.plan)}\n"
        f"Feedback: {feedback.get('feedback', '')}\n"
        f"Score: {feedback.get('score', 0)}\n\n"
        f"In one sentence, what should the next plan avoid or improve?"
    )
    return _llm(prompt, max_tokens=80).strip()


def run_lats(
    property_details: dict[str, Any],
    initial_plans: list[dict] | None = None,
    iterations: int = 3,
) -> dict[str, Any]:
    """
    LATS over property pricing strategies.

    Returns best plan + full MCTS trace.
    """
    if initial_plans is None:
        base_price = property_details.get("price", 2_500_000)
        initial_plans = [
            {"price": int(base_price * 0.9),  "marketing": "Social_Ads"},
            {"price": int(base_price * 1.05), "marketing": "Exclusive_Listing"},
        ]

    roots = [MCTSNode(plan=p) for p in initial_plans]
    all_nodes: list[MCTSNode] = list(roots)
    best_node: MCTSNode | None = None
    best_score = -1.0

    print(f"[LATS] Starting MCTS with {len(roots)} root nodes, "
          f"{iterations} iterations")

    for iteration in range(iterations):
        print(f"\n[LATS] Iteration {iteration + 1}/{iterations}")

        # 1. SELECT — UCB
        candidate = max(all_nodes, key=lambda n: n.ucb())
        print(f"  [SELECT] plan={candidate.plan} ucb={candidate.ucb():.3f}")

        # 2. EXPAND — LLM generates children
        children = _expand_node(candidate, property_details)
        candidate.children.extend(children)
        all_nodes.extend(children)

        # 3. EVALUATE — grounded environment feedback
        for child in children:
            fb = grounded_environment_feedback(child.plan)
            score = fb["score"]
            print(f"  [EVAL] price={child.plan.get('price'):,} "
                  f"score={score:.3f} passed={fb['passed']} "
                  f"| {fb['feedback'][:60]}")

            # Reflect on failures
            if not fb["passed"]:
                child.reflection = _reflect(child, fb)
                print(f"  [REFLECT] {child.reflection}")

            # 4. BACKPROPAGATE
            node = child
            while node is not None:
                node.visits += 1
                node.total_score += score
                node = node.parent

            if score > best_score:
                best_score = score
                best_node = child

    selected = best_node.plan if best_node else initial_plans[0]
    print(f"\n[LATS] Best plan: {selected} (score={best_score:.3f})")

    return {
        "lats_evaluated_trees": [
            {"plan": n.plan, "visits": n.visits,
             "avg_score": round(n.avg_score, 3),
             "reflection": n.reflection}
            for n in all_nodes
        ],
        "selected_listing_plan": selected,
        "lats_best_score": best_score,
    }


# ----------------------------------------------------------------
# LangGraph node wrapper
# ----------------------------------------------------------------

def lats_pricing_evaluator_node(state: dict) -> dict:
    """
    Drop-in replacement for the old hardcoded lats_pricing_evaluator_node.
    Uses real MCTS + grounded DB feedback.
    """
    property_details = state.get("property_details", {})
    result = run_lats(property_details, iterations=3)
    return result