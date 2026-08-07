"""
Consolidation Layer
====================
بيشتغل بشكل دوري — بيراجع الـ episodic store
ويحول الأحداث المهمة لحقائق في الـ semantic store.
مش بيشتغل في real-time — ده pass منفصل.

الفرق عن الـ router:
- الـ router بيقرر: promote أو drop (مش بيكتب في semantic)
- الـ consolidation بس اللي بيكتب في semantic
- الـ consolidation بيشتغل دوري مش في كل turn
"""

import re

try:
    from .episodic_store import EpisodicStore
    from .semantic_store import SemanticStore
except ImportError:
    from episodic_store import EpisodicStore
    from semantic_store import SemanticStore


# ----------------------------------------------------------------
# Patterns بتدل على معلومات مهمة
# ----------------------------------------------------------------
BUDGET_PATTERNS = [
    r"budget\s+is\s+([\d,.]+[mM]?)",
    r"can\s+go\s+to\s+([\d,.]+[mM]?)",
    r"max\s+is\s+([\d,.]+[mM]?)",
    r"offer_amount[\"']?\s*:\s*([\d]+)",
    r"actually\s+my\s+budget\s+is\s+([\d,.]+)",
]

DEADLINE_PATTERNS = [
    r"before\s+(\w+\s+\d+)",
    r"close\s+by\s+(\w+\s+\d+)",
    r"lease\s+ends\s+(\w+\s+\d+)",
    r"must\s+close\s+(\w+\s+\d+)",
]


def _extract_budget(content: str) -> str | None:
    for pattern in BUDGET_PATTERNS:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")
    return None


def _extract_deadline(content: str) -> str | None:
    for pattern in DEADLINE_PATTERNS:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def consolidate(customer_id: int,
                episodic: EpisodicStore,
                semantic: SemanticStore) -> None:
    """
    Pass دوري — بيراجع كل الـ episodes للعميل
    ويحدث الـ semantic store بالحقائق المستخرجة.

    التحسين الأساسي:
    - بيجمع كل القيم الأول من كل الـ episodes
    - بياخد الأخيرة بس (الأحدث)
    - بيكتب في semantic بس لو القيمة اتغيرت فعلاً
    - ده بيمنع التكرار في الـ version history
    """
    print(f"\n[CONSOLIDATION] starting pass for customer={customer_id}")

    episodes = episodic.get(customer_id)
    if not episodes:
        print(f"[CONSOLIDATION] no episodes found for customer={customer_id}")
        return

    # ----------------------------------------------------------------
    # اجمع كل القيم من كل الـ episodes
    # ----------------------------------------------------------------
    budgets = []
    deadlines = []

    for episode in episodes:
        content = str(episode.content)

        budget = _extract_budget(content)
        if budget:
            budgets.append((budget, episode.timestamp))

        deadline = _extract_deadline(content)
        if deadline:
            deadlines.append((deadline, episode.timestamp))

    # ----------------------------------------------------------------
    # اكتب الأحدث بس — وبس لو اتغيرت عن القيمة الحالية
    # ----------------------------------------------------------------
    if budgets:
        # الأخيرة في القائمة = الأحدث
        latest_budget, ts = budgets[-1]
        current = semantic.get_fact(customer_id, "budget")

        if current != latest_budget:
            print(f"[CONSOLIDATION] budget changed: '{current}' → '{latest_budget}'")
            semantic.set_fact(customer_id, "budget", latest_budget)
        else:
            print(f"[CONSOLIDATION] budget unchanged: '{current}' — skipping write")

    if deadlines:
        latest_deadline, ts = deadlines[-1]
        current = semantic.get_fact(customer_id, "deadline")

        if current != latest_deadline:
            print(f"[CONSOLIDATION] deadline changed: '{current}' → '{latest_deadline}'")
            semantic.set_fact(customer_id, "deadline", latest_deadline)
        else:
            print(f"[CONSOLIDATION] deadline unchanged: '{current}' — skipping write")

    # ----------------------------------------------------------------
    # اطبع ملخص الحقائق الحالية
    # ----------------------------------------------------------------
    all_facts = semantic.get_all(customer_id)
    print(f"[CONSOLIDATION] done — current facts for customer={customer_id}: {all_facts}\n")