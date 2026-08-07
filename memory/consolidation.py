"""
Consolidation Layer
====================
بيشتغل بشكل دوري — بيراجع الـ episodic store
ويحول الأحداث المهمة لحقائق في الـ semantic store.
مش بيشتغل في real-time — ده pass منفصل.
"""

import re

try:
    from .episodic_store import EpisodicStore
    from .semantic_store import SemanticStore
except ImportError:  # Fallback for direct execution
    from episodic_store import EpisodicStore
    from semantic_store import SemanticStore


# Patterns بتدل على معلومات مهمة
BUDGET_PATTERNS = [
    r"budget\s+is\s+([\d,.]+[mM]?)",
    r"can\s+go\s+to\s+([\d,.]+[mM]?)",
    r"max\s+is\s+([\d,.]+[mM]?)",
    r"offer_amount[\"']?\s*:\s*([\d]+)",
]

REJECTION_PATTERNS = [
    r"property[_\s]+(\d+)",
    r"property_id[\"']?\s*:\s*(\d+)",
]

DEADLINE_PATTERNS = [
    r"before\s+(\w+\s+\d+)",
    r"close\s+by\s+(\w+\s+\d+)",
    r"lease\s+ends\s+(\w+\s+\d+)",
]


def _extract_budget(content: str) -> str | None:
    for pattern in BUDGET_PATTERNS:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)
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
    """
    print(f"\n[CONSOLIDATION] starting pass for customer={customer_id}")

    episodes = episodic.get(customer_id)
    if not episodes:
        print(f"[CONSOLIDATION] no episodes found for customer={customer_id}")
        return

    for episode in episodes:
        content = str(episode.content)

        # استخرج الميزانية
        budget = _extract_budget(content)
        if budget:
            semantic.set_fact(customer_id, "budget", budget)

        # استخرج الـ deadline
        deadline = _extract_deadline(content)
        if deadline:
            semantic.set_fact(customer_id, "deadline", deadline)

    # اطبع ملخص الحقائق الحالية
    all_facts = semantic.get_all(customer_id)
    print(f"[CONSOLIDATION] done — current facts for customer={customer_id}: {all_facts}\n")