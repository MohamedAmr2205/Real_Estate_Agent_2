"""
Promote-or-Drop Router
=======================
لما البافر يتملى، كل رسالة بتتحكم فيها:
- Promote  → روح Episodic Store (معلومة مهمة)
- Drop     → اتنسى (small talk)
"""

try:
    from .short_term import Turn
except ImportError:  # Fallback for direct execution
    from short_term import Turn

# الكلمات اللي بتدل على معلومة مهمة
PROMOTE_KEYWORDS = [
    # ميزانية
    "budget", "price", "afford", "max", "million",
    # رفض
    "rejected", "don't like", "dislike", "not interested", "already seen",
    # مواعيد
    "deadline", "close by", "before", "lease ends", "must close",
    # متطلبات
    "bedroom", "bathroom", "garage", "garden", "floor",
]

# الكلمات اللي بتدل على small talk
DROP_KEYWORDS = [
    "hello", "hi", "thanks", "thank you", "ok", "okay",
    "sure", "great", "bye", "goodbye", "sounds good",
]


def _should_promote(turn: Turn) -> tuple[bool, str]:
    """
    قرر: الـ turn ده مهم ولا لأ؟
    بيرجع (True/False, السبب)
    """
    content = str(turn.content).lower()

    # لو tool output → دايماً مهم
    if turn.role == "tool":
        return True, "tool output always promoted"

    # لو فيه keyword مهم
    for keyword in PROMOTE_KEYWORDS:
        if keyword in content:
            return True, f"contains important keyword: '{keyword}'"

    # لو فيه keyword تافه
    for keyword in DROP_KEYWORDS:
        if keyword in content:
            return False, f"small talk: '{keyword}'"

    # لو مش واضح → احتفظ بيه
    return True, "uncertain — promoted by default"


def route_turn(turn: Turn) -> dict:
    """
    خد turn وقرر مصيره.
    بيرجع dict فيه القرار والسبب.
    """
    promote, reason = _should_promote(turn)

    decision = {
        "turn": turn,
        "action": "promote" if promote else "drop",
        "reason": reason,
    }

    # لوج القرار
    print(f"[ROUTER] customer={turn.customer_id} "
          f"role={turn.role} "
          f"action={decision['action']} "
          f"reason={reason}")

    return decision


def route_overflow(turns: list[Turn]) -> tuple[list[Turn], list[Turn]]:
    """
    خد list من الـ turns القديمة وقسمها:
    - promoted: هتروح Episodic Store
    - dropped: هتتنسى
    """
    promoted = []
    dropped = []

    for turn in turns:
        decision = route_turn(turn)
        if decision["action"] == "promote":
            promoted.append(turn)
        else:
            dropped.append(turn)

    print(f"[ROUTER] total={len(turns)} "
          f"promoted={len(promoted)} "
          f"dropped={len(dropped)}")

    return promoted, dropped