"""
Episodic Store
==============
بيحفظ الـ turns المهمة per customer_id.
في المرحلة دي بنستخدم list عادية —
الـ vector DB هييجي بعدين.
"""

from dataclasses import dataclass, field
from datetime import datetime

try:
    from .short_term import Turn
except ImportError:  # Fallback for direct execution
    from short_term import Turn


@dataclass
class Episode:
    """حدث مهم اتحفظ من الـ router"""
    customer_id: int
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reason: str = ""  # سبب الـ promote من الـ router


class EpisodicStore:
    """
    بيحفظ episodes مفلترة بـ customer_id
    """

    def __init__(self):
        self._store: list[Episode] = []

    def add(self, turn: Turn, reason: str = "") -> Episode:
        """احفظ turn كـ episode"""
        episode = Episode(
            customer_id=turn.customer_id,
            role=turn.role,
            content=str(turn.content),
            timestamp=turn.timestamp,
            reason=reason,
        )
        self._store.append(episode)
        print(f"[EPISODIC] saved episode for customer={turn.customer_id} "
              f"reason='{reason}'")
        return episode

    def get(self, customer_id: int) -> list[Episode]:
        """جيب كل الـ episodes الخاصة بعميل معين"""
        return [e for e in self._store if e.customer_id == customer_id]

    def summary(self, customer_id: int) -> str:
        """ملخص نصي للـ episodes عشان الـ agent يقراه"""
        episodes = self.get(customer_id)
        if not episodes:
            return "No past episodes for this customer."
        lines = []
        for e in episodes:
            lines.append(f"[{e.timestamp[:10]}] {e.role}: {e.content[:100]}")
        return "\n".join(lines)