"""
Short-Term Memory: Rolling Buffer + Scratchpad
===============================================
البافر: بيحفظ آخر N رسالة في الجلسة الحالية
الـ Scratchpad: منفصل - بيحفظ الـ sub-goal الحالي للـ agent
                محميش من الـ pruning
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Turn:
    """رسالة واحدة في الجلسة"""
    role: str           # "user" أو "agent" أو "tool"
    content: Any        # محتوى الرسالة
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    customer_id: int = None


class ShortTermMemory:
    """
    Rolling buffer + scratchpad للجلسة الحالية
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._buffer: deque[Turn] = deque(maxlen=max_turns)
        self._scratchpad: dict[int, str] = {}  # customer_id -> sub-goal

    # ----------------------------------------------------------------
    # Buffer
    # ----------------------------------------------------------------
    def add_turn(self, customer_id: int, role: str, content: Any) -> Turn:
        """اضيف turn جديد للبافر"""
        turn = Turn(role=role, content=content, customer_id=customer_id)
        self._buffer.append(turn)
        return turn

    def get_turns(self, customer_id: int) -> list[Turn]:
        """جيب كل الـ turns الخاصة بعميل معين"""
        return [t for t in self._buffer if t.customer_id == customer_id]

    def is_full(self, customer_id: int) -> bool:
        """البافر اتملى؟"""
        return len(self.get_turns(customer_id)) >= self.max_turns

    # ----------------------------------------------------------------
    # Scratchpad — منفصل ومحميش من الـ pruning
    # ----------------------------------------------------------------
    def set_goal(self, customer_id: int, goal: str) -> None:
        """حدد الـ sub-goal الحالي للعميل"""
        self._scratchpad[customer_id] = goal

    def get_goal(self, customer_id: int) -> str | None:
        """جيب الـ sub-goal الحالي"""
        return self._scratchpad.get(customer_id)

    def clear_goal(self, customer_id: int) -> None:
        """امسح الـ goal لما الـ sub-task يخلص"""
        self._scratchpad.pop(customer_id, None)