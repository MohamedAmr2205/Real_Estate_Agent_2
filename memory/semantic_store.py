"""
Semantic Store
==============
بيحفظ حقائق ثابتة عن العميل مع versioning —
مش بيمسح القديم، بيضيف version جديدة.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Fact:
    """حقيقة واحدة عن العميل"""
    customer_id: int
    key: str
    value: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    superseded: bool = False  # True = دي version قديمة


class SemanticStore:

    def __init__(self):
        self._store: list[Fact] = []

    def set_fact(self, customer_id: int, key: str, value: str) -> Fact:
        """
        احفظ حقيقة جديدة —
        لو في version قديمة، اعملها superseded مش بتمسحها
        """
        # mark القديمة كـ superseded
        for fact in self._store:
            if fact.customer_id == customer_id and fact.key == key and not fact.superseded:
                fact.superseded = True
                print(f"[SEMANTIC] versioned old fact: {key}='{fact.value}' → superseded")

        # ضيف الجديدة
        new_fact = Fact(customer_id=customer_id, key=key, value=value)
        self._store.append(new_fact)
        print(f"[SEMANTIC] new fact: customer={customer_id} {key}='{value}'")
        return new_fact

    def get_fact(self, customer_id: int, key: str) -> str | None:
        """جيب الـ version الحالية"""
        for fact in reversed(self._store):
            if fact.customer_id == customer_id and fact.key == key and not fact.superseded:
                return fact.value
        return None

    def get_history(self, customer_id: int, key: str) -> list[Fact]:
        """جيب كل الـ versions (القديمة والحالية)"""
        return [f for f in self._store
                if f.customer_id == customer_id and f.key == key]

    def get_all(self, customer_id: int) -> dict:
        """جيب كل الحقائق الحالية للعميل"""
        result = {}
        for fact in self._store:
            if fact.customer_id == customer_id and not fact.superseded:
                result[fact.key] = fact.value
        return result