"""
Strategy 4 — Zone-Based Pruning

Splits the transcript into 4 zones and applies a different retention
rule to each, instead of one uniform rule across the whole session:

  Zone 1 — Opening (first `opening_size` turns): kept almost entirely.
           Early turns are where a client typically states budget,
           must-have features, and deadlines — exactly the facts our
           test suite plants there.
  Zone 2 — Early-middle: lightly thinned (keep every other turn).
  Zone 3 — Late-middle: heavily thinned (keep every 3rd turn) — this is
           usually the densest tool-call noise, once the agent has
           already acted on most of it.
  Zone 4 — Recent (last `recent_size` turns): kept entirely — the agent
           needs full fidelity on what just happened.

No LLM calls — this is pure Python, like sliding window and masking,
but structurally aware of WHERE in the conversation facts tend to live
instead of only looking at recency.
"""

from .base import ContextStrategy, PruneResult, Turn


class ZoneBasedPruningStrategy(ContextStrategy):
    name = "zone_based_pruning"

    def __init__(self, opening_size: int = 4, recent_size: int = 6):
        self.opening_size = opening_size
        self.recent_size = recent_size

    def prune(self, turns: list[Turn], scratchpad: str | None) -> PruneResult:
        n = len(turns)
        if n <= self.opening_size + self.recent_size:
            return PruneResult(turns=turns, strategy_name=self.name, extra_llm_calls=0)

        opening = turns[: self.opening_size]
        recent = turns[-self.recent_size:]
        middle = turns[self.opening_size: -self.recent_size]

        # split the middle into early-middle (lightly thinned) and
        # late-middle (heavily thinned)
        midpoint = len(middle) // 2
        early_middle = middle[:midpoint]
        late_middle = middle[midpoint:]

        kept_early_middle = early_middle[::2]   # keep every other turn
        kept_late_middle = late_middle[::3]     # keep every 3rd turn

        kept = [*opening, *kept_early_middle, *kept_late_middle, *recent]
        return PruneResult(turns=kept, strategy_name=self.name, extra_llm_calls=0)