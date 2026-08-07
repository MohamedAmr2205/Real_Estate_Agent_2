"""
Strategy 1 — Sliding Window

The simplest possible strategy: keep only the last N turns, drop
everything older. No LLM calls, near-zero latency — but anything
important said before the window closes is gone permanently.

This is the strategy we EXPECT to perform worst on our test suite: our
transcripts plant a critical fact early (e.g. a budget change on turn 3)
and ask about it on a late turn (turn 30+). A window of 10 recent turns
has no way to see turn 3 anymore. We keep it in the comparison as the
baseline everything else has to beat.
"""

from .base import ContextStrategy, PruneResult, Turn


class SlidingWindowStrategy(ContextStrategy):
    name = "sliding_window"

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def prune(self, turns: list[Turn], scratchpad: str | None) -> PruneResult:
        kept = turns[-self.window_size:] if len(turns) > self.window_size else turns
        return PruneResult(turns=kept, strategy_name=self.name, extra_llm_calls=0)