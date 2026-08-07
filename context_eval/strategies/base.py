"""
Shared interface for all 4 context window management strategies.

Every strategy takes the SAME input shape (a long list of Turn objects
from memory.short_term, plus the untouchable scratchpad) and returns the
SAME output shape (a condensed list of Turn objects to actually send to
the model). This is what makes it possible to run all four against the
same test suite and produce one comparison table — the eval script
doesn't need to know which strategy it's calling.

CRITICAL RULE (per the lab spec): none of these strategies may touch or
drop the scratchpad. The scratchpad is Person 1's (memory/short_term.py)
protected working state — pruning the transcript must never destroy it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from memory.short_term import Turn


@dataclass
class PruneResult:
    """What a strategy hands back to the eval harness."""
    turns: list[Turn]          # the condensed context actually sent to the model
    strategy_name: str
    extra_llm_calls: int = 0   # strategies that call an LLM (e.g. summarization)
                                # report it here so the eval script can count
                                # the real cost, not just token counts


class ContextStrategy(ABC):
    """Base class every strategy subclasses."""

    name: str = "base"

    @abstractmethod
    def prune(self, turns: list[Turn], scratchpad: str | None) -> PruneResult:
        """
        turns: the FULL rolling buffer for one customer session (can be
               long — this is the thing being pruned).
        scratchpad: the current sub-goal text (read-only context for the
               strategy to use if helpful — e.g. zone-based pruning may
               keep turns relevant to the active goal — but never
               mutated or dropped by any strategy).

        Returns a PruneResult with the condensed turns that would
        actually be sent to the model as context.
        """
        raise NotImplementedError