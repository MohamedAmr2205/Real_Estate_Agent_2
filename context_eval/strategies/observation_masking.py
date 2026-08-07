"""
Strategy 2 — Observation & Tool-Output Masking

Insight: in a tool-heavy agent (like ours — generate_cma, search_properties,
search_knowledge_base all return large JSON), the transcript bloat is
almost entirely TOOL OUTPUT, not dialogue. A negotiation call has maybe
20 short human/agent turns but could carry 15 large CMA/offer-history
JSON blobs.

This strategy keeps ALL user/agent turns (the actual conversation is
cheap and often where critical facts like a budget change live), but
masks — replaces with a short placeholder — any tool-output turn older
than `keep_last_n_tool_outputs`. Recent tool outputs stay in full (the
agent likely still needs the most recent CMA numbers); old ones are
masked because their raw content is rarely needed again once the agent
has already acted on them.

This is the strategy the lab's own worked example picked as the winner
for a tool-heavy vet system — we expect something similar here, since
our transcripts have the same shape (dialogue carries the facts, tools
carry the bulk).
"""

from .base import ContextStrategy, PruneResult, Turn


class ObservationMaskingStrategy(ContextStrategy):
    name = "observation_masking"

    def __init__(self, keep_last_n_tool_outputs: int = 3):
        self.keep_last_n_tool_outputs = keep_last_n_tool_outputs

    def prune(self, turns: list[Turn], scratchpad: str | None) -> PruneResult:
        tool_indices = [i for i, t in enumerate(turns) if t.role == "tool"]
        # the most recent N tool-output turns stay untouched
        keep_full_indices = set(tool_indices[-self.keep_last_n_tool_outputs:])

        masked_turns: list[Turn] = []
        for i, turn in enumerate(turns):
            if turn.role == "tool" and i not in keep_full_indices:
                masked_turns.append(
                    Turn(
                        role="tool",
                        content=f"[masked tool output — {len(str(turn.content))} chars omitted]",
                        timestamp=turn.timestamp,
                        customer_id=turn.customer_id,
                    )
                )
            else:
                masked_turns.append(turn)

        return PruneResult(turns=masked_turns, strategy_name=self.name, extra_llm_calls=0)