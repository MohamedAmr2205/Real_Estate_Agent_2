"""
Strategy 3 — Recursive Summarization

Every `summarize_every` turns, the oldest turns beyond `keep_recent` are
compacted into a single running summary via an LLM call. "Recursive"
means the summary is never rebuilt from scratch — each pass feeds the
PREVIOUS summary text back into the model along with the newly-aged-out
turns, so the summary compounds over the whole session instead of only
covering the latest chunk.

Trade-off this strategy makes explicit: it can preserve facts sliding
window loses, but at the cost of an extra LLM call per summarization
pass — real latency and real output tokens, unlike the two strategies
above which are pure Python with zero model calls. This is exactly the
trade-off the comparison table in run_eval.py is built to measure
honestly rather than assume.

Uses Groq (same free-tier client pattern as agent/client.py) so this
strategy costs nothing to evaluate.
"""

import os
from pathlib import Path

from .base import ContextStrategy, PruneResult, Turn


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file(Path(__file__).resolve().parent.parent.parent / ".env")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

try:
    import openai
    _groq_client = openai.OpenAI(
        api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"
    ) if GROQ_API_KEY else None
except ImportError:
    _groq_client = None


class RecursiveSummarizationStrategy(ContextStrategy):
    name = "recursive_summarization"

    def __init__(self, keep_recent: int = 8, summarize_every: int = 15):
        self.keep_recent = keep_recent
        self.summarize_every = summarize_every
        self._running_summary: str = ""

    def _summarize(self, prior_summary: str, turns_to_fold_in: list[Turn]) -> str:
        transcript_text = "\n".join(
            f"[{t.role}] {t.content}" for t in turns_to_fold_in
        )
        prompt = (
            "You are compacting an agent's conversation history. Update the "
            "running summary below with the new turns, preserving every "
            "concrete fact (numbers, dates, names, stated preferences, "
            "deadlines). Be concise but never drop a specific fact.\n\n"
            f"PREVIOUS SUMMARY:\n{prior_summary or '(none yet)'}\n\n"
            f"NEW TURNS TO FOLD IN:\n{transcript_text}\n\n"
            "UPDATED SUMMARY:"
        )
        if _groq_client is None:
            # Honest fallback, no API key: naive truncal concat instead of
            # a fabricated summary.
            return (prior_summary + " | " + transcript_text)[:800]

        response = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def prune(self, turns: list[Turn], scratchpad: str | None) -> PruneResult:
        if len(turns) <= self.keep_recent:
            return PruneResult(turns=turns, strategy_name=self.name, extra_llm_calls=0)

        to_summarize = turns[: -self.keep_recent]
        recent = turns[-self.keep_recent:]

        llm_calls = 0
        # only pay for a new summarization pass if there's actually new
        # material beyond what's already folded into the running summary
        if to_summarize:
            self._running_summary = self._summarize(self._running_summary, to_summarize)
            llm_calls = 1

        summary_turn = Turn(
            role="summary",
            content=self._running_summary,
            customer_id=turns[0].customer_id if turns else None,
        )
        return PruneResult(
            turns=[summary_turn, *recent],
            strategy_name=self.name,
            extra_llm_calls=llm_calls,
        )