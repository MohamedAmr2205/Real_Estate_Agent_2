"""
Runs all 4 context management strategies against the frozen long-context
test suite and produces the comparison table (accuracy, tokens, latency)
that justifies which strategy we actually ship.

For each (strategy, transcript) pair:
  1. Prune the transcript's turns (everything except the final query turn).
  2. Ask the client's own model (Groq) to answer the query using ONLY the
     pruned context — this measures whether the strategy actually
     preserved the critical fact, not just whether it exists somewhere
     in the raw transcript.
  3. Check whether the expected keyword appears in the model's answer.
  4. Record token estimates and latency.

Token counts are ESTIMATED (chars / 4) since we don't have a tokenizer
library in this environment — documented here rather than presented as
exact. Latency is real wall-clock time.

Run with:
    python context_eval/run_eval.py
Requires GROQ_API_KEY in .env (same key used elsewhere in this project).
Falls back to a keyword-presence check on the pruned context (no live
model call) if no key is configured, clearly labeled in the output.
"""

import json
import os
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memory.short_term import Turn

from context_eval.strategies.sliding_window import SlidingWindowStrategy
from context_eval.strategies.observation_masking import ObservationMaskingStrategy
from context_eval.strategies.recursive_summarization import RecursiveSummarizationStrategy
from context_eval.strategies.zone_based_pruning import ZoneBasedPruningStrategy


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file(Path(__file__).resolve().parent.parent / ".env")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

try:
    import openai
    _groq_client = openai.OpenAI(
        api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"
    ) if GROQ_API_KEY else None
except ImportError:
    _groq_client = None


def estimate_tokens(text: str) -> int:
    """Rough chars/4 estimate — no tokenizer dependency in this project."""
    return max(1, len(text) // 4)


def load_transcripts(data_dir: Path) -> list[dict]:
    records = []
    for path in sorted(data_dir.glob("transcript_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def turns_from_record(record: dict) -> list[Turn]:
    return [Turn(**t) for t in record["turns"]]


def format_context(turns: list[Turn]) -> str:
    return "\n".join(f"[{t.role}] {str(t.content)[:300]}" for t in turns)


def generate_answer(context_text: str, query: str) -> tuple[str, float]:
    """Returns (answer_text, latency_seconds)."""
    prompt = (
        "Answer the user's question using ONLY the conversation context "
        "below. If the answer isn't in the context, say so explicitly.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION: {query}\nANSWER:"
    )
    start = time.perf_counter()
    if _groq_client is None:
        # Honest fallback: no live model call, just report whether the
        # fact string-matches somewhere in the pruned context.
        answer = context_text  # accuracy check below will search this directly
    else:
        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
    latency = time.perf_counter() - start
    return answer, latency


STRATEGY_FACTORIES = {
    "sliding_window": lambda: SlidingWindowStrategy(window_size=10),
    "observation_masking": lambda: ObservationMaskingStrategy(keep_last_n_tool_outputs=3),
    "recursive_summarization": lambda: RecursiveSummarizationStrategy(keep_recent=8, summarize_every=15),
    "zone_based_pruning": lambda: ZoneBasedPruningStrategy(opening_size=4, recent_size=6),
}


def run_evaluation(data_dir: Path) -> dict:
    transcripts = load_transcripts(data_dir)
    results = {}

    for strat_name, factory in STRATEGY_FACTORIES.items():
        correct = 0
        input_tok_total = 0
        output_tok_total = 0
        latency_total = 0.0

        for record in transcripts:
            turns = turns_from_record(record)
            context_turns, query_turn = turns[:-1], turns[-1]
            query = query_turn.content
            expected = record["expected_keyword"]

            strategy = print(f"  Running {strat_name} on transcript {record['seed']}...")
            time.sleep(2)
            strategy = factory()  # fresh instance per transcript — recursive()  # fresh instance per transcript — recursive
                                   # summarization must not carry state across
                                   # different customers' sessions
            t0 = time.perf_counter()
            prune_result = strategy.prune(context_turns, scratchpad="finding the right property")
            prune_latency = time.perf_counter() - t0

            context_text = format_context(prune_result.turns)
            answer_text, answer_latency = generate_answer(context_text, query)

            is_correct = expected.lower() in answer_text.lower()
            correct += int(is_correct)

            input_tokens = estimate_tokens(context_text + query)
            output_tokens = estimate_tokens(answer_text)
            if prune_result.extra_llm_calls:
                # the summarization pass itself used max_tokens=300
                output_tokens += 300 * prune_result.extra_llm_calls

            input_tok_total += input_tokens
            output_tok_total += output_tokens
            latency_total += prune_latency + answer_latency

        n = len(transcripts)
        results[strat_name] = {
            "correct": correct,
            "total": n,
            "avg_input_tokens": round(input_tok_total / n),
            "avg_output_tokens": round(output_tok_total / n),
            "avg_latency_s": round(latency_total / n, 2),
        }

    return results


def render_markdown_table(results: dict) -> str:
    lines = [
        "| Strategy | Detail recalled correctly | Avg. input tokens/run | Avg. output tokens/run | Avg. latency |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['correct']}/{r['total']} | {r['avg_input_tokens']} "
            f"| {r['avg_output_tokens']} | {r['avg_latency_s']}s |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "test_transcripts" / "data"
    if not _groq_client:
        print("[NOTE] No GROQ_API_KEY found — running in fallback mode "
              "(accuracy checks the pruned context directly instead of a "
              "live model answer). Set GROQ_API_KEY in .env for the real "
              "evaluation this table should be built from.\n")

    results = run_evaluation(data_dir)
    table = render_markdown_table(results)
    print(table)

    out_path = Path(__file__).resolve().parent / "results.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Context Management Strategy Comparison\n\n")
        f.write(table + "\n")
    print(f"\nWritten to {out_path}")