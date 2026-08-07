"""
Long-context test suite generator.

Builds synthetic negotiation-call transcripts shaped like our real
agent's tool-heavy sessions: a critical fact stated early (a budget
change, a hard deadline, a rejected property) buried under many large
tool-output turns (CMA results, property search results), then a final
question that can only be answered correctly if that early fact
survived whatever pruning strategy was applied.

Per the lab's guardrail: input tokens are cheap, output tokens are
expensive — so we lean on LARGE REALISTIC INPUT (big fake tool-output
payloads) rather than trying to generate huge model output. All content
here is synthetic Python data, zero LLM calls to build the suite itself.

IMPORTANT: this suite is generated ONCE and then frozen to JSON files in
test_transcripts/data/. Do not regenerate after evaluation starts —
changing the test cases mid-evaluation invalidates the comparison table.
"""

import json
import random
from dataclasses import asdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from memory.short_term import Turn


# A bank of critical facts to plant — one per transcript variation, so
# the suite covers different KINDS of facts an agent must not lose.
CRITICAL_FACTS = [
    {
        "planted_statement": "Actually, my budget just went up to 4,200,000 after my bonus came through.",
        "query": "What is the client's current maximum budget?",
        "expected_keyword": "4,200,000",
    },
    {
        "planted_statement": "I need to close before August 15th — my current lease ends then.",
        "query": "What is the client's closing deadline?",
        "expected_keyword": "August 15",
    },
    {
        "planted_statement": "We already toured the Smouha villa last month and didn't like the noise from the road.",
        "query": "Has the client already seen and rejected any property? Which one and why?",
        "expected_keyword": "Smouha",
    },
    {
        "planted_statement": "My partner is severely allergic to mold, so any property with past water damage is a hard no.",
        "query": "Is there any health-related constraint on which properties we can show this client?",
        "expected_keyword": "mold",
    },
    {
        "planted_statement": "We're paying cash, no mortgage contingency needed.",
        "query": "Does this client need a financing contingency in any offer?",
        "expected_keyword": "cash",
    },
]


def _fake_cma_tool_output(i: int) -> dict:
    """A large, realistic-shaped fake CMA tool result to bloat the transcript."""
    return {
        "subject_property_id": random.randint(1, 5),
        "comparable_count": 8,
        "comparables": [
            {
                "property_id": 100 + j,
                "title": f"Comparable listing {j} (batch {i})",
                "price": random.randint(1_500_000, 5_500_000),
                "bedrooms": random.randint(2, 5),
                "bathrooms": random.randint(1, 4),
                "area_sqft": random.randint(900, 3200),
                "notes": "Recently renovated kitchen, close to transit, "
                         "quiet street, good school district access." * 2,
            }
            for j in range(8)
        ],
        "average_price_per_sqft": round(random.uniform(1200, 2100), 2),
    }


def build_transcript(fact_index: int, noise_turns: int, seed: int) -> list[Turn]:
    """
    Builds one full transcript:
      turn 0: greeting
      turn 1-2: small talk / context
      turn 3: THE CRITICAL FACT (planted)
      turns 4..N-2: tool-heavy noise (large CMA payloads + short agent replies)
      turn N-1: the query that requires recalling the planted fact
    """
    random.seed(seed)
    fact = CRITICAL_FACTS[fact_index % len(CRITICAL_FACTS)]
    customer_id = 1000 + seed

    turns: list[Turn] = [
        Turn(role="agent", content="Hi! Happy to help you find a property today.", customer_id=customer_id),
        Turn(role="user", content="Thanks, we've been looking for a while now.", customer_id=customer_id),
        Turn(role="agent", content="Let's go over what you're looking for.", customer_id=customer_id),
        Turn(role="user", content=fact["planted_statement"], customer_id=customer_id),
    ]

    for i in range(noise_turns):
        turns.append(Turn(role="tool", content=_fake_cma_tool_output(i), customer_id=customer_id))
        turns.append(Turn(
            role="agent",
            content=f"Checked comparable set {i} — a few options in range, let's keep looking.",
            customer_id=customer_id,
        ))

    turns.append(Turn(role="user", content=fact["query"], customer_id=customer_id))
    return turns


def build_and_freeze_suite(output_dir: Path, n_variations: int = 10) -> None:
    """
    Generates n_variations transcripts (varying which fact is planted and
    how much noise separates it from the query) and writes them to JSON
    so the suite is reproducible and frozen for the whole evaluation run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(n_variations):
        fact_index = seed % len(CRITICAL_FACTS)
        noise_turns = 12 + (seed % 4) * 4  # vary length: 12, 16, 20, 24 tool-noise pairs
        transcript = build_transcript(fact_index=fact_index, noise_turns=noise_turns, seed=seed)

        record = {
            "seed": seed,
            "fact_index": fact_index,
            "expected_keyword": CRITICAL_FACTS[fact_index]["expected_keyword"],
            "query": CRITICAL_FACTS[fact_index]["query"],
            "turns": [asdict(t) for t in transcript],
        }
        with open(output_dir / f"transcript_{seed:02d}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)

    print(f"Wrote {n_variations} frozen transcripts to {output_dir}")


if __name__ == "__main__":
    build_and_freeze_suite(Path(__file__).resolve().parent / "data", n_variations=10)