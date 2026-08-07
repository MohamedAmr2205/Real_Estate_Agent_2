# Context Evaluation

## Overview

This module evaluates different context window management strategies for long-running AI agent conversations. The goal is to identify which strategy preserves important information while minimizing token usage and latency.

Each strategy receives the same conversation history and returns a pruned version of the context that will be sent to the language model. All strategies implement the same interface, making them directly comparable.

---

## Implemented Strategies

### 1. Sliding Window

Keeps only the most recent conversation turns.

**Advantages**
- Very fast.
- No additional LLM calls.
- Lowest latency.

**Disadvantages**
- Loses important information from earlier in the conversation.
- Performs poorly when critical facts appear early.

---

### 2. Observation Masking

Preserves the complete user-agent conversation while replacing older tool outputs with lightweight placeholders.

Recent tool outputs remain unchanged.

**Advantages**
- Preserves important conversational facts.
- Significantly reduces context size.
- No extra LLM calls.

**Best suited for**
- Tool-heavy agents where JSON responses dominate the context.

---

### 3. Recursive Summarization

Maintains a running summary of older conversation history using an LLM.

Older turns are recursively summarized while the newest turns remain unchanged.

**Advantages**
- Preserves long-term information.
- Lowest average input token usage among the accurate strategies.

**Disadvantages**
- Requires additional LLM calls.
- Highest latency.

---

### 4. Zone-Based Pruning

Divides the conversation into four regions:

- Opening
- Early Middle
- Late Middle
- Recent

Different pruning rules are applied to each section.

**Advantages**
- Preserves important opening information.
- Keeps recent conversation intact.
- No extra LLM calls.

---

## Evaluation Process

Each strategy is evaluated using the same benchmark.

For every transcript:

1. Load the conversation history.
2. Remove the final user question.
3. Apply the pruning strategy.
4. Build the reduced context.
5. Ask the Groq model to answer using only the reduced context.
6. Compare the generated answer against the expected keyword.
7. Measure:
   - Recall Accuracy
   - Input Tokens
   - Output Tokens
   - End-to-End Latency

Token counts are estimated using:

```
estimated_tokens = characters / 4
```

Latency is measured using real execution time.

---

## Test Suite

The evaluation uses **10 synthetic long-conversation transcripts**.

Each transcript contains:

- Early conversation facts
- Long dialogue history
- Tool outputs
- Final query requiring long-term memory

This benchmark evaluates whether a pruning strategy preserves the information required to answer the final question.

---

## Results

| Strategy | Recall | Avg Input Tokens | Avg Output Tokens | Avg Latency |
|-----------|-------:|-----------------:|------------------:|------------:|
| Sliding Window | 0/10 | 499 | 39 | 0.30 s |
| Observation Masking | 10/10 | 819 | 26 | 7.54 s |
| Recursive Summarization | 10/10 | **480** | 329 | 23.87 s |
| Zone-Based Pruning | 10/10 | 1168 | 26 | 13.15 s |

---

## Discussion

### Sliding Window

Although it achieved the lowest latency, it failed to retain long-term information and therefore could not correctly answer any benchmark question.

### Observation Masking

Successfully preserved all required information while maintaining relatively low latency and moderate token usage.

### Recursive Summarization

Produced perfect recall with the smallest average input context, but required additional LLM calls, increasing execution time.

### Zone-Based Pruning

Also achieved perfect recall without additional summarization, though it retained a larger context than the other successful approaches.

---

## Selected Strategy

**Observation Masking** was selected as the preferred context management strategy.

Reasons:

- Perfect recall accuracy (10/10)
- No additional summarization cost
- Lower latency than Recursive Summarization
- Maintains conversational context while reducing unnecessary tool outputs
- Well suited for tool-based AI agents

---

## Running the Evaluation

```bash
python context_eval/run_eval.py
```

The script automatically:

- Loads the benchmark transcripts
- Runs all four strategies
- Evaluates each strategy using Groq
- Computes token estimates
- Measures latency
- Generates:

```
context_eval/results.md
```

---

## Project Structure

```
context_eval/
│
├── run_eval.py
├── results.md
├── strategies/
│   ├── base.py
│   ├── sliding_window.py
│   ├── observation_masking.py
│   ├── recursive_summarization.py
│   └── zone_based_pruning.py
│
└── test_transcripts/
```

---

## Future Improvements

- Evaluate on larger real-world datasets.
- Use tokenizer-based token counting instead of estimation.
- Add semantic similarity metrics.
- Evaluate memory quality across conversations with hundreds of turns.
- Benchmark additional context compression strategies.