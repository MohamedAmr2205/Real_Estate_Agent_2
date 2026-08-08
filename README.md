# Cornerstone Realty — MCP Server Lab

## The Company

Cornerstone Realty is a mid-sized real estate brokerage with agents and
brokers working across multiple properties, clients, and offer
negotiations at once. Agents want an AI assistant that can search
listings, schedule showings, draft compliant marketing copy, track
offers, and remember context across long negotiations — without giving
that assistant raw access to the underlying database, and without losing
critical facts as conversations grow long.

## The Problem

Real estate transactions involve real legal and financial stakes:

- **Accepting or submitting an offer** is a binding step with real money
  attached — not something that should happen silently or be reversible
  by mistake.
- **Only the listing agent (or a broker) may accept an offer** on a given
  property — any other agent doing so would be acting without authority.
- **Below-market offers** (significantly under asking price) carry real
  negotiation risk and should not be finalized without a broker's
  explicit sign-off.
- **Listing agent assignments change** — a broker may reassign who lists
  a property mid-session, and that agent's tool access should update
  immediately, not after a reconnect.
- **Fair Housing advertising compliance** is a legal requirement for any
  listing description a model drafts — this can't be left to chance.
- **Comparative Market Analyses (CMA)** take real time to compute
  (gathering comparable properties, computing statistics) and shouldn't
  leave the caller blocked with no feedback.
- **Negotiations run long.** A single customer's session can span dozens
  of turns and multiple large tool outputs (CMAs, offer histories, policy
  lookups) — the agent needs to manage that context without silently
  losing the budget the customer stated on turn 3 by the time they close
  on turn 40.
- **Policy questions need grounded, verified answers** — an agent that
  confidently states a made-up commission rate or closing deadline is a
  liability, not a convenience.

This is the exact shape of problem the lab asks for: real state changes,
real authorization boundaries, real memory-management trade-offs, and
genuine risk — not a "customer chatbot" with nothing at stake.

## Why this needs all 9 protocol concerns (8 required + Sampling)

| Concern | Why it's genuinely needed here |
|---|---|
| Capability negotiation | `submit_offer` depends on elicitation support; a client that never checked would silently assume it, exactly the failure mode this concern prevents. |
| Notifications | Listing agent reassignment is a real runtime event that changes who can call `accept_offer` on a property — polling would be unreliable and wasteful. |
| Elicitation | A below-threshold offer needs a real human (broker) decision before it becomes binding — this cannot be silently approved or silently dropped. |
| Resources | Fair Housing policy is reference data the model should read and reason over, not an action it performs. |
| Prompts | Listing descriptions and offer rejections are common, recurring drafting tasks worth a canned, parameterized starting point. |
| Transport | A single-office stdio process can't serve agents working from multiple Meridian office locations — Streamable HTTP is the real deployment target. |
| Progress tracking | CMA generation genuinely takes multiple steps (load subject, search comparables, compute stats) worth surfacing incrementally. |
| Defensive tool design | `accept_offer` is legally binding and must not be callable by just any authenticated agent — the authorization check has real teeth. |
| Sampling | Offer risk analysis benefits from the client's own model reasoning over the facts, without the server owning or running its own LLM. |

## Repository structure

```
Real_Estate_Agent_2/
├── README.md               — this file
├── .env.example
├── .gitignore
├── requirements.txt
├── db/                      — schema, seed data, ERD, engine documentation
├── mcp_server/              — the MCP server (tools, resources, prompts)
├── agent/                   — the MCP client that drives the server
├── demo/                    — full demo transcript covering every concern
├── memory/                  — short-term buffer, scratchpad, promote/drop
│                              router, episodic + semantic memory
├── context_eval/            — 4 context-window management strategies,
│                              long-context test suite, comparison table
├── rag/                     — vector store, embedding pipeline, 4
│                              retrieval architectures (naive, hybrid,
│                              agentic, graph), Self-RAG verification
└── retrieval_eval/          — domain-specific retrieval test set (12
                               queries) and comparison harness
```

Detailed documentation for each part lives in its own README where
available:
- [`db/README.md`](db/README.md) — engine choice, schema, seed data rationale
- [`mcp_server/README.md`](mcp_server/README.md) — exactly where each
  protocol concern lives in the code, tool read-only/write comparison,
  and capability fallback behavior
- [`agent/README.md`](agent/README.md) — how to run the client, and how
  it negotiates capabilities

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/MohamedAmr2205/Real_Estate_Agent_2
cd Real_Estate_Agent_2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the env template and add your keys
cp .env.example .env
#    GROQ_API_KEY   → free at https://console.groq.com (no credit card required)
#                      used by: agent/client.py sampling, recursive_summarization
#    GEMINI_API_KEY → https://aistudio.google.com/apikey
#                      used by: rag/ embeddings (naive/hybrid/agentic/graph retrieval)

# 4. Build the database
python db/build_db.py

# 5. Run the full agent demo (server is started automatically by the client)
python agent/client.py

# 6. Run the context-window management comparison (4 strategies)
python context_eval/run_eval.py

# 7. Run the retrieval architecture comparison (4 strategies incl. Graph RAG bonus)
python retrieval_eval/eval.py
```

See [`demo/transcript.md`](demo/transcript.md) for a full recorded run
showing every protocol concern firing end-to-end, including memory
routing, consolidation, and Self-RAG-verified knowledge base search.

## Tool comparison: read-only vs. write, and capability dependencies

| Tool | Type | Elicitation required? | Authorization required |
|---|---|---|---|
| `search_properties` | Read-only | No | None |
| `get_property` | Read-only | No | None |
| `generate_cma` | Read-only (long-running) | No | None |
| `submit_offer` | Write | Yes, if offer < 85% of list price | None |
| `accept_offer` | Write | No | Listing agent for that property, or Broker |
| `assign_listing_agent` | Write | No | Broker only |
| `explain_offer_risk` | Read-only (uses sampling) | No | None |
| `search_knowledge_base` | Read-only (RAG + Self-RAG) | No | Role-based section filtering (Broker sees more) |

**If a client connects without elicitation support:** it still sees
`submit_offer` and `accept_offer` in the server's declared tool list
(FastMCP doesn't conditionally hide tools per-client), but the client
itself is responsible for not calling them — `agent/client.py`
demonstrates this via the `SUPPORTS_ELICITATION` flag, which controls
whether the elicitation callback is registered at all.

## Context window management (`context_eval/`)

Four strategies for keeping a long negotiation transcript within budget
without losing facts the agent still needs — all sharing one interface
(`ContextStrategy.prune`) so they can be swapped and benchmarked
identically. None of them may touch or drop the scratchpad — pruning the
transcript must never destroy the customer's active sub-goal.

| Strategy | Idea | LLM calls |
|---|---|---|
| `sliding_window` | Keep only the last N turns | 0 |
| `observation_masking` | Keep all dialogue, mask old tool-output turns | 0 |
| `recursive_summarization` | Fold aged-out turns into a running LLM summary | 1 per pass |
| `zone_based_pruning` | Four retention zones (opening / early-mid / late-mid / recent) instead of one uniform rule | 0 |

Run `python context_eval/run_eval.py` for the full comparison table
(detail recall, avg input/output tokens, avg latency) against a frozen
10-transcript test suite, written to `context_eval/results.md`.

## Memory systems (`memory/`)

- **Short-term buffer + scratchpad** — a rolling buffer of recent turns
  plus a protected scratchpad holding the customer's active sub-goal,
  never mutated by pruning or routing.
- **Promote-or-drop router** — once the buffer fills, each turn is
  routed to either `drop` (small talk) or `promote` (kept — tool output
  is always promoted; turns containing important keywords like "budget"
  are promoted with a stated reason).
- **Episodic memory** — promoted turns are saved as episodes per
  customer.
- **Semantic memory consolidation** — a periodic pass (separate from the
  router) extracts durable facts (e.g. `budget`) from episodic memory,
  versions them on conflict (old value marked `superseded`, new value
  timestamped), and never writes to semantic memory directly from a
  single turn.
- **Self-RAG-verified recall** — both episodic and semantic recall are
  verified (`ISREL`/`ISSUP`-style checks) before being trusted, the same
  verification approach used for RAG retrieval below.

## Retrieval architectures (`rag/`, `retrieval_eval/`)

Four retrieval architectures share one interface (`RetrieverConfig` /
`RetrievalResult`) over a real vector store (`rag/vector_store.py`): a
cosine-similarity ANN index (HNSW via `hnswlib`, falling back to exact
brute-force search), a metadata payload store, and a metadata index used
for pre-search filtering — not a bare list of vectors.

| Architecture | Approach | Wins on |
|---|---|---|
| `naive` | Pure vector similarity, top-k | Simple factual lookups (Q01–Q03) |
| `hybrid` | Vector + BM25 via Reciprocal Rank Fusion, metadata pre-filtering | Exact-identifier and metadata queries (Q04–Q06, Q10) |
| `agentic` | Iterative LLM-guided re-querying across up to 3 rounds | Multi-hop questions needing cross-policy reasoning (Q07–Q09) |
| `graph` *(bonus)* | Single-shot vector seed + graph expansion over policy-code and domain-entity edges | Entity-linked multi-hop questions, with zero query-time LLM cost |

All four are verified with **Self-RAG-style checks** (`rag/verification.py`):
a post-retrieval relevance check (`ISREL`, query-coverage based) and a
post-generation groundedness check (`ISSUP`, answer-support based), each
with a real threshold and a visible consequence when either fails — the
result is withheld rather than returned unverified.

Run `python retrieval_eval/eval.py` for the full comparison table
(verification pass rate, avg relevance, avg groundedness, avg latency,
avg chunks) against `retrieval_eval/test_set.json` — 12 domain-specific
queries covering general lookups, exact-identifier queries, multi-hop
reasoning, metadata filtering, and two deliberate Self-RAG failure cases
(irrelevant context, unsupported answer).

**Decision:** Hybrid search ships as the default for
`search_knowledge_base` — highest verified accuracy at low latency
among the filtered strategies. Agentic RAG is reserved for multi-hop
queries that explicitly require cross-policy reasoning. Graph RAG is a
genuinely applicable alternative for entity-linked queries at comparable
latency with zero query-time LLM cost, though it trails Hybrid/Agentic
on overall verified accuracy in this test set. Naive RAG serves as the
baseline and is not production-suitable due to insufficient retrieval
precision on exact-identifier and metadata queries.

## Team ownership

| Concern | Owner | Issue # |
|---|---|---|
| DB schema & ERD | Rewan | #1 |
| Seed data | Rewan | #2 |
| Capability negotiation | Mohamed Amr | #3 |
| Notifications | Mohamed Amr | #4 |
| Elicitation | Mohamed Amr | #5 |
| Resources | Mohamed Amr | #6 |
| Prompts | Mohamed Amr | #7 |
| Transport | Mohamed Amr | #8 |
| Progress tracking | Mohamed Amr | #9 |
| Sampling | Mohamed Amr | #10 |
| Defensive tool design | Mohamed Amr | #11 |
| Context window management (4 strategies) | Mohamed Amr | — |
| Graph RAG (bonus) | Mohamed Amr | — |
| Memory systems (scratchpad, router, consolidation) | Omar Ghanem | — |
| Retrieval architectures (naive/hybrid/agentic) | Omar Fekry | — |

> Issue numbers for the newer additions (context window management,
> memory systems, retrieval architectures, Graph RAG) are not yet linked
> — add them here once the corresponding GitHub issues are opened.