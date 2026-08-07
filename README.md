# Cornerstone Realty — MCP Server Lab

## The Company

Cornerstone Realty is a mid-sized real estate brokerage with agents and
brokers working across multiple properties, clients, and offer
negotiations at once. Agents want an AI assistant that can search
listings, schedule showings, draft compliant marketing copy, and track
offers — without giving that assistant raw access to the underlying
database.

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
- **Comparative Market Analyses (CMA)** take real time to compute (
  gathering comparable properties, computing statistics) and shouldn't
  leave the caller blocked with no feedback.

This is the exact shape of problem the lab asks for: real state changes,
real authorization boundaries, and genuine risk — not a "customer
chatbot" with nothing at stake.

## Why this needs all 9 concerns (8 required + Sampling from the rubric)

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
├── README.md              — this file
├── .env.example
├── .gitignore
├── db/                    — schema, seed data, ERD, engine documentation
├── mcp_server/             — the MCP server (tools, resources, prompts)
├── agent/                  — the MCP client that drives the server
└── demo/                   — full demo transcript covering every concern
```

Detailed documentation for each part lives in its own README:
- [`db/README.md`](db/README.md) — engine choice, schema, seed data rationale
- [`mcp_server/README.md`](mcp_server/README.md) — exactly where each
  protocol concern lives in the code, tool read-only/write comparison,
  and capability fallback behavior
- [`agent/README.md`](agent/README.md) — how to run the client, and how
  it negotiates capabilities

## Quick start

```bash
# 1. Build the database
python db/build_db.py

# 2. Install dependencies
pip install "mcp<2" jsonschema openai

# 3. Add a free Groq API key to .env (see .env.example)
#    https://console.groq.com — no credit card required

# 4. Run the full demo (server is started automatically by the client)
python agent/client.py
```

See [`demo/transcript.md`](demo/transcript.md) for a full recorded run
showing every concern firing.

## Comparison note: read-only vs. write, and capability dependencies

| Tool | Type | Elicitation required? | Authorization required |
|---|---|---|---|
| `search_properties` | Read-only | No | None |
| `get_property` | Read-only | No | None |
| `generate_cma` | Read-only (long-running) | No | None |
| `submit_offer` | Write | Yes, if offer < 85% of list price | None |
| `accept_offer` | Write | No | Listing agent for that property, or Broker |
| `assign_listing_agent` | Write | No | Broker only |
| `explain_offer_risk` | Read-only (uses sampling) | No | None |

**If a client connects without elicitation support:** it still sees
`submit_offer` and `accept_offer` in the server's declared tool list
(FastMCP doesn't conditionally hide tools per-client), but the client
itself is responsible for not calling them — `agent/client.py`
demonstrates this via the `SUPPORTS_ELICITATION` flag, which controls
whether the elicitation callback is registered at all.

## Team ownership

| Concern | Owner | Issue # |
|---|---|---|
| DB schema & ERD | REWAN | #1 |
| Seed data | REWAN  | #2 |
| Capability negotiation | MOHAMED | #3 |
| Notifications | MOHAMED | #4 |
| Elicitation | MOHAMED | #5 |
| Resources | MOHAMED | #6 |
| Prompts | MOHAMED | #7 |
| Transport | MOHAMED | #8 |
| Progress tracking | MOHAMED | #9 |
| Sampling | MOHAMED | #10 |
| Defensive tool design | MOHAMED | #11 |
