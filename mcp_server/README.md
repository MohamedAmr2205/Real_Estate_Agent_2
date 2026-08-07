# mcp_server/ — Meridian Realty MCP Server

Built with the official Python MCP SDK (`mcp[cli]`, pinned to `mcp<2` —
see note at the bottom). Run locally with:

```bash
cd mcp_server
python server.py
```

## Where every protocol concern lives

Everything below is implemented in `server.py`, organized into
numbered, commented sections so it can be found without reading the
whole file top to bottom. Business logic (validation, DB access,
authorization) is kept in separate files under `tools/` and
`validation.py` so each concern's *decision logic* and its *protocol
plumbing* are easy to review independently.

| Concern | File / location | Trigger |
|---|---|---|
| **Capability negotiation** | `server.py` SECTION 1 — `FastMCP(...)` declares tools/resources/prompts capabilities during `initialize`. Client-side check happens in `agent/client.py`. | Client checks declared capabilities before offering `submit_offer`/`accept_offer`; falls back to read-only tools if elicitation isn't supported. |
| **Notifications** | `server.py` SECTION 7 — `assign_listing_agent()` → `await ctx.session.send_tool_list_changed()` | Fires the moment a Broker reassigns a property's listing agent (`tools/assign_listing_agent.py`). |
| **Elicitation** | `server.py` SECTION 5 — `submit_offer()` → `await ctx.elicit(...)` | Fires when `offer_amount` is below 85% of list price (`validation.offer_is_high_risk`). |
| **Resources** | `server.py` SECTION 2 — `fair_housing_policy()` reads `resources/fair_housing_policy.md` | Read via `resources/read`, not a tool call. Referenced by `draft_listing_description`. |
| **Prompts** | `server.py` SECTION 3, logic in `prompts/listing_prompts.py` | `draft_listing_description(property_id)`, `draft_offer_rejection(offer_id)` — discoverable via `prompts/list`. |
| **Transport** | `server.py` SECTION 10 — `mcp.run(transport=...)` | stdio during development (current); Streamable HTTP added in a later commit — see "Transport choice" below. |
| **Progress tracking** | `server.py` SECTION 8 — `generate_cma()` forwards each step from `tools/generate_cma.py`'s generator to `await ctx.report_progress(...)` | 4 real progress steps: loading subject, searching comparables, computing stats, finalizing. |
| **Defensive tool design** | `server.py` SECTION 6 — `accept_offer()`, logic in `tools/accept_offer.py` + `validation.py` | Typed schema (no `**kwargs`), independent status check (offer must be `Pending`), and an authorization check run inside the handler (`assert_can_accept_offer` — caller must be the listing agent or a Broker). |
| **Sampling** | `server.py` SECTION 9 — `explain_offer_risk()` → `await ctx.session.create_message(...)` | Asks the **client's** model to analyze offer risk; the server only prepares facts/prompt in `tools/explain_offer_risk.py` and never runs its own LLM call. |

## Tools: read-only vs. write, and elicitation requirements

| Tool | Type | Requires elicitation? | Requires which role? |
|---|---|---|---|
| `search_properties` | Read-only | No | Any connected agent |
| `get_property` | Read-only | No | Any connected agent |
| `submit_offer` | Write | Yes, if offer < 85% of list price | Any connected agent |
| `accept_offer` | Write | No | Listing agent for that property, or Broker |
| `assign_listing_agent` | Write | No | Broker only |
| `generate_cma` | Read-only (long-running) | No | Any connected agent |
| `explain_offer_risk` | Read-only (uses sampling) | No | Any connected agent |

## What happens if a client connects without a capability a tool needs

`submit_offer` depends on `elicitation/create` being supported by the
connecting client. `agent/client.py` checks the server's declared
capabilities during the `initialize` handshake; if elicitation support
isn't present, the client does not surface `submit_offer` or
`accept_offer` at all — only the read-only tools (`search_properties`,
`get_property`) are exposed for that session. This is a client-side
decision informed by server-declared capabilities, not something the
server silently works around.

`explain_offer_risk` similarly depends on `sampling/createMessage`
support; a client without it should not offer that tool either.

## Transport choice, justified

Development transport is **stdio** — the simplest option for local
iteration while the tool set was still being built (see early commit
history). Meridian Realty has agents working across multiple office
locations, so a single local stdio process can't serve them; the actual
deployment target is **Streamable HTTP**, added in a later commit once
the core tools were stable, so multiple remote agent sessions can
connect to one running server instance.

## A note on the `mcp` package version

This server is pinned to `mcp<2`. The official SDK's v2.0.0 (released
2026-07-28) renamed `FastMCP` to `MCPServer` as part of a breaking
architectural rewrite. This codebase targets the stable, well-documented
v1.x API (`mcp.server.fastmcp.FastMCP`) rather than the just-released v2,
since v2 shipped the day before this server was built and its
documentation/tutorials were not yet mature enough to build against
reliably.