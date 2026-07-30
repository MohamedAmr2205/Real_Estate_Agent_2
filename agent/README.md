# agent/ — Meridian Realty MCP Client

`client.py` is the "host" side of the protocol: it spawns
`mcp_server/server.py` over stdio, performs a real `initialize`/
`initialized` handshake, and exercises every tool so each protocol
concern can be demonstrated live end-to-end (not just described).

## How to run it

From the repository root:

```bash
# 1. Build the database (only needed once, or after schema/seed changes)
python db/build_db.py

# 2. Install client dependencies
pip install "mcp<2" jsonschema openai

# 3. Add your Groq API key to .env (see .env.example)
#    GROQ_API_KEY=gsk_...
#    Free tier, no credit card required: https://console.groq.com

# 4. Run the client (it starts the server itself, no separate process needed)
python agent/client.py
```

During the run, when `submit_offer` is called with a below-threshold
offer, the terminal will pause and ask:

```
Confirm this offer should be submitted? (y/n):
```

Type `y` or `n` — this is a genuine `elicitation/create` round trip, not
a simulated pause.

## What `client.py` actually does

1. **Capability negotiation** — connects via `mcp.client.stdio.stdio_client`,
   calls `session.initialize()`, and prints the server's declared
   capabilities. This client declares its own support for elicitation and
   sampling by *passing* `elicitation_callback` and `sampling_callback` to
   `ClientSession` — in the MCP spec, elicitation and sampling are
   **client** capabilities, not server ones. The server only calls
   `ctx.elicit()` / `ctx.session.create_message()` because this client
   opted in by providing those callbacks.

   `SUPPORTS_ELICITATION` (top of `client.py`) can be flipped to `False`
   to demonstrate the fallback path: the client still sees `submit_offer`
   and `accept_offer` in the server's tool list, but deliberately does
   not call them, since it never declared elicitation support and the
   server-side `ctx.elicit()` call would have nothing to talk to.

2. **Notifications** — a `message_handler` is registered on the session
   that watches for `notifications/tools/list_changed` and prints when
   one arrives, then the client re-fetches `list_tools()` to show the
   reaction, rather than periodically polling.

3. **Elicitation** — `elicitation_callback` genuinely blocks on
   `input()` at the terminal, waiting for a real decision before
   returning a result back to the server.

4. **Sampling** — `sampling_callback` sends the server's prepared prompt
   to Groq's API (Llama 3.3 70B) — the **client's** model, not anything
   the server owns — and returns the actual generated analysis.

5. **Progress tracking** — a `progress_callback` passed to
   `call_tool("generate_cma", ...)` prints each intermediate step as it
   streams in.

6. Every other tool (`search_properties`, `get_property`,
   `accept_offer`) is called directly through `session.call_tool(...)`
   to prove the agent genuinely discovers and calls server tools, rather
   than the tool list only ever being described in prose.

## Why Groq instead of a paid API

Groq offers a genuinely free tier (no credit card) suitable for this
class project — the choice has nothing to do with the server, which has
no dependency on any specific provider. `sampling/createMessage` simply
asks "the client's model," whatever that happens to be; swapping Groq
for Anthropic/OpenAI/etc. only requires changing `sampling_callback` in
`client.py`.

## Transport

This client connects over **stdio** (`StdioServerParameters` spawns
`mcp_server/server.py` as a subprocess). See `mcp_server/README.md` →
"Transport choice, justified" for why stdio is used for local
development, and the plan for a Streamable HTTP transport for the real
multi-office deployment.