"""
rag/query_decomposition.py
===========================
ADD-ON LAB — Option A: Query Decomposition wrapping search_knowledge_base.

Compound questions ("What documents are required for an offer, and what's
the commission rate?") often only get partially answered by a single
search_knowledge_base call, because the embedding/BM25 signal for the
whole compound question dilutes relevance for each sub-part individually.

decompose_and_search():
  1. Makes ONE real LLM call (Groq — same client pattern already used in
     context_eval/strategies/recursive_summarization.py) to split the
     incoming query into 2-4 sub-questions.
  2. Calls the EXISTING search_knowledge_base once per sub-question
     unchanged — this wraps it, it does not replace it, and every
     role-based filter / Self-RAG verification search_knowledge_base
     already does still applies per sub-question.
  3. Returns the combined chunks tagged with which sub-question each one
     answers. It does NOT merge them into a single final answer — that's
     left to the calling model, per the lab spec.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable


# ----------------------------------------------------------------
# Env + Groq client (same load pattern as recursive_summarization.py)
# ----------------------------------------------------------------
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


DECOMPOSE_PROMPT = """\
Break the following question into 2-4 simpler sub-questions that, \
together, fully answer it. If the question is already simple, just \
return it as-is as a single sub-question.

STRICT RULES:
- Every sub-question must ask about something the ORIGINAL question \
explicitly mentions. Do not introduce new topics, angles, or facts \
(e.g. fees, location, timing) that the original question did not raise.
- Do not add clarifying or speculative sub-questions "just in case."
- Never create a sub-question that just asks to restate or identify \
something already named in the original question (e.g. if the question \
already names "Clause 7" or "the Smouha villa contract," do not add a \
separate sub-question asking what the clause number or contract name is).
- Prefer fewer, closer-to-the-original sub-questions over more, \
broader ones.

Question: {query}

Return ONLY a numbered list, one sub-question per line. Example:
1. ...
2. ...
"""


def _call_llm(prompt: str, original_query: str) -> str:
    """
    Real Groq call. Honest fallback if no GROQ_API_KEY is configured:
    treat the query as already-simple (a single sub-question) rather
    than fabricating a decomposition.
    """
    if _groq_client is None:
        return f"1. {original_query}"

    response = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def decompose_query(query: str) -> list[str]:
    """Turn one (possibly compound) query into a list of sub-questions."""
    raw = _call_llm(DECOMPOSE_PROMPT.format(query=query), query)

    sub_questions: list[str] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\.\)]\s*|^-\s*", "", line, count=1)
        if line:
            sub_questions.append(line.strip())

    return sub_questions or [query]  # fallback: treat as one question


def decompose_and_search(
    query: str,
    search_tool: Callable[..., dict[str, Any]],
    property_id: int,
    caller_agent_id: int,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    The new tool logic. `search_tool` is the real search_knowledge_base
    function (passed in so this module has no hard import-time dependency
    on the server — mirrors how server.py itself imports it from
    rag/knowledge_base.py).

    Same role-based filtering and Self-RAG verification as a normal
    search_knowledge_base call apply to EACH sub-question independently,
    since we're calling the real tool, not bypassing it.
    """
    sub_questions = decompose_query(query)

    tagged_results: list[dict[str, str]] = []
    notes: list[str] = []
    for sub_q in sub_questions:
        result = search_tool(sub_q, property_id, caller_agent_id, top_k)
        if result.get("note"):
            notes.append(f"[{sub_q}] {result['note']}")
        for chunk_text in result.get("results", []):
            tagged_results.append({"sub_question": sub_q, "chunk": chunk_text})

    return {
        "sub_questions": sub_questions,
        "results": tagged_results,
        "notes": notes,
    }


# ----------------------------------------------------------------
# Standalone demo — a compound question the plain tool only partially
# answers, vs. decompose_and_search answering it fully.
# Run directly: python rag/query_decomposition.py
# Requires GEMINI_API_KEY (for search_knowledge_base's embeddings) and
# ideally GROQ_API_KEY (for the real decomposition call) in .env.
# ----------------------------------------------------------------
if __name__ == "__main__":
    import sys
    PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

    from rag.knowledge_base import index_property_notes, search_knowledge_base

    index_property_notes()

    demo_query = (
        "What documents are required to submit a property offer, and "
        "what is the standard commission rate for listing agents?"
    )
    PROPERTY_ID = 1
    CALLER_AGENT_ID = 1  # regular agent, not Broker

    print("=== Plain search_knowledge_base (sees only the raw compound question) ===")
    plain = search_knowledge_base(demo_query, PROPERTY_ID, CALLER_AGENT_ID, top_k=3)
    for r in plain.get("results", []):
        print(f"  - {r[:100]}...")
    print(f"  ({len(plain.get('results', []))} chunk(s) returned — note whether both "
          f"topics [offer documents] AND [commission rate] actually appear above)\n")

    print("=== decompose_and_search (same tool, split into sub-questions) ===")
    combined = decompose_and_search(
        demo_query, search_knowledge_base, PROPERTY_ID, CALLER_AGENT_ID, top_k=2
    )
    print(f"Sub-questions: {combined['sub_questions']}\n")
    for r in combined["results"]:
        print(f"  [{r['sub_question']}] -> {r['chunk'][:100]}...")
    print(f"  ({len(combined['results'])} chunk(s) returned, tagged by sub-question)")