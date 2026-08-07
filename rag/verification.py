"""
rag/verification.py  ← REPLACE the existing file with this
=============================================================
الإصلاح الرئيسي في verify_relevance:
- قبل: Jaccard overlap (intersection/union) — دايماً منخفض لأن الـ context
  كبير جداً مقارنة بالـ query القصيرة
- بعد: Query Coverage score — كم% من كلمات الـ query موجودة في الـ context
  ده أكثر منطقية: query تسأل عن "documents required" → لو "documents"
  و"required" موجودين في الـ context → score عالي
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class VerificationConfig:
    relevance_threshold: float = 0.35   # query coverage: 35% of query terms in context
    groundedness_threshold: float = 0.25
    use_llm: bool = False
    llm_model: str = "claude-sonnet-4-6"


@dataclass
class RelevanceVerdict:
    relevant: bool
    score: float
    details: str = ""


@dataclass
class GroundednessVerdict:
    grounded: bool
    confidence: float
    explanation: str = ""


# ----------------------------------------------------------------
# Token helpers
# ----------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "it", "its", "this", "that",
    "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "with", "by", "from", "as", "what",
    "how", "when", "where", "which", "who", "will", "would",
    "can", "could", "should", "may", "might", "say", "says",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _query_coverage(query: str, context: str) -> float:
    """
    Query Coverage score:
    What fraction of the query's content terms appear in the context?

    This is asymmetric — we care about the query terms, not the context terms.
    A short query against a long relevant context → high score.
    A short query against irrelevant context → low score.

    Much better than Jaccard for RAG relevance checking because:
    - Jaccard penalizes long contexts (many extra context terms lower the score)
    - Coverage only asks: "does the context cover what the query needs?"
    """
    query_terms = _tokens(query)
    if not query_terms:
        return 0.0
    context_terms = _tokens(context)
    if not context_terms:
        return 0.0
    covered = query_terms & context_terms
    return len(covered) / len(query_terms)


def _overlap_score(a: str, b: str) -> float:
    """Jaccard-style overlap — kept for groundedness checking."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ----------------------------------------------------------------
# verify_relevance — ISREL check
# ----------------------------------------------------------------

def verify_relevance(
    query: str,
    retrieved_context: str,
    config: VerificationConfig | None = None,
) -> RelevanceVerdict:
    """
    ISREL check: is the retrieved context relevant to the query?

    Uses query coverage (not Jaccard) so short queries against long
    relevant contexts still score high.

    Threshold: 0.35 means at least 35% of the query's content words
    must appear in the retrieved context.
    """
    config = config or VerificationConfig()

    if not query.strip():
        return RelevanceVerdict(
            relevant=False, score=0.0,
            details="[SELF-RAG ISREL] FAIL — empty query"
        )
    if not retrieved_context.strip():
        return RelevanceVerdict(
            relevant=False, score=0.0,
            details="[SELF-RAG ISREL] FAIL — no retrieved context"
        )

    if config.use_llm:
        return _llm_relevance(query, retrieved_context, config)

    score = _query_coverage(query, retrieved_context)
    passed = score >= config.relevance_threshold

    query_terms = _tokens(query)
    context_terms = _tokens(retrieved_context)
    covered = query_terms & context_terms
    missing = query_terms - context_terms

    details = (
        f"[SELF-RAG ISREL] {'PASS' if passed else 'FAIL'} "
        f"— coverage={score:.3f} threshold={config.relevance_threshold} "
        f"query_terms={len(query_terms)} covered={len(covered)}"
    )
    if not passed and missing:
        details += f" | missing: {sorted(missing)[:5]}"

    return RelevanceVerdict(relevant=passed, score=round(score, 4), details=details)


# ----------------------------------------------------------------
# verify_groundedness — ISSUP check
# ----------------------------------------------------------------

def verify_groundedness(
    retrieved_context: str,
    generated_answer: str,
    config: VerificationConfig | None = None,
) -> GroundednessVerdict:
    """
    ISSUP check: is the generated answer supported by the retrieved context?

    Uses answer coverage: what fraction of the answer's content words
    appear in the retrieved context? Low → hallucination risk.
    """
    config = config or VerificationConfig()

    if not generated_answer.strip():
        return GroundednessVerdict(
            grounded=False, confidence=0.0,
            explanation="[SELF-RAG ISSUP] FAIL — empty answer"
        )
    if not retrieved_context.strip():
        return GroundednessVerdict(
            grounded=False, confidence=0.0,
            explanation="[SELF-RAG ISSUP] FAIL — no context to ground against"
        )

    if config.use_llm:
        return _llm_groundedness(retrieved_context, generated_answer, config)

    answer_tokens = _tokens(generated_answer)
    context_tokens = _tokens(retrieved_context)

    if not answer_tokens:
        return GroundednessVerdict(
            grounded=False, confidence=0.0,
            explanation="[SELF-RAG ISSUP] FAIL — answer has no content tokens"
        )

    supported = answer_tokens & context_tokens
    confidence = len(supported) / len(answer_tokens)
    passed = confidence >= config.groundedness_threshold

    explanation = (
        f"[SELF-RAG ISSUP] {'PASS' if passed else 'FAIL'} "
        f"— {len(supported)}/{len(answer_tokens)} answer tokens found in context "
        f"(confidence={confidence:.3f} threshold={config.groundedness_threshold})"
    )
    if not passed:
        unsupported = answer_tokens - context_tokens
        explanation += f" | unsupported: {sorted(unsupported)[:8]}"

    return GroundednessVerdict(
        grounded=passed,
        confidence=round(confidence, 4),
        explanation=explanation,
    )


# ----------------------------------------------------------------
# Convenience wrapper
# ----------------------------------------------------------------

@dataclass
class VerificationReport:
    relevance: RelevanceVerdict
    groundedness: GroundednessVerdict

    @property
    def passed(self) -> bool:
        return self.relevance.relevant and self.groundedness.grounded

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status}\n"
            f"  {self.relevance.details}\n"
            f"  {self.groundedness.explanation}"
        )


def verify(
    query: str,
    retrieved_context: str,
    generated_answer: str,
    config: VerificationConfig | None = None,
) -> VerificationReport:
    config = config or VerificationConfig()
    return VerificationReport(
        relevance=verify_relevance(query, retrieved_context, config),
        groundedness=verify_groundedness(retrieved_context, generated_answer, config),
    )


# ----------------------------------------------------------------
# Optional LLM-backed verification (Anthropic API)
# ----------------------------------------------------------------

def _llm_relevance(query: str, context: str,
                   config: VerificationConfig) -> RelevanceVerdict:
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            f"Query: {query}\n\n"
            f"Retrieved context:\n{context[:2000]}\n\n"
            "Is this retrieved context relevant to the query? "
            "Reply with ONLY 'yes' or 'no' followed by a brief reason."
        )
        msg = client.messages.create(
            model=config.llm_model,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip().lower()
        relevant = text.startswith("yes")
        score = 0.9 if relevant else 0.1
        return RelevanceVerdict(
            relevant=relevant, score=score,
            details=f"[SELF-RAG ISREL LLM] {'PASS' if relevant else 'FAIL'} — {text[:120]}"
        )
    except Exception as e:
        score = _query_coverage(query, context)
        passed = score >= config.relevance_threshold
        return RelevanceVerdict(
            relevant=passed, score=round(score, 4),
            details=f"[SELF-RAG ISREL fallback] LLM error: {e} — coverage={score:.3f}"
        )


def _llm_groundedness(context: str, answer: str,
                      config: VerificationConfig) -> GroundednessVerdict:
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            f"Context:\n{context[:2000]}\n\n"
            f"Answer:\n{answer}\n\n"
            "Is every claim in the answer supported by the context above? "
            "Reply with ONLY 'yes' or 'no' followed by a brief reason."
        )
        msg = client.messages.create(
            model=config.llm_model,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip().lower()
        grounded = text.startswith("yes")
        confidence = 0.9 if grounded else 0.1
        return GroundednessVerdict(
            grounded=grounded, confidence=confidence,
            explanation=f"[SELF-RAG ISSUP LLM] {'PASS' if grounded else 'FAIL'} — {text[:120]}"
        )
    except Exception as e:
        answer_t = _tokens(answer)
        context_t = _tokens(context)
        conf = len(answer_t & context_t) / max(len(answer_t), 1)
        passed = conf >= config.groundedness_threshold
        return GroundednessVerdict(
            grounded=passed, confidence=round(conf, 4),
            explanation=f"[SELF-RAG ISSUP fallback] LLM error: {e} — confidence={conf:.3f}"
        )