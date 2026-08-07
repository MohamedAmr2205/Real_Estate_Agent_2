"""rag/verification.py — Simple relevance and groundedness verification."""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class VerificationConfig:
    relevance_threshold: float = 0.5
    groundedness_threshold: float = 0.5


def verify_relevance(query: str, candidate_text: str, config: VerificationConfig | None = None) -> RelevanceVerdict:
    config = config or VerificationConfig()
    relevant = bool(query.strip()) and bool(candidate_text.strip())
    score = 1.0 if relevant else 0.0
    details = "Non-empty query and candidate text" if relevant else "Missing query or candidate text"
    return RelevanceVerdict(relevant=relevant, score=score, details=details)


def verify_groundedness(context: str, answer: str, config: VerificationConfig | None = None) -> GroundednessVerdict:
    config = config or VerificationConfig()
    grounded = bool(answer.strip()) and bool(context.strip())
    confidence = 1.0 if grounded else 0.0
    explanation = "Answer is grounded in provided context" if grounded else "Answer lacks sufficient context"
    return GroundednessVerdict(grounded=grounded, confidence=confidence, explanation=explanation)
