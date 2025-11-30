from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class Query:
    text: str
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentSegment:
    identifier: str
    text: str
    language: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_score(self, score: float) -> "DocumentSegment":
        return DocumentSegment(
            identifier=self.identifier,
            text=self.text,
            language=self.language,
            score=score,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class RetrievalCandidate:
    segment: DocumentSegment
    dense_score: float
    rerank_score: Optional[float]

    def final_score(self, alpha: float) -> float:
        if self.rerank_score is None:
            return self.dense_score
        return alpha * self.dense_score + (1 - alpha) * self.rerank_score


@dataclass(frozen=True)
class TranslationRequest:
    segment: DocumentSegment
    target_language: str


@dataclass(frozen=True)
class TranslationResult:
    translated_text: str
    confidence: float
    sentences: Sequence[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SentenceAlignment:
    source_sentence: str
    target_sentence: str
    slot_matches: Dict[str, Sequence[str]]


@dataclass(frozen=True)
class EvidenceBlock:
    segment: DocumentSegment
    translated_text: Optional[str]
    alignment: Sequence[SentenceAlignment]
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)
