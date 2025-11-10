from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Optional, Protocol, Sequence

from .translation import (
    AlignmentMode,
    RegexSentenceSplitter,
    SentenceSplitter,
    SupportsBackTranslation,
    Translator,
    align_sentences,
    estimate_alignment_quality,
)
from .types import DocumentSegment, TranslationRequest, TranslationResult


@dataclass(frozen=True)
class ConfidenceEstimate:
    value: float
    details: Mapping[str, float]
    preview: Optional[TranslationResult] = None


class ConfidenceEstimator(Protocol):
    def estimate(self, segment: DocumentSegment, target_language: str) -> ConfidenceEstimate:
        ...


@dataclass
class HeuristicConfidenceEstimator:
    translator: Translator
    splitter: SentenceSplitter = field(default_factory=RegexSentenceSplitter)
    alignment: AlignmentMode = AlignmentMode.GREEDY
    back_translation_weight: float = 0.2
    slot_weight: float = 0.4
    length_weight: float = 0.4

    def estimate(self, segment: DocumentSegment, target_language: str) -> ConfidenceEstimate:
        request = TranslationRequest(segment=segment, target_language=target_language)
        preview = self.translator.translate(request)
        length_score = _length_ratio(segment.text, preview.translated_text)
        source_sentences = self.splitter.split(segment.text)
        alignments = align_sentences(
            source_sentences,
            preview.sentences,
            mode=self.alignment,
        )
        coverage, slot_score = estimate_alignment_quality(alignments, len(source_sentences))
        slot_score = slot_score * coverage
        back_score = 0.0
        if isinstance(self.translator, SupportsBackTranslation):
            back_text = self.translator.back_translate(
                preview.translated_text, segment.language
            )
            back_score = _normalized_similarity(segment.text, back_text)
        details: MutableMapping[str, float] = {
            "length_ratio": length_score,
            "slot_consistency": slot_score,
            "coverage": coverage,
        }
        if back_score:
            details["back_translation"] = back_score
        raw = (
            self.length_weight * length_score
            + self.slot_weight * slot_score
            + self.back_translation_weight * back_score
        )
        confidence = max(0.0, min(1.0, (raw + preview.confidence) / 2))
        details["preview_confidence"] = preview.confidence
        details["combined"] = confidence
        return ConfidenceEstimate(value=confidence, details=dict(details), preview=preview)


@dataclass(frozen=True)
class TranslationCandidate:
    segment: DocumentSegment
    relevance: float
    confidence: float
    cost: float
    pivot_language: str
    confidence_details: Mapping[str, float] = field(default_factory=dict)
    preview: Optional[TranslationResult] = None

    @property
    def efficiency(self) -> float:
        if self.cost <= 0:
            return float("inf")
        return (self.relevance * self.confidence) / self.cost


@dataclass(frozen=True)
class TranslationPlan:
    selected: Sequence[TranslationCandidate]
    skipped: Sequence[TranslationCandidate]
    budget: float
    spent: float
    total_candidates: int


class TranslationSelector:
    def __init__(self, budget: float) -> None:
        self._budget = max(0.0, budget)

    def select(self, candidates: Iterable[TranslationCandidate]) -> TranslationPlan:
        pool = sorted(candidates, key=lambda c: c.efficiency, reverse=True)
        chosen: list[TranslationCandidate] = []
        skipped: list[TranslationCandidate] = []
        remaining = self._budget
        for candidate in pool:
            if candidate.cost <= remaining:
                chosen.append(candidate)
                remaining -= candidate.cost
            else:
                skipped.append(candidate)
        spent = self._budget - remaining
        return TranslationPlan(
            selected=tuple(chosen),
            skipped=tuple(skipped),
            budget=self._budget,
            spent=spent,
            total_candidates=len(pool),
        )


def _length_ratio(source: str, target: str) -> float:
    src = max(len(source.strip()), 1)
    tgt = max(len(target.strip()), 1)
    return min(src, tgt) / max(src, tgt)


def _normalized_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return _levenshtein_ratio(a, b)


def _levenshtein_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        for j, cb in enumerate(b, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    distance = previous_row[-1]
    return 1.0 - distance / max(len(a), len(b), 1)
