from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Protocol, Any, Optional

from .types import DocumentSegment


@dataclass(frozen=True)
class ConfidenceEstimate:
    value: float
    details: dict[str, Any] = field(default_factory=dict)
    preview: Optional[str] = None

class ConfidenceEstimator(Protocol):
    def estimate(self, segment: "DocumentSegment", target_language: str) -> ConfidenceEstimate:
        ...

@dataclass(frozen=True)
class TranslationCandidate:
    segment: DocumentSegment
    relevance: float
    confidence: float
    cost: float

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


class TranslationSelector:
    def __init__(self, budget: float, min_efficiency: float = 1e-6) -> None:
        self._budget = max(0.0, budget)
        self._min_eff = min_efficiency

    def select(self, candidates: Iterable[TranslationCandidate]) -> TranslationPlan:
        pool = sorted(candidates, key=lambda c: c.efficiency, reverse=True)
        chosen, skipped, remaining = [], [], self._budget
        for c in pool:
            if c.efficiency < self._min_eff: 
                skipped.append(c); continue
            if c.cost <= remaining:
                chosen.append(c); remaining -= c.cost
            else:
                skipped.append(c)
        for c in sorted((x for x in skipped if x.cost <= remaining), key=lambda x: x.efficiency, reverse=True):
            chosen.append(c); remaining -= c.cost
        spent = self._budget - remaining
        return TranslationPlan(tuple(chosen), tuple([x for x in skipped if x not in chosen]), self._budget, spent)
