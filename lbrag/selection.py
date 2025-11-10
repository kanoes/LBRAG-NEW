from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .types import DocumentSegment


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
    def __init__(self, budget: float) -> None:
        self._budget = max(0.0, budget)

    def select(self, candidates: Iterable[TranslationCandidate]) -> TranslationPlan:
        pool = sorted(candidates, key=lambda c: c.efficiency, reverse=True)
        chosen: List[TranslationCandidate] = []
        skipped: List[TranslationCandidate] = []
        remaining = self._budget
        for candidate in pool:
            if candidate.cost <= remaining:
                chosen.append(candidate)
                remaining -= candidate.cost
            else:
                skipped.append(candidate)
        spent = self._budget - remaining
        return TranslationPlan(selected=tuple(chosen), skipped=tuple(skipped), budget=self._budget, spent=spent)
