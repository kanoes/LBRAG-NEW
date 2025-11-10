from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

from .types import DocumentSegment, Query, RetrievalCandidate


class Retriever(Protocol):
    def retrieve(self, query: Query, top_k: int) -> Sequence[RetrievalCandidate]:
        ...


class Reranker(Protocol):
    def score(self, query: Query, candidates: Sequence[DocumentSegment]) -> Sequence[float]:
        ...


class ScoreNormalization(Enum):
    NONE = "none"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    TEMPERATURE = "temperature"


@dataclass
class RetrievalConfig:
    alpha: float = 0.5
    top_k: int = 20
    normalization: ScoreNormalization = ScoreNormalization.MIN_MAX
    temperature: float = 1.0
    diversify_languages: bool = True
    fallback_threshold: float = 0.05
    fallback_top_k: int = 10
    per_language_quota: Optional[int] = None


@dataclass
class HybridRetriever:
    retrievers: Mapping[str, Retriever]
    reranker: Optional[Reranker] = None
    fallback_retrievers: Sequence[Retriever] = field(default_factory=tuple)
    config: RetrievalConfig = field(default_factory=RetrievalConfig)

    def retrieve(self, query: Query) -> Sequence[RetrievalCandidate]:
        merged = self._collect_candidates(query)
        ranked = self._apply_reranker(query, merged)
        ranked.sort(key=lambda c: c[1], reverse=True)
        if self.config.diversify_languages:
            ranked = self._diversify_by_language(ranked)
        results = [candidate for candidate, _ in ranked[: self.config.top_k]]
        if self._needs_fallback(ranked):
            fallback = self._run_fallback(query)
            seen = {candidate.segment.identifier for candidate in results}
            for candidate in fallback:
                if candidate.segment.identifier not in seen:
                    results.append(candidate)
                    seen.add(candidate.segment.identifier)
                if len(results) >= self.config.top_k:
                    break
        return tuple(results)

    def _collect_candidates(self, query: Query) -> MutableMapping[str, RetrievalCandidate]:
        merged: MutableMapping[str, RetrievalCandidate] = {}
        for language, retriever in self.retrievers.items():
            _ = language
            for candidate in retriever.retrieve(query, self.config.top_k):
                existing = merged.get(candidate.segment.identifier)
                if existing is None or candidate.dense_score > existing.dense_score:
                    merged[candidate.segment.identifier] = candidate
        return merged

    def _apply_reranker(
        self, query: Query, merged: Mapping[str, RetrievalCandidate]
    ) -> List[tuple[RetrievalCandidate, float]]:
        if not merged:
            return []
        candidates = list(merged.values())
        dense_scores = [c.dense_score for c in candidates]
        rerank_scores: Optional[List[float]] = None
        if self.reranker is not None:
            segments = [c.segment for c in candidates]
            rerank_scores = list(self.reranker.score(query, segments))
        dense_scores = self._normalize(dense_scores)
        if rerank_scores is not None:
            rerank_scores = self._normalize(rerank_scores)
        augmented: List[tuple[RetrievalCandidate, float]] = []
        for idx, candidate in enumerate(candidates):
            dense = dense_scores[idx]
            rerank = rerank_scores[idx] if rerank_scores is not None else None
            enriched = RetrievalCandidate(
                segment=candidate.segment,
                dense_score=dense,
                rerank_score=rerank,
            )
            augmented.append((enriched, enriched.final_score(self.config.alpha)))
        return augmented

    def _normalize(self, scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        mode = self.config.normalization
        if mode is ScoreNormalization.NONE:
            return list(scores)
        if mode is ScoreNormalization.MIN_MAX:
            min_v = min(scores)
            max_v = max(scores)
            if max_v == min_v:
                return [0.5] * len(scores)
            return [(s - min_v) / (max_v - min_v) for s in scores]
        if mode is ScoreNormalization.Z_SCORE:
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / max(len(scores) - 1, 1)
            std = variance**0.5
            if std == 0:
                return [0.0] * len(scores)
            normalized = [(s - mean) / std for s in scores]
            min_v = min(normalized)
            max_v = max(normalized)
            if max_v == min_v:
                return [0.5] * len(scores)
            return [(s - min_v) / (max_v - min_v) for s in normalized]
        if mode is ScoreNormalization.TEMPERATURE:
            temperature = max(self.config.temperature, 1e-5)
            exps = [math.exp(s / temperature) for s in scores]
            total = sum(exps)
            if total == 0:
                return [0.0] * len(scores)
            return [e / total for e in exps]
        return list(scores)

    def _diversify_by_language(
        self, ranked: List[tuple[RetrievalCandidate, float]]
    ) -> List[tuple[RetrievalCandidate, float]]:
        buckets: Dict[str, List[tuple[RetrievalCandidate, float]]] = {}
        for candidate, score in ranked:
            language = candidate.segment.language
            buckets.setdefault(language, []).append((candidate, score))
        order: List[tuple[RetrievalCandidate, float]] = []
        indices: Dict[str, int] = {lang: 0 for lang in buckets}
        while True:
            added = False
            for language, items in buckets.items():
                idx = indices[language]
                quota = self.config.per_language_quota
                if quota is not None and idx >= quota:
                    continue
                if idx < len(items):
                    order.append(items[idx])
                    indices[language] = idx + 1
                    added = True
            if not added:
                break
        return order

    def _needs_fallback(self, ranked: Sequence[tuple[RetrievalCandidate, float]]) -> bool:
        if not self.fallback_retrievers:
            return False
        if not ranked:
            return True
        top_score = ranked[0][1]
        return top_score < self.config.fallback_threshold

    def _run_fallback(self, query: Query) -> List[RetrievalCandidate]:
        merged: List[RetrievalCandidate] = []
        for retriever in self.fallback_retrievers:
            merged.extend(retriever.retrieve(query, self.config.fallback_top_k))
        return merged[: self.config.fallback_top_k]
