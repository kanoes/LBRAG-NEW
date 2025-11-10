from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, MutableMapping, Optional, Protocol, Sequence

from .types import DocumentSegment, Query, RetrievalCandidate


class Retriever(Protocol):
    def retrieve(self, query: Query, top_k: int) -> Sequence[RetrievalCandidate]:
        ...


class Reranker(Protocol):
    def score(self, query: Query, candidates: Sequence[DocumentSegment]) -> Sequence[float]:
        ...


@dataclass
class RetrievalConfig:
    alpha: float = 0.5
    top_k: int = 20


class HybridRetriever:
    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        reranker: Optional[Reranker] = None,
        config: RetrievalConfig = RetrievalConfig(),
    ) -> None:
        self._retrievers = dict(retrievers)
        self._reranker = reranker
        self._config = config

    def retrieve(self, query: Query) -> Sequence[RetrievalCandidate]:
        merged = self._collect_candidates(query)
        ranked = self._apply_reranker(query, merged)
        ranked.sort(key=lambda c: c[1], reverse=True)
        top = [cand for cand, _ in ranked[: self._config.top_k]]
        return top

    def _collect_candidates(
        self, query: Query
    ) -> MutableMapping[str, RetrievalCandidate]:
        merged: MutableMapping[str, RetrievalCandidate] = {}
        for language, retriever in self._retrievers.items():
            _ = language  # explicit unused binding to keep order intention clear
            for candidate in retriever.retrieve(query, self._config.top_k):
                existing = merged.get(candidate.segment.identifier)
                if existing is None or candidate.dense_score > existing.dense_score:
                    merged[candidate.segment.identifier] = candidate
        return merged

    def _apply_reranker(
        self,
        query: Query,
        merged: Mapping[str, RetrievalCandidate],
    ) -> List[tuple[RetrievalCandidate, float]]:
        if not merged:
            return []
        candidates = list(merged.values())
        if self._reranker is None:
            return [(c, c.final_score(self._config.alpha)) for c in candidates]
        segments = [c.segment for c in candidates]
        rerank_scores = self._reranker.score(query, segments)
        augmented: List[tuple[RetrievalCandidate, float]] = []
        for candidate, rerank_score in zip(candidates, rerank_scores):
            enriched = RetrievalCandidate(
                segment=candidate.segment,
                dense_score=candidate.dense_score,
                rerank_score=rerank_score,
            )
            augmented.append((enriched, enriched.final_score(self._config.alpha)))
        return augmented
