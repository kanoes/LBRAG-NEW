from __future__ import annotations

from typing import Iterable, Sequence


def response_language_consistency(tokens: Sequence[str], allowed_tokens: Iterable[str]) -> float:
    allowed = set(allowed_tokens)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in allowed)
    return hits / len(tokens)


def cost_normalized_bridging_efficiency(
    baseline_score: float,
    bridged_score: float,
    translation_tokens: float,
) -> float:
    improvement = bridged_score - baseline_score
    if translation_tokens <= 0:
        return 0.0
    return improvement / translation_tokens
