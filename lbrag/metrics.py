from __future__ import annotations
from typing import Iterable, Sequence, Dict


def response_language_consistency(
    tokens: Sequence[str], allowed_tokens: Iterable[str]
) -> float:
    allowed = set(allowed_tokens)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in allowed)
    return hits / len(tokens)


def response_language_consistency_prob(
    lang_posteriors: Sequence[Dict[str, float]], target_lang: str, tau: float = 0.7
) -> float:
    if not lang_posteriors:
        return 0.0
    hits, total = 0, 0
    for post in lang_posteriors:
        if post.get("_neutral"):
            continue
        total += 1
        hits += 1 if post.get(target_lang, 0.0) >= tau else 0
    return hits / total if total else 1.0


def cost_normalized_bridging_efficiency(
    baseline_score: float, bridged_score: float, translation_tokens: float
) -> float:
    improvement = bridged_score - baseline_score
    if translation_tokens <= 0:
        return 0.0
    return improvement / translation_tokens
