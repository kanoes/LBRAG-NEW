from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

from .translation import neutral_characters


LANGUAGE_SCRIPTS = {
    "en": ("LATIN",),
    "es": ("LATIN",),
    "fr": ("LATIN",),
    "de": ("LATIN",),
    "ja": ("HIRAGANA", "KATAKANA", "CJK UNIFIED"),
    "zh": ("CJK UNIFIED",),
    "ko": ("HANGUL", "CJK UNIFIED"),
    "ru": ("CYRILLIC",),
}

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def response_language_consistency(
    tokens: Sequence[str],
    language: str,
    neutral_tokens: Iterable[str] | None = None,
) -> float:
    neutral = set(neutral_tokens or neutral_characters())
    allowed_scripts = LANGUAGE_SCRIPTS.get(language, ())
    counted = 0
    hits = 0
    for token in tokens:
        if _is_neutral(token, neutral):
            continue
        counted += 1
        if _matches_language(token, allowed_scripts, language):
            hits += 1
    if counted == 0:
        return 1.0
    return hits / counted


def cost_normalized_bridging_efficiency(
    baseline_score: float,
    bridged_score: float,
    translation_tokens: float,
) -> float:
    improvement = bridged_score - baseline_score
    if translation_tokens <= 0:
        return 0.0
    return improvement / translation_tokens


def simple_language_tokenize(text: str, language: str) -> Sequence[str]:
    if language in {"ja", "zh"}:
        return tuple(text.strip())
    tokens = TOKEN_PATTERN.findall(text)
    return tuple(token.strip() for token in tokens if token.strip())


def _is_neutral(token: str, neutral: Iterable[str]) -> bool:
    return all(ch in neutral for ch in token)


def _matches_language(token: str, scripts: Sequence[str], language: str) -> bool:
    if not token:
        return False
    if not scripts:
        return _fallback_language_match(token, language)
    for ch in token:
        name = unicodedata.name(ch, "")
        if any(script in name for script in scripts):
            continue
        if ch.isdigit():
            continue
        if ch in neutral_characters():
            continue
        return False
    return True


def _fallback_language_match(token: str, language: str) -> bool:
    if language == "en":
        return token.isascii()
    return True
