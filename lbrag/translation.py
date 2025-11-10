from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Iterable, Mapping, Protocol, Sequence

from .types import SentenceAlignment, TranslationRequest, TranslationResult


class Translator(Protocol):
    def translate(self, request: TranslationRequest) -> TranslationResult:
        ...

    def estimate_cost(self, request: TranslationRequest) -> float:
        ...


class SupportsBackTranslation(Protocol):
    def back_translate(self, text: str, source_language: str) -> str:
        ...


class SentenceSplitter(Protocol):
    def split(self, text: str) -> Sequence[str]:
        ...


@dataclass
class RegexSentenceSplitter:
    pattern: re.Pattern[str] = re.compile(r"(?<=[.!?。！？])\s+|。|\n")

    def split(self, text: str) -> Sequence[str]:
        cleaned = text.strip()
        if not cleaned:
            return tuple()
        parts = [piece.strip() for piece in self.pattern.split(cleaned) if piece.strip()]
        return tuple(parts) if parts else (cleaned,)


class AlignmentMode(Enum):
    GREEDY = "greedy"
    HUNGARIAN = "hungarian"


def align_sentences(
    source_sentences: Sequence[str],
    target_sentences: Sequence[str],
    mode: AlignmentMode = AlignmentMode.GREEDY,
    similarity: str = "ratio",
) -> Sequence[SentenceAlignment]:
    if not source_sentences or not target_sentences:
        return tuple()
    if mode is AlignmentMode.HUNGARIAN:
        return _hungarian_alignment(source_sentences, target_sentences, similarity)
    return _greedy_alignment(source_sentences, target_sentences)


def greedy_sentence_alignment(
    source_sentences: Sequence[str], target_sentences: Sequence[str]
) -> Sequence[SentenceAlignment]:
    return align_sentences(
        source_sentences,
        target_sentences,
        mode=AlignmentMode.GREEDY,
    )


def _greedy_alignment(
    source_sentences: Sequence[str], target_sentences: Sequence[str]
) -> Sequence[SentenceAlignment]:
    alignments: list[SentenceAlignment] = []
    tgt_iter = iter(enumerate(target_sentences))
    current = next(tgt_iter, None)
    for idx, source in enumerate(source_sentences):
        if current is None:
            break
        target_idx, target_sentence = current
        alignments.append(
            SentenceAlignment(
                source_sentence=source,
                target_sentence=target_sentence,
                source_index=idx,
                target_index=target_idx,
                slot_matches=extract_slot_matches(source, target_sentence),
            )
        )
        current = next(tgt_iter, None)
    return tuple(alignments)


def _hungarian_alignment(
    source_sentences: Sequence[str],
    target_sentences: Sequence[str],
    similarity: str = "ratio",
) -> Sequence[SentenceAlignment]:
    size = max(len(source_sentences), len(target_sentences))
    if size == 0:
        return tuple()
    cost_matrix = [[1.0] * size for _ in range(size)]
    for i, source in enumerate(source_sentences):
        for j, target in enumerate(target_sentences):
            cost_matrix[i][j] = 1.0 - _sentence_similarity(source, target, similarity)
    assignment = _hungarian(cost_matrix)
    alignments: list[SentenceAlignment] = []
    for source_idx, target_idx in assignment:
        if source_idx >= len(source_sentences) or target_idx >= len(target_sentences):
            continue
        source = source_sentences[source_idx]
        target = target_sentences[target_idx]
        alignments.append(
            SentenceAlignment(
                source_sentence=source,
                target_sentence=target,
                source_index=source_idx,
                target_index=target_idx,
                slot_matches=extract_slot_matches(source, target),
            )
        )
    alignments.sort(key=lambda a: a.source_index)
    return tuple(alignments)


def _sentence_similarity(source: str, target: str, mode: str) -> float:
    if not source or not target:
        return 0.0
    ratio = SequenceMatcher(None, source, target).ratio()
    if mode == "ratio":
        return ratio
    if mode == "partial":
        return SequenceMatcher(None, source, target).quick_ratio()
    return ratio


def _hungarian(cost_matrix: Sequence[Sequence[float]]) -> Sequence[tuple[int, int]]:
    n = len(cost_matrix)
    m = len(cost_matrix[0]) if cost_matrix else 0
    size = max(n, m)
    # Pad matrix to square
    padded = [list(row) + [1.0] * (size - m) for row in cost_matrix]
    for _ in range(size - n):
        padded.append([1.0] * size)
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = padded[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = []
    for j in range(1, size + 1):
        if p[j] != 0:
            assignment.append((p[j] - 1, j - 1))
    return assignment


def extract_slot_matches(source: str, target: str) -> dict[str, Sequence[str]]:
    slots: dict[str, Sequence[str]] = {}
    patterns: Mapping[str, Iterable[str]] = {
        "numbers": re.findall(r"[-+]?[0-9]+(?:[.,][0-9]+)?", source),
        "dates": re.findall(
            r"\b(?:\d{4}[\-/年年]\d{1,2}[\-/月月]\d{1,2})\b", source
        ),
        "latin_named": re.findall(r"\b[A-Z][A-Za-z0-9_-]+\b", source),
        "cjk_named": re.findall(r"[\u4e00-\u9fff]{2,5}", source),
    }
    for key, values in patterns.items():
        filtered = [value for value in values if value and value in target]
        if filtered:
            slots[key] = tuple(filtered)
    return slots


def estimate_alignment_quality(
    alignments: Sequence[SentenceAlignment], total_sentences: int
) -> tuple[float, float]:
    if total_sentences <= 0:
        return 0.0, 0.0
    coverage = len({a.source_index for a in alignments}) / total_sentences
    total_slots = 0
    matched_slots = 0
    for alignment in alignments:
        for values in alignment.slot_matches.values():
            total_slots += 1
            if values:
                matched_slots += 1
    slot_consistency = matched_slots / total_slots if total_slots else 1.0
    return coverage, slot_consistency


def neutral_characters() -> set[str]:
    return set("0123456789%$€¥,.:-_/\\@#&")
