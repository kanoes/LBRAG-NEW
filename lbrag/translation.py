from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .types import SentenceAlignment, TranslationRequest, TranslationResult


class Translator(Protocol):
    def translate(self, request: TranslationRequest) -> TranslationResult:
        ...


class SentenceSplitter(Protocol):
    def split(self, text: str) -> Sequence[str]:
        ...


@dataclass
class SimpleSentenceSplitter:
    pattern: re.Pattern[str] = re.compile(r"(?<=[.!?。！？])\s+")

    def split(self, text: str) -> Sequence[str]:
        parts = [t.strip() for t in self.pattern.split(text) if t.strip()]
        if not parts:
            return (text.strip(),) if text.strip() else tuple()
        return tuple(parts)

@dataclass
class RegexSentenceSplitter:
    pattern: re.Pattern[str] = re.compile(r"(?<=[。．！？!?]|[.!?])")

    def split(self, text: str) -> Sequence[str]:
        parts = [t.strip() for t in self.pattern.split(text) if t.strip()]
        return tuple(parts) if parts else ((text.strip(),) if text.strip() else tuple())


def greedy_sentence_alignment(
    source_sentences: Sequence[str],
    target_sentences: Sequence[str],
) -> Sequence[SentenceAlignment]:
    if not source_sentences or not target_sentences:
        return tuple()
    alignments: list[SentenceAlignment] = []
    target_iter = iter(target_sentences)
    current_target = next(target_iter, None)
    for source in source_sentences:
        if current_target is None:
            break
        alignments.append(
            SentenceAlignment(
                source_sentence=source,
                target_sentence=current_target,
                slot_matches=_extract_slot_matches(source, current_target),
            )
        )
        current_target = next(target_iter, None)
    return tuple(alignments)


def _extract_slot_matches(source: str, target: str) -> dict[str, Sequence[str]]:
    slots: dict[str, Sequence[str]] = {}
    patterns = {
        "numbers": re.findall(r"[-+]?[0-9]+(?:[.,][0-9]+)?", source),
        "dates": re.findall(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b", source),
        "upper": re.findall(r"\b[A-Z][A-Za-z0-9_-]+\b", source),
    }
    for key, values in patterns.items():
        if values:
            target_values = []
            for value in values:
                if value in target:
                    target_values.append(value)
            slots[key] = tuple(target_values)
    return slots


def estimate_alignment_quality(alignments: Sequence[SentenceAlignment], total_sentences: int) -> tuple[float, float]:
    if total_sentences == 0:
        return 0.0, 0.0
    coverage = len(alignments) / total_sentences
    if not alignments:
        return coverage, 0.0
    match_hits = 0
    match_total = 0
    for alignment in alignments:
        for values in alignment.slot_matches.values():
            match_total += 1
            if values:
                match_hits += 1
    consistency = match_hits / match_total if match_total else 1.0
    return coverage, consistency
