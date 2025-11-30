from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Callable
from .prompting import PromptBuilder
from .retrieval import HybridRetriever
from .selection import TranslationCandidate, TranslationSelector
from .translation import (
    SimpleSentenceSplitter,
    SentenceSplitter,
    Translator,
    estimate_alignment_quality,
    greedy_sentence_alignment,
)
from .integrations import estimate_kappa
from .types import EvidenceBlock, Query, RetrievalCandidate, TranslationRequest

PivotFn = Callable[[Sequence[RetrievalCandidate], str], str]


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class WeightingConfig:
    beta_search: float = 0.6
    beta_alignment: float = 0.2
    beta_slots: float = 0.2

    def normalize(self) -> "WeightingConfig":
        total = self.beta_search + self.beta_alignment + self.beta_slots
        if total == 0:
            return WeightingConfig(1.0, 0.0, 0.0)
        return WeightingConfig(
            beta_search=self.beta_search / total,
            beta_alignment=self.beta_alignment / total,
            beta_slots=self.beta_slots / total,
        )


@dataclass(frozen=True)
class PipelineOutput:
    answer: str
    evidence: Sequence[EvidenceBlock]
    prompt: str


def default_pivot(
    cands: Sequence[RetrievalCandidate], lq: str, tau: float = 0.6
) -> str:
    total = max(len(cands), 1)
    lq_ratio = sum(1 for c in cands if c.segment.language == lq) / total
    return lq if lq_ratio >= tau else "en"


class LBRAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        retriever_alpha: float | None,
        translator: Translator,
        generator: Generator,
        prompt_builder: PromptBuilder,
        translation_selector: TranslationSelector,
        weighting: WeightingConfig = WeightingConfig(),
        sentence_splitter: Optional[SentenceSplitter] = None,
        pivot_selector: PivotFn = default_pivot,
    ) -> None:
        self._retriever = retriever
        self._alpha = (
            retriever_alpha if retriever_alpha is not None else retriever._config.alpha
        )
        self._translator = translator
        self._generator = generator
        self._prompt_builder = prompt_builder
        self._selector = translation_selector
        self._weighting = weighting.normalize()
        self._splitter = sentence_splitter or SimpleSentenceSplitter()
        self._pivot_selector = pivot_selector

    def run(self, query: Query) -> PipelineOutput:
        candidates = self._retriever.retrieve(query)
        self._pivot = self._pivot_selector(candidates, query.language)
        plan = self._selector.select(self._to_translation_candidates(candidates))
        evidence_blocks = self._build_evidence_blocks(query, candidates, plan.selected)
        prompt = self._prompt_builder.build(query.text, evidence_blocks, query.language)
        answer = self._generator.generate(prompt)
        return PipelineOutput(answer=answer, evidence=evidence_blocks, prompt=prompt)

    def _to_translation_candidates(
        self, candidates: Sequence[RetrievalCandidate]
    ) -> Sequence[TranslationCandidate]:
        translated_candidates = []
        for candidate in candidates:
            metadata = candidate.segment.metadata
            confidence = float(metadata.get("translation_confidence", 1.0))
            cost = float(
                metadata.get(
                    "translation_cost", self._estimate_cost(candidate.segment.text)
                )
            )
            translated_candidates.append(
                TranslationCandidate(
                    segment=candidate.segment,
                    relevance=candidate.final_score(alpha=self._alpha),
                    confidence=max(0.0, min(1.0, confidence)),
                    cost=max(cost, 1.0),
                )
            )
        return tuple(translated_candidates)

    def _estimate_cost(self, text: str) -> float:
        tokens = max(len(text) // 3, 1)
        return float(tokens)

    def _build_evidence_blocks(
        self,
        query: Query,
        candidates: Sequence[RetrievalCandidate],
        selected: Sequence[TranslationCandidate],
    ) -> Sequence[EvidenceBlock]:
        selected_ids = {candidate.segment.identifier for candidate in selected}
        blocks = []
        for candidate in candidates:
            if candidate.segment.identifier in selected_ids:
                block = self._translate_and_align(candidate)
            else:
                block = self._build_untranslated_block(candidate)
            blocks.append(block)
        blocks.sort(key=lambda b: b.weight, reverse=True)
        return tuple(blocks)

    def _translate_and_align(self, candidate: RetrievalCandidate) -> EvidenceBlock:
        segment = candidate.segment
        request = TranslationRequest(segment=segment, target_language=self._pivot)
        result = self._translator.translate(request)
        source_sentences = self._splitter.split(segment.text)
        alignments = greedy_sentence_alignment(source_sentences, result.sentences)
        coverage, slots = estimate_alignment_quality(alignments, len(source_sentences))
        back_trans = None
        if hasattr(self._translator, "back_translate"):
            try:
                back_trans = self._translator.back_translate(
                    result.translated_text, segment.language
                )
            except Exception:
                back_trans = None
        kappa = estimate_kappa(segment.text, result.translated_text, back_trans, slots)
        weight = self._compute_weight(candidate, coverage, slots)
        metadata = {
            "translation_confidence": result.confidence,
            "coverage": coverage,
            "slot_consistency": slots,
            "kappa": kappa,
            "token_count": result.metadata.get("token_count"),
            "used_pivot": self._pivot,
        }
        return EvidenceBlock(
            segment=segment,
            translated_text=result.translated_text,
            alignment=alignments,
            weight=weight,
            metadata=metadata,
        )

    def _build_untranslated_block(self, candidate: RetrievalCandidate) -> EvidenceBlock:
        weight = self._compute_weight(candidate, 0.0, 0.0)
        return EvidenceBlock(
            segment=candidate.segment,
            translated_text=None,
            alignment=tuple(),
            weight=weight,
            metadata={"translation_confidence": 0.0},
        )

    def _compute_weight(
        self, candidate: RetrievalCandidate, coverage: float, slots: float
    ) -> float:
        w = self._weighting
        score = candidate.final_score(alpha=self._alpha)
        return (
            w.beta_search * score + w.beta_alignment * coverage + w.beta_slots * slots
        )
