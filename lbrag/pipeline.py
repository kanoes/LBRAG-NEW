from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Protocol, Sequence

from .prompting import PromptBuilder
from .retrieval import HybridRetriever, RetrievalConfig
from .selection import (
    ConfidenceEstimator,
    HeuristicConfidenceEstimator,
    TranslationCandidate,
    TranslationPlan,
    TranslationSelector,
)
from .translation import (
    AlignmentMode,
    RegexSentenceSplitter,
    SentenceSplitter,
    Translator,
    align_sentences,
    estimate_alignment_quality,
)
from .types import (
    EvidenceBlock,
    Query,
    RetrievalCandidate,
    SentenceAlignment,
    TranslationRequest,
    TranslationResult,
)


class Generator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class PivotStrategy(Enum):
    AUTO = "auto"
    ENGLISH = "english"
    QUERY = "query"
    CUSTOM = "custom"


@dataclass
class PipelineConfig:
    pivot_strategy: PivotStrategy = PivotStrategy.AUTO
    custom_pivot_language: Optional[str] = None
    alignment_mode: AlignmentMode = AlignmentMode.GREEDY
    enforce_sentence_ids: bool = True


@dataclass
class PipelineDiagnostics:
    translation_plan: TranslationPlan
    total_translation_tokens: float
    confidence_by_segment: Mapping[str, float]
    language_distribution: Mapping[str, int]


@dataclass(frozen=True)
class PipelineOutput:
    answer: str
    evidence: Sequence[EvidenceBlock]
    prompt: str
    diagnostics: PipelineDiagnostics


@dataclass
class WeightingConfig:
    beta_search: float = 0.6
    beta_alignment: float = 0.2
    beta_slots: float = 0.2

    def normalized(self) -> WeightingConfig:
        total = self.beta_search + self.beta_alignment + self.beta_slots
        if total == 0:
            return WeightingConfig(1.0, 0.0, 0.0)
        return WeightingConfig(
            beta_search=self.beta_search / total,
            beta_alignment=self.beta_alignment / total,
            beta_slots=self.beta_slots / total,
        )


@dataclass
class LBRAGPipeline:
    retriever: HybridRetriever
    translator: Translator
    generator: Generator
    prompt_builder: PromptBuilder
    translation_selector: TranslationSelector
    confidence_estimator: Optional[ConfidenceEstimator] = None
    weighting_alpha: Optional[float] = None
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    sentence_splitter: SentenceSplitter = field(default_factory=RegexSentenceSplitter)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)

    def __post_init__(self) -> None:
        if self.confidence_estimator is None:
            self.confidence_estimator = HeuristicConfidenceEstimator(
                translator=self.translator,
                splitter=self.sentence_splitter,
                alignment=self.pipeline_config.alignment_mode,
            )
        self._preview_cache: Dict[str, TranslationResult] = {}
        self._weighting = self.weighting.normalized()

    def run(self, query: Query) -> PipelineOutput:
        assert self.confidence_estimator is not None
        candidates = self.retriever.retrieve(query)
        alpha = self.weighting_alpha if self.weighting_alpha is not None else self._resolve_alpha()
        translation_candidates = self._build_translation_candidates(query, candidates, alpha)
        plan = self.translation_selector.select(tuple(translation_candidates.values()))
        evidence_blocks = self._build_evidence_blocks(query, candidates, plan.selected, alpha)
        prompt = self.prompt_builder.build(query.text, evidence_blocks, query.language)
        answer = self.generator.generate(prompt)
        diagnostics = PipelineDiagnostics(
            translation_plan=plan,
            total_translation_tokens=plan.spent,
            confidence_by_segment={cid: translation_candidates[cid].confidence for cid in translation_candidates},
            language_distribution=self._compute_language_distribution(candidates),
        )
        return PipelineOutput(answer=answer, evidence=evidence_blocks, prompt=prompt, diagnostics=diagnostics)

    def _resolve_alpha(self) -> float:
        config: RetrievalConfig = getattr(self.retriever, "config", RetrievalConfig())
        return config.alpha

    def _build_translation_candidates(
        self,
        query: Query,
        candidates: Sequence[RetrievalCandidate],
        alpha: float,
    ) -> Dict[str, TranslationCandidate]:
        prepared: Dict[str, TranslationCandidate] = {}
        for candidate in candidates:
            segment = candidate.segment
            pivot_language = self._decide_pivot_language(query, segment)
            if pivot_language == segment.language:
                confidence = 1.0
                cost = 0.0
                preview = None
                details = {"combined": 1.0}
            else:
                estimate = self.confidence_estimator.estimate(segment, pivot_language)
                preview = estimate.preview
                if preview is not None:
                    self._preview_cache[segment.identifier] = preview
                confidence = estimate.value
                details = dict(estimate.details)
                request = TranslationRequest(segment=segment, target_language=pivot_language)
                cost = estimate.preview.metadata.get("token_count", 0.0) if estimate.preview else 0.0
                if cost <= 0:
                    cost = self.translator.estimate_cost(request)
            prepared[segment.identifier] = TranslationCandidate(
                segment=segment,
                relevance=candidate.final_score(alpha=alpha),
                confidence=confidence,
                cost=max(cost, 0.0),
                pivot_language=pivot_language,
                confidence_details=details,
                preview=preview,
            )
        return prepared

    def _build_evidence_blocks(
        self,
        query: Query,
        candidates: Sequence[RetrievalCandidate],
        selected: Sequence[TranslationCandidate],
        alpha: float,
    ) -> Sequence[EvidenceBlock]:
        selected_ids = {candidate.segment.identifier for candidate in selected}
        lookup = {candidate.segment.identifier: candidate for candidate in selected}
        blocks = []
        for candidate in candidates:
            segment_id = candidate.segment.identifier
            if segment_id in selected_ids:
                block = self._translate_and_align(query, candidate, lookup[segment_id], alpha)
            else:
                block = self._build_untranslated_block(candidate, alpha)
            blocks.append(block)
        blocks.sort(key=lambda block: block.weight, reverse=True)
        return tuple(blocks)

    def _translate_and_align(
        self,
        query: Query,
        candidate: RetrievalCandidate,
        translation_candidate: TranslationCandidate,
        alpha: float,
    ) -> EvidenceBlock:
        segment = candidate.segment
        pivot_language = translation_candidate.pivot_language
        translation = self._preview_cache.get(segment.identifier)
        if translation is None:
            if pivot_language == segment.language:
                sentences = self.sentence_splitter.split(segment.text)
                translation = TranslationResult(
                    translated_text=segment.text,
                    confidence=translation_candidate.confidence,
                    sentences=sentences,
                    metadata={"token_count": 0.0},
                )
            else:
                request = TranslationRequest(segment=segment, target_language=pivot_language)
                translation = self.translator.translate(request)
        if translation is None:
            request = TranslationRequest(segment=segment, target_language=pivot_language)
            translation = self.translator.translate(request)
        self._preview_cache[segment.identifier] = translation
        source_sentences = self.sentence_splitter.split(segment.text)
        alignments = align_sentences(
            source_sentences,
            translation.sentences,
            mode=self.pipeline_config.alignment_mode,
        )
        coverage, slot_score = estimate_alignment_quality(alignments, len(source_sentences))
        metadata = {
            "translation_confidence": translation.confidence,
            "coverage": coverage,
            "slot_consistency": slot_score,
            "relevance": candidate.final_score(alpha=alpha),
            "pivot_language": pivot_language,
            "token_count": translation.metadata.get("token_count", 0.0),
            "confidence_details": translation_candidate.confidence_details,
        }
        weight = self._compute_weight(candidate, coverage, slot_score, alpha)
        return EvidenceBlock(
            segment=segment,
            translated_text=translation.translated_text,
            alignment=self._inject_sentence_ids(segment.identifier, alignments) if self.pipeline_config.enforce_sentence_ids else alignments,
            weight=weight,
            pivot_language=pivot_language,
            metadata=metadata,
        )

    def _build_untranslated_block(self, candidate: RetrievalCandidate, alpha: float) -> EvidenceBlock:
        metadata = {
            "translation_confidence": 0.0,
            "coverage": 0.0,
            "slot_consistency": 0.0,
            "relevance": candidate.final_score(alpha=alpha),
            "pivot_language": candidate.segment.language,
        }
        return EvidenceBlock(
            segment=candidate.segment,
            translated_text=None,
            alignment=tuple(),
            weight=self._compute_weight(candidate, 0.0, 0.0, alpha),
            pivot_language=candidate.segment.language,
            metadata=metadata,
        )

    def _compute_weight(
        self,
        candidate: RetrievalCandidate,
        coverage: float,
        slot_score: float,
        alpha: float,
    ) -> float:
        search_weight = candidate.final_score(alpha=alpha)
        w = self._weighting
        score = (
            w.beta_search * search_weight
            + w.beta_alignment * coverage
            + w.beta_slots * slot_score
        )
        return min(1.0, max(0.0, score))

    def _decide_pivot_language(self, query: Query, segment) -> str:
        strategy = self.pipeline_config.pivot_strategy
        if strategy is PivotStrategy.ENGLISH:
            return "en"
        if strategy is PivotStrategy.QUERY:
            return query.language
        if strategy is PivotStrategy.CUSTOM and self.pipeline_config.custom_pivot_language:
            return self.pipeline_config.custom_pivot_language
        if strategy is PivotStrategy.AUTO:
            if segment.language == query.language:
                return query.language
            if segment.language == "en":
                return "en"
            if query.language != "en":
                return query.language
            return "en"
        return query.language

    def _inject_sentence_ids(
        self, segment_id: str, alignments: Sequence[SentenceAlignment]
    ) -> Sequence[SentenceAlignment]:
        enriched: list[SentenceAlignment] = []
        for alignment in alignments:
            target_sentence = f"[{segment_id}:s{alignment.target_index}] {alignment.target_sentence}"
            enriched.append(
                SentenceAlignment(
                    source_sentence=alignment.source_sentence,
                    target_sentence=target_sentence,
                    source_index=alignment.source_index,
                    target_index=alignment.target_index,
                    slot_matches=alignment.slot_matches,
                )
            )
        return tuple(enriched)

    def _compute_language_distribution(
        self, candidates: Sequence[RetrievalCandidate]
    ) -> Mapping[str, int]:
        distribution: Dict[str, int] = {}
        for candidate in candidates:
            distribution[candidate.segment.language] = distribution.get(candidate.segment.language, 0) + 1
        return distribution
