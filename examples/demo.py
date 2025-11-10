from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lbrag import (
    DocumentSegment,
    HybridRetriever,
    LBRAGPipeline,
    PipelineOutput,
    PromptBuilder,
    PromptTemplate,
    Query,
    RetrievalCandidate,
    RetrievalConfig,
    TranslationRequest,
    TranslationResult,
    TranslationSelector,
    WeightingConfig,
)
from lbrag.pipeline import Generator
from lbrag.retrieval import Retriever
from lbrag.translation import SimpleSentenceSplitter


def _mock_segments() -> Sequence[DocumentSegment]:
    return (
        DocumentSegment(
            identifier="jp-1",
            text="火星は太陽系で四番目の惑星であり、赤い惑星として知られている。",
            language="ja",
        ),
        DocumentSegment(
            identifier="en-1",
            text="Mars is the fourth planet from the Sun and is known as the Red Planet.",
            language="en",
        ),
    )


class StaticRetriever(Retriever):
    def __init__(self, segments: Sequence[DocumentSegment], dense_score: float) -> None:
        self._segments = segments
        self._dense_score = dense_score

    def retrieve(self, query: Query, top_k: int) -> Sequence[RetrievalCandidate]:
        return tuple(
            RetrievalCandidate(segment=segment, dense_score=self._dense_score, rerank_score=None)
            for segment in self._segments[:top_k]
        )


class EchoTranslator:
    def translate(self, request: TranslationRequest) -> TranslationResult:
        text = request.segment.text
        if request.segment.language == "ja":
            translated = "Mars is the fourth planet from the Sun, known as the Red Planet."
        else:
            translated = text
        splitter = SimpleSentenceSplitter()
        sentences = splitter.split(translated)
        return TranslationResult(translated_text=translated, confidence=0.85, sentences=sentences)


class EchoGenerator(Generator):
    def generate(self, prompt: str) -> str:
        return "Mars is the fourth planet from the Sun."


@dataclass
class DummyReranker:
    score_value: float

    def score(self, query: Query, candidates: Sequence[DocumentSegment]) -> Sequence[float]:
        return tuple(self.score_value for _ in candidates)


def build_pipeline() -> LBRAGPipeline:
    segments = _mock_segments()
    retrievers = {
        "ja": StaticRetriever((segments[0],), dense_score=0.9),
        "en": StaticRetriever((segments[1],), dense_score=0.8),
    }
    reranker = DummyReranker(0.95)
    hybrid = HybridRetriever(retrievers, reranker=reranker, config=RetrievalConfig(alpha=0.4, top_k=4))
    selector = TranslationSelector(budget=32)
    template = PromptTemplate(
        system_instruction="You answer in {language} using evidence only.",
        citation_instruction="Cite the evidence identifiers in your reply.",
        answer_instruction="Respond in {language} with a concise summary.",
    )
    prompt_builder = PromptBuilder(template)
    generator = EchoGenerator()
    translator = EchoTranslator()
    pipeline = LBRAGPipeline(
        retriever=hybrid,
        translator=translator,
        generator=generator,
        prompt_builder=prompt_builder,
        translation_selector=selector,
        weighting=WeightingConfig(beta_search=0.6, beta_alignment=0.3, beta_slots=0.1),
    )
    return pipeline


def run_demo() -> PipelineOutput:
    pipeline = build_pipeline()
    query = Query(text="火星について教えて", language="ja")
    return pipeline.run(query)


if __name__ == "__main__":
    output = run_demo()
    print(output.prompt)
    print("\nAnswer:\n" + output.answer)
