from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence
from openai import OpenAI
from .retrieval import Reranker, Retriever
from .selection import ConfidenceEstimate, ConfidenceEstimator
from .translation import (
    RegexSentenceSplitter,
    SentenceSplitter,
    SupportsBackTranslation,
    Translator,
)
from .types import (
    DocumentSegment,
    Query,
    RetrievalCandidate,
    TranslationRequest,
    TranslationResult,
)


class OpenAIChatGenerator:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        system_instruction: str | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._model = model
        self._system = system_instruction or "You are a helpful multilingual assistant."

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()


@dataclass
class OpenAITranslator(Translator, SupportsBackTranslation):
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    splitter: SentenceSplitter = field(default_factory=RegexSentenceSplitter)

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))

    def translate(self, request: TranslationRequest) -> TranslationResult:
        prompt = (
            "Translate the following passage to {lang} preserving numbers, dates, and names.\n\n"
            "Source ({source_lang}):\n{source}\n"
        ).format(
            lang=request.target_language,
            source_lang=request.segment.language,
            source=request.segment.text,
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You translate precisely without omitting information.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        translated = (response.choices[0].message.content or "").strip()
        sentences = self.splitter.split(translated)
        metadata = {"token_count": self.estimate_cost(request)}
        return TranslationResult(
            translated_text=translated,
            confidence=1.0,
            sentences=sentences,
            metadata=metadata,
        )

    def estimate_cost(self, request: TranslationRequest) -> float:
        tokens = max(len(request.segment.text) // 3, 1)
        return float(tokens)

    def back_translate(self, text: str, source_language: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You translate precisely without omitting information.",
                },
                {
                    "role": "user",
                    "content": f"Translate this passage to {source_language}:\n{text}",
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()


@dataclass
class OpenAIEmbeddingRetriever(Retriever):
    documents: Sequence[DocumentSegment]
    embedding_model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    exclude_same_language: bool = False

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
        self._vectors = self._embed_documents(self.documents)

    def retrieve(self, query: Query, top_k: int) -> Sequence[RetrievalCandidate]:
        embedding = (
            self._client.embeddings.create(model=self.embedding_model, input=query.text)
            .data[0]
            .embedding
        )
        scored = []
        for vector, segment in zip(self._vectors, self.documents):
            if self.exclude_same_language and segment.language == query.language:
                continue
            score = self._dot(vector, embedding)
            scored.append(
                RetrievalCandidate(
                    segment=segment, dense_score=score, rerank_score=None
                )
            )
        scored.sort(key=lambda c: c.dense_score, reverse=True)
        return tuple(scored[:top_k])

    def _embed_documents(
        self, documents: Sequence[DocumentSegment]
    ) -> Sequence[Sequence[float]]:
        texts = [doc.text for doc in documents]
        response = self._client.embeddings.create(
            model=self.embedding_model, input=texts
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))


@dataclass
class QdrantRetriever(Retriever):
    collection: str
    url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    embedding_model: str = "text-embedding-3-small"
    api_key_openai: Optional[str] = None

    def __post_init__(self) -> None:
        self._openai = OpenAI(
            api_key=self.api_key_openai or os.getenv("OPENAI_API_KEY")
        )

    def retrieve(self, query: Query, top_k: int) -> Sequence[RetrievalCandidate]:
        embedding = (
            self._openai.embeddings.create(model=self.embedding_model, input=query.text)
            .data[0]
            .embedding
        )
        payload = {"vector": embedding, "limit": top_k, "with_payload": True}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        import requests
        response = requests.post(
            f"{self.url}/collections/{self.collection}/points/search",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        points = response.json().get("result", [])
        candidates = []
        for point in points:
            payload = point.get("payload", {})
            identifier = str(point.get("id", payload.get("id")))
            text = payload.get("text") or payload.get("content", "")
            language = payload.get("language", query.language)
            segment = DocumentSegment(
                identifier=identifier, text=text, language=language, metadata=payload
            )
            candidates.append(
                RetrievalCandidate(
                    segment=segment,
                    dense_score=float(point.get("score", 0.0)),
                    rerank_score=None,
                )
            )
        return tuple(candidates[:top_k])


@dataclass
class OpenAIListwiseReranker(Reranker):
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))

    def score(
        self, query: Query, candidates: Sequence[DocumentSegment]
    ) -> Sequence[float]:
        if not candidates:
            return tuple()
        prompt = self._build_prompt(query, candidates)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Score relevance between 0 and 1."},
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        scores = self._parse_scores(content, len(candidates))
        return tuple(scores)

    def _build_prompt(self, query: Query, candidates: Sequence[DocumentSegment]) -> str:
        lines = [f"Query ({query.language}): {query.text}", "\nCandidates:"]
        for idx, candidate in enumerate(candidates, start=1):
            lines.append(f"[{idx}] ({candidate.language}) {candidate.text}")
        lines.append("\nProvide scores as JSON array of floats in candidate order.")
        return "\n".join(lines)

    @staticmethod
    def _parse_scores(content: str, expected: int) -> Sequence[float]:
        try:
            m = re.search(r"\[[^\]]+\]", content, flags=re.S)
            if m:
                arr = json.loads(m.group(0))
                if isinstance(arr, list) and len(arr) >= expected:
                    return [float(arr[i]) for i in range(expected)]
        except Exception:
            pass
        nums = re.findall(r"(?:^|\s)(0(?:\.\d+)?|1(?:\.0+)?)", content)
        if len(nums) >= expected:
            return [float(nums[i]) for i in range(expected)]
        return [0.0] * expected


@dataclass
class StaticConfidenceEstimator(ConfidenceEstimator):
    confidence: float = 1.0

    def estimate(
        self, segment: DocumentSegment, target_language: str
    ) -> ConfidenceEstimate:
        return ConfidenceEstimate(
            value=self.confidence, details={"static": self.confidence}, preview=None
        )


def estimate_kappa(
    source: str, translated: str, back_trans: str | None, slot_consistency: float
) -> float:
    ls = len(source) if source else 1
    lt = len(translated) if translated else 1
    len_keep = min(lt / ls, ls / lt)
    len_keep = max(0.0, min(1.0, len_keep))
    bt = 0.0
    if back_trans:
        s = set(source.split())
        b = set(back_trans.split())
        u = s | b
        bt = (len(s & b) / len(u)) if u else 0.0
    return 0.4 * len_keep + 0.3 * bt + 0.3 * slot_consistency
