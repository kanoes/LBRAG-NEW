import json
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from lbrag import (
    Query,
    DocumentSegment,
    LBRAGPipeline,
    WeightingConfig,
    PromptBuilder,
    PromptTemplate,
    PipelineOutput,
)
from lbrag.selection import TranslationSelector
from lbrag.integrations import (
    OpenAIChatGenerator,
    OpenAITranslator,
    OpenAIEmbeddingRetriever,
    OpenAIListwiseReranker,
)
from lbrag.retrieval import HybridRetriever, RetrievalConfig
from lbrag.types import EvidenceBlock

import dotenv

dotenv.load_dotenv()


@dataclass
class Sample:
    id: str
    question: str
    question_lang: str
    answer: str | None


def load_samples(path: str) -> List[Sample]:
    print(f"[load_samples] path={path}")
    def to_samples(objs: Sequence[dict]) -> List[Sample]:
        samples: List[Sample] = []
        for obj in objs:
            samples.append(
                Sample(
                    id=str(obj["id"]),
                    question=obj["question"],
                    question_lang=obj["question_lang"],
                    answer=obj.get("answer"),
                )
            )
        return samples

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw_stripped = raw.strip()
    if not raw_stripped:
        print("[load_samples] empty file")
        return []
    try:
        obj = json.loads(raw_stripped)
        if isinstance(obj, dict) and "data" in obj:
            records = obj["data"]
        elif isinstance(obj, list):
            records = obj
        else:
            records = [obj]
        samples = to_samples(records)
        print(f"[load_samples] loaded {len(samples)} samples (JSON array/dict)")
        return samples
    except json.JSONDecodeError:
        samples: List[Sample] = []
        with open(path, "r", encoding="utf-8") as f2:
            for line in f2:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.extend(to_samples([obj]))
        print(f"[load_samples] loaded {len(samples)} samples (JSONL)")
        return samples


@dataclass
class DirectPipeline:
    generator: OpenAIChatGenerator
    template: PromptTemplate

    def run(self, query: Query) -> PipelineOutput:
        system_instruction = self.template.system_instruction.format(
            language=query.language
        )
        answer_instruction = self.template.answer_instruction.format(
            language=query.language
        )
        prompt_parts = [
            system_instruction,
            f"Question: {query.text}",
            "Evidence (ranked):",
            "- [NONE] No external documents are available.",
            "Use only your internal knowledge. If you are not sure, say you are not sure.",
            self.template.citation_instruction,
            answer_instruction,
        ]
        prompt = "\n\n".join(prompt_parts)
        answer = self.generator.generate(prompt)
        return PipelineOutput(answer=answer, evidence=tuple(), prompt=prompt)


def build_prompt_builder() -> PromptBuilder:
    template = PromptTemplate(
        system_instruction=(
            "You are a careful multilingual assistant. "
            "Always answer in {language} only. "
            "Do not mix other languages. "
            "If the evidence is insufficient, say so."
        ),
        citation_instruction="Use [ID] to cite evidence when you rely on it.",
        answer_instruction="Answer only with a short answer in {language}, without explanation.",
    )
    return PromptBuilder(template)


def samples_to_documents(samples: Sequence[Sample]) -> List[DocumentSegment]:
    print(f"[samples_to_documents] building {len(samples)} documents")
    docs: List[DocumentSegment] = []
    for s in samples:
        ans = s.answer or ""
        text = s.question + "\n\n" + ans
        docs.append(
            DocumentSegment(
                identifier=s.id,
                text=text,
                language=s.question_lang,
                metadata={"source": "mkqa"},
            )
        )
    return docs


def build_systems(samples: Sequence[Sample]) -> Dict[str, object]:
    print("[build_systems] start")
    docs = samples_to_documents(samples)
    print("[build_systems] creating OpenAIEmbeddingRetriever (embedding all docs)...")
    base_retriever = OpenAIEmbeddingRetriever(documents=docs)
    print("[build_systems] embeddings ready")
    print("[build_systems] creating reranker...")
    reranker = OpenAIListwiseReranker()
    hybrid = HybridRetriever(
        retrievers={"mkqa": base_retriever},
        reranker=reranker,
        config=RetrievalConfig(alpha=0.5, top_k=20),
    )
    translator = OpenAITranslator()
    generator = OpenAIChatGenerator()
    builder = build_prompt_builder()
    wcfg = WeightingConfig(0.6, 0.2, 0.2)
    selector_multi = TranslationSelector(budget=0.0)
    selector_full = TranslationSelector(budget=1e9)
    selector_lbrag = TranslationSelector(budget=35.0)

    def always_en_pivot(cands, lq: str) -> str:
        return "en"

    direct = DirectPipeline(generator=generator, template=builder._template)

    multi = LBRAGPipeline(
        retriever=hybrid,
        retriever_alpha=None,
        translator=translator,
        generator=generator,
        prompt_builder=builder,
        translation_selector=selector_multi,
        weighting=wcfg,
    )

    cross = LBRAGPipeline(
        retriever=hybrid,
        retriever_alpha=None,
        translator=translator,
        generator=generator,
        prompt_builder=builder,
        translation_selector=selector_full,
        weighting=wcfg,
        pivot_selector=always_en_pivot,
    )

    lbrag = LBRAGPipeline(
        retriever=hybrid,
        retriever_alpha=None,
        translator=translator,
        generator=generator,
        prompt_builder=builder,
        translation_selector=selector_lbrag,
        weighting=wcfg,
    )

    systems = {
        "direct": direct,
        "multi": multi,
        "cross": cross,
        "lbrag": lbrag,
    }
    print(f"[build_systems] done, systems={list(systems.keys())}")
    return systems


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff\u3040-\u30ff]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score_lang(pred: str, gold: str, lang: str) -> float:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if not pred_norm and not gold_norm:
        return 1.0
    if not pred_norm or not gold_norm:
        return 0.0
    if lang in ("ja", "zh", "zh_cn", "zh_tw"):
        p_tokens = list(pred_norm.replace(" ", ""))
        g_tokens = list(gold_norm.replace(" ", ""))
    else:
        p_tokens = pred_norm.split()
        g_tokens = gold_norm.split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for t in g_tokens:
        counts[t] = counts.get(t, 0) + 1
    overlap = 0
    for t in p_tokens:
        c = counts.get(t, 0)
        if c > 0:
            overlap += 1
            counts[t] = c - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def compute_rlc(text: str, lang: str) -> float:
    total = 0
    hits = 0
    for ch in text:
        if ch.isspace():
            continue
        if ch.isdigit():
            continue
        if ch in ".,!?;:()[]{}-–—'\"/":
            continue
        total += 1
        if lang.startswith("ja"):
            if ("\u3040" <= ch <= "\u309f") or ("\u30a0" <= ch <= "\u30ff") or ("\u4e00" <= ch <= "\u9fff"):
                hits += 1
        elif lang.startswith("zh"):
            if "\u4e00" <= ch <= "\u9fff":
                hits += 1
        elif lang in ("en", "de", "es", "fr", "it", "pt"):
            if ch.isalpha():
                hits += 1
        else:
            if ch.isalpha() or ("\u4e00" <= ch <= "\u9fff") or ("\u3040" <= ch <= "\u30ff"):
                hits += 1
    if total == 0:
        return 1.0
    return hits / total


def rlc_binary(text: str, lang: str, threshold: float = 0.6) -> float:
    score = compute_rlc(text, lang)
    return 1.0 if score >= threshold else 0.0


def total_translation_tokens(evidence: Sequence[EvidenceBlock]) -> float:
    total = 0.0
    for block in evidence:
        v = evidence_block_token_count(block)
        if v is None:
            continue
        try:
            total += float(v)
        except Exception:
            continue
    return total


def evidence_block_token_count(block: EvidenceBlock):
    if block.metadata is None:
        return None
    return block.metadata.get("token_count")


def run_experiment(data_path: str, max_samples: int | None = None) -> None:
    print("[run_experiment] start")
    samples = load_samples(data_path)
    if max_samples is not None:
        samples = samples[:max_samples]
        print(f"[run_experiment] truncated to {len(samples)} samples")
    else:
        print(f"[run_experiment] total samples={len(samples)}")
    systems = build_systems(samples)
    agg: Dict[str, Dict[str, float]] = {}
    for name in systems:
        agg[name] = {
            "em": 0.0,
            "f1": 0.0,
            "rlc": 0.0,
            "rlc_ok": 0.0,
            "cost": 0.0,
            "n": 0.0,
        }
    total_samples = len(samples)
    for idx, s in enumerate(samples, start=1):
        print(f"[run_experiment] sample {idx}/{total_samples} id={s.id} lang={s.question_lang}")
        gold_text = s.answer or ""
        q = Query(text=s.question, language=s.question_lang, metadata={"id": s.id})
        for name, pipe in systems.items():
            print(f"  [system:{name}] running...", end="", flush=True)
            out: PipelineOutput = pipe.run(q)  # type: ignore
            ans = out.answer or ""
            em = exact_match(ans, gold_text)
            f1 = f1_score_lang(ans, gold_text, s.question_lang)
            rlc = compute_rlc(ans, s.question_lang)
            rlc_ok = rlc_binary(ans, s.question_lang)
            cost = total_translation_tokens(out.evidence)
            agg[name]["em"] += em
            agg[name]["f1"] += f1
            agg[name]["rlc"] += rlc
            agg[name]["rlc_ok"] += rlc_ok
            agg[name]["cost"] += cost
            agg[name]["n"] += 1.0
            print(" done")
    baseline_f1 = 0.0
    if "direct" in agg and agg["direct"]["n"] > 0:
        baseline_f1 = agg["direct"]["f1"] / max(agg["direct"]["n"], 1.0)
    print("=== Experiment E1 Results ===")
    print(f"Data: {data_path}")
    print(f"Baseline (for CNBE): direct (no-RAG), F1={baseline_f1:.3f}")
    print("")
    for name in systems:
        n = max(agg[name]["n"], 1.0)
        em = agg[name]["em"] / n
        f1 = agg[name]["f1"] / n
        rlc = agg[name]["rlc"] / n
        rlc_ok = agg[name]["rlc_ok"] / n
        cost = agg[name]["cost"] / n
        if name == "direct" or cost <= 0.0:
            cnbe = 0.0
        else:
            cnbe = (f1 - baseline_f1) / cost if cost > 0.0 else 0.0
        print(
            f"{name:6s}",
            "EM={:.3f}".format(em),
            "F1={:.3f}".format(f1),
            "RLC={:.3f}".format(rlc),
            "RLC_OK={:.3f}".format(rlc_ok),
            "AvgTranslateTokens={:.1f}".format(cost),
            "CNBE={:.5f}".format(cnbe),
        )


if __name__ == "__main__":
    path = "experiments/data/samples_mkqa_multi.json"
    run_experiment(path, max_samples=10)
