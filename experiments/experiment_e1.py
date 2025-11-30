# experiments/experiment_e1.py

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from lbrag import (
    Query,
    DocumentSegment,  # 即使当前文件不直接用，也保留以防后续扩展
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
    TavilyRetriever,  # ✅ 使用现成的向量检索后端
)
from lbrag.retrieval import HybridRetriever, RetrievalConfig
from lbrag.types import EvidenceBlock  # RetrievalCandidate 不再需要

import dotenv

dotenv.load_dotenv()


# =========================
# 数据结构
# =========================

@dataclass
class Sample:
    id: str
    question: str
    question_lang: str
    answer: str


# =========================
# 读取 samples.json
# =========================

def load_samples(path: str) -> List[Sample]:
    def to_samples(objs: Sequence[dict]) -> List[Sample]:
        samples: List[Sample] = []
        for obj in objs:
            samples.append(
                Sample(
                    id=str(obj["id"]),
                    question=obj["question"],
                    question_lang=obj["question_lang"],
                    answer=obj["answer"],
                )
            )
        return samples

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw_stripped = raw.strip()
    if not raw_stripped:
        return []
    try:
        obj = json.loads(raw_stripped)
        if isinstance(obj, dict) and "data" in obj:
            records = obj["data"]
        elif isinstance(obj, list):
            records = obj
        else:
            records = [obj]
        return to_samples(records)
    except json.JSONDecodeError:
        samples: List[Sample] = []
        with open(path, "r", encoding="utf-8") as f2:
            for line in f2:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.extend(to_samples([obj]))
        return samples


# =========================
# no-RAG baseline：只用 LLM 自身知识
# =========================

@dataclass
class DirectPipeline:
    """
    不做检索，也不翻译。
    只用同一个 OpenAIChatGenerator，作为 no-RAG baseline。
    """
    generator: OpenAIChatGenerator
    template: PromptTemplate

    def run(self, query: Query) -> PipelineOutput:
        system_instruction = self.template.system_instruction.format(
            language=query.language
        )
        answer_instruction = self.template.answer_instruction.format(
            language=query.language
        )

        # 这里明确告诉模型：没有外部文档，只能靠自身知识
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


# =========================
# 构造 PromptBuilder
# =========================

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


# =========================
# 构造 RAG 系统 & baseline
# =========================

def build_systems() -> Dict[str, object]:
    """
    使用 Tavily 作为全局向量检索后端（真实 RAG）。
    不再依赖 per-query 的 docs_map / FixedRetriever。
    """

    # 真实的向量检索：Tavily API（需要环境变量 TAVILY_API_KEY）
    base_retriever = TavilyRetriever(
        api_key=None,          # None = 用环境变量 TAVILY_API_KEY
        include_domains=None,  # 例如可以改成 ["wikipedia.org"] 只查维基
    )

    # HybridRetriever 现在只是一个统一入口：里边只有一个 "web" 检索器，
    # 以后如果你要加别的检索器（比如本地 Qdrant），也可以继续往里塞。
    hybrid = HybridRetriever(
        retrievers={"web": base_retriever},
        reranker=None,  # 如需二次重排，可以换成 OpenAIListwiseReranker
        config=RetrievalConfig(alpha=0.5, top_k=20),
    )

    translator = OpenAITranslator()
    generator = OpenAIChatGenerator()
    builder = build_prompt_builder()
    wcfg = WeightingConfig(0.6, 0.2, 0.2)

    selector_multi = TranslationSelector(budget=0.0)      # multi：不翻译（相当于 MRAG / no-translate）
    selector_full = TranslationSelector(budget=1e9)       # cross：全部翻译（cRAG 风格）
    selector_lbrag = TranslationSelector(budget=35.0)     # lbrag：有预算约束

    # pivot 永远用英语的版本（和 CrossRAG 类似）
    def always_en_pivot(cands, lq: str) -> str:
        return "en"

    # ========== no-RAG baseline ==========
    direct = DirectPipeline(generator=generator, template=builder._template)

    # ========== 各种 RAG pipeline ==========
    multi = LBRAGPipeline(
        retriever=hybrid,
        retriever_alpha=None,      # 用 HybridRetriever._config.alpha
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

    return {
        "direct": direct,   # ✅ no-RAG baseline
        "multi": multi,     # MRAG 风格（多语检索，不翻译或少翻）
        "cross": cross,     # CrossRAG 风格（多语检索 + pivot 翻译）
        "lbrag": lbrag,     # 你自己的 LBRAG（带翻译预算）
    }


# =========================
# 评测指标（基本延续你原来的逻辑）
# =========================

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
    """
    简单的“语言一致性”打分：越像目标语言，分数越高（0~1）。
    这个是 char-level 的粗略度量，和你之前的版本保持一致风格。
    """
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
            # 其他语言就宽松一些
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
    # 兼容一下 metadata 里可能没有 token_count 的情况
    if block.metadata is None:
        return None
    return block.metadata.get("token_count")


# =========================
# 主实验逻辑
# =========================

def run_experiment(data_path: str, max_samples: int | None = None) -> None:
    samples = load_samples(data_path)
    if max_samples is not None:
        samples = samples[:max_samples]

    # 不再需要 docs_map / FixedRetriever
    systems = build_systems()

    # 聚合指标
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

    for s in samples:
        q = Query(text=s.question, language=s.question_lang, metadata={"id": s.id})
        for name, pipe in systems.items():
            out: PipelineOutput = pipe.run(q)  # type: ignore
            ans = out.answer or ""

            em = exact_match(ans, s.answer)
            f1 = f1_score_lang(ans, s.answer, s.question_lang)
            rlc = compute_rlc(ans, s.question_lang)
            rlc_ok = rlc_binary(ans, s.question_lang)
            cost = total_translation_tokens(out.evidence)

            agg[name]["em"] += em
            agg[name]["f1"] += f1
            agg[name]["rlc"] += rlc
            agg[name]["rlc_ok"] += rlc_ok
            agg[name]["cost"] += cost
            agg[name]["n"] += 1.0

    # ==== CNBE 基线：现在用 direct(no-RAG) 的 F1 作为基线 ====
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

        # direct 或 cost<=0 的系统，CNBE 设为 0
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
    # 这里改成你刚才生成的 MKQA 多语言样本
    path = "experiments/data/samples_mkqa_multi.json"
    run_experiment(path, max_samples=100)
