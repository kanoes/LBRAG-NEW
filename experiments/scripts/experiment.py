import os
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence
from datetime import datetime
import time
import random
import math
from statistics import mean, stdev


import matplotlib.pyplot as plt

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
from utils.llm import LLMClient, format_usage_summary

import dotenv

dotenv.load_dotenv()


@dataclass
class Sample:
    id: str
    question: str
    question_lang: str
    answer: str | None
    quid: int | None = None


def load_samples(path: str) -> List[Sample]:
    print(f"[load_samples] path={path}")

    def to_samples(objs: Sequence[dict]) -> List[Sample]:
        samples: List[Sample] = []
        for obj in objs:
            quid_val = obj.get("quid")
            try:
                quid = int(quid_val) if quid_val is not None else None
            except Exception:
                quid = None
            samples.append(
                Sample(
                    id=str(obj["id"]),
                    question=obj["question"],
                    question_lang=obj["question_lang"],
                    answer=obj.get("answer"),
                    quid=quid,
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
            "- [NONE] No external documents are available.",          self.template.citation_instruction,
            answer_instruction,
        ]
        prompt = "\n\n".join(prompt_parts)
        answer = self.generator.generate(prompt)
        return PipelineOutput(answer=answer, evidence=tuple(), prompt=prompt)


@dataclass
class OpenAISemanticJudge:
    model: str = "gpt-4o"
    api_key: str | None = None
    llm_client: LLMClient | None = None

    def __post_init__(self) -> None:
        self._llm = self.llm_client or LLMClient(api_key=self.api_key)

    def score(
        self,
        question: str,
        gold_answer: str,
        pred_answer: str,
        target_lang: str,
    ) -> float:
        if not gold_answer.strip():
            return 0.0
        if not pred_answer.strip():
            return 0.0

        system_msg = (
            "You are a strict but fair evaluation assistant. "
            "Given a question, a gold reference answer, and a model's predicted answer, "
            "you judge how semantically equivalent the predicted answer is to the gold answer. "
            "You only care about factual content relevant to the question, not style or wording. "
            "Return a JSON object with fields 'score' (0.0 to 1.0) and 'explanation' (short text)."
        )

        user_msg = f"""
[Question] (language: {target_lang})
{question}

[Gold Answer]
{gold_answer}

[Predicted Answer]
{pred_answer}

Please evaluate how semantically equivalent the predicted answer is to the gold answer,
on a scale from 0.0 (completely wrong or unrelated) to 1.0 (fully correct and equivalent).

Guidelines:
- Small wording differences or rephrasings should still get a high score.
- Numeric answers with the same value but different units formatting (e.g. "4429m" vs "4429 メートル") should be treated as equivalent.
- If the predicted answer misses key parts of the gold answer, lower the score.
- If the predicted answer contradicts the gold answer, score near 0.
- Only consider content needed to answer the question.

Respond ONLY with a JSON object like:
{{"score": 0.94, "explanation": "..."}}.
        """.strip()

        try:
            content, _ = self._llm.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        except Exception:
            return 0.0

        try:
            m = re.search(r"\{.*\}", content, flags=re.S)
            if not m:
                return 0.0
            obj = json.loads(m.group(0))
            score = float(obj.get("score", 0.0))
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0
            return score
        except Exception:
            return 0.0


def build_prompt_builder() -> PromptBuilder:
    template = PromptTemplate(
        system_instruction=(
            "You are a careful multilingual assistant. "
            "Always answer in {language} only. "
            "Do not mix other languages. "
            "Based on the evidence provided, give your best answer. "
        ),
        citation_instruction=(
            "Do NOT include any citations, IDs, brackets like [..], or evidence markers in your final answer."
        ),
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


def build_systems(samples: Sequence[Sample], llm_client: LLMClient, data_dir: str) -> Dict[str, object]:
    print("[build_systems] start")
    docs = samples_to_documents(samples)
    print("[build_systems] creating OpenAIEmbeddingRetriever (embedding all docs)...")
    base_retriever = OpenAIEmbeddingRetriever(
        documents=docs, exclude_same_language=0.5, llm_client=llm_client, cache_dir=data_dir
    )
    print("[build_systems] embeddings ready")
    print("[build_systems] creating reranker...")
    reranker = OpenAIListwiseReranker(llm_client=llm_client)
    hybrid = HybridRetriever(
        retrievers={"mkqa": base_retriever},
        reranker=reranker,
        config=RetrievalConfig(alpha=0.5, top_k=10),
    )
    translator = OpenAITranslator(llm_client=llm_client)
    generator = OpenAIChatGenerator(llm_client=llm_client)
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
    pattern = r"[^" \
              r"0-9" \
              r"a-z\u00c0-\u024f" \
              r"\u0400-\u04ff" \
              r"\u0590-\u05ff" \
              r"\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff" \
              r"\u0e00-\u0e7f" \
              r"\u1100-\u11ff\uac00-\ud7af" \
              r"\u3040-\u309f\u30a0-\u30ff" \
              r"\u4e00-\u9fff" \
              r"]+"
    s = re.sub(pattern, " ", s)
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
    if precision + recall == 0:
        return 0.0
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


def ensure_dirs(base_dir: str) -> Dict[str, str]:
    answers_dir = os.path.join(base_dir, "answers")
    metrics_dir = os.path.join(base_dir, "metrics")
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(answers_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return {
        "answers": answers_dir,
        "metrics": metrics_dir,
        "figures": figures_dir,
    }


def plot_metrics(metrics: Dict[str, Dict[str, float]], figures_dir: str, run_id: str):
    systems = ["direct", "multi", "cross", "lbrag"]
    systems = [s for s in systems if s in metrics]

    f1_vals = [metrics[s]["f1"] for s in systems]
    em_vals = [metrics[s]["em"] for s in systems]
    rlc_vals = [metrics[s]["rlc"] for s in systems]
    cost_vals = [metrics[s]["cost"] for s in systems]
    cnbe_vals = [metrics[s]["cnbe"] for s in systems]
    semantic_score_vals = [metrics[s]["semantic_score"] for s in systems]
    
    f1_stds = [metrics[s].get("f1_std", 0) for s in systems]
    em_stds = [metrics[s].get("em_std", 0) for s in systems]
    rlc_stds = [metrics[s].get("rlc_std", 0) for s in systems]
    cost_stds = [metrics[s].get("cost_std", 0) for s in systems]
    cnbe_stds = [metrics[s].get("cnbe_std", 0) for s in systems]
    semantic_score_stds = [metrics[s].get("semantic_score_std", 0) for s in systems]

    plt.style.use("ggplot")
    x = range(len(systems))
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], em_vals, width, yerr=em_stds, 
           label="EM", capsize=5, alpha=0.8)
    ax.bar([i + width/2 for i in x], f1_vals, width, yerr=f1_stds, 
           label="F1", capsize=5, alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"EM / F1 by system ({run_id})")
    ax.legend()
    for i, (v, std) in enumerate(zip(em_vals, em_stds)):
        ax.text(i - width/2, v + std + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, (v, std) in enumerate(zip(f1_vals, f1_stds)):
        ax.text(i + width/2, v + std + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"{run_id}_em_f1.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, rlc_vals, yerr=rlc_stds, capsize=5, alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems)
    ax.set_ylabel("RLC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Response Language Consistency ({run_id})")
    for i, (v, std) in enumerate(zip(rlc_vals, rlc_stds)):
        ax.text(i, v + std + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"{run_id}_rlc.png"), dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x, cost_vals, yerr=cost_stds, capsize=5, alpha=0.8, 
            label="Avg translation tokens", color='skyblue')
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(systems)
    ax1.set_ylabel("Avg translation tokens")
    ax1.set_title(f"Translation cost & CNBE ({run_id})")

    ax2 = ax1.twinx()
    ax2.errorbar(list(x), cnbe_vals, yerr=cnbe_stds, marker="o", 
                 label="CNBE", capsize=5, color='orange', linewidth=2)
    ax2.set_ylabel("CNBE")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"{run_id}_cost_cnbe.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, semantic_score_vals, yerr=semantic_score_stds, capsize=5, alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems)
    ax.set_ylabel("Semantic Agreement Score (SAS)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"LLM-based Semantic Agreement ({run_id})")
    for i, (v, std) in enumerate(zip(semantic_score_vals, semantic_score_stds)):
        ax.text(i, v + std + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"{run_id}_semantic.png"), dpi=150)
    plt.close(fig)


def select_samples_by_quid(
    samples: Sequence[Sample],
    max_samples: int | None = None,
    seed: int = 42,
) -> List[Sample]:
    rng = random.Random(seed)
    groups: Dict[int | str, List[Sample]] = {}
    order: List[int | str] = []

    for s in samples:
        if s.quid is not None:
            key: int | str = s.quid
        else:
            key = s.id.rsplit("_", 1)[0]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s)

    total_quids = len(order)
    if max_samples is None:
        target = total_quids
    else:
        target = min(max_samples, total_quids)

    selected: List[Sample] = []
    for key in order[:target]:
        cand_group = groups[key]
        choice = rng.choice(cand_group)
        selected.append(choice)

    if max_samples is not None and max_samples > total_quids:
        print(
            f"[select_samples_by_quid] requested max_samples={max_samples}, "
            f"but only {total_quids} quids available; using {total_quids}."
        )

    print(
        f"[select_samples_by_quid] selected {len(selected)} samples "
        f"from {total_quids} quids"
    )
    return selected


def run_experiment(data_path: str, num_test_queries: int | None = None) -> None:
    print("[run_experiment] start")
    start_time = datetime.now()
    answer_rows: List[dict] = []
    llm_client = LLMClient()

    all_samples = load_samples(data_path)
    print(f"[run_experiment] loaded {len(all_samples)} samples as knowledge base")
    
    data_dir = os.path.dirname(data_path)
    systems = build_systems(all_samples, llm_client=llm_client, data_dir=data_dir)
    
    test_queries = select_samples_by_quid(all_samples, max_samples=num_test_queries, seed=42)
    if num_test_queries is not None:
        test_queries = test_queries[:num_test_queries]
        print(f"[run_experiment] selected {len(test_queries)} test queries")
    else:
        print(f"[run_experiment] using all {len(test_queries)} samples as test queries")
    semantic_judge = OpenAISemanticJudge(llm_client=llm_client)
    agg: Dict[str, Dict[str, float]] = {}
    for name in systems:
        agg[name] = {
            "em": 0.0,
            "f1": 0.0,
            "rlc": 0.0,
            "rlc_ok": 0.0,
            "cost": 0.0,
            "semantic_score": 0.0,
            "n": 0.0,
        }
    per_sample: Dict[str, Dict[str, List[float]]] = {}
    for name in systems:
        per_sample[name] = {
            "em": [],
            "f1": [],
            "rlc": [],
            "rlc_ok": [],
            "cost": [],
            "semantic_score": [],
        }

    total_queries = len(test_queries)
    for idx, s in enumerate(test_queries, start=1):
        print(f"[run_experiment] test query {idx}/{total_queries} id={s.id} lang={s.question_lang}")
        gold_text = s.answer or ""
        q = Query(text=s.question, language=s.question_lang, metadata={"id": s.id})
        for name, pipe in systems.items():
            print(f"  [system:{name}] running...", end="", flush=True)
            out: PipelineOutput = pipe.run(q)  # type: ignore
            ans = out.answer or ""
            print(f" retrieved {len(out.evidence)} evidence blocks", end="", flush=True)
            em = exact_match(ans, gold_text)
            f1 = f1_score_lang(ans, gold_text, s.question_lang)
            rlc = compute_rlc(ans, s.question_lang)
            rlc_ok = rlc_binary(ans, s.question_lang)
            cost = total_translation_tokens(out.evidence)
            semantic_score = semantic_judge.score(s.question, gold_text, ans, s.question_lang)
            agg[name]["em"] += em
            agg[name]["f1"] += f1
            agg[name]["rlc"] += rlc
            agg[name]["rlc_ok"] += rlc_ok
            agg[name]["cost"] += cost
            agg[name]["n"] += 1.0
            agg[name]["semantic_score"] += semantic_score
            per_sample[name]["em"].append(em)
            per_sample[name]["f1"].append(f1)
            per_sample[name]["rlc"].append(rlc)
            per_sample[name]["rlc_ok"].append(rlc_ok)
            per_sample[name]["cost"].append(cost)
            per_sample[name]["semantic_score"].append(semantic_score)
            evidence_list = []
            for ev_block in out.evidence:
                ev_info = {
                    "id": ev_block.segment.identifier,
                    "language": ev_block.segment.language,
                    "original_text": ev_block.segment.text[:200] + "..." if len(ev_block.segment.text) > 200 else ev_block.segment.text,
                    "translated_text": (ev_block.translated_text[:200] + "..." if ev_block.translated_text and len(ev_block.translated_text) > 200 else ev_block.translated_text) if ev_block.translated_text else None,
                    "weight": ev_block.weight,
                    "metadata": ev_block.metadata,
                }
                evidence_list.append(ev_info)
            
            answer_rows.append(
                {
                    "sample_id": s.id,
                    "sample_lang": s.question_lang,
                    "system": name,
                    "question": s.question,
                    "gold_answer": gold_text,
                    "pred_answer": ans,
                    "em": em,
                    "f1": f1,
                    "rlc": rlc,
                    "rlc_ok": rlc_ok,
                    "translate_tokens": cost,
                    "semantic_score": semantic_score,
                    "num_evidence": len(out.evidence),
                    "evidence": evidence_list,
                    "prompt": out.prompt,
                }
            )
            print(" done")

    for name in systems:
        if name == "direct":
            continue
        
        cnbe_list: List[float] = []
        for i in range(len(per_sample[name]["f1"])):
            f1_rag = per_sample[name]["f1"][i]
            f1_baseline = per_sample["direct"]["f1"][i]
            cost_i = per_sample[name]["cost"][i]
            
            if cost_i and cost_i > 0.0:
                # CNBE = (F1_improvement) / cost
                cnbe_list.append((f1_rag - f1_baseline) / cost_i)
            else:
                cnbe_list.append(0.0)
        
        per_sample[name]["cnbe"] = cnbe_list
    
    per_sample["direct"]["cnbe"] = [0.0] * len(per_sample["direct"]["f1"])
    
    def _mean_std(xs: List[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        if len(xs) < 2:
            return float(xs[0]), 0.0
        return float(mean(xs)), float(stdev(xs))
    
    print("\n" + "="*100)
    print("Experiment Results Summary")
    print("="*100)
    
    metrics_out: Dict[str, Dict[str, float]] = {}
    for name in systems:
        n = float(len(per_sample[name]["f1"])) if name in per_sample else max(agg[name]["n"], 1.0)

        em_mean, em_std = _mean_std(per_sample[name]["em"])
        f1_mean, f1_std = _mean_std(per_sample[name]["f1"])
        rlc_mean, rlc_std = _mean_std(per_sample[name]["rlc"])
        rlc_ok_mean, rlc_ok_std = _mean_std(per_sample[name]["rlc_ok"])
        cost_mean, cost_std = _mean_std(per_sample[name]["cost"])
        sem_mean, sem_std = _mean_std(per_sample[name]["semantic_score"])
        cnbe_mean, cnbe_std = _mean_std(per_sample[name]["cnbe"])

        metrics_out[name] = {
            "em": em_mean,
            "em_std": em_std,
            "f1": f1_mean,
            "f1_std": f1_std,
            "rlc": rlc_mean,
            "rlc_std": rlc_std,
            "rlc_ok": rlc_ok_mean,
            "rlc_ok_std": rlc_ok_std,
            "cost": cost_mean,
            "cost_std": cost_std,
            "cnbe": cnbe_mean,
            "cnbe_std": cnbe_std,
            "semantic_score": sem_mean,
            "semantic_score_std": sem_std,
            "n": n,
        }

        print(
            f"{name:8s} | "
            f"EM={em_mean:.3f}±{em_std:.3f} | "
            f"F1={f1_mean:.3f}±{f1_std:.3f} | "
            f"RLC={rlc_mean:.3f}±{rlc_std:.3f} | "
            f"Cost={cost_mean:.1f}±{cost_std:.1f} | "
            f"CNBE={cnbe_mean:.5f}±{cnbe_std:.5f} | "
            f"Sem={sem_mean:.3f}±{sem_std:.3f}"
        )
    print("="*100 + "\n")

    baseline_f1 = 0.0
    if "direct" in per_sample and per_sample["direct"]["f1"]:
        baseline_f1 = float(mean(per_sample["direct"]["f1"]))
    
    end_time = datetime.now()
    run_id = end_time.strftime("%Y%m%d_%H%M")
    base_dir = os.path.join("experiments", "results", run_id)
    dirs = ensure_dirs(base_dir)
    answers_path = os.path.join(dirs["answers"], f"{run_id}_answers.jsonl")
    metrics_path = os.path.join(dirs["metrics"], f"{run_id}_metrics.json")
    
    print("=== Experiment E1 Results ===")
    print(f"Run ID: {run_id}")
    print(f"Results dir: {base_dir}")
    print(f"Data: {data_path}")
    print(f"Baseline (for CNBE): direct (no-RAG), F1={baseline_f1:.3f}")
    print("")

    
    meta = {
        "run_id": run_id,
        "data_path": data_path,
        "knowledge_base_size": len(all_samples),
        "num_test_queries": num_test_queries,
        "actual_test_queries": len(test_queries),
        "baseline_f1": baseline_f1,
        "metrics": metrics_out,
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "llm_usage": format_usage_summary(llm_client.usage),
    }
    
    with open(answers_path, "w", encoding="utf-8") as fout:
        for row in answer_rows:
            out_row = {"run_id": run_id, **row}
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    
    with open(metrics_path, "w", encoding="utf-8") as fmeta:
        json.dump(meta, fmeta, ensure_ascii=False, indent=2)

    plot_metrics(metrics_out, dirs["figures"], run_id)
    print(f"[run_experiment] answers saved to {answers_path}")
    print(f"[run_experiment] metrics saved to {metrics_path}")
    print(f"[run_experiment] figures saved to {dirs['figures']}")
    print(f"[run_experiment] LLM usage summary: {format_usage_summary(llm_client.usage)}")


if __name__ == "__main__":
    start_time = time.time()
    path = "experiments/data/20251208_1/mkqa_samples.json"
    num_test_queries = 1
    run_experiment(path, num_test_queries=num_test_queries)
    end_time = time.time()
    print(f"Test queries: {num_test_queries}, Time taken: {end_time - start_time} seconds")
