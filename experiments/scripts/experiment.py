import os
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence
from datetime import datetime
import time
import random

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
    base_retriever = OpenAIEmbeddingRetriever(documents=docs, exclude_same_language=True)
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

    plt.style.use("ggplot")

    x = range(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width/2 for i in x], em_vals, width, label="EM")
    ax.bar([i + width/2 for i in x], f1_vals, width, label="F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"EM / F1 by system ({run_id})")
    ax.legend()
    for i, v in enumerate(em_vals):
        ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(f1_vals):
        ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"{run_id}_em_f1.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, rlc_vals)
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems)
    ax.set_ylabel("RLC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Response Language Consistency ({run_id})")
    for i, v in enumerate(rlc_vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"{run_id}_rlc.png"))
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x, cost_vals, label="Avg translation tokens")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(systems)
    ax1.set_ylabel("Avg translation tokens")
    ax1.set_title(f"Translation cost & CNBE ({run_id})")

    ax2 = ax1.twinx()
    ax2.plot(list(x), cnbe_vals, marker="o", label="CNBE")
    ax2.set_ylabel("CNBE")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"{run_id}_cost_cnbe.png"))
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


def run_experiment(data_path: str, max_samples: int | None = None) -> None:
    print("[run_experiment] start")
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    base_dir = os.path.join("experiments", "results", run_id)
    dirs = ensure_dirs(base_dir)
    answers_path = os.path.join(dirs["answers"], f"{run_id}_answers.jsonl")
    metrics_path = os.path.join(dirs["metrics"], f"{run_id}_metrics.json")

    samples = load_samples(data_path)
    samples = select_samples_by_quid(samples, max_samples=max_samples, seed=42)
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
    with open(answers_path, "w", encoding="utf-8") as fout:
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
                row = {
                    "run_id": run_id,
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
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(" done")

    baseline_f1 = 0.0
    if "direct" in agg and agg["direct"]["n"] > 0:
        baseline_f1 = agg["direct"]["f1"] / max(agg["direct"]["n"], 1.0)
    print("=== Experiment E1 Results ===")
    print(f"Run ID: {run_id}")
    print(f"Results dir: {base_dir}")
    print(f"Data: {data_path}")
    print(f"Baseline (for CNBE): direct (no-RAG), F1={baseline_f1:.3f}")
    print("")

    metrics_out: Dict[str, Dict[str, float]] = {}
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
        metrics_out[name] = {
            "em": em,
            "f1": f1,
            "rlc": rlc,
            "rlc_ok": rlc_ok,
            "cost": cost,
            "cnbe": cnbe,
            "n": n,
        }
        print(
            f"{name:6s}",
            "EM={:.3f}".format(em),
            "F1={:.3f}".format(f1),
            "RLC={:.3f}".format(rlc),
            "RLC_OK={:.3f}".format(rlc_ok),
            "AvgTranslateTokens={:.1f}".format(cost),
            "CNBE={:.5f}".format(cnbe),
        )

    meta = {
        "run_id": run_id,
        "data_path": data_path,
        "max_samples": max_samples,
        "baseline_f1": baseline_f1,
        "metrics": metrics_out,
    }
    with open(metrics_path, "w", encoding="utf-8") as fmeta:
        json.dump(meta, fmeta, ensure_ascii=False, indent=2)

    plot_metrics(metrics_out, dirs["figures"], run_id)
    print(f"[run_experiment] answers saved to {answers_path}")
    print(f"[run_experiment] metrics saved to {metrics_path}")
    print(f"[run_experiment] figures saved to {dirs['figures']}")


if __name__ == "__main__":
    start_time = time.time()
    path = "experiments/data/samples_mkqa_multi.json"
    max_samples = 10
    run_experiment(path, max_samples=max_samples)
    end_time = time.time()
    print(f"Test sample {max_samples} Time taken: {end_time - start_time} seconds")
