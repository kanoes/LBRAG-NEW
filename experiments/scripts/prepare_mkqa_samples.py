from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset


MKQA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"


def build_mkqa_samples(
    split: str = "train",          # 这里只保留 train，实际也只有这一种
    langs: List[str] | None = None,
    max_per_lang: int | None = None,
) -> List[Dict]:
    """
    从 MKQA JSONL 抽取样本，转换为格式：
    [
      {id, question, question_lang, answer},
      ...
    ]
    """
    if langs is None:
        # 注意：官方支持的语言码在 mkqa.py 里，中文是 zh_cn/zh_hk/zh_tw，没有 "zh"
        # 你可以按需改
        langs = ["en", "ja", "de", "es", "zh_cn"]

    print("Loading MKQA jsonl from GitHub (no remote code)...")
    # 这里用内置的 "json" builder，就不会触发脚本加载
    ds = load_dataset(
        "json",
        data_files={"train": MKQA_URL},
        split="train",   # 只有一个 split
    )

    samples: List[Dict] = []
    lang_counts: Dict[str, int] = {lang: 0 for lang in langs}

    for ex in ds:
        # 原脚本里是 example_id
        ex_id = str(ex.get("example_id", ""))
        queries = ex["queries"]      # {lang: string}
        answers = ex["answers"]      # {lang: [ {type, entity, text, aliases} ]}

        for lang in langs:
            q = queries.get(lang)
            ans_list = answers.get(lang)

            if not q or not ans_list:
                continue

            if max_per_lang is not None and lang_counts[lang] >= max_per_lang:
                continue

            # 这里用第一个答案的 text 字段
            first_answer = ans_list[0]
            gold_text = first_answer.get("text", "")

            sample = {
                "id": f"mkqa_{ex_id}_{lang}",
                "question": q,
                "question_lang": lang,
                "answer": gold_text,
            }
            samples.append(sample)
            lang_counts[lang] += 1

    print("Collected samples per language:")
    for lang in langs:
        print(f"  {lang}: {lang_counts[lang]}")

    return samples


def main():
    # 你可以按需调整语言列表和数量
    langs = ["en", "ja", "de", "es", "zh_cn"]

    samples = build_mkqa_samples(
        split="train",
        langs=langs,
        max_per_lang=500,  # 每种语言最多 500 条，方便快速实验
    )

    # 用 pathlib 处理路径，自动建目录
    output_path = Path("experiments/data/samples_mkqa_multi.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 这行很关键！

    print(f"Saving {len(samples)} samples to {output_path} ...")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
