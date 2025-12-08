from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Sequence
from datetime import datetime
import time

from datasets import load_dataset

MKQA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"


def build_mkqa_samples(
    split: str = "train",
    langs: List[str] | None = None,
    max_quid: int | None = None,
) -> List[Dict]:
    if langs is None:
        langs = ["en", "ja", "de", "es", "zh_cn"]

    print("Loading MKQA jsonl from GitHub (no remote code)...")
    ds = load_dataset(
        "json",
        data_files={"train": MKQA_URL},
        split=split,
    )

    samples: List[Dict] = []
    exid_to_quid: Dict[str, int] = {}
    next_quid = 1

    for ex in ds:
        ex_id = str(ex.get("example_id", ""))
        if not ex_id:
            continue

        queries = ex["queries"]
        answers = ex["answers"]

        available_langs = []
        for lang in langs:
            q = queries.get(lang)
            ans_list = answers.get(lang)
            if q and ans_list:
                available_langs.append(lang)

        if not available_langs:
            continue

        if ex_id not in exid_to_quid:
            if max_quid is not None and len(exid_to_quid) >= max_quid:
                break
            exid_to_quid[ex_id] = next_quid
            next_quid += 1
        quid = exid_to_quid[ex_id]

        for lang in available_langs:
            q = queries[lang]
            ans_list = answers[lang]

            first_answer = ans_list[0]
            gold_text = first_answer.get("text", "")

            sample = {
                "id": f"mkqa_{ex_id}_{lang}",
                "quid": quid,
                "question": q,
                "question_lang": lang,
                "answer": gold_text,
            }
            samples.append(sample)

    print(f"Total unique quids (semantic questions): {len(exid_to_quid)}")
    print(f"Total samples: {len(samples)}")
    return samples


def summarize_dataset(
    samples: Sequence[Dict],
    langs: Sequence[str],
    repo_path: Path,
) -> None:
    quid_to_langs: Dict[int, set] = {}
    lang_to_quids: Dict[str, set] = {lang: set() for lang in langs}

    for s in samples:
        quid = int(s.get("quid", -1))
        lang = s.get("question_lang")
        if quid < 0 or lang is None:
            continue
        if quid not in quid_to_langs:
            quid_to_langs[quid] = set()
        quid_to_langs[quid].add(lang)
        if lang in lang_to_quids:
            lang_to_quids[lang].add(quid)
        else:
            lang_to_quids[lang] = {quid}

    total_quids = len(quid_to_langs)

    coverage_counts: Dict[int, int] = {}
    for _, langset in quid_to_langs.items():
        k = len(langset)
        coverage_counts[k] = coverage_counts.get(k, 0) + 1

    lines: List[str] = []
    lines.append(f"Total unique semantic questions (quids): {total_quids}")
    lines.append("")
    lines.append("Language coverage distribution per question:")
    for k in sorted(coverage_counts.keys(), reverse=True):
        v = coverage_counts[k]
        lines.append(f"  {k} languages: {v}")
    lines.append("")
    lines.append("Per-language question counts:")
    for lang in langs:
        cnt = len(lang_to_quids.get(lang, set()))
        lines.append(f"  {lang}: {cnt}")

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    with repo_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Dataset summary written to {repo_path}")


def main():
    langs = [
        "ar",
        "de",
        "en",
        "es",
        "fi",
        "fr",
        "he",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sv",
        "th",
        "tr",
        "vi",
        "zh_cn",
    ]

    max_quid = 10000

    start_time = time.time()

    samples = build_mkqa_samples(
        split="train",
        langs=langs,
        max_quid=max_quid,
    )
    end_time = time.time()

    base_data_dir = Path(__file__).resolve().parent.parent / "data"
    base_data_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")

    run_idx = 1
    while True:
        run_dir = base_data_dir / f"{date_str}_{run_idx}"
        if not run_dir.exists():
            break
        run_idx += 1

    run_dir.mkdir(parents=True, exist_ok=False)

    output_path = run_dir / "mkqa_samples.json"
    print(f"Saving {len(samples)} samples to {output_path} ...")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print("Done.")

    repo_path = run_dir / "dataset_repo.txt"
    summarize_dataset(samples, langs, repo_path)
    print(
        f"Taken {end_time - start_time} seconds for building {max_quid} MKQA samples(24 languages)"
    )


if __name__ == "__main__":
    main()
