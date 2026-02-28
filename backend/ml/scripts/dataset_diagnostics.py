from __future__ import annotations

import argparse
import json
from collections import Counter

from .common import load_jsonl


def run(dataset_path: str, exclude_gold_seed: bool = False, task_filter: str = "all") -> None:
    rows = load_jsonl(dataset_path)
    if not rows:
        raise ValueError("Dataset is empty.")

    task_counts: Counter[str] = Counter()
    total_candidates = 0
    total_positives = 0
    samples_no_positive = 0
    samples_le_one_positive = 0
    samples_no_candidates = 0

    for row in rows:
        task = str(row.get("task", "")).strip().lower()
        if task_filter == "non_rewrite" and task == "rewrite":
            continue
        if task_filter == "rewrite" and task != "rewrite":
            continue
        task_counts[task] += 1
        candidates = row.get("candidates", [])
        if exclude_gold_seed:
            candidates = [
                c for c in candidates if str(c.get("source", "")).strip().lower() != "gold_seed"
            ]
        pos_count = sum(1 for c in candidates if int(c.get("label", 0)) >= 2)

        total_candidates += len(candidates)
        total_positives += pos_count
        if not candidates:
            samples_no_candidates += 1
        if pos_count == 0:
            samples_no_positive += 1
        if pos_count <= 1:
            samples_le_one_positive += 1

    sample_count = sum(task_counts.values())
    if sample_count == 0:
        raise ValueError("No rows remain after applying task filter.")
    report = {
        "dataset": dataset_path,
        "rows": sample_count,
        "exclude_gold_seed": exclude_gold_seed,
        "task_filter": task_filter,
        "avg_candidates_per_row": round(total_candidates / sample_count, 3),
        "avg_positives_per_row": round(total_positives / sample_count, 3),
        "samples_with_no_candidates": samples_no_candidates,
        "samples_with_no_positive": samples_no_positive,
        "samples_with_le_1_positive": samples_le_one_positive,
        "oracle_hit_at_1_upper_bound": round((sample_count - samples_no_positive) / sample_count, 4),
        "task_distribution": dict(task_counts),
    }
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect label/coverage diagnostics for ranker dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to grouped JSONL dataset (e.g., split test/val file).",
    )
    parser.add_argument(
        "--exclude-gold-seed",
        action="store_true",
        help="Ignore candidates with source=gold_seed while computing diagnostics.",
    )
    parser.add_argument(
        "--task-filter",
        choices=["non_rewrite", "rewrite", "all"],
        default="all",
        help="Filter diagnostics by task bucket.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        dataset_path=args.dataset,
        exclude_gold_seed=args.exclude_gold_seed,
        task_filter=args.task_filter,
    )
