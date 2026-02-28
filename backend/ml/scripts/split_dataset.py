from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Callable

from .common import load_jsonl, write_jsonl

DEFAULT_DATASET = "backend/ml/data/dataset_ranker.jsonl"
DEFAULT_OUT_DIR = "backend/ml/data/splits"
_WORD_RE = re.compile(r"[a-zA-Z]+")
_SPACE_RE = re.compile(r"\s+")


def _positive_count(row: dict) -> int:
    candidates = row.get("candidates", [])
    return sum(1 for item in candidates if int(item.get("label", 0)) >= 2)


def _norm(value: str) -> str:
    return _SPACE_RE.sub(" ", (value or "").strip().lower())


def _shape(text: str) -> str:
    lowered = _norm(text)
    return _WORD_RE.sub("w", lowered)


def _family_key(row: dict) -> str:
    task = str(row.get("task", "")).strip().lower()
    payload = row.get("input", {}) or {}
    if task in {"suggest_blank", "suggest_selection", "suggest_sentence", "rewrite"}:
        sentence = str(payload.get("sentence", ""))
        has_selection = "1" if payload.get("selection") else "0"
        return f"{task}|{_shape(sentence)}|sel={has_selection}"
    if task == "lexical":
        lexical_task = _norm(str(payload.get("lexical_task", "")))
        word = _norm(str(payload.get("word", "")))
        return f"{task}|{lexical_task}|{word}"
    if task == "constraints":
        relation = _norm(str(payload.get("relation", "")))
        rhyme_with = _norm(str(payload.get("rhyme_with", "")))
        meaning_of = _norm(str(payload.get("meaning_of", "")))
        return f"{task}|{relation}|{rhyme_with}|{meaning_of}"
    if task == "oneword":
        query = str(payload.get("query", ""))
        return f"{task}|{_shape(query)}"
    return f"{task}|misc"


def _split_rows_random(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    filtered = [row for row in rows if row.get("id")]
    if len(filtered) < 12:
        raise ValueError("Need at least 12 grouped rows before splitting train/val/test.")
    filtered.sort(key=lambda row: (row.get("task", ""), _positive_count(row), row.get("id")))
    random.Random(seed).shuffle(filtered)
    total = len(filtered)
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, train_end + int(total * val_ratio))
    if val_end >= total:
        val_end = total - 1
    train_rows = filtered[:train_end]
    val_rows = filtered[train_end:val_end]
    test_rows = filtered[val_end:]
    if not val_rows or not test_rows:
        raise ValueError("Split produced empty val/test partitions. Adjust ratios or add rows.")
    return train_rows, val_rows, test_rows


def _split_rows_hard(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    family_key_fn: Callable[[dict], str],
) -> tuple[list[dict], list[dict], list[dict]]:
    filtered = [row for row in rows if row.get("id")]
    if len(filtered) < 12:
        raise ValueError("Need at least 12 grouped rows before splitting train/val/test.")

    families: dict[str, list[dict]] = {}
    for row in filtered:
        key = family_key_fn(row)
        families.setdefault(key, []).append(row)

    family_keys = sorted(families.keys())
    random.Random(seed).shuffle(family_keys)

    total = len(filtered)
    target_train = int(total * train_ratio)
    target_val = int(total * val_ratio)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for key in family_keys:
        bucket = families[key]
        if len(train_rows) < target_train:
            train_rows.extend(bucket)
        elif len(val_rows) < target_val:
            val_rows.extend(bucket)
        else:
            test_rows.extend(bucket)

    # Safety rebalance for tiny datasets.
    if not val_rows and train_rows:
        pivot = max(1, len(train_rows) // 10)
        val_rows = train_rows[-pivot:]
        train_rows = train_rows[:-pivot]
    if not test_rows and val_rows:
        pivot = max(1, len(val_rows) // 4)
        test_rows = val_rows[-pivot:]
        val_rows = val_rows[:-pivot]

    if not val_rows or not test_rows:
        raise ValueError("Hard split produced empty val/test partitions. Add rows or adjust ratios.")
    return train_rows, val_rows, test_rows


def _split_rows(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    split_mode: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    if split_mode == "hard":
        return _split_rows_hard(rows, train_ratio, val_ratio, seed, _family_key)
    return _split_rows_random(rows, train_ratio, val_ratio, seed)


def _split_tiny_bucket(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    if not rows:
        return [], [], []
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if count == 1:
        return shuffled, [], []
    if count == 2:
        return [shuffled[0]], [shuffled[1]], []
    if count == 3:
        return [shuffled[0]], [shuffled[1]], [shuffled[2]]
    if count == 4:
        return [shuffled[0], shuffled[1]], [shuffled[2]], [shuffled[3]]
    train_cut = max(1, int(count * 0.6))
    val_cut = max(train_cut + 1, int(count * 0.8))
    return shuffled[:train_cut], shuffled[train_cut:val_cut], shuffled[val_cut:]


def _split_rows_task_stratified(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    split_mode: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        task = str(row.get("task", "")).strip().lower() or "unknown"
        by_task.setdefault(task, []).append(row)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for idx, task in enumerate(sorted(by_task.keys())):
        task_rows = by_task[task]
        try:
            t_rows, v_rows, te_rows = _split_rows(
                task_rows,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed + idx,
                split_mode=split_mode,
            )
        except ValueError:
            t_rows, v_rows, te_rows = _split_tiny_bucket(task_rows, seed=seed + idx)
        train_rows.extend(t_rows)
        val_rows.extend(v_rows)
        test_rows.extend(te_rows)

    if not val_rows or not test_rows:
        return _split_rows(rows, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed, split_mode=split_mode)
    return train_rows, val_rows, test_rows


def split_dataset(
    dataset_path: str,
    out_dir: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    regression_size: int,
    split_mode: str,
    stratify_by_task: bool,
) -> None:
    rows = load_jsonl(dataset_path)
    if stratify_by_task:
        train_rows, val_rows, test_rows = _split_rows_task_stratified(
            rows,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            split_mode=split_mode,
        )
    else:
        train_rows, val_rows, test_rows = _split_rows(
            rows,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            split_mode=split_mode,
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / "train.jsonl"
    val_path = out_path / "val.jsonl"
    test_path = out_path / "test.jsonl"
    regression_path = out_path / "regression.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)
    write_jsonl(regression_path, test_rows[: max(1, min(regression_size, len(test_rows)))])

    train_ids = {row.get("id") for row in train_rows}
    val_ids = {row.get("id") for row in val_rows}
    test_ids = {row.get("id") for row in test_rows}
    overlap = len((train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids))

    print(f"Split mode: {split_mode}")
    print(f"Task stratified: {stratify_by_task}")
    print(f"Train rows: {len(train_rows)} -> {train_path.as_posix()}")
    print(f"Val rows: {len(val_rows)} -> {val_path.as_posix()}")
    print(f"Test rows: {len(test_rows)} -> {test_path.as_posix()}")
    print(f"Regression rows: {min(regression_size, len(test_rows))} -> {regression_path.as_posix()}")
    print(f"ID overlap across splits: {overlap}")
    train_tasks = {}
    val_tasks = {}
    test_tasks = {}
    for row in train_rows:
        task = str(row.get("task", "")).strip().lower()
        train_tasks[task] = train_tasks.get(task, 0) + 1
    for row in val_rows:
        task = str(row.get("task", "")).strip().lower()
        val_tasks[task] = val_tasks.get(task, 0) + 1
    for row in test_rows:
        task = str(row.get("task", "")).strip().lower()
        test_tasks[task] = test_tasks.get(task, 0) + 1
    print(f"Task distribution train: {train_tasks}")
    print(f"Task distribution val: {val_tasks}")
    print(f"Task distribution test: {test_tasks}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split ranker dataset into train/val/test JSONL files.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Input grouped JSONL dataset.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for split files.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--split-mode",
        choices=["random", "hard"],
        default="random",
        help="random=id-randomized split, hard=template-family holdout split",
    )
    parser.add_argument(
        "--no-stratify-by-task",
        action="store_true",
        help="Disable task-stratified split (enabled by default).",
    )
    parser.add_argument("--regression-size", type=int, default=200, help="Rows to freeze for regression tests.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ratio = min(max(args.train_ratio, 0.6), 0.9)
    val_ratio = min(max(args.val_ratio, 0.05), 0.2)
    split_dataset(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=args.seed,
        regression_size=args.regression_size,
        split_mode=args.split_mode,
        stratify_by_task=not args.no_stratify_by_task,
    )
