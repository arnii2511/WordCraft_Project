from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .eval_reranker import _feature_text, _prob_to_score
from .common import load_jsonl

DEFAULT_DATASET = "backend/ml/data/splits/test.jsonl"
DEFAULT_ARTIFACT = "backend/ml/models/reranker.pkl"


def _len_bucket(value: str) -> str:
    size = len((value or "").split())
    if size <= 1:
        return "1"
    if size <= 3:
        return "2-3"
    if size <= 6:
        return "4-6"
    return "7+"


def _extract_pattern(task: str, row: dict[str, Any]) -> str:
    payload = row.get("input", {}) or {}
    context = str(payload.get("context", "neutral") or "neutral").strip().lower()
    mode = str(payload.get("mode", "") or "").strip().lower()
    if task == "constraints":
        relation = str(payload.get("relation", "synonym") or "synonym").strip().lower()
        rhyme_with = str(payload.get("rhyme_with", "") or "").strip().lower()
        meaning_of = str(payload.get("meaning_of", "") or "").strip().lower()
        return (
            f"relation={relation}|ctx={context}|rhyme_words={_len_bucket(rhyme_with)}"
            f"|target_words={_len_bucket(meaning_of)}"
        )
    if task == "lexical":
        lexical_task = str(payload.get("lexical_task", "") or "").strip().lower()
        return f"lexical_task={lexical_task}|ctx={context}"
    if task == "oneword":
        query = str(payload.get("query", "") or "").strip().lower()
        return f"query_words={_len_bucket(query)}|ctx={context}"
    if task in {"suggest_blank", "suggest_selection", "suggest_sentence"}:
        sentence = str(payload.get("sentence", "") or "").strip().lower()
        has_blank = int("[blank]" in sentence or "____" in sentence)
        has_selection = int(bool(payload.get("selection")))
        return (
            f"mode={mode or 'write'}|ctx={context}|blank={has_blank}|sel={has_selection}"
            f"|sent_words={_len_bucket(sentence)}"
        )
    if task == "rewrite":
        sentence = str(payload.get("sentence", "") or "").strip().lower()
        return f"rewrite|ctx={context}|sent_words={_len_bucket(sentence)}"
    return f"task={task}|ctx={context}"


def _flatten_rows(
    rows: list[dict[str, Any]],
    exclude_gold_seed: bool,
    task_filter: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    flat: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = str(row.get("task", "")).strip().lower()
        if task_filter == "non_rewrite" and task == "rewrite":
            continue
        if task_filter == "rewrite" and task != "rewrite":
            continue
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            continue
        by_id[sample_id] = row
        for candidate in row.get("candidates", []):
            if exclude_gold_seed and str(candidate.get("source", "")).strip().lower() == "gold_seed":
                continue
            text = (candidate.get("text") or "").strip()
            if not text:
                continue
            flat.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "feature_text": _feature_text(row, candidate),
                    "label": int(candidate.get("label", 0)),
                    "baseline_score": float(candidate.get("model_score", 0.0) or 0.0),
                    "candidate": text,
                }
            )
    return flat, by_id


def _rank_map(
    flat_rows: list[dict[str, Any]],
    scorer: str,
    artifact_path: str | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if scorer == "baseline":
        for row in flat_rows:
            grouped[row["sample_id"]].append(
                {
                    "label": int(row["label"]),
                    "score": float(row["baseline_score"]),
                    "candidate": row["candidate"],
                    "task": row["task"],
                }
            )
        return grouped

    if not artifact_path:
        raise ValueError("artifact path is required for scorer=reranker")
    artifact = pickle.loads(Path(artifact_path).read_bytes())
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    x = [row["feature_text"] for row in flat_rows]
    x_vec = vectorizer.transform(x)
    probabilities = model.predict_proba(x_vec)
    scores = _prob_to_score(probabilities, model.classes_)

    for row, score in zip(flat_rows, scores):
        grouped[row["sample_id"]].append(
            {
                "label": int(row["label"]),
                "score": float(score),
                "candidate": row["candidate"],
                "task": row["task"],
            }
        )
    return grouped


def _task_metrics(task_rows: list[list[dict[str, Any]]]) -> dict[str, Any]:
    samples = len(task_rows)
    if samples == 0:
        return {}
    hit1 = hit3 = hit5 = 0
    no_positive = 0
    avg_candidates = 0.0
    avg_positives = 0.0
    for ranked in task_rows:
        ranked_sorted = sorted(ranked, key=lambda x: x["score"], reverse=True)
        positives = [row for row in ranked_sorted if row["label"] >= 2]
        avg_candidates += len(ranked_sorted)
        avg_positives += len(positives)
        if not positives:
            no_positive += 1
            continue
        if ranked_sorted[:1] and ranked_sorted[0]["label"] >= 2:
            hit1 += 1
        if any(row["label"] >= 2 for row in ranked_sorted[:3]):
            hit3 += 1
        if any(row["label"] >= 2 for row in ranked_sorted[:5]):
            hit5 += 1
    return {
        "samples": samples,
        "avg_candidates_per_sample": round(avg_candidates / samples, 3),
        "avg_positives_per_sample": round(avg_positives / samples, 3),
        "oracle_fail_no_positive": no_positive,
        "oracle_coverage": round((samples - no_positive) / samples, 4),
        "hit_at_1": round(hit1 / samples, 4),
        "hit_at_3": round(hit3 / samples, 4),
        "hit_at_5": round(hit5 / samples, 4),
    }


def generate_report(
    dataset_path: str,
    scorer: str,
    artifact_path: str | None,
    exclude_gold_seed: bool,
    task_filter: str,
    top_n: int,
) -> dict[str, Any]:
    grouped_rows = load_jsonl(dataset_path)
    flat_rows, by_id = _flatten_rows(
        grouped_rows,
        exclude_gold_seed=exclude_gold_seed,
        task_filter=task_filter,
    )
    if not flat_rows:
        raise ValueError("No candidate rows available after filters.")

    ranked_by_sample = _rank_map(flat_rows, scorer=scorer, artifact_path=artifact_path)

    by_task_samples: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    pattern_failures: dict[str, Counter[str]] = defaultdict(Counter)
    pattern_oracle_failures: dict[str, Counter[str]] = defaultdict(Counter)

    for sample_id, ranked in ranked_by_sample.items():
        if not ranked:
            continue
        task = str(ranked[0]["task"]).strip().lower()
        by_task_samples[task].append(ranked)

        ranked_sorted = sorted(ranked, key=lambda x: x["score"], reverse=True)
        positives = [row for row in ranked_sorted if row["label"] >= 2]
        row_meta = by_id.get(sample_id, {})
        pattern = _extract_pattern(task, row_meta)
        if not positives:
            pattern_oracle_failures[task][pattern] += 1
        elif ranked_sorted[0]["label"] < 2:
            pattern_failures[task][pattern] += 1

    task_reports: dict[str, Any] = {}
    for task in sorted(by_task_samples.keys()):
        metrics = _task_metrics(by_task_samples[task])
        top_rank_fail = [
            {"pattern": key, "count": count}
            for key, count in pattern_failures[task].most_common(top_n)
        ]
        top_oracle_fail = [
            {"pattern": key, "count": count}
            for key, count in pattern_oracle_failures[task].most_common(top_n)
        ]
        task_reports[task] = {
            **metrics,
            "top_ranking_fail_patterns": top_rank_fail,
            "top_oracle_fail_patterns": top_oracle_fail,
        }

    return {
        "dataset": dataset_path,
        "scorer": scorer,
        "artifact": artifact_path if scorer == "reranker" else None,
        "exclude_gold_seed": exclude_gold_seed,
        "task_filter": task_filter,
        "total_rows": len(grouped_rows),
        "total_candidate_rows": len(flat_rows),
        "per_task": task_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-task failure report with top failure patterns (oracle fail vs ranking fail)."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Grouped JSONL dataset path.")
    parser.add_argument(
        "--scorer",
        choices=["reranker", "baseline"],
        default="reranker",
        help="Use reranker artifact scores or baseline model_score.",
    )
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Reranker artifact path.")
    parser.add_argument(
        "--exclude-gold-seed",
        action="store_true",
        help="Exclude source=gold_seed candidates.",
    )
    parser.add_argument(
        "--task-filter",
        choices=["non_rewrite", "rewrite", "all"],
        default="non_rewrite",
        help="Task slice for report.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Top failure patterns to show per task bucket.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = generate_report(
        dataset_path=args.dataset,
        scorer=args.scorer,
        artifact_path=args.artifact,
        exclude_gold_seed=args.exclude_gold_seed,
        task_filter=args.task_filter,
        top_n=max(3, min(20, args.top_n)),
    )
    print(json.dumps(report, indent=2))
