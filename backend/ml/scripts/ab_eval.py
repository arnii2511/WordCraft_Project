from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import RankingMetrics, compute_ranking_metrics, load_jsonl
from .eval_reranker import _feature_text, _prob_to_score

DEFAULT_DATASET = "backend/ml/data/splits/test.jsonl"
DEFAULT_ARTIFACT = "backend/ml/models/reranker.pkl"


def _flatten_rows(rows: list[dict[str, Any]], exclude_gold_seed: bool, task_filter: str) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        task = str(row.get("task", "")).strip().lower()
        if task_filter == "non_rewrite" and task == "rewrite":
            continue
        if task_filter == "rewrite" and task != "rewrite":
            continue
        sample_id = row.get("id")
        if not sample_id:
            continue
        for candidate in row.get("candidates", []):
            if exclude_gold_seed and str(candidate.get("source", "")).strip().lower() == "gold_seed":
                continue
            text = (candidate.get("text") or "").strip()
            if not text:
                continue
            flat.append(
                {
                    "sample_id": str(sample_id),
                    "feature_text": _feature_text(row, candidate),
                    "label": int(candidate.get("label", 0)),
                    "baseline_score": float(candidate.get("model_score", 0.0) or 0.0),
                }
            )
    return flat


def _group_from_scores(flat_rows: list[dict[str, Any]], scores: np.ndarray | list[float]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(flat_rows, scores):
        grouped[row["sample_id"]].append(
            {
                "label": int(row["label"]),
                "pred_score": float(score),
            }
        )
    return grouped


def _as_dict(metrics: RankingMetrics) -> dict[str, float | int]:
    return {
        "precision_at_1": metrics.precision_at_1,
        "precision_at_3": metrics.precision_at_3,
        "precision_at_5": metrics.precision_at_5,
        "hit_at_1": metrics.hit_at_1,
        "hit_at_3": metrics.hit_at_3,
        "hit_at_5": metrics.hit_at_5,
        "ndcg_at_5": metrics.ndcg_at_5,
        "mrr": metrics.mrr,
        "samples": metrics.samples,
    }


def ab_eval(dataset_path: str, artifact_path: str, exclude_gold_seed: bool, task_filter: str) -> None:
    rows = load_jsonl(dataset_path)
    flat_rows = _flatten_rows(rows, exclude_gold_seed=exclude_gold_seed, task_filter=task_filter)
    if not flat_rows:
        raise ValueError("No candidate rows available for A/B evaluation.")

    artifact = pickle.loads(Path(artifact_path).read_bytes())
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    x = [row["feature_text"] for row in flat_rows]
    x_vec = vectorizer.transform(x)
    probabilities = model.predict_proba(x_vec)
    reranker_scores = _prob_to_score(probabilities, model.classes_)
    baseline_scores = [row["baseline_score"] for row in flat_rows]

    grouped_baseline = _group_from_scores(flat_rows, baseline_scores)
    grouped_reranker = _group_from_scores(flat_rows, reranker_scores)
    baseline_metrics = compute_ranking_metrics(grouped_baseline)
    reranker_metrics = compute_ranking_metrics(grouped_reranker)

    keys = [
        "precision_at_1",
        "precision_at_3",
        "precision_at_5",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "ndcg_at_5",
        "mrr",
    ]
    base_dict = _as_dict(baseline_metrics)
    rerank_dict = _as_dict(reranker_metrics)
    deltas = {key: round(float(rerank_dict[key]) - float(base_dict[key]), 4) for key in keys}

    payload = {
        "dataset": dataset_path,
        "artifact": artifact_path,
        "rows": len(flat_rows),
        "exclude_gold_seed": bool(exclude_gold_seed),
        "task_filter": task_filter,
        "baseline": base_dict,
        "reranker": rerank_dict,
        "delta_reranker_minus_baseline": deltas,
    }
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B evaluation: baseline model_score vs trained reranker.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Grouped dataset (typically unseen test split).")
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Reranker model artifact path.")
    parser.add_argument(
        "--exclude-gold-seed",
        action="store_true",
        help="Exclude candidates with source=gold_seed from A/B evaluation.",
    )
    parser.add_argument(
        "--task-filter",
        choices=["non_rewrite", "rewrite", "all"],
        default="non_rewrite",
        help="non_rewrite excludes rewrite task from default A/B ranker evaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ab_eval(
        dataset_path=args.dataset,
        artifact_path=args.artifact,
        exclude_gold_seed=args.exclude_gold_seed,
        task_filter=args.task_filter,
    )
