from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import load_jsonl
from .eval_reranker import _feature_text, _prob_to_score

DEFAULT_DATASET = "backend/ml/data/splits/test.jsonl"
DEFAULT_ARTIFACT = "backend/ml/models/reranker.pkl"


def _gain(label: int) -> float:
    return float((2 ** max(0, int(label))) - 1)


def _dcg(labels: list[int]) -> float:
    value = 0.0
    for idx, label in enumerate(labels, start=1):
        value += _gain(label) / math.log2(idx + 1)
    return value


def _flatten_rows(rows: list[dict[str, Any]], exclude_gold_seed: bool) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        task = str(row.get("task", "")).strip().lower()
        if not sample_id or not task:
            continue
        for candidate in row.get("candidates", []):
            if exclude_gold_seed and str(candidate.get("source", "")).strip().lower() == "gold_seed":
                continue
            text = str(candidate.get("text", "")).strip()
            if not text:
                continue
            flat.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "feature_text": _feature_text(row, candidate),
                    "label": int(candidate.get("label", 0)),
                    "baseline_score": float(candidate.get("model_score", 0.0) or 0.0),
                    "rhyme": bool(candidate.get("rhyme", False)),
                    "relation_match": bool(candidate.get("relation_match", False)),
                }
            )
    return flat


def _predict_scores(flat_rows: list[dict[str, Any]], artifact_path: str | None, scorer: str) -> np.ndarray:
    if scorer == "baseline":
        return np.asarray([row["baseline_score"] for row in flat_rows], dtype=np.float32)
    if not artifact_path:
        raise ValueError("artifact path is required for scorer=reranker")

    artifact = pickle.loads(Path(artifact_path).read_bytes())
    artifact_type = artifact.get("artifact_type", "sklearn_tfidf_logreg")

    if artifact_type == "sklearn_tfidf_logreg":
        vectorizer = artifact["vectorizer"]
        model = artifact["model"]
        x_vec = vectorizer.transform([row["feature_text"] for row in flat_rows])
        probabilities = model.predict_proba(x_vec)
        return _prob_to_score(probabilities, model.classes_).astype(np.float32)

    if artifact_type.startswith("cross_encoder"):
        from sentence_transformers.cross_encoder import CrossEncoder

        model = CrossEncoder(artifact["model_path"])
        values = model.predict([row["feature_text"] for row in flat_rows])
        scores = np.asarray(values, dtype=np.float32)
        if scores.ndim == 2:
            scores = scores.reshape(-1)
        return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)

    raise ValueError(f"Unsupported artifact_type: {artifact_type}")


def _group_rows(flat_rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(flat_rows, scores):
        grouped[row["sample_id"]].append(
            {
                "task": row["task"],
                "label": int(row["label"]),
                "pred_score": float(score),
                "rhyme": bool(row.get("rhyme", False)),
                "relation_match": bool(row.get("relation_match", False)),
            }
        )
    return grouped


def _init_metric_bucket() -> dict[str, float]:
    return {
        "samples": 0.0,
        "hit@1": 0.0,
        "hit@3": 0.0,
        "hit@5": 0.0,
        "precision@1": 0.0,
        "precision@3": 0.0,
        "precision@5": 0.0,
        "ndcg@1": 0.0,
        "ndcg@3": 0.0,
        "ndcg@5": 0.0,
        "mrr": 0.0,
        "constraint_pass@1": 0.0,
        "constraint_pass@3": 0.0,
    }


def _finalize_metric_bucket(bucket: dict[str, float]) -> dict[str, float]:
    samples = max(1.0, bucket["samples"])
    output = {}
    for key, value in bucket.items():
        if key == "samples":
            output[key] = int(value)
        else:
            output[key] = round(value / samples, 4)
    return output


def _evaluate_per_task(grouped_rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    by_task: dict[str, dict[str, float]] = defaultdict(_init_metric_bucket)

    for rows in grouped_rows.values():
        ranked = sorted(rows, key=lambda item: item["pred_score"], reverse=True)
        if not ranked:
            continue
        task = str(ranked[0]["task"]).strip().lower()
        bucket = by_task[task]
        bucket["samples"] += 1.0

        top1 = ranked[:1]
        top3 = ranked[:3]
        top5 = ranked[:5]

        rel1 = sum(1 for row in top1 if row["label"] >= 2)
        rel3 = sum(1 for row in top3 if row["label"] >= 2)
        rel5 = sum(1 for row in top5 if row["label"] >= 2)

        bucket["precision@1"] += rel1 / 1.0
        bucket["precision@3"] += rel3 / 3.0
        bucket["precision@5"] += rel5 / 5.0
        bucket["hit@1"] += 1.0 if rel1 > 0 else 0.0
        bucket["hit@3"] += 1.0 if rel3 > 0 else 0.0
        bucket["hit@5"] += 1.0 if rel5 > 0 else 0.0

        labels1 = [int(row["label"]) for row in top1]
        labels3 = [int(row["label"]) for row in top3]
        labels5 = [int(row["label"]) for row in top5]
        ideal = sorted((int(row["label"]) for row in ranked), reverse=True)
        ideal1 = ideal[:1]
        ideal3 = ideal[:3]
        ideal5 = ideal[:5]

        idcg1 = _dcg(ideal1)
        idcg3 = _dcg(ideal3)
        idcg5 = _dcg(ideal5)
        if idcg1 > 0:
            bucket["ndcg@1"] += _dcg(labels1) / idcg1
        if idcg3 > 0:
            bucket["ndcg@3"] += _dcg(labels3) / idcg3
        if idcg5 > 0:
            bucket["ndcg@5"] += _dcg(labels5) / idcg5

        reciprocal = 0.0
        for idx, row in enumerate(ranked, start=1):
            if row["label"] >= 2:
                reciprocal = 1.0 / idx
                break
        bucket["mrr"] += reciprocal

        if task == "constraints":
            pass1 = any(item["rhyme"] and item["relation_match"] for item in top1)
            pass3 = any(item["rhyme"] and item["relation_match"] for item in top3)
            bucket["constraint_pass@1"] += 1.0 if pass1 else 0.0
            bucket["constraint_pass@3"] += 1.0 if pass3 else 0.0

    result: dict[str, dict[str, float]] = {}
    for task, bucket in sorted(by_task.items()):
        result[task] = _finalize_metric_bucket(bucket)
    return result


def build_dashboard(
    dataset_path: str,
    artifact_path: str | None,
    scorer: str,
    exclude_gold_seed: bool,
) -> dict[str, Any]:
    rows = load_jsonl(dataset_path)
    flat_rows = _flatten_rows(rows, exclude_gold_seed=exclude_gold_seed)
    if not flat_rows:
        raise ValueError("No candidate rows found in dataset.")

    scores = _predict_scores(flat_rows, artifact_path=artifact_path, scorer=scorer)
    grouped = _group_rows(flat_rows, scores)
    per_task = _evaluate_per_task(grouped)

    return {
        "dataset": dataset_path,
        "scorer": scorer,
        "artifact": artifact_path if scorer == "reranker" else None,
        "rows": len(rows),
        "candidate_rows": len(flat_rows),
        "exclude_gold_seed": bool(exclude_gold_seed),
        "per_task_metrics": per_task,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-level ranking metrics dashboard.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Grouped JSONL dataset path.")
    parser.add_argument(
        "--scorer",
        choices=["reranker", "baseline"],
        default="reranker",
        help="Evaluate reranker artifact or baseline model_score ranking.",
    )
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Model artifact path.")
    parser.add_argument(
        "--exclude-gold-seed",
        action="store_true",
        help="Exclude source=gold_seed candidate rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    artifact = None if args.scorer == "baseline" else args.artifact
    report = build_dashboard(
        dataset_path=args.dataset,
        artifact_path=artifact,
        scorer=args.scorer,
        exclude_gold_seed=args.exclude_gold_seed,
    )
    print(json.dumps(report, indent=2))
