from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .common import RankingMetrics, compute_ranking_metrics, load_jsonl

DEFAULT_DATASET = "backend/ml/data/dataset_ranker.jsonl"
DEFAULT_ARTIFACT = "backend/ml/models/reranker.pkl"
DEFAULT_VAL_SPLIT = "backend/ml/data/splits/val.jsonl"


def _feature_text(row: dict[str, Any], candidate: dict[str, Any]) -> str:
    payload = row.get("input", {})
    task = row.get("task", "")
    mode = payload.get("mode", "")
    context = payload.get("context", "")
    input_text = row.get("input_text", "")
    candidate_text = candidate.get("text", "")
    pos = candidate.get("pos", "") or ""
    reason = candidate.get("reason", "") or ""
    source = candidate.get("source", "") or ""
    return (
        f"task={task} mode={mode} context={context} "
        f"input={input_text} candidate={candidate_text} pos={pos} source={source} reason={reason}"
    )


def _flatten_rows(
    rows: list[dict[str, Any]],
    exclude_gold_seed: bool = False,
    task_filter: str = "non_rewrite",
) -> list[dict[str, Any]]:
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
                    "sample_id": sample_id,
                    "feature_text": _feature_text(row, candidate),
                    "label": int(candidate.get("label", 0)),
                }
            )
    return flat


def _prob_to_score(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_values = classes.astype(float)
    return probabilities @ class_values


def _predict_scores(artifact: dict[str, Any], feature_rows: list[str]) -> tuple[np.ndarray, np.ndarray]:
    artifact_type = artifact.get("artifact_type", "sklearn_tfidf_logreg")

    if artifact_type == "sklearn_tfidf_logreg":
        vectorizer = artifact["vectorizer"]
        model = artifact["model"]
        x_vec = vectorizer.transform(feature_rows)
        y_pred = model.predict(x_vec)
        probabilities = model.predict_proba(x_vec)
        pred_scores = _prob_to_score(probabilities, model.classes_)
        return y_pred.astype(np.int64), pred_scores.astype(np.float32)

    if artifact_type.startswith("cross_encoder"):
        from sentence_transformers.cross_encoder import CrossEncoder

        model = CrossEncoder(artifact["model_path"])
        values = np.asarray(model.predict(feature_rows), dtype=np.float32)
        if values.ndim == 2:
            values = values.reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-values))
        # Map to coarse relevance labels for compatibility with classification metrics.
        y_pred = np.where(probs >= 0.75, 3, np.where(probs >= 0.5, 2, np.where(probs >= 0.3, 1, 0)))
        return y_pred.astype(np.int64), probs.astype(np.float32)

    raise ValueError(f"Unsupported artifact_type: {artifact_type}")


def evaluate(
    dataset_path: str,
    artifact_path: str,
    exclude_gold_seed: bool = False,
    task_filter: str = "non_rewrite",
) -> None:
    dataset_rows = load_jsonl(dataset_path)
    flat_rows = _flatten_rows(
        dataset_rows,
        exclude_gold_seed=exclude_gold_seed,
        task_filter=task_filter,
    )
    if not flat_rows:
        raise ValueError("Dataset has no labeled candidate rows.")

    artifact = pickle.loads(Path(artifact_path).read_bytes())
    x = [row["feature_text"] for row in flat_rows]
    y = np.array([row["label"] for row in flat_rows], dtype=np.int64)
    y_pred, pred_scores = _predict_scores(artifact, x)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(flat_rows, pred_scores):
        grouped[row["sample_id"]].append({"label": row["label"], "pred_score": float(score)})

    ranking_metrics: RankingMetrics = compute_ranking_metrics(grouped)
    metrics = {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "macro_f1": round(float(f1_score(y, y_pred, average="macro")), 4),
        "precision_at_1": ranking_metrics.precision_at_1,
        "precision_at_3": ranking_metrics.precision_at_3,
        "precision_at_5": ranking_metrics.precision_at_5,
        "hit_at_1": ranking_metrics.hit_at_1,
        "hit_at_3": ranking_metrics.hit_at_3,
        "hit_at_5": ranking_metrics.hit_at_5,
        "ndcg_at_5": ranking_metrics.ndcg_at_5,
        "mrr": ranking_metrics.mrr,
        "ranking_samples": ranking_metrics.samples,
        "rows": len(flat_rows),
        "exclude_gold_seed": bool(exclude_gold_seed),
        "task_filter": task_filter,
        "artifact_type": artifact.get("artifact_type", "sklearn_tfidf_logreg"),
    }
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WordCraft reranker artifact.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset JSONL path.")
    parser.add_argument(
        "--use-default-val",
        action="store_true",
        help="Evaluate on backend/ml/data/splits/val.jsonl.",
    )
    parser.add_argument(
        "--exclude-gold-seed",
        action="store_true",
        help="Exclude candidates with source=gold_seed from evaluation rows.",
    )
    parser.add_argument(
        "--task-filter",
        choices=["non_rewrite", "rewrite", "all"],
        default="non_rewrite",
        help="non_rewrite excludes rewrite task from ranker evaluation (default).",
    )
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Model artifact path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = DEFAULT_VAL_SPLIT if args.use_default_val else args.dataset
    evaluate(
        dataset_path=dataset_path,
        artifact_path=args.artifact,
        exclude_gold_seed=args.exclude_gold_seed,
        task_filter=args.task_filter,
    )
