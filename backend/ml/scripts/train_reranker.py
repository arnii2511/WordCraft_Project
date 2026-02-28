from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .common import RankingMetrics, compute_ranking_metrics, load_jsonl

DEFAULT_DATASET = "backend/ml/data/dataset_ranker.jsonl"
DEFAULT_ARTIFACT = "backend/ml/models/reranker.pkl"
DEFAULT_TRAIN_SPLIT = "backend/ml/data/splits/train.jsonl"
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


def _flatten_rows(rows: list[dict[str, Any]], exclude_gold_seed: bool = False) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        task = str(row.get("task", "")).strip().lower()
        if task == "rewrite":
            continue
        sample_id = row.get("id")
        if not sample_id:
            continue
        candidates = row.get("candidates", [])
        for candidate in candidates:
            if exclude_gold_seed and str(candidate.get("source", "")).strip().lower() == "gold_seed":
                continue
            text = (candidate.get("text") or "").strip()
            if not text:
                continue
            label = int(candidate.get("label", 0))
            flat.append(
                {
                    "sample_id": sample_id,
                    "feature_text": _feature_text(row, candidate),
                    "label": label,
                    "source": str(candidate.get("source", "")).strip().lower(),
                }
            )
    return flat


def _sample_weight(row: dict[str, Any]) -> float:
    label = int(row.get("label", 0))
    source = str(row.get("source", "")).strip().lower()
    weight = 1.0
    if label >= 3:
        weight += 1.0
    elif label == 2:
        weight += 0.6
    elif label == 1:
        weight += 0.2
    if source in {"user_feedback", "implicit_insert", "implicit_copy", "implicit_favorite"}:
        weight += 0.8
    elif source in {"gold_seed", "seed"}:
        weight += 0.3
    return float(weight)


def _has_positive(rows: list[dict[str, Any]]) -> bool:
    return any(int(row["label"]) >= 2 for row in rows)


def _split_by_sample(flat_rows: list[dict[str, Any]], test_size: float, random_state: int) -> tuple[list, list]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flat_rows:
        grouped[row["sample_id"]].append(row)

    sample_ids = sorted(grouped.keys())
    valid_ids = [sample_id for sample_id in sample_ids if _has_positive(grouped[sample_id])]
    if len(valid_ids) < 4:
        raise ValueError("Need at least 4 labeled samples with positives to train/evaluate reranker.")

    train_ids, test_ids = train_test_split(
        valid_ids,
        test_size=test_size,
        random_state=random_state,
    )

    train_rows = [row for sample_id in train_ids for row in grouped[sample_id]]
    test_rows = [row for sample_id in test_ids for row in grouped[sample_id]]
    return train_rows, test_rows


def _flatten_grouped(rows: list[dict[str, Any]], exclude_gold_seed: bool = False) -> list[dict[str, Any]]:
    flat = _flatten_rows(rows, exclude_gold_seed=exclude_gold_seed)
    if not flat:
        raise ValueError("Provided split has no candidate rows.")
    return flat


def _prob_to_score(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_values = classes.astype(float)
    return probabilities @ class_values


def _group_for_ranking(rows: list[dict[str, Any]], pred_scores: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(rows, pred_scores):
        grouped[row["sample_id"]].append(
            {
                "label": int(row["label"]),
                "pred_score": float(score),
            }
        )
    return grouped


def train(
    artifact_path: str,
    test_size: float,
    random_state: int,
    dataset_path: str | None = None,
    train_path: str | None = None,
    val_path: str | None = None,
    exclude_gold_seed: bool = False,
) -> None:
    if train_path and val_path:
        train_rows = _flatten_grouped(load_jsonl(train_path), exclude_gold_seed=exclude_gold_seed)
        test_rows = _flatten_grouped(load_jsonl(val_path), exclude_gold_seed=exclude_gold_seed)
        source_info = {"train_path": train_path, "val_path": val_path}
    else:
        if not dataset_path:
            raise ValueError("dataset_path is required when train/val split paths are not provided.")
        dataset_rows = load_jsonl(dataset_path)
        flat_rows = _flatten_rows(dataset_rows, exclude_gold_seed=exclude_gold_seed)
        if not flat_rows:
            raise ValueError("Dataset has no candidate rows. Build dataset first.")
        train_rows, test_rows = _split_by_sample(flat_rows, test_size=test_size, random_state=random_state)
        source_info = {"dataset_path": dataset_path, "split_strategy": "grouped_random"}

    x_train = [row["feature_text"] for row in train_rows]
    y_train = np.array([int(row["label"]) for row in train_rows], dtype=np.int64)
    w_train = np.array([_sample_weight(row) for row in train_rows], dtype=np.float64)
    x_test = [row["feature_text"] for row in test_rows]
    y_test = np.array([int(row["label"]) for row in test_rows], dtype=np.int64)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=120000, min_df=1)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
    )
    model.fit(x_train_vec, y_train, sample_weight=w_train)

    y_pred = model.predict(x_test_vec)
    test_accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    probabilities = model.predict_proba(x_test_vec)
    pred_scores = _prob_to_score(probabilities, model.classes_)
    grouped_for_ranking = _group_for_ranking(test_rows, pred_scores)
    ranking_metrics: RankingMetrics = compute_ranking_metrics(grouped_for_ranking)

    artifact = {
        "vectorizer": vectorizer,
        "model": model,
        "metadata": {
            **source_info,
            "exclude_gold_seed": exclude_gold_seed,
            "train_examples": len(train_rows),
            "test_examples": len(test_rows),
            "metrics": {
                "accuracy": round(test_accuracy, 4),
                "macro_f1": round(macro_f1, 4),
                "precision_at_1": ranking_metrics.precision_at_1,
                "precision_at_3": ranking_metrics.precision_at_3,
                "precision_at_5": ranking_metrics.precision_at_5,
                "hit_at_1": ranking_metrics.hit_at_1,
                "hit_at_3": ranking_metrics.hit_at_3,
                "hit_at_5": ranking_metrics.hit_at_5,
                "ndcg_at_5": ranking_metrics.ndcg_at_5,
                "mrr": ranking_metrics.mrr,
                "ranking_samples": ranking_metrics.samples,
            },
        },
    }

    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    print(json.dumps(artifact["metadata"]["metrics"], indent=2))
    print(f"Saved artifact: {output_path.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WordCraft NLP reranker.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Ranker dataset JSONL path.")
    parser.add_argument("--train", default=None, help="Optional grouped train split JSONL path.")
    parser.add_argument("--val", default=None, help="Optional grouped validation split JSONL path.")
    parser.add_argument(
        "--use-default-splits",
        action="store_true",
        help="Use backend/ml/data/splits/train.jsonl and val.jsonl if present.",
    )
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Output model artifact path.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split ratio by sample.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--exclude-gold-seed",
        action="store_true",
        help="Exclude candidates with source=gold_seed from training/evaluation folds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_path = args.train
    val_path = args.val
    if args.use_default_splits and train_path is None and val_path is None:
        train_path = DEFAULT_TRAIN_SPLIT
        val_path = DEFAULT_VAL_SPLIT
    train(
        artifact_path=args.artifact,
        test_size=min(max(args.test_size, 0.1), 0.45),
        random_state=args.random_state,
        dataset_path=args.dataset,
        train_path=train_path,
        val_path=val_path,
        exclude_gold_seed=args.exclude_gold_seed,
    )
