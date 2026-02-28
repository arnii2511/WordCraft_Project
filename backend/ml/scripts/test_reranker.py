from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .common import compute_ranking_metrics, load_jsonl
from .eval_reranker import _flatten_rows, _prob_to_score

DEFAULT_TEST_SPLIT = "backend/ml/data/splits/test.jsonl"
DEFAULT_ARTIFACT = "backend/ml/models/reranker.pkl"


def test_model(
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
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    x = [row["feature_text"] for row in flat_rows]
    y = np.array([row["label"] for row in flat_rows], dtype=np.int64)
    x_vec = vectorizer.transform(x)
    y_pred = model.predict(x_vec)
    probabilities = model.predict_proba(x_vec)
    pred_scores = _prob_to_score(probabilities, model.classes_)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(flat_rows, pred_scores):
        grouped[row["sample_id"]].append({"label": row["label"], "pred_score": float(score)})

    ranking = compute_ranking_metrics(grouped)
    report = {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "macro_f1": round(float(f1_score(y, y_pred, average="macro")), 4),
        "hit_at_1": ranking.hit_at_1,
        "hit_at_3": ranking.hit_at_3,
        "hit_at_5": ranking.hit_at_5,
        "ndcg_at_5": ranking.ndcg_at_5,
        "mrr": ranking.mrr,
        "samples": ranking.samples,
        "rows": len(flat_rows),
        "exclude_gold_seed": bool(exclude_gold_seed),
        "task_filter": task_filter,
    }
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test reranker artifact on frozen test split.")
    parser.add_argument("--dataset", default=DEFAULT_TEST_SPLIT, help="Grouped test split JSONL path.")
    parser.add_argument(
        "--exclude-gold-seed",
        action="store_true",
        help="Exclude candidates with source=gold_seed from test rows.",
    )
    parser.add_argument(
        "--task-filter",
        choices=["non_rewrite", "rewrite", "all"],
        default="non_rewrite",
        help="non_rewrite excludes rewrite task from default ranker test (default).",
    )
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Model artifact path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_model(
        dataset_path=args.dataset,
        artifact_path=args.artifact,
        exclude_gold_seed=args.exclude_gold_seed,
        task_filter=args.task_filter,
    )
