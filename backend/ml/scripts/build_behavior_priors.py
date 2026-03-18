from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pymongo import MongoClient

from backend.config import MONGODB_DB, MONGODB_URI

DEFAULT_OUTPUT = "backend/ml/models/behavior_priors.json"

_IMPLICIT_SCORE = {
    "implicit_favorite": 0.88,
    "implicit_insert": 0.92,
    "implicit_copy": 0.78,
}


def _normalize_rating(rating: int) -> float:
    clipped = max(1, min(5, int(rating)))
    return (clipped - 1.0) / 4.0


def _safe_candidate(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_task(value: Any) -> str:
    return str(value or "").strip().lower()


def _score_bucket(rows: list[dict[str, Any]], smoothing_alpha: float) -> dict[str, Any]:
    ratings = [_normalize_rating(int(item.get("rating", 0))) for item in rows if item.get("rating") is not None]
    implicit = [
        _IMPLICIT_SCORE.get(str(item.get("source", "")).strip().lower())
        for item in rows
        if str(item.get("source", "")).strip().lower() in _IMPLICIT_SCORE
    ]
    implicit = [value for value in implicit if value is not None]

    rating_sum = float(sum(ratings))
    implicit_sum = float(sum(implicit))
    rating_count = len(ratings)
    implicit_count = len(implicit)
    total = rating_count + implicit_count

    if total == 0:
        return {"score": 0.5, "count": 0, "rating_count": 0, "implicit_count": 0}

    # Bayesian smoothing keeps low-data candidates from extreme scores.
    raw = (rating_sum + implicit_sum + (0.5 * smoothing_alpha)) / (total + smoothing_alpha)
    score = max(0.0, min(1.0, raw))
    return {
        "score": round(score, 4),
        "count": total,
        "rating_count": rating_count,
        "implicit_count": implicit_count,
    }


def build_behavior_priors(output_path: str, smoothing_alpha: float = 6.0) -> None:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
    database = client[MONGODB_DB]

    docs = list(
        database.feedback_ratings.find(
            {},
            {
                "task": 1,
                "candidate": 1,
                "rating": 1,
                "source": 1,
            },
        )
    )

    by_task_candidate: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    by_candidate_global: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for doc in docs:
        task = _safe_task(doc.get("task"))
        candidate = _safe_candidate(doc.get("candidate"))
        if not task or not candidate:
            continue
        by_task_candidate[task][candidate].append(doc)
        by_candidate_global[candidate].append(doc)

    task_candidate_scores: dict[str, dict[str, dict[str, Any]]] = {}
    for task, candidate_map in by_task_candidate.items():
        task_candidate_scores[task] = {}
        for candidate, rows in candidate_map.items():
            task_candidate_scores[task][candidate] = _score_bucket(rows, smoothing_alpha=smoothing_alpha)

    global_scores: dict[str, dict[str, Any]] = {}
    for candidate, rows in by_candidate_global.items():
        global_scores[candidate] = _score_bucket(rows, smoothing_alpha=smoothing_alpha)

    output = {
        "meta": {
            "mongodb_db": MONGODB_DB,
            "events": len(docs),
            "tasks": len(task_candidate_scores),
            "candidates": len(global_scores),
            "smoothing_alpha": smoothing_alpha,
        },
        "task_candidate": task_candidate_scores,
        "global_candidate": global_scores,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote behavior priors: {out_path.as_posix()}")
    print(json.dumps(output["meta"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build behavior priors from feedback events.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=6.0,
        help="Bayesian smoothing strength for sparse candidate events.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_behavior_priors(
        output_path=args.output,
        smoothing_alpha=max(0.1, float(args.smoothing_alpha)),
    )
