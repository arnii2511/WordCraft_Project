from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pymongo import MongoClient

from backend.config import MONGODB_DB, MONGODB_URI

from .common import build_input_text, load_jsonl, write_jsonl

DEFAULT_OUTPUT = "backend/ml/data/dataset_feedback.jsonl"
DEFAULT_APPEND_TO = "backend/ml/data/dataset_ranker.jsonl"


def _mapped_task(task: str, input_payload: dict[str, Any]) -> str:
    if task == "editor_suggestion":
        if input_payload.get("selection"):
            return "suggest_selection"
        sentence = (input_payload.get("sentence") or "").strip()
        if "[BLANK]" in sentence or "____" in sentence:
            return "suggest_blank"
        return "suggest_sentence"
    if task == "editor_rewrite":
        return "rewrite"
    return task


def _label_from_avg(avg_rating: float) -> int:
    if avg_rating < 3.0:
        return 0
    if avg_rating == 3.0:
        return 1
    if avg_rating < 5.0:
        return 2
    return 3


def _safe_iso(dt: Any) -> str:
    if isinstance(dt, datetime):
        return dt.isoformat()
    return datetime.utcnow().isoformat()


def _build_feedback_rows(raw_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in raw_docs:
        key = doc.get("input_key")
        if not key:
            continue
        grouped[str(key)].append(doc)

    rows: list[dict[str, Any]] = []
    for key, docs in grouped.items():
        first = docs[0]
        input_payload = first.get("input_payload") or {}
        mapped_task = _mapped_task(first.get("task", "suggest_sentence"), input_payload)
        context = first.get("context")
        mode = first.get("mode")

        candidate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for doc in docs:
            candidate = (doc.get("candidate") or "").strip()
            if not candidate:
                continue
            candidate_groups[candidate.lower()].append(doc)

        candidates: list[dict[str, Any]] = []
        for group_docs in candidate_groups.values():
            base = group_docs[0]
            ratings = [int(item.get("rating", 0)) for item in group_docs if item.get("rating")]
            if not ratings:
                continue
            avg_rating = round(sum(ratings) / len(ratings), 2)
            label = _label_from_avg(avg_rating)
            source_counter = Counter(str(item.get("source", "")).strip().lower() for item in group_docs)
            dominant_source = source_counter.most_common(1)[0][0] if source_counter else "user_feedback"
            if not dominant_source:
                dominant_source = "user_feedback"
            candidates.append(
                {
                    "text": base.get("candidate"),
                    "label": label,
                    "model_score": round(float(base.get("model_score") or 0.0), 4),
                    "pos": base.get("pos"),
                    "reason": base.get("reason") or "User-rated candidate.",
                    "source": dominant_source,
                    "rating_avg": avg_rating,
                    "rating_count": len(ratings),
                }
            )

        if not candidates:
            continue

        candidates.sort(key=lambda item: (item["label"], item["rating_avg"], item["rating_count"]), reverse=True)
        created_values = [_safe_iso(doc.get("created_at")) for doc in docs]
        created_at = min(created_values) if created_values else datetime.utcnow().isoformat()

        input_data = {**input_payload}
        if context and "context" not in input_data:
            input_data["context"] = context
        if mode and "mode" not in input_data:
            input_data["mode"] = mode

        row = {
            "id": f"fb_{key[:16]}",
            "task": mapped_task,
            "input": input_data,
            "input_text": build_input_text(mapped_task, input_data),
            "candidates": candidates,
            "stats": {
                "candidate_count": len(candidates),
                "positive_count": sum(1 for item in candidates if item["label"] >= 2),
                "feedback_events": len(docs),
            },
            "note": f"Aggregated from {len(docs)} user rating events.",
            "created_at": created_at,
        }
        rows.append(row)

    rows.sort(key=lambda item: item["id"])
    return rows


def _merge_rows(base_rows: list[dict[str, Any]], feedback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        row_id = row.get("id")
        if row_id:
            merged[str(row_id)] = row
    for row in feedback_rows:
        row_id = row.get("id")
        if row_id:
            merged[str(row_id)] = row
    return [merged[key] for key in sorted(merged.keys())]


def export_feedback_dataset(output_path: str, append_to: str | None) -> None:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
    database = client[MONGODB_DB]
    total_feedback_docs = database.feedback_ratings.count_documents({})
    print(f"Mongo DB: {MONGODB_DB}")
    print(f"Feedback docs in collection: {total_feedback_docs}")
    docs = list(
        database.feedback_ratings.find(
            {},
            {
                "input_key": 1,
                "task": 1,
                "candidate": 1,
                "rating": 1,
                "context": 1,
                "mode": 1,
                "input_payload": 1,
                "model_score": 1,
                "reason": 1,
                "pos": 1,
                "created_at": 1,
            },
        )
    )

    feedback_rows = _build_feedback_rows(docs)
    write_jsonl(output_path, feedback_rows)
    print(f"Feedback dataset rows: {len(feedback_rows)}")
    print(f"Wrote: {Path(output_path).as_posix()}")

    if append_to:
        base_path = Path(append_to)
        base_rows = load_jsonl(base_path) if base_path.exists() else []
        merged_rows = _merge_rows(base_rows, feedback_rows)
        write_jsonl(base_path, merged_rows)
        print(f"Merged dataset rows: {len(merged_rows)}")
        print(f"Updated: {base_path.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Mongo feedback ratings into ranker dataset rows.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output feedback dataset JSONL path.")
    parser.add_argument(
        "--append-to",
        default=None,
        help="Optional target JSONL file to merge feedback rows into (e.g. dataset_ranker.jsonl).",
    )
    parser.add_argument(
        "--append-default",
        action="store_true",
        help="Merge into default base dataset file backend/ml/data/dataset_ranker.jsonl.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    append_to = DEFAULT_APPEND_TO if args.append_default else args.append_to
    export_feedback_dataset(output_path=args.output, append_to=append_to)
