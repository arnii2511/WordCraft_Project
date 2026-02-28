from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

from .common import load_jsonl, write_jsonl

DEFAULT_SEED = "backend/ml/data/seed_gold.jsonl"
DEFAULT_CSV_PATTERN = "C:/Users/*/Downloads/wordcraft_dataset_12k.csv"


def _parse_json_field(raw: str, fallback: Any) -> Any:
    text = (raw or "").strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        return fallback


def _normalize_row(raw: dict[str, str]) -> dict[str, Any] | None:
    sample_id = (raw.get("id") or "").strip()
    task = (raw.get("task") or "").strip()
    if not sample_id or not task:
        return None

    row: dict[str, Any] = {
        "id": sample_id,
        "task": task,
        "input": _parse_json_field(raw.get("input", ""), {}),
        "expected": _parse_json_field(raw.get("expected", ""), {"positives": [], "acceptable": [], "negatives": []}),
    }

    input_text = (raw.get("input_text") or "").strip()
    if input_text:
        row["input_text"] = input_text

    candidates = _parse_json_field(raw.get("candidates", ""), None)
    if isinstance(candidates, list):
        row["candidates"] = candidates

    stats = _parse_json_field(raw.get("stats", ""), None)
    if isinstance(stats, dict):
        row["stats"] = stats

    note = (raw.get("note") or "").strip()
    if note:
        row["note"] = note
    return row


def _resolve_csv_path(csv_path: str | None, csv_pattern: str) -> Path:
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path.as_posix()}")
        return path
    matches = sorted(glob.glob(csv_pattern))
    if not matches:
        raise FileNotFoundError(f"No CSV matched pattern: {csv_pattern}")
    return Path(matches[0])


def import_csv_into_seed(
    seed_path: str,
    csv_path: str | None,
    csv_pattern: str,
    prefer_csv: bool,
) -> None:
    seed_file = Path(seed_path)
    existing_rows = load_jsonl(seed_file) if seed_file.exists() else []
    merged: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        row_id = str(row.get("id", "")).strip()
        if row_id:
            merged[row_id] = row

    csv_file = _resolve_csv_path(csv_path, csv_pattern)
    imported = 0
    skipped = 0
    replaced = 0
    with csv_file.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            normalized = _normalize_row(raw)
            if not normalized:
                skipped += 1
                continue
            row_id = normalized["id"]
            if row_id in merged:
                if prefer_csv:
                    merged[row_id] = normalized
                    replaced += 1
                else:
                    skipped += 1
                continue
            merged[row_id] = normalized
            imported += 1

    output_rows = [merged[key] for key in sorted(merged.keys())]
    write_jsonl(seed_file, output_rows)
    print(f"CSV: {csv_file.as_posix()}")
    print(f"Imported new rows: {imported}")
    print(f"Replaced rows: {replaced}")
    print(f"Skipped rows: {skipped}")
    print(f"Total seed rows: {len(output_rows)}")
    print(f"Updated: {seed_file.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import WordCraft seed rows from CSV into seed_gold.jsonl.")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Seed JSONL target path.")
    parser.add_argument("--csv", default=None, help="Explicit CSV path.")
    parser.add_argument("--csv-pattern", default=DEFAULT_CSV_PATTERN, help="Glob pattern used when --csv is omitted.")
    parser.add_argument(
        "--prefer-csv",
        action="store_true",
        help="Replace existing rows when IDs overlap.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import_csv_into_seed(
        seed_path=args.seed,
        csv_path=args.csv,
        csv_pattern=args.csv_pattern,
        prefer_csv=args.prefer_csv,
    )
