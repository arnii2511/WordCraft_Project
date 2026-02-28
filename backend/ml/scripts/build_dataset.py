from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from backend.services.nlp.constraints_service import get_constraint_matches
from backend.services.nlp.engine import generate_suggestions
from backend.services.nlp.lexical_service import get_lexical_results
from backend.services.nlp.oneword_service import get_one_word_substitutions

from .common import build_input_text, load_jsonl, normalize_text, write_jsonl

DEFAULT_SEED = "backend/ml/data/seed_gold.jsonl"
DEFAULT_OUTPUT = "backend/ml/data/dataset_ranker.jsonl"


def _as_normalized_set(values: list[str] | None) -> set[str]:
    return {normalize_text(value) for value in (values or []) if normalize_text(value)}


def _label_for_candidate(
    text: str,
    positives: list[str],
    acceptable: set[str],
    negatives: set[str],
) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 0
    if normalized in negatives:
        return 0
    for idx, target in enumerate(positives):
        if normalized == normalize_text(target):
            return 3 if idx == 0 else 2
    if normalized in acceptable:
        return 1
    return 0


def _candidate_row(
    text: str,
    *,
    model_score: float = 0.0,
    reason: str = "",
    pos: str | None = None,
    source: str = "model",
) -> dict[str, Any]:
    return {
        "text": text,
        "label": 0,
        "model_score": round(float(model_score or 0.0), 4),
        "pos": pos,
        "reason": reason,
        "source": source,
    }


def _run_task(task: str, payload: dict[str, Any], limit: int) -> tuple[list[dict[str, Any]], str | None]:
    if task in {"suggest_blank", "suggest_selection", "suggest_sentence"}:
        output = generate_suggestions(
            text=payload.get("sentence", ""),
            context=payload.get("context", "neutral"),
            mode=payload.get("mode", "write"),
            selection=payload.get("selection"),
            trigger=payload.get("trigger", "auto"),
            max_suggestions=limit,
        )
        suggestions = output.get("suggestions", [])[:limit]
        rows = [
            _candidate_row(
                item.get("word", ""),
                model_score=item.get("score", 0.0),
                reason=item.get("note", ""),
                pos=item.get("pos"),
                source=item.get("source", "suggest"),
            )
            for item in suggestions
            if item.get("word")
        ]
        return rows, output.get("explanation")

    if task == "rewrite":
        output = generate_suggestions(
            text=payload.get("sentence", ""),
            context=payload.get("context", "neutral"),
            mode="rewrite",
            selection=None,
            trigger="button",
        )
        rewrites = output.get("rewrites", [])[:limit]
        rows = []
        for idx, rewrite in enumerate(rewrites):
            score = max(0.0, 1.0 - (idx * 0.12))
            rows.append(
                _candidate_row(
                    rewrite,
                    model_score=score,
                    reason="Rewrite proposal from transform mode.",
                    source="rewrite",
                )
            )
        return rows, output.get("explanation")

    if task == "lexical":
        _, details = get_lexical_results(
            word=payload.get("word", ""),
            task=payload.get("lexical_task", "synonyms"),
            context=payload.get("context"),
            max_results=limit,
        )
        rows = [
            _candidate_row(
                item.get("word", ""),
                model_score=item.get("score", 0.0),
                reason=item.get("reason", ""),
                pos=item.get("pos"),
                source="lexical",
            )
            for item in details
            if item.get("word")
        ]
        return rows, None

    if task == "constraints":
        results, note = get_constraint_matches(
            rhyme_with=payload.get("rhyme_with", ""),
            relation=payload.get("relation", "synonym"),
            meaning_of=payload.get("meaning_of", ""),
            context=payload.get("context"),
            limit=limit,
        )
        rows = [
            _candidate_row(
                item.get("word", ""),
                model_score=item.get("score", 0.0),
                reason=item.get("reason", ""),
                source="constraints",
            )
            for item in results
            if item.get("word")
        ]
        return rows, note

    if task == "oneword":
        results, note = get_one_word_substitutions(
            query=payload.get("query", ""),
            context=payload.get("context"),
            limit=limit,
        )
        rows = [
            _candidate_row(
                item.get("word", ""),
                model_score=item.get("score", 0.0),
                reason=item.get("reason", ""),
                source="oneword",
            )
            for item in results
            if item.get("word")
        ]
        return rows, note

    return [], f"Unsupported task: {task}"


def _enrich_and_label(
    record: dict[str, Any],
    generated: list[dict[str, Any]],
    *,
    inject_gold_positives: bool = True,
) -> dict[str, Any]:
    expected = record.get("expected", {})
    positives = expected.get("positives", []) or []
    acceptable = _as_normalized_set(expected.get("acceptable", []))
    negatives = _as_normalized_set(expected.get("negatives", []))

    by_text: dict[str, dict[str, Any]] = {}
    for candidate in generated:
        normalized = normalize_text(candidate.get("text", ""))
        if not normalized:
            continue
        by_text[normalized] = candidate

    # Optionally inject gold positives so supervised training always has positives.
    if inject_gold_positives:
        for positive in positives:
            normalized = normalize_text(positive)
            if normalized in by_text:
                continue
            by_text[normalized] = _candidate_row(
                positive,
                model_score=0.0,
                reason="Injected gold positive from seed dataset.",
                source="gold_seed",
            )

    labeled_candidates: list[dict[str, Any]] = []
    for normalized, candidate in by_text.items():
        label = _label_for_candidate(
            candidate.get("text", ""),
            positives=positives,
            acceptable=acceptable,
            negatives=negatives,
        )
        candidate["label"] = label
        candidate["text"] = candidate.get("text", "").strip()
        labeled_candidates.append(candidate)

    labeled_candidates.sort(key=lambda item: (item["label"], item["model_score"]), reverse=True)
    positive_count = sum(1 for item in labeled_candidates if item["label"] >= 2)

    return {
        **record,
        "input_text": build_input_text(record.get("task", ""), record.get("input", {})),
        "candidates": labeled_candidates,
        "stats": {
            "candidate_count": len(labeled_candidates),
            "positive_count": positive_count,
        },
    }


def build_dataset(
    seed_path: str,
    output_path: str,
    limit: int,
    keep_empty: bool,
    use_reranker: bool,
    inject_gold_positives: bool,
) -> None:
    if not use_reranker:
        os.environ["WORDCRAFT_DISABLE_RERANKER"] = "1"
    seed_rows = load_jsonl(seed_path)
    built_rows: list[dict[str, Any]] = []
    skipped = 0

    for row in seed_rows:
        task = row.get("task", "")
        payload = row.get("input", {})
        generated_candidates, note = _run_task(task=task, payload=payload, limit=limit)
        enriched = _enrich_and_label(
            row,
            generated_candidates,
            inject_gold_positives=inject_gold_positives,
        )
        if note:
            enriched["note"] = note
        if not keep_empty and enriched["stats"]["candidate_count"] == 0:
            skipped += 1
            continue
        built_rows.append(enriched)

    write_jsonl(output_path, built_rows)
    print(f"Built dataset rows: {len(built_rows)}")
    print(f"Skipped empty rows: {skipped}")
    print(f"Inject gold positives: {inject_gold_positives}")
    print(f"Output: {Path(output_path).as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supervised NLP reranker dataset.")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Path to seed gold JSONL file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=12, help="Max candidates per row.")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep rows that produce zero candidates.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Allow runtime reranker during candidate generation (off by default).",
    )
    parser.add_argument(
        "--inject-gold-positives",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Inject expected positives into candidate list (true by default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        seed_path=args.seed,
        output_path=args.output,
        limit=max(1, min(24, args.limit)),
        keep_empty=args.keep_empty,
        use_reranker=args.use_reranker,
        inject_gold_positives=args.inject_gold_positives.lower() == "true",
    )
