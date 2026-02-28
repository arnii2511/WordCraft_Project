from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import math

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def build_input_text(task: str, payload: dict[str, Any]) -> str:
    if task in {"suggest_blank", "suggest_selection", "suggest_sentence", "rewrite"}:
        sentence = (payload.get("sentence") or "").strip()
        context = (payload.get("context") or "neutral").strip()
        mode = (payload.get("mode") or "write").strip()
        selection = payload.get("selection")
        selection_text = ""
        if isinstance(selection, dict):
            selection_text = (selection.get("text") or "").strip()
        parts = [f"sentence={sentence}", f"context={context}", f"mode={mode}"]
        if selection_text:
            parts.append(f"selection={selection_text}")
        return " | ".join(parts)

    if task == "lexical":
        return " | ".join(
            [
                f"word={(payload.get('word') or '').strip()}",
                f"lexical_task={(payload.get('lexical_task') or '').strip()}",
                f"context={(payload.get('context') or 'neutral').strip()}",
            ]
        )

    if task == "constraints":
        return " | ".join(
            [
                f"rhyme_with={(payload.get('rhyme_with') or '').strip()}",
                f"relation={(payload.get('relation') or '').strip()}",
                f"meaning_of={(payload.get('meaning_of') or '').strip()}",
                f"context={(payload.get('context') or 'neutral').strip()}",
            ]
        )

    if task == "oneword":
        return " | ".join(
            [
                f"query={(payload.get('query') or '').strip()}",
                f"context={(payload.get('context') or 'neutral').strip()}",
            ]
        )

    return json.dumps(payload, ensure_ascii=True)


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


@dataclass
class RankingMetrics:
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    ndcg_at_5: float
    mrr: float
    samples: int


def compute_ranking_metrics(grouped_rows: dict[str, list[dict[str, Any]]]) -> RankingMetrics:
    if not grouped_rows:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    p1_total = 0.0
    p3_total = 0.0
    p5_total = 0.0
    h1_total = 0.0
    h3_total = 0.0
    h5_total = 0.0
    ndcg5_total = 0.0
    mrr_total = 0.0
    count = 0

    def _gain(label: int) -> float:
        return float((2 ** max(0, int(label))) - 1)

    def _dcg(labels: list[int]) -> float:
        score = 0.0
        for idx, label in enumerate(labels, start=1):
            score += _gain(label) / math.log2(idx + 1)
        return score

    for rows in grouped_rows.values():
        ranked = sorted(rows, key=lambda row: row["pred_score"], reverse=True)
        if not ranked:
            continue
        count += 1

        p1_total += 1.0 if ranked[0]["label"] >= 2 else 0.0
        h1_total += 1.0 if ranked[0]["label"] >= 2 else 0.0

        top3 = ranked[:3]
        relevant_in_top3 = sum(1 for row in top3 if row["label"] >= 2)
        p3_total += relevant_in_top3 / 3.0
        h3_total += 1.0 if relevant_in_top3 > 0 else 0.0

        top5 = ranked[:5]
        relevant_in_top5 = sum(1 for row in top5 if row["label"] >= 2)
        p5_total += relevant_in_top5 / 5.0
        h5_total += 1.0 if relevant_in_top5 > 0 else 0.0

        ranked_labels = [int(row["label"]) for row in top5]
        ideal_labels = sorted((int(row["label"]) for row in rows), reverse=True)[:5]
        ideal_dcg = _dcg(ideal_labels)
        if ideal_dcg > 0.0:
            ndcg5_total += _dcg(ranked_labels) / ideal_dcg

        reciprocal_rank = 0.0
        for idx, row in enumerate(ranked, start=1):
            if row["label"] >= 2:
                reciprocal_rank = 1.0 / idx
                break
        mrr_total += reciprocal_rank

    if count == 0:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    return RankingMetrics(
        precision_at_1=round(p1_total / count, 4),
        precision_at_3=round(p3_total / count, 4),
        precision_at_5=round(p5_total / count, 4),
        hit_at_1=round(h1_total / count, 4),
        hit_at_3=round(h3_total / count, 4),
        hit_at_5=round(h5_total / count, 4),
        ndcg_at_5=round(ndcg5_total / count, 4),
        mrr=round(mrr_total / count, 4),
        samples=count,
    )
