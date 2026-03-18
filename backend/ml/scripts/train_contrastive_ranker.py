from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from .common import load_jsonl

DEFAULT_TRAIN_SPLIT = "backend/ml/data/splits/train.jsonl"
DEFAULT_MODEL_DIR = "backend/ml/models/contrastive_ranker"


def _build_query(row: dict[str, Any]) -> str:
    payload = row.get("input", {}) or {}
    task = str(row.get("task", "")).strip().lower()
    context = str(payload.get("context", "") or "neutral").strip().lower()
    text = str(row.get("input_text", "") or "").strip()
    return f"task={task} context={context} input={text}"


def _triplets(rows: list[dict[str, Any]], exclude_gold_seed: bool) -> list[InputExample]:
    items: list[InputExample] = []
    for row in rows:
        task = str(row.get("task", "")).strip().lower()
        if task == "rewrite":
            continue
        query = _build_query(row)
        positives = []
        negatives = []
        for candidate in row.get("candidates", []):
            if exclude_gold_seed and str(candidate.get("source", "")).strip().lower() == "gold_seed":
                continue
            text = str(candidate.get("text", "")).strip()
            if not text:
                continue
            label = int(candidate.get("label", 0))
            if label >= 2:
                positives.append(text)
            elif label <= 1:
                negatives.append(text)
        if not positives or not negatives:
            continue
        for pos in positives[:4]:
            for neg in negatives[:4]:
                items.append(InputExample(texts=[query, pos, neg]))
    return items


def train_contrastive(
    train_path: str,
    output_dir: str,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    exclude_gold_seed: bool,
) -> None:
    rows = load_jsonl(train_path)
    triplets = _triplets(rows, exclude_gold_seed=exclude_gold_seed)
    if not triplets:
        raise ValueError("No triplets generated. Check dataset labels/filters.")

    model = SentenceTransformer(base_model)
    train_loader = DataLoader(triplets, shuffle=True, batch_size=max(4, int(batch_size)))
    train_loss = losses.TripletLoss(model=model)

    warmup_steps = max(1, int(0.1 * len(train_loader) * max(1, int(epochs))))
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=max(1, int(epochs)),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": max(1e-6, float(learning_rate))},
        show_progress_bar=True,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Triplets: {len(triplets)}")
    print(f"Saved contrastive model: {out_path.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train triplet contrastive model for ranking features.")
    parser.add_argument("--train", default=DEFAULT_TRAIN_SPLIT, help="Grouped train split JSONL path.")
    parser.add_argument("--output-dir", default=DEFAULT_MODEL_DIR, help="Model output directory.")
    parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer base model.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument(
        "--exclude-gold-seed",
        action="store_true",
        help="Exclude source=gold_seed rows from triplet generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_contrastive(
        train_path=args.train,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=max(1, int(args.epochs)),
        batch_size=max(4, int(args.batch_size)),
        learning_rate=max(1e-6, float(args.learning_rate)),
        exclude_gold_seed=args.exclude_gold_seed,
    )
