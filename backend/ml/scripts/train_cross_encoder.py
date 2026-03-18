from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers.cross_encoder import CrossEncoder

from .common import compute_ranking_metrics, load_jsonl

DEFAULT_TRAIN_SPLIT = "backend/ml/data/splits/train.jsonl"
DEFAULT_VAL_SPLIT = "backend/ml/data/splits/val.jsonl"
DEFAULT_ARTIFACT = "backend/ml/models/reranker_cross_encoder.pkl"
DEFAULT_MODEL_DIR = "backend/ml/models/cross_encoder_ranker"


def _feature_text(row: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, str]:
    payload = row.get("input", {}) or {}
    task = str(row.get("task", "")).strip().lower()
    mode = str(payload.get("mode", "") or "").strip().lower()
    context = str(payload.get("context", "") or "neutral").strip().lower()
    input_text = str(row.get("input_text", "") or "").strip()
    if not input_text:
        input_text = json.dumps(payload, ensure_ascii=True)
    query = f"task={task} mode={mode} context={context} input={input_text}"
    candidate_text = str(candidate.get("text", "")).strip()
    return query, candidate_text


def _group_rows(rows: list[dict[str, Any]], exclude_gold_seed: bool) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for row in rows:
        task = str(row.get("task", "")).strip().lower()
        if task == "rewrite":
            continue
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            continue
        candidates = []
        for candidate in row.get("candidates", []):
            if exclude_gold_seed and str(candidate.get("source", "")).strip().lower() == "gold_seed":
                continue
            text = str(candidate.get("text", "")).strip()
            if not text:
                continue
            query_text, candidate_text = _feature_text(row, candidate)
            candidates.append(
                {
                    "query": query_text,
                    "candidate": candidate_text,
                    "label": int(candidate.get("label", 0)),
                }
            )
        positives = [item for item in candidates if item["label"] >= 2]
        negatives = [item for item in candidates if item["label"] <= 1]
        if not positives or not negatives:
            continue
        grouped.append({"id": sample_id, "task": task, "candidates": candidates})
    return grouped


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _forward_scores(model: CrossEncoder, query: str, candidates: list[str], device: torch.device) -> torch.Tensor:
    encoded = model.tokenizer(
        [query] * len(candidates),
        candidates,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = _to_device(encoded, device)
    output = model.model(**encoded, return_dict=True)
    logits = output.logits
    if logits.ndim == 2:
        logits = logits.squeeze(-1)
    return logits.float()


def _listwise_loss(model: CrossEncoder, group: dict[str, Any], device: torch.device, label_temp: float) -> torch.Tensor:
    query = group["candidates"][0]["query"]
    candidates = [item["candidate"] for item in group["candidates"]]
    labels = torch.tensor([max(0, int(item["label"])) for item in group["candidates"]], dtype=torch.float32, device=device)
    logits = _forward_scores(model, query, candidates, device=device)
    target = torch.softmax(labels * label_temp, dim=0)
    log_probs = torch.log_softmax(logits, dim=0)
    return -(target * log_probs).sum()


def _pairwise_loss(model: CrossEncoder, group: dict[str, Any], device: torch.device, margin: float) -> torch.Tensor:
    positives = [item for item in group["candidates"] if item["label"] >= 2]
    negatives = [item for item in group["candidates"] if item["label"] <= 1]
    if not positives or not negatives:
        return torch.tensor(0.0, device=device)
    query = positives[0]["query"]
    random.shuffle(positives)
    random.shuffle(negatives)
    pairs = min(len(positives), len(negatives), 6)
    loss_total = torch.tensor(0.0, device=device)
    for idx in range(pairs):
        pos = positives[idx]["candidate"]
        neg = negatives[idx]["candidate"]
        pos_score = _forward_scores(model, query, [pos], device=device)[0]
        neg_score = _forward_scores(model, query, [neg], device=device)[0]
        loss_total = loss_total + torch.relu(torch.tensor(margin, device=device) - (pos_score - neg_score))
    return loss_total / max(1, pairs)


def _predict_group_scores(model: CrossEncoder, grouped_rows: list[dict[str, Any]], device: torch.device) -> dict[str, list[dict[str, Any]]]:
    grouped_scores: dict[str, list[dict[str, Any]]] = {}
    model.model.eval()
    with torch.no_grad():
        for row in grouped_rows:
            row_id = row["id"]
            query = row["candidates"][0]["query"]
            candidates = [item["candidate"] for item in row["candidates"]]
            logits = _forward_scores(model, query, candidates, device=device)
            values = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            grouped_scores[row_id] = [
                {"label": int(item["label"]), "pred_score": float(score)}
                for item, score in zip(row["candidates"], values)
            ]
    return grouped_scores


def train_cross_encoder(
    *,
    train_path: str,
    val_path: str,
    artifact_path: str,
    model_output_dir: str,
    base_model: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    objective: str,
    label_temperature: float,
    margin: float,
    exclude_gold_seed: bool,
) -> None:
    train_rows = _group_rows(load_jsonl(train_path), exclude_gold_seed=exclude_gold_seed)
    val_rows = _group_rows(load_jsonl(val_path), exclude_gold_seed=exclude_gold_seed)
    if not train_rows or not val_rows:
        raise ValueError("Train/val splits must contain grouped rows with positives and negatives.")

    model = CrossEncoder(base_model, num_labels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(max(1, int(epochs))):
        random.shuffle(train_rows)
        model.model.train()
        running = 0.0
        steps = 0
        for group in train_rows:
            optimizer.zero_grad()
            if objective == "pairwise":
                loss = _pairwise_loss(model, group, device=device, margin=margin)
            else:
                loss = _listwise_loss(
                    model,
                    group,
                    device=device,
                    label_temp=max(0.2, float(label_temperature)),
                )
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(loss.detach().cpu().item())
            steps += 1
        avg_loss = running / max(1, steps)
        print(f"Epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}")

    output_dir = Path(model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    grouped_scores = _predict_group_scores(model, val_rows, device=device)
    ranking = compute_ranking_metrics(grouped_scores)
    metrics = {
        "precision_at_1": ranking.precision_at_1,
        "precision_at_3": ranking.precision_at_3,
        "precision_at_5": ranking.precision_at_5,
        "hit_at_1": ranking.hit_at_1,
        "hit_at_3": ranking.hit_at_3,
        "hit_at_5": ranking.hit_at_5,
        "ndcg_at_5": ranking.ndcg_at_5,
        "mrr": ranking.mrr,
        "ranking_samples": ranking.samples,
        "objective": objective,
    }

    artifact = {
        "artifact_type": f"cross_encoder_{objective}",
        "model_path": output_dir.as_posix(),
        "metadata": {
            "base_model": base_model,
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "exclude_gold_seed": bool(exclude_gold_seed),
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "metrics": metrics,
        },
    }
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(artifact_path).open("wb") as handle:
        pickle.dump(artifact, handle)

    print(json.dumps(metrics, indent=2))
    print(f"Saved artifact: {Path(artifact_path).as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cross-encoder reranker with listwise/pairwise objective.")
    parser.add_argument("--train", default=DEFAULT_TRAIN_SPLIT, help="Grouped train split JSONL path.")
    parser.add_argument("--val", default=DEFAULT_VAL_SPLIT, help="Grouped validation split JSONL path.")
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Output artifact pickle path.")
    parser.add_argument("--model-output-dir", default=DEFAULT_MODEL_DIR, help="Directory to save trained cross-encoder.")
    parser.add_argument(
        "--base-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HF/SBERT cross-encoder base model.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument(
        "--objective",
        choices=["listwise", "pairwise"],
        default="listwise",
        help="Ranking objective.",
    )
    parser.add_argument(
        "--label-temperature",
        type=float,
        default=1.5,
        help="Listwise target softmax temperature over labels.",
    )
    parser.add_argument("--margin", type=float, default=0.3, help="Pairwise hinge margin.")
    parser.add_argument(
        "--exclude-gold-seed",
        action="store_true",
        help="Exclude source=gold_seed candidates from training/eval.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_cross_encoder(
        train_path=args.train,
        val_path=args.val,
        artifact_path=args.artifact,
        model_output_dir=args.model_output_dir,
        base_model=args.base_model,
        epochs=max(1, int(args.epochs)),
        lr=max(1e-6, float(args.lr)),
        weight_decay=max(0.0, float(args.weight_decay)),
        objective=args.objective,
        label_temperature=max(0.2, float(args.label_temperature)),
        margin=max(0.05, float(args.margin)),
        exclude_gold_seed=args.exclude_gold_seed,
    )
