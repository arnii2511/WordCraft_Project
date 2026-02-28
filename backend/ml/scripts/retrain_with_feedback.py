from __future__ import annotations

import argparse
from pathlib import Path

from .eval_reranker import evaluate
from .export_feedback_dataset import export_feedback_dataset
from .split_dataset import split_dataset
from .test_reranker import test_model
from .train_reranker import train

DEFAULT_BASE_DATASET = "backend/ml/data/dataset_ranker.jsonl"
DEFAULT_FEEDBACK_OUTPUT = "backend/ml/data/dataset_feedback.jsonl"
DEFAULT_SPLIT_DIR = "backend/ml/data/splits"
DEFAULT_ARTIFACT = "backend/ml/models/reranker.pkl"


def run(args: argparse.Namespace) -> None:
    append_target = args.base_dataset if args.append_feedback else None
    export_feedback_dataset(output_path=args.feedback_output, append_to=append_target)

    split_dataset(
        dataset_path=args.base_dataset,
        out_dir=args.split_dir,
        train_ratio=min(max(args.train_ratio, 0.6), 0.9),
        val_ratio=min(max(args.val_ratio, 0.05), 0.2),
        seed=args.seed,
        regression_size=max(50, args.regression_size),
        split_mode=args.split_mode,
    )

    train_path = str(Path(args.split_dir) / "train.jsonl")
    val_path = str(Path(args.split_dir) / "val.jsonl")
    test_path = str(Path(args.split_dir) / "test.jsonl")

    train(
        artifact_path=args.artifact,
        test_size=0.25,
        random_state=args.seed,
        dataset_path=args.base_dataset,
        train_path=train_path,
        val_path=val_path,
        exclude_gold_seed=args.exclude_gold_seed,
    )

    print("\nValidation metrics:")
    evaluate(
        dataset_path=val_path,
        artifact_path=args.artifact,
        exclude_gold_seed=args.exclude_gold_seed,
        task_filter=args.task_filter,
    )
    print("\nTest metrics:")
    test_model(
        dataset_path=test_path,
        artifact_path=args.artifact,
        exclude_gold_seed=args.exclude_gold_seed,
        task_filter=args.task_filter,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh ranker with Mongo feedback, rebuild splits, retrain, and evaluate."
    )
    parser.add_argument("--base-dataset", default=DEFAULT_BASE_DATASET, help="Main dataset JSONL path.")
    parser.add_argument(
        "--feedback-output",
        default=DEFAULT_FEEDBACK_OUTPUT,
        help="Standalone exported feedback JSONL path.",
    )
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Output split directory.")
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Output model artifact path.")
    parser.add_argument(
        "--append-feedback",
        action="store_true",
        help="Append exported feedback rows into --base-dataset before splitting/training.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "hard"],
        default="hard",
        help="hard is recommended for realistic generalization checks.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split/training.")
    parser.add_argument("--regression-size", type=int, default=200, help="Regression set rows.")
    parser.add_argument(
        "--exclude-gold-seed",
        action="store_true",
        help="Exclude source=gold_seed rows in train/eval for strict realism.",
    )
    parser.add_argument(
        "--task-filter",
        choices=["non_rewrite", "rewrite", "all"],
        default="non_rewrite",
        help="Evaluation/test task slice. non_rewrite is recommended for lexical ranker tracking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
