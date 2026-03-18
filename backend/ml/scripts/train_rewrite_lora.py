from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _require_modules():
    try:
        import torch  # noqa: F401
        from datasets import Dataset  # noqa: F401
        from peft import LoraConfig, get_peft_model  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "LoRA dependencies missing. Install: pip install transformers datasets peft accelerate bitsandbytes"
        ) from exc


def _load_rewrite_rows(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            source = str(item.get("source_sentence", "")).strip()
            target = str(item.get("target_rewrite", "")).strip()
            context = str(item.get("context", "neutral")).strip()
            if not source or not target:
                continue
            rows.append({"source_sentence": source, "target_rewrite": target, "context": context})
    return rows


def train_rewrite_lora(
    *,
    dataset_path: str,
    output_dir: str,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> None:
    _require_modules()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    rows = _load_rewrite_rows(dataset_path)
    if not rows:
        raise ValueError("No rewrite rows found. Expected JSONL with source_sentence + target_rewrite.")

    def _prompt(row: dict[str, Any]) -> str:
        return (
            "Task: Rewrite sentence while preserving meaning.\n"
            f"Context: {row['context']}\n"
            f"Input: {row['source_sentence']}\n"
            f"Output: {row['target_rewrite']}"
        )

    dataset = Dataset.from_list([{"text": _prompt(row)} for row in rows])
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(base_model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=max(1, int(batch_size)),
        learning_rate=max(1e-6, float(learning_rate)),
        num_train_epochs=max(1, int(epochs)),
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA rewrite adapter: {Path(output_dir).as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning scaffold for rewrite generation.")
    parser.add_argument(
        "--dataset",
        default="backend/ml/data/rewrite_train.jsonl",
        help="Rewrite training JSONL path (source_sentence, target_rewrite, context).",
    )
    parser.add_argument("--output-dir", default="backend/ml/models/rewrite_lora", help="Output model directory.")
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF base model.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max sequence length.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_rewrite_lora(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        learning_rate=max(1e-6, float(args.learning_rate)),
        max_length=max(128, int(args.max_length)),
    )
