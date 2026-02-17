from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CONTEXT_PATH = Path(__file__).resolve().parent / "data" / "context_vocab.json"


def load_contexts(path: Path | None = None) -> dict:
    context_path = path or DEFAULT_CONTEXT_PATH
    with context_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    contexts: dict[str, dict] = {}
    for name, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        description = str(payload.get("description", "")).strip()
        words = payload.get("words", [])
        if not description or not isinstance(words, list):
            continue
        cleaned_words = []
        for word in words:
            if not isinstance(word, str):
                continue
            cleaned = word.strip().lower()
            if cleaned:
                cleaned_words.append(cleaned)
        if not cleaned_words:
            continue
        contexts[name.strip().lower()] = {
            "description": description,
            "words": sorted(set(cleaned_words)),
        }

    if not contexts:
        raise ValueError(f"No valid contexts found in {context_path}")

    return contexts
