from __future__ import annotations

import re
from typing import NamedTuple

BLANK_TOKEN = "[BLANK]"
BLANK_RE = re.compile(
    r"(\_{2,}|\.{3,}|\(\s*blank\s*\)|\[\s*blank\s*\]|<\s*blank\s*>|\{\s*blank\s*\})",
    re.IGNORECASE,
)


class PreprocessResult(NamedTuple):
    cleaned_text: str
    normalized_text: str
    tokens: list[str]
    blank_index: int | None
    blank_present: bool


def preprocess_text(text: str) -> PreprocessResult:
    if not text:
        return PreprocessResult("", "", [], None, False)

    normalized = text.strip()
    normalized = BLANK_RE.sub(BLANK_TOKEN, normalized)
    normalized = re.sub(
        rf"\s*{re.escape(BLANK_TOKEN)}\s*",
        f" {BLANK_TOKEN} ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()

    tokens = normalized.split(" ") if normalized else []
    blank_positions = [i for i, token in enumerate(tokens) if token == BLANK_TOKEN]
    if len(blank_positions) > 1:
        for index in blank_positions[1:]:
            tokens[index] = ""
        tokens = [token for token in tokens if token]
        normalized = " ".join(tokens)

    blank_index = None
    for i, token in enumerate(tokens):
        if token == BLANK_TOKEN:
            blank_index = i
            break

    placeholder = "BLANKTOKEN"
    normalized_for_lower = normalized.replace(BLANK_TOKEN, placeholder)
    cleaned = normalized_for_lower.lower().replace(placeholder.lower(), BLANK_TOKEN)

    return PreprocessResult(
        cleaned_text=cleaned,
        normalized_text=normalized,
        tokens=tokens,
        blank_index=blank_index,
        blank_present=blank_index is not None,
    )


def extract_focus_word(tokens: list[str], blank_index: int | None) -> str | None:
    if not tokens:
        return None
    if blank_index is not None:
        if blank_index > 0:
            return tokens[blank_index - 1].strip(".,!?;:") or None
        return None

    for token in reversed(tokens):
        cleaned = token.strip(".,!?;:")
        if cleaned:
            return cleaned
    return None
