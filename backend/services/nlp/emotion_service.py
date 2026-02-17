from __future__ import annotations

from pathlib import Path

NRC_PATH = Path(__file__).resolve().parent / "data" / "NRC_emotion_lexicon.txt"

_LEXICON: dict[str, set[str]] = {}
_LOADED = False

CONTEXT_EMOTION_MAP: dict[str, set[str]] = {
    "neutral": set(),
    "hopeful": {"anticipation", "joy", "trust"},
    "horror": {"fear", "sadness"},
    "nostalgia": {"sadness", "joy"},
    "romantic": {"joy", "trust", "anticipation"},
    "academic": {"trust"},
    "joyful": {"joy"},
    "melancholic": {"sadness"},
    "mysterious": {"anticipation", "fear"},
    "formal": {"trust"},
}


def load_lexicon(path: Path | None = None) -> None:
    global _LEXICON, _LOADED
    if _LOADED:
        return
    target = path or NRC_PATH
    if not target.exists():
        _LOADED = True
        return
    lexicon: dict[str, set[str]] = {}
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, value = parts
            if value != "1":
                continue
            bucket = lexicon.setdefault(word, set())
            bucket.add(emotion)
    _LEXICON = lexicon
    _LOADED = True


def emotion_score(word: str, context: str) -> float:
    if not _LOADED:
        load_lexicon()
    if not word:
        return 0.0
    context_key = (context or "").lower()
    emotions = CONTEXT_EMOTION_MAP.get(context_key, set())
    if not emotions:
        return 0.0
    word_emotions = _LEXICON.get(word, set())
    if not word_emotions:
        return 0.0
    overlap = word_emotions.intersection(emotions)
    return len(overlap) / len(emotions)
