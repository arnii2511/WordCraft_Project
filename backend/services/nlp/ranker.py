from __future__ import annotations

from typing import Iterable

import numpy as np

from . import embeddings
from . import wordnet_service

BLANK_TOKEN = "[BLANK]"
BLANK_PLACEHOLDER = "BLANKTOKEN"

COPULAR_VERBS = {
    "be",
    "seem",
    "feel",
    "become",
    "remain",
    "appear",
    "look",
    "sound",
    "smell",
    "taste",
    "grow",
    "get",
}
IRREGULAR_ADVERBS = {"well", "fast", "hard", "late", "early", "straight", "right"}
_DETERMINERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "our",
    "their",
}
_PREPOSITIONS = {"in", "on", "at", "into", "with", "by", "for", "from", "to", "of", "over", "under"}

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

_SPACY_NLP = None


def _get_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if spacy is None:
        return None
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    except OSError:
        _SPACY_NLP = None
    return _SPACY_NLP


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _scale_similarity(similarity: float) -> float:
    return (similarity + 1.0) / 2.0


def infer_expected_pos(text_with_placeholder: str) -> set[str] | None:
    nlp = _get_spacy()
    if nlp is None:
        tokens = text_with_placeholder.split()
        if BLANK_PLACEHOLDER not in tokens:
            return None
        idx = tokens.index(BLANK_PLACEHOLDER)
        prev = tokens[idx - 1].lower() if idx > 0 else ""
        next_token = tokens[idx + 1].lower() if idx + 1 < len(tokens) else ""
        if prev == "to":
            return {"VERB"}
        if prev in COPULAR_VERBS:
            return {"ADJ"}
        if prev in _DETERMINERS:
            return {"NOUN", "ADJ"}
        if prev.endswith(("ed", "ing")):
            return {"ADV"}
        if next_token in _PREPOSITIONS:
            if prev.endswith(("ed", "ing")):
                return {"ADV"}
            return {"NOUN"}
        return None
    doc = nlp(text_with_placeholder)
    blank_index = None
    for i, token in enumerate(doc):
        if token.text == BLANK_PLACEHOLDER:
            blank_index = i
            break
    if blank_index is None:
        return None

    prev_token = doc[blank_index - 1] if blank_index > 0 else None
    next_token = doc[blank_index + 1] if blank_index + 1 < len(doc) else None

    if prev_token is not None:
        if prev_token.lower_ == "to":
            return {"VERB"}
        if prev_token.lemma_.lower() in COPULAR_VERBS:
            return {"ADJ"}
        if prev_token.pos_ in {"DET", "PRON"}:
            return {"NOUN", "ADJ"}
        if prev_token.pos_ in {"VERB", "AUX"}:
            return {"ADV"}
        if prev_token.pos_ == "ADJ":
            return {"NOUN"}
        if prev_token.pos_ == "ADP":
            return {"NOUN"}

    if next_token is not None:
        if next_token.pos_ in {"NOUN", "PROPN"}:
            return {"ADJ"}
        if next_token.pos_ in {"VERB", "AUX"}:
            return {"ADV"}
        if next_token.pos_ == "ADP":
            return {"NOUN"}

    return None


def describe_slot_hint(text_with_placeholder: str, expected_pos: set[str] | None) -> str | None:
    if not expected_pos:
        return None
    nlp = _get_spacy()
    if nlp is None:
        return f"Fits expected {', '.join(sorted(expected_pos))} slot."
    doc = nlp(text_with_placeholder)
    idx = None
    for i, token in enumerate(doc):
        if token.text == BLANK_PLACEHOLDER:
            idx = i
            break
    if idx is None:
        return f"Fits expected {', '.join(sorted(expected_pos))} slot."
    prev_token = doc[idx - 1] if idx > 0 else None
    if prev_token is None:
        return f"Fits expected {', '.join(sorted(expected_pos))} slot."
    if prev_token.lower_ == "to":
        return "Fits infinitive slot after 'to'."
    if prev_token.lemma_.lower() in COPULAR_VERBS:
        return f"Fits descriptive slot after '{prev_token.text}'."
    if prev_token.pos_ in {"VERB", "AUX"}:
        return f"Fits manner slot after '{prev_token.text}'."
    return f"Fits expected {', '.join(sorted(expected_pos))} slot."


def _grammatical_fit(word: str, expected_pos: set[str] | None) -> float:
    if expected_pos is None:
        return 1.0

    tags = wordnet_service.get_pos_tags(word)
    if not tags:
        return 0.4
    if tags & expected_pos:
        if expected_pos == {"ADV"}:
            if word.endswith("ly") or word in IRREGULAR_ADVERBS:
                return 1.0
            return 0.45
        if expected_pos == {"VERB"}:
            if word.endswith(("e", "ed", "ing")) or "VERB" in tags:
                return 1.0
            return 0.55
        return 1.0
    if "ADJ" in expected_pos and "ADV" in tags:
        return 0.45
    if "ADV" in expected_pos and "ADJ" in tags:
        return 0.45
    return 0.0


def _resolve_pos(word: str, expected_pos: set[str] | None) -> str:
    tags = wordnet_service.get_pos_tags(word)
    if expected_pos:
        overlap = tags & expected_pos
        if overlap:
            return sorted(overlap)[0]
    primary = wordnet_service.get_primary_pos(word)
    if primary:
        return primary
    if tags:
        return sorted(tags)[0]
    nlp = _get_spacy()
    if nlp is None:
        return "X"
    doc = nlp(word)
    if not doc:
        return "X"
    pos = doc[0].pos_.upper()
    if pos in {"NOUN", "VERB", "ADJ", "ADV"}:
        return pos
    return "X"


def rank_candidates(
    cleaned_text: str,
    context_key: str,
    candidates: Iterable[str],
    context_description: str,
    blank_present: bool,
    emotion_scores: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    top_k: int = 5,
    source_map: dict[str, set[str]] | None = None,
    strict_pos: bool = False,
    expected_pos_override: set[str] | None = None,
    context_words: set[str] | None = None,
) -> list[dict]:
    sentence_vector = embeddings.embed_sentence(cleaned_text)
    context_vector = embeddings.get_context_centroid(context_key)

    expected_pos = expected_pos_override
    slot_hint = None
    if expected_pos is None and blank_present:
        pos_text = cleaned_text.replace(BLANK_TOKEN, BLANK_PLACEHOLDER)
        expected_pos = infer_expected_pos(pos_text)
        slot_hint = describe_slot_hint(pos_text, expected_pos)

    default_weights = {
        "semantic": 0.42,
        "context": 0.24,
        "emotion": 0.08,
        "grammar": 0.18,
        "frequency": 0.08,
    }
    active_weights = default_weights if weights is None else {**default_weights, **weights}

    scored: list[dict] = []
    context_vocab = context_words or set()
    for word in candidates:
        if not wordnet_service.is_valid_word(word):
            continue
        grammar = _grammatical_fit(word, expected_pos)
        if strict_pos and expected_pos and grammar < 0.95:
            continue

        word_vector = embeddings.get_word_embedding(word)
        semantic = _scale_similarity(_cosine_similarity(sentence_vector, word_vector))
        context_sim = _scale_similarity(_cosine_similarity(context_vector, word_vector))
        emotion = emotion_scores.get(word, 0.0) if emotion_scores else 0.0
        frequency = wordnet_service.estimate_frequency(word)
        source_score = 0.0
        if source_map:
            sources = source_map.get(word, set())
            if "wordnet" in sources:
                source_score += 0.05
            if "conceptnet" in sources:
                source_score += 0.03
            if "context" in sources:
                source_score += 0.03
            if "derivational" in sources:
                source_score += 0.02
        if context_vocab and word in context_vocab:
            context_sim = max(context_sim, 0.62)

        score = (
            active_weights["semantic"] * semantic
            + active_weights["context"] * context_sim
            + active_weights["emotion"] * emotion
            + active_weights["grammar"] * grammar
            + active_weights["frequency"] * frequency
            + source_score
        )

        reasons: list[str] = []
        if expected_pos and grammar >= 0.95:
            reasons.append(slot_hint or "Fits the grammatical slot.")
        elif expected_pos and grammar < 0.2:
            reasons.append("Weak grammatical fit.")

        if context_sim >= 0.62:
            reasons.append(f"Matches {context_key} tone.")
        elif context_description:
            reasons.append(f"Aligned with {context_description}.")

        if semantic >= 0.62:
            reasons.append("Strong semantic fit.")
        elif semantic >= 0.54:
            reasons.append("Good semantic match.")
        else:
            reasons.append("Lexical alternative for this context.")

        if frequency < 0.06:
            reasons.append("Rare word.")

        pos = _resolve_pos(word, expected_pos)
        scored.append(
            {
                "word": word,
                "score": round(float(score), 4),
                "pos": pos,
                "note": " ".join(reasons),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
