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
GENERIC_EDITOR_WORDS = {
    "good",
    "bad",
    "nice",
    "great",
    "thing",
    "things",
    "stuff",
    "person",
    "people",
    "feeling",
    "feelings",
    "very",
    "really",
}
FORMAL_CONTEXTS = {"formal", "academic", "scholarly", "professional", "literary"}
ADVANCED_CONTEXTS = FORMAL_CONTEXTS | {"advanced"}
SOURCE_PRIORITY = [
    "selection",
    "definition",
    "wordnet",
    "conceptnet",
    "derivational",
    "context",
    "slot",
    "pattern",
    "neutral",
    "fallback",
]
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


def _quality_bucket(
    score: float,
    grammar: float,
    semantic: float,
    context_sim: float,
    blank_present: bool,
    expected_pos: set[str] | None,
) -> int:
    if blank_present and expected_pos:
        if grammar >= 0.95 and semantic >= 0.52:
            return 4
        if grammar >= 0.7 and semantic >= 0.5:
            return 3
        if grammar >= 0.45 and (semantic >= 0.56 or context_sim >= 0.6):
            return 2
        return 0
    if score >= 0.56 and semantic >= 0.52:
        return 3
    if score >= 0.44 and (semantic >= 0.48 or context_sim >= 0.56):
        return 2
    if score >= 0.32:
        return 1
    return 0


def _generic_penalty(
    context_key: str,
    word: str,
    *,
    semantic: float,
    context_sim: float,
    frequency: float,
    sources: set[str],
    blank_present: bool,
    expected_pos: set[str] | None,
    vocabulary_preference: str = "balanced",
) -> float:
    penalty = 0.0
    if word in GENERIC_EDITOR_WORDS:
        penalty += 0.10
    if not blank_present and not expected_pos and word in GENERIC_EDITOR_WORDS:
        penalty += 0.06
    if context_key in FORMAL_CONTEXTS and word in GENERIC_EDITOR_WORDS:
        penalty += 0.08
    if blank_present and expected_pos and word in {"good", "bad", "nice"}:
        penalty += 0.06
    if frequency >= 0.65 and semantic < 0.62 and context_sim < 0.62 and "definition" not in sources:
        penalty += 0.04
    if vocabulary_preference == "advanced" and word in GENERIC_EDITOR_WORDS:
        penalty += 0.08
    if vocabulary_preference == "advanced" and frequency >= 0.72 and semantic < 0.68:
        penalty += 0.04
    return penalty


def _precision_bonus(
    context_key: str,
    *,
    semantic: float,
    context_sim: float,
    frequency: float,
    sources: set[str],
    blank_present: bool,
    expected_pos: set[str] | None,
    vocabulary_preference: str = "balanced",
) -> float:
    bonus = 0.0
    if "definition" in sources and semantic >= 0.58:
        bonus += 0.03
    if not blank_present and not expected_pos and "definition" in sources and semantic >= 0.62:
        bonus += 0.03
    if context_key in FORMAL_CONTEXTS and 0.02 <= frequency <= 0.38 and (semantic >= 0.58 or context_sim >= 0.62):
        bonus += 0.04
    if vocabulary_preference == "advanced":
        if "definition" in sources and semantic >= 0.6:
            bonus += 0.04
        if 0.015 <= frequency <= 0.32 and (semantic >= 0.58 or context_sim >= 0.6):
            bonus += 0.04
        if context_key in ADVANCED_CONTEXTS and 0.01 <= frequency <= 0.28 and semantic >= 0.56:
            bonus += 0.03
    return bonus


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
    vocabulary_preference: str = "balanced",
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
        sources = set()
        source_score = 0.0
        if source_map:
            sources = source_map.get(word, set())
            if "definition" in sources:
                source_score += 0.04
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

        precision_bonus = _precision_bonus(
            context_key,
            semantic=semantic,
            context_sim=context_sim,
            frequency=frequency,
            sources=sources,
            blank_present=blank_present,
            expected_pos=expected_pos,
            vocabulary_preference=vocabulary_preference,
        )
        generic_penalty = _generic_penalty(
            context_key,
            word,
            semantic=semantic,
            context_sim=context_sim,
            frequency=frequency,
            sources=sources,
            blank_present=blank_present,
            expected_pos=expected_pos,
            vocabulary_preference=vocabulary_preference,
        )

        score = (
            active_weights["semantic"] * semantic
            + active_weights["context"] * context_sim
            + active_weights["emotion"] * emotion
            + active_weights["grammar"] * grammar
            + active_weights["frequency"] * frequency
            + source_score
            + precision_bonus
            - generic_penalty
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
        quality_bucket = _quality_bucket(
            score=score,
            grammar=grammar,
            semantic=semantic,
            context_sim=context_sim,
            blank_present=blank_present,
            expected_pos=expected_pos,
        )
        primary_source = "model"
        if sources:
            for src in SOURCE_PRIORITY:
                if src in sources:
                    primary_source = src
                    break
            else:
                primary_source = sorted(sources)[0]
        scored.append(
            {
                "word": word,
                "score": round(float(score), 4),
                "pos": pos,
                "note": " ".join(reasons),
                "source": primary_source,
                "_quality_bucket": quality_bucket,
            }
        )

    scored = [item for item in scored if item["_quality_bucket"] > 0]
    scored.sort(key=lambda item: (item["_quality_bucket"], item["score"]), reverse=True)
    cleaned = []
    for item in scored[:top_k]:
        next_item = {key: value for key, value in item.items() if not key.startswith("_")}
        cleaned.append(next_item)
    return cleaned
