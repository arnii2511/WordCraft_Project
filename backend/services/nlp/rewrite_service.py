from __future__ import annotations

import re
from typing import Iterable

from .pipeline import estimate_semantic_drift
from .wordnet_service import get_wordnet

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

_NLP = None

COMMON_REPLACEMENTS = [
    ("in order to", "to"),
    ("due to the fact that", "because"),
    ("at this point in time", "now"),
    ("in the event that", "if"),
    ("a lot of", "many"),
]

FILLER_WORDS = {"very", "really", "just", "quite", "basically", "actually", "literally"}

TONE_ADVERBS: dict[str, str] = {
    "nostalgia": "softly",
    "horror": "grimly",
    "romantic": "warmly",
    "academic": "systematically",
    "joyful": "brightly",
    "melancholic": "quietly",
    "hopeful": "steadily",
    "mysterious": "subtly",
    "formal": "carefully",
    "neutral": "clearly",
}


def _get_spacy():
    global _NLP
    if _NLP is not None:
        return _NLP
    if spacy is None:
        return None
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        _NLP = None
    return _NLP


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _match_leading_case(original: str, updated: str) -> str:
    if not original or not updated:
        return updated
    if original[0].isupper() and updated[0].islower():
        return updated[0].upper() + updated[1:]
    return updated


def _apply_common_replacements(text: str) -> str:
    updated = text
    for src, dst in COMMON_REPLACEMENTS:
        updated = re.sub(rf"\b{re.escape(src)}\b", dst, updated, flags=re.IGNORECASE)
    return _normalize_spacing(updated)


def _remove_fillers(text: str) -> str:
    nlp = _get_spacy()
    if nlp is None:
        pattern = r"\b(" + "|".join(sorted(FILLER_WORDS)) + r")\b"
        return _normalize_spacing(re.sub(pattern, "", text, flags=re.IGNORECASE))
    doc = nlp(text)
    parts: list[str] = []
    for token in doc:
        if token.lower_ in FILLER_WORDS and token.pos_ in {"ADV", "PART"}:
            continue
        parts.append(token.text_with_ws)
    return _normalize_spacing("".join(parts))


def _is_complete_by_dependencies(sentence: str) -> bool:
    nlp = _get_spacy()
    if nlp is None:
        return False
    doc = nlp(sentence)
    has_verb = any(token.pos_ in {"VERB", "AUX"} for token in doc)
    has_subject = any(token.dep_ in {"nsubj", "nsubjpass"} for token in doc)
    return has_verb and has_subject


def is_sentence_complete(sentence: str, blank_present: bool) -> bool:
    if blank_present:
        return False
    tokens = [t for t in sentence.split() if t]
    if len(tokens) < 4:
        return False
    if sentence.strip().endswith((".", "!", "?")):
        return True
    return _is_complete_by_dependencies(sentence)


def _ensure_terminal_punct(text: str, original: str) -> str:
    if not text:
        return text
    if original.strip().endswith((".", "!", "?")):
        return text if text.endswith((".", "!", "?")) else f"{text}."
    return text


def _are_synonyms(base: str, candidate: str) -> bool:
    wn = get_wordnet()
    if wn is None:
        return False
    for synset in wn.synsets(base):
        for lemma in synset.lemma_names():
            cleaned = lemma.replace("_", " ").lower()
            if cleaned == candidate:
                return True
    return False


def _replace_with_suggestion(text: str, suggestion: str) -> str:
    if not suggestion:
        return text
    nlp = _get_spacy()
    if nlp is None:
        return text
    doc = nlp(text)
    suggestion_doc = nlp(suggestion)
    if not suggestion_doc:
        return text
    suggestion_pos = suggestion_doc[0].pos_
    if suggestion_pos not in {"ADJ", "ADV"}:
        return text

    suggestion_lower = suggestion.lower()
    for token in doc:
        if token.pos_ != suggestion_pos or not token.is_alpha:
            continue
        if token.lower_ == suggestion_lower:
            return text
        if not _are_synonyms(token.lemma_.lower(), suggestion_lower):
            continue
        replacement = suggestion_lower.capitalize() if token.text[0].isupper() else suggestion_lower
        parts: list[str] = []
        for idx, current in enumerate(doc):
            if idx == token.i:
                parts.append(replacement + current.whitespace_)
            else:
                parts.append(current.text_with_ws)
        return _normalize_spacing("".join(parts))
    return text


def _inject_tone_adverb(text: str, context: str) -> str:
    nlp = _get_spacy()
    if nlp is None:
        return text
    tone_word = TONE_ADVERBS.get((context or "").lower())
    if not tone_word:
        return text
    doc = nlp(text)
    for token in doc:
        if token.pos_ in {"VERB", "AUX"} and token.i > 0:
            parts: list[str] = []
            for idx, current in enumerate(doc):
                if idx == token.i:
                    parts.append(f"{tone_word} {current.text_with_ws}")
                else:
                    parts.append(current.text_with_ws)
            return _normalize_spacing("".join(parts))
    return text


def rewrite_sentence(
    sentence: str,
    context: str,
    mode: str,
    blank_present: bool,
    allow_rewrite: bool,
    suggestions: Iterable[str] | None = None,
) -> str:
    if not sentence or not allow_rewrite:
        return ""
    if not is_sentence_complete(sentence, blank_present):
        return ""

    base = _normalize_spacing(sentence)
    base = _apply_common_replacements(base)

    if mode == "write":
        rewritten = base
    elif mode == "edit":
        rewritten = _remove_fillers(base)
    elif mode == "rewrite":
        rewritten = _remove_fillers(base)
        rewritten = _inject_tone_adverb(rewritten, context)
        if suggestions:
            rewritten = _replace_with_suggestion(rewritten, next(iter(suggestions), ""))
    else:
        rewritten = base

    if estimate_semantic_drift(base, rewritten) > 0.52:
        rewritten = base

    rewritten = _ensure_terminal_punct(rewritten, sentence)
    rewritten = _match_leading_case(sentence, rewritten)
    return rewritten


def rewrite_variants(
    sentence: str,
    context: str,
    mode: str,
    blank_present: bool,
    allow_rewrite: bool,
    suggestions: Iterable[str] | None = None,
    max_variants: int = 3,
) -> list[str]:
    base = rewrite_sentence(
        sentence=sentence,
        context=context,
        mode=mode,
        blank_present=blank_present,
        allow_rewrite=allow_rewrite,
        suggestions=suggestions,
    )
    if not base:
        return []

    variants = [base]
    if suggestions and mode == "rewrite":
        for suggestion in suggestions:
            variant = _replace_with_suggestion(base, suggestion)
            if not variant or variant in variants:
                continue
            if estimate_semantic_drift(base, variant) <= 0.52:
                variants.append(variant)
            if len(variants) >= max_variants:
                break
    return variants


__all__ = ["rewrite_sentence", "rewrite_variants", "is_sentence_complete"]
