from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .blank_detector import extract_focus_word, preprocess_text
from .conceptnet_service import get_related_words
from .emotion_service import emotion_score
from .ranker import BLANK_TOKEN, infer_expected_pos
from .wordnet_service import (
    get_pos_tags,
    get_derivational_forms,
    get_synonyms,
    get_wordnet,
    is_valid_word,
)

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

_SPACY_NLP = None
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")

DRAFT_VERBS = {
    "consider",
    "explore",
    "remember",
    "reflect",
    "discover",
    "imagine",
    "reveal",
    "become",
    "feel",
    "linger",
}
_IRREGULAR_ADV = {"well", "fast", "hard", "late", "early", "straight", "right", "near"}


@dataclass
class IntentDecision:
    intent: str
    cleaned_text: str
    blank_present: bool
    selection_text: str
    focus_terms: list[str]
    expected_pos: set[str] | None
    strict_pos: bool
    focus_window: str


@dataclass
class PipelineResult:
    decision: IntentDecision
    candidates: set[str]
    source_map: dict[str, set[str]]
    emotion_scores: dict[str, float]
    context_description: str


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


def _tokenize(text: str) -> list[str]:
    nlp = _get_spacy()
    if nlp is None:
        return [item.lower() for item in _TOKEN_RE.findall(text or "")]
    doc = nlp(text or "")
    tokens: list[str] = []
    for token in doc:
        if token.is_stop or not token.is_alpha:
            continue
        lemma = token.lemma_.lower().strip()
        if lemma:
            tokens.append(lemma)
    return tokens


def _extract_content_terms(text: str, limit: int = 6) -> list[str]:
    nlp = _get_spacy()
    if nlp is None:
        tokens = _tokenize(text)
        return list(dict.fromkeys(tokens))[:limit]
    doc = nlp(text or "")
    words: list[str] = []
    for token in doc:
        if token.is_stop or not token.is_alpha:
            continue
        if token.pos_ not in {"NOUN", "VERB", "ADJ", "ADV"}:
            continue
        lemma = token.lemma_.lower().strip()
        if lemma and lemma not in words:
            words.append(lemma)
        if len(words) >= limit:
            break
    return words


def _add_candidate(source_map: dict[str, set[str]], word: str, source: str) -> None:
    cleaned = (word or "").strip().lower()
    if not cleaned:
        return
    if cleaned == BLANK_TOKEN.lower():
        return
    if " " in cleaned:
        return
    if not is_valid_word(cleaned):
        return
    source_map.setdefault(cleaned, set()).add(source)


def _get_selection_text(selection: Any | None) -> str:
    if selection is None:
        return ""
    if isinstance(selection, dict):
        return (selection.get("text") or "").strip()
    return (getattr(selection, "text", "") or "").strip()


def _infer_selection_pos(selection_text: str) -> set[str] | None:
    tokens = _tokenize(selection_text)
    if not tokens:
        return None
    last = tokens[-1]
    tags = get_pos_tags(last)
    if tags:
        return tags
    nlp = _get_spacy()
    if nlp is None:
        return None
    doc = nlp(selection_text)
    if not doc:
        return None
    mapped = doc[-1].pos_.upper()
    if mapped in {"NOUN", "VERB", "ADJ", "ADV"}:
        return {mapped}
    return None


def detect_intent(
    text: str,
    selection: Any | None = None,
) -> IntentDecision:
    processed = preprocess_text(text or "")
    cleaned_text = processed.cleaned_text
    selection_text = _get_selection_text(selection)
    expected_pos: set[str] | None = None
    strict_pos = False
    focus_window = cleaned_text

    if selection_text:
        focus_terms = _extract_content_terms(selection_text, limit=4)
        if not focus_terms:
            focus_terms = _tokenize(selection_text)[:3]
        expected_pos = _infer_selection_pos(selection_text)
        strict_pos = expected_pos is not None and len(expected_pos) == 1
        return IntentDecision(
            intent="selection",
            cleaned_text=cleaned_text,
            blank_present=processed.blank_present,
            selection_text=selection_text,
            focus_terms=focus_terms,
            expected_pos=expected_pos,
            strict_pos=strict_pos,
            focus_window=selection_text,
        )

    if processed.blank_present:
        focus_word = extract_focus_word(processed.tokens, processed.blank_index) or ""
        focus_terms = _extract_content_terms(cleaned_text, limit=5)
        if focus_word:
            focus_terms = [focus_word.lower()] + [w for w in focus_terms if w != focus_word.lower()]
        pos_text = cleaned_text.replace(BLANK_TOKEN, "BLANKTOKEN")
        expected_pos = infer_expected_pos(pos_text)
        strict_pos = expected_pos is not None
        return IntentDecision(
            intent="blank",
            cleaned_text=cleaned_text,
            blank_present=True,
            selection_text="",
            focus_terms=focus_terms,
            expected_pos=expected_pos,
            strict_pos=strict_pos,
            focus_window=cleaned_text,
        )

    focus_terms = _extract_content_terms(cleaned_text, limit=6)
    if not focus_terms:
        focus_terms = _tokenize(cleaned_text)[:6]
    return IntentDecision(
        intent="sentence",
        cleaned_text=cleaned_text,
        blank_present=False,
        selection_text="",
        focus_terms=focus_terms,
        expected_pos=None,
        strict_pos=False,
        focus_window=cleaned_text,
    )


def _expand_wordnet(source_map: dict[str, set[str]], terms: list[str], mode: str) -> None:
    for term in terms[:6]:
        for synonym in get_synonyms([term], max_synonyms_per_word=8):
            _add_candidate(source_map, synonym, "wordnet")
            for deriv in get_derivational_forms(synonym, max_results=4):
                _add_candidate(source_map, deriv, "derivational")
        if mode in {"write", "rewrite"}:
            for deriv in get_derivational_forms(term, max_results=8):
                _add_candidate(source_map, deriv, "derivational")


def _expand_conceptnet(source_map: dict[str, set[str]], terms: list[str], mode: str) -> None:
    if mode not in {"write", "rewrite", "transform"}:
        return
    for term in terms[:3]:
        for related in get_related_words(term, max_terms=10):
            _add_candidate(source_map, related, "conceptnet")


def _add_slot_fallbacks(
    source_map: dict[str, set[str]],
    expected_pos: set[str] | None,
    context_words: list[str],
) -> None:
    if not expected_pos:
        return
    if expected_pos == {"ADV"}:
        for word in context_words[:24]:
            if word in _IRREGULAR_ADV:
                _add_candidate(source_map, word, "slot")
                continue
            if word.endswith("y") and len(word) > 3:
                _add_candidate(source_map, f"{word[:-1]}ily", "slot")
            _add_candidate(source_map, f"{word}ly", "slot")
    elif expected_pos == {"VERB"}:
        for word in DRAFT_VERBS:
            _add_candidate(source_map, word, "slot")
    elif expected_pos == {"ADJ"}:
        for word in context_words[:20]:
            _add_candidate(source_map, word, "slot")


def build_pipeline(
    text: str,
    context_key: str,
    mode: str,
    contexts: dict[str, dict],
    selection: Any | None = None,
) -> PipelineResult:
    decision = detect_intent(text, selection=selection)
    context_payload = contexts.get(context_key, {})
    context_words = context_payload.get("words", [])
    context_description = context_payload.get("description", "the selected tone")
    source_map: dict[str, set[str]] = {}

    for word in context_words:
        _add_candidate(source_map, word, "context")

    if decision.focus_terms:
        _expand_wordnet(source_map, decision.focus_terms, mode=mode)
        _expand_conceptnet(source_map, decision.focus_terms, mode=mode)

    if decision.intent == "blank":
        for verb in DRAFT_VERBS:
            _add_candidate(source_map, verb, "pattern")
        _add_slot_fallbacks(source_map, decision.expected_pos, context_words)
    if decision.intent == "selection":
        for term in decision.focus_terms[:3]:
            _add_candidate(source_map, term, "selection")

    if mode == "edit":
        neutral_words = contexts.get("neutral", {}).get("words", [])
        for word in neutral_words:
            _add_candidate(source_map, word, "neutral")

    if not source_map and context_words:
        for word in context_words:
            _add_candidate(source_map, word, "fallback")

    candidates = set(source_map.keys())
    if len(candidates) > 320:
        candidates = set(sorted(candidates)[:320])
    emotion_scores = {word: emotion_score(word, context_key) for word in candidates}

    return PipelineResult(
        decision=decision,
        candidates=candidates,
        source_map=source_map,
        emotion_scores=emotion_scores,
        context_description=context_description,
    )


def estimate_semantic_drift(original: str, rewritten: str) -> float:
    wn = get_wordnet()
    if wn is None:
        return 0.0
    before = set(_extract_content_terms(original, limit=10))
    after = set(_extract_content_terms(rewritten, limit=10))
    if not before or not after:
        return 0.0
    overlap = len(before & after) / max(1, len(before))
    return 1.0 - overlap
