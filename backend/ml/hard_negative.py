from __future__ import annotations

import re
from typing import Any

from backend.ml.scripts.common import build_input_text, normalize_text
from backend.services.nlp import embeddings
from backend.services.nlp.conceptnet_service import get_related_words
from backend.services.nlp.context_loader import load_contexts
from backend.services.nlp.wordnet_service import (
    get_antonyms,
    get_derivational_forms,
    get_pos_tags,
    get_synonyms_for_word,
    get_wordnet,
    is_valid_word,
)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_BLANK_RE = re.compile(r"_{3,}|\[blank\]|<blank>|\(blank\)", re.IGNORECASE)
_CONTEXT_CACHE: dict[str, dict] | None = None


def _tokens(text: str) -> list[str]:
    values: list[str] = []
    for token in _TOKEN_RE.findall(text or ""):
        cleaned = token.strip("-'").lower()
        if cleaned:
            values.append(cleaned)
    return values


def _context_words(context: str | None) -> set[str]:
    global _CONTEXT_CACHE
    if not context:
        return set()
    key = context.strip().lower()
    if not key:
        return set()
    if _CONTEXT_CACHE is None:
        try:
            _CONTEXT_CACHE = load_contexts()
        except Exception:
            _CONTEXT_CACHE = {}
    payload = _CONTEXT_CACHE.get(key)
    if not payload:
        return set()
    return set(payload.get("words", []))


def _infer_expected_pos(task: str, payload: dict[str, Any], positives: list[str]) -> set[str]:
    tags: set[str] = set()
    for positive in positives:
        tags |= get_pos_tags(positive)
    if tags:
        return tags

    if task != "suggest_blank":
        return set()

    sentence = str(payload.get("sentence", "") or "")
    lowered = sentence.lower()
    match = _BLANK_RE.search(lowered)
    if not match:
        return set()

    left = lowered[: match.start()]
    if re.search(r"\bto\s*$", left):
        return {"VERB"}
    if re.search(r"\b(is|are|was|were|be|been|feel|feels|seem|seems|look|looks)\s*$", left):
        return {"ADJ"}
    if re.search(r"\b\w+(ed|ing)\s*$", left):
        return {"ADV"}
    return set()


def _collect_terms(task: str, payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    if task in {"suggest_blank", "suggest_selection", "suggest_sentence", "rewrite"}:
        values.extend(_tokens(str(payload.get("sentence", "") or "")))
    elif task == "lexical":
        values.extend(_tokens(str(payload.get("word", "") or "")))
    elif task == "constraints":
        values.extend(_tokens(str(payload.get("rhyme_with", "") or "")))
        values.extend(_tokens(str(payload.get("meaning_of", "") or "")))
    elif task == "oneword":
        values.extend(_tokens(str(payload.get("query", "") or "")))
    else:
        values.extend(_tokens(build_input_text(task, payload)))
    return list(dict.fromkeys(values))


def _cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    denom = float((a @ a) ** 0.5 * (b @ b) ** 0.5)
    if denom <= 0.0:
        return 0.0
    return float((a @ b) / denom)


def _semantic_similarity(query_text: str, candidate: str) -> float:
    q_vec = embeddings.embed_sentence(query_text)
    c_vec = embeddings.get_word_embedding(candidate)
    return (_cosine(q_vec, c_vec) + 1.0) / 2.0


def _tone_similarity(context: str | None, candidate: str, context_words: set[str]) -> float:
    if candidate in context_words:
        return 1.0
    if not context:
        return 0.5
    c_vec = embeddings.get_context_centroid(context.strip().lower())
    w_vec = embeddings.get_word_embedding(candidate)
    return (_cosine(c_vec, w_vec) + 1.0) / 2.0


def generate_hard_negatives(
    *,
    task: str,
    payload: dict[str, Any],
    positives: list[str],
    existing_texts: set[str],
    max_items: int = 8,
) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []

    terms = _collect_terms(task, payload)
    if not terms and not positives:
        return []

    expected_pos = _infer_expected_pos(task, payload, positives)
    context = str(payload.get("context", "") or "").strip().lower()
    context_words = _context_words(context)
    query_text = build_input_text(task, payload)
    positives_norm = {normalize_text(value) for value in positives if normalize_text(value)}

    pool: set[str] = set()
    antonym_hint: set[str] = set()

    for term in terms[:12]:
        for synonym in get_synonyms_for_word(term, max_results=14):
            pool.add(normalize_text(synonym))
        for antonym in get_antonyms(term, max_results=12):
            normalized = normalize_text(antonym)
            if normalized:
                pool.add(normalized)
                antonym_hint.add(normalized)
        for derived in get_derivational_forms(term, max_results=10):
            pool.add(normalize_text(derived))
        for related in get_related_words(term, max_terms=10):
            pool.add(normalize_text(related))

    for positive in positives[:8]:
        for synonym in get_synonyms_for_word(positive, max_results=14):
            pool.add(normalize_text(synonym))
        for antonym in get_antonyms(positive, max_results=12):
            normalized = normalize_text(antonym)
            if normalized:
                pool.add(normalized)
                antonym_hint.add(normalized)
        for related in get_related_words(positive, max_terms=10):
            pool.add(normalize_text(related))

    # Light WordNet neighborhood expansion to get semantic-near terms.
    wn = get_wordnet()
    if wn is not None:
        for term in terms[:6]:
            for synset in wn.synsets(term)[:6]:
                for lemma in synset.lemma_names():
                    pool.add(normalize_text(lemma.replace("_", " ")))
                for hypernym in synset.hypernyms()[:2]:
                    for lemma in hypernym.lemma_names():
                        pool.add(normalize_text(lemma.replace("_", " ")))

    scored: list[tuple[float, str, str]] = []
    for candidate in pool:
        if not candidate or " " in candidate:
            continue
        if candidate in positives_norm or candidate in existing_texts:
            continue
        if not is_valid_word(candidate):
            continue

        if expected_pos:
            candidate_pos = get_pos_tags(candidate)
            if candidate_pos and not (candidate_pos & expected_pos):
                continue

        semantic = _semantic_similarity(query_text, candidate)
        tone_fit = _tone_similarity(context, candidate, context_words)
        if semantic < 0.44:
            continue

        tone_mismatch = 1.0 - tone_fit
        if tone_mismatch < 0.25 and candidate not in antonym_hint:
            continue

        score = (0.62 * semantic) + (0.28 * tone_mismatch) + (0.10 if candidate in antonym_hint else 0.0)
        if score < 0.48:
            continue

        reason = "Semantic near-miss with tone/context mismatch."
        if candidate in antonym_hint:
            reason = "Semantic near-miss (antonym/contrast) likely to confuse ranker."
        scored.append((score, candidate, reason))

    scored.sort(key=lambda item: item[0], reverse=True)
    output: list[dict[str, Any]] = []
    for score, candidate, reason in scored[: max_items]:
        output.append(
            {
                "text": candidate,
                "label": 0,
                "model_score": round(min(0.35, max(0.02, score * 0.35)), 4),
                "pos": None,
                "reason": reason,
                "source": "hard_negative",
            }
        )
    return output


__all__ = ["generate_hard_negatives"]
