from __future__ import annotations

from typing import Iterable

import numpy as np

from . import embeddings
from .context_loader import load_contexts
from .homonym_service import get_homophones
from .rhyme_service import get_rhymes
from .wordnet_service import (
    estimate_frequency,
    get_antonyms,
    get_primary_pos,
    get_synonyms_for_word,
    is_valid_word,
)

_CONTEXT_CACHE: dict[str, dict] | None = None


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _scale(similarity: float) -> float:
    return (similarity + 1.0) / 2.0


def _context_words(context: str | None) -> set[str]:
    global _CONTEXT_CACHE
    if not context:
        return set()
    if _CONTEXT_CACHE is None:
        try:
            _CONTEXT_CACHE = load_contexts()
        except Exception:
            _CONTEXT_CACHE = {}
    payload = _CONTEXT_CACHE.get(context.strip().lower())
    if not payload:
        return set()
    return set(payload.get("words", []))


def _context_similarity(context: str | None, candidate: str) -> float:
    if not context:
        return 0.0
    context_vec = embeddings.get_context_centroid(context.strip().lower())
    candidate_vec = embeddings.get_word_embedding(candidate)
    return _scale(_cosine_similarity(context_vec, candidate_vec))


def _semantic_similarity(base_word: str, candidate: str) -> float:
    base_vec = embeddings.get_word_embedding(base_word)
    cand_vec = embeddings.get_word_embedding(candidate)
    return _scale(_cosine_similarity(base_vec, cand_vec))


def _reason_for(task: str, candidate: str, context: str | None, semantic: float, context_fit: float) -> str:
    parts: list[str] = []
    if task == "synonyms":
        parts.append("WordNet synonym.")
    elif task == "antonyms":
        parts.append("WordNet antonym.")
    elif task == "rhymes":
        parts.append("Phonetic rhyme match.")
    elif task == "homonyms":
        parts.append("Pronunciation match.")

    if semantic >= 0.62:
        parts.append("Strong semantic fit.")
    elif semantic >= 0.54:
        parts.append("Good semantic match.")

    if context and context_fit >= 0.62:
        parts.append(f"Aligned with {context.lower()} tone.")
    return " ".join(parts)


def _rank_candidates(
    base_word: str,
    task: str,
    candidates: Iterable[str],
    context: str | None = None,
    max_results: int = 10,
) -> list[dict]:
    context_vocab = _context_words(context)
    scored: list[dict] = []
    for candidate in candidates:
        cleaned = (candidate or "").strip().lower()
        if not is_valid_word(cleaned):
            continue
        semantic = _semantic_similarity(base_word, cleaned)
        context_fit = _context_similarity(context, cleaned)
        if cleaned in context_vocab:
            context_fit = max(context_fit, 0.66)
        frequency = estimate_frequency(cleaned)
        phonetic = 1.0 if task in {"rhymes", "homonyms"} else 0.0
        score = 0.58 * semantic + 0.18 * context_fit + 0.16 * frequency + 0.08 * phonetic
        scored.append(
            {
                "word": cleaned,
                "score": round(float(score), 4),
                "pos": get_primary_pos(cleaned),
                "reason": _reason_for(task, cleaned, context, semantic, context_fit),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:max_results]


def get_lexical_results(
    word: str,
    task: str,
    context: str | None = None,
    max_results: int = 10,
) -> tuple[list[str], list[dict]]:
    cleaned = (word or "").strip().lower()
    if not cleaned:
        return [], []

    if task == "synonyms":
        raw_candidates = get_synonyms_for_word(cleaned, max_results=max_results * 2)
    elif task == "antonyms":
        raw_candidates = get_antonyms(cleaned, max_results=max_results * 2)
    elif task == "homonyms":
        raw_candidates = get_homophones(cleaned, max_results=max_results * 2)
    elif task == "rhymes":
        raw_candidates = get_rhymes(cleaned, max_results=max_results * 2)
    else:
        raw_candidates = []

    details = _rank_candidates(
        base_word=cleaned,
        task=task,
        candidates=raw_candidates,
        context=context,
        max_results=max_results,
    )
    return [entry["word"] for entry in details], details


__all__ = ["get_lexical_results"]
