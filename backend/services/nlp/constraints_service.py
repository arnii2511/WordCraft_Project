from __future__ import annotations

import re

import numpy as np

from . import embeddings
from .context_loader import load_contexts
from .wordnet_service import (
    estimate_frequency,
    get_derivational_forms,
    get_synonyms_for_word,
    get_wordnet,
    is_valid_word,
)

try:
    import pronouncing
except ImportError:  # pragma: no cover
    pronouncing = None

_CONTEXT_CACHE: dict[str, dict] | None = None


def _clean_word(word: str) -> str:
    return (word or "").replace("_", " ").strip().lower()


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _scale(similarity: float) -> float:
    return (similarity + 1.0) / 2.0


def _get_context_words(context: str | None) -> set[str]:
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


def _collect_rhymes(word: str) -> list[str]:
    if not word or pronouncing is None:
        return []
    rhymes = pronouncing.rhymes(word)
    cleaned = []
    for rhyme in rhymes:
        candidate = _clean_word(rhyme)
        if " " in candidate:
            continue
        if candidate and is_valid_word(candidate) and candidate not in cleaned:
            cleaned.append(candidate)
    return cleaned


def _collect_meaning(word: str, relation: str) -> set[str]:
    wn = get_wordnet()
    if wn is None or not word:
        return set()
    results: set[str] = set()
    if relation == "synonym":
        if is_valid_word(word):
            results.add(word)
        for synonym in get_synonyms_for_word(word, max_results=30):
            if is_valid_word(synonym):
                results.add(synonym)
        for deriv in get_derivational_forms(word, max_results=20):
            if is_valid_word(deriv):
                results.add(deriv)
    else:
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    cleaned = _clean_word(antonym.name())
                    if " " in cleaned:
                        continue
                    if cleaned and is_valid_word(cleaned):
                        results.add(cleaned)
    return results


def _semantic_similarity(word: str, target: str) -> float:
    wv = embeddings.get_word_embedding(word)
    tv = embeddings.get_word_embedding(target)
    return _scale(_cosine_similarity(wv, tv))


def _context_similarity(word: str, context: str | None) -> float:
    if not context:
        return 0.0
    cv = embeddings.get_context_centroid(context.strip().lower())
    wv = embeddings.get_word_embedding(word)
    return _scale(_cosine_similarity(cv, wv))


def _rhyme_quality(word: str, rhyme_with: str) -> float:
    if pronouncing is None:
        return 1.0 if word == rhyme_with else 0.0
    phones_a = pronouncing.phones_for_word(word)
    phones_b = pronouncing.phones_for_word(rhyme_with)
    if not phones_a or not phones_b:
        return 1.0 if word in _collect_rhymes(rhyme_with) else 0.0
    tail_a = re.sub(r"\d", "", phones_a[0].split()[-1])
    tail_b = re.sub(r"\d", "", phones_b[0].split()[-1])
    if tail_a == tail_b:
        return 1.0
    return 0.0


def _build_reason(
    rhyme_with: str,
    meaning_of: str,
    relation: str,
    rhyme_match: bool,
    relation_match: bool,
    semantic: float,
    context_hit: bool,
    best_effort: bool,
) -> str:
    parts: list[str] = []
    if rhyme_match:
        parts.append(f"Rhymes with '{rhyme_with}'.")
    elif best_effort:
        parts.append(f"Closest rhyme candidate for '{rhyme_with}'.")

    rel_label = "synonym" if relation == "synonym" else "antonym"
    if relation_match:
        parts.append(f"{rel_label.capitalize()} of '{meaning_of}'.")
    elif semantic >= 0.58:
        parts.append(f"Near {rel_label} meaning to '{meaning_of}'.")

    if context_hit:
        parts.append("Tone-aligned with selected context.")
    return " ".join(parts) if parts else "Best available match for the provided constraints."


def get_constraint_matches(
    rhyme_with: str,
    relation: str,
    meaning_of: str,
    context: str | None = None,
    limit: int = 10,
) -> tuple[list[dict], str | None]:
    rhyme_base = _clean_word(rhyme_with)
    meaning_base = _clean_word(meaning_of)

    rhyme_candidates = _collect_rhymes(rhyme_base)
    meaning_candidates = _collect_meaning(meaning_base, relation)
    rhyme_set = set(rhyme_candidates)
    context_words = _get_context_words(context)

    exact_matches = list(rhyme_set & meaning_candidates)
    note: str | None = None
    best_effort = False

    if exact_matches:
        candidate_pool = exact_matches
    else:
        best_effort = True
        if rhyme_candidates:
            candidate_pool = list(dict.fromkeys(rhyme_candidates + list(meaning_candidates)))
            note = "No strict rhyme+meaning overlap. Showing best-effort ranked matches."
        else:
            candidate_pool = list(meaning_candidates)
            note = "No rhyme candidates found. Showing strongest meaning matches."

    if len(candidate_pool) > 260:
        candidate_pool = candidate_pool[:260]
    if not candidate_pool:
        return [], "No matches found for the provided constraints."

    results = []
    for candidate in candidate_pool:
        rhyme_match = candidate in rhyme_set
        relation_match = candidate in meaning_candidates
        rhyme_score = _rhyme_quality(candidate, rhyme_base) if rhyme_base else 0.0
        semantic = _semantic_similarity(candidate, meaning_base) if meaning_base else 0.0
        relation_score = 1.0 if relation_match else semantic
        context_score = _context_similarity(candidate, context)
        if candidate in context_words:
            context_score = max(context_score, 0.68)
        frequency = estimate_frequency(candidate)
        score = (
            0.40 * rhyme_score
            + 0.32 * relation_score
            + 0.16 * semantic
            + 0.07 * context_score
            + 0.05 * frequency
        )
        if rhyme_match and relation_match:
            score += 0.08
        results.append(
            {
                "word": candidate,
                "score": round(float(min(score, 0.99)), 4),
                "rhyme": rhyme_match,
                "relation_match": relation_match,
                "reason": _build_reason(
                    rhyme_with=rhyme_base,
                    meaning_of=meaning_base,
                    relation=relation,
                    rhyme_match=rhyme_match,
                    relation_match=relation_match,
                    semantic=semantic,
                    context_hit=candidate in context_words or context_score >= 0.62,
                    best_effort=best_effort,
                ),
            }
        )

    results.sort(key=lambda item: (item["score"], item["relation_match"], item["rhyme"]), reverse=True)
    capped = max(1, min(10, int(limit or 10)))
    return results[:capped], note


__all__ = ["get_constraint_matches"]
