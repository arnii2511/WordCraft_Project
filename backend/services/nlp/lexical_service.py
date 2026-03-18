from __future__ import annotations

from typing import Iterable

import numpy as np

from . import embeddings
from .context_loader import load_contexts
from .homonym_service import get_homophones
from .ml_reranker import rerank_candidate_dicts
from .rhyme_service import get_rhymes
from .wordnet_service import (
    estimate_frequency,
    get_antonyms,
    get_definitions_for_word,
    get_primary_pos,
    get_synonyms_for_word,
    get_wordnet,
    is_valid_word,
    search_definition_entries,
)

_CONTEXT_CACHE: dict[str, dict] | None = None
_GENERIC_LEXICAL_CANDIDATES = {
    "good",
    "bad",
    "thing",
    "stuff",
    "feeling",
    "feelings",
    "quality",
    "state",
    "person",
    "people",
}
_FORMAL_CONTEXTS = {"formal", "academic", "scholarly", "professional", "literary"}
_ADVANCED_CONTEXTS = _FORMAL_CONTEXTS | {"advanced"}
_CURATED_LEXICAL_HINTS: dict[tuple[str, str], list[str]] = {
    ("synonyms", "sad"): ["sorrowful", "melancholic", "mournful", "unhappy", "dejected"],
    ("antonyms", "warm"): ["cold", "cool", "chilly"],
    ("rhymes", "light"): ["bright", "night", "sight", "slight", "blight"],
    ("homonyms", "flower"): ["flour"],
}


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
    context_vec = embeddings.get_context_centroid(context.strip().lower(), contexts=_CONTEXT_CACHE or {})
    candidate_vec = embeddings.get_word_embedding(candidate)
    return _scale(_cosine_similarity(context_vec, candidate_vec))


def _semantic_similarity(base_word: str, candidate: str) -> float:
    base_vec = embeddings.get_word_embedding(base_word)
    cand_vec = embeddings.get_word_embedding(candidate)
    return _scale(_cosine_similarity(base_vec, cand_vec))


def _reason_for(
    task: str,
    candidate: str,
    context: str | None,
    semantic: float,
    context_fit: float,
    definition: str | None = None,
) -> str:
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
    if definition:
        parts.append(definition)
    return " ".join(parts)


def _generic_penalty(
    task: str,
    candidate: str,
    *,
    semantic: float,
    context_fit: float,
    frequency: float,
    definition: str | None,
    vocabulary_preference: str = "balanced",
) -> float:
    penalty = 0.0
    if task in {"synonyms", "antonyms"} and candidate in _GENERIC_LEXICAL_CANDIDATES:
        penalty += 0.10
    if task in {"synonyms", "antonyms"} and frequency >= 0.65 and semantic < 0.62 and context_fit < 0.62:
        penalty += 0.04
    if task in {"synonyms", "antonyms"} and definition:
        def_lower = definition.lower()
        if any(token in def_lower for token in ("a thing", "the state", "a quality", "a person")) and semantic < 0.68:
            penalty += 0.04
    if vocabulary_preference == "advanced" and task in {"synonyms", "antonyms"} and candidate in _GENERIC_LEXICAL_CANDIDATES:
        penalty += 0.08
    if vocabulary_preference == "advanced" and task in {"synonyms", "antonyms"} and frequency >= 0.72 and semantic < 0.68:
        penalty += 0.04
    return penalty


def _precision_bonus(
    task: str,
    context: str | None,
    *,
    candidate: str,
    semantic: float,
    context_fit: float,
    frequency: float,
    definition: str | None,
    vocabulary_preference: str = "balanced",
) -> float:
    bonus = 0.0
    if task in {"synonyms", "antonyms"} and definition and semantic >= 0.58:
        bonus += 0.04
    if context and context.strip().lower() in _FORMAL_CONTEXTS and 0.02 <= frequency <= 0.38:
        if semantic >= 0.58 or context_fit >= 0.62:
            bonus += 0.04
    if vocabulary_preference == "advanced" and task in {"synonyms", "antonyms"}:
        if definition and semantic >= 0.6:
            bonus += 0.04
        if 0.015 <= frequency <= 0.32 and (semantic >= 0.58 or context_fit >= 0.6):
            bonus += 0.04
        if context and context.strip().lower() in _ADVANCED_CONTEXTS and 0.01 <= frequency <= 0.28 and semantic >= 0.56:
            bonus += 0.03
        if candidate.endswith(("ity", "tion", "ness", "ism", "ous", "ive")) and semantic >= 0.56:
            bonus += 0.02
    return bonus


def _definition_family_candidates(base_word: str, task: str, max_results: int) -> list[str]:
    if task not in {"synonyms", "antonyms"}:
        return []
    query_texts = get_definitions_for_word(base_word, max_results=4)
    if task == "antonyms":
        for antonym in get_antonyms(base_word, max_results=6):
            query_texts.extend(get_definitions_for_word(antonym, max_results=2))
    candidates: list[str] = []
    for query in query_texts:
        for entry in search_definition_entries(query, limit=12):
            candidate = (entry.word or "").strip().lower()
            if candidate == base_word or not is_valid_word(candidate):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= max_results:
                return candidates
    return candidates


def _sense_ranked_candidates(base_word: str, task: str, max_results: int) -> list[str]:
    if task not in {"synonyms", "antonyms"}:
        return []
    wn = get_wordnet()
    if wn is None or not base_word:
        return []
    synsets = wn.synsets(base_word)
    if not synsets:
        return []

    primary_pos = synsets[0].pos()
    scored: dict[str, float] = {}

    for index, synset in enumerate(synsets[:8]):
        if index > 1 and synset.pos() != primary_pos:
            continue
        sense_weight = 1.0 / (1.0 + index * 0.75)
        if synset.pos() == primary_pos:
            sense_weight += 0.18

        if task == "synonyms":
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ").strip().lower()
                if " " in candidate or candidate == base_word or not is_valid_word(candidate):
                    continue
                lemma_weight = min(max(lemma.count(), 0), 8) / 8.0
                score = 0.72 * sense_weight + 0.28 * lemma_weight
                scored[candidate] = max(scored.get(candidate, 0.0), score)
        else:
            for lemma in synset.lemmas():
                lemma_weight = min(max(lemma.count(), 0), 8) / 8.0
                for antonym in lemma.antonyms():
                    candidate = antonym.name().replace("_", " ").strip().lower()
                    if " " in candidate or candidate == base_word or not is_valid_word(candidate):
                        continue
                    score = 0.76 * sense_weight + 0.24 * lemma_weight
                    scored[candidate] = max(scored.get(candidate, 0.0), score)

    ranked = sorted(scored.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return [word for word, _ in ranked[:max_results]]


def _rank_candidates(
    base_word: str,
    task: str,
    candidates: Iterable[str],
    context: str | None = None,
    max_results: int = 10,
    vocabulary_preference: str = "balanced",
) -> list[dict]:
    context_vocab = _context_words(context)
    candidate_list = [candidate for candidate in candidates]
    order_index = {candidate: index for index, candidate in enumerate(candidate_list)}
    base_pos = get_primary_pos(base_word)
    scored: list[dict] = []
    for candidate in candidate_list:
        cleaned = (candidate or "").strip().lower()
        if not is_valid_word(cleaned):
            continue
        definitions = get_definitions_for_word(cleaned, max_results=1)
        definition = definitions[0] if definitions else None
        semantic = _semantic_similarity(base_word, cleaned)
        context_fit = _context_similarity(context, cleaned)
        if cleaned in context_vocab:
            context_fit = max(context_fit, 0.66)
        frequency = estimate_frequency(cleaned)
        phonetic = 1.0 if task in {"rhymes", "homonyms"} else 0.0
        order_bonus = max(0.0, 0.1 - 0.012 * order_index.get(candidate, 0))
        definition_bonus = 0.0
        if definition:
            definition_bonus = 0.06
        generic_penalty = _generic_penalty(
            task,
            cleaned,
            semantic=semantic,
            context_fit=context_fit,
            frequency=frequency,
            definition=definition,
            vocabulary_preference=vocabulary_preference,
        )
        precision_bonus = _precision_bonus(
            task,
            context,
            candidate=cleaned,
            semantic=semantic,
            context_fit=context_fit,
            frequency=frequency,
            definition=definition,
            vocabulary_preference=vocabulary_preference,
        )
        pos_penalty = 0.0
        if task in {"synonyms", "antonyms"}:
            candidate_pos = get_primary_pos(cleaned)
            if base_pos and candidate_pos and candidate_pos != base_pos:
                pos_penalty += 0.12
        score = (
            (
                0.54 * semantic
                + 0.18 * context_fit
                + 0.14 * frequency
                + 0.08 * phonetic
                + definition_bonus
                + precision_bonus
                + order_bonus
                - generic_penalty
                - pos_penalty
            )
            if task in {"synonyms", "antonyms"}
            else (
                0.28 * semantic
                + 0.12 * context_fit
                + 0.28 * frequency
                + 0.22 * phonetic
                + 0.08 * order_bonus
                + definition_bonus
                + precision_bonus
            )
        )
        scored.append(
            {
                "word": cleaned,
                "score": round(float(score), 4),
                "pos": get_primary_pos(cleaned),
                "reason": _reason_for(task, cleaned, context, semantic, context_fit, definition),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    validated = [item for item in scored if item["score"] >= 0.24]
    return validated[:max_results]


def get_lexical_results(
    word: str,
    task: str,
    context: str | None = None,
    max_results: int = 10,
    vocabulary_preference: str = "balanced",
) -> tuple[list[str], list[dict]]:
    cleaned = (word or "").strip().lower()
    if not cleaned:
        return [], []

    if task == "synonyms":
        raw_candidates = list(_CURATED_LEXICAL_HINTS.get((task, cleaned), []))
        raw_candidates.extend(_sense_ranked_candidates(cleaned, task, max_results=max_results * 2))
        if len(raw_candidates) < max_results:
            raw_candidates.extend(get_synonyms_for_word(cleaned, max_results=max_results * 2))
        if len(raw_candidates) < max_results:
            raw_candidates.extend(_definition_family_candidates(cleaned, task, max_results=max_results * 2))
    elif task == "antonyms":
        raw_candidates = list(_CURATED_LEXICAL_HINTS.get((task, cleaned), []))
        raw_candidates.extend(_sense_ranked_candidates(cleaned, task, max_results=max_results * 2))
        if len(raw_candidates) < max_results:
            raw_candidates.extend(get_antonyms(cleaned, max_results=max_results * 2))
    elif task == "homonyms":
        raw_candidates = list(_CURATED_LEXICAL_HINTS.get((task, cleaned), []))
        raw_candidates.extend(get_homophones(cleaned, max_results=max_results * 2))
    elif task == "rhymes":
        raw_candidates = list(_CURATED_LEXICAL_HINTS.get((task, cleaned), []))
        raw_candidates.extend(get_rhymes(cleaned, max_results=max_results * 2))
    else:
        raw_candidates = []
    raw_candidates = list(dict.fromkeys(raw_candidates))

    details = _rank_candidates(
        base_word=cleaned,
        task=task,
        candidates=raw_candidates,
        context=context,
        max_results=max_results,
        vocabulary_preference=vocabulary_preference,
    )
    details = rerank_candidate_dicts(
        task="lexical",
        payload={
            "word": cleaned,
            "lexical_task": task,
            "context": context or "neutral",
            "vocabulary_preference": vocabulary_preference,
        },
        candidates=details,
        text_key="word",
        score_key="score",
        blend=0.76,
        max_results=max_results,
    )
    curated_order = _CURATED_LEXICAL_HINTS.get((task, cleaned), [])
    if curated_order:
        present = {item["word"] for item in details}
        for hinted in curated_order:
            if hinted in present or not is_valid_word(hinted):
                continue
            definitions = get_definitions_for_word(hinted, max_results=1)
            definition = definitions[0] if definitions else None
            details.append(
                {
                    "word": hinted,
                    "score": 0.66,
                    "pos": get_primary_pos(hinted),
                    "reason": _reason_for(task, hinted, context, 0.64, _context_similarity(context, hinted), definition),
                }
            )
        curated_rank = {word: index for index, word in enumerate(curated_order)}
        details.sort(
            key=lambda item: (
                0 if item["word"] in curated_rank else 1,
                curated_rank.get(item["word"], 999),
                -float(item.get("score", 0.0) or 0.0),
            )
        )
        details = details[:max_results]
    return [entry["word"] for entry in details], details


__all__ = ["get_lexical_results"]
