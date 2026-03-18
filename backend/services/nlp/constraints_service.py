from __future__ import annotations

import os
import re

import numpy as np

from . import embeddings
from .conceptnet_service import get_related_words
from .context_loader import load_contexts
from .ml_reranker import rerank_candidate_dicts
from .wordnet_service import (
    get_antonyms,
    estimate_frequency,
    get_definitions_for_word,
    get_derivational_forms,
    get_synonyms_for_word,
    get_wordnet,
    is_valid_word,
    search_definition_entries,
)

try:
    import pronouncing
except ImportError:  # pragma: no cover
    pronouncing = None

_CONTEXT_CACHE: dict[str, dict] | None = None
_GENERIC_CONSTRAINT_WORDS = {
    "good",
    "bad",
    "thing",
    "stuff",
    "feeling",
    "feelings",
    "state",
    "quality",
    "person",
    "people",
}
_CANONICAL_ANTONYM_HINTS = {
    "warm": {"cold", "cool"},
    "soft": {"hard", "firm"},
    "bright": {"dark", "dim"},
    "sad": {"glad", "happy"},
}


def _enable_conceptnet_runtime() -> bool:
    value = os.getenv("WORDCRAFT_ENABLE_CONCEPTNET_RUNTIME", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        for antonym in _CANONICAL_ANTONYM_HINTS.get(word, set()):
            if is_valid_word(antonym):
                results.add(antonym)
        for antonym in get_antonyms(word, max_results=30):
            if is_valid_word(antonym):
                results.add(antonym)
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    cleaned = _clean_word(antonym.name())
                    if " " in cleaned:
                        continue
                    if cleaned and is_valid_word(cleaned):
                        results.add(cleaned)
    return results


def _collect_semantic_expansion(base_word: str, relation: str, max_terms: int = 120) -> set[str]:
    if not base_word:
        return set()
    expanded: set[str] = set()
    frontier: list[str] = [base_word]
    seen: set[str] = set()

    while frontier and len(expanded) < max_terms:
        term = _clean_word(frontier.pop(0))
        if not term or term in seen:
            continue
        seen.add(term)
        if is_valid_word(term):
            expanded.add(term)

        if relation == "synonym":
            for synonym in get_synonyms_for_word(term, max_results=20):
                cleaned = _clean_word(synonym)
                if cleaned and cleaned not in seen:
                    frontier.append(cleaned)
        else:
            for antonym in _collect_meaning(term, "antonym"):
                cleaned = _clean_word(antonym)
                if cleaned and cleaned not in seen:
                    frontier.append(cleaned)

        for deriv in get_derivational_forms(term, max_results=12):
            cleaned = _clean_word(deriv)
            if cleaned and cleaned not in seen:
                frontier.append(cleaned)

        if _enable_conceptnet_runtime():
            for related in get_related_words(term, max_terms=10):
                cleaned = _clean_word(related)
                if cleaned and cleaned not in seen and " " not in cleaned and is_valid_word(cleaned):
                    frontier.append(cleaned)

        if len(expanded) >= max_terms:
            break

    return set(list(expanded)[:max_terms])


def _collect_rhyme_expansion(seed_terms: set[str], max_terms: int = 200) -> set[str]:
    results: set[str] = set()
    for term in list(seed_terms)[:40]:
        for rhyme in _collect_rhymes(term):
            results.add(rhyme)
            if len(results) >= max_terms:
                return results
    return results


def _collect_definition_family(base_word: str, relation: str, max_terms: int = 80) -> set[str]:
    if not base_word:
        return set()
    seed_queries: list[str] = []
    seed_queries.extend(get_definitions_for_word(base_word, max_results=4))

    if relation == "antonym":
        antonym_terms = list(_collect_meaning(base_word, "antonym"))[:10]
        for term in antonym_terms:
            seed_queries.extend(get_definitions_for_word(term, max_results=2))

    collected: list[str] = []
    for query in seed_queries:
        for entry in search_definition_entries(query, limit=18):
            candidate = _clean_word(entry.word)
            if candidate == base_word or not is_valid_word(candidate):
                continue
            if candidate not in collected:
                collected.append(candidate)
            if len(collected) >= max_terms:
                return set(collected)
    return set(collected)


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
    near_relation_match: bool,
    semantic: float,
    context_hit: bool,
    fallback_stage: str | None,
) -> str:
    parts: list[str] = []
    if rhyme_match:
        parts.append(f"Rhymes with '{rhyme_with}'.")
    elif fallback_stage:
        parts.append(f"Closest rhyme candidate for '{rhyme_with}'.")

    rel_label = "synonym" if relation == "synonym" else "antonym"
    if relation_match:
        if relation == "synonym":
            parts.append(f"Synonym of '{meaning_of}'.")
        else:
            parts.append(f"Opposes the meaning of '{meaning_of}'.")
    elif near_relation_match:
        if relation == "synonym":
            parts.append(f"From the synonym family of '{meaning_of}'.")
        else:
            parts.append(f"From the antonym family of '{meaning_of}'.")
    elif semantic >= 0.58:
        if relation == "synonym":
            parts.append(f"Near synonym meaning to '{meaning_of}'.")
        else:
            parts.append(f"Near opposite meaning to '{meaning_of}'.")
    elif relation == "antonym":
        parts.append(f"Ranked against antonym constraint for '{meaning_of}'.")
    else:
        parts.append(f"Ranked against synonym constraint for '{meaning_of}'.")

    if context_hit:
        parts.append("Tone-aligned with selected context.")
    return " ".join(parts) if parts else "Best available match for the provided constraints."


def _quality_bucket(item: dict, fallback_stage: str | None) -> int:
    rhyme_match = bool(item.get("rhyme"))
    relation_match = bool(item.get("relation_match"))
    near_relation_match = bool(item.get("near_relation_match"))
    semantic = float(item.get("_semantic", 0.0) or 0.0)

    if rhyme_match and relation_match:
        return 5
    if rhyme_match and near_relation_match:
        return 4
    if fallback_stage == "meaning_only":
        if relation_match:
            return 4
        if near_relation_match:
            return 3
        if semantic >= 0.64:
            return 2
        return 0
    if rhyme_match and semantic >= 0.58:
        return 3
    if relation_match:
        return 2
    if near_relation_match:
        return 1
    return 0


def _post_validate_results(results: list[dict], fallback_stage: str | None, limit: int) -> tuple[list[dict], str | None]:
    if not results:
        return results, None

    audited: list[dict] = []
    dropped = 0
    for item in results:
        bucket = _quality_bucket(item, fallback_stage)
        next_item = {**item, "_bucket": bucket}
        if bucket <= 0:
            dropped += 1
            continue
        audited.append(next_item)

    if not audited:
        fallback = sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)[: max(1, min(3, limit))]
        cleaned = []
        for item in fallback:
            next_item = {key: value for key, value in item.items() if not str(key).startswith("_")}
            cleaned.append(next_item)
        return cleaned, "Constraint audit found weak matches only. Showing the strongest available fallback candidates."

    audited.sort(key=lambda item: (item.get("_bucket", 0), item.get("score", 0.0)), reverse=True)
    cleaned = []
    for item in audited[: max(1, limit)]:
        next_item = {key: value for key, value in item.items() if not str(key).startswith("_")}
        cleaned.append(next_item)

    audit_note = None
    if dropped > 0:
        audit_note = f"Validated and filtered {dropped} low-confidence matches."
    return cleaned, audit_note


def _prune_weak_fallbacks(
    results: list[dict],
    *,
    fallback_stage: str | None,
    vocabulary_preference: str,
    relation: str,
) -> list[dict]:
    if not results or fallback_stage not in {"rhyme_first", "relation_family_rhymes"}:
        return results
    semantic_floor = 0.54 if vocabulary_preference == "balanced" else 0.6
    pruned = [
        item
        for item in results
        if item.get("relation_match")
        or item.get("near_relation_match")
        or float(item.get("_semantic", 0.0) or 0.0) >= semantic_floor
    ]
    if pruned:
        return pruned
    if relation == "antonym" and vocabulary_preference == "advanced":
        return []
    return results[:3]


def get_constraint_matches(
    rhyme_with: str,
    relation: str,
    meaning_of: str,
    context: str | None = None,
    limit: int = 10,
    vocabulary_preference: str = "balanced",
) -> tuple[list[dict], str | None]:
    rhyme_base = _clean_word(rhyme_with)
    meaning_base = _clean_word(meaning_of)

    rhyme_candidates = _collect_rhymes(rhyme_base)
    meaning_candidates = _collect_meaning(meaning_base, relation)
    definition_family = _collect_definition_family(meaning_base, relation)
    semantic_expansion = _collect_semantic_expansion(meaning_base, relation)
    semantic_expansion |= definition_family
    rhyme_from_semantic = _collect_rhyme_expansion(semantic_expansion)
    rhyme_from_relation = _collect_rhyme_expansion(meaning_candidates)
    rhyme_set = set(rhyme_candidates)
    context_words = _get_context_words(context)

    exact_matches = list(dict.fromkeys([word for word in rhyme_candidates if word in meaning_candidates]))
    if rhyme_base in meaning_candidates and rhyme_base not in exact_matches:
        exact_matches.insert(0, rhyme_base)
    near_relation_rhymes = list(
        dict.fromkeys([word for word in rhyme_candidates if word in semantic_expansion and word not in meaning_candidates])
    )
    note: str | None = None
    fallback_stage: str | None = None

    if exact_matches:
        candidate_pool = exact_matches + near_relation_rhymes
    else:
        if rhyme_candidates:
            fallback_stage = "rhyme_first"
            candidate_pool = list(dict.fromkeys(rhyme_candidates))
            note = "No exact synonym/antonym rhymes. Showing direct rhymes ranked by relation closeness."
        elif rhyme_from_relation or rhyme_from_semantic:
            fallback_stage = "relation_family_rhymes"
            candidate_pool = list(dict.fromkeys(list(rhyme_from_relation) + list(rhyme_from_semantic)))
            note = "No direct rhymes found. Showing rhymes from the synonym/antonym family ranked by meaning fit."
        else:
            fallback_stage = "meaning_only"
            candidate_pool = list(dict.fromkeys(list(meaning_candidates) + list(semantic_expansion)))
            note = "No rhyme candidates found. Showing strongest meaning matches."

    if len(candidate_pool) > 320:
        candidate_pool = candidate_pool[:320]
    if not candidate_pool:
        return [], "No matches found for the provided constraints."

    results = []
    for candidate in candidate_pool:
        rhyme_match = candidate in rhyme_set
        relation_match = candidate in meaning_candidates
        near_relation_match = candidate in semantic_expansion and not relation_match
        rhyme_score = _rhyme_quality(candidate, rhyme_base) if rhyme_base else 0.0
        semantic = _semantic_similarity(candidate, meaning_base) if meaning_base else 0.0
        if relation_match:
            relation_score = 1.0
        elif near_relation_match:
            relation_score = max(0.72, semantic)
        else:
            relation_score = semantic
        context_score = _context_similarity(candidate, context)
        if candidate in context_words:
            context_score = max(context_score, 0.68)
        frequency = estimate_frequency(candidate)
        generic_penalty = 0.0
        precision_bonus = 0.0
        if vocabulary_preference == "advanced":
            if candidate in _GENERIC_CONSTRAINT_WORDS:
                generic_penalty += 0.08
            if frequency >= 0.72 and semantic < 0.68:
                generic_penalty += 0.04
            if 0.015 <= frequency <= 0.32 and (relation_score >= 0.62 or semantic >= 0.58):
                precision_bonus += 0.04
            if context and context.strip().lower() in {"formal", "academic", "professional", "literary", "advanced"}:
                if 0.01 <= frequency <= 0.28 and (relation_score >= 0.58 or semantic >= 0.56):
                    precision_bonus += 0.03
            if relation_match and not rhyme_match and fallback_stage == "meaning_only":
                precision_bonus += 0.03
            if near_relation_match and rhyme_match:
                precision_bonus += 0.02
            if rhyme_match and not relation_match and semantic < 0.56:
                generic_penalty += 0.05
        if fallback_stage == "meaning_only":
            if relation == "antonym":
                score = (
                    0.60 * relation_score
                    + 0.16 * semantic
                    + 0.14 * context_score
                    + 0.10 * frequency
                )
            else:
                score = (
                    0.52 * relation_score
                    + 0.22 * semantic
                    + 0.16 * context_score
                    + 0.10 * frequency
                )
        else:
            if relation == "antonym":
                score = (
                    0.44 * rhyme_score
                    + 0.34 * relation_score
                    + 0.10 * semantic
                    + 0.07 * context_score
                    + 0.05 * frequency
                )
            else:
                score = (
                    0.48 * rhyme_score
                    + 0.28 * relation_score
                    + 0.12 * semantic
                    + 0.07 * context_score
                    + 0.05 * frequency
                )
        if rhyme_match and relation_match:
            score += 0.18 if relation == "antonym" else 0.16
        elif rhyme_match and near_relation_match:
            score += 0.08
        score += precision_bonus
        score -= generic_penalty
        results.append(
            {
                "word": candidate,
                "score": round(float(min(score, 0.99)), 4),
                "rhyme": rhyme_match,
                "relation_match": relation_match,
                "near_relation_match": near_relation_match,
                "_semantic": round(float(semantic), 4),
                "_rhyme_score": round(float(rhyme_score), 4),
                "_relation_score": round(float(relation_score), 4),
                "reason": _build_reason(
                    rhyme_with=rhyme_base,
                    meaning_of=meaning_base,
                    relation=relation,
                    rhyme_match=rhyme_match,
                    relation_match=relation_match,
                    near_relation_match=near_relation_match,
                    semantic=semantic,
                    context_hit=candidate in context_words or context_score >= 0.62,
                    fallback_stage=fallback_stage,
                ),
            }
        )

    results.sort(key=lambda item: (item["score"], item["relation_match"], item["rhyme"]), reverse=True)
    results = _prune_weak_fallbacks(
        results,
        fallback_stage=fallback_stage,
        vocabulary_preference=vocabulary_preference,
        relation=relation,
    )
    if not results:
        return [], "No strong Smart Match results satisfied the rhyme and meaning constraints."
    capped = max(1, min(10, int(limit or 10)))
    reranked = rerank_candidate_dicts(
        task="constraints",
        payload={
            "rhyme_with": rhyme_base,
            "relation": relation,
            "meaning_of": meaning_base,
            "context": context or "neutral",
            "vocabulary_preference": vocabulary_preference,
        },
        candidates=results,
        text_key="word",
        score_key="score",
        blend=0.72,
        max_results=capped,
    )
    validated, audit_note = _post_validate_results(reranked, fallback_stage=fallback_stage, limit=capped)
    if note and audit_note:
        return validated, f"{note} {audit_note}"
    return validated, note or audit_note


__all__ = ["get_constraint_matches"]
