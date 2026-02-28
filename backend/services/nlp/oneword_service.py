from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from . import embeddings
from .conceptnet_service import get_related_words
from .context_loader import load_contexts
from .ml_reranker import rerank_candidate_dicts
from .wordnet_service import estimate_frequency, get_wordnet, is_valid_word

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_CONTEXT_CACHE: dict[str, dict] | None = None

_PERSON_PATTERNS = ("a person who", "someone who", "one who", "an individual who")
_ABSTRACT_PATTERNS = ("quality of", "state of being", "act of")
_ABSTRACT_LEXNAMES = {
    "noun.attribute",
    "noun.state",
    "noun.feeling",
    "noun.cognition",
}
_SELF_HINT_TERMS = {"self", "ego", "egot", "vain", "conceit", "narciss", "obsess", "selfish"}
_REFLEXIVE_MAP = {
    "myself": "self",
    "yourself": "self",
    "himself": "self",
    "herself": "self",
    "itself": "self",
    "ourselves": "self",
    "yourselves": "self",
    "themselves": "self",
}


@dataclass
class CandidateMeta:
    definitions: set[str]
    sources: set[str]
    lemma_count: int
    pos_tags: set[str]
    lexnames: set[str]


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _scale(similarity: float) -> float:
    return (similarity + 1.0) / 2.0


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text or ""):
        lowered = token.lower().strip("-'")
        lowered = _REFLEXIVE_MAP.get(lowered, lowered)
        if lowered.endswith("self") and len(lowered) > 4:
            lowered = "self"
        if lowered:
            tokens.append(lowered)
    return tokens


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _clean_word(word: str) -> str:
    return (word or "").strip().lower().replace("_", " ")


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


def _query_hints(query: str) -> tuple[bool, bool, bool]:
    lowered = (query or "").strip().lower()
    tokens = _tokenize(lowered)
    person_hint = any(pattern in lowered for pattern in _PERSON_PATTERNS)
    abstract_hint = any(pattern in lowered for pattern in _ABSTRACT_PATTERNS)
    self_hint = "self" in tokens or any(
        token.startswith(prefix)
        for token in tokens
        for prefix in ("ego", "obsess", "vain", "narciss", "conceit")
    )
    return person_hint, abstract_hint, self_hint


def _add_candidate(
    pool: dict[str, CandidateMeta],
    raw_word: str,
    definition: str,
    source: str,
    pos: str,
    lexname: str,
    lemma_count: int = 0,
) -> None:
    cleaned = _clean_word(raw_word)
    if not cleaned or " " in cleaned:
        return
    if not is_valid_word(cleaned):
        return
    entry = pool.setdefault(
        cleaned,
        CandidateMeta(definitions=set(), sources=set(), lemma_count=0, pos_tags=set(), lexnames=set()),
    )
    if definition:
        entry.definitions.add(definition.strip())
    entry.sources.add(source)
    if pos:
        entry.pos_tags.add(pos)
    if lexname:
        entry.lexnames.add(lexname)
    entry.lemma_count += max(0, lemma_count)


def _seed_terms(query: str, query_tokens: list[str]) -> list[str]:
    terms = [query.strip().lower().replace(" ", "_")]
    terms.extend(query_tokens[:8])
    return list(dict.fromkeys([term for term in terms if term]))


def _collect_wordnet_candidates(query: str, query_tokens: list[str]) -> dict[str, CandidateMeta]:
    wn = get_wordnet()
    if wn is None:
        return {}
    pool: dict[str, CandidateMeta] = {}
    for term in _seed_terms(query, query_tokens):
        synsets = wn.synsets(term)
        for synset in synsets[:14]:
            definition = synset.definition()
            for lemma in synset.lemmas():
                _add_candidate(
                    pool=pool,
                    raw_word=lemma.name(),
                    definition=definition,
                    source="wordnet",
                    pos=synset.pos(),
                    lexname=synset.lexname(),
                    lemma_count=lemma.count(),
                )
            for related in synset.hypernyms()[:4]:
                related_definition = related.definition()
                for lemma in related.lemmas():
                    _add_candidate(
                        pool=pool,
                        raw_word=lemma.name(),
                        definition=related_definition,
                        source="hypernym",
                        pos=related.pos(),
                        lexname=related.lexname(),
                        lemma_count=lemma.count(),
                    )
    return pool


def _inject_self_seed_candidates(pool: dict[str, CandidateMeta]) -> None:
    seed_rows = [
        ("narcissist", "a self-obsessed person", "noun.person"),
        ("egotist", "a self-centered and conceited person", "noun.person"),
        ("egocentric", "focused excessively on oneself", "noun.person"),
        ("vain", "excessively proud or self-admiring", "adj.all"),
    ]
    for word, definition, lexname in seed_rows:
        _add_candidate(
            pool=pool,
            raw_word=word,
            definition=definition,
            source="seed",
            pos="n" if lexname == "noun.person" else "a",
            lexname=lexname,
            lemma_count=2,
        )


def _collect_conceptnet(query: str, query_tokens: Iterable[str]) -> set[str]:
    related: set[str] = set()
    for term in _seed_terms(query, list(query_tokens))[:5]:
        for word in get_related_words(term, max_terms=12):
            cleaned = _clean_word(word)
            if " " in cleaned:
                continue
            if is_valid_word(cleaned):
                related.add(cleaned)
    return related


def _best_definition(definitions: Iterable[str], query_tokens: set[str]) -> tuple[str, float]:
    best = ""
    best_overlap = 0.0
    for definition in definitions:
        overlap = len(_token_set(definition) & query_tokens) / max(1, len(query_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best = definition
    return best, best_overlap


def _pos_score(meta: CandidateMeta, person_hint: bool, abstract_hint: bool) -> float:
    score = 0.0
    if person_hint:
        if "n" in meta.pos_tags:
            score += 0.5
        if "noun.person" in meta.lexnames:
            score += 0.5
        return score
    if abstract_hint:
        if "n" in meta.pos_tags:
            score += 0.45
        if meta.lexnames & _ABSTRACT_LEXNAMES:
            score += 0.55
        return score
    if "n" in meta.pos_tags:
        return 0.5
    if "a" in meta.pos_tags or "s" in meta.pos_tags:
        return 0.45
    return 0.2


def _source_score(sources: set[str]) -> float:
    score = 0.0
    if "wordnet" in sources:
        score += 0.6
    if "hypernym" in sources:
        score += 0.3
    if "conceptnet" in sources:
        score += 0.2
    if "seed" in sources:
        score += 0.35
    return min(score, 1.0)


def _shorten(text: str, max_chars: int = 84) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1].rstrip()}…"


def _build_reason(
    meaning_line: str,
    person_hint: bool,
    abstract_hint: bool,
    context: str | None,
    context_hit: bool,
    semantic: float,
) -> str:
    parts: list[str] = []
    if meaning_line:
        parts.append(f"Matches '{meaning_line}'.")
    elif person_hint:
        parts.append("Fits a person-focused noun description.")
    elif abstract_hint:
        parts.append("Fits an abstract-quality description.")
    else:
        parts.append("Strong one-word substitution candidate.")

    if semantic >= 0.62:
        parts.append("Strong semantic fit.")
    elif semantic >= 0.54:
        parts.append("Good semantic fit.")

    if context and context_hit:
        parts.append(f"Boosted for {context.lower()} tone.")
    return " ".join(parts)


def get_one_word_substitutions(
    query: str,
    context: str | None = None,
    limit: int = 10,
) -> tuple[list[dict], str | None]:
    cleaned_query = (query or "").strip().lower()
    if not cleaned_query:
        return [], "Please provide a phrase or description."

    query_tokens = _tokenize(cleaned_query)
    token_set = set(query_tokens)
    if not token_set:
        return [], "Please provide a phrase or description."

    person_hint, abstract_hint, self_hint = _query_hints(cleaned_query)
    context_words = _context_words(context)
    context_stems = set(_tokenize(" ".join(context_words)))
    query_vec = embeddings.embed_sentence(cleaned_query)

    candidates = _collect_wordnet_candidates(cleaned_query, query_tokens)
    if self_hint:
        _inject_self_seed_candidates(candidates)
    concept_related = _collect_conceptnet(cleaned_query, query_tokens)
    for word in concept_related:
        _add_candidate(
            pool=candidates,
            raw_word=word,
            definition="",
            source="conceptnet",
            pos="",
            lexname="",
            lemma_count=0,
        )

    results: list[dict] = []
    for word, meta in candidates.items():
        if word in token_set:
            continue
        definition, overlap = _best_definition(meta.definitions, token_set)
        meaning_line = _shorten(definition) if definition else ""
        meaning_terms = _token_set(meaning_line)
        self_topic_hit = bool(meaning_terms & _SELF_HINT_TERMS) or any(
            word.startswith(prefix) for prefix in ("ego", "narciss", "vain", "conceit")
        )
        if self_hint and not self_topic_hit:
            continue
        candidate_vec = embeddings.get_word_embedding(word)
        semantic = _scale(_cosine_similarity(query_vec, candidate_vec))
        definition_sem = 0.0
        if definition:
            def_vec = embeddings.embed_sentence(definition)
            definition_sem = _scale(_cosine_similarity(query_vec, def_vec))
        semantic = max(semantic, definition_sem)

        pos_score = _pos_score(meta, person_hint, abstract_hint)
        source_score = _source_score(meta.sources)
        context_fit = 0.0
        if context:
            context_vec = embeddings.get_context_centroid(context.strip().lower())
            context_fit = _scale(_cosine_similarity(context_vec, candidate_vec))
        if word in context_words:
            context_fit = max(context_fit, 0.7)
        if meaning_line and _token_set(meaning_line) & context_stems:
            context_fit = max(context_fit, 0.64)

        self_focus = 0.0
        if self_hint:
            self_terms = _SELF_HINT_TERMS
            if meaning_terms & self_terms or any(word.startswith(item) for item in ("ego", "narciss", "vain")):
                self_focus = 0.14
            else:
                self_focus = -0.24

        if self_hint and person_hint and "noun.person" not in meta.lexnames:
            self_focus -= 0.12

        frequency = estimate_frequency(word)
        score = (
            0.42 * semantic
            + 0.22 * overlap
            + 0.16 * pos_score
            + 0.10 * context_fit
            + 0.06 * source_score
            + 0.04 * frequency
            + self_focus
        )
        score = max(0.0, min(0.99, score))
        results.append(
            {
                "word": word,
                "score": round(float(score), 4),
                "reason": _build_reason(
                    meaning_line=meaning_line,
                    person_hint=person_hint,
                    abstract_hint=abstract_hint,
                    context=context,
                    context_hit=context_fit >= 0.62,
                    semantic=semantic,
                ),
                "meaning": meaning_line or None,
            }
        )

    results.sort(key=lambda item: (item["score"], item["word"]), reverse=True)
    capped = max(1, min(10, int(limit or 10)))
    top_results = rerank_candidate_dicts(
        task="oneword",
        payload={"query": cleaned_query, "context": context or "neutral"},
        candidates=results,
        text_key="word",
        score_key="score",
        blend=0.74,
        max_results=capped,
    )
    if top_results:
        return top_results, "Top one-word substitutions ranked by grammar-safe meaning match."
    return [], "No one-word substitutions found for that description."


__all__ = ["get_one_word_substitutions"]
