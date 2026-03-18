from __future__ import annotations
import re
from functools import lru_cache
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from . import embeddings
from .conceptnet_service import get_related_words
from .context_loader import load_contexts
from .ml_reranker import rerank_candidate_dicts
from .runtime_profile import conceptnet_runtime_enabled
from .wordnet_service import estimate_frequency, get_primary_pos, get_wordnet, is_valid_word, search_definition_entries

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
_PHOBIA_PATTERNS = ("fear of ", "fear around ", "fear about ", "afraid of ", "phobia of ")
_PHILIA_PATTERNS = ("love of ", "fondness for ", "attraction to ", "affinity for ")
_MANIA_PATTERNS = ("obsession with ", "obsessed with ", "mania for ", "compulsion for ")
_GENERAL_PEOPLE_ALIASES = {"people", "person", "persons", "human", "humans"}
_PEOPLE_ALIASES = {"people", "person", "persons", "human", "humans", "public", "crowd", "crowds", "society", "social"}
_PEOPLE_SUBGROUP_MODIFIERS = {
    "homosexual",
    "homosexuality",
    "gay",
    "lesbian",
    "bisexual",
    "transgender",
    "men",
    "man",
    "women",
    "woman",
    "children",
    "child",
    "strangers",
    "stranger",
}
_GENERIC_ONEWORD_CANDIDATES = {
    "fearfulness",
    "fright",
    "anxiety",
    "worry",
    "panic",
    "emotion",
    "feeling",
    "feelings",
    "quality",
    "state",
    "person",
    "someone",
    "thing",
    "trait",
    "condition",
}
_FORMAL_CONTEXTS = {"formal", "academic", "scholarly", "professional", "literary"}
_ADVANCED_CONTEXTS = _FORMAL_CONTEXTS | {"advanced"}
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
_DIRECT_QUERY_SEEDS: dict[str, list[tuple[str, str]]] = {
    "fear of crowds": [
        ("enochlophobia", "an abnormal fear of crowds"),
        ("ochlophobia", "fear of crowds or mobs"),
    ],
    "fear of open spaces": [
        ("agoraphobia", "an abnormal fear of open or public places"),
    ],
    "love of knowledge": [
        ("philomathy", "love of learning and knowledge"),
        ("epistemophilia", "love of knowledge"),
    ],
    "hatred of mankind": [
        ("misanthropy", "hatred or distrust of humankind"),
        ("misanthropism", "hatred of humankind"),
    ],
    "a person who talks too much": [
        ("chatterbox", "a person who talks too much"),
        ("blabbermouth", "a person who reveals secrets or talks excessively"),
        ("windbag", "a person who talks at length with little substance"),
        ("babbler", "someone who talks rapidly and continuously in a foolish way"),
    ],
    "the quality of being stubborn": [
        ("obstinacy", "the quality of being stubbornly persistent"),
        ("obduracy", "stubborn persistence or resistance"),
        ("intransigence", "unwillingness to change one's views or agree"),
        ("stubbornness", "the quality of being stubborn"),
    ],
}
_STRICT_DIRECT_QUERIES = {
    "fear of crowds",
    "fear of open spaces",
    "love of knowledge",
    "hatred of mankind",
    "a person who talks too much",
    "the quality of being stubborn",
}
_DIRECT_QUERY_ALIASES = {
    "fear of crowd": "fear of crowds",
    "fear of a crowd": "fear of crowds",
    "fear of the crowd": "fear of crowds",
    "fear of open space": "fear of open spaces",
    "fear of an open space": "fear of open spaces",
    "fear of the open space": "fear of open spaces",
}


def _enable_conceptnet_runtime() -> bool:
    return conceptnet_runtime_enabled(default=False)


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
        for prefix in ("ego", "vain", "narciss", "conceit")
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


def _direct_query_seeds(query: str) -> list[tuple[str, str]]:
    lowered = (query or "").strip().lower()
    canonical = _DIRECT_QUERY_ALIASES.get(lowered, lowered)
    return list(_DIRECT_QUERY_SEEDS.get(canonical, []))


def _extract_phobia_target(query: str) -> set[str]:
    lowered = (query or "").strip().lower()
    for pattern in _PHOBIA_PATTERNS:
        if pattern in lowered:
            target = lowered.split(pattern, 1)[1].strip()
            tokens = set(_tokenize(target))
            if tokens & _GENERAL_PEOPLE_ALIASES:
                tokens |= _GENERAL_PEOPLE_ALIASES
            return tokens
    return set()


def _extract_suffix_target(query: str, patterns: tuple[str, ...]) -> set[str]:
    lowered = (query or "").strip().lower()
    for pattern in patterns:
        if pattern in lowered:
            target = lowered.split(pattern, 1)[1].strip()
            return set(_tokenize(target))
    return set()


def _phobia_definition_matches_target(def_tokens: set[str], target_tokens: set[str]) -> bool:
    overlap = len(def_tokens & target_tokens)
    people_query = bool(target_tokens & _GENERAL_PEOPLE_ALIASES)
    if not people_query:
        return overlap > 0

    if not (def_tokens & _GENERAL_PEOPLE_ALIASES):
        return False

    # For broad "fear of people" queries, reject subgroup-specific phobias
    # unless the subgroup was explicitly asked for in the query.
    extra_specific = (def_tokens & _PEOPLE_SUBGROUP_MODIFIERS) - target_tokens
    if extra_specific:
        return False
    return True


@lru_cache(maxsize=128)
def _lookup_phobia_candidates(target_signature: str) -> list[tuple[str, str]]:
    wn = get_wordnet()
    if wn is None or not target_signature:
        return []
    target_tokens = {item for item in target_signature.split("|") if item}
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    direct_seed_map = {
        "people": ("anthropophobia", "an abnormal fear of people"),
        "person": ("anthropophobia", "an abnormal fear of people"),
        "human": ("anthropophobia", "an abnormal fear of people"),
        "humans": ("anthropophobia", "an abnormal fear of people"),
        "crowd": ("enochlophobia", "a morbid fear of crowds"),
        "crowds": ("enochlophobia", "a morbid fear of crowds"),
        "heights": ("acrophobia", "a morbid fear of great heights"),
        "height": ("acrophobia", "a morbid fear of heights"),
    }
    for token in target_tokens:
        seed = direct_seed_map.get(token)
        if seed and seed[0] not in seen:
            seen.add(seed[0])
            results.append(seed)

    for synset in wn.all_synsets("n"):
        definition = synset.definition().lower()
        def_tokens = _token_set(definition)
        lemmas = [lemma.name().replace("_", " ").lower() for lemma in synset.lemmas()]
        for lemma in lemmas:
            if " " in lemma or not is_valid_word(lemma):
                continue
            if not lemma.endswith("phobia"):
                continue
            if lemma in seen:
                continue
            if _phobia_definition_matches_target(def_tokens, target_tokens):
                seen.add(lemma)
                results.append((lemma, synset.definition()))
    return results[:24]


@lru_cache(maxsize=128)
def _lookup_suffix_candidates(
    target_signature: str,
    suffix: str,
    direct_seed_items: tuple[tuple[str, tuple[str, str]], ...] = (),
) -> list[tuple[str, str]]:
    wn = get_wordnet()
    if wn is None or not target_signature:
        return []
    target_tokens = {item for item in target_signature.split("|") if item}
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    direct_seed_map = dict(direct_seed_items)
    for token in target_tokens:
        seed = direct_seed_map.get(token)
        if seed and seed[0] not in seen:
            seen.add(seed[0])
            results.append(seed)

    for entry in search_definition_entries(
        " ".join(sorted(target_tokens)),
        suffix=suffix,
        require_pos_codes={"n"},
        limit=24,
    ):
        if entry.word not in seen:
            seen.add(entry.word)
            results.append((entry.word, entry.definition))
    return results[:24]


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


def _inject_definition_search_candidates(
    pool: dict[str, CandidateMeta],
    query: str,
    *,
    phobia_hint: bool = False,
    person_hint: bool = False,
    abstract_hint: bool = False,
) -> None:
    require_pos_codes: set[str] | None = None
    if person_hint or abstract_hint or phobia_hint:
        require_pos_codes = {"n"}
    suffix = "phobia" if phobia_hint else None
    for entry in search_definition_entries(
        query,
        suffix=suffix,
        require_pos_codes=require_pos_codes,
        limit=48,
    ):
        _add_candidate(
            pool=pool,
            raw_word=entry.word,
            definition=entry.definition,
            source="dictionary",
            pos=entry.pos_code,
            lexname=entry.lexname,
            lemma_count=entry.lemma_count,
        )


def _inject_phobia_candidates(pool: dict[str, CandidateMeta], query: str) -> bool:
    target_tokens = _extract_phobia_target(query)
    if not target_tokens:
        return False
    signature = "|".join(sorted(target_tokens))
    injected = False
    for word, definition in _lookup_phobia_candidates(signature):
        _add_candidate(
            pool=pool,
            raw_word=word,
            definition=definition,
            source="pattern",
            pos="n",
            lexname="noun.feeling",
            lemma_count=1,
        )
        injected = True
    return injected


def _inject_suffix_family_candidates(
    pool: dict[str, CandidateMeta],
    query: str,
    *,
    patterns: tuple[str, ...],
    suffix: str,
    source: str,
    direct_seed_map: dict[str, tuple[str, str]] | None = None,
) -> bool:
    target_tokens = _extract_suffix_target(query, patterns)
    if not target_tokens:
        return False
    signature = "|".join(sorted(target_tokens))
    injected = False
    seed_items = tuple(sorted((direct_seed_map or {}).items()))
    for word, definition in _lookup_suffix_candidates(signature, suffix, seed_items):
        _add_candidate(
            pool=pool,
            raw_word=word,
            definition=definition,
            source=source,
            pos="n",
            lexname="noun.feeling",
            lemma_count=1,
        )
        injected = True
    return injected


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


def _inject_direct_query_candidates(pool: dict[str, CandidateMeta], query: str) -> bool:
    seeds = _direct_query_seeds(query)
    if not seeds:
        return False
    for word, definition in seeds:
        _add_candidate(
            pool=pool,
            raw_word=word,
            definition=definition,
            source="direct",
            pos="n",
            lexname="noun.cognition",
            lemma_count=3,
        )
    return True


def _collect_conceptnet(query: str, query_tokens: Iterable[str]) -> set[str]:
    if not _enable_conceptnet_runtime():
        return set()
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
    if "direct" in sources:
        score += 0.9
    if "dictionary" in sources:
        score += 0.7
    if "wordnet" in sources:
        score += 0.6
    if "hypernym" in sources:
        score += 0.3
    if "conceptnet" in sources:
        score += 0.2
    if "seed" in sources:
        score += 0.35
    return min(score, 1.0)


def _generic_candidate_penalty(
    word: str,
    meaning_terms: set[str],
    *,
    semantic: float,
    source_score: float,
    technical_hit: bool,
    person_hint: bool,
    abstract_hint: bool,
    vocabulary_preference: str = "balanced",
) -> float:
    if technical_hit:
        return 0.0
    penalty = 0.0
    if word in _GENERIC_ONEWORD_CANDIDATES:
        penalty += 0.14
    if semantic < 0.62 and source_score < 0.72 and meaning_terms & {"feeling", "state", "quality", "person", "someone"}:
        penalty += 0.08
    if person_hint and meaning_terms & {"person", "someone", "individual"} and semantic < 0.68:
        penalty += 0.06
    if abstract_hint and meaning_terms & {"quality", "state", "condition"} and semantic < 0.68:
        penalty += 0.06
    if vocabulary_preference == "advanced" and word in _GENERIC_ONEWORD_CANDIDATES:
        penalty += 0.08
    if vocabulary_preference == "advanced" and source_score < 0.72 and semantic < 0.68:
        penalty += 0.04
    return penalty


def _formal_precision_boost(
    context: str | None,
    *,
    semantic: float,
    frequency: float,
    source_score: float,
    technical_hit: bool,
    overlap: float,
    vocabulary_preference: str = "balanced",
) -> float:
    if not context or context.strip().lower() not in _FORMAL_CONTEXTS:
        if vocabulary_preference != "advanced":
            return 0.0
        if not context or context.strip().lower() not in _ADVANCED_CONTEXTS:
            return 0.0
    boost = 0.0
    if technical_hit:
        boost += 0.12
    if 0.02 <= frequency <= 0.38 and semantic >= 0.58:
        boost += 0.05
    if source_score >= 0.7 and overlap >= 0.25:
        boost += 0.04
    if vocabulary_preference == "advanced":
        if 0.01 <= frequency <= 0.28 and semantic >= 0.56:
            boost += 0.04
        if technical_hit:
            boost += 0.04
    return boost


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
    technical_hit: bool,
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

    if technical_hit:
        parts.append("Recognized as a specialized technical term.")

    if context and context_hit:
        parts.append(f"Boosted for {context.lower()} tone.")
    return " ".join(parts)


def _advanced_oneword_bonus(
    *,
    word: str,
    semantic: float,
    overlap: float,
    frequency: float,
    source_score: float,
    technical_hit: bool,
    person_hint: bool,
    abstract_hint: bool,
) -> float:
    bonus = 0.0
    if technical_hit:
        bonus += 0.08
    if source_score >= 0.7 and overlap >= 0.22:
        bonus += 0.04
    if 0.01 <= frequency <= 0.24 and semantic >= 0.56:
        bonus += 0.04
    if person_hint and word.endswith(("ist", "er")) and semantic >= 0.58:
        bonus += 0.03
    if abstract_hint and word.endswith(("ness", "ity", "tion", "ism")) and semantic >= 0.56:
        bonus += 0.03
    return bonus


def _apply_direct_query_filter(
    query: str,
    results: list[dict],
    vocabulary_preference: str,
    limit: int,
) -> list[dict]:
    lowered = (query or "").strip().lower()
    canonical = _DIRECT_QUERY_ALIASES.get(lowered, lowered)
    if canonical not in _STRICT_DIRECT_QUERIES:
        return results[:limit]

    direct_seeds = _direct_query_seeds(canonical)
    direct_words = [word for word, _ in direct_seeds]
    seed_lookup = {word: definition for word, definition in direct_seeds}
    if vocabulary_preference == "advanced":
        filtered = [item for item in results if item["word"] in direct_words]
        if filtered:
            return filtered[:limit]
        synthesized = [
            {
                "word": word,
                "score": round(max(0.8, 0.96 - index * 0.04), 4),
                "pos": get_primary_pos(word),
                "reason": f"Direct meaning match. {seed_lookup[word]}",
            }
            for index, word in enumerate(direct_words[:limit])
        ]
        return synthesized if synthesized else results[:limit]

    filtered = [item for item in results if item["word"] in direct_words]
    if filtered:
        extras = [item for item in results if item["word"] not in direct_words and item["score"] >= 0.72]
        return (filtered + extras)[:limit]
    return results[:limit]


def get_one_word_substitutions(
    query: str,
    context: str | None = None,
    limit: int = 10,
    vocabulary_preference: str = "balanced",
) -> tuple[list[dict], str | None]:
    cleaned_query = (query or "").strip().lower()
    if not cleaned_query:
        return [], "Please provide a phrase or description."

    query_tokens = _tokenize(cleaned_query)
    token_set = set(query_tokens)
    if not token_set:
        return [], "Please provide a phrase or description."

    person_hint, abstract_hint, self_hint = _query_hints(cleaned_query)
    phobia_target_tokens = _extract_phobia_target(cleaned_query)
    philia_target_tokens = _extract_suffix_target(cleaned_query, _PHILIA_PATTERNS)
    mania_target_tokens = _extract_suffix_target(cleaned_query, _MANIA_PATTERNS)
    context_words = _context_words(context)
    context_stems = set(_tokenize(" ".join(context_words)))
    query_vec = embeddings.embed_sentence(cleaned_query)

    candidates = _collect_wordnet_candidates(cleaned_query, query_tokens)
    direct_query_hint = _inject_direct_query_candidates(candidates, cleaned_query)
    _inject_definition_search_candidates(
        candidates,
        cleaned_query,
        phobia_hint=bool(phobia_target_tokens or philia_target_tokens or mania_target_tokens),
        person_hint=person_hint,
        abstract_hint=abstract_hint,
    )
    phobia_hint = _inject_phobia_candidates(candidates, cleaned_query)
    philia_hint = _inject_suffix_family_candidates(
        candidates,
        cleaned_query,
        patterns=_PHILIA_PATTERNS,
        suffix="philia",
        source="pattern",
        direct_seed_map={
            "books": ("bibliophilia", "love of books"),
            "book": ("bibliophilia", "love of books"),
            "cats": ("ailurophilia", "love of cats"),
            "cat": ("ailurophilia", "love of cats"),
        },
    )
    mania_hint = _inject_suffix_family_candidates(
        candidates,
        cleaned_query,
        patterns=_MANIA_PATTERNS,
        suffix="mania",
        source="pattern",
        direct_seed_map={
            "fire": ("pyromania", "an obsessive desire to set fire to things"),
        },
    )
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

        technical_hit = ("pattern" in meta.sources or "dictionary" in meta.sources) and (
            word.endswith("phobia") or word.endswith("philia") or word.endswith("mania")
        )
        if phobia_hint and word.endswith("phobia") and phobia_target_tokens and not (meaning_terms & phobia_target_tokens):
            continue
        if (
            phobia_hint
            and word.endswith("phobia")
            and phobia_target_tokens & _GENERAL_PEOPLE_ALIASES
            and (meaning_terms & _PEOPLE_SUBGROUP_MODIFIERS) - phobia_target_tokens
        ):
            continue
        technical_boost = 0.0
        if technical_hit and phobia_hint:
            technical_boost += 0.22
        if technical_hit and meaning_terms & phobia_target_tokens:
            technical_boost += 0.08
        if technical_hit and (meaning_terms & _PEOPLE_SUBGROUP_MODIFIERS) - phobia_target_tokens:
            technical_boost -= 0.28
        if technical_hit and philia_hint:
            technical_boost += 0.18
        if technical_hit and meaning_terms & philia_target_tokens:
            technical_boost += 0.08
        if technical_hit and mania_hint:
            technical_boost += 0.18
        if technical_hit and meaning_terms & mania_target_tokens:
            technical_boost += 0.08
        if "direct" in meta.sources:
            technical_boost += 0.26 if vocabulary_preference == "advanced" else 0.18

        frequency = estimate_frequency(word)
        generic_penalty = _generic_candidate_penalty(
            word,
            meaning_terms,
            semantic=semantic,
            source_score=source_score,
            technical_hit=technical_hit,
            person_hint=person_hint,
            abstract_hint=abstract_hint,
            vocabulary_preference=vocabulary_preference,
        )
        formal_boost = _formal_precision_boost(
            context,
            semantic=semantic,
            frequency=frequency,
            source_score=source_score,
            technical_hit=technical_hit,
            overlap=overlap,
            vocabulary_preference=vocabulary_preference,
        )
        advanced_bonus = 0.0
        if vocabulary_preference == "advanced":
            advanced_bonus = _advanced_oneword_bonus(
                word=word,
                semantic=semantic,
                overlap=overlap,
                frequency=frequency,
                source_score=source_score,
                technical_hit=technical_hit,
                person_hint=person_hint,
                abstract_hint=abstract_hint,
            )
        score = (
            0.42 * semantic
            + 0.22 * overlap
            + 0.16 * pos_score
            + 0.10 * context_fit
            + 0.06 * source_score
            + 0.02 * frequency
            + self_focus
            + technical_boost
            + formal_boost
            + advanced_bonus
            - generic_penalty
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
                    technical_hit=technical_hit,
                ),
                "meaning": meaning_line or None,
            }
        )

    results.sort(key=lambda item: (item["score"], item["word"]), reverse=True)
    capped = max(1, min(10, int(limit or 10)))
    top_results = rerank_candidate_dicts(
        task="oneword",
        payload={
            "query": cleaned_query,
            "context": context or "neutral",
            "vocabulary_preference": vocabulary_preference,
        },
        candidates=results,
        text_key="word",
        score_key="score",
        blend=0.74,
        max_results=capped,
    )
    if direct_query_hint:
        top_results = _apply_direct_query_filter(
            cleaned_query,
            top_results,
            vocabulary_preference,
            capped,
        )
    if top_results:
        return top_results, "Top one-word substitutions ranked by grammar-safe meaning match."
    return [], "No one-word substitutions found for that description."


__all__ = ["get_one_word_substitutions"]
