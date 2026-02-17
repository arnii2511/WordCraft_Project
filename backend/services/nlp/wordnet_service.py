from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_WORD_RE = re.compile(r"^[a-zA-Z][a-zA-Z\-]*$")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "by",
    "with",
    "from",
}


def get_wordnet():
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None
    try:
        wn.synsets("test")
    except LookupError:
        return None
    return wn


def is_valid_word(word: str) -> bool:
    if not word:
        return False
    if len(word) < 3:
        return False
    if word in STOPWORDS:
        return False
    return bool(_WORD_RE.match(word))


def get_synonyms(words: Iterable[str], max_synonyms_per_word: int = 6) -> set[str]:
    wn = get_wordnet()
    if wn is None:
        return set()
    synonyms: set[str] = set()
    for word in words:
        synsets = wn.synsets(word)
        count = 0
        for synset in synsets:
            for lemma in synset.lemma_names():
                cleaned = lemma.replace("_", " ").strip().lower()
                if " " in cleaned:
                    continue
                if cleaned and cleaned != word:
                    synonyms.add(cleaned)
                    count += 1
                    if count >= max_synonyms_per_word:
                        break
            if count >= max_synonyms_per_word:
                break
    return synonyms


def get_synonyms_for_word(word: str, max_results: int = 8) -> list[str]:
    if not word:
        return []
    synonyms = get_synonyms([word], max_synonyms_per_word=max_results)
    filtered = [syn for syn in synonyms if syn != word]
    return filtered[:max_results]


def get_antonyms(word: str, max_results: int = 8) -> list[str]:
    wn = get_wordnet()
    if wn is None or not word:
        return []
    antonyms: list[str] = []
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                cleaned = antonym.name().replace("_", " ").lower()
                if cleaned and cleaned != word and cleaned not in antonyms:
                    antonyms.append(cleaned)
                    if len(antonyms) >= max_results:
                        return antonyms
    return antonyms


def get_primary_pos(word: str) -> str | None:
    wn = get_wordnet()
    if wn is None:
        return None
    synsets = wn.synsets(word)
    if not synsets:
        return None
    pos = synsets[0].pos()
    mapping = {"n": "NOUN", "v": "VERB", "a": "ADJ", "s": "ADJ", "r": "ADV"}
    return mapping.get(pos)


def get_pos_tags(word: str) -> set[str]:
    wn = get_wordnet()
    if wn is None or not word:
        return set()
    mapping = {"n": "NOUN", "v": "VERB", "a": "ADJ", "s": "ADJ", "r": "ADV"}
    tags: set[str] = set()
    for synset in wn.synsets(word):
        mapped = mapping.get(synset.pos())
        if mapped:
            tags.add(mapped)
    return tags


def get_derivational_forms(word: str, max_results: int = 12) -> list[str]:
    wn = get_wordnet()
    if wn is None or not word:
        return []
    forms: list[str] = []
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            for related in lemma.derivationally_related_forms():
                cleaned = related.name().replace("_", " ").lower().strip()
                if " " in cleaned:
                    continue
                if cleaned and cleaned != word and is_valid_word(cleaned) and cleaned not in forms:
                    forms.append(cleaned)
                    if len(forms) >= max_results:
                        return forms
    return forms


def estimate_frequency(word: str) -> float:
    """Estimate a light frequency prior using WordNet lemma counts."""
    wn = get_wordnet()
    if wn is None or not word:
        return 0.0
    counts = Counter()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            cleaned = lemma.name().replace("_", " ").lower().strip()
            if cleaned != word:
                continue
            counts[cleaned] += max(0, lemma.count())
    total = counts.get(word, 0)
    return min(float(total), 20.0) / 20.0
