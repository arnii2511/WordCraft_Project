from __future__ import annotations

import re

try:
    import pronouncing
except ImportError:  # pragma: no cover
    pronouncing = None


def get_homophones(word: str, max_results: int = 8) -> list[str]:
    if not word or pronouncing is None:
        return []
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return []
    pattern = "^" + re.escape(phones[0]) + "$"
    matches = pronouncing.search(pattern)
    results = [match for match in matches if match.lower() != word.lower()]
    return results[:max_results]


__all__ = ["get_homophones"]
