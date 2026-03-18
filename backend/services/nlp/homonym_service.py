from __future__ import annotations

import re

try:
    import pronouncing
except ImportError:  # pragma: no cover
    pronouncing = None


def get_homophones(word: str, max_results: int = 8) -> list[str]:
    if not word or pronouncing is None:
        return []
    cleaned = word.lower().strip()
    phones = pronouncing.phones_for_word(cleaned)
    if not phones:
        return []
    pattern = "^" + re.escape(phones[0]) + "$"
    matches = pronouncing.search(pattern)
    seen: set[str] = set()
    ranked: list[tuple[tuple[int, str], str]] = []
    for match in matches:
        candidate = match.strip().lower()
        if not candidate or candidate == cleaned or candidate in seen:
            continue
        seen.add(candidate)
        length_gap = abs(len(candidate) - len(cleaned))
        ranked.append(((length_gap, candidate), candidate))
    ranked.sort(key=lambda item: item[0])
    return [candidate for _, candidate in ranked[:max_results]]


__all__ = ["get_homophones"]
