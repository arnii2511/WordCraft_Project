from __future__ import annotations

try:
    import pronouncing
except ImportError:  # pragma: no cover
    pronouncing = None


def get_rhymes(word: str, max_results: int = 6) -> list[str]:
    if not word or pronouncing is None:
        return []
    cleaned = word.lower().strip()
    phones = pronouncing.phones_for_word(cleaned)
    base_syllables = pronouncing.syllable_count(phones[0]) if phones else None
    seen: set[str] = set()
    ranked: list[tuple[tuple[int, int, str], str]] = []
    for rhyme in pronouncing.rhymes(cleaned):
        candidate = rhyme.strip().lower()
        if not candidate or candidate == cleaned or candidate in seen:
            continue
        seen.add(candidate)
        candidate_phones = pronouncing.phones_for_word(candidate)
        candidate_syllables = (
            pronouncing.syllable_count(candidate_phones[0]) if candidate_phones else 99
        )
        syllable_gap = abs((base_syllables or candidate_syllables) - candidate_syllables)
        length_gap = abs(len(candidate) - len(cleaned))
        ranked.append(((syllable_gap, length_gap, candidate), candidate))
    ranked.sort(key=lambda item: item[0])
    return [candidate for _, candidate in ranked[:max_results]]
