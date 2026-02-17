from __future__ import annotations

try:
    import pronouncing
except ImportError:  # pragma: no cover
    pronouncing = None


def get_rhymes(word: str, max_results: int = 6) -> list[str]:
    if not word or pronouncing is None:
        return []
    rhymes = pronouncing.rhymes(word)
    return rhymes[:max_results]
