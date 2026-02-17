from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Iterable

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)

CONCEPTNET_BASE = "https://api.conceptnet.io/c/en/"


def _normalize_terms(terms: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for term in terms:
        if not term:
            continue
        lowered = term.strip().lower()
        if " " in lowered:
            continue
        cleaned.append(lowered)
    return cleaned


@lru_cache(maxsize=512)
def get_related_words(word: str, max_terms: int = 8, timeout: float = 4.0) -> list[str]:
    if not word:
        return []
    if requests is None:
        return []
    try:
        response = requests.get(
            f"{CONCEPTNET_BASE}{word}",
            params={"limit": 20},
            timeout=timeout,
        )
        if response.status_code != 200:
            return []
        payload = response.json()
        edges = payload.get("edges", [])
        terms: list[str] = []
        for edge in edges:
            start = edge.get("start", {}).get("label")
            end = edge.get("end", {}).get("label")
            if start:
                terms.append(start)
            if end:
                terms.append(end)
        cleaned = _normalize_terms(terms)
        unique = list(dict.fromkeys(cleaned))
        return unique[:max_terms]
    except requests.RequestException as exc:
        logger.debug("ConceptNet request failed: %s", exc)
        return []


def warm_cache(words: Iterable[str]) -> None:
    if requests is None:
        return
    for word in words:
        if not word:
            continue
        try:
            get_related_words(word)
            time.sleep(0.05)
        except Exception:
            continue
