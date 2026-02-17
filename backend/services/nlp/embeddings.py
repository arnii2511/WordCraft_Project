from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled by runtime fallback
    SentenceTransformer = None

_model = None
_word_embeddings: dict[str, np.ndarray] = {}
_context_centroids: dict[str, np.ndarray] = {}
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_FALLBACK_DIM = 192


def load_model(model_name: str = DEFAULT_MODEL_NAME):
    global _model
    if _model is not None:
        return _model
    if SentenceTransformer is None:
        return None
    try:
        _model = SentenceTransformer(model_name)
    except Exception:
        _model = None
    return _model


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def _fallback_embed(text: str) -> np.ndarray:
    vec = np.zeros(_FALLBACK_DIM, dtype=np.float32)
    tokens = _TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        value = int(digest, 16)
        index = value % _FALLBACK_DIM
        sign = 1.0 if ((value >> 1) & 1) else -1.0
        weight = 1.0 + ((value >> 8) % 5) * 0.1
        vec[index] += sign * weight
    return _normalize(vec)


def encode_texts(texts: Iterable[str]) -> np.ndarray:
    model = load_model()
    text_list = list(texts)
    if model is None:
        if not text_list:
            return np.empty((0, _FALLBACK_DIM), dtype=np.float32)
        return np.vstack([_fallback_embed(text) for text in text_list])
    return model.encode(text_list, convert_to_numpy=True, normalize_embeddings=True)


def embed_sentence(text: str) -> np.ndarray:
    embeddings = encode_texts([text])
    return embeddings[0]


def ensure_context_embeddings(contexts: dict[str, dict]) -> None:
    if _context_centroids:
        return

    all_words: list[str] = []
    for payload in contexts.values():
        all_words.extend(payload.get("words", []))
    unique_words = sorted(set(all_words))

    if unique_words:
        vectors = encode_texts(unique_words)
        for word, vector in zip(unique_words, vectors):
            _word_embeddings[word] = vector

    for name, payload in contexts.items():
        words = payload.get("words", [])
        vectors = [_word_embeddings[word] for word in words if word in _word_embeddings]
        if vectors:
            centroid = np.mean(vectors, axis=0)
            _context_centroids[name] = _normalize(centroid)


def get_context_centroid(context: str) -> np.ndarray | None:
    return _context_centroids.get(context)


def get_word_embedding(word: str) -> np.ndarray | None:
    cached = _word_embeddings.get(word)
    if cached is not None:
        return cached
    vectors = encode_texts([word])
    embedding = vectors[0]
    _word_embeddings[word] = embedding
    return embedding
