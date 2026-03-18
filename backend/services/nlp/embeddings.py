from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

from .runtime_profile import transformers_enabled

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled by runtime fallback
    SentenceTransformer = None

try:
    import torch
except ImportError:  # pragma: no cover - handled by runtime fallback
    torch = None

_model = None
_model_device = None
_word_embeddings: dict[str, np.ndarray] = {}
_context_centroids: dict[str, np.ndarray] = {}
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_FALLBACK_DIM = 192


def _resolve_device() -> str | None:
    import os

    forced = (os.getenv("WORDCRAFT_EMBEDDINGS_DEVICE") or "").strip().lower()
    if forced:
        return forced
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return None


def load_model(model_name: str = DEFAULT_MODEL_NAME):
    global _model, _model_device
    if _model is not None:
        return _model
    if not transformers_enabled():
        return None
    if SentenceTransformer is None:
        return None
    try:
        _model_device = _resolve_device()
        if _model_device:
            _model = SentenceTransformer(model_name, device=_model_device)
        else:
            _model = SentenceTransformer(model_name)
    except Exception:
        _model = None
        _model_device = None
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


def ensure_word_embeddings(words: Iterable[str]) -> None:
    unique_words = sorted({(word or "").strip().lower() for word in words if (word or "").strip()})
    missing_words = [word for word in unique_words if word not in _word_embeddings]
    if not missing_words:
        return
    vectors = encode_texts(missing_words)
    for word, vector in zip(missing_words, vectors):
        _word_embeddings[word] = vector


def ensure_context_embeddings(contexts: dict[str, dict]) -> None:
    if _context_centroids:
        return
    all_words: list[str] = []
    for payload in contexts.values():
        all_words.extend(payload.get("words", []))
    ensure_word_embeddings(all_words)
    for name, payload in contexts.items():
        ensure_context_centroid(name, contexts=contexts)


def ensure_context_centroid(context: str, contexts: dict[str, dict] | None = None) -> np.ndarray | None:
    key = (context or "").strip().lower()
    if not key:
        return None
    cached = _context_centroids.get(key)
    if cached is not None:
        return cached
    if contexts is None:
        return None
    payload = contexts.get(key)
    if not payload:
        return None
    words = [(word or "").strip().lower() for word in payload.get("words", []) if (word or "").strip()]
    if not words:
        return None
    ensure_word_embeddings(words)
    vectors = [_word_embeddings[word] for word in words if word in _word_embeddings]
    if not vectors:
        return None
    centroid = _normalize(np.mean(vectors, axis=0))
    _context_centroids[key] = centroid
    return centroid


def get_context_centroid(context: str, contexts: dict[str, dict] | None = None) -> np.ndarray | None:
    key = (context or "").strip().lower()
    if not key:
        return None
    cached = _context_centroids.get(key)
    if cached is not None:
        return cached
    return ensure_context_centroid(key, contexts=contexts)


def get_word_embedding(word: str) -> np.ndarray | None:
    cached = _word_embeddings.get(word)
    if cached is not None:
        return cached
    vectors = encode_texts([word])
    embedding = vectors[0]
    _word_embeddings[word] = embedding
    return embedding
