from __future__ import annotations

import pytest

from backend.services.nlp import embeddings
from backend.services.nlp import engine
from backend.services.nlp import oneword_service
from backend.services.nlp import pipeline
from backend.services.nlp import conceptnet_service


@pytest.fixture(autouse=True)
def isolate_nlp_runtime(monkeypatch: pytest.MonkeyPatch):
    # Keep tests deterministic and CI-safe: no remote ConceptNet calls, no heavy model loads.
    monkeypatch.setattr(embeddings, "SentenceTransformer", None, raising=False)
    monkeypatch.setattr(conceptnet_service, "get_related_words", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pipeline, "get_related_words", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(oneword_service, "get_related_words", lambda *_args, **_kwargs: [])

    embeddings._model = None
    embeddings._word_embeddings.clear()
    embeddings._context_centroids.clear()

    engine._CONTEXTS = None
    engine._INITIALIZED = False
    engine._INITIALIZATION_ERROR = None
