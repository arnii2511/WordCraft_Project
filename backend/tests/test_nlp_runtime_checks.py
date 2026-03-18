from __future__ import annotations

import numpy as np

from backend.services.nlp.constraints_service import get_constraint_matches
from backend.services.nlp import embeddings as embeddings_module
from backend.services.nlp import oneword_service
from backend.services.nlp.ml_reranker import rerank_candidate_dicts
from backend.services.nlp.engine import generate_suggestions
from backend.services.nlp.lexical_service import get_lexical_results
from backend.services.nlp.oneword_service import get_one_word_substitutions
from backend.services.nlp.runtime_profile import (
    cross_encoder_enabled_for_task,
    retrieval_enabled_for_task,
    semantic_embeddings_enabled_for_task,
)


def test_suggest_runtime_shape_for_blank():
    payload = generate_suggestions("He walked ____ into the room.", "horror", mode="write")
    suggestions = payload.get("suggestions", [])

    assert suggestions, "Expected non-empty blank-fill suggestions"
    for item in suggestions:
        assert item.get("word")
        assert isinstance(item.get("score"), float)
        assert item.get("pos")
        assert item.get("note")


def test_suggest_runtime_shape_for_selection():
    payload = generate_suggestions(
        "She felt happy about the result.",
        "melancholic",
        mode="write",
        selection={"text": "happy", "start": 10, "end": 15},
    )
    suggestions = payload.get("suggestions", [])
    assert suggestions, "Expected non-empty selection suggestions"
    assert payload.get("explanation"), "Expected explainable output"


def test_rewrite_runtime_controls():
    rewrite_with_blank = generate_suggestions(
        "He ____ the door.",
        "formal",
        mode="rewrite",
        trigger="button",
    )
    assert not rewrite_with_blank.get("rewrites"), "Rewrite should be suppressed when blank exists"

    rewrite_complete = generate_suggestions(
        "The moon rose over the town.",
        "nostalgia",
        mode="rewrite",
        trigger="button",
    )
    assert rewrite_complete.get("rewrites"), "Rewrite should be generated for complete sentence"


def test_lexical_runtime_details():
    words, details = get_lexical_results("bright", "synonyms", context="formal", max_results=6)
    assert words, "Expected lexical words"
    assert details, "Expected lexical scored details"
    for item in details:
        assert item.get("word")
        assert isinstance(item.get("score"), float)
        assert item.get("reason")


def test_constraints_runtime_shape():
    results, note = get_constraint_matches("night", "synonym", "sad", "nostalgia", 6)
    assert results, "Expected constraints results"
    for item in results:
        assert item.get("word")
        assert isinstance(item.get("score"), float)
        assert item.get("reason")


def test_oneword_runtime_shape():
    results, note = get_one_word_substitutions(
        "a person who loves themselves too much",
        "formal",
        8,
    )
    assert results, "Expected one-word substitutions"
    for item in results:
        assert item.get("word")
        assert isinstance(item.get("score"), float)
        assert item.get("reason")


def test_free_host_profile_limits_rewrite_variants(monkeypatch):
    monkeypatch.setenv("WORDCRAFT_ML_PROFILE", "free")
    payload = generate_suggestions(
        "The moon rose over the town.",
        "nostalgia",
        mode="rewrite",
        trigger="button",
    )
    assert payload.get("rewrites"), "Rewrite should still be available on free-host profile"
    assert len(payload.get("rewrites", [])) <= 1


def test_free_host_profile_disables_transformers(monkeypatch):
    monkeypatch.setenv("WORDCRAFT_ML_PROFILE", "free")

    embeddings_module._model = None
    assert embeddings_module.load_model() is None


def test_free_host_profile_disables_retrieval(monkeypatch):
    monkeypatch.setenv("WORDCRAFT_ML_PROFILE", "free")
    monkeypatch.setenv("WORDCRAFT_ENABLE_RETRIEVAL", "1")
    import backend.ml.retrieval as retrieval_module

    assert retrieval_module._is_enabled() is False


def test_free_host_profile_skips_lexical_reranker(monkeypatch):
    monkeypatch.setenv("WORDCRAFT_ML_PROFILE", "free")
    candidates = [
        {"word": "sorrowful", "score": 0.61, "reason": "seed"},
        {"word": "melancholic", "score": 0.82, "reason": "seed"},
    ]
    output = rerank_candidate_dicts(
        task="lexical",
        payload={"word": "sad", "lexical_task": "synonyms", "context": "formal"},
        candidates=candidates,
        text_key="word",
        score_key="score",
        max_results=2,
    )
    assert output == candidates


def test_context_centroid_is_built_lazily(monkeypatch):
    embeddings_module._word_embeddings.clear()
    embeddings_module._context_centroids.clear()

    calls: list[list[str]] = []

    def fake_encode_texts(texts):
        batch = list(texts)
        calls.append(batch)
        return np.vstack(
            [np.full((192,), float(index + 1), dtype=np.float32) for index, _ in enumerate(batch)]
        )

    monkeypatch.setattr(embeddings_module, "encode_texts", fake_encode_texts)
    contexts = {
        "formal": {"words": ["precise", "measured"]},
        "hopeful": {"words": ["bright", "uplifted"]},
    }

    centroid = embeddings_module.get_context_centroid("formal", contexts=contexts)

    assert centroid is not None
    assert set(embeddings_module._context_centroids) == {"formal"}
    assert "hopeful" not in embeddings_module._context_centroids
    assert calls == [["measured", "precise"]]


def test_oneword_direct_suffix_seeds_survive_without_wordnet(monkeypatch):
    monkeypatch.setattr(oneword_service, "get_wordnet", lambda: None)
    oneword_service._lookup_phobia_candidates.cache_clear()
    oneword_service._lookup_suffix_candidates.cache_clear()

    heights_results, _ = get_one_word_substitutions("fear of heights", "formal", 5, "advanced")
    books_results, _ = get_one_word_substitutions("love of books", "formal", 5, "advanced")

    assert heights_results and heights_results[0]["word"] == "acrophobia"
    assert books_results and books_results[0]["word"] == "bibliophilia"


def test_lexical_stays_symbolic_by_default():
    assert semantic_embeddings_enabled_for_task("lexical") is False
    assert rerank_candidate_dicts(
        task="lexical",
        payload={"word": "sad", "lexical_task": "synonyms", "context": "formal"},
        candidates=[{"word": "melancholic", "score": 0.82, "reason": "seed"}],
        text_key="word",
        score_key="score",
        max_results=1,
    ) == [{"word": "melancholic", "score": 0.82, "reason": "seed"}]


def test_retrieval_and_cross_encoder_are_suggest_only(monkeypatch):
    monkeypatch.setenv("WORDCRAFT_ENABLE_RETRIEVAL", "1")
    assert retrieval_enabled_for_task("suggest_sentence") is True
    assert retrieval_enabled_for_task("oneword") is False
    assert retrieval_enabled_for_task("constraints") is False
    assert cross_encoder_enabled_for_task("suggest_blank") is True
    assert cross_encoder_enabled_for_task("constraints") is False
