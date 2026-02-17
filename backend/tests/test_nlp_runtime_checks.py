from __future__ import annotations

from backend.services.nlp.constraints_service import get_constraint_matches
from backend.services.nlp.engine import generate_suggestions
from backend.services.nlp.lexical_service import get_lexical_results
from backend.services.nlp.oneword_service import get_one_word_substitutions


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
