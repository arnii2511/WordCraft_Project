from __future__ import annotations

from backend.services.nlp.constraints_service import get_constraint_matches
from backend.services.nlp.engine import generate_suggestions
from backend.services.nlp.oneword_service import get_one_word_substitutions


def test_regression_blank_adv_slot():
    payload = generate_suggestions("He walked ____ into the room.", "horror", mode="write")
    suggestions = payload.get("suggestions", [])
    assert suggestions, "Expected suggestions for blank sentence"

    top_three = suggestions[:3]
    assert any(item.get("pos") == "ADV" for item in top_three), "Expected ADV in top blank-fill suggestions"
    assert any("grammatical slot" in (item.get("note") or "").lower() for item in top_three)


def test_regression_selection_adj_focus():
    payload = generate_suggestions(
        "She felt happy about the result.",
        "melancholic",
        mode="write",
        selection={"text": "happy", "start": 10, "end": 15},
    )
    suggestions = payload.get("suggestions", [])
    assert suggestions, "Expected suggestions for selection"
    assert suggestions[0].get("pos") == "ADJ", "Selection on adjective should prioritize ADJ candidates"


def test_regression_oneword_self_focused():
    results, note = get_one_word_substitutions(
        "a person who loves themselves too much",
        "formal",
        10,
    )
    assert results, "Expected one-word candidates"
    top_words = {item["word"] for item in results[:5]}
    assert top_words & {"narcissist", "egotist", "egocentric"}, (
        "Self-focused query should keep core candidates in top results"
    )


def test_regression_constraints_best_effort():
    results, note = get_constraint_matches("night", "synonym", "sad", "nostalgia", 10)
    assert results, "Expected best-effort constraints results"
    assert note is not None, "Expected best-effort note when strict overlap is absent"
    assert "best-effort" in note.lower() or "no strict" in note.lower()
