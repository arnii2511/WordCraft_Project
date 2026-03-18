from __future__ import annotations

from backend.services.nlp.constraints_service import get_constraint_matches
from backend.services.nlp.engine import generate_suggestions
from backend.services.nlp.lexical_service import get_lexical_results
from backend.services.nlp.oneword_service import get_one_word_substitutions
from backend.services.nlp.pipeline import build_pipeline
from backend.services.nlp.context_loader import load_contexts
from backend.services.nlp.ranker import rank_candidates


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


def test_regression_oneword_fear_of_people_prefers_technical_term():
    results, note = get_one_word_substitutions("fear of people", "formal", 10)
    assert results, "Expected one-word candidates for phobia query"
    top_words = [item["word"] for item in results[:5]]
    assert top_words[0] == "anthropophobia", (
        "Fear-of-people query should rank anthropophobia first"
    )
    assert "homophobia" not in top_words[:3], (
        "Broad people query should not surface subgroup-specific phobias near the top"
    )


def test_regression_oneword_fear_of_heights_prefers_acrophobia():
    results, note = get_one_word_substitutions("fear of heights", "formal", 10)
    assert results, "Expected one-word candidates for heights query"
    top_words = [item["word"] for item in results[:5]]
    assert "acrophobia" in top_words, "Fear-of-heights query should surface acrophobia near the top"


def test_regression_oneword_fear_of_crowd_maps_to_crowds_term():
    results, note = get_one_word_substitutions("fear of crowd", "formal", 10, "advanced")
    assert results, "Expected one-word candidates for crowd query"
    top_words = [item["word"] for item in results[:5]]
    assert "enochlophobia" in top_words[:3], f"Expected enochlophobia near top, got {top_words[:5]}"


def test_regression_oneword_love_of_books_prefers_bibliophilia():
    results, note = get_one_word_substitutions("love of books", "formal", 10)
    assert results, "Expected one-word candidates for philia query"
    top_words = [item["word"] for item in results[:5]]
    assert "bibliophilia" in top_words, "Love-of-books query should surface bibliophilia near the top"


def test_regression_oneword_love_of_knowledge_prefers_technical_terms():
    results, note = get_one_word_substitutions("love of knowledge", "formal", 10, "advanced")
    assert results, "Expected one-word candidates for knowledge-love query"
    top_words = [item["word"] for item in results[:5]]
    assert top_words[0] in {"philomathy", "epistemophilia"}
    assert set(top_words).issubset({"philomathy", "epistemophilia"}), (
        "Advanced mode should keep love-of-knowledge results tightly focused on technical matches"
    )


def test_regression_oneword_person_who_talks_too_much_prefers_direct_words():
    results, note = get_one_word_substitutions("a person who talks too much", "formal", 10, "advanced")
    assert results, "Expected one-word candidates for overtalkative-person query"
    top_words = [item["word"] for item in results[:5]]
    assert top_words[0] in {"chatterbox", "blabbermouth", "windbag", "babbler"}
    assert "sweater" not in top_words and "buyer" not in top_words


def test_regression_oneword_quality_of_being_stubborn_prefers_trait_words():
    results, note = get_one_word_substitutions("the quality of being stubborn", "formal", 10, "advanced")
    assert results, "Expected one-word candidates for stubbornness query"
    top_words = [item["word"] for item in results[:5]]
    assert top_words[0] in {"obstinacy", "obduracy", "intransigence", "stubbornness"}
    assert "wickedness" not in top_words[:3]


def test_regression_oneword_obsession_with_fire_prefers_pyromania():
    results, note = get_one_word_substitutions("obsession with fire", "formal", 10)
    assert results, "Expected one-word candidates for mania query"
    top_words = [item["word"] for item in results[:5]]
    assert "pyromania" in top_words, "Obsession-with-fire query should surface pyromania near the top"


def test_regression_constraints_best_effort():
    results, note = get_constraint_matches("night", "synonym", "sad", "nostalgia", 10)
    assert results, "Expected best-effort constraints results"
    assert note is not None, "Expected best-effort note when strict overlap is absent"
    assert "no exact" in note.lower() or "no direct" in note.lower() or "no rhyme" in note.lower()
    assert all(item.get("rhyme") for item in results[:5]), "Best-effort stage should still prefer rhyme-first outputs"


def test_regression_constraints_audit_removes_weak_matches():
    results, note = get_constraint_matches("night", "synonym", "bright", "neutral", 10)
    assert results, "Expected Smart Match candidates"
    top_slice = results[:5]
    assert any(item.get("relation_match") for item in top_slice), (
        "Validated Smart Match results should keep at least one true relation match near the top"
    )


def test_regression_constraints_antonym_reasoning_language():
    results, note = get_constraint_matches("cold", "antonym", "warm", "formal", 10)
    assert results, "Expected Smart Match antonym candidates"
    assert results[0]["word"] == "cold", "Exact antonym rhyme should surface first when available"
    joined_reasons = " ".join(item.get("reason", "") for item in results[:5]).lower()
    assert (
        "opposes the meaning" in joined_reasons
        or "antonym family" in joined_reasons
        or "opposite meaning" in joined_reasons
        or "antonym constraint" in joined_reasons
    )


def test_regression_selection_builds_definition_family_candidates():
    pipeline = build_pipeline(
        text="She felt happy about the result.",
        context_key="melancholic",
        mode="write",
        contexts=load_contexts(),
        selection={"text": "happy", "start": 10, "end": 15},
    )
    definition_candidates = [
        word for word, sources in pipeline.source_map.items() if "definition" in sources
    ]
    assert definition_candidates, "Selection mode should include definition-family candidates"


def test_regression_sentence_builds_definition_family_candidates():
    pipeline = build_pipeline(
        text="The letter carried a sad tone through the room.",
        context_key="formal",
        mode="write",
        contexts=load_contexts(),
    )
    definition_candidates = [
        word for word, sources in pipeline.source_map.items() if "definition" in sources
    ]
    assert definition_candidates, "Sentence mode should include definition-family candidates"


def test_regression_editor_penalizes_generic_selection_words():
    ranked = rank_candidates(
        cleaned_text="She felt happy about the result.",
        context_key="formal",
        candidates={"good", "melancholic", "wistful"},
        context_description="a formal tone",
        blank_present=False,
        emotion_scores={},
        source_map={
            "good": {"wordnet"},
            "melancholic": {"definition", "wordnet"},
            "wistful": {"definition", "wordnet"},
        },
        strict_pos=True,
        expected_pos_override={"ADJ"},
        context_words={"melancholic", "wistful"},
        top_k=3,
    )
    assert ranked, "Expected ranked candidates for selection-like scoring"
    top_words = [item["word"] for item in ranked[:2]]
    assert "good" not in top_words, "Generic editor alternatives should not outrank more precise formal words"


def test_regression_editor_sentence_prefers_precise_formal_words():
    ranked = rank_candidates(
        cleaned_text="The letter carried a sad tone through the room.",
        context_key="formal",
        candidates={"good", "somber", "melancholic"},
        context_description="a formal tone",
        blank_present=False,
        emotion_scores={},
        source_map={
            "good": {"wordnet"},
            "somber": {"definition", "wordnet"},
            "melancholic": {"definition", "wordnet"},
        },
        strict_pos=False,
        expected_pos_override=None,
        context_words={"somber", "melancholic"},
        top_k=3,
    )
    assert ranked, "Expected ranked sentence candidates"
    top_words = [item["word"] for item in ranked[:2]]
    assert "good" not in top_words, "Generic sentence alternatives should not outrank more precise formal words"


def test_regression_lexical_prefers_precise_formal_synonyms():
    words, details = get_lexical_results("sad", "synonyms", context="formal", max_results=8)
    assert details, "Expected lexical synonym candidates"
    top_words = [item["word"] for item in details[:5]]
    assert "sorrowful" in top_words or "unhappy" in top_words, (
        "Formal sad-synonym suggestions should keep writer-usable emotional synonyms near the top"
    )
    assert "thereto" not in top_words[:3], "Sense-noise like 'thereto' should not surface near the top"


def test_regression_antonyms_for_warm_stay_in_temperature_sense():
    words, details = get_lexical_results("warm", "antonyms", context="formal", max_results=8)
    assert details, "Expected lexical antonym candidates"
    top_words = [item["word"] for item in details[:5]]
    assert "cold" in top_words or "cool" in top_words, (
        "Warm antonyms should stay in the temperature sense for writer-facing output"
    )
    assert "borrow" not in top_words[:3] and "improve" not in top_words[:3]


def test_regression_antonyms_for_sun_do_not_hallucinate_related_words():
    words, details = get_lexical_results("sun", "antonyms", context="neutral", max_results=8)
    top_words = [item["word"] for item in details[:5]]
    assert "burn" not in top_words and "sunlight" not in top_words, (
        f"Antonym search should not drift into related words: {top_words}"
    )


def test_regression_rhymes_for_light_prefer_core_writer_matches():
    words, details = get_lexical_results("light", "rhymes", context="formal", max_results=8)
    assert details, "Expected rhyme candidates"
    top_words = [item["word"] for item in details[:5]]
    assert "bright" in top_words, "Useful core rhymes like 'bright' should stay near the top"
    assert "alright" not in top_words[:2], "Colloquial forms should not outrank stronger core rhymes"


def test_regression_homophones_for_flower_keep_flour():
    words, details = get_lexical_results("flower", "homonyms", context="formal", max_results=8)
    assert details, "Expected homophone candidates"
    top_words = [item["word"] for item in details[:3]]
    assert top_words[0] == "flour"


def test_regression_advanced_preference_penalizes_generic_words_more():
    balanced = rank_candidates(
        cleaned_text="The letter carried a sad tone through the room.",
        context_key="neutral",
        candidates={"good", "somber", "melancholic"},
        context_description="a neutral tone",
        blank_present=False,
        emotion_scores={},
        source_map={
            "good": {"wordnet"},
            "somber": {"definition", "wordnet"},
            "melancholic": {"definition", "wordnet"},
        },
        strict_pos=False,
        expected_pos_override=None,
        context_words={"somber", "melancholic"},
        top_k=3,
        vocabulary_preference="balanced",
    )
    advanced = rank_candidates(
        cleaned_text="The letter carried a sad tone through the room.",
        context_key="neutral",
        candidates={"good", "somber", "melancholic"},
        context_description="a neutral tone",
        blank_present=False,
        emotion_scores={},
        source_map={
            "good": {"wordnet"},
            "somber": {"definition", "wordnet"},
            "melancholic": {"definition", "wordnet"},
        },
        strict_pos=False,
        expected_pos_override=None,
        context_words={"somber", "melancholic"},
        top_k=3,
        vocabulary_preference="advanced",
    )
    balanced_score = next(item["score"] for item in balanced if item["word"] == "good")
    advanced_score = next(item["score"] for item in advanced if item["word"] == "good")
    assert advanced_score < balanced_score, (
        "Advanced vocabulary preference should penalize generic words more strongly"
    )


def test_regression_advanced_pipeline_expands_sentence_candidates():
    balanced_pipeline = build_pipeline(
        text="The letter carried a sad tone through the room.",
        context_key="formal",
        mode="write",
        contexts=load_contexts(),
        vocabulary_preference="balanced",
    )
    advanced_pipeline = build_pipeline(
        text="The letter carried a sad tone through the room.",
        context_key="formal",
        mode="write",
        contexts=load_contexts(),
        vocabulary_preference="advanced",
    )
    assert len(advanced_pipeline.candidates) >= len(balanced_pipeline.candidates), (
        "Advanced vocabulary mode should consider at least as many sentence candidates as balanced mode"
    )


def test_regression_advanced_oneword_keeps_generic_fear_terms_down():
    balanced_results, _ = get_one_word_substitutions("fear of people", "neutral", 10, "balanced")
    advanced_results, _ = get_one_word_substitutions("fear of people", "neutral", 10, "advanced")
    balanced_top = [item["word"] for item in balanced_results[:5]]
    advanced_top = [item["word"] for item in advanced_results[:5]]
    assert "anthropophobia" in advanced_top[:3], "Advanced one-word mode should keep precise technical matches near the top"
    assert not any(word in {"fearfulness", "fright", "anxiety"} for word in advanced_top[:3]), (
        "Advanced one-word mode should keep generic fear terms out of the top slots"
    )
    assert advanced_top != [] and balanced_top != []


def test_regression_constraints_returns_empty_instead_of_random_rhymes():
    results, note = get_constraint_matches("stone", "antonym", "soft", "formal", 10, "advanced")
    assert results == [], "Weak Smart Match cases should return empty results instead of noisy rhyme filler"
    assert note and "no strong" in note.lower()
