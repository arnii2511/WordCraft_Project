from __future__ import annotations

import re

try:
    import spacy
except ImportError:  # pragma: no cover - handled by runtime fallback
    spacy = None

try:
    from nltk.corpus import wordnet as wn
except ImportError:  # pragma: no cover - handled by runtime fallback
    wn = None

_SPACY_NLP = None

BLANK_RE = re.compile(r"(\_{2,}|\(\s*blank\s*\)|\[\s*blank\s*\]|\.{3,})", re.IGNORECASE)

CONTEXT_TEMPLATES = {
    "nostalgia": [
        "like a memory from the past",
        "as if remembering old days",
        "with the softness of yesterday",
    ],
    "horror": [
        "as though something watched from the dark",
        "with a chill that would not leave",
        "as if shadows were listening",
    ],
    "romantic": [
        "with a warmth that lingered",
        "as if drawn by gentle affection",
        "like a quiet promise",
    ],
    "academic": [
        "in a manner consistent with prior studies",
        "with measured, analytical clarity",
        "as supported by the evidence",
    ],
    "joyful": [
        "with a brightness that lifted the moment",
        "as if the world had opened up",
        "in a burst of lighthearted ease",
    ],
    "melancholic": [
        "with a sadness that hung in the air",
        "as if the day remembered its losses",
        "in a quiet, fading hush",
    ],
    "hopeful": [
        "as if something better was just ahead",
        "with a calm, rising optimism",
        "like the first light after rain",
    ],
    "mysterious": [
        "as if a secret lay just beneath the words",
        "with a quiet, hidden pull",
        "as though the answer was out of reach",
    ],
    "formal": [
        "in a manner that maintains proper tone",
        "with due consideration and restraint",
        "as appropriate to the context",
    ],
}

SOFTEN_CONTEXTS = {"nostalgia", "romantic", "melancholic", "hopeful"}


def _get_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if spacy is None:
        return None
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    except OSError:
        _SPACY_NLP = None
    return _SPACY_NLP


def _choose_template(context: str, seed: str) -> str:
    options = CONTEXT_TEMPLATES.get(context.lower(), [])
    if not options:
        return ""
    index = abs(hash(seed)) % len(options)
    return options[index]


def _get_wordnet():
    if wn is None:
        return None
    try:
        wn.synsets("test")
    except LookupError:
        return None
    return wn


def _choose_verb_synonym(verb: str) -> str:
    wordnet = _get_wordnet()
    if wordnet is None:
        return verb
    synsets = wordnet.synsets(verb, pos=wordnet.VERB)
    for synset in synsets:
        for lemma in synset.lemma_names():
            candidate = lemma.replace("_", " ").strip().lower()
            if " " in candidate:
                continue
            if candidate and candidate != verb:
                return candidate
    return verb


def _conjugate(lemma: str, tag: str) -> str:
    if tag in {"VBD", "VBN"}:
        if lemma.endswith("e"):
            return f"{lemma}d"
        return f"{lemma}ed"
    if tag == "VBZ":
        if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
            return f"{lemma[:-1]}ies"
        if lemma.endswith(("s", "sh", "ch", "x", "z")):
            return f"{lemma}es"
        return f"{lemma}s"
    if tag == "VBG":
        if lemma.endswith("e") and not lemma.endswith("ee"):
            return f"{lemma[:-1]}ing"
        return f"{lemma}ing"
    return lemma


def _replace_first_blank(text: str, word: str) -> str:
    return BLANK_RE.sub(word, text, count=1)


def _append_phrase(text: str, phrase: str) -> str:
    if not phrase:
        return text
    stripped = text.strip()
    if not stripped:
        return phrase
    if stripped[-1] in ".!?":
        return f"{stripped[:-1]}, {phrase}{stripped[-1]}"
    return f"{stripped}, {phrase}"


def _soften_verb(text: str) -> str:
    nlp = _get_spacy()
    if nlp is None:
        return text
    doc = nlp(text)
    replacement_index = None
    replacement = None
    for token in doc:
        if token.pos_ in {"VERB", "AUX"} and token.is_alpha:
            synonym = _choose_verb_synonym(token.lemma_.lower())
            replacement = _conjugate(synonym, token.tag_)
            replacement_index = token.i
            break
    if replacement_index is None or replacement is None:
        return text

    parts = []
    for token in doc:
        if token.i == replacement_index:
            parts.append(replacement + token.whitespace_)
        else:
            parts.append(token.text_with_ws)
    return "".join(parts)


def generate_rewrite(
    original_text: str,
    context: str,
    top_word: str | None,
    blank_present: bool,
) -> str:
    if not original_text:
        return ""

    rewritten = original_text.strip()
    if blank_present and top_word:
        rewritten = _replace_first_blank(rewritten, top_word)

    if context.lower() in SOFTEN_CONTEXTS:
        rewritten = _soften_verb(rewritten)

    phrase = _choose_template(context, rewritten)
    rewritten = _append_phrase(rewritten, phrase)
    return rewritten
