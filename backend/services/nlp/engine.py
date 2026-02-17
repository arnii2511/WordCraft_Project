from __future__ import annotations

import logging
from typing import Any

from . import embeddings, explanation, ranker
from .context_loader import load_contexts
from .emotion_service import load_lexicon
from .pipeline import build_pipeline
from .rewrite_service import is_sentence_complete, rewrite_variants

logger = logging.getLogger(__name__)

_CONTEXTS: dict[str, dict] | None = None
_INITIALIZED = False
_INITIALIZATION_ERROR: Exception | None = None


def initialize() -> None:
    global _CONTEXTS, _INITIALIZED, _INITIALIZATION_ERROR
    if _INITIALIZED or _INITIALIZATION_ERROR:
        return
    try:
        _CONTEXTS = load_contexts()
        embeddings.ensure_context_embeddings(_CONTEXTS)
        load_lexicon()
        _INITIALIZED = True
    except Exception as exc:  # pragma: no cover
        _INITIALIZATION_ERROR = exc
        logger.exception("NLP initialization failed", exc_info=exc)


def _fallback_response(detected_blank: bool) -> dict[str, Any]:
    return {
        "suggestions": [],
        "rewrite": "",
        "rewrites": [],
        "explanation": "Unable to generate suggestions at the moment.",
        "detected_blank": detected_blank,
    }


def _resolve_context_key(context: str, contexts: dict[str, dict]) -> str:
    key = (context or "").strip().lower()
    if key in contexts:
        return key
    if "neutral" in contexts:
        return "neutral"
    return next(iter(contexts))


def _mode_weights(mode: str, intent: str) -> dict[str, float]:
    if mode == "edit":
        return {"semantic": 0.34, "context": 0.12, "emotion": 0.04, "grammar": 0.36, "frequency": 0.14}
    if mode == "rewrite":
        return {"semantic": 0.44, "context": 0.24, "emotion": 0.08, "grammar": 0.16, "frequency": 0.08}
    if intent == "blank":
        return {"semantic": 0.34, "context": 0.16, "emotion": 0.08, "grammar": 0.34, "frequency": 0.08}
    return {"semantic": 0.45, "context": 0.23, "emotion": 0.08, "grammar": 0.16, "frequency": 0.08}


def _should_rewrite(
    text: str,
    mode: str,
    trigger: str,
    blank_present: bool,
) -> bool:
    if blank_present:
        return False
    if not is_sentence_complete(text or "", blank_present=False):
        return False
    if mode in {"write", "edit"}:
        return True
    return mode == "rewrite" and trigger == "button"


def generate_suggestions(
    text: str,
    context: str,
    mode: str = "write",
    selection: Any | None = None,
    trigger: str = "auto",
) -> dict[str, Any]:
    initialize()
    if not _INITIALIZED or _CONTEXTS is None:
        return _fallback_response(False)

    mode_key = (mode or "write").strip().lower()
    context_key = _resolve_context_key(context, _CONTEXTS)
    raw_text = text or ""

    pipeline = build_pipeline(
        text=raw_text,
        context_key=context_key,
        mode=mode_key,
        contexts=_CONTEXTS,
        selection=selection,
    )
    cleaned_text = pipeline.decision.cleaned_text
    blank_present = pipeline.decision.blank_present
    if not cleaned_text:
        return _fallback_response(blank_present)
    if not pipeline.candidates:
        return _fallback_response(blank_present)

    ranked = ranker.rank_candidates(
        cleaned_text=cleaned_text,
        context_key=context_key,
        candidates=pipeline.candidates,
        context_description=pipeline.context_description,
        blank_present=blank_present and pipeline.decision.intent != "selection",
        emotion_scores=pipeline.emotion_scores,
        weights=_mode_weights(mode_key, pipeline.decision.intent),
        top_k=5,
        source_map=pipeline.source_map,
        strict_pos=pipeline.decision.strict_pos,
        expected_pos_override=pipeline.decision.expected_pos,
        context_words=set(_CONTEXTS.get(context_key, {}).get("words", [])),
    )
    top_words = [item["word"] for item in ranked]

    allow_rewrite = _should_rewrite(
        text=raw_text,
        mode=mode_key,
        trigger=trigger,
        blank_present=blank_present,
    )
    rewrite_candidates = rewrite_variants(
        sentence=raw_text,
        context=context_key,
        mode=mode_key,
        blank_present=blank_present,
        allow_rewrite=allow_rewrite,
        suggestions=top_words,
    )
    rewrite_text = rewrite_candidates[0] if rewrite_candidates else ""

    explanation_text = explanation.generate_explanation(
        context=context_key,
        description=pipeline.context_description,
        words=top_words,
        mode=mode_key,
        blank_present=blank_present and pipeline.decision.intent != "selection",
        selection_present=pipeline.decision.intent == "selection",
        intent=pipeline.decision.intent,
    )

    return {
        "suggestions": ranked,
        "rewrite": rewrite_text,
        "rewrites": rewrite_candidates,
        "explanation": explanation_text,
        "detected_blank": blank_present,
    }
