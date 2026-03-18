from __future__ import annotations

import os


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


_SUGGESTION_TASKS = {"suggest_blank", "suggest_selection", "suggest_sentence"}


def ml_profile() -> str:
    return (os.getenv("WORDCRAFT_ML_PROFILE") or "").strip().lower()


def is_free_host_profile() -> bool:
    return ml_profile() in {"free", "free-host", "render-free"}


def transformers_enabled() -> bool:
    if _truthy(os.getenv("WORDCRAFT_DISABLE_TRANSFORMERS")):
        return False
    return not is_free_host_profile()


def spacy_enabled() -> bool:
    if _truthy(os.getenv("WORDCRAFT_DISABLE_SPACY")):
        return False
    return not is_free_host_profile()


def conceptnet_runtime_enabled(default: bool = False) -> bool:
    if is_free_host_profile():
        return False
    value = os.getenv("WORDCRAFT_ENABLE_CONCEPTNET_RUNTIME")
    if value is None:
        return default
    return _truthy(value)


def retrieval_enabled() -> bool:
    if is_free_host_profile():
        return False
    value = os.getenv("WORDCRAFT_ENABLE_RETRIEVAL")
    if value is None:
        return False
    return _truthy(value)


def cross_encoder_enabled() -> bool:
    if _truthy(os.getenv("WORDCRAFT_DISABLE_CROSS_ENCODER")):
        return False
    return not is_free_host_profile()


def rewrite_variant_limit(default: int = 3) -> int:
    return 1 if is_free_host_profile() else default


def semantic_embeddings_enabled_for_task(task: str) -> bool:
    normalized = (task or "").strip().lower()
    if _truthy(os.getenv("WORDCRAFT_DISABLE_SEMANTIC_EMBEDDINGS")):
        return False
    if normalized == "lexical" and not _truthy(os.getenv("WORDCRAFT_ENABLE_LEXICAL_EMBEDDINGS")):
        return False
    if not is_free_host_profile():
        return True
    return normalized in _SUGGESTION_TASKS


def reranker_enabled_for_task(task: str) -> bool:
    normalized = (task or "").strip().lower()
    if normalized == "lexical":
        return False
    if not is_free_host_profile():
        return True
    return normalized in _SUGGESTION_TASKS


def retrieval_enabled_for_task(task: str) -> bool:
    if not retrieval_enabled():
        return False
    normalized = (task or "").strip().lower()
    return normalized in _SUGGESTION_TASKS


def cross_encoder_enabled_for_task(task: str) -> bool:
    if not cross_encoder_enabled():
        return False
    normalized = (task or "").strip().lower()
    return normalized in _SUGGESTION_TASKS
