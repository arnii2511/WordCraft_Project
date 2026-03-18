from __future__ import annotations

import os


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

