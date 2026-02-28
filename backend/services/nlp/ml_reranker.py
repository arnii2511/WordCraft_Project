from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ARTIFACT = "backend/ml/models/reranker.pkl"

_CACHE_LOCK = threading.Lock()
_CACHED_MTIME: float | None = None
_CACHED_ARTIFACT: dict[str, Any] | None = None


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _artifact_path() -> Path:
    path = os.getenv("WORDCRAFT_RERANKER_ARTIFACT", DEFAULT_ARTIFACT)
    return Path(path)


def _load_artifact() -> dict[str, Any] | None:
    global _CACHED_MTIME, _CACHED_ARTIFACT
    artifact_path = _artifact_path()
    if not artifact_path.exists():
        return None

    mtime = artifact_path.stat().st_mtime
    with _CACHE_LOCK:
        if _CACHED_ARTIFACT is not None and _CACHED_MTIME == mtime:
            return _CACHED_ARTIFACT
        try:
            payload = pickle.loads(artifact_path.read_bytes())
        except Exception:
            _CACHED_ARTIFACT = None
            _CACHED_MTIME = None
            return None
        if "vectorizer" not in payload or "model" not in payload:
            _CACHED_ARTIFACT = None
            _CACHED_MTIME = None
            return None
        _CACHED_ARTIFACT = payload
        _CACHED_MTIME = mtime
        return _CACHED_ARTIFACT


def _feature_text(task: str, payload: dict[str, Any], candidate: dict[str, Any], text_key: str) -> str:
    mode = payload.get("mode", "")
    context = payload.get("context", "")
    reason = candidate.get("reason") or candidate.get("note") or ""
    pos = candidate.get("pos") or ""
    source = candidate.get("source") or ""
    candidate_text = candidate.get(text_key, "")
    return (
        f"task={task} mode={mode} context={context} "
        f"input={payload} candidate={candidate_text} pos={pos} source={source} reason={reason}"
    )


def _prob_to_score(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_values = classes.astype(float)
    return probabilities @ class_values


def rerank_candidate_dicts(
    task: str,
    payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    text_key: str = "word",
    score_key: str = "score",
    blend: float = 0.75,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates
    if _truthy_env(os.getenv("WORDCRAFT_DISABLE_RERANKER")):
        return candidates[:max_results] if max_results else candidates

    artifact = _load_artifact()
    if artifact is None:
        return candidates[:max_results] if max_results else candidates

    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    texts: list[str] = []
    active_rows: list[dict[str, Any]] = []
    for item in candidates:
        word = (item.get(text_key) or "").strip()
        if not word:
            continue
        texts.append(_feature_text(task, payload, item, text_key=text_key))
        active_rows.append(item)

    if not active_rows:
        return candidates[:max_results] if max_results else candidates

    try:
        x_vec = vectorizer.transform(texts)
        prob = model.predict_proba(x_vec)
        raw_scores = _prob_to_score(prob, model.classes_)
    except Exception:
        return candidates[:max_results] if max_results else candidates

    max_label = float(max(model.classes_)) if len(model.classes_) else 3.0
    if max_label <= 0.0:
        max_label = 3.0
    safe_blend = min(max(blend, 0.0), 1.0)

    reranked: list[dict[str, Any]] = []
    for item, raw_score in zip(active_rows, raw_scores):
        ml_score = float(raw_score) / max_label
        ml_score = min(max(ml_score, 0.0), 1.0)
        base_score = float(item.get(score_key, 0.0) or 0.0)
        combined = (safe_blend * ml_score) + ((1.0 - safe_blend) * base_score)
        next_item = {**item}
        next_item["ml_score"] = round(ml_score, 4)
        next_item[score_key] = round(combined, 4)
        reranked.append(next_item)

    reranked.sort(key=lambda row: row.get(score_key, 0.0), reverse=True)
    if max_results is not None:
        return reranked[: max(1, max_results)]
    return reranked


__all__ = ["rerank_candidate_dicts"]
