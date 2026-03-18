from __future__ import annotations

import json
import os
import pickle
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from backend.ml.retrieval import retrieve_candidates
from backend.services.nlp import embeddings
from backend.services.nlp.context_loader import load_contexts
from backend.services.nlp.runtime_profile import cross_encoder_enabled, retrieval_enabled

DEFAULT_ARTIFACT = "backend/ml/models/reranker.pkl"
DEFAULT_BEHAVIOR_PRIORS = "backend/ml/models/behavior_priors.json"

_CACHE_LOCK = threading.Lock()
_CACHED_MTIME: float | None = None
_CACHED_ARTIFACT: dict[str, Any] | None = None
_CACHED_BEHAVIOR_PRIORS_MTIME: float | None = None
_CACHED_BEHAVIOR_PRIORS: dict[str, Any] | None = None
_CACHED_CONTEXTS: dict[str, dict] | None = None
_CACHED_CROSS_ENCODERS: dict[str, Any] = {}
_BLANK_RE = re.compile(r"_{3,}|\[blank\]|<blank>|\(blank\)", re.IGNORECASE)


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _artifact_path() -> Path:
    path = os.getenv("WORDCRAFT_RERANKER_ARTIFACT", DEFAULT_ARTIFACT)
    return Path(path)


def _behavior_path() -> Path:
    path = os.getenv("WORDCRAFT_BEHAVIOR_PRIORS", DEFAULT_BEHAVIOR_PRIORS)
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
        artifact_type = payload.get("artifact_type", "sklearn_tfidf_logreg")
        if artifact_type == "sklearn_tfidf_logreg":
            if "vectorizer" not in payload or "model" not in payload:
                _CACHED_ARTIFACT = None
                _CACHED_MTIME = None
                return None
        elif artifact_type.startswith("cross_encoder"):
            if "model_path" not in payload:
                _CACHED_ARTIFACT = None
                _CACHED_MTIME = None
                return None
        else:
            _CACHED_ARTIFACT = None
            _CACHED_MTIME = None
            return None
        _CACHED_ARTIFACT = payload
        _CACHED_MTIME = mtime
        return _CACHED_ARTIFACT


def _load_behavior_priors() -> dict[str, Any] | None:
    global _CACHED_BEHAVIOR_PRIORS_MTIME, _CACHED_BEHAVIOR_PRIORS
    path = _behavior_path()
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    with _CACHE_LOCK:
        if _CACHED_BEHAVIOR_PRIORS is not None and _CACHED_BEHAVIOR_PRIORS_MTIME == mtime:
            return _CACHED_BEHAVIOR_PRIORS
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _CACHED_BEHAVIOR_PRIORS = None
            _CACHED_BEHAVIOR_PRIORS_MTIME = None
            return None
        _CACHED_BEHAVIOR_PRIORS = payload
        _CACHED_BEHAVIOR_PRIORS_MTIME = mtime
        return _CACHED_BEHAVIOR_PRIORS


def _load_contexts() -> dict[str, dict]:
    global _CACHED_CONTEXTS
    if _CACHED_CONTEXTS is not None:
        return _CACHED_CONTEXTS
    try:
        _CACHED_CONTEXTS = load_contexts()
    except Exception:
        _CACHED_CONTEXTS = {}
    return _CACHED_CONTEXTS


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


def _cross_encoder_scores(artifact: dict[str, Any], texts: list[str]) -> np.ndarray:
    if not cross_encoder_enabled():
        raise RuntimeError("cross-encoder disabled for current runtime profile")
    model_path = artifact.get("model_path")
    if not model_path:
        raise ValueError("cross-encoder artifact missing model_path")
    from sentence_transformers.cross_encoder import CrossEncoder

    with _CACHE_LOCK:
        model = _CACHED_CROSS_ENCODERS.get(model_path)
        if model is None:
            model = CrossEncoder(model_path)
            _CACHED_CROSS_ENCODERS[model_path] = model
    predictions = model.predict(texts)
    values = np.asarray(predictions, dtype=np.float32)
    if values.ndim == 2:
        values = values.reshape(-1)
    # Map raw score to [0, 1].
    return 1.0 / (1.0 + np.exp(-values))


def _parse_expected_pos(payload: dict[str, Any]) -> str | None:
    sentence = str(payload.get("sentence", "") or "").lower()
    match = _BLANK_RE.search(sentence)
    if not match:
        return None
    left = sentence[: match.start()]
    if re.search(r"\bto\s*$", left):
        return "VERB"
    if re.search(r"\b(is|are|was|were|be|been|seem|seems|feel|feels|look|looks)\s*$", left):
        return "ADJ"
    if re.search(r"\b\w+(ed|ing)\s*$", left):
        return "ADV"
    return None


def _pos_match(payload: dict[str, Any], candidate: dict[str, Any]) -> bool | None:
    expected = _parse_expected_pos(payload)
    if not expected:
        return None
    found = str(candidate.get("pos", "") or "").upper().strip()
    if not found:
        return None
    return found == expected


def _safe_norm(vec: np.ndarray | None) -> float:
    if vec is None:
        return 0.0
    return float(np.linalg.norm(vec))


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = _safe_norm(a) * _safe_norm(b)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _semantic_similarity(payload: dict[str, Any], candidate_text: str) -> float:
    query_text = json.dumps(payload, ensure_ascii=True)
    q_vec = embeddings.embed_sentence(query_text)
    c_vec = embeddings.get_word_embedding(candidate_text)
    return max(0.0, min(1.0, (_cosine(q_vec, c_vec) + 1.0) / 2.0))


def _tone_similarity(payload: dict[str, Any], candidate_text: str) -> float:
    context = str(payload.get("context", "") or "").strip().lower()
    if not context:
        return 0.5
    contexts = _load_contexts()
    context_words = set(contexts.get(context, {}).get("words", []))
    if candidate_text in context_words:
        return 1.0
    c_vec = embeddings.get_context_centroid(context)
    w_vec = embeddings.get_word_embedding(candidate_text)
    return max(0.0, min(1.0, (_cosine(c_vec, w_vec) + 1.0) / 2.0))


def _lookup_behavior_score(task: str, candidate_text: str) -> float | None:
    priors = _load_behavior_priors()
    if not priors:
        return None
    task_key = str(task or "").strip().lower()
    candidate_key = str(candidate_text or "").strip().lower()
    per_task = priors.get("task_candidate", {}).get(task_key, {})
    value = per_task.get(candidate_key)
    if isinstance(value, dict):
        score = value.get("score")
        if score is not None:
            return float(score)
    global_map = priors.get("global_candidate", {})
    value = global_map.get(candidate_key)
    if isinstance(value, dict):
        score = value.get("score")
        if score is not None:
            return float(score)
    return None


def _calibrate_probability(probability: float, temperature: float) -> float:
    p = min(max(probability, 1e-6), 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    temp = max(0.2, min(8.0, float(temperature)))
    scaled = 1.0 / (1.0 + np.exp(-(logit / temp)))
    return float(min(max(scaled, 0.0), 1.0))


def _normalize_scores(rows: list[dict[str, Any]], score_key: str) -> None:
    if not rows:
        return
    values = [float(row.get(score_key, 0.0) or 0.0) for row in rows]
    low = min(values)
    high = max(values)
    if high - low < 1e-6:
        return
    for row in rows:
        value = float(row.get(score_key, 0.0) or 0.0)
        row[score_key] = round((value - low) / (high - low), 4)


def _merge_retrieval_candidates(
    task: str,
    payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    text_key: str,
    score_key: str,
    max_extra: int = 24,
) -> list[dict[str, Any]]:
    existing = {str(item.get(text_key, "")).strip().lower() for item in candidates if item.get(text_key)}
    retrieved = retrieve_candidates(
        task=task,
        payload=payload,
        existing_texts=existing,
        top_k=200,
        max_new_candidates=max_extra,
    )
    if not retrieved:
        return candidates
    merged = list(candidates)
    for item in retrieved:
        word = (item.get("word") or "").strip()
        if not word:
            continue
        merged.append(
            {
                text_key: word,
                score_key: float(item.get("score", 0.0) or 0.0),
                "reason": item.get("reason", "Retrieved from similar examples."),
                "source": item.get("source", "retrieval"),
            }
        )
    return merged


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

    if retrieval_enabled():
        candidates = _merge_retrieval_candidates(
            task=task,
            payload=payload,
            candidates=candidates,
            text_key=text_key,
            score_key=score_key,
        )

    artifact = _load_artifact()
    if artifact is None:
        return candidates[:max_results] if max_results else candidates

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
        artifact_type = artifact.get("artifact_type", "sklearn_tfidf_logreg")
        if artifact_type == "sklearn_tfidf_logreg":
            vectorizer = artifact["vectorizer"]
            model = artifact["model"]
            x_vec = vectorizer.transform(texts)
            prob = model.predict_proba(x_vec)
            raw_scores = _prob_to_score(prob, model.classes_)
            max_prob = np.max(prob, axis=1)
            max_label = float(max(model.classes_)) if len(model.classes_) else 3.0
        elif artifact_type.startswith("cross_encoder"):
            raw_scores = _cross_encoder_scores(artifact, texts)
            max_prob = raw_scores
            max_label = 1.0
        else:
            return candidates[:max_results] if max_results else candidates
    except Exception:
        return candidates[:max_results] if max_results else candidates

    if max_label <= 0.0:
        max_label = 3.0
    safe_blend = min(max(blend, 0.0), 1.0)
    behavior_blend = min(max(float(os.getenv("WORDCRAFT_BEHAVIOR_BLEND", "0.3")), 0.0), 0.6)
    temperature = float(os.getenv("WORDCRAFT_CONFIDENCE_TEMPERATURE", "1.0"))

    reranked: list[dict[str, Any]] = []
    for idx, (item, raw_score) in enumerate(zip(active_rows, raw_scores)):
        ml_score = float(raw_score) / max_label
        ml_score = min(max(ml_score, 0.0), 1.0)
        base_score = float(item.get(score_key, 0.0) or 0.0)
        behavior_score = _lookup_behavior_score(task, str(item.get(text_key, "")).strip().lower())
        if behavior_score is None:
            blended_ml = (safe_blend * ml_score) + ((1.0 - safe_blend) * base_score)
            combined = blended_ml
            behavior_value = None
        else:
            combined = (0.7 * ml_score) + (0.3 * float(behavior_score))
            behavior_value = float(behavior_score)

        confidence = _calibrate_probability(float(max_prob[idx]), temperature=temperature)
        semantic_similarity = _semantic_similarity(payload, str(item.get(text_key, "")).strip().lower())
        tone_match = _tone_similarity(payload, str(item.get(text_key, "")).strip().lower())
        pos_match = _pos_match(payload, item)

        next_item = {**item}
        next_item["ml_score"] = round(ml_score, 4)
        if behavior_value is not None:
            next_item["behavior_score"] = round(behavior_value, 4)
        next_item["confidence"] = round(confidence, 4)
        next_item["score_breakdown"] = {
            "ml_score": round(ml_score, 4),
            "base_score": round(base_score, 4),
            "behavior_score": round(behavior_value, 4) if behavior_value is not None else None,
            "pos_match": pos_match,
            "semantic_similarity": round(semantic_similarity, 4),
            "tone_match": round(tone_match, 4),
            "blend": {
                "model_blend": round(safe_blend, 4),
                "behavior_blend": round(behavior_blend, 4),
            },
        }
        next_item[score_key] = round(combined, 4)
        reranked.append(next_item)

    if _truthy_env(os.getenv("WORDCRAFT_NORMALIZE_FINAL_SCORE", "1")):
        _normalize_scores(reranked, score_key=score_key)
    reranked.sort(key=lambda row: row.get(score_key, 0.0), reverse=True)
    if max_results is not None:
        return reranked[: max(1, max_results)]
    return reranked


__all__ = ["rerank_candidate_dicts"]
