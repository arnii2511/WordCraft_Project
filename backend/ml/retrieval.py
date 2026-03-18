from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from backend.ml.scripts.common import build_input_text, load_jsonl, normalize_text
from backend.services.nlp import embeddings

try:  # pragma: no cover
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

DEFAULT_DATASET = "backend/ml/data/dataset_ranker.jsonl"

_LOCK = threading.Lock()
_CACHE_DATASET_PATH: str | None = None
_CACHE_DATASET_MTIME: float | None = None
_CACHE_PAYLOAD: dict[str, Any] | None = None


def _dataset_path() -> Path:
    value = os.getenv("WORDCRAFT_RETRIEVAL_DATASET", DEFAULT_DATASET)
    return Path(value)


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_enabled() -> bool:
    return _truthy(os.getenv("WORDCRAFT_ENABLE_RETRIEVAL", "0"))


def _candidate_text(candidate: dict[str, Any]) -> str:
    return str(candidate.get("text", "") or "").strip().lower()


def _extract_row_candidates(row: dict[str, Any]) -> list[tuple[str, float]]:
    candidates = row.get("candidates", []) or []
    positives = [item for item in candidates if int(item.get("label", 0)) >= 2]
    source = positives if positives else candidates
    if not source:
        return []
    ranked = sorted(
        source,
        key=lambda item: (int(item.get("label", 0)), float(item.get("model_score", 0.0) or 0.0)),
        reverse=True,
    )
    output: list[tuple[str, float]] = []
    for candidate in ranked[:8]:
        text = _candidate_text(candidate)
        if not text or " " in text:
            continue
        label = float(int(candidate.get("label", 0)))
        # Keep small prior weight from historical label relevance.
        weight = max(0.05, min(1.0, (label + 1.0) / 4.0))
        output.append((text, weight))
    return output


def _row_input_text(row: dict[str, Any]) -> str:
    existing = str(row.get("input_text", "") or "").strip()
    if existing:
        return existing
    task = str(row.get("task", "") or "").strip().lower()
    payload = row.get("input", {}) or {}
    return build_input_text(task, payload)


def _build_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    text_rows: list[str] = []

    for row in rows:
        task = str(row.get("task", "") or "").strip().lower()
        if task == "rewrite":
            continue
        candidates = _extract_row_candidates(row)
        if not candidates:
            continue
        query_text = _row_input_text(row)
        if not query_text:
            continue
        entries.append(
            {
                "id": str(row.get("id", "")),
                "task": task,
                "input": row.get("input", {}) or {},
                "query_text": query_text,
                "candidates": candidates,
            }
        )
        text_rows.append(query_text)

    if not entries:
        return {"entries": [], "matrix": np.empty((0, 1), dtype=np.float32), "faiss_index": None}

    matrix = embeddings.encode_texts(text_rows).astype(np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    faiss_index = None
    if faiss is not None:
        try:
            faiss_index = faiss.IndexFlatIP(matrix.shape[1])
            faiss_index.add(matrix)
        except Exception:
            faiss_index = None

    return {"entries": entries, "matrix": matrix, "faiss_index": faiss_index}


def _load_cached_index() -> dict[str, Any] | None:
    global _CACHE_DATASET_PATH, _CACHE_DATASET_MTIME, _CACHE_PAYLOAD
    if not _is_enabled():
        return None

    path = _dataset_path()
    if not path.exists():
        return None

    mtime = path.stat().st_mtime
    with _LOCK:
        if (
            _CACHE_PAYLOAD is not None
            and _CACHE_DATASET_PATH == path.as_posix()
            and _CACHE_DATASET_MTIME == mtime
        ):
            return _CACHE_PAYLOAD

        rows = load_jsonl(path)
        payload = _build_index(rows)
        _CACHE_PAYLOAD = payload
        _CACHE_DATASET_PATH = path.as_posix()
        _CACHE_DATASET_MTIME = mtime
        return _CACHE_PAYLOAD


def _search(index_payload: dict[str, Any], query_text: str, top_k: int) -> list[tuple[int, float]]:
    entries = index_payload.get("entries", [])
    if not entries:
        return []
    matrix = index_payload.get("matrix")
    if matrix is None or len(entries) == 0:
        return []

    top_k = max(1, min(top_k, len(entries)))
    query_vec = embeddings.embed_sentence(query_text).astype(np.float32).reshape(1, -1)

    faiss_index = index_payload.get("faiss_index")
    if faiss_index is not None:
        distances, indices = faiss_index.search(query_vec, top_k)
        output: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            output.append((int(idx), float(score)))
        return output

    # Numpy fallback (inner product since embeddings are normalized).
    sims = matrix @ query_vec[0]
    if sims.ndim == 0:
        sims = np.array([float(sims)])
    idxs = np.argsort(-sims)[:top_k]
    return [(int(idx), float(sims[idx])) for idx in idxs]


def retrieve_candidates(
    *,
    task: str,
    payload: dict[str, Any],
    existing_texts: set[str],
    top_k: int = 200,
    max_new_candidates: int = 30,
) -> list[dict[str, Any]]:
    index_payload = _load_cached_index()
    if index_payload is None:
        return []

    query_text = build_input_text(task, payload)
    neighbors = _search(index_payload, query_text=query_text, top_k=max(10, top_k))
    if not neighbors:
        return []

    entries: list[dict[str, Any]] = index_payload["entries"]
    score_by_word: dict[str, float] = {}
    count_by_word: dict[str, int] = {}

    for idx, similarity in neighbors:
        if idx < 0 or idx >= len(entries):
            continue
        entry = entries[idx]
        for candidate_text, candidate_weight in entry.get("candidates", []):
            normalized = normalize_text(candidate_text)
            if not normalized or normalized in existing_texts or " " in normalized:
                continue
            score = max(0.0, min(1.0, (similarity + 1.0) / 2.0)) * float(candidate_weight)
            score_by_word[normalized] = score_by_word.get(normalized, 0.0) + score
            count_by_word[normalized] = count_by_word.get(normalized, 0) + 1

    if not score_by_word:
        return []

    ranked = sorted(
        score_by_word.items(),
        key=lambda item: (item[1] / max(1, count_by_word.get(item[0], 1)), count_by_word.get(item[0], 0)),
        reverse=True,
    )

    output: list[dict[str, Any]] = []
    for word, total in ranked[: max(1, min(80, max_new_candidates))]:
        avg = total / max(1, count_by_word.get(word, 1))
        output.append(
            {
                "word": word,
                "score": round(max(0.02, min(0.55, avg)), 4),
                "reason": "Retrieved from semantically similar historical examples.",
                "source": "retrieval",
            }
        )
    return output


def export_retrieval_manifest(path: str = "backend/ml/models/retrieval_manifest.json") -> None:
    payload = _load_cached_index()
    if payload is None:
        raise RuntimeError("Retrieval index is unavailable. Check dataset path and retrieval settings.")
    entries = payload.get("entries", [])
    output = {
        "enabled": _is_enabled(),
        "dataset": _dataset_path().as_posix(),
        "entries": len(entries),
        "faiss": bool(payload.get("faiss_index") is not None),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


__all__ = ["retrieve_candidates", "export_retrieval_manifest"]
