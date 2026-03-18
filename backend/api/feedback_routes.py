from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from ..auth import get_optional_user
from ..db import db
from ..models.request_models import FeedbackRequest, FeedbackResponse, ImplicitFeedbackRequest

router = APIRouter(prefix="/feedback", tags=["feedback"])


def _quality_bucket(rating: int) -> tuple[str, int]:
    if rating < 3:
        return "bad", 0
    if rating == 3:
        return "average", 1
    if rating == 4:
        return "good", 2
    return "good", 3


def _input_key(
    task: str,
    payload: dict[str, Any],
    context: str | None,
    mode: str | None,
    vocabulary_preference: str | None,
) -> str:
    normalized = {
        "task": task,
        "context": (context or "").strip().lower(),
        "mode": (mode or "").strip().lower(),
        "vocabulary_preference": (vocabulary_preference or "").strip().lower(),
        "input_payload": payload or {},
    }
    digest = hashlib.sha1(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def _implicit_mapping(action: str) -> tuple[int, str, int, str]:
    normalized = (action or "").strip().lower()
    if normalized == "inserted":
        return 5, "good", 3, "implicit_insert"
    if normalized == "copied":
        return 4, "good", 2, "implicit_copy"
    return 4, "good", 2, "implicit_favorite"


@router.post("", response_model=FeedbackResponse)
async def create_feedback(payload: FeedbackRequest, current_user=Depends(get_optional_user)):
    quality, label = _quality_bucket(payload.rating)
    now = datetime.now(timezone.utc)
    key = _input_key(
        payload.task,
        payload.input_payload,
        payload.context,
        payload.mode,
        payload.vocabulary_preference,
    )

    doc = {
        "task": payload.task,
        "candidate": payload.candidate.strip(),
        "rating": payload.rating,
        "quality": quality,
        "label": label,
        "context": payload.context,
        "mode": payload.mode,
        "input_payload": payload.input_payload or {},
        "input_key": key,
        "input_text": payload.input_text,
        "vocabulary_preference": payload.vocabulary_preference,
        "source": payload.source,
        "pos": payload.pos,
        "model_score": payload.model_score,
        "reason": payload.reason,
        "session_id": payload.session_id,
        "created_at": now,
        "user_id": current_user.get("_id") if current_user else None,
    }

    result = await db.feedback_ratings.insert_one(doc)
    return {
        "id": str(result.inserted_id),
        "task": payload.task,
        "candidate": payload.candidate.strip(),
        "rating": payload.rating,
        "quality": quality,
        "label": label,
        "message": "Feedback saved",
    }


@router.post("/implicit", response_model=FeedbackResponse)
async def create_implicit_feedback(payload: ImplicitFeedbackRequest, current_user=Depends(get_optional_user)):
    rating, quality, label, source = _implicit_mapping(payload.action)
    now = datetime.now(timezone.utc)
    key = _input_key(
        payload.task,
        payload.input_payload,
        payload.context,
        payload.mode,
        payload.vocabulary_preference,
    )

    doc = {
        "task": payload.task,
        "candidate": payload.candidate.strip(),
        "rating": rating,
        "quality": quality,
        "label": label,
        "context": payload.context,
        "mode": payload.mode,
        "input_payload": payload.input_payload or {},
        "input_key": key,
        "input_text": payload.input_text,
        "vocabulary_preference": payload.vocabulary_preference,
        "source": source,
        "pos": payload.pos,
        "model_score": payload.model_score,
        "reason": payload.reason or f"Implicit feedback from {payload.action} action.",
        "session_id": payload.session_id,
        "created_at": now,
        "user_id": current_user.get("_id") if current_user else None,
    }
    result = await db.feedback_ratings.insert_one(doc)
    return {
        "id": str(result.inserted_id),
        "task": payload.task,
        "candidate": payload.candidate.strip(),
        "rating": rating,
        "quality": quality,
        "label": label,
        "message": "Implicit feedback saved",
    }


@router.get("/stats")
async def feedback_stats():
    total = await db.feedback_ratings.count_documents({})
    task_cursor = db.feedback_ratings.aggregate(
        [
            {"$group": {"_id": "$task", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
    )
    source_cursor = db.feedback_ratings.aggregate(
        [
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
    )
    by_task = [{ "task": item.get("_id"), "count": item.get("count", 0)} async for item in task_cursor]
    by_source = [{ "source": item.get("_id"), "count": item.get("count", 0)} async for item in source_cursor]
    return {
        "total_feedback_events": total,
        "by_task": by_task,
        "by_source": by_source,
    }
