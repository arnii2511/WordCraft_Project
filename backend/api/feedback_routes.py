from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from ..auth import get_optional_user
from ..db import db
from ..models.request_models import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


def _quality_bucket(rating: int) -> tuple[str, int]:
    if rating < 3:
        return "bad", 0
    if rating == 3:
        return "average", 1
    if rating == 4:
        return "good", 2
    return "good", 3


def _input_key(task: str, payload: dict[str, Any], context: str | None, mode: str | None) -> str:
    normalized = {
        "task": task,
        "context": (context or "").strip().lower(),
        "mode": (mode or "").strip().lower(),
        "input_payload": payload or {},
    }
    digest = hashlib.sha1(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


@router.post("", response_model=FeedbackResponse)
async def create_feedback(payload: FeedbackRequest, current_user=Depends(get_optional_user)):
    quality, label = _quality_bucket(payload.rating)
    now = datetime.now(timezone.utc)
    key = _input_key(payload.task, payload.input_payload, payload.context, payload.mode)

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
