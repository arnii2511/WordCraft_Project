import hashlib
import json
from datetime import datetime, timezone
import logging

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..db import db
from ..models.saved_word_models import SavedWordCreate, SavedWordResponse
from ..serializers import serialize_saved_word

router = APIRouter(prefix="/saved-words", tags=["saved-words"])
logger = logging.getLogger(__name__)


def _ensure_object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Saved word not found") from exc


def _infer_feedback_task(payload: SavedWordCreate) -> str:
    source = (payload.source or "").strip().lower()
    item_type = (payload.type or "").strip().lower()
    if source == "oneword" or item_type == "oneword":
        return "oneword"
    if source == "constraints" or item_type == "smart_match":
        return "constraints"
    if source == "lexical" or item_type in {"synonyms", "antonyms", "rhymes", "homonyms"}:
        return "lexical"
    if item_type == "rewrite":
        return "editor_rewrite"
    return "editor_suggestion"


def _infer_input_payload(payload: SavedWordCreate) -> dict:
    source = (payload.source or "").strip().lower()
    item_type = (payload.type or "").strip().lower()
    related = (payload.related_to or "").strip()
    if source == "lexical" or item_type in {"synonyms", "antonyms", "rhymes", "homonyms"}:
        return {"word": related, "lexical_task": item_type}
    if source == "oneword" or item_type == "oneword":
        return {"query": related}
    if source == "constraints" or item_type == "smart_match":
        relation = ""
        meaning_of = ""
        rhyme_with = ""
        for segment in related.split("|"):
            segment = segment.strip()
            if segment.startswith("rhyme:"):
                rhyme_with = segment.split(":", 1)[1].strip()
            elif ":" in segment and not relation:
                relation, meaning_of = [part.strip() for part in segment.split(":", 1)]
        return {
            "rhyme_with": rhyme_with,
            "relation": relation or "synonym",
            "meaning_of": meaning_of,
        }
    return {"related_to": related}


def _input_key(task: str, payload: dict, context: str | None) -> str:
    normalized = {
        "task": task,
        "context": (context or "").strip().lower(),
        "mode": "",
        "input_payload": payload or {},
    }
    return hashlib.sha1(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()


@router.post("", response_model=SavedWordResponse)
async def create_saved_word(payload: SavedWordCreate, current_user=Depends(get_current_user)):
    now = datetime.now(timezone.utc)
    doc = {
        "user_id": current_user["_id"],
        "word": payload.word,
        "source": payload.source,
        "type": payload.type,
        "context": payload.context,
        "related_to": payload.related_to,
        "created_at": now,
    }
    result = await db.saved_words.insert_one(doc)
    doc["_id"] = result.inserted_id

    # Saving to favorites is an implicit positive signal.
    try:
        feedback_task = _infer_feedback_task(payload)
        feedback_input = _infer_input_payload(payload)
        await db.feedback_ratings.insert_one(
            {
                "task": feedback_task,
                "candidate": payload.word.strip(),
                "rating": 4,
                "quality": "good",
                "label": 2,
                "context": payload.context,
                "mode": None,
                "input_payload": feedback_input,
                "input_key": _input_key(feedback_task, feedback_input, payload.context),
                "input_text": payload.related_to,
                "source": "implicit_favorite",
                "pos": None,
                "model_score": None,
                "reason": "Implicit positive feedback from save/favorite action.",
                "session_id": None,
                "created_at": now,
                "user_id": current_user["_id"],
            }
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to write implicit favorite feedback: %s", exc)

    return serialize_saved_word(doc)


@router.get("", response_model=list[SavedWordResponse])
async def list_saved_words(current_user=Depends(get_current_user)):
    cursor = db.saved_words.find({"user_id": current_user["_id"]}).sort("created_at", -1)
    return [serialize_saved_word(item) async for item in cursor]


@router.delete("/{saved_word_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_saved_word(saved_word_id: str, current_user=Depends(get_current_user)):
    word_id = _ensure_object_id(saved_word_id)
    result = await db.saved_words.delete_one({"_id": word_id, "user_id": current_user["_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Saved word not found")
    return None
