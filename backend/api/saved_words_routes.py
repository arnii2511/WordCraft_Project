from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..db import db
from ..models.saved_word_models import SavedWordCreate, SavedWordResponse
from ..serializers import serialize_saved_word

router = APIRouter(prefix="/saved-words", tags=["saved-words"])


def _ensure_object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Saved word not found") from exc


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
