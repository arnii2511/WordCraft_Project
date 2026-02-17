from datetime import datetime, timezone

from bson import ObjectId
from pymongo import ReturnDocument
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..db import db
from ..models.document_models import DocumentCreate, DocumentResponse, DocumentUpdate
from ..serializers import serialize_document

router = APIRouter(prefix="/documents", tags=["documents"])


def _ensure_object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc


@router.post("", response_model=DocumentResponse)
async def create_document(payload: DocumentCreate, current_user=Depends(get_current_user)):
    now = datetime.now(timezone.utc)
    doc = {
        "user_id": current_user["_id"],
        "title": payload.title,
        "content_html": payload.content_html,
        "content_text": payload.content_text,
        "context": payload.context,
        "mode": payload.mode,
        "created_at": now,
        "updated_at": now,
    }
    result = await db.documents.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize_document(doc)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(current_user=Depends(get_current_user)):
    cursor = db.documents.find({"user_id": current_user["_id"]}).sort("updated_at", -1)
    docs = [serialize_document(doc) async for doc in cursor]
    return docs


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, current_user=Depends(get_current_user)):
    doc_id = _ensure_object_id(document_id)
    doc = await db.documents.find_one({"_id": doc_id, "user_id": current_user["_id"]})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return serialize_document(doc)


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    payload: DocumentUpdate,
    current_user=Depends(get_current_user),
):
    doc_id = _ensure_object_id(document_id)
    raw_data = (
        payload.model_dump(exclude_unset=True)
        if hasattr(payload, "model_dump")
        else payload.dict(exclude_unset=True)
    )
    update_data = {key: value for key, value in raw_data.items() if value is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided")
    update_data["updated_at"] = datetime.now(timezone.utc)

    result = await db.documents.find_one_and_update(
        {"_id": doc_id, "user_id": current_user["_id"]},
        {"$set": update_data},
        return_document=ReturnDocument.AFTER,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    return serialize_document(result)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, current_user=Depends(get_current_user)):
    doc_id = _ensure_object_id(document_id)
    result = await db.documents.delete_one({"_id": doc_id, "user_id": current_user["_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return None
