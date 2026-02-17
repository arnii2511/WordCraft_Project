from datetime import datetime
from typing import Any


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def serialize_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(user.get("_id")),
        "email": user.get("email"),
        "username": user.get("username"),
        "phone": user.get("phone"),
        "bio": user.get("bio"),
        "interests": user.get("interests"),
        "created_at": _iso(user.get("created_at")),
    }


def serialize_document(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(doc.get("_id")),
        "title": doc.get("title"),
        "content_html": doc.get("content_html", ""),
        "content_text": doc.get("content_text", ""),
        "context": doc.get("context", "neutral"),
        "mode": doc.get("mode", "write"),
        "created_at": _iso(doc.get("created_at")),
        "updated_at": _iso(doc.get("updated_at")),
    }


def serialize_saved_word(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(doc.get("_id")),
        "word": doc.get("word"),
        "source": doc.get("source", "suggest"),
        "type": doc.get("type", "suggestion"),
        "context": doc.get("context"),
        "related_to": doc.get("related_to"),
        "created_at": _iso(doc.get("created_at")),
    }
