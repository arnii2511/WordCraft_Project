import logging

from fastapi import APIRouter

from ..models.request_models import OneWordRequest, OneWordResponse
from ..services.nlp.oneword_service import get_one_word_substitutions

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/oneword", response_model=OneWordResponse)
async def oneword_endpoint(payload: OneWordRequest):
    try:
        results, note = get_one_word_substitutions(
            query=payload.query,
            context=payload.context,
            limit=payload.limit,
            vocabulary_preference=payload.vocabulary_preference,
        )
        return {
            "query": payload.query,
            "results": results,
            "note": note,
        }
    except Exception as exc:  # pragma: no cover - defensive API fallback
        logger.exception("One-word substitution failed for payload=%s", payload.model_dump(), exc_info=exc)
        return {
            "query": payload.query,
            "results": [],
            "note": "One-word substitution is temporarily unavailable. Please try again in a moment.",
        }
