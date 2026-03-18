import logging

from fastapi import APIRouter

from ..models.request_models import SuggestionRequest, SuggestionResponse
from ..services.nlp.engine import generate_suggestions

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/suggest", response_model=SuggestionResponse)
async def suggest_words(request: SuggestionRequest):
    try:
        return generate_suggestions(
            request.sentence,
            request.context,
            mode=request.mode,
            selection=request.selection,
            trigger=request.trigger,
            vocabulary_preference=request.vocabulary_preference,
        )
    except Exception as exc:  # pragma: no cover - defensive API fallback
        logger.exception("Suggestion generation failed for payload=%s", request.model_dump(), exc_info=exc)
        return {
            "suggestions": [],
            "rewrite": "",
            "rewrites": [],
            "explanation": "Suggestions are temporarily unavailable. Please try again in a moment.",
            "detected_blank": False,
        }


@router.post("/suggestions", response_model=SuggestionResponse)
async def suggest_words_legacy(request: SuggestionRequest):
    return await suggest_words(request)
