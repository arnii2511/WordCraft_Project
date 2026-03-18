import logging

from fastapi import APIRouter

from ..models.request_models import LexicalRequest, LexicalResponse
from ..services.nlp.lexical_service import get_lexical_results

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/lexical", response_model=LexicalResponse)
async def lexical_tools(request: LexicalRequest):
    try:
        results, details = get_lexical_results(
            request.word,
            request.task,
            context=request.context,
            vocabulary_preference=request.vocabulary_preference,
        )
        return {
            "word": request.word,
            "task": request.task,
            "results": results,
            "details": details,
        }
    except Exception as exc:  # pragma: no cover - defensive API fallback
        logger.exception("Lexical lookup failed for payload=%s", request.model_dump(), exc_info=exc)
        return {
            "word": request.word,
            "task": request.task,
            "results": [],
            "details": [],
        }
