import logging

from fastapi import APIRouter

from ..models.request_models import ConstraintRequest, ConstraintResponse
from ..services.nlp.constraints_service import get_constraint_matches

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/constraints", response_model=ConstraintResponse)
async def constraints_endpoint(payload: ConstraintRequest):
    try:
        results, note = get_constraint_matches(
            rhyme_with=payload.rhyme_with,
            relation=payload.relation,
            meaning_of=payload.meaning_of,
            context=payload.context,
            limit=payload.limit,
            vocabulary_preference=payload.vocabulary_preference,
        )
        return {"results": results, "notes": note}
    except Exception as exc:  # pragma: no cover - defensive API fallback
        logger.exception("Smart Match failed for payload=%s", payload.model_dump(), exc_info=exc)
        return {
            "results": [],
            "notes": "Smart Match is temporarily unavailable. Please try again in a moment.",
        }
