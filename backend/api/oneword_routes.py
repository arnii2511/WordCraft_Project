from fastapi import APIRouter

from ..models.request_models import OneWordRequest, OneWordResponse
from ..services.nlp.oneword_service import get_one_word_substitutions

router = APIRouter()


@router.post("/oneword", response_model=OneWordResponse)
async def oneword_endpoint(payload: OneWordRequest):
    results, note = get_one_word_substitutions(
        query=payload.query,
        context=payload.context,
        limit=payload.limit,
    )
    return {
        "query": payload.query,
        "results": results,
        "note": note,
    }
