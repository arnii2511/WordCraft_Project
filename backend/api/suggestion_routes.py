from fastapi import APIRouter

from ..models.request_models import SuggestionRequest, SuggestionResponse
from ..services.nlp.engine import generate_suggestions

router = APIRouter()


@router.post("/suggest", response_model=SuggestionResponse)
async def suggest_words(request: SuggestionRequest):
    return generate_suggestions(
        request.sentence,
        request.context,
        mode=request.mode,
        selection=request.selection,
        trigger=request.trigger,
        vocabulary_preference=request.vocabulary_preference,
    )


@router.post("/suggestions", response_model=SuggestionResponse)
async def suggest_words_legacy(request: SuggestionRequest):
    return generate_suggestions(
        request.sentence,
        request.context,
        mode=request.mode,
        selection=request.selection,
        trigger=request.trigger,
        vocabulary_preference=request.vocabulary_preference,
    )
