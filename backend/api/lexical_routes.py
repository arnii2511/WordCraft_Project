from fastapi import APIRouter

from ..models.request_models import LexicalRequest, LexicalResponse
from ..services.nlp.lexical_service import get_lexical_results

router = APIRouter()


@router.post("/lexical", response_model=LexicalResponse)
async def lexical_tools(request: LexicalRequest):
    results, details = get_lexical_results(request.word, request.task, context=request.context)
    return {
        "word": request.word,
        "task": request.task,
        "results": results,
        "details": details,
    }
