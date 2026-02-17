from fastapi import APIRouter

from ..models.request_models import ConstraintRequest, ConstraintResponse
from ..services.nlp.constraints_service import get_constraint_matches

router = APIRouter()


@router.post("/constraints", response_model=ConstraintResponse)
async def constraints_endpoint(payload: ConstraintRequest):
    results, note = get_constraint_matches(
        rhyme_with=payload.rhyme_with,
        relation=payload.relation,
        meaning_of=payload.meaning_of,
        context=payload.context,
        limit=payload.limit,
    )
    return {"results": results, "notes": note}
