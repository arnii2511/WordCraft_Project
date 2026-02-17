from pydantic import BaseModel, Field


class SavedWordCreate(BaseModel):
    word: str = Field(..., min_length=1)
    source: str = "suggest"
    type: str = "suggestion"
    context: str | None = None
    related_to: str | None = None


class SavedWordResponse(BaseModel):
    id: str
    word: str
    source: str
    type: str
    context: str | None = None
    related_to: str | None = None
    created_at: str | None = None
