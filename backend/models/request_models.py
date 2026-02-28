from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SelectionSpan(BaseModel):
    text: str
    start: int
    end: int


class SuggestionRequest(BaseModel):
    sentence: str = Field(..., min_length=1)
    context: str = Field(default="neutral")
    mode: Literal["write", "edit", "rewrite"] = "write"
    selection: SelectionSpan | None = None
    trigger: Literal["auto", "button"] = "auto"


class SuggestionItem(BaseModel):
    word: str
    score: float
    pos: str | None = None
    note: str | None = None


class SuggestionResponse(BaseModel):
    suggestions: list[SuggestionItem]
    rewrite: str = ""
    rewrites: list[str] = []
    explanation: str = ""
    detected_blank: bool = False


class LexicalRequest(BaseModel):
    word: str = Field(..., min_length=1)
    task: Literal["synonyms", "antonyms", "homonyms", "rhymes"]
    context: str | None = None


class LexicalResult(BaseModel):
    word: str
    score: float
    pos: str | None = None
    reason: str


class LexicalResponse(BaseModel):
    word: str
    task: Literal["synonyms", "antonyms", "homonyms", "rhymes"]
    results: list[str]
    details: list[LexicalResult] = []


class ConstraintRequest(BaseModel):
    rhyme_with: str = Field(..., min_length=1)
    relation: Literal["synonym", "antonym"]
    meaning_of: str = Field(..., min_length=1)
    context: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class ConstraintResult(BaseModel):
    word: str
    score: float
    rhyme: bool
    relation_match: bool
    reason: str


class ConstraintResponse(BaseModel):
    results: list[ConstraintResult]
    notes: str | None = None


class OneWordRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: str | None = None
    limit: int = Field(default=10, ge=1, le=10)


class OneWordResult(BaseModel):
    word: str
    score: float
    reason: str
    meaning: str | None = None


class OneWordResponse(BaseModel):
    query: str
    results: list[OneWordResult]
    note: str | None = None


class FeedbackRequest(BaseModel):
    task: Literal["editor_suggestion", "editor_rewrite", "lexical", "constraints", "oneword"]
    candidate: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5)
    context: str | None = None
    mode: Literal["write", "edit", "rewrite"] | None = None
    input_payload: dict[str, Any] = Field(default_factory=dict)
    source: str = Field(default="ui")
    pos: str | None = None
    model_score: float | None = None
    reason: str | None = None
    session_id: str | None = None
    input_text: str | None = None


class FeedbackResponse(BaseModel):
    id: str
    task: str
    candidate: str
    rating: int
    quality: Literal["bad", "average", "good"]
    label: int
    message: str
