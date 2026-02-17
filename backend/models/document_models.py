from typing import Literal

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1)
    content_html: str = ""
    content_text: str = ""
    context: str = "neutral"
    mode: Literal["write", "edit", "rewrite"] = "write"


class DocumentUpdate(BaseModel):
    title: str | None = None
    content_html: str | None = None
    content_text: str | None = None
    context: str | None = None
    mode: Literal["write", "edit", "rewrite"] | None = None


class DocumentResponse(BaseModel):
    id: str
    title: str
    content_html: str
    content_text: str
    context: str
    mode: Literal["write", "edit", "rewrite"]
    created_at: str | None = None
    updated_at: str | None = None
