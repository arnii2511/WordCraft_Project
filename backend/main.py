import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.auth_routes import router as auth_router
from .api.oneword_routes import router as oneword_router
from .api.documents_routes import router as documents_router
from .api.lexical_routes import router as lexical_router
from .api.constraints_routes import router as constraints_router
from .api.saved_words_routes import router as saved_words_router
from .api.suggestion_routes import router as suggestion_router
from .api.feedback_routes import router as feedback_router
from .config import CORS_ORIGINS
from .db import db

app = FastAPI(title="WordCraft API", version="0.1.0")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(suggestion_router)
app.include_router(lexical_router)
app.include_router(constraints_router)
app.include_router(oneword_router)
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(saved_words_router)
app.include_router(feedback_router)


async def _ensure_indexes_background() -> None:
    try:
        await asyncio.wait_for(db.users.create_index("email", unique=True), timeout=8)
        await asyncio.wait_for(db.documents.create_index([("user_id", 1), ("updated_at", -1)]), timeout=8)
        await asyncio.wait_for(db.saved_words.create_index([("user_id", 1), ("created_at", -1)]), timeout=8)
        await asyncio.wait_for(db.feedback_ratings.create_index([("input_key", 1), ("created_at", -1)]), timeout=8)
        await asyncio.wait_for(db.feedback_ratings.create_index([("task", 1), ("created_at", -1)]), timeout=8)
    except Exception as exc:  # pragma: no cover
        logger.warning("Index creation skipped/deferred during startup: %s", exc)


@app.on_event("startup")
async def startup_indexes():
    # Do not block boot on index setup; Render health checks require fast port binding.
    asyncio.create_task(_ensure_indexes_background())


@app.get("/health")
async def health_check():
    return {"status": "ok"}
