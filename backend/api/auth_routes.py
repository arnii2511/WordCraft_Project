from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import create_access_token, get_current_user, get_password_hash, verify_password
from ..db import db
from ..models.auth_models import AuthResponse, LoginRequest, RegisterRequest, UserProfile
from ..serializers import serialize_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
async def register_user(payload: RegisterRequest):
    existing = await db.users.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_doc = {
        "email": payload.email,
        "username": payload.username,
        "password_hash": get_password_hash(payload.password),
        "phone": payload.phone,
        "bio": payload.bio,
        "interests": payload.interests,
        "created_at": datetime.now(timezone.utc),
    }
    result = await db.users.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id

    token = create_access_token(str(result.inserted_id))
    return {"token": token, "user": serialize_user(user_doc)}


@router.post("/login", response_model=AuthResponse)
async def login_user(payload: LoginRequest):
    user = await db.users.find_one({"email": payload.email})
    if not user or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(str(user["_id"]))
    return {"token": token, "user": serialize_user(user)}


@router.get("/me", response_model=UserProfile)
async def get_me(current_user=Depends(get_current_user)):
    return serialize_user(current_user)
