from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import create_access_token, get_current_user, get_password_hash, verify_password
from ..db import db
from ..models.auth_models import (
    AuthResponse,
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    UpdateProfileRequest,
    UserProfile,
)
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
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found. Please register first.",
        )
    if not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )

    token = create_access_token(str(user["_id"]))
    return {"token": token, "user": serialize_user(user)}


@router.get("/me", response_model=UserProfile)
async def get_me(current_user=Depends(get_current_user)):
    return serialize_user(current_user)


@router.put("/me", response_model=UserProfile)
async def update_me(
    payload: UpdateProfileRequest,
    current_user=Depends(get_current_user),
):
    updates = payload.model_dump(exclude_unset=True)
    if "email" in updates and updates["email"] != current_user.get("email"):
        existing = await db.users.find_one({"email": updates["email"]})
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

    if updates:
        await db.users.update_one(
            {"_id": current_user["_id"]},
            {"$set": updates},
        )
        current_user = await db.users.find_one({"_id": current_user["_id"]})
    return serialize_user(current_user)


@router.post("/change-password")
async def change_password(
    payload: ChangePasswordRequest,
    current_user=Depends(get_current_user),
):
    if not verify_password(payload.current_password, current_user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if verify_password(payload.new_password, current_user.get("password_hash", "")):
        raise HTTPException(
            status_code=400,
            detail="New password must be different from current password",
        )

    await db.users.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"password_hash": get_password_hash(payload.new_password)}},
    )
    return {"message": "Password updated successfully"}
