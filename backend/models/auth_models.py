from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3)
    username: str = Field(..., min_length=2)
    password: str = Field(..., min_length=6)
    phone: str | None = None
    bio: str | None = None
    interests: str | None = None


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)


class UserProfile(BaseModel):
    id: str
    email: str
    username: str
    phone: str | None = None
    bio: str | None = None
    interests: str | None = None
    created_at: str | None = None


class AuthResponse(BaseModel):
    token: str
    user: UserProfile
