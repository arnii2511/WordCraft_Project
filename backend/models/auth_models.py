import re

import dns.resolver
from pydantic import BaseModel, Field, field_validator

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
INDIA_PHONE_RE = re.compile(r"^\+91\s[6-9]\d{9}$")
STRONG_PASSWORD_RE = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$"
)


def _domain_looks_resolvable(email: str) -> bool:
    domain = email.split("@", 1)[1]
    resolver = dns.resolver.Resolver()
    resolver.timeout = 1.5
    resolver.lifetime = 2.5
    try:
        resolver.resolve(domain, "MX")
        return True
    except Exception:
        try:
            resolver.resolve(domain, "A")
            return True
        except Exception:
            return False


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3)
    username: str = Field(..., min_length=2)
    password: str = Field(..., min_length=8)
    phone: str = Field(..., min_length=14)
    bio: str | None = None
    interests: str | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not EMAIL_RE.match(cleaned):
            raise ValueError("Enter a valid email address")
        if not _domain_looks_resolvable(cleaned):
            raise ValueError("Email domain does not appear to exist")
        return cleaned

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if not STRONG_PASSWORD_RE.match(value):
            raise ValueError(
                "Password must be 8+ chars with uppercase, lowercase, number, and special character"
            )
        return value

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, value: str) -> str:
        cleaned = value.strip()
        if not INDIA_PHONE_RE.match(cleaned):
            raise ValueError("Phone must be in India format: +91 9876543210")
        return cleaned


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not EMAIL_RE.match(cleaned):
            raise ValueError("Enter a valid email address")
        return cleaned


class UpdateProfileRequest(BaseModel):
    email: str | None = None
    username: str | None = None
    phone: str | None = None
    bio: str | None = None
    interests: str | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip().lower()
        if not EMAIL_RE.match(cleaned):
            raise ValueError("Enter a valid email address")
        if not _domain_looks_resolvable(cleaned):
            raise ValueError("Email domain does not appear to exist")
        return cleaned

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("Username must be at least 2 characters")
        return cleaned

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        if not INDIA_PHONE_RE.match(cleaned):
            raise ValueError("Phone must be in India format: +91 9876543210")
        return cleaned


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8)

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if not STRONG_PASSWORD_RE.match(value):
            raise ValueError(
                "Password must be 8+ chars with uppercase, lowercase, number, and special character"
            )
        return value


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
