"""
POST /api/auth/login — authenticate and receive a JWT access token.
"""
from __future__ import annotations
import hashlib

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from auth.security import verify_password, create_access_token
from cache.redis_client import rate_limit_check
from db.models import User
from db.session import get_session_dependency

router = APIRouter()

LOGIN_ATTEMPTS_PER_MINUTE = 10


@router.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(),
          session=Depends(get_session_dependency)):
    email_key = hashlib.sha256(form_data.username.strip().lower().encode()).hexdigest()[:16]
    if not rate_limit_check(f"login:{email_key}", LOGIN_ATTEMPTS_PER_MINUTE, 60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please wait a minute and try again.",
            headers={"Retry-After": "60"},
        )

    user = session.query(User).filter_by(email=form_data.username).first()

    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect email or password",
                            headers={"WWW-Authenticate": "Bearer"})
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    token = create_access_token(user.id, user.email, user.role)
    return {"access_token": token, "token_type": "bearer"}