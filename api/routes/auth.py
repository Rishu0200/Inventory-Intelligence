"""
POST /api/auth/login — authenticate and receive a JWT access token.
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from db.session import get_session_dependency
from db.models import User
from auth.security import verify_password, create_access_token

router = APIRouter()


@router.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(),
          session=Depends(get_session_dependency)):
    user = session.query(User).filter_by(email=form_data.username).first()

    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect email or password",
                            headers={"WWW-Authenticate": "Bearer"})
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    token = create_access_token(user.id, user.email, user.role)
    return {"access_token": token, "token_type": "bearer"}