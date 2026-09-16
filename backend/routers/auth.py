from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth import create_session_token, get_current_user, verify_google_credential

router = APIRouter(prefix="/api/auth", tags=["auth"])


class GoogleLoginRequest(BaseModel):
    credential: str


@router.post("/google")
def login_with_google(body: GoogleLoginRequest):
    email, picture = verify_google_credential(body.credential)
    token = create_session_token(email, picture)
    return {"access_token": token, "token_type": "bearer", "email": email, "picture": picture}


@router.get("/me")
def read_current_user(user: dict = Depends(get_current_user)):
    return user
