"""後台登入：驗證 Google ID token + email 白名單，發我們自己的 session JWT。"""

import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from jose import JWTError, jwt

load_dotenv()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
ADMIN_EMAILS = {
    e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()
}
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 12

_bearer = HTTPBearer(auto_error=False)


def verify_google_credential(credential: str) -> tuple[str, str]:
    """驗證前端 Google Identity Services 送來的 ID token，回傳已通過白名單檢查的 (email, 頭像網址)。"""
    try:
        payload = google_id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Google 登入驗證失敗")

    email = (payload.get("email") or "").lower()
    if not payload.get("email_verified") or not email:
        raise HTTPException(status_code=401, detail="Google 帳號 email 未驗證")
    if email not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="此帳號不在後台白名單內")

    return email, payload.get("picture") or ""


def create_session_token(email: str, picture: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(
        {"sub": email, "picture": picture, "exp": expire}, JWT_SECRET, algorithm=JWT_ALGORITHM
    )


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """受保護路由的 dependency：從 Authorization: Bearer <token> 解出目前登入者資訊。"""
    if creds is None:
        raise HTTPException(status_code=401, detail="未登入", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="登入已過期或無效", headers={"WWW-Authenticate": "Bearer"})

    email = payload.get("sub")
    if email not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="此帳號不在後台白名單內")
    return {"email": email, "picture": payload.get("picture") or ""}
