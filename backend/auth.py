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


# ── 模型 API 寫入權限（2026-11-16）────────────────────────────────────────────────
# /api/model/* 的寫入端點（建立任務、推論、刪除、取消、收藏、實驗層）只接受兩種身分其一：
#   1. 管理員：後台 Google 登入後的 session JWT（Authorization: Bearer <session>），擁有全部權限範圍。
#   2. 服務金鑰：X-Hermesnote-Key: hnk_...（給 MCP 等程式）。後端只保存金鑰的 SHA-256，
#      設定在 .env 的 MODEL_API_KEYS（JSON 陣列）：[{"name": "mcp-training", "sha256": "<64 hex>", "scopes": ["model:train"]}]
# 讀取（GET）與 WebSocket 進度不受影響。驗證失敗一律在建立任何資料之前回 401／403。
import hashlib  # noqa: E402
import hmac  # noqa: E402
import json as _json  # noqa: E402
import logging  # noqa: E402

from fastapi import Header  # noqa: E402

MODEL_SCOPES = {
    "model:train": "POST /api/model/train（Phase 1／2 訓練任務）",
    "model:infer": "POST /api/model/infer（推論與 Phase 3）",
    "model:manage": "刪除／取消任務、收藏模型產物",
    "model:experiment": "實驗層寫入（建立實驗、修改設定、組別、框架、執行、暖機建議、特徵記憶類型）",
}
SERVICE_KEY_HEADER = "X-Hermesnote-Key"
_log = logging.getLogger("hermesnote.auth")
_keys_cache: tuple[str, list[dict]] | None = None


def _service_keys() -> list[dict]:
    """解析 MODEL_API_KEYS；格式錯誤的項目直接略過並記錄（不印出內容），不會讓整個後端無法啟動。"""
    global _keys_cache
    raw = os.getenv("MODEL_API_KEYS", "").strip()
    if _keys_cache is not None and _keys_cache[0] == raw:
        return _keys_cache[1]
    keys: list[dict] = []
    if raw:
        try:
            items = _json.loads(raw)
        except ValueError:
            _log.error("MODEL_API_KEYS 不是合法 JSON，服務金鑰全部停用")
            items = []
        for it in items if isinstance(items, list) else []:
            name, digest, scopes = it.get("name"), str(it.get("sha256", "")).lower(), it.get("scopes")
            if (not isinstance(name, str) or not name or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest) or not isinstance(scopes, list)):
                _log.error("MODEL_API_KEYS 有一筆格式不正確（name／sha256／scopes），已略過")
                continue
            unknown = [s for s in scopes if s not in MODEL_SCOPES]
            if unknown:
                _log.error("MODEL_API_KEYS 金鑰 %s 有未知的權限範圍 %s，已略過這些範圍", name, unknown)
            keys.append({"name": name, "sha256": digest, "scopes": [s for s in scopes if s in MODEL_SCOPES]})
    _keys_cache = (raw, keys)
    return keys


def _match_service_key(presented: str) -> dict | None:
    digest = hashlib.sha256(presented.encode("utf-8")).hexdigest()
    found = None
    for k in _service_keys():  # 全部比完，不提早結束
        if hmac.compare_digest(digest, k["sha256"]):
            found = k
    return found


def require_model_writer(scope: str):
    """產生 dependency：通過時回傳 principal {"kind", "id", "scopes"}，id 例如 "admin:<email>"、"key:mcp-training"。"""
    if scope not in MODEL_SCOPES:
        raise ValueError(f"未知的權限範圍 {scope}")

    async def dep(
        creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
        service_key: str | None = Header(default=None, alias=SERVICE_KEY_HEADER),
    ) -> dict:
        if service_key:
            k = _match_service_key(service_key)
            if k is None:
                raise HTTPException(status_code=401, detail="服務金鑰無效")
            if scope not in k["scopes"]:
                raise HTTPException(status_code=403, detail=f"服務金鑰 {k['name']} 沒有 {scope} 權限")
            return {"kind": "key", "id": f"key:{k['name']}", "scopes": list(k["scopes"])}
        if creds is None:
            raise HTTPException(status_code=401, detail=f"需要登入或服務金鑰（{SERVICE_KEY_HEADER}）",
                                headers={"WWW-Authenticate": "Bearer"})
        if not JWT_SECRET:
            raise HTTPException(status_code=401, detail="後端未設定 JWT_SECRET，管理員登入停用")
        user = get_current_user(creds)
        return {"kind": "admin", "id": f"admin:{user['email']}", "scopes": sorted(MODEL_SCOPES)}

    return dep


def principal_id(principal) -> str | None:
    """路由內取得驗證身分；直接呼叫路由函式（單元測試）時 principal 是 Depends 物件，回傳 None。"""
    return principal.get("id") if isinstance(principal, dict) else None
