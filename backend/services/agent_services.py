"""NAS 上的 RAG／MCP 服務客戶端（後台 Dashboard 專用）。

- 服務 token 只在後端：先讀環境變數 RAG_SERVICE_TOKEN／MCP_SERVICE_TOKEN，沒有再讀
  AGENT_SERVICES_SECRETS_FILE（NAS 上 MCP 的 settings.json，以唯讀掛載進 backend 容器）。
  token 不回傳給前端、不寫日誌、不出現在錯誤訊息。
- 只呼叫這裡寫死的路徑，不提供可任意指定 URL／path 的代理。
- 每次呼叫都記下觀測時間與耗時；失敗時回報原因，不把「查不到」當成「沒有」。
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv()

SECRETS_FILE = os.getenv("AGENT_SERVICES_SECRETS_FILE", "/run/secrets/rag-mcp/settings.json")
HEALTH_TIMEOUT = 5
READ_TIMEOUT = 20
# 搜尋要先在 RAG 服務端做 embedding（CPU），第一次可能較慢
SEARCH_TIMEOUT = 60
MCP_PROTOCOL_VERSION = "2025-06-18"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ServiceError(Exception):
    """上游服務呼叫失敗；reason 是可以顯示給管理員的原因（不含 token）。"""

    def __init__(self, service: str, reason: str, http_status: int | None = None, upstream_detail=None):
        super().__init__(reason)
        self.service = service
        self.reason = reason
        self.http_status = http_status
        self.upstream_detail = upstream_detail

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "reason": self.reason,
            "http_status": self.http_status,
            "upstream_detail": self.upstream_detail,
            "observed_at": now_iso(),
        }


def _secrets_file() -> dict:
    try:
        return json.loads(Path(SECRETS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def service_config(service: str) -> dict:
    """回傳 {base_url, token, token_source}；token_source 只說明來源，不含值。"""
    secrets = _secrets_file()
    if service == "rag":
        base = os.getenv("RAG_BASE_URL") or secrets.get("rag_url") or ""
        env_token = os.getenv("RAG_SERVICE_TOKEN")
        file_token = secrets.get("rag_token")
    elif service == "mcp":
        base = os.getenv("MCP_BASE_URL") or ""
        env_token = os.getenv("MCP_SERVICE_TOKEN")
        file_token = secrets.get("mcp_token")
    else:
        raise ValueError(service)
    token = env_token or file_token or ""
    source = "env" if env_token else ("secrets_file" if file_token else None)
    return {"base_url": base.rstrip("/"), "token": token, "token_source": source}


def _request(service: str, method: str, path: str, *, body=None, timeout=READ_TIMEOUT,
             auth=True, headers=None) -> tuple[requests.Response, float]:
    cfg = service_config(service)
    if not cfg["base_url"]:
        raise ServiceError(service, "後端未設定服務位址")
    if auth and not cfg["token"]:
        raise ServiceError(service, "後端未設定服務憑證")
    h = dict(headers or {})
    if auth:
        h["Authorization"] = "Bearer " + cfg["token"]
    started = time.monotonic()
    try:
        resp = requests.request(method, cfg["base_url"] + path, json=body, headers=h, timeout=timeout)
    except requests.Timeout:
        raise ServiceError(service, f"逾時（{timeout} 秒未回應）")
    except requests.ConnectionError:
        raise ServiceError(service, "無法連線")
    except requests.RequestException as exc:
        raise ServiceError(service, f"請求失敗：{type(exc).__name__}")
    return resp, round((time.monotonic() - started) * 1000)


def _upstream_detail(resp: requests.Response):
    try:
        data = resp.json()
    except ValueError:
        return None
    if isinstance(data, dict):
        return data.get("detail") or data.get("error")
    return None


# ---------------------------------------------------------------- 狀態檢查

def _probe(service: str, path: str, auth: bool) -> dict:
    """單一檢查：不丟例外，回傳 {ok, http_status, latency_ms, error, body, observed_at}。"""
    result = {"ok": False, "http_status": None, "latency_ms": None, "error": None, "body": None,
              "observed_at": now_iso()}
    try:
        resp, ms = _request(service, "GET", path, timeout=HEALTH_TIMEOUT, auth=auth)
    except ServiceError as exc:
        result["error"] = exc.reason
        return result
    result.update(http_status=resp.status_code, latency_ms=ms)
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        result["body"] = {k: v for k, v in body.items() if "token" not in k.lower()}
    if resp.status_code == 200:
        result["ok"] = True
    elif resp.status_code == 401:
        result["error"] = "服務拒絕後端憑證（401）"
    elif resp.status_code == 503:
        result["error"] = "服務回報未就緒（503）"
    else:
        result["error"] = f"HTTP {resp.status_code}"
    return result


def service_status(service: str) -> dict:
    """存活（/health，免認證）與就緒（/ready，需認證）分開回報。"""
    cfg = service_config(service)
    return {
        "service": service,
        "base_url": cfg["base_url"] or None,
        "credential_configured": bool(cfg["token"]),
        "credential_source": cfg["token_source"],
        "alive": _probe(service, "/health", auth=False),
        "ready": _probe(service, "/ready", auth=True),
    }


# ---------------------------------------------------------------- RAG（固定路徑）

def _rag(method: str, path: str, body=None, timeout=READ_TIMEOUT) -> dict:
    resp, ms = _request("rag", method, path, body=body, timeout=timeout)
    if resp.status_code != 200:
        reasons = {401: "服務拒絕後端憑證（401）", 404: "找不到指定來源或版本（404）",
                   422: "查詢參數不符合服務規則（422）", 503: "服務未就緒（503）"}
        raise ServiceError("rag", reasons.get(resp.status_code, f"HTTP {resp.status_code}"),
                           resp.status_code, _upstream_detail(resp))
    try:
        data = resp.json()
    except ValueError:
        raise ServiceError("rag", "回應不是 JSON")
    if isinstance(data, dict):
        data["latency_ms"] = ms
    return data


def _query(params: dict) -> str:
    clean = {k: v for k, v in params.items() if v is not None and v != ""}
    return ("?" + urlencode(clean)) if clean else ""


def _uuid(value: str, service: str = "rag") -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise ServiceError(service, "source_id 格式不正確", 422)


def rag_knowledge_bases() -> dict:
    return _rag("GET", "/api/research/v1/knowledge-bases")


def rag_summary() -> dict:
    return _rag("GET", "/api/research/v1/summary")


def rag_sources(offset: int, limit: int, source_type: str | None, status: str | None) -> dict:
    return _rag("GET", "/api/research/v1/sources" + _query(
        {"offset": offset, "limit": limit, "source_type": source_type, "status": status}))


def rag_source_content(source_id: str, version: str | None, chunk_index: int | None,
                       offset: int, limit: int) -> dict:
    return _rag("GET", f"/api/research/v1/sources/{_uuid(source_id)}/content" + _query(
        {"version": version, "chunk_index": chunk_index, "offset": offset, "limit": limit}))


def rag_ingestion_jobs(source_id: str | None, offset: int, limit: int) -> dict:
    sid = _uuid(source_id) if source_id else None
    return _rag("GET", "/api/research/v1/ingestion-jobs" + _query(
        {"source_id": sid, "offset": offset, "limit": limit}))


def rag_search(query: str, limit: int, source_type: str | None, source_id: str | None) -> dict:
    filters = {}
    if source_type:
        filters["source_type"] = source_type
    if source_id:
        filters["source_id"] = _uuid(source_id)
    return _rag("POST", "/api/research/v1/search",
                body={"query": query, "limit": limit, "filters": filters}, timeout=SEARCH_TIMEOUT)


# ---------------------------------------------------------------- MCP

def mcp_tool_catalog() -> dict:
    resp, ms = _request("mcp", "GET", "/api/mcp/v1/tools")
    if resp.status_code != 200:
        raise ServiceError("mcp", "服務拒絕後端憑證（401）" if resp.status_code == 401 else f"HTTP {resp.status_code}",
                           resp.status_code, _upstream_detail(resp))
    data = resp.json()
    data["latency_ms"] = ms
    data["observed_at"] = now_iso()
    return data


def _rpc(session: dict, method: str, params: dict | None = None, notify: bool = False) -> dict | None:
    """MCP Streamable HTTP 一次 JSON-RPC 呼叫（服務端為 stateless＋JSON 回應）。"""
    payload = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        payload["params"] = params
    if not notify:
        session["next_id"] += 1
        payload["id"] = session["next_id"]
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
               "MCP-Protocol-Version": session.get("protocol_version", MCP_PROTOCOL_VERSION)}
    if session.get("session_id"):
        headers["Mcp-Session-Id"] = session["session_id"]
    resp, _ = _request("mcp", "POST", "/mcp/", body=payload, timeout=READ_TIMEOUT, headers=headers)
    if resp.headers.get("mcp-session-id"):
        session["session_id"] = resp.headers["mcp-session-id"]
    if notify:
        if resp.status_code not in (200, 202, 204):
            raise ServiceError("mcp", f"{method}：HTTP {resp.status_code}", resp.status_code)
        return None
    if resp.status_code != 200:
        raise ServiceError("mcp", f"{method}：HTTP {resp.status_code}", resp.status_code, _upstream_detail(resp))
    text = resp.text
    if "text/event-stream" in resp.headers.get("content-type", ""):
        # 只取最後一個 data: 事件
        lines = [ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:")]
        text = lines[-1] if lines else ""
    try:
        message = json.loads(text)
    except ValueError:
        raise ServiceError("mcp", f"{method}：回應不是 JSON-RPC")
    if message.get("error"):
        err = message["error"]
        raise ServiceError("mcp", f"{method}：{err.get('message', 'JSON-RPC error')}", upstream_detail=err)
    return message.get("result")


def _tool_payload(result: dict) -> tuple[object, str]:
    """MCP 工具回應可能用 structuredContent，或把 JSON 放在 text content；兩種都讀。"""
    structured = result.get("structuredContent")
    if structured is not None:
        return structured, "structuredContent"
    for item in result.get("content") or []:
        if item.get("type") == "text":
            try:
                return json.loads(item.get("text", "")), "text"
            except ValueError:
                return item.get("text"), "text"
    return None, "none"


def mcp_connection_test() -> dict:
    """唯讀連線測試：initialize → tools/list → 呼叫 knowledge_bases。每步記錄結果與耗時。"""
    steps = []
    session = {"next_id": 0}

    def step(name: str, fn):
        started = time.monotonic()
        entry = {"step": name, "ok": False, "latency_ms": None, "detail": None, "error": None}
        try:
            entry["detail"] = fn()
            entry["ok"] = True
        except ServiceError as exc:
            entry["error"] = exc.reason
        entry["latency_ms"] = round((time.monotonic() - started) * 1000)
        steps.append(entry)
        return entry["ok"]

    def initialize():
        result = _rpc(session, "initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "hermesnote-admin", "version": "1"}})
        session["protocol_version"] = result.get("protocolVersion", MCP_PROTOCOL_VERSION)
        _rpc(session, "notifications/initialized", notify=True)
        return {"protocol_version": session["protocol_version"], "server": result.get("serverInfo")}

    def list_tools():
        result = _rpc(session, "tools/list", {})
        return {"tools": [t.get("name") for t in result.get("tools", [])]}

    def call_kb():
        result = _rpc(session, "tools/call", {"name": "knowledge_bases", "arguments": {}})
        payload, carrier = _tool_payload(result)
        kbs = payload.get("data") if isinstance(payload, dict) else None
        if result.get("isError"):
            raise ServiceError("mcp", "工具回報錯誤", upstream_detail=payload)
        return {"carrier": carrier,
                "knowledge_bases": [{"id": k.get("id"), "state": k.get("state"),
                                     "sources_scanned": k.get("sources_scanned")} for k in (kbs or [])]}

    if step("initialize", initialize) and step("tools/list", list_tools):
        step("tools/call knowledge_bases", call_kb)
    return {"ok": all(s["ok"] for s in steps) and len(steps) == 3, "steps": steps, "observed_at": now_iso()}
