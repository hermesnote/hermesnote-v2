"""後台 Dashboard／知識庫／MCP 服務頁的整合 API（全部限管理員）。

工作來源以 provider 整合：每個 provider 回傳統一的工作項目，並各自帶連線狀態
（connected／error／not_connected）。查不到的來源標為 error 並附原因，不當成沒有工作；
尚未接入的來源（HA、SDK）標為 not_connected，不提供任何工作項目。
"""

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from auth import get_current_user
from services import agent_services as svc
from services import job_store as backtest_store
from training import job_store as training_store

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(get_current_user)])

ACTIVE = {"pending", "running"}
NOT_CONNECTED_PROVIDERS = [
    {"id": "hermes_agent", "label": "Hermes Agent（HA Profile）",
     "note": "HA 看板／Profile 的資料契約尚未建立，這裡不讀取 HA runtime。"},
    {"id": "agent_sdk", "label": "Claude／Codex SDK",
     "note": "SDK 執行服務尚未建置。"},
]


def _raise(exc: svc.ServiceError):
    status = exc.http_status if exc.http_status in (404, 422) else 502
    raise HTTPException(status_code=status, detail=exc.to_dict())


async def _call(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except svc.ServiceError as exc:
        _raise(exc)


# ---------------------------------------------------------------- 工作來源 providers

def _model_title(job: dict) -> str:
    spec = job.get("graph_spec") or {}
    models = [n for n in spec.get("nodes", []) if n.get("type") == "model"]
    tf = next((n.get("timeframe") for n in spec.get("nodes", []) if n.get("timeframe")), None)
    name = "＋".join(m.get("key", "?") for m in models) or job.get("job_type") or "模型"
    kind = "推論" if job.get("job_type") == "inference" else "訓練"
    phase = f"Phase {job['phase']}" if job.get("phase") else None
    return " · ".join(x for x in [f"{name} {kind}", tf, phase] if x)


def _total_epochs(job: dict) -> int | None:
    for node in (job.get("graph_spec") or {}).get("nodes", []):
        params = node.get("params") or {}
        for key in ("epochs", "n_estimators"):
            if isinstance(params.get(key), int):
                return params[key]
    return None


async def _model_provider() -> dict:
    jobs = await training_store.list_jobs(50)
    items = []
    for j in jobs:
        progress = None
        if j["status"] == "running":
            rows = await training_store.get_progress(j["job_id"])
            total = _total_epochs(j)
            if rows:
                current = max(r["epoch"] for r in rows) + 1
                progress = {"current": current, "total": total, "unit": "輪"}
        done = j["status"] == "done"
        items.append({
            "provider": "model_training",
            "kind": "模型推論" if j.get("job_type") == "inference" else "模型訓練",
            "id": j["job_id"],
            "title": _model_title(j),
            "status": j["status"],
            "progress": progress,
            "error": j.get("error"),
            "created_at": j["created_at"],
            "updated_at": j.get("finished_at") or j.get("started_at") or j["created_at"],
            "delivery": ({"state": "available", "label": "結果可查看"} if done and j.get("result")
                         else {"state": "missing", "label": "無結果紀錄"} if done else None),
            "link": f"/model?job={j['job_id']}",
        })
    return {"items": items}


async def _backtest_provider() -> dict:
    jobs = await backtest_store.list_jobs(30)
    items = []
    for j in jobs:
        done = j["status"] == "done"
        items.append({
            "provider": "backtest",
            "kind": "量化回測",
            "id": j["job_id"],
            "title": f"{j['symbol']} {j['timeframe']} · {j['start_date']}～{j['end_date']}",
            "status": j["status"],
            "progress": None,
            "error": j.get("error"),
            "created_at": j["created_at"],
            "updated_at": j.get("finished_at") or j["created_at"],
            "delivery": ({"state": "available", "label": "結果可查看"} if done and j.get("summary")
                         else {"state": "missing", "label": "無結果紀錄"} if done else None),
            "link": f"/quant?job={j['job_id']}",
        })
    return {"items": items}


def _rag_provider() -> dict:
    """知識庫索引：待核對文件（needs_review）列為需要處理；索引工作只顯示最近幾筆。

    歷史 ingestion job 的累積 failed 筆數不等於目前未解決的文件數，所以「需要處理」
    只看目前版本的文件狀態。
    """
    review = svc.rag_sources(0, 50, None, "needs_review")
    jobs = svc.rag_ingestion_jobs(None, 0, 10)
    items = []
    for doc in review.get("data") or []:
        warnings = (doc.get("metadata") or {}).get("warnings") or []
        pages = sorted({w["locator"].get("page") for w in warnings if (w.get("locator") or {}).get("page")})
        items.append({
            "provider": "rag",
            "kind": "知識庫文件",
            "id": doc["source_id"],
            "title": doc["title"],
            "status": "needs_review",
            "progress": None,
            "error": (f"品質警告 {len(warnings)} 則" + (f"（第 {'、'.join(map(str, pages))} 頁）" if pages else "")),
            "created_at": doc.get("updated_at"),
            "updated_at": doc.get("updated_at"),
            "delivery": {"state": "indexed_with_warnings", "label": "已索引，待核對"},
            "link": f"/admin/rag?source={doc['source_id']}",
        })
    for job in jobs.get("data") or []:
        if job.get("status") in ("running", "pending", "queued") or job.get("finished_at") is None:
            items.append({
                "provider": "rag",
                "kind": "索引工作",
                "id": job["job_id"],
                "title": f"索引 {job.get('source_id', '')[:8]}（{job.get('stage')}）",
                "status": "running",
                "progress": None,
                "error": job.get("error"),
                "created_at": job.get("started_at"),
                "updated_at": job.get("started_at"),
                "delivery": None,
                "link": f"/admin/rag?tab=jobs",
            })
    return {"items": items}


async def _run_provider(pid: str, label: str, coro_fn) -> dict:
    started = time.monotonic()
    entry = {"id": pid, "label": label, "state": "connected", "error": None, "items": [],
             "observed_at": svc.now_iso(), "latency_ms": None}
    try:
        result = await coro_fn()
        entry["items"] = result["items"]
    except svc.ServiceError as exc:
        entry.update(state="error", error=exc.reason)
    except Exception as exc:  # 資料庫等其他失敗：保留「無法確認」
        entry.update(state="error", error=f"{type(exc).__name__}")
    entry["latency_ms"] = round((time.monotonic() - started) * 1000)
    return entry


def _service_entry(status: dict, label: str) -> dict:
    alive, ready = status["alive"], status["ready"]
    if not status["credential_configured"]:
        state = "unconfigured"
    elif alive["ok"] and ready["ok"]:
        state = "ready"
    elif alive["ok"]:
        state = "degraded"
    else:
        state = "down"
    return {"id": status["service"], "label": label, "state": state, **status}


@router.get("/overview")
async def overview():
    providers = await asyncio.gather(
        _run_provider("model_training", "模型訓練／推論", _model_provider),
        _run_provider("backtest", "量化回測", _backtest_provider),
        _run_provider("rag", "知識庫索引", lambda: asyncio.to_thread(_rag_provider)),
    )
    rag_status, mcp_status = await asyncio.gather(
        asyncio.to_thread(svc.service_status, "rag"), asyncio.to_thread(svc.service_status, "mcp"))
    services = [_service_entry(rag_status, "RAG 知識庫服務"), _service_entry(mcp_status, "MCP 服務")]

    connected = [p for p in providers if p["state"] == "connected"]
    items = [i for p in connected for i in p["items"]]
    counts = {
        "running": sum(1 for i in items if i["status"] == "running"),
        "pending": sum(1 for i in items if i["status"] == "pending"),
        "attention": sum(1 for i in items if i["status"] in ("failed", "needs_review", "blocked", "awaiting_approval")),
        "service_issues": sum(1 for s in services if s["state"] != "ready"),
    }
    return {
        "observed_at": svc.now_iso(),
        "counts": counts,
        "counted_providers": [p["id"] for p in connected],
        "unavailable_providers": [{"id": p["id"], "label": p["label"], "error": p["error"]}
                                  for p in providers if p["state"] != "connected"],
        "providers": list(providers) + [{**p, "state": "not_connected", "items": [], "error": None,
                                         "observed_at": None, "latency_ms": None}
                                        for p in NOT_CONNECTED_PROVIDERS],
        "services": services,
    }


@router.get("/services")
async def services_status():
    rag_status, mcp_status = await asyncio.gather(
        asyncio.to_thread(svc.service_status, "rag"), asyncio.to_thread(svc.service_status, "mcp"))
    return [_service_entry(rag_status, "RAG 知識庫服務"), _service_entry(mcp_status, "MCP 服務")]


# ---------------------------------------------------------------- RAG（固定路徑代理）

@router.get("/rag/knowledge-bases")
async def rag_knowledge_bases():
    return await _call(svc.rag_knowledge_bases)


@router.get("/rag/summary")
async def rag_summary():
    return await _call(svc.rag_summary)


@router.get("/rag/sources")
async def rag_sources(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100),
                      source_type: str | None = None, status: str | None = None):
    return await _call(svc.rag_sources, offset, limit, source_type, status)


@router.get("/rag/sources/{source_id}/content")
async def rag_source_content(source_id: str, version: str | None = None,
                             chunk_index: int | None = Query(None, ge=0),
                             offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=50)):
    return await _call(svc.rag_source_content, source_id, version, chunk_index, offset, limit)


@router.get("/rag/ingestion-jobs")
async def rag_ingestion_jobs(source_id: str | None = None, offset: int = Query(0, ge=0),
                             limit: int = Query(20, ge=1, le=50)):
    return await _call(svc.rag_ingestion_jobs, source_id, offset, limit)


class SearchBody(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=30)
    source_type: str | None = None
    source_id: str | None = None


@router.post("/rag/search")
async def rag_search(body: SearchBody):
    return await _call(svc.rag_search, body.query, body.limit, body.source_type, body.source_id)


# ---------------------------------------------------------------- MCP

@router.get("/mcp/status")
async def mcp_status():
    rag_status, mcp_status_ = await asyncio.gather(
        asyncio.to_thread(svc.service_status, "rag"), asyncio.to_thread(svc.service_status, "mcp"))
    return {
        "service": _service_entry(mcp_status_, "MCP 服務"),
        "upstream_rag": _service_entry(rag_status, "RAG 知識庫服務"),
        "protocol_endpoint": (mcp_status_["base_url"] + "/mcp/") if mcp_status_["base_url"] else None,
        # 服務健康不代表任何 Agent 已接上；目前 HA Profile 尚未設定 mcp_servers
        "agent_connections": [{"id": p["id"], "label": p["label"], "state": "not_connected", "note": p["note"]}
                              for p in NOT_CONNECTED_PROVIDERS],
        "observed_at": svc.now_iso(),
    }


@router.get("/mcp/tools")
async def mcp_tools():
    return await _call(svc.mcp_tool_catalog)


@router.post("/mcp/test")
async def mcp_test():
    return await asyncio.to_thread(svc.mcp_connection_test)
