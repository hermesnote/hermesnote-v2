"""模型訓練 API：送訓練任務、查狀態、WebSocket 即時進度推送、三個 registry 查詢。

送出訓練任務只負責寫進 model_training_jobs（status='pending'），完全不在 web 容器裡執行訓練——
真正執行是另一個容器（hermesnote-training）的 worker 撿去跑，兩邊只共用這張表，
見 docs/record/archive/2026-09-08.md 的容器架構定案。
"""

import asyncio
import json
from datetime import date

import asyncpg
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from services import model_artifacts
from services.hermesnote_db import HERMESNOTE_DATABASE_URL
from services.quotes import fetch_ohlcv_rows
from training import job_store
from training.inference_store import list_predictions
from training.phases import build_phase2_graph_spec, final_model_node_id, validate_parent_for_phase
from training.preview_store import PREVIEW_CHANNEL_PREFIX, get_preview_sample, list_preview_samples
from training.progress import CHANNEL_PREFIX
from training.registry import architectures, decision_rules, features, labeling_rules, outcomes
from training.registry.labeling_rules import validate_combination

router = APIRouter(prefix="/api/model", tags=["model_training"])


@router.get("/candles")
async def get_candles(symbol: str = "tx", timeframe: str = "15m", start: date = None, end: date = None):
    """給訓練頁 B 區塊畫 K 線窗格用的原始價量（跟訓練/回測無關，純粹讀 quotes DB）。

    fetch_ohlcv_rows 只有 ORDER BY datetime, built_at，沒有去重——同一根 K 棒偶爾因為
    資料管線重跑被寫入兩筆（見 docs/architecture.md 的已知資料品質問題）。這裡依 datetime
    去重、保留最後一筆（= 最新 built_at，因為上游已經照 built_at ASC 排過），不然
    lightweight-charts 收到重複時間戳記會直接丟 assertion 錯誤拒畫。
    """
    rows = await fetch_ohlcv_rows(symbol, timeframe, start, end)
    by_time: dict[int, dict] = {}
    for r in rows:
        t = int(r["datetime"].timestamp())
        by_time[t] = {
            "time": t,
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "volume": float(r["volume"]) if r["volume"] is not None else None,
        }
    return [by_time[t] for t in sorted(by_time)]


class TrainRequest(BaseModel):
    start: str  # "YYYY-MM-DD"
    end: str
    nodes: list[dict]  # 見 training/graph.py 的節點格式
    phase: int | None = None  # 教授三階段協定：1（一般訓練也可以標 1）或 2；3 走 /infer，不走這支
    parent_job_id: str | None = None  # phase=2 時必填，必須是一個已完成的 Phase 1 job


@router.post("/train")
async def start_training(body: TrainRequest):
    if body.phase is not None and body.phase not in (1, 2):
        raise HTTPException(status_code=400, detail="phase 只能是 1 或 2；Phase 3 是推論，請用 POST /api/model/infer")

    if body.phase == 2:
        # Phase 2：完整繼承 Phase 1 的 graph_spec，只覆寫 split_strategy，
        # 不接受呼叫端自己送的 nodes/start/end——一旦允許微調，就不是「同一組實驗改切分方式」了。
        if not body.parent_job_id:
            raise HTTPException(status_code=400, detail="Phase 2 必須指定 parent_job_id（一個已完成的 Phase 1 job）")
        parent_job = await job_store.get_job(body.parent_job_id)
        try:
            validate_parent_for_phase(parent_job, expected_parent_phase=1, this_phase=2)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        graph_spec = build_phase2_graph_spec(parent_job["graph_spec"])
        job_id = await job_store.create_job(graph_spec, phase=2, parent_job_id=body.parent_job_id)
        return {"job_id": job_id}

    # 後端也要擋不合法的 outcome/labeling_rule 組合，不能只靠前端擋——
    # Agent 可能直接打 API 送 graph_spec，不會經過前端表單。
    for n in body.nodes:
        if n.get("type") == "label":
            try:
                validate_combination(n["outcome"], n.get("labeling_rule", "fixed_threshold"))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
    graph_spec = {"start": body.start, "end": body.end, "nodes": body.nodes}
    job_id = await job_store.create_job(graph_spec, phase=body.phase)
    return {"job_id": job_id}


@router.get("/jobs")
async def list_training_jobs(limit: int = 20):
    return await job_store.list_jobs(limit)


@router.get("/jobs/{job_id}")
async def get_training_job(job_id: str):
    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="找不到這個訓練任務")
    return job


@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: str):
    """刪掉一筆訓練紀錄（含它的進度、已保存的模型權重，DB 是 CASCADE）。

    主要給「送出後容器崩潰、狀態永遠卡在 running/pending 的孤兒 job」清理用——這裡
    只是刪 DB 紀錄，如果那個 job 對應的 process 真的還活著（不是孤兒、是正常在跑的
    訓練），這支 API 沒辦法把它強制中斷，訓練還是會繼續跑到結束，只是完成時找不到
    對應的 job 列可以寫結果，會在 log 裡留下一筆寫入失敗，不影響其他 job；真正的
    「中途喊停」需要另外做一個 worker 會去檢查的取消旗標，目前還沒做。
    """
    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="找不到這個訓練任務")
    if await job_store.has_children(job_id):
        raise HTTPException(status_code=409, detail="這筆有 Phase 2/3 的後續任務指著它，先刪後續的那筆")
    await job_store.delete_job(job_id)
    return {"deleted": True}


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str):
    """把一筆 pending/running 的 job 標記成失敗（使用者手動中斷）。

    只處理 DB 狀態，不會真的把訓練容器裡的 process 砍掉——這裡的 worker 是單一
    長駐 process、同時間只跑一個 job，沒有子行程可以單獨終止，要真的中斷運算
    得整個訓練容器一起重啟（`docker restart hermesnote-training`），這是操作
    NAS 本身的動作，刻意不讓 web 後端有權限觸發（見 docs/record 這次的討論：
    掛 docker.sock 進對外服務容器的安全代價太大，選擇留給人手動做）。這支 API
    存在的目的只是讓歷史紀錄立刻反映「使用者已經放棄這筆」，不用等重啟容器後
    才發現它其實已經死了。
    """
    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="找不到這個訓練任務")
    if job["status"] not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"這筆狀態是 {job['status']}，不是還在跑，不用取消")
    await job_store.mark_failed(job_id, "使用者手動中斷")
    return {"cancelled": True}


@router.get("/jobs/{job_id}/progress")
async def get_training_progress(job_id: str):
    """給前端初次載入頁面補歷史用；之後的新進度靠 WebSocket 推，不用這支輪詢。"""
    return await job_store.get_progress(job_id)


@router.get("/jobs/{job_id}/preview_samples")
async def get_preview_samples(job_id: str, node_id: str, limit: int = 50, before_id: int | None = None):
    """完成後歷史瀏覽用：依 id 游標分頁往回查這個模型節點的即時視窗預覽抽樣紀錄（新到舊）。
    這些都是「訓練當下抽樣紀錄」（source 標明 train/val），不是最終模型對全期間資料的推論——
    兩者是不同性質的資料，不要混用同一支端點；最終推論走既有的 POST /api/model/infer。
    """
    return await list_preview_samples(job_id, node_id, limit=limit, before_id=before_id)


@router.get("/jobs/{job_id}/inference_predictions")
async def get_inference_predictions(
    job_id: str, node_id: str,
    start: int | None = None, end: int | None = None,
    cursor: int | None = None, limit: int = 1000,
):
    """最終模型推論的逐列結果（job_type='infer' 那種 job 存進 model_inference_predictions 的
    完整內容），依時間區間（start/end，unix 秒）＋ id 游標分頁查——這是「最終模型推論」，
    跟 /preview_samples（訓練時抽樣紀錄）是不同性質的資料，不要混用；也因此不接受一次撈
    超過 5000 筆，全期間資料本來就該分批查，不是一次載入。
    """
    limit = min(limit, 5000)
    return await list_predictions(job_id, node_id, start=start, end=end, cursor=cursor, limit=limit)


@router.websocket("/jobs/{job_id}/stream")
async def stream_training_progress(websocket: WebSocket, job_id: str):
    """開一條專用的 asyncpg 連線，同時 LISTEN 兩個 channel：
      - 進度（epoch 級 loss/accuracy）：NOTIFY payload 本身就是完整內容，直接轉推。
      - 即時視窗預覽：NOTIFY payload 只帶 sample_id（PostgreSQL 的 NOTIFY payload 上限是
        8000 bytes，bars 隨 window 變大很容易超過，不能塞在 NOTIFY 裡），收到之後這裡才去
        model_training_preview_samples 查完整內容，再轉推給瀏覽器。
    """
    await websocket.accept()
    progress_channel = f"{CHANNEL_PREFIX}{job_id.replace('-', '_')}"
    preview_channel = f"{PREVIEW_CHANNEL_PREFIX}{job_id.replace('-', '_')}"

    conn = await asyncpg.connect(HERMESNOTE_DATABASE_URL)
    queue: asyncio.Queue = asyncio.Queue()

    def _on_notify(_conn, _pid, _channel, payload):
        queue.put_nowait(payload)

    await conn.add_listener(progress_channel, _on_notify)
    await conn.add_listener(preview_channel, _on_notify)
    try:
        while True:
            payload = await queue.get()
            try:
                msg = json.loads(payload)
            except (TypeError, ValueError):
                continue
            if msg.get("type") == "preview":
                sample = await get_preview_sample(conn, msg["sample_id"])
                if sample is not None:
                    await websocket.send_text(json.dumps({"type": "preview", **sample}))
            else:
                await websocket.send_text(payload)
    except WebSocketDisconnect:
        pass
    finally:
        await conn.remove_listener(progress_channel, _on_notify)
        await conn.remove_listener(preview_channel, _on_notify)
        await conn.close()


class InferRequest(BaseModel):
    target_job_id: str  # 要載入哪個 job 的模型
    target_node_id: str | None = None  # 那個 job 裡的哪個 model node；不給就自動找最終輸出節點
    start: str  # 推論用的新資料區間，"YYYY-MM-DD"
    end: str
    phase: int | None = None  # 教授三階段協定：3 = Phase 3 holdout 推論；None = 一般推論，不掛三階段協定


@router.post("/infer")
async def start_inference(body: InferRequest):
    target_node_id = body.target_node_id
    if target_node_id is None:
        target_job = await job_store.get_job(body.target_job_id)
        if target_job is None:
            raise HTTPException(status_code=404, detail="找不到指定的 job")
        try:
            target_node_id = final_model_node_id(target_job["graph_spec"]["nodes"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{e}；請明確指定 target_node_id")

    if body.phase is not None and body.phase != 3:
        raise HTTPException(status_code=400, detail="/infer 只接受 phase=3（Phase 3 holdout 推論）或不指定 phase")
    if body.phase == 3:
        parent_job = await job_store.get_job(body.target_job_id)
        try:
            validate_parent_for_phase(parent_job, expected_parent_phase=2, this_phase=3)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # holdout 要是真的沒看過的期間：Phase 3 的起始日期不能早於 Phase 1/2 訓練期間的結束日期，
        # 不然這個「holdout」其實跟訓練資料重疊，不是乾淨的未見測試——這是 3.2 那項「框定不能碰
        # holdout 的邊界」在目前的任務血緣機制下能做到的具體檢查（之前排在第 3 步因為那時候
        # 還沒有 Phase 血緣可以框，這裡才是真的能落地的地方）。
        if body.start <= parent_job["graph_spec"]["end"]:
            raise HTTPException(
                status_code=400,
                detail=f"Phase 3 的 holdout 起始日期（{body.start}）必須晚於 Phase 1/2 訓練期間的結束日期"
                       f"（{parent_job['graph_spec']['end']}），不然這段資料跟訓練期間重疊，不是乾淨的未見測試",
            )

    if not await model_artifacts.artifact_exists(body.target_job_id, target_node_id):
        raise HTTPException(status_code=404, detail="找不到指定的模型產物，確認 job_id/node_id 是否正確、該次訓練是否成功")
    infer_spec = {
        "target_job_id": body.target_job_id, "target_node_id": target_node_id,
        "start": body.start, "end": body.end,
    }
    job_id = await job_store.create_job(
        infer_spec, job_type="infer",
        phase=body.phase, parent_job_id=body.target_job_id if body.phase == 3 else None,
    )
    return {"job_id": job_id}


@router.get("/artifacts")
async def list_model_artifacts(job_id: str | None = None):
    return await model_artifacts.list_artifacts(job_id)


class KeepArtifactRequest(BaseModel):
    kept: bool


@router.post("/artifacts/{job_id}/{node_id}/keep")
async def keep_model_artifact(job_id: str, node_id: str, body: KeepArtifactRequest):
    ok = await model_artifacts.mark_kept(job_id, node_id, body.kept)
    if not ok:
        raise HTTPException(status_code=404, detail="找不到這個模型產物")
    return {"ok": True}


@router.get("/registry/features")
def get_feature_registry():
    return features.list_available()


@router.get("/registry/outcomes")
def get_outcome_registry():
    return outcomes.list_available()


@router.get("/registry/labeling_rules")
def get_labeling_rule_registry():
    return labeling_rules.list_available()


@router.get("/registry/architectures")
def get_architecture_registry():
    return architectures.list_available()


@router.get("/registry/decision_rules")
def get_decision_rule_registry():
    """F 項契約：只回傳目前登記的決策規則清單供查詢，這輪沒有對應的送出端點——
    模型輸出→決策規則→回測這條完整路徑排在下一輪，見開發排程。"""
    return decision_rules.list_available()
