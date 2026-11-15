"""模型訓練任務（model_training_jobs）的持久化，比照 services/job_store.py 的量化回測寫法。"""

import json
import uuid

from services.hermesnote_db import get_conn
from training import json_safe
from training.provenance import PIPELINE_VERSION

# 2026-09-18 修正：`result` 這個 JSONB 欄位本來應該只放摘要（final_metrics/device/
# training_meta 這類），但 graph.py 曾經把逐樣本長度的 prediction_source 陣列也塞進去
# （5 筆真實規模的 LSTM 任務因此每筆 result 壓縮後就有 2.3~2.4MB，展開成 HTTP 回應
# 高達 22MB）——這裡在 SQL 查詢端直接把每個節點物件底下的 prediction_source 這個 key
# 剔除掉，不管是新寫入的任務（worker.py 那邊已經不會再寫這個 key，見下方 worker.py 的
# 修正）還是資料庫裡本來就已經肥大的舊任務，查詢當下就會拿到剔除後的小物件，不需要
# 對舊資料做任何遷移/清洗。real-world 驗證：3e4d15c1 那筆的 result 欄位，投影前
# pg_column_size 是 2,440,674 bytes，投影後只剩 812 bytes。
#
# 用 jsonb_each 展開 result（頂層 key 是動態的 model node id，數量/名稱都不固定，
# 不能寫死），對每個節點物件各自做 `value - 'prediction_source'`（jsonb 的「刪除
# 指定 key」運算子，key 不存在時原樣不變，不會報錯），再用 jsonb_object_agg 組回同樣
# 形狀的物件。result 本身可能是 NULL（job 還沒完成）或 '{}'（理論上不會發生，但
# jsonb_each 對空物件會產生 0 列，object_agg 對 0 列的結果是 NULL，所以額外包一層
# COALESCE 讓「原本是空物件」跟「原本是 NULL」不會被搞混）。
_RESULT_PROJECTION = """
    CASE
        WHEN result IS NULL THEN NULL
        ELSE COALESCE(
            (SELECT jsonb_object_agg(key, value - 'prediction_source') FROM jsonb_each(result)),
            '{}'::jsonb
        )
    END
"""


# 流程版本與資料紀錄（2026-11 M1 遷移新增）。舊任務這些欄位是 NULL：pipeline_version=None 代表 2026-11-r1 以前。
# 列表不帶 data_report 全文（每個時間框架一份報告），只帶提示；單筆查詢才帶完整內容。
_PROVENANCE_COLUMNS = ", pipeline_version, warnings, env_fingerprint, data_report"


def _provenance(row, with_report: bool) -> dict:
    def load(v):
        return json.loads(v) if v else None

    report = load(row["data_report"])
    warnings = load(row["warnings"]) or []
    # 執行時的提示（data_report.warnings）優先：worker 用的是實際執行的規則
    if report and report.get("warnings") is not None:
        warnings = report["warnings"]
    out = {"pipeline_version": row["pipeline_version"], "warnings": warnings}
    if with_report:
        out["env_fingerprint"] = load(row["env_fingerprint"])
        out["data_report"] = report
    return out


async def create_job(
    graph_spec: dict, job_type: str = "train", phase: int | None = None, parent_job_id: str | None = None,
    warnings: list | None = None,
) -> str:
    """`graph_spec` 這個欄位名稱是歷史包袱——job_type='infer' 時裡面存的不是節點圖，
    是 `{"target_job_id","target_node_id","start","end"}` 這種推論請求，重用同一個
    JSONB 欄位跟同一套 job 佇列/狀態機制，不用另外建第二套 job 系統。
    """
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO model_training_jobs (graph_spec, status, job_type, phase, parent_job_id, pipeline_version, warnings)
            VALUES ($1, 'pending', $2, $3, $4, $5, $6)
            RETURNING id
            """,
            json.dumps(graph_spec), job_type, phase,
            uuid.UUID(parent_job_id) if parent_job_id else None,
            PIPELINE_VERSION, json.dumps(warnings or []),
        )
        return str(row["id"])
    finally:
        await conn.close()


async def fetch_next_pending() -> dict | None:
    """撿最早的一筆 pending 任務。訓練同時間只能跑一筆（GPU 記憶體通常被單一訓練跑滿），
    這裡不做 SELECT ... FOR UPDATE SKIP LOCKED 之類的多 worker 搶佔設計——
    目前設計只會有一個訓練容器、一個 worker process，不存在多 worker 互搶的情境。
    """
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            """
            SELECT id, graph_spec, job_type, phase FROM model_training_jobs
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT 1
            """
        )
    finally:
        await conn.close()
    if row is None:
        return None
    return {"id": str(row["id"]), "graph_spec": json.loads(row["graph_spec"]), "job_type": row["job_type"],
            "phase": row["phase"]}


async def mark_running(job_id: str, device: str, env_fingerprint: dict | None = None) -> None:
    """執行當下的流程版本以 worker 的程式為準（建立時記的是 web 容器的版本，兩者部署期間可能短暫不同）。"""
    conn = await get_conn()
    try:
        await conn.execute(
            "UPDATE model_training_jobs SET status = 'running', device = $2, started_at = now(), "
            "pipeline_version = $3, env_fingerprint = $4 WHERE id = $1",
            uuid.UUID(job_id), device, PIPELINE_VERSION, json_safe.dumps(env_fingerprint or {}),
        )
    finally:
        await conn.close()


async def save_data_report(job_id: str, report: dict) -> None:
    """讀取報告（各時間框架的排除筆數、rule_version 分布、缺格數、期間提示）；讀完資料、訓練前寫入，
    所以訓練失敗的任務也查得到當時讀到什麼。"""
    conn = await get_conn()
    try:
        await conn.execute(
            "UPDATE model_training_jobs SET data_report = $2 WHERE id = $1",
            uuid.UUID(job_id), json_safe.dumps(report),
        )
    finally:
        await conn.close()


async def mark_done(job_id: str, result: dict) -> None:
    conn = await get_conn()
    try:
        await conn.execute(
            "UPDATE model_training_jobs SET status = 'done', result = $2, finished_at = now() WHERE id = $1",
            uuid.UUID(job_id), json_safe.dumps(result),
        )
    finally:
        await conn.close()


async def mark_failed(job_id: str, error: str) -> None:
    conn = await get_conn()
    try:
        await conn.execute(
            "UPDATE model_training_jobs SET status = 'failed', error = $2, finished_at = now() WHERE id = $1",
            uuid.UUID(job_id), error,
        )
    finally:
        await conn.close()


async def list_jobs(limit: int = 20) -> list[dict]:
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            f"""
            SELECT id, graph_spec, status, error, {_RESULT_PROJECTION} AS result, device, job_type,
                   phase, parent_job_id, created_at, started_at, finished_at{_PROVENANCE_COLUMNS}
            FROM model_training_jobs
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
    finally:
        await conn.close()
    return [
        {
            "job_id": str(r["id"]),
            "graph_spec": json.loads(r["graph_spec"]),
            "status": r["status"],
            "error": r["error"],
            "result": json.loads(r["result"]) if r["result"] else None,
            "device": r["device"],
            "job_type": r["job_type"],
            "phase": r["phase"],
            "parent_job_id": str(r["parent_job_id"]) if r["parent_job_id"] else None,
            "created_at": r["created_at"].isoformat(),
            "started_at": r["started_at"].isoformat() if r["started_at"] else None,
            "finished_at": r["finished_at"].isoformat() if r["finished_at"] else None,
            **_provenance(r, with_report=False),
        }
        for r in rows
    ]


async def get_job(job_id: str) -> dict | None:
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            f"""
            SELECT id, graph_spec, status, error, {_RESULT_PROJECTION} AS result, device, job_type,
                   phase, parent_job_id, created_at, started_at, finished_at{_PROVENANCE_COLUMNS}
            FROM model_training_jobs WHERE id = $1
            """,
            uuid.UUID(job_id),
        )
    finally:
        await conn.close()
    if row is None:
        return None
    return {
        "job_id": str(row["id"]),
        "graph_spec": json.loads(row["graph_spec"]),
        "status": row["status"],
        "error": row["error"],
        "result": json.loads(row["result"]) if row["result"] else None,
        "device": row["device"],
        "job_type": row["job_type"],
        "phase": row["phase"],
        "parent_job_id": str(row["parent_job_id"]) if row["parent_job_id"] else None,
        "created_at": row["created_at"].isoformat(),
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
        **_provenance(row, with_report=True),
    }


async def has_children(job_id: str) -> bool:
    """有沒有別的 job 用 parent_job_id 指到這筆（Phase 2 指 Phase 1、Phase 3 指 Phase 2）。
    刪除前先擋一下，不然會撞到 parent_job_id 的 FK 限制（沒有 ON DELETE CASCADE，
    血緣鏈是刻意設計成不能被動刪掉的），與其讓使用者看到一串 Postgres 的錯誤，
    不如先擋下來給清楚的說明。
    """
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            "SELECT 1 FROM model_training_jobs WHERE parent_job_id = $1 LIMIT 1", uuid.UUID(job_id)
        )
    finally:
        await conn.close()
    return row is not None


async def delete_job(job_id: str) -> bool:
    """刪掉這筆 job；`model_training_progress`／`model_artifacts` 都有掛
    `ON DELETE CASCADE` 在 job_id 上，刪 job 會連帶把進度紀錄跟已保存的模型權重
    一起清掉——如果這筆有收藏（kept）的模型，權重也會一起沒了，呼叫端要注意。
    """
    conn = await get_conn()
    try:
        result = await conn.execute("DELETE FROM model_training_jobs WHERE id = $1", uuid.UUID(job_id))
    finally:
        await conn.close()
    return result != "DELETE 0"


async def get_progress(job_id: str) -> list[dict]:
    """給前端初次載入頁面用（WebSocket 只會推之後的新進度，頁面剛打開時要先補歷史）。"""
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            """
            -- SELECT * 而不是列出 metrics 欄：遷移（新增 metrics 欄）還沒套用之前，讀取歷史進度
            -- 仍然能運作（舊列本來就沒有擴充指標），不會因為欄位還不存在就整支 500
            SELECT *
            FROM model_training_progress
            WHERE job_id = $1
            ORDER BY id ASC
            """,
            uuid.UUID(job_id),
        )
    finally:
        await conn.close()
    return [
        {
            "epoch": r["epoch"],
            "loss": r["loss"],
            "accuracy": r["accuracy"],
            "val_loss": r["val_loss"],
            "val_accuracy": r["val_accuracy"],
            "window_meta": json.loads(r["window_meta"]) if r["window_meta"] else None,
            # 擴充指標（各自有明確名稱，如 rmse／dir_acc／joint_loss），舊 architecture 的列是 None
            "metrics": json.loads(r["metrics"]) if r.get("metrics") else None,
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]
