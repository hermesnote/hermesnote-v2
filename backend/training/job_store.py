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
_PROVENANCE_COLUMNS = (", pipeline_version, warnings, env_fingerprint, data_report"
                       # 實驗層（2026-11-16 M2～M7 遷移新增）；舊任務都是 NULL
                       ", experiment_id, arm_id, frame_id, model_seed, split_seed, gate_check, requested_by")


def _provenance(row, with_report: bool) -> dict:
    def load(v):
        return json.loads(v) if v else None

    report = load(row["data_report"])
    warnings = load(row["warnings"]) or []
    # 執行時的提示（data_report.warnings）優先：worker 用的是實際執行的規則
    if report and report.get("warnings") is not None:
        warnings = report["warnings"]
    out = {"pipeline_version": row["pipeline_version"], "warnings": warnings,
           "experiment_id": str(row["experiment_id"]) if row["experiment_id"] else None,
           "arm_id": str(row["arm_id"]) if row["arm_id"] else None,
           "frame_id": str(row["frame_id"]) if row["frame_id"] else None,
           "model_seed": row["model_seed"], "split_seed": row["split_seed"], "gate_check": load(row["gate_check"]),
           "requested_by": row["requested_by"]}
    if with_report:
        out["env_fingerprint"] = load(row["env_fingerprint"])
        out["data_report"] = report
    return out


async def create_job(
    graph_spec: dict, job_type: str = "train", phase: int | None = None, parent_job_id: str | None = None,
    warnings: list | None = None, *, experiment_id: str | None = None, arm_id: str | None = None,
    frame_id: str | None = None, model_seed: int | None = None, split_seed: int | None = None,
    gate_check: dict | None = None, requested_by: str | None = None,
) -> str:
    """`graph_spec` 這個欄位名稱是歷史包袱——job_type='infer' 時裡面存的不是節點圖，
    是 `{"target_job_id","target_node_id","start","end"}` 這種推論請求，重用同一個
    JSONB 欄位跟同一套 job 佇列/狀態機制，不用另外建第二套 job 系統。
    """
    conn = await get_conn()
    try:
        cols = ["graph_spec", "status", "job_type", "phase", "parent_job_id", "pipeline_version", "warnings"]
        vals = [json.dumps(graph_spec), "pending", job_type, phase, uuid.UUID(parent_job_id) if parent_job_id else None,
                PIPELINE_VERSION, json.dumps(warnings or [])]
        # 實驗層欄位只在有值時才寫（舊呼叫端的 SQL 與之前相同）
        for c, v in (("experiment_id", experiment_id), ("arm_id", arm_id), ("frame_id", frame_id)):
            if v is not None:
                cols.append(c)
                vals.append(uuid.UUID(str(v)))
        for c, v in (("model_seed", model_seed), ("split_seed", split_seed)):
            if v is not None:
                cols.append(c)
                vals.append(int(v))
        if gate_check is not None:
            cols.append("gate_check")
            vals.append(json_safe.dumps(gate_check))
        if requested_by is not None:  # 驗證身分（admin:<email>／key:<名稱>），見 auth.require_model_writer
            cols.append("requested_by")
            vals.append(requested_by)
        ph = ", ".join(f"${i + 1}" for i in range(len(cols)))
        row = await conn.fetchrow(
            f"INSERT INTO model_training_jobs ({', '.join(cols)}) VALUES ({ph}) RETURNING id", *vals)
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
    所以訓練失敗的任務也查得到當時讀到什麼。worker 程序中只在任務仍屬於本實例時寫入（見 training/fence.py）。"""
    clause, extra = _owned_clause(3)
    conn = await get_conn()
    try:
        status = await conn.execute(
            "UPDATE model_training_jobs SET data_report = $2 WHERE id = $1" + clause,
            uuid.UUID(job_id), json_safe.dumps(report), *extra,
        )
    finally:
        await conn.close()
    if extra and not status.endswith(" 1"):
        from training import fence

        raise fence.LostOwnership(f"任務 {job_id} 已不屬於 worker 實例 {fence.INSTANCE}，讀取報告不寫入")


# ── worker 排他與任務狀態（2026-11-18：中斷後殘留 running 與失鎖後舊 worker 的處理） ───────────────────
# 1. 單一實例鎖（WorkerLock）：worker 啟動時用一條專用連線取得 PostgreSQL session 層級 advisory lock，整個程序期間持有。
#    連線斷開時資料庫自動釋放；伺服器端 keepalive 30／10／3 讓網路中斷的連線約 1 分鐘內被判定失效。
# 2. 認領與回收都在「持有鎖的那條連線」上執行，同一條 SQL 內確認仍持有鎖：失去鎖的 worker 無法再認領或回收。
#    認領是原子操作（FOR UPDATE SKIP LOCKED），同時寫入 worker 身分（env_fingerprint.worker）。
# 3. 回收：持有鎖的 worker 啟動時，把「running、有 worker 身分、實例 ID 不是自己」的任務改成 failed。
#    沒有 worker 身分的舊任務不動。
# 4. 寫入的所有權檢查（training/fence.py）：worker 的每一筆寫入都先確認任務仍是 running 且屬於本實例；
#    任務列本身的寫入（讀取報告、完成、失敗）直接把同樣的條件放在 UPDATE 的 WHERE。
WORKER_LOCK_KEY = 4848_2026_1118  # pg_advisory_lock 的 bigint 鍵（訓練 worker 專用）
INTERRUPTED_ERROR = "worker 中斷，原因未判定"
LOCK_QUERY_TIMEOUT_SEC = 5
RESOURCE_PROBE = None  # worker 設定：回傳 data_report.resources 內容的函式（完成／失敗寫入時一併記錄）

_HOLDS_LOCK = ("EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid() AND granted "
               f"AND ((classid::bigint << 32) | objid::bigint) = {WORKER_LOCK_KEY})")


class WorkerLock:
    """持有 advisory lock 的連線。同一條 asyncpg 連線不能同時執行兩個查詢，所有使用都經過 self._mutex。"""

    def __init__(self, conn):
        import asyncio

        self.conn = conn
        self._mutex = asyncio.Lock()

    async def alive(self) -> bool:
        """仍持有鎖嗎？連線失效、查詢逾時（網路中斷）都視為失去鎖。"""
        import asyncio

        try:
            if self.conn.is_closed():
                return False
            async with self._mutex:
                return bool(await asyncio.wait_for(self.conn.fetchval("SELECT " + _HOLDS_LOCK),
                                                   LOCK_QUERY_TIMEOUT_SEC))
        except Exception:  # noqa: BLE001
            return False

    async def claim_next_pending(self, device: str, env_fingerprint: dict) -> dict | None:
        """原子認領最早的 pending 任務；沒有可認領的任務、被搶先、或已失去鎖時回傳 None。"""
        async with self._mutex:
            row = await self.conn.fetchrow(
                f"""
                UPDATE model_training_jobs SET status = 'running', device = $1, started_at = now(),
                       pipeline_version = $2, env_fingerprint = $3
                WHERE id = (SELECT id FROM model_training_jobs WHERE status = 'pending'
                            ORDER BY created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED)
                  AND status = 'pending' AND {_HOLDS_LOCK}
                RETURNING id, graph_spec, job_type, phase
                """,
                device, PIPELINE_VERSION, json_safe.dumps(env_fingerprint or {}), timeout=30,
            )
        if row is None:
            return None
        return {"id": str(row["id"]), "graph_spec": json.loads(row["graph_spec"]), "job_type": row["job_type"],
                "phase": row["phase"]}

    async def reclaim_orphans(self, my_instance: str) -> list[dict]:
        """回收其他實例殘留的 running。回傳 [{id, instance}]；只動 status／finished_at／error。"""
        async with self._mutex:
            rows = await self.conn.fetch(
                f"""
                UPDATE model_training_jobs
                SET status = 'failed', finished_at = now(),
                    error = $2 || '（原 worker 實例 ' || (env_fingerprint->'worker'->>'instance')
                            || '，由實例 ' || $1 || ' 啟動時回收）'
                WHERE status = 'running'
                  AND env_fingerprint ? 'worker'
                  AND (env_fingerprint->'worker'->>'instance') IS NOT NULL
                  AND (env_fingerprint->'worker'->>'instance') <> $1
                  AND {_HOLDS_LOCK}
                RETURNING id, env_fingerprint->'worker'->>'instance' AS instance
                """,
                my_instance, INTERRUPTED_ERROR, timeout=60,
            )
        return [{"id": str(r["id"]), "instance": r["instance"]} for r in rows]

    async def close(self) -> None:
        await self.conn.close()


async def acquire_worker_lock() -> "WorkerLock | None":
    """取得鎖就回傳 WorkerLock；鎖被其他程序持有時回傳 None。"""
    import asyncpg

    from services.hermesnote_db import HERMESNOTE_DATABASE_URL

    conn = await asyncpg.connect(HERMESNOTE_DATABASE_URL, timeout=15, server_settings={
        "application_name": "hermesnote-training-worker-lock",
        "tcp_keepalives_idle": "30", "tcp_keepalives_interval": "10", "tcp_keepalives_count": "3"})
    if await conn.fetchval("SELECT pg_try_advisory_lock($1)", WORKER_LOCK_KEY):
        return WorkerLock(conn)
    await conn.close()
    return None


def _owned_clause(first_param: int) -> tuple[str, list]:
    """任務列寫入的所有權條件：worker 程序（fence.INSTANCE 已設定）時必須仍是 running 且為本實例認領。"""
    from training import fence

    if fence.INSTANCE is None:
        return "", []
    return (f" AND status = 'running' AND (env_fingerprint->'worker'->>'instance') = ${first_param}",
            [fence.INSTANCE])


def _resources() -> str | None:
    if RESOURCE_PROBE is None:
        return None
    try:
        r = RESOURCE_PROBE()
    except Exception:  # noqa: BLE001 — 紀錄失敗不影響任務狀態
        return None
    return json_safe.dumps(r) if r else None


async def _finish(job_id: str, sets: str, args: list) -> bool:
    res = _resources()
    if res is not None:
        sets += (", data_report = COALESCE(data_report, '{}'::jsonb) || "
                 f"jsonb_build_object('resources', ${len(args) + 2}::jsonb)")
        args = args + [res]
    clause, extra = _owned_clause(len(args) + 2)
    conn = await get_conn()
    try:
        status = await conn.execute(f"UPDATE model_training_jobs SET {sets} WHERE id = $1" + clause,
                                    uuid.UUID(job_id), *args, *extra)
    finally:
        await conn.close()
    return status.endswith(" 1")


async def mark_done(job_id: str, result: dict) -> bool:
    """worker 寫入完成。worker 程序中只有任務仍是 running 且為本實例認領時生效；回傳是否寫入。"""
    ok = await _finish(job_id, "status = 'done', result = $2, finished_at = now()", [json_safe.dumps(result)])
    if not ok:
        print(f"[worker] 任務 {job_id} 已不是本 worker 執行中的狀態（可能已取消或被回收），完成結果不寫入", flush=True)
    return ok


async def mark_failed_by_worker(job_id: str, error: str) -> bool:
    """worker 寫入失敗。條件同 mark_done；使用者手動中斷（API）仍用 mark_failed。"""
    return await _finish(job_id, "status = 'failed', error = $2, finished_at = now()", [error])


async def mark_failed(job_id: str, error: str) -> None:
    conn = await get_conn()
    try:
        await conn.execute(
            "UPDATE model_training_jobs SET status = 'failed', error = $2, finished_at = now() WHERE id = $1",
            uuid.UUID(job_id), error,
        )
    finally:
        await conn.close()


async def list_job_summaries(limit: int = 1000) -> list[dict]:
    """訓練頁下拉選單用的輕量清單：只有 ID、類型、狀態、建立時間（不讀 graph_spec／result，一百多筆也只有十幾 KB）。"""
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, job_type, status, created_at FROM model_training_jobs ORDER BY created_at DESC LIMIT $1",
            limit,
        )
    finally:
        await conn.close()
    return [{"job_id": str(r["id"]), "job_type": r["job_type"], "status": r["status"],
             "created_at": r["created_at"].isoformat() if r["created_at"] else None} for r in rows]


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
