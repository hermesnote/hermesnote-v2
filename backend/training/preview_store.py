"""即時視窗預覽的持久化，跟 training/progress.py（epoch 級 loss/accuracy）是刻意分開的兩張表、
兩條 NOTIFY channel——原因見下面 save_preview_sample 的說明：PostgreSQL 的 NOTIFY payload
有 8000 bytes 上限（https://www.postgresql.org/docs/current/sql-notify.html），一個 window 的
bars（60 根以上的 OHLC）序列化後很容易超過，絕對不能塞進 NOTIFY payload 本身，只能先存進表裡，
NOTIFY 只帶「去哪裡查」的小訊息，讓 WebSocket 端點收到通知後自己去資料表撈完整內容再轉推。

同樣用同步 psycopg2（不是 asyncpg），理由跟 progress.py 一樣：呼叫端是訓練用的背景執行緒，
沒有事件迴圈可以 await。
"""

import json
import os
import queue
import threading

import asyncpg
import psycopg2
from dotenv import load_dotenv

from services.hermesnote_db import HERMESNOTE_DATABASE_URL

load_dotenv()

_SYNC_DATABASE_URL = os.getenv("HERMESNOTE_DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")

PREVIEW_CHANNEL_PREFIX = "model_training_preview_"

# 訓練執行緒呼叫 save_preview_sample() 時，不能直接在那個執行緒裡連 DB——try/except 只能
# 接住「已經拋出的例外」，接不住「連線/查詢一直卡著沒回應」這種情況，DB 一忙，訓練執行緒
# 就跟著卡住，這正是這輪要修的問題。改成：訓練執行緒只把資料丟進一個有容量上限的佇列
# （非阻塞、滿了就直接丟棄這一筆，不等待），真正的 DB 寫入交給一個獨立的背景執行緒處理，
# 那邊才加連線/查詢逾時。佇列上限也保證不會因為 DB 一直忙碌而無限堆積記憶體。
_QUEUE_MAXSIZE = 50
_CONNECT_TIMEOUT_SEC = 3
_STATEMENT_TIMEOUT_MS = 3000

_queue: "queue.Queue[tuple[str, str, dict]]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
_worker_lock = threading.Lock()
_worker_started = False
_dropped_count = 0  # 佇列滿了被丟棄的筆數，只用來偶爾印一次 log，不是給前端用的統計


def _ensure_worker_started() -> None:
    global _worker_started
    if _worker_started:
        return
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_worker_loop, name="preview-sample-writer", daemon=True)
        t.start()
        _worker_started = True


def _worker_loop() -> None:
    while True:
        job_id, node_id, preview = _queue.get()
        try:
            _write_sample_sync(job_id, node_id, preview)
        except Exception as e:
            print(f"[preview_store] 背景寫入失敗（已忽略，不影響訓練本身）：{e}", flush=True)


def _write_sample_sync(job_id: str, node_id: str, preview: dict) -> None:
    """真正連 DB 寫入的地方，只在背景執行緒裡跑——connect_timeout 擋連不上的情況，
    statement_timeout 擋「連上了但查詢卡住」的情況（例如 DB 當下負載很高），兩種逾時
    都只會讓這一筆預覽寫入失敗，不會拖累其他佇列裡的項目，也不會拖累訓練本身
    （訓練執行緒早就在 save_preview_sample() 那一刻就繼續往下跑了）。
    """
    conn = psycopg2.connect(_SYNC_DATABASE_URL, connect_timeout=_CONNECT_TIMEOUT_SEC)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {_STATEMENT_TIMEOUT_MS}")
            cur.execute(
                """
                INSERT INTO model_training_preview_samples
                    (job_id, node_id, epoch, source, decision_ts, target_ts, horizon,
                     task_type, n_classes, labeling_rule, bars, actual, predicted)
                VALUES (%s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    job_id, node_id, preview["epoch"], preview["source"],
                    preview["decision_ts"], preview["target_ts"], preview["horizon"],
                    preview["task_type"], preview["n_classes"], preview.get("labeling_rule"),
                    json.dumps(preview["bars"]), preview["actual"], preview["predicted"],
                ),
            )
            sample_id = cur.fetchone()[0]
            payload = json.dumps({"type": "preview", "node_id": node_id, "sample_id": sample_id})
            cur.execute(f"NOTIFY {PREVIEW_CHANNEL_PREFIX}{job_id.replace('-', '_')}, %s", (payload,))
    finally:
        conn.close()


def save_preview_sample(job_id: str, node_id: str, preview: dict) -> None:
    """訓練執行緒呼叫的入口——只做一次非阻塞的 queue.put_nowait()，不連 DB、不等待，
    幾乎是常數時間（頂多是拷貝 bars 這個 list 的參考，不是深拷貝）。佇列滿了就直接丟棄
    這一筆，安靜跳過，不阻塞、不重試、不讓例外往外拋——下一筆節流通過的預覽很快就會來，
    偶爾漏一筆不影響整體體驗，但讓訓練執行緒卡住才是真正不能接受的事。
    """
    global _dropped_count
    _ensure_worker_started()
    try:
        _queue.put_nowait((job_id, node_id, preview))
    except queue.Full:
        _dropped_count += 1
        if _dropped_count % 20 == 1:  # 不要每一筆都印，佇列持續滿的話這裡會洗版
            print(f"[preview_store] 預覽佇列已滿，已丟棄 {_dropped_count} 筆（累計），DB 可能忙碌或太慢", flush=True)


def _row_to_dict(row) -> dict:
    return {
        "id": row["id"], "node_id": row["node_id"], "epoch": row["epoch"], "source": row["source"],
        "decision_ts": int(row["decision_ts"].timestamp()), "target_ts": int(row["target_ts"].timestamp()),
        "horizon": row["horizon"], "task_type": row["task_type"], "n_classes": row["n_classes"],
        "labeling_rule": row["labeling_rule"], "bars": json.loads(row["bars"]),
        "actual": row["actual"], "predicted": row["predicted"],
        "created_at": row["created_at"].isoformat(),
    }


async def get_preview_sample(conn: "asyncpg.Connection", sample_id: int) -> dict | None:
    """WebSocket 端點收到 NOTIFY（只帶 sample_id）之後，用這支查完整內容（含 bars）
    再轉推給瀏覽器——這是刻意把「小通知」跟「大內容」分開查詢的地方。
    """
    row = await conn.fetchrow("SELECT * FROM model_training_preview_samples WHERE id = $1", sample_id)
    return _row_to_dict(row) if row else None


async def list_preview_samples(
    job_id: str, node_id: str, limit: int = 50, before_id: int | None = None
) -> list[dict]:
    """完成後歷史瀏覽用：依 id 游標分頁往回查（新到舊），不是把全期間資料一次全部丟給前端——
    訓練時間長、抽樣頻率高的 job 可能累積幾千筆，一次查完既浪費頻寬也沒必要。
    """
    conn = await asyncpg.connect(HERMESNOTE_DATABASE_URL)
    try:
        if before_id is not None:
            rows = await conn.fetch(
                """
                SELECT * FROM model_training_preview_samples
                WHERE job_id = $1 AND node_id = $2 AND id < $3
                ORDER BY id DESC LIMIT $4
                """,
                job_id, node_id, before_id, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT * FROM model_training_preview_samples
                WHERE job_id = $1 AND node_id = $2
                ORDER BY id DESC LIMIT $3
                """,
                job_id, node_id, limit,
            )
    finally:
        await conn.close()
    return [_row_to_dict(r) for r in rows]
