"""訓練進度寫入：用同步 psycopg2（不是 asyncpg），因為呼叫端是在訓練用的背景執行緒裡
（`asyncio.to_thread` 丟出去跑同步/GPU 運算的那條），沒有事件迴圈可以 await，用同步連線最簡單，
不用另外搞執行緒↔事件迴圈的橋接。

每寫一筆進度，同時發一個 Postgres NOTIFY，讓 web 後端的 WebSocket 端點能即時收到、推給瀏覽器，
不需要輪詢（見 docs/record/archive/2026-09-08-training-plan.md）。
"""

import json
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

_SYNC_DATABASE_URL = os.getenv("HERMESNOTE_DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")

CHANNEL_PREFIX = "model_training_progress_"


def write_progress(job_id: str, node_id: str, epoch: int, metrics: dict) -> None:
    """epoch 級的 loss/accuracy 進度，跟即時視窗預覽（見 training/preview_store.py）是完全
    分開的兩條管道——預覽的 bars 資料量會隨 window 變大，不能跟這裡的小 payload 混在一起，
    PostgreSQL 的 NOTIFY payload 上限是 8000 bytes（見官方文件），這裡的 metrics 本來就很小，
    穩定維持在限制以內。
    """
    conn = psycopg2.connect(_SYNC_DATABASE_URL)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_training_progress
                    (job_id, epoch, loss, accuracy, val_loss, val_accuracy, window_meta)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job_id, epoch,
                    metrics.get("loss"), metrics.get("accuracy"),
                    metrics.get("val_loss"), metrics.get("val_accuracy"),
                    json.dumps({"node_id": node_id}),
                ),
            )
            payload = json.dumps({"node_id": node_id, "epoch": epoch, **metrics})
            cur.execute(f"NOTIFY {CHANNEL_PREFIX}{job_id.replace('-', '_')}, %s", (payload,))
    finally:
        conn.close()
