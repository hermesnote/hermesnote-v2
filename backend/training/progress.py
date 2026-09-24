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

from training import json_safe

load_dotenv()

_SYNC_DATABASE_URL = os.getenv("HERMESNOTE_DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")

CHANNEL_PREFIX = "model_training_progress_"


# 固定四欄（圖表用）與只給 log 看的診斷欄位（不是訓練指標）：不算「擴充指標」。
CORE_METRIC_KEYS = ("loss", "accuracy", "val_loss", "val_accuracy")
DIAGNOSTIC_KEYS = ("gpu_mem_mb",)


def split_metrics(metrics: dict) -> tuple[dict, dict]:
    """把一輪的 metrics 拆成 (固定四欄, 擴充指標)。擴充指標（例如 rmse、val_mae、雙頭的 joint_loss／
    mse／bce／dir_acc 及其 val_ 版本）存進 `metrics JSONB` 欄，各自有明確名稱，不借用 loss／accuracy
    欄裝別的東西；沒有擴充指標時該欄是 NULL。None 不當成 0 寫入。"""
    core = {k: metrics.get(k) for k in CORE_METRIC_KEYS}
    extras = {
        k: v for k, v in metrics.items()
        if k not in CORE_METRIC_KEYS and k not in DIAGNOSTIC_KEYS and v is not None
    }
    return core, extras


def write_progress(job_id: str, node_id: str, epoch: int, metrics: dict) -> None:
    """epoch 級的 loss/accuracy 進度，跟即時視窗預覽（見 training/preview_store.py）是完全
    分開的兩條管道——預覽的 bars 資料量會隨 window 變大，不能跟這裡的小 payload 混在一起，
    PostgreSQL 的 NOTIFY payload 上限是 8000 bytes（見官方文件），這裡的 metrics 本來就很小，
    穩定維持在限制以內。

    NOTIFY 訊息是扁平格式（loss、accuracy…在最上層），擴充指標另外以 `metrics` 巢狀物件附上
    （跟資料庫歷史查詢 GET /progress 的形狀一致）。
    """
    core, extras = split_metrics(metrics)
    conn = psycopg2.connect(_SYNC_DATABASE_URL)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_training_progress
                    (job_id, epoch, loss, accuracy, val_loss, val_accuracy, window_meta, metrics)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job_id, epoch, core["loss"], core["accuracy"], core["val_loss"], core["val_accuracy"],
                    json.dumps({"node_id": node_id}), json_safe.dumps(extras) if extras else None,
                ),
            )
            payload_dict = {"node_id": node_id, "epoch": epoch, **metrics}
            if extras:
                payload_dict["metrics"] = extras
            payload = json_safe.dumps(payload_dict)
            cur.execute(f"NOTIFY {CHANNEL_PREFIX}{job_id.replace('-', '_')}, %s", (payload,))
    finally:
        conn.close()
