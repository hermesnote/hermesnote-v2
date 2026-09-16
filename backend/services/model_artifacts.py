"""模型產物的 web 後端查詢/操作（async，給 FastAPI route handler 用）。

實際的保存/推論載入在訓練容器的同步路徑（見 `training/artifacts.py`），這支只負責
「查清單」跟「切換 kept 標記」這種可以在 async 事件迴圈裡直接做、不需要碰 GPU/torch 的操作。
"""

import json
import uuid

from services.hermesnote_db import get_conn


async def list_artifacts(job_id: str | None = None) -> list[dict]:
    conn = await get_conn()
    try:
        if job_id:
            rows = await conn.fetch(
                """
                SELECT job_id, node_id, architecture_key, task_type, kept, created_at
                FROM model_artifacts WHERE job_id = $1 ORDER BY created_at DESC
                """,
                uuid.UUID(job_id),
            )
        else:
            rows = await conn.fetch(
                """
                SELECT job_id, node_id, architecture_key, task_type, kept, created_at
                FROM model_artifacts ORDER BY created_at DESC
                """
            )
    finally:
        await conn.close()
    return [
        {
            "job_id": str(r["job_id"]), "node_id": r["node_id"], "architecture_key": r["architecture_key"],
            "task_type": r["task_type"], "kept": r["kept"], "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


async def mark_kept(job_id: str, node_id: str, kept: bool) -> bool:
    conn = await get_conn()
    try:
        result = await conn.execute(
            "UPDATE model_artifacts SET kept = $3 WHERE job_id = $1 AND node_id = $2",
            uuid.UUID(job_id), node_id, kept,
        )
        return result != "UPDATE 0"
    finally:
        await conn.close()


async def artifact_exists(job_id: str, node_id: str) -> bool:
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            "SELECT 1 FROM model_artifacts WHERE job_id = $1 AND node_id = $2",
            uuid.UUID(job_id), node_id,
        )
        return row is not None
    finally:
        await conn.close()
