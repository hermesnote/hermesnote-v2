"""最終模型推論結果的持久化——跟 training/preview_store.py（訓練時的抽樣紀錄）是完全不同
性質的資料，不要混用同一張表或同一個查詢端點：抽樣紀錄是訓練途中隨機抽的少量展示樣本，
這裡存的是「模型對一整段期間所有列」的真正推論結果，量級可以到百萬列，一律分批寫入、
分頁查詢，絕對不整包塞進 model_training_jobs.result 這個 JSONB 欄位（那樣會讓單一 job
的資料筆巨大、GET /jobs/{job_id} 回應也會跟著肥大）。

output_type 決定怎麼解讀 predicted/probabilities 這兩欄：
  'class'       — 純量類別（沒有 predict_proba 的分類），predicted 是類別索引，probabilities 是 NULL
  'probability' — predict_proba 的完整機率向量，predicted 是 argmax 類別索引（方便排序/篩選用），
                  probabilities 存完整向量——原始機率不能只留 argmax 就丟掉，之後機率門檻策略要用
  'regression'  — 連續數值，predicted 是那個數值，probabilities 是 NULL
"""

import json
import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

load_dotenv()

_SYNC_DATABASE_URL = os.getenv("HERMESNOTE_DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")

_CONNECT_TIMEOUT_SEC = 5
_STATEMENT_TIMEOUT_MS = 30000  # 分批寫入百萬列本身就需要一點時間，逾時值比預覽的 3 秒寬鬆很多
CHUNK_SIZE = 5000


def row_to_output(row: list[float], task_type: str, is_probability: bool) -> tuple[str, float, str | None]:
    """把 predict()/predict_proba() 某一列的原始輸出，轉成要存進 DB 的 (output_type, predicted, probabilities_json)。
    這裡不做任何「只留摘要」的簡化——is_probability 時完整機率向量原封不動存進 probabilities，
    predicted 只是額外方便查詢用的 argmax，不是取代機率向量。
    """
    if is_probability:
        predicted_class = max(range(len(row)), key=lambda i: row[i])
        return "probability", float(predicted_class), json.dumps([float(x) for x in row])
    if task_type == "regression":
        return "regression", float(row[0]), None
    return "class", float(row[0]), None


class _PredictionWriter:
    """串流寫入用：呼叫端算完一批（chunk）預測就呼叫一次 write_chunk()，不用先把整個
    推論期間的 predictions 全部算完、存成一個大 list，才回頭分批寫 DB——那樣「分批」
    只解決了寫入 DB 這一步，算預測本身那個大 list 早就把記憶體吃掉了。這裡從一開始就
    是「算一批、寫一批、丟掉這一批」，尖峰記憶體只跟 chunk 大小有關，不跟全期間筆數
    成正比。同一條 DB 連線重複使用，不是每批重新連線。
    """

    def __init__(self, job_id: str, node_id: str):
        self.job_id = job_id
        self.node_id = node_id
        self.written = 0
        self._conn = psycopg2.connect(_SYNC_DATABASE_URL, connect_timeout=_CONNECT_TIMEOUT_SEC)
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {_STATEMENT_TIMEOUT_MS}")

    def write_chunk(self, timestamps: list[int], predictions: list[list[float]], task_type: str, is_probability: bool) -> None:
        if not timestamps:
            return
        rows = []
        for ts, pred in zip(timestamps, predictions):
            output_type, predicted, probabilities = row_to_output(pred, task_type, is_probability)
            rows.append((self.job_id, self.node_id, ts, output_type, predicted, probabilities))
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO model_inference_predictions
                    (job_id, node_id, ts, output_type, predicted, probabilities)
                VALUES %s
                """,
                rows,
                template="(%s, %s, to_timestamp(%s), %s, %s, %s)",
            )
        self.written += len(rows)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "_PredictionWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_prediction_writer(job_id: str, node_id: str) -> _PredictionWriter:
    return _PredictionWriter(job_id, node_id)


def save_predictions_chunked(
    job_id: str, node_id: str, timestamps: list[int], predictions: list[list[float]],
    task_type: str = "classification", is_probability: bool = False,
) -> int:
    """一次性版本（呼叫端已經有完整 timestamps/predictions 在手上時用）——內部一樣是
    分批寫入，只是「算預測」那一步不是這支的責任。串流情境（inference.py 的
    run_inference_chunked）請直接用 open_prediction_writer()，不要繞過這支重新組一個
    完整 list 再呼叫這裡，那樣就白做了省記憶體的事。
    """
    if not timestamps:
        return 0
    writer = open_prediction_writer(job_id, node_id)
    try:
        for i in range(0, len(timestamps), CHUNK_SIZE):
            writer.write_chunk(
                timestamps[i:i + CHUNK_SIZE], predictions[i:i + CHUNK_SIZE], task_type, is_probability,
            )
    finally:
        writer.close()
    return writer.written


async def list_predictions(
    job_id: str, node_id: str,
    start: int | None = None, end: int | None = None,
    cursor: int | None = None, limit: int = 1000,
) -> list[dict]:
    """最終模型推論的逐列結果，依時間區間（start/end，unix 秒）＋ id 游標分頁查——
    絕對不是把全期間（可能百萬列）一次全部丟給前端，這是「最終模型推論」專用的查詢，
    跟 preview_store.list_preview_samples（訓練時抽樣紀錄）刻意分開，回應結構也不同，
    使用端不應該混用。probabilities 有值時完整回傳，不是只給 predicted 那個 argmax。
    """
    import asyncpg

    from services.hermesnote_db import HERMESNOTE_DATABASE_URL

    conditions = ["job_id = $1", "node_id = $2"]
    params: list = [job_id, node_id]
    if start is not None:
        params.append(start)
        conditions.append(f"ts >= to_timestamp(${len(params)})")
    if end is not None:
        params.append(end)
        conditions.append(f"ts < to_timestamp(${len(params)})")
    if cursor is not None:
        params.append(cursor)
        conditions.append(f"id > ${len(params)}")
    params.append(limit)
    query = f"""
        SELECT id, ts, output_type, predicted, probabilities FROM model_inference_predictions
        WHERE {' AND '.join(conditions)}
        ORDER BY id ASC LIMIT ${len(params)}
    """
    conn = await asyncpg.connect(HERMESNOTE_DATABASE_URL)
    try:
        rows = await conn.fetch(query, *params)
    finally:
        await conn.close()
    return [
        {
            "id": r["id"], "ts": int(r["ts"].timestamp()), "output_type": r["output_type"],
            "predicted": r["predicted"],
            "probabilities": json.loads(r["probabilities"]) if r["probabilities"] else None,
        }
        for r in rows
    ]
