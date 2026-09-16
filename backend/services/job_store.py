"""量化回測任務的持久化：建任務、標記狀態、存結果、查詢（含分段查 K 棒）。"""

import json
import uuid
from datetime import date, datetime, timezone

from services.hermesnote_db import get_conn


def _to_dt(unix_ts: int) -> datetime:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)


def _to_unix(dt: datetime) -> int:
    return int(dt.timestamp())


async def create_job(
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    long_entry_tree: dict,
    long_exit_tree: dict,
    short_entry_tree: dict | None,
    short_exit_tree: dict | None,
) -> str:
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO quant_backtest_jobs
                (symbol, timeframe, start_date, end_date, long_entry_tree, long_exit_tree, short_entry_tree, short_exit_tree, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending')
            RETURNING id
            """,
            symbol, timeframe, start, end,
            json.dumps(long_entry_tree), json.dumps(long_exit_tree),
            json.dumps(short_entry_tree) if short_entry_tree else None,
            json.dumps(short_exit_tree) if short_exit_tree else None,
        )
        return str(row["id"])
    finally:
        await conn.close()


async def mark_running(job_id: str) -> None:
    conn = await get_conn()
    try:
        await conn.execute(
            "UPDATE quant_backtest_jobs SET status = 'running' WHERE id = $1", uuid.UUID(job_id)
        )
    finally:
        await conn.close()


async def mark_failed(job_id: str, error: str) -> None:
    conn = await get_conn()
    try:
        await conn.execute(
            "UPDATE quant_backtest_jobs SET status = 'failed', error = $2, finished_at = now() WHERE id = $1",
            uuid.UUID(job_id), error,
        )
    finally:
        await conn.close()


async def delete_job(job_id: str) -> bool:
    """刪掉一筆回測任務。candles/trades 兩張表對 job_id 都是 ON DELETE CASCADE，
    刪 job 這筆自動連帶清掉，不用分開刪。回傳是否真的刪到東西（job_id 存在過）。"""
    conn = await get_conn()
    try:
        result = await conn.execute("DELETE FROM quant_backtest_jobs WHERE id = $1", uuid.UUID(job_id))
        return result != "DELETE 0"
    finally:
        await conn.close()


async def save_result(
    job_id: str,
    candles: list[dict],
    trades: list[dict],
    summary: dict,
    indicators: list[dict],
    rules: list[dict],
    bar_count: int,
) -> None:
    jid = uuid.UUID(job_id)
    conn = await get_conn()
    try:
        async with conn.transaction():
            await conn.execute(
                """
                UPDATE quant_backtest_jobs
                SET status = 'done', summary = $2, indicators = $3, rules = $4, bar_count = $5, finished_at = now()
                WHERE id = $1
                """,
                jid, json.dumps(summary), json.dumps(indicators), json.dumps(rules), bar_count,
            )

            candle_records = [
                (jid, _to_dt(c["time"]), c["open"], c["high"], c["low"], c["close"], c.get("volume"))
                for c in candles
            ]
            if candle_records:
                await conn.copy_records_to_table(
                    "quant_backtest_candles",
                    records=candle_records,
                    columns=["job_id", "ts", "open", "high", "low", "close", "volume"],
                )

            trade_records = [
                (
                    jid, t["direction"], _to_dt(t["entry_time"]), t["entry_price"], t["entry_reason"],
                    _to_dt(t["exit_time"]) if t["exit_time"] is not None else None,
                    t["exit_price"], t["exit_reason"], t["points"], t["pnl"], t["return_pct"], t["status"],
                )
                for t in trades
            ]
            if trade_records:
                await conn.copy_records_to_table(
                    "quant_backtest_trades",
                    records=trade_records,
                    columns=[
                        "job_id", "direction", "entry_time", "entry_price", "entry_reason",
                        "exit_time", "exit_price", "exit_reason", "points", "pnl", "return_pct", "status",
                    ],
                )
    finally:
        await conn.close()


async def list_jobs(limit: int = 20) -> list[dict]:
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            """
            SELECT id, symbol, timeframe, start_date, end_date, status, error, summary, bar_count, created_at, finished_at
            FROM quant_backtest_jobs
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
            "symbol": r["symbol"],
            "timeframe": r["timeframe"],
            "start_date": r["start_date"].isoformat(),
            "end_date": r["end_date"].isoformat(),
            "status": r["status"],
            "error": r["error"],
            "summary": json.loads(r["summary"]) if r["summary"] else None,
            "bar_count": r["bar_count"],
            "created_at": r["created_at"].isoformat(),
            "finished_at": r["finished_at"].isoformat() if r["finished_at"] else None,
        }
        for r in rows
    ]


async def get_job(job_id: str) -> dict | None:
    """任務本身的中繼資料+摘要，不含交易明細（明細量可能很大，見 get_trades 分頁查詢）。"""
    jid = uuid.UUID(job_id)
    conn = await get_conn()
    try:
        row = await conn.fetchrow("SELECT * FROM quant_backtest_jobs WHERE id = $1", jid)
        if row is None:
            return None
        trade_count = await conn.fetchval(
            "SELECT COUNT(*) FROM quant_backtest_trades WHERE job_id = $1", jid
        )
    finally:
        await conn.close()

    return {
        "job_id": str(row["id"]),
        "symbol": row["symbol"],
        "timeframe": row["timeframe"],
        "start_date": row["start_date"].isoformat(),
        "end_date": row["end_date"].isoformat(),
        "status": row["status"],
        "error": row["error"],
        "summary": json.loads(row["summary"]) if row["summary"] else None,
        "indicators": json.loads(row["indicators"]) if row["indicators"] else None,
        "rules": json.loads(row["rules"]) if row["rules"] else None,
        "bar_count": row["bar_count"],
        "trade_count": trade_count,
        "created_at": row["created_at"].isoformat(),
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
    }


def _trade_row_to_dict(t) -> dict:
    return {
        "direction": t["direction"],
        "entry_time": _to_unix(t["entry_time"]),
        "entry_price": t["entry_price"],
        "entry_reason": t["entry_reason"],
        "exit_time": _to_unix(t["exit_time"]) if t["exit_time"] else None,
        "exit_price": t["exit_price"],
        "exit_reason": t["exit_reason"],
        "points": t["points"],
        "pnl": t["pnl"],
        "return_pct": t["return_pct"],
        "status": t["status"],
    }


async def get_trades(job_id: str, offset: int = 0, limit: int = 200) -> dict:
    """分頁查詢交易明細：186,304 筆這種規模不能一次全部塞進一個回應。"""
    jid = uuid.UUID(job_id)
    conn = await get_conn()
    try:
        total = await conn.fetchval("SELECT COUNT(*) FROM quant_backtest_trades WHERE job_id = $1", jid)
        rows = await conn.fetch(
            """
            SELECT * FROM quant_backtest_trades WHERE job_id = $1
            ORDER BY entry_time LIMIT $2 OFFSET $3
            """,
            jid, limit, offset,
        )
    finally:
        await conn.close()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "trades": [_trade_row_to_dict(t) for t in rows],
    }


async def get_trades_in_range(job_id: str, start: int, end: int, limit: int = 5000) -> list[dict]:
    """給圖表標記用：只抓目前已載入的 K 棒時間窗內的交易，不管總交易數多少，數量都是有界的。"""
    jid = uuid.UUID(job_id)
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            """
            SELECT * FROM quant_backtest_trades
            WHERE job_id = $1 AND (
                (entry_time >= $2 AND entry_time <= $3) OR
                (exit_time >= $2 AND exit_time <= $3)
            )
            ORDER BY entry_time LIMIT $4
            """,
            jid, _to_dt(start), _to_dt(end), limit,
        )
    finally:
        await conn.close()
    return [_trade_row_to_dict(t) for t in rows]


async def save_strategy(job_id: str, name: str | None) -> str:
    """把某筆回測用的條件樹收藏起來，供模型訓練那邊當 Feature Node 取用。
    回測有效不代表對模型預測有用，這裡只是把規則存起來方便重複取用，不是背書。
    """
    jid = uuid.UUID(job_id)
    conn = await get_conn()
    try:
        job = await conn.fetchrow(
            "SELECT symbol, timeframe, long_entry_tree, long_exit_tree, short_entry_tree, short_exit_tree "
            "FROM quant_backtest_jobs WHERE id = $1",
            jid,
        )
        if job is None:
            raise ValueError(f"找不到回測任務 {job_id}")
        row = await conn.fetchrow(
            """
            INSERT INTO quant_saved_strategies
                (job_id, name, symbol, timeframe, long_entry_tree, long_exit_tree, short_entry_tree, short_exit_tree)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            """,
            jid, name, job["symbol"], job["timeframe"],
            job["long_entry_tree"], job["long_exit_tree"], job["short_entry_tree"], job["short_exit_tree"],
        )
        return str(row["id"])
    finally:
        await conn.close()


async def get_saved_strategy_id_for_job(job_id: str) -> str | None:
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            "SELECT id FROM quant_saved_strategies WHERE job_id = $1 AND deleted_at IS NULL", uuid.UUID(job_id)
        )
    finally:
        await conn.close()
    return str(row["id"]) if row else None


async def unsave_strategy(saved_id: str) -> bool:
    """軟刪除：只標記 deleted_at，不真的刪列。已經有模型訓練任務引用這個 saved_id 當
    Feature Node（quant_saved_strategy），真刪除會讓那些舊任務的設定解析失敗；標記後
    UI 清單濾掉即可，舊任務仍能查到原始規則。"""
    conn = await get_conn()
    try:
        result = await conn.execute(
            "UPDATE quant_saved_strategies SET deleted_at = now() WHERE id = $1 AND deleted_at IS NULL",
            uuid.UUID(saved_id),
        )
    finally:
        await conn.close()
    return result != "UPDATE 0"


async def list_saved_strategies() -> list[dict]:
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, job_id, name, symbol, timeframe, created_at FROM quant_saved_strategies "
            "WHERE deleted_at IS NULL ORDER BY created_at DESC"
        )
    finally:
        await conn.close()
    return [
        {
            "saved_id": str(r["id"]), "job_id": str(r["job_id"]), "name": r["name"],
            "symbol": r["symbol"], "timeframe": r["timeframe"], "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


async def get_candles(
    job_id: str, before: int | None = None, around: int | None = None, limit: int = 2000
) -> list[dict]:
    """三種查法：都不給 = 抓最新一段；before = 那個時間點之前一段（往回滑用）；
    around = 跳到某個時間點附近（時間軸拉桿用），從那個時間點開始往後抓一段。"""
    jid = uuid.UUID(job_id)
    conn = await get_conn()
    try:
        if before is not None:
            rows = await conn.fetch(
                """
                SELECT ts, open, high, low, close, volume FROM quant_backtest_candles
                WHERE job_id = $1 AND ts < $2
                ORDER BY ts DESC LIMIT $3
                """,
                jid, _to_dt(before), limit,
            )
            rows = list(reversed(rows))
        elif around is not None:
            rows = await conn.fetch(
                """
                SELECT ts, open, high, low, close, volume FROM quant_backtest_candles
                WHERE job_id = $1 AND ts >= $2
                ORDER BY ts ASC LIMIT $3
                """,
                jid, _to_dt(around), limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT ts, open, high, low, close, volume FROM quant_backtest_candles
                WHERE job_id = $1
                ORDER BY ts DESC LIMIT $2
                """,
                jid, limit,
            )
            rows = list(reversed(rows))
    finally:
        await conn.close()

    return [
        {
            "time": _to_unix(r["ts"]),
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["volume"],
        }
        for r in rows
    ]
