"""quotes DB 存取共用邏輯（唯讀）。"""

import os
import re
from datetime import date

import asyncpg
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

QUOTES_DATABASE_URL = os.getenv("QUOTES_DATABASE_URL")

_IDENT_RE = re.compile(r"^[A-Za-z0-9]+$")


def quotes_table(symbol: str, timeframe: str) -> str:
    """symbol / timeframe 會被直接拼進 SQL 表名，先擋掉非白名單字元避免 SQL injection。"""
    if not _IDENT_RE.match(symbol) or not _IDENT_RE.match(timeframe):
        raise HTTPException(status_code=400, detail="symbol/timeframe 格式不正確")
    return f"market.future_taifex_{symbol.lower()}_{timeframe}"


async def fetch_ohlcv_rows(symbol: str, timeframe: str, start: date, end: date) -> list[dict]:
    table = quotes_table(symbol, timeframe)
    conn = await asyncpg.connect(QUOTES_DATABASE_URL)
    try:
        # 同一根 K 棒可能因為資料管線重跑被寫入多次（rule_version 相同、built_at 不同），
        # 用 built_at 排序，讓去重時保留最新一次計算的版本
        rows = await conn.fetch(
            f"""
            SELECT datetime, open, high, low, close, volume
            FROM {table}
            WHERE datetime >= $1 AND datetime < $2
            ORDER BY datetime ASC, built_at ASC
            """,
            start,
            end,
        )
    finally:
        await conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="這段期間查無資料")
    return [dict(r) for r in rows]


async def fetch_ohlcv_dataframe(symbol: str, timeframe: str, start: date, end: date) -> "pd.DataFrame":
    """跟 fetch_ohlcv_rows 撈同一份資料，但給訓練 worker 用——不繞「asyncpg Record → 兩百多萬個
    Python dict → pandas.DataFrame(list of dict)」這條路。1m 級時間框架（十幾年份、兩百多萬列）
    這樣繞，light 是慢，重則整個 process 被系統 OOM killer 砍掉（2026-09-11 實測復現：本機同一組
    設定量到的實際記憶體用量跟 NAS 上 `dmesg` 記錄的 OOM 擊殺數字幾乎一模一樣，兩邊互相印證）。

    asyncpg 的 Record 本身可以當 tuple 拆，直接餵 `pd.DataFrame(records, columns=...)` 讓 pandas
    自己攤成欄位陣列，中間不用先物化成一堆 dict 物件——這條路徑對百萬列等級的資料明顯更省記憶體。
    """
    import pandas as pd

    table = quotes_table(symbol, timeframe)
    conn = await asyncpg.connect(QUOTES_DATABASE_URL)
    try:
        rows = await conn.fetch(
            f"""
            SELECT datetime, open, high, low, close, volume
            FROM {table}
            WHERE datetime >= $1 AND datetime < $2
            ORDER BY datetime ASC, built_at ASC
            """,
            start,
            end,
        )
    finally:
        await conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="這段期間查無資料")
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
