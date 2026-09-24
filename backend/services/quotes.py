"""quotes DB 存取共用邏輯（唯讀）。"""

import os
import re
from datetime import date

import asyncpg
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

QUOTES_DATABASE_URL = os.getenv("QUOTES_DATABASE_URL")

# 表名由 symbol／timeframe 直接拼進 SQL，所以一律先過白名單。timeframe 允許底線，才接得住
# 既有的 `daily_day`／`daily_full` 日線資料表（`market.future_taifex_tx_daily_day`）；仍然不允許
# 空白、引號、點、分號、連字號等任何可以跳出識別字的字元，也不允許空字串。
# 注意：worker 用 `tf.split("_", 1)` 把 "tx_daily_day" 拆成 symbol="tx"、timeframe="daily_day"，
# symbol 自己不含底線，所以拆分不會有歧義。
_TIMEFRAME_RE = re.compile(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*")
_SYMBOL_RE = re.compile(r"[A-Za-z0-9]+")  # symbol 本身不含底線（拆分 "tx_daily_day" 依賴這一點）


def quotes_table(symbol: str, timeframe: str) -> str:
    """symbol / timeframe 會被直接拼進 SQL 表名，先擋掉非白名單字元避免 SQL injection。"""
    # 用 fullmatch：正則裡的 `$` 會放行尾端換行字元，fullmatch 不會
    if not _SYMBOL_RE.fullmatch(symbol) or not _TIMEFRAME_RE.fullmatch(timeframe):
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
        # bar_end_ts 不是每張行情表都有（例如 market.future_taifex_spread 就沒有），直接 SELECT 會 SQL 報錯。
        # 先問 information_schema 這張表有沒有這欄：有就一起撈，沒有就只撈 OHLCV、回傳的 DataFrame 不帶
        # bar_end_ts 欄——下游（training/time_semantics.py）把「沒有這欄」視為資訊可用時間未知（None），
        # 不猜、不補。表名已過 quotes_table 白名單，schema／表名拆開後用參數綁定查詢。
        schema, _, name = table.partition(".")
        has_end = bool(await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = $2 AND column_name = 'bar_end_ts')",
            schema, name,
        ))
        cols = ["datetime", "open", "high", "low", "close", "volume"] + (["bar_end_ts"] if has_end else [])
        rows = await conn.fetch(
            f"""
            SELECT {", ".join(cols)}
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
    # 兩個時間欄位語意不同，不能混用：
    #   datetime    K 棒的「識別時間」（棒起點）——決定這根棒在時間軸上的位置、對齊與切分都用它。
    #               日線的 datetime 是 08:45，但那根日棒要收盤（bar_end_ts，13:45）才有完整的 OHLCV。
    #   bar_end_ts  這根棒的「資訊可用時間」——完整日棒的資訊最早只能在這個時間之後才用來決策，
    #               不能當成 08:45 就已經知道。訓練樣本、預覽與推論結果的「決策時刻」用它標示
    #               （見 training/graph.py 的 available_ts）。
    # 這個欄位只是附加給呼叫端標時間用，特徵計算只取 open/high/low/close/volume，不受影響。
    return pd.DataFrame(rows, columns=cols)
