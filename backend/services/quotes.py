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


# ── 訓練讀取規則（流程版本 2026-11-r1 起，見 training/provenance.py）──────────────────────────────
# 1. 排除 session='other'：研究期間只有 5 個星期六各一根 13:45 的零星成交（1～5 口），在 30m／60m 會和
#    日盤那根撞同一個 datetime，舊的「同時間保留最後一筆」可能把真正的日盤 K 棒換成這一根。
#    只排除 'other'，不寫成 IN ('day','night')——daily_full 的 session 是 'full'。
# 2. 依 bar_start_ts 排序並驗證唯一；datetime（graph 目前的索引）也驗證唯一。仍有重複就報錯並列出明細，
#    不再靜默保留最後一筆。
# 3. 讀取報告（排除筆數與日期、rule_version／session 分布、缺格數）放在 DataFrame.attrs["read_report"]，
#    由 worker 寫進任務紀錄。統計在 SQL 端算，不把字串欄位撈進記憶體（1m 十幾年份兩三百萬列）。
OPTIONAL_COLUMNS = ("bar_end_ts", "bar_start_ts", "session", "trading_day", "rule_version")
EXCLUDED_SESSIONS = ("other",)
_INTRADAY_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60}
_DUP_DETAIL_LIMIT = 20


class QuoteIntegrityError(ValueError):
    """讀取的行情違反唯一性等前提；訊息含明細，worker 會把任務標記為失敗。"""


def duplicate_details(values, label: str) -> list[str]:
    """回傳重複值的明細（最多 _DUP_DETAIL_LIMIT 筆），沒有重複回空串列。"""
    import pandas as pd

    s = pd.Series(pd.to_datetime(values))
    counts = s[s.duplicated(keep=False)].value_counts().sort_index()
    return [f"{label}={t} 出現 {n} 次" for t, n in list(counts.items())[:_DUP_DETAIL_LIMIT]] + (
        [f"……共 {len(counts)} 個重複時間"] if len(counts) > _DUP_DETAIL_LIMIT else [])


async def fetch_ohlcv_dataframe(symbol: str, timeframe: str, start: date, end: date) -> "pd.DataFrame":
    """跟 fetch_ohlcv_rows 撈同一份資料，但給訓練 worker 用——不繞「asyncpg Record → 兩百多萬個
    Python dict → pandas.DataFrame(list of dict)」這條路。1m 級時間框架（十幾年份、兩百多萬列）
    這樣繞，light 是慢，重則整個 process 被系統 OOM killer 砍掉（2026-09-11 實測復現：本機同一組
    設定量到的實際記憶體用量跟 NAS 上 `dmesg` 記錄的 OOM 擊殺數字幾乎一模一樣，兩邊互相印證）。

    asyncpg 的 Record 本身可以當 tuple 拆，直接餵 `pd.DataFrame(records, columns=...)` 讓 pandas
    自己攤成欄位陣列，中間不用先物化成一堆 dict 物件——這條路徑對百萬列等級的資料明顯更省記憶體。

    讀取規則見上方說明（排除 other、唯一性驗證、讀取報告）。
    """
    import pandas as pd

    table = quotes_table(symbol, timeframe)
    conn = await asyncpg.connect(QUOTES_DATABASE_URL)
    try:
        # 選用欄位不是每張行情表都有（例如 market.future_taifex_spread 就沒有 bar_end_ts），直接 SELECT 會
        # SQL 報錯。先問 information_schema：沒有 bar_end_ts 時回傳的 DataFrame 不帶這欄——下游
        # （training/time_semantics.py）把「沒有這欄」視為資訊可用時間未知（None），不猜、不補。
        # 表名已過 quotes_table 白名單，schema／表名拆開後用參數綁定查詢。
        schema, _, name = table.partition(".")
        present = {r["column_name"] for r in await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = $2 AND column_name = ANY($3::text[])",
            schema, name, list(OPTIONAL_COLUMNS),
        )}
        has_session = "session" in present
        has_start = "bar_start_ts" in present
        where = "datetime >= $1 AND datetime < $2"
        kept = where + (" AND session IS DISTINCT FROM 'other'" if has_session else "")
        cols = ["datetime", "open", "high", "low", "close", "volume"] + [
            c for c in ("bar_end_ts", "bar_start_ts") if c in present]
        order = "bar_start_ts ASC, built_at ASC" if has_start else "datetime ASC, built_at ASC"
        rows = await conn.fetch(f"SELECT {', '.join(cols)} FROM {table} WHERE {kept} ORDER BY {order}", start, end)

        report = {"table": table, "start": str(start), "end": str(end), "n_rows": len(rows),
                  "order_key": "bar_start_ts" if has_start else "datetime",
                  "excluded_sessions": list(EXCLUDED_SESSIONS) if has_session else [],
                  "n_excluded": 0, "excluded_trading_days": []}
        if has_session:
            day_col = "trading_day" if "trading_day" in present else "datetime::date"
            excluded = await conn.fetch(
                f"SELECT {day_col} AS d, count(*) AS n FROM {table} WHERE {where} AND session = 'other' "
                "GROUP BY 1 ORDER BY 1", start, end)
            report["n_excluded"] = int(sum(r["n"] for r in excluded))
            report["excluded_trading_days"] = [str(r["d"]) for r in excluded]
            report["sessions"] = {r["s"]: int(r["n"]) for r in await conn.fetch(
                f"SELECT session AS s, count(*) AS n FROM {table} WHERE {kept} GROUP BY 1 ORDER BY 1", start, end)}
        if "rule_version" in present:
            report["rule_versions"] = {r["v"]: int(r["n"]) for r in await conn.fetch(
                f"SELECT rule_version AS v, count(*) AS n FROM {table} WHERE {kept} GROUP BY 1 ORDER BY 1", start, end)}
        minutes = _INTRADAY_MINUTES.get(timeframe)
        if minutes and has_start and has_session and "trading_day" in present:
            # 缺格：同一交易日、同一時段內，相鄰兩根 bar_start_ts 的間隔超過一個週期。只統計、不補棒。
            gap = await conn.fetchrow(
                f"""SELECT count(*) FILTER (WHERE g > $3) AS n_gaps,
                           COALESCE(sum(EXTRACT(EPOCH FROM g) / ($4 * 60) - 1) FILTER (WHERE g > $3), 0) AS n_missing
                    FROM (SELECT bar_start_ts - lag(bar_start_ts) OVER (PARTITION BY trading_day, session
                                                                         ORDER BY bar_start_ts) AS g
                          FROM {table} WHERE {kept}) s""",
                start, end, pd.Timedelta(minutes=minutes).to_pytimedelta(), minutes)
            report["n_gaps"] = int(gap["n_gaps"])
            report["n_missing_bars"] = int(round(float(gap["n_missing"])))
    finally:
        await conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="這段期間查無資料")
    # 兩個時間欄位語意不同，不能混用：
    #   datetime    K 棒的「識別時間」——目前 graph 的索引。舊版資料（2025 年以前）是桶內第一筆成交的秒數，
    #               不一定等於桶起點；同一週期內順序與 bar_start_ts 相同（已核對 datetime 都落在自己的桶內）。
    #   bar_end_ts  這根棒的「資訊可用時間」（桶起點 + 週期）——完整的 OHLCV 最早只能在這個時間之後才用來決策。
    #   bar_start_ts 桶起點；跨時間框架對齊要用它（M4），目前只用來排序與驗證唯一。
    # 這些欄位只是附加給呼叫端標時間用，特徵計算只取 open/high/low/close/volume，不受影響。
    df = pd.DataFrame(rows, columns=cols)
    problems = duplicate_details(df["datetime"], "datetime")
    if has_start:
        problems = duplicate_details(df["bar_start_ts"], "bar_start_ts") + problems
    if problems:
        raise QuoteIntegrityError(f"{table} 在 {start}～{end} 有重複 K 棒（已排除 session='other'）：" + "；".join(problems))
    report["first_datetime"] = str(df["datetime"].iloc[0])
    report["last_datetime"] = str(df["datetime"].iloc[-1])
    df.attrs["read_report"] = report
    return df
