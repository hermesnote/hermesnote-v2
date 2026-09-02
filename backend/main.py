import asyncio
import os
from datetime import date

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.backtest_demo import compute_rsi_threshold_backtest

load_dotenv()

QUOTES_DATABASE_URL = os.getenv("QUOTES_DATABASE_URL")

app = FastAPI(title="hermesnote-v2 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/backtest/demo")
async def backtest_demo(
    symbol: str = "TX",
    timeframe: str = "15m",
    start: date = date(2026, 6, 1),
    end: date = date(2026, 9, 1),
):
    table = f"market.future_taifex_{symbol.lower()}_{timeframe}"

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

    row_dicts = [dict(r) for r in rows]

    # TA-Lib / vectorbt 是同步、CPU-bound 的計算，丟到執行緒池跑，不卡住 event loop
    result = await asyncio.to_thread(compute_rsi_threshold_backtest, row_dicts)
    result["meta"] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "bar_count": len(row_dicts),
    }
    return result
