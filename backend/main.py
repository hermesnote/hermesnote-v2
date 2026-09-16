import asyncio
from datetime import date

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from indicators.signals import OSCILLATOR_DEFAULTS, is_crossover, is_oscillator, is_pattern
from routers.auth import router as auth_router
from routers.indicators import router as indicators_router
from routers.model_training import router as model_training_router
from routers.strategy import router as strategy_router
from services.backtest_demo import compute_indicator_backtest
from services.quotes import QUOTES_DATABASE_URL, fetch_ohlcv_rows, quotes_table

load_dotenv()

app = FastAPI(title="hermesnote-v2 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)
app.include_router(indicators_router)
app.include_router(strategy_router)
app.include_router(model_training_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/backtest/date-range")
async def backtest_date_range(symbol: str = "TX", timeframe: str = "15m"):
    table = quotes_table(symbol, timeframe)
    conn = await asyncpg.connect(QUOTES_DATABASE_URL)
    try:
        row = await conn.fetchrow(f"SELECT MIN(datetime) AS min_dt, MAX(datetime) AS max_dt FROM {table}")
    finally:
        await conn.close()

    if row is None or row["min_dt"] is None:
        raise HTTPException(status_code=404, detail="查無資料")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "min_date": row["min_dt"].date().isoformat(),
        "max_date": row["max_dt"].date().isoformat(),
    }


@app.get("/api/backtest/demo")
async def backtest_demo(
    symbol: str = "TX",
    timeframe: str = "15m",
    start: date = date(2026, 6, 1),
    end: date = date(2026, 9, 1),
    indicator: str = "RSI",
    timeperiod: int = 14,
    oversold: float | None = None,
    overbought: float | None = None,
):
    if not is_pattern(indicator) and not is_oscillator(indicator) and not is_crossover(indicator):
        raise HTTPException(status_code=400, detail=f"{indicator} 尚未接上訊號規則")

    if is_oscillator(indicator):
        defaults = OSCILLATOR_DEFAULTS[indicator]
        if oversold is None:
            oversold = defaults["oversold"]
        if overbought is None:
            overbought = defaults["overbought"]

    row_dicts = await fetch_ohlcv_rows(symbol, timeframe, start, end)

    # TA-Lib / vectorbt 是同步、CPU-bound 的計算，丟到執行緒池跑，不卡住 event loop
    result = await asyncio.to_thread(
        compute_indicator_backtest, row_dicts, timeframe, indicator, timeperiod, oversold, overbought
    )
    result["meta"] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "bar_count": len(row_dicts),
    }
    return result
