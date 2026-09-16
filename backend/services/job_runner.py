"""量化回測背景執行：送出後立刻回 job_id，實際運算在背景跑完再存進 DB。

單機、單一 uvicorn worker 的簡易做法（`asyncio.create_task`，不是正式的任務佇列）——
對個人專案來說夠用，重開伺服器會遺失還在跑的任務，之後真的需要跨重啟存活再換成
真正的 job queue（Celery/RQ 之類）。
"""

import asyncio
from datetime import date

from services import job_store
from services.backtest_demo import compute_strategy_backtest
from services.quotes import fetch_ohlcv_rows

_background_tasks: set[asyncio.Task] = set()


async def _run(
    job_id: str,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    long_entry_tree: dict,
    long_exit_tree: dict,
    short_entry_tree: dict | None,
    short_exit_tree: dict | None,
) -> None:
    try:
        await job_store.mark_running(job_id)
        rows = await fetch_ohlcv_rows(symbol, timeframe, start, end)
        result = await asyncio.to_thread(
            compute_strategy_backtest, rows, timeframe, long_entry_tree, long_exit_tree, short_entry_tree, short_exit_tree
        )
        await job_store.save_result(
            job_id,
            result["candles"],
            result["trades"],
            result["summary"],
            result["indicators"],
            result["rules"],
            len(rows),
        )
    except Exception as e:  # noqa: BLE001 — 背景任務要把任何失敗記進 DB，不能讓例外憑空消失
        await job_store.mark_failed(job_id, str(e))


def start(
    job_id: str,
    symbol: str,
    timeframe: str,
    start_date: date,
    end_date: date,
    long_entry_tree: dict,
    long_exit_tree: dict,
    short_entry_tree: dict | None,
    short_exit_tree: dict | None,
) -> None:
    task = asyncio.create_task(
        _run(job_id, symbol, timeframe, start_date, end_date, long_entry_tree, long_exit_tree, short_entry_tree, short_exit_tree)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
