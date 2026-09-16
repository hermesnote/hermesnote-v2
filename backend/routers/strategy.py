from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import job_runner, job_store

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


class StrategyRequest(BaseModel):
    symbol: str = "TX"
    timeframe: str = "15m"
    start: date
    end: date
    long_entry_tree: dict  # 見 indicators/tree.py 的節點格式
    long_exit_tree: dict
    short_entry_tree: dict | None = None  # 兩棵都沒給 = 純多方
    short_exit_tree: dict | None = None


@router.post("/strategy")
async def backtest_strategy(body: StrategyRequest):
    """立刻回 job_id，實際運算在背景跑（見 services/job_runner.py），不阻塞這個請求。"""
    job_id = await job_store.create_job(
        body.symbol, body.timeframe, body.start, body.end,
        body.long_entry_tree, body.long_exit_tree, body.short_entry_tree, body.short_exit_tree,
    )
    job_runner.start(
        job_id, body.symbol, body.timeframe, body.start, body.end,
        body.long_entry_tree, body.long_exit_tree, body.short_entry_tree, body.short_exit_tree,
    )
    return {"job_id": job_id}


@router.get("/jobs")
async def list_backtest_jobs(limit: int = 20):
    return await job_store.list_jobs(limit)


@router.get("/jobs/{job_id}")
async def get_backtest_job(job_id: str):
    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="找不到這個回測任務")
    return job


@router.delete("/jobs/{job_id}")
async def delete_backtest_job(job_id: str):
    deleted = await job_store.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="找不到這個回測任務")
    return {"deleted": True}


@router.get("/jobs/{job_id}/candles")
async def get_backtest_candles(
    job_id: str, before: int | None = None, around: int | None = None, limit: int = 2000
):
    return await job_store.get_candles(job_id, before, around, limit)


@router.get("/jobs/{job_id}/trades")
async def get_backtest_trades(job_id: str, offset: int = 0, limit: int = 200):
    """給下方交易明細表用，分頁——交易數可能是幾十萬筆，不能一次全部回傳。"""
    return await job_store.get_trades(job_id, offset, limit)


@router.get("/jobs/{job_id}/trades/range")
async def get_backtest_trades_in_range(job_id: str, start: int, end: int):
    """給圖表標記用，只抓某個時間窗內的交易（配合 K 棒分段載入）。"""
    return await job_store.get_trades_in_range(job_id, start, end)


class SaveStrategyRequest(BaseModel):
    name: str | None = None


@router.post("/jobs/{job_id}/save")
async def save_strategy(job_id: str, body: SaveStrategyRequest):
    """收藏這筆回測用的條件樹，之後模型訓練那邊可以當 Feature Node 取用。"""
    try:
        saved_id = await job_store.save_strategy(job_id, body.name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"saved_id": saved_id}


@router.delete("/jobs/{job_id}/save")
async def unsave_strategy_by_job(job_id: str):
    """取消收藏（用 job_id 找對應的收藏紀錄刪掉，前端不用自己記 saved_id）。"""
    saved_id = await job_store.get_saved_strategy_id_for_job(job_id)
    if saved_id is None:
        raise HTTPException(status_code=404, detail="這筆回測沒有被收藏過")
    await job_store.unsave_strategy(saved_id)
    return {"deleted": True}


@router.get("/saved-strategies")
async def list_saved_strategies():
    """給模型訓練那邊的 Feature Node 選單用，列出所有收藏的策略。"""
    return await job_store.list_saved_strategies()
