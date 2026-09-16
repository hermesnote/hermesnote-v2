"""量化回測 pipeline：DB -> TA-Lib 指標 -> 策略層(通用訊號模板 / 條件樹) -> vectorbt 模擬。

策略層邏輯在 indicators/signals.py（單一指標的三套觸發模板）跟 indicators/tree.py
（多空各自的進場/出場條件樹，可巢狀 AND/OR/加權），這支只負責：撈資料整理成 DataFrame
-> 呼叫策略層 -> 跑 vectorbt（支援多空雙向）-> 整理成前端要的格式。
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

from indicators.signals import (
    compute_crossover_signals,
    compute_oscillator_signals,
    compute_pattern_signals,
    is_crossover,
    is_oscillator,
    is_pattern,
)
from indicators.tree import collect_indicator_keys, evaluate_node


def prepare_ohlcv(rows: list[dict]) -> pd.DataFrame:
    """rows: [{datetime, open, high, low, close, volume}, ...] -> 整理好、去重、排序的 DataFrame。"""
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    # 原始資料在盤別交界處偶爾會有重複的 datetime（例如日盤/夜盤交接），
    # lightweight-charts 要求時間嚴格遞增，這裡保留每個時間點最後一筆
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


_RULE_LABELS = {
    "long_entry": "多方進場",
    "long_exit": "多方出場",
    "short_entry": "空方進場",
    "short_exit": "空方出場",
}

# 資料庫/前端使用的 timeframe 字串 -> vectorbt/pandas 看得懂的 freq alias。
# 年化報酬、Sharpe 這類跟頻率有關的統計量都靠這個算對，之前這裡是寫死 "15min"，
# 任何非 15 分鐘的回測算出來的這些指標其實都是錯的。
_TIMEFRAME_TO_FREQ = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min",
    "1d": "1D",
}


def _resolve_freq(timeframe: str) -> str:
    freq = _TIMEFRAME_TO_FREQ.get(timeframe)
    if freq is None:
        raise ValueError(f"未知的 timeframe：{timeframe!r}，目前支援：{sorted(_TIMEFRAME_TO_FREQ)}")
    return freq


def run_backtest(
    df: pd.DataFrame,
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series | None,
    short_exits: pd.Series | None,
    reasons: dict[str, str],
    indicators_meta: list[dict],
    timeframe: str,
    fees: float = 0.0,
) -> dict:
    """跑 vectorbt（entries/exits 做多，short_entries/short_exits 做空，兩者都給就是多空雙向）
    並整理成前端要的格式。reasons 的 key 是 long_entry/long_exit/short_entry/short_exit，
    每一筆交易顯示的是「這個方向當時設定的規則敘述」，不是逐筆判斷是哪個子條件觸發的。
    fees 是 vectorbt 的比例手續費（0.001 = 0.1%），預設 0 維持現有零成本模擬行為，
    之後要接滑價可以另外用 slippage 參數，這裡先只把 fees 從寫死改成可傳入。"""
    pf = vbt.Portfolio.from_signals(
        df["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=1_000_000, fees=fees, freq=_resolve_freq(timeframe),
    )

    candles = [
        {
            "time": int(ts.timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for ts, row in df.iterrows()
    ]

    trades_df = pf.trades.records_readable
    trades = []
    for _, t in trades_df.iterrows():
        direction = str(t["Direction"])  # "Long" or "Short"
        entry_reason = reasons.get("long_entry" if direction == "Long" else "short_entry", "")
        exit_reason = reasons.get("long_exit" if direction == "Long" else "short_exit", "")

        entry_ts = pd.Timestamp(t["Entry Timestamp"])
        exit_ts = pd.Timestamp(t["Exit Timestamp"]) if pd.notna(t["Exit Timestamp"]) else None
        entry_price = float(t["Avg Entry Price"])
        exit_price = float(t["Avg Exit Price"]) if pd.notna(t["Avg Exit Price"]) else None
        trades.append({
            "direction": direction,
            "entry_time": int(entry_ts.timestamp()),
            "entry_price": entry_price,
            "entry_reason": entry_reason,
            "exit_time": int(exit_ts.timestamp()) if exit_ts is not None else None,
            "exit_price": exit_price,
            "exit_reason": exit_reason if exit_price is not None else None,
            "points": (exit_price - entry_price) if exit_price is not None else None,
            "pnl": float(t["PnL"]),
            "return_pct": float(t["Return"]) * 100,
            "status": str(t["Status"]),
        })

    stats = pf.stats()

    def _stat(key):
        v = stats.get(key)
        if pd.notna(v) and np.isfinite(v):
            return float(v)
        return None

    summary = {
        "total_trades": int(stats["Total Trades"]),
        "win_rate_pct": _stat("Win Rate [%]"),
        "total_return_pct": _stat("Total Return [%]"),
        "max_drawdown_pct": _stat("Max Drawdown [%]"),
        "sharpe_ratio": _stat("Sharpe Ratio"),
        "profit_factor": _stat("Profit Factor"),
        "avg_win_pct": _stat("Avg Winning Trade [%]"),
        "avg_loss_pct": _stat("Avg Losing Trade [%]"),
        "best_trade_pct": _stat("Best Trade [%]"),
        "worst_trade_pct": _stat("Worst Trade [%]"),
    }

    rules = [
        {"action": "entry" if k.endswith("entry") else "exit", "text": f"{_RULE_LABELS[k]}：{v}"}
        for k, v in reasons.items()
        if v
    ]

    return {
        "candles": candles,
        "trades": trades,
        "summary": summary,
        "indicators": indicators_meta,
        "rules": rules,
    }


def compute_indicator_backtest(
    rows: list[dict],
    timeframe: str,
    indicator: str = "RSI",
    timeperiod: int = 14,
    oversold: float | None = None,
    overbought: float | None = None,
    fees: float = 0.0,
) -> dict:
    """單一指標、純多方的簡易回測（舊的 /api/backtest/demo 用）。"""
    df = prepare_ohlcv(rows)

    if is_pattern(indicator):
        entries, exits, entry_reason, exit_reason = compute_pattern_signals(df, indicator)
        indicator_params: dict = {}
    elif is_oscillator(indicator):
        entries, exits, entry_reason, exit_reason = compute_oscillator_signals(
            df, indicator, timeperiod, oversold, overbought
        )
        indicator_params = {"timeperiod": timeperiod, "oversold": oversold, "overbought": overbought}
    elif is_crossover(indicator):
        entries, exits, entry_reason, exit_reason = compute_crossover_signals(df, indicator)
        indicator_params = {}
    else:
        raise ValueError(f"{indicator} 尚未接上訊號規則")

    indicators_meta = [{"key": indicator, "label": indicator, "params": indicator_params}]
    reasons = {"long_entry": entry_reason, "long_exit": exit_reason}
    return run_backtest(df, entries, exits, None, None, reasons, indicators_meta, timeframe, fees)


def compute_strategy_backtest(
    rows: list[dict],
    timeframe: str,
    long_entry_tree: dict,
    long_exit_tree: dict,
    short_entry_tree: dict | None,
    short_exit_tree: dict | None,
    fees: float = 0.0,
) -> dict:
    """多空各自一棵進場/出場條件樹（見 indicators/tree.py）。空方兩棵樹沒給就是純多方。"""
    df = prepare_ohlcv(rows)

    long_entries, long_entry_reason = evaluate_node(df, long_entry_tree)
    long_exits, long_exit_reason = evaluate_node(df, long_exit_tree)

    short_entries = short_exits = None
    short_entry_reason = short_exit_reason = ""
    trees = [long_entry_tree, long_exit_tree]
    if short_entry_tree and short_exit_tree:
        short_entries, short_entry_reason = evaluate_node(df, short_entry_tree)
        short_exits, short_exit_reason = evaluate_node(df, short_exit_tree)
        trees += [short_entry_tree, short_exit_tree]

    keys: set[str] = set()
    for t in trees:
        keys |= set(collect_indicator_keys(t))
    indicators_meta = [{"key": k, "label": k, "params": {}} for k in sorted(keys)]

    reasons = {
        "long_entry": long_entry_reason,
        "long_exit": long_exit_reason,
        "short_entry": short_entry_reason,
        "short_exit": short_exit_reason,
    }
    return run_backtest(df, long_entries, long_exits, short_entries, short_exits, reasons, indicators_meta, timeframe, fees)
