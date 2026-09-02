"""量化回測 demo pipeline：DB -> TA-Lib 指標 -> 策略層(自訂規則) -> vectorbt 模擬。

這支只是驗證整條鏈路能不能通，策略邏輯（RSI 門檻）是先求可動、不是最終策略。
"""

import pandas as pd
import talib
import vectorbt as vbt


def compute_rsi_threshold_backtest(rows: list[dict]) -> dict:
    """rows: [{datetime, open, high, low, close, volume}, ...]，依 datetime 升冪排序。"""
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    # 原始資料在盤別交界處偶爾會有重複的 datetime（例如日盤/夜盤交接），
    # lightweight-charts 要求時間嚴格遞增，這裡保留每個時間點最後一筆
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # 1) TA-Lib 算指標
    df["rsi"] = talib.RSI(df["close"].values, timeperiod=14)

    # 2) 策略層（自訂規則，不是 TA-Lib 也不是 vectorbt 的事）
    entries = (df["rsi"] < 30) & (df["rsi"].shift(1) >= 30)
    exits = (df["rsi"] > 70) & (df["rsi"].shift(1) <= 70)
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # 3) vectorbt 模擬
    pf = vbt.Portfolio.from_signals(
        df["close"], entries, exits,
        init_cash=1_000_000, fees=0.0, freq="15min",
    )

    candles = [
        {
            "time": int(ts.timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for ts, row in df.iterrows()
    ]

    trades_df = pf.trades.records_readable
    trades = []
    for _, t in trades_df.iterrows():
        entry_ts = pd.Timestamp(t["Entry Timestamp"])
        exit_ts = pd.Timestamp(t["Exit Timestamp"]) if pd.notna(t["Exit Timestamp"]) else None
        trades.append({
            "direction": str(t["Direction"]),
            "entry_time": int(entry_ts.timestamp()),
            "entry_price": float(t["Avg Entry Price"]),
            "exit_time": int(exit_ts.timestamp()) if exit_ts is not None else None,
            "exit_price": float(t["Avg Exit Price"]) if pd.notna(t["Avg Exit Price"]) else None,
            "pnl": float(t["PnL"]),
            "return_pct": float(t["Return"]) * 100,
            "status": str(t["Status"]),
        })

    stats = pf.stats()
    summary = {
        "total_trades": int(stats["Total Trades"]),
        "win_rate_pct": float(stats["Win Rate [%]"]) if pd.notna(stats["Win Rate [%]"]) else None,
        "total_return_pct": float(stats["Total Return [%]"]),
        "max_drawdown_pct": float(stats["Max Drawdown [%]"]),
        "sharpe_ratio": float(stats["Sharpe Ratio"]) if pd.notna(stats["Sharpe Ratio"]) else None,
    }

    return {"candles": candles, "trades": trades, "summary": summary}
