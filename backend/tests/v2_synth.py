"""v2 測試用的合成行情：日盤 08:45～13:45（每天 300 根 1m）＋可選夜盤 15:00～05:00，週末休市。
欄位與正式 quotes 表相同：datetime、bar_start_ts、bar_end_ts、trading_day、session、OHLCV。較粗時間框架由 1m
在同一時段內彙總（K 棒 bar_end_ts＝bar_start_ts＋週期，時段尾端不足一根的取時段結束）。"""

import numpy as np
import pandas as pd


def _session_minutes(day: pd.Timestamp, night: bool):
    out = []
    s = day + pd.Timedelta(hours=8, minutes=45)
    out += [(s + pd.Timedelta(minutes=i), "day", day) for i in range(300)]
    if night:
        s = day + pd.Timedelta(hours=15)
        nxt = day + pd.offsets.BDay(1)
        out += [(s + pd.Timedelta(minutes=i), "night", nxt) for i in range(14 * 60)]
    return out


def make_1m(days: int = 12, start: str = "2024-01-01", seed: int = 0, night: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start, periods=days)
    rows = []
    for d in bdays:
        rows += _session_minutes(d, night)
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    open_ = close + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + rng.random(n) * 0.1
    low = np.minimum(open_, close) - rng.random(n) * 0.1
    vol = rng.integers(1, 100, n).astype(float)
    start_ts = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame({
        "datetime": start_ts, "bar_start_ts": start_ts, "bar_end_ts": start_ts + pd.Timedelta(minutes=1),
        "trading_day": pd.DatetimeIndex([r[2] for r in rows]).date, "session": [r[1] for r in rows],
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
    })


def resample(m1: pd.DataFrame, minutes: int | None) -> pd.DataFrame:
    """minutes=None → 日線（daily_day：只用日盤）。"""
    df = m1.copy()
    if minutes is None:
        df = df[df["session"] == "day"]
        key = pd.to_datetime(df["trading_day"])
        g = df.groupby(key)
        out = pd.DataFrame({
            "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
            "volume": g["volume"].sum()})
        out["bar_start_ts"] = out.index + pd.Timedelta(hours=8, minutes=45)
        out["bar_end_ts"] = out.index + pd.Timedelta(hours=13, minutes=45)
        out["trading_day"] = out.index.date
        out["session"] = "day"
        out["datetime"] = out["bar_start_ts"]
        return out.reset_index(drop=True)
    sess_start = df.groupby(["trading_day", "session"])["bar_start_ts"].transform("min")
    sess_end = df.groupby(["trading_day", "session"])["bar_end_ts"].transform("max")
    bucket = sess_start + ((df["bar_start_ts"] - sess_start) // pd.Timedelta(minutes=minutes)) * pd.Timedelta(minutes=minutes)
    df["_b"] = bucket
    df["_se"] = sess_end
    g = df.groupby("_b", sort=True)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(), "trading_day": g["trading_day"].first(), "session": g["session"].first(),
        "_se": g["_se"].first()})
    out["bar_start_ts"] = out.index
    out["bar_end_ts"] = np.minimum(out.index + pd.Timedelta(minutes=minutes), out["_se"])
    out["datetime"] = out["bar_start_ts"]
    return out.drop(columns="_se").reset_index(drop=True)


def market(days: int = 12, seed: int = 0, night: bool = False) -> dict:
    m1 = make_1m(days, seed=seed, night=night)
    return {"tx_1m": m1, "tx_5m": resample(m1, 5), "tx_15m": resample(m1, 15), "tx_60m": resample(m1, 60),
            "tx_daily_day": resample(m1, None)}
