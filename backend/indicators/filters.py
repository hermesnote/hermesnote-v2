"""B 組「輔助型」條件：任何一條 TA-Lib 輸出數值，都能問「現在是不是大於/小於某個門檻」。

跟 signals.py 的差別：這裡不判斷「進出場的那一刻」，只回答「現在符不符合這個狀態」，
用來當組合引擎（combo.py）裡的閘門條件，不會單獨拿來決定進出場時機。
"""

import operator as op

import pandas as pd
from talib import abstract

_OPS = {
    ">": op.gt,
    ">=": op.ge,
    "<": op.lt,
    "<=": op.le,
}


def _ohlcv_inputs(df: pd.DataFrame) -> dict:
    return {
        "open": df["open"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "close": df["close"].values,
        "volume": df["volume"].values,
    }


def compute_filter_series(df: pd.DataFrame, key: str, output_name: str | None, params: dict) -> pd.Series:
    """算出指定指標（可指定要用哪一條輸出線）的數值序列。"""
    fn = abstract.Function(key)
    kwargs = {k: v for k, v in (params or {}).items() if k in fn.parameters}
    out = fn(_ohlcv_inputs(df), **kwargs)

    if isinstance(out, list):
        out_dict = dict(zip(fn.output_names, out))
        chosen = output_name or fn.output_names[0]
        if chosen not in out_dict:
            raise ValueError(f"{key} 沒有輸出線 {chosen}，可用的是 {fn.output_names}")
        values = out_dict[chosen]
    else:
        values = out

    return pd.Series(values, index=df.index)


def compute_filter_condition(
    df: pd.DataFrame,
    key: str,
    output_name: str | None,
    params: dict,
    operator_symbol: str,
    threshold: float,
) -> pd.Series:
    """回傳 True/False 序列：這個指標的數值 [operator] threshold 是否成立。"""
    if operator_symbol not in _OPS:
        raise ValueError(f"不支援的比較符號：{operator_symbol}")
    series = compute_filter_series(df, key, output_name, params)
    return _OPS[operator_symbol](series, threshold).fillna(False)
