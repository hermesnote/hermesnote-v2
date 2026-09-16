"""通用訊號模板：把 TA-Lib 指標值轉成進出場訊號，不用每個指標各自手刻一套邏輯。

三套模板（對應 A 組「觸發型」指標）：
- Pattern Recognition（61 個）：TA-Lib 輸出已經標準化成「看漲(>0)/看跌(<0)/無(=0)」，
  直接用正負號當進出場訊號。
- 有明確上下界的震盪指標：共用「跌破下界進場、突破上界出場」的門檻穿越模板，
  只是每個指標的常見門檻不同，需要人工查證後填進 OSCILLATOR_DEFAULTS。
- 雙線交叉型（MACD 這類）：快線穿越慢線，共用「快線上穿進場、下穿出場」的模板。

其餘 TA-Lib 指標（B 組「輔助型」：疊加均線、量能、波動度、週期、統計、數學函數）沒有自己的
進出場時機，不接這裡的觸發模板——它們的用法是當篩選條件（見 indicators/filters.py），
在組合引擎（indicators/combo.py）裡當閘門，不是這支要處理的事。
"""

import pandas as pd
from talib import abstract

PATTERN_PREFIX = "CDL"

# key -> 常見超賣/超買門檻（人工查證，不是機械生成）
OSCILLATOR_DEFAULTS = {
    "RSI": {"oversold": 30, "overbought": 70},
    "CCI": {"oversold": -100, "overbought": 100},
    "WILLR": {"oversold": -80, "overbought": -20},
    "MFI": {"oversold": 20, "overbought": 80},
    "ULTOSC": {"oversold": 30, "overbought": 70},
    "CMO": {"oversold": -50, "overbought": 50},
    "STOCH": {"oversold": 20, "overbought": 80},
}

# key -> 哪兩條輸出線做交叉比較（人工查證，不是機械生成）
CROSSOVER_DEFAULTS = {
    "MACD": {"fast_output": "macd", "slow_output": "macdsignal"},
}


def is_pattern(key: str) -> bool:
    return key.startswith(PATTERN_PREFIX)


def is_oscillator(key: str) -> bool:
    return key in OSCILLATOR_DEFAULTS


def is_crossover(key: str) -> bool:
    return key in CROSSOVER_DEFAULTS


def signal_supported(key: str) -> bool:
    """A 組：有沒有自己的進出場觸發模板。"""
    return is_pattern(key) or is_oscillator(key) or is_crossover(key)


def _ohlcv_inputs(df: pd.DataFrame) -> dict:
    return {
        "open": df["open"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "close": df["close"].values,
        "volume": df["volume"].values,
    }


def compute_pattern_signals(df: pd.DataFrame, key: str):
    """回傳 (entries, exits, entry_reason, exit_reason)。看漲型態進場、看跌型態出場。"""
    fn = abstract.Function(key)
    out = pd.Series(fn(_ohlcv_inputs(df)), index=df.index)
    entries = (out > 0).fillna(False)
    exits = (out < 0).fillna(False)
    entry_reason = f"{key} 出現看漲型態"
    exit_reason = f"{key} 出現看跌型態"
    return entries, exits, entry_reason, exit_reason


def compute_oscillator_signals(df: pd.DataFrame, key: str, timeperiod: int, oversold: float, overbought: float):
    """回傳 (entries, exits, entry_reason, exit_reason)。跌破下界進場、突破上界出場。"""
    fn = abstract.Function(key)
    kwargs = {"timeperiod": timeperiod} if "timeperiod" in fn.parameters else {}
    out = fn(_ohlcv_inputs(df), **kwargs)
    if isinstance(out, list):
        out = out[0]  # 例如 STOCH 回傳 (slowk, slowd)，取第一個當主訊號線
    series = pd.Series(out, index=df.index)

    entries = ((series < oversold) & (series.shift(1) >= oversold)).fillna(False)
    exits = ((series > overbought) & (series.shift(1) <= overbought)).fillna(False)
    entry_reason = f"{key} 由上往下跌破 {oversold:g}"
    exit_reason = f"{key} 由下往上突破 {overbought:g}"
    return entries, exits, entry_reason, exit_reason


def compute_crossover_signals(df: pd.DataFrame, key: str, params: dict | None = None):
    """回傳 (entries, exits, entry_reason, exit_reason)。快線上穿慢線進場、下穿出場。"""
    cfg = CROSSOVER_DEFAULTS[key]
    fn = abstract.Function(key)
    kwargs = {k: v for k, v in (params or {}).items() if k in fn.parameters}
    out = fn(_ohlcv_inputs(df), **kwargs)
    out_dict = dict(zip(fn.output_names, out)) if isinstance(out, list) else {fn.output_names[0]: out}

    fast = pd.Series(out_dict[cfg["fast_output"]], index=df.index)
    slow = pd.Series(out_dict[cfg["slow_output"]], index=df.index)

    entries = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).fillna(False)
    exits = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).fillna(False)
    entry_reason = f"{key} {cfg['fast_output']} 上穿 {cfg['slow_output']}"
    exit_reason = f"{key} {cfg['fast_output']} 下穿 {cfg['slow_output']}"
    return entries, exits, entry_reason, exit_reason
