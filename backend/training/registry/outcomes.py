"""Outcome Node 登記表：輸入 OHLCV 資料 + horizon，輸出「原始數值」，跟要不要分類無關。

這是從舊版 `labels.py` 的 `nbar_return` 拆出來的——舊版把「算報酬率」跟「切成幾類」焊死
在同一個 function 裡，導致沒辦法只算報酬率不切類別（回歸模式），也沒辦法換算「未來價格」。
拆開後，outcome 只負責算出一個連續數值，要不要切類別、切成幾類是 labeling_rule 的事
（見 `labeling_rules.py`），兩者可以自由組合。

每個函式回傳 (raw, valid_length)：
  raw           — 長度跟輸入資料筆數對齊的原始數值陣列，最後 horizon 筆因為看不到未來是 NaN
  valid_length  — 有效可訓練的樣本數（= 總筆數 - horizon）
"""

import numpy as np

OUTCOME_REGISTRY: dict = {}


def register(key: str):
    def deco(fn):
        OUTCOME_REGISTRY[key] = fn
        return fn
    return deco


def _future_close(df, horizon: int):
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    future = np.full(n, np.nan)
    future[: n - horizon] = close[horizon:]
    return close, future, n


@register("simple_return")
def simple_return(df, params: dict) -> tuple[np.ndarray, int]:
    """T+horizon 收盤價相對 T 的簡單報酬率（百分比）。params: horizon（預設 1）。"""
    horizon = int(params.get("horizon", 1))
    close, future, n = _future_close(df, horizon)
    raw = (future - close) / close * 100.0
    return raw, n - horizon


@register("log_return")
def log_return(df, params: dict) -> tuple[np.ndarray, int]:
    """T+horizon 收盤價相對 T 的對數報酬率。params: horizon（預設 1）。"""
    horizon = int(params.get("horizon", 1))
    close, future, n = _future_close(df, horizon)
    raw = np.full(n, np.nan)
    valid = ~np.isnan(future)
    raw[valid] = np.log(future[valid] / close[valid])
    return raw, n - horizon


@register("future_price")
def future_price(df, params: dict) -> tuple[np.ndarray, int]:
    """T+horizon 的實際收盤價（絕對值，不是相對報酬率）。params: horizon（預設 1）。
    注意：這是價格的絕對值域，不能直接套百分比門檻分類（labeling_rule 那邊會擋掉這個組合）。
    """
    horizon = int(params.get("horizon", 1))
    _, future, n = _future_close(df, horizon)
    return future, n - horizon


def list_available() -> list[dict]:
    """給 GET /api/model/registry/outcomes 用，Swagger 自動文件化，Agent/MCP 查詢用。"""
    return [{"key": k, "description": (fn.__doc__ or "").strip()} for k, fn in OUTCOME_REGISTRY.items()]


def compute_outcome(key: str, df, params: dict | None = None) -> tuple[np.ndarray, int]:
    if key not in OUTCOME_REGISTRY:
        raise ValueError(f"未登記的 outcome key: {key!r}，目前有：{sorted(OUTCOME_REGISTRY)}")
    return OUTCOME_REGISTRY[key](df, params or {})
