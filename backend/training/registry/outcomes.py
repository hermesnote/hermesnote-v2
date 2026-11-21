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
# 宣告式 metadata（不參與計算）：unit 是 raw 數值的單位、signed 是「帶正負號的報酬」
# （正負有意義，才適合拿 >=／< 門檻定義方向事件）。具名輸出的 metadata（training/output_specs.py）
# 從這裡取單位——simple_return 是百分比、log_return 是自然對數比值，同一個 0.0 門檻在兩者
# 上意義不同，單位必須明確記錄，不能只靠使用者記得。
OUTCOME_META: dict = {}


def register(key: str, unit: str | None = None, signed: bool = False, *, gap_check: bool = False,
             v2_only: bool = False, classes=None):
    """gap_check：v2 圖預設對 T～T+H 做缺格檢查（見 engine_v2.Ctx.labels，Label params.gaps 可覆寫）；
    v2_only：只能用在 mode=v2 的圖（缺格檢查需要 v2 的 K 棒時間）；
    classes：`(params) -> [類別名稱…]`，類別型目標宣告類別順序（索引＝位置），寫進輸出說明。"""
    def deco(fn):
        OUTCOME_REGISTRY[key] = fn
        OUTCOME_META[key] = {"unit": unit, "signed": signed, "gap_check": gap_check, "v2_only": v2_only}
        if classes is not None:
            OUTCOME_CLASSES[key] = classes
        return fn
    return deco


OUTCOME_CLASSES: dict = {}
# outcome key → (df, params) -> {原因代碼: 布林遮罩}：該 outcome 自己的起點條件（engine_v2.label_exclusions 套用）
OUTCOME_EXCLUSIONS: dict = {}


def _future_close(df, horizon: int):
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    future = np.full(n, np.nan)
    future[: n - horizon] = close[horizon:]
    return close, future, n


@register("simple_return", unit="percent", signed=True)
def simple_return(df, params: dict) -> tuple[np.ndarray, int]:
    """T+horizon 收盤價相對 T 的簡單報酬率（百分比）。params: horizon（預設 1）。"""
    horizon = int(params.get("horizon", 1))
    close, future, n = _future_close(df, horizon)
    raw = (future - close) / close * 100.0
    return raw, n - horizon


@register("log_return", unit="log_ratio", signed=True)
def log_return(df, params: dict) -> tuple[np.ndarray, int]:
    """T+horizon 收盤價相對 T 的對數報酬率。params: horizon（預設 1）。"""
    horizon = int(params.get("horizon", 1))
    close, future, n = _future_close(df, horizon)
    raw = np.full(n, np.nan)
    valid = ~np.isnan(future)
    raw[valid] = np.log(future[valid] / close[valid])
    return raw, n - horizon


@register("future_price", unit="price", signed=False)
def future_price(df, params: dict) -> tuple[np.ndarray, int]:
    """T+horizon 的實際收盤價（絕對值，不是相對報酬率）。params: horizon（預設 1）。
    注意：這是價格的絕對值域，不能直接套百分比門檻分類（labeling_rule 那邊會擋掉這個組合）。
    """
    horizon = int(params.get("horizon", 1))
    _, future, n = _future_close(df, horizon)
    return future, n - horizon


# ── 2026-11-21：首次越界、區間最高／最低 ─────────────────────────────────────
def _horizon(params: dict) -> int:
    return int(params.get("horizon", 1))


def _future_block(x: np.ndarray, horizon: int) -> np.ndarray:
    """(n, H)：第 t 列是 x[t+1..t+H]；超出尾端為 NaN。"""
    n = len(x)
    out = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        out[: max(n - k, 0), k - 1] = x[k:]
    return out


def first_crossing_classes(params: dict) -> list[str]:
    h = _horizon(params)
    return [f"up_{k}" for k in range(1, h + 1)] + [f"down_{k}" for k in range(1, h + 1)] + ["none"]


@register("first_crossing", unit="class", gap_check=True, v2_only=True, classes=first_crossing_classes)
def first_crossing(df, params: dict) -> tuple[np.ndarray, int]:
    """首次越界（類別）：以 T 收完時已知的上界／下界（params.bounds，見 training/registry/bounds.py），
    看 T+1～T+H 每根收盤第一次穿越哪一側、在第幾根。收盤 > 上界＝上穿、< 下界＝下穿，恰好等於仍算區間內。
    類別索引：0～H-1＝上穿第 1～H 根、H～2H-1＝下穿第 1～H 根、2H＝期限內未越界（共 2H+1 類，
    Label 的 labeling_rule 用 class_index、n_classes＝2H+1）。T 的上下界是 NaN（暖機）或期末不足 H 根時沒有答案。
    params: horizon、bounds。"""
    from training.registry import bounds as bd

    h = _horizon(params)
    upper, lower = bd.compute(df, params["bounds"])
    close = df["close"].to_numpy(dtype=float)
    fut = _future_block(close, h)
    up = fut > upper[:, None]
    down = fut < lower[:, None]
    n = len(close)
    raw = np.full(n, float(2 * h))
    decided = np.zeros(n, dtype=bool)
    for k in range(h):
        hit_up = up[:, k] & ~decided
        hit_down = down[:, k] & ~decided
        raw[hit_up] = k
        raw[hit_down] = h + k
        decided |= hit_up | hit_down
    raw[np.isnan(upper) | np.isnan(lower)] = np.nan
    raw[max(n - h, 0):] = np.nan
    return raw, n - h


def _start_outside(df, params: dict) -> dict:
    """first_crossing 的起點條件：T 收盤必須在上下界之內或恰在線上，否則排除（label_start_outside）。"""
    from training.registry import bounds as bd

    upper, lower = bd.compute(df, params["bounds"])
    close = df["close"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return {"label_start_outside": (close > upper) | (close < lower)}


OUTCOME_EXCLUSIONS["first_crossing"] = _start_outside


@register("future_high_offset", unit="percent", signed=True, gap_check=True, v2_only=True)
def future_high_offset(df, params: dict) -> tuple[np.ndarray, int]:
    """區間最高（回歸，百分比）：100 × (max(High[T+1..T+H]) / Close[T] − 1)，不含 T、保留正負號。params: horizon。"""
    h = _horizon(params)
    close = df["close"].to_numpy(dtype=float)
    raw = 100.0 * (_future_block(df["high"].to_numpy(dtype=float), h).max(axis=1) / close - 1.0)
    raw[max(len(close) - h, 0):] = np.nan
    return raw, len(close) - h


@register("future_low_offset", unit="percent", signed=True, gap_check=True, v2_only=True)
def future_low_offset(df, params: dict) -> tuple[np.ndarray, int]:
    """區間最低（回歸，百分比）：100 × (min(Low[T+1..T+H]) / Close[T] − 1)，不含 T、保留正負號。params: horizon。"""
    h = _horizon(params)
    close = df["close"].to_numpy(dtype=float)
    raw = 100.0 * (_future_block(df["low"].to_numpy(dtype=float), h).min(axis=1) / close - 1.0)
    raw[max(len(close) - h, 0):] = np.nan
    return raw, len(close) - h


def list_available() -> list[dict]:
    """給 GET /api/model/registry/outcomes 用，Swagger 自動文件化，Agent/MCP 查詢用。"""
    return [
        {"key": k, "description": (fn.__doc__ or "").strip(), **OUTCOME_META.get(k, {})}
        for k, fn in OUTCOME_REGISTRY.items()
    ]


def compute_outcome(key: str, df, params: dict | None = None) -> tuple[np.ndarray, int]:
    if key not in OUTCOME_REGISTRY:
        raise ValueError(f"未登記的 outcome key: {key!r}，目前有：{sorted(OUTCOME_REGISTRY)}")
    return OUTCOME_REGISTRY[key](df, params or {})
