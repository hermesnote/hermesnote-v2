"""Labeling Rule 登記表：把 Outcome Node 的原始數值轉成模型實際要學習的目標 y。

跟 outcome 分離之後，同一個 outcome（例如 simple_return）可以配不同的 labeling_rule
（分類切門檻、或直接回歸），不用重寫 outcome 的計算邏輯。

每個函式簽名 `(raw, valid_length, params) -> (y, task_type, meta)`：
  y          — 陣列，跟 raw 同長度（有效範圍外維持 NaN，呼叫端用 valid_length 對齊）
  task_type  — "classification" 或 "regression"，決定下游 Model Node 要用哪種 loss/輸出頭
  meta       — 補充資訊（例如分類的 n_classes），目前只給前端顯示用，不影響計算
"""

import numpy as np

LABELING_RULE_REGISTRY: dict = {}

# outcome key -> 這個 outcome 不合法的 labeling_rule 集合，並附上不合法的理由。
# 目前只有一條規則：future_price 是價格的絕對值域，不能直接拿百分比門檻分類。
ILLEGAL_COMBINATIONS: dict[str, dict[str, str]] = {
    "future_price": {
        "fixed_threshold": "future_price 是價格的絕對值域，不能直接套百分比門檻分類，"
        "改用 identity（回歸）或先換 outcome 成 simple_return/log_return",
    },
}


def register(key: str):
    def deco(fn):
        LABELING_RULE_REGISTRY[key] = fn
        return fn
    return deco


def validate_combination(outcome_key: str, labeling_rule_key: str) -> None:
    """檢查 outcome + labeling_rule 是不是合法組合，不合法直接丟例外（後端擋，不是只靠前端擋）。"""
    reason = ILLEGAL_COMBINATIONS.get(outcome_key, {}).get(labeling_rule_key)
    if reason:
        raise ValueError(f"不合法的組合：outcome={outcome_key!r} + labeling_rule={labeling_rule_key!r}：{reason}")


@register("fixed_threshold")
def fixed_threshold(raw: np.ndarray, valid_length: int, params: dict) -> tuple[np.ndarray, str, dict]:
    """固定門檻分類。params: n_classes（2=漲/不漲，3=跌/平/漲，預設 2）、
    threshold_pct（漲跌幅門檻，二分類跟三分類都吃這個參數，預設 0.0）。
    """
    n_classes = int(params.get("n_classes", 2))
    threshold_pct = float(params.get("threshold_pct", 0.0))

    y = np.full(len(raw), np.nan)
    if n_classes == 2:
        y[:valid_length] = (raw[:valid_length] > threshold_pct).astype(float)
    elif n_classes == 3:
        up = raw > threshold_pct
        down = raw < -threshold_pct
        cls = np.where(up, 2, np.where(down, 0, 1)).astype(float)
        y[:valid_length] = cls[:valid_length]
    else:
        raise ValueError(f"fixed_threshold 只支援 n_classes=2 或 3，收到 {n_classes}")
    return y, "classification", {"n_classes": n_classes}


@register("identity")
def identity(raw: np.ndarray, valid_length: int, params: dict) -> tuple[np.ndarray, str, dict]:
    """回歸：不轉換，直接把 outcome 的原始數值當學習目標。"""
    y = raw.copy()
    y[valid_length:] = np.nan
    return y, "regression", {}


def list_available() -> list[dict]:
    """給 GET /api/model/registry/labeling_rules 用，Swagger 自動文件化，Agent/MCP 查詢用。"""
    return [{"key": k, "description": (fn.__doc__ or "").strip()} for k, fn in LABELING_RULE_REGISTRY.items()]


def apply_labeling_rule(key: str, raw: np.ndarray, valid_length: int, params: dict | None = None):
    if key not in LABELING_RULE_REGISTRY:
        raise ValueError(f"未登記的 labeling_rule key: {key!r}，目前有：{sorted(LABELING_RULE_REGISTRY)}")
    return LABELING_RULE_REGISTRY[key](raw, valid_length, params or {})
