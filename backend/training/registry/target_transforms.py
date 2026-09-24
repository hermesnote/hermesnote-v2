"""Target Transform 登記表：由 Label 的連續目標「推導」出另一個學習目標（例如方向標籤）。

跟 outcomes.py（算原始數值）、labeling_rules.py（把原始數值轉成主要學習目標）一樣是可插拔積木：
Outcome／horizon 留在 Label Node，「怎麼從連續目標推導方向」由這裡的明確設定決定，不藏在
architecture 的程式碼裡。推導永遠在**目標的原始單位**上做（例如 log-return 的 0.0 門檻），
不受 architecture 內部目標縮放影響——縮放只改訓練回歸頭用的尺度，方向標籤與門檻的尺度固定是原單位。

每個函式簽名 `(y, rule) -> np.ndarray`：y 是原始單位的連續目標，rule 是已驗證的事件規則
（`op`、`threshold`，見 training/output_specs.py 的 build_event_rule），回傳 0.0／1.0 的 float 陣列。
"""

import numpy as np

TARGET_TRANSFORM_REGISTRY: dict = {}

_OPS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
}


def register(key: str):
    def deco(fn):
        TARGET_TRANSFORM_REGISTRY[key] = fn
        return fn
    return deco


@register("sign_threshold")
def sign_threshold(y: np.ndarray, rule: dict) -> np.ndarray:
    """事件標籤：目標 `op` 門檻 為 1.0，否則 0.0。例：{"op": ">=", "threshold": 0.0} → 報酬 >= 0 為 1。
    門檻的單位就是目標的單位（不換算，見 build_event_rule）。"""
    y = np.asarray(y, dtype=float)
    return _OPS[rule["op"]](y, rule["threshold"]).astype(float)


def apply_target_transform(key: str, y: np.ndarray, rule: dict) -> np.ndarray:
    if key not in TARGET_TRANSFORM_REGISTRY:
        raise ValueError(f"未登記的 target transform: {key!r}，目前有：{sorted(TARGET_TRANSFORM_REGISTRY)}")
    return TARGET_TRANSFORM_REGISTRY[key](y, rule)


def list_available() -> list[dict]:
    """給 GET /api/model/registry/target_transforms 用。"""
    return [{"key": k, "description": (fn.__doc__ or "").strip()} for k, fn in TARGET_TRANSFORM_REGISTRY.items()]
