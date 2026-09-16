"""Decision Rule Node 登記表：把模型輸出（帶時間戳＋task_type）轉成回測引擎看得懂的訊號。

這是 F 項的契約，這輪只定義介面＋兩個最小驗證 function，不做後台完整串接 UI（模型訓練完
→ 選一條決策規則 → 送進回測 job 這整條路，排在下一輪）。這裡刻意不碰 `indicators/tree.py`
本身——它現有的 `entries`/`exits` 布林序列格式已經很成熟（回測系統就是吃這個），這個模組只負責
「模型輸出 → 這個格式」的轉換，回測引擎完全不用改。

每個函式簽名一致：`(predictions, timestamps, task_type, params) -> dict`
  predictions — 模型輸出，分類的機率模式是 (n, n_classes)，類別/回歸模式是 (n,) 或 (n, 1)
  timestamps  — 跟 predictions 逐列對應的時間戳（np.datetime64 陣列）
  task_type   — "classification" 或 "regression"，決定這個規則能不能用（不合用要丟例外，
                不是硬套出沒意義的數字）
  params      — 規則本身的參數（例如機率門檔值）
回傳 dict，key 是 "long_entry"／"long_exit"／"short_entry"／"short_exit"（四個方向的語意跟
`indicators/tree.py` 的 `TRIGGER_LABELS` 完全對應），value 是 (n,) 的布林陣列；沒有的方向可以
省略，不用四個都給——保留多空進出場四種語意，不收斂成單一「做多」布林值。
"""

import numpy as np

DECISION_RULE_REGISTRY: dict = {}


def register(key: str):
    def deco(fn):
        DECISION_RULE_REGISTRY[key] = fn
        return fn
    return deco


@register("probability_threshold")
def probability_threshold(predictions: np.ndarray, timestamps: np.ndarray, task_type: str, params: dict) -> dict:
    """分類機率門檻：只適用分類模型的機率輸出（`output_mode="probability"`），不是類別輸出。
    params: class_index（要看哪一類的機率，例如二分類的「漲」通常是 index 1，預設 1）、
    entry_threshold（機率超過這個門檻進場，預設 0.6）、exit_threshold（低於這個門檻出場，預設 0.4）。
    只產生多方進出場訊號（long_entry/long_exit）——要不要同時做空方，由呼叫端決定要不要另外
    用 class_index=0（或其他類別）再呼叫一次，這裡不擅自幫多空各配一組。
    """
    if task_type != "classification":
        raise ValueError("probability_threshold 只適用分類模型，回歸模型請用 regression_sign")
    predictions = np.asarray(predictions)
    if predictions.ndim != 2:
        raise ValueError(
            f"probability_threshold 需要機率輸出（(n, n_classes) 的 2D 陣列），"
            f"收到的是 {predictions.ndim}D——這個模型訓練時 output_mode 是不是沒設成 probability？"
        )
    class_index = int(params.get("class_index", 1))
    entry_threshold = float(params.get("entry_threshold", 0.6))
    exit_threshold = float(params.get("exit_threshold", 0.4))
    if not (0 <= class_index < predictions.shape[1]):
        raise ValueError(f"class_index={class_index} 超出範圍，這個模型只有 {predictions.shape[1]} 類")

    prob = predictions[:, class_index]
    return {
        "long_entry": prob > entry_threshold,
        "long_exit": prob < exit_threshold,
    }


@register("regression_sign")
def regression_sign(predictions: np.ndarray, timestamps: np.ndarray, task_type: str, params: dict) -> dict:
    """回歸值正負號當方向：只適用「有方向意義」的回歸值（報酬率，simple_return/log_return），
    不適用 future_price（多半是正值，正負號沒有意義，價格回歸要先跟一個明確基準價相減才能
    取得方向，這裡不做這個轉換，直接擋掉避免誤用）。
    params: outcome_key（這個模型訓練時用的 outcome，僅供這裡的合法性檢查用，不影響計算）。
    """
    if task_type != "regression":
        raise ValueError("regression_sign 只適用回歸模型，分類模型請用 probability_threshold")
    outcome_key = params.get("outcome_key")
    if outcome_key == "future_price":
        raise ValueError(
            "regression_sign 不能用在 outcome=future_price 的模型——價格是絕對值域，"
            "正負號沒有方向意義，要嘛換 outcome 成 simple_return/log_return，要嘛先跟明確的基準價相減"
        )
    predictions = np.asarray(predictions).reshape(-1)
    return {
        "long_entry": predictions > 0,
        "long_exit": predictions <= 0,
    }


def list_available() -> list[dict]:
    """給 GET /api/model/registry/decision_rules 用，Swagger 自動文件化，Agent/MCP 查詢用。"""
    return [{"key": k, "description": (fn.__doc__ or "").strip()} for k, fn in DECISION_RULE_REGISTRY.items()]


def apply_decision_rule(key: str, predictions: np.ndarray, timestamps: np.ndarray, task_type: str, params: dict | None = None) -> dict:
    if key not in DECISION_RULE_REGISTRY:
        raise ValueError(f"未登記的 decision_rule key: {key!r}，目前有：{sorted(DECISION_RULE_REGISTRY)}")
    return DECISION_RULE_REGISTRY[key](predictions, timestamps, task_type, params or {})
