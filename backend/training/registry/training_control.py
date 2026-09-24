"""訓練控制（輪數／早停／best-last 追蹤）的共用定義，所有 architecture 同一套語意。

- 輪數各自用自己的名稱表達：神經網路是 `epochs`（完整走過訓練集幾次），XGBoost 是
  `n_estimators`（boosting rounds＝長幾棵樹），兩者不是同一件事，不混用同一個欄位名。
- 早停一律寫在 `early_stopping` 物件裡：`patience` 是整數，**0＝關閉早停**（跑滿設定的
  輪數），正整數＝連續幾輪監控指標沒有改善（超過 `min_delta`）就停。沒有另外的
  enabled 開關——單一欄位、單一語意。
- 不管早停開或關，「監控指標最好的那一輪（best）」一律追蹤，best 與最後一輪（last）
  都保存、都各自完整評估；早停只決定要不要提前結束，不決定有沒有 best。
"""


def early_stopping_fields(monitor: dict | None) -> list:
    """回傳 `early_stopping.*` 的 FieldSpec。monitor=None 代表這個 architecture 的監控指標
    固定（例如 XGBoost 由 eval_metric 決定），schema 不開放選擇、也不提供 min_delta。"""
    fields = [
        {"name": "early_stopping.patience", "type": "int", "required": True, "min": 0,
         "label": "早停 patience", "description": "0＝關閉早停、跑滿設定輪數；正整數＝連續幾輪沒改善就停"},
    ]
    if monitor is not None:
        fields.insert(0, {"name": "early_stopping.monitor", "type": "enum", "required": True,
                          "label": "監控指標", "description": "決定 best 那一輪，以及早停依據", **monitor})
        fields.append({"name": "early_stopping.min_delta", "type": "float", "default": 0.0, "min": 0,
                       "label": "最小改善量", "description": "改善幅度要超過這個值才算進步"})
    return fields


class BestTracker:
    """逐輪追蹤監控指標：記錄 best 那一輪、判斷是否該早停。mode 是 "min" 或 "max"。"""

    def __init__(self, mode: str, patience: int, min_delta: float = 0.0):
        self.mode, self.patience, self.min_delta = mode, int(patience), float(min_delta)
        self.best_value = None
        self.best_epoch = -1
        self.wait = 0

    def update(self, epoch: int, value: float) -> tuple[bool, bool]:
        """回傳 (這一輪是不是新的 best, 是不是該停了)。"""
        if self.best_value is None or (
            value < self.best_value - self.min_delta if self.mode == "min" else value > self.best_value + self.min_delta
        ):
            self.best_value, self.best_epoch, self.wait = value, epoch, 0
            return True, False
        self.wait += 1
        return False, self.patience > 0 and self.wait >= self.patience
