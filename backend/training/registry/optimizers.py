"""優化器元件登記表：優化器是可共用元件，不寫死在任何 architecture 裡。

architecture 在自己的 `slots` 宣告「optimizer 位置」（例如 LSTM），params_schema 用
`{"name": "optimizer", "type": "component_ref", "component_registry": "optimizer", "required": True}`
引用；這裡的每個優化器宣告自己的參數（名稱、預設值、合法範圍）、相容的 (family, slot)，
以及建立邏輯。相容性由 `architectures.validate()` 依兩邊登記資訊實際比對。

每個優化器宣告：
  key                 唯一識別字串（連同補齊預設值後的完整設定存進 model_config.config.optimizer）
  label／description  顯示用
  params_schema       FieldSpec 陣列（UI 選了這個優化器後依它長出欄位；後端依它驗證並補預設值）
  build               (parameters, cfg) -> torch.optim.Optimizer；cfg 是驗證後的完整設定（不含 type）
  slot_compatibility  [(family, slot_name), ...]：只列真的驗證過的組合

新增優化器（例如 AdamW、SGD）＝在這裡加一個 @register class；引用它的模型只要把
(family, "optimizer") 加進 slot_compatibility（或新模型宣告 optimizer slot），訓練端都經
`build_optimizer()` 建立，不用改模型程式。
"""

OPTIMIZER_REGISTRY: dict = {}


def register(key: str, *, label: str, slot_compatibility: list):
    def deco(spec_cls):
        OPTIMIZER_REGISTRY[key] = {
            "key": key, "label": label, "description": (spec_cls.__doc__ or "").strip(),
            "params_schema": list(spec_cls.params_schema), "build": spec_cls.build,
            "slot_compatibility": [tuple(p) for p in slot_compatibility],
        }
        return spec_cls
    return deco


@register("adam", label="Adam", slot_compatibility=[("lstm", "optimizer")])
class Adam:
    """Adam（torch.optim.Adam）。weight_decay 是 Adam 原始的 L2 形式（加進梯度，不是 AdamW 的解耦式）。"""

    params_schema = [
        {"name": "lr", "type": "float", "required": True, "exclusive_min": 0, "label": "學習率 lr"},
        {"name": "weight_decay", "type": "float", "required": True, "min": 0, "label": "weight_decay（L2）"},
        {"name": "beta1", "type": "float", "default": 0.9, "min": 0, "exclusive_max": 1, "label": "beta1",
         "description": "一階動差衰減率"},
        {"name": "beta2", "type": "float", "default": 0.999, "min": 0, "exclusive_max": 1, "label": "beta2",
         "description": "二階動差衰減率"},
        {"name": "eps", "type": "float", "default": 1e-8, "exclusive_min": 0, "label": "eps",
         "description": "分母的數值穩定項"},
    ]

    @staticmethod
    def build(parameters, cfg: dict):
        import torch

        return torch.optim.Adam(parameters, lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]), eps=cfg["eps"],
                                weight_decay=cfg["weight_decay"])


def build_optimizer(parameters, cfg: dict):
    """cfg＝validate() 解析後的 `{"type": key, ...完整參數}`。"""
    entry = OPTIMIZER_REGISTRY[cfg["type"]]
    return entry["build"](parameters, {k: v for k, v in cfg.items() if k != "type"})


def list_available() -> list[dict]:
    return [
        {"key": k, "label": s["label"], "description": s["description"], "params_schema": s["params_schema"],
         "slot_compatibility": [list(p) for p in s["slot_compatibility"]]}
        for k, s in OPTIMIZER_REGISTRY.items()
    ]
