# 模型擴充指南（通用模型組裝框架）

> 對象：要新增模型架構（Transformer、Mamba…）、新 Attention 型態或新可插拔元件的開發者。使用 API 送訓練請看 [api.md](api.md)。
> 對應原始碼：`backend/training/registry/`（`architectures.py`、`param_schema.py`、`training_control.py`、`attention.py`、`optimizers.py`、`lstm.py`、`xgboost_model.py`），前端 `frontend/src/pages/admin/sections/SchemaForm.tsx`／`schemaFormLogic.ts`。

## 框架怎麼運作

```
register(key, family, label, params_schema, extra_checks, capabilities, slots, outputs, describe_outputs, metric_specs, lazy_windows)
        │
        ├─ GET /api/model/registry/architectures ── list_available()：key／capabilities／slots／params_schema／outputs
        ├─ GET /api/model/registry/components/{registry} ── 元件（attention、optimizer…）的 params_schema／slot_compatibility
        │        └─ 後台 SchemaForm 依 params_schema 產生表單；元件欄位只列與 (family, slot) 相容的元件，
        │           選定後依該元件的 params_schema 長出欄位（不為個別架構或元件寫表單）
        │
        ├─ 提交驗證 validate_graph_spec → describe_outputs → validate(key, params, context)
        │        1. validate_with_schema：型別、必填、範圍、enum_cases、visible_if、未知欄位、component_ref 子參數
        │        2. slot 相容性：元件的 slot_compatibility 含 (family, slot)，output_kind 在 accepts_output_kind 內
        │        3. extra_checks(cfg, context)：schema 表達不了的模型專屬規則
        │
        └─ 訓練 train_model(key, …) → 再驗證一次 → train 函式（拿到解析後的完整 cfg）
           推論 load_model(key, weights, model_config)
```

`graph.py`、`inference.py`、API router、前端表單都只透過上面這組介面驅動模型，新增架構不需要修改它們。

## 新增一個架構（例如 Transformer）

新增 `backend/training/registry/transformer.py`，並在 `architectures.py` 檔尾的匯入區加一行 `from training.registry import transformer`。

```python
from training.registry.architectures import register, register_loader
from training.registry.optimizers import build_optimizer
from training.registry.training_control import BestTracker, early_stopping_fields

PARAMS_SCHEMA = [
    {"name": "d_model", "type": "int", "required": True, "min": 1, "label": "d_model"},
    {"name": "n_layers", "type": "int", "required": True, "min": 1, "max": 12, "label": "層數"},
    {"name": "optimizer", "type": "component_ref", "component_registry": "optimizer", "required": True, "label": "優化器"},
    {"name": "batch_size", "type": "int", "required": True, "min": 1, "label": "batch_size"},
    {"name": "epochs", "type": "int", "required": True, "min": 1, "label": "epochs"},
    *early_stopping_fields({"enum_values": ["val_loss", "val_accuracy"],
                            "enum_cases": [{"when": {"field": "$task_type", "equals": "regression"},
                                            "values": ["val_loss", "val_rmse"]}]}),
    {"name": "seed", "type": "int", "required": True, "min": 0, "label": "seed"},
]

@register("transformer", family="transformer", label="Transformer", params_schema=PARAMS_SCHEMA,
          capabilities={"bidirectional": "fixed_off", "attention": "fixed_on", "dual_head": "fixed_off",
                        "optimizer": "configurable"},
          slots=[{"slot_name": "optimizer", "component_registry": "optimizer", "cardinality": "exactly_one"}],
          metric_specs=metric_specs)   # 見下方「宣告指標」
def train_transformer(X_train, y_train, X_val, y_val, cfg, on_epoch, task_type="classification", *,
                      preview_hook=None, n_classes=None) -> dict:
    ...
    opt = build_optimizer(model.parameters(), cfg["optimizer"])   # 共用機制建立，不直接 new torch.optim.*
    ...

@register_loader("transformer")
def load_transformer(weights: bytes, model_config: dict) -> dict:
    ...
```

**train 函式契約**（所有架構一致，前端與文件都依賴這個形狀）：

| 回傳鍵 | 內容 |
|---|---|
| `final_metrics` | best 那一輪的 `val_*` 指標＋`last`（最後一輪同一組指標）＋訓練控制（`monitor`、`patience`、best／實際輪數、`stopped_early`） |
| `evaluation` | `{"best": 報告, "last": 報告}`，用 `training/evaluation.py` 的 `classification_evaluation`／`regression_evaluation`，對整個驗證集算 |
| `weights`／`weights_last` | best／last 兩組權重（bytes） |
| `predict`（＋分類 `predict_proba`；多輸出 `predict_outputs`） | 以 best 權重推論 |
| `model_config` | 重建與重載需要的全部設定（可 JSON 化），至少含解析後的 `config` |
| `device` | `"cpu"`／`"cuda"` |

- `cfg` 已經是驗證後的完整設定（含預設值），不要在 train 函式裡再補預設。
- 分類的類別數用參數 `n_classes`（來自 Label），不要放進模型參數。
- 要用既有優化器：在 `optimizers.py` 把 `("transformer", "optimizer")` 加進該優化器的 `slot_compatibility`（只加真的驗證過的組合），訓練時用 `build_optimizer()`；`model_config` 保存 `cfg` 即自動包含優化器名稱與補齊預設後的完整設定。
- 早停：`BestTracker(mode, patience, min_delta).update(epoch, value) -> (improved, stop)`；`patience=0` 永遠不停、只追蹤 best。
- 每輪呼叫 `on_epoch(epoch, {"loss", "accuracy", "val_loss", "val_accuracy", ...擴充指標})`；擴充指標會自動存進進度的 `metrics` 欄。
- 多個具名輸出：`register(..., outputs=(DEFAULT_OUTPUT, "regression", ...), describe_outputs=fn)`，`fn(node, label_node, task_type)` 依這次設定回傳 `output_specs`（參考 `lstm.describe`）。
- `lazy_windows=True`：訓練迴圈逐 batch 取樣時，graph.py 會傳入省記憶體的 LazyWindowed（參考 LSTM）。

**宣告指標（`metric_specs`）**：前台與 Agent 依它呈現逐輪曲線與評估結果，新架構必須宣告：

```python
from training.registry import metrics as M

def metric_specs(node, label_node, task_type):
    return M.build_specs(
        round_unit="epoch",
        heads=[{"key": "default", "label": "輸出", "task": task_type}],
        series=[{"id": "loss", "metric": "cross_entropy", "head": "default", "train": "loss", "val": "val_loss"}],
        evaluation={"default": M.CLASSIFICATION_EVAL if task_type == "classification" else M.REGRESSION_EVAL})
```

- `series[].train`／`val` 必須是 `on_epoch` 真的回報的欄位名（測試會實際訓練一次核對，見 `tests/test_evaluation_records.py`）。
- 引用的指標必須已在 `training/registry/metrics.py` 登記；沒有的先登記（見「新增指標」）。
- 評估報告沿用 `training/evaluation.py` 的分類／回歸報告，worker 會轉成 `evaluations`；前端不用改。

**必寫的測試**（參考 `tests/test_lstm.py`、`tests/test_xgboost.py`、`tests/test_model_framework.py`）：schema 驗證（必填、範圍、不適用欄位）、訓練後 best／last 與 evaluation 都在、`weights`／`weights_last` 重載後推論與訓練當下一致、`patience=0` 跑滿、`model_config` 可 JSON 化、經 `run_graph` 能被下游引用。

## 新增一個 Attention 型態

在 `training/registry/attention.py` 加一個 class，用 `@register(...)` 宣告：

```python
@register("scaled_dot", query_source="self", combine="concat_summary",
          slot_compatibility=[("lstm", "attention")], label="Scaled dot-product（self）")
class ScaledDotAttention:
    """說明文字（會出現在 registry 與 UI）。"""
    params_schema = [{"name": "heads", "type": "int", "required": True, "min": 1, "label": "heads"}]

    @staticmethod
    def validate_params(params) -> dict: ...          # 最後一道把關，回傳解析後設定
    @staticmethod
    def context_dim(hidden_size, cfg) -> int: ...
    @staticmethod
    def build(hidden_size, cfg): ...                  # nn.Module，forward(seq)->(context, weights)
```

- `slot_compatibility` 只列**真的驗證過**的 `(family, slot)`；沒列的組合提交時會 400。
- `output_kind`：`"context_vector"`（序列進、向量出，目前 LSTM 的 attention slot 只接受這種）或 `"sequence"`。
- 建網路時 `build_checked()` 會用假序列跑一次 forward，形狀跟 `context_dim` 不一致會直接報錯。
- 前端不用改：選了這個型態後，SchemaForm 讀它的 `params_schema` 長出專屬欄位。

## 新增一個優化器（例如 AdamW、SGD）

在 `training/registry/optimizers.py` 加一個 class：

```python
@register("adamw", label="AdamW", slot_compatibility=[("lstm", "optimizer")])
class AdamW:
    """AdamW（解耦式 weight decay）。"""
    params_schema = [
        {"name": "lr", "type": "float", "required": True, "exclusive_min": 0, "label": "學習率 lr"},
        {"name": "weight_decay", "type": "float", "required": True, "min": 0, "label": "weight_decay（解耦）"},
        {"name": "beta1", "type": "float", "default": 0.9, "min": 0, "exclusive_max": 1, "label": "beta1"},
        {"name": "beta2", "type": "float", "default": 0.999, "min": 0, "exclusive_max": 1, "label": "beta2"},
        {"name": "eps", "type": "float", "default": 1e-8, "exclusive_min": 0, "label": "eps"},
    ]

    @staticmethod
    def build(parameters, cfg):          # cfg：驗證後、補齊預設值的完整設定（不含 type）
        import torch
        return torch.optim.AdamW(parameters, lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
                                 eps=cfg["eps"], weight_decay=cfg["weight_decay"])
```

- 參數、預設值、合法範圍只寫在 `params_schema`；後端驗證、補預設值、UI 欄位都讀它。
- `slot_compatibility` 決定哪些模型可以用；模型程式、API、前端都不用改。
- 測試參考 `tests/test_optimizers.py`：預設與自訂參數確實傳進 `torch.optim`（檢查 `param_groups`）、產物保存完整設定、不相容組合被拒。

## 新增指標

在 `training/registry/metrics.py` 登記一行，再由模型的 `metric_specs` 引用：

```python
register("sharpe", label="Sharpe ratio", direction="max", unit="ratio", fmt="decimal", description="…")
```

- `shape` 決定前端元件與評估報告裡的資料契約：

  | shape | 評估報告裡的值 | 前端元件 |
  |---|---|---|
  | `scalar` | 數值或 `null`（附 `<key>_unavailable_reason`） | 逐輪曲線、`MetricCompare` |
  | `per_class` | `{類別: {欄位: 值}}`，欄位自訂 | `PerClassTable` |
  | `distribution` | `{類別: 數值}` | `DistributionBars` |
  | `matrix` | 二維陣列；標籤放 `<key>_labels`；登記時給 `axes={"row": …, "col": …}` | `MatrixHeatmap` |

- 衍生指標：登記時給 `derive={"from": 來源指標, "method": "matrix_diagonal_ratio"}`（方法在 `metrics.DERIVATIONS`，需要新方法就加一個函式）。
- 新基準：評估報告放 `baseline_<名稱>` 物件（`method`＋同名指標數值），在 `metrics.register_baseline()` 登記名稱與標籤（未登記時以名稱本身顯示）。
- 逐輪指標：在模型 `on_epoch` 回報對應欄位（例如 `sharpe`／`val_sharpe`）並加進 `series`；評估指標：在評估報告產生對應鍵並加進 `evaluation[head]`。
- 評估轉換（`training/evaluation_records.py`）與前端（`frontend/src/components/metrics/`）都依登記的 shape 分派，沒有固定指標清單；同形態的新指標不用改程式。未登記的鍵不會出現在評估清單（`tests/test_evaluation_records.py` 核對 `evaluation.py` 產生的鍵都已登記）。

## 新增一種可插拔位置（新的元件登記表）

例如在序列摘要後加 Pooling slot：

1. 新增 `training/registry/pooling.py`（仿 `attention.py`／`optimizers.py`：`POOLING_REGISTRY`、`register`、`list_available`）。
2. `architectures._component_modules()` 與 `component_registries()` 加上 `pooling`（`param_schema` 的 `component_ref`、slot 相容性檢查、`GET /registry/components/pooling` 都從這裡查）。
3. 架構的 `slots` 宣告 `{"slot_name": "pooling", "component_registry": "pooling", "cardinality": "zero_or_one", "accepts_output_kind": [...]}`，`params_schema` 加 `{"name": "pooling", "type": "component_ref", "component_registry": "pooling", "default": null}`。
4. API 與前端不用改：`GET /api/model/registry/components/{registry}` 是通用端點；後台依架構 `slots` 宣告的 `component_registry` 名稱自動抓元件清單。

## 未來：多元件組合（本版未實作，只記錄接點）

本版每個 slot 最多一個元件（`cardinality="zero_or_one"`），payload 送清單一律 400。要支援多個 Attention 時的接點：

| 位置 | 要做的事 |
|---|---|
| slot 宣告（`lstm.py` 的 `register(..., slots=...)`） | `cardinality` 改為 `"zero_or_many"` |
| `param_schema.py` 的 `component_ref` 驗證 | 目前遇到清單直接回「本版本只支援單一元件」；改為依 slot 的 cardinality 允許清單、逐項驗證 |
| `architectures._slot_compatibility_problems` | 對清單每一項檢查相容性；加上相鄰元件的 `output_kind` 能否銜接 |
| 元件登記的 `combine` | 新增 `"sequential"`（兩個 `sequence` 元件串接：前一個的序列輸出餵下一個）與 `"parallel_concat"`（多個 `context_vector` 元件各自對同一序列算向量後串接） |
| `lstm.build_net` | 依 `combine` 分派組合方式，計算合併後的 context 維度 |
| 前端 `SchemaForm` | `component_ref` 支援多選清單（每項各自的子參數） |

`context_vector` 元件沒有時間維度，無法串接到下一個序列消費者，只能並聯；要串接必須先有 `output_kind="sequence"` 的型態。

## 訓練控制與保存的共通規則

- 輪數各自表達：神經網路用 `epochs`，boosting 用 `n_estimators`；`early_stopping.patience`：0＝關閉。
- best／last 一律都保存（`model_artifacts.weights`／`weights_last`）且各自完整評估；`POST /api/model/infer` 的 `use_weights` 選用哪一組。
- 寫進 JSONB 的結果會把非有限數值轉成 `null`（`training/json_safe.py`）；評估指標算不出來時要自己給 `*_unavailable_reason`，不要回 NaN。
- Phase 1／2／3 的流程、切分與評估契約由 `graph.py`／router 負責，架構模組不處理切分。
