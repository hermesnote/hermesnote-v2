# 模型訓練／推論 API

> 對應原始碼：`backend/routers/model_training.py`（送出/查詢）＋ `backend/training/graph.py`（graph_spec 節點格式的權威定義）。連線位址見 `../index.md`。
>
> 這裡只保證**格式跟現在程式碼一致**，不代表已經做到論文等級的嚴謹（見文末「已知限制」）。新增模型／元件的方法見 [extending.md](extending.md)。

## 流程總覽（訓練）

1. `GET /api/model/registry/{features,outcomes,labeling_rules,architectures,metrics}` 與 `GET /api/model/registry/components/{attention,optimizer}` → 查目前登記的可用選項、每個模型與元件的參數 schema（永遠用這幾支查即時清單，不要背下面範例裡的值）
2. 組出 `graph_spec`（見下方「節點格式」），`POST /api/model/train` → 立刻回 `job_id`（訓練在 `hermesnote-training` 容器背景跑，不卡住這個請求）
3. 輪詢 `GET /api/model/jobs/{job_id}` → 看 `status`，`done` 才有 `result`
4. 訓練中想看即時進度：`GET /api/model/jobs/{job_id}/progress`（補歷史用）或直接開 WebSocket `GET /api/model/jobs/{job_id}/stream`（即時推）
5. 訓練完成後，模型產物已自動存進 `model_artifacts`（不用另外呼叫「儲存」），可以直接拿去 `POST /api/model/infer`
6. 人要看圖表的話，瀏覽器開 `{前台網址}/model?job={job_id}`

## `POST /api/model/train`

**請求 body：**

```json
{
  "start": "2020-01-01",
  "end": "2020-12-31",
  "nodes": [
    { "id": "feat_ohlcv", "type": "feature", "key": "ohlcv", "timeframe": "tx_1m" },
    { "id": "feat_rsi", "type": "feature", "key": "talib_indicator", "timeframe": "tx_1m",
      "params": { "name": "RSI", "timeperiod": 14 } },
    { "id": "label1", "type": "label", "outcome": "simple_return", "labeling_rule": "fixed_threshold",
      "timeframe": "tx_1m", "params": { "horizon": 1, "n_classes": 2, "threshold_pct": 0.0 } },
    { "id": "model1", "type": "model", "key": "lstm",
      "inputs": ["feat_ohlcv", "feat_rsi"], "label": "label1",
      "window": 60, "val_ratio": 0.2, "split_strategy": "random", "output_mode": "probability",
      "params": { "lstm_layers": [{"units": 64, "dropout": 0.2}], "heads": "single",
                  "shared": {"dense": 32, "dropout": 0.2},
                  "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0}, "batch_size": 128,
                  "epochs": 300, "early_stopping": {"monitor": "val_loss", "patience": 0}, "seed": 42 } }
  ],
  "phase": null,
  "parent_job_id": null
}
```

| 欄位 | 型別 | 說明 |
|------|------|------|
| `start` / `end` | string，`YYYY-MM-DD` | 資料範圍 |
| `nodes` | array | 節點圖，見下方「節點格式」；目前 `phase=2` 仍須送 `[]` 佔位以通過 schema 驗證，後端忽略此值並繼承 Phase 1（見下面 Phase 說明） |
| `phase` | `1`／`2`／`null` | 教授三階段協定用（Reproducer／Experimenter 一律從 `1` 開始）；`3`（holdout 推論）不走這支，走 `/infer` |
| `parent_job_id` | string 或 `null` | `phase=2` 時必填，必須是一個已完成的 Phase 1 job id |

**Phase 2 特例（已知的 schema 落差，2026-09-17 對照 OpenAPI 實測確認）**：`phase=2` 時，後端處理邏輯上**不會使用**呼叫端送的 `nodes`/`start`/`end`（會直接繼承 `parent_job_id` 那筆 Phase 1 的 `graph_spec`，只把切分方式覆寫成 chronological），但 `TrainRequest` 這個 Pydantic 模型的 `start`/`end`/`nodes` 三個欄位目前**沒有標成可省略**，OpenAPI／FastAPI 的請求驗證仍然會要求這三個欄位存在，省略會直接被擋在驗證層（422），連後端邏輯都還沒執行到。這是程式碼本身的 schema 沒跟語意對齊，不是文件寫錯——已回報，之後可能會把這三個欄位改成可選；**在那之前，請用下面的請求範例（帶佔位值，會被忽略）：**

```json
{ "start": "", "end": "", "nodes": [], "phase": 2, "parent_job_id": "<一個已完成的 Phase 1 job_id>" }
```

`start`/`end`/`nodes` 只要型別對（字串／字串／陣列）就會通過驗證，實際值不影響結果——後端在 `phase==2` 分支裡完全不會讀這三個欄位。

**回應：** `{ "job_id": "..." }`

### 節點格式（`nodes` 陣列裡的三種節點）

每個節點必有 `id`（自訂字串，同一個 graph_spec 裡唯一）跟 `type`（`"feature"`／`"label"`／`"model"`）。

**Feature Node**（`type: "feature"`）：

| 欄位 | 說明 |
|------|------|
| `key` | 用哪個特徵函式，即時清單見 `GET /api/model/registry/features`（目前有 `ohlcv`／`raw_price`／`raw_volume`／`talib_indicator`） |
| `timeframe` | `"{symbol}_{timeframe}"`，例如 `"tx_1m"`／`"tx_15m"` |
| `params` | 依 `key` 而定：`ohlcv`/`raw_price`/`raw_volume` 不需要；`talib_indicator` 必須有 `name`（TA-Lib 指標代號，跟 `../indicators.md` 的 `key` 同一套），其餘 key 當該指標的參數（例如 `{"name": "RSI", "timeperiod": 14}`） |

**Label Node**（`type: "label"`，一個 graph_spec 可以有多個，各自被不同 Model Node 引用）：

| 欄位 | 說明 |
|------|------|
| `outcome` | 算原始數值用哪個函式，即時清單見 `GET /api/model/registry/outcomes`（目前有 `simple_return`／`log_return`，都吃 `params.horizon`，預設 1） |
| `labeling_rule` | 把原始數值轉成學習目標，即時清單見 `GET /api/model/registry/labeling_rules`（每項帶 `task_type`）：`"fixed_threshold"`（分類：`params.n_classes` 2 或 3＝模型的類別數、`params.threshold_pct` 門檻，預設 2／0.0）、`"identity"`（回歸：原始數值直接當目標） |
| `timeframe` | 同 Feature Node |
| `params` | 依 `outcome`/`labeling_rule` 而定，見上面兩個登記表 |

注意：不是每個 `outcome`＋`labeling_rule` 組合都合法（例如 `future_price` 不能配 `fixed_threshold`，價格絕對值域不能套百分比門檻），後端會直接擋、回 400 並附原因，不是只靠前端表單擋——Agent 直接組 graph_spec 送出時一樣會被檢查。

**Model Node**（`type: "model"`，可以有多個，彼此可以互相接——把上游 Model Node 的預測結果當下游的輸入特徵，等於模型融合）：

| 欄位 | 說明 |
|------|------|
| `key` | `"lstm"` 或 `"xgboost"`；即時清單見 `GET /api/model/registry/architectures`（見下方「模型設定」） |
| `inputs` | 輸入引用陣列，順序就是特徵欄位串接的順序（訓練與推論一致）。每一項是節點 id 字串（Feature Node 或 Model Node，取 `default` 輸出），或具名輸出引用 `{"node": "<Model Node id>", "output": "<輸出名>"}`，見下方「具名輸出」 |
| `label` | 引用哪個 Label Node 的 id（決定 `task_type` 是 classification 還是 regression；分類的類別數由該 Label 的 `params.n_classes` 決定，**不是模型參數**） |
| `window` | 滑動窗格大小（幾根K棒組一個樣本），預設 `60` |
| `val_ratio` | 驗證集比例，預設 `0.2` |
| `split_strategy` | `"random"`（預設，Phase 1 用）或 `"chronological"`（依時間先後切，Phase 2 由後端依母任務自動覆寫，不用自己送） |
| `output_mode` | 只適用「單頭分類」的 `default` 輸出：不給＝類別，`"probability"`＝完整機率向量（推論也存機率）。雙頭（`heads="dual"`）送了會 400 |
| `params` | 模型參數，形狀完全由該 architecture 登記的 `params_schema` 決定，見下節 |

## 模型設定（統一入口：schema 驅動）

LSTM 與 XGBoost 用**同一套**登記／參數 schema／表單／後端驗證機制；每個架構只接受自己 schema 裡的欄位。後台表單就是讀同一份 schema 產生的，後端驗證是最終把關（提交時一次列出全部問題，HTTP 400、不建立 job）。

### 1. 查能力：`GET /api/model/registry/architectures`

每一項：

```json
{"key": "lstm", "family": "lstm", "label": "LSTM",
 "outputs": ["default", "regression", "direction_probability"],
 "capabilities": {"bidirectional": "configurable", "attention": "configurable", "dual_head": "configurable",
                  "optimizer": "configurable"},
 "slots": [{"slot_name": "attention", "component_registry": "attention", "cardinality": "zero_or_one",
            "accepts_output_kind": ["context_vector"]},
           {"slot_name": "optimizer", "component_registry": "optimizer", "cardinality": "exactly_one"}],
 "params_schema": [ ...FieldSpec... ], "lazy_windows": true, "description": "..."}
```

```json
{"key": "xgboost", "family": "xgboost", "label": "XGBoost", "outputs": ["default"],
 "capabilities": {"bidirectional": "fixed_off", "attention": "fixed_off", "dual_head": "fixed_off", "optimizer": "fixed_off"},
 "slots": [], "params_schema": [ ...FieldSpec... ], "lazy_windows": false, "description": "..."}
```

- `capabilities`（三態 `fixed_on`／`fixed_off`／`configurable`）只說「這個 key **允許**設定什麼」，不代表某一次提交實際選了什麼；實際選擇看那次的 `params` 或產物的 `model_config`。
- `slots`：可插拔位置，`slot_name` 就是 params 裡的欄位名稱，`component_registry` 是元件登記表名稱。`cardinality="zero_or_one"`＝可不選、最多一個（attention）；`"exactly_one"`＝必選一個（optimizer）；`accepts_output_kind`（只有元件輸出會接進網路的位置才有）＝接受的元件輸出形狀。
- `outputs`：可能提供的具名輸出（靜態超集）；實際輸出依設定而定，提交時驗證。

可共用元件：`GET /api/model/registry/components/{registry}`，`registry` 取自 slots 的 `component_registry`（目前 `attention`、`optimizer`；未登記的名稱回 404）。

```json
GET /api/model/registry/components/attention
[{"key": "additive", "label": "Additive（無 query，tanh 打分）",
  "params_schema": [{"name": "dim", "type": "int", "required": true, "min": 1, "label": "attention.dim"}],
  "query_source": "none", "combine": "concat_summary", "output_kind": "context_vector",
  "slot_compatibility": [["lstm", "attention"]], "description": "..."}]
```

```json
GET /api/model/registry/components/optimizer
[{"key": "adam", "label": "Adam",
  "params_schema": [
    {"name": "lr", "type": "float", "required": true, "exclusive_min": 0, "label": "學習率 lr"},
    {"name": "weight_decay", "type": "float", "required": true, "min": 0, "label": "weight_decay（L2）"},
    {"name": "beta1", "type": "float", "default": 0.9, "min": 0, "exclusive_max": 1, "label": "beta1"},
    {"name": "beta2", "type": "float", "default": 0.999, "min": 0, "exclusive_max": 1, "label": "beta2"},
    {"name": "eps", "type": "float", "default": 1e-08, "exclusive_min": 0, "label": "eps"}],
  "slot_compatibility": [["lstm", "optimizer"]], "description": "..."}]
```

後端實際檢查相容性：元件的 `slot_compatibility` 要包含 `(architecture family, slot)`；有 `accepts_output_kind` 的位置，元件 `output_kind` 也要在清單內。不相容、未登記的 `type`、必選位置送 `null` 都是 400。Attention 在建網路時再用假序列跑一次 forward，確認介面形狀跟宣告一致。後台表單同樣只列出與所選架構相容的元件。

### 2. 讀 schema：`params_schema`（FieldSpec 陣列）

```json
{"name": "early_stopping.monitor", "type": "enum", "required": true,
 "enum_values": ["val_loss", "val_accuracy"],
 "enum_cases": [{"when": {"field": "heads", "equals": "dual"},
                 "values": ["val_joint_loss", "val_mse", "val_bce", "val_rmse", "val_mae", "val_dir_acc"]},
                {"when": {"field": "$task_type", "equals": "regression"}, "values": ["val_loss", "val_rmse", "val_mae"]}]}
```

| 屬性 | 意義 |
|---|---|
| `name` | dot path，對應 params 裡的巢狀位置（`shared.dense` → `{"shared": {"dense": …}}`） |
| `type` | `int`／`float`／`bool`／`string`／`enum`／`component_ref`（值是 `{"type": 元件key, ...該元件 params_schema 的欄位}`；`default` 為 `null` 的欄位（例如 `attention`）才可以送 `null`＝不用，必填的（例如 `optimizer`）不行；元件來源見 `component_registry`）／`array_of_object`（配 `item_schema`、`min_items`／`max_items`） |
| `required` | `true`／`false` 或條件；**沒有 `default` 的必填欄位＝研究值，平台不代填** |
| `default` | 省略時採用的值（解析後的完整設定會存進產物） |
| `min`／`max`／`exclusive_min`／`exclusive_max` | 數值範圍（exclusive＝不含端點） |
| `enum_values`＋`enum_cases` | 可選值；`enum_cases` 依序比對，第一個條件成立的取代預設清單 |
| `visible_if` | 條件不成立時此欄位**不適用**：送了會 400（不是靜默忽略） |
| `derived_from` | 值由情境決定（例如 `"$outcome_unit"`＝Label outcome 的單位）；可省略，送了必須相等 |

條件語法：`{"field": "heads", "equals": "dual"}`／`{"field": …, "not_equals": …}`／`{"all": [...]}`／`{"any": [...]}`。`field` 以 `$` 開頭的是**情境**而不是 params 欄位：`$task_type`（由 Label 的 labeling_rule 決定，見 `registry/labeling_rules` 的 `task_type`）、`$outcome_unit`（由 outcome 決定，見 `registry/outcomes` 的 `unit`）。未知欄位、型別錯誤、超出範圍、不適用卻送了，一律 400 並指出欄位路徑。

### 3. LSTM（`key="lstm"`）

方向（單／雙向）、Attention（無／一個）、輸出頭（單／雙頭）三軸**各自獨立**，同一種 payload 形狀表達全部 8 種組合。

| 欄位 | 必填 | 說明 |
|---|---|---|
| `lstm_layers` | ✓ | 1～8 項 `{units, dropout}`，依序堆疊；每層後一個 dropout |
| `bidirectional` | 預設 `false` | 雙向時序列摘要＝兩個方向**各自跑完整條序列**的最終 hidden state 串接（`h_n[0]`、`h_n[1]`），不是 `out[:, -1]` |
| `attention` | 預設 `null` | `null`＝不用；`{"type": "additive", "dim": 64}`＝用一個。**送清單一律 400**（本版不支援多元件） |
| `heads` | ✓ | `"single"`＝依 Label 分類或回歸；`"dual"`＝回歸頭＋方向事件頭（Label 必須是連續、帶正負號的目標，`labeling_rule="identity"`；分類 Label 只能 `single`） |
| `direction_rule.op`／`.threshold` | 雙頭必填 | 方向事件：`目標 op threshold`（`op` ∈ `> >= < <=`，門檻在目標原單位） |
| `direction_rule.threshold_unit` | 可省略 | 固定等於 Label outcome 的單位（不做單位換算） |
| `loss_weights.regression`／`.direction` | 雙頭必填 | 聯合損失權重（≥0，不可同時為 0） |
| `target_scaling` | 回歸（單頭回歸或雙頭）時可用，預設 `"none"` | `"standardize"`＝訓練時對回歸目標做 train-only 標準化；輸出與 RMSE／MAE 一律原單位 |
| `class_weight` | 單頭分類時可用，預設 `"none"` | `"balanced"`＝依訓練集類別頻率反比加權 |
| `shared.dense`／`shared.dropout` | ✓ | 序列摘要（＋Attention context）之後的共享 Dense 層 |
| `optimizer` | ✓ | 優化器元件 `{"type": "adam", ...}`，參數見下方「優化器」；不能是 `null` |
| `batch_size` | ✓ | mini-batch 大小（不屬於優化器參數） |
| `epochs` | ✓ | 完整走過訓練集的輪數上限 |
| `early_stopping.monitor` | ✓ | 決定 best 那一輪（與早停依據）；可選值：單頭分類 `val_loss`／`val_accuracy`、單頭回歸 `val_loss`／`val_rmse`／`val_mae`、雙頭 `val_joint_loss`／`val_mse`／`val_bce`／`val_rmse`／`val_mae`／`val_dir_acc` |
| `early_stopping.patience` | ✓ | **`0`＝關閉早停、跑滿 `epochs`**；正整數＝連續幾輪沒改善就停 |
| `early_stopping.min_delta` | 預設 `0` | 視為「改善」的最小幅度 |
| `seed` | ✓ | 權重初始化與洗牌的亂數種子 |

結構：`輸入 → 疊層 LSTM（每層後 dropout）→ 序列摘要（單向：最終 hidden；雙向：兩方向最終 hidden 串接）→［有 Attention：與 context 串接］→ shared Dense＋ReLU＋dropout → 單頭 Linear(k 或 1)／雙頭 回歸 Linear(1)＋方向 Linear(1)`。雙頭損失＝`loss_weights.regression × MSE ＋ loss_weights.direction × BCEWithLogits`。

**範例 A：單向、無 Attention、單頭分類（Label 用 `fixed_threshold`）**

```json
{"id": "m1", "type": "model", "key": "lstm", "inputs": ["f_ohlcv"], "label": "l_cls",
 "window": 60, "val_ratio": 0.2, "output_mode": "probability",
 "params": {
   "lstm_layers": [{"units": 64, "dropout": 0.2}],
   "heads": "single", "class_weight": "none",
   "shared": {"dense": 32, "dropout": 0.2},
   "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0}, "batch_size": 64,
   "epochs": 300, "early_stopping": {"monitor": "val_loss", "patience": 0},
   "seed": 42}}
```

**範例 B：雙向＋Attention、單頭回歸（Label 用 `identity`）**

```json
"params": {
  "lstm_layers": [{"units": 64, "dropout": 0.2}, {"units": 32, "dropout": 0.2}],
  "bidirectional": true, "attention": {"type": "additive", "dim": 32},
  "heads": "single", "target_scaling": "standardize",
  "shared": {"dense": 32, "dropout": 0.2},
  "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0}, "batch_size": 64,
  "epochs": 300, "early_stopping": {"monitor": "val_rmse", "patience": 0},
  "seed": 42}
```

**範例 C：單向＋Attention、雙頭（Label：`log_return`＋`identity`）**——研究草案 v003 所述論文設定的格式示範，數值**不等於已核准實驗**，由研究端決定：

```json
{
  "start": "2011-01-03", "end": "2026-09-18",
  "nodes": [
    {"id": "f_ohlcv", "type": "feature", "key": "ohlcv", "timeframe": "tx_daily_day"},
    {"id": "l1", "type": "label", "outcome": "log_return", "labeling_rule": "identity",
     "timeframe": "tx_daily_day", "params": {"horizon": 1}},
    {"id": "m1", "type": "model", "key": "lstm", "inputs": ["f_ohlcv"], "label": "l1",
     "window": 60, "val_ratio": 0.2,
     "params": {
       "lstm_layers": [{"units": 128, "dropout": 0.2}, {"units": 64, "dropout": 0.2}],
       "bidirectional": false, "attention": {"type": "additive", "dim": 64},
       "heads": "dual",
       "direction_rule": {"op": ">=", "threshold": 0.0, "threshold_unit": "log_ratio"},
       "loss_weights": {"regression": 0.8, "direction": 0.2},
       "target_scaling": "none",
       "shared": {"dense": 32, "dropout": 0.2},
       "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0}, "batch_size": 64,
       "epochs": 300, "early_stopping": {"monitor": "val_rmse", "patience": 0, "min_delta": 0.0},
       "seed": 42}}
  ],
  "phase": 1, "parent_job_id": null
}
```

**範例 D：雙向、無 Attention、雙頭**——在範例 C 的 `params` 改 `"bidirectional": true, "attention": null`，其餘不變。

### 優化器（`optimizer` 元件）

優化器是可共用元件：模型在 `slots` 宣告 optimizer 位置、在 `params_schema` 用 `component_ref` 引用；參數、預設值、合法範圍與建立邏輯集中在元件登記（`training/registry/optimizers.py`）。本版只有 Adam（`torch.optim.Adam`，`weight_decay` 是加進梯度的 L2 形式，不是 AdamW 的解耦式）：

| 欄位 | 必填 | 預設 | 範圍 |
|---|---|---|---|
| `lr` | ✓ | — | > 0 |
| `weight_decay` | ✓ | — | ≥ 0 |
| `beta1` | | 0.9 | [0, 1) |
| `beta2` | | 0.999 | [0, 1) |
| `eps` | | 1e-8 | > 0 |

```json
"optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0}
"optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 1e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1e-9}
```

**產物保存**：`model_config.config.optimizer` 記錄優化器名稱（`type`）與補齊預設值後的完整實際設定，例如只送 `lr`／`weight_decay` 時存成 `{"type": "adam", "lr": 0.001, "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.999, "eps": 1e-08}`。Phase 2 繼承 Phase 1 的 params 原樣使用，優化器設定一併繼承。XGBoost 沒有 optimizer 位置（`capabilities.optimizer="fixed_off"`），送了是 400。

### 4. XGBoost（`key="xgboost"`）

| 欄位 | 必填 | 說明 |
|---|---|---|
| `n_estimators` | ✓ | boosting rounds（長幾棵樹）；跟 LSTM 的 `epochs` 是不同概念，各自表達 |
| `max_depth` | ✓ | |
| `learning_rate` | ✓ | > 0 |
| `subsample`／`colsample_bytree` | 預設 `1.0` | (0, 1] |
| `early_stopping.patience` | ✓ | `0`＝關閉、長滿 `n_estimators`；正整數＝XGBoost 原生早停 |
| `seed` | ✓ | |

監控指標固定、不開放選擇（分類：驗證集 `logloss`，多分類 `mlogloss`；回歸：`rmse`），送 `early_stopping.monitor` 會 400。分類一律輸出機率（二元 `binary:logistic`、多元 `multi:softprob`），類別數取 Label 的 `n_classes`（驗證集沒出現某類也不會出錯）。

```json
{"id": "x1", "type": "model", "key": "xgboost", "inputs": ["f_ohlcv"], "label": "l_cls",
 "window": 60, "val_ratio": 0.2, "output_mode": "probability",
 "params": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
            "subsample": 1.0, "colsample_bytree": 1.0,
            "early_stopping": {"patience": 0}, "seed": 42}}
```

### 5. 研究操作方針（Reproducer／Experimenter）

本階段新送出的訓練一律**明確帶 `"patience": 0`**（關閉早停），LSTM `epochs: 300`（XGBoost 以 `n_estimators` 表達輪數），跑滿後比較 best／last 兩組結果。早停仍是平台選項；要重新啟用，由 Orchestrator 說明理由並確認後再調整。平台不替研究值代填預設：schema 裡沒有 `default` 的必填欄位都要自己給。

### 6. 送出前自我檢查清單（Agent）

1. `GET /registry/architectures`、`/registry/outcomes`、`/registry/labeling_rules`，以及所選架構 `slots` 裡每個 `component_registry` 的 `GET /registry/components/{name}`，取即時 schema。
2. 由 Label 的 `labeling_rule` 得 `$task_type`、由 outcome 得 `$outcome_unit`。
3. 依 schema 只放**目前可見**的欄位；必填都給值；enum 值取 `enum_cases` 第一個成立條件的清單；元件欄位送 `{"type": 相容元件的 key, ...該元件 params_schema 的欄位}`。
4. `POST /api/model/train`；400 時 `detail` 會列出全部問題（欄位路徑＋原因），修正後重送。

### 具名輸出（Model Node 之間的連接契約）

一個 Model Node 可以有多個輸出，下游用 `inputs` 選要接哪一個。`xgboost` 與單頭 `lstm` 只有 `default`（`output_mode="probability"` 時是 `(n, k)` 機率陣列）；雙頭 `lstm` 有 `default`（＝回歸值）、`regression`、`direction_probability`。

**引用規則：**

| 寫法 | 目標 | 結果 |
|------|------|------|
| `"m1"`（字串） | Feature Node 或 Model Node | 接受，取 `default` |
| `{"node": "m1", "output": "..."}` | Model Node | 接受；`output` 必須是該節點（依實際設定）宣告的輸出名稱 |
| 物件形式 | Feature Node | 拒絕（特徵只有 `default`，請用字串） |
| 字串或物件 | Label Node | 拒絕（Label 只能用 `label` 欄位） |

物件的鍵只允許 `node`、`output`。同一個上游可以被同一個下游用不同輸出引用多次，也可以被不同下游各自引用不同輸出。範例（雙頭 LSTM 的方向機率接給 XGBoost）：`"inputs": [{"node": "m1", "output": "direction_probability"}, "f_ohlcv"]`。

**輸出名稱是通用語意**，不含 outcome 名稱。輸出實際是什麼，看 metadata（`output_specs`）：

```json
{
  "regression": {"kind": "regression", "columns": 1,
                 "target": {"label_node": "l1", "outcome": "log_return", "unit": "log_ratio",
                            "signed": true, "horizon": 1, "labeling_rule": "identity", "task_type": "regression"}},
  "direction_probability": {"kind": "probability", "columns": 1, "derived_from": "regression",
                 "event": {"of": "regression", "op": ">=", "threshold": 0.0, "threshold_unit": "log_ratio"},
                 "classes": ["not_event", "event"],
                 "description": "P(目標 >= 0.0 log_ratio)，即符合設定事件規則的機率"}
}
```

- `kind`：`regression`／`class`／`probability`；`columns` 是輸出欄數，形狀為 `(n, columns)`。
- **方向機率是「符合設定規則的機率」**，`op` 不一定是 `>=`；一律看 `event` 欄位。
- **門檻單位固定等於目標的單位**（`simple_return` 是 `percent`，`log_return` 是 `log_ratio`），不支援單位換算。
- `output_specs` 存進 `job.result[node_id]`，也隨模型產物保存（`model_config.output_specs`），推論載入時據此驗證下游引用。

**提交前驗證（所有建立／繼承圖的入口共用）**：`POST /train`（含 Phase 2 繼承後）與 worker 撿到任務時（讀資料之前）都會驗證，失敗時 API 回 400、**不建立 job**，訊息一次列出全部問題：節點或輸出不存在、物件引用 Feature、`inputs` 引用 Label、自我引用、成環、architecture／outcome／labeling_rule 未登記或組合不合法、模型參數不符 schema（`Model 節點 m1 的模型設定不合法：params.…`）。Label 的 `horizon` 必須是 ≥1 的整數、`n_classes` ≥2 的整數、`threshold_pct` 有限數值；Model 的 `window` ≥1 的整數、`val_ratio` 在 (0, 1)、`output_mode` 必須是字串。

**已知限制**：具名輸出只解決「下游選哪個輸出」，**不代表多模型串接已符合研究驗證要求**——上游輸出仍是對自己訓練過的資料的樣本內預測（非 OOF，見文末）。

### 日線資料與時間語意

- **日線 timeframe**：`tx_daily_day`（日盤）、`tx_daily_full`（日夜全盤）；對應資料表 `market.future_taifex_tx_daily_day`／`..._daily_full`。`timeframe` 允許以底線連接的英數字（`daily_day`），`symbol` 本身不含底線；空白、引號、點、分號、連字號等一律 400（表名是拼進 SQL 的，白名單不放寬）。訓練、推論、行情查詢共用同一個驗證。
- **日夜盤選擇、換月（連續月）調整、研究期間都是研究決定，不是平台預設**：平台只讀資料表現有內容，不做任何調整（實測資料 186 次換月未調整，換月日的報酬絕對值明顯偏大，要不要處理由研究端決定）。
- **兩個時間不能混用**：
  - `datetime`＝K 棒的**識別時間**（棒起點；日線固定 08:45）——對齊、切分、樣本 `ts` 一律用它。
  - `bar_end_ts`＝**資訊可用時間**（日線是 13:45）——完整日棒在此之前並不知道。
  - 因此樣本另外帶 `available_ts`：**決策可用時間＝全部輸入（Feature 與上游 Model，沿依賴鏈傳遞）在樣本最後一根棒的可用時間的最晚者**；**目標可用時間（`target_available_ts`）取自 Label 節點自己資料的 `bar_end_ts`**。預覽的 `decision_available_ts`／`target_available_ts`、推論結果的 `available_ts` 都是這個語意。任何一個輸入未知就整個是 `null`。這只是標時間，**不改變任何計算與切分**。

## 讀取訓練結果

`GET /api/model/jobs/{id}`（與 `GET /api/model/jobs` 清單）回傳的訓練任務，每個模型節點都有兩份通用資料，**所有架構同一種形狀**：

- `result[node_id].evaluations`：評估清單（完成後才有）
- `result[node_id].metric_specs`，以及任務最上層的 `metric_specs[node_id]`（進行中也有，依提交設定算）：這個節點會產生哪些逐輪序列、哪些評估指標、每個指標的意義

```json
{
  "job_id": "…", "job_type": "train", "phase": 1, "status": "done",
  "metric_specs": {"m1": { "...": "見下方 metric_specs" }},
  "result": {"m1": {
    "final_metrics": {"val_loss": 0.68, "val_accuracy": 0.56, "last": {"val_loss": 0.69, "val_accuracy": 0.54},
                      "monitor": "val_loss", "monitor_mode": "min", "patience": 0, "min_delta": 0.0,
                      "best_epoch": 41, "stopped_epoch": 299, "epochs_run": 300, "stopped_early": false},
    "evaluations": [ { "...": "見下方 evaluations" } ],
    "metric_specs": { "...": "同最上層 metric_specs[m1]" },
    "output_specs": {"default": {"...": "..."}},
    "training_meta": {"split_strategy": "random", "n_train": 2640, "n_val": 660, "n_excluded_boundary": 0,
                      "n_samples": 3300, "val_ratio": 0.2, "split_seed": 42},
    "device": "cuda"
  }}
}
```

- `final_metrics`：訓練迴圈裡的 best（監控指標最好那一輪，索引由 0 起算）驗證指標＋`last`＋訓練控制資訊（LSTM：`best_epoch`／`epochs_run`；XGBoost：`best_round`／`rounds_run`）。`patience=0` 時 `stopped_early` 一定是 `false`。
- **best／last 兩組權重都保存**（`model_artifacts.weights`＝best、`weights_last`＝last），也都各自完整評估（`evaluations`）。
- 產物的 `model_config`（推論時載入）記錄這次實際的設定：LSTM 有 `config`（解析後完整 params，含預設值與優化器）、`mode`、`n_classes`、`target_scaling`、`sequence_summary`、`attention_meta`、`output_specs`；XGBoost 有 `config`、`is_cls`、`n_classes`、`best_round`、`rounds_run`。

### `evaluations`：評估清單

每筆紀錄是「某個模型節點的某個輸出頭，在某個資料集、某個評估時點」的完整評估，比較條件寫在紀錄裡：

```json
{
  "id": "m1/default/val/best",
  "model_node": "m1", "head": "default", "head_label": "輸出", "task": "classification",
  "dataset": {
    "split": "val", "label": "驗證集", "selection": "random", "val_ratio": 0.2,
    "n_samples": 660, "n_train": 2640, "n_total": 3300, "n_excluded_boundary": 0,
    "phase": 1, "task_data_range": {"start": "2015-01-05", "end": "2024-12-31"},
    "description": "任務資料範圍內的 3300 個樣本隨機打亂（固定種子 42），抽 20%（660 筆）為驗證集、其餘 2640 筆訓練；驗證樣本分散在整個期間，不是一段連續時間"
  },
  "point": {"kind": "best", "round": 41, "round_unit": "epoch", "weights": "best", "label": "最佳（第 42 Epoch）",
            "selected_by": {"monitor": "val_loss", "mode": "min", "patience": 0}},
  "metrics": {
    "accuracy": 0.5591, "balanced_accuracy": 0.5427, "macro_f1": 0.5404, "roc_auc": 0.5712, "average_precision": 0.5895,
    "per_class": {"0": {"precision": 0.4979, "recall": 0.4069, "f1": 0.4478, "support": 290},
                  "1": {"precision": 0.5934, "recall": 0.6784, "f1": 0.6331, "support": 370}},
    "class_distribution": {"0": 290, "1": 370},
    "confusion_matrix": {"labels": [0, 1], "values": [[118, 172], [119, 251]], "row_axis": "實際", "col_axis": "預測"}
  },
  "unavailable": {},
  "baselines": [{"key": "majority_class", "label": "訓練集多數類別", "available": true,
                 "method": "predict_majority_class_from_train",
                 "metrics": {"accuracy": 0.5606, "balanced_accuracy": 0.5, "macro_f1": 0.3592}, "detail": {"class": 1}}]
}
```

| 欄位 | 說明 |
|---|---|
| `id` | `模型節點/輸出頭/資料集/評估時點`，同一任務內唯一 |
| `head` | `default`（單頭）；雙頭拆成 `regression` 與 `direction` 兩筆 |
| `dataset.split` | 目前只有 `val`（驗證集）；預留 `holdout`（Phase 3）、`train` |
| `dataset.selection` | `random`（Phase 1：任務資料範圍內所有樣本以固定種子 42 打亂後抽 `val_ratio`，驗證樣本**分散在整個期間**，不是一段時間）或 `chronological`（Phase 2：依時間排序取最後 `val_ratio`，並排除標籤跨越驗證起點的訓練樣本 `n_excluded_boundary`） |
| `dataset.task_data_range` | 這個任務 graph_spec 的 `start`／`end`（**任務資料範圍**，不是驗證集的時間範圍） |
| `dataset.n_samples`／`n_train`／`n_total` | 驗證集樣本數／訓練樣本數／切分前樣本總數 |
| `point.kind` | `best`／`last`；預留 `checkpoint`。`round` 由 0 起算，`label` 是顯示用（由 1 起算）；`weights` 對應 `/infer` 的 `use_weights` |
| `point.selected_by` | best 的挑選依據（監控指標、方向、patience） |
| `metrics` | 鍵＝指標登記表的 key（報告裡凡是已登記的指標都納入，沒有固定清單）；值的形狀依定義的 `shape`：`scalar` 數值（或 `null`）、`per_class` 為 `{類別: {欄位: 值}}`（欄位由資料決定）、`distribution` 為 `{類別: 數值}`、`matrix` 為 `{labels, values, row_axis, col_axis}`（軸名稱來自定義的 `axes`）。衍生指標依定義的 `derive` 推算，例如 `accuracy` 由混淆矩陣對角線／總數 |
| `unavailable` | 算不出的指標與原因（該指標在 `metrics` 裡是 `null`），例如驗證集缺類別時的多分類 ROC-AUC、多分類 AP |
| `baselines` | 適用的簡單基準：分類 `majority_class`（訓練集多數類別）；回歸 `train_mean`（預測訓練集平均）、`zero`（預測零）。`metrics` 只含已登記的指標，其他欄位（例如多數類別是哪一類）在 `detail`；`available=false` 時附 `reason` |

回歸頭的 `metrics` 是 `{"mse", "rmse", "mae", "r2"}`（`mse` 在目標原單位²，`rmse`／`mae` 在目標原單位，單位見 `metric_specs.heads[].target_unit`）。**`average_precision`（AP）不是梯形積分的 PR-AUC。** 數值不會出現 NaN（JSONB 一律存 `null`）。

### `metric_specs`：指標描述（模型引用的共用指標定義）

```json
{
  "round_unit": "epoch",
  "heads": [{"key": "default", "label": "輸出", "task": "classification"}],
  "series": [
    {"id": "loss", "metric": "cross_entropy", "head": "default", "train": "loss", "val": "val_loss"},
    {"id": "accuracy", "metric": "accuracy", "head": "default", "train": "accuracy", "val": "val_accuracy"}
  ],
  "evaluation": {"default": ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "average_precision",
                             "per_class", "class_distribution", "confusion_matrix"]},
  "definitions": {"cross_entropy": {"key": "cross_entropy", "label": "Cross-entropy", "shape": "scalar",
                                    "direction": "min", "unit": "loss_space", "format": "decimal", "description": "…"}}
}
```

- `round_unit`：`epoch`（LSTM）或 `boosting_round`（XGBoost）。
- `series`：逐輪序列。`train`／`val` 是逐輪紀錄（`/progress`、WebSocket）裡的欄位名——固定四欄 `loss`／`accuracy`／`val_loss`／`val_accuracy` 在最上層，其他在 `metrics` 物件裡。`unit` 有值時覆寫定義的單位（例：LSTM 回歸的訓練 MSE 在損失空間）。
- 同一個 `loss` 欄位在不同模型意義不同，一律看 `series[].metric`：LSTM 分類 `cross_entropy`、LSTM 回歸 `mse`（損失空間）、XGBoost 分類 `logloss`／`mlogloss`、XGBoost 回歸 `rmse`；雙頭 LSTM 的序列是 `joint_loss`（聯合）、`rmse`／`mse`（回歸頭）、`dir_acc`／`bce`（方向頭）。
- `definitions`：引用指標的定義——`direction`（`min` 越低越好／`max` 越高越好）、`unit`（`loss_space`／`target`／`target_squared`／`ratio`／`count`）、`shape`、`format`（`percent`／`decimal`／`int`）；矩陣另有 `axes`（`{row, col}`），衍生指標另有 `derive`（`{from, method}`）。全部指標：`GET /api/model/registry/metrics`。
- 解析時依 `definitions[key].shape` 處理 `metrics[key]`，不要依指標名稱寫死；新增的同形態指標會照同一契約出現。

### 從 `evaluation.best／last` 改讀 `evaluations`（HA 解析腳本更新）

**2026-09-30 起新任務不再保存 `evaluation`，只有 `evaluations`。** 舊紀錄的 `evaluation` 原樣保留並照常回傳（資料庫未改寫），同時由 API 讀取時轉換出 `evaluations`；所以**一律改讀 `evaluations`**，新舊紀錄都適用。

| 舊（`result[node].evaluation`） | 新（`result[node].evaluations[]`） |
|---|---|
| `evaluation.best`／`evaluation.last` | `point.kind == "best"`／`"last"` 的紀錄 |
| 雙頭 `evaluation.best.direction`／`.regression` | `head == "direction"`／`"regression"` 的兩筆紀錄 |
| 單頭報告本身 | `head == "default"` 的紀錄 |
| `report.macro_f1` 等數值 | `record.metrics.macro_f1`（鍵名不變；多了 `accuracy`） |
| `report.confusion_matrix` ＋ `confusion_matrix_labels` | `record.metrics.confusion_matrix.values` ＋ `.labels` |
| `report.per_class`／`class_distribution` | `record.metrics.per_class`／`class_distribution` |
| `report.roc_auc_unavailable_reason` 等 | `record.unavailable.roc_auc` |
| `report.baseline_majority_class` | `record.baselines` 中 `key == "majority_class"`（數值在 `.metrics`，類別在 `.detail.class`） |
| `report.baseline_mean`／`baseline_zero` | `record.baselines` 中 `key == "train_mean"`／`"zero"` |
| `report.baseline_*_unavailable_reason` | 該基準 `available == false` 的 `reason` |
| （沒有） | `dataset`（資料集、切分、樣本數、Phase、任務資料範圍）、`point.round`／`selected_by` |

```python
def pick(result_node, head="default", kind="best", split="val"):
    for rec in result_node["evaluations"]:
        if rec["head"] == head and rec["point"]["kind"] == kind and rec["dataset"]["split"] == split:
            return rec
    return None

# 舊：r["evaluation"]["best"]["macro_f1"]
rec = pick(r, "default", "best")
macro_f1 = rec["metrics"]["macro_f1"]
# 舊：r["evaluation"]["best"]["direction"]["roc_auc"]（雙頭）
roc = pick(r, "direction", "best")["metrics"]["roc_auc"]
# 舊：r["evaluation"]["last"]["regression"]["baseline_mean"]["rmse"]
b = next(x for x in pick(r, "regression", "last")["baselines"] if x["key"] == "train_mean")
baseline_rmse = b["metrics"]["rmse"]
# 舊：r["evaluation"]["best"]["confusion_matrix"]
cm = pick(r, "default", "best")["metrics"]["confusion_matrix"]["values"]
```

### 逐輪指標

`GET .../progress` 與 WebSocket 進度訊息是 `{node_id, epoch, loss, accuracy, val_loss, val_accuracy, metrics}`；要畫哪些序列、各欄位意義看 `metric_specs.series`。`metrics` 是擴充指標：單頭回歸 `rmse`／`val_rmse`／`val_mae`；雙頭 `joint_loss`／`mse`／`bce`／`rmse`／`dir_acc` 與各自的 `val_` 版本；單頭分類與 XGBoost 為 `null`。

**推論（雙頭）**：`/inference_predictions` 逐列 `output_type="dual"`，`predicted` 是回歸值（目標原單位）、`probabilities`＝`[P(不符合事件), P(符合事件)]`。

```json
{"id": 7, "ts": 1725148800, "available_ts": 1725169500, "output_type": "dual",
 "predicted": 0.0031, "probabilities": [0.42, 0.58]}
```

**常見 400 訊息**（一次列出全部）：`params.shared.dense 為必填（沒有預設值）`、`params.lstm_layers[1].units 為必填（沒有預設值）`、`params.early_stopping.monitor 必須是 [...] 之一`、`params.heads 必須是 ['single'] 之一，收到 'dual'（…分類 Label 只能 single）`、`params.class_weight 在目前的設定下不適用…請移除`、`params.attention：本版本只支援單一元件（零或一個），不支援多元件組合`、`params.xxx 不是這個模型認得的欄位`、`params.direction_rule：…`（`threshold_unit` 與目標單位不同）、下游引用了不存在的輸出名稱。

## `GET /api/model/jobs?limit=20`

列出最近的任務（新到舊）。

## `GET /api/model/jobs/{job_id}`

```json
{
  "job_id": "...", "graph_spec": { ... }, "status": "done", "error": null,
  "metric_specs": { "model1": {...} },
  "result": { "model1": { "final_metrics": {...}, "evaluations": [...], "metric_specs": {...}, "output_specs": {...},
                          "device": "cuda", "training_meta": {...} } },
  "device": "cuda", "job_type": "train", "phase": 1, "parent_job_id": null, "created_at": "..."
}
```

`status`：`pending`／`running`／`done`／`failed`（看 `error`）。worker 同時間只跑一筆；`pending` 表示還在排隊，或 GPU 實際可用顯存不足預留量（`TRAINING_GPU_MIN_FREE_MB`，預設 4096MiB，已先釋放 worker 自己的快取才判斷）而每 30 秒重試——延後原因（total／used／free、worker 自身佔用、門檻）記在 training 容器 log（`docker logs hermesnote-training`），API 不回傳。`result` 的 key 是每個 Model Node 的 `id`；`job_type='infer'` 的 job，`result[node_id]` 改成 `{"count": N, "start_ts": unix秒, "end_ts": unix秒}`（推論結果本身不在這裡，見下方 `/inference_predictions`）。

## `DELETE /api/model/jobs/{job_id}`

刪掉一筆任務（含進度、已存的模型權重，DB CASCADE）。如果有 Phase 2/3 的後續任務指著它，回 409，要求先刪後續那筆。主要給孤兒 job（送出後容器崩潰、狀態卡死）清理用，不能中斷真的還在跑的訓練（見下面 `/cancel`）。

## `POST /api/model/jobs/{job_id}/cancel`

把一筆 `pending`/`running` 的任務標記成 `failed`（人工中斷）。**只改 DB 狀態，不會真的砍掉背景 process**——worker 是單一長駐行程、同時間只跑一個 job，沒有子行程可以單獨終止，真的要中斷運算需要重啟整個 training 容器（NAS 端操作，這支 API 沒有權限做）。呼叫這支的目的只是讓歷史紀錄立刻反映「已放棄」，不用等重啟才發現它其實已經死了。

## `GET /api/model/jobs/{job_id}/progress`

給初次載入頁面補歷史用（陣列，每輪一筆 `{node_id, epoch, loss, accuracy, val_loss, val_accuracy, metrics, created_at}`，`metrics` 見上方「讀取訓練結果」）；訓練中的新進度建議走 WebSocket，不要每輪都輪詢這支。

## `GET /api/model/jobs/{job_id}/stream`（WebSocket）

訓練中即時推送兩種訊息：
- 進度：`{node_id, epoch, loss, accuracy, val_loss, val_accuracy, metrics}`
- 即時抽樣預覽：`{type: "preview", node_id, epoch, source: "train"|"val", decision_ts, target_ts, decision_available_ts, target_available_ts, horizon, task_type, n_classes, labeling_rule, bars: [...], actual, predicted}`——訓練途中隨機抽某個正在處理的 window 展示用，`bars` 是真實 OHLC，**不是模型的正式推論結果**，只用來人工肉眼核對訓練有沒有在正常學習。

## `GET /api/model/jobs/{job_id}/preview_samples?node_id=...&limit=50&before_id=...`

完成後回頭查訓練途中記錄過的抽樣預覽（依 `id` 游標分頁，新到舊）。回傳格式跟上面 WebSocket 的 `preview` 訊息一樣（多一個 `id` 欄位）。**這是訓練時的抽樣展示，不是模型對全期間資料的正式推論**——正式推論結果一律走下面的 `/infer` + `/inference_predictions`。

## `POST /api/model/infer`

用已經訓練好、存在 `model_artifacts` 的模型，對一段新資料做推論（不重新訓練）。

**請求 body：**

```json
{
  "target_job_id": "3e4d15c1-3e30-4bf8-a553-09a60c506248",
  "target_node_id": "model1",
  "start": "2026-09-01",
  "end": "2026-09-08",
  "phase": null,
  "use_weights": "best"
}
```

| 欄位 | 說明 |
|------|------|
| `target_job_id` | 要載入哪個訓練 job 的模型 |
| `target_node_id` | 那個 job 裡的哪個 Model Node；不給就自動找最終輸出節點（沒有下游節點引用的那個），找不到（多個候選）會回 400 要求明確指定 |
| `start` / `end` | 推論用的新資料區間，`YYYY-MM-DD` |
| `phase` | `null`（一般推論）或 `3`（Phase 3 holdout，`target_job_id` 必須是一筆已完成的 Phase 2 job，且 `start` 必須晚於該 Phase 1/2 訓練期間的 `end`，不然會回 400——holdout 不能跟訓練資料重疊） |
| `use_weights` | `"best"`（預設）或 `"last"`：載入哪一組權重（見「讀取訓練結果」）；infer job 的 `graph_spec` 與 `result` 都會記錄這個值 |

**回應：** `{ "job_id": "..." }`——一樣是非同步 job，用 `GET /api/model/jobs/{job_id}` 輪詢，`status='done'` 後結果已經分批存進 DB，用下面這支查。

## `GET /api/model/jobs/{job_id}/inference_predictions?node_id=...&start=...&end=...&cursor=...&limit=1000`

`job_type='infer'` 的 job 完成後，逐列推論結果依時間區間（`start`/`end`，unix 秒，可省略）＋ `id` 游標分頁查（`limit` 上限 5000，全期間資料本來就該分批查，不是一次全撈）。

```json
[
  { "id": 1, "ts": 1725148800, "output_type": "probability", "predicted": 1, "probabilities": [0.12, 0.88] },
  { "id": 2, "ts": 1725152400, "output_type": "class", "predicted": 0, "probabilities": null }
]
```

`output_type` 決定怎麼解讀：
- `"class"` / `"regression"`：`predicted` 是唯一數值，`probabilities` 是 `null`
- `"probability"`：`predicted` 是 argmax 類別索引（方便排序/篩選），`probabilities` 是完整機率向量——**原始機率沒有被丟掉**，之後要做機率門檻策略可以直接用這欄，不用只看 `predicted`

## `GET /api/model/artifacts?job_id=...`

列出已保存的模型產物（不給 `job_id` 列全部）。`POST /api/model/artifacts/{job_id}/{node_id}/keep` body `{"kept": true}` 只是標記/顯示用（收藏），不影響能不能被 `/infer` 載入——沒標記的一樣可以推論。

## 用同樣設定重跑（沒有專用端點）

沒有「重跑」端點，重跑＝把該筆紀錄的設定原樣重新送出一次，會新增一筆 job，原紀錄不動。後台歷史紀錄的「重跑」按鈕做的就是下面這件事：

| 原紀錄 | 重送什麼 |
|---|---|
| Phase 1／沒標 Phase 的訓練 | `GET /jobs/{id}` 取回 `graph_spec`，以其 `start`／`end`／`nodes` 加上原本的 `phase`（`1` 或 `null`）`POST /train` |
| Phase 2 | `POST /train`，`phase=2`、`parent_job_id` 用**同一個** Phase 1 母任務（`start`／`end`／`nodes` 仍是佔位值，後端會完整繼承母任務的設定並覆寫成時間切分） |
| Phase 3（或一般推論） | 該筆 infer job 的 `graph_spec` 內有 `target_job_id`、`target_node_id`、`start`、`end`、`use_weights`，原樣 `POST /infer`，Phase 3 仍帶 `phase=3` |

- 資料範圍、特徵、Label、模型參數、切分方式全部不變；要換時間框架或特徵，就是另一個實驗，請用新的 payload 當新的 Phase 1 送出。
- `seed` 在 payload 裡，隨設定一起重送；Phase 1 的隨機切分種子固定。CPU 上同設定重跑的指標一致（有測試）；GPU 上不保證逐位相同，只保證設定相同。
- Phase 2 要求母任務仍存在且已完成；母任務被刪除就不能重跑 Phase 2。Phase 3 要求來源 Phase 2 的模型產物仍在，且 `start` 仍須晚於該 Phase 1/2 訓練期間的結束日（不合會回 400）。

## 已知限制（截至這份文件更新時的程式碼狀態）

- **正規化**已接（train-only fit 的 MinMax），但**跨時間框架融合**（例如 15m 模型接 1h 模型）還沒做重採樣對齊，時間戳交集在兩邊尺度完全不同時會直接交集出空集合報錯。
- 上游 Model Node 接下游時，上游對自己訓練用的「全範圍」窗格是用同一個已 fit 過的模型直接預測（不是 OOF/out-of-fold），下游等於看到上游對 train 段的過擬合結果——這是已知的資料洩漏風險，尚未處理；具名輸出不改變這一點。
- 推論時欄位順序依 `inputs` 的原始順序（與訓練相同）。**具名輸出修正部署之前的推論實作是「先全部 Feature、再接上游模型」**，只有 `inputs` 剛好是這個排列時才跟訓練對得上；如果有把模型排在特徵前面或夾在中間的舊產物，推論結果可能錯位，需重新訓練。管理後台產生的圖一律特徵在前，不受影響。
- 進階 `labeling_rule`（波動度門檻、分位數切分、成本導向、triple barrier）還沒做，目前只有 `fixed_threshold`（分類）跟 `identity`（回歸）。
- `/cancel` 只改 DB 狀態，無法真的中斷背景運算中的訓練。
- **混合模型目前僅支援節點輸出串接**：上游模型節點凍結後的輸出，當作下游模型節點的一個輸入特徵，兩個節點各自獨立訓練，沒有梯度互通。**不支援**聯合訓練（end-to-end 跨節點反向傳播）、集成投票／加權融合這類結構、共享編碼器的多任務結構——這些都還沒實作，不要從「支援具名輸出引用」推論出更廣的能力。
- **Attention 本版本只支援零或一個**，不支援多個 Attention 的串接或並聯；送清單一律 400，不是靜默忽略（未來擴充點見 [extending.md](extending.md)）。
- 後台畫布上「模組→模組」的連線只接上游的 `default` 輸出；要接具名輸出（例如雙頭的 `direction_probability`）目前要直接用 API 送 `{"node", "output"}` 引用。
- 沒有「只驗證不送出」的端點：提交驗證失敗的 400 不會建立 job，可直接當驗證用。

以上限制以程式碼為準，`GET /api/model/registry/*` 的即時清單跟本文件的落差，一律以即時清單為準。
