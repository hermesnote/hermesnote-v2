# 模型訓練／推論 API

> 對應原始碼：`backend/routers/model_training.py`（送出/查詢）＋ `backend/training/graph.py`（graph_spec 節點格式的權威定義）。連線位址見 `../index.md`。
>
> 這裡只保證**格式跟現在程式碼一致**，不代表已經做到論文等級的嚴謹（見文末「已知限制」）。新增模型／元件的方法見 [extending.md](extending.md)。

## 版本與變更（MCP 同步用）

每筆任務都記錄 `pipeline_version`（流程版本）。會改變結果的修正都會升一版；**不同流程版本的結果不互相比較**。舊任務（`pipeline_version: null`）保留、不重算。

| 流程版本 | 部署日 | 對 API 呼叫端的變化 |
|---|---|---|
| `null`（舊） | 2026-11 以前 | — |
| `2026-11-r1`（M1） | 2026-11-15 | ① `POST /train`、`POST /infer` 回應多 `warnings`；② `GET /jobs`、`GET /jobs/{id}` 多 `pipeline_version`、`warnings`，單筆另有 `data_report`、`env_fingerprint`；③ 跨時間框架的圖回 400（M4 前暫不開放）；④ 讀到重複 K 棒時任務 `failed`（不再靜默去重）；⑤ 分鐘資料讀取排除 `session='other'` |

| `2026-11-r2`（M2～M7） | 2026-11-16 | ⓪ **寫入端點需要認證**（401／403，見「認證」）；① 新圖格式 `mode: "v2"`（樣本框架、往前取暖機、轉換元件、跨時間框架 as-of、多分支、串接 OOF），見「v2 圖與實驗層」；**舊格式的圖行為與 r1 逐位元組相同**，但帶 v2 專用欄位（`settings`、`warmup`、`transform`、`branches`、`upstream_policy`、`normalization`）會 400；② **Phase 3（`POST /infer` + `phase: 3`）改成 best、last 都推論並評估**，`result[node]` 多 `evaluations`（`dataset.split="holdout"`）、`metric_specs`、`holdout`；③ 新端點：實驗層 `/api/model/experiments…`、逐筆結果 `/jobs/{id}/records`、`/jobs/{id}/evaluations`、`/jobs/{id}/holdout_usage`、`/jobs/{id}/upstream_folds`、`/jobs/{id}/cascade_audit`、`/warmup/suggest`、`/registry/v2`；④ `GET /jobs`、`/jobs/{id}` 多 `experiment_id`、`arm_id`、`frame_id`、`model_seed`、`split_seed`、`gate_check`、`requested_by`（舊任務皆 `null`）；⑤ 新訓練的 `result[node]` 多 `train_stats`（v2 另有 `training_meta.frame／exclusions／warmup／branches／cascade`）；⑥ 舊格式的 Phase 1→2→3 血緣規則不變 |

M2～M7 以同一個流程版本 `2026-11-r2` 交付（一次遷移、一次部署）。**MCP 端請以這張表判斷要同步哪些欄位**；欄位的即時定義仍以 `GET /api/model/registry/*` 與實際回應為準。

### 認證（2026-11-16 起，寫入端點必須帶）

`/api/model/*` 的**寫入端點**只接受下面兩種身分其一；**讀取（GET）與 WebSocket 不需要認證**。驗證失敗時在建立任何資料之前就回錯，不會產生任務。

| 身分 | 標頭 | 給誰 | 權限範圍 |
|---|---|---|---|
| 服務金鑰 | `X-Hermesnote-Key: hnk_…` | MCP 等程式 | 依金鑰設定（見下表），**只給需要的範圍** |
| 管理員 | `Authorization: Bearer <session JWT>` | 後台網頁（Google 登入＋email 白名單，12 小時到期） | 全部 |

| 權限範圍 | 端點 |
|---|---|
| `model:train` | `POST /api/model/train` |
| `model:infer` | `POST /api/model/infer` |
| `model:manage` | `DELETE /api/model/jobs/{id}`、`POST /api/model/jobs/{id}/cancel`、`POST /api/model/artifacts/{job}/{node}/keep` |
| `model:experiment` | `POST /experiments`、`PATCH /experiments/{id}/settings`、`POST /experiments/{id}/arms`、`POST /experiments/{id}/frames`、`POST /experiments/{id}/arms/{arm}/run`、`POST /warmup/suggest`、`POST /registry/v2/feature_memory` |

| 情況 | 回應 |
|---|---|
| 兩種都沒帶、金鑰錯、JWT 偽造／過期／格式錯 | **401**（`detail` 說明，不會建立任務） |
| 帶了金鑰就只看金鑰：金鑰錯時**不會**改用同時帶的 JWT | 401 |
| 金鑰正確但沒有該權限範圍 | **403** `服務金鑰 <名稱> 沒有 <範圍> 權限` |
| JWT 有效但 email 不在白名單 | 403 |

- 通過後後端記錄身分：任務 `requested_by`（`key:<金鑰名稱>`／`admin:<email>`，舊任務為 `null`，`GET /jobs`、`/jobs/{id}` 回傳）、實驗 `created_by`、設定修改紀錄 `settings.history[].by`、holdout 使用紀錄 `detail.requested_by`。
- 門檻放行的**確認人由後端依驗證身分記錄**（`gate_check.override.confirmed_by`）；請求裡的 `gate_override.confirmed_by` 改為選填，只當備註存在 `note_confirmed_by`。
- 金鑰設定（網站後端 `.env`）：`MODEL_API_KEYS=[{"name": "mcp-training", "sha256": "<金鑰明文的 SHA-256 hex>", "scopes": ["model:train"]}]`。後端只存雜湊；格式錯的項目會被略過並記錄錯誤（不印出內容）。輪替：加入新金鑰 → 呼叫端換新 → 移除舊項目，每次修改需重啟網站後端容器。
- **MCP 需要的變更**：所有會呼叫上述寫入端點的工具（目前為 `submit_training` → `POST /train`）在 HTTP 請求加 `X-Hermesnote-Key`；金鑰放 MCP 執行環境，不寫進程式、log 或回應。401／403 應原樣回報給呼叫者，不要重送（`request_id` 防重送規則不變）。讀取工具不用改。

### 錯誤與失敗（呼叫端怎麼判斷）

| 情況 | 形式 | 內容 |
|---|---|---|
| 圖不合法（含跨時間框架、參數、引用、成環） | HTTP 400，**不建立任務** | `detail`：全部問題以「；」串成一行 |
| 請求格式不對（缺欄位、型別錯） | HTTP 422 | FastAPI 標準驗證錯誤 |
| Phase 血緣不合法（母任務不存在／不是對應 Phase／未完成；Phase 3 期間與訓練重疊） | HTTP 400 | `detail` 說明原因 |
| 推論找不到任務或模型產物 | HTTP 404 | — |
| 刪除仍有後續 Phase 任務的紀錄 | HTTP 409 | — |
| 期間超過研究資料上限（迄日晚於 `2026-01-01`） | HTTP 200，**任務照常建立** | 回應 `warnings: [{"code": "period_after_research_end", "message": "..."}]`，同時寫進任務紀錄 |
| 該期間查無資料 | 任務 `status: "failed"` | `error`：`這段期間查無資料` |
| 行情有重複 K 棒 | 任務 `status: "failed"` | `error`：`market.future_taifex_tx_… 在 … 有重複 K 棒（已排除 session='other'）：bar_start_ts=… 出現 2 次；…`（最多列 20 筆）。研究期間（2026 年以前）實測沒有重複；2026 年資料有已知的重複夜盤 |
| 驗證通過但執行時失敗（例如 OOM 以外的例外） | 任務 `status: "failed"` | `error` 是例外訊息 |
| v2 圖不合法（暖機、轉換、分支、上游策略、settings） | HTTP 400，不建立任務 | 例：`Feature 節點 o：累積型指標必須選擇 transform（…）`、`Feature 節點 a：… 請設定 warmup.bars`、`Model 節點 m 分支 hour：… 必須在 aggregate 指定彙總方式` |
| 舊格式的圖帶 v2 專用欄位 | HTTP 400 | `節點 r：warmup 只在 mode="v2" 的圖有效；…` |
| 寫入端點沒帶憑證／憑證無效 | HTTP **401**，不建立任務 | 見上方「認證」 |
| 服務金鑰沒有該權限範圍 | HTTP **403**，不建立任務 | `服務金鑰 <名稱> 沒有 <範圍> 權限` |
| 實驗：門檻未通過 | HTTP **409**，不建立任務 | `detail: {"message": "...", "gate_check": {...}}`；附 `gate_override: {"reason": "..."}` 重送即建立，放行紀錄寫進任務 `gate_check.override`（確認人＝驗證身分） |
| 實驗：修改建立後不可改的設定（期間、Label、切分…） | HTTP 400 | `detail` 列出可修改的鍵 |
| 實驗：配對比較的兩個任務流程版本／框架版本／對齊設定不同 | HTTP 400 | 「系統拒絕跨版本比較」 |
| v2：框架內可用的訓練點或驗證點為 0 | 任務 `failed` | `error`：`節點 m：框架內可用的訓練點 0、驗證點 …（排除原因：{…}）` |
| v2 實驗：保存的框架雜湊與依資料重建的不同 | 任務 `failed` | `error`：`樣本框架 … 的雜湊與這次依資料重建的不同 … 請建立新版本框架` |
| v2：排除比例超過門檻（預設 5%） | 任務 `done` | `warnings` 多 `{"code": "exclusion_ratio_high"}`；另有 `upstream_in_sample`（上游用舊行為）、`cumulative_load_dependent`（legacy_cumulative） |

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

**回應（`2026-11-r1` 起）：**

```json
{ "job_id": "...", "warnings": [] }
```

期間迄日晚於 `2026-01-01` 時（只提示、任務照常建立；Phase 2 依繼承來的期間判斷）：

```json
{ "job_id": "...", "warnings": [{ "code": "period_after_research_end",
  "message": "期間迄日 2026-03-01 超過研究資料上限 2026-01-01：2026 年起的行情來自新匯入管線，重複夜盤、1～4 月缺口與時段定義尚未核對，結果請另外標示。" }] }
```

**跨時間框架（`2026-11-r1` 起暫不開放，M4 完成後開放）**：同一個 Model Node 的 Feature 輸入、上游模型鏈與 Label 只要用到兩個以上 `timeframe`，回 400：
`Model 節點 m：輸入與 Label 用到多個時間框架 ['tx_15m', 'tx_60m']；跨時間框架對齊（依 bar_start_ts 對齊、依 bar_end_ts 判定可用）完成前暫不開放`。

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

列出最近的任務（新到舊）。每筆含 `pipeline_version`、`warnings`（不含 `data_report`／`env_fingerprint` 全文，要看請查單筆）。

## `GET /api/model/job-summaries?limit=1000`

輕量任務清單（2026-11-18 起；訓練頁下拉選單用）：`[{job_id, job_type, status, created_at}]`，新到舊，`limit` 1～5000（預設 1000）。不含圖、結果與資料紀錄，一百多筆也只有十幾 KB；要完整內容請查單筆。

## `GET /api/model/jobs/{job_id}`

```json
{
  "job_id": "...", "graph_spec": { ... }, "status": "done", "error": null,
  "metric_specs": { "model1": {...} },
  "result": { "model1": { "final_metrics": {...}, "evaluations": [...], "metric_specs": {...}, "output_specs": {...},
                          "device": "cuda", "training_meta": {...} } },
  "device": "cuda", "job_type": "train", "phase": 1, "parent_job_id": null, "created_at": "...",
  "started_at": "...", "finished_at": "...",
  "pipeline_version": "2026-11-r1",
  "warnings": [],
  "env_fingerprint": { "pipeline_version": "2026-11-r1", "python": "3.12.x", "numpy": "...", "pandas": "...",
                       "torch": "2.10.0+cu128", "cuda": "12.8", "gpu": "NVIDIA GeForce RTX 5060 Ti",
                       "xgboost": "3.2.0", "sklearn": "1.9.0", "talib": "0.7.1", "talib_c": "0.7.1 (...)" },
  "data_report": {
    "pipeline_version": "2026-11-r1",
    "period": { "start": "2011-01-03", "end": "2025-01-01" },
    "research_data_end": "2026-01-01",
    "warnings": [],
    "available_time": { "source": "bar_end_ts", "publication_delay_sec": 0, "delay_status": "assumed" },
    "timeframes": {
      "tx_1m": { "table": "market.future_taifex_tx_1m", "start": "2011-01-03", "end": "2025-01-01",
                 "n_rows": 2544388, "order_key": "bar_start_ts",
                 "excluded_sessions": ["other"], "n_excluded": 5,
                 "excluded_trading_days": ["2012-03-03", "2012-12-22", "2014-12-27", "2016-01-30", "2016-09-10"],
                 "sessions": { "day": 1026183, "night": 1518205 },
                 "rule_versions": { "taifex_front_outright_v1_2026-04-30": 2537283,
                                    "taifex_front_outright_v1_tick_precise_2026-08-09": 7105 },
                 "n_gaps": 33262, "n_missing_bars": 44823,
                 "first_datetime": "2011-01-03 08:45:00", "last_datetime": "2024-12-31 23:59:03" }
    }
  }
}
```

**資料紀錄欄位（`2026-11-r1` 起；舊任務都是 `null`／空）：**

| 欄位 | 何時寫入 | 說明 |
|---|---|---|
| `pipeline_version` | 建立時（web）；開始執行時以 worker 的版本覆寫 | 流程版本，見「版本與變更」 |
| `warnings` | 建立時；讀完資料後以 `data_report.warnings` 為準 | 只提示、不擋任務 |
| `env_fingerprint` | 開始執行時 | 實際執行環境的套件／CUDA／GPU／TA-Lib C 版本；另有 `worker`（認領這筆任務的 worker 程序：`instance` 每次啟動不同、`pid`、`host`、`name`、`started_at`），用於中斷後回收，舊任務沒有 |
| `data_report.resources` | 任務結束時（成功或失敗；被系統終止則沒有） | worker 程序峰值記憶體：`peak_rss_bytes`／`peak_rss_gib`（`VmHWM`），`scope` 為 `this_job`（每筆開始時歸零）或 `process_lifetime`（無法歸零時） |
| `data_report` | 讀完資料、訓練開始前（訓練失敗的任務也查得到） | 每個時間框架讀到什麼、排除了什麼。`n_excluded`：排除的 `session='other'` 列數；`n_gaps`：同一交易日同一時段內相鄰 K 棒間隔超過一個週期的次數；`n_missing_bars`：缺的 K 棒總數（**只統計，不補棒，也不擋訓練**）；日線表沒有 `n_gaps`；`available_time`：目前的資訊可用時間假設（發布延遲 0 秒為未實測假設） |

`status`：`pending`／`running`／`done`／`failed`（看 `error`）。worker 同時間只跑一筆；`pending` 表示還在排隊，或 GPU 實際可用顯存不足預留量（`TRAINING_GPU_MIN_FREE_MB`，預設 4096MiB，已先釋放 worker 自己的快取才判斷）而每 30 秒重試——延後原因（total／used／free、worker 自身佔用、門檻）記在 training 容器 log（`docker logs hermesnote-training`），API 不回傳。`result` 的 key 是每個 Model Node 的 `id`；`job_type='infer'` 的 job，`result[node_id]` 改成 `{"count": N, "start_ts": unix秒, "end_ts": unix秒}`（推論結果本身不在這裡，見下方 `/inference_predictions`）。

## `DELETE /api/model/jobs/{job_id}`

刪掉一筆任務（含進度、已存的模型權重，DB CASCADE）。如果有 Phase 2/3 的後續任務指著它，回 409，要求先刪後續那筆。主要給孤兒 job（送出後容器崩潰、狀態卡死）清理用，不能中斷真的還在跑的訓練（見下面 `/cancel`）。

## `POST /api/model/jobs/{job_id}/cancel`

把一筆 `pending`/`running` 的任務標記成 `failed`（人工中斷）。**只改 DB 狀態，不會真的砍掉背景 process**——worker 是單一長駐行程、同時間只跑一個 job，沒有子行程可以單獨終止，真的要中斷運算需要重啟整個 training 容器（NAS 端操作，這支 API 沒有權限做）。呼叫這支的目的只是讓歷史紀錄立刻反映「已放棄」，不用等重啟才發現它其實已經死了。

**狀態規則（2026-11-18 起）**：
- worker 只認領 `pending` 的任務（原子操作），已取消的 `pending` 不會再被執行。
- 執行中被取消的任務：worker 的每一筆寫入（進度、預覽、讀取報告、模型產物、推論結果、樣本框架、逐筆結果、評估、holdout 使用紀錄、串接折次、完成狀態）都會先確認任務仍是 `running` 且屬於自己，所以取消後的下一次寫入（通常是下一個 epoch 的進度）就會中止訓練、不再寫入任何資料，狀態維持取消時的 `failed`（舊版會跑完並改回 `done`）。取消前已寫出的資料保留。
- worker 中斷（程序被終止、容器重啟）時任務會停在 `running`；下一個 worker 實例取得單一實例鎖後，把「有 `env_fingerprint.worker`、且不是自己認領」的 `running` 改成 `failed`，`error` 以「worker 中斷，原因未判定」開頭並記原實例 ID。沒有 `env_fingerprint.worker` 的舊任務不會自動回收。

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

**回應：** `{ "job_id": "...", "warnings": [] }`（`warnings` 規則同 `/train`，依推論期間判斷）——一樣是非同步 job，用 `GET /api/model/jobs/{job_id}` 輪詢，`status='done'` 後結果已經分批存進 DB，用下面這支查。

**Phase 3 範例**（`target_job_id` 是已完成的 Phase 2；`start` 必須晚於 Phase 1/2 的 `end`）：

```json
{ "target_job_id": "<Phase 2 job_id>", "start": "2025-01-02", "end": "2026-01-01", "phase": 3, "use_weights": "best" }
```

回應 `{ "job_id": "...", "warnings": [] }`；建立的任務 `job_type: "infer"`、`phase: 3`、`parent_job_id` 為該 Phase 2；同時寫一筆 holdout 使用紀錄（`kind: "inference"`）。

**Phase 3 的執行內容（`2026-11-r2` 起）**：
1. 逐列推論照舊寫進 `model_inference_predictions`（用 `use_weights` 那組），`/inference_predictions` 讀法不變，MCP 現有讀法不受影響。
2. **best 與 last 兩組權重都推論**，各自依產物裡的答案定義（Label 的 outcome／labeling_rule／horizon）對答案、用與訓練相同的評估函式評估；尾端答案尚未成熟的時間點照常預測與保存，但**不納入評估**（`label_unmatured`）。
3. 逐筆結果寫進 `model_prediction_records`（`dataset: "holdout"`，best、last 各一列），評估寫進 `model_evaluations`，並寫一筆 holdout 使用紀錄（`kind: "evaluation"`）。測試期間可以重跑，每次都記錄，不限次數。
4. v2 產物：依產物保存的歷史需求**往 `start` 之前載入暖機資料**，測試期間第一根就有預測；輸入不足的時間點不預測，記原因（`holdout.skipped_reasons`）。舊格式產物：推論沿用舊路徑（期間開頭的暖機時間點照舊輸出）；舊產物沒有保存訓練集統計時，基準標示無法計算；舊格式的雙頭模型不做 Phase 3 評估（舊推論路徑只回回歸值）。

**Phase 3 完成後的 `GET /jobs/{job_id}`（節錄）**：

```json
{
  "job_type": "infer", "phase": 3, "status": "done", "pipeline_version": "2026-11-r2",
  "result": {
    "m": {
      "count": 5124, "start_ts": 1735779600, "end_ts": 1767159000, "use_weights": "best",
      "holdout": {"n_points": 5124, "n_predicted": 5124, "n_included": 5123, "n_unmatured": 1, "n_skipped": 0,
                  "skipped_reasons": {}, "cutoff_ts": "2025-12-31 13:45:00", "baseline_stats": "train_stats"},
      "metric_specs": {"...": "同訓練任務"},
      "evaluations": [
        {"id": "m/default/holdout/best", "model_node": "m", "head": "default",
         "dataset": {"split": "holdout", "label": "Holdout", "phase": 3, "n_samples": 5123, "n_unmatured": 1,
                     "n_skipped": 0, "task_data_range": {"start": "2025-01-02", "end": "2026-01-01"}, "description": "..."},
         "point": {"kind": "best", "weights": "best", "label": "最佳權重（best）"},
         "metrics": {"accuracy": 0.51, "balanced_accuracy": 0.50, "macro_f1": 0.49, "roc_auc": 0.51},
         "unavailable": {}, "baselines": [{"key": "majority_class", "available": true, "metrics": {"accuracy": 0.52}}]},
        {"id": "m/default/holdout/last", "...": "同上，point.kind=last"}
      ]
    }
  }
}
```

數值只是格式示意。`evaluations` 的格式與訓練任務相同（同一套指標鍵與 `metric_specs`），只差 `dataset.split="holdout"`、`point.kind` 是權重。`GET /jobs/{id}/evaluations` 回同一份內容（每組權重一列，含 `n_included`、`excluded_reasons`、`cutoff_ts`）。

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

## v2 圖與實驗層（`2026-11-r2`，M2～M7）

### 兩種用法

| 用法 | 入口 | 適合 |
|---|---|---|
| 直接送 v2 圖 | `POST /api/model/train`，`graph_spec` 加 `"mode": "v2"` | 單次實驗、MCP 直接提交。樣本框架在任務內建立（不保存時間點，雜湊記在結果） |
| 實驗層 | `POST /api/model/experiments` → `.../arms` → `.../arms/{arm}/run` | 多組別公平比較：共同樣本框架（保存、可版本化）、模型種子清單、階段門檻、配對比較、holdout 使用紀錄 |

Phase 2 仍用 `POST /train` + `phase: 2` + `parent_job_id`（v2 母任務會繼承 `mode`／`settings`，只把切分改成 chronological）；Phase 3 仍用 `POST /infer` + `phase: 3`。

### v2 圖格式

```json
{
  "mode": "v2",
  "start": "2016-01-01", "end": "2024-01-01",
  "settings": {
    "split_seed": 42, "val_ratio": 0.2,
    "warmup": {"epsilon": 0.01},
    "exclusion_warn_ratio": 0.05,
    "alignment": {"allow_previous_session": true, "max_sessions_back": 1, "max_age_sec": null,
                  "delays": {"tx_60m": {"sec": 0, "status": "assumed", "note": ""}}},
    "cascade": {"strategy": "forward_chaining", "k": 5, "final": "refit_full"}
  },
  "nodes": [
    {"id": "f", "type": "feature", "key": "ohlcv", "timeframe": "tx_15m"},
    {"id": "r", "type": "feature", "key": "talib_indicator", "timeframe": "tx_15m", "params": {"name": "RSI", "timeperiod": 14}},
    {"id": "o", "type": "feature", "key": "talib_indicator", "timeframe": "tx_15m", "params": {"name": "OBV"},
     "transform": {"type": "rolling_sum", "window": 20}},
    {"id": "a", "type": "feature", "key": "talib_indicator", "timeframe": "tx_60m", "params": {"name": "ADX", "timeperiod": 14},
     "warmup": {"bars": 120}},
    {"id": "v1", "type": "feature", "key": "raw_volume", "timeframe": "tx_1m"},
    {"id": "l60", "type": "label", "timeframe": "tx_60m", "outcome": "log_return", "labeling_rule": "fixed_threshold",
     "params": {"horizon": 1, "n_classes": 2}},
    {"id": "up", "type": "model", "key": "xgboost", "inputs": ["a"], "label": "l60", "window": 10, "output_mode": "probability",
     "params": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "early_stopping": {"patience": 20}, "seed": 42},
     "upstream_policy": {"strategy": "forward_chaining", "k": 5, "final": "refit_full", "extra_purge_sec": 0}},
    {"id": "l", "type": "label", "timeframe": "tx_15m", "outcome": "log_return", "labeling_rule": "fixed_threshold",
     "params": {"horizon": 1, "n_classes": 2}},
    {"id": "m", "type": "model", "key": "lstm", "inputs": ["f", "r", "o", "v1", "a", "up"], "label": "l", "params": {"...": "LSTM 參數"},
     "branches": [
       {"id": "main", "inputs": ["f", "r", "o", "v1"], "window": 60, "aggregate": {"v1": "sum"}},
       {"id": "hour", "inputs": ["a", "up"], "window": 8, "encoder": "lstm"}
     ]}
  ]
}
```

| 欄位 | 規則 |
|---|---|
| 決策時間框架 | ＝最終 Model Node 的 Label 時間框架。樣本＝這個時間框架在 `[start, end)` 的每一根 K 棒（以 `bar_start_ts` 識別），決策時刻＝該 K 棒 `bar_end_ts`＋延遲 |
| `settings.split_seed`／`val_ratio` | 實驗層切分（只做一次，所有組別共用）。Phase 1 random；Phase 2 chronological（答案 K 棒跨進驗證起點的訓練點排除 `boundary`） |
| Feature `warmup.bars` | 研究者設定的暖機（從資料載入起點算），只能比程式基本值長。**composite／unknown 必填**；查基本值：`POST /registry/v2/feature_memory`，實測建議：`POST /warmup/suggest` |
| Feature `transform` | `none`／`diff`／`pct_return`／`log_return`／`rolling_sum`（`window`）／`fixed_anchor`（`anchor` 日期）／`legacy_cumulative`。**累積型（OBV、AD）必填**，只能用 diff／rolling_sum／fixed_anchor／legacy_cumulative |
| 往前載入 | worker 依暖機＋窗格＋上游需求，自動往 `start` 之前多載入（`data_report.history_requirements`、各時間框架 `lookback`）；推論同一規則 |
| Model `branches` | 不給＝一個 `main` 分支（`inputs`、`window`，時間框架＝決策時間框架）。分支的 `inputs` 合起來必須剛好等於節點 `inputs`。`encoder`：LSTM 用 `lstm`（預設）或 `last_value`；XGBoost 用 `flatten`（預設）或 `last_value`（window 固定 1）；`normalization`：目前只有 `[{"method": "global_minmax"}]`（預設） |
| 跨時間框架 | 分支內較粗的特徵／上游預測：as-of（取可用時間 ≤ 該根分支 K 棒可用時間的最後一根）；較細的特徵：必須在分支 `aggregate` 指定 `last／mean／sum／std／max／min`（只用完全落在分支 K 棒內且已收完的子 K 棒） |
| `alignment` | `allow_previous_session`（預設 true）、`max_sessions_back`（預設 1，0～5，`null`＝不限）、`max_age_sec`（預設 `null`）：套在每個分支最後一根可用 K 棒與 as-of 值上，超過就該時間點不產生樣本（`stale_branch`）。`delays`：依「資料表×時間框架」設定，預設 0 秒 `assumed` |
| `upstream_policy` | 被下游使用的上游模型：`forward_chaining`（預設，K=5）／`purged_kfold`／`blending`／`in_sample`（舊行為，任務帶 `upstream_in_sample` 提示）；`final`：`refit_full`（預設）／`last_fold`；`extra_purge_sec` 只能加長自動清除範圍 |

**排除原因代碼**（`GET /registry/v2` 的 `exclusion_reasons`）：`label_unavailable`、`label_unmatured`、`boundary`、`insufficient_history`、`warmup_not_ready`、`stale_branch`、`missing_availability`、`upstream_unavailable`、`upstream_purged`、`outside_frame`、`mapped_excluded`。

**v2 訓練結果新增欄位**（`result[node]`）：`train_stats`；`training_meta.pipeline: "v2"`、`frame {hash, counts}`、`exclusions {train, val, frame}`、`exclusion_ratio`、`warmup {特徵: {memory_type, base_bars, researcher_bars, bars, transform, total_bars}}`、`branches {分支: {timeframe, window, encoder, n_from_previous_session, max_sessions_back, max_age_sec, median_age_sec}}`、`cascade {policy, purge_sec, n_folds, …}`（上游）、`evaluation_dataset`；最終節點另有 `frame`（完整摘要）與 `n_prediction_records`。`evaluations` 格式不變（`dataset.split: "val"`）。

### 實驗層端點

**建立實驗** `POST /api/model/experiments`

```json
{"name": "15m RSI vs RSI+1h ADX", "description": "…",
 "settings": {
   "start": "2016-01-01", "end": "2024-01-01", "holdout": {"start": "2024-01-01", "end": "2026-01-01"},
   "label": {"timeframe": "tx_15m", "outcome": "log_return", "labeling_rule": "fixed_threshold", "params": {"horizon": 1, "n_classes": 2}},
   "split_seed": 42, "val_ratio": 0.2, "warmup": {"epsilon": 0.01}, "exclusion_warn_ratio": 0.05,
   "alignment": {"allow_previous_session": true, "max_sessions_back": 1, "max_age_sec": null, "delays": {}},
   "cascade": {"strategy": "forward_chaining", "k": 5, "final": "refit_full"},
   "gates": [{"phase": 2, "metric": "macro_f1", "op": ">=", "value": 0.5, "weights": "best", "head": "default"}]}}
```

回應 `{"experiment_id": "…", "warnings": [...], "pipeline_version": "2026-11-r2"}`。`holdout.start` 不能早於 `end`。

| 端點 | 用途 |
|---|---|
| `GET /experiments`、`GET /experiments/{id}` | 清單；單筆含 `arms`、`frames`（phase、version、status、hash、counts）、`jobs` |
| `PATCH /experiments/{id}/settings` `{"settings": {...}, "note": "…"}` | 只能改 `alignment`、`gates`、`exclusion_warn_ratio`、`warmup`、`cascade`（影響之後的任務；修改紀錄存在 `settings.history`） |
| `POST /experiments/{id}/arms` `{"name", "nodes": [Feature／Model…], "model_seeds": [42, 43]}` | 組別。最終模型的 `label` 必須是 `"label"`（實驗注入）；模型種子覆寫每個 Model Node 的 `params.seed` |
| `POST /experiments/{id}/arms/{arm_id}/run` `{"phase": 1}` | Phase 1：每個種子一筆任務 |
| 同上 `{"phase": 2, "parent_job_ids": null, "gate_override": null}` | Phase 2：預設每筆已完成的 Phase 1 一對一銜接；先檢查門檻，未過回 409，附 `gate_override: {"reason"}` 放行（確認人由後端依驗證身分記錄） |
| 同上 `{"phase": 3}` | Phase 3：每筆已完成的 Phase 2 一筆 infer 任務（期間＝`settings.holdout`），best／last 都評估 |
| `POST /experiments/{id}/frames` `{"phase": 1, "note": "…"}` | 建立新版本框架（舊結果保留；跨版本比較會被拒絕） |
| `GET /frames/{frame_id}/points?status=&cursor=&limit=` | 框架時間點（ts、決策時刻、答案時間、狀態、原因） |
| `GET /experiments/{id}/compare?job_a=&job_b=&weights=best` | 配對比較（共同時間點）。分類：各自 accuracy、只有 A／B 答對數、McNemar 精確 p；回歸：各自 MSE、平方誤差差的配對 t。Phase 3 的比較寫 holdout 使用紀錄 |
| `GET /experiments/{id}/holdout_usage` | holdout 使用紀錄（inference／evaluation／comparison） |

**確認人來自驗證身分**：`gate_check.override.confirmed_by` 是後端依金鑰名稱或登入 email 記錄的身分（`key:…`／`admin:…`）；請求裡送的 `confirmed_by` 只存成 `note_confirmed_by` 備註，不當身分。認證上線前建立的放行紀錄沒有 `note_confirmed_by`，其 `confirmed_by` 是未驗證的文字。

`run` 回應：`{"jobs": [{"job_id", "model_seed", "phase", "parent_job_id"?, "gate_check"?}], "frame": {...}}`。`gate_check`：`{"configured", "passed", "status": "passed|overridden|not_configured", "conditions": [{metric, op, value, weights, head, actual, ok, reason}], "override"?: {reason, confirmed_by, note_confirmed_by, at}}`。

### 任務層端點

| 端點 | 內容 |
|---|---|
| `GET /jobs/{id}/records?node_id=&dataset=val|holdout|train&weights=best|last&cursor=&limit=1000` | 逐筆結果（id 游標，上限 5000）：`sample_ts`、`decision_available_ts`、`window_start_ts`、`target_ts`、`target_available_ts`、`y_true`、`y_pred`、`proba`、`included`、`exclusion_reason`、`branches`（各分支最後一根時間、可用時間、`age_sec`、`sessions_back`、`previous_session`）、`upstream`（上游折次與可用時間）、`frame_hash`。`weights: null` 的列＝輸入不足未預測 |
| `GET /jobs/{id}/evaluations` | Phase 3 評估（每組權重一列） |
| `GET /jobs/{id}/holdout_usage` | 這筆任務相關的使用紀錄 |
| `GET /jobs/{id}/upstream_folds` | 串接的上游折次清單（訓練範圍、清除區間、預測範圍、樣本數、權重指紋；`fold: -1`＝最終上游模型） |
| `GET /jobs/{id}/cascade_audit?format=json|csv&sample_n=5` | 獨立核對（只讀紀錄重算 5 條規則）：`{"rules": {"r1": {"checked", "violations"}, …}, "passed", "violations": [...], "sample_chains": [...]}`；`csv` 回違反清單 |
| `POST /warmup/suggest` `{"feature": {key, params, timeframe}, "epsilon": 0.01, "end": "2026-01-01", "n_bars": 3000}` | `{"program": {memory_type, base_bars, needs_researcher, detail}, "empirical": {"bars", "trials", "tolerance"}}`（只是建議） |
| `GET /registry/v2`、`POST /registry/v2/feature_memory` | v2 元件清單；單一特徵的記憶類型與基本暖機 |

### 資料表（`2026-11-16` 遷移新增）

`model_experiments`、`model_experiment_arms`、`model_sample_frames`、`model_sample_frame_points`、`model_prediction_records`、`model_evaluations`、`model_holdout_usage`、`model_upstream_folds`；`model_training_jobs` 加 `experiment_id`、`arm_id`、`frame_id`、`model_seed`、`split_seed`、`gate_check`、`requested_by`；`model_experiments` 含 `created_by`；`model_artifacts` 加 `fingerprint`、`warmup_policy`、`train_class_counts`（存 `train_stats`）、`history_requirements`。


## 已知限制（截至這份文件更新時的程式碼狀態）

- **正規化**已接（train-only fit 的 MinMax；v2 的步驟框架目前也只有 `global_minmax`）。**跨時間框架**：舊格式的圖仍回 400；`mode: "v2"` 的圖開放（as-of 對齊與子 K 棒彙總，見「v2 圖與實驗層」）。
- **舊格式的圖**：上游 Model Node 接下游時仍是樣本內預測（有洩漏，為了相容行為不變）。要防洩漏請改用 `mode: "v2"`（預設 `forward_chaining` OOF）。
- v2：一個 Model Node 只能有一個 Label；只支援 TX 系列；編碼器只有 `lstm`／`flatten`／`last_value`，合併只有 concat；所有 lstm 分支共用節點的 `lstm_layers`／`bidirectional`／`attention`；串接的 Phase 3（與一般推論）上游模型固定用 best 權重，`use_weights` 只影響最終模型（最終模型 best、last 都會評估）；沒有超參數搜尋、跨商品、形成中 K 棒。發布延遲預設 0 秒且為未實測假設。GPU 訓練不保證逐位元組重現；遞迴型指標只保證在 ε 內一致（固定錨點除外）。
- 推論時欄位順序依 `inputs` 的原始順序（與訓練相同）。**具名輸出修正部署之前的推論實作是「先全部 Feature、再接上游模型」**，只有 `inputs` 剛好是這個排列時才跟訓練對得上；如果有把模型排在特徵前面或夾在中間的舊產物，推論結果可能錯位，需重新訓練。管理後台產生的圖一律特徵在前，不受影響。
- 進階 `labeling_rule`（波動度門檻、分位數切分、成本導向、triple barrier）還沒做，目前只有 `fixed_threshold`（分類）跟 `identity`（回歸）。
- `/cancel` 只改 DB 狀態，無法真的中斷背景運算中的訓練。
- **混合模型目前僅支援節點輸出串接**：上游模型節點凍結後的輸出，當作下游模型節點的一個輸入特徵，兩個節點各自獨立訓練，沒有梯度互通。**不支援**聯合訓練（end-to-end 跨節點反向傳播）、集成投票／加權融合這類結構、共享編碼器的多任務結構——這些都還沒實作，不要從「支援具名輸出引用」推論出更廣的能力。
- **Attention 本版本只支援零或一個**，不支援多個 Attention 的串接或並聯；送清單一律 400，不是靜默忽略（未來擴充點見 [extending.md](extending.md)）。
- 後台畫布上「模組→模組」的連線只接上游的 `default` 輸出；要接具名輸出（例如雙頭的 `direction_probability`）目前要直接用 API 送 `{"node", "output"}` 引用。
- 沒有「只驗證不送出」的端點：提交驗證失敗的 400 不會建立 job，可直接當驗證用。

以上限制以程式碼為準，`GET /api/model/registry/*` 的即時清單跟本文件的落差，一律以即時清單為準。
