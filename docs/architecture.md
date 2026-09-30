# Hermesnote V2 — Architecture

> 讀者：接手實作的 AI。本檔說明「東西在哪、誰負責什麼、資料怎麼流、怎麼擴充、哪裡不能動」。
> 需求與不變條件見 `spec.md`（`REQ-*`／`INV-*`）；每個設計的理由見 `decisions.md`（`D-xxx`）；歷程見 `record/`。
> 最後整理：2026-09-29。與程式不一致時以程式為準，並回報差異。

## 1. 系統拓樸

```
瀏覽器 ──> Nginx Proxy Manager（hermesnote.com）
              ├─ /      → 192.168.0.44:8082  hermesnote-frontend（nginx:alpine，bind mount frontend/dist）
              └─ /api   → 192.168.0.44:8000  hermesnote-backend（FastAPI／uvicorn，不掛 GPU）
HA（Hermes Agent）──HTTP──> hermesnote-backend
hermesnote-backend ──寫 pending job──> PostgreSQL hermesnote ──輪詢──> hermesnote-training（worker，掛 GPU）
hermesnote-training ──唯讀──> PostgreSQL quotes（報價）
```

| 元件 | 位置 | 說明 |
|---|---|---|
| NAS | TrueNAS 192.168.0.44，RTX 5060 Ti 16GB | 部署根目錄 `/mnt/Hermesnote/web/hermes/`（與 repo 對稱：`frontend/dist`、`backend/`、`docs/agent-api/`、`docker-compose.yml`、`nginx.conf`） |
| 容器 | `docker-compose.yml` | `hermesnote-frontend`、`hermesnote-backend`、`hermesnote-training`；backend 與 training 共用 `backend/.env`。GPU 用 `deploy.resources.reservations.devices`（`runtime: nvidia` 在此機器無效） |
| DB | PostgreSQL（NAS:5432） | `hermesnote`（讀寫）、`quotes`（唯讀）（INV-03） |
| 本機開發 | Windows，conda env `hermesnote-backend`（有 GPU 的 torch）、Node | 前端 dev server port 5174（`.claude/launch.json`） |

## 2. 技術棧與版本鎖

| 層 | 技術 | 鎖定原因 |
|---|---|---|
| 前端 | React 19、TypeScript ~5.8、Vite ^6.3.1、react-router-dom 7、lightweight-charts 5、@xyflow/react 12 | Vite 8 的原生綁定在開發機安裝失敗（D-035） |
| 後端 | Python 3.12（Docker `python:3.12-slim`）、FastAPI、asyncpg（web）、psycopg2（training 同步路徑） | **沒有 ORM、沒有 Alembic**；schema 用 SQL 檔管理（見 §4） |
| 回測 | TA-Lib 0.7.1（C library 在 Dockerfile 編譯）、vectorbt 1.1.0、plotly **5.24.1** | vectorbt 與新版 plotly 不相容（D-035） |
| 訓練 | torch 2.10.0（cu128，training 映像）、xgboost 3.2.0、scikit-learn 1.9.0（評估指標）、numpy、pandas | training 映像另有 `backend/training/requirements.txt` 與 `training/Dockerfile` |
| 認證 | Google Identity Services＋`python-jose` JWT | D-007 |

## 3. 模組職責

### 3.1 後端 `backend/`

| 路徑 | 職責 | 邊界 |
|---|---|---|
| `main.py` | 建 app、掛 router、`/health`、`/api/backtest/date-range`、`/api/backtest/demo` | 不放業務邏輯 |
| `auth.py`、`routers/auth.py` | Google token 驗證、白名單、JWT、`get_current_user` | 目前只有 `/api/auth/me` 用到 auth dependency（OPEN-01） |
| `routers/indicators.py` | `GET /api/indicators` | — |
| `routers/strategy.py` | `/api/backtest/*`：送出、job 查詢、分頁、收藏 | 運算交給 `services/job_runner.py` |
| `routers/model_training.py` | `/api/model/*`：train、infer、jobs、progress、stream（WebSocket）、artifacts、registry | **不做訓練**，只驗證 graph_spec 並寫 pending job |
| `indicators/` | `registry.py`（TA-Lib 161 函式機械式登記）、`descriptions.py`（中文說明）、`signals.py`（觸發模板）、`filters.py`（過濾門檻）、`tree.py`（四棵條件樹求值） | `docs/library.md` 與 `descriptions.py` 同步維護 |
| `services/quotes.py` | 讀 `quotes` DB（表名白名單、去重、`bar_end_ts` 有無判斷） | 唯一的報價讀取入口 |
| `services/backtest_demo.py`、`job_runner.py`、`job_store.py` | 回測運算（條件樹＋vectorbt）、背景執行（`asyncio.create_task`，重啟會遺失執行中任務）、`quant_*` 表存取 | — |
| `services/hermesnote_db.py` | `get_conn()`：asyncpg 連線（JSONB 回來是字串，讀取端必須 `json.loads`） | — |
| `services/model_artifacts.py` | web 端的產物清單／收藏（async） | 推論載入用 `training/artifacts.py`（sync） |
| `training/worker.py` | training 容器的長駐程序：輪詢 pending → GPU 可用顯存判斷 → 執行 → 標 done／failed → 釋放快取 | 同時間一筆（INV-08） |
| `training/graph.py` | 節點圖執行引擎：資料對齊、Label、正規化（train-only fit）、切分、呼叫 `train_model`、具名輸出串接 | 架構無關；不得寫死任何模型 |
| `training/graph_refs.py` | `validate_graph_spec`：提交前驗證（引用、成環、登記、參數） | API 與 worker 共用（INV-05） |
| `training/phases.py` | Phase 2 繼承、Phase 3 父任務檢查 | INV-01 |
| `training/output_specs.py`、`time_semantics.py` | 具名輸出 metadata、識別時間／可用時間 | — |
| `training/evaluation.py` | 分類／回歸完整評估與基準 | 算不出→`null`＋原因（INV-07） |
| `training/evaluation_records.py` | 評估報告 → `evaluations` 清單（worker 保存新任務；API 讀取時轉換舊紀錄，`enrich_job`）；補 `metric_specs` | 只重組，不重算（accuracy 由混淆矩陣推算） |
| `training/inference.py`、`inference_store.py` | 載入產物推論（`use_weights`），分批寫入 | — |
| `training/artifacts.py` | 產物寫入／讀取（sync psycopg2） | 無條件讀寫 `weights_last`（需 migration，已套用） |
| `training/job_store.py`、`progress.py`、`preview_store.py` | job 狀態、逐輪進度＋NOTIFY、抽樣預覽＋NOTIFY | JSONB 寫入經 `json_safe.dumps` |
| `training/registry/` | 見 §6 | — |
| `tests/` | 187 項 unittest（CPU、合成資料）＋`contract_harness.py`（資料契約／推論／訓練決定性基準比對） | 測試固定 CPU（`CUDA_VISIBLE_DEVICES=-1`） |

### 3.2 前端 `frontend/src/`

| 路徑 | 職責 |
|---|---|
| `App.tsx` | 路由：`/`、`/quant`、`/model`、`/hermes`（`Layout`）；`/admin/login`；`/admin/{model,quant,trading,ledger}`（`RequireAuth`＋`AdminLayout`） |
| `auth/` | `AuthContext`（token 存 localStorage，開頁驗 `/api/auth/me`）、`RequireAuth` |
| `pages/QuantBacktestPage.tsx` | 回測結果（K 線、成交量、MA、交易標記、分頁載入） |
| `pages/ModelTrainingPage.tsx` | 訓練進度（REST 補歷史＋WebSocket 即時）、best／last、評估報告、推論結果瀏覽 |
| `components/metrics/` | 通用指標視覺化（前後台共用）：`MetricsDashboard`（逐輪曲線＋逐輪明細＋評估）、`EvaluationView`（依輸出頭／資料集／評估時點篩選、比較條件）、`MetricSeriesChart`、`EvaluationParts`（數值比較＋基準、逐類別表、分布、熱圖）、`metricsModel.ts`（資料契約型別） |
| `pages/admin/sections/QuantSettings.tsx`、`TreeBuilder.tsx` | 回測條件樹編輯 |
| `pages/admin/sections/ModelSettings.tsx` | 訓練模組畫布 → graph_spec、歷史紀錄、Phase 2／3、重跑、推論 |
| `pages/admin/sections/SchemaForm.tsx`、`schemaFormLogic.ts` | 通用參數表單；語意鏡像後端 `param_schema.py`，後端是最終把關 |

## 4. 資料庫

**`hermesnote`**（schema：`backend/quant_init.sql`、`backend/model_init.sql`；變更：`backend/migrations/<日期>_<主題>.sql`，冪等、需核准後套用，OPS-02）

| 表 | 用途 | 關聯 |
|---|---|---|
| `quant_backtest_jobs`／`_candles`／`_trades` | 回測任務與結果 | 子表 `ON DELETE CASCADE` |
| `quant_saved_strategies` | 收藏的條件樹 | — |
| `model_training_jobs` | 訓練／推論任務（`job_type`、`phase`、`parent_job_id`、`graph_spec`、`result` 摘要） | `parent_job_id` 為 NO ACTION（有子任務時不可刪） |
| `model_training_progress` | 逐輪指標（固定四欄＋`metrics` JSONB） | CASCADE |
| `model_training_preview_samples` | 訓練中抽樣預覽 | CASCADE |
| `model_artifacts` | 權重（`weights`／`weights_last` bytea）、`model_config`、特徵／目標定義、graph 快照 | CASCADE，PK `(job_id, node_id)` |
| `model_inference_predictions` | 推論逐列結果 | CASCADE |

**`quotes`**（唯讀）：`market.future_taifex_{symbol}_{timeframe}`（例：`tx_1m`、`tx_15m`、`tx_daily_day`、`tx_daily_full`），欄位含 OHLCV、`built_at`、`bar_end_ts`（`spread` 系列沒有）；保證金 `market.margin_requirements`。

## 5. 資料流程

**回測**：`POST /api/backtest/strategy` → 寫 `quant_backtest_jobs`（pending）→ `job_runner` 背景：讀 quotes → 條件樹求值 → vectorbt → 寫 candles／trades → 前端輪詢 `GET /jobs/{id}`，再分頁讀結果。

**訓練**：
1. `POST /api/model/train` → `validate_graph_spec`（失敗 400，不建 job）→ `model_training_jobs` pending（Phase 2 由後端從母任務產生 graph_spec）。
2. worker 輪詢 → `gpu_capacity()`（先釋放自身快取，再看 `nvidia-smi` 可用顯存 ≥ `TRAINING_GPU_MIN_FREE_MB`，預設 4096；不足則延後並記錄原因）→ `mark_running` → 再驗證 → 讀 quotes → `run_graph`（`asyncio.to_thread`）。
3. 每輪：`write_progress` 寫表＋`NOTIFY`；backend 的 `/jobs/{id}/stream` 以 `LISTEN` 轉成 WebSocket。
4. 完成：`save_artifacts_for_job` → `mark_done`（`result` 只留摘要，含 `evaluations`＋`metric_specs`）→ 釋放 CUDA 快取。
5. 讀取：`GET /jobs`、`/jobs/{id}` 經 `enrich_job` 補 `metric_specs`（進行中也有），舊紀錄的 `evaluation.best／last` 即時轉成 `evaluations`（資料庫不改寫）。

**推論**：`POST /api/model/infer`（Phase 3 檢查 holdout 邊界）→ infer job → worker 載入產物 → `sliding_window_view` 分批預測 → 分批寫 `model_inference_predictions`。

## 6. 模型組裝框架（`backend/training/registry/`）

| 檔案 | 內容 |
|---|---|
| `architectures.py` | `register(key, family, label, params_schema, extra_checks, capabilities, slots, outputs, describe_outputs, lazy_windows)`；`validate()`＝schema → slot 相容性 → `extra_checks`；`train_model`／`load_model`／`describe_outputs`；`component_registries()`／`list_components()` |
| `param_schema.py` | FieldSpec 驗證器（dot path、條件 `all`／`any`／`$情境`、`enum_cases`、`derived_from`、`component_ref`、`array_of_object`、補預設值） |
| `lstm.py`、`xgboost_model.py` | 目前的兩個架構 |
| `attention.py`、`optimizers.py` | 可共用元件（`slot_compatibility`；attention 另有 `output_kind` 與 `build_checked` 形狀檢查；optimizer 經 `build_optimizer()` 建立） |
| `training_control.py` | `early_stopping_fields()`、`BestTracker` |
| `metrics.py` | 指標定義登記表（可共用元件）；`build_specs()` 組模型的 `metric_specs`；`GET /api/model/registry/metrics` |
| `features.py`、`outcomes.py`、`labeling_rules.py`、`target_transforms.py`、`decision_rules.py` | 其他登記表 |

train 函式契約（所有架構一致）：回傳 `final_metrics`（best＋`last`）、`evaluation {best,last}`、`weights`、`weights_last`、`predict`（分類另有 `predict_proba`，多輸出另有 `predict_outputs`）、`model_config`（含解析後 `config`）、`device`。

## 7. 擴充點與禁區

| 要做的事 | 做法 | 不需要改 |
|---|---|---|
| 新架構（Transformer、Mamba…） | 新增 `registry/<name>.py` 並 `register()`，在 `architectures.py` 檔尾匯入；需要優化器就宣告 optimizer slot 並用 `build_optimizer()` | `graph.py`、`inference.py`、router、前端 |
| 新指標 | 在 `metrics.py` 登記（shape／direction／unit／format），模型 `metric_specs` 引用，並在訓練迴圈或評估報告產生對應鍵 | 前端頁面、API |
| 新 Attention／優化器 | 在 `attention.py`／`optimizers.py` 加一個 `@register` class，`slot_compatibility` 只列驗證過的組合 | 模型程式、API、前端 |
| 新元件登記表（新 slot） | 新增模組並加進 `architectures._component_modules()`／`component_registries()`；架構宣告 slot | API（`/registry/components/{name}` 通用）、前端（依 slots 自動抓） |
| 新 API 行為 | 改 router＋同輪更新 `docs/agent-api/`（REQ-API-01） | — |
| DB 變更 | 新增 `migrations/<日期>_<主題>.sql`（冪等）並同步 `*_init.sql`；套用需核准 | — |

**禁區**：
- 不得在 `graph.py`／`worker.py` 寫死特定模型或特定元件。
- 不得繞過 `validate_graph_spec` 建立 job。
- 不得改變 Phase 1／2／3 行為與 train-only fit（INV-01／02）。
- 不得讓 JSONB 寫入出現 NaN（一律經 `training/json_safe.py`）。
- 前端 `SchemaForm` 只鏡像後端語意，不得在前端新增後端沒有的規則。

詳細步驟與範例：`docs/agent-api/training/extending.md`。

## 8. 已知陷阱

| 項目 | 規則 |
|---|---|
| `load_dotenv()` | 讀環境變數的模組（`auth.py`、`services/quotes.py`、`training/progress.py` 等）必須在檔案開頭自行呼叫，不能依賴 `main.py` 的載入順序 |
| quotes 重複列 | 同一根 K 棒可能有多筆（`built_at` 不同）；查詢 `ORDER BY datetime, built_at` 後依 datetime 去重保留最後一筆（REQ-BT-06） |
| asyncpg JSONB | 回傳原始 JSON 字串，讀取端必須 `json.loads` |
| NOTIFY 上限 | payload 上限 8000 bytes；預覽的大資料走資料表，不塞進 NOTIFY |
| frontend 容器 | 只有 bind mount，換檔不會重讀設定；`deploy.ps1` 每次 `restart frontend` |
| 部署腳本 | `deploy.ps1` 先確認三容器都在跑，否則中止，不建立容器；`backend/` 以 `/MIR` 同步（NAS 上多出的檔案會被刪除，工作目錄的未追蹤檔也會被帶上去） |
| 測試環境 | `tests/contract_harness.py` 匯入時設定 `CUDA_VISIBLE_DEVICES=-1`，同一測試程序內不會有 GPU |

## 9. 已知限制

見 `spec.md` §6（OPEN-01～08）。
