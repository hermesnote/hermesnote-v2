# Hermesnote V2 — Spec

> 讀者：接手實作的 AI（Claude Code 等）。本檔是「系統必須怎樣」的規範，不是歷史回顧。
> 規範用語：**必須**／**不得**＝硬性要求；**應**＝預設做法，偏離要在回報中說明。
> 相關：`architecture.md`（東西在哪、怎麼擴充）、`decisions.md`（`D-xxx`，每條規則的理由）、`record/index.md`（逐輪交付紀錄）。
> 最後整理：2026-09-29。

## 0. 使用方式

1. 修改前：找出受影響的 `REQ-*` 與 `INV-*`；`INV-*` 不得違反，需要違反時先停下來問 Hermes。
2. 細節以「唯一來源」為準，本檔不重抄：API 契約 → `docs/agent-api/`；模型參數 → `GET /api/model/registry/*` 的 `params_schema`；資料表 → `backend/*_init.sql`＋`backend/migrations/`。
3. 驗收：每條需求列出測試或端點。完成改動後跑 `cd backend && python -m unittest discover -s tests -t .` 與 `python -m tests.contract_harness compare <baseline>`（改動前先 `save` 基準）。
4. 程式與本檔不一致時：不自行選邊，回報差異並問 Hermes。
5. 本檔以外的新需求：實作完成後補進本檔（新增 `REQ-*`），重大取捨補進 `decisions.md`。

## 1. 定位與範圍

| ID | 規範 |
|---|---|
| SCOPE-01 | Hermesnote V2 是**研究敘事型個人網站**：呈現 Hermes 的研究（台指期模型訓練、量化回測）與結果。不是純求職作品集，也不是多人平台。（D-001） |
| SCOPE-02 | 實驗（訓練、推論、回測）由 Hermes Agent（HA）透過 HTTP API 送出；後台表單是同一套 API 的人工操作介面。（D-010） |
| SCOPE-03 | 公開頁面只呈現、不提供操作；所有設定與送出在需要登入的後台。（D-007） |
| SCOPE-04 | 只有 Hermes（白名單 email）使用後台，**不得**設計外部註冊或多租戶。（D-007） |

## 2. 不變條件（違反＝錯誤）

| ID | 規範 | 驗收 |
|---|---|---|
| INV-01 | 教授三階段協定：Phase 1＝訓練期間內 random 切分；Phase 2＝繼承已完成 Phase 1 的 graph_spec 原樣（含全部 params），只把 `split_strategy` 改為 chronological；Phase 3＝對 Phase 2 模型做 holdout 推論，`start` 必須晚於 Phase 1/2 訓練期間的 `end`。**不得**更動此協定。（D-002） | `tests/test_phases_dual.py` |
| INV-02 | 防 data leakage：前處理（正規化、目標縮放）只用訓練集 fit；分類基準的多數類別由訓練集決定；對齊與切分用識別時間，不得用未來資訊。（D-003、D-019） | `tests/test_available_time.py`、`tests/contract_harness.py` |
| INV-03 | `quotes` DB 只能 SELECT；`hermesnote` DB 可讀寫（含 DDL），新資料表用用途前綴（`quant_`、`model_`），不加版本後綴。（D-033） | — |
| INV-04 | 研究值不由平台代填：`params_schema` 中沒有 `default` 的必填欄位，缺少就拒絕。（D-020） | `tests/test_model_framework.py` |
| INV-05 | 提交驗證失敗必須回 400、一次列出全部問題、**不建立 job**；worker 讀資料前再驗證一次。（D-018） | `tests/test_named_outputs.py` |
| INV-06 | 所有模型都保存 best 與 last 兩組權重並各自完整評估；`early_stopping.patience=0`＝關閉早停、跑滿輪數。（D-024） | `tests/test_lstm.py`、`tests/test_xgboost.py` |
| INV-07 | 寫入 JSONB 的數值不得含 NaN／Infinity（存 `null`）；算不出的評估指標必須是 `null`＋`*_unavailable_reason`。（D-027） | `tests/test_xgboost.py`、`tests/test_artifacts_sql.py` |
| INV-08 | 同一 worker 同時間只執行一筆 job；開始前依實際可用顯存判斷，不得把 worker 自身快取當外部佔用。（D-031） | `tests/test_worker_gpu.py` |
| INV-09 | 已撤回的功能不得重新引入：scrub 維護閘門、暫停／checkpoint／硬中斷恢復。（D-021） | — |

## 3. 功能需求（逐條摘要）

### 3.1 量化回測

| ID | 規範 | 驗收／來源 |
|---|---|---|
| REQ-BT-01 | `GET /api/indicators` 提供 TA-Lib 全部 161 個指標（參數 schema、輸出線、中文說明、`signal_supported`／`signal_type`）。 | `docs/agent-api/indicators.md` |
| REQ-BT-02 | 策略＝四棵條件樹（多進／多出／空進／空出，空方可為 null）；節點為條件（觸發型或 `filter`）或群組（`and`／`or`／`weighted`，可無限巢狀）；觸發方向依所在樹自動決定；`filter` 只擋進場不擋出場。（D-008） | `docs/agent-api/backtest/tree-schema.md` |
| REQ-BT-03 | `POST /api/backtest/strategy` 立即回 `job_id`，背景執行；結果存 `quant_backtest_jobs`／`_candles`／`_trades`，刪 job 連帶刪結果。（D-009） | `docs/agent-api/backtest/api.md` |
| REQ-BT-04 | 結果查詢必須有界分頁（candles、trades、trades/range），單次回應大小有上限。（D-009） | 同上 |
| REQ-BT-05 | 回測可收藏；收藏的條件樹可當訓練的 Feature Node（`quant_saved_strategy`）。 | 同上 |
| REQ-BT-06 | 讀 OHLCV 必須依 `datetime` 去重、保留最新 `built_at`。（D-034） | `backend/services/quotes.py` |

> 量化回測目前沒有自動化測試（見 OPEN-07）。

### 3.2 模型訓練／推論

| ID | 規範 | 驗收／來源 |
|---|---|---|
| REQ-TR-01 | 訓練以 graph_spec（Feature／Label／Model Node 組成的 DAG）表達；Model Node 可接上游 Model Node 的 `default` 或具名輸出。（D-012、D-018） | `tests/test_named_outputs.py`、`docs/agent-api/training/api.md` |
| REQ-TR-02 | Label＝outcome（原始數值）＋labeling_rule（學習目標）；task_type 與分類類別數由 Label 決定，不是模型參數。（D-014、D-025） | `tests/test_lstm_dual.py` |
| REQ-TR-03 | 可選項與參數契約可查：`GET /api/model/registry/{features,outcomes,labeling_rules,architectures,target_transforms,decision_rules}`、`/registry/components/{registry}`。 | `docs/agent-api/training/api.md` |
| REQ-TR-04 | 模型參數完全由登記的 `params_schema` 驗證：未知欄位、型別／範圍錯誤、條件不適用卻送了，都是 400。UI 與 API 用同一份 schema。（D-023） | `tests/test_model_framework.py` |
| REQ-TR-05 | 模型入口：`lstm`（方向單／雙向 × Attention 零或一個 × 單頭／雙頭，8 種組合）與 `xgboost`。雙向序列摘要取兩方向各自最終 hidden state。（D-023） | `tests/test_lstm.py`、`tests/test_xgboost.py` |
| REQ-TR-06 | 可插拔元件（attention、optimizer）獨立登記，模型以 slot＋`slot_compatibility` 引用，相容性由程式檢查。optimizer 必選；目前只有 Adam（lr、weight_decay 必填；beta1、beta2、eps 有預設）。（D-029） | `tests/test_optimizers.py` |
| REQ-TR-07 | 輪數：神經網路用 `epochs`，boosting 用 `n_estimators`；`final_metrics` 頂層＝best，`last`＝最後一輪。（D-024） | `tests/test_lstm.py` |
| REQ-TR-08 | 產物在訓練成功當下存進 `model_artifacts`（`weights`、`weights_last`、`model_config` 含解析後完整設定與優化器設定、`output_specs`）。（D-015） | `tests/test_artifacts_sql.py` |
| REQ-TR-09 | 評估對整個驗證集計算，結果以 `evaluations` 清單提供：每筆標明模型節點、輸出頭、資料集（切分方式、Phase、任務資料範圍、樣本數、說明）、評估時點（best／last 與輪次、挑選依據）、指標與適用基準（多數類別／訓練集平均／預測零）。舊紀錄的 `evaluation.best／last` 由 API 讀取時轉換，不改寫資料庫，數值必須與原報告相同。（D-036） | `tests/test_evaluation_records.py` |
| REQ-TR-15 | 指標是可共用元件（`training/registry/metrics.py`：label、shape、direction、unit、format），模型以 `metric_specs` 引用並宣告逐輪序列與評估指標；API 回傳每個模型節點的 `metric_specs`（進行中也有）。宣告的序列欄位必須真的由訓練迴圈回報。（D-036） | `tests/test_evaluation_records.py` |
| REQ-TR-10 | 每輪進度寫 `model_training_progress` 並經 LISTEN/NOTIFY → WebSocket 推送；訓練中抽樣預覽另存，與正式推論分開。（D-013、D-016） | `tests/test_lstm_dual.py`（ProgressWrite） |
| REQ-TR-11 | `POST /api/model/infer`：載入產物（`use_weights` 選 best／last）對新區間推論，逐列分批寫入 `model_inference_predictions`；`job.result` 只存摘要。（D-016、D-017） | `tests/test_lstm_dual.py`（InferenceAndStorage） |
| REQ-TR-12 | 樣本同時帶識別時間（`ts`）與資訊可用時間（`available_ts`）；日線 `tx_daily_day`／`tx_daily_full` 可用。（D-019） | `tests/test_available_time.py` |
| REQ-TR-13 | job 管理：`/cancel` 只改 DB 狀態；`DELETE` 連帶刪除進度與產物，有 Phase 2/3 子任務時回 409。 | `docs/agent-api/training/api.md` |
| REQ-TR-14 | 訓練數值決定性：CPU 上同設定重跑結果一致。 | `python -m tests.contract_harness compare` |

### 3.3 Agent API（給 HA）

| ID | 規範 | 驗收／來源 |
|---|---|---|
| REQ-API-01 | `docs/agent-api/` 是 HA 的操作手冊，API 行為改變時必須同輪更新；部署時由 `deploy.ps1` 同步到 NAS `/mnt/Hermesnote/web/hermes/docs/agent-api`。（D-010） | 手冊內範例 payload 應經後端驗證 |
| REQ-API-02 | 研究操作方針：Reproducer／Experimenter 送出的訓練明確帶 `patience: 0`、LSTM `epochs: 300`、優化器參數寫完整；重新啟用早停需 Orchestrator 說明理由並經確認。（D-032） | `docs/agent-api/training/api.md` |

### 3.4 前端

| ID | 規範 | 驗收／來源 |
|---|---|---|
| REQ-UI-01 | 公開頁：`/`、`/quant?job=`（回測結果）、`/model?job=`（訓練進度、best／last、評估報告）、`/hermes`（履歷）；只讀。 | `frontend/src/App.tsx` |
| REQ-UI-02 | 後台 `/admin/*` 需 Google 登入＋email 白名單（`ADMIN_EMAILS`），session JWT 12 小時。（D-007） | `backend/auth.py` |
| REQ-UI-03 | `/admin/quant`：遞迴條件樹編輯器，左 LONG 右 SHORT。 | `pages/admin/sections/QuantSettings.tsx`、`TreeBuilder.tsx` |
| REQ-UI-05 | 指標視覺化前後台共用 `frontend/src/components/metrics/`：依 `metric_specs` 畫逐輪曲線（train／val、best 標記、基準參考線），依 `evaluations` 呈現比較條件、數值比較＋基準、逐類別表、分布、混淆矩陣熱圖；不得依模型種類或指標名稱在頁面寫死判斷。（D-036） | 實際畫面確認（見 record 2026-09-30） |
| REQ-UI-04 | `/admin/model`：畫布（React Flow）組訓練模組 → graph_spec；模型參數表單由 `params_schema` 產生，元件只列相容者；表單只建立 Phase 1，Phase 2／3 與重跑從歷史紀錄觸發。（D-023） | `pages/admin/sections/ModelSettings.tsx`、`SchemaForm.tsx` |

## 4. AI 操作規則

| ID | 規範 |
|---|---|
| OPS-01 | 部署只用 repo 根目錄 `deploy.ps1`，且必須取得 Hermes 當輪明確核准。（D-011） |
| OPS-02 | 正式 DB 的 migration、刪除資料、NAS 清理必須取得核准；執行前確認無 pending／running job。 |
| OPS-03 | 後端重啟、服務狀態檢查、呼叫本機 API 屬 Hermes 主責；只有當輪明確授權（例如「確認正式服務正常」）才做。 |
| OPS-04 | 不得碰 HA 的容器、workspace、runtime、授權設定（含「julie」）；`docs/agent-workflow/` 是 Codex 主責區，不得修改。 |
| OPS-05 | commit／push 依 Hermes 指示；只提交與該輪工作相關的檔案。 |
| OPS-06 | 每輪交付：更新 `docs/record/index.md`（一行摘要）＋`docs/record/archive/<日期>-<主題>.md`（實作、驗證、部署清單、執行紀錄）。 |
| OPS-07 | V1（`D:\hermesnote`）已封存，只有 D-002、D-003 兩項原則可參考，不得沿用其程式或其他設計。（D-004） |
| OPS-08 | 本機 GPU（RTX 3070）與 NAS（RTX 5060 Ti）環境不同，本機 GPU 訓練不算有效驗證；驗證以 CPU 測試與 contract harness 為主。 |
| OPS-09 | 回覆一律繁體中文（台灣）。 |

## 5. 非目標

- 外部註冊、多租戶、公開操作台（SCOPE-03／04）。
- 聯合訓練（跨節點反向傳播）、集成投票、共享編碼器；多模型只支援節點輸出串接。
- 多個 Attention 組合（接點見 `docs/agent-api/training/extending.md`）。
- 即時交易與券商串接（`/admin/trading`、`/admin/ledger` 目前是佔位頁）。

## 6. 待確認（OPEN）

| ID | 內容 |
|---|---|
| OPEN-01 | Agent API 全部端點沒有應用層認證，存取控制只靠網路層；是否加憑證機制未定。 |
| OPEN-02 | Phase 2 請求仍須送 `start`／`end`／`nodes` 佔位值（schema 未改成可省略）。 |
| OPEN-03 | 上游模型輸出給下游時是樣本內預測（非 OOF），有已知洩漏風險，尚未處理。 |
| OPEN-04 | 跨時間框架融合沒有重採樣對齊。 |
| OPEN-05 | 1m → 各時間框架聚合與每日報價匯入管線不在本 repo，來源與工具待確認。 |
| OPEN-06 | HA 提交責任、Critic 審查時機仍在對齊（見 `docs/agent-workflow/index.md`）。 |
| OPEN-07 | 量化回測沒有自動化測試。 |
| OPEN-08 | `hermesnote` DB 有 `model_references` 表（0 筆），程式未使用，用途待確認。 |
