# 2026-09-23（七續）單一模型設定介面＋通用擴充框架

> 狀態：程式與本地文件完成；**尚未部署、尚未套用 DB migration、NAS 測試資產尚未清理**（待一次核准，清單見文末）。
> 取代：同日五續（`lstm_custom`）、六續（獨立選擇／schema UI）與 2026-09-21／09-23 首輪的 `lstm_attention_dual`／`lstm_bidirectional` 設計；那幾則記錄僅供追溯。

## 目標

一個 LSTM 入口涵蓋層數／寬度、單雙向、Attention、單雙頭、訓練參數；LSTM 與 XGBoost 共用同一套登記、參數 schema、表單渲染與後端驗證；未來 Transformer／Mamba／新 Attention 走同一框架擴充。Phase 1／2／3 流程、切分與評估契約不變。完成後供 HA 跑第一份正式 Reproducer。

## 實作

**後端（`backend/training/registry/`）**
- `param_schema.py`（新）：FieldSpec 驗證器，UI 與後端共用語意——dot path、`required`／`visible_if` 條件（`all`／`any`、`$task_type`／`$outcome_unit` 情境）、`enum_cases`、`exclusive_min/max`、`derived_from`、`component_ref`（子參數用元件自己的 schema 驗證；清單一律拒絕）、`array_of_object`；不適用卻送了、未知欄位都是錯誤。可選值依情境縮小時，錯誤訊息附欄位說明。
- `architectures.py`：`register(key, family, label, params_schema, extra_checks, capabilities, slots, outputs, describe_outputs, lazy_windows)`；`validate()`＝schema → slot 相容性（元件 `slot_compatibility`＋`output_kind`）→ `extra_checks`；`list_available()` 公開 schema／能力／slots。
- `lstm.py`（新，唯一 LSTM）：三軸獨立 8 種組合；雙向摘要用 `h_n` 兩方向最終狀態；`heads` 在分類 Label 時只能 `single`；監控指標依模式切換（預設順序 `val_loss` 在前）。
- `xgboost_model.py`（新）：改用原生 `xgb.train`，類別數固定取 Label 的 `n_classes`（修正 3 類、`threshold_pct=0` 時 sklearn 包裝因 y 缺類別直接報錯）；best＝截到 best round、last＝全部樹；多分類改 `multi:softprob`，機率可重載。
- `training_control.py`：`early_stopping_fields()`（`patience` 0＝關閉）＋`BestTracker`；`n_classes` 改由 Label 決定、不再是模型參數。
- `attention.py`：元件宣告 `params_schema`／`output_kind`／`slot_compatibility`，`build_checked()` 建網路時實際跑一次形狀檢查；多元件組合只記錄接點。
- 刪除：`lstm_attention_dual`、`lstm_bidirectional`、`lstm_custom` 三個舊 key 與 `attention_dual.py`、`lstm_custom.py`。
- 移除只為舊版／未遷移環境相容的分支：`artifacts.py` 的 `weights_last` 欄位存在檢查（改為部署前必須先套用遷移）、`progress.py` 的「metrics 欄可能不存在」分支（正式庫已有此欄）、推論 `use_weights` 的「只記非預設值」與舊 infer job 預設值。
- NaN 修正（harness 驗證時抓到）：多分類驗證集缺類別時 sklearn 的 OvR ROC-AUC 回傳 NaN，而 `json.dumps` 寫出的 NaN 字面值 PostgreSQL JSONB 不接受，訓練跑完會寫不進結果。`evaluation.py` 改回 `null`＋原因；新增 `training/json_safe.py`，job 結果、每輪擴充指標、產物 `model_config` 寫入前一律把非有限值轉 `null`。
- 提交錯誤前綴「輸出設定不合法」改為「模型設定不合法」。

**前端**
- `SchemaForm.tsx`＋`schemaFormLogic.ts`（新）：通用表單，語意鏡像後端；LSTM、XGBoost 共用。
- `ModelSettings.tsx`：刪除 `DualHeadEditor`、`ARCHITECTURE_PARAM_FIELDS`、扁平參數欄位、`dualConfig`、`epochs`／`bidirectional` 專用欄位；每個架構各存一份表單值；`params` 純由 schema 產生（不再包 `epochs`／`n_classes`）；`output_mode` 只在單頭分類送；提交失敗時顯示後端 400 的完整問題清單；推論一律送 `use_weights`。
- `ModelTrainingPage.tsx`：雙頭判斷改看送出的 `params.heads`；所有架構顯示 best／last 摘要與完整評估報告；模型參數顯示改成完整 params。
- `evaluationRows.ts`（新）：兩頁共用的評估報告顯示邏輯（單頭分類／單頭回歸／雙頭三種形狀）。

**測試**：新增 `test_lstm.py`（8 組合、雙向摘要正確性、單頭、訓練控制）、`test_xgboost.py`、`test_model_framework.py`（validator、registry、相容性、擴充性）、`test_artifacts_sql.py`（取代 `test_artifacts_weights_last_compat.py`）；`test_attention_dual.py` 改寫為 `test_lstm_dual.py`；刪除舊 key 專屬測試。

## 驗證

- 後端：172 項測試全過（3 skipped）；`contract_harness compare` ALL MATCH（資料契約／推論／訓練決定性，含 XGBoost 改用原生 API 後數值不變）。
- 8 種 LSTM 組合：訓練、宣告輸出形狀、best／last 權重與評估、保存重載推論逐值一致、`sequence_summary` 記錄；經 `run_graph` 雙頭輸出 `direction_probability` 被下游 XGBoost 具名引用。
- XGBoost：二元／三元（含驗證集缺類別）／回歸，`patience` 0／正整數，best／last、重載一致。
- 前端：`tsc -b`、`npm run build` 通過。
- **UI 實際操作**（Browser pane，暫時 harness 掛真正的 `ModelSettings`／`ModelTrainingPage`，registry 回應用後端 `list_available()` 匯出、訓練紀錄用後端真的跑 `run_graph` 的結果；不打任何本機 API；驗證後 harness 已移除，保存在 session scratchpad）：
  - 表單依 schema 顯示／隱藏：分類 Label 時 `heads` 只有 `single`、有 `class_weight`／`output_mode`；雙頭時出現 `direction_rule`／`loss_weights`／`target_scaling`、監控指標換成雙頭清單、`threshold_unit` 跟隨 Label（`log_ratio`）；選 Attention 後長出 `dim`；XGBoost 只顯示自己的 7 個欄位、沒有 epochs；切換架構後已填值保留。
  - 擷取 4 份 UI payload 送後端：單向單頭分類、雙向＋Attention＋雙頭回歸、XGBoost 3 類 → 提交驗證通過並實際訓練，best／last／evaluation 齊全；第 2 層留空的 payload 被拒（`lstm_layers[1].units 為必填`）。
  - 訓練頁：雙頭（最新一輪指標、best／last 摘要、方向頭／回歸頭報告）與 XGBoost 3 類（缺類別 ROC-AUC 顯示原因）正確顯示。
- 文件：api.md 內 6 份範例 payload 全數用後端 `validate_graph_spec`／`validate` 驗證通過。
- 唯讀查證正式 `hermesnote` DB：2026-09-21 遷移欄位（`metrics`、`*_available_ts`、`output_type`、`probabilities`）已存在；`model_artifacts.weights_last` **不存在**；子表對 `model_training_jobs` 皆 `ON DELETE CASCADE`，`parent_job_id` 為 `NO ACTION`；現有 jobs 12（train done 10、failed 1、infer failed 1）、artifacts 10（lstm 7／lstm_attention_dual 2／xgboost 1，無收藏）、progress 420、preview 368、inference 0。

## 需要人工登入才能做的驗證

後台需要 Google 帳號登入、本機 API 呼叫屬 Hermes 操作邊界，以下由 Hermes 在部署後執行：開 `/admin/model` → 新增模組 → 依序送出（a）LSTM 單向單頭分類 `patience 0`、（b）LSTM 雙向＋Attention＋雙頭（`log_return`＋`identity`）、（c）XGBoost 分類 → 確認歷史紀錄的完整評估報告、`/model?job=` 的 best／last 顯示、以 last 權重推論一次。

## 部署／遷移／清理清單（待一次核准，未執行）

1. 套用 `backend/migrations/2026-09-23_weights_last_and_evaluation.sql`（`ALTER TABLE model_artifacts ADD COLUMN IF NOT EXISTS weights_last BYTEA`）。**必須在部署新後端／training 容器之前**。
2. 清理正式測試資產（hermesnote DB），依相依順序：
   1. `DELETE FROM model_training_jobs WHERE parent_job_id IS NOT NULL;`（Phase 2／3 子任務；`NO ACTION` FK 必須先刪）
   2. `DELETE FROM model_training_jobs;`（其餘 job；`model_artifacts`、`model_training_progress`、`model_training_preview_samples`、`model_inference_predictions` 以 CASCADE 一併刪除）
   3. 刪除後核對五張表皆為 0 筆；`model_references`（0 筆）不動。
3. 本機 `npm run build`（frontend）後執行 `deploy.ps1`：robocopy 同步 `frontend/dist`、`backend`、`docs/agent-api`（HA 讀的 `training/api.md` 與新增的 `training/extending.md` 一併更新到 NAS `/mnt/Hermesnote/web/hermes/docs/agent-api`），再 `docker compose build backend training && up -d && restart frontend`。training 容器的 `requirements.txt` 含 `scikit-learn==1.9.0`（評估指標用；先前輪次加入、尚未部署，這次建置會安裝），本輪沒有其他新相依。注意 `backend` 是 `/MIR` 同步，工作目錄裡未追蹤的 `backend/tmp_ind_full.json`（非本輪檔案）也會被複製上去，是否先移除由 Hermes 決定。
4. 部署後 Hermes 執行上一節的人工驗證，再交 HA 跑第一份 Reproducer。
