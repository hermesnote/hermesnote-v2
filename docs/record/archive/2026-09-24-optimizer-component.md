# 2026-09-24 優化器改為可共用元件（本輪實作 Adam）

> 狀態：**已部署**（2026-09-24 Hermes 核准後執行，含 2026-09-23 七續的項目）；部署後需登入的人工驗證待 Hermes 執行。
> 前一輪：[2026-09-23-unified-model-entry.md](2026-09-23-unified-model-entry.md)

## 起因

2026-09-23 七續統一模型入口時移除了 Adam 的 betas／eps 設定，而且沒有保存實際值：`model_config` 只記 `lr`／`weight_decay`／`batch_size`，betas／eps 隱含依賴 PyTorch 預設，不利 Reproducer 追溯。這輪把優化器做成跟 Attention 同一套機制的可共用元件。

## 實作

**後端**
- `training/registry/optimizers.py`（新）：`OPTIMIZER_REGISTRY`、`register(key, label, slot_compatibility)`、`build_optimizer(parameters, cfg)`、`list_available()`。Adam 的參數、預設值與合法範圍集中在 `params_schema`：`lr`（必填，>0）、`weight_decay`（必填，≥0）、`beta1`（0.9，[0,1)）、`beta2`（0.999，[0,1)）、`eps`（1e-8，>0）。
- `lstm.py`：`optimizer` 改成 `component_ref`（`component_registry="optimizer"`，必填、不可 null）；`batch_size` 移到頂層；slots 新增 `{"slot_name": "optimizer", "component_registry": "optimizer", "cardinality": "exactly_one"}`；capabilities 新增 `optimizer: configurable`；訓練改用 `build_optimizer()`。XGBoost 宣告 `optimizer: fixed_off`。
- `architectures.py`：`component_registries()` 加入 optimizer；新增 `list_components(registry)`；slot 相容檢查只在 slot 有宣告 `accepts_output_kind` 時比對 `output_kind`（optimizer 沒有輸出形狀）。
- `param_schema.py`：`component_ref` 只有 `default` 為 `null` 的欄位可以送 `null`（attention 可以、optimizer 不行）；元件子參數依元件自己的 schema 驗證並補預設值。
- API：`GET /api/model/registry/attention_types` 改成通用的 `GET /api/model/registry/components/{registry}`（`attention`／`optimizer`，未登記回 404）。
- 產物：`model_config.config.optimizer` 保存 `type` 與補齊預設後的完整實際設定，例如 `{"type": "adam", "lr": 0.001, "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.999, "eps": 1e-08}`。

**前端**
- `ModelSettings.tsx`：依架構 `slots` 宣告的 `component_registry` 自動抓 `/registry/components/{name}`（不再寫死 attention）。
- `SchemaForm.tsx`／`schemaFormLogic.ts`：元件下拉只列與 `(family, slot)` 相容的元件（跟後端同一條規則）；必填元件（optimizer）預選第一個相容元件、沒有「不使用」選項；選定後依元件 `params_schema` 長出欄位（beta1／beta2／eps 帶預設值，lr／weight_decay 必填）。

**測試**：新增 `tests/test_optimizers.py`（8 項）；既有測試 payload 改成新形狀。

## 驗證

- 後端 180 項測試全過（3 skipped）；`contract_harness compare` ALL MATCH（預設 Adam 參數下，訓練數值與改動前逐位元組一致）。
- 預設與自訂參數確實生效：攔截訓練實際建立的 `torch.optim` 物件，`param_groups` 的 `betas`／`eps`／`lr`／`weight_decay` 與設定相符；只改 beta1 時權重不同。
- 完整保存：預設／自訂兩種情況 `model_config.config.optimizer` 都含名稱與五個參數，可 JSON 化；經 `run_graph` 的 random 與 chronological 切分路徑一致。
- 相容性：beta1=1、beta2<0、eps=0、lr=0、weight_decay<0、未知參數、未登記 type、`null`、缺欄位都被拒；登記成只相容其他 family 的優化器被拒（「相容」）；XGBoost 送 optimizer 被拒。
- 擴充性：測試中臨時登記 SGD，不改模型程式即可被 LSTM 使用並保存設定。
- UI（Browser pane 暫時 harness，registry／components 用後端 `list_available()` 匯出，不打本機 API；驗證後已移除）：LSTM 表單出現「優化器＝Adam」與五個欄位；XGBoost 只有自己的 7 個欄位、沒有優化器。擷取兩份 UI payload（預設、自訂 beta1=0.5／beta2=0.95／eps=1e-6／weight_decay=1e-4）送後端驗證並實際訓練，產物設定與送出值一致；API 省略選填欄位時補上預設值。
- 前端 `tsc -b`、`npm run build` 通過；api.md 6 份範例 payload 經後端驗證通過。

## 取捨

1. `batch_size` 從 `optimizer.batch_size` 移到頂層（不屬於優化器參數）——payload 破壞性變更；舊紀錄已確認可捨棄。
2. `lr`、`weight_decay` 維持必填不給預設（研究值不代填）；beta1／beta2／eps 給 PyTorch 預設。
3. `/registry/attention_types` 由通用 `/registry/components/{registry}` 取代，不保留舊端點；HA 文件同步更新。
4. `weight_decay` 是 Adam 原始 L2 形式（非 AdamW 解耦式），文件已標明；要 AdamW 應登記新元件，不改 Adam 語意。

## 部署／遷移／清理清單（待一次核准，未執行；含七續尚未執行項目）

1. **DB migration**（hermesnote DB）：套用 `backend/migrations/2026-09-23_weights_last_and_evaluation.sql`（新增 `model_artifacts.weights_last`）。**必須在部署新後端／training 容器之前**。本輪沒有新增 migration（優化器設定存在既有 JSONB `model_config`）。
2. **清理正式測試資產**（hermesnote DB），依相依順序：
   1. `DELETE FROM model_training_jobs WHERE parent_job_id IS NOT NULL;`
   2. `DELETE FROM model_training_jobs;`（`model_artifacts`、`model_training_progress`、`model_training_preview_samples`、`model_inference_predictions` 以 CASCADE 一併刪除）
   3. 核對上述五張表皆 0 筆；`model_references` 不動。
3. **建置與部署**：本機 `npm run build`（frontend）→ `deploy.ps1`：robocopy 同步 `frontend/dist`、`backend`、`docs/agent-api`，再 `docker compose build backend training && up -d && restart frontend`。training 容器會安裝 `scikit-learn==1.9.0`（七續前加入、尚未部署）；本輪沒有新相依。`backend/tmp_ind_full.json`（非本輪檔案）會隨 `/MIR` 同步，是否先移除由 Hermes 決定。
4. **HA 文件**：步驟 3 的 `docs/agent-api` 同步會把 `training/api.md`（元件端點改為 `/registry/components/{registry}`、optimizer payload、產物保存說明）、`training/extending.md`（新增優化器章節）、`index.md` 更新到 NAS `/mnt/Hermesnote/web/hermes/docs/agent-api`；部署後核對該路徑檔案時間。HA 若有快取舊的 `/registry/attention_types` 用法，需改用新端點。
5. **部署後人工驗證（需登入）**：`/admin/model` 新增 LSTM 模組確認優化器欄位（Adam 預選、beta／eps 預設）→ 送出一筆預設、一筆自訂 beta 的訓練 → `GET /api/model/artifacts` 或產物紀錄核對 `model_config.config.optimizer`；再依七續清單驗證雙頭與 XGBoost，最後交 HA 跑第一份 Reproducer（`patience: 0`、`epochs: 300`、optimizer 明確寫出五個參數）。

## 執行紀錄（2026-09-24，Hermes 核准後）

1. Migration：執行前確認無 pending／running 任務；套用後 `model_artifacts.weights_last` 存在（bytea、可為空）。
2. 清理：單一交易，刪除前 jobs 12／artifacts 10／progress 420／preview 368／inference 0；子任務 0 筆、其餘 12 筆；刪除後五張表皆 0，`model_references` 維持 0，COMMIT。
3. 部署：`deploy.ps1` 前置檢查三個容器皆在跑 → 前端建置 → robocopy 同步（NAS 上舊的 `attention_dual.py`、`test_attention_dual.py` 與舊前端 bundle 依 `/MIR` 移除）→ `docker compose build backend training`（training 安裝 scikit-learn 1.9.0）→ backend／training 重建並啟動、frontend 重啟。
4. HA 文件：NAS `/mnt/Hermesnote/web/hermes/docs/agent-api/` 的 `training/api.md`、`training/extending.md`、`index.md` 已更新（大小與本機一致）。
5. 未做（依協作邊界由 Hermes 執行）：服務狀態／API 檢查、登入後的人工驗證、交 HA 跑 Reproducer。`backend/tmp_ind_full.json` 未處理，已隨同步到 NAS。
6. 提交（Hermes 指示）：把 2026-09-17 以來已部署的訓練系統工作提交到 `main`（後端 `backend/`、前端模型頁與 `SchemaForm`、`docs/agent-api`、`docs/architecture.md`、`docs/record/`、`deploy.ps1`）。刻意不納入：`frontend/public/interview-resume.html`、`docs/index.md`、`docs/Hermes-Agent-Guide/`、`docs/agent-workflow/`（Codex 主責區）、`docs/deploy.bat`、`docs/temp/`、`output/`、`backend/tmp_ind_full.json`——非本工作範圍，留在工作目錄由 Hermes 決定。
