# 2026-09-23：架構分層選擇、Attention 可插拔元件、Early Stopping 開關、V1 評估指標補齊

延續同月稍早的雙頭 Attention LSTM 能力擴充。中途曾一度誤解 Reproducer 方向、做出一批「為了完全貼合論文而擴充平台切分契約」的變更（TA-Lib input 選擇、`date_range` 切分、四個新特徵、成交量截尾、metadata 端點）——已依 Hermes 指示**完整撤回**（獨立 patch 保存於 [2026-09-23-daily-full-tuning-extension-withdrawn.patch](../../temp/2026-09-23-daily-full-tuning-extension-withdrawn.patch)），三階段 Phase 1/2/3 契約與既有切分模式維持原樣不變。本輪是收斂範圍後的正式交付。**程式與遷移檔已完成，尚未部署。**

## 1. 架構分層選擇（family／variant）

- `registry/architectures.py` 的 `register()` 新增 `family`／`variant`／`label` metadata；`key` 本身維持扁平字串、格式不變——分層只在查詢端（`GET /registry/architectures`），不影響已存在的訓練紀錄、Phase 2 繼承、HA 現有 script。
- 新增 `lstm_bidirectional`：LSTM 家族底下的獨立選項，結構與 `lstm` 完全共用，差異只是雙向固定開啟；不接受 `params.bidirectional`（提交前與執行期皆擋）。既有 `key="lstm"`＋`bidirectional` 相容路徑不受影響。
- 後台「建立訓練」的架構下拉選單改依 family 分組（`<optgroup>`）。

## 2. Attention 可插拔元件

- 新增 `training/registry/attention.py`：跟 features/outcomes 同一套登記模式，每個型態宣告 `validate_params`／`build`（回傳 `nn.Module`，統一 `forward(seq)->(context, weights)` 契約）／`query_source`／`combine`／`compatible_families`／`label`。
- 現有的 query-free additive attention（`e_t=vᵀtanh(W_a h_t+b_a)`）從 `attention_dual.py` 抽成獨立元件 `"additive"`，數學完全沒變（已用測試核對 state_dict 結構調整後行為一致；沒有正式庫產物依賴舊的 state_dict 佈局）。
- `lstm_attention_dual` 的 `params.attention.type` 改為查 registry 動態驗證（不再寫死 `"additive"`），選擇連同該型態的完整宣告存進 `model_config.attention_meta`。
- 新增 `GET /api/model/registry/attention_types`。後台新增設定入口顯示可選型態。
- 混合 Attention：沒有明確的合併規則前不假造，未登記任何型態。

## 3. Early stopping 開關 + best／last 雙權重

- `params.early_stopping.enabled`（必填布林）：`true` 維持原行為；`false` 固定跑滿 `epochs`（可設 300），`monitor`／`min_delta` 仍必填（追蹤但不中斷），`patience` 非必填。
- 不論開關與否，「最佳一輪」與「最後一輪」的權重都分開保存並各自完整評估（`final_metrics` 頂層＝best，`final_metrics.last`＝last）。
- 新增 `model_artifacts.weights_last`（可為空，遷移檔 `migrations/2026-09-23_weights_last_and_evaluation.sql`）；`training/artifacts.py` 對缺欄位環境（遷移未套用）做讀取期防禦（比照既有 `bar_end_ts` 的 `information_schema` 檢查手法），避免破壞尚未遷移的環境。
- `POST /infer` 新增可選 `use_weights`（`"best"`／`"last"`），沒有保存對應權重回 400；不影響既有 payload（預設值不寫進 job 存檔，保持既有格式逐位元組相同）。後台推論表單新增對應下拉選單。

## 4. V1 評估指標補齊

- 新增 `training/evaluation.py`：`classification_evaluation`（類別分布、混淆矩陣、precision/recall/F1、macro-F1、balanced accuracy、ROC-AUC、PR-AUC）、`regression_evaluation`（MSE/RMSE/MAE/R²＋「預測訓練集平均值」簡單基準）。任何算不出來的指標回 `null` 加 `*_unavailable_reason`，不假裝算過。
- 目前只有 `lstm_attention_dual` 的訓練流程實際呼叫這兩個函式（方向頭走分類報告、回歸頭走回歸報告，best／last 各一份），結果放進 `job.result.evaluation`（O(類別數) 大小，跟既有輕量摘要原則不衝突）；`lstm`／`xgboost` 尚未接線，如實記錄為缺口，之後可直接重用同一組函式。
- 後台歷史紀錄與前台即時訓練頁都新增「展開完整評估報告」區塊。

## 測試與驗證

- 新增 `tests/test_architecture_families.py`（14 項）：family/variant metadata、`lstm_bidirectional` 與 `lstm+bidirectional=True` 逐位元組相同、拒收 `bidirectional` 參數、既有相容路徑不受影響。
- `tests/test_attention_dual.py` 擴充至 49 項：Attention registry 選用與 metadata 保存、early stopping 開關兩種狀態、best／last 雙權重的載入與各自評估、完整評估報告的欄位與「無法定義原因」語意。
- 全部既有＋新增測試（`test_named_outputs`／`test_attention_dual`／`test_available_time`／`test_phases_dual`／`test_architecture_families`）共 **148 項全過**；改動前後基準比對（`contract_harness.py --db`，含正式庫既有產物）ALL MATCH；遷移檔以 pglast 檢查語法；前端 `tsc`／`npm run build` 通過。
- 撤回的擴充內容：撤回後逐項 `grep` 確認無殘留，148 項回歸測試與基準比對在撤回後、本輪交付前都各自重跑過一次，全部通過。

## DB 遷移（先遷移、後部署；可為空的新增欄位，舊程式不受影響）

`backend/migrations/2026-09-23_weights_last_and_evaluation.sql`：`model_artifacts.weights_last BYTEA`（同步附加在 `model_init.sql`）。**尚未套用到正式庫。**

## 仍待研究端／後續決定

- `evaluation` 報告尚未接上 `lstm`／`xgboost`（只是還沒接線，不是設計限制）。
- 完整圖形化設定表單目前只有 `lstm_attention_dual` 做了專屬區塊；未來新架構（Transformer、混合模型）比照 family/variant 機制登記即可，UI 端要另外做對應表單。
- Attention 打分式是否含 query，仍待對照論文原文核對（沿用 2026-09-21 那次的既有記錄，未解決）。
