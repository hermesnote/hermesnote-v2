# 2026-09-21：具名輸出引用（Model Node 之間的平台契約）

背景：研究端要評估以「雙輸出（回歸＋方向機率）聯合訓練」的論文架構。討論後決議，先落地**平台契約**
（具名輸出如何被下游引用），雙頭 architecture 本身、進度 `metrics` 欄、`dual` 推論列、日線資料
正則等等，等研究設定確認後再做。本輪已實作並通過測試，**尚未部署**。

## 設計決議（摘要）

- 輸出名稱用通用語意（`default`／`regression`／`direction_probability`），不含 outcome 名稱；
  輸出實際是什麼由 metadata（`output_specs`）描述：Outcome、單位、horizon、事件規則。
- 方向機率描述為「符合設定規則的機率」，`op` 可配置（`> >= < <=`）；門檻單位固定等於目標單位，
  不做單位換算（`simple_return` 是百分比、`log_return` 是自然對數比值，單位由 outcome registry 宣告）。
- `inputs` 每一項：字串（等於 `default`，Feature 或 Model 皆可，舊圖行為不變）或
  `{"node", "output"}`（只能指向 Model Node，輸出名須在該節點依實際設定宣告的名單內）。
- `default` 的描述反映實際 `output_mode`：回歸 `(n,1)`、分類機率模式 `(n,k)`、分類其他 `(n,1)` 類別索引。

## 修改內容（皆在 `backend/`，前台只放寬型別）

| 位置 | 內容 |
|---|---|
| 新 `training/graph_refs.py` | `normalize_input_ref`／`input_refs`（唯一的 `inputs` 解析器）、`validate_graph_spec`（所有建立／繼承圖入口共用，一次回報全部問題） |
| 新 `training/output_specs.py` | 輸出 metadata 產生與驗證、事件規則（`build_event_rule`）、輸出形狀檢查 |
| `registry/architectures.py` | `register(key, outputs=, describe_outputs=)`、`describe_outputs()`、`list_available()` 帶 `outputs` |
| `registry/outcomes.py`／`labeling_rules.py` | 宣告式 metadata：outcome 的 `unit`／`signed`、labeling_rule 的 `task_type`（不參與計算） |
| `graph.py` | 用共用解析器；具名輸出存進 `named_outputs`；訓練後產生 `output_specs`（同時存進 `model_config` 隨產物保存）並檢查宣告與實際輸出欄數一致 |
| `inference.py` | 欄位改依 `inputs` 原始順序組（見下方「順帶修正」）；上游輸出可指定具名輸出，快取鍵（節點, 輸出）；舊產物沒有 `output_specs` 時要求具名輸出會回明確錯誤 |
| `artifacts.py`／`phases.py` | 改用共用解析器（`phases.py` 的集合推導遇到物件引用原本會 `TypeError`） |
| `routers/model_training.py` | `POST /train`（一般與 Phase 2 繼承後）共用驗證，失敗回 400、不建 job |
| `worker.py` | `run_one` 讀資料之前先驗證；`job.result` 白名單新增 `output_specs` |
| `frontend ModelTrainingPage.tsx` | `ModelNode.inputs` 型別放寬（`ModelSettings` 仍只產生字串） |
| `docs/agent-api/training/api.md` | 補具名輸出契約、metadata 欄位、驗證規則、已知限制 |

**沒有 DB 變更**：每條邊選的輸出已存在 `graph_spec_snapshot`，`output_specs` 存在 `model_config`
（皆為既有 JSONB）。

## 順帶修正的既有風險

`graph.py` 訓練時依 `inputs` 順序串接欄位，`inference.py` 卻是「先全部 Feature、再接上游模型」。
只有 `inputs` 剛好是特徵在前的排列才對得上；API 直接提交、把模型排在特徵前面或中間的圖，推論欄位會
悄悄錯位。管理後台產生的圖一律特徵在前，所以之前沒踩到。現在兩邊用同一個解析器、同一個順序。
測試有對照組證明測試能分辨錯位。若有這種排列的舊產物，需重新訓練（正式庫現有 8 個產物皆不受影響）。

## 驗證

- **前後比對**（`backend/tests/contract_harness.py`，改動前先擷取基準）：資料契約（傳進 architecture 的
  X／y 與切分，逐值相同）、載入舊權重的預測（本機合成產物＋正式庫既有 LSTM／XGBoost 產物，逐值相同）、
  重新訓練指標（固定隨機性；在舊程式碼上實測 LSTM／XGBoost 跑跑之間的波動皆為 0，所以要求逐值相同）——
  **全部一致**。
- **驗收測試**（`backend/tests/test_named_outputs.py`，`python -m unittest tests.test_named_outputs`，
  58 項全過（含下方補正的回歸案例）），對應驗收清單：舊 graph 與輸出模式（對真實 lstm／xgboost 逐模式比對宣告與實際輸出欄數，
  含三分類機率）、Phase 2 繼承、訓練／推論欄位順序（含交錯與具名輸出）、**具名引用 6a 合法（同上游多輸出）
  須成功、6b 非法引用與成環須拒絕**（API 回 400 且不建 job、worker 讀資料前就擋、Phase 2 繼承後再驗）、
  事件規則與單位、`list_available` 與 `job.result` 摘要。新具名輸出用測試專用註冊的確定性小模型驗證，
  不需要雙頭模型。
- 正式庫全部 10 個訓練任務的 `graph_spec` 與 8 個產物快照，皆通過新驗證（唯讀查詢）。
- 前台 `npm run build` 通過。

## GC 審查補正：metadata 產生前的參數型別驗證

GC 重現 Label `params={"horizon": null}`（`TypeError`）與 `params=[1]`（`AttributeError`）會在產生
metadata 時丟出未處理例外。`validate_graph_spec` 現在在呼叫 `describe_outputs`／`label_metadata`
**之前**先驗參數型別（`params` 須為物件；Label 的 `horizon`／`n_classes`／`threshold_pct`、Model 的
`params.n_classes`、`window`、`val_ratio`、`output_mode`，有給才檢查、沒給維持預設），並把 architecture
自己的描述函式丟出的任何意外例外也一律轉成驗證問題。統一結果：`GraphValidationError` → API 400、不建 job，
worker 讀資料前拒絕，Phase 2 繼承後同樣被擋。新增回歸測試共 9 項（總計 58 項全過）：兩個重現案例與其他
壞型別／合法型別仍通過、API 與 Phase 2 與 worker 三個入口、描述函式意外例外；改動前後比對仍全一致；
正式庫 17 份已存 graph_spec／產物快照仍全數通過。仍未部署。

## 部署範圍與未做

部署：`hermesnote-backend`＋`hermesnote-training`；前台只有型別放寬，隨下次前台部署即可。

未做（待研究設定）：雙頭 architecture、`metrics` JSONB 進度欄、推論 `dual` 輸出、日線
`quotes_table` 正則放寬（`daily_day` 目前被拒）、換月處理。上游輸出仍是樣本內預測（非 OOF），
具名輸出不改變這一點。
