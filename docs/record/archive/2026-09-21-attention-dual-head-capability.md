# 2026-09-21：Attention＋雙頭 LSTM 能力擴充（日線讀取、雙頭訓練／推論、指標、時間語意）

延續同日的具名輸出平台契約（[named-outputs-contract](2026-09-21-named-outputs-contract.md)），本輪完成讓 HA 經 API 用日線資料訓練、保存、重載、評估
Attention＋雙頭 LSTM 所需的全部能力。研究設定來源：HA workspace `temp/attention-dual-head-lstm-research-settings-v003.md`
（草案≠已核准實驗；論文原文明示／研究端建議／未確認項目分開處理）。**正式庫遷移已於 2026-09-21 套用（Hermes 核准），部署由 Hermes 執行；未提交任何訓練。**

## 內容

| 項目 | 結果 |
|---|---|
| 日線讀取 | `timeframe` 白名單放行底線（`daily_day`／`daily_full`），symbol 仍只允許英數；`fullmatch` 擋尾端換行；資料表與資料不動；以唯讀方式驗證過真實讀取路徑（`RUN_DB_TESTS=1`） |
| 時間語意 | 識別時間（`datetime`，日線 08:45）與資訊可用時間（`bar_end_ts`，13:45）分開；樣本 `ts` 仍是識別時間，另帶 `available_ts`（預覽 `decision_available_ts`／`target_available_ts`、推論列 `available_ts`）；只標時間，不改計算與切分 |
| 時間語意修正（GC 檢閱後） | 決策可用時間沿依賴鏈取全部輸入（Feature＋上游 Model）最晚者，目標可用時間改取 Label 自己資料的 `bar_end_ts`（原本誤用 Feature 的時間）；任一輸入未知則為未知；訓練與推論同一規則。缺 `bar_end_ts` 的資料表由 `fetch_ohlcv_dataframe` 先查 `information_schema`，沒有就不 SELECT、視為未知（僅 `spread` 系列表缺此欄，且非 OHLCV 表）。新增 `tests/test_available_time.py` 16 項（含 GC 的 5h／7h 案例、三層鏈、推論鏈、缺欄位假連線、DB 唯讀真實 schema） |
| 新架構 | `lstm_attention_dual`：疊層 LSTM（層寬由 payload）＋每層輸出後 dropout、query-free additive attention、與最後 hidden 串接、shared Dense+ReLU、回歸頭＋方向 logit 頭、`w_reg·MSE + w_dir·BCEWithLogits`；公式與 dropout 位置寫入 `model_config.formula` |
| 目標／方向 | Outcome／horizon 留在 Label Node；方向規則由 `target_transforms.sign_threshold` 明確指定，在原單位計算；`target_scaling`（可選 standardize，train-only fit）保存並還原，方向標籤與門檻不受縮放影響 |
| 必填不預設 | `shared.dense`、`epochs`、`seed`、optimizer、early stopping monitor／patience 等研究端未確認值一律必填 |
| 指標 | `joint_loss`／`mse`／`bce`／`rmse`／`dir_acc` 各自命名，逐輪存 `model_training_progress.metrics`（JSONB）；early stopping（val_rmse、patience 2、還原最佳權重）；`final_metrics` 對應還原後的最佳模型並區分 `best_epoch`／`stopped_epoch`；`job.result` 保持輕量 |
| 推論 | 儲存→重載→`dual` 推論→分批入庫→API 讀回；`predicted`＝回歸值、`probabilities`＝`[P(非事件), P(事件)]`；下游可用具名輸出任選其一；訓練與推論欄位順序一致 |
| 提交前驗證 | 參數（型別／範圍／必填／未知欄位）、目標單位、輸出引用皆在 POST /train 與 worker 讀資料前驗證，失敗 400 不建 job |
| 前台 | `ModelTrainingPage`／`ModelSettings` 辨識雙頭模型：聯合損失與方向準確率不再被標成 CrossEntropy／accuracy，每輪擴充指標、最佳／停止輪、預覽與推論的識別時間 vs 資訊可用時間、`dual` 推論列。完整的圖形化設定表單另列後續 |
| 後台重跑 | 後台歷史紀錄每筆（非執行中）新增「重跑」：Phase 1／未標 Phase 原樣重送該筆 graph_spec；Phase 2 沿用同一個 Phase 1 母任務；Phase 3／推論用同一模型與同一區間重新推論（Phase 3 仍帶 phase=3）；新增一筆紀錄、原紀錄不動；改時間框架或特徵屬另一個實驗，走「建立訓練」。純前端改動，後端不變。新增 `tests/test_phases_dual.py` 11 項（雙頭走 Phase 1→2→3、路由血緣與 holdout 邊界、重跑等價） |
| 文件 | `docs/agent-api/training/api.md` 新增日線與時間語意、雙頭架構契約、完整 payload 範本、結果範例、錯誤訊息、部署順序 |

## DB 遷移（先遷移、後部署；皆為可為空的新增欄位，舊程式不受影響）

`backend/migrations/2026-09-21_attention_dual_head.sql`（同步附加在 `backend/model_init.sql`，皆冪等）：
`model_training_progress.metrics JSONB`、`model_training_preview_samples.decision_available_ts／target_available_ts`、
`model_inference_predictions.available_ts`。**已套用到正式庫（2026-09-21，四欄位皆確認存在）。**

## 驗收

離線合成資料：`tests.test_named_outputs`＋`tests.test_attention_dual` 共 116 項（含 `tests.test_available_time` 16 項）（2 項為需 `RUN_DB_TESTS=1` 的唯讀真實日線讀取測試，另跑一次通過）；
改動前後基準比對（資料契約、含正式庫既有產物的權重載入預測、固定種子 CPU 重訓指標）全一致；遷移檔以 pglast 檢查語法；前端 `npm run build`＋`tsc` 通過。
測試一律以 monkeypatch 隔離，不寫正式庫。

## 仍待研究端決定（平台不代為決定）

日夜盤（`daily_day`／`daily_full`）選擇、換月（連續月）調整（實測 186 次換月未調整，換月日報酬明顯偏大）、研究期間與切分、shared Dense 寬度、epochs、seed、
Attention 式 1 是否含 query（尚未對論文原文核對，目前採不含 query 並如實記錄）、混合精度（草案標 optional，未實作）。
HA 建議的固定技術特徵（GARCH／HMM 等）未核准，平台僅提供組合既有 Feature，不實作、不宣稱重現論文。

## 已知限制（未解決，如實保留）

上游模型輸出仍是對自身訓練資料的樣本內預測（非 OOF），多模型串接的洩漏風險不因具名輸出而解決；`/cancel` 仍只改 DB 狀態。
