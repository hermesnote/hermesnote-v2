# 2026-09-23（續）：週日 scrub 維護閘門、統一評估能力、基準補齊、early_stopping 相容性修正

延續同日稍早的架構分層選擇／Attention 元件／best-last 雙權重／V1 評估指標交付。這輪補齊五項後續要求。**程式與遷移檔已完成，尚未部署。**

## 1. 週日 scrub 維護保護

- 新增 `training/scrub_window.py`：`is_scrub_window()` 判定台北時間週日 00:00–08:00。
- 新增 `training/pause.py`（`TrainingPaused` 訊號）、`training/checkpoints.py`（checkpoint 存取，新表 `model_training_checkpoints`，遷移 `migrations/2026-09-23_scrub_window_checkpoints.sql`）。
- `lstm`／`lstm_bidirectional`／`lstm_attention_dual`／`xgboost` 都支援：每輪（或每棵樹）結束後檢查維護窗口，是就存 checkpoint、拋出 `TrainingPaused`；`graph.py`／`worker.py` 接住後把 job 標成新狀態 `paused`（不是 `failed`），不呼叫 `mark_done`。
- `worker.py` 的 `main_loop`：窗口期間不撿任何新工作（含已暫停要接續的）；窗口結束後優先接續 `paused` 任務，從 checkpoint 的下一輪開始，經測試確認不重跑、不跳輪。單一 worker process 天生序列化，`paused` 跟 `pending` 分開查詢，不會重複派工。
- XGBoost 用官方支援的 `fit(xgb_model=...)` 接續機制，實測確認 `n_estimators` 在接續時是「這次還要再長幾棵」，不是總數（10 棵存檔 + 15 接續 = 25 棵，不是 15 棵）。
- 已知簡化：resume 後 optimizer 動量重新初始化；雙頭模型 resume 後 early stopping 的「目前最好」追蹤重新累積；多節點融合圖只精準接續被暫停的那個節點，其餘已完成節點 resume 時會重跑（不做跨節點部分完成快照）。均已寫進文件，不是隱藏限制。
- `POST /jobs/{id}/cancel` 接受 `paused` 狀態；前後台 UI 的狀態列與可點擊判斷都加上 `paused`。

## 2. 統一評估能力

- `training/evaluation.py` 的 `classification_evaluation`／`regression_evaluation` 接上 `lstm`、`lstm_bidirectional`（同一份訓練函式）、`xgboost`（分類＋回歸），不再只有雙頭模型才有；`job.result[node].evaluation` 對這幾個架構是分類或回歸報告本身，雙頭模型維持 `{"best":..., "last":...}` 巢狀（見稍早交付）。
- 確認並測試：`lstm`／`lstm_bidirectional` 的 `patience=0` 已跑滿設定的 epochs（含 300 輪），無需額外開關；XGBoost `patience=0` 同理跑滿 `n_estimators`。

## 3. 基準補齊 + AP 正名

- 分類新增 **`baseline_majority_class`**（train 集決定多數類別，不是驗證集，避免偷看答案）。
- 回歸原本只有「預測訓練集平均值」，改成兩個獨立欄位：**`baseline_mean`**（原本那個）與新增的 **`baseline_zero`**（預測零報酬，不需要訓練集，報酬類目標最單純的參照）。
- `pr_auc` 更名為 **`average_precision`**（AP）：`sklearn.average_precision_score` 是 step function 累加，跟梯形積分算出來的 PR-AUC 是不同的量，不能混用同一個名字誤導比較對象。

## 4. `early_stopping.enabled` 舊設定相容性修正

- 上一輪把 `enabled` 訂為必填，會讓沒有這個欄位的舊 payload／舊產物的 `model_config`／Phase 2 母任務被新驗證拒絕——**已修正**：`enabled` 改為可省略，**省略＝`true`**，維持這個開關加進來之前「提前停止一定開著」的既有語意；要跑滿 epochs（`enabled=false`）是新研究設定要主動選的行為，不是預設值。
- 新增測試涵蓋：舊格式 payload 驗證＋實際訓練行為不變、舊 `model_config`（缺 `enabled` 欄位）仍可正常載入推論、Phase 1（舊格式）繼承成 Phase 2 後原樣通過驗證且不代為補值。

## 測試與驗證

新增／擴充：`tests/test_evaluation_and_epochs.py`（10 項，LSTM／BiLSTM／XGBoost 的評估接線＋300 epochs／patience=0 驗證）、`tests/test_scrub_window.py`（13 項，時區判定、四種架構的暫停／接續、graph.py 節點 id 標記、worker.py 暫停處理、`/cancel` 接受 paused）、`tests/test_attention_dual.py` 新增 backward-compat 與基準重命名相關測試。

全部既有＋新增測試（`test_named_outputs`／`test_attention_dual`／`test_available_time`／`test_phases_dual`／`test_architecture_families`／`test_evaluation_and_epochs`／`test_scrub_window`）共 **175 項全過**；改動前後基準比對（`contract_harness.py --db`，含正式庫既有產物）ALL MATCH；兩份遷移檔以 pglast 檢查語法；前端 `tsc`／`npm run build` 通過。

**UI 驗證的實際範圍**：前端型別檢查與建置通過；後台雙頭表單產生的 payload（含 early stopping 開／關兩種分支）已用真實後端 `validate_params` 交叉驗證通過，證明表單產生的格式跟後端契約一致。**沒有做到瀏覽器登入後的實際點擊操作**——本機起後端服務、呼叫本機 API 屬於 Hermes 的操作邊界（見既有協作邊界記錄），管理後台需要 Google OAuth 登入、本工具沒有帳密，兩者都不是我能自行跨過的限制；建議部署後由 Hermes 實機點擊確認一次雙頭表單、早停開關、評估報告展開、best／last 推論選擇。

## DB 遷移（先遷移、後部署；新增表／欄位皆不影響舊程式）

`backend/migrations/2026-09-23_scrub_window_checkpoints.sql`：新增 `model_training_checkpoints` 表（同步附加在 `model_init.sql`）。連同稍早的 `2026-09-23_weights_last_and_evaluation.sql`，**尚未套用到正式庫**。

## 仍待決定／已知限制

- 多節點融合圖的 checkpoint 目前只精準到被暫停的那個節點，其餘已完成節點 resume 會重跑；如果之後常態出現大型多節點圖橫跨維護窗口，值得再投入做跨節點快照。
- resume 後 optimizer 動量不接續、雙頭模型 early stopping 追蹤不接續，均為已知簡化。
- `evaluation` 現在四個架構都有，但 `lstm`／`xgboost` 目前沒有 best/last 概念（沒有這個需求，維持單一報告）。
