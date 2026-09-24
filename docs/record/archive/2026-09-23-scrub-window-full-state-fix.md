# 2026-09-23（三續）：checkpoint 三項問題補正——完整狀態恢復、多節點重用、提前暫停邊界

補正同日稍早交付的週日 scrub 維護閘門功能的三個問題。**程式已完成，尚未部署。**

## 1. 完整恢復訓練狀態

- `train_lstm`／`train_lstm_bidirectional`／`train_lstm_attention_dual` 的 checkpoint 從「只存模型權重」改成打包保存：模型權重、optimizer 狀態（Adam 動量／二階動差）、RNG 狀態、best 權重／指標／輪次、early stopping 等待計數。resume 時原樣還原，不是重新初始化 optimizer 或從頭累積 best。
- **踩到的真實 bug**：一開始改成用獨立的 `torch.Generator` 承接洗牌用的隨機數，實測發現這會讓 dropout（仍吃全域 RNG）的抽樣序列跟著改變，導致「連續訓練」跟「暫停接續」的最終權重對不上，也讓 `contract_harness.py` 既有的基準比對出現差異。改回**還原全域 RNG 本身**（`torch.set_rng_state`），洗牌與 dropout 維持同一條串流，問題排除，基準比對恢復 ALL MATCH。
- XGBoost 的 early stopping 改成手動追蹤（`best_metric`／`best_iteration`／`wait` 三個純量，存進 checkpoint 的 `extra`，JSON 可序列化），不再用原生 `early_stopping_rounds`——原生機制的內部狀態綁在單一次 `.fit()` 呼叫，continue 訓練（resume）時會重置，沒辦法正確接續「已經連續幾輪沒進步」。`predict`／`predict_proba` 改用手動追蹤的 `best_iteration` 算 `iteration_range`，維持跟原生機制一樣的載入／推論行為。
- 驗收：新增「連續訓練」對「中途暫停恢復」的直接比對測試，涵蓋 LSTM、雙頭 LSTM（best／last／final_metrics 全部比對）、XGBoost（final_metrics／best_iteration），以及「最後一輪才暫停」的邊界情況（resume 進來 0 輪可跑，不出錯不多跑）。

## 2. 多節點圖：已完成節點直接重用

- `TrainingPaused` 新增 `completed_results`：graph.py 在拋出例外前，把「這次暫停之前已經完整訓練完成」的其他節點結果（濾掉不可序列化的 callable）一併附上。
- `worker.py` 在捕捉到暫停時，立刻對這些已完成節點呼叫 `save_artifacts_for_job`（不等整個 job 完成），並把摘要併入 `job_store.mark_paused` 存的 partial result。
- resume 時讀回這份 partial result，得到「哪些節點不用重跑」的清單，傳給 `run_graph(completed_node_ids=...)`；graph.py 對這些節點改成載入已保存產物、對本次資料範圍做一次推論重建輸出（給下游融合節點用），**不呼叫 `train_model()`，不觸發 `on_epoch`／`on_preview`**，不會重跑、不會產生重複的訓練進度紀錄。
- 驗收：新增測試確認「上游節點完成、下游節點暫停」情境下，resume 時上游節點的訓練函式呼叫次數是 0（真的沒有重跑），下游能正確拿到上游的輸出當輸入。

## 3. 收緊 scrub 時間邊界

- `is_scrub_window()` 新增 `lead_minutes` 參數；新增 `SCRUB_LEAD_MINUTES`（5 分鐘）常數與 `should_pause_for_scrub()`（套用緩衝的版本，worker.py／訓練迴圈都改用這個，不是精確版本）。明確寫出時間保證：安全點（一輪／一棵樹／一個推論 chunk）完整跑完才會反應，緩衝讓「即將進入窗口」提前視為已進入，短於緩衝的安全點完全不會跨進窗口，長於緩衝的最多跨進「一個安全點的執行時間」，不是無上限；退出邊界（08:00）精確、無緩衝。
- **執行中推論**現在也涵蓋：`run_inference_chunked` 新增 `should_pause`／`resume_from_index`，每寫完一個 chunk 就檢查一次，暫停時拋出新的 `InferencePaused`（帶著下一筆要接續的位置，不需要保存模型狀態）；`worker.py` 對應處理，checkpoint 存進同一張表（`weights` 欄位放空值佔位）。
- 驗收：新增邊界測試（週日 00:00／08:00 精確邊界、lead_minutes 提前觸發、退出邊界不受影響）與推論暫停測試（暫停後接續，逐筆比對跟連續跑一次完全相同、沒有重複寫入的列）。

## 測試與驗證

新增／擴充 `tests/test_scrub_window.py`（24 項，含連續訓練 vs 暫停恢復比對、多節點重用、推論暫停恢復、時間邊界）。全部既有＋新增測試（`test_named_outputs`／`test_attention_dual`／`test_available_time`／`test_phases_dual`／`test_architecture_families`／`test_evaluation_and_epochs`／`test_scrub_window`）共 **186 項全過**；改動前後基準比對（`contract_harness.py --db`，含正式庫既有產物）ALL MATCH。沒有新增 DB 遷移（`model_training_checkpoints` 表沿用稍早已建立的定義，這輪只改欄位裡存的內容格式，不改 schema）。

## 已知限制（如實保留）

- resume 後 optimizer／RNG 狀態雖然完整還原，但只涵蓋 CPU 路徑實測；GPU（CUDA RNG）路徑的還原邏輯已寫（`torch.cuda.get_rng_state_all`／`set_rng_state_all`），本機沒有 GPU 可以實測，正式驗證留給 NAS 部署後。
- 多節點重用只精準到「節點層級」（整個節點跳過重訓），不是節點內部更細的部分完成快照。
- 時間保證仍是「以安全點為粒度」，不是絕對的秒級保證；lead_minutes 緩衝是實務上的風險緩解，不是數學上的絕對邊界。
