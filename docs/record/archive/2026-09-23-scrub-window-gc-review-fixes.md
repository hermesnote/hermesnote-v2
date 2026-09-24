# 2026-09-23（四續）：GC 檢閱三項問題修正——JSONB 讀回、CUDA RNG、scrub 硬性截止

修正 GC 對稍早交付的週日 scrub 維護閘門／checkpoint 全狀態恢復功能的檢閱發現。**保留已完成的完整狀態保存、多節點重用及評估功能，只修正下列三項。程式已完成，尚未部署。**

**後續補強（同日）**：Hermes 收斂需求為「集中補齊硬中斷的恢復路徑」，在上面三項的基礎上補齊了兩個實際的落地缺口（第 3 節的「硬性截止」原本只設計了骨架，這裡補上讓它真正可靠運作的兩塊拼圖），並新增用真正子程序＋真正強制終止驗證的測試（不是只 mock `should_pause()`）。詳見下面「後續補強：節點級即時落地與推論精確接續」與「真正強制終止子程序的測試」兩節。

## 1. JSONB 讀回錯誤

`checkpoints.py` 的 `load_checkpoint()`／`load_checkpoints_for_job()` 原本對 `extra` 欄位無條件呼叫 `json.loads()`，但 psycopg2 對 JSONB 欄位預設就會自動解碼成 Python 物件（`dict`／`list`）——這個專案其餘讀 JSONB 的地方（`training/artifacts.py` 的 `load_artifact()`）本來就直接當 dict 用，沒有另外 `json.loads()`，這裡是漏改的那一處。GC 用一個回傳 dict 的假游標隔離重現，兩個讀取函式都拋 `TypeError`。

修正：新增 `_decode_extra()`，dict／list 已解碼直接回傳，str／bytes／bytearray 才呼叫 `json.loads()`，None／空字串視為沒有內容。兩個讀取函式都改用它，`save_checkpoint()` 寫入端不受影響。

新增 `tests/test_scrub_window.py::CheckpointExtraDecodeTests`（5 項）：dict／JSON 字串／None 三種格式各自的讀取正確性，以及一個完整模擬「存進去是 dict → 資料庫回讀時原樣拿到 dict」的存讀往返測試（假造的游標模擬 psycopg2 JSONB 自動解碼行為，不是只測讀取端）。

## 2. CUDA RNG 恢復問題

`train_lstm()`／`train_lstm_attention_dual()` resume 時原本用 `torch.load(io.BytesIO(weights), map_location=device)` 整包載入 checkpoint；`device=="cuda"` 時，checkpoint 裡的 `rng_state`（`torch.get_rng_state()` 的回傳值，介面上定義一定是 CPU byte tensor）也會被一併搬到 GPU，再傳給只接受 CPU tensor 的 `torch.set_rng_state()`，行為不對（本機沒有 GPU，這個問題只會在真實 GPU 上出現）。

修正：改成 `map_location="cpu"` 載入整包 checkpoint——`rng_state`／`cuda_rng_state`／`shuffle_rng_state`（雙頭模型的獨立洗牌 generator）三者依 torch 定義本來就都是 CPU tensor，這樣載入後直接可用，不需要另外搬。`model_state`／`optimizer_state` 不需要手動搬裝置：`model.load_state_dict` 用 `copy_()` 寫入已在 device 上的既有參數，`opt.load_state_dict` 內部會把還原的 state 轉成跟對應參數同一個 device，兩者都能正確跨裝置還原。

`device=="cpu"` 時這個改動是 no-op（`map_location="cpu"` 跟 `map_location=device` 結果相同），所以現有全部測試（含「連續訓練 vs 中途暫停恢復」逐位元組比對）不受影響、全部維持通過——這個修正的正確性只能靠邏輯推導與 CPU 路徑驗證，真正的 GPU 驗證留給 NAS 部署後。

## 3. scrub 硬性截止邊界

原本的 `SCRUB_LEAD_MINUTES`（5 分鐘）提前緩衝只是軟性機制——單一安全點（一輪／一棵樹／一個推論 chunk）執行時間比緩衝還長時，仍然可能跨過週日 00:00 才反應，不滿足「絕對禁止跨過截止時間運算」的要求。

修正方向：**軟性機制無法單獨給出硬性保證，補上部署層的硬性第二層**，兩層合起來才是真正的「00:00 之後絕對不會有計算繼續執行」：

1. 第一層（既有，維持不變）：程式自己配合，在安全點暫停並 checkpoint，`SCRUB_LEAD_MINUTES` 提前緩衝降低跨界機率。
2. 第二層（新增）：部署層在週日 00:00:00 台北時間對 training worker 容器執行 `docker kill`（SIGKILL，不等待 graceful shutdown），不管當下算到哪裡直接砍斷；容器重啟策略設成不自動重啟，改由另一個排程在 08:00 之後 `docker start`。

process 被砍掉後，job 會卡在 `status='running'` 沒有任何 process 在跑——新增 `training/job_store.py::fetch_orphaned_running()`＋`training/artifacts.py::list_completed_node_ids()`＋`training/worker.py::_reap_orphaned_running()`（worker process 啟動時呼叫一次）：
- `train` job：從 `model_artifacts` 直接反推哪些節點已經真的落地完成（不信任 `job.result`，process 被殺掉來不及寫入的話那份是空的／過期的），轉成 `paused`，讓下一輪 resume 接續——已完成節點不會重跑，正在訓練的那個節點如果剛好也有 checkpoint 就從那接續，否則從頭訓練這一個節點（不是整個 job 從頭開始）。
- `infer` job：目前的暫停機制只在正常收到暫停訊號時才會存「下一筆要接續的位置」，process 被直接殺掉來不及存，沒有可信的接續點——標記 `failed`，訊息說明原因，使用者重新送出推論請求即可（成本遠低於訓練，直接重來是合理取捨）。

`training/scrub_window.py` 模組開頭文件字串完整重寫，明確列出兩層保證的分工與第二層需要的**四項部署操作**（NAS/主機排程指令、容器重啟策略、時區一致性要求）。`docs/agent-api/training/api.md` 對應更新，也補上 CUDA RNG 修正的說明與研究端「Reproducer／Experimenter 現階段預設關閉 early stopping、未來啟用需 Orchestrator 說明理由」的操作方針（平台本身的相容預設不受影響，仍是省略＝`true`）。

新增 `tests/test_scrub_window.py::OrphanedRunningRecoveryTests`（3 項，其中 infer 那一項後續改寫，見下一節）：train job 有／沒有已完成節點兩種情況都正確轉成 `paused`。

## 後續補強：節點級即時落地與推論精確接續

原本第 3 節的「硬性截止」只設計了骨架（部署層 `docker kill`＋worker 啟動時偵測孤兒 `running` job），實際能不能可靠復原，取決於兩個之前沒補齊的缺口：

1. **多節點訓練的落地時機太晚**：`graph.py` 原本只在整個 `run_graph()` 跑完，或某個節點觸發 `TrainingPaused` 時，才把「這次之前已完成」的節點產物存進 `model_artifacts`。這代表如果 process 剛好在「節點 A 訓練完、節點 B 才剛開始」這個時間點被硬性砍掉（沒有經過任何 `TrainingPaused`），A 的完整訓練成果從來沒被寫進資料庫，重啟後偵測不到它已經完成，只能整個重跑——跟「硬性截止不該讓已完成的進度倒退」的要求不符。

   修正：`run_graph()` 新增 `on_node_complete(node_id, model_result)` 回呼，每個 Model Node **自己**完整訓練完成的那一刻（不是等整個 job 或所有節點都跑完）就呼叫一次；`worker.py` 接上這個回呼，立刻對這一個節點呼叫 `save_artifacts_for_job`（該函式本來就接受任意子集的 `results`，不需要改動它）。回呼失敗（例如資料庫暫時不可用）不會讓訓練本身跟著死掉，只是印警告——`run_one_sync` 結尾仍會對全部節點的最終結果再存一次（`ON CONFLICT DO UPDATE`，冪等），這一步只是「儘早保存」的最佳化，不是唯一的保存路徑。

2. **推論的硬性截止復原原本設計成「標記失敗、要求重新送出」**：因為 `InferencePaused` 帶的 `resume_from_index` 只有在正常收到暫停訊號時才會被存進 checkpoint，process 被直接砍掉來不及存，看起來似乎沒有「可信的接續點」。但實際檢查 `run_inference_chunked` 才發現：每算完一批（chunk）就呼叫 `on_chunk` 把這批寫進 `model_inference_predictions`，這一步跟有沒有正常收到暫停訊號完全無關——不管是正常暫停還是硬性截止，已經寫完的 chunk 一定已經落地（單一 `execute_values` INSERT，Postgres 保證單一陳述式的原子性，不會有半個 chunk 的殘留列）。

   修正：新增 `inference_store.count_predictions(job_id, node_id)`，直接 `COUNT(*)` 反推「已經寫了幾筆」當 `resume_from_index`——跟正常暫停時存的那個索引意義完全相同，不需要事先存過 checkpoint 也能精確接續。`worker.py` 的 `_reap_orphaned_running()` 對 `infer` job 改成：查這個數字、存成一個合成的 checkpoint（沿用既有的 `model_training_checkpoints` 表與既有的 resume 讀取邏輯，不需要另外改 `run_one_infer`）、標記 `paused`——不再是標記 `failed` 要求使用者重新送出。

`training/scrub_window.py`、`docs/agent-api/training/api.md` 都同步更新，明確列出硬性截止實際可能損失的進度範圍（多節點訓練：只有「正在訓練、還沒完成」的那一個節點會損失，範圍限定在它自己最後一個安全點之後；推論：只有「正在算、還沒寫完」的那一個 chunk 會損失），不誇大、不隱藏。

## 真正強制終止子程序的測試

新增 `tests/test_hard_kill_recovery.py`（2 項），用 `subprocess.Popen` 真正啟動獨立的 Python 子程序、真正呼叫 `Process.kill()`（Windows 上是 `TerminateProcess`，沒有任何機會執行收尾程式碼）強制終止，不是像其餘測試那樣用 `should_pause()` 回傳 `True` 模擬「程式自己配合暫停」的軟性路徑：

- **多節點訓練**（`_hard_kill_train_harness.py`）：兩個節點 up→down，up 用快速假架構、down 用真的會 `sleep` 的慢速架構製造可中斷視窗；等 up 完成並確認產物已經落地保存之後，才真正砍掉子程序；驗證被砍後 up 的產物完好、down 沒有任何殘留；接著啟動第二個獨立子程序模擬 resume，驗證：橫跨兩個獨立 process 的完整呼叫紀錄裡，up 的訓練函式總共只被呼叫一次（resume 那次完全沒有重跑），down 被呼叫兩次（中斷前一次沒有完成、resume 後重新從頭訓練一次），最終兩個節點都有完整結果。
- **推論**（`_hard_kill_infer_harness.py`）：對一批合成資料跑 `run_inference_chunked`，每個 chunk 前插入真的 `sleep`；等已經寫完一部分 chunk 之後才真正砍掉；驗證被砍後已寫入的列完好無缺；resume 時用「目前已寫入的行數」當 `resume_from_index` 接續，驗證最終全部時間戳恰好各出現一次、順序跟原本推論順序一致、總筆數等於完整跑一次該有的筆數。

持久化層用本地檔案取代真實資料庫（這個環境連不到 NAS 上的正式 Postgres，見下方「已知限制」），但驅動的是 `training/graph.py`、`training/inference.py` 的正式程式碼（`run_graph()`、`run_inference_chunked()`），不是為測試另外寫的精簡版——驗證的是真正的 `on_node_complete`／`completed_node_ids`／`resume_from_index` 復原邏輯本身，只是把「存進 model_artifacts／model_inference_predictions」換成「寫本地檔案」。

## 測試與驗證

`tests/test_scrub_window.py` 新增 10 項（`CheckpointExtraDecodeTests` 5 項＋`OrphanedRunningRecoveryTests` 3 項，其中 infer 那一項改寫為驗證「用 `COUNT(*)` 反推 `resume_from_index` 並標記 `paused`」＋`OnNodeCompleteImmediateSaveTests` 2 項，驗證 `on_node_complete` 依拓樸順序、每個節點完成立刻各呼叫一次，且回呼本身出錯不會讓訓練跟著死）；另外新增 `tests/test_hard_kill_recovery.py`（2 項，真正子程序＋真正強制終止，見上一節）。全部既有＋新增測試（`test_named_outputs`／`test_attention_dual`／`test_available_time`／`test_phases_dual`／`test_architecture_families`／`test_evaluation_and_epochs`／`test_scrub_window`／`test_hard_kill_recovery`）共 **198 項全過**。CUDA RNG 修正在 CPU 路徑上是 no-op，沒有另外跑 `contract_harness.py` 基準比對（這個修正不影響 CPU 訓練的任何數值路徑，全部既有的逐位元組比對測試維持通過就是等價的驗證）。

## 已知限制（如實保留）

- 第二層硬性截止的部署操作（NAS Cron Job／容器重啟策略調整）尚未由 Hermes 實際設定，目前只有程式碼與說明文件；真正生效需要照 `scrub_window.py` 文件字串列出的四個步驟在 NAS 上操作。
- CUDA RNG 修正的正確性只驗證了邏輯（`map_location="cpu"` 是三個 RNG tensor 正確的載入方式），沒有真實 GPU 可以實測，留給 NAS 部署後驗證。
- 真正子程序強制終止測試用本地檔案取代真實資料庫——這個開發環境目前連不到 NAS 上的正式 `hermesnote` Postgres（`HERMESNOTE_DATABASE_URL` 在這裡實測解析成 `localhost`，連線被拒），沒辦法對真正的 DB 連線做同等的強制終止測試；驗證的是復原邏輯本身（跟持久化層是檔案還是 Postgres 無關），不是「真的對正式 NAS DB 跑過一次」。
- 硬性截止可能損失的進度範圍已在 `scrub_window.py`／`api.md` 明確列出：多節點訓練損失範圍限定在「正在訓練、還沒完成的那一個節點，從它自己最後一個安全點之後」；推論損失範圍限定在「正在算、還沒寫完的那一個 chunk」——兩者都不是「整個 job 從頭開始」，但也不是零損失，如實記錄不誇大。
