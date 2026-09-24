# 2026-09-18：歷史訓練紀錄巨量 result 效能修正

承接同日 `docs/agent-workflow/record/2026-09-18.md` 記錄的多方排查過程（使用者／Codex／HA
協作定位）。完整排查過程（HUD 誤判、CPU 欄位解析錯誤更正、逐步排除各種假設的過程）見該檔案，
這裡只記錄實際的程式修正。

## 根因

`backend/training/graph.py` 的 Model Node 訓練分支，把 `prediction_source`（逐樣本
train/val/unused 標記，長度等於整個訓練期間的樣本數，例如 2.9M 筆）直接放進
`model_result["prediction_source"]`，`training/worker.py` 的 `run_one_sync()` 存進
`model_training_jobs.result` 之前，原本只用黑名單濾掉 `predict`/`predict_proba`/`weights`
這三個不可序列化欄位，`prediction_source` 因此被原封不動存進這個本來該是輕量摘要的欄位。

5 筆真實規模的 LSTM 訓練任務因此 `result` 欄位（壓縮後）高達 2.3～2.4MB；前台選取這些
歷史紀錄時，`GET /api/model/jobs/{job_id}` 對這個欄位做 `json.loads()` 解析＋重新序列化
成 HTTP 回應，量到單次回應 22,163,941 bytes、耗時 1.40 秒、backend worker 單核心接近
滿載——這是「進前台讀歷史紀錄，NAS 資源就飆高」的直接觸發點，跟訓練是否在跑、
WebSocket 連線、HUD 都無關。（實際重現與量測見上述 agent-workflow 記錄；WebSocket
生命週期是否另有問題，維持獨立、未證實的觀察方向，不併入這次修正。）

## 修改內容

1. **[backend/training/worker.py](../../../backend/training/worker.py)**：`run_one_sync()`
   尾端把黑名單（`_non_serializable`，只排除不可序列化欄位）改成白名單（`_summary_keys =
   {"final_metrics", "device", "training_meta"}`，只放行這三個前台實際會讀的欄位）——
   之後任何 architecture 或 graph.py 新增的欄位，不管是不是像 `prediction_source` 這種
   跟樣本數等長的陣列，預設都不會被存進這個摘要欄位，除非明確加進清單，避免同一種錯誤
   重演。完整訓練產物（權重／`preprocessing_state`／`model_config`）不受影響，繼續完整存
   進 `model_artifacts`（`save_artifacts_for_job()` 用的是過濾前的 `results`）。
2. **[backend/training/job_store.py](../../../backend/training/job_store.py)**：
   `get_job()`（原本 `SELECT *`）與 `list_jobs()` 都改成在 SQL 查詢端用
   `jsonb_each` + `jsonb_object_agg(key, value - 'prediction_source')` 投影掉每個節點
   物件底下的 `prediction_source`，不管是新任務還是資料庫裡已經存在的舊任務都會被投影，
   Postgres 端就把這個欄位剔除，不會整包傳到 Python 再刪——**不需要對舊資料做任何遷移
   或清洗**，舊的 5 筆肥大任務下次查詢就會拿到已經投影過的小欄位。

沒有新增資料表、沒有刪除任何舊任務／模型權重／DB 資料。上一輪的 XGBoost 進度指標修正
（`architectures.py` 的 evals_log key 對齊、loss/accuracy 種類區分）維持不動，兩輪修正
互不衝突（見下方驗證）。

## 驗證結果（全部針對造成 22MB 回應的同一筆真實任務 `3e4d15c1-...`，實際連線正式 NAS DB）

| | 修正前（已量測） | 修正後（本輪量測） |
|---|---|---|
| `GET /jobs/{id}` 耗時 | 1.40 秒 | 0.16～0.18 秒 |
| `GET /jobs/{id}` 回應大小 | 22,163,941 bytes | 1,696 bytes |
| `result` 欄位 `pg_column_size`（SQL 端直接測） | 2,440,674 bytes | 812 bytes |
| `list_jobs(limit=20)`（9 筆真實任務） | 未量測（預期同樣受影響） | 0.47 秒／17,530 bytes，確認無 `prediction_source` |

修正後 `result` 仍保留 `final_metrics`（`loss`/`val_loss`/`val_accuracy` 皆有值）、
`training_meta`（`n_train=2,332,882`／`n_val=583,220`，證實原本 `prediction_source`
陣列長度正是這個量級）、`device`——前台曲線（讀 `model_training_progress`，完全不同
資料表）、模型載入/推論（讀 `model_artifacts`，完全不同資料表）、預覽端點
（`model_training_preview_samples`，同上）均未受這次改動影響，未重新測試是因為程式路徑
上跟這次修改完全無交集，非遺漏。

同時用合成資料重跑上一輪的 `train_xgboost()` 分類流程，確認新的白名單過濾邏輯跟上一輪
XGBoost 指標修正（`loss`/`accuracy`/`val_loss`/`val_accuracy` 逐輪正確回報）互不干擾，
`final_metrics`/`device` 正確通過白名單、`prediction_source` 這類假設性大欄位正確被擋下。

`architectures.py`／`job_store.py`／`worker.py` 語法檢查通過。前端本輪未變動，不需要
重新 build。

## 部署範圍（尚未部署，等使用者確認）

只需要重建 **`hermesnote-training`**（`worker.py` 的白名單邏輯在這個容器執行）跟
**`hermesnote-backend`**（`job_store.py` 的 SQL 投影邏輯在這個容器執行）；`hermesnote-frontend`
這輪沒有變動，不需要重建。
