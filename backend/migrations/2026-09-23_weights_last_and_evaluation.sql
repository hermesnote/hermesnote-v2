-- 2026-09-23：best／last 雙權重與完整評估報告支援。
--
-- model_artifacts.weights＝best（監控指標最好那一輪，/infer 預設載入）；weights_last＝最後一輪。
-- 2026-09-23 統一模型入口後，lstm／xgboost 每次訓練都寫入兩組權重，training/artifacts.py 無條件
-- 讀寫這個欄位——**部署新版後端／training 容器之前必須先套用這個遷移**，否則產物寫入會失敗。
-- 可為空（冪等），套用本身不影響既有資料。
--
-- 完整評估報告（分類/回歸指標，見 backend/training/evaluation.py）存在 job.result（既有
-- model_training_jobs.result JSONB 欄位）裡，不需要額外的資料表或欄位。

BEGIN;

ALTER TABLE model_artifacts
    ADD COLUMN IF NOT EXISTS weights_last BYTEA;

COMMIT;
