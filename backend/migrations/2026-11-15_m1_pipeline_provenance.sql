-- 2026-11-15（M1）：任務的流程版本、環境指紋、期間提示與讀取報告。僅新增 nullable 欄位。
--
-- 性質：全部 `ADD COLUMN IF NOT EXISTS`、不帶預設值、不改既有資料、不刪任何東西；可重複執行；
--   PostgreSQL 11 以上只改 catalog，不重寫整張表。包在單一 transaction。
--   舊任務這四欄維持 NULL：pipeline_version=NULL 代表流程版本 2026-11-r1 以前（不重算、不改寫）。
--
-- 套用順序：**先套用本遷移，再部署 backend／training 新程式碼。**
--   新版 web 建立任務時寫 pipeline_version／warnings，新版 worker 執行時寫 env_fingerprint／data_report，
--   讀取任務的 SQL 也會 SELECT 這四欄——遷移之前部署新程式碼，建立與查詢任務都會失敗。
--   舊程式碼不認得這四欄，先遷移對舊程式碼沒有影響。
--
-- 套用方式（範例，由有權限的人在 NAS 上執行；本檔案尚未對正式資料庫執行過）：
--   psql "$HERMESNOTE_DATABASE_URL" -v ON_ERROR_STOP=1 -f backend/migrations/2026-11-15_m1_pipeline_provenance.sql
--
-- 回復（新程式碼依賴這四欄，回復前要先退回舊程式碼）：
--   ALTER TABLE model_training_jobs DROP COLUMN IF EXISTS pipeline_version, DROP COLUMN IF EXISTS warnings,
--                                   DROP COLUMN IF EXISTS env_fingerprint, DROP COLUMN IF EXISTS data_report;

BEGIN;

ALTER TABLE model_training_jobs
    ADD COLUMN IF NOT EXISTS pipeline_version TEXT,   -- 例如 '2026-11-r1'（training/provenance.py）
    ADD COLUMN IF NOT EXISTS warnings JSONB,          -- 建立時的提示，例如期間超過研究資料上限
    ADD COLUMN IF NOT EXISTS env_fingerprint JSONB,   -- worker 執行時的套件／CUDA／TA-Lib C 版本
    ADD COLUMN IF NOT EXISTS data_report JSONB;       -- 讀取報告：排除的 other、rule_version 分布、缺格數

COMMIT;

-- 套用後驗證（唯讀）：
--   SELECT column_name, data_type FROM information_schema.columns
--   WHERE table_name = 'model_training_jobs'
--     AND column_name IN ('pipeline_version', 'warnings', 'env_fingerprint', 'data_report')
--   ORDER BY column_name;   -- 預期 4 列
