-- 2026-09-21：Attention＋雙頭 LSTM 能力擴充所需的資料表變更（僅新增 nullable 欄位）
--
-- 性質：全部是 `ADD COLUMN IF NOT EXISTS`、不帶預設值、不改既有資料、不刪任何東西——
--   * 可重複執行（idempotent），重跑不會出錯也不會重複加欄位。
--   * 不帶 DEFAULT 的 nullable 欄位在 PostgreSQL 11 以上只改 catalog，不重寫整張表，鎖定時間極短；
--     正式庫是 PostgreSQL 17，訓練中也可以套用（只會短暫等待表上的鎖）。
--   * 包在單一 transaction：任何一句失敗整份 rollback，不留半套。
--
-- 套用順序：**先套用本遷移，再部署 backend／training 新程式碼。**
--   新版 training 寫進度時（只有雙頭模型這類有擴充指標的才會寫 metrics 欄）、寫預覽與推論時會用到
--   下面這些欄位；舊程式碼完全不認得它們，所以「先遷移、後部署」對兩邊都安全。
--   backend 讀取進度／推論結果的 SQL 已改成 SELECT * 並用 row.get()，遷移之前也不會 500，
--   但預覽與推論寫入在遷移之前會因欄位不存在而失敗。
--
-- 套用方式（範例，由有權限的人在 NAS 上執行；本檔案尚未對正式資料庫執行過）：
--   psql "$HERMESNOTE_DATABASE_URL" -v ON_ERROR_STOP=1 -f backend/migrations/2026-09-21_attention_dual_head.sql
--
-- 回復（如需要；這些欄位新舊程式碼都不依賴其內容，通常不需要回復）：
--   ALTER TABLE model_training_progress         DROP COLUMN IF EXISTS metrics;
--   ALTER TABLE model_training_preview_samples  DROP COLUMN IF EXISTS decision_available_ts,
--                                               DROP COLUMN IF EXISTS target_available_ts;
--   ALTER TABLE model_inference_predictions     DROP COLUMN IF EXISTS available_ts;

BEGIN;

-- 每輪訓練的「擴充指標」（各自有明確名稱，不借用 loss／accuracy 欄裝別的東西）：
-- 例如雙頭模型的 joint_loss、mse、bce、rmse、dir_acc 及其 val_ 版本。既有 LSTM／XGBoost 只有舊四欄，
-- 這欄維持 NULL。
ALTER TABLE model_training_progress
    ADD COLUMN IF NOT EXISTS metrics JSONB;

-- 預覽樣本的「資訊可用時間」：decision_ts／target_ts 是 K 棒的識別時間（棒起點），
-- 日線的 08:45 那根棒要 13:45 收盤才有完整資訊，決策時刻不能標成 08:45。舊列維持 NULL。
ALTER TABLE model_training_preview_samples
    ADD COLUMN IF NOT EXISTS decision_available_ts TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS target_available_ts TIMESTAMPTZ;

-- 推論結果的「資訊可用時間」：ts 是樣本最後一根輸入棒的識別時間，available_ts 才是該樣本最早可用於決策的時間。
-- 另外，雙頭模型的推論列使用既有欄位存放（不需要新欄位）：
--   output_type='dual'、predicted＝回歸頭預測值（目標原單位）、probabilities＝[P(不符合事件), P(符合事件)]。
ALTER TABLE model_inference_predictions
    ADD COLUMN IF NOT EXISTS available_ts TIMESTAMPTZ;

COMMIT;

-- 套用後驗證（唯讀）：
--   SELECT table_name, column_name, data_type FROM information_schema.columns
--   WHERE (table_name, column_name) IN (
--       ('model_training_progress', 'metrics'),
--       ('model_training_preview_samples', 'decision_available_ts'),
--       ('model_training_preview_samples', 'target_available_ts'),
--       ('model_inference_predictions', 'available_ts'))
--   ORDER BY table_name, column_name;   -- 預期 4 列
