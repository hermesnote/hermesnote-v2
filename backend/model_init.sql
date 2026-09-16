-- hermesnote-v2 模型訓練：任務佇列（web 容器寫入 pending，訓練容器 worker 撿去跑）
-- 建在 hermesnote DB（限定可用，見 CLAUDE.md），一律 model_ 前綴，比照 quant_backtest_jobs 的做法

CREATE TABLE IF NOT EXISTS model_training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_spec JSONB NOT NULL,          -- 節點圖規格（Feature Node / Model Node + DAG），見 docs/record/archive/2026-09-08.md
    status TEXT NOT NULL DEFAULT 'pending',  -- pending / running / done / failed
    error TEXT,
    result JSONB,                        -- 訓練完成後的結果摘要（metrics、產出模型檔路徑等）；job_type='infer' 時存推論結果
    device TEXT,                          -- worker 撿到任務時記錄實際用 cuda 還是 cpu
    job_type TEXT NOT NULL DEFAULT 'train',  -- 'train'（送 graph_spec 重新訓練）或 'infer'（載入 model_artifacts 做推論，不重訓）
    phase INTEGER,                        -- 教授三階段協定：1=random / 2=chronological / 3=holdout 推論；NULL 代表不屬於三階段流程的一般訓練
    parent_job_id UUID REFERENCES model_training_jobs(id),  -- Phase 2 必須指向一個 Phase 1 job；Phase 3 必須指向一個 Phase 2 job
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_model_training_jobs_status ON model_training_jobs(status, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_model_training_jobs_parent ON model_training_jobs(parent_job_id);

-- 訓練進度：一個 epoch/checkpoint 一列，避免塞進 model_training_jobs 的單一 JSONB 越寫越大
-- （比照 quant_backtest_candles 的設計理由，見 docs/record/archive/2026-09-08-training-plan.md）
CREATE TABLE IF NOT EXISTS model_training_progress (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES model_training_jobs(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    loss DOUBLE PRECISION,
    accuracy DOUBLE PRECISION,
    val_loss DOUBLE PRECISION,
    val_accuracy DOUBLE PRECISION,
    window_meta JSONB,     -- 這個 epoch/窗格對應的資料範圍等中繼資訊，前端點擊清單跳轉圖表用
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_model_training_progress_job ON model_training_progress(job_id, epoch ASC);

-- 模型產物：訓練成功當下自動存一份，不是等使用者按「儲存」才存
-- （job 結束後模型只在記憶體，等按鈕才存來源早就沒了）；kept 只決定要不要在 UI 上
-- 標成「已收藏／長期保留」，不影響這筆產物存不存在或能不能被載入推論。
-- 權重直接存 bytea 在 DB 裡，不用檔案系統路徑——training 容器跟 backend 容器目前沒有共用
-- volume（見 docker-compose.yml），存檔案兩邊會互相讀不到，容器重建也會把檔案沖掉；
-- 兩個容器本來就都連得到這個 DB，存 bytea 不用動 docker-compose，不影響「container 歸
-- container」這條原則。
CREATE TABLE IF NOT EXISTS model_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES model_training_jobs(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,                -- graph_spec 裡這個 model node 的 id
    architecture_key TEXT NOT NULL,       -- 'lstm' / 'xgboost'
    task_type TEXT NOT NULL,              -- 'classification' / 'regression'
    weights BYTEA NOT NULL,               -- LSTM: state_dict 序列化；XGBoost: Booster.save_raw() 原生格式
    model_config JSONB NOT NULL,          -- 重建模型結構要的資訊（units/layers/n_features/out_dim/window/n_classes...）
    feature_schema JSONB NOT NULL,        -- 這個 node 依序引用的 Feature Node 定義快照
    target_spec JSONB NOT NULL,           -- outcome/labeling_rule/horizon/n_classes/threshold_pct 快照
    graph_spec_snapshot JSONB NOT NULL,   -- 當次完整 graph_spec（含所有上游節點定義），重建依賴鏈用
    preprocessing_state JSONB NOT NULL DEFAULT '{}'::jsonb,  -- 這輪固定 {"method":"none"}，之後接正規化才會有內容
    depends_on_node_ids JSONB NOT NULL DEFAULT '[]'::jsonb,  -- 這個 model node 依賴的上游「其他 model node」的 id 列表
    kept BOOLEAN NOT NULL DEFAULT false,  -- 使用者在 UI 按「儲存模型」= true，純粹是否標記/顯示用途
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_artifacts_job_node ON model_artifacts(job_id, node_id);

-- 即時視窗預覽（訓練途中「正在處理哪個 window」的抽樣紀錄）：跟 model_training_progress
-- 刻意分成兩張表——這裡的 bars 資料量會隨 window 變大，PostgreSQL 的 NOTIFY payload 上限是
-- 8000 bytes（https://www.postgresql.org/docs/current/sql-notify.html），不能跟 epoch 級的
-- 小 payload 混在同一個 NOTIFY 裡，也不適合塞進 model_training_progress 的 window_meta 欄位。
-- 完成後的「歷史瀏覽」直接查這張表（依 id 游標分頁），不用另外設計儲存格式。
CREATE TABLE IF NOT EXISTS model_training_preview_samples (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES model_training_jobs(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    source TEXT NOT NULL,              -- 'train' / 'val'——抽樣當下這個 window 屬於哪一邊
    decision_ts TIMESTAMPTZ NOT NULL,  -- 決策時刻 T
    target_ts TIMESTAMPTZ NOT NULL,    -- 預測目標 T+horizon
    horizon INTEGER NOT NULL,
    task_type TEXT NOT NULL,           -- 'classification' / 'regression'
    n_classes INTEGER,                 -- 分類才有值
    labeling_rule TEXT,                -- 用哪個 labeling_rule 算出來的類別，前端依此決定類別的實際意義（漲/跌/平）
    bars JSONB NOT NULL,               -- 這個 window 的真實 OHLC K 線
    actual DOUBLE PRECISION NOT NULL,  -- 實際答案（展示比對用，不影響訓練）
    predicted DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_model_preview_samples_job_node
    ON model_training_preview_samples(job_id, node_id, id DESC);

-- 最終模型推論的逐列結果（job_type='infer' 那種 job）：跟上面 model_training_preview_samples
-- 是完全不同性質的資料——那張表是訓練時的抽樣展示，這張是模型對一整段期間所有列的真正
-- 推論結果，量級可以到百萬列，一律分批寫入（見 training/inference_store.py 的
-- open_prediction_writer／save_predictions_chunked）、依時間區間＋游標分頁查，不整包塞進
-- model_training_jobs.result。output_type 決定怎麼解讀 predicted/probabilities：
--   'class'／'regression' — predicted 是唯一的數值，probabilities 是 NULL
--   'probability'         — predicted 是 argmax 類別索引（方便排序/篩選），probabilities 是
--                           完整機率向量（JSONB array）——不能只留 argmax 就把原始機率丟掉，
--                           之後機率門檻策略要用完整分布，不是只有預測類別。
CREATE TABLE IF NOT EXISTS model_inference_predictions (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES model_training_jobs(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    output_type TEXT NOT NULL,
    predicted DOUBLE PRECISION NOT NULL,
    probabilities JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_model_inference_predictions_job_node_ts
    ON model_inference_predictions(job_id, node_id, ts ASC);
CREATE INDEX IF NOT EXISTS idx_model_inference_predictions_job_node_id
    ON model_inference_predictions(job_id, node_id, id ASC);

-- 防呆用：如果正式環境曾經套用過這張表「舊版」定義（沒有 output_type/probabilities，
-- 見 2026-09-15 那版 model_init.sql），上面的 CREATE TABLE IF NOT EXISTS 對已存在的表
-- 是完全不動作的，不會自動補欄位——這裡明確用 ALTER TABLE ADD COLUMN IF NOT EXISTS
-- 補上，同一支腳本不管是全新環境還是曾經套用過舊版，重複執行都是安全、可預期的結果。
-- output_type 給預設值 'class' 只是防呆（目前確認這張表在正式環境從未寫入過任何資料，
-- 理論上不會有既有列需要這個預設值），不影響 inference_store.py 寫入時一律明確指定值。
ALTER TABLE model_inference_predictions ADD COLUMN IF NOT EXISTS output_type TEXT NOT NULL DEFAULT 'class';
ALTER TABLE model_inference_predictions ADD COLUMN IF NOT EXISTS probabilities JSONB;
