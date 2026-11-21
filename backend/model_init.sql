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

-- 2026-09-21（Attention＋雙頭 LSTM 能力擴充）新增的欄位，語意見 migrations/2026-09-21_attention_dual_head.sql。
-- 既有環境請用該遷移檔；這裡放同樣的 idempotent 語句，讓從頭建置的環境跟遷移後的環境一致。
ALTER TABLE model_training_progress ADD COLUMN IF NOT EXISTS metrics JSONB;
ALTER TABLE model_training_preview_samples ADD COLUMN IF NOT EXISTS decision_available_ts TIMESTAMPTZ;
ALTER TABLE model_training_preview_samples ADD COLUMN IF NOT EXISTS target_available_ts TIMESTAMPTZ;
ALTER TABLE model_inference_predictions ADD COLUMN IF NOT EXISTS available_ts TIMESTAMPTZ;

-- 2026-09-23（best／last 雙權重）新增的欄位，語意見 migrations/2026-09-23_weights_last_and_evaluation.sql
-- （training/artifacts.py 無條件讀寫此欄，部署前必須先套用）。
ALTER TABLE model_artifacts ADD COLUMN IF NOT EXISTS weights_last BYTEA;

-- （2026-09-23 週日 scrub 維護閘門的 model_training_checkpoints 建表語句已隨功能撤回移除
--  ——那個遷移同樣從未套用過正式庫，撤回範圍內的東西不留在 schema 定義檔裡，見
--  docs/record/archive/2026-09-23-scrub-withdrawal-and-lstm-custom.md。）

-- 2026-11-15（M1：流程版本與資料紀錄）新增的欄位，語意見 migrations/2026-11-15_m1_pipeline_provenance.sql
-- （新版 web／training 會讀寫這四欄，部署前必須先套用）。
ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS pipeline_version TEXT;
ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS warnings JSONB;
ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS env_fingerprint JSONB;
ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS data_report JSONB;

-- 2026-11-16 M2～M7：實驗層、樣本框架、逐筆結果、Phase 3 評估、holdout 使用紀錄、串接折次
-- （內容同 migrations/2026-11-16_m2_m7_experiments_frames_records.sql，部署前必須先套用）

CREATE TABLE IF NOT EXISTS model_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    settings JSONB NOT NULL,           -- 期間、holdout、Label、切分種子、val_ratio、對齊規則、暖機 ε、排除警示、門檻、串接預設
    pipeline_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_experiment_arms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES model_experiments(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    spec JSONB NOT NULL,               -- 組別的節點（Feature／Model；Label 由實驗層注入）
    model_seeds JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_model_experiment_arms_exp ON model_experiment_arms(experiment_id);

CREATE TABLE IF NOT EXISTS model_sample_frames (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES model_experiments(id) ON DELETE CASCADE,
    phase SMALLINT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'defined',   -- defined（尚未產生時間點）／ready
    definition JSONB,
    hash TEXT,
    counts JSONB,
    n_points INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ready_at TIMESTAMPTZ,
    UNIQUE (experiment_id, phase, version)
);

CREATE TABLE IF NOT EXISTS model_sample_frame_points (
    frame_id UUID NOT NULL REFERENCES model_sample_frames(id) ON DELETE CASCADE,
    ts TIMESTAMP NOT NULL,                 -- 決策 K 棒 bar_start_ts
    decision_available_ts TIMESTAMP,       -- 決策時刻（bar_end_ts＋延遲）
    target_ts TIMESTAMP,
    target_available_ts TIMESTAMP,
    status TEXT NOT NULL,                  -- train／val／holdout／excluded
    reason TEXT,
    PRIMARY KEY (frame_id, ts)
);

CREATE TABLE IF NOT EXISTS model_prediction_records (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES model_training_jobs(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    experiment_id UUID,
    arm_id UUID,
    frame_hash TEXT,
    phase SMALLINT,
    dataset TEXT NOT NULL,                 -- train（只在串接時存，供核對上游來源）／val／holdout
    weights TEXT,                          -- best／last；NULL＝沒有預測的跳過點
    model_seed INT,
    sample_ts TIMESTAMP NOT NULL,
    decision_available_ts TIMESTAMP,
    window_start_ts TIMESTAMP,
    target_ts TIMESTAMP,
    target_available_ts TIMESTAMP,
    y_true DOUBLE PRECISION,
    y_pred DOUBLE PRECISION,
    proba JSONB,
    included BOOLEAN NOT NULL,             -- 是否納入評估
    exclusion_reason TEXT,
    branches JSONB,                        -- 各分支實際採用的資料（最後一根時間、可用時間、距決策時刻、是否前一時段）
    upstream JSONB                         -- 上游預測來源（折次、可用時間）
);
CREATE INDEX IF NOT EXISTS idx_model_prediction_records_job ON model_prediction_records(job_id, node_id, dataset, weights, sample_ts);
CREATE INDEX IF NOT EXISTS idx_model_prediction_records_exp ON model_prediction_records(experiment_id, arm_id);

CREATE TABLE IF NOT EXISTS model_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES model_training_jobs(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    phase SMALLINT,
    dataset TEXT NOT NULL,
    weights TEXT NOT NULL,
    evaluations JSONB NOT NULL,            -- 與 job.result[node].evaluations 相同格式（逐輸出頭）
    n_included INT,
    n_excluded INT,
    excluded_reasons JSONB,
    cutoff_ts TIMESTAMP,                   -- 已成熟答案的最晚時間
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_model_evaluations_job ON model_evaluations(job_id);

CREATE TABLE IF NOT EXISTS model_holdout_usage (
    id BIGSERIAL PRIMARY KEY,
    experiment_id UUID,
    arm_id UUID,
    job_id UUID REFERENCES model_training_jobs(id) ON DELETE SET NULL,
    parent_job_id UUID,
    kind TEXT NOT NULL,                    -- inference／evaluation／comparison
    period_start DATE,
    period_end DATE,
    detail JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_model_holdout_usage_exp ON model_holdout_usage(experiment_id, created_at);

CREATE TABLE IF NOT EXISTS model_upstream_folds (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES model_training_jobs(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,                 -- 上游模型節點
    fold INT NOT NULL,                     -- 1..K；-1＝最終上游模型
    strategy TEXT NOT NULL,
    frame_hash TEXT,
    n_train INT,
    train_start TIMESTAMP,
    train_end TIMESTAMP,
    train_max_label_end TIMESTAMP,
    before_max_le TIMESTAMP,               -- 清除區間之前訓練點最晚的答案可用時間
    after_min_ws TIMESTAMP,                -- 清除區間之後訓練點最早的窗格起點
    purge_start TIMESTAMP,
    purge_end TIMESTAMP,
    predict_start TIMESTAMP,
    predict_end TIMESTAMP,
    n_pred INT,
    fingerprint TEXT,
    detail JSONB
);
CREATE INDEX IF NOT EXISTS idx_model_upstream_folds_job ON model_upstream_folds(job_id, node_id);

ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS experiment_id UUID;
ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS arm_id UUID;
ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS frame_id UUID;
ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS model_seed INT;
ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS split_seed INT;
ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS gate_check JSONB;
-- 驗證身分（admin:<email>／key:<服務金鑰名稱>）：誰建立這筆任務；舊任務為 NULL
ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS requested_by TEXT;
ALTER TABLE model_experiments ADD COLUMN IF NOT EXISTS created_by TEXT;
CREATE INDEX IF NOT EXISTS idx_model_training_jobs_experiment ON model_training_jobs(experiment_id, arm_id);

ALTER TABLE model_artifacts ADD COLUMN IF NOT EXISTS fingerprint TEXT;
ALTER TABLE model_artifacts ADD COLUMN IF NOT EXISTS warmup_policy JSONB;
ALTER TABLE model_artifacts ADD COLUMN IF NOT EXISTS train_class_counts JSONB;   -- train_stats（類別計數／目標平均）
ALTER TABLE model_artifacts ADD COLUMN IF NOT EXISTS history_requirements JSONB;
