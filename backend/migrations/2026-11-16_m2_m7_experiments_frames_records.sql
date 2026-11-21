-- M2～M7：實驗層、共同樣本框架、逐筆結果、Phase 3 評估、holdout 使用紀錄、串接折次（只新增，不修改／刪除既有資料）
-- 套用前須經 Hermes 同意；套用順序：先遷移、後部署。可重複執行（IF NOT EXISTS）。

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
