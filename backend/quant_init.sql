-- hermesnote-v2 量化回測：任務/結果持久化
-- 建在 hermesnote DB（限定可用，見 CLAUDE.md），不動舊站原有的表，一律 quant_ 前綴

CREATE TABLE IF NOT EXISTS quant_backtest_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    long_entry_tree JSONB NOT NULL,
    long_exit_tree JSONB NOT NULL,
    short_entry_tree JSONB,
    short_exit_tree JSONB,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending / running / done / failed
    error TEXT,
    summary JSONB,
    indicators JSONB,
    rules JSONB,
    bar_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS quant_backtest_candles (
    job_id UUID NOT NULL REFERENCES quant_backtest_jobs(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    PRIMARY KEY (job_id, ts)
);

CREATE TABLE IF NOT EXISTS quant_backtest_trades (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES quant_backtest_jobs(id) ON DELETE CASCADE,
    direction TEXT NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    entry_reason TEXT,
    exit_time TIMESTAMPTZ,
    exit_price DOUBLE PRECISION,
    exit_reason TEXT,
    points DOUBLE PRECISION,
    pnl DOUBLE PRECISION NOT NULL,
    return_pct DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quant_backtest_trades_job ON quant_backtest_trades(job_id);
CREATE INDEX IF NOT EXISTS idx_quant_backtest_jobs_status ON quant_backtest_jobs(status, created_at DESC);

-- 收藏的回測策略：讓模型訓練那邊可以當 Feature Node 取用（見 docs/record 2026-09-09）。
-- 跟 job 綁 ON DELETE CASCADE——刪掉那筆回測紀錄，收藏也跟著清掉，不留孤兒資料。
-- 回測有效不代表對模型預測有用，這兩件事不對等，這裡只是把驗證過的規則轉存方便重複取用。
CREATE TABLE IF NOT EXISTS quant_saved_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES quant_backtest_jobs(id) ON DELETE CASCADE,
    name TEXT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    long_entry_tree JSONB,
    long_exit_tree JSONB,
    short_entry_tree JSONB,
    short_exit_tree JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_quant_saved_strategies_job ON quant_saved_strategies(job_id);
