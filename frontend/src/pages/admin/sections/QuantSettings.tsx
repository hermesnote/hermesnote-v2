import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  StrategyGroupEditor,
  describeNode,
  emptyGroup,
  type GroupNode,
  type RegistryGroup,
  type TreeNode,
} from "./TreeBuilder";
import "./QuantSettings.css";

const API_BASE = import.meta.env.VITE_API_BASE as string;
const TIMEFRAMES = ["1m", "5m", "15m", "30m", "60m", "1d"];

const CATEGORIES: Record<string, { label: string; symbols: { value: string; label: string }[] }> = {
  index_futures: {
    label: "期貨指數",
    symbols: [
      { value: "TX", label: "TX（台指期）" },
      { value: "MTX", label: "MTX（小台指）" },
      { value: "TMF", label: "TMF（微台指）" },
    ],
  },
  stock_futures: { label: "個股期貨", symbols: [] },
};

const CUSTOM_GROUPS = [
  { key: "custom_raw_price", label: "Raw - Price", description: "原始開高低收價格，不做任何轉換的基礎資料" },
  { key: "custom_raw_volume", label: "Raw - Volume", description: "原始成交量，不做任何轉換的基礎資料" },
  { key: "custom_raw_candle", label: "Raw - Candle Geometry", description: "K棒幾何特徵：實體、上影線、下影線佔整根K棒的比例" },
  { key: "custom_structure_levels", label: "Structure - Key Levels", description: "自動偵測近期支撐/壓力價位與樞紐點" },
  { key: "custom_structure_events", label: "Structure - Events", description: "價格突破/跌破關鍵價位、或突破後回踩測試的事件" },
  { key: "custom_structure_states", label: "Structure - States", description: "判斷目前是盤整、均線糾結、或價格相對關鍵位置的狀態" },
];

type JobSummary = {
  job_id: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  status: "pending" | "running" | "done" | "failed";
  error: string | null;
  summary: {
    total_trades: number;
    win_rate_pct: number | null;
    total_return_pct: number;
    max_drawdown_pct: number;
  } | null;
  bar_count: number | null;
  created_at: string;
};

function serializeNode(node: TreeNode): unknown {
  if (node.type === "condition") {
    const base: Record<string, unknown> = {
      type: "condition",
      kind: node.kind,
      key: node.key,
      params: node.params,
      weight: node.weight,
    };
    if (node.kind === "filter") {
      base.output_name = node.outputName;
      base.operator = node.operator;
      base.threshold_value = node.thresholdValue;
    }
    return base;
  }
  return {
    type: "group",
    mode: node.mode,
    threshold: node.threshold,
    weight: node.weight,
    children: node.children.map(serializeNode),
  };
}

export default function QuantSettings() {
  const navigate = useNavigate();
  const [category, setCategory] = useState("index_futures");
  const [symbol, setSymbol] = useState("TX");
  const [timeframe, setTimeframe] = useState("15m");
  const [start, setStart] = useState("2026-06-01");
  const [end, setEnd] = useState("2026-09-01");
  const [dateRange, setDateRange] = useState<{ min: string; max: string } | null>(null);

  const [registry, setRegistry] = useState<RegistryGroup[]>([]);
  const [longEntryTree, setLongEntryTree] = useState<GroupNode>(() => emptyGroup("or"));
  const [longExitTree, setLongExitTree] = useState<GroupNode>(() => emptyGroup("or"));
  const [shortEntryTree, setShortEntryTree] = useState<GroupNode>(() => emptyGroup("or"));
  const [shortExitTree, setShortExitTree] = useState<GroupNode>(() => emptyGroup("or"));
  const [enableShort, setEnableShort] = useState(false);

  const [runningJobId, setRunningJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState("");
  const [history, setHistory] = useState<JobSummary[]>([]);
  const [savedJobIds, setSavedJobIds] = useState<Set<string>>(new Set());

  function refreshSaved() {
    fetch(`${API_BASE}/api/backtest/saved-strategies`)
      .then((res) => res.json())
      .then((json: { job_id: string }[]) => setSavedJobIds(new Set(json.map((s) => s.job_id))))
      .catch(() => {});
  }

  function handleToggleSave(jobId: string) {
    const method = savedJobIds.has(jobId) ? "DELETE" : "POST";
    fetch(`${API_BASE}/api/backtest/jobs/${jobId}/save`, {
      method,
      headers: method === "POST" ? { "Content-Type": "application/json" } : undefined,
      body: method === "POST" ? JSON.stringify({}) : undefined,
    })
      .then((res) => {
        if (!res.ok) throw new Error();
        refreshSaved();
      })
      .catch(() => setSubmitError("收藏/取消收藏失敗，請稍後再試"));
  }

  useEffect(() => {
    fetch(`${API_BASE}/api/indicators`)
      .then((res) => res.json())
      .then((json: { groups: RegistryGroup[] }) => {
        setRegistry([
          ...json.groups,
          {
            group: "Custom",
            group_zh: "Custom（自訂指標，尚未建立計算邏輯）",
            indicators: CUSTOM_GROUPS.map((c) => ({
              key: c.key,
              display_name: c.label,
              description: c.description,
              parameters: {},
              output_names: [],
              signal_supported: false,
              signal_type: null,
            })),
          },
        ]);
      })
      .catch(() => setRegistry([]));
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/backtest/date-range?symbol=${symbol}&timeframe=${timeframe}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((json: { min_date: string; max_date: string }) => {
        setDateRange({ min: json.min_date, max: json.max_date });
        setStart(json.min_date);
        setEnd(json.max_date);
      })
      .catch(() => setDateRange(null));
  }, [symbol, timeframe]);

  function refreshHistory() {
    fetch(`${API_BASE}/api/backtest/jobs?limit=10`)
      .then((res) => res.json())
      .then((json: JobSummary[]) => setHistory(json))
      .catch(() => {});
  }

  function handleDelete(jobId: string) {
    if (!window.confirm("確定要刪除這筆回測紀錄嗎？刪除後無法復原，K 棒跟交易明細會一起清掉。")) return;
    fetch(`${API_BASE}/api/backtest/jobs/${jobId}`, { method: "DELETE" })
      .then((res) => {
        if (!res.ok) throw new Error();
        refreshHistory();
      })
      .catch(() => setSubmitError("刪除失敗，請稍後再試"));
  }

  useEffect(() => {
    refreshHistory();
    refreshSaved();
  }, []);

  // 有任務在跑就每 2 秒輪詢一次狀態，跑完/失敗就停止輪詢、更新歷史紀錄
  useEffect(() => {
    if (!runningJobId) return;
    const timer = setInterval(() => {
      fetch(`${API_BASE}/api/backtest/jobs/${runningJobId}`)
        .then((res) => res.json())
        .then((job: { status: string; error: string | null }) => {
          if (job.status === "done" || job.status === "failed") {
            setRunningJobId(null);
            if (job.status === "failed") setSubmitError(job.error || "回測失敗");
            refreshHistory();
          }
        })
        .catch(() => {});
    }, 2000);
    return () => clearInterval(timer);
  }, [runningJobId]);

  function handleCategoryChange(next: string) {
    setCategory(next);
    const firstSymbol = CATEGORIES[next].symbols[0];
    setSymbol(firstSymbol ? firstSymbol.value : "");
  }

  const longReady = longEntryTree.children.length > 0 && longExitTree.children.length > 0;
  const shortReady = !enableShort || (shortEntryTree.children.length > 0 && shortExitTree.children.length > 0);
  const canSubmit = longReady && shortReady && !runningJobId;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitError("");

    const body = {
      symbol,
      timeframe,
      start,
      end,
      long_entry_tree: serializeNode(longEntryTree),
      long_exit_tree: serializeNode(longExitTree),
      short_entry_tree: enableShort ? serializeNode(shortEntryTree) : null,
      short_exit_tree: enableShort ? serializeNode(shortExitTree) : null,
    };

    fetch(`${API_BASE}/api/backtest/strategy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`送出失敗（${res.status}）`);
        return res.json();
      })
      .then((json: { job_id: string }) => {
        setRunningJobId(json.job_id);
        refreshHistory();
      })
      .catch((err) => setSubmitError(err.message || "送出失敗"));
  }

  const totalIndicators = registry.reduce((n, g) => n + g.indicators.length, 0);
  const supportedCount = registry.reduce((n, g) => n + g.indicators.filter((i) => i.signal_supported).length, 0);

  return (
    <div className="quant-settings">
      <div className="quant-settings-block">
        <h2 className="quant-settings-block-title">建立回測</h2>
        <div className="quant-settings-card">
          <form className="quant-settings-form-v2" onSubmit={handleSubmit}>
            <div className="quant-settings-config-row">
              <label className="quant-settings-field">
                <span>類別</span>
                <select value={category} onChange={(e) => handleCategoryChange(e.target.value)}>
                  {Object.entries(CATEGORIES).map(([key, c]) => (
                    <option key={key} value={key}>{c.label}</option>
                  ))}
                </select>
              </label>

              <label className="quant-settings-field">
                <span>商品</span>
                {CATEGORIES[category].symbols.length > 0 ? (
                  <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
                    {CATEGORIES[category].symbols.map((s) => (
                      <option key={s.value} value={s.value}>{s.label}</option>
                    ))}
                  </select>
                ) : (
                  <select disabled><option>尚無資料</option></select>
                )}
              </label>

              <label className="quant-settings-field">
                <span>時間框架</span>
                <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
                  {TIMEFRAMES.map((tf) => (
                    <option key={tf} value={tf}>{tf}</option>
                  ))}
                </select>
              </label>

              <label className="quant-settings-field">
                <span>開始日期</span>
                <input type="date" value={start} min={dateRange?.min} max={dateRange?.max} onChange={(e) => setStart(e.target.value)} />
              </label>

              <label className="quant-settings-field">
                <span>結束日期</span>
                <input type="date" value={end} min={dateRange?.min} max={dateRange?.max} onChange={(e) => setEnd(e.target.value)} />
              </label>
            </div>

            <div className="quant-settings-sides">
              <div className="quant-settings-side">
                <div className="quant-settings-side-title">做多 LONG</div>

                <div className="quant-settings-indicators">
                  <div className="quant-settings-indicators-head"><span>進場運算式</span></div>
                  <div className="quant-settings-preview">{describeNode(longEntryTree)}</div>
                  <StrategyGroupEditor group={longEntryTree} onChange={setLongEntryTree} registry={registry} triggerKind="long_entry" />
                </div>

                <div className="quant-settings-indicators">
                  <div className="quant-settings-indicators-head"><span>出場運算式</span></div>
                  <div className="quant-settings-preview">{describeNode(longExitTree)}</div>
                  <StrategyGroupEditor group={longExitTree} onChange={setLongExitTree} registry={registry} triggerKind="long_exit" />
                </div>
              </div>

              <div className="quant-settings-side">
                <div className="quant-settings-side-title">
                  做空 SHORT
                  <label className="quant-settings-short-toggle">
                    <input type="checkbox" checked={enableShort} onChange={(e) => setEnableShort(e.target.checked)} />
                    啟用
                  </label>
                </div>

                {enableShort ? (
                  <>
                    <div className="quant-settings-indicators">
                      <div className="quant-settings-indicators-head"><span>進場運算式</span></div>
                      <div className="quant-settings-preview">{describeNode(shortEntryTree)}</div>
                      <StrategyGroupEditor group={shortEntryTree} onChange={setShortEntryTree} registry={registry} triggerKind="short_entry" />
                    </div>

                    <div className="quant-settings-indicators">
                      <div className="quant-settings-indicators-head"><span>出場運算式</span></div>
                      <div className="quant-settings-preview">{describeNode(shortExitTree)}</div>
                      <StrategyGroupEditor group={shortExitTree} onChange={setShortExitTree} registry={registry} triggerKind="short_exit" />
                    </div>
                  </>
                ) : (
                  <div className="quant-settings-indicators-empty">未啟用——這次回測只做多，期貨本身可以直接雙向，要測試放空策略再勾選。</div>
                )}
              </div>
            </div>

            <div className="quant-settings-indicators-note">
              指標清單來自後端 TA-Lib 登記表（共 {totalIndicators} 個，其中 {supportedCount} 個已接上訊號規則，能當「觸發條件」；其餘的、以及任何已接上的，都能當「過濾條件」）。
              每個觸發指標算出兩個方向的事件：放進「多方進場／空方出場」用的是同一個上升方向，放進「多方出場／空方進場」用的是同一個下降方向，系統自動接對，不用你自己選方向。群組可以無限巢狀，每層自己選 AND／OR／加權。
            </div>

            <div className="quant-settings-submit-row">
              <button type="submit" className="quant-settings-submit" disabled={!canSubmit}>
                {runningJobId ? "回測中…" : "建立回測"}
              </button>
              {submitError && <div className="quant-settings-submit-msg is-error">{submitError}</div>}
            </div>
          </form>
        </div>
      </div>

      <div className="quant-settings-block">
        <h2 className="quant-settings-block-title">歷史紀錄</h2>
        <div className="quant-settings-card">
          {history.length === 0 && <div className="quant-settings-empty">尚無回測紀錄。</div>}
          {history.map((job) => (
            <div
              key={job.job_id}
              className={"quant-settings-history-row" + (job.status !== "done" ? " is-disabled" : "")}
              onClick={() => job.status === "done" && navigate(`/quant?job=${job.job_id}`)}
            >
              <div className="quant-settings-history-meta">
                <div className="quant-settings-history-time">
                  {new Date(job.created_at).toLocaleString("zh-TW")}
                  {job.status !== "done" && ` ・ ${job.status === "failed" ? "失敗" : job.status === "running" ? "執行中" : "等待中"}`}
                </div>
                <div className="quant-settings-history-config">
                  {job.symbol} ・ {job.timeframe} ・ {job.start_date} ~ {job.end_date}
                  {job.bar_count != null && ` ・ ${job.bar_count.toLocaleString()} 根`}
                </div>
                {job.error && <div className="quant-settings-history-error">{job.error}</div>}
              </div>
              {job.summary && (
                <div className="quant-settings-history-stats">
                  <div className="quant-settings-history-stat"><span>交易數</span><strong>{job.summary.total_trades}</strong></div>
                  <div className="quant-settings-history-stat"><span>勝率</span><strong>{job.summary.win_rate_pct?.toFixed(1) ?? "—"}%</strong></div>
                  <div className="quant-settings-history-stat">
                    <span>報酬</span>
                    <strong className={job.summary.total_return_pct >= 0 ? "is-positive" : "is-negative"}>
                      {job.summary.total_return_pct > 0 ? "+" : ""}
                      {job.summary.total_return_pct.toFixed(2)}%
                    </strong>
                  </div>
                  <div className="quant-settings-history-stat"><span>最大回撤</span><strong className="is-negative">{job.summary.max_drawdown_pct.toFixed(2)}%</strong></div>
                </div>
              )}
              <button
                type="button"
                className={"quant-settings-history-save" + (savedJobIds.has(job.job_id) ? " is-saved" : "")}
                onClick={(e) => {
                  e.stopPropagation();
                  handleToggleSave(job.job_id);
                }}
                aria-label={savedJobIds.has(job.job_id) ? "取消收藏（模型訓練會找不到這個特徵來源）" : "收藏這個策略，之後模型訓練可以當特徵來源"}
                title="收藏後，模型訓練的 Feature Node 可以選用這個策略重算出的進出場訊號當特徵"
              >
                {savedJobIds.has(job.job_id) ? "★ 已收藏" : "☆ 收藏"}
              </button>
              <button
                type="button"
                className="quant-settings-history-delete"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(job.job_id);
                }}
                aria-label="刪除這筆回測紀錄"
              >
                刪除
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
