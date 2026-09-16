import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  createSeriesMarkers,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type IChartApi,
  type Time,
} from "lightweight-charts";
import "./ModelTrainingPage.css";

const API_BASE = import.meta.env.VITE_API_BASE as string;

const TZ_OFFSET_SEC = -new Date().getTimezoneOffset() * 60;
function toChartTime(unixSec: number): Time {
  return (unixSec + TZ_OFFSET_SEC) as Time;
}

// ── 資料型別：跟 graph_spec 實際存的欄位對齊，不是憑印象猜的 ──────────────────
type FeatureNode = { id: string; type: "feature"; key: string; timeframe: string; params?: Record<string, unknown> };
// Label Node 存的是 outcome／labeling_rule 兩個獨立欄位，不是塞在 key 裡——
// 舊版這裡型別跟畫面都讀 labelNode.key，讀到的其實是 undefined，這次修掉。
type LabelNode = {
  id: string; type: "label"; outcome: string; labeling_rule: string; timeframe: string;
  params?: { horizon?: number; n_classes?: number; threshold_pct?: number };
};
type ModelNode = {
  id: string; type: "model"; key: string; inputs: string[]; label: string;
  window: number; val_ratio: number; output_mode?: string; params?: Record<string, unknown>;
};
type AnyNode = FeatureNode | LabelNode | ModelNode;

type JobSummary = { job_id: string; job_type: "train" | "infer"; status: "pending" | "running" | "done" | "failed"; created_at: string };

type JobResultEntry = {
  final_metrics?: Record<string, number>;
  device?: string;
  training_meta?: { n_train?: number; n_val?: number; split_strategy?: string };
  // job_type==='infer' 才會有——全期間逐列結果已分批存進 model_inference_predictions，
  // 這裡只留摘要（筆數、時間範圍），完整內容改用 /inference_predictions 分頁查。
  count?: number;
  start_ts?: number | null;
  end_ts?: number | null;
};

// job_type==='train' 的 graph_spec 是完整節點圖；job_type==='infer' 完全是另一種形狀
// ——載入已保存模型對新區間做推論用，沒有 nodes，不能用同一套「模型節點/特徵」畫面邏輯，
// 用 job_type 分流，不能靠鴨子定型（duck typing）猜測。
type TrainGraphSpec = { start: string; end: string; nodes: AnyNode[] };
type InferGraphSpec = { target_job_id: string; target_node_id: string; start: string; end: string };

type JobDetail = {
  job_id: string;
  job_type: "train" | "infer";
  graph_spec: TrainGraphSpec | InferGraphSpec;
  status: "pending" | "running" | "done" | "failed";
  error: string | null;
  result: Record<string, JobResultEntry> | null;
  device: string | null;
  created_at?: string;
};

// 最終模型推論的逐列結果（job_type='infer'）——跟 WindowPreview（訓練途中的抽樣展示）
// 是完全不同性質的資料，這裡沒有 bars／actual，只有模型真正的推論輸出，量級可以到百萬列，
// 一律依游標分頁載入，不整段抓。
type InferPrediction = {
  id: number;
  ts: number;
  output_type: "class" | "probability" | "regression";
  predicted: number;
  probabilities: number[] | null;
};

// 即時視窗預覽：後端「正在處理的 batch/迭代」抽一筆出來展示，不是固定一個樣本看到底——
// 每次收到都可能是不同的 window（隨機訓練下日期本來就會跳動，是正常現象）。bars 是真實
// OHLC，decision_ts/target_ts 是真實時間戳，actual/predicted 只給展示比對用，完全不影響
// 訓練本身（見 training/graph.py 的 preview_hook 說明）。
type WindowBar = { time: number; open: number; high: number; low: number; close: number };
type WindowPreview = {
  id?: number; // 只有歷史瀏覽（查 model_training_preview_samples）回來的紀錄才有
  epoch: number;
  source: "train" | "val";
  decision_ts: number;
  target_ts: number;
  horizon: number;
  task_type: "classification" | "regression";
  n_classes: number | null;
  labeling_rule?: string | null;
  bars: WindowBar[];
  actual: number;
  predicted: number;
};

// 進度一定要用 (job, node, epoch) 一起識別，多模型節點時不能只靠 epoch 合併，
// 不然兩個模型節點剛好跑到同一個 epoch 數字，資料會互相蓋掉。
type ProgressPoint = {
  node_id: string;
  epoch: number;
  loss: number | null;
  accuracy: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
  created_at: string;
};

function extractNodeId(raw: { node_id?: string; window_meta?: unknown }): string {
  const wm = raw.window_meta as { node_id?: string } | null | undefined;
  return raw.node_id ?? wm?.node_id ?? "default";
}

function fmt(n: number | null | undefined, digits = 4): string {
  return n === null || n === undefined ? "—" : n.toFixed(digits);
}

function fmtPct(n: number | null | undefined, digits = 2): string {
  return n === null || n === undefined ? "—" : `${(n * 100).toFixed(digits)}%`;
}

// 隨機訓練時樣本是打亂的，同一個節點抽到的 window 在真實日曆上可能跳來跳去（這一筆是
// 今天，下一筆可能是三年前），一定要顯示真實日期時間，不能只顯示「第幾筆」這種暗示
// 連續性的相對編號——這是隨機切分下唯一不會誤導的表示法。
function formatPreviewTime(unixSec: number): string {
  return new Date(unixSec * 1000).toLocaleString("zh-TW");
}

// 分類的類別要顯示實際意義（漲/不漲、跌/平/漲），不是「類別 0」這種需要使用者自己回頭
// 對照 threshold_pct 才看得懂的數字——這個對應只在 fixed_threshold 這個 labeling_rule
// 下成立（見 training/registry/labeling_rules.py 的 fixed_threshold 說明），不是通用規則，
// 換了 labeling_rule 就不能沿用這個對應，退回顯示「類別 N」。
function formatPreviewValue(v: number, taskType: "classification" | "regression", nClasses: number | null, labelingRule?: string | null): string {
  if (taskType !== "classification") return v.toFixed(4);
  if (labelingRule === "fixed_threshold") {
    if (nClasses === 2) return ["不漲", "漲"][v] ?? `類別 ${v}`;
    if (nClasses === 3) return ["跌", "平", "漲"][v] ?? `類別 ${v}`;
  }
  return `類別 ${v}`;
}

function secondsAgo(iso: string | null, nowMs: number): string {
  if (!iso) return "—";
  const diff = Math.max(0, Math.round((nowMs - new Date(iso).getTime()) / 1000));
  if (diff < 60) return `${diff} 秒前`;
  if (diff < 3600) return `${Math.floor(diff / 60)} 分鐘前`;
  return `${Math.floor(diff / 3600)} 小時前`;
}

function mergeGrouped(
  base: Record<string, ProgressPoint[]>,
  extra: Record<string, ProgressPoint[]>
): Record<string, ProgressPoint[]> {
  const merged: Record<string, ProgressPoint[]> = {};
  const nodeIds = new Set([...Object.keys(base), ...Object.keys(extra)]);
  for (const nodeId of nodeIds) {
    const byEpoch = new Map<number, ProgressPoint>();
    for (const p of base[nodeId] ?? []) byEpoch.set(p.epoch, p);
    for (const p of extra[nodeId] ?? []) byEpoch.set(p.epoch, p);
    merged[nodeId] = Array.from(byEpoch.values()).sort((a, b) => a.epoch - b.epoch);
  }
  return merged;
}

async function fetchProgressGrouped(id: string): Promise<Record<string, ProgressPoint[]>> {
  const hist: ProgressPoint[] = await fetch(`${API_BASE}/api/model/jobs/${id}/progress`).then((r) => r.json());
  const grouped: Record<string, ProgressPoint[]> = {};
  for (const raw of hist) {
    const nodeId = extractNodeId(raw);
    (grouped[nodeId] ??= []).push({ ...raw, node_id: nodeId });
  }
  for (const nodeId of Object.keys(grouped)) grouped[nodeId].sort((a, b) => a.epoch - b.epoch);
  return grouped;
}

const STATUS_LABEL: Record<string, string> = {
  pending: "等待中", running: "訓練中", done: "已完成", failed: "失敗",
};

const HISTORY_PAGE_SIZE = 30;

export default function ModelTrainingPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const jobIdParam = searchParams.get("job");

  // 「追蹤目前訓練」：自動跟著最新的 pending/running 任務；沒有這種任務就明確顯示
  // 「目前無訓練」，不會把舊的已完成任務偽裝成正在跑。
  // 「訓練紀錄」：使用者手動選定一筆，鎖住不被新任務搶走，只在有新任務時給提示。
  const [mode, setMode] = useState<"latest" | "history">(jobIdParam ? "history" : "latest");
  const [historyJobId, setHistoryJobId] = useState<string | null>(jobIdParam);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobList, setJobList] = useState<JobSummary[]>([]);
  const [latestActiveJobId, setLatestActiveJobId] = useState<string | null>(null); // 目前最新的 pending/running job
  const [hasNewerJob, setHasNewerJob] = useState(false); // 歷史模式下用來顯示「有新任務」提示

  const [job, setJob] = useState<JobDetail | null>(null);
  // infer job 的 graph_spec 沒有 nodes（型別見上方 InferGraphSpec 說明），一律當成空節點圖，
  // 不進入「模型節點/特徵」這套訓練專用畫面邏輯——這兩個要放在元件很前面宣告，因為底下
  // loadInferPage／它的觸發 effect 在原始碼順序上比原本宣告的位置早，用到還沒宣告的
  // const 會直接 TDZ ReferenceError（實測抓到的 bug：Cannot access 'isInferJob' before
  // initialization）。
  const isInferJob = job?.job_type === "infer";
  const trainGraphSpec = job && !isInferJob ? (job.graph_spec as TrainGraphSpec) : null;
  const inferGraphSpec = job && isInferJob ? (job.graph_spec as InferGraphSpec) : null;
  const [status, setStatus] = useState<string>("讀取中…");
  const [wsStatus, setWsStatus] = useState<"connecting" | "connected" | "error" | "closed">("closed");
  const [progressByNode, setProgressByNode] = useState<Record<string, ProgressPoint[]>>({});
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [showFullEpochList, setShowFullEpochList] = useState(false);
  const [showJobIdDetail, setShowJobIdDetail] = useState(false);
  const [expandedFeature, setExpandedFeature] = useState<Record<string, boolean>>({});
  const [now, setNow] = useState(() => Date.now());

  // 主要區域：「訓練即時預覽」（正在處理的抽樣 window，持續更換）或「完成後歷史瀏覽」
  // （查 model_training_preview_samples，依 id 游標分頁）——兩者共用同一個 K 線 chart
  // instance，不是各自一張圖，避免留一張跟核心功能無關的大歷史圖。
  const [viewMode, setViewMode] = useState<"live" | "history">("live");
  const [livePreviewByNode, setLivePreviewByNode] = useState<Record<string, WindowPreview>>({});
  const [historySamples, setHistorySamples] = useState<WindowPreview[]>([]);
  const [historySelectedIdx, setHistorySelectedIdx] = useState(0);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyHasMore, setHistoryHasMore] = useState(true);
  const [historyError, setHistoryError] = useState(false); // 明確跟「查得到、但確實沒有紀錄」分開
  // 歷史查詢要綁定「當下的 job + 模型節點」，不能只靠 activeJobId 的世代（loadGenRef）——
  // 同一個 job 內快速切換模型節點不會讓 loadGenRef 變動，光比對它擋不住「切到新節點後，
  // 舊節點還沒回來的查詢結果被誤放進新節點畫面」這個情境。這裡另外開一個世代編號，
  // job 或 selectedNodeId 任一個變動都會遞增，涵蓋兩種情境。
  const historyLoadGenRef = useRef(0);
  // 記錄「已經幫哪個 job+節點組合觸發過歷史載入」——不能靠 historySamples.length===0 這種
  // 用 state 判斷「要不要載入」的寫法：換節點的 reset effect 呼叫 setHistorySamples([]) 之後，
  // 同一輪 render 裡其他 effect 讀到的 historySamples 還是「這次 render 開始時」的舊值
  // （state 更新要等下一次 render 才會反映），如果載入判斷剛好也在這一輪跑、又是拿
  // historySamples.length 當條件，就會誤判「還有資料，不用載入」而漏掉這次切換——
  // 這是實測抓到的真的 bug（切模型節點到歷史瀏覽分頁，新節點永遠不會發出查詢）。
  // 用 ref 記錄「目前這個 key 是否已經觸發過」就不會有這個 state 時序問題。
  const historyLoadedKeyRef = useRef<string | null>(null);

  // 最終模型推論（job_type='infer'）的逐列瀏覽——跟上面的訓練抽樣歷史瀏覽是完全不同的
  // 資料來源／端點（/inference_predictions vs /preview_samples），量級可能到百萬列，一律
  // 依 id 游標分頁載入，不整段抓。用同一個 loadGenRef（job 世代）擋過期回應即可，infer job
  // 沒有「模型節點切換」這件事（一個 infer job 只對應一個 target_node_id），不需要再疊一層
  // 節點世代。
  const [inferPredictions, setInferPredictions] = useState<InferPrediction[]>([]);
  const [inferLoading, setInferLoading] = useState(false);
  const [inferHasMore, setInferHasMore] = useState(true);
  const [inferError, setInferError] = useState(false);
  const [inferSelectedIdx, setInferSelectedIdx] = useState(0);
  const inferLoadedKeyRef = useRef<string | null>(null);

  const lossChartRef = useRef<HTMLDivElement>(null);
  const lossChartApiRef = useRef<IChartApi | null>(null);
  const lossSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const valLossSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  const accChartRef = useRef<HTMLDivElement>(null);
  const accChartApiRef = useRef<IChartApi | null>(null);
  const accSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const valAccSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  // hover 中的 epoch（每張圖各自獨立）：null 代表沒在 hover，圖例退回顯示最新一輪的數值。
  const [lossHoverEpoch, setLossHoverEpoch] = useState<number | null>(null);
  const [accHoverEpoch, setAccHoverEpoch] = useState<number | null>(null);

  // 使用者是否手動縮放/捲動過任一張圖——一旦動過，「已完成輪數自動貼齊滿版寬度」這個
  // 預設行為就要讓步給使用者自己的視野，不能每次新資料進來就搶回去；換節點/任務時歸零。
  const chartUserInteractedRef = useRef(false);

  // 即時視窗預覽／歷史瀏覽共用的主圖表（K 線 + 決策點 T 的 marker）。
  const previewChartRef = useRef<HTMLDivElement>(null);
  const previewChartApiRef = useRef<IChartApi | null>(null);
  const previewSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const previewMarkersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const finalizeRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 每次「開始載入某個 job」都遞增一次，即使換回同一個 job id 也一樣——這是初始載入、
  // 3 秒輪詢、finalizeProgressSync 及其重試、WebSocket 全部共用的唯一有效性依據，取代原本
  // 各自比對 activeJobIdRef 的寫法。A → B → A 快速切換時，光比對 id 會誤判第一次那個早該
  // 作廢的 A session 還有效；比對這個世代編號則不會，因為每次載入都是全新的數字。
  const loadGenRef = useRef(0);
  const [progressSyncError, setProgressSyncError] = useState(false);

  // Loss/Accuracy 圖表是共用同一個 series instance 顯示「目前選定的節點」，不是每個節點
  // 各自一份，所以「換節點/換 job」永遠都要整包 setData，不能靠「這個節點以前是不是畫過」
  // 判斷——chartContextRef 記的是「圖表上現在顯示的是哪個 job+node」，renderStateRef 記的
  // 才是「這個 job+node 上一次同步到畫面時，各 epoch 各是什麼值」，兩者用途不同。
  const chartContextRef = useRef<{ jobId: string | null; nodeId: string | null }>({ jobId: null, nodeId: null });
  const renderStateRef = useRef<Record<string, { points: Map<number, ProgressPoint>; maxEpoch: number | null }>>({});

  // 每秒跳動一次，只用來重新計算「N 秒前」這種相對時間文字，不觸發任何資料重抓。
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  // ── 建立圖表 instance，只在掛載時建一次，之後只換資料不重建。容器 div 一定要無條件
  // 掛載（不能包在 `job &&` 這種條件式渲染裡）——首次進頁 job 是 null，如果容器跟著 job
  // 一起有條件掛載，這個只執行一次的 effect 第一次跑的時候容器根本不存在，之後 job 出現、
  // 容器真的掛上去了，這個 [] 依賴的 effect 也不會重新執行，圖表就永遠是空的。
  useEffect(() => {
    if (!lossChartRef.current) return;
    const chart = createChart(lossChartRef.current, {
      layout: { background: { color: "transparent" }, textColor: "#a8adb8" },
      grid: { vertLines: { color: "#23272d" }, horzLines: { color: "#23272d" } },
      autoSize: true,
      localization: { timeFormatter: (t: number) => `第 ${t} 輪` },
      timeScale: { timeVisible: false, tickMarkFormatter: (t: number) => String(t) },
    });
    // 不設定 title——lightweight-charts 會在最後一筆資料旁邊常駐畫出「title + 數值」的
    // 色塊標籤，priceLineVisible/lastValueVisible 都關不掉這個，只有不設定 title 才會消失。
    // 這正是使用者要求移除的「遮擋曲線的右側大標籤」，改用上面的固定圖例（.model-chart-legend）
    // 取代同樣的資訊。
    lossSeriesRef.current = chart.addSeries(LineSeries, {
      color: "#f0616b", priceLineVisible: false, lastValueVisible: false,
    });
    valLossSeriesRef.current = chart.addSeries(LineSeries, {
      color: "#f0a13a", lineStyle: 2, priceLineVisible: false, lastValueVisible: false,
    });
    lossChartApiRef.current = chart;
    chart.subscribeCrosshairMove((param) => {
      setLossHoverEpoch(param.time === undefined ? null : (param.time as number) - 1);
    });
    const markInteracted = () => { chartUserInteractedRef.current = true; };
    lossChartRef.current.addEventListener("wheel", markInteracted, { passive: true });
    lossChartRef.current.addEventListener("mousedown", markInteracted);
    lossChartRef.current.addEventListener("touchstart", markInteracted, { passive: true });
    return () => chart.remove();
  }, []);

  useEffect(() => {
    if (!accChartRef.current) return;
    const chart = createChart(accChartRef.current, {
      layout: { background: { color: "transparent" }, textColor: "#a8adb8" },
      grid: { vertLines: { color: "#23272d" }, horzLines: { color: "#23272d" } },
      autoSize: true,
      localization: { timeFormatter: (t: number) => `第 ${t} 輪`, priceFormatter: (p: number) => `${(p * 100).toFixed(1)}%` },
      timeScale: { timeVisible: false, tickMarkFormatter: (t: number) => String(t) },
    });
    accSeriesRef.current = chart.addSeries(LineSeries, {
      color: "#3ecf8e", priceLineVisible: false, lastValueVisible: false,
      priceFormat: { type: "custom", formatter: (p: number) => `${(p * 100).toFixed(1)}%`, minMove: 0.0001 },
    });
    valAccSeriesRef.current = chart.addSeries(LineSeries, {
      color: "#3ab0cf", lineStyle: 2, priceLineVisible: false, lastValueVisible: false,
      priceFormat: { type: "custom", formatter: (p: number) => `${(p * 100).toFixed(1)}%`, minMove: 0.0001 },
    });
    accChartApiRef.current = chart;
    chart.subscribeCrosshairMove((param) => {
      setAccHoverEpoch(param.time === undefined ? null : (param.time as number) - 1);
    });
    const markInteracted = () => { chartUserInteractedRef.current = true; };
    accChartRef.current.addEventListener("wheel", markInteracted, { passive: true });
    accChartRef.current.addEventListener("mousedown", markInteracted);
    accChartRef.current.addEventListener("touchstart", markInteracted, { passive: true });
    return () => chart.remove();
  }, []);

  useEffect(() => {
    if (!previewChartRef.current) return;
    const chart = createChart(previewChartRef.current, {
      layout: { background: { color: "transparent" }, textColor: "#a8adb8" },
      grid: { vertLines: { color: "#23272d" }, horzLines: { color: "#23272d" } },
      autoSize: true,
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#f0616b", downColor: "#3ecf8e",
      borderVisible: false, wickUpColor: "#f0616b", wickDownColor: "#3ecf8e",
    });
    previewSeriesRef.current = series;
    previewMarkersRef.current = createSeriesMarkers(series, []);
    previewChartApiRef.current = chart;
    return () => chart.remove();
  }, []);

  // ── 追蹤模式：輪詢找目前最新的 pending/running job，沒有就是「目前無訓練」──
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const list: JobSummary[] = await fetch(`${API_BASE}/api/model/jobs?limit=10`).then((r) => r.json());
        if (cancelled) return;
        setJobList(list);
        const active = list.find((j) => j.status === "pending" || j.status === "running") ?? null;
        setLatestActiveJobId(active?.job_id ?? null);
        if (mode === "history" && historyJobId) {
          const newer = list.find((j) => (j.status === "pending" || j.status === "running") && j.job_id !== historyJobId);
          setHasNewerJob(!!newer);
        } else {
          setHasNewerJob(false);
        }
      } catch {
        // 輪詢失敗不代表任務出事，安靜重試就好，不用把這個誤判成訓練失敗。
      }
    }
    poll();
    const t = setInterval(poll, 4000);
    return () => { cancelled = true; clearInterval(t); };
  }, [mode, historyJobId]);

  useEffect(() => {
    setActiveJobId(mode === "history" ? historyJobId : latestActiveJobId);
  }, [mode, historyJobId, latestActiveJobId]);

  useEffect(() => {
    if (mode === "history" && historyJobId) {
      if (searchParams.get("job") !== historyJobId) setSearchParams({ job: historyJobId }, { replace: true });
    } else if (mode === "latest" && searchParams.get("job")) {
      setSearchParams({}, { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, historyJobId]);

  function connectStream(id: string, gen: number) {
    if (gen !== loadGenRef.current) return; // 世代已經過期（包含頁面已卸載的情況），不要再建立新連線
    if (reconnectTimerRef.current) { clearTimeout(reconnectTimerRef.current); reconnectTimerRef.current = null; }
    wsRef.current?.close();
    setWsStatus("connecting");
    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsHost = API_BASE ? API_BASE.replace(/^https?:\/\//, "") : window.location.host;
    const ws = new WebSocket(`${wsProtocol}://${wsHost}/api/model/jobs/${id}/stream`);
    // 光靠 id 比對不夠：A → B → A 時，第一顆（早該淘汰的）A 連線的 id 又會跟目前顯示的
    // job 對上。改成比對 gen（跟初始載入、輪詢、finalizeProgressSync 共用同一個世代編號）
    // 加上 wsRef.current 是不是「這一顆」，才能確定這是目前唯一有效的連線。
    const isCurrent = () => gen === loadGenRef.current && wsRef.current === ws;
    ws.onopen = () => { if (isCurrent()) setWsStatus("connected"); };
    ws.onmessage = (evt) => {
      if (!isCurrent()) return; // 舊連線的訊息，不管 job id 是否剛好又相同，都不該處理
      try {
        const msg = JSON.parse(evt.data) as { type?: string; node_id: string; epoch: number; sample_id?: number } & Record<string, unknown>;
        if (msg.type === "preview") {
          // 即時視窗預覽：跟 epoch 進度是分開的訊息（見 backend routers/model_training.py
          // 的說明，NOTIFY payload 有大小限制，bars 不能跟小 payload 混在一起）。
          const preview: WindowPreview = {
            epoch: msg.epoch, source: msg.source as "train" | "val",
            decision_ts: msg.decision_ts as number, target_ts: msg.target_ts as number,
            horizon: msg.horizon as number, task_type: msg.task_type as "classification" | "regression",
            n_classes: (msg.n_classes as number | null) ?? null, labeling_rule: msg.labeling_rule as string | null,
            bars: msg.bars as WindowBar[], actual: msg.actual as number, predicted: msg.predicted as number,
          };
          setLivePreviewByNode((prev) => ({ ...prev, [msg.node_id]: preview }));
          return;
        }
        const nodeId = extractNodeId(msg as { node_id?: string });
        const next: ProgressPoint = {
          node_id: nodeId, epoch: msg.epoch,
          loss: (msg.loss as number | null) ?? null, accuracy: (msg.accuracy as number | null) ?? null,
          val_loss: (msg.val_loss as number | null) ?? null, val_accuracy: (msg.val_accuracy as number | null) ?? null,
          created_at: new Date().toISOString(),
        };
        setProgressByNode((prev) => mergeGrouped(prev, { [nodeId]: [next] }));
        setStatus("running");
      } catch {
        // 解析失敗（例如訓練端算出 NaN，Python json.dumps 印出字面 NaN token，JSON.parse
        // 嚴格拒絕）不代表訓練真的掛了，交給輪詢保底去查真實狀態，這裡不吃掉整個畫面。
      }
    };
    ws.onerror = () => { if (isCurrent()) setWsStatus("error"); };
    ws.onclose = () => {
      if (gen !== loadGenRef.current) return; // 這個世代已經過期，跟目前顯示的 job 無關，不用理它
      setWsStatus((prev) => (prev === "error" ? prev : "closed"));
      // 非預期斷線（訓練還在跑）就試著重連，斷線不等於訓練失敗，見狀態列的說明。
      if (wsRef.current === ws) {
        reconnectTimerRef.current = setTimeout(() => {
          if (gen === loadGenRef.current) connectStream(id, gen);
        }, 2000);
      }
    };
    wsRef.current = ws;
  }

  // ── 主要資料載入：只在 activeJobId 真的換掉時整套重新抓，換 job 時務必清乾淨
  // 舊的 WebSocket／重連計時器／圖表資料，不然舊任務的回應會汙染新任務畫面。
  // 這個 effect 本身（含它的 cleanup）是唯一允許無條件關閉 wsRef 的地方；
  // 其他非同步回呼一律先檢查 gen === loadGenRef.current 才能動 wsRef／狀態。 ──
  useEffect(() => {
    // gen 遞增取代原本的閉包 cancelled 旗標——理由同上面 connectStream 的說明：A → B → A
    // 快速切換時，只比對 id 會誤判舊 session 還有效。這裡遞增的 gen 同時也是輪詢／
    // finalizeProgressSync 用來判斷自己是否過期的唯一依據，三邊共用同一個數字。
    const gen = ++loadGenRef.current;
    setProgressByNode({});
    setSelectedNodeId(null);
    setProgressSyncError(false);
    setLivePreviewByNode({});
    setHistorySamples([]);
    setHistorySelectedIdx(0);
    setHistoryHasMore(true);
    setInferPredictions([]);
    setInferSelectedIdx(0);
    setInferHasMore(true);
    setInferError(false);
    inferLoadedKeyRef.current = null;
    // job 也要在這裡無條件清掉，不能只在「沒有 activeJobId」那個分支清——A 換到 B（兩個都是
    // 真的 job）時，如果 job 狀態留著 A 不動，直到 B 的 fetch 回來之前，左側設定欄全部還在
    // 顯示 A 的內容，使用者會看到「殘留舊曲線」，尤其 B 剛好還沒有任何進度時，看起來會像是
    // 「怎麼還在顯示上一筆訓練的完整曲線」。
    setJob(null);
    if (reconnectTimerRef.current) { clearTimeout(reconnectTimerRef.current); reconnectTimerRef.current = null; }
    wsRef.current?.close();
    wsRef.current = null;
    setWsStatus("closed");
    chartContextRef.current = { jobId: null, nodeId: null };
    renderStateRef.current = {};
    chartUserInteractedRef.current = false;
    lossSeriesRef.current?.setData([]);
    valLossSeriesRef.current?.setData([]);
    accSeriesRef.current?.setData([]);
    valAccSeriesRef.current?.setData([]);

    if (!activeJobId) {
      setStatus(mode === "latest" ? "目前無訓練" : "尚無訓練紀錄");
      return;
    }
    setStatus("讀取中…");

    const id = activeJobId;
    async function load() {
      const res = await fetch(`${API_BASE}/api/model/jobs/${id}`);
      if (gen !== loadGenRef.current) return;
      if (!res.ok) { setStatus("找不到這筆訓練紀錄"); return; }
      const detail: JobDetail = await res.json();
      if (gen !== loadGenRef.current) return;
      setJob(detail);
      setStatus(detail.status);

      if (detail.status === "pending" || detail.status === "running") connectStream(id, gen);

      const grouped = await fetchProgressGrouped(id);
      if (gen !== loadGenRef.current) return;
      setProgressByNode((prev) => mergeGrouped(grouped, prev)); // WS 可能已經先推了新的，prev 優先

      // infer job 的 graph_spec 是 {target_job_id, target_node_id, start, end}，沒有 nodes——
      // 這種 job 不走「模型節點」選取邏輯，主要瀏覽區改成 InferPredictionsPanel（見下方）。
      if (detail.job_type !== "infer") {
        // detail.job_type !== 'infer' 只窄化了 job_type 本身，graph_spec 仍然是
        // TrainGraphSpec | InferGraphSpec 這個型別——這兩個欄位對 TS 來說是同一個物件上
        // 互不相干的欄位，不會因為判斷其中一個就自動窄化另一個（不是真正的 discriminated
        // union）。這裡明確轉型成 TrainGraphSpec，跟上面 trainGraphSpec/inferGraphSpec
        // 的做法一致。
        const graphSpec = detail.graph_spec as TrainGraphSpec;
        const modelNodes = graphSpec.nodes.filter((n): n is ModelNode => n.type === "model");
        setSelectedNodeId((prev) => prev ?? modelNodes[0]?.id ?? null);
      }
    }
    load();
    return () => {
      // 立刻讓這個世代作廢，不管是換 job 還是元件真的卸載——換 job 的話下一次 effect
      // 執行時還會再遞增一次沒關係，反正只比對是否相等；但卸載時後面不會再有下一次
      // effect 來遞增 loadGenRef 了，如果這裡不主動讓 gen 過期，尚未返回的初始請求、
      // 輪詢、finalizeProgressSync 的重試，回來時看到的 loadGenRef.current 還是同一個
      // gen，檢查就會誤判成「還有效」，在頁面已經關閉之後才建立 WS 或發出重試請求。
      loadGenRef.current++;
      if (reconnectTimerRef.current) { clearTimeout(reconnectTimerRef.current); reconnectTimerRef.current = null; }
      if (finalizeRetryTimerRef.current) { clearTimeout(finalizeRetryTimerRef.current); finalizeRetryTimerRef.current = null; }
      wsRef.current?.close();
      wsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJobId]);

  // 換節點（同一個 job 內）也要清即時預覽/歷史瀏覽狀態，不然會看到上一個節點的抽樣視窗
  // 短暫殘留，或歷史清單裡混進不同節點的紀錄。世代編號也要在這裡遞增（不是只在換 job
  // 時才動），這樣「切換節點」本身就會讓還沒回來的舊查詢結果作廢。
  useEffect(() => {
    historyLoadGenRef.current++;
    historyLoadedKeyRef.current = null; // 這個 job+節點組合還沒（重新）觸發過歷史載入
    setLivePreviewByNode({});
    setHistorySamples([]);
    setHistorySelectedIdx(0);
    setHistoryHasMore(true);
    setHistoryError(false);
    maybeLoadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedNodeId, activeJobId]);

  // 保底：WebSocket 只有訓練端主動推進度時才會更新畫面，斷線重連需要時間、訊息也可能
  // 解析失敗，這裡除了查真實的 job 狀態，也重新補抓一次完整進度並合併——不是只查狀態，
  // 不然會出現「狀態顯示已完成，但曲線／逐輪紀錄還缺資料」的情況。狀態變成 done/failed
  // 時再多做最後一次進度核對，確保收尾前漏掉的最後幾筆也補齊。
  // job detail 跟 progress 是分開兩個請求，Promise.all 只保證同時發出，不保證誰先查到
  // 「真的結束」的那一刻——很可能 progress 那邊還停在倒數第二輪，job detail 已經查到
  // done，這時候如果直接收尾關 WS、停輪詢，最後一輪就永遠補不回來了。所以確認終止狀態
  // 之後，還要再單獨抓一次 progress、確定合併成功，才真正收尾；抓失敗就重試，不能放棄。
  // gen 涵蓋這裡所有的請求、重試、以及收尾的關閉連線操作——不是只擋初始載入，輪詢跟這個
  // 函式本身的重試都要用同一個世代編號比對，A → B → A 時延遲回來的舊一輪輪詢／補抓才不會
  // 誤判成還有效，甚至去關掉新一輪 A 才剛建立的 WS。
  async function finalizeProgressSync(id: string, gen: number, attempt = 0) {
    if (gen !== loadGenRef.current) return; // 世代已經過期，連請求都不要發
    let grouped: Record<string, ProgressPoint[]>;
    try {
      grouped = await fetchProgressGrouped(id);
    } catch {
      if (gen !== loadGenRef.current) return;
      if (attempt < 5) {
        finalizeRetryTimerRef.current = setTimeout(() => {
          finalizeRetryTimerRef.current = null;
          finalizeProgressSync(id, gen, attempt + 1);
        }, 1500);
        return; // 還有重試機會，先不關連線、也還不算同步失敗
      }
      // 重試用完仍然失敗：「訓練已完成」跟「進度同步完成」是兩件事，不能因為抓不到
      // 最後幾筆就假裝沒事——明確標成同步失敗，讓畫面能提示使用者、給重新同步的入口，
      // 同時還是要確實把連線／計時器關掉，訓練本身已經結束了，沒有理由讓 WS 繼續掛著。
      setProgressSyncError(true);
      if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
      if (reconnectTimerRef.current) { clearTimeout(reconnectTimerRef.current); reconnectTimerRef.current = null; }
      return;
    }
    if (gen !== loadGenRef.current) return;
    setProgressByNode((prev) => mergeGrouped(prev, grouped));
    setProgressSyncError(false);
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    if (reconnectTimerRef.current) { clearTimeout(reconnectTimerRef.current); reconnectTimerRef.current = null; }
  }

  function retryProgressSync() {
    if (!activeJobId) return;
    if (finalizeRetryTimerRef.current) { clearTimeout(finalizeRetryTimerRef.current); finalizeRetryTimerRef.current = null; }
    setProgressSyncError(false);
    finalizeProgressSync(activeJobId, loadGenRef.current);
  }

  useEffect(() => {
    if (!activeJobId || (status !== "pending" && status !== "running")) return;
    const id = activeJobId;
    const gen = loadGenRef.current; // 這一輪輪詢從頭到尾都認這個世代編號，不是拿 activeJobIdRef 比對 id
    const timer = setInterval(() => {
      if (gen !== loadGenRef.current) return; // 世代已過期，理論上 cleanup 已經清掉這個 interval，這裡雙重保險
      // job 狀態跟 progress 分開兩個獨立請求，不是包在同一個 Promise.all 裡——如果 progress
      // 端點持續故障，Promise.all 整組會直接 reject，連「訓練是否已經 done」都偵測不到，
      // 「訓練已完成」跟「進度同步完成」就完全沒有分開的意義了。狀態偵測要獨立於進度同步
      // 是否成功，才能保證使用者至少看得到正確的 done/failed，再由 finalizeProgressSync
      // 自己的重試/失敗提示機制去處理進度資料本身抓不抓得到。
      fetch(`${API_BASE}/api/model/jobs/${id}`).then((r) => r.json() as Promise<JobDetail>)
        .then((detail) => {
          if (gen !== loadGenRef.current) return;
          setJob((prev) => (prev && prev.job_id === detail.job_id ? detail : prev));
          if (detail.status === "done" || detail.status === "failed") {
            setStatus(detail.status);
            finalizeProgressSync(id, gen); // 確認結束後再補抓一次，不是這裡就直接關 WS／停輪詢
          }
        })
        .catch(() => {});
      fetchProgressGrouped(id)
        .then((grouped) => {
          if (gen !== loadGenRef.current) return;
          setProgressByNode((prev) => mergeGrouped(prev, grouped));
        })
        .catch(() => {});
    }, 3000);
    return () => clearInterval(timer);
  }, [activeJobId, status]);

  const progressForSelected = selectedNodeId ? progressByNode[selectedNodeId] ?? [] : [];
  const latest = progressForSelected[progressForSelected.length - 1];

  // 歷史瀏覽：依 id 游標分頁往回查，不是把全期間資料一次全部丟給前端。
  // 取消防護要同時比對 loadGenRef（job 世代）跟 historyLoadGenRef（job+節點世代）——
  // 同一個 job 內快速切換模型節點不會動到 loadGenRef，只比對它擋不住「切到新節點後，
  // 舊節點還沒回來的查詢結果被誤放進新節點畫面」，一定要兩個都比對才夠。
  async function loadHistoryPage(reset: boolean) {
    if (!activeJobId || !selectedNodeId) return;
    const gen = loadGenRef.current;
    const historyGen = historyLoadGenRef.current;
    const id = activeJobId, nodeId = selectedNodeId;
    const isStale = () => gen !== loadGenRef.current || historyGen !== historyLoadGenRef.current;
    setHistoryLoading(true);
    setHistoryError(false);
    try {
      const beforeId = reset ? undefined : historySamples[historySamples.length - 1]?.id;
      const params = new URLSearchParams({ node_id: nodeId, limit: String(HISTORY_PAGE_SIZE) });
      if (beforeId !== undefined) params.set("before_id", String(beforeId));
      const res = await fetch(`${API_BASE}/api/model/jobs/${id}/preview_samples?${params}`);
      if (isStale()) return; // job 或選定的模型節點已經換了，這批回應作廢，不能寫進畫面
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const rows: WindowPreview[] = await res.json();
      if (isStale()) return;
      setHistorySamples((prev) => (reset ? rows : [...prev, ...rows]));
      setHistoryHasMore(rows.length === HISTORY_PAGE_SIZE);
      if (reset) setHistorySelectedIdx(0);
    } catch {
      // 明確跟「查得到、但確實是空的」分開——載入失敗要讓使用者知道可以重試，
      // 不能跟「這個任務真的沒有紀錄」用同一句話帶過。
      if (!isStale()) setHistoryError(true);
    } finally {
      if (!isStale()) setHistoryLoading(false);
    }
  }

  // 只在「進入歷史分頁」跟「換 job/節點」兩種時機各觸發一次判斷，判斷本身用
  // historyLoadedKeyRef（見上面宣告處的說明），不是用 historySamples.length===0，
  // 避免同一輪 render 裡的 state 時序問題讓某些切換永遠不會真的發出查詢。
  function maybeLoadHistory() {
    if (viewMode !== "history" || !activeJobId || !selectedNodeId) return;
    const key = `${activeJobId}:${selectedNodeId}`;
    if (historyLoadedKeyRef.current === key) return;
    historyLoadedKeyRef.current = key;
    loadHistoryPage(true);
  }

  useEffect(() => {
    maybeLoadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode]);

  // 最終模型推論的分頁載入——只在 job_type==='infer' 且已經完成（有結果可查）時觸發，
  // 用 job_id 當 key（一個 infer job 只有一個 target_node_id，不用像訓練歷史那樣再疊節點）。
  // 用 gen（loadGenRef）擋過期回應，跟其餘非同步流程共用同一套世代機制。
  async function loadInferPage(reset: boolean) {
    if (!activeJobId || !isInferJob || !inferGraphSpec) return;
    const gen = loadGenRef.current;
    const id = activeJobId;
    const isStale = () => gen !== loadGenRef.current;
    setInferLoading(true);
    setInferError(false);
    try {
      const cursor = reset ? undefined : inferPredictions[inferPredictions.length - 1]?.id;
      const params = new URLSearchParams({ node_id: inferGraphSpec.target_node_id, limit: String(HISTORY_PAGE_SIZE) });
      if (cursor !== undefined) params.set("cursor", String(cursor));
      const res = await fetch(`${API_BASE}/api/model/jobs/${id}/inference_predictions?${params}`);
      if (isStale()) return;
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const rows: InferPrediction[] = await res.json();
      if (isStale()) return;
      setInferPredictions((prev) => (reset ? rows : [...prev, ...rows]));
      setInferHasMore(rows.length === HISTORY_PAGE_SIZE);
      if (reset) setInferSelectedIdx(0);
    } catch {
      if (!isStale()) setInferError(true);
    } finally {
      if (!isStale()) setInferLoading(false);
    }
  }

  useEffect(() => {
    if (!isInferJob || !activeJobId || status !== "done") return;
    if (inferLoadedKeyRef.current === activeJobId) return;
    inferLoadedKeyRef.current = activeJobId;
    loadInferPage(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isInferJob, activeJobId, status]);

  const displayedPreview: WindowPreview | null =
    viewMode === "live"
      ? (selectedNodeId ? livePreviewByNode[selectedNodeId] ?? null : null)
      : historySamples[historySelectedIdx] ?? null;

  // 抽樣視窗圖只有 window 根K線，資料量小，每次都整包 setData／setMarkers 沒有效能疑慮。
  useEffect(() => {
    if (!previewSeriesRef.current) return;
    if (!displayedPreview) {
      previewSeriesRef.current.setData([]);
      previewMarkersRef.current?.setMarkers([]);
      return;
    }
    previewSeriesRef.current.setData(
      displayedPreview.bars.map((b) => ({ time: toChartTime(b.time), open: b.open, high: b.high, low: b.low, close: b.close }))
    );
    previewMarkersRef.current?.setMarkers([
      { time: toChartTime(displayedPreview.decision_ts), position: "belowBar", color: "#e8b339", shape: "arrowUp", text: "T（決策點）" },
    ]);
    previewChartApiRef.current?.timeScale().fitContent();
  }, [displayedPreview]);

  // 圖表同步只有一個 effect，內部按情境分流，不再拆成「先 update() 後 setData()」兩個各自
  // 判斷 selectedNodeId 的 effect——那樣兩個 effect 誰先跑完全看宣告順序，換節點時「增量
  // update()」那個永遠先跑，會拿新節點的資料去 update() 舊節點還留在圖表上的 series，一旦
  // 新節點的 epoch 比舊節點畫到的位置早（例如舊節點已經到第 100 輪，新節點才第 1 輪），
  // lightweight-charts 的 update() 會直接丟例外（時間往回退）。全部併進同一個 effect 之後，
  // 順序不再是問題，改成用資料本身判斷屬於哪一種情境：
  //   1) 首次載入 / 切換節點或任務 → chartContextRef 的 job/node 對不上了 → 整包 setData + fitContent
  //   2) 純粹尾端追加新 epoch（沒有任何一筆是回頭補洞或改值）→ 用 update() 逐筆疊上去；
  //      使用者沒手動縮放過就順便 fitContent()，讓已完成輪數預設充分使用圖表寬度——
  //      動過縮放/捲動之後就尊重使用者的視野，不再搶回去。
  //   3) 歷史補洞（epoch 比目前畫到的還舊）或既有 epoch 的數值被訂正 → 整包 setData 重畫，
  //      使用者手動縮放過才需要保留原本可視範圍，否則一樣直接 fitContent()。
  useEffect(() => {
    if (!selectedNodeId || !activeJobId) return;
    const key = `${activeJobId}:${selectedNodeId}`;
    const pts = progressByNode[selectedNodeId] ?? [];

    const contextChanged = chartContextRef.current.jobId !== activeJobId || chartContextRef.current.nodeId !== selectedNodeId;
    chartContextRef.current = { jobId: activeJobId, nodeId: selectedNodeId };
    if (contextChanged) chartUserInteractedRef.current = false;

    let state = renderStateRef.current[key];
    if (!state) {
      state = { points: new Map(), maxEpoch: null };
      renderStateRef.current[key] = state;
    }

    function setFullData() {
      lossSeriesRef.current?.setData(pts.filter((p) => p.loss !== null).map((p) => ({ time: (p.epoch + 1) as unknown as Time, value: p.loss as number })));
      valLossSeriesRef.current?.setData(pts.filter((p) => p.val_loss !== null).map((p) => ({ time: (p.epoch + 1) as unknown as Time, value: p.val_loss as number })));
      accSeriesRef.current?.setData(pts.filter((p) => p.accuracy !== null).map((p) => ({ time: (p.epoch + 1) as unknown as Time, value: p.accuracy as number })));
      valAccSeriesRef.current?.setData(pts.filter((p) => p.val_accuracy !== null).map((p) => ({ time: (p.epoch + 1) as unknown as Time, value: p.val_accuracy as number })));
    }
    function fitBoth() {
      lossChartApiRef.current?.timeScale().fitContent();
      accChartApiRef.current?.timeScale().fitContent();
    }

    if (contextChanged) {
      // 首次顯示這個節點，或切換了節點/任務：圖表上現在顯示的內容跟這份資料完全無關，
      // 一定要整包換掉，而且視野也該重新 fit（不同節點的輪數/尺度通常差很多）。
      setFullData();
      fitBoth();
    } else {
      let needsFullRedraw = false;
      const appended: ProgressPoint[] = [];
      for (const p of pts) {
        const prev = state.points.get(p.epoch);
        if (!prev) {
          if (state.maxEpoch !== null && p.epoch <= state.maxEpoch) { needsFullRedraw = true; break; }
          appended.push(p);
        } else if (
          prev.loss !== p.loss || prev.val_loss !== p.val_loss ||
          prev.accuracy !== p.accuracy || prev.val_accuracy !== p.val_accuracy
        ) {
          needsFullRedraw = true; break; // 既有 epoch 的數值被訂正，不是單純新增
        }
      }
      if (needsFullRedraw) {
        if (chartUserInteractedRef.current) {
          const lossRange = lossChartApiRef.current?.timeScale().getVisibleLogicalRange();
          const accRange = accChartApiRef.current?.timeScale().getVisibleLogicalRange();
          setFullData();
          if (lossRange) lossChartApiRef.current?.timeScale().setVisibleLogicalRange(lossRange);
          if (accRange) accChartApiRef.current?.timeScale().setVisibleLogicalRange(accRange);
        } else {
          setFullData();
          fitBoth();
        }
      } else if (appended.length > 0) {
        for (const p of appended.sort((a, b) => a.epoch - b.epoch)) {
          const t = (p.epoch + 1) as unknown as Time;
          if (p.loss !== null) lossSeriesRef.current?.update({ time: t, value: p.loss });
          if (p.val_loss !== null) valLossSeriesRef.current?.update({ time: t, value: p.val_loss });
          if (p.accuracy !== null) accSeriesRef.current?.update({ time: t, value: p.accuracy });
          if (p.val_accuracy !== null) valAccSeriesRef.current?.update({ time: t, value: p.val_accuracy });
        }
        if (!chartUserInteractedRef.current) fitBoth();
      }
      // appended.length === 0 && !needsFullRedraw：完全沒有新資料，不用碰圖表
    }

    for (const p of pts) state.points.set(p.epoch, p);
    if (pts.length) state.maxEpoch = pts[pts.length - 1].epoch;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progressByNode, selectedNodeId, activeJobId]);

  const modelNodes = useMemo(() => trainGraphSpec?.nodes.filter((n): n is ModelNode => n.type === "model") ?? [], [trainGraphSpec]);
  const modelNode = modelNodes.find((n) => n.id === selectedNodeId) ?? modelNodes[0];

  // 依「選定的模型節點」自己的 label／inputs 取對應設定，不是整個 graph 隨便抓
  // 第一個 label/feature 節點——多模型節點時，B 模型不該顯示成 A 模型的 horizon
  // 或特徵（Codex 抓到的 bug）。
  const modelLabelNode = useMemo(
    () => trainGraphSpec?.nodes.find((n): n is LabelNode => n.type === "label" && n.id === modelNode?.label),
    [trainGraphSpec, modelNode]
  );
  const modelFeatureNodes = useMemo(() => {
    if (!modelNode) return [];
    const inputSet = new Set(modelNode.inputs);
    return trainGraphSpec?.nodes.filter((n): n is FeatureNode => n.type === "feature" && inputSet.has(n.id)) ?? [];
  }, [trainGraphSpec, modelNode]);

  const resultEntry = modelNode ? job?.result?.[modelNode.id] : undefined;
  // infer job 的 result 用 target_node_id 當 key（見 training/worker.py 的 run_one_infer），
  // 不是 modelNode.id——infer job 根本沒有 modelNode。
  const inferResultEntry = inferGraphSpec ? job?.result?.[inferGraphSpec.target_node_id] : undefined;
  const selectedInferPrediction = inferPredictions[inferSelectedIdx] ?? null;
  const totalEpochs = Number(modelNode?.params?.epochs ?? modelNode?.params?.n_estimators ?? 0) || null;
  const lastProgressAt = latest?.created_at ?? null;

  const featureLabel = (n: FeatureNode) => (n.key === "talib_indicator" ? String(n.params?.name ?? n.key) : n.key);

  // Loss/Accuracy 圖例：預設顯示最新一輪，滑鼠移到圖上時改顯示那一輪的 train/val 數值——
  // 兩張圖各自獨立判斷 hover 狀態。
  const lossLegendPoint = (lossHoverEpoch !== null ? progressForSelected.find((p) => p.epoch === lossHoverEpoch) : undefined) ?? latest;
  const accLegendPoint = (accHoverEpoch !== null ? progressForSelected.find((p) => p.epoch === accHoverEpoch) : undefined) ?? latest;

  return (
    <div className="model-page">
      {/* 頂部：任務選擇 + 狀態列 */}
      <div className="model-topbar">
        <div className="model-mode-switch">
          <select
            className="model-mode-select"
            value={mode === "latest" ? "latest" : (historyJobId ?? "")}
            onChange={(e) => {
              if (e.target.value === "latest") { setMode("latest"); }
              else { setMode("history"); setHistoryJobId(e.target.value); }
            }}
          >
            <option value="latest">追蹤目前訓練</option>
            <optgroup label="訓練紀錄">
              {jobList.map((j) => (
                <option key={j.job_id} value={j.job_id}>
                  {new Date(j.created_at).toLocaleString("zh-TW")}　{STATUS_LABEL[j.status] ?? j.status}{j.job_type === "infer" ? "［推論］" : ""}
                </option>
              ))}
            </optgroup>
          </select>
          {mode === "history" && hasNewerJob && (
            <button type="button" className="model-switch-hint" onClick={() => setMode("latest")}>
              有新任務在跑，切換查看 →
            </button>
          )}
        </div>

        <div className={`model-status${status === "failed" ? " is-error" : ""}`}>
          {job ? (
            <>
              {STATUS_LABEL[status] ?? status}
              {totalEpochs && latest ? `｜已完成 ${latest.epoch + 1}／${totalEpochs} 輪` : ""}
              {(status === "pending" || status === "running") && `｜最後進度回報：${secondsAgo(lastProgressAt, now)}`}
              {(status === "pending" || status === "running") && wsStatus === "error" && (
                <span className="model-status-warn">　連線中斷，正在重新連線（訓練本身不受影響）</span>
              )}
              {status === "failed" && job.error && <div className="model-status-error">{job.error}</div>}
              {(status === "done" || status === "failed") && progressSyncError && (
                <div className="model-status-error">
                  進度資料同步失敗，可能缺少最後幾筆紀錄。
                  <button type="button" className="model-jobid-toggle" onClick={retryProgressSync}>重新同步</button>
                </div>
              )}
            </>
          ) : (
            status
          )}
          <button type="button" className="model-jobid-toggle" onClick={() => setShowJobIdDetail((v) => !v)}>
            詳細資訊 {showJobIdDetail ? "▲" : "▼"}
          </button>
          {showJobIdDetail && job && (
            <div className="model-jobid-detail">
              job_id：<code>{job.job_id}</code>
              <button type="button" onClick={() => navigator.clipboard?.writeText(job.job_id)}>複製</button>
            </div>
          )}
        </div>
      </div>

      <div className="model-body">
        {!job && <div className="model-empty-overlay">{status}</div>}

        <div className="model-left">
          {job && isInferJob && inferGraphSpec && (
            // infer job：載入已保存模型對新區間做推論，沒有訓練節點圖／特徵/Loss/Accuracy
            // 這些訓練專用資訊，設定欄只顯示這個推論任務實際引用了哪個訓練 job/節點、
            // 查詢區間，以及寫入結果的摘要（完整逐列結果在下方 InferPredictionsPanel 瀏覽）。
            <div className="model-left-section">
              <div className="model-left-title">推論設定</div>
              <div className="model-info-row"><span>來源訓練 job</span><strong>{inferGraphSpec.target_job_id}</strong></div>
              <div className="model-info-row"><span>模型節點</span><strong>{inferGraphSpec.target_node_id}</strong></div>
              <div className="model-info-row"><span>推論期間</span><strong>{inferGraphSpec.start} ~ {inferGraphSpec.end}</strong></div>
              <div className="model-info-row"><span>Device</span><strong>{job.device ?? "—"}</strong></div>
              {inferResultEntry && (
                <>
                  <div className="model-info-row"><span>推論筆數</span><strong>{inferResultEntry.count ?? "—"}</strong></div>
                  <div className="model-info-row"><span>結果時間範圍</span><strong>
                    {inferResultEntry.start_ts ? formatPreviewTime(inferResultEntry.start_ts) : "—"}
                    {" ~ "}
                    {inferResultEntry.end_ts ? formatPreviewTime(inferResultEntry.end_ts) : "—"}
                  </strong></div>
                </>
              )}
            </div>
          )}
          {job && !isInferJob && (
            <>
              <div className="model-left-section">
                <div className="model-left-title">資料</div>
                <div className="model-info-row"><span>商品／時間框架</span><strong>{modelFeatureNodes[0]?.timeframe ?? "—"}</strong></div>
                <div className="model-info-row"><span>期間</span><strong>{trainGraphSpec?.start} ~ {trainGraphSpec?.end}</strong></div>
                {resultEntry?.training_meta && (
                  <>
                    <div className="model-info-row"><span>訓練樣本數</span><strong>{resultEntry.training_meta.n_train ?? "—"}</strong></div>
                    <div className="model-info-row"><span>驗證樣本數</span><strong>{resultEntry.training_meta.n_val ?? "—"}</strong></div>
                  </>
                )}
              </div>

              <div className="model-left-section">
                <div className="model-left-title">預測目標</div>
                <div className="model-info-row"><span>outcome</span><strong>{modelLabelNode?.outcome ?? "—"}</strong></div>
                <div className="model-info-row"><span>labeling_rule</span><strong>{modelLabelNode?.labeling_rule ?? "—"}</strong></div>
                <div className="model-info-row"><span>horizon</span><strong>{String(modelLabelNode?.params?.horizon ?? "—")}</strong></div>
                <div className="model-info-row"><span>n_classes</span><strong>{String(modelLabelNode?.params?.n_classes ?? "—")}</strong></div>
                {modelLabelNode?.params?.threshold_pct !== undefined && (
                  <div className="model-info-row"><span>threshold_pct</span><strong>{String(modelLabelNode.params.threshold_pct)}</strong></div>
                )}
              </div>

              <div className="model-left-section">
                <div className="model-left-title">
                  模型{modelNodes.length > 1 && (
                    <select className="model-node-select" value={selectedNodeId ?? ""} onChange={(e) => setSelectedNodeId(e.target.value)}>
                      {modelNodes.map((n) => <option key={n.id} value={n.id}>{n.id}</option>)}
                    </select>
                  )}
                </div>
                <div className="model-info-row"><span>architecture</span><strong>{modelNode?.key ?? "—"}</strong></div>
                <div className="model-info-row"><span>window</span><strong>{modelNode?.window ?? "—"}</strong></div>
                <div className="model-info-row"><span>epochs</span><strong>{String(modelNode?.params?.epochs ?? "—")}</strong></div>
                <div className="model-info-row"><span>batch_size</span><strong>{String(modelNode?.params?.batch_size ?? "—")}</strong></div>
                <div className="model-info-row"><span>units</span><strong>{String(modelNode?.params?.units ?? "—")}</strong></div>
                <div className="model-info-row"><span>layers</span><strong>{String(modelNode?.params?.layers ?? "—")}</strong></div>
                <div className="model-info-row"><span>Device</span><strong>{resultEntry?.device ?? job.device ?? "—"}</strong></div>
                {modelNode?.params && (
                  <details className="model-params-detail">
                    <summary>其他參數</summary>
                    {Object.entries(modelNode.params)
                      .filter(([k]) => !["epochs", "batch_size", "units", "layers", "n_classes"].includes(k))
                      .map(([k, v]) => (
                        <div className="model-info-row" key={k}><span>{k}</span><strong>{String(v)}</strong></div>
                      ))}
                  </details>
                )}
              </div>

              <div className="model-left-section">
                <div className="model-left-title-row">
                  <div className="model-left-title">特徵（{modelFeatureNodes.length}）</div>
                  {modelFeatureNodes.length > 0 && (
                    <div className="model-feature-bulk">
                      <button type="button" onClick={() => setExpandedFeature(Object.fromEntries(modelFeatureNodes.map((n) => [n.id, true])))}>全部展開</button>
                      <button type="button" onClick={() => setExpandedFeature({})}>全部收合</button>
                    </div>
                  )}
                </div>
                <div className="model-feature-tags">
                  {modelFeatureNodes.map((n) => (
                    <button
                      type="button"
                      key={n.id}
                      className="model-feature-tag"
                      onClick={() => setExpandedFeature((p) => ({ ...p, [n.id]: !p[n.id] }))}
                    >
                      {featureLabel(n)}
                    </button>
                  ))}
                  {modelFeatureNodes.length === 0 && <span className="model-info-row"><span>—</span></span>}
                </div>
                {modelFeatureNodes.filter((n) => expandedFeature[n.id] && n.params).map((n) => (
                  <div className="model-feature-params" key={n.id}>
                    <div className="model-left-title">{featureLabel(n)} 參數</div>
                    {Object.entries(n.params ?? {}).filter(([k]) => k !== "name").map(([k, v]) => (
                      <div className="model-info-row" key={k}><span>{k}</span><strong>{String(v)}</strong></div>
                    ))}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="model-right">
          {/* infer job：全期間最終推論的逐列瀏覽——跟下面訓練 job 專用的「即時預覽／歷史瀏覽」
              是完全不同的資料來源與畫面，這裡沒有 bars／actual（推論本身不重新標記真實答案），
              只有模型真正輸出的 predicted／probabilities，沿用同樣的 .model-primary-wrap／
              .model-history-list 版面（依使用者指示：版面沿用既有改善，不再另開美化工作）。 */}
          {isInferJob && (
            <div className="model-primary-wrap">
              <div className="model-primary-header">
                <div className="model-left-title">全期間推論結果</div>
                {inferResultEntry?.count !== undefined && (
                  <span className="model-preview-badge">共 {inferResultEntry.count} 筆</span>
                )}
              </div>
              <div className="model-preview-body">
                <div className="model-preview-info">
                  {selectedInferPrediction ? (
                    <>
                      <div className="model-preview-info-row"><span>時間</span><strong>{formatPreviewTime(selectedInferPrediction.ts)}</strong></div>
                      <div className="model-preview-info-row"><span>輸出類型</span><strong>{selectedInferPrediction.output_type}</strong></div>
                      <div className="model-preview-info-row"><span>predicted</span><strong>{selectedInferPrediction.predicted}</strong></div>
                      {selectedInferPrediction.probabilities && (
                        <div className="model-preview-info-row">
                          <span>機率分佈</span>
                          <strong>{selectedInferPrediction.probabilities.map((p) => `${(p * 100).toFixed(1)}%`).join(" / ")}</strong>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="model-preview-chart-overlay">
                      {status !== "done"
                        ? "推論任務尚未完成"
                        : inferLoading
                          ? "載入中…"
                          : inferError
                            ? "載入失敗"
                            : "此任務尚無推論結果"}
                    </div>
                  )}
                </div>
              </div>
              <div className="model-history-list">
                {inferPredictions.length === 0 && !inferLoading && inferError && (
                  <div className="model-history-empty">
                    載入失敗，可能是網路或伺服器暫時的問題。
                    <button type="button" className="model-history-retry" onClick={() => loadInferPage(true)}>重試</button>
                  </div>
                )}
                {inferPredictions.length === 0 && !inferLoading && !inferError && status === "done" && (
                  <div className="model-history-empty">此任務尚無推論結果</div>
                )}
                {inferPredictions.map((p, i) => (
                  <div
                    key={p.id}
                    className={`model-history-row${i === inferSelectedIdx ? " is-selected" : ""}`}
                    onClick={() => setInferSelectedIdx(i)}
                  >
                    <span>{formatPreviewTime(p.ts)}</span>
                    <span>{p.output_type}</span>
                    <span>predicted {p.predicted}</span>
                    <span>{p.probabilities ? p.probabilities.map((v) => `${(v * 100).toFixed(1)}%`).join(" / ") : "—"}</span>
                  </div>
                ))}
                {inferHasMore && inferPredictions.length > 0 && (
                  <button type="button" className="model-history-more" disabled={inferLoading} onClick={() => loadInferPage(false)}>
                    {inferLoading ? "載入中…" : "載入更多"}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* B：主要區域——訓練即時預覽（正在處理的抽樣 window，持續更換）／完成後歷史瀏覽，
              兩者共用同一個 K 線圖表，不是留一張跟核心功能無關的大歷史圖再把功能塞進小圖。
              只有 job_type==='train' 才有這些——infer job 沒有訓練過程，改看上面的分支。
              用 style 隱藏、不是條件式掛載：底下 previewChartRef 容器的 lightweight-charts
              instance 只在整個元件掛載時建立一次（見上面該 effect 的說明），容器一旦被
              條件式移出 DOM 就再也不會重建，之後切回訓練 job 會拿到已經失效的 chart 參照。 */}
          <div className="model-primary-wrap" style={isInferJob ? { display: "none" } : undefined}>
            <div className="model-primary-header">
              <div className="model-mode-tabs">
                <button type="button" className={viewMode === "live" ? "is-active" : ""} onClick={() => setViewMode("live")}>訓練即時預覽</button>
                <button type="button" className={viewMode === "history" ? "is-active" : ""} onClick={() => setViewMode("history")}>完成後歷史瀏覽</button>
              </div>
              {viewMode === "live" && displayedPreview && <span className="model-preview-badge">抽樣視窗（持續更換，非代表全部）</span>}
            </div>

            <div className="model-preview-body">
              <div className="model-preview-chart-wrap">
                {!displayedPreview && (
                  <div className="model-preview-chart-overlay">
                    {!job
                      ? ""
                      : viewMode === "live"
                        ? (status === "pending" || status === "running")
                          ? "等待訓練回報第一筆抽樣視窗…"
                          // 已結束的任務不會再建立 WebSocket 連線（見 connectStream 只在
                          // pending/running 時才呼叫），「等待回報」在這裡是誤導的說法——
                          // 訓練已經結束了，不會再有新的即時預覽進來。
                          : "訓練已結束，若曾記錄過抽樣視窗，請切換「完成後歷史瀏覽」查看"
                        : historyLoading
                          ? "載入中…"
                          : historyError
                            ? "載入失敗"
                            // 查詢成功但確實是空的：這個任務（可能是這個功能上線前建立的舊任務）
                            // 從頭到尾沒有記錄過任何一筆抽樣，不是「還在等」，要明確說清楚。
                            : "此任務未記錄預覽"}
                  </div>
                )}
                <div ref={previewChartRef} className="model-preview-chart" />
              </div>
              <div className="model-preview-info">
                {displayedPreview && (
                  <>
                    <div className="model-preview-info-row"><span>來源</span><strong>{displayedPreview.source === "train" ? "訓練集 (train)" : "驗證集 (val)"}</strong></div>
                    <div className="model-preview-info-row"><span>第幾輪</span><strong>第 {displayedPreview.epoch + 1} 輪</strong></div>
                    <div className="model-preview-info-row"><span>決策時刻 T</span><strong>{formatPreviewTime(displayedPreview.decision_ts)}</strong></div>
                    <div className="model-preview-info-row"><span>預測目標 T+{displayedPreview.horizon}</span><strong>{formatPreviewTime(displayedPreview.target_ts)}</strong></div>
                    <div className="model-preview-info-row"><span>預測</span><strong>{formatPreviewValue(displayedPreview.predicted, displayedPreview.task_type, displayedPreview.n_classes, displayedPreview.labeling_rule)}</strong></div>
                    <div className="model-preview-info-row"><span>實際答案</span><strong>{formatPreviewValue(displayedPreview.actual, displayedPreview.task_type, displayedPreview.n_classes, displayedPreview.labeling_rule)}</strong></div>
                    {displayedPreview.task_type === "classification" && (
                      <div className={`model-preview-verdict ${displayedPreview.predicted === displayedPreview.actual ? "is-correct" : "is-wrong"}`}>
                        {displayedPreview.predicted === displayedPreview.actual ? "這一筆預測正確" : "這一筆預測錯誤"}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {viewMode === "history" && (
              <div className="model-history-list">
                {historySamples.length === 0 && !historyLoading && historyError && (
                  <div className="model-history-empty">
                    載入失敗，可能是網路或伺服器暫時的問題。
                    <button type="button" className="model-history-retry" onClick={() => loadHistoryPage(true)}>重試</button>
                  </div>
                )}
                {historySamples.length === 0 && !historyLoading && !historyError && (
                  <div className="model-history-empty">此任務未記錄預覽</div>
                )}
                {historySamples.map((s, i) => (
                  <div
                    key={s.id ?? i}
                    className={`model-history-row${i === historySelectedIdx ? " is-selected" : ""}`}
                    onClick={() => setHistorySelectedIdx(i)}
                  >
                    <span>{formatPreviewTime(s.decision_ts)}</span>
                    <span>{s.source === "train" ? "train" : "val"}</span>
                    <span>預測 {formatPreviewValue(s.predicted, s.task_type, s.n_classes, s.labeling_rule)}</span>
                    <span>實際 {formatPreviewValue(s.actual, s.task_type, s.n_classes, s.labeling_rule)}</span>
                  </div>
                ))}
                {historyHasMore && (
                  <button type="button" className="model-history-more" disabled={historyLoading} onClick={() => loadHistoryPage(false)}>
                    {historyLoading ? "載入中…" : "載入更早的紀錄"}
                  </button>
                )}
              </div>
            )}
          </div>

          {/* C：訓練指標——Loss 跟 Accuracy 分成兩張獨立圖表，橫軸是訓練輪次。infer job
              沒有訓練過程，一樣用 style 隱藏（理由同上，lossChartRef/accChartRef 容器也是
              只在元件掛載時建立一次的 chart instance）。 */}
          <div className="model-progress-wrap" style={isInferJob ? { display: "none" } : undefined}>
            <div className="model-progress-header">
              <div className="model-left-title">訓練指標</div>
              {job && totalEpochs && latest && (status === "pending" || status === "running") && (
                <div className="model-compact-progress">
                  正在第 {latest.epoch + 1}／{totalEpochs} 輪｜最後回報：{secondsAgo(lastProgressAt, now)}
                </div>
              )}
            </div>
            <div className="model-progress-charts">
              <div className="model-progress-chart-block">
                <div className="model-chart-label">Loss</div>
                <div className="model-progress-chart-inner">
                  {job && progressForSelected.length === 0 && (status === "pending" || status === "running") && (
                    <div className="model-progress-waiting">等待第一輪回報…</div>
                  )}
                  {job && progressForSelected.length > 0 && (
                    <div className="model-chart-legend">
                      <span className="model-legend-item"><i className="model-legend-swatch is-solid" style={{ background: "#f0616b" }} />train {fmt(lossLegendPoint?.loss, 4)}</span>
                      <span className="model-legend-item"><i className="model-legend-swatch is-dashed" style={{ borderColor: "#f0a13a" }} />val {fmt(lossLegendPoint?.val_loss, 4)}</span>
                      {lossLegendPoint && <span className="model-legend-epoch">第 {lossLegendPoint.epoch + 1} 輪</span>}
                    </div>
                  )}
                  <div ref={lossChartRef} className="model-progress-chart" />
                </div>
              </div>
              <div className="model-progress-chart-block">
                <div className="model-chart-label">Accuracy</div>
                <div className="model-progress-chart-inner">
                  {job && progressForSelected.length === 0 && (status === "pending" || status === "running") && (
                    <div className="model-progress-waiting">等待第一輪回報…</div>
                  )}
                  {job && progressForSelected.length > 0 && (
                    <div className="model-chart-legend">
                      <span className="model-legend-item"><i className="model-legend-swatch is-solid" style={{ background: "#3ecf8e" }} />train {fmtPct(accLegendPoint?.accuracy)}</span>
                      <span className="model-legend-item"><i className="model-legend-swatch is-dashed" style={{ borderColor: "#3ab0cf" }} />val {fmtPct(accLegendPoint?.val_accuracy)}</span>
                      {accLegendPoint && <span className="model-legend-epoch">第 {accLegendPoint.epoch + 1} 輪</span>}
                    </div>
                  )}
                  <div ref={accChartRef} className="model-progress-chart" />
                </div>
              </div>
            </div>

            {job && (
              <>
                <button type="button" className="model-epoch-toggle" onClick={() => setShowFullEpochList((v) => !v)}>
                  {showFullEpochList ? "收起" : "展開"}逐輪明細（{progressForSelected.length} 筆）
                </button>
                {showFullEpochList && (
                  <div className="model-epoch-list">
                    <div className="model-epoch-row model-epoch-header">
                      <span>epoch</span><span>loss</span><span>accuracy</span><span>val_loss</span><span>val_accuracy</span>
                    </div>
                    {progressForSelected.map((p) => (
                      <div className="model-epoch-row" key={p.epoch}>
                        <span>{p.epoch}</span>
                        <span>{fmt(p.loss)}</span>
                        <span>{fmtPct(p.accuracy)}</span>
                        <span>{fmt(p.val_loss)}</span>
                        <span>{fmtPct(p.val_accuracy)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
