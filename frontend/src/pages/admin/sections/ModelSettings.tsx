import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  addEdge,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type Connection,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./ModelSettings.css";
import "./TreeBuilder.css";
import { IndicatorAdder, findIndicator, type RegistryGroup } from "./TreeBuilder";

const API_BASE = import.meta.env.VITE_API_BASE as string;
const TIMEFRAMES = ["1m", "5m", "15m", "30m", "60m", "1d"];
const SYMBOLS = [
  { value: "tx", label: "TX（台指期）" },
  { value: "mtx", label: "MTX（小台指）" },
  { value: "tmf", label: "TMF（微台指）" },
];
// 這幾個 Custom 特徵本身就是後端真正的 feature key（不用像 TA-Lib 指標那樣包一層
// talib_indicator + name），選到這幾個要走不同的節點格式，兩個地方都要判斷，抽出來共用。
const CUSTOM_FEATURE_KEYS = ["ohlcv", "raw_price", "raw_volume"];

type RegistryEntry = { key: string; description: string };
type SavedStrategy = { saved_id: string; job_id: string; name: string | null; symbol: string; timeframe: string; created_at: string };
// group：整組加入時標記這個特徵屬於哪個 TA-Lib 分類（例如「動量指標」），純 UI 用途，
// 讓畫面能把同一組加進來的特徵收在一起顯示、一鍵整組移除；不影響 buildGraphSpec() 送出的節點格式。
// params 是一個彈性的 key→數字 字典，對應該指標實際的參數schema（例如 RSI 是
// {timeperiod:14}，MACD 是 {fastperiod,slowperiod,signalperiod}，ACOS 這種沒有參數的
// 就是空物件 {}）——不是每個 TA-Lib 指標都只有 timeperiod 這一個參數，之前寫死成單一
// timeperiod 欄位，選到 MACD/STOCH 這類指標送出去會在後端 talib.abstract 那層直接
// KeyError('timeperiod')，因為那個指標的參數表裡根本沒有這個 key。
type FeatureNodeConfig = { id: string; key: string; name: string; params: Record<string, number>; savedId: string; group?: string };

type TrainingModule = {
  id: string;
  name: string;
  symbol: string;
  timeframe: string;
  start: string;
  end: string;
  featureNodes: FeatureNodeConfig[];
  outcomeKey: string;
  labelingRuleKey: string;
  horizon: number;
  nClasses: number;
  thresholdPct: number;
  architectureKey: string;
  window: number;
  valRatio: number;
  splitStrategy: "random" | "chronological";
  epochs: number;
  outputMode: "class" | "probability";
  // Architecture 超參數，依 architectureKey 決定哪些欄位有意義（見 ARCHITECTURE_PARAM_FIELDS）
  units: number;
  layers: number;
  dropout: number;
  dense: number;
  headDropout: number;
  l2Lambda: number;
  patience: number;
  bidirectional: boolean;
  classWeight: "none" | "balanced";
  batchSize: number;
  learningRate: number;
  nEstimators: number;
  maxDepth: number;
  subsample: number;
  colsampleBytree: number;
};

// outcome key -> 不合法的 labeling_rule 集合（跟後端 training/registry/labeling_rules.py 的
// ILLEGAL_COMBINATIONS 保持一致：future_price 是價格絕對值域，不能直接套百分比門檻分類）。
const ILLEGAL_LABELING_RULES: Record<string, string[]> = {
  future_price: ["fixed_threshold"],
};

// 每個 architecture 對應要顯示哪些超參數欄位；純數字輸入框為主，不用花俏 UI，
// 只有 bidirectional（開關）跟 class_weight（none/balanced 兩種選項）不是數字。
type ArchParamField =
  | { key: keyof TrainingModule; label: string; type: "number"; step?: number }
  | { key: keyof TrainingModule; label: string; type: "checkbox" }
  | { key: keyof TrainingModule; label: string; type: "select"; options: { value: string; label: string }[] };

const ARCHITECTURE_PARAM_FIELDS: Record<string, ArchParamField[]> = {
  lstm: [
    { key: "units", label: "units", type: "number" },
    { key: "layers", label: "layers", type: "number" },
    { key: "dropout", label: "dropout（LSTM 層）", type: "number", step: 0.05 },
    { key: "dense", label: "dense（隱藏層寬度）", type: "number" },
    { key: "headDropout", label: "head_dropout（Dense 後）", type: "number", step: 0.05 },
    // bidirectional 改放共用第一排（跟 architecture/window 同一行），這裡不重複渲染。
    { key: "l2Lambda", label: "l2_lambda（L2 正則化）", type: "number", step: 0.001 },
    { key: "batchSize", label: "batch_size", type: "number" },
    { key: "learningRate", label: "learning_rate", type: "number", step: 0.0001 },
    { key: "patience", label: "patience（0=不啟用）", type: "number" },
    {
      key: "classWeight", label: "class_weight", type: "select",
      options: [{ value: "none", label: "none" }, { value: "balanced", label: "balanced" }],
    },
  ],
  xgboost: [
    { key: "nEstimators", label: "n_estimators", type: "number" },
    { key: "maxDepth", label: "max_depth", type: "number" },
    { key: "learningRate", label: "learning_rate", type: "number", step: 0.01 },
    { key: "subsample", label: "subsample", type: "number", step: 0.05 },
    { key: "colsampleBytree", label: "colsample_bytree", type: "number", step: 0.05 },
    { key: "patience", label: "patience（0=不啟用）", type: "number" },
  ],
};

type JobSummary = {
  job_id: string;
  graph_spec: { start: string; end: string; nodes: Record<string, unknown>[] };
  status: "pending" | "running" | "done" | "failed";
  error: string | null;
  result: Record<string, {
    final_metrics?: Record<string, number>;
    /** 舊格式（一次性回傳全部預測值）；新版 infer job 改用下面的 count 摘要，不再整包塞這裡 */
    predictions?: number[];
    timestamps?: number[];
    /** 新版 infer job 的摘要：全期間逐列結果已分批存進 model_inference_predictions，這裡只留筆數與時間範圍 */
    count?: number;
    start_ts?: number | null;
    end_ts?: number | null;
  }> | null;
  device: string | null;
  job_type: "train" | "infer";
  phase: 1 | 2 | 3 | null;
  parent_job_id: string | null;
  created_at: string;
};

type ArtifactSummary = { job_id: string; node_id: string; architecture_key: string; task_type: string; kept: boolean; created_at: string };

let nextModuleId = 1;
let nextFeatureId = 1;

function newModule(displayNumber: number): TrainingModule {
  return {
    id: `mod${nextModuleId++}`,
    name: `模組 ${displayNumber}`,
    symbol: "tx",
    timeframe: "15m",
    start: "2026-06-01",
    end: "2026-09-01",
    // 不預設帶 OHLCV——有些文獻設計會刻意不用原始價量當特徵，新模組一律從空清單開始，
    // 要用什麼特徵（含 OHLCV）都透過「＋加特徵」或「＋整組加入」自己加。
    featureNodes: [],
    outcomeKey: "simple_return",
    labelingRuleKey: "fixed_threshold",
    horizon: 1,
    nClasses: 2,
    thresholdPct: 0,
    architectureKey: "lstm",
    window: 60,
    valRatio: 0.2,
    splitStrategy: "random",
    epochs: 10,
    outputMode: "class",
    units: 64,
    layers: 2,
    dropout: 0.2,
    dense: 32,
    headDropout: 0.2,
    l2Lambda: 0,
    patience: 0,
    bidirectional: false,
    classWeight: "none",
    batchSize: 64,
    learningRate: 0.001,
    nEstimators: 100,
    maxDepth: 6,
    subsample: 1,
    colsampleBytree: 1,
  };
}

// ── React Flow 卡片節點：顯示模組摘要，左邊接點收上游輸入、右邊接點輸出給下游 ──────────
function ModuleCardNode({ data }: NodeProps) {
  const m = data.module as TrainingModule;
  const onEdit = data.onEdit as () => void;
  const onDelete = data.onDelete as () => void;
  return (
    <div className="model-flow-card" onDoubleClick={onEdit}>
      <button
        type="button"
        className="model-flow-card-delete"
        title="刪除這個模組"
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
      >
        ×
      </button>
      <Handle type="target" position={Position.Left} />
      <div className="model-flow-card-title">{m.name}</div>
      <div className="model-flow-card-line">{m.symbol.toUpperCase()} {m.timeframe} {m.architectureKey}</div>
      <div className="model-flow-card-line">
        {m.outcomeKey}　{m.labelingRuleKey === "identity" ? "回歸" : `${m.nClasses}類`}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { moduleCard: ModuleCardNode };

export default function ModelSettings() {
  const navigate = useNavigate();

  const [outcomes, setOutcomes] = useState<RegistryEntry[]>([]);
  const [labelingRules, setLabelingRules] = useState<RegistryEntry[]>([]);
  const [architectures, setArchitectures] = useState<RegistryEntry[]>([]);
  const [savedStrategies, setSavedStrategies] = useState<SavedStrategy[]>([]);
  const [indicatorRegistry, setIndicatorRegistry] = useState<RegistryGroup[]>([]);

  const [modules, setModules] = useState<TrainingModule[]>([]);
  const [editingModule, setEditingModule] = useState<TrainingModule | null>(null);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const [runningJobId, setRunningJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState("");
  const [history, setHistory] = useState<JobSummary[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactSummary[]>([]);
  const [inferTarget, setInferTarget] = useState<{ jobId: string; nodeId?: string; isPhase3?: boolean } | null>(null);
  const [inferStart, setInferStart] = useState("2026-09-01");
  const [inferEnd, setInferEnd] = useState("2026-09-08");
  const [inferError, setInferError] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/api/model/registry/outcomes`).then((r) => r.json()).then(setOutcomes);
    fetch(`${API_BASE}/api/model/registry/labeling_rules`).then((r) => r.json()).then(setLabelingRules);
    fetch(`${API_BASE}/api/model/registry/architectures`).then((r) => r.json()).then(setArchitectures);
    fetch(`${API_BASE}/api/backtest/saved-strategies`).then((r) => r.json()).then(setSavedStrategies).catch(() => {});
    // 特徵庫是「以 TA-Lib 分類為主幹，再加一個 Custom 分類」，不是把 Custom 當成
    // 跟 talib_indicator 平行、另外獨立出來的一種 feature key——Custom 底下目前有三個
    // 平等的選項：OHLCV（價量一起）、Raw Price（只要價）、Raw Volume（只要量），
    // 各自獨立才能做消融實驗（Candle Geometry、Structure 那三個還沒做，不放進來）。
    // Custom 放在陣列最前面、OHLCV 又放在 Custom 的第一個，這樣搜尋框空白時預設列表
    // 第一個看到的就是 OHLCV，不用使用者自己往下找。
    // 名稱刻意不再加 "custom_" 前綴——IndicatorAdder 本來就會秀分類徽章（Custom（本專案
    // 自訂特徵）），名字裡再重複一次「custom」是多餘的視覺雜訊。
    fetch(`${API_BASE}/api/indicators`)
      .then((r) => r.json())
      .then((json: { groups: RegistryGroup[] }) => {
        setIndicatorRegistry([
          {
            group: "custom",
            group_zh: "Custom（本專案自訂特徵）",
            indicators: [
              { key: "ohlcv", display_name: "OHLCV", description: "開高低收＋成交量一起加入，價跟量綁在同一個特徵、不拆開", parameters: {}, output_names: ["open", "high", "low", "close", "volume"], signal_supported: false, signal_type: null },
              { key: "raw_price", display_name: "Raw Price", description: "原始開高低收，不做任何轉換", parameters: {}, output_names: ["open", "high", "low", "close"], signal_supported: false, signal_type: null },
              { key: "raw_volume", display_name: "Raw Volume", description: "原始成交量，不做任何轉換", parameters: {}, output_names: ["volume"], signal_supported: false, signal_type: null },
            ],
          },
          ...json.groups,
        ]);
      })
      .catch(() => {});
  }, []);

  function refreshHistory() {
    fetch(`${API_BASE}/api/model/jobs?limit=20`)
      .then((res) => res.json())
      .then((json: JobSummary[]) => setHistory(json))
      .catch(() => {});
  }
  function refreshArtifacts() {
    fetch(`${API_BASE}/api/model/artifacts`)
      .then((res) => res.json())
      .then((json: ArtifactSummary[]) => setArtifacts(json))
      .catch(() => {});
  }
  useEffect(() => { refreshHistory(); refreshArtifacts(); }, []);

  function handleDeleteJob(jobId: string) {
    // 這裡只是刪 DB 紀錄，不是真的中斷還在跑的訓練——主要拿來清掉容器崩潰後
    // 卡在 running/pending 永遠不會結束的孤兒 job。如果這筆有收藏的模型，
    // 權重會跟著 DB 紀錄一起被刪掉（CASCADE），先跟使用者確認一次再送。
    if (!window.confirm("確定要刪除這筆訓練紀錄？如果有收藏的模型，權重也會一併刪除，且無法復原。")) return;
    fetch(`${API_BASE}/api/model/jobs/${jobId}`, { method: "DELETE" })
      .then((res) => {
        if (!res.ok) return res.json().then((j) => { throw new Error(j.detail || "刪除失敗"); });
        refreshHistory();
        refreshArtifacts();
        if (runningJobId === jobId) setRunningJobId(null);
      })
      .catch((err) => setSubmitError(err.message || "刪除失敗"));
  }

  function handleCancelJob(jobId: string) {
    // 這裡只把 DB 狀態標成失敗，讓歷史紀錄立刻反映「不等了」——真的要把還在跑的
    // process 砍掉，需要另外在 NAS 上手動重啟訓練容器（docker restart hermesnote-training），
    // 這支 API 沒有權限（也刻意不給它權限）觸發那個動作。
    if (!window.confirm("標記這筆為已中斷？\n\n注意：這只會更新畫面上的狀態，不會真的中斷背景還在跑的訓練——如果它真的卡住了，需要你自己在 NAS 上重啟訓練容器（docker restart hermesnote-training）才會真的釋放 GPU/記憶體。")) return;
    fetch(`${API_BASE}/api/model/jobs/${jobId}/cancel`, { method: "POST" })
      .then((res) => {
        if (!res.ok) return res.json().then((j) => { throw new Error(j.detail || "取消失敗"); });
        refreshHistory();
        if (runningJobId === jobId) setRunningJobId(null);
      })
      .catch((err) => setSubmitError(err.message || "取消失敗"));
  }

  function handleToggleKeep(jobId: string, nodeId: string, currentlyKept: boolean) {
    fetch(`${API_BASE}/api/model/artifacts/${jobId}/${nodeId}/keep`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kept: !currentlyKept }),
    })
      .then((res) => { if (!res.ok) throw new Error(); refreshArtifacts(); })
      .catch(() => setSubmitError("儲存/取消儲存模型失敗，請稍後再試"));
  }

  function handleStartInfer() {
    if (!inferTarget) return;
    setInferError("");
    fetch(`${API_BASE}/api/model/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_job_id: inferTarget.jobId,
        ...(inferTarget.nodeId ? { target_node_id: inferTarget.nodeId } : {}),
        start: inferStart, end: inferEnd,
        ...(inferTarget.isPhase3 ? { phase: 3 } : {}),
      }),
    })
      .then(async (res) => {
        if (!res.ok) throw new Error((await res.json().catch(() => ({})))?.detail || "推論任務建立失敗");
        setInferTarget(null);
        refreshHistory();
      })
      .catch((e) => setInferError(e.message || "推論任務建立失敗"));
  }

  useEffect(() => {
    if (!runningJobId) return;
    const timer = setInterval(() => {
      fetch(`${API_BASE}/api/model/jobs/${runningJobId}`)
        .then((res) => res.json())
        .then((job: { status: string; error: string | null }) => {
          if (job.status === "done" || job.status === "failed") {
            setRunningJobId(null);
            if (job.status === "failed") {
              setSubmitError(job.error || "訓練失敗");
            } else {
              // 訓練成功送出去、已經進了歷史紀錄，畫布上的模組卡片留著沒有意義了——
              // 失敗才留著，方便使用者原地修一修再送一次，不用重新拉卡片。
              setModules([]);
              setNodes([]);
              setEdges([]);
            }
            refreshHistory();
          }
        })
        .catch(() => {});
    }, 2000);
    return () => clearInterval(timer);
  }, [runningJobId]);

  // ── 卡片節點跟 modules 陣列同步：module 內容改變時，更新對應卡片顯示 ──────────
  function syncNodeForModule(m: TrainingModule, position?: { x: number; y: number }) {
    setNodes((prev) => {
      const existing = prev.find((n) => n.id === m.id);
      const pos = position ?? existing?.position ?? { x: 40 + prev.length * 60, y: 40 + prev.length * 40 };
      const node: Node = {
        id: m.id, type: "moduleCard", position: pos,
        data: { module: m, onEdit: () => setEditingModule(m), onDelete: () => removeModule(m.id) },
      };
      if (existing) return prev.map((n) => (n.id === m.id ? node : n));
      return [...prev, node];
    });
  }

  function openNewModule() {
    setEditingModule(newModule(modules.length + 1));
  }

  function saveModule(m: TrainingModule) {
    setModules((prev) => {
      const exists = prev.some((x) => x.id === m.id);
      return exists ? prev.map((x) => (x.id === m.id ? m : x)) : [...prev, m];
    });
    syncNodeForModule(m);
    setEditingModule(null);
  }

  function removeModule(id: string) {
    setModules((prev) => prev.filter((m) => m.id !== id));
    setNodes((prev) => prev.filter((n) => n.id !== id));
    setEdges((prev) => prev.filter((e) => e.source !== id && e.target !== id));
  }

  const onConnect = useCallback(
    (conn: Connection) => setEdges((prev) => addEdge({ ...conn, animated: true }, prev)),
    [setEdges]
  );

  // 有模組卻一個特徵都沒加，送到後端 graph.py 會在 np.concatenate([]) 直接炸掉
  // （"need at least one array to concatenate"，這個錯誤訊息完全看不出真正原因）——
  // 這種一目了然的表單缺漏，前端擋掉比丟一串 numpy 錯誤訊息給使用者猜好。
  const emptyFeatureModules = modules.filter((m) => m.featureNodes.length === 0);
  // 「＋加特徵」現在只會新增一列空白的、還沒選指標的節點（key="talib_indicator"、
  // name=""），要靠使用者自己去搜尋挑選——如果挑完就直接存模組/送訓練，
  // 這種沒選到指標名稱的節點送到後端會變成呼叫 ta.Function("")，TA-Lib
  // 直接丟 "not supported by TA-LIB." 這種看不出真正原因的錯誤，要擋在前端。
  const unpickedFeatureModules = modules.filter((m) =>
    m.featureNodes.some((f) => f.key !== "quant_saved_strategy" && !f.name)
  );
  const canSubmit = modules.length > 0 && emptyFeatureModules.length === 0 && unpickedFeatureModules.length === 0 && !runningJobId;

  function buildGraphSpec() {
    const nodesOut: Record<string, unknown>[] = [];
    // 全部模組共用同一段 start/end（第一個模組的區間），送出前不同模組各自的時間框架仍然各自宣告
    const start = modules[0]?.start ?? "2026-06-01";
    const end = modules[0]?.end ?? "2026-09-01";

    for (const m of modules) {
      const tf = `${m.symbol}_${m.timeframe}`;
      const featureIds = m.featureNodes.map((f) => `${m.id}_${f.id}`);
      for (const f of m.featureNodes) {
        nodesOut.push({
          id: `${m.id}_${f.id}`, type: "feature", key: f.key, timeframe: tf,
          params:
            f.key === "talib_indicator" ? { name: f.name, ...f.params } :
            f.key === "quant_saved_strategy" ? { saved_id: f.savedId } :
            undefined,
        });
      }
      nodesOut.push({
        id: `${m.id}_lbl`, type: "label", outcome: m.outcomeKey, labeling_rule: m.labelingRuleKey, timeframe: tf,
        params: { horizon: m.horizon, n_classes: m.nClasses, threshold_pct: m.thresholdPct },
      });
      // inputs = 自己的 feature nodes + 畫布上「別的模組 → 這個模組」的邊（上游模組的輸出當額外特徵）
      const upstreamModuleIds = edges.filter((e) => e.target === m.id).map((e) => e.source);
      const archParams: Record<string, number | boolean | string> =
        m.architectureKey === "xgboost"
          ? {
              n_estimators: m.nEstimators, max_depth: m.maxDepth, learning_rate: m.learningRate,
              subsample: m.subsample, colsample_bytree: m.colsampleBytree, patience: m.patience,
            }
          : {
              units: m.units, layers: m.layers, dropout: m.dropout, dense: m.dense,
              head_dropout: m.headDropout, bidirectional: m.bidirectional, l2_lambda: m.l2Lambda,
              batch_size: m.batchSize, learning_rate: m.learningRate, patience: m.patience,
              class_weight: m.classWeight,
            };
      nodesOut.push({
        id: m.id, type: "model", key: m.architectureKey,
        inputs: [...featureIds, ...upstreamModuleIds],
        label: `${m.id}_lbl`, window: m.window, val_ratio: m.valRatio,
        // 這個表單建立的一律是 Phase 1（random），chronological 只能透過歷史紀錄的
        // 「進到 Phase 2」自動帶入，不讓使用者在這裡自己選——上一輪抓到的設計矛盾修正。
        split_strategy: "random",
        output_mode: m.outputMode,
        params: { epochs: m.epochs, n_classes: m.nClasses, ...archParams },
      });
    }
    return { start, end, nodes: nodesOut };
  }

  function handleSubmit() {
    if (!canSubmit) return;
    setSubmitError("");
    const graphSpec = buildGraphSpec();

    fetch(`${API_BASE}/api/model/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // 從這個表單建立的訓練，一律標成教授三階段協定的 Phase 1（random 切分）——
      // 之後要進 Phase 2/3，從歷史紀錄的「進到下一階段」按鈕觸發，不是從這裡重新填表單。
      body: JSON.stringify({ ...graphSpec, phase: 1 }),
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

  function handleAdvancePhase2(parentJobId: string) {
    setSubmitError("");
    fetch(`${API_BASE}/api/model/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start: "", end: "", nodes: [], phase: 2, parent_job_id: parentJobId }),
    })
      .then(async (res) => {
        if (!res.ok) throw new Error((await res.json().catch(() => ({})))?.detail || "建立 Phase 2 任務失敗");
        refreshHistory();
      })
      .catch((e) => setSubmitError(e.message || "建立 Phase 2 任務失敗"));
  }

  return (
    <div className="model-settings">
      <div className="model-settings-block">
        <h2 className="model-settings-block-title">建立訓練</h2>
        <div className="model-settings-card">
          <div className="model-settings-mock-note">
            每個「訓練模組」是一個完整的時間框架+特徵+Label+模型設定。只有一個模組時直接跑；
            兩個以上，拉線把某個模組的輸出接到另一個模組，代表「接到的那個模組會把上游的輸出當額外特徵」——
            線怎麼接，最終訓練的組合方式就怎麼跑，畫布只是產生設定的介面，不是另一套邏輯。
          </div>

          <div className="model-flow-toolbar">
            <button type="button" className="model-settings-add-btn" onClick={openNewModule}>
              ＋加訓練模組
            </button>
            <span className="model-flow-count">{modules.length} 個模組</span>
          </div>

          <div className="model-flow-canvas">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onEdgeClick={(_e, edge) => setEdges((eds) => eds.filter((e) => e.id !== edge.id))}
              nodeTypes={nodeTypes}
              fitView
            >
              <Background />
              <Controls />
            </ReactFlow>
          </div>

          {editingModule && (
            <ModuleEditor
              module={editingModule}
              outcomes={outcomes}
              labelingRules={labelingRules}
              architectures={architectures}
              savedStrategies={savedStrategies}
              indicatorRegistry={indicatorRegistry}
              onCancel={() => setEditingModule(null)}
              onSave={saveModule}
              onDelete={modules.some((m) => m.id === editingModule.id) ? () => { removeModule(editingModule.id); setEditingModule(null); } : undefined}
            />
          )}

          <div className="model-settings-submit-row">
            <button type="button" className="model-settings-submit" disabled={!canSubmit} onClick={handleSubmit}>
              {runningJobId ? "訓練中…" : "建立訓練"}
            </button>
            {emptyFeatureModules.length > 0 && (
              <div className="model-settings-submit-msg is-error">
                {emptyFeatureModules.map((m) => m.name).join("、")} 還沒有加特徵，點卡片進去用「＋加特徵」或「＋整組加入」補上再送
              </div>
            )}
            {unpickedFeatureModules.length > 0 && (
              <div className="model-settings-submit-msg is-error">
                {unpickedFeatureModules.map((m) => m.name).join("、")} 有特徵列還沒搜尋挑選指標，點卡片進去補選再送
              </div>
            )}
            {submitError && <div className="model-settings-submit-msg is-error">{submitError}</div>}
          </div>
        </div>
      </div>

      <div className="model-settings-block">
        <h2 className="model-settings-block-title">歷史紀錄</h2>
        <div className="model-settings-card">
          {history.length === 0 && <div className="model-settings-empty">尚無訓練紀錄。</div>}
          {history.map((job) => {
            const isInfer = job.job_type === "infer";
            const clickable = job.status !== "pending" && !isInfer;
            return (
              <div
                key={job.job_id}
                className={"model-settings-history-row" + (!clickable ? " is-disabled" : "")}
                onClick={() => clickable && navigate(`/model?job=${job.job_id}`)}
              >
                <div className="model-settings-history-meta">
                  <div className="model-settings-history-time">
                    {job.phase && <span className="model-settings-history-phase">Phase {job.phase}</span>}
                    {isInfer && "［推論］"}
                    {new Date(job.created_at).toLocaleString("zh-TW")}
                    {job.status !== "done" && ` ・ ${job.status === "failed" ? "失敗" : job.status === "running" ? "執行中" : "等待中"}`}
                  </div>
                  <div className="model-settings-history-config">
                    {job.graph_spec.start} ~ {job.graph_spec.end}　device: {job.device ?? "—"}
                  </div>
                  {job.error && <div className="model-settings-history-error">{job.error}</div>}
                </div>

                {(job.status === "pending" || job.status === "running") && (
                  <button
                    type="button"
                    className="model-settings-history-delete"
                    onClick={(e) => { e.stopPropagation(); handleCancelJob(job.job_id); }}
                  >
                    停止
                  </button>
                )}
                <button
                  type="button"
                  className="model-settings-history-delete"
                  onClick={(e) => { e.stopPropagation(); handleDeleteJob(job.job_id); }}
                >
                  刪除
                </button>

                {isInfer && job.status === "done" && Object.entries(job.result || {}).map(([nodeId, r]) => (
                  <div key={nodeId} className="model-settings-history-stats">
                    <div className="model-settings-history-stat"><span>推論筆數</span><strong>{r.count ?? r.predictions?.length ?? 0}</strong></div>
                  </div>
                ))}

                {!isInfer && job.status === "done" && Object.entries(job.result || {}).map(([nodeId, r]) => {
                  const artifact = artifacts.find((a) => a.job_id === job.job_id && a.node_id === nodeId);
                  return (
                    <div key={nodeId} className="model-settings-history-stats">
                      {r.final_metrics && Object.entries(r.final_metrics).map(([k, v]) => (
                        <div key={k} className="model-settings-history-stat"><span>{k}</span><strong>{typeof v === "number" ? v.toFixed(4) : "—"}</strong></div>
                      ))}
                      {artifact && (
                        <>
                          <button
                            type="button"
                            className={"model-settings-history-save" + (artifact.kept ? " is-saved" : "")}
                            onClick={(e) => { e.stopPropagation(); handleToggleKeep(job.job_id, nodeId, artifact.kept); }}
                          >
                            {artifact.kept ? "★ 已收藏" : "☆ 收藏模型"}
                          </button>
                          <button
                            type="button"
                            className="model-settings-add-btn"
                            onClick={(e) => { e.stopPropagation(); setInferTarget({ jobId: job.job_id, nodeId }); }}
                          >
                            用這個模型推論
                          </button>
                        </>
                      )}
                      {job.phase === 1 && (
                        <button type="button" className="model-settings-add-btn" onClick={(e) => { e.stopPropagation(); handleAdvancePhase2(job.job_id); }}>
                          進到 Phase 2
                        </button>
                      )}
                      {job.phase === 2 && artifact && (
                        <button
                          type="button"
                          className="model-settings-add-btn"
                          onClick={(e) => { e.stopPropagation(); setInferTarget({ jobId: job.job_id, isPhase3: true }); }}
                        >
                          進到 Phase 3（holdout）
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>

        {inferTarget && (
          <div className="model-editor-overlay" onClick={() => setInferTarget(null)}>
            <div className="model-editor-panel" onClick={(e) => e.stopPropagation()} style={{ width: 420 }}>
              <div className="model-settings-block-title">{inferTarget.isPhase3 ? "進到 Phase 3（holdout 推論）" : "用已存模型做推論"}</div>
              <div className="model-settings-mock-note">
                {inferTarget.isPhase3
                  ? `載入 Phase 2 job（${inferTarget.jobId.slice(0, 8)}…）已保存的權重，對下面指定的 holdout 期間做預測評估，不重新訓練。`
                  : `載入節點 ${inferTarget.nodeId}（job ${inferTarget.jobId.slice(0, 8)}…）已保存的權重，對下面指定的新資料區間直接算預測，不重新訓練。`}
              </div>
              <div className="model-settings-config-row">
                <label className="model-settings-field">
                  <span>開始日期</span>
                  <input type="date" value={inferStart} onChange={(e) => setInferStart(e.target.value)} />
                </label>
                <label className="model-settings-field">
                  <span>結束日期</span>
                  <input type="date" value={inferEnd} onChange={(e) => setInferEnd(e.target.value)} />
                </label>
              </div>
              {inferError && <div className="model-settings-submit-msg is-error">{inferError}</div>}
              <div className="model-settings-submit-row">
                <button type="button" className="model-settings-add-btn" onClick={() => setInferTarget(null)}>取消</button>
                <button type="button" className="model-settings-submit" onClick={handleStartInfer}>送出推論任務</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── 訓練模組的詳細設定表單（就是原本那份表單，現在包成單一模組用） ──────────
function ModuleEditor({
  module, outcomes, labelingRules, architectures, savedStrategies, indicatorRegistry, onCancel, onSave, onDelete,
}: {
  module: TrainingModule;
  outcomes: RegistryEntry[];
  labelingRules: RegistryEntry[];
  architectures: RegistryEntry[];
  savedStrategies: SavedStrategy[];
  indicatorRegistry: RegistryGroup[];
  onCancel: () => void;
  onSave: (m: TrainingModule) => void;
  onDelete?: () => void;
}) {
  const [m, setM] = useState<TrainingModule>(module);
  const [groupPickerOpen, setGroupPickerOpen] = useState(false);
  const [dateRange, setDateRange] = useState<{ min: string; max: string } | null>(null);

  // 開始/結束日期跟量化回測一樣，對照資料庫實際涵蓋範圍：選了商品/時間框架就直接
  // 帶成資料庫最早～最新那一筆，不是只在超出範圍才夾回邊界。
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/backtest/date-range?symbol=${m.symbol}&timeframe=${m.timeframe}`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((json: { min_date: string; max_date: string }) => {
        if (cancelled) return;
        setDateRange({ min: json.min_date, max: json.max_date });
        setM((prev) => ({ ...prev, start: json.min_date, end: json.max_date }));
      })
      .catch(() => { if (!cancelled) setDateRange(null); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [m.symbol, m.timeframe]);

  function addFeatureNode() {
    // 新增一個空的「指標」類型特徵，還沒指定是哪個指標——使用者用同一個跨全部分類
    // （TA-Lib＋Custom）的搜尋挑選，不預設塞一個特定指標名稱進去。
    setM((prev) => ({ ...prev, featureNodes: [...prev.featureNodes, { id: `f${nextFeatureId++}`, key: "talib_indicator", name: "", params: {}, savedId: "" }] }));
  }
  function removeFeatureNode(id: string) {
    // 空清單現在是合法起始狀態（不強制帶 OHLCV），這裡不用再擋「至少留一個」。
    setM((prev) => ({ ...prev, featureNodes: prev.featureNodes.filter((f) => f.id !== id) }));
  }
  function updateFeatureNode(id: string, patch: Partial<FeatureNodeConfig>) {
    setM((prev) => ({ ...prev, featureNodes: prev.featureNodes.map((f) => (f.id === id ? { ...f, ...patch } : f)) }));
  }
  // 整組加入：一次把某個 TA-Lib 分類（例如「動量指標」）底下全部指標加成特徵，
  // 各自用該指標「自己真正的」預設參數（ind.parameters 是 registry 給的完整 schema，
  // 不是每個都只有 timeperiod——MACD/STOCH 這類多參數指標也能正確帶出全部欄位），
  // 全部標上同一個 group，之後想做消融實驗（整組拿掉重跑 Phase 1）可以一鍵整組移除。
  function addFeatureGroup(group: RegistryGroup) {
    const newNodes: FeatureNodeConfig[] = group.indicators.map((ind) => {
      // Custom 分類底下這幾個本身就是後端真正的 feature key，跟單一加特徵那邊的
      // 邏輯一致，不能整組都寫死成 talib_indicator。
      const isCustom = CUSTOM_FEATURE_KEYS.includes(ind.key);
      return {
        id: `f${nextFeatureId++}`,
        key: isCustom ? ind.key : "talib_indicator",
        name: ind.key,
        params: { ...ind.parameters },
        savedId: "",
        group: group.group_zh,
      };
    });
    setM((prev) => ({ ...prev, featureNodes: [...prev.featureNodes, ...newNodes] }));
  }
  function removeFeatureGroup(group: string) {
    setM((prev) => ({ ...prev, featureNodes: prev.featureNodes.filter((f) => f.group !== group) }));
  }

  return (
    <div className="model-editor-overlay">
      <div className="model-editor-panel">
        <div className="model-settings-config-row">
          <label className="model-settings-field">
            <span>模組名稱</span>
            <input value={m.name} onChange={(e) => setM({ ...m, name: e.target.value })} />
          </label>
          <label className="model-settings-field">
            <span>商品</span>
            <select value={m.symbol} onChange={(e) => setM({ ...m, symbol: e.target.value })}>
              {SYMBOLS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </label>
          <label className="model-settings-field">
            <span>時間框架</span>
            <select value={m.timeframe} onChange={(e) => setM({ ...m, timeframe: e.target.value })}>
              {TIMEFRAMES.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
            </select>
          </label>
          <label className="model-settings-field">
            <span>開始日期</span>
            <input type="date" value={m.start} min={dateRange?.min} max={dateRange?.max} onChange={(e) => setM({ ...m, start: e.target.value })} />
          </label>
          <label className="model-settings-field">
            <span>結束日期</span>
            <input type="date" value={m.end} min={dateRange?.min} max={dateRange?.max} onChange={(e) => setM({ ...m, end: e.target.value })} />
          </label>
          {dateRange && (
            <div className="model-settings-indicators-note" style={{ alignSelf: "flex-end", marginBottom: 7 }}>
              資料庫涵蓋範圍：{dateRange.min} ~ {dateRange.max}
            </div>
          )}
        </div>

        <div className="model-settings-indicators">
          <div className="model-settings-indicators-head">
            <span>特徵（Feature Node，可加多個）</span>
            <div style={{ display: "flex", gap: 8, position: "relative" }}>
              <button type="button" className="model-settings-add-btn" onClick={addFeatureNode}>＋加特徵</button>
              <button type="button" className="model-settings-add-btn" onClick={() => setGroupPickerOpen((v) => !v)}>＋整組加入</button>
              {groupPickerOpen && (
                <div className="model-settings-group-picker">
                  {indicatorRegistry.length === 0 && <div className="model-settings-group-picker-empty">指標清單載入中…</div>}
                  {indicatorRegistry.map((g) => (
                    <button
                      type="button"
                      key={g.group}
                      className="model-settings-group-picker-option"
                      onClick={() => { addFeatureGroup(g); setGroupPickerOpen(false); }}
                    >
                      {g.group_zh}（{g.indicators.length} 個）
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {(() => {
            const groupNames = Array.from(new Set(m.featureNodes.map((f) => f.group).filter((g): g is string => !!g)));
            const renderFeatureRow = (f: FeatureNodeConfig) => (
              <div key={f.id} className="model-settings-indicator-row">
                <div style={{ flex: 1 }}>
                  <div className="model-settings-indicator-params">
                    {/* 特徵庫是「以 TA-Lib 分類為主幹＋一個 Custom 分類」的單一目錄，不是
                        talib_indicator／ohlcv／raw_price／raw_volume 這些互相獨立、平起平坐
                        的 feature key 選項——Custom 只是 IndicatorAdder 搜尋得到的眾多分類
                        裡多一個而已，跟「動量指標」「價格轉換」這些 TA-Lib 分類同一層。
                        這裡只留一個「指標／收藏的策略」的類型切換，實際挑哪個指標一律走
                        同一個跨全部分類搜尋的 IndicatorAdder，不再讓使用者先選一個原始的
                        後端 key 名稱。 */}
                    <label>
                      <span>類型</span>
                      <select
                        value={f.key === "quant_saved_strategy" ? "quant_saved_strategy" : "indicator"}
                        onChange={(e) =>
                          updateFeatureNode(f.id, e.target.value === "quant_saved_strategy"
                            ? { key: "quant_saved_strategy", name: "", params: {} }
                            : { key: "talib_indicator", name: "", params: {} })
                        }
                        style={{ width: 140 }}
                      >
                        <option value="indicator">指標</option>
                        <option value="quant_saved_strategy">收藏的策略</option>
                      </select>
                    </label>
                    {f.key !== "quant_saved_strategy" && (
                      <>
                        <label>
                          <span>指標名稱</span>
                          <div className="model-settings-indicator-pick">
                            <span className="model-settings-indicator-picked">{f.name || "尚未選擇"}</span>
                            <IndicatorAdder
                              registry={indicatorRegistry}
                              label="搜尋指標（TA-Lib／Custom）"
                              // 同一個模組裡，別的特徵列已經選過的指標不能再選第二次
                              // （自己這一列目前選的除外，不然編輯到一半會把自己濾掉）。
                              excludeKeys={m.featureNodes.filter((other) => other.id !== f.id).map((other) => other.name).filter(Boolean)}
                              onAdd={(key) => {
                                const meta = findIndicator(indicatorRegistry, key);
                                // Custom 底下這幾個本身就是後端真正的 feature key（不用再包
                                // 一層 talib_indicator+name）；其餘一律是 TA-Lib 指標，key
                                // 統一是 talib_indicator，name 才是實際指標名稱。
                                // 換指標要整組換掉參數，不是保留舊的——舊指標的參數名稱
                                // （例如 RSI 的 timeperiod）對新指標（例如 MACD）通常
                                // 根本不存在，留著只會送出無意義甚至會報錯的欄位。
                                const isCustom = CUSTOM_FEATURE_KEYS.includes(key);
                                updateFeatureNode(f.id, {
                                  key: isCustom ? key : "talib_indicator",
                                  name: key,
                                  params: { ...(meta?.parameters ?? {}) },
                                });
                              }}
                            />
                          </div>
                        </label>
                        {f.name && Object.keys(f.params).length === 0 && (
                          <span className="model-settings-indicator-rule">（這個指標沒有參數）</span>
                        )}
                        {Object.entries(f.params).map(([key, value]) => (
                          <label key={key}>
                            <span>{key}</span>
                            <input
                              type="number"
                              value={value}
                              onChange={(e) => updateFeatureNode(f.id, { params: { ...f.params, [key]: Number(e.target.value) } })}
                            />
                          </label>
                        ))}
                      </>
                    )}
                    {f.key === "quant_saved_strategy" && (
                      <label>
                        <span>收藏的策略（在 /admin/quant 收藏）</span>
                        <select value={f.savedId} onChange={(e) => updateFeatureNode(f.id, { savedId: e.target.value })} style={{ width: 260 }}>
                          <option value="">請選擇</option>
                          {savedStrategies.map((s) => (
                            <option key={s.saved_id} value={s.saved_id}>
                              {s.name || `${s.symbol} ${s.timeframe}`}（{new Date(s.created_at).toLocaleDateString("zh-TW")}）
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                  </div>
                </div>
                <button type="button" className="model-settings-remove-btn" onClick={() => removeFeatureNode(f.id)}>×</button>
              </div>
            );
            return (
              <>
                {m.featureNodes.filter((f) => !f.group).map(renderFeatureRow)}
                {groupNames.map((group) => (
                  <div key={group} className="model-settings-feature-group">
                    <div className="model-settings-feature-group-head">
                      <span>{group}（{m.featureNodes.filter((f) => f.group === group).length} 個）</span>
                      <button type="button" className="model-settings-group-remove-btn" onClick={() => removeFeatureGroup(group)}>移除整組</button>
                    </div>
                    {m.featureNodes.filter((f) => f.group === group).map(renderFeatureRow)}
                  </div>
                ))}
              </>
            );
          })()}
        </div>

        <div className="model-settings-indicators">
          <div className="model-settings-indicators-head"><span>Target / Label Node</span></div>
          <div className="model-settings-indicator-row">
            <div className="model-settings-indicator-params">
              <label>
                <span>outcome（原始數值）</span>
                <select
                  value={m.outcomeKey}
                  style={{ width: 140 }}
                  onChange={(e) => {
                    const outcomeKey = e.target.value;
                    const illegal = ILLEGAL_LABELING_RULES[outcomeKey] ?? [];
                    const labelingRuleKey = illegal.includes(m.labelingRuleKey)
                      ? (labelingRules.find((r) => !illegal.includes(r.key))?.key ?? m.labelingRuleKey)
                      : m.labelingRuleKey;
                    setM({ ...m, outcomeKey, labelingRuleKey });
                  }}
                >
                  {outcomes.map((o) => <option key={o.key} value={o.key}>{o.key}</option>)}
                </select>
              </label>
              <label><span>horizon</span><input type="number" value={m.horizon} min={1} onChange={(e) => setM({ ...m, horizon: Number(e.target.value) })} /></label>
              <label>
                <span>labeling_rule（怎麼變學習目標）</span>
                <select value={m.labelingRuleKey} style={{ width: 150 }} onChange={(e) => setM({ ...m, labelingRuleKey: e.target.value })}>
                  {labelingRules
                    .filter((r) => !(ILLEGAL_LABELING_RULES[m.outcomeKey] ?? []).includes(r.key))
                    .map((r) => <option key={r.key} value={r.key}>{r.key === "identity" ? "identity（回歸）" : r.key}</option>)}
                </select>
              </label>
              {m.labelingRuleKey === "fixed_threshold" && (
                <>
                  <label>
                    <span>n_classes</span>
                    <select value={m.nClasses} onChange={(e) => setM({ ...m, nClasses: Number(e.target.value) })}>
                      <option value={2}>2（漲/不漲）</option>
                      <option value={3}>3（跌/平/漲）</option>
                    </select>
                  </label>
                  <label><span>threshold_pct</span><input type="number" step="0.01" value={m.thresholdPct} onChange={(e) => setM({ ...m, thresholdPct: Number(e.target.value) })} /></label>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="model-settings-indicators">
          <div className="model-settings-indicators-head"><span>Model Node</span></div>
          <div className="model-settings-indicator-row">
            <div className="model-settings-indicator-params">
              <label>
                <span>architecture</span>
                <select value={m.architectureKey} onChange={(e) => setM({ ...m, architectureKey: e.target.value })} style={{ width: 120 }}>
                  {architectures.map((a) => <option key={a.key} value={a.key}>{a.key}</option>)}
                </select>
              </label>
              <label><span>window</span><input type="number" value={m.window} min={5} onChange={(e) => setM({ ...m, window: Number(e.target.value) })} /></label>
              <label><span>val_ratio</span><input type="number" step="0.05" value={m.valRatio} onChange={(e) => setM({ ...m, valRatio: Number(e.target.value) })} /></label>
              {/* split_strategy 不給使用者選：這個表單建立的一律是 Phase 1（random），
                  chronological 只能透過歷史紀錄的「進到 Phase 2」按鈕自動帶入，見 buildGraphSpec()。 */}
              <label><span>epochs</span><input type="number" value={m.epochs} min={1} onChange={(e) => setM({ ...m, epochs: Number(e.target.value) })} /></label>
              {m.labelingRuleKey === "fixed_threshold" && (
                <label>
                  <span>輸出</span>
                  <select value={m.outputMode} onChange={(e) => setM({ ...m, outputMode: e.target.value as "class" | "probability" })}>
                    <option value="class">類別</option>
                    <option value="probability">機率（softmax）</option>
                  </select>
                </label>
              )}
              {m.architectureKey === "lstm" && (
                <label>
                  <span>bidirectional</span>
                  <input
                    type="checkbox"
                    checked={m.bidirectional}
                    onChange={(e) => setM({ ...m, bidirectional: e.target.checked })}
                  />
                </label>
              )}
              {/* 架構專屬超參數：跟上面共用參數放同一個 flex-wrap 容器裡，自然換行，
                  不另外分區塊／不加子標題——上一輪把它拆成獨立的 grid 區塊，
                  結果在 flex 容器裡被擠成一欄窄窄的直排，比原本還醜，這輪改回最單純的做法。 */}
              {(ARCHITECTURE_PARAM_FIELDS[m.architectureKey] ?? []).map((field) => {
                if (field.type === "checkbox") {
                  return (
                    <label key={String(field.key)}>
                      <span>{field.label}</span>
                      <input
                        type="checkbox"
                        checked={m[field.key] as boolean}
                        onChange={(e) => setM({ ...m, [field.key]: e.target.checked })}
                      />
                    </label>
                  );
                }
                if (field.type === "select") {
                  return (
                    <label key={String(field.key)}>
                      <span>{field.label}</span>
                      <select value={m[field.key] as string} onChange={(e) => setM({ ...m, [field.key]: e.target.value })}>
                        {field.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                      </select>
                    </label>
                  );
                }
                return (
                  <label key={String(field.key)}>
                    <span>{field.label}</span>
                    <input
                      type="number"
                      step={field.step}
                      value={m[field.key] as number}
                      onChange={(e) => setM({ ...m, [field.key]: Number(e.target.value) })}
                    />
                  </label>
                );
              })}
            </div>
          </div>
        </div>

        <div className="model-settings-submit-row">
          {onDelete && <button type="button" className="model-settings-history-delete" onClick={onDelete}>刪除這個模組</button>}
          <button type="button" className="model-settings-add-btn" onClick={onCancel}>取消</button>
          <button type="button" className="model-settings-submit" onClick={() => onSave(m)}>儲存模組</button>
        </div>
      </div>
    </div>
  );
}
