// 後台「模型實驗」（M7）：實驗層設定（期間、holdout、Label、切分、對齊規則：沿用前一時段／最多往前幾個時段／
// 資料過舊上限／發布延遲、暖機 ε、排除警示、階段門檻、串接預設）、組別、樣本框架版本、三階段執行（門檻未過時需填理由確認）、
// 任務的排除原因與提示、各分支實際採用資料的摘要、評估（val／holdout、best／last）、配對比較、holdout 使用紀錄、串接核對。
// API：docs/agent-api/training/api.md 的「實驗層」一節。
import { useCallback, useEffect, useMemo, useState } from "react";
import { modelWriteFetch, useAuth } from "../../../auth/AuthContext";
import "./ModelExperiments.css";

const API = `${import.meta.env.VITE_API_BASE as string}/api/model`;
const TFS = ["tx_1m", "tx_5m", "tx_15m", "tx_30m", "tx_60m", "tx_daily_day", "tx_daily_full"];

type Json = Record<string, any>;
type Exp = Json & { id: string; name: string; settings: Json; arms?: Json[]; frames?: Json[]; jobs?: Json[] };

async function call(path: string, init?: RequestInit): Promise<any> {
  // 寫入帶後台登入 token（後端 /api/model 寫入端點要求管理員或服務金鑰）；讀取照舊
  const write = init?.method && init.method !== "GET";
  const r = write ? await modelWriteFetch(`${API}${path}`, init)
    : await fetch(`${API}${path}`, { headers: { "Content-Type": "application/json" }, ...init });
  const text = await r.text();
  const body = text ? JSON.parse(text) : null;
  if (!r.ok) {
    const d = body?.detail;
    const err: any = new Error(typeof d === "string" ? d : d?.message ?? `HTTP ${r.status}`);
    err.detail = d;
    err.status = r.status;
    throw err;
  }
  return body;
}

const DEFAULT_SETTINGS = {
  start: "2015-01-01", end: "2024-01-01",
  holdout: { start: "2024-01-01", end: "2026-01-01" },
  label: { timeframe: "tx_60m", outcome: "log_return", labeling_rule: "fixed_threshold", params: { horizon: 1, n_classes: 2, threshold_pct: 0 } },
  split_seed: 42, val_ratio: 0.2,
  alignment: { allow_previous_session: true, max_sessions_back: 1, max_age_sec: null, delays: {} },
  warmup: { epsilon: 0.01 }, exclusion_warn_ratio: 0.05,
  gates: [], cascade: { strategy: "forward_chaining", k: 5, final: "refit_full" },
};

const DEFAULT_ARM = [
  { id: "f", type: "feature", key: "ohlcv", timeframe: "tx_60m" },
  { id: "r", type: "feature", key: "talib_indicator", timeframe: "tx_60m", params: { name: "RSI", timeperiod: 14 } },
  { id: "m", type: "model", key: "xgboost", inputs: ["f", "r"], label: "label", window: 20,
    params: { n_estimators: 200, max_depth: 4, learning_rate: 0.05, early_stopping: { patience: 20 }, seed: 42 } },
];

function JsonBox({ value, onChange, rows = 10 }: { value: string; onChange: (v: string) => void; rows?: number }) {
  let ok = true;
  try { JSON.parse(value); } catch { ok = false; }
  return <textarea className={`mx-json${ok ? "" : " is-bad"}`} rows={rows} value={value} spellCheck={false}
    onChange={(e) => onChange(e.target.value)} />;
}

function Err({ e }: { e: any }) {
  if (!e) return null;
  return <div className="mx-error">{String(e.message ?? e)}</div>;
}

function Warnings({ list }: { list?: Json[] }) {
  if (!list || list.length === 0) return null;
  return <ul className="mx-warnings">{list.map((w, i) => <li key={i}><code>{w.code}</code> {w.message}</li>)}</ul>;
}

function fmt(v: any): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}

// ── 對齊規則與門檻編輯（建立與修改共用）────────────────────────────────
function AlignmentEditor({ value, onChange }: { value: Json; onChange: (v: Json) => void }) {
  const a = value ?? {};
  const set = (k: string, v: any) => onChange({ ...a, [k]: v });
  const delays: Json = a.delays ?? {};
  return (
    <div className="mx-grid">
      <label className="mx-check">
        <input type="checkbox" checked={a.allow_previous_session !== false}
          onChange={(e) => set("allow_previous_session", e.target.checked)} />
        沿用前一交易時段的資料（allow_previous_session）
      </label>
      <label>最多往前幾個時段（0～5，空白＝不限）
        <input type="number" min={0} max={5} value={a.max_sessions_back ?? ""}
          onChange={(e) => set("max_sessions_back", e.target.value === "" ? null : Number(e.target.value))} />
      </label>
      <label>資料過舊上限（秒，空白＝不限）
        <input type="number" min={1} value={a.max_age_sec ?? ""}
          onChange={(e) => set("max_age_sec", e.target.value === "" ? null : Number(e.target.value))} />
      </label>
      <div className="mx-delays">
        <div className="mx-sub">發布延遲（秒；未設定＝0 秒、未實測假設）</div>
        {TFS.map((tf) => {
          const d = delays[tf];
          return (
            <div key={tf} className="mx-delay-row">
              <code>{tf}</code>
              <input type="number" min={0} value={d?.sec ?? ""} placeholder="0"
                onChange={(e) => {
                  const nd = { ...delays };
                  if (e.target.value === "") delete nd[tf];
                  else nd[tf] = { sec: Number(e.target.value), status: d?.status ?? "assumed", note: d?.note ?? "" };
                  set("delays", nd);
                }} />
              <select value={d?.status ?? "assumed"} disabled={!d}
                onChange={(e) => set("delays", { ...delays, [tf]: { ...d, status: e.target.value } })}>
                <option value="assumed">assumed（未實測）</option>
                <option value="measured">measured（已實測）</option>
              </select>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function GatesEditor({ value, onChange }: { value: Json[]; onChange: (v: Json[]) => void }) {
  const gates = value ?? [];
  const upd = (i: number, k: string, v: any) => onChange(gates.map((g, j) => (j === i ? { ...g, [k]: v } : g)));
  return (
    <div>
      {gates.length === 0 && <div className="mx-muted">未設定門檻（進入下一階段時只顯示「未設定」）</div>}
      {gates.map((g, i) => (
        <div key={i} className="mx-gate-row">
          進入 Phase
          <select value={g.phase} onChange={(e) => upd(i, "phase", Number(e.target.value))}>
            <option value={2}>2（檢查 Phase 1）</option><option value={3}>3（檢查 Phase 2）</option>
          </select>
          <input placeholder="指標鍵，例如 macro_f1" value={g.metric} onChange={(e) => upd(i, "metric", e.target.value)} />
          <select value={g.op} onChange={(e) => upd(i, "op", e.target.value)}>
            {[">", ">=", "<", "<="].map((o) => <option key={o}>{o}</option>)}
          </select>
          <input type="number" step="any" value={g.value} onChange={(e) => upd(i, "value", Number(e.target.value))} />
          <select value={g.weights ?? "best"} onChange={(e) => upd(i, "weights", e.target.value)}>
            <option value="best">best</option><option value="last">last</option>
          </select>
          <button className="mx-btn-ghost" onClick={() => onChange(gates.filter((_, j) => j !== i))}>移除</button>
        </div>
      ))}
      <button className="mx-btn-ghost" onClick={() => onChange([...gates, { phase: 2, metric: "macro_f1", op: ">=", value: 0.5, weights: "best" }])}>
        ＋ 新增門檻
      </button>
    </div>
  );
}

// ── 建立實驗 ─────────────────────────────────────────────────────────
function CreateExperiment({ onCreated }: { onCreated: (id: string) => void }) {
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [text, setText] = useState(JSON.stringify({ ...DEFAULT_SETTINGS, alignment: undefined, gates: undefined }, null, 2));
  const [alignment, setAlignment] = useState<Json>(DEFAULT_SETTINGS.alignment);
  const [gates, setGates] = useState<Json[]>([]);
  const [err, setErr] = useState<any>(null);
  const [notice, setNotice] = useState<Json[] | null>(null);
  const submit = async () => {
    setErr(null);
    try {
      const settings = { ...JSON.parse(text), alignment, gates };
      const r = await call("/experiments", { method: "POST", body: JSON.stringify({ name, description: desc, settings }) });
      setNotice(r.warnings);
      onCreated(r.experiment_id);
    } catch (e) { setErr(e); }
  };
  return (
    <section className="mx-card">
      <h3>建立實驗</h3>
      <div className="mx-muted">實驗層＝各組別必須一致的設定：期間、holdout、Label（答案定義）、切分種子、val_ratio、對齊規則、暖機 ε、排除警示門檻、串接預設。</div>
      <div className="mx-grid">
        <label>名稱<input value={name} onChange={(e) => setName(e.target.value)} /></label>
        <label>說明<input value={desc} onChange={(e) => setDesc(e.target.value)} /></label>
      </div>
      <div className="mx-sub">共用設定（JSON）</div>
      <JsonBox value={text} onChange={setText} rows={16} />
      <div className="mx-sub">時間對齊規則</div>
      <AlignmentEditor value={alignment} onChange={setAlignment} />
      <div className="mx-sub">階段門檻（只提示並記錄；未通過時需填理由確認才能繼續）</div>
      <GatesEditor value={gates} onChange={setGates} />
      <button className="mx-btn" disabled={!name} onClick={submit}>建立</button>
      <Err e={err} />
      <Warnings list={notice ?? undefined} />
    </section>
  );
}

// ── 任務明細 ─────────────────────────────────────────────────────────
function JobDetail({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<Json | null>(null);
  const [evals, setEvals] = useState<Json[]>([]);
  const [audit, setAudit] = useState<Json | null>(null);
  const [records, setRecords] = useState<Json[] | null>(null);
  const [err, setErr] = useState<any>(null);
  useEffect(() => {
    setJob(null); setAudit(null); setRecords(null);
    call(`/jobs/${jobId}`).then(setJob).catch(setErr);
    call(`/jobs/${jobId}/evaluations`).then(setEvals).catch(() => setEvals([]));
  }, [jobId]);
  if (err) return <Err e={err} />;
  if (!job) return <div className="mx-muted">載入中…</div>;
  const nodes = job.result ?? {};
  const evalRows: Json[] = [];
  for (const [nid, r] of Object.entries<any>(nodes)) for (const e of r?.evaluations ?? []) evalRows.push({ nid, ...e });
  for (const row of evals) for (const e of row.evaluations ?? []) if (!evalRows.find((x) => x.id === e.id)) evalRows.push({ nid: row.node_id, ...e });
  const metricKeys = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "rmse", "mae", "r2"];
  return (
    <div className="mx-detail">
      <div className="mx-kv">
        <span>狀態</span><b>{job.status}</b>
        <span>Phase</span><b>{fmt(job.phase)}</b>
        <span>流程版本</span><b>{fmt(job.pipeline_version)}</b>
        <span>模型種子</span><b>{fmt(job.model_seed)}</b>
        <span>門檻</span><b>{job.gate_check ? `${job.gate_check.status}${job.gate_check.override ? `（確認人 ${job.gate_check.override.confirmed_by}${job.gate_check.override.note_confirmed_by === undefined ? "［舊紀錄，未驗證文字］" : ""}：${job.gate_check.override.reason}）` : ""}` : "—"}</b>
      </div>
      {job.error && <div className="mx-error">{job.error}</div>}
      <Warnings list={job.warnings} />
      {job.gate_check?.conditions?.length > 0 && (
        <table className="mx-table"><thead><tr><th>門檻</th><th>實際</th><th>結果</th></tr></thead><tbody>
          {job.gate_check.conditions.map((c: Json, i: number) => (
            <tr key={i}><td>{c.metric} {c.op} {c.value}（{c.weights}）</td><td>{fmt(c.actual)}</td><td>{c.ok ? "通過" : "未通過"}</td></tr>))}
        </tbody></table>
      )}
      {Object.entries<any>(nodes).map(([nid, r]) => {
        const tm = r?.training_meta ?? {};
        const hd = r?.holdout;
        return (
          <div key={nid} className="mx-node">
            <div className="mx-sub">節點 {nid}{tm.pipeline === "v2" ? "（v2）" : ""}</div>
            {tm.frame && <div className="mx-muted">樣本框架 <code>{String(tm.frame.hash ?? "").slice(0, 12)}</code>：{fmt(tm.frame.counts)}</div>}
            {tm.exclusions && (
              <div className="mx-muted">排除原因：train {fmt(tm.exclusions.train)}；val {fmt(tm.exclusions.val)}；框架 {fmt(tm.exclusions.frame)}
                （排除比例 {(Number(tm.exclusion_ratio ?? 0) * 100).toFixed(1)}%）</div>
            )}
            {tm.branches && (
              <table className="mx-table"><thead><tr><th>分支</th><th>時間框架</th><th>window</th><th>編碼器</th>
                <th>沿用前一時段筆數</th><th>最多往前時段</th><th>距決策時刻（中位／最大，秒）</th></tr></thead><tbody>
                {Object.entries<any>(tm.branches).map(([bid, b]) => (
                  <tr key={bid}><td>{bid}</td><td>{b.timeframe}</td><td>{b.window}</td><td>{b.encoder}</td>
                    <td>{b.n_from_previous_session}</td><td>{fmt(b.max_sessions_back)}</td>
                    <td>{fmt(b.median_age_sec)}／{fmt(b.max_age_sec)}</td></tr>))}
              </tbody></table>
            )}
            {tm.warmup && (
              <div className="mx-muted">暖機：{Object.entries<any>(tm.warmup).map(([f, w]) =>
                `${f}=${w.total_bars} 根（${w.memory_type}${w.researcher_bars != null ? "，研究者設定" : ""}${w.transform?.type && w.transform.type !== "none" ? `，${w.transform.type}` : ""}）`).join("；")}</div>
            )}
            {tm.cascade && <div className="mx-muted">串接：{fmt(tm.cascade.policy)}，清除 {fmt(tm.cascade.purge_sec)} 秒，有樣本外預測 {tm.cascade.n_rows_with_oof} 列</div>}
            {hd && <div className="mx-muted">Holdout：納入 {hd.n_included}、未成熟 {hd.n_unmatured}、輸入不足未預測 {hd.n_skipped} {fmt(hd.skipped_reasons)}；成熟截止 {fmt(hd.cutoff_ts)}</div>}
          </div>
        );
      })}
      {evalRows.length > 0 && (
        <table className="mx-table"><thead><tr><th>評估</th>{metricKeys.map((k) => <th key={k}>{k}</th>)}</tr></thead><tbody>
          {evalRows.map((e) => (
            <tr key={e.id}><td>{e.nid}／{e.head}／{e.dataset?.split}／{e.point?.kind}</td>
              {metricKeys.map((k) => <td key={k}>{fmt(e.metrics?.[k])}</td>)}</tr>))}
        </tbody></table>
      )}
      <div className="mx-actions">
        <button className="mx-btn-ghost" onClick={() => call(`/jobs/${jobId}/cascade_audit`).then(setAudit).catch(setErr)}>串接核對</button>
        <a className="mx-btn-ghost" href={`${API}/jobs/${jobId}/cascade_audit?format=csv`} target="_blank" rel="noreferrer">違反清單 CSV</a>
        <button className="mx-btn-ghost" onClick={() => call(`/jobs/${jobId}/records?limit=50`).then(setRecords).catch(setErr)}>逐筆結果（前 50 筆）</button>
      </div>
      {audit && (
        <div className={`mx-audit ${audit.passed ? "is-ok" : "is-bad"}`}>
          {audit.passed ? "全部規則違反 0 筆" : "有違反"}：{Object.entries<any>(audit.rules).map(([k, v]) => `${k} ${v.violations}/${v.checked}`).join("，")}
          （紀錄 {audit.n_records}、折 {audit.n_folds}）
        </div>
      )}
      {records && (
        <table className="mx-table mx-small"><thead><tr><th>資料集</th><th>權重</th><th>時間</th><th>答案</th><th>預測</th><th>納入</th><th>原因</th></tr></thead><tbody>
          {records.map((r) => (<tr key={r.id}><td>{r.dataset}</td><td>{fmt(r.weights)}</td><td>{r.sample_ts}</td><td>{fmt(r.y_true)}</td>
            <td>{fmt(r.y_pred)}</td><td>{r.included ? "是" : "否"}</td><td>{fmt(r.exclusion_reason)}</td></tr>))}
        </tbody></table>
      )}
    </div>
  );
}

// ── 實驗明細 ─────────────────────────────────────────────────────────
function ExperimentView({ id, onChanged }: { id: string; onChanged: () => void }) {
  const { email } = useAuth();
  const [exp, setExp] = useState<Exp | null>(null);
  const [err, setErr] = useState<any>(null);
  const [armName, setArmName] = useState("");
  const [armNodes, setArmNodes] = useState(JSON.stringify(DEFAULT_ARM, null, 2));
  const [seeds, setSeeds] = useState("42");
  const [selJob, setSelJob] = useState<string | null>(null);
  const [cmp, setCmp] = useState<{ a: string; b: string; weights: string }>({ a: "", b: "", weights: "best" });
  const [cmpResult, setCmpResult] = useState<Json | null>(null);
  const [usage, setUsage] = useState<Json[]>([]);
  const [align, setAlign] = useState<Json | null>(null);
  const [gates, setGates] = useState<Json[] | null>(null);
  const [warnRatio, setWarnRatio] = useState<number | null>(null);
  const [gatePrompt, setGatePrompt] = useState<{ arm: string; phase: number; detail: Json } | null>(null);
  const [reason, setReason] = useState("");
  const [suggestIn, setSuggestIn] = useState(JSON.stringify({ key: "talib_indicator", timeframe: "tx_60m", params: { name: "MACD" } }));
  const [suggest, setSuggest] = useState<Json | null>(null);

  const load = useCallback(() => {
    call(`/experiments/${id}`).then((e: Exp) => {
      setExp(e); setAlign(e.settings.alignment ?? DEFAULT_SETTINGS.alignment); setGates(e.settings.gates ?? []);
      setWarnRatio(e.settings.exclusion_warn_ratio ?? 0.05);
    }).catch(setErr);
    call(`/experiments/${id}/holdout_usage`).then(setUsage).catch(() => setUsage([]));
  }, [id]);
  useEffect(() => { load(); }, [load]);

  const run = async (arm: string, phase: number, override?: Json) => {
    setErr(null);
    try {
      await call(`/experiments/${id}/arms/${arm}/run`, { method: "POST", body: JSON.stringify({ phase, gate_override: override }) });
      setGatePrompt(null); setReason(""); load(); onChanged();
    } catch (e: any) {
      if (e.status === 409 && e.detail?.gate_check) setGatePrompt({ arm, phase, detail: e.detail });
      else setErr(e);
    }
  };
  if (!exp) return <Err e={err} />;
  const s = exp.settings;
  const jobsByArm = (arm: string) => (exp.jobs ?? []).filter((j) => j.arm_id === arm);
  return (
    <div>
      <section className="mx-card">
        <h3>{exp.name} <span className="mx-muted">（{exp.pipeline_version}）</span></h3>
        <div className="mx-kv">
          <span>期間</span><b>{s.start}～{s.end}</b>
          <span>Holdout</span><b>{s.holdout ? `${s.holdout.start}～${s.holdout.end}` : "未設定"}</b>
          <span>Label</span><b>{s.label?.timeframe}／{s.label?.outcome}／{s.label?.labeling_rule} {fmt(s.label?.params)}</b>
          <span>切分</span><b>種子 {s.split_seed ?? 42}，val_ratio {s.val_ratio ?? 0.2}</b>
        </div>
        <details>
          <summary>修改對齊規則、門檻、排除警示門檻（只影響之後建立的任務；比較時設定不同會被拒絕）</summary>
          {align && <AlignmentEditor value={align} onChange={setAlign} />}
          {gates && <GatesEditor value={gates} onChange={setGates} />}
          <label>排除比例警示門檻（0～1）
            <input type="number" step="0.01" min={0} max={1} value={warnRatio ?? 0.05}
              onChange={(e) => setWarnRatio(Number(e.target.value))} /></label>
          <button className="mx-btn" onClick={() => call(`/experiments/${id}/settings`, {
            method: "PATCH", body: JSON.stringify({ settings: { alignment: align, gates, exclusion_warn_ratio: warnRatio } }),
          }).then(load).catch(setErr)}>儲存設定</button>
          {(s.history ?? []).length > 0 && <div className="mx-muted">修改紀錄：{s.history.map((h: Json) => `${h.at.slice(0, 19)} ${h.changed.join("、")}`).join("；")}</div>}
        </details>
        <Err e={err} />
      </section>

      {gatePrompt && (
        <section className="mx-card mx-gate">
          <h3>門檻未通過：Phase {gatePrompt.phase}</h3>
          <div>{gatePrompt.detail.message}</div>
          <table className="mx-table"><tbody>{gatePrompt.detail.gate_check.conditions.map((c: Json, i: number) => (
            <tr key={i}><td>{c.metric} {c.op} {c.value}（{c.weights}）</td><td>實際 {fmt(c.actual)}</td><td>{c.ok ? "通過" : "未通過"}</td></tr>))}</tbody></table>
          <label>放行理由（必填，會連同確認人與時間記錄在任務上）<input value={reason} onChange={(e) => setReason(e.target.value)} /></label>
          <div className="mx-muted">確認人由後端依目前登入的帳號記錄（admin:{email ?? "—"}），不採用前端填寫的文字。</div>
          <button className="mx-btn" disabled={!reason} onClick={() => run(gatePrompt.arm, gatePrompt.phase, { reason })}>確認放行</button>
          <button className="mx-btn-ghost" onClick={() => setGatePrompt(null)}>取消</button>
        </section>
      )}

      <section className="mx-card">
        <h3>樣本框架</h3>
        <table className="mx-table"><thead><tr><th>Phase</th><th>版本</th><th>狀態</th><th>雜湊</th><th>時間點</th><th>計數</th></tr></thead><tbody>
          {(exp.frames ?? []).map((f) => (<tr key={f.id}><td>{f.phase}</td><td>{f.version}</td><td>{f.status}</td>
            <td><code>{String(f.hash ?? "").slice(0, 12)}</code></td><td>{fmt(f.n_points)}</td><td>{fmt(f.counts)}</td></tr>))}
        </tbody></table>
        <div className="mx-actions">{[1, 2, 3].map((p) => (
          <button key={p} className="mx-btn-ghost" onClick={() => call(`/experiments/${id}/frames`, { method: "POST", body: JSON.stringify({ phase: p }) }).then(load).catch(setErr)}>
            建立 Phase {p} 新版本框架</button>))}</div>
        <div className="mx-muted">框架在第一個任務執行時由 worker 依資料產生；之後的任務核對雜湊，不同就失敗並提示建立新版本。不同版本的結果不比較。</div>
      </section>

      <section className="mx-card">
        <h3>組別</h3>
        {(exp.arms ?? []).map((a) => (
          <div key={a.id} className="mx-arm">
            <div className="mx-arm-head"><b>{a.name}</b> <span className="mx-muted">種子 {JSON.stringify(a.model_seeds)}</span>
              {[1, 2, 3].map((p) => <button key={p} className="mx-btn-ghost" onClick={() => run(a.id, p)}>執行 Phase {p}</button>)}
            </div>
            <table className="mx-table"><thead><tr><th>任務</th><th>Phase</th><th>種子</th><th>狀態</th><th>門檻</th><th>提示</th></tr></thead><tbody>
              {jobsByArm(a.id).map((j) => (
                <tr key={j.id} className={selJob === j.id ? "is-sel" : ""} onClick={() => setSelJob(j.id)}>
                  <td><code>{j.id.slice(0, 8)}</code></td><td>{j.phase}</td><td>{fmt(j.model_seed)}</td><td>{j.status}</td>
                  <td>{j.gate_check?.status ?? "—"}</td><td>{(j.warnings ?? []).map((w: Json) => w.code).join("、") || "—"}</td></tr>))}
            </tbody></table>
          </div>
        ))}
        <div className="mx-sub">新增組別（Feature／Model 節點；最終模型的 label 用 "label"，由實驗注入）</div>
        <div className="mx-grid">
          <label>名稱<input value={armName} onChange={(e) => setArmName(e.target.value)} /></label>
          <label>模型種子（逗號分隔）<input value={seeds} onChange={(e) => setSeeds(e.target.value)} /></label>
        </div>
        <JsonBox value={armNodes} onChange={setArmNodes} rows={14} />
        <button className="mx-btn" disabled={!armName} onClick={() => call(`/experiments/${id}/arms`, {
          method: "POST", body: JSON.stringify({ name: armName, nodes: JSON.parse(armNodes), model_seeds: seeds.split(",").map((x) => Number(x.trim())) }),
        }).then(() => { setArmName(""); load(); }).catch(setErr)}>新增組別</button>
        <details>
          <summary>暖機建議（程式基本值＋實測建議；複合型指標必須由研究者設定 warmup.bars）</summary>
          <JsonBox value={suggestIn} onChange={setSuggestIn} rows={3} />
          <button className="mx-btn-ghost" onClick={() => call("/warmup/suggest", {
            method: "POST", body: JSON.stringify({ feature: JSON.parse(suggestIn), epsilon: s.warmup?.epsilon ?? 0.01 }),
          }).then(setSuggest).catch(setErr)}>計算建議</button>
          {suggest && <div className="mx-muted">程式：{suggest.program.memory_type}，基本 {suggest.program.base_bars} 根（{suggest.program.detail}）；
            實測建議 {suggest.empirical.bars} 根（容許差 {fmt(suggest.empirical.tolerance)}）</div>}
        </details>
      </section>

      {selJob && <section className="mx-card"><h3>任務 <code>{selJob.slice(0, 8)}</code></h3><JobDetail jobId={selJob} /></section>}

      <section className="mx-card">
        <h3>配對比較（同一框架版本、同一 Phase、共同時間點）</h3>
        <div className="mx-grid">
          {(["a", "b"] as const).map((k) => (
            <label key={k}>任務 {k.toUpperCase()}
              <select value={cmp[k]} onChange={(e) => setCmp({ ...cmp, [k]: e.target.value })}>
                <option value="">—</option>
                {(exp.jobs ?? []).filter((j) => j.status === "done").map((j) => (
                  <option key={j.id} value={j.id}>{(exp.arms ?? []).find((a) => a.id === j.arm_id)?.name}／P{j.phase}／seed {j.model_seed}／{j.id.slice(0, 8)}</option>))}
              </select></label>))}
          <label>權重<select value={cmp.weights} onChange={(e) => setCmp({ ...cmp, weights: e.target.value })}>
            <option>best</option><option>last</option></select></label>
        </div>
        <button className="mx-btn" disabled={!cmp.a || !cmp.b} onClick={() => call(
          `/experiments/${id}/compare?job_a=${cmp.a}&job_b=${cmp.b}&weights=${cmp.weights}`).then(setCmpResult).catch(setErr)}>比較</button>
        {cmpResult && <pre className="mx-pre">{JSON.stringify(cmpResult, null, 2)}</pre>}
      </section>

      <section className="mx-card">
        <h3>Holdout 使用紀錄（{usage.length}）</h3>
        <table className="mx-table mx-small"><thead><tr><th>時間</th><th>類型</th><th>任務</th><th>期間</th><th>內容</th></tr></thead><tbody>
          {usage.map((u) => (<tr key={u.id}><td>{u.created_at.slice(0, 19)}</td><td>{u.kind}</td><td><code>{String(u.job_id ?? "").slice(0, 8)}</code></td>
            <td>{fmt(u.period_start)}～{fmt(u.period_end)}</td><td>{fmt(u.detail)}</td></tr>))}
        </tbody></table>
      </section>
    </div>
  );
}

export default function ModelExperiments() {
  const [list, setList] = useState<Exp[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [registry, setRegistry] = useState<Json | null>(null);
  const [err, setErr] = useState<any>(null);
  const refresh = useCallback(() => call("/experiments").then(setList).catch(setErr), []);
  useEffect(() => { refresh(); call("/registry/v2").then(setRegistry).catch(() => {}); }, [refresh]);
  const reasons = useMemo(() => Object.entries<string>(registry?.exclusion_reasons ?? {}), [registry]);
  return (
    <div className="mx-root">
      <header className="mx-header">
        <h2>模型實驗</h2>
        <button className="mx-btn" onClick={() => setCreating((v) => !v)}>{creating ? "收起" : "＋ 建立實驗"}</button>
      </header>
      <Err e={err} />
      {creating && <CreateExperiment onCreated={(id) => { setCreating(false); refresh(); setSel(id); }} />}
      <div className="mx-layout">
        <aside className="mx-list">
          {list.map((e) => (
            <button key={e.id} className={`mx-list-item${sel === e.id ? " is-sel" : ""}`} onClick={() => setSel(e.id)}>
              <b>{e.name}</b><span className="mx-muted">{e.settings?.label?.timeframe} · 組別 {e.n_arms} · 任務 {e.n_jobs}</span>
            </button>))}
          {list.length === 0 && <div className="mx-muted">尚無實驗</div>}
          {reasons.length > 0 && (
            <details className="mx-reasons"><summary>排除原因代碼</summary>
              <ul>{reasons.map(([k, v]) => <li key={k}><code>{k}</code> {v}</li>)}</ul></details>)}
        </aside>
        <main className="mx-main">{sel ? <ExperimentView id={sel} onChanged={refresh} /> : <div className="mx-muted">選擇左側的實驗</div>}</main>
      </div>
    </div>
  );
}
