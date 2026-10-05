import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import {
  adminFetch,
  usePolling,
  type AdminApiError,
  type Envelope,
  type KnowledgeBase,
  type PollState,
  type RagChunk,
  type RagContent,
  type RagHit,
  type RagJob,
  type RagSource,
  type RagSummary,
} from "../dashboard/adminApi";
import {
  Badge,
  ErrorNote,
  PanelHead,
  Pager,
  SOURCE_TYPE_LABEL,
  formatTime,
  locatorLabel,
  shortId,
} from "../dashboard/parts";
import "../dashboard/dashboard.css";

type Tab = "overview" | "documents" | "search" | "jobs";
const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "總覽" },
  { id: "documents", label: "文件" },
  { id: "search", label: "搜尋" },
  { id: "jobs", label: "索引工作" },
];
const PAGE = 20;

function citation(title: string, sourceId: string, version: string, chunkIndex: number, loc: RagChunk["locator"]) {
  return `${title}（${locatorLabel(loc)}）· source ${sourceId} · version ${version.slice(0, 12)} · chunk #${chunkIndex}`;
}

// ---------------------------------------------------------------- 文件詳細（片段、版本、定位、品質警告）

function SourceDetail({ sourceId, version, focusChunk, onClose }: {
  sourceId: string;
  version?: string;
  focusChunk?: number;
  onClose: () => void;
}) {
  const [doc, setDoc] = useState<RagContent | null>(null);
  const [chunks, setChunks] = useState<RagChunk[]>([]);
  const [error, setError] = useState<AdminApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [pageFilter, setPageFilter] = useState<number | null>(null);
  const focusRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setChunks([]);
    setDoc(null);
    (async () => {
      // 依 next_cursor 逐頁讀完整份文件的索引片段（每頁最多 50）
      const all: RagChunk[] = [];
      let offset = 0;
      try {
        for (let i = 0; i < 40; i++) {
          const q = new URLSearchParams({ offset: String(offset), limit: "50" });
          if (version) q.set("version", version);
          const res = await adminFetch<Envelope<RagContent>>(`/rag/sources/${sourceId}/content?${q}`);
          if (cancelled) return;
          if (i === 0) setDoc(res.data);
          all.push(...res.data.chunks);
          setChunks([...all]);
          if (!res.next_cursor) break;
          offset = Number(res.next_cursor);
        }
      } catch (e) {
        if (!cancelled) setError(e as AdminApiError);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sourceId, version]);

  useEffect(() => {
    if (focusChunk != null && focusRef.current) focusRef.current.scrollIntoView({ block: "center" });
  }, [focusChunk, chunks.length]);

  const warnings = doc?.metadata?.warnings ?? [];
  const pages = useMemo(
    () => [...new Set(chunks.map((c) => c.locator.page).filter((p): p is number => p != null))].sort((a, b) => a - b),
    [chunks],
  );
  const shown = pageFilter != null ? chunks.filter((c) => c.locator.page === pageFilter) : chunks;

  return (
    <section className="dash-panel dash-detail">
      <div className="dash-panel-head">
        <h2 className="dash-panel-title">{doc?.title ?? "文件內容"}</h2>
        <div className="dash-panel-meta">
          <button className="dash-btn is-ghost" onClick={onClose}>關閉</button>
        </div>
      </div>
      <ErrorNote error={error} />
      {doc && (
        <dl className="dash-kv">
          <dt>類型</dt><dd>{SOURCE_TYPE_LABEL[doc.source_type] ?? doc.source_type}</dd>
          <dt>狀態</dt><dd><Badge status={doc.status} /> {doc.is_current ? "目前版本" : <Badge label="非目前版本" tone="warn" />}</dd>
          <dt>source_id</dt><dd className="dash-mono">{doc.source_id}</dd>
          <dt>version</dt><dd className="dash-mono" title={doc.version}>{doc.version}</dd>
          <dt>片段</dt><dd>{chunks.length}{loading ? "（讀取中…）" : ""}</dd>
          <dt>索引模型</dt><dd className="dash-mono">{String(doc.metadata?.model_id ?? "—")}</dd>
        </dl>
      )}
      {warnings.length > 0 && (
        <div className="dash-warnings">
          <strong>品質警告（{warnings.length}）</strong>
          <ul>
            {warnings.map((w, idx) => (
              <li key={idx}>
                {w.locator?.page != null ? (
                  <button className="dash-btn is-link" onClick={() => setPageFilter(w.locator!.page!)}>
                    {locatorLabel(w.locator)}
                  </button>
                ) : (
                  <span>{locatorLabel(w.locator)}</span>
                )}
                ：{w.reason}
              </li>
            ))}
          </ul>
        </div>
      )}
      {doc?.source_type === "pdf" && (
        <div className="dash-faint dash-note">
          這裡顯示的是索引時抽取的文字；PDF 原檔預覽需先補受控來源 API，目前未提供。
        </div>
      )}
      {pages.length > 0 && (
        <div className="dash-filters">
          <label>
            頁碼
            <select value={pageFilter ?? ""} onChange={(e) => setPageFilter(e.target.value ? Number(e.target.value) : null)}>
              <option value="">全部</option>
              {pages.map((p) => <option key={p} value={p}>第 {p} 頁</option>)}
            </select>
          </label>
          {pageFilter != null && <span className="dash-faint">顯示 {shown.length} 個片段</span>}
        </div>
      )}
      <div className="dash-chunks">
        {shown.map((c) => {
          const focused = c.chunk_index === focusChunk;
          return (
            <div key={c.chunk_index} ref={focused ? focusRef : undefined} className={`dash-chunk${focused ? " is-focus" : ""}${c.quality_warnings.length ? " is-warn" : ""}`}>
              <div className="dash-chunk-head">
                <span className="dash-mono">#{c.chunk_index}</span>
                <span>{locatorLabel(c.locator)}</span>
                {c.quality_warnings.length > 0 && <Badge label="品質警告" tone="warn" />}
                {focused && <Badge label="搜尋命中" tone="info" />}
              </div>
              {c.quality_warnings.map((w, idx) => <div key={idx} className="dash-list-error">{w.reason}</div>)}
              <pre className="dash-chunk-text">{c.content}</pre>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- 總覽

function OverviewTab({ summary, kb }: { summary: PollState<Envelope<RagSummary>>; kb: KnowledgeBase | undefined }) {
  const data = summary.data?.data;
  const total = data?.documents.reduce((s, d) => s + d.count, 0);
  const byType = useMemo(() => {
    const map = new Map<string, { status: string; count: number }[]>();
    for (const d of data?.documents ?? []) map.set(d.source_type, [...(map.get(d.source_type) ?? []), d]);
    return [...map.entries()];
  }, [data]);
  return (
    <section className="dash-panel">
      <PanelHead title="知識庫總覽" lastSuccessAt={summary.lastSuccessAt} loading={summary.loading} onRefresh={summary.refresh} />
      <ErrorNote error={summary.error} hasStale={!!data} />
      {data && (
        <>
          <div className="dash-summary is-compact">
            <div className="dash-stat"><div className="dash-stat-value">{total}</div><div className="dash-stat-label">來源（目前版本）</div></div>
            <div className="dash-stat"><div className="dash-stat-value">{data.chunks}</div><div className="dash-stat-label">片段 chunks</div></div>
            <div className="dash-stat"><div className="dash-stat-value">{data.health.sources_scanned}</div><div className="dash-stat-label">本輪掃描</div></div>
            <div className="dash-stat"><div className="dash-stat-value">{data.health.failures}</div><div className="dash-stat-label">本輪失敗</div></div>
          </div>
          <table className="dash-table">
            <thead><tr><th>類型</th><th>狀態</th><th>數量</th></tr></thead>
            <tbody>
              {byType.flatMap(([type, rows]) => rows.map((r) => (
                <tr key={type + r.status}>
                  <td>{SOURCE_TYPE_LABEL[type] ?? type}</td>
                  <td><Badge status={r.status} /></td>
                  <td>{r.count}</td>
                </tr>
              )))}
            </tbody>
          </table>
          <dl className="dash-kv">
            <dt>索引狀態</dt><dd><Badge status={data.health.state} /></dd>
            <dt>最後掃描</dt><dd>{formatTime(data.health.last_scan_at)}</dd>
            {data.health.scan_error && (<><dt>掃描錯誤</dt><dd className="dash-bad">{data.health.scan_error}</dd></>)}
            <dt>知識庫</dt><dd>{kb?.label ?? data.knowledge_base}</dd>
          </dl>
          <div className="dash-faint dash-note">
            目前掛載的來源類型：PDF 文獻、研究卡、文獻索引。實驗紀錄等其他類型尚未進庫。服務目前沒有重新索引／刪除 API。
          </div>
        </>
      )}
    </section>
  );
}

// ---------------------------------------------------------------- 文件清單

function DocumentsTab({ onOpen }: { onOpen: (s: RagSource) => void }) {
  const [sourceType, setSourceType] = useState("");
  const [status, setStatus] = useState("");
  const [offset, setOffset] = useState(0);
  const q = new URLSearchParams({ offset: String(offset), limit: String(PAGE) });
  if (sourceType) q.set("source_type", sourceType);
  if (status) q.set("status", status);
  const list = usePolling<Envelope<RagSource[]>>(`/rag/sources?${q}`, 0);
  return (
    <section className="dash-panel">
      <PanelHead title="文件清單" lastSuccessAt={list.lastSuccessAt} loading={list.loading} onRefresh={list.refresh} />
      <div className="dash-filters">
        <label>
          類型
          <select value={sourceType} onChange={(e) => { setSourceType(e.target.value); setOffset(0); }}>
            <option value="">全部</option>
            {Object.entries(SOURCE_TYPE_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </label>
        <label>
          狀態
          <select value={status} onChange={(e) => { setStatus(e.target.value); setOffset(0); }}>
            <option value="">全部</option>
            <option value="indexed">已索引</option>
            <option value="needs_review">待核對</option>
          </select>
        </label>
      </div>
      <ErrorNote error={list.error} hasStale={!!list.data} />
      <table className="dash-table is-clickable">
        <thead><tr><th>標題</th><th>類型</th><th>狀態</th><th>警告</th><th>版本</th><th>更新</th></tr></thead>
        <tbody>
          {list.data?.data.map((s) => (
            <tr key={s.source_id} onClick={() => onOpen(s)}>
              <td>{s.title}</td>
              <td className="dash-faint">{SOURCE_TYPE_LABEL[s.source_type] ?? s.source_type}</td>
              <td><Badge status={s.status} /></td>
              <td>{s.metadata?.warnings?.length ? <Badge label={`${s.metadata.warnings.length}`} tone="warn" /> : <span className="dash-faint">—</span>}</td>
              <td className="dash-mono dash-faint" title={s.version}>{shortId(s.version, 10)}</td>
              <td className="dash-faint">{formatTime(s.updated_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {list.data && list.data.data.length === 0 && <div className="dash-empty">沒有符合條件的文件</div>}
      <Pager offset={offset} limit={PAGE} nextCursor={list.data?.next_cursor ?? null} onChange={setOffset} />
    </section>
  );
}

// ---------------------------------------------------------------- 搜尋

function SearchTab({ onOpenHit }: { onOpenHit: (h: RagHit) => void }) {
  const [query, setQuery] = useState("");
  const [sourceType, setSourceType] = useState("");
  const [limit, setLimit] = useState(10);
  const [hits, setHits] = useState<RagHit[] | null>(null);
  const [error, setError] = useState<AdminApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [meta, setMeta] = useState<{ at: string; ms?: number; q: string } | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  async function run(e?: FormEvent) {
    e?.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await adminFetch<Envelope<RagHit[]>>("/rag/search", {
        method: "POST",
        body: { query: query.trim(), limit, source_type: sourceType || null },
      });
      setHits(res.data);
      setMeta({ at: res.observed_at, ms: res.latency_ms, q: query.trim() });
    } catch (err) {
      setError(err as AdminApiError);
    } finally {
      setLoading(false);
    }
  }

  function copy(text: string, key: string) {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(key);
      window.setTimeout(() => setCopied(null), 1500);
    });
  }

  return (
    <section className="dash-panel">
      <PanelHead title="搜尋" />
      <form className="dash-filters" onSubmit={run}>
        <input className="dash-input is-wide" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="輸入中文或英文查詢，例如：注意力機制 LSTM" maxLength={1000} />
        <label>
          類型
          <select value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
            <option value="">全部</option>
            {Object.entries(SOURCE_TYPE_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </label>
        <label>
          筆數
          <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
            {[5, 10, 20, 30].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <button className="dash-btn" type="submit" disabled={loading || !query.trim()}>{loading ? "搜尋中…" : "搜尋"}</button>
      </form>
      <ErrorNote error={error} hasStale={!!hits} />
      {meta && (
        <div className="dash-faint dash-note">
          「{meta.q}」共 {hits?.length ?? 0} 筆 · {formatTime(meta.at)}{meta.ms != null ? ` · ${meta.ms} ms` : ""} · 混合排序（向量＋關鍵字）
        </div>
      )}
      <div className="dash-chunks">
        {hits?.map((h, idx) => {
          const key = `${h.source_id}:${h.chunk_index}`;
          const cite = citation(h.title, h.source_id, h.version, h.chunk_index, h.locator);
          return (
            <div key={key} className={`dash-chunk${h.quality_warnings.length ? " is-warn" : ""}`}>
              <div className="dash-chunk-head">
                <span className="dash-mono">{idx + 1}.</span>
                <strong>{h.title}</strong>
                <span className="dash-faint">{SOURCE_TYPE_LABEL[h.source_type] ?? h.source_type}</span>
                <span>{locatorLabel(h.locator)}</span>
                <span className="dash-mono dash-faint">#{h.chunk_index} · score {h.score.toFixed(4)}</span>
                {h.quality_warnings.length > 0 && <Badge label="品質警告" tone="warn" />}
              </div>
              {h.quality_warnings.map((w, i) => <div key={i} className="dash-list-error">{w.reason}</div>)}
              <pre className="dash-chunk-text">{h.content}</pre>
              <div className="dash-chunk-actions">
                <button className="dash-btn is-ghost" onClick={() => onOpenHit(h)}>回查原片段</button>
                <button className="dash-btn is-ghost" onClick={() => copy(cite, key)}>{copied === key ? "已複製" : "複製引用"}</button>
                <span className="dash-faint dash-mono dash-cite">{cite}</span>
              </div>
            </div>
          );
        })}
        {hits && hits.length === 0 && <div className="dash-empty">沒有結果</div>}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- 索引工作

function JobsTab({ titles }: { titles: Map<string, string> }) {
  const [offset, setOffset] = useState(0);
  const jobs = usePolling<Envelope<RagJob[]>>(`/rag/ingestion-jobs?offset=${offset}&limit=${PAGE}`, 60_000);
  return (
    <section className="dash-panel">
      <PanelHead title="索引工作" lastSuccessAt={jobs.lastSuccessAt} loading={jobs.loading} onRefresh={jobs.refresh} />
      <div className="dash-faint dash-note">
        歷史工作全部保留；過去失敗的筆數不等於目前未解決的文件數，目前狀態以「文件」頁為準。新來源首次處理失敗只會出現在這裡。
      </div>
      <ErrorNote error={jobs.error} hasStale={!!jobs.data} />
      <table className="dash-table">
        <thead><tr><th>開始</th><th>來源</th><th>階段</th><th>狀態</th><th>片段</th><th>耗時</th><th>錯誤</th></tr></thead>
        <tbody>
          {jobs.data?.data.map((j) => {
            const ms = j.started_at && j.finished_at ? new Date(j.finished_at).getTime() - new Date(j.started_at).getTime() : null;
            return (
              <tr key={j.job_id}>
                <td className="dash-faint">{formatTime(j.started_at)}</td>
                <td title={j.source_id}>{titles.get(j.source_id) ?? <span className="dash-mono">{shortId(j.source_id)}</span>}</td>
                <td className="dash-faint">{j.stage}</td>
                <td><Badge status={j.status} /></td>
                <td>{j.chunk_count ?? "—"}</td>
                <td className="dash-faint">{ms != null ? `${(ms / 1000).toFixed(1)} 秒` : "—"}</td>
                <td className="dash-bad">{j.error ?? ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <Pager offset={offset} limit={PAGE} nextCursor={jobs.data?.next_cursor ?? null} onChange={setOffset} />
    </section>
  );
}

// ---------------------------------------------------------------- 頁面

export default function RagKnowledge() {
  const [params, setParams] = useSearchParams();
  const tab = (params.get("tab") as Tab) || "overview";
  const openSource = params.get("source");
  const openVersion = params.get("version") ?? undefined;
  const focusChunk = params.get("chunk") != null ? Number(params.get("chunk")) : undefined;

  const kbs = usePolling<Envelope<KnowledgeBase[]>>("/rag/knowledge-bases", 60_000);
  const summary = usePolling<Envelope<RagSummary>>("/rag/summary", 60_000);
  const [kbId, setKbId] = useState("research");
  const kb = kbs.data?.data.find((k) => k.id === kbId);

  // 索引工作只回 source_id；載入一次目前版本清單換成標題
  const [titles, setTitles] = useState<Map<string, string>>(new Map());
  const loadTitles = useCallback(() => {
    adminFetch<Envelope<RagSource[]>>("/rag/sources?offset=0&limit=100")
      .then((res) => setTitles(new Map(res.data.map((s) => [s.source_id, `${s.title}（${SOURCE_TYPE_LABEL[s.source_type] ?? s.source_type}）`]))))
      .catch(() => { /* 只影響顯示標題，失敗時改顯示 id */ });
  }, []);
  useEffect(loadTitles, [loadTitles]);

  function set(next: Record<string, string | null>) {
    const p = new URLSearchParams(params);
    for (const [k, v] of Object.entries(next)) {
      if (v == null) p.delete(k);
      else p.set(k, v);
    }
    setParams(p, { replace: false });
  }

  return (
    <div className="dash">
      <div className="dash-panel-head">
        <h2 className="dash-panel-title is-page">知識庫</h2>
        <div className="dash-panel-meta">
          <label className="dash-filters is-inline">
            知識庫
            <select value={kbId} onChange={(e) => setKbId(e.target.value)}>
              {(kbs.data?.data ?? [{ id: "research", label: "Research" } as KnowledgeBase]).map((k) => (
                <option key={k.id} value={k.id}>{k.label}</option>
              ))}
            </select>
          </label>
          {kb && <Badge status={kb.state} />}
          {kb && <span className="dash-faint">最後掃描 {formatTime(kb.last_scan_at)}</span>}
        </div>
      </div>
      <ErrorNote error={kbs.error} hasStale={!!kbs.data} />

      <nav className="dash-tabs">
        {TABS.map((t) => (
          <button key={t.id} className={`dash-tab${tab === t.id ? " is-active" : ""}`} onClick={() => set({ tab: t.id })}>
            {t.label}
          </button>
        ))}
      </nav>

      <div className={openSource ? "dash-split" : ""}>
        <div className="dash-split-main">
          {tab === "overview" && <OverviewTab summary={summary} kb={kb} />}
          {tab === "documents" && <DocumentsTab onOpen={(s) => set({ source: s.source_id, version: s.version, chunk: null })} />}
          {tab === "search" && (
            <SearchTab onOpenHit={(h) => set({ source: h.source_id, version: h.version, chunk: String(h.chunk_index) })} />
          )}
          {tab === "jobs" && <JobsTab titles={titles} />}
        </div>
        {openSource && (
          <SourceDetail
            key={`${openSource}:${openVersion ?? ""}`}
            sourceId={openSource}
            version={openVersion}
            focusChunk={focusChunk}
            onClose={() => set({ source: null, version: null, chunk: null })}
          />
        )}
      </div>
    </div>
  );
}
