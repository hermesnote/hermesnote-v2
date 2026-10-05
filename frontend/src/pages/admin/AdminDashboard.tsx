import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { usePolling, type Overview, type WorkItem } from "./dashboard/adminApi";
import { Badge, ErrorNote, PanelHead, ServiceCard, formatTime, relativeTime } from "./dashboard/parts";
import "./dashboard/dashboard.css";

const ATTENTION = new Set(["failed", "needs_review", "blocked", "awaiting_approval"]);

function byUpdatedDesc(a: WorkItem, b: WorkItem) {
  return (b.updated_at ?? "").localeCompare(a.updated_at ?? "");
}

function ProgressCell({ item }: { item: WorkItem }) {
  const p = item.progress;
  if (!p) return <span className="dash-faint">—</span>;
  const pct = p.total ? Math.min(100, Math.round((p.current / p.total) * 100)) : null;
  return (
    <div className="dash-progress" title={pct != null ? `${pct}%` : ""}>
      {pct != null && <div className="dash-progress-bar"><div style={{ width: `${pct}%` }} /></div>}
      <span>{p.current}{p.total ? ` / ${p.total}` : ""} {p.unit}</span>
    </div>
  );
}

function ItemLink({ item, children }: { item: WorkItem; children: ReactNode }) {
  // 站內路由（/model?job=、/quant?job=、/admin/rag?source=）
  return <Link className="dash-link" to={item.link}>{children}</Link>;
}

export default function AdminDashboard() {
  const poll = usePolling<Overview>("/overview", 30_000);
  const data = poll.data;

  const connected = data?.providers.filter((p) => p.state === "connected") ?? [];
  const items = connected.flatMap((p) => p.items);
  const active = items.filter((i) => i.status === "running" || i.status === "pending").sort(byUpdatedDesc);
  const attention = items.filter((i) => ATTENTION.has(i.status)).sort(byUpdatedDesc);
  const recentDone = items.filter((i) => i.status === "done").sort(byUpdatedDesc).slice(0, 8);
  const unavailable = data?.unavailable_providers ?? [];
  const notConnected = data?.providers.filter((p) => p.state === "not_connected") ?? [];
  const countedLabels = connected.map((p) => p.label).join("、");

  const summary = [
    { key: "running", label: "執行中", value: data?.counts.running, tone: "info" },
    { key: "pending", label: "待處理", value: data?.counts.pending, tone: "warn" },
    { key: "attention", label: "需要處理", value: data?.counts.attention, tone: "bad" },
    { key: "service_issues", label: "服務異常", value: data?.counts.service_issues, tone: "bad" },
  ];

  return (
    <div className="dash">
      <PanelHead title="工作總覽" lastSuccessAt={poll.lastSuccessAt} loading={poll.loading} onRefresh={poll.refresh} />
      <ErrorNote error={poll.error} hasStale={!!data} />

      <section className="dash-summary">
        {summary.map((s) => (
          <div key={s.key} className={`dash-stat is-${s.tone}${s.value ? " is-active" : ""}`}>
            <div className="dash-stat-value">{s.value ?? "—"}</div>
            <div className="dash-stat-label">{s.label}</div>
          </div>
        ))}
        <div className="dash-summary-note">
          {data ? (
            <>
              <div>統計來源：{countedLabels || "無"}</div>
              {unavailable.length > 0 && (
                <div className="dash-bad">無法確認（未計入）：{unavailable.map((u) => `${u.label}（${u.error}）`).join("、")}</div>
              )}
              <div className="dash-faint">尚未接入：{notConnected.map((p) => p.label).join("、")}</div>
            </>
          ) : poll.loading ? "查詢中…" : "尚無資料"}
        </div>
      </section>

      <div className="dash-grid">
        <section className="dash-panel dash-main">
          <PanelHead title="正在進行的工作" />
          {active.length === 0 ? (
            <div className="dash-empty">
              {data ? `已接入的來源目前沒有執行中或待處理的工作${unavailable.length ? "（部分來源無法確認）" : ""}` : "—"}
            </div>
          ) : (
            <table className="dash-table">
              <thead>
                <tr><th>類型</th><th>名稱</th><th>狀態</th><th>進度</th><th>更新時間</th><th /></tr>
              </thead>
              <tbody>
                {active.map((i) => (
                  <tr key={`${i.provider}:${i.id}`}>
                    <td className="dash-faint">{i.kind}</td>
                    <td>{i.title}</td>
                    <td><Badge status={i.status} /></td>
                    <td><ProgressCell item={i} /></td>
                    <td title={formatTime(i.updated_at)}>{relativeTime(i.updated_at)}</td>
                    <td><ItemLink item={i}>詳細</ItemLink></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <aside className="dash-side">
          <section className="dash-panel">
            <PanelHead title="需要處理" />
            {attention.length === 0 ? (
              <div className="dash-empty">{data ? "已接入的來源沒有失敗或待核對項目" : "—"}</div>
            ) : (
              <ul className="dash-list">
                {attention.slice(0, 10).map((i) => (
                  <li key={`${i.provider}:${i.id}`}>
                    <div className="dash-list-head">
                      <Badge status={i.status} />
                      <span className="dash-faint">{i.kind}</span>
                      <span className="dash-faint dash-right" title={formatTime(i.updated_at)}>{relativeTime(i.updated_at)}</span>
                    </div>
                    <ItemLink item={i}>{i.title}</ItemLink>
                    {i.error && <div className="dash-list-error">{i.error}</div>}
                  </li>
                ))}
                {attention.length > 10 && <li className="dash-faint">另有 {attention.length - 10} 項</li>}
              </ul>
            )}
          </section>

          <section className="dash-panel">
            <PanelHead title="服務狀態" />
            {data?.services.map((s) => <ServiceCard key={s.id} service={s} compact />)}
            <div className="dash-probe">
              <span className="dash-probe-label">Agent 已連線</span>
              <Badge status="not_connected" />
              <span className="dash-faint">HA Profile／SDK 尚未接入；服務正常不代表 Agent 已連上</span>
            </div>
          </section>
        </aside>
      </div>

      <div className="dash-grid">
        <section className="dash-panel dash-main">
          <PanelHead title="最近完成與交付" />
          {recentDone.length === 0 ? (
            <div className="dash-empty">—</div>
          ) : (
            <table className="dash-table">
              <thead>
                <tr><th>類型</th><th>名稱</th><th>工作狀態</th><th>交付狀態</th><th>完成時間</th><th /></tr>
              </thead>
              <tbody>
                {recentDone.map((i) => (
                  <tr key={`${i.provider}:${i.id}`}>
                    <td className="dash-faint">{i.kind}</td>
                    <td>{i.title}</td>
                    <td><Badge status={i.status} /></td>
                    <td>{i.delivery ? <Badge label={i.delivery.label} tone={i.delivery.state === "available" ? "ok" : "warn"} /> : "—"}</td>
                    <td title={formatTime(i.updated_at)}>{relativeTime(i.updated_at)}</td>
                    <td><ItemLink item={i}>查看</ItemLink></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <aside className="dash-side">
          <section className="dash-panel">
            <PanelHead title="快速入口" />
            <div className="dash-quick">
              <Link to="/admin/rag" className="dash-quick-link">知識庫</Link>
              <Link to="/admin/mcp" className="dash-quick-link">MCP 服務</Link>
              <Link to="/admin/model" className="dash-quick-link">模型訓練</Link>
              <Link to="/admin/quant" className="dash-quick-link">量化回測</Link>
            </div>
          </section>
          <section className="dash-panel">
            <PanelHead title="工作來源" />
            <ul className="dash-list">
              {data?.providers.map((p) => (
                <li key={p.id} className="dash-source">
                  <div className="dash-list-head">
                    <span>{p.label}</span>
                    <span className="dash-right"><Badge status={p.state} /></span>
                  </div>
                  <div className="dash-faint">
                    {p.state === "connected" && `${p.items.length} 筆 · ${p.latency_ms} ms`}
                    {p.state === "error" && p.error}
                    {p.state === "not_connected" && (p.note ?? "")}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        </aside>
      </div>
    </div>
  );
}
