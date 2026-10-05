import type { ReactNode } from "react";
import type { AdminApiError, Locator, Probe, ServiceEntry } from "./adminApi";

export type Tone = "ok" | "warn" | "bad" | "idle" | "info";

const STATUS: Record<string, { label: string; tone: Tone }> = {
  running: { label: "執行中", tone: "info" },
  pending: { label: "待處理", tone: "warn" },
  queued: { label: "排隊中", tone: "warn" },
  done: { label: "完成", tone: "ok" },
  indexed: { label: "已索引", tone: "ok" },
  failed: { label: "失敗", tone: "bad" },
  cancelled: { label: "已取消", tone: "idle" },
  needs_review: { label: "待核對", tone: "warn" },
  blocked: { label: "封鎖", tone: "bad" },
  awaiting_approval: { label: "待核准", tone: "warn" },
  complete: { label: "完成", tone: "ok" },
  // 服務／來源狀態
  ready: { label: "就緒", tone: "ok" },
  degraded: { label: "存活但未就緒", tone: "warn" },
  down: { label: "無法連線", tone: "bad" },
  unconfigured: { label: "後端未設定憑證", tone: "bad" },
  connected: { label: "已接入", tone: "ok" },
  error: { label: "無法確認", tone: "bad" },
  not_connected: { label: "尚未接入", tone: "idle" },
};

export function statusInfo(status: string): { label: string; tone: Tone } {
  return STATUS[status] ?? { label: status, tone: "idle" };
}

export function Badge({ status, label, tone }: { status?: string; label?: string; tone?: Tone }) {
  const info = status ? statusInfo(status) : { label: label ?? "", tone: tone ?? "idle" };
  return <span className={`dash-badge is-${tone ?? info.tone}`}>{label ?? info.label}</span>;
}

export function formatTime(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const d = typeof value === "string" ? new Date(value) : value;
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString("zh-TW", { hour12: false, month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function relativeTime(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const d = typeof value === "string" ? new Date(value) : value;
  const sec = Math.round((Date.now() - d.getTime()) / 1000);
  if (Number.isNaN(sec)) return String(value);
  if (sec < 60) return "剛剛";
  if (sec < 3600) return `${Math.floor(sec / 60)} 分鐘前`;
  if (sec < 86400) return `${Math.floor(sec / 3600)} 小時前`;
  return `${Math.floor(sec / 86400)} 天前`;
}

export function locatorLabel(loc: Locator | null | undefined): string {
  if (!loc) return "—";
  if (loc.page != null) return `第 ${loc.page} 頁`;
  if (loc.line_start != null) return loc.line_end && loc.line_end !== loc.line_start ? `第 ${loc.line_start}–${loc.line_end} 行` : `第 ${loc.line_start} 行`;
  return JSON.stringify(loc);
}

export const SOURCE_TYPE_LABEL: Record<string, string> = {
  pdf: "PDF 文獻",
  research_card: "研究卡",
  literature_index: "文獻索引",
};

export function shortId(id: string | null | undefined, n = 8): string {
  return id ? id.slice(0, n) : "—";
}

/** 區塊標頭：標題＋最後成功查詢時間＋重新整理。 */
export function PanelHead({ title, lastSuccessAt, loading, onRefresh, extra }: {
  title: string;
  lastSuccessAt?: Date | null;
  loading?: boolean;
  onRefresh?: () => void;
  extra?: ReactNode;
}) {
  return (
    <div className="dash-panel-head">
      <h2 className="dash-panel-title">{title}</h2>
      <div className="dash-panel-meta">
        {extra}
        {lastSuccessAt !== undefined && (
          <span title={lastSuccessAt ? formatTime(lastSuccessAt) : ""}>
            {lastSuccessAt ? `最後成功查詢 ${formatTime(lastSuccessAt)}` : "尚未成功查詢"}
          </span>
        )}
        {onRefresh && (
          <button className="dash-btn is-ghost" onClick={onRefresh} disabled={loading}>
            {loading ? "查詢中…" : "重新整理"}
          </button>
        )}
      </div>
    </div>
  );
}

/** 查詢失敗：顯示原因與時間；有舊資料時註明目前顯示的是舊資料。 */
export function ErrorNote({ error, hasStale }: { error: AdminApiError | null; hasStale?: boolean }) {
  if (!error) return null;
  return (
    <div className="dash-error" role="alert">
      <strong>無法確認</strong>
      <span>{error.service ? `${error.service.toUpperCase()}：` : ""}{error.message}</span>
      <span className="dash-faint">（{formatTime(error.observedAt)}）</span>
      {hasStale && <span className="dash-faint">下方為上一次成功查詢的資料</span>}
    </div>
  );
}

function ProbeRow({ label, probe, hint }: { label: string; probe: Probe; hint: string }) {
  return (
    <div className="dash-probe">
      <span className="dash-probe-label" title={hint}>{label}</span>
      <Badge label={probe.ok ? "正常" : "異常"} tone={probe.ok ? "ok" : "bad"} />
      <span className="dash-faint">
        {probe.ok ? `${probe.latency_ms} ms` : probe.error ?? `HTTP ${probe.http_status}`}
      </span>
    </div>
  );
}

/** 服務狀態：存活（/health）、就緒（/ready）、憑證設定分開呈現。 */
export function ServiceCard({ service, compact }: { service: ServiceEntry; compact?: boolean }) {
  return (
    <div className="dash-service">
      <div className="dash-service-head">
        <span className="dash-service-name">{service.label}</span>
        <Badge status={service.state} />
      </div>
      <ProbeRow label="服務存活" probe={service.alive} hint="/health：程序是否在跑，不代表整條鏈可用" />
      <ProbeRow label="服務就緒" probe={service.ready} hint="/ready：需認證；資料庫／上游與索引是否就緒" />
      {!compact && (
        <div className="dash-probe">
          <span className="dash-probe-label">後端憑證</span>
          <Badge label={service.credential_configured ? "已設定" : "未設定"} tone={service.credential_configured ? "ok" : "bad"} />
          <span className="dash-faint">
            {service.credential_source === "env" ? "來源：環境變數" : service.credential_source === "secrets_file" ? "來源：NAS 服務設定檔（唯讀掛載）" : ""}
          </span>
        </div>
      )}
      <div className="dash-faint dash-service-foot">
        {service.base_url ?? "未設定位址"} · 檢查於 {formatTime(service.alive.observed_at)}
      </div>
    </div>
  );
}

export function Pager({ offset, limit, nextCursor, onChange, total }: {
  offset: number;
  limit: number;
  nextCursor: string | null;
  onChange: (offset: number) => void;
  total?: number;
}) {
  const page = Math.floor(offset / limit) + 1;
  return (
    <div className="dash-pager">
      <button className="dash-btn is-ghost" disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - limit))}>上一頁</button>
      <span className="dash-faint">第 {page} 頁{total != null ? ` · 共 ${total} 筆` : ""}</span>
      <button className="dash-btn is-ghost" disabled={!nextCursor} onClick={() => nextCursor && onChange(Number(nextCursor))}>下一頁</button>
    </div>
  );
}
