// 後台整合 API（/api/admin/*）：一律帶管理員 session token；服務 token 只在後端，前端拿不到
import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE as string;
const TOKEN_KEY = "hermesnote_admin_token";

export class AdminApiError extends Error {
  status: number;
  service: string | null;
  observedAt: string | null;
  upstreamDetail: unknown;

  constructor(status: number, message: string, service: string | null, observedAt: string | null, upstreamDetail: unknown) {
    super(message);
    this.status = status;
    this.service = service;
    this.observedAt = observedAt;
    this.upstreamDetail = upstreamDetail;
  }
}

export async function adminFetch<T>(path: string, init?: { method?: string; body?: unknown }): Promise<T> {
  const token = localStorage.getItem(TOKEN_KEY);
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/admin${path}`, {
      method: init?.method ?? "GET",
      headers: {
        Authorization: `Bearer ${token ?? ""}`,
        ...(init?.body !== undefined ? { "Content-Type": "application/json" } : {}),
      },
      body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
    });
  } catch {
    throw new AdminApiError(0, "無法連線到網站後端", null, new Date().toISOString(), null);
  }
  if (!res.ok) {
    let detail: unknown = null;
    try {
      detail = (await res.json())?.detail;
    } catch {
      /* 非 JSON 錯誤 */
    }
    if (detail && typeof detail === "object" && "reason" in detail) {
      const d = detail as { reason: string; service: string; observed_at: string; upstream_detail: unknown };
      throw new AdminApiError(res.status, d.reason, d.service, d.observed_at, d.upstream_detail);
    }
    const msg =
      res.status === 401 ? "登入已過期，請重新登入" :
      res.status === 403 ? "此帳號沒有後台權限" :
      typeof detail === "string" ? detail : `HTTP ${res.status}`;
    throw new AdminApiError(res.status, msg, null, new Date().toISOString(), detail);
  }
  return res.json() as Promise<T>;
}

export type PollState<T> = {
  data: T | null;
  error: AdminApiError | null;
  loading: boolean;
  lastSuccessAt: Date | null;
  lastAttemptAt: Date | null;
  refresh: () => void;
};

/** 低頻輪詢；失敗時保留上一次成功的資料，並標出錯誤與最後成功時間。 */
export function usePolling<T>(path: string | null, intervalMs: number): PollState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<AdminApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastSuccessAt, setLastSuccessAt] = useState<Date | null>(null);
  const [lastAttemptAt, setLastAttemptAt] = useState<Date | null>(null);
  const seq = useRef(0);

  const load = useCallback(() => {
    if (!path) return;
    const mine = ++seq.current;
    setLoading(true);
    adminFetch<T>(path)
      .then((d) => {
        if (mine !== seq.current) return;
        setData(d);
        setError(null);
        setLastSuccessAt(new Date());
      })
      .catch((e: AdminApiError) => {
        if (mine !== seq.current) return;
        setError(e);
      })
      .finally(() => {
        if (mine !== seq.current) return;
        setLoading(false);
        setLastAttemptAt(new Date());
      });
  }, [path]);

  useEffect(() => {
    load();
    if (!intervalMs) return;
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") load();
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [load, intervalMs]);

  return { data, error, loading, lastSuccessAt, lastAttemptAt, refresh: load };
}

// ---------------------------------------------------------------- 型別

export type Probe = {
  ok: boolean;
  http_status: number | null;
  latency_ms: number | null;
  error: string | null;
  body: Record<string, unknown> | null;
  observed_at: string;
};

export type ServiceEntry = {
  id: string;
  label: string;
  state: "ready" | "degraded" | "down" | "unconfigured";
  base_url: string | null;
  credential_configured: boolean;
  credential_source: "env" | "secrets_file" | null;
  alive: Probe;
  ready: Probe;
};

export type WorkItem = {
  provider: string;
  kind: string;
  id: string;
  title: string;
  status: string;
  progress: { current: number; total: number | null; unit: string } | null;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
  delivery: { state: string; label: string } | null;
  link: string;
};

export type Provider = {
  id: string;
  label: string;
  state: "connected" | "error" | "not_connected";
  error: string | null;
  note?: string;
  items: WorkItem[];
  observed_at: string | null;
  latency_ms: number | null;
};

export type Overview = {
  observed_at: string;
  counts: { running: number; pending: number; attention: number; service_issues: number };
  counted_providers: string[];
  unavailable_providers: { id: string; label: string; error: string | null }[];
  providers: Provider[];
  services: ServiceEntry[];
};

export type Envelope<T> = {
  ok: boolean;
  data: T;
  error: unknown;
  observed_at: string;
  provenance: unknown[];
  next_cursor: string | null;
  latency_ms?: number;
};

export type Locator = { page?: number; line_start?: number; line_end?: number };
export type QualityWarning = { reason: string; locator?: Locator };

export type RagSource = {
  source_id: string;
  version: string;
  title: string;
  source_type: string;
  status: string;
  metadata: { warnings?: QualityWarning[]; model_id?: string; pipeline?: string; [k: string]: unknown } | null;
  updated_at: string;
};

export type RagChunk = {
  chunk_index: number;
  content: string;
  locator: Locator;
  quality_warnings: QualityWarning[];
};

export type RagContent = {
  source_id: string;
  version: string;
  title: string;
  source_type: string;
  status: string;
  is_current: boolean;
  metadata: RagSource["metadata"];
  chunks: RagChunk[];
};

export type RagHit = RagChunk & {
  source_id: string;
  version: string;
  title: string;
  source_type: string;
  score: number;
  metadata: RagSource["metadata"];
};

export type RagJob = {
  job_id: string;
  source_id: string;
  version: string;
  stage: string;
  status: string;
  error: string | null;
  chunk_count: number | null;
  started_at: string | null;
  finished_at: string | null;
};

export type KnowledgeBase = {
  id: string;
  label: string;
  state: string;
  last_scan_at: string | null;
  scan_error: string | null;
  sources_scanned: number;
  failures: number;
};

export type RagSummary = {
  knowledge_base: string;
  health: Omit<KnowledgeBase, "id" | "label">;
  documents: { source_type: string; status: string; count: number }[];
  chunks: number;
};
