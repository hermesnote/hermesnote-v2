import { useState } from "react";
import { adminFetch, usePolling, type AdminApiError, type ServiceEntry } from "../dashboard/adminApi";
import { Badge, ErrorNote, PanelHead, ServiceCard, formatTime } from "../dashboard/parts";
import "../dashboard/dashboard.css";

type McpStatus = {
  service: ServiceEntry;
  upstream_rag: ServiceEntry;
  protocol_endpoint: string | null;
  agent_connections: { id: string; label: string; state: string; note: string }[];
  observed_at: string;
};

type JsonSchema = {
  type?: string;
  properties?: Record<string, JsonSchema & { default?: unknown; title?: string; anyOf?: JsonSchema[] }>;
  required?: string[];
  anyOf?: JsonSchema[];
};

type Tool = {
  name: string;
  title?: string | null;
  description?: string | null;
  inputSchema?: JsonSchema;
  annotations?: { readOnlyHint?: boolean; destructiveHint?: boolean; openWorldHint?: boolean } | null;
};

type TestResult = {
  ok: boolean;
  observed_at: string;
  steps: { step: string; ok: boolean; latency_ms: number | null; detail: unknown; error: string | null }[];
};

function typeLabel(s: JsonSchema): string {
  if (s.anyOf) return s.anyOf.map(typeLabel).join(" | ");
  return s.type ?? "any";
}

function ToolCard({ tool }: { tool: Tool }) {
  const [raw, setRaw] = useState(false);
  const props = tool.inputSchema?.properties ?? {};
  const required = new Set(tool.inputSchema?.required ?? []);
  const a = tool.annotations ?? {};
  return (
    <div className="dash-tool">
      <div className="dash-chunk-head">
        <strong className="dash-mono">{tool.name}</strong>
        {a.readOnlyHint && <Badge label="唯讀" tone="ok" />}
        {a.destructiveHint === false && <Badge label="非破壞性" tone="ok" />}
        {a.openWorldHint === false && <Badge label="封閉範圍" tone="idle" />}
      </div>
      {tool.description && <div className="dash-tool-desc">{tool.description}</div>}
      {Object.keys(props).length > 0 ? (
        <table className="dash-table is-dense">
          <thead><tr><th>參數</th><th>型別</th><th>必填</th><th>預設</th></tr></thead>
          <tbody>
            {Object.entries(props).map(([k, v]) => (
              <tr key={k}>
                <td className="dash-mono">{k}</td>
                <td className="dash-mono dash-faint">{typeLabel(v)}</td>
                <td>{required.has(k) ? "是" : ""}</td>
                <td className="dash-mono dash-faint">{v.default !== undefined ? JSON.stringify(v.default) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="dash-faint">無輸入參數</div>
      )}
      <button className="dash-btn is-link" onClick={() => setRaw((v) => !v)}>{raw ? "收起" : "完整輸入定義（JSON Schema）"}</button>
      {raw && <pre className="dash-chunk-text dash-mono">{JSON.stringify(tool.inputSchema, null, 2)}</pre>}
    </div>
  );
}

export default function McpService() {
  const status = usePolling<McpStatus>("/mcp/status", 30_000);
  const tools = usePolling<{ service: string; tools: Tool[]; observed_at: string; latency_ms: number }>("/mcp/tools", 0);
  const [test, setTest] = useState<TestResult | null>(null);
  const [testError, setTestError] = useState<AdminApiError | null>(null);
  const [testing, setTesting] = useState(false);

  async function runTest() {
    setTesting(true);
    setTestError(null);
    try {
      setTest(await adminFetch<TestResult>("/mcp/test", { method: "POST" }));
    } catch (e) {
      setTestError(e as AdminApiError);
    } finally {
      setTesting(false);
    }
  }

  const s = status.data;
  return (
    <div className="dash">
      <PanelHead title="MCP 服務" lastSuccessAt={status.lastSuccessAt} loading={status.loading} onRefresh={status.refresh} />
      <ErrorNote error={status.error} hasStale={!!s} />

      <div className="dash-grid">
        <div className="dash-main">
          <section className="dash-panel">
            <PanelHead title="服務狀態" />
            <div className="dash-services">
              {s && <ServiceCard service={s.service} />}
              {s && <ServiceCard service={s.upstream_rag} />}
            </div>
            <div className="dash-faint dash-note">
              MCP 就緒（/ready）會檢查上游 RAG；存活只代表程序在跑。
            </div>
          </section>

          <section className="dash-panel">
            <PanelHead
              title={`工具${tools.data ? `（${tools.data.tools.length}）` : ""}`}
              lastSuccessAt={tools.lastSuccessAt}
              loading={tools.loading}
              onRefresh={tools.refresh}
            />
            <ErrorNote error={tools.error} hasStale={!!tools.data} />
            <div className="dash-tools">
              {tools.data?.tools.map((t) => <ToolCard key={t.name} tool={t} />)}
            </div>
          </section>
        </div>

        <aside className="dash-side">
          <section className="dash-panel">
            <PanelHead title="連線測試" />
            <div className="dash-faint dash-note">
              以 MCP 協定唯讀測試：initialize → tools/list → 呼叫 knowledge_bases。不寫入任何資料。
            </div>
            <button className="dash-btn" onClick={runTest} disabled={testing}>{testing ? "測試中…" : "執行連線測試"}</button>
            <ErrorNote error={testError} />
            {test && (
              <div className="dash-test">
                <div className="dash-list-head">
                  <Badge label={test.ok ? "通過" : "未通過"} tone={test.ok ? "ok" : "bad"} />
                  <span className="dash-faint">{formatTime(test.observed_at)}</span>
                </div>
                <ol className="dash-steps">
                  {test.steps.map((st) => (
                    <li key={st.step}>
                      <div className="dash-list-head">
                        <Badge label={st.ok ? "OK" : "失敗"} tone={st.ok ? "ok" : "bad"} />
                        <span className="dash-mono">{st.step}</span>
                        <span className="dash-faint dash-right">{st.latency_ms} ms</span>
                      </div>
                      {st.error && <div className="dash-list-error">{st.error}</div>}
                      {st.detail != null && <pre className="dash-chunk-text dash-mono is-small">{JSON.stringify(st.detail, null, 2)}</pre>}
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </section>

          <section className="dash-panel">
            <PanelHead title="端點" />
            <dl className="dash-kv">
              <dt>服務位址</dt><dd className="dash-mono">{s?.service.base_url ?? "—"}</dd>
              <dt>協定入口</dt><dd className="dash-mono">{s?.protocol_endpoint ?? "—"}</dd>
              <dt>傳輸</dt><dd>Streamable HTTP（stateless、JSON 回應）</dd>
              <dt>認證</dt><dd>Bearer（除 /health 外）；憑證由網站後端保存</dd>
            </dl>
            <div className="dash-faint dash-note">協定入口給 MCP 客戶端連線用，不是瀏覽器操作頁。</div>
          </section>

          <section className="dash-panel">
            <PanelHead title="Agent 連線" />
            <ul className="dash-list">
              {s?.agent_connections.map((a) => (
                <li key={a.id}>
                  <div className="dash-list-head">
                    <span>{a.label}</span>
                    <span className="dash-right"><Badge status={a.state} /></span>
                  </div>
                  <div className="dash-faint">{a.note}</div>
                </li>
              ))}
            </ul>
            <div className="dash-faint dash-note">服務健康不能推論已有哪些 Agent 接入；接線後才會顯示實際連線來源。</div>
          </section>
        </aside>
      </div>
    </div>
  );
}
