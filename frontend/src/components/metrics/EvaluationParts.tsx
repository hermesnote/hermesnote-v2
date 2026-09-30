// 評估結果的共用呈現元件：由 EvaluationView 依指標定義的 shape 分派，元件以 metricKey 參數讀資料，
// 同一 shape 的新指標直接沿用（資料契約見 backend training/registry/metrics.py）——
//   scalar       MetricCompare（各評估紀錄並排＋基準，依 direction 標出較佳者，附相對條）
//   per_class    PerClassTable      {類別: {欄位: 值}}
//   distribution DistributionBars   {類別: 數值}
//   matrix       MatrixHeatmap      {labels, values, row_axis, col_axis}
import { better, directionHint, formatValue, num, type Baseline, type EvaluationRecord, type MetricDef } from "./metricsModel";

export function MetricCompare({ keys, defs, records, baselines }: {
  keys: string[];
  defs: Record<string, MetricDef>;
  records: EvaluationRecord[];
  baselines: Baseline[]; // 取自第一筆紀錄（同一資料集的基準）
}) {
  const cols = [
    ...records.map((r) => ({ key: r.id, label: `${r.point.label}`, sub: r.dataset.label, get: (m: string) => r.metrics[m], reason: (m: string) => r.unavailable[m] })),
    ...baselines.filter((b) => b.available).map((b) => ({ key: `baseline:${b.key}`, label: `基準：${b.label}`, sub: b.method ?? "", get: (m: string) => b.metrics?.[m], reason: () => undefined, baseline: true })),
  ];
  const rows = keys.filter((k) => cols.some((c) => c.get(k) !== undefined || c.reason(k)));
  if (!rows.length) return null;
  return (
    <table className="mv-table mv-compare" data-testid="metric-compare">
      <thead>
        <tr><th>指標</th>{cols.map((c) => <th key={c.key}>{c.label}<div className="mv-dim">{c.sub}</div></th>)}</tr>
      </thead>
      <tbody>
        {rows.map((k) => {
          const def = defs[k];
          const values = cols.map((c) => num(c.get(k)));
          const present = values.filter((v): v is number => v !== null);
          const max = present.length ? Math.max(...present.map(Math.abs)) : 0;
          // 模型紀錄之間比較（不含基準）找出較佳者；基準另外標示是否被模型超越
          const modelVals = values.slice(0, records.length);
          let bestIdx = -1;
          modelVals.forEach((v, i) => {
            if (v === null) return;
            if (bestIdx < 0 || better(def, v, modelVals[bestIdx]) === "a") bestIdx = i;
          });
          return (
            <tr key={k}>
              <td title={def?.description}>{def?.label ?? k}<div className="mv-dim">{directionHint(def)}</div></td>
              {cols.map((c, i) => {
                const v = values[i];
                const reason = c.reason(k);
                const isBaseline = "baseline" in c;
                const beatsAll = !isBaseline && v !== null && records.length > 1 && i === bestIdx && def?.direction;
                return (
                  <td key={c.key} className={isBaseline ? "mv-baseline" : beatsAll ? "mv-better" : undefined} title={reason}>
                    {v === null ? (reason ? <span className="mv-dim">無法計算</span> : "—") : formatValue(def, v)}
                    {v !== null && max > 0 && <div className="mv-bar"><span style={{ width: `${(Math.abs(v) / max) * 100}%` }} /></div>}
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

type PerClass = Record<string, Record<string, unknown>>;

const cell = (v: unknown) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : v === null || v === undefined ? "—" : String(v));

/** shape="per_class"：{類別: {欄位: 值}}；欄位由資料決定（依第一個類別的欄位順序，再補其他類別出現的欄位）。 */
export function PerClassTable({ records, metricKey, def }: { records: EvaluationRecord[]; metricKey: string; def?: MetricDef }) {
  const withData = records.filter((r) => r.metrics[metricKey] && typeof r.metrics[metricKey] === "object");
  if (!withData.length) return null;
  const first = withData[0].metrics[metricKey] as PerClass;
  const classes = Object.keys(first);
  const fields: string[] = [];
  for (const r of withData) for (const row of Object.values(r.metrics[metricKey] as PerClass)) for (const f of Object.keys(row ?? {})) if (!fields.includes(f)) fields.push(f);
  return (
    <div className="mv-block">
      <div className="mv-block-title" title={def?.description}>{def?.label ?? metricKey}</div>
      <table className="mv-table" data-testid={`per-class-${metricKey}`}>
        <thead>
          <tr><th>類別</th>{withData.map((r) => <th key={r.id} colSpan={fields.length}>{r.point.label}</th>)}</tr>
          <tr><th />{withData.map((r) => fields.map((f) => <th key={r.id + f} className="mv-dim">{f}</th>))}</tr>
        </thead>
        <tbody>
          {classes.map((c) => (
            <tr key={c}>
              <td>{c}</td>
              {withData.map((r) => fields.map((f) => <td key={r.id + f}>{cell((r.metrics[metricKey] as PerClass)[c]?.[f])}</td>))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** shape="distribution"：{類別: 數值}。 */
export function DistributionBars({ record, metricKey, def }: { record: EvaluationRecord; metricKey: string; def?: MetricDef }) {
  const dist = record.metrics[metricKey] as Record<string, number> | undefined;
  if (!dist || typeof dist !== "object") return null;
  const total = Object.values(dist).reduce((a, b) => a + (typeof b === "number" ? b : 0), 0);
  return (
    <div className="mv-block" data-testid={`distribution-${metricKey}`}>
      <div className="mv-block-title" title={def?.description}>{def?.label ?? metricKey}<span className="mv-dim">（{record.dataset.label}，合計 {formatValue(def, total)}）</span></div>
      {Object.entries(dist).map(([c, n]) => (
        <div className="mv-dist-row" key={c}>
          <span>{c}</span>
          <div className="mv-bar is-wide"><span style={{ width: `${total ? (n / total) * 100 : 0}%` }} /></div>
          <span>{formatValue(def, n)}（{total ? ((n / total) * 100).toFixed(1) : 0}%）</span>
        </div>
      ))}
    </div>
  );
}

type Matrix = { labels: (string | number)[]; values: number[][]; row_axis?: string; col_axis?: string };

/** shape="matrix"：{labels, values, row_axis, col_axis}；顏色依列比例，對角線另外標示。 */
export function MatrixHeatmap({ record, metricKey, def }: { record: EvaluationRecord; metricKey: string; def?: MetricDef }) {
  const m = record.metrics[metricKey] as Matrix | undefined;
  if (!m || !Array.isArray(m.values)) return null;
  const labels = m.labels ?? m.values.map((_, i) => i);
  return (
    <div className="mv-block" data-testid={`matrix-${metricKey}`}>
      <div className="mv-block-title" title={def?.description}>{def?.label ?? metricKey}<span className="mv-dim">（{record.point.label}；列＝{m.row_axis ?? "列"}、欄＝{m.col_axis ?? "欄"}，顏色依列比例）</span></div>
      <div className="mv-heatmap" style={{ gridTemplateColumns: `auto repeat(${labels.length}, minmax(56px, 1fr))` }}>
        <div />
        {labels.map((l) => <div key={`c${l}`} className="mv-heat-head">{m.col_axis ?? ""} {l}</div>)}
        {m.values.map((row, i) => {
          const rowSum = row.reduce((a, b) => a + b, 0);
          return [
            <div key={`r${i}`} className="mv-heat-head">{m.row_axis ?? ""} {labels[i]}</div>,
            ...row.map((v, j) => {
              const share = rowSum ? v / rowSum : 0;
              return (
                <div key={`${i}-${j}`} className={"mv-heat-cell" + (i === j ? " is-diag" : "")}
                  style={{ background: `rgba(${i === j ? "62, 207, 142" : "240, 97, 107"}, ${0.08 + share * 0.72})` }}
                  title={`${m.row_axis ?? "列"} ${labels[i]} → ${m.col_axis ?? "欄"} ${labels[j]}：${formatValue(def, v)}（${(share * 100).toFixed(1)}%）`}>
                  <strong>{formatValue(def, v)}</strong><span>{(share * 100).toFixed(1)}%</span>
                </div>
              );
            }),
          ];
        })}
      </div>
    </div>
  );
}
