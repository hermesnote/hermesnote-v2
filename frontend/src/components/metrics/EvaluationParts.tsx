// 評估結果的共用呈現元件：依指標定義的 shape 選用——
//   scalar       MetricCompare（各評估紀錄並排＋基準，依 direction 標出較佳者，附相對條）
//   per_class    PerClassTable
//   distribution DistributionBars
//   matrix       MatrixHeatmap
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

type PerClass = Record<string, { precision: number; recall: number; f1: number; support: number }>;

export function PerClassTable({ records, def }: { records: EvaluationRecord[]; def?: MetricDef }) {
  const withData = records.filter((r) => r.metrics.per_class);
  if (!withData.length) return null;
  const classes = Object.keys(withData[0].metrics.per_class as PerClass);
  return (
    <div className="mv-block">
      <div className="mv-block-title">{def?.label ?? "逐類別指標"}</div>
      <table className="mv-table" data-testid="per-class-table">
        <thead>
          <tr><th>類別</th>{withData.map((r) => <th key={r.id} colSpan={4}>{r.point.label}</th>)}</tr>
          <tr><th />{withData.map((r) => ["precision", "recall", "f1", "support"].map((f) => <th key={r.id + f} className="mv-dim">{f}</th>))}</tr>
        </thead>
        <tbody>
          {classes.map((c) => (
            <tr key={c}>
              <td>{c}</td>
              {withData.map((r) => {
                const row = (r.metrics.per_class as PerClass)[c];
                return [row?.precision, row?.recall, row?.f1].map((v, i) => <td key={r.id + i}>{typeof v === "number" ? v.toFixed(4) : "—"}</td>)
                  .concat(<td key={r.id + "s"}>{row?.support ?? "—"}</td>);
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DistributionBars({ record, def }: { record: EvaluationRecord; def?: MetricDef }) {
  const dist = record.metrics.class_distribution as Record<string, number> | undefined;
  if (!dist) return null;
  const total = Object.values(dist).reduce((a, b) => a + b, 0);
  return (
    <div className="mv-block">
      <div className="mv-block-title">{def?.label ?? "類別分布"}<span className="mv-dim">（{record.dataset.label}，{total} 筆）</span></div>
      {Object.entries(dist).map(([c, n]) => (
        <div className="mv-dist-row" key={c}>
          <span>類別 {c}</span>
          <div className="mv-bar is-wide"><span style={{ width: `${total ? (n / total) * 100 : 0}%` }} /></div>
          <span>{n}（{total ? ((n / total) * 100).toFixed(1) : 0}%）</span>
        </div>
      ))}
    </div>
  );
}

type Matrix = { labels: (string | number)[]; values: number[][]; row_axis?: string; col_axis?: string };

export function MatrixHeatmap({ record, def }: { record: EvaluationRecord; def?: MetricDef }) {
  const m = record.metrics.confusion_matrix as Matrix | undefined;
  if (!m) return null;
  return (
    <div className="mv-block" data-testid="matrix-heatmap">
      <div className="mv-block-title">{def?.label ?? "矩陣"}<span className="mv-dim">（{record.point.label}；列＝{m.row_axis ?? "列"}、欄＝{m.col_axis ?? "欄"}，顏色依列比例）</span></div>
      <div className="mv-heatmap" style={{ gridTemplateColumns: `auto repeat(${m.labels.length}, minmax(56px, 1fr))` }}>
        <div />
        {m.labels.map((l) => <div key={`c${l}`} className="mv-heat-head">{m.col_axis ?? ""} {l}</div>)}
        {m.values.map((row, i) => {
          const rowSum = row.reduce((a, b) => a + b, 0);
          return [
            <div key={`r${i}`} className="mv-heat-head">{m.row_axis ?? ""} {m.labels[i]}</div>,
            ...row.map((v, j) => {
              const share = rowSum ? v / rowSum : 0;
              return (
                <div key={`${i}-${j}`} className={"mv-heat-cell" + (i === j ? " is-diag" : "")}
                  style={{ background: `rgba(${i === j ? "62, 207, 142" : "240, 97, 107"}, ${0.08 + share * 0.72})` }}
                  title={`${m.row_axis ?? "列"} ${m.labels[i]} → ${m.col_axis ?? "欄"} ${m.labels[j]}：${v}（${(share * 100).toFixed(1)}%）`}>
                  <strong>{v}</strong><span>{(share * 100).toFixed(1)}%</span>
                </div>
              );
            }),
          ];
        })}
      </div>
    </div>
  );
}
