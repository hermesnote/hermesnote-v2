// 通用指標視覺化的總控元件：依 metric_specs（模型引用的指標描述）與 evaluations（評估清單）自動組合，
// 不依模型種類寫死。前台 /model 與後台歷史紀錄共用。
//   MetricsDashboard  逐輪曲線（每個 series 一張）＋逐輪明細＋EvaluationView
//   EvaluationView    依輸出頭／資料集／評估時點篩選；比較條件、數值比較＋基準、逐類別、分布、熱圖
import { useMemo, useState } from "react";
import MetricSeriesChart, { type RefLine } from "./MetricSeriesChart";
import { DistributionBars, MatrixHeatmap, MetricCompare, PerClassTable } from "./EvaluationParts";
import {
  formatValue, pointValue, ROUND_NOUN, shapeOf, type EvaluationRecord, type MetricDef, type MetricSpecs, type ProgressPoint, type SeriesSpec,
} from "./metricsModel";
import "./metrics.css";

const SELECTION_LABEL: Record<string, string> = { random: "隨機切分（Phase 1）", chronological: "時間切分（Phase 2）" };

/** 沒有 metric_specs 的紀錄（未宣告的架構）：用逐輪資料裡出現的欄位各畫一張，名稱照原樣。 */
function fallbackSpecs(points: ProgressPoint[]): MetricSpecs {
  const keys = new Set<string>();
  for (const p of points) {
    for (const k of ["loss", "accuracy"]) if (pointValue(p, k) !== null || pointValue(p, `val_${k}`) !== null) keys.add(k);
    for (const k of Object.keys(p.metrics ?? {})) keys.add(k.replace(/^val_/, ""));
  }
  const series: SeriesSpec[] = [...keys].map((k) => ({ id: k, metric: k, head: "default", train: k, val: `val_${k}` }));
  const definitions: Record<string, MetricDef> = Object.fromEntries([...keys].map((k) => [k, { key: k, label: k, shape: "scalar", direction: null, unit: "", format: "decimal" }]));
  return { round_unit: "epoch", heads: [{ key: "default", label: "輸出", task: "" }], series, evaluation: {}, definitions };
}

export function EvaluationView({ specs, evaluations, compact = false }: {
  specs: MetricSpecs | null | undefined;
  evaluations: EvaluationRecord[];
  compact?: boolean;
}) {
  const heads = useMemo(() => {
    const present = [...new Set(evaluations.map((e) => e.head))];
    const order = (specs?.heads ?? []).map((h) => h.key);
    const rank = (h: string) => (order.indexOf(h) < 0 ? 99 : order.indexOf(h));
    return present.sort((a, b) => rank(a) - rank(b));
  }, [evaluations, specs]);
  const [head, setHead] = useState<string | null>(null);
  const activeHead = head && heads.includes(head) ? head : heads[0];
  const splits = [...new Set(evaluations.filter((e) => e.head === activeHead).map((e) => e.dataset.split))];
  const [split, setSplit] = useState<string | null>(null);
  const activeSplit = split && splits.includes(split) ? split : splits[0];
  const records = evaluations.filter((e) => e.head === activeHead && e.dataset.split === activeSplit);
  const pointKinds = [...new Set(records.map((r) => r.point.kind))];
  const [hiddenPoints, setHiddenPoints] = useState<string[]>([]);
  const shown = records.filter((r) => !hiddenPoints.includes(r.point.kind));

  if (!evaluations.length) return null;
  const defs = specs?.definitions ?? {};
  const headSpec = specs?.heads.find((h) => h.key === activeHead);
  const declared = specs?.evaluation[activeHead ?? ""] ?? Object.keys(records[0]?.metrics ?? {});
  const shapeFor = (k: string) => shapeOf(defs[k], records.map((r) => r.metrics[k]).find((v) => v !== undefined));
  const byShape = (shape: string) => declared.filter((k) => shapeFor(k) === shape);
  const scalarKeys = byShape("scalar");
  const ds = shown[0]?.dataset ?? records[0]?.dataset;
  const primary = shown.find((r) => r.point.kind === "best") ?? shown[0];
  const stop = (e: { stopPropagation: () => void }) => e.stopPropagation();

  return (
    <div className={"mv-eval" + (compact ? " is-compact" : "")} data-testid="evaluation-view" onClick={stop}>
      <div className="mv-filters">
        {heads.length > 1 && (
          <div className="mv-tabs">
            {heads.map((h) => (
              <button type="button" key={h} className={h === activeHead ? "is-active" : ""} onClick={() => setHead(h)}>
                {specs?.heads.find((x) => x.key === h)?.label ?? evaluations.find((e) => e.head === h)?.head_label ?? h}
              </button>
            ))}
          </div>
        )}
        {splits.length > 1 && (
          <div className="mv-tabs">
            {splits.map((s) => (
              <button type="button" key={s} className={s === activeSplit ? "is-active" : ""} onClick={() => setSplit(s)}>
                {evaluations.find((e) => e.dataset.split === s)?.dataset.label ?? s}
              </button>
            ))}
          </div>
        )}
        <div className="mv-points">
          {pointKinds.map((k) => (
            <label key={k}>
              <input type="checkbox" checked={!hiddenPoints.includes(k)}
                onChange={() => setHiddenPoints((prev) => prev.includes(k) ? prev.filter((x) => x !== k) : [...prev, k])} />
              {records.find((r) => r.point.kind === k)?.point.label ?? k}
            </label>
          ))}
        </div>
      </div>

      {ds && (
        <div className="mv-context" data-testid="evaluation-context">
          <div><span>評估資料集</span><strong>{ds.label}{ds.phase ? `（Phase ${ds.phase}）` : ""}</strong></div>
          <div><span>任務資料範圍</span><strong>{ds.task_data_range?.start ?? "—"} ～ {ds.task_data_range?.end ?? "—"}</strong></div>
          <div><span>切分方式</span><strong>{SELECTION_LABEL[ds.selection] ?? ds.selection}{ds.val_ratio !== undefined ? `，驗證比例 ${(ds.val_ratio * 100).toFixed(0)}%` : ""}</strong></div>
          <div><span>樣本數</span><strong>驗證 {ds.n_samples ?? "—"}／訓練 {ds.n_train ?? "—"}{ds.n_total ? `／全部 ${ds.n_total}` : ""}{ds.n_excluded_boundary ? `（邊界排除 ${ds.n_excluded_boundary}）` : ""}</strong></div>
          {ds.description && <div className="mv-context-wide"><span>驗證集怎麼來</span><strong>{ds.description}</strong></div>}
          {primary?.point.selected_by?.monitor && (
            <div className="mv-context-wide"><span>best 挑選依據</span><strong>
              監控 {primary.point.selected_by.monitor}（{primary.point.selected_by.mode === "max" ? "越高越好" : "越低越好"}）；
              早停 {primary.point.selected_by.patience === 0 ? "關閉（patience=0，跑滿輪數）" : `patience=${primary.point.selected_by.patience}`}
            </strong></div>
          )}
          {headSpec?.target_unit && <div><span>目標單位</span><strong>{headSpec.target_unit}</strong></div>}
        </div>
      )}

      <MetricCompare keys={scalarKeys} defs={defs} records={shown} baselines={primary?.baselines ?? []} />
      {(primary?.baselines ?? []).filter((b) => !b.available).map((b) => (
        <div key={b.key} className="mv-dim">基準「{b.label}」無法計算：{b.reason}</div>
      ))}
      {!compact && byShape("per_class").map((k) => <PerClassTable key={k} records={shown} metricKey={k} def={defs[k]} />)}
      <div className="mv-grid">
        {!compact && primary && byShape("distribution").map((k) => <DistributionBars key={k} record={primary} metricKey={k} def={defs[k]} />)}
        {byShape("matrix").flatMap((k) => (compact ? (primary ? [primary] : []) : shown).map((r) => (
          <MatrixHeatmap key={`${k}:${r.id}`} record={r} metricKey={k} def={defs[k]} />
        )))}
      </div>
    </div>
  );
}

export default function MetricsDashboard({ specs, points, evaluations, waitingText }: {
  specs: MetricSpecs | null | undefined;
  points: ProgressPoint[];
  evaluations: EvaluationRecord[];
  waitingText?: string | null;
}) {
  const [showTable, setShowTable] = useState(false);
  const effective = specs ?? fallbackSpecs(points);
  const noun = ROUND_NOUN[effective.round_unit] ?? effective.round_unit;
  const headLabel = (h: string) => effective.heads.find((x) => x.key === h)?.label;
  const targetUnit = (h: string) => effective.heads.find((x) => x.key === h)?.target_unit ?? null;
  // best 那一輪：所有輸出頭共用同一組權重，取任一驗證集 best 紀錄的輪次
  const bestRound = evaluations.find((e) => e.point.kind === "best" && e.dataset.split === "val")?.point.round ?? null;
  // 基準參考線：同輸出頭、驗證集 best 紀錄的基準裡有同一指標，且單位一致（損失空間的曲線不畫原單位基準）
  const refLinesOf = (s: SeriesSpec): RefLine[] => {
    const def = effective.definitions[s.metric];
    if (s.unit && s.unit !== def?.unit) return [];
    const rec = evaluations.find((e) => e.point.kind === "best" && e.dataset.split === "val" && e.head === s.head);
    return (rec?.baselines ?? []).flatMap((b) => {
      const v = b.available ? b.metrics?.[s.metric] : null;
      return typeof v === "number" ? [{ label: `基準：${b.label}`, value: v }] : [];
    });
  };
  const columns = effective.series.flatMap((s) => [s.train, s.val].filter(Boolean).map((k) => ({ key: k as string, def: effective.definitions[s.metric] })));

  return (
    <div className="mv-dashboard" data-testid="metrics-dashboard">
      {waitingText && points.length === 0 && <div className="mv-empty">{waitingText}</div>}
      <div className="mv-series-grid">
        {effective.series.map((s) => (
          <MetricSeriesChart key={s.id} series={s} def={effective.definitions[s.metric]} points={points}
            roundUnit={effective.round_unit} headLabel={effective.heads.length > 1 ? headLabel(s.head) : undefined}
            targetUnit={targetUnit(s.head)} bestRound={bestRound} refLines={refLinesOf(s)} />
        ))}
      </div>
      {points.length > 0 && (
        <>
          <button type="button" className="mv-toggle" onClick={() => setShowTable((v) => !v)}>
            {showTable ? "收起" : "展開"}逐輪明細（{points.length} 筆）
          </button>
          {showTable && (
            <div className="mv-epoch-table">
              <table className="mv-table">
                <thead><tr><th>{noun}</th>{columns.map((c) => <th key={c.key}>{c.key}</th>)}</tr></thead>
                <tbody>
                  {points.map((p) => (
                    <tr key={p.epoch}><td>{p.epoch + 1}</td>{columns.map((c) => <td key={c.key}>{formatValue(c.def, pointValue(p, c.key))}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
      {evaluations.length > 0 && (
        <>
          <div className="mv-section-title">評估結果</div>
          <EvaluationView specs={specs} evaluations={evaluations} />
        </>
      )}
    </div>
  );
}
