// 通用指標視覺化的資料契約（對應後端 training/registry/metrics.py 與 training/evaluation_records.py）。
// 前端只依資料描述（shape／direction／unit／format）選元件與格式，不依模型種類或指標名稱寫死判斷。

export type MetricDef = {
  key: string;
  label: string;
  shape: "scalar" | "per_class" | "distribution" | "matrix";
  direction: "min" | "max" | null;
  unit: string; // loss_space／target／target_squared／ratio／count
  format: "percent" | "decimal" | "int";
  description?: string;
};

export type SeriesSpec = { id: string; metric: string; head: string; train?: string; val?: string; unit?: string };
export type HeadSpec = { key: string; label: string; task: string; target_unit?: string | null };

export type MetricSpecs = {
  round_unit: "epoch" | "boosting_round" | string;
  heads: HeadSpec[];
  series: SeriesSpec[];
  evaluation: Record<string, string[]>;
  definitions: Record<string, MetricDef>;
};

export type EvalDataset = {
  split: string; label: string; selection: string; val_ratio?: number;
  n_samples?: number | null; n_train?: number | null; n_total?: number | null; n_excluded_boundary?: number;
  phase?: number | null; task_data_range?: { start?: string; end?: string }; description?: string;
};

export type EvalPoint = {
  kind: string; round?: number | null; round_unit?: string; weights?: string; label: string;
  selected_by?: { monitor?: string | null; mode?: string | null; patience?: number | null };
};

export type Baseline = {
  key: string; label: string; available: boolean; method?: string; reason?: string;
  metrics?: Record<string, number | null>; detail?: Record<string, unknown>;
};

export type EvaluationRecord = {
  id: string; model_node: string; head: string; head_label?: string; task: string;
  dataset: EvalDataset; point: EvalPoint;
  metrics: Record<string, unknown>;
  unavailable: Record<string, string>;
  baselines: Baseline[];
};

// 逐輪紀錄：固定四欄在最上層，其他指標在 metrics
export type ProgressPoint = {
  epoch: number;
  loss: number | null; accuracy: number | null; val_loss: number | null; val_accuracy: number | null;
  metrics?: Record<string, number | null> | null;
  created_at?: string;
};

export const ROUND_NOUN: Record<string, string> = { epoch: "Epoch", boosting_round: "Boosting 輪次" };

const UNIT_LABEL: Record<string, string> = {
  loss_space: "損失空間", target: "目標原單位", target_squared: "目標原單位²", ratio: "比例", count: "筆數",
};

export function unitLabel(unit: string | undefined, targetUnit?: string | null): string {
  if (!unit) return "";
  if (unit === "target" && targetUnit) return targetUnit;
  if (unit === "target_squared" && targetUnit) return `${targetUnit}²`;
  return UNIT_LABEL[unit] ?? unit;
}

export function pointValue(p: ProgressPoint, key: string | undefined): number | null {
  if (!key) return null;
  const top = (p as unknown as Record<string, unknown>)[key];
  const v = top !== undefined ? top : p.metrics?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function formatValue(def: MetricDef | undefined, v: unknown): string {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) return "—";
  if (def?.format === "percent") return `${(v * 100).toFixed(2)}%`;
  if (def?.format === "int") return String(Math.round(v));
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 1e5)) return v.toExponential(3);
  return v.toFixed(a < 1 ? 5 : 4);
}

/** 依 direction 比較：回傳比較好的那個（相同或無方向回 null）。 */
export function better(def: MetricDef | undefined, a: number | null, b: number | null): "a" | "b" | null {
  if (!def?.direction || a === null || b === null || a === b) return null;
  return (def.direction === "max" ? a > b : a < b) ? "a" : "b";
}

export function directionHint(def: MetricDef | undefined): string {
  return def?.direction === "max" ? "越高越好" : def?.direction === "min" ? "越低越好" : "";
}

export function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
