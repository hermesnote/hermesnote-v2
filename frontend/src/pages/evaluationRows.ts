// 完整評估報告（backend/training/evaluation.py）→ 顯示用的 [名稱, 數值] 列。後台歷史紀錄與訓練頁共用，
// 各自用自己的版面渲染。報告形狀依模型輸出而定：單頭分類＝分類報告、單頭回歸＝回歸報告、
// 雙頭＝{direction: 分類報告, regression: 回歸報告}。算不出來的指標是 null＋`*_unavailable_reason`，照實顯示原因。

export type EvaluationReport = Record<string, unknown>;
export type EvaluationPair = { best: EvaluationReport; last: EvaluationReport };

const num = (v: unknown, digits: number) => (typeof v === "number" ? v.toFixed(digits) : "—");
const orReason = (ev: EvaluationReport, key: string, digits = 4) =>
  ev[key] === null || ev[key] === undefined
    ? `無法計算（${String(ev[`${key}_unavailable_reason`] ?? "未提供")}）`
    : num(ev[key], digits);

function classificationRows(ev: EvaluationReport, prefix: string): [string, string][] {
  const base = ev.baseline_majority_class as Record<string, number> | null;
  return [
    [`${prefix}macro-F1／balanced accuracy`, `${num(ev.macro_f1, 4)}／${num(ev.balanced_accuracy, 4)}`],
    [`${prefix}ROC-AUC／AP`, `${orReason(ev, "roc_auc")}／${orReason(ev, "average_precision")}`],
    [`${prefix}混淆矩陣（列＝實際、欄＝預測，類別 ${(ev.confusion_matrix_labels as number[]).join("/")}）`,
      (ev.confusion_matrix as number[][]).map((row) => `[${row.join(",")}]`).join(" ")],
    [`${prefix}基準：訓練集多數類別 macro-F1`, base ? num(base.macro_f1, 4) : orReason(ev, "baseline_majority_class")],
  ];
}

function regressionRows(ev: EvaluationReport, prefix: string): [string, string][] {
  const mean = ev.baseline_mean as Record<string, number> | null;
  const zero = ev.baseline_zero as Record<string, number> | null;
  return [
    [`${prefix}MSE／RMSE／MAE`, `${num(ev.mse, 6)}／${num(ev.rmse, 6)}／${num(ev.mae, 6)}`],
    [`${prefix}R²`, orReason(ev, "r2")],
    [`${prefix}基準 RMSE（訓練集平均／預測零）`,
      `${mean ? num(mean.rmse, 6) : orReason(ev, "baseline_mean")}／${zero ? num(zero.rmse, 6) : "—"}`],
  ];
}

export function evaluationRows(ev: EvaluationReport): [string, string][] {
  if ("direction" in ev && "regression" in ev) {
    return [
      ...classificationRows(ev.direction as EvaluationReport, "方向頭 "),
      ...regressionRows(ev.regression as EvaluationReport, "回歸頭 "),
    ];
  }
  return "confusion_matrix" in ev ? classificationRows(ev, "") : regressionRows(ev, "");
}
