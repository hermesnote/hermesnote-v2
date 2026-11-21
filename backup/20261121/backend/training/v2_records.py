"""逐筆結果（model_prediction_records）的列產生器與 Phase 3 評估組裝（M3）。

逐筆結果以產生器輸出，worker 用 COPY 串流寫入，不在記憶體裡累積成整份清單。
每個時間點：best、last 各一列（branches／upstream 只寫在 best 那列，避免重複）；輸入不足而沒有預測的時間點
寫一列 weights=NULL、included=false、exclusion_reason＝原因代碼。holdout 尾端答案未成熟的點照常預測、
included=false、exclusion_reason=label_unmatured。
"""

import json

import numpy as np
import pandas as pd

from training.engine_v2 import branch_meta_json, upstream_json
from training.evaluation_records import SPLIT_LABEL, _report_to_metrics
from training.holdout_eval import evaluate_outputs

COLUMNS = ["job_id", "node_id", "experiment_id", "arm_id", "frame_hash", "phase", "dataset", "weights", "model_seed",
           "sample_ts", "decision_available_ts", "window_start_ts", "target_ts", "target_available_ts",
           "y_true", "y_pred", "proba", "included", "exclusion_reason", "branches", "upstream"]


def _dt(v):
    return None if v is None or np.isnat(v) else pd.Timestamp(v).to_pydatetime()


def _f(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _pred_cols(out: dict, i: int):
    if out.get("dual") is not None:
        p = float(out["dual"]["direction_probability"][i])
        return float(out["dual"]["regression"][i]), json.dumps([1 - p, p])
    if out.get("proba") is not None:
        return float(out["pred"][i]), json.dumps([float(x) for x in out["proba"][i]])
    return float(out["pred"][i]), None


def prediction_rows(entry: dict, *, job_id, node_id: str, dataset: str, phase=None, model_seed=None,
                    experiment_id=None, arm_id=None):
    rows, asm = entry["rows"], entry["asm"]
    pts = np.asarray(entry["points"])
    common = (job_id, node_id, experiment_id, arm_id, entry.get("frame_hash"), phase, dataset)

    def base(r):
        return (_dt(rows.ts[r]), _dt(rows.tau[r]), _dt(asm.window_start[r]), _dt(rows.target_ts[r]), _dt(rows.le[r]))

    for w, out in entry["by_weights"].items():
        for i, r in enumerate(pts):
            fr_reason = rows.frame_reason[r]
            included = dataset != "holdout" or fr_reason is None
            y_pred, proba = _pred_cols(out, i)
            yield (*common, w, model_seed, *base(r), _f(rows.y[r]), y_pred, proba, bool(included),
                   None if included else fr_reason,
                   branch_meta_json(asm, r) if w == "best" else None, upstream_json(asm, r) if w == "best" else None)
    for r in np.asarray(entry.get("skipped", [])):
        yield (*common, None, model_seed, *base(r), _f(rows.y[r]), None, None, False, asm.reason[r],
               branch_meta_json(asm, r), upstream_json(asm, r))
    for r in np.asarray((entry.get("train_upstream") or {}).get("points", [])):
        yield (job_id, node_id, experiment_id, arm_id, entry.get("frame_hash"), phase, "train", None, model_seed,
               *base(r), _f(rows.y[r]), None, None, False, None, None, upstream_json(asm, r))


def evaluate_entry(entry: dict, *, task_type: str, model_config: dict, stats: dict | None) -> tuple[dict, dict]:
    """holdout 逐點預測 → ({best: 報告, last: 報告}, 統計)。只評估答案已成熟的點。"""
    rows = entry["rows"]
    pts = np.asarray(entry["points"])
    inc = np.array([rows.frame_reason[r] is None for r in pts], dtype=bool)
    y_true = rows.y[pts[inc]]
    raw = {}
    for w, out in entry["by_weights"].items():
        sub = {"pred": np.asarray(out["pred"])[inc],
               "proba": None if out.get("proba") is None else np.asarray(out["proba"])[inc],
               "dual": None if out.get("dual") is None else {k: np.asarray(v)[inc] for k, v in out["dual"].items()}}
        raw[w] = evaluate_outputs(task_type, model_config, y_true, sub, stats) if inc.any() else None
    skipped = np.asarray(entry.get("skipped", []))
    reasons = pd.Series([entry["asm"].reason[r] for r in skipped]).value_counts().to_dict() if len(skipped) else {}
    matured_le = rows.le[pts[inc]]
    stats_out = {
        "n_points": int(len(pts) + len(skipped)), "n_predicted": int(len(pts)), "n_included": int(inc.sum()),
        "n_unmatured": int((~inc).sum()), "n_skipped": int(len(skipped)),
        "skipped_reasons": {str(k): int(v) for k, v in reasons.items()},
        "cutoff_ts": str(matured_le.max())[:19] if len(matured_le) else None,
        "baseline_stats": "train_stats" if stats else "unavailable",
    }
    return raw, stats_out


def holdout_evaluations(node_id: str, raw: dict, *, specs: dict | None, phase, period: dict, stats_out: dict) -> list:
    """與訓練時 evaluations 相同格式；dataset.split=holdout、point＝best／last 權重。"""
    heads = {h["key"]: h for h in (specs or {}).get("heads", [])}
    dataset = {"split": "holdout", "label": SPLIT_LABEL["holdout"], "selection": "holdout", "phase": phase,
               "n_samples": stats_out["n_included"], "n_unmatured": stats_out["n_unmatured"],
               "n_skipped": stats_out["n_skipped"], "task_data_range": period,
               "description": (f"測試期間 {period.get('start')}～{period.get('end')}：預測 {stats_out['n_predicted']} 筆，"
                               f"其中 {stats_out['n_included']} 筆答案已成熟納入評估（成熟截止 {stats_out['cutoff_ts']}），"
                               f"{stats_out['n_unmatured']} 筆未成熟不評估；輸入不足未預測 {stats_out['n_skipped']} 筆")}
    out = []
    for kind in ("best", "last"):
        report = raw.get(kind)
        if not report:
            continue
        point = {"kind": kind, "weights": kind, "label": "最佳權重（best）" if kind == "best" else "最後權重（last）"}
        parts = [(h, report[h]) for h in heads if h != "default" and isinstance(report.get(h), dict)]
        for head, rep in parts or [("default", report)]:
            metrics, unavailable, baselines = _report_to_metrics(rep)
            out.append({"id": f"{node_id}/{head}/holdout/{kind}", "model_node": node_id, "head": head,
                        "head_label": heads.get(head, {}).get("label", head), "task": heads.get(head, {}).get("task"),
                        "dataset": dataset, "point": point, "metrics": metrics, "unavailable": unavailable,
                        "baselines": baselines})
    return out
