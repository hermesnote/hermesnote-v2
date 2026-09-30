"""評估清單（`evaluations`）與指標描述（`metric_specs`）：新任務完成時由 worker 產生並保存；
舊紀錄（只有 `evaluation: {best, last}`）在 API 讀取時用同一套函式即時轉換，不改寫資料庫。

每筆評估紀錄標明比較條件：
  model_node／head        哪個模型節點、哪個輸出頭（default／regression／direction）
  dataset                 split（val；未來 holdout／train）、切分方式、Phase、任務資料範圍、樣本數與說明
  point                   評估時點（best／last；未來 checkpoint），對應輪次、權重版本、best 的挑選依據
  metrics                 指標值（鍵＝metrics.py 登記的指標；形狀依定義的 shape）
  unavailable             算不出來的指標與原因
  baselines               適用的簡單基準（方法、數值）

本模組只做資料重組與由混淆矩陣推算 accuracy，不重新計算其他指標；數值與原始報告逐一相同。
"""

from training.registry import labeling_rules
from training.registry import architectures

ROUND_LABEL = {"epoch": "Epoch", "boosting_round": "Boosting 輪次"}
HEAD_LABEL = {"default": "輸出", "regression": "回歸頭", "direction": "方向頭"}
SPLIT_LABEL = {"val": "驗證集", "holdout": "Holdout", "train": "訓練集"}

_CLS_SCALARS = ["balanced_accuracy", "macro_f1", "roc_auc", "average_precision"]
_REG_SCALARS = ["mse", "rmse", "mae", "r2"]
_BASELINES = {
    "baseline_majority_class": ("majority_class", "訓練集多數類別", ["accuracy", "balanced_accuracy", "macro_f1"]),
    "baseline_mean": ("train_mean", "預測訓練集平均", ["mse", "rmse", "mae", "r2"]),
    "baseline_zero": ("zero", "預測零", ["mse", "rmse", "mae", "r2"]),
}


def _task_type(label_node: dict) -> str | None:
    return labeling_rules.LABELING_RULE_TASK_TYPE.get(label_node.get("labeling_rule", "fixed_threshold"))


def _accuracy_from_matrix(matrix) -> float | None:
    total = sum(sum(row) for row in matrix)
    return sum(matrix[i][i] for i in range(len(matrix))) / total if total else None


def _report_to_metrics(report: dict) -> tuple[dict, dict, list]:
    """單一分類或回歸報告 → (metrics, unavailable, baselines)。"""
    metrics, unavailable, baselines = {}, {}, []
    if "confusion_matrix" in report:
        matrix = report["confusion_matrix"]
        labels = report.get("confusion_matrix_labels") or list(range(len(matrix)))
        metrics["accuracy"] = _accuracy_from_matrix(matrix)
        for k in _CLS_SCALARS:
            if k in report:
                metrics[k] = report[k]
        metrics["per_class"] = report.get("per_class")
        metrics["class_distribution"] = report.get("class_distribution")
        metrics["confusion_matrix"] = {"labels": labels, "values": matrix, "row_axis": "實際", "col_axis": "預測"}
    else:
        for k in _REG_SCALARS:
            if k in report:
                metrics[k] = report[k]
    for k, v in report.items():
        if k.endswith("_unavailable_reason"):
            name = k[: -len("_unavailable_reason")]
            if name in _BASELINES:
                continue
            unavailable[name] = v
    for raw_key, (key, label, keys) in _BASELINES.items():
        if raw_key not in report:
            continue
        b = report[raw_key]
        if b is None:
            baselines.append({"key": key, "label": label, "available": False,
                              "reason": report.get(f"{raw_key}_unavailable_reason")})
            continue
        entry = {"key": key, "label": label, "available": True, "method": b.get("method"),
                 "metrics": {k: b[k] for k in keys if k in b}}
        if "class" in b:
            entry["detail"] = {"class": b["class"]}
        baselines.append(entry)
    return metrics, unavailable, baselines


def _dataset(training_meta: dict, node: dict, phase, task_range: dict) -> dict:
    tm = training_meta or {}
    strategy = tm.get("split_strategy") or node.get("split_strategy", "random")
    val_ratio = tm.get("val_ratio", node.get("val_ratio", 0.2))
    n_val, n_train = tm.get("n_val"), tm.get("n_train")
    excluded = tm.get("n_excluded_boundary", 0) or 0
    n_samples = tm.get("n_samples") or ((n_val or 0) + (n_train or 0) + excluded or None)
    pct = f"{float(val_ratio) * 100:g}%"
    if strategy == "chronological":
        desc = (f"任務資料範圍內的樣本依時間排序，最後 {pct}（{n_val} 筆）為驗證集；"
                f"訓練 {n_train} 筆" + (f"，另排除 {excluded} 筆標籤跨越驗證起點的樣本" if excluded else ""))
    else:
        seed = tm.get("split_seed", 42)
        desc = (f"任務資料範圍內的 {n_samples} 個樣本隨機打亂（固定種子 {seed}），抽 {pct}（{n_val} 筆）為驗證集、"
                f"其餘 {n_train} 筆訓練；驗證樣本分散在整個期間，不是一段連續時間")
    return {"split": "val", "label": SPLIT_LABEL["val"], "selection": strategy, "val_ratio": val_ratio,
            "n_samples": n_val, "n_train": n_train, "n_total": n_samples, "n_excluded_boundary": excluded,
            "phase": phase, "task_data_range": task_range, "description": desc}


def _points(final_metrics: dict, round_unit: str) -> dict:
    fm = final_metrics or {}
    if round_unit == "boosting_round":
        best, runs = fm.get("best_round"), fm.get("rounds_run")
    else:
        best, runs = fm.get("best_epoch"), fm.get("epochs_run")
    unit = ROUND_LABEL.get(round_unit, round_unit)
    last = runs - 1 if isinstance(runs, int) else None
    # XGBoost 的 final_metrics 沒有 monitor_mode；它的監控指標（logloss／mlogloss／rmse）都是越低越好
    mode = fm.get("monitor_mode") or ("min" if fm.get("monitor") else None)
    selected_by = {"monitor": fm.get("monitor"), "mode": mode, "patience": fm.get("patience")}
    return {
        "best": {"kind": "best", "round": best, "round_unit": round_unit, "weights": "best",
                 "label": f"最佳（第 {best + 1} {unit}）" if isinstance(best, int) else "最佳",
                 "selected_by": selected_by},
        "last": {"kind": "last", "round": last, "round_unit": round_unit, "weights": "last",
                 "label": f"最後（第 {last + 1} {unit}）" if isinstance(last, int) else "最後"},
    }


def build_evaluations(node_id: str, raw: dict, *, node: dict, final_metrics: dict, training_meta: dict,
                      phase, task_range: dict, round_unit: str) -> list[dict]:
    """`raw`＝訓練回傳的 `{"best": 報告, "last": 報告}`（報告是分類、回歸或雙頭 {direction, regression}）。"""
    dataset = _dataset(training_meta, node, phase, task_range)
    points = _points(final_metrics, round_unit)
    records = []
    for kind in ("best", "last"):
        report = (raw or {}).get(kind)
        if not report:
            continue
        parts = ([("direction", report["direction"]), ("regression", report["regression"])]
                 if "direction" in report and "regression" in report else [("default", report)])
        for head, rep in parts:
            metrics, unavailable, baselines = _report_to_metrics(rep)
            records.append({
                "id": f"{node_id}/{head}/val/{kind}", "model_node": node_id, "head": head,
                "head_label": HEAD_LABEL.get(head, head),
                "task": "classification" if "confusion_matrix" in rep else "regression",
                "dataset": dataset, "point": points[kind],
                "metrics": metrics, "unavailable": unavailable, "baselines": baselines,
            })
    return records


def node_context(graph_spec: dict, node_id: str) -> tuple[dict, dict, str | None]:
    nodes = {n["id"]: n for n in graph_spec.get("nodes", [])}
    node = nodes[node_id]
    label_node = nodes.get(node.get("label"), {})
    return node, label_node, _task_type(label_node)


def summarize_node(node_id: str, result: dict, graph_spec: dict, phase) -> dict:
    """新任務：run_graph 的節點結果 → 保存用的 metric_specs＋evaluations（取代舊的 evaluation）。"""
    node, label_node, task_type = node_context(graph_spec, node_id)
    specs = architectures.metric_specs(node["key"], node, label_node, task_type)
    out = {"metric_specs": specs} if specs is not None else {}
    if result.get("evaluation") is not None:
        out["evaluations"] = build_evaluations(
            node_id, result["evaluation"], node=node, final_metrics=result.get("final_metrics"),
            training_meta=result.get("training_meta"), phase=phase,
            task_range={"start": graph_spec.get("start"), "end": graph_spec.get("end")},
            round_unit=(specs or {}).get("round_unit", "epoch"))
    return out


def enrich_job(job: dict) -> dict:
    """API 回應前：每個訓練任務的模型節點補上 metric_specs（進行中也有，依提交設定算）與 evaluations
    （新紀錄用保存的；舊紀錄由 evaluation.best／last 即時轉換）。不改寫資料庫，原有欄位原樣保留。"""
    if job.get("job_type") != "train":
        return job
    spec = job.get("graph_spec") or {}
    specs_by_node = {}
    for n in spec.get("nodes", []):
        if n.get("type") != "model":
            continue
        try:
            node, label_node, task_type = node_context(spec, n["id"])
            specs_by_node[n["id"]] = architectures.metric_specs(node["key"], node, label_node, task_type)
        except Exception:
            specs_by_node[n["id"]] = None  # 未登記的舊架構 key 等：沒有描述，前端退回通用顯示
    job["metric_specs"] = specs_by_node
    for node_id, r in (job.get("result") or {}).items():
        if not isinstance(r, dict):
            continue
        if r.get("metric_specs") is None and specs_by_node.get(node_id) is not None:
            r["metric_specs"] = specs_by_node[node_id]
        if "evaluations" not in r and r.get("evaluation") is not None and node_id in {n["id"] for n in spec.get("nodes", [])}:
            node, _, _ = node_context(spec, node_id)
            r["evaluations"] = build_evaluations(
                node_id, r["evaluation"], node=node, final_metrics=r.get("final_metrics"),
                training_meta=r.get("training_meta"), phase=job.get("phase"),
                task_range={"start": spec.get("start"), "end": spec.get("end")},
                round_unit=(r.get("metric_specs") or {}).get("round_unit", "epoch"))
    return job
