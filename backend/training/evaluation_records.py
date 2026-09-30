"""評估清單（`evaluations`）與指標描述（`metric_specs`）：新任務完成時由 worker 產生並保存；
舊紀錄（只有 `evaluation: {best, last}`）在 API 讀取時用同一套函式即時轉換，不改寫資料庫。

每筆評估紀錄標明比較條件：
  model_node／head        哪個模型節點、哪個輸出頭（metric_specs.heads 宣告；單頭為 default）
  dataset                 split（val；未來 holdout／train）、切分方式、Phase、任務資料範圍、樣本數與說明
  point                   評估時點（best／last；未來 checkpoint），對應輪次、權重版本、best 的挑選依據
  metrics                 指標值（鍵＝metrics.py 登記的指標；形狀依定義的 shape）
  unavailable             算不出來的指標與原因
  baselines               適用的簡單基準（方法、數值）

本模組只做資料重組（依指標登記的 shape 與資料契約）與登記的衍生指標推算（例：accuracy 由混淆矩陣
推算），不重新計算其他指標；數值與原始報告逐一相同。
"""

from training.registry import architectures, labeling_rules
from training.registry import metrics as metrics_registry

ROUND_LABEL = {"epoch": "Epoch", "boosting_round": "Boosting 輪次"}
SPLIT_LABEL = {"val": "驗證集", "holdout": "Holdout", "train": "訓練集"}

def _task_type(label_node: dict) -> str | None:
    return labeling_rules.LABELING_RULE_TASK_TYPE.get(label_node.get("labeling_rule", "fixed_threshold"))


def _metric_value(key: str, definition: dict, value, report: dict):
    """依定義的 shape 把報告裡的值轉成資料契約的形狀（見 registry/metrics.py）。"""
    if definition["shape"] == "matrix":
        labels = report.get(f"{key}_labels") or list(range(len(value)))
        axes = definition.get("axes") or {}
        return {"labels": labels, "values": value, "row_axis": axes.get("row", "列"), "col_axis": axes.get("col", "欄")}
    return value


def _report_to_metrics(report: dict) -> tuple[dict, dict, list]:
    """單一報告 → (metrics, unavailable, baselines)。不用固定清單：報告裡凡是已登記的指標都納入，
    形狀依定義的 shape；衍生指標依登記的 derive 推算；`baseline_*` 依 BASELINE_REGISTRY 命名。
    未登記的鍵不進評估清單（tests 核對 evaluation.py 產生的鍵都已登記）。"""
    registry = metrics_registry.METRIC_REGISTRY
    metrics, unavailable, baselines = {}, {}, []
    for k, v in report.items():
        if k in registry:
            metrics[k] = _metric_value(k, registry[k], v, report)
    for k, d in registry.items():
        derive = d.get("derive")
        if k not in metrics and derive and report.get(derive["from"]) is not None:
            metrics[k] = metrics_registry.DERIVATIONS[derive["method"]](report[derive["from"]])
    for k, v in report.items():
        if k.endswith("_unavailable_reason") and not k.startswith("baseline_"):
            unavailable[k[: -len("_unavailable_reason")]] = v
    for raw_key, b in report.items():
        if not raw_key.startswith("baseline_") or raw_key.endswith("_unavailable_reason"):
            continue
        name = raw_key[len("baseline_"):]
        meta = metrics_registry.BASELINE_REGISTRY.get(raw_key) or {"key": name, "label": name}
        if b is None:
            baselines.append({"key": meta["key"], "label": meta["label"], "available": False,
                              "reason": report.get(f"{raw_key}_unavailable_reason")})
            continue
        entry = {"key": meta["key"], "label": meta["label"], "available": True, "method": b.get("method"),
                 "metrics": {k: v for k, v in b.items() if k in registry}}
        detail = {k: v for k, v in b.items() if k not in registry and k != "method"}
        if detail:
            entry["detail"] = detail
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
                      phase, task_range: dict, specs: dict | None) -> list[dict]:
    """`raw`＝訓練回傳的 `{"best": 報告, "last": 報告}`。報告若含 metric_specs 宣告的輸出頭鍵（例：雙頭的
    `direction`／`regression`），逐頭拆成各自一筆；否則整份屬於 `default`。"""
    heads = {h["key"]: h for h in (specs or {}).get("heads", [])}
    round_unit = (specs or {}).get("round_unit", "epoch")
    dataset = _dataset(training_meta, node, phase, task_range)
    points = _points(final_metrics, round_unit)
    records = []
    for kind in ("best", "last"):
        report = (raw or {}).get(kind)
        if not report:
            continue
        head_parts = [(h, report[h]) for h in heads if h != "default" and isinstance(report.get(h), dict)]
        for head, rep in head_parts or [("default", report)]:
            metrics, unavailable, baselines = _report_to_metrics(rep)
            records.append({
                "id": f"{node_id}/{head}/val/{kind}", "model_node": node_id, "head": head,
                "head_label": heads.get(head, {}).get("label", head), "task": heads.get(head, {}).get("task"),
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
            task_range={"start": graph_spec.get("start"), "end": graph_spec.get("end")}, specs=specs)
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
                task_range={"start": spec.get("start"), "end": spec.get("end")}, specs=r.get("metric_specs"))
    return job
