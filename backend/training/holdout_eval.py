"""Phase 3（holdout）評估與階段門檻（M3）。

一致性：Phase 3 用的評估函式就是訓練時對驗證集用的同一組（training/evaluation.py 的
classification_evaluation／regression_evaluation），輸出頭拆法也跟訓練一樣（雙頭＝direction＋regression）。
基準需要的「訓練集統計」在訓練時存進產物（train_stats：類別計數、目標平均），Phase 3 據此重建，
不重新讀訓練資料。舊產物沒有 train_stats 時，基準標示無法計算的原因，其他指標照算。
"""

import numpy as np

from training.evaluation import classification_evaluation, regression_evaluation


def train_stats(task_type: str, y_train, n_classes: int | None = None, model_params: dict | None = None) -> dict:
    """訓練點的統計（Phase 3 基準用）：分類＝類別計數；回歸＝平均；雙頭另存方向事件計數。"""
    y = np.asarray(y_train)
    if task_type == "classification":
        k = int(n_classes or (int(y.max()) + 1 if len(y) else 2))
        return {"task_type": task_type, "n": int(len(y)), "class_counts": np.bincount(y.astype(int), minlength=k).tolist()}
    out = {"task_type": task_type, "n": int(len(y)), "mean": float(np.mean(y)) if len(y) else None}
    params = model_params or {}
    if params.get("heads") == "dual" and params.get("direction_rule"):
        from training.registry.target_transforms import apply_target_transform

        rule = {k: v for k, v in params["direction_rule"].items() if k != "threshold_unit"}
        cls = apply_target_transform("sign_threshold", y.astype(float), rule).astype(int)
        out["direction_counts"] = np.bincount(cls, minlength=2).tolist()
    return out


def _y_from_counts(counts):
    return None if counts is None else np.repeat(np.arange(len(counts)), counts)


def evaluate_outputs(task_type: str, model_config: dict, y_true, outputs: dict, stats: dict | None) -> dict:
    """outputs：{"pred": (n,), "proba": (n,k) 或 None, "dual": {"regression", "direction_probability"} 或 None}。
    回傳格式與訓練時 evaluation[best|last] 相同。"""
    stats = stats or {}
    y_true = np.asarray(y_true)
    if outputs.get("dual") is not None:
        rule = {k: v for k, v in (((model_config or {}).get("config") or {}).get("direction_rule") or {}).items()
                if k != "threshold_unit"}
        from training.registry.target_transforms import apply_target_transform

        cls = apply_target_transform("sign_threshold", y_true.astype(float), rule).astype(int)
        p = np.asarray(outputs["dual"]["direction_probability"], dtype=float)
        mean = stats.get("mean")
        return {
            "direction": classification_evaluation(cls, (p >= 0.5).astype(int), y_proba=np.stack([1 - p, p], axis=1),
                                                   n_classes=2, y_train=_y_from_counts(stats.get("direction_counts"))),
            "regression": regression_evaluation(y_true.astype(float), outputs["dual"]["regression"],
                                                y_train=None if mean is None else np.array([mean])),
        }
    if task_type == "classification":
        n_classes = (model_config or {}).get("n_classes") or (outputs["proba"].shape[1] if outputs.get("proba") is not None else 2)
        return classification_evaluation(y_true.astype(int), np.asarray(outputs["pred"]).astype(int),
                                         y_proba=outputs.get("proba"), n_classes=int(n_classes),
                                         y_train=_y_from_counts(stats.get("class_counts")))
    mean = stats.get("mean")
    return regression_evaluation(y_true.astype(float), np.asarray(outputs["pred"], dtype=float),
                                 y_train=None if mean is None else np.array([mean]))


def predict_outputs(loaded: dict, X, task_type: str, dual: bool = False, chunk: int = 5000) -> dict:
    """對 X（PackedWindows 或陣列）分批預測，回傳 evaluate_outputs 需要的格式。"""
    preds, probas, regs, dirs = [], [], [], []
    for s in range(0, len(X), chunk):
        xb = np.asarray(X[np.arange(s, min(s + chunk, len(X)))]) if hasattr(X, "subset") else np.asarray(X[s:s + chunk])
        if dual:
            o = loaded["predict_outputs"](xb)
            regs.append(np.asarray(o["regression"], dtype=float).reshape(-1))
            dirs.append(np.asarray(o["direction_probability"], dtype=float).reshape(-1))
        elif task_type == "classification" and "predict_proba" in loaded:
            p = np.asarray(loaded["predict_proba"](xb), dtype=float)
            probas.append(p)
            preds.append(p.argmax(axis=1))
        else:
            preds.append(np.asarray(loaded["predict"](xb), dtype=float).reshape(-1))
    if dual:
        r = np.concatenate(regs) if regs else np.zeros(0)
        d = np.concatenate(dirs) if dirs else np.zeros(0)
        return {"pred": r, "proba": None, "dual": {"regression": r, "direction_probability": d}}
    return {"pred": np.concatenate(preds) if preds else np.zeros(0),
            "proba": np.concatenate(probas) if probas else None, "dual": None}


# ── 階段門檻：只提示、記錄，研究者確認後可放行 ─────────────────────────────────────
OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}


def gate_issues(gates) -> list[str]:
    if gates is None:
        return []
    if not isinstance(gates, list):
        return ["gates 必須是陣列"]
    out = []
    for i, g in enumerate(gates):
        if not isinstance(g, dict):
            out.append(f"gates[{i}] 必須是物件")
            continue
        if g.get("op") not in OPS:
            out.append(f"gates[{i}].op 必須是 {sorted(OPS)} 之一")
        if g.get("weights", "best") not in ("best", "last"):
            out.append(f"gates[{i}].weights 必須是 best 或 last")
        if g.get("phase") not in (2, 3):
            out.append(f"gates[{i}].phase 必須是 2（進入 Phase 2 前檢查 Phase 1）或 3（進入 Phase 3 前檢查 Phase 2）")
        if not isinstance(g.get("metric"), str):
            out.append(f"gates[{i}].metric 必須是指標鍵（例如 macro_f1、rmse）")
        if isinstance(g.get("value"), bool) or not isinstance(g.get("value"), (int, float)):
            out.append(f"gates[{i}].value 必須是數值")
    return out


def check_gates(gates: list, parent_result: dict, node_id: str, entering_phase: int) -> dict:
    """依母任務的 evaluations 檢查門檻。回傳 {passed, configured, conditions[...]}。"""
    conds = [g for g in (gates or []) if g.get("phase") == entering_phase]
    evals = ((parent_result or {}).get(node_id) or {}).get("evaluations") or []
    out = []
    for g in conds:
        kind = g.get("weights", "best")
        head = g.get("head", "default")
        rec = next((e for e in evals if e.get("head") == head and (e.get("point") or {}).get("kind") == kind), None)
        actual = None if rec is None else (rec.get("metrics") or {}).get(g["metric"])
        ok = isinstance(actual, (int, float)) and OPS[g["op"]](float(actual), float(g["value"]))
        out.append({**g, "weights": kind, "head": head, "actual": actual, "ok": bool(ok),
                    "reason": None if rec is not None and actual is not None else "母任務沒有這個指標的評估紀錄"})
    return {"configured": bool(conds), "passed": all(c["ok"] for c in out), "conditions": out,
            "entering_phase": entering_phase}
