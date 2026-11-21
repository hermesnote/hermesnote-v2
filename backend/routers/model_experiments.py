"""實驗層 API（M2～M7）：實驗、組別、樣本框架、三階段執行、階段門檻、逐筆結果、Phase 3 評估、配對比較、
holdout 使用紀錄、串接核對、暖機建議，以及 v2 的元件登記表。詳見 docs/agent-api/training/api.md。

實驗＝「必須一致」的設定（期間、holdout、Label、切分種子、val_ratio、對齊規則、暖機 ε、排除警示門檻、串接預設、
階段門檻）；組別＝「可作為變因」的設定（特徵、轉換、暖機、分支、模型、模型種子清單）。組別的每個任務都用
mode=v2 的圖，並帶 frame_ref 指向實驗的樣本框架（第一次執行時由 worker 產生並保存，之後核對雜湊）。
"""

from datetime import date, datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from auth import principal_id, require_model_writer
from training import experiment_store as es
from training import job_store
from training.graph_refs import GraphValidationError, validate_graph_spec
from training.holdout_eval import check_gates, gate_issues
from training.phases import build_phase2_graph_spec, final_model_node_id
from training.provenance import PIPELINE_VERSION, period_warnings

router = APIRouter(prefix="/api/model", tags=["model_experiments"])

LABEL_ID = "label"
SHARED_KEYS = ("split_seed", "val_ratio", "alignment", "warmup", "exclusion_warn_ratio", "cascade")
MUTABLE_KEYS = ("alignment", "gates", "exclusion_warn_ratio", "warmup", "cascade")


def _bad(msg, code=400):
    raise HTTPException(status_code=code, detail=msg)


def _date(v, name):
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        _bad(f"{name} 必須是 YYYY-MM-DD")


def _settings_issues(s: dict) -> list[str]:
    out = []
    if not isinstance(s, dict):
        return ["settings 必須是物件"]
    for k in ("start", "end"):
        if k not in s:
            out.append(f"settings.{k} 必填（YYYY-MM-DD）")
    if out:
        return out
    if _date(s["start"], "settings.start") >= _date(s["end"], "settings.end"):
        out.append("settings.start 必須早於 settings.end")
    h = s.get("holdout")
    if h is not None:
        if not isinstance(h, dict) or "start" not in h or "end" not in h:
            out.append("settings.holdout 必須是 {start, end}")
        elif _date(h["start"], "holdout.start") < _date(s["end"], "settings.end"):
            out.append("settings.holdout.start 不能早於 settings.end（測試期間不能與訓練期間重疊）")
        elif _date(h["start"], "holdout.start") >= _date(h["end"], "holdout.end"):
            out.append("settings.holdout.start 必須早於 holdout.end")
    lab = s.get("label")
    if not isinstance(lab, dict) or not isinstance(lab.get("timeframe"), str) or not isinstance(lab.get("outcome"), str):
        out.append("settings.label 必須是 {timeframe, outcome, labeling_rule, params}")
    out += gate_issues(s.get("gates"))
    return out


def _label_node(s: dict) -> dict:
    lab = s["label"]
    return {"id": LABEL_ID, "type": "label", "timeframe": lab["timeframe"], "outcome": lab["outcome"],
            "labeling_rule": lab.get("labeling_rule", "fixed_threshold"), "params": lab.get("params") or {}}


def arm_graph(exp: dict, arm_nodes: list, *, seed, arm_id=None, frame=None) -> dict:
    """組別節點＋實驗 Label → v2 圖。模型種子覆寫每個 Model Node 的 params.seed。"""
    s = exp["settings"]
    nodes = []
    for n in arm_nodes:
        n = {**n}
        if n.get("type") == "label" and n.get("id") == LABEL_ID:
            continue
        if n.get("type") == "model":
            n.pop("split_strategy", None)
            if seed is not None:
                n["params"] = {**(n.get("params") or {}), "seed": int(seed)}
        nodes.append(n)
    nodes.append(_label_node(s))
    g = {"mode": "v2", "start": s["start"], "end": s["end"], "settings": {k: s[k] for k in SHARED_KEYS if k in s},
         "nodes": nodes,
         "experiment": {"experiment_id": exp["id"], "arm_id": arm_id, "model_seed": seed}}
    if frame is not None:
        g["frame_ref"] = {"frame_id": frame["id"], "phase": frame["phase"], "version": frame["version"]}
    return g


def _arm_issues(exp: dict, nodes: list, seeds: list) -> list[str]:
    if not isinstance(nodes, list) or not nodes:
        return ["nodes 必須是非空陣列"]
    if not isinstance(seeds, list) or not seeds or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds):
        return ["model_seeds 必須是 ≥ 0 整數的非空陣列（次數由研究設計決定）"]
    g = arm_graph(exp, nodes, seed=seeds[0])
    try:
        validate_graph_spec(g)
    except GraphValidationError as e:
        return e.issues
    try:
        fid = final_model_node_id(g["nodes"])
    except ValueError as e:
        return [str(e)]
    final = next(n for n in g["nodes"] if n["id"] == fid)
    if final.get("label") != LABEL_ID:
        return [f"最終模型 {fid} 的 label 必須是實驗的 Label（\"{LABEL_ID}\"）——答案定義屬於實驗層，組別之間必須一致"]
    return []


class ExperimentRequest(BaseModel):
    name: str
    description: str | None = None
    settings: dict


@router.post("/experiments")
async def create_experiment(body: ExperimentRequest, principal: dict = Depends(require_model_writer("model:experiment"))):
    issues = _settings_issues(body.settings)
    if not issues:
        probe = {"id": "probe", "settings": body.settings}
        g = arm_graph(probe, [{"id": "f", "type": "feature", "key": "ohlcv", "timeframe": body.settings["label"]["timeframe"]},
                              {"id": "m", "type": "model", "key": "xgboost", "inputs": ["f"], "label": LABEL_ID,
                               "params": {"n_estimators": 1, "max_depth": 1, "learning_rate": 0.1,
                                          "early_stopping": {"patience": 0}, "seed": 0}}], seed=0)
        try:
            validate_graph_spec(g)
        except GraphValidationError as e:
            issues = [i for i in e.issues if not i.startswith("Feature 節點 f") and not i.startswith("Model 節點 m")] or e.issues
    if issues:
        _bad("；".join(issues))
    exp_id = await es.create_experiment(body.name, body.description, body.settings, created_by=principal_id(principal))
    warnings = period_warnings(body.settings["start"], body.settings["end"])
    if body.settings.get("holdout"):
        warnings += period_warnings(body.settings["holdout"]["start"], body.settings["holdout"]["end"])
    return {"experiment_id": exp_id, "warnings": warnings, "pipeline_version": PIPELINE_VERSION}


@router.get("/experiments")
async def list_experiments():
    return await es.list_experiments()


@router.get("/experiments/{exp_id}")
async def get_experiment(exp_id: str):
    exp = await es.get_experiment(exp_id)
    if exp is None:
        _bad("找不到這個實驗", 404)
    return exp


class SettingsPatch(BaseModel):
    settings: dict
    note: str | None = None


@router.patch("/experiments/{exp_id}/settings")
async def patch_settings(exp_id: str, body: SettingsPatch, principal: dict = Depends(require_model_writer("model:experiment"))):
    """只允許改對齊規則、門檻、排除警示門檻、暖機 ε、串接預設（影響之後建立的任務；每個任務都保存當時的設定快照，
    比較時設定不同會被拒絕）。期間、Label、切分屬於框架定義，不能改——要改請建立新實驗或新版本框架。"""
    exp = await es.get_experiment(exp_id)
    if exp is None:
        _bad("找不到這個實驗", 404)
    bad = sorted(set(body.settings) - set(MUTABLE_KEYS))
    if bad:
        _bad(f"這些設定不能在建立後修改：{bad}（可修改：{list(MUTABLE_KEYS)}）")
    new = {**exp["settings"], **body.settings}
    hist = list(new.get("history") or [])
    hist.append({"at": datetime.now(timezone.utc).isoformat(), "changed": sorted(body.settings), "note": body.note,
                 "by": principal_id(principal)})
    new["history"] = hist
    issues = _settings_issues(new)
    if not issues:
        probe = {"id": exp_id, "settings": new}
        g = arm_graph(probe, [{"id": "f", "type": "feature", "key": "ohlcv", "timeframe": new["label"]["timeframe"]},
                              {"id": "m", "type": "model", "key": "xgboost", "inputs": ["f"], "label": LABEL_ID,
                               "params": {"n_estimators": 1, "max_depth": 1, "learning_rate": 0.1,
                                          "early_stopping": {"patience": 0}, "seed": 0}}], seed=0)
        try:
            validate_graph_spec(g)
        except GraphValidationError as e:
            issues = e.issues
    if issues:
        _bad("；".join(issues))
    await es.update_settings(exp_id, new)
    return {"ok": True, "settings": new}


class ArmRequest(BaseModel):
    name: str
    nodes: list[dict]
    model_seeds: list[int] = [42]


@router.post("/experiments/{exp_id}/arms")
async def create_arm(exp_id: str, body: ArmRequest, principal: dict = Depends(require_model_writer("model:experiment"))):
    exp = await es.get_experiment(exp_id)
    if exp is None:
        _bad("找不到這個實驗", 404)
    issues = _arm_issues(exp, body.nodes, body.model_seeds)
    if issues:
        _bad("；".join(issues))
    return {"arm_id": await es.create_arm(exp_id, body.name, {"nodes": body.nodes}, body.model_seeds)}


class FrameRequest(BaseModel):
    phase: int
    note: str | None = None


@router.post("/experiments/{exp_id}/frames")
async def new_frame(exp_id: str, body: FrameRequest, principal: dict = Depends(require_model_writer("model:experiment"))):
    """建立新版本的樣本框架（例如新組別需要更長歷史、或資料更新）：舊結果保留，系統拒絕跨版本比較。"""
    if body.phase not in (1, 2, 3):
        _bad("phase 必須是 1、2 或 3")
    if await es.get_experiment(exp_id) is None:
        _bad("找不到這個實驗", 404)
    return await es.new_frame_version(exp_id, body.phase, body.note)


@router.get("/frames/{frame_id}/points")
async def frame_points(frame_id: str, status: str | None = None, cursor: str | None = None, limit: int = 1000):
    return await es.frame_points(frame_id, status, cursor, min(limit, 5000))


class GateOverride(BaseModel):
    reason: str
    # 只當備註；實際確認人由後端從驗證身分記錄（override.confirmed_by），不採用呼叫端送的文字
    confirmed_by: str | None = None


class RunRequest(BaseModel):
    phase: int
    parent_job_ids: list[str] | None = None
    gate_override: GateOverride | None = None


async def _gate(exp, parent, entering, override, who=None):
    node_id = final_model_node_id(parent["graph_spec"]["nodes"])
    gc = check_gates(exp["settings"].get("gates"), parent.get("result"), node_id, entering)
    if gc["configured"] and not gc["passed"]:
        if override is None:
            raise HTTPException(status_code=409, detail={
                "message": f"母任務 {parent['job_id']} 未通過進入 Phase {entering} 的門檻；若仍要繼續，請附 gate_override（reason；確認人由後端依登入身分記錄）",
                "gate_check": gc})
        gc["override"] = {"reason": override.reason, "confirmed_by": who or "unauthenticated",
                          "note_confirmed_by": override.confirmed_by,
                          "at": datetime.now(timezone.utc).isoformat()}
    gc["status"] = "passed" if gc["passed"] else ("overridden" if gc.get("override") else "not_configured")
    if not gc["configured"]:
        gc["status"] = "not_configured"
    return gc


@router.post("/experiments/{exp_id}/arms/{arm_id}/run")
async def run_arm(exp_id: str, arm_id: str, body: RunRequest, principal: dict = Depends(require_model_writer("model:experiment"))):
    exp = await es.get_experiment(exp_id)
    arm = await es.get_arm(arm_id)
    if exp is None or arm is None or arm["experiment_id"] != exp_id:
        _bad("找不到這個實驗或組別", 404)
    s = exp["settings"]
    jobs = [j for j in exp["jobs"] if j.get("arm_id") == arm_id]
    created = []
    if body.phase == 1:
        frame = await es.latest_frame(exp_id, 1)
        for seed in arm["model_seeds"]:
            g = arm_graph(exp, arm["spec"]["nodes"], seed=seed, arm_id=arm_id, frame=frame)
            g["nodes"] = [{**n, "split_strategy": "random"} if n["type"] == "model" else n for n in g["nodes"]]
            validate_graph_spec(g)
            jid = await job_store.create_job(g, phase=1, warnings=period_warnings(s["start"], s["end"]),
                                             experiment_id=exp_id, arm_id=arm_id, frame_id=frame["id"],
                                             model_seed=seed, split_seed=s.get("split_seed", 42),
                                             requested_by=principal_id(principal))
            created.append({"job_id": jid, "model_seed": seed, "phase": 1})
        return {"jobs": created, "frame": frame}
    if body.phase not in (2, 3):
        _bad("phase 必須是 1、2 或 3")
    want = body.phase - 1
    parents = body.parent_job_ids or [j["id"] for j in jobs if j.get("phase") == want and j["status"] == "done"
                                      and j.get("job_type", "train") == "train"]
    if not parents:
        _bad(f"這個組別沒有已完成的 Phase {want} 任務")
    frame = await es.latest_frame(exp_id, body.phase)
    for pid in parents:
        parent = await job_store.get_job(pid)
        if parent is None or parent.get("arm_id") != arm_id or parent.get("phase") != want or parent["status"] != "done":
            _bad(f"任務 {pid} 不是這個組別已完成的 Phase {want} 任務")
        gc = await _gate(exp, parent, body.phase, body.gate_override, principal_id(principal))
        seed = parent.get("model_seed")
        if body.phase == 2:
            g = build_phase2_graph_spec(parent["graph_spec"])
            g["frame_ref"] = {"frame_id": frame["id"], "phase": 2, "version": frame["version"]}
            g["settings"] = {k: s[k] for k in SHARED_KEYS if k in s}
            validate_graph_spec(g)
            jid = await job_store.create_job(g, phase=2, parent_job_id=pid, warnings=period_warnings(s["start"], s["end"]),
                                             experiment_id=exp_id, arm_id=arm_id, frame_id=frame["id"], model_seed=seed,
                                             split_seed=s.get("split_seed", 42), gate_check=gc,
                                             requested_by=principal_id(principal))
        else:
            h = s.get("holdout")
            if not h:
                _bad("實驗沒有設定 holdout 期間（settings.holdout），無法執行 Phase 3")
            node_id = final_model_node_id(parent["graph_spec"]["nodes"])
            spec = {"target_job_id": pid, "target_node_id": node_id, "start": h["start"], "end": h["end"],
                    "use_weights": "best", "experiment": {"experiment_id": exp_id, "arm_id": arm_id, "model_seed": seed},
                    "frame_ref": {"frame_id": frame["id"], "phase": 3, "version": frame["version"]}}
            jid = await job_store.create_job(spec, job_type="infer", phase=3, parent_job_id=pid,
                                             warnings=period_warnings(h["start"], h["end"]), experiment_id=exp_id,
                                             arm_id=arm_id, frame_id=frame["id"], model_seed=seed, gate_check=gc,
                                             requested_by=principal_id(principal))
            await es.write_holdout_usage("inference", job_id=jid, parent_job_id=pid, experiment_id=exp_id, arm_id=arm_id,
                                         start=h["start"], end=h["end"],
                                         detail={"weights": ["best", "last"], "requested_by": principal_id(principal)})
        created.append({"job_id": jid, "parent_job_id": pid, "model_seed": seed, "phase": body.phase, "gate_check": gc})
    return {"jobs": created, "frame": frame}


@router.get("/experiments/{exp_id}/holdout_usage")
async def holdout_usage(exp_id: str):
    return await es.holdout_usage(exp_id=exp_id)


def _paired(a: list, b: list, task_type: str) -> dict:
    """同一批時間點上兩個任務的配對比較。分類：各自正確率與 McNemar 精確檢定；回歸：各自 MSE 與平方誤差差的配對 t。"""
    import math

    ma = {r["sample_ts"]: r for r in a}
    mb = {r["sample_ts"]: r for r in b}
    common = sorted(set(ma) & set(mb))
    out = {"n_a": len(a), "n_b": len(b), "n_common": len(common)}
    if not common:
        return out
    ya = np.array([ma[t]["y_true"] for t in common], dtype=float)
    yb = np.array([mb[t]["y_true"] for t in common], dtype=float)
    out["label_mismatch"] = int((~np.isclose(ya, yb, equal_nan=True)).sum())
    pa = np.array([ma[t]["y_pred"] for t in common], dtype=float)
    pb = np.array([mb[t]["y_pred"] for t in common], dtype=float)
    if task_type == "classification":
        ca, cb = pa == ya, pb == ya
        n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
        n = n01 + n10
        p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n)
        out.update({"accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()),
                    "only_a_correct": n10, "only_b_correct": n01, "mcnemar_exact_p": p})
    else:
        ea, eb = (pa - ya) ** 2, (pb - ya) ** 2
        d = ea - eb
        sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        t = float(d.mean() / (sd / math.sqrt(len(d)))) if len(d) > 1 and sd > 0 else None
        out.update({"mse_a": float(ea.mean()), "mse_b": float(eb.mean()), "mean_sq_err_diff": float(d.mean()),
                    "paired_t": t, "p_normal_approx": None if t is None else float(math.erfc(abs(t) / math.sqrt(2)))})
    return out


@router.get("/experiments/{exp_id}/compare")
async def compare(exp_id: str, job_a: str, job_b: str, weights: str = "best"):
    """兩個任務（同一實驗、同一 Phase）在共同時間點上的配對比較。框架雜湊、流程版本、對齊設定不同時拒絕。"""
    if weights not in ("best", "last"):
        _bad("weights 必須是 best 或 last")
    ja, jb = await job_store.get_job(job_a), await job_store.get_job(job_b)
    if ja is None or jb is None:
        _bad("找不到指定的任務", 404)
    for j in (ja, jb):
        if j.get("experiment_id") != exp_id or j["status"] != "done":
            _bad(f"任務 {j['job_id']} 不是這個實驗已完成的任務")
    if ja.get("phase") != jb.get("phase"):
        _bad("兩個任務的 Phase 不同，不能比較")
    if ja.get("pipeline_version") != jb.get("pipeline_version"):
        _bad(f"流程版本不同（{ja.get('pipeline_version')} vs {jb.get('pipeline_version')}），系統拒絕跨版本比較")
    if ja.get("frame_id") != jb.get("frame_id"):
        _bad("兩個任務使用不同的樣本框架版本，系統拒絕跨版本比較")
    sa = (ja["graph_spec"].get("settings") or {}).get("alignment")
    sb = (jb["graph_spec"].get("settings") or {}).get("alignment") if jb.get("job_type") == "train" else None
    if ja.get("job_type") == "train" and sa != sb:
        _bad("兩個任務的對齊設定（alignment）不同，不能直接比較")
    dataset = "holdout" if ja.get("phase") == 3 else "val"
    ra = await es.records_for_compare(job_a, dataset, weights)
    rb = await es.records_for_compare(job_b, dataset, weights)
    hashes = {r["frame_hash"] for r in ra + rb if r["frame_hash"]}
    if len(hashes) > 1:
        _bad("逐筆結果的樣本框架雜湊不一致，拒絕比較")
    exp = await es.get_experiment(exp_id)
    from training.registry.labeling_rules import LABELING_RULE_TASK_TYPE

    task_type = LABELING_RULE_TASK_TYPE.get(exp["settings"]["label"].get("labeling_rule", "fixed_threshold"))
    res = _paired(ra, rb, task_type)
    if dataset == "holdout":
        await es.write_holdout_usage("comparison", experiment_id=exp_id, job_id=job_a, parent_job_id=job_b,
                                     detail={"weights": weights, "n_common": res["n_common"]})
    return {"dataset": dataset, "weights": weights, "frame_hash": next(iter(hashes), None), "task_type": task_type, **res}


# ── 任務層：逐筆結果、評估、串接核對 ─────────────────────────────────────
@router.get("/jobs/{job_id}/records")
async def job_records(job_id: str, node_id: str | None = None, dataset: str | None = None, weights: str | None = None,
                      cursor: int | None = None, limit: int = 1000):
    return await es.list_records(job_id, node_id, dataset, weights, cursor, min(limit, 5000))


@router.get("/jobs/{job_id}/evaluations")
async def job_evaluations(job_id: str):
    return await es.list_evaluations(job_id)


@router.get("/jobs/{job_id}/upstream_folds")
async def job_upstream_folds(job_id: str):
    return await es.upstream_folds(job_id)


@router.get("/jobs/{job_id}/holdout_usage")
async def job_holdout_usage(job_id: str):
    return await es.holdout_usage(job_id=job_id)


@router.get("/jobs/{job_id}/cascade_audit")
async def cascade_audit(job_id: str, format: str = Query("json", pattern="^(json|csv)$"), sample_n: int = 5):
    from training.cascade_audit import audit, violations_csv

    job = await job_store.get_job(job_id)
    if job is None:
        _bad("找不到這個任務", 404)
    data = await es.audit_inputs(job_id)
    strategy = next((n.get("split_strategy") for n in job["graph_spec"].get("nodes", []) if n.get("type") == "model"), None)
    rep = audit(data["records"], data["folds"], split_strategy=strategy, sample_n=min(sample_n, 50))
    if format == "csv":
        return PlainTextResponse(violations_csv(rep), media_type="text/csv")
    return rep


# ── 暖機建議（實測）與 v2 元件登記表 ───────────────────────────────────────
class WarmupSuggestRequest(BaseModel):
    feature: dict  # Feature Node（key／params／timeframe）
    epsilon: float = 0.01
    end: str | None = None  # 取樣截止日（預設 2026-01-01）
    n_bars: int = 3000
    tolerance: float | None = None  # 預設＝epsilon × 特徵標準差


@router.post("/warmup/suggest")
async def warmup_suggest(body: WarmupSuggestRequest, principal: dict = Depends(require_model_writer("model:experiment"))):
    """程式依記憶類型給的基本值，加上以實際資料測試的建議值（複合型指標用）：同一段資料從不同起點開始算，
    找出差異落在容許值內所需的根數。只是建議，研究者確認後填進 warmup.bars。"""
    from datetime import timedelta

    from services.quotes import fetch_ohlcv_dataframe
    from training import alignment as al
    from training.registry import warmup as wu
    from training.registry.features import compute_feature

    f = body.feature
    try:
        base = wu.suggest(f.get("key"), f.get("params"), body.epsilon)
    except Exception as e:  # noqa: BLE001
        _bad(f"無法判定這個特徵的暖機：{e}")
    n = max(500, min(int(body.n_bars), 20000))
    end = _date(body.end or "2026-01-01", "end")
    sym, tf = f["timeframe"].split("_", 1)
    df = await fetch_ohlcv_dataframe(sym, tf, end - timedelta(days=1), end, lookback_bars=n)
    T = al.TfData(f["timeframe"], df)
    full = np.asarray(compute_feature(f["key"], T.ohlcv, f.get("params")), dtype=float)
    full = full.reshape(-1, 1) if full.ndim == 1 else full
    scale = np.nanstd(full, axis=0)
    tol = body.tolerance if body.tolerance is not None else float(body.epsilon * np.nanmax(scale))
    offsets = [int(x) for x in np.linspace(n // 10, n // 2, 4)]
    parts = {}
    for s in offsets:
        p = np.asarray(compute_feature(f["key"], T.ohlcv.iloc[s:], f.get("params")), dtype=float)
        parts[s] = p.reshape(-1, 1) if p.ndim == 1 else p
    emp = wu.empirical_suggestion(full, parts, tol)
    return {"program": base, "empirical": emp, "sample": {"timeframe": f["timeframe"], "n_bars": len(T), "end": str(end)},
            "note": "實測值只是建議，請研究者確認後設定 warmup.bars（不能小於 program.base_bars）"}


@router.get("/registry/v2")
def registry_v2():
    """v2 圖可用的元件：記憶類型、轉換、編碼器、正規化、上游策略、對齊預設與彙總方式、排除原因代碼。"""
    from training import alignment as al
    from training import cascade as cc
    from training import frame as fr
    from training import v2_inputs as vi
    from training.registry import transforms as tr
    from training.registry import warmup as wu

    return {"pipeline_version": PIPELINE_VERSION, "memory_types": wu.MEMORY_TYPES, "transforms": tr.list_available(),
            "encoders": vi.ENCODERS, "default_encoder": vi.DEFAULT_ENCODER, "normalizations": vi.NORMALIZATIONS,
            "upstream_policies": cc.POLICIES, "upstream_finals": cc.FINALS, "default_upstream_policy": cc.DEFAULT_POLICY,
            "alignment_defaults": al.DEFAULT_ALIGNMENT, "aggregations": list(al.AGGREGATIONS),
            "exclusion_reasons": fr.REASONS, "warmup_epsilon": {"default": wu.DEFAULT_EPSILON, "range": wu.EPSILON_RANGE}}


@router.post("/registry/v2/feature_memory")
async def feature_memory(feature: dict, epsilon: float = 0.01, principal: dict = Depends(require_model_writer("model:experiment"))):
    """單一特徵的記憶類型與程式基本暖機（不讀資料）。"""
    from training.registry import warmup as wu

    try:
        return wu.suggest(feature.get("key"), feature.get("params"), epsilon)
    except Exception as e:  # noqa: BLE001
        _bad(str(e))
