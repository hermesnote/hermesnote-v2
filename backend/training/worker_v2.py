"""worker 的 v2 任務與 Phase 3 評估（M2～M5）。training/worker.py 依圖的格式分派到這裡；舊格式的訓練與
一般推論仍走原本的路徑，行為不變。

訓練（v2）：依歷史需求往前載入 → 寫讀取報告 → 建框架（實驗框架：第一次寫入、之後核對雜湊）→ 訓練 →
  保存產物 → 寫驗證集逐筆結果、上游折次 → 摘要與提示。
推論（v2 產物）：同一套組裝；Phase 3 時 best 與 last 都推論、都評估，寫逐筆結果、評估紀錄、holdout 使用紀錄。
Phase 3（舊產物）：沿用舊推論路徑（逐列寫進 model_inference_predictions，相容 MCP），另外對 best／last 對答案與評估。
"""

import asyncio
import json
from datetime import date

import numpy as np

from training import job_store
from training import v2_inputs as vi
from training.provenance import data_report


def _tfs_needed(graph_spec: dict) -> set:
    return {n["timeframe"] for n in graph_spec["nodes"] if n["type"] in ("feature", "label")}


async def load_v2(graph_spec: dict, start: date, end: date) -> tuple[dict, dict]:
    from services.quotes import fetch_ohlcv_dataframe

    req = vi.history_requirements(graph_spec)
    dfs = {}
    for tf in sorted(_tfs_needed(graph_spec)):
        symbol, timeframe = tf.split("_", 1)
        anchor = req["anchors"].get(tf)
        dfs[tf] = await fetch_ohlcv_dataframe(symbol, timeframe, start, end, lookback_bars=req["bars"].get(tf, 2),
                                              load_from=date.fromisoformat(anchor) if anchor else None, with_session=True)
    return dfs, req


def _report(dfs, spec_for_period, req, graph_spec) -> dict:
    from training import alignment as al

    r = data_report(dfs, spec_for_period)
    a = al.resolve_alignment(vi.settings_of(graph_spec))
    r["pipeline"] = "v2"
    r["history_requirements"] = req
    r["alignment"] = {**{k: v for k, v in a.items() if k != "delays"}, "delays": al.delay_report(a, dfs.keys())}
    r["available_time"] = {"source": "bar_end_ts", "delays": r["alignment"]["delays"],
                           "delay_status": "assumed" if all(d["status"] == "assumed" for d in r["alignment"]["delays"].values())
                           else "mixed"}
    return r


async def run_train_job(job_id: str, graph_spec: dict, phase) -> None:
    from training import experiment_store as es
    from training.artifacts import save_artifacts_for_job
    from training.engine_v2 import run_train_v2
    from training.preview_store import save_preview_sample
    from training.progress import write_progress
    from training.v2_records import prediction_rows
    from training.worker import _summarize

    start, end = date.fromisoformat(graph_spec["start"]), date.fromisoformat(graph_spec["end"])
    dfs, req = await load_v2(graph_spec, start, end)
    report = _report(dfs, graph_spec, req, graph_spec)
    await job_store.save_data_report(job_id, report)

    ref = graph_spec.get("frame_ref") or {}
    stored = await es.get_frame(ref["frame_id"]) if ref.get("frame_id") else None

    def hook(frame):
        if stored and stored.get("status") == "ready" and stored.get("hash") != frame.hash:
            raise ValueError(f"樣本框架 {ref['frame_id']} 的雜湊與這次依資料重建的不同（保存 {stored['hash'][:12]}…、"
                             f"重建 {frame.hash[:12]}…）：資料或定義已變更，請建立新版本框架，舊結果不與新框架比較")

    def on_epoch(nid, epoch, metrics):
        write_progress(job_id, nid, epoch, metrics)

    def on_preview(nid, preview):
        save_preview_sample(job_id, nid, preview)

    # 讀取報告已算完；引擎建好時間軸後清空原始資料（consume_dfs），訓練期間不再持有逐列文字欄位（2026-11-18）
    results, frame, extras = await asyncio.to_thread(run_train_v2, graph_spec, dfs, phase=phase, on_epoch=on_epoch,
                                                     on_preview=on_preview, frame_hook=hook, consume_dfs=True)
    if stored and stored.get("status") != "ready":
        await es.save_frame(ref["frame_id"], frame)
    await asyncio.to_thread(save_artifacts_for_job, job_id, graph_spec, results)
    meta = graph_spec.get("experiment") or {}
    n_records = 0
    for nid, entry in extras["records"].items():
        n_records += await es.write_prediction_records(prediction_rows(
            entry, job_id=job_store_uuid(job_id), node_id=nid, dataset="val", phase=phase,
            model_seed=meta.get("model_seed"), experiment_id=meta.get("experiment_id"), arm_id=meta.get("arm_id")))
    from training.v2_records import upstream_oof_rows

    n_oof = {}
    for nid, entry in extras.get("upstream_oof", {}).items():
        n_oof[nid] = await es.write_prediction_records(upstream_oof_rows(
            entry, job_id=job_store_uuid(job_id), node_id=nid, phase=phase, model_seed=meta.get("model_seed"),
            experiment_id=meta.get("experiment_id"), arm_id=meta.get("arm_id"), frame_hash=frame.hash))
    for nid, info in extras["folds"].items():
        await es.write_upstream_folds(job_id, nid, info, frame.hash)
    summary = _summarize(results, graph_spec, phase)
    for nid, r in summary.items():
        r["frame"] = frame.summary() if nid in extras["records"] else results[nid]["training_meta"]["frame"]
        r["n_prediction_records"] = n_records if nid in extras["records"] else 0
        if nid in n_oof:
            r["n_upstream_oof_records"] = n_oof[nid]
    report["warnings"] = (report.get("warnings") or []) + extras["warnings"]
    await job_store.save_data_report(job_id, report)
    await job_store.mark_done(job_id, summary)
    print(f"[worker] v2 job {job_id} done（框架 {frame.hash[:12]}…，逐筆結果 {n_records} 列）", flush=True)


def job_store_uuid(job_id):
    import uuid

    return uuid.UUID(str(job_id))


async def _artifact_chain(target_job_id: str, target_node_id: str) -> tuple[dict, dict]:
    from training.artifacts import load_artifact

    first = await asyncio.to_thread(load_artifact, target_job_id, target_node_id)
    if first is None:
        raise ValueError(f"找不到 job {target_job_id} 節點 {target_node_id} 的模型產物")
    snap = first["graph_spec_snapshot"]
    snap = json.loads(snap) if isinstance(snap, str) else snap
    arts = {}
    for n in snap["nodes"]:
        if n["type"] == "model":
            a = first if n["id"] == target_node_id else await asyncio.to_thread(load_artifact, target_job_id, n["id"])
            if a is not None:
                for k in ("model_config", "preprocessing_state", "train_class_counts"):
                    if isinstance(a.get(k), str):
                        a[k] = json.loads(a[k])
                arts[n["id"]] = a
    return snap, arts


async def run_infer_job(job_id: str, infer_spec: dict, phase) -> bool:
    """v2 產物或 Phase 3 由這裡處理（回傳 True）；舊產物的一般推論回傳 False，交回 worker 原路徑。"""
    from training.artifacts import load_artifact

    art = await asyncio.to_thread(load_artifact, infer_spec["target_job_id"], infer_spec["target_node_id"])
    if art is None:
        raise ValueError("找不到指定的模型產物")
    snap = art["graph_spec_snapshot"]
    snap = json.loads(snap) if isinstance(snap, str) else snap
    if vi.is_v2(snap):
        await _infer_v2(job_id, infer_spec, phase)
        return True
    if phase == 3:
        await _phase3_legacy(job_id, infer_spec, snap, art)
        return True
    return False


async def _infer_v2(job_id: str, spec: dict, phase) -> None:
    from training import experiment_store as es
    from training.engine_v2 import run_infer_v2
    from training.evaluation_records import node_context
    from training.inference_store import open_prediction_writer
    from training.registry import architectures
    from training.v2_records import evaluate_entry, holdout_evaluations, prediction_rows

    tj, tn = spec["target_job_id"], spec["target_node_id"]
    snap, arts = await _artifact_chain(tj, tn)
    start, end = date.fromisoformat(spec["start"]), date.fromisoformat(spec["end"])
    snap = {**snap, "start": spec["start"], "end": spec["end"]}
    dfs, req = await load_v2(snap, start, end)
    report = _report(dfs, spec, req, snap)
    await job_store.save_data_report(job_id, report)
    weights = ("best", "last") if phase == 3 else (spec.get("use_weights", "best"),)
    out = await asyncio.to_thread(run_infer_v2, snap, arts, dfs, spec["start"], spec["end"], weights=weights,
                                  consume_dfs=True)
    # 實驗層的 Phase 3：與訓練相同的框架規則——第一次使用時保存，之後雜湊必須相符（不符就失敗，不寫任何結果）
    ref = spec.get("frame_ref") or {}
    if ref.get("frame_id"):
        stored = await es.get_frame(ref["frame_id"])
        if stored and stored.get("status") == "ready" and stored.get("hash") != out["frame"].hash:
            raise ValueError(f"樣本框架 {ref['frame_id']} 的雜湊與這次依資料重建的不同（保存 {stored['hash'][:12]}…、"
                             f"重建 {out['frame'].hash[:12]}…）：資料或定義已變更，請建立新版本框架，舊結果不與新框架比較")
        if stored and stored.get("status") != "ready":
            await es.save_frame(ref["frame_id"], out["frame"])
    rows, use = out["rows"], out["points"]
    # 相容：一般推論與 Phase 3 都照舊寫 model_inference_predictions（所選權重），只寫有預測的點
    w0 = spec.get("use_weights", "best")
    pred = out["by_weights"][w0 if w0 in out["by_weights"] else "best"]

    def _write():
        with open_prediction_writer(job_id, tn) as writer:
            ts = [int(np.datetime64(t, "s").astype("int64")) for t in rows.ts[use]]
            av = [int(np.datetime64(t, "s").astype("int64")) for t in rows.tau[use]]
            if pred.get("dual") is not None:
                vals = np.column_stack([pred["dual"]["regression"], pred["dual"]["direction_probability"]]).tolist()
                kind, is_prob = "dual", False
            elif pred.get("proba") is not None:
                vals, kind, is_prob = pred["proba"].tolist(), None, True
            else:
                vals, kind, is_prob = np.asarray(pred["pred"]).reshape(-1, 1).tolist(), None, False
            for i in range(0, len(ts), 5000):
                writer.write_chunk(ts[i:i + 5000], vals[i:i + 5000], out["task_type"], is_prob, kind, av[i:i + 5000])
    await asyncio.to_thread(_write)
    meta = spec.get("experiment") or {}
    entry = {"rows": rows, "asm": out["asm"], "points": use, "skipped": out["skipped"], "by_weights": out["by_weights"],
             "frame_hash": out["frame"].hash}
    summary = {"count": int(len(use)), "use_weights": w0, "pipeline": "v2",
               "start_ts": int(np.datetime64(rows.ts[use[0]], "s").astype("int64")) if len(use) else None,
               "end_ts": int(np.datetime64(rows.ts[use[-1]], "s").astype("int64")) if len(use) else None,
               "skipped": {str(k): int(v) for k, v in
                           zip(*np.unique([str(out["asm"].reason[r]) for r in out["skipped"]], return_counts=True))}
               if len(out["skipped"]) else {},
               "frame": out["frame"].summary(), "output_specs": out["model_config"].get("output_specs")}
    if phase == 3:
        n = await es.write_prediction_records(prediction_rows(
            entry, job_id=job_store_uuid(job_id), node_id=tn, dataset="holdout", phase=3,
            model_seed=meta.get("model_seed"), experiment_id=meta.get("experiment_id"), arm_id=meta.get("arm_id")))
        art = arts[tn]
        stats = art.get("train_class_counts")
        raw, st = evaluate_entry(entry, task_type=out["task_type"], model_config=art["model_config"], stats=stats)
        node, label_node, task_type = node_context(snap, tn)
        specs = architectures.metric_specs(node["key"], node, label_node, task_type)
        evals = holdout_evaluations(tn, raw, specs=specs, phase=3, period={"start": spec["start"], "end": spec["end"]},
                                    stats_out=st)
        for w in raw:
            await es.write_evaluation(job_id, tn, 3, "holdout", w, [e for e in evals if e["point"]["kind"] == w], st)
        await es.write_holdout_usage("evaluation", job_id=job_id, parent_job_id=tj, experiment_id=meta.get("experiment_id"),
                                     arm_id=meta.get("arm_id"), start=spec["start"], end=spec["end"],
                                     detail={"weights": list(raw), "n_included": st["n_included"]})
        summary.update({"evaluations": evals, "metric_specs": specs, "holdout": st, "n_prediction_records": n})
    report["warnings"] = (report.get("warnings") or []) + out["warnings"]
    await job_store.save_data_report(job_id, report)
    await job_store.mark_done(job_id, {tn: summary})


async def _phase3_legacy(job_id: str, spec: dict, snap: dict, art: dict) -> None:
    """舊格式產物的 Phase 3：推論沿用舊路徑（含 model_inference_predictions），再對 best／last 對答案、評估。"""
    import pandas as pd

    from training import experiment_store as es
    from training.evaluation_records import node_context
    from training.graph import _ensure_time_indexed
    from training.holdout_eval import evaluate_outputs
    from training.inference import dual_output_names, run_inference
    from training.registry import architectures
    from training.registry.labeling_rules import apply_labeling_rule
    from training.registry.outcomes import compute_outcome
    from training.worker import _load_dataframes_for_infer, run_infer_sync
    from training.v2_records import holdout_evaluations

    tj, tn = spec["target_job_id"], spec["target_node_id"]
    start, end = date.fromisoformat(spec["start"]), date.fromisoformat(spec["end"])
    dfs = await _load_dataframes_for_infer(tj, tn, start, end)
    node, label_node, task_type = node_context(snap, tn)
    if label_node["timeframe"] not in dfs:
        from services.quotes import fetch_ohlcv_dataframe

        sym, tf = label_node["timeframe"].split("_", 1)
        dfs[label_node["timeframe"]] = await fetch_ohlcv_dataframe(sym, tf, start, end)
    await job_store.save_data_report(job_id, data_report(dfs, spec))
    summary = await asyncio.to_thread(run_infer_sync, job_id, tj, tn, dfs, spec.get("use_weights", "best"))

    def _eval():
        ldf = _ensure_time_indexed(dfs[label_node["timeframe"]])
        params = label_node.get("params") or {}
        raw, vl = compute_outcome(label_node["outcome"], ldf, params)
        y, _tt, _ = apply_labeling_rule(label_node.get("labeling_rule", "fixed_threshold"), raw, vl, params)
        y_by_ts = pd.Series(y, index=ldf.index)
        mc = art["model_config"] if isinstance(art["model_config"], dict) else json.loads(art["model_config"])
        dual = dual_output_names(mc)
        out = {}
        for w in ("best", "last"):
            if w == "last" and art.get("weights_last") is None:
                continue
            res = run_inference(tj, tn, lambda tf: dfs[tf], use_weights=w)
            ts = pd.to_datetime(res["timestamps"], unit="s")
            preds = np.asarray(res["predictions"], dtype=float)
            out[w] = (ts, preds)
        if not out:
            return None
        ts = next(iter(out.values()))[0]
        yv = y_by_ts.reindex(ts).to_numpy()
        inc = ~np.isnan(yv)
        # 舊產物推論不往前取暖機：期間開頭的窗格可能含 NaN 特徵，預測值非有限數時不納入評估（記原因）
        finite = np.ones(len(ts), dtype=bool)
        for _w, (_t, p) in out.items():
            finite &= np.isfinite(p.reshape(len(p), -1)).all(axis=1)
        inc &= finite
        stats = art.get("train_class_counts")
        stats = json.loads(stats) if isinstance(stats, str) else stats
        reports, rows = {}, []
        for w, (_ts, p) in out.items():
            if dual:
                # 舊路徑的 run_inference 只回 default（回歸值），沒有方向機率，雙頭的 Phase 3 評估請用 v2 圖
                reports[w] = None
                continue
            if task_type == "classification" and p.ndim == 2 and p.shape[1] > 1:
                o = {"pred": p.argmax(axis=1), "proba": p, "dual": None}
            else:
                o = {"pred": p.reshape(-1), "proba": None, "dual": None}
            if o is not None:
                sub = {"pred": o["pred"][inc], "proba": None if o["proba"] is None else o["proba"][inc], "dual": None}
                reports[w] = evaluate_outputs(task_type, mc, yv[inc], sub, stats) if inc.any() else None
                for i in range(len(ts)):
                    rows.append((job_store_uuid(job_id), tn, None, None, None, 3, "holdout", w, None,
                                 ts[i].to_pydatetime(), None, None, None, None,
                                 None if np.isnan(yv[i]) else float(yv[i]),
                                 float(o["pred"][i]) if np.isfinite(o["pred"][i]) else None,
                                 None if o["proba"] is None or not finite[i] else json.dumps([float(x) for x in o["proba"][i]]),
                                 bool(inc[i]), None if inc[i] else ("prediction_not_finite" if not finite[i] else "label_unmatured"),
                                 None, None))
        st = {"n_points": int(len(ts)), "n_predicted": int(len(ts)), "n_included": int(inc.sum()),
              "n_unmatured": int((~inc & finite).sum()), "n_skipped": int((~finite).sum()),
              "skipped_reasons": {"prediction_not_finite": int((~finite).sum())} if (~finite).any() else {},
              "cutoff_ts": str(ts[inc].max())[:19] if inc.any() else None,
              "baseline_stats": "train_stats" if stats else "unavailable（舊產物沒有保存訓練集統計）",
              "note": "舊格式產物：推論期間開頭沒有往前取暖機，暖機中的時間點照舊輸出（與 v2 不同）"}
        return reports, rows, st

    got = await asyncio.to_thread(_eval)
    if got is not None:
        reports, rows, st = got
        await es.write_prediction_records(iter(rows))
        specs = architectures.metric_specs(node["key"], node, label_node, task_type)
        evals = holdout_evaluations(tn, {k: v for k, v in reports.items() if v}, specs=specs, phase=3,
                                    period={"start": spec["start"], "end": spec["end"]}, stats_out=st)
        for w in reports:
            if reports[w]:
                await es.write_evaluation(job_id, tn, 3, "holdout", w, [e for e in evals if e["point"]["kind"] == w], st)
        await es.write_holdout_usage("evaluation", job_id=job_id, parent_job_id=tj, start=spec["start"], end=spec["end"],
                                     detail={"weights": [w for w in reports if reports[w]], "legacy": True})
        summary.update({"evaluations": evals, "metric_specs": specs, "holdout": st})
    await job_store.mark_done(job_id, {tn: summary})
