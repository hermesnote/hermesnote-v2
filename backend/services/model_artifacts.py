"""模型產物的 web 後端查詢/操作（async，給 FastAPI route handler 用）。

實際的保存/推論載入在訓練容器的同步路徑（見 `training/artifacts.py`），這支只負責
「查清單」跟「切換 kept 標記」這種可以在 async 事件迴圈裡直接做、不需要碰 GPU/torch 的操作。
"""

import json
import uuid

from services.hermesnote_db import get_conn


async def list_artifacts(job_id: str | None = None) -> list[dict]:
    conn = await get_conn()
    try:
        if job_id:
            rows = await conn.fetch(
                """
                SELECT job_id, node_id, architecture_key, task_type, kept, created_at
                FROM model_artifacts WHERE job_id = $1 ORDER BY created_at DESC
                """,
                uuid.UUID(job_id),
            )
        else:
            rows = await conn.fetch(
                """
                SELECT job_id, node_id, architecture_key, task_type, kept, created_at
                FROM model_artifacts ORDER BY created_at DESC
                """
            )
    finally:
        await conn.close()
    return [
        {
            "job_id": str(r["job_id"]), "node_id": r["node_id"], "architecture_key": r["architecture_key"],
            "task_type": r["task_type"], "kept": r["kept"], "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


async def mark_kept(job_id: str, node_id: str, kept: bool) -> bool:
    conn = await get_conn()
    try:
        result = await conn.execute(
            "UPDATE model_artifacts SET kept = $3 WHERE job_id = $1 AND node_id = $2",
            uuid.UUID(job_id), node_id, kept,
        )
        return result != "UPDATE 0"
    finally:
        await conn.close()


async def artifact_exists(job_id: str, node_id: str) -> bool:
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            "SELECT 1 FROM model_artifacts WHERE job_id = $1 AND node_id = $2",
            uuid.UUID(job_id), node_id,
        )
        return row is not None
    finally:
        await conn.close()


# ── 舊任務 v2 metadata 唯讀補充（2026-11-21）────────────────────────────────
# 2026-11-21 以前 worker 的摘要白名單沒有保留 v2，已完成的任務 job.result 缺 v2.input_layout。
# 這裡只在 API 回應時從同任務已保存的 model_artifacts 推回，不改寫資料庫；每個欄位標明來源，推不出的列入 unknown。
_V2_ARTIFACT_KEYS = ["node_id", "model_config", "preprocessing_state", "graph_spec_snapshot",
                     "warmup_policy", "history_requirements", "fingerprint"]


async def v2_artifact_sources(job_id: str) -> dict[str, dict]:
    conn = await get_conn()
    try:
        rows = await conn.fetch(f"SELECT {', '.join(_V2_ARTIFACT_KEYS)} FROM model_artifacts WHERE job_id = $1",
                                uuid.UUID(job_id))
    finally:
        await conn.close()
    out = {}
    for r in rows:
        out[r["node_id"]] = {k: (json.loads(r[k]) if isinstance(r[k], str) and k != "fingerprint" and k != "node_id"
                                 else r[k]) for k in _V2_ARTIFACT_KEYS}
    return out


def _branch_widths(inputs: list, art: dict) -> tuple[list | None, str | None]:
    """各輸入分支的欄數。單一分支用 model_config.n_features；否則用 preprocessing_state 各分支 min 的長度。"""
    mc = art.get("model_config") or {}
    nf = mc.get("n_features")
    pre = art.get("preprocessing_state") or {}
    by_id = {b.get("id"): b for b in pre.get("branches") or []} if pre.get("version") == 2 else {}
    pw = []
    for b in inputs:
        st = [s for s in (by_id.get(b["id"]) or {}).get("steps") or [] if s.get("method") == "global_minmax"]
        pw.append(len(st[0]["min"]) if st else None)
    if len(inputs) == 1 and isinstance(nf, int):
        if pw[0] is not None and pw[0] != nf:
            return None, None  # 兩個來源不一致：維持未知
        return [nf], "model_artifacts.model_config.n_features + model_artifacts.graph_spec_snapshot"
    if inputs and all(w is not None for w in pw):
        return pw, "model_artifacts.preprocessing_state（各分支 global_minmax min 長度）+ model_artifacts.graph_spec_snapshot"
    return None, None


def v2_from_artifact(node_id: str, art: dict | None, frame_hash: str | None = None) -> dict:
    """由模型產物推回 job.result[node].v2（回應用）。backfilled=true；source 逐欄標來源；unknown 列出推不出的欄位。"""
    from training import v2_inputs as vi

    out = {"backfilled": True, "input_layout": None, "source": {}, "unknown": []}
    if art is None:
        out["unknown"] = ["input_layout", "model_input_format", "mask_branches", "warmup_policy", "frame_hash",
                          "alignment", "decision_timeframe", "history_requirements", "weights_fingerprint",
                          "upstream_policy"]
        return out
    src = out["source"]
    mc = art.get("model_config") or {}
    snap = art.get("graph_spec_snapshot") or {}
    nodes = {n["id"]: n for n in snap.get("nodes") or []}
    node = nodes.get(node_id)
    branches = None
    if node is not None:
        try:
            branches = vi.branches_of(node, nodes)
        except Exception:
            branches = None
    if mc.get("input_layout"):
        out["input_layout"] = mc["input_layout"]
        src["input_layout"] = "model_artifacts.model_config.input_layout"
    elif branches is not None:
        inputs = [b for b in branches if b["role"] == "input"]
        widths, how = _branch_widths(inputs, art)
        if widths is not None:
            off, lb = 0, []
            for b, k in zip(inputs, widths):
                lb.append({"id": b["id"], "timeframe": b["timeframe"], "window": b["window"], "encoder": b["encoder"],
                           "cols": [off, off + k]})
                off += k
            out["input_layout"] = {"version": 1, "w_max": max(b["window"] for b in inputs), "f_total": off,
                                   "branches": lb}
            src["input_layout"] = how
    if out["input_layout"] is None:
        out["unknown"].append("input_layout")
    # 模型實際收到的格式：model_config 有 input_layout＝多分支格式；沒有＝單分支相容格式（權重格式同舊版）
    out["model_input_format"] = "input_layout" if mc.get("input_layout") else "legacy_single_branch"
    src["model_input_format"] = "model_artifacts.model_config.input_layout（有無）"
    if branches is not None:
        out["mask_branches"] = [b["id"] for b in branches if b["role"] == "mask"]
        src["mask_branches"] = "model_artifacts.graph_spec_snapshot"
        try:
            out["decision_timeframe"] = vi.decision_tf(node, nodes)
            src["decision_timeframe"] = "model_artifacts.graph_spec_snapshot"
        except Exception:
            out["unknown"].append("decision_timeframe")
    else:
        out["unknown"] += ["mask_branches", "decision_timeframe"]
    for key, col in (("warmup_policy", "warmup_policy"), ("history_requirements", "history_requirements"),
                     ("weights_fingerprint", "fingerprint")):
        if art.get(col) is not None:
            out[key] = art[col]
            src[key] = f"model_artifacts.{col}"
        else:
            out["unknown"].append(key)
    if frame_hash:
        out["frame_hash"] = frame_hash
        src["frame_hash"] = "job.result.training_meta.frame.hash"
    else:
        out["unknown"].append("frame_hash")
    out["unknown"] += ["alignment", "upstream_policy"]
    return out


async def backfill_v2(job: dict) -> dict:
    """GET /jobs/{id} 回應前：v2 訓練任務中缺 v2 的節點，用模型產物唯讀補上（不寫回資料庫）。"""
    result = job.get("result")
    if job.get("job_type") != "train" or (job.get("graph_spec") or {}).get("mode") != "v2" or not isinstance(result, dict):
        return job
    missing = [nid for nid, r in result.items() if isinstance(r, dict) and "v2" not in r]
    if not missing:
        return job
    arts = await v2_artifact_sources(str(job.get("job_id") or job.get("id")))
    for nid in missing:
        r = result[nid]
        fh = ((r.get("training_meta") or {}).get("frame") or {}).get("hash")
        r["v2"] = v2_from_artifact(nid, arts.get(nid), fh)
    return job
