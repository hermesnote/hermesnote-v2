"""舊版 v2 組裝（2026-11-18 A4 之前，原樣）：用 np.concatenate 合併各輸入，作為新舊逐位元比對的對照組。
只給測試用；正式程式在 training/v2_inputs.py。"""

import numpy as np

from training import alignment as al
from training.graph_refs import normalize_input_ref
from training.v2_inputs import Assembled, _CODES, branches_of


def legacy_assemble(model_node: dict, nodes: dict, tfs: dict, tau: np.ndarray, feature_cache, upstream_series: dict,
             alignment: dict) -> Assembled:
    """tau：每個時間點的決策時刻（datetime64）。feature_cache(fid) → 該特徵在自己時間框架上的值 (n_f, k)。
    upstream_series[(node, output)] → {"available", "values", "fold"}（上游在自己決策 K 棒上的輸出）。"""
    tau = np.asarray(tau, dtype="datetime64[ns]")
    n = len(tau)
    res = Assembled()
    res.branches = branches_of(model_node, nodes)
    reason = np.full(n, None, dtype=object)
    w_max = max(b["window"] for b in res.branches)
    off, layout_b = 0, []
    win_start = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    hits = {c: np.zeros(n, dtype=bool) for c in _CODES}
    for b in res.branches:
        T = tfs[b["timeframe"]]
        cols, row_bad = [], {c: np.zeros(len(T), dtype=bool) for c in _CODES}
        ups = {}
        for item in b["inputs"]:
            ref = normalize_input_ref(item)
            src = nodes[ref.node]
            if src["type"] == "feature":
                F = tfs[src["timeframe"]]
                vals = feature_cache(ref.node)
                if src["timeframe"] == T.tf:
                    v = vals
                elif al.is_coarser(src["timeframe"], T.tf):
                    v, pos = al.asof_values(F.available, vals, T.available)
                    st = al.staleness(F, pos, T.available, alignment)
                    row_bad["stale_branch"] |= st["stale"]
                    row_bad["insufficient_history"] |= pos < 0
                else:
                    v, present = al.aggregate_subbars(F, vals, T, b["aggregate"][ref.node])
                    row_bad["insufficient_history"] |= ~present
                nanrow = np.isnan(v).any(axis=1)
                row_bad["warmup_not_ready"] |= nanrow & ~row_bad["stale_branch"] & ~row_bad["insufficient_history"]
            else:
                s = upstream_series[(ref.node, ref.output)]
                v, pos = al.asof_values(s["available"], s["values"], T.available)
                row_bad["upstream_unavailable"] |= np.isnan(v).any(axis=1)
                fold_rows = np.where(pos >= 0, np.asarray(s["fold"])[np.where(pos >= 0, pos, 0)], -2)
                ups[ref.node] = {"fold_rows": fold_rows, "pos_rows": pos, "src_available": s["available"]}
            cols.append(np.asarray(v, dtype=float))
        base = np.concatenate(cols, axis=1) if len(cols) > 1 else cols[0]
        e = al.asof_positions(T, tau)
        st = al.staleness(T, e, tau, alignment)
        w = b["window"]
        lo = e - w + 1
        bad_hist = (e < 0) | (lo < 0)
        lo_c, e_c = np.clip(lo, 0, None), np.clip(e, 0, None)
        for code in _CODES:
            cum = np.r_[0, np.cumsum(row_bad[code])]
            hit = (cum[e_c + 1] - cum[lo_c]) > 0
            if code == "stale_branch":
                hit = hit | st["stale"]
            if code == "insufficient_history":
                hit = hit | bad_hist
            hits[code] |= hit & ~bad_hist if code != "insufficient_history" else hit
        res.end_pos[b["id"]] = e
        res.bases[b["id"]] = base
        p = np.where(e >= 0, e, 0)
        res.branch_meta[b["id"]] = {
            "last_ts": np.where(e >= 0, T.index.values[p], np.datetime64("NaT")),
            "available": np.where(e >= 0, T.available[p], np.datetime64("NaT")),
            "age_sec": st["age_sec"], "sessions_back": st["sessions_back"],
        }
        ws = np.where(e >= 0, T.index.values[lo_c], np.datetime64("NaT"))
        win_start = np.where(np.isnat(win_start) | (ws < win_start), ws, win_start)
        if ups:
            res.upstream[b["id"]] = {
                u: {"fold_at": np.where(e >= 0, d["fold_rows"][p], -2),
                    "folds_in_window": (lo_c, e_c, d["fold_rows"]),
                    "avail_at": np.where((e >= 0) & (d["pos_rows"][p] >= 0),
                                         d["src_available"][np.where(d["pos_rows"][p] >= 0, d["pos_rows"][p], 0)],
                                         np.datetime64("NaT"))}
                for u, d in ups.items()}
        layout_b.append({"id": b["id"], "timeframe": b["timeframe"], "window": w, "encoder": b["encoder"],
                         "cols": [off, off + base.shape[1]]})
        off += base.shape[1]
    for code in ("insufficient_history", "stale_branch", "upstream_unavailable", "warmup_not_ready"):
        reason[hits[code] & (reason == None)] = code  # noqa: E711 — 優先序：歷史不足 > 過舊 > 上游 > 暖機
    res.reason = reason
    res.window_start = win_start
    res.layout = {"version": 1, "w_max": w_max, "f_total": off, "branches": layout_b}
    return res
