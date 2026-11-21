"""v2 圖（graph_spec.mode == "v2"）的輸入組裝：訓練與推論共用同一套函式（M2／M4／M6）。

  1. 特徵：compute_feature → 暖機遮罩（從資料載入起點算的前 N 根設成 NaN，N＝max(程式基本值, 研究者設定)）
     → 轉換元件（training/registry/transforms.py）。
  2. 分支：每個 Model Node 有一或多個分支（未設定 branches 時＝一個 main 分支，內容等於 inputs／window）。
     分支有自己的時間框架與 window；其他時間框架的特徵／上游預測用 as-of（較粗）或子 K 棒彙總（較細）
     對到分支的每一根 K 棒（training/alignment.py）。
  3. 對每個決策時間點，每個分支取「可用時間 ≤ 決策時刻」的最後 window 根，檢查歷史是否足夠、窗格內有無
     暖機中的值、上游預測是否存在、最後一根是否過舊；不合格的時間點不產生樣本，記錄原因代碼。
  4. 各分支各自正規化（步驟清單，目前只有 global_minmax：只用訓練樣本窗格涵蓋的列擬合）。
  5. 打包成 PackedWindows：(n, W_max, F_total)，分支 b 佔自己的欄位、右對齊在最後 W_b 步，左側補 0；
     input_layout 記錄每個分支的欄位範圍與 window，模型（training/registry/lstm.py、xgboost_model.py）依此拆開。
"""

import numpy as np

from training import alignment as al
from training.graph_refs import input_refs, normalize_input_ref
from training.registry import transforms as tr
from training.registry import warmup as wu
from training.registry.features import compute_feature

ENCODERS = {
    "lstm": {"label": "LSTM 序列編碼（沿用節點的 lstm_layers／bidirectional／attention 設定）", "families": ["lstm"]},
    "flatten": {"label": "窗格攤平（XGBoost）", "families": ["xgboost"]},
    "gru": {"label": "GRU 序列編碼（節點的 gru_layers／bidirectional）", "families": ["gru"]},
    "cnn_lstm": {"label": "CNN-LSTM 序列編碼（節點的 conv_layers／pool_size／lstm_layers）", "families": ["cnn_lstm"]},
    "transformer": {"label": "Transformer Encoder 序列編碼（節點的 transformer 設定）", "families": ["transformer"]},
    "last_value": {"label": "只取最後一根（window 固定 1）",
                   "families": ["lstm", "xgboost", "gru", "cnn_lstm", "transformer"]},
}
DEFAULT_ENCODER = {"lstm": "lstm", "xgboost": "flatten", "gru": "gru", "cnn_lstm": "cnn_lstm", "transformer": "transformer"}
# 分支角色（2026-11-21）：input＝輸入模型；mask＝只套用它的樣本可用性（暖機、上游樣本外預測可用與清除），
# 不輸入模型。用來讓「沒加上游機率」的對照組與「有加」組使用完全相同的樣本（見 api.md「共用樣本遮罩」）。
BRANCH_ROLES = ("input", "mask")
NORMALIZATIONS = {"global_minmax": "全域 MinMax：只用訓練樣本窗格涵蓋的列擬合 min／max，套用到全部（含驗證、推論）"}


def is_v2(graph_spec: dict) -> bool:
    return isinstance(graph_spec, dict) and graph_spec.get("mode") == "v2"


def settings_of(graph_spec: dict) -> dict:
    return graph_spec.get("settings") or {}


def epsilon_of(graph_spec: dict) -> float:
    return float((settings_of(graph_spec).get("warmup") or {}).get("epsilon", wu.DEFAULT_EPSILON))


# ── 暖機 ────────────────────────────────────────────────────────────────
def feature_warmup(node: dict, epsilon: float) -> dict:
    """單一 Feature Node 的暖機政策：程式基本值、研究者設定、轉換額外根數、實際使用值。"""
    sug = wu.suggest(node["key"], node.get("params"), epsilon)
    researcher = (node.get("warmup") or {}).get("bars")
    base = int(sug["base_bars"])
    used = max(base, int(researcher)) if researcher is not None else base
    extra = tr.extra_warmup(node.get("transform"))
    return {"memory_type": sug["memory_type"], "base_bars": base, "researcher_bars": researcher,
            "bars": used, "transform": node.get("transform") or {"type": "none"}, "transform_extra": extra,
            "total_bars": used + extra, "method": sug["method"], "epsilon": epsilon, "detail": sug["detail"],
            "needs_researcher": sug["needs_researcher"]}


def feature_issues(nid: str, node: dict, epsilon: float) -> list[str]:
    out = []
    w = node.get("warmup")
    if w is not None and (not isinstance(w, dict) or set(w) - {"bars"}):
        return [f"Feature 節點 {nid}：warmup 只接受 {{\"bars\": 整數}}"]
    researcher = (w or {}).get("bars")
    if researcher is not None and (isinstance(researcher, bool) or not isinstance(researcher, int) or researcher < 0):
        return [f"Feature 節點 {nid}：warmup.bars 必須是 ≥ 0 的整數"]
    try:
        sug = wu.suggest(node.get("key"), node.get("params"), epsilon)
    except Exception as e:  # noqa: BLE001 — TA-Lib 參數錯誤等，回報成驗證問題
        return [f"Feature 節點 {nid}：無法判定暖機（{type(e).__name__}：{e}）"]
    out += [f"Feature 節點 {nid}：{m}" for m in tr.validate(node.get("transform"), sug["memory_type"])]
    if sug["needs_researcher"] and sug["memory_type"] != "cumulative" and researcher is None:
        out.append(f"Feature 節點 {nid}：{sug['detail']}（暖機需要研究者設定 warmup.bars）")
    if researcher is not None and researcher < sug["base_bars"]:
        out.append(f"Feature 節點 {nid}：warmup.bars={researcher} 比程式的基本值 {sug['base_bars']} 短（只能加長）")
    return out


def compute_feature_v2(node: dict, data: "al.TfData", epsilon: float) -> np.ndarray:
    raw = np.asarray(compute_feature(node["key"], data.ohlcv, node.get("params")), dtype=float)
    raw = raw.reshape(-1, 1) if raw.ndim == 1 else raw.copy()
    w = feature_warmup(node, epsilon)["bars"]
    raw[:w] = np.nan
    return tr.apply(raw, node.get("transform"), index=data.index)


# ── 分支 ────────────────────────────────────────────────────────────────
def family_of(model_node: dict) -> str:
    from training.registry.architectures import ARCHITECTURE_META

    return (ARCHITECTURE_META.get(model_node.get("key")) or {}).get("family", model_node.get("key"))


def decision_tf(model_node: dict, nodes: dict) -> str:
    return nodes[model_node["label"]]["timeframe"]


def branches_of(model_node: dict, nodes: dict) -> list[dict]:
    """正規化後的分支清單：[{id, inputs, timeframe, window, encoder, aggregate, normalization}]。"""
    fam = family_of(model_node)
    raw = model_node.get("branches")
    if not raw:
        raw = [{"id": "main", "inputs": model_node.get("inputs") or [], "window": model_node.get("window", 60),
                "timeframe": decision_tf(model_node, nodes), "normalization": model_node.get("normalization")}]
    out = []
    for b in raw:
        enc = b.get("encoder") or DEFAULT_ENCODER.get(fam, "lstm")
        inputs = list(b.get("inputs") or [])
        tf = b.get("timeframe")
        if tf is None:
            for item in inputs:
                ref = normalize_input_ref(item)
                n = nodes.get(ref.node)
                if n is None:
                    continue
                tf = n["timeframe"] if n["type"] == "feature" else decision_tf(n, nodes)
                break
        out.append({"id": str(b.get("id")), "inputs": inputs, "timeframe": tf,
                    "window": 1 if enc == "last_value" else int(b.get("window", 60)), "encoder": enc,
                    "aggregate": dict(b.get("aggregate") or {}), "role": b.get("role", "input"),
                    "normalization": b.get("normalization") or [{"method": "global_minmax"}]})
    return out


def branch_issues(nid: str, model_node: dict, nodes: dict) -> list[str]:
    out = []
    fam = family_of(model_node)
    raw = model_node.get("branches")
    if raw is not None and (not isinstance(raw, list) or not raw):
        return [f"Model 節點 {nid}：branches 必須是非空陣列"]
    try:
        branches = branches_of(model_node, nodes)
    except Exception as e:  # noqa: BLE001
        return [f"Model 節點 {nid}：branches 格式不對（{e}）"]
    ids = [b["id"] for b in branches]
    if len(set(ids)) != len(ids):
        out.append(f"Model 節點 {nid}：分支 id 重複 {ids}")
    used: list = []
    if not any(b["role"] == "input" for b in branches):
        out.append(f"Model 節點 {nid}：至少要有一個 role=input 的分支")
    for b in branches:
        tag = f"Model 節點 {nid} 分支 {b['id']}"
        if b["role"] not in BRANCH_ROLES:
            out.append(f"{tag}：role 必須是 {list(BRANCH_ROLES)} 之一，收到 {b['role']!r}")
        if b["encoder"] not in ENCODERS:
            out.append(f"{tag}：未登記的 encoder {b['encoder']!r}，目前有 {sorted(ENCODERS)}")
        elif fam not in ENCODERS[b["encoder"]]["families"]:
            out.append(f"{tag}：encoder {b['encoder']!r} 不適用 {fam}（可用：{[k for k, v in ENCODERS.items() if fam in v['families']]}）")
        if not 1 <= b["window"] <= 1000:
            out.append(f"{tag}：window 必須在 1～1000")
        if not b["inputs"]:
            out.append(f"{tag}：inputs 不能是空的")
        if b["timeframe"] is None:
            out.append(f"{tag}：無法決定分支的時間框架")
        for step in b["normalization"]:
            if not isinstance(step, dict) or step.get("method") not in NORMALIZATIONS:
                out.append(f"{tag}：normalization 步驟必須是 {sorted(NORMALIZATIONS)}，收到 {step!r}")
        for item in b["inputs"]:
            try:
                ref = normalize_input_ref(item)
            except ValueError as e:
                out.append(f"{tag}：{e}")
                continue
            used.append(item if isinstance(item, str) else (ref.node, ref.output))
            n = nodes.get(ref.node)
            if n is None or b["timeframe"] is None:
                continue
            src_tf = n["timeframe"] if n["type"] == "feature" else (decision_tf(n, nodes) if n["type"] == "model" else None)
            if src_tf is None:
                continue
            if src_tf != b["timeframe"] and not al.is_coarser(src_tf, b["timeframe"]):
                method = b["aggregate"].get(ref.node)
                if method not in al.AGGREGATIONS:
                    out.append(f"{tag}：{ref.node}（{src_tf}）比分支時間框架 {b['timeframe']} 細，"
                               f"必須在 aggregate 指定彙總方式 {list(al.AGGREGATIONS)}")
                if n["type"] == "model":
                    out.append(f"{tag}：上游模型 {ref.node} 的時間框架比分支細，不支援彙總上游預測")
    if raw:
        declared = [i if isinstance(i, str) else (normalize_input_ref(i).node, normalize_input_ref(i).output)
                    for i in (model_node.get("inputs") or [])]
        if sorted(map(str, declared)) != sorted(map(str, used)):
            out.append(f"Model 節點 {nid}：branches 的 inputs 合起來必須剛好等於節點的 inputs（各出現一次）")
    if fam == "xgboost" and any(b["encoder"] == "lstm" for b in branches):
        out.append(f"Model 節點 {nid}：XGBoost 分支請用 flatten 或 last_value")
    return out


# ── 歷史需求 ─────────────────────────────────────────────────────────────
def _convert(target_tf: str, bars: int, of_tf: str) -> int:
    if bars <= 0:
        return 0
    if al.tf_part(target_tf) == al.tf_part(of_tf):
        return int(bars)
    return al.bars_needed_in(target_tf, bars, of_tf)


def history_requirements(graph_spec: dict) -> dict:
    """每個時間框架要往期間起點之前載入幾根（暖機＋窗格＋上游預測所需），以及固定錨點日。
    訓練與推論都依同一份計算載入，結果存進產物。"""
    from training.graph import _topo_order

    eps = epsilon_of(graph_spec)
    nodes = {n["id"]: n for n in graph_spec["nodes"]}
    order = _topo_order(nodes)
    models = [nid for nid in order if nodes[nid]["type"] == "model"]
    demand = {nid: 0 for nid in models}  # 該模型在自己決策時間框架上，期間起點之前還要多產生幾根的輸出
    req: dict = {}
    anchors: dict = {}

    def need(tf, bars):
        req[tf] = max(req.get(tf, 0), int(bars))

    for nid in reversed(models):
        m = nodes[nid]
        dtf = decision_tf(m, nodes)
        need(dtf, demand[nid] + 2)
        for b in branches_of(m, nodes):
            rows_b = b["window"] + _convert(b["timeframe"], demand[nid], dtf)
            need(b["timeframe"], rows_b + 2)
            for item in b["inputs"]:
                ref = normalize_input_ref(item)
                n = nodes[ref.node]
                if n["type"] == "feature":
                    w = feature_warmup(n, eps)
                    need(n["timeframe"], w["total_bars"] + _convert(n["timeframe"], rows_b, b["timeframe"]) + 2)
                    t = n.get("transform") or {}
                    if t.get("type") == "fixed_anchor":
                        a = str(t["anchor"])
                        anchors[n["timeframe"]] = min(anchors.get(n["timeframe"], a), a)
                elif n["type"] == "model":
                    demand[ref.node] = max(demand[ref.node], _convert(decision_tf(n, nodes), rows_b, b["timeframe"]))
    return {"bars": req, "anchors": anchors, "epsilon": eps}


# ── 組裝 ────────────────────────────────────────────────────────────────
class PackedWindows:
    """多分支窗格的惰性打包（見模組說明）；索引時才切出 (k, W_max, F_total)，跟 LazyWindowed 同一個用途。"""

    def __init__(self, parts: list, w_max: int, f_total: int, sample_idx: np.ndarray):
        self.parts = parts  # [(base (n_b, F_b) float32, end_pos (n_points,), W_b, col_offset)]
        self.w_max, self.f_total = int(w_max), int(f_total)
        self.sample_idx = np.asarray(sample_idx, dtype=np.int64)

    @property
    def shape(self):
        return (len(self.sample_idx), self.w_max, self.f_total)

    def __len__(self):
        return len(self.sample_idx)

    def subset(self, local_idx) -> "PackedWindows":
        return PackedWindows(self.parts, self.w_max, self.f_total, self.sample_idx[local_idx])

    def __getitem__(self, local_idx):
        pts = self.sample_idx[local_idx]
        single = np.ndim(pts) == 0
        pts = np.atleast_1d(pts)
        out = np.zeros((len(pts), self.w_max, self.f_total), dtype=np.float32)
        steps = None
        for base, end_pos, w, off in self.parts:
            e = end_pos[pts]
            steps = (e[:, None] - w + 1) + np.arange(w)[None, :]
            out[:, self.w_max - w:, off:off + base.shape[1]] = base[steps]
        return out[0] if single else out

    def __array__(self, dtype=None):
        arr = self[np.arange(len(self))]
        return arr.astype(dtype) if dtype is not None else arr


class Assembled:
    def __init__(self):
        self.reason = None          # (n_points,) object：None＝可用
        self.end_pos = {}           # 分支 id → (n_points,) 每個時間點在分支上的最後一根位置
        self.bases = {}             # 分支 id → (n_b, F_b) 原始（未正規化）值
        self.branch_meta = {}       # 分支 id → {last_ts, available, age_sec, sessions_back}
        self.upstream = {}          # 分支 id → {上游節點: {"fold_at": (n_points,), "fold_rows": (n_b,), "avail_at": (n_points,)}}
        self.layout = None
        self.branches = []
        self.window_start = None    # (n_points,) 所有分支窗格中最早一根的 bar_start（串接清除規則用）


_CODES = ("warmup_not_ready", "upstream_unavailable", "stale_branch", "insufficient_history")


def assemble(model_node: dict, nodes: dict, tfs: dict, tau: np.ndarray, feature_cache, upstream_series: dict,
             alignment: dict, release=None) -> Assembled:
    """tau：每個時間點的決策時刻（datetime64）。feature_cache(fid) → 該特徵在自己時間框架上的值 (n_f, k)。
    upstream_series[(node, output)] → {"available", "values", "fold"}（上游在自己決策 K 棒上的輸出）。
    release(fid)：這次引用用完特徵後呼叫（呼叫端據此在最後一次引用後釋放快取；2026-11-18 A4）。
    合併矩陣（base）在每個分支的輸入都取得後才配置，逐塊複製、複製完即丟掉該塊，
    與 np.concatenate 逐位元相同，但峰值只多一塊而不是多一份完整矩陣。"""
    tau = np.asarray(tau, dtype="datetime64[ns]")
    n = len(tau)
    res = Assembled()
    res.branches = branches_of(model_node, nodes)
    reason = np.full(n, None, dtype=object)
    w_max = max(b["window"] for b in res.branches if b["role"] == "input")
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
                if release is not None:
                    release(ref.node)  # 快取可能被移除；vals 仍持有這份資料直到複製進 base
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
            v = vals = None
        n_rows = cols[0].shape[0] if cols else len(T)
        base = np.empty((n_rows, sum(c.shape[1] for c in cols)), dtype=float)
        o = 0
        for i in range(len(cols)):
            k = cols[i].shape[1]
            base[:, o:o + k] = cols[i]
            o += k
            cols[i] = None  # 這一塊已複製，立即釋放
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
        is_mask = b["role"] == "mask"
        if not is_mask:
            res.end_pos[b["id"]] = e
            res.bases[b["id"]] = base
        base = base if not is_mask else None
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
        if not is_mask:
            layout_b.append({"id": b["id"], "timeframe": b["timeframe"], "window": w, "encoder": b["encoder"],
                             "cols": [off, off + base.shape[1]]})
            off += base.shape[1]
    for code in ("insufficient_history", "stale_branch", "upstream_unavailable", "warmup_not_ready"):
        reason[hits[code] & (reason == None)] = code  # noqa: E711 — 優先序：歷史不足 > 過舊 > 上游 > 暖機
    res.reason = reason
    res.window_start = win_start
    res.layout = {"version": 1, "w_max": w_max, "f_total": off, "branches": layout_b}
    res.mask_branches = [b["id"] for b in res.branches if b["role"] == "mask"]
    res.branches = [b for b in res.branches if b["role"] == "input"]
    return res


def is_legacy_layout(layout: dict, family: str) -> bool:
    """單一分支、序列編碼器：模型結構等於舊版（不傳 input_layout，權重格式與舊版相同）。"""
    bs = layout["branches"]
    return len(bs) == 1 and bs[0]["encoder"] == DEFAULT_ENCODER.get(family)


# ── 正規化 ───────────────────────────────────────────────────────────────
# 分批處理（2026-11-18 全期間 OOM 修正）：舊寫法會複製一整份 base[mask]，套用時再產生兩份 float64 暫存，
# 2011～2025 tx_1m × 178 欄時峰值約為 base 的 4 倍。改成每批最多 NORM_CHUNK_BYTES 位元組（float64 計），
# 每個值的運算與型別轉換順序不變：min／max 一樣用 fmin／fmax 歸約（np.nanmin／nanmax 對 ndarray 就是這兩個），
# 套用一樣是 float64 的 (x − min) / scale，算完才轉 float32。
NORM_CHUNK_BYTES = 64 * 2 ** 20


def _chunk_rows(n_cols: int) -> int:
    return max(1, NORM_CHUNK_BYTES // max(1, int(n_cols) * 8))


def _masked_minmax(base: np.ndarray, mask: np.ndarray):
    """等同 np.nanmin(base[mask], axis=0)、np.nanmax(base[mask], axis=0)，但不複製整份 base[mask]。"""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        rows = base[idx]  # 空集合：沿用原本的錯誤行為（歸約沒有單位元素）
        return np.nanmin(rows, axis=0), np.nanmax(rows, axis=0)
    step = _chunk_rows(base.shape[1])
    mn = mx = None
    for s in range(0, len(idx), step):
        part = base[idx[s:s + step]]
        pmn, pmx = np.fmin.reduce(part, axis=0), np.fmax.reduce(part, axis=0)
        mn = pmn if mn is None else np.fmin(mn, pmn)
        mx = pmx if mx is None else np.fmax(mx, pmx)
    if np.isnan(mn).any():
        import warnings

        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
    return mn, mx


def fit_normalization(asm: Assembled, train_points: np.ndarray) -> dict:
    state = {"version": 2, "branches": []}
    for b in asm.branches:
        base, e, w = asm.bases[b["id"]], asm.end_pos[b["id"]][train_points], b["window"]
        cover = np.zeros(len(base) + 1, dtype=np.int64)
        np.add.at(cover, e - w + 1, 1)
        np.add.at(cover, e + 1, -1)
        mask = np.cumsum(cover[:-1]) > 0
        steps = []
        minmax = None
        for step in b["normalization"]:
            if step["method"] == "global_minmax":
                if minmax is None:
                    minmax = _masked_minmax(base, mask)
                mn, mx = minmax
                steps.append({"method": "global_minmax", "min": mn.tolist(),
                              "scale": np.where(mx > mn, mx - mn, 1.0).tolist()})
        state["branches"].append({"id": b["id"], "steps": steps})
    return state


def apply_normalization(asm: Assembled, state: dict) -> dict:
    out = {}
    by_id = {s["id"]: s for s in state["branches"]}
    for b in asm.branches:
        base = asm.bases[b["id"]]
        steps = [(np.asarray(s["min"]), np.asarray(s["scale"])) for s in by_id[b["id"]]["steps"]
                 if s["method"] == "global_minmax"]
        res = np.empty(base.shape, dtype=np.float32)
        step = _chunk_rows(base.shape[1] if base.ndim == 2 else 1)
        for s in range(0, len(base), step):
            x = base[s:s + step]
            for mn, sc in steps:
                x = (x - mn) / sc
            res[s:s + step] = x
        out[b["id"]] = res
    return out


def packed(asm: Assembled, normalized: dict, points: np.ndarray) -> PackedWindows:
    parts = []
    for lb in asm.layout["branches"]:
        parts.append((normalized[lb["id"]], np.asarray(asm.end_pos[lb["id"]]), lb["window"], lb["cols"][0]))
    return PackedWindows(parts, asm.layout["w_max"], asm.layout["f_total"], points)


def feature_refs(model_node: dict, nodes: dict) -> list[str]:
    return [r.node for r in input_refs(model_node) if nodes[r.node]["type"] == "feature"]
