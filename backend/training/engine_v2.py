"""v2 訓練與推論引擎（graph_spec.mode == "v2"，流程版本 2026-11-r2 起）。舊格式的圖仍走 training/graph.py，
結果逐位元組不變；v2 是另一條路徑，所有修正（共同樣本框架、往前取暖機、訓練／推論一致、跨時間框架 as-of、
串接 OOF、多分支）都只在這裡生效，並由流程版本區隔。

訓練（run_train_v2）
  1. 決策時間框架＝最終模型 Label 的時間框架；依實驗層的切分（Phase 1 random／Phase 2 chronological，切分種子
     在 settings.split_seed）建立樣本框架（training/frame.py）。上游模型若決策時間框架或答案定義不同，由下游框架推導。
  2. 依拓樸順序訓練每個 Model Node：輸入用 training/v2_inputs.py 組裝（訓練與推論同一支），框架的 train 點且
     輸入有效的列訓練、val 點且有效的列驗證；無效的點記錄原因代碼。
  3. 被下游使用的上游模型依 upstream_policy 產生樣本外預測（training/cascade.py），下游逐筆檢查防洩漏。
  4. 最終模型的驗證集逐筆結果（best 與 last 兩組權重）、串接時下游訓練樣本的上游來源，回傳給 worker 寫進資料庫。
推論（run_infer_v2）
  依產物保存的圖快照、正規化參數與歷史需求，用同一套組裝；上游用最終上游模型；輸入不足的時間點不預測、記原因。
"""

import hashlib
import json
import time

import numpy as np
import pandas as pd

from training import alignment as al
from training import cascade as cc
from training import frame as fr
from training import v2_inputs as vi
from training.graph_refs import input_refs
from training.holdout_eval import predict_outputs, train_stats
from training.output_specs import DEFAULT_OUTPUT, check_output_shape
from training.provenance import PIPELINE_VERSION
from training.registry.architectures import LAZY_WINDOW_ARCHS, describe_outputs, load_model, train_model
from training.registry.labeling_rules import apply_labeling_rule, validate_combination
from training.registry.outcomes import compute_outcome

NAT = np.datetime64("NaT", "ns")
PREVIEW_THROTTLE_SEC = 1.5


class Ctx:
    def __init__(self, graph_spec: dict, dfs: dict, consume_dfs: bool = False):
        from training.graph import _topo_order

        self.spec = graph_spec
        self.nodes = {n["id"]: n for n in graph_spec["nodes"]}
        self.order = _topo_order(self.nodes)
        self.models = [i for i in self.order if self.nodes[i]["type"] == "model"]
        self.settings = vi.settings_of(graph_spec)
        self.alignment = al.resolve_alignment(self.settings)
        self.eps = vi.epsilon_of(graph_spec)
        self.tfs = {tf: al.TfData(tf, df, al.delay_sec(self.alignment, tf)) for tf, df in dfs.items()}
        if consume_dfs:
            # TfData 已複製需要的欄位（時間、價量、時段），原始 DataFrame（含逐列文字欄位）之後不再使用：
            # 清空呼叫端的 dict 讓它可以被釋放（worker 用；讀取報告在此之前已算完）。2026-11-18 OOM 修正
            dfs.clear()
        self._feat: dict = {}
        self._labels: dict = {}
        self.label_excl: dict = {}
        self.warnings: list = []
        # 特徵快取的釋放（2026-11-18 A3／A4）：每個特徵記住還有幾次引用（所有 Model Node 的所有分支）尚未組裝，
        # 最後一次引用取用後就從快取移除（組裝結果 base 有自己的一份）。feature_computations 記每個特徵實際計算
        # 幾次，正常應該都是 1（測試用來確認沒有因釋放而重算）。
        self._feat_pending: dict = {}
        for m in self.models:
            for b in vi.branches_of(self.nodes[m], self.nodes):
                for item in b["inputs"]:
                    fid = normalize(item)
                    if self.nodes[fid]["type"] == "feature":
                        self._feat_pending[fid] = self._feat_pending.get(fid, 0) + 1
        self.feature_computations: dict = {}

    def feature(self, fid: str) -> np.ndarray:
        if fid not in self._feat:
            n = self.nodes[fid]
            self._feat[fid] = vi.compute_feature_v2(n, self.tfs[n["timeframe"]], self.eps)
            self.feature_computations[fid] = self.feature_computations.get(fid, 0) + 1
        return self._feat[fid]

    def feature_used(self, fid: str) -> None:
        """組裝取用一次特徵後呼叫：沒有其他尚未組裝的引用時釋放快取。"""
        left = self._feat_pending.get(fid, 0) - 1
        self._feat_pending[fid] = left
        if left <= 0:
            self._feat.pop(fid, None)

    def labels(self, lid: str):
        if lid not in self._labels:
            ln = self.nodes[lid]
            d = self.tfs[ln["timeframe"]]
            params = ln.get("params") or {}
            rule = ln.get("labeling_rule", "fixed_threshold")
            validate_combination(ln["outcome"], rule)
            raw, vl = compute_outcome(ln["outcome"], d.ohlcv, params)
            y, task_type, _ = apply_labeling_rule(rule, raw, vl, params)
            y = np.asarray(y, dtype=float)
            h = int(params.get("horizon", 1))
            excl = label_exclusions(d, h, ln["outcome"], params)
            if excl is not None:
                y[excl != None] = np.nan  # noqa: E711
                self.label_excl[lid] = excl
            self._labels[lid] = (y, task_type, h)
        return self._labels[lid]

    def label_reasons(self, lid: str):
        """這個 Label 每列的答案排除原因（見 label_exclusions；沒有要檢查時是 None）。"""
        self.labels(lid)
        return self.label_excl.get(lid)


LABEL_EXCLUSION_ORDER = ("label_gap", "label_cross_session", "label_start_outside")


def label_exclusions(d: "al.TfData", h: int, outcome: str, params: dict):
    """新目標的答案排除（2026-11-21，Dot 規格）：回傳每列的原因代碼（object 陣列，None＝不排除），
    沒有要檢查時回 None（既有目標預設不檢查，行為不變）。只標「T+H 仍在資料內」的列；超出資料尾端的列維持
    label_unavailable／label_unmatured（真值尚未成熟），與下列原因分開：

      label_gap            同一時段內 T→T+H 任一步不符合該週期的時間格：前一根 bar_end_ts ≠ 下一根 bar_start_ts、
                           K 棒長度 ≠ 週期、或起點不在「時段第一根起點 + k×週期」上
      label_cross_session  T～T+H 跨到下一個交易時段（日盤→夜盤、夜盤→日盤、跨日）；cross_session=true 時不排除
                           （相鄰時段之間不檢查時間格，時段內照樣檢查）
      label_start_outside  outcome 宣告的起點條件不成立（first_crossing：T 收盤已在上下界之外）

    日線（tx_daily_*）：每根各自一個時段，H 根＝資料中接下來的 H 個交易日；不檢查時間格與跨時段（缺一整天偵測不到）。
    params.gaps：exclude（新目標預設）／ignore（不做 label_gap 與 label_cross_session）。"""
    from training.registry.outcomes import OUTCOME_EXCLUSIONS, OUTCOME_META

    n = len(d)
    meta = OUTCOME_META.get(outcome, {})
    policy = params.get("gaps") or ("exclude" if meta.get("gap_check") else "ignore")
    extra = OUTCOME_EXCLUSIONS.get(outcome)
    if policy != "exclude" and extra is None:
        return None
    reason = np.full(n, None, dtype=object)
    t = np.arange(n)
    inside = t + h < n
    masks = {}
    if policy == "exclude" and n >= 2 and "daily" not in d.tf:
        period = np.timedelta64(al._nominal_period(d.tf).value, "ns")
        so = d.session_ord
        start, end = d.index.values, d.end
        first = d.session_start[so]  # 每根所屬時段第一根的起點
        grid_ok = ((start - first) % period) == np.timedelta64(0, "ns")
        same = so[1:] == so[:-1]
        step_ok = (end[:-1] == start[1:]) & ((end[:-1] - start[:-1]) == period) & grid_ok[:-1] & grid_ok[1:]
        bad_gap = same & ~step_ok
        bad_cross = ~same if not params.get("cross_session") else np.zeros(n - 1, dtype=bool)
        e = np.minimum(t + h, n - 1)
        for code, bad in (("label_gap", bad_gap), ("label_cross_session", bad_cross)):
            cum = np.concatenate([[0], np.cumsum(bad)])
            masks[code] = ((cum[e] - cum[t]) > 0) & inside
    if extra is not None:
        for code, m in extra(d.ohlcv, params).items():
            masks[code] = np.asarray(m, dtype=bool) & inside
    for code in reversed(LABEL_EXCLUSION_ORDER):  # 優先序：缺格 > 跨時段 > 起點在界外
        if code in masks:
            reason[masks[code]] = code
    return reason


def sample_set(ts) -> dict:
    """樣本集合摘要（2026-11-21）：筆數＋樣本 ID（決策 K 棒 bar_start_ts，奈秒整數、排序後）的 SHA-256。
    兩個任務的 train／val／holdout 雜湊相同＝樣本完全一致。"""
    import hashlib

    a = np.sort(np.asarray(ts, dtype="datetime64[ns]").astype(np.int64))
    return {"n": int(len(a)), "hash": hashlib.sha256(a.tobytes()).hexdigest()}


def label_signature(label_node: dict) -> dict:
    return {"timeframe": label_node["timeframe"], "outcome": label_node["outcome"],
            "labeling_rule": label_node.get("labeling_rule", "fixed_threshold"), "params": label_node.get("params") or {}}


def final_node(ctx: Ctx) -> str:
    from training.phases import final_model_node_id

    return final_model_node_id(list(ctx.nodes.values()))


def frame_definition_extra(ctx: Ctx, label_node: dict) -> dict:
    tf = label_node["timeframe"]
    return {"label": label_signature(label_node), "pipeline_version": PIPELINE_VERSION,
            "decision_delay": al.delay_report(ctx.alignment, [tf])[tf]}


def build_main_frame(ctx: Ctx, *, phase, split_strategy: str, start, end) -> fr.Frame:
    fid = final_node(ctx)
    ln = ctx.nodes[ctx.nodes[fid]["label"]]
    y, _tt, h = ctx.labels(ln["id"])
    s = ctx.settings
    return fr.build_frame(ctx.tfs[ln["timeframe"]], y, h, start, end, phase=phase, split_strategy=split_strategy,
                          val_ratio=float(s.get("val_ratio", ctx.nodes[fid].get("val_ratio", 0.2))),
                          split_seed=int(s.get("split_seed", 42)), definition_extra=frame_definition_extra(ctx, ln),
                          label_reasons=ctx.label_reasons(ln["id"]))


def frame_for(ctx: Ctx, nid: str, main: fr.Frame, main_nid: str, start, end) -> fr.Frame:
    ln = ctx.nodes[ctx.nodes[nid]["label"]]
    main_ln = ctx.nodes[ctx.nodes[main_nid]["label"]]
    if label_signature(ln) == label_signature(main_ln):
        return main
    y, _tt, h = ctx.labels(ln["id"])
    return fr.derive_frame(main, ctx.tfs[ln["timeframe"]], y, h, start, end,
                           definition_extra=frame_definition_extra(ctx, ln), label_reasons=ctx.label_reasons(ln["id"]))


class Rows:
    """一個模型節點在自己決策時間框架「每一列」上的框架狀態、答案時間。"""

    def __init__(self, T: "al.TfData", frame: fr.Frame, y: np.ndarray, h: int):
        n = len(T)
        self.n = n
        self.pos = np.searchsorted(T.index.values, frame.ts)
        if len(frame) and not np.array_equal(T.index.values[np.clip(self.pos, 0, n - 1)], frame.ts):
            raise ValueError(f"{T.tf}：框架時間點對不到資料（資料或框架已變更）")
        self.status = np.full(n, "context", dtype=object)
        self.status[self.pos] = frame.status
        self.frame_reason = np.full(n, None, dtype=object)
        self.frame_reason[self.pos] = frame.reason
        tgt = np.arange(n) + h
        ok = tgt < n
        t = np.where(ok, tgt, 0)
        self.target_ts = np.where(ok, T.index.values[t], NAT)
        self.le = np.where(ok & ~np.isnan(y), T.available[t], NAT)
        self.y = y
        self.tau = T.available
        self.ts = T.index.values


SPAN_QUANTILE = 0.99  # 自動清除範圍用跨度的 99 百分位（排除週末／長假造成的極端值）；超出的樣本由逐筆檢查排除


def _q(td: np.ndarray) -> np.timedelta64:
    if len(td) == 0:
        return np.timedelta64(0, "ns")
    return np.timedelta64(int(np.quantile(td.astype("int64"), SPAN_QUANTILE)), "ns")


def _window_span(T: "al.TfData", w: int) -> np.timedelta64:
    if len(T) < w or w < 1:
        return np.timedelta64(0, "ns")
    return _q(T.available[w - 1:] - T.index.values[:len(T) - w + 1])


def _label_span(T: "al.TfData", h: int) -> np.timedelta64:
    if len(T) <= h:
        return np.timedelta64(0, "ns")
    return _q(T.available[h:] - T.available[:-h])


def _node_span(ctx: Ctx, nid: str) -> np.timedelta64:
    node = ctx.nodes[nid]
    return max((_window_span(ctx.tfs[b["timeframe"]], b["window"]) for b in vi.branches_of(node, ctx.nodes)),
               default=np.timedelta64(0, "ns"))


def _cast_y(y, task_type):
    return np.asarray(y, dtype=(int if task_type == "classification" else float))


def _dual(specs: dict) -> bool:
    return "regression" in specs and "direction_probability" in specs


def _output_arrays(out: dict, node: dict, task_type: str, specs: dict) -> dict:
    """預測結果 → 各輸出名的 (n, k) 陣列（default 的語意與舊版 graph.py 相同）。"""
    arrays = {}
    if out.get("dual") is not None:
        arrays[DEFAULT_OUTPUT] = np.asarray(out["dual"]["regression"]).reshape(-1, 1)
        arrays["regression"] = arrays[DEFAULT_OUTPUT]
        arrays["direction_probability"] = np.asarray(out["dual"]["direction_probability"]).reshape(-1, 1)
    elif task_type == "classification" and node.get("output_mode") == "probability" and out.get("proba") is not None:
        arrays[DEFAULT_OUTPUT] = np.asarray(out["proba"], dtype=float)
    else:
        arrays[DEFAULT_OUTPUT] = np.asarray(out["pred"], dtype=float).reshape(-1, 1)
    return {k: v for k, v in arrays.items() if k in specs or k == DEFAULT_OUTPUT}


def _fingerprint(weights) -> str | None:
    return hashlib.sha256(bytes(weights)).hexdigest() if weights is not None else None


class Trainer:
    def __init__(self, ctx: Ctx, *, phase, split_strategy: str, start, end, on_epoch=None, on_preview=None):
        self.ctx = ctx
        self.phase, self.strategy = phase, split_strategy
        self.start, self.end = start, end
        self.on_epoch, self.on_preview = on_epoch, on_preview
        self.series: dict = {}
        self.fold_tables: dict = {}
        self.results: dict = {}
        self.records: dict = {}
        self.upstream_oof: dict = {}
        self.folds: dict = {}

    # ── 共用：組裝、擬合、預測 ───────────────────────────────────────────
    def assemble(self, nid: str) -> vi.Assembled:
        ctx = self.ctx
        node = ctx.nodes[nid]
        T = ctx.tfs[vi.decision_tf(node, ctx.nodes)]
        return vi.assemble(node, ctx.nodes, ctx.tfs, T.available, ctx.feature, self.series, ctx.alignment,
                           release=ctx.feature_used)

    def fit(self, nid, asm, rows: Rows, tr, va, task_type, n_classes, *, emit: bool, preview_rows=None):
        node = self.ctx.nodes[nid]
        norm = vi.fit_normalization(asm, tr)
        nz = vi.apply_normalization(asm, norm)
        Xtr, Xva = vi.packed(asm, nz, tr), vi.packed(asm, nz, va)
        if node["key"] not in LAZY_WINDOW_ARCHS:
            Xtr, Xva = np.asarray(Xtr), np.asarray(Xva)
        layout = None if vi.is_legacy_layout(asm.layout, vi.family_of(node)) else asm.layout
        on_epoch = (lambda e, m: self.on_epoch(nid, e, m)) if (emit and self.on_epoch) else None
        hook = self._preview_hook(nid, asm, rows, tr, task_type) if emit else None
        res = train_model(node["key"], Xtr, _cast_y(rows.y[tr], task_type), Xva, _cast_y(rows.y[va], task_type),
                          node.get("params", {}), on_epoch, task_type=task_type, preview_hook=hook,
                          n_classes=n_classes, input_layout=layout)
        return res, norm, nz

    def _preview_hook(self, nid, asm, rows: Rows, tr, task_type):
        if self.on_preview is None:
            return None
        ctx = self.ctx
        b0 = asm.branches[0]
        T0 = ctx.tfs[b0["timeframe"]]
        if not {"open", "high", "low", "close"}.issubset(T0.ohlcv.columns):
            return None
        node = ctx.nodes[nid]
        ln = ctx.nodes[node["label"]]
        lp = ln.get("params") or {}
        last = [0.0]

        def hook(epoch, local_idx, predicted, source="train"):
            now = time.time()
            if now - last[0] < PREVIEW_THROTTLE_SEC or local_idx >= len(tr):
                return
            last[0] = now
            try:
                r = int(tr[local_idx])
                e = int(asm.end_pos[b0["id"]][r])
                lo = e - b0["window"] + 1
                bars = T0.ohlcv.iloc[lo:e + 1]
                self.on_preview(nid, {
                    "epoch": epoch, "source": source,
                    "decision_ts": int(pd.Timestamp(rows.ts[r]).timestamp()),
                    "target_ts": int(pd.Timestamp(rows.target_ts[r]).timestamp()) if not np.isnat(rows.target_ts[r]) else None,
                    "decision_available_ts": int(pd.Timestamp(rows.tau[r]).timestamp()),
                    "target_available_ts": int(pd.Timestamp(rows.le[r]).timestamp()) if not np.isnat(rows.le[r]) else None,
                    "horizon": int(lp.get("horizon", 1)), "task_type": task_type,
                    "n_classes": int(lp.get("n_classes", 2)) if task_type == "classification" else None,
                    "labeling_rule": ln.get("labeling_rule"),
                    "bars": [{"time": int(pd.Timestamp(t).timestamp()), "open": float(o), "high": float(h),
                              "low": float(l), "close": float(c)}
                             for t, o, h, l, c in zip(bars.index, bars["open"], bars["high"], bars["low"], bars["close"])],
                    "actual": int(rows.y[r]) if task_type == "classification" else float(rows.y[r]),
                    "predicted": predicted,
                })
            except Exception:
                pass

        return hook

    @staticmethod
    def predict(pred: dict, asm, nz, rows_idx, task_type, dual: bool) -> dict:
        return predict_outputs(pred, vi.packed(asm, nz, rows_idx), task_type, dual)

    # ── 主流程 ───────────────────────────────────────────────────────────
    def run(self, main: fr.Frame) -> dict:
        ctx = self.ctx
        main_nid = final_node(ctx)
        consumers = {m: [] for m in ctx.models}
        for m in ctx.models:
            for r in input_refs(ctx.nodes[m]):
                if r.node in consumers:
                    consumers[r.node].append(m)
        for nid in ctx.models:
            self.train_node(nid, frame_for(ctx, nid, main, main_nid, self.start, self.end), consumers[nid],
                            is_final=(nid == main_nid), main=main)
        return self.results

    def train_node(self, nid: str, frame: fr.Frame, consumers: list, *, is_final: bool, main: fr.Frame):
        ctx = self.ctx
        node = ctx.nodes[nid]
        label_node = ctx.nodes[node["label"]]
        y, task_type, h = ctx.labels(label_node["id"])
        T = ctx.tfs[label_node["timeframe"]]
        rows = Rows(T, frame, y, h)
        n_classes = int((label_node.get("params") or {}).get("n_classes", 2)) if task_type == "classification" else None
        specs = describe_outputs(node["key"], node, label_node, task_type)
        dual = _dual(specs)
        asm = self.assemble(nid)

        # 下游：逐筆防洩漏檢查（上游用 in_sample 時只提示、不排除）
        upstream_used = {}
        for bid, ups in asm.upstream.items():
            for u, info in ups.items():
                lo, e, fold_rows = info["folds_in_window"]
                fids = [int(f) for f in np.unique(fold_rows)]
                used = cc.folds_in_window(fold_rows, lo, e, fids)
                prev = upstream_used.setdefault(u, {})
                for f, m in used.items():
                    prev[f] = prev.get(f, np.zeros(len(m), bool)) | m
        purged = np.zeros(rows.n, dtype=bool)
        for u, used in upstream_used.items():
            policy = cc.resolve_policy(ctx.nodes[u], ctx.settings)
            if policy["strategy"] == "in_sample":
                continue
            bad = cc.check_downstream(is_train=rows.status == "train", ws=asm.window_start, tau=rows.tau, le=rows.le,
                                      used=used, folds=self.fold_tables[u], strategy=policy["strategy"])
            purged |= bad
        in_frame = np.isin(rows.status, fr.INCLUDED)
        asm.reason[purged & in_frame & (asm.reason == None)] = "upstream_purged"  # noqa: E711
        valid = asm.reason == None  # noqa: E711
        tr = np.nonzero((rows.status == "train") & valid)[0]
        va = np.nonzero((rows.status == "val") & valid)[0]
        if len(tr) == 0 or len(va) == 0:
            raise ValueError(f"節點 {nid}：框架內可用的訓練點 {len(tr)}、驗證點 {len(va)}，至少各要 1 筆"
                             f"（排除原因：{self._excl(rows, asm)}）")

        policy = cc.resolve_policy(node, ctx.settings)
        cascade_meta = None
        if consumers and policy["strategy"] != "in_sample":
            res, norm, cascade_meta = self.train_upstream(nid, node, asm, rows, tr, task_type, n_classes, specs, dual,
                                                          policy, consumers)
            eval_note = "最終上游模型的折內驗證段（從框架 train 點切出；框架 val 不進上游訓練）"
        else:
            res, norm, nz = self.fit(nid, asm, rows, tr, va, task_type, n_classes, emit=True)
            eval_note = "框架 val 點"
            if consumers:
                ctx.warnings.append({"code": "upstream_in_sample", "node": nid,
                                     "message": f"上游模型 {nid} 使用 in_sample（舊行為），下游看到的是樣本內預測，結果可能虛高"})
                ok_rows = np.nonzero(valid)[0]
                out = self.predict(res, asm, nz, ok_rows, task_type, dual)
                self._set_series(nid, T, _output_arrays(out, node, task_type, specs), ok_rows, 0, rows.n)
                self.fold_tables[nid] = {}
            nz = None  # 之後的驗證逐筆結果會依 preprocessing_state 重算，先放掉這份，避免同時存兩份 float32
        stats = train_stats(task_type, rows.y[tr], n_classes, node.get("params"))
        sample_sets = {"train": sample_set(rows.ts[tr]), "val": sample_set(rows.ts[va]),
                       "mask_branches": list(getattr(asm, "mask_branches", []))}
        excl = self._excl(rows, asm)
        n_frame = int(in_frame.sum())
        n_bad = int((in_frame & ~valid).sum())
        ratio = n_bad / n_frame if n_frame else 0.0
        warn_ratio = float(ctx.settings.get("exclusion_warn_ratio", 0.05))
        if ratio > warn_ratio:
            ctx.warnings.append({"code": "exclusion_ratio_high", "node": nid,
                                 "message": f"節點 {nid}：框架內 {n_bad}/{n_frame}（{ratio:.1%}）個時間點因輸入不足被排除，"
                                            f"超過警示門檻 {warn_ratio:.0%}；原因 {excl}"})
        warmups = {fid: vi.feature_warmup(ctx.nodes[fid], ctx.eps)
                   for b in asm.branches for fid in [normalize(i) for i in b["inputs"]] if ctx.nodes[fid]["type"] == "feature"}
        for fid, w in warmups.items():
            if (w["transform"] or {}).get("type") == "legacy_cumulative":
                ctx.warnings.append({"code": "cumulative_load_dependent", "node": fid,
                                     "message": f"特徵 {fid} 使用 legacy_cumulative：數值與資料載入起點有關，訓練與推論只保證同一載入規則下一致"})
        res["task_type"] = task_type
        res["preprocessing_state"] = norm
        res["output_specs"] = specs
        res.setdefault("model_config", {})["output_specs"] = specs
        res["train_stats"] = stats
        res["training_meta"] = {
            "sample_sets": sample_sets,
            "split_strategy": frame.definition.get("split_strategy"), "pipeline": "v2",
            "n_train": int(len(tr)), "n_val": int(len(va)),
            "n_excluded_boundary": int((rows.frame_reason == "boundary").sum()),
            "n_samples": n_frame, "val_ratio": frame.definition.get("val_ratio"),
            "split_seed": frame.definition.get("split_seed"),
            "preprocessing": "v2 steps: " + ", ".join(sorted({s["method"] for b in asm.branches for s in b["normalization"]})),
            "evaluation_dataset": eval_note,
            "frame": {"hash": frame.hash, "counts": frame.counts(), "derived": "derived_from" in frame.definition},
            "exclusions": excl, "exclusion_ratio": ratio,
            "warmup": warmups,
            "branches": self._branch_summary(asm, valid & in_frame),
            "time_semantics": {"sample_ts": "bar_start_ts_of_decision_bar", "available_ts": True, "target_available_ts": True},
            **({"cascade": cascade_meta} if cascade_meta else {}),
        }
        res["v2"] = {"input_layout": asm.layout, "warmup_policy": warmups, "frame_hash": frame.hash,
                     "model_input_format": ("legacy_single_branch" if vi.is_legacy_layout(asm.layout, vi.family_of(node))
                                            else "input_layout"),
                     "mask_branches": list(getattr(asm, "mask_branches", [])),
                     "alignment": ctx.alignment, "decision_timeframe": T.tf,
                     "history_requirements": vi.history_requirements(ctx.spec),
                     "weights_fingerprint": _fingerprint(res.get("weights")),
                     "upstream_policy": policy if consumers else None}
        self.results[nid] = res
        if is_final:
            self.records[nid] = self._val_records(nid, node, asm, rows, va, res, task_type, dual, frame)
            if upstream_used:
                self.records[nid]["train_upstream"] = self._train_upstream_records(asm, rows, tr)

    def train_upstream(self, nid, node, asm, rows: Rows, tr, task_type, n_classes, specs, dual, policy, consumers):
        ctx = self.ctx
        T = ctx.tfs[vi.decision_tf(node, ctx.nodes)]
        valid_rows = np.nonzero(asm.reason == None)[0]  # noqa: E711
        d_span = max(_node_span(ctx, c) + _label_span(ctx.tfs[vi.decision_tf(ctx.nodes[c], ctx.nodes)],
                                                      ctx.labels(ctx.nodes[c]["label"])[2]) for c in consumers)
        purge = cc.auto_purge(d_span, _node_span(ctx, nid), policy["extra_purge_sec"])
        plan = cc.fold_plan(policy, predict_rows=valid_rows, train_rows=tr, ws=asm.window_start, tau=rows.tau,
                            le=rows.le, purge=purge)
        inner = "chronological" if self.strategy == "chronological" else "random"
        val_ratio = float(ctx.settings.get("val_ratio", node.get("val_ratio", 0.2)))
        seed = int(ctx.settings.get("split_seed", 42))
        values: dict = {}
        fold_of_row = np.full(rows.n, cc.FOLD_NONE, dtype=np.int64)
        table, meta, last = {}, [], None
        for f in plan:
            itr, iva = cc.inner_split(f["train_rows"], rows.ts, rows.target_ts, inner, val_ratio, seed)
            entry = {"fold": f["fold"], "strategy": policy["strategy"], "purge_start": f["purge_start"],
                     "purge_end": f["purge_end"], "predict_start": rows.tau[f["predict_rows"]].min(),
                     "predict_end": rows.tau[f["predict_rows"]].max(), "n_pred": 0}
            used_rows = np.r_[itr, iva].astype(np.int64)
            entry.update(cc.fold_bounds(used_rows, asm.window_start, rows.tau, rows.le, (f["purge_start"], f["purge_end"])))
            if len(itr) == 0 or len(iva) == 0:
                entry["skipped"] = "折內訓練點不足"
                meta.append(entry)
                table[f["fold"]] = entry
                continue
            res, norm, nz = self.fit(nid, asm, rows, itr, iva, task_type, n_classes, emit=False)
            out = self.predict(res, asm, nz, f["predict_rows"], task_type, dual)
            for name, arr in _output_arrays(out, node, task_type, specs).items():
                values.setdefault(name, np.full((rows.n, arr.shape[1]), np.nan))[f["predict_rows"]] = arr
            fold_of_row[f["predict_rows"]] = f["fold"]
            nz = None  # 下一折會重算，先放掉這一折的 float32
            entry["n_pred"] = int(len(f["predict_rows"]))
            entry["fingerprint"] = _fingerprint(res.get("weights"))
            meta.append(entry)
            table[f["fold"]] = entry
            last = (res, norm)
            print(f"[engine_v2] 上游 {nid} 折 {f['fold']}：訓練 {len(itr)}／折內驗證 {len(iva)}、預測 {len(f['predict_rows'])}",
                  flush=True)
        if policy["final"] == "last_fold" and last is not None:
            res, norm = last
            final_rows = None
        else:
            itr, iva = cc.inner_split(tr, rows.ts, rows.target_ts, inner, val_ratio, seed)
            res, norm, _ = self.fit(nid, asm, rows, itr, iva, task_type, n_classes, emit=True)
            final_rows = np.r_[itr, iva].astype(np.int64)
        fin = {"fold": cc.FOLD_FINAL, "strategy": policy["strategy"], "final": policy["final"], "n_pred": 0,
               "purge_start": None, "purge_end": None, "predict_start": None, "predict_end": None,
               "fingerprint": _fingerprint(res.get("weights"))}
        if final_rows is not None:
            fin.update(cc.fold_bounds(final_rows, asm.window_start, rows.tau, rows.le, None))
        meta.append(fin)
        table[cc.FOLD_FINAL] = fin
        if not values:  # 所有折都跳過：下游沒有任何樣本外預測
            values = {DEFAULT_OUTPUT: np.full((rows.n, 1), np.nan)}
        self._set_series_full(nid, T, values, fold_of_row)
        if policy.get("save_oof"):
            self.upstream_oof[nid] = {"rows": rows, "asm": asm, "values": values, "fold": fold_of_row,
                                      "task_type": task_type,
                                      "fold_info": {k: {"train_end": v.get("train_end"),
                                                        "train_max_label_end": v.get("train_max_label_end")}
                                                    for k, v in table.items()}}
        self.fold_tables[nid] = table
        self.folds[nid] = {"consumers": consumers, "policy": policy, "purge_sec": float(purge / np.timedelta64(1, "s")),
                           "folds": meta}
        return res, norm, {"policy": policy, "purge_sec": float(purge / np.timedelta64(1, "s")),
                           "n_folds": len(plan), "n_rows_with_oof": int((fold_of_row > 0).sum()),
                           "n_rows_without_oof": int(len(valid_rows) - (fold_of_row > 0).sum())}

    def _set_series(self, nid, T, arrays: dict, rows_idx, fold_id, n):
        values = {k: np.full((n, v.shape[1]), np.nan) for k, v in arrays.items()}
        for k, v in arrays.items():
            values[k][rows_idx] = v
        fold = np.full(n, cc.FOLD_NONE, dtype=np.int64)
        fold[rows_idx] = fold_id
        self._set_series_full(nid, T, values, fold)

    def _set_series_full(self, nid, T, values: dict, fold):
        for name, arr in values.items():
            self.series[(nid, name)] = {"available": T.available, "values": arr, "fold": fold}

    def _excl(self, rows: Rows, asm) -> dict:
        out = {}
        for st in ("train", "val", "holdout"):
            m = rows.status == st
            if not m.any():
                continue
            r = pd.Series(asm.reason[m])
            out[st] = {str(k): int(v) for k, v in r[r.notna()].value_counts().items()}
        fr_r = pd.Series(rows.frame_reason[rows.frame_reason != None])  # noqa: E711
        out["frame"] = {str(k): int(v) for k, v in fr_r.value_counts().items()}
        return out

    @staticmethod
    def _branch_summary(asm, mask) -> dict:
        out = {}
        for b in asm.branches:
            m = asm.branch_meta[b["id"]]
            back = m["sessions_back"][mask]
            age = m["age_sec"][mask]
            out[b["id"]] = {"timeframe": b["timeframe"], "window": b["window"], "encoder": b["encoder"],
                            "n_from_previous_session": int((back > 0).sum()),
                            "max_sessions_back": int(back.max()) if len(back) else None,
                            "max_age_sec": float(np.nanmax(age)) if len(age) and not np.all(np.isnan(age)) else None,
                            "median_age_sec": float(np.nanmedian(age)) if len(age) and not np.all(np.isnan(age)) else None}
        return out

    def _val_records(self, nid, node, asm, rows: Rows, va, res, task_type, dual, frame) -> dict:
        out = {"dataset": "val", "points": va, "rows": rows, "asm": asm, "frame_hash": frame.hash, "by_weights": {}}
        nz = vi.apply_normalization(asm, res["preprocessing_state"])
        out["by_weights"]["best"] = self.predict(res, asm, nz, va, task_type, dual)
        if res.get("weights_last") is not None:
            last = load_model(node["key"], bytes(res["weights_last"]), res["model_config"])
            out["by_weights"]["last"] = self.predict(last, asm, nz, va, task_type, dual)
        skipped = np.nonzero((rows.status == "val") & (asm.reason != None))[0]  # noqa: E711
        out["skipped"] = skipped
        return out

    @staticmethod
    def _train_upstream_records(asm, rows: Rows, tr) -> dict:
        return {"points": tr}


def normalize(item) -> str:
    from training.graph_refs import normalize_input_ref

    return normalize_input_ref(item).node


def run_train_v2(graph_spec: dict, dfs: dict, *, phase, on_epoch=None, on_preview=None, frame_hook=None,
                 consume_dfs: bool = False):
    """訓練入口；回傳 (results, frame, extras)。frame_hook(frame) 在訓練前呼叫（worker 用來核對／保存框架）。"""
    ctx = Ctx(graph_spec, dfs, consume_dfs=consume_dfs)
    fid = final_node(ctx)
    strategy = ctx.nodes[fid].get("split_strategy", "random")
    main = build_main_frame(ctx, phase=phase, split_strategy=strategy, start=graph_spec["start"], end=graph_spec["end"])
    if frame_hook is not None:
        frame_hook(main)
    trainer = Trainer(ctx, phase=phase, split_strategy=strategy, start=graph_spec["start"], end=graph_spec["end"],
                      on_epoch=on_epoch, on_preview=on_preview)
    results = trainer.run(main)
    return results, main, {"records": trainer.records, "folds": trainer.folds, "warnings": ctx.warnings, "ctx": ctx,
                           "upstream_oof": trainer.upstream_oof}


# ── 推論（Phase 3 與一般推論）─────────────────────────────────────────────
def run_infer_v2(graph_spec: dict, artifacts: dict, dfs: dict, start, end, *, weights=("best", "last"),
                 consume_dfs: bool = False) -> dict:
    """artifacts：{model_node_id: load_artifact(...) 的 dict}（整條依賴鏈）。回傳最終節點的逐點預測、跳過原因與框架。"""
    ctx = Ctx(graph_spec, dfs, consume_dfs=consume_dfs)
    fid = final_node(ctx)
    ln = ctx.nodes[ctx.nodes[fid]["label"]]
    y, task_type, h = ctx.labels(ln["id"])
    frame = fr.build_frame(ctx.tfs[ln["timeframe"]], y, h, start, end, phase=3, split_strategy="holdout", val_ratio=0,
                           split_seed=0, definition_extra=frame_definition_extra(ctx, ln), label_reasons=ctx.label_reasons(ln["id"]))
    series: dict = {}
    out: dict = {"frame": frame, "node_id": fid, "task_type": task_type}
    for nid in ctx.models:
        node = ctx.nodes[nid]
        art = artifacts[nid]
        T = ctx.tfs[vi.decision_tf(node, ctx.nodes)]
        asm = vi.assemble(node, ctx.nodes, ctx.tfs, T.available, ctx.feature, series, ctx.alignment,
                          release=ctx.feature_used)
        state = art.get("preprocessing_state") or {}
        if state.get("version") != 2:
            raise ValueError(f"節點 {nid} 的產物不是 v2 格式（preprocessing_state.version={state.get('version')!r}）")
        nz = vi.apply_normalization(asm, state)
        mc = art["model_config"]
        specs = mc.get("output_specs") or {}
        dual = _dual(specs)
        tt = art["task_type"]
        if nid != fid:
            ok = np.nonzero(asm.reason == None)[0]  # noqa: E711
            loaded = load_model(art["architecture_key"], bytes(art["weights"]), mc)
            pred = predict_outputs(loaded, vi.packed(asm, nz, ok), tt, dual)
            arrays = _output_arrays(pred, node, tt, specs)
            fold = np.full(len(T), cc.FOLD_NONE, dtype=np.int64)
            fold[ok] = cc.FOLD_FINAL
            for name, arr in arrays.items():
                full = np.full((len(T), arr.shape[1]), np.nan)
                full[ok] = arr
                series[(nid, name)] = {"available": T.available, "values": full, "fold": fold}
            continue
        rows = Rows(T, frame, y, h)
        pts = rows.pos
        valid = asm.reason[pts] == None  # noqa: E711
        use = pts[valid]
        out.update({"rows": rows, "asm": asm, "points": use, "skipped": pts[~valid], "dual": dual,
                    "model_config": mc, "by_weights": {}})
        for w in weights:
            blob = art["weights"] if w == "best" else art.get("weights_last")
            if blob is None:
                raise ValueError(f"節點 {nid} 的產物沒有保存 {w} 權重")
            loaded = load_model(art["architecture_key"], bytes(blob), mc)
            out["by_weights"][w] = predict_outputs(loaded, vi.packed(asm, nz, use), tt, dual)
    out["warnings"] = ctx.warnings
    return out


def branch_meta_json(asm, r: int) -> str:
    d = {}
    for b in asm.branches:
        m = asm.branch_meta[b["id"]]
        back = int(m["sessions_back"][r])
        d[b["id"]] = {"last_ts": None if np.isnat(m["last_ts"][r]) else str(m["last_ts"][r])[:19],
                      "available_ts": None if np.isnat(m["available"][r]) else str(m["available"][r])[:19],
                      "age_sec": None if np.isnan(m["age_sec"][r]) else float(m["age_sec"][r]),
                      "sessions_back": back, "previous_session": back > 0}
    return json.dumps(d, ensure_ascii=False)


def upstream_json(asm, r: int) -> str | None:
    if not asm.upstream:
        return None
    d = {}
    for bid, ups in asm.upstream.items():
        for u, info in ups.items():
            lo, e, fold_rows = info["folds_in_window"]
            used = sorted({int(x) for x in fold_rows[lo[r]:e[r] + 1]})
            av = info["avail_at"][r]
            d[u] = {"branch": bid, "fold_at_decision": int(info["fold_at"][r]), "folds_in_window": used,
                    "pred_available_ts": None if np.isnat(av) else str(av)[:19]}
    return json.dumps(d)
