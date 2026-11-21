"""特徵庫 164 項實測（純計算，不連資料庫、不寫任何服務狀態）。

路徑：graph_refs.validate_graph_spec（送訓練時的同一個驗證）→ engine_v2.Ctx / build_main_frame / Rows
→ Trainer.assemble（v2_inputs.assemble：compute_feature_v2＝compute_feature→暖機遮罩→轉換）
→ fit_normalization（框架 train 點）→ apply_normalization → packed → np.asarray（訓練輸入矩陣）。
只差最後一步 train_model 不呼叫。
用法：python feature_acceptance.py <輸出資料夾>
"""

import json
import sys
import traceback

import numpy as np
import pandas as pd
import talib
import talib.abstract as ta

from training import engine_v2 as ev
from training import frame as fr
from training import v2_inputs as vi
from training.graph_refs import validate_graph_spec
from training.registry import warmup as wu
from training.registry.features import compute_feature

OUT = sys.argv[1]
TF = "tx_5m"
WINDOW = 10
START, END = "2026-02-02", "2026-04-01"


# ── 合成資料：台指原始價格尺度＋邊界條件 ─────────────────────────────────
def make_data(seed=7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-02", "2026-03-31")
    rows = []
    price = 22000.0
    edge = {}
    for di, d in enumerate(days):
        if di == 20:
            price += 600  # 跳空
        for k in range(60):
            ts = d + pd.Timedelta(hours=8, minutes=45 + 5 * k)
            o = price
            c = round(o + rng.normal(0, 18))
            h = max(o, c) + abs(round(rng.normal(0, 8)))
            lo = min(o, c) - abs(round(rng.normal(0, 8)))
            v = float(rng.integers(800, 6000))
            tag = None
            # 樣本期前（暖機區）與樣本期內都放邊界條件
            if di in (8, 30) and 10 <= k < 34:          # 連續 24 根平盤（O=H=L=C 且不動）
                o = h = lo = c = price
                tag = "flat"
            if di in (9, 32) and 5 <= k < 25:           # 連續 20 根零量（價格照動）
                v = 0.0
                tag = "zero_volume"
            if di in (10, 34) and 20 <= k < 32:         # 平盤＋零量
                o = h = lo = c = price
                v = 0.0
                tag = "flat_zero_volume"
            if di == 36 and k == 0:                     # 單根極端長 K
                h, lo = o + 450, o - 450
                tag = "wide_bar"
            rows.append({"bar_start_ts": ts, "open": o, "high": h, "low": lo, "close": c, "volume": v})
            if tag:
                edge.setdefault(tag, 0)
                edge[tag] += 1
            price = c
    df = pd.DataFrame(rows)
    df["bar_end_ts"] = df["bar_start_ts"] + pd.Timedelta(minutes=5)
    return df, edge


DF, EDGE = make_data()


# ── 圖 ───────────────────────────────────────────────────────────────
LABEL = {"id": "l", "type": "label", "outcome": "log_return", "labeling_rule": "fixed_threshold", "timeframe": TF,
         "params": {"horizon": 1, "n_classes": 2}}
LSTM = {"lstm_layers": [{"units": 4, "dropout": 0.0}], "heads": "single", "shared": {"dense": 4, "dropout": 0.0},
        "optimizer": {"type": "adam", "lr": 1e-3, "weight_decay": 0.0}, "batch_size": 32, "epochs": 1,
        "early_stopping": {"monitor": "val_loss", "patience": 0}, "seed": 0}


def graph(feats: list[dict]) -> dict:
    return {"mode": "v2", "start": START, "end": END, "settings": {"split_seed": 42, "val_ratio": 0.2},
            "nodes": feats + [LABEL, {"id": "m", "type": "model", "key": "lstm", "inputs": [f["id"] for f in feats],
                                      "label": "l", "window": WINDOW, "params": LSTM}]}


def build_matrix(g: dict) -> dict:
    """送訓練前的驗證＋完整組裝到訓練輸入矩陣。驗證不過時拋出 ValueError（訊息即 API 400 的內容）。"""
    validate_graph_spec(g)
    ctx = ev.Ctx(g, {TF: DF})
    main = ev.build_main_frame(ctx, phase=1, split_strategy="random", start=START, end=END)
    tr_ = ev.Trainer(ctx, phase=1, split_strategy="random", start=START, end=END)
    nid = "m"
    ln = ctx.nodes["l"]
    y, task_type, h = ctx.labels("l")
    rows = ev.Rows(ctx.tfs[TF], main, y, h)
    asm = tr_.assemble(nid)
    valid = asm.reason == None  # noqa: E711
    in_frame = np.isin(rows.status, fr.INCLUDED)
    tr_pts = np.nonzero((rows.status == "train") & valid)[0]
    va_pts = np.nonzero((rows.status == "val") & valid)[0]
    excl = pd.Series(asm.reason[in_frame & ~valid]).value_counts().to_dict()
    out = {"n_frame": int(in_frame.sum()), "n_train": len(tr_pts), "n_val": len(va_pts), "excluded": excl,
           "valid_points": set(np.nonzero(in_frame & valid)[0].tolist()), "layout": asm.layout}
    base = np.concatenate([asm.bases[b["id"]] for b in asm.branches], axis=1)
    out["base_inf"] = int(np.isinf(base).sum())
    if len(tr_pts) == 0:
        out["shape"] = None
        return out
    norm = vi.fit_normalization(asm, tr_pts)
    nz = vi.apply_normalization(asm, norm)
    Xtr = np.asarray(vi.packed(asm, nz, tr_pts))
    Xva = np.asarray(vi.packed(asm, nz, va_pts)) if len(va_pts) else np.zeros((0,) + Xtr.shape[1:], np.float32)
    out["shape"] = [list(Xtr.shape), list(Xva.shape)]
    out["nan"] = int(np.isnan(Xtr).sum() + np.isnan(Xva).sum())
    out["inf"] = int(np.isinf(Xtr).sum() + np.isinf(Xva).sum())
    # 訓練點正規化後應在 [0,1]；常數欄（擬合 min==max）另計
    out["const_cols"] = int(sum(np.sum(np.asarray(s["steps"][0]["scale"]) == 1.0) for s in norm["branches"]))
    out["train_range"] = [float(np.nanmin(Xtr)), float(np.nanmax(Xtr))] if Xtr.size else None
    return out


def empirical_bars(node: dict) -> int:
    """比照 POST /warmup/suggest：同一段資料從不同起點開始算，找出差異 ≤ ε·max(std) 的根數（ε 0.01）。"""
    full = np.asarray(compute_feature(node["key"], DF, node.get("params")), dtype=float)
    full = full.reshape(-1, 1) if full.ndim == 1 else full
    n = len(DF)
    tol = float(wu.DEFAULT_EPSILON * np.nanmax(np.nanstd(full, axis=0)))
    parts = {}
    for s in [int(x) for x in np.linspace(n // 10, n // 2, 4)]:
        p = np.asarray(compute_feature(node["key"], DF.iloc[s:].reset_index(drop=True), node.get("params")), dtype=float)
        parts[s] = p.reshape(-1, 1) if p.ndim == 1 else p
    return int(wu.empirical_suggestion(full, parts, tol)["bars"])


# ── 項目清單 ─────────────────────────────────────────────────────────────
ITEMS = []
for g, fs in talib.get_function_groups().items():
    for f in sorted(fs):
        fn = ta.Function(f)
        ins = []
        for v in fn.input_names.values():
            ins += v if isinstance(v, list) else [v]
        ITEMS.append({"name": f, "group": g, "key": "talib_indicator", "params": {"name": f}, "inputs": ins,
                      "param_defaults": dict(fn.parameters), "outputs": list(fn.output_names)})
for k, outs in (("ohlcv", ["open", "high", "low", "close", "volume"]), ("raw_price", ["open", "high", "low", "close"]),
                ("raw_volume", ["volume"])):
    ITEMS.append({"name": k, "group": "Custom", "key": k, "params": {}, "inputs": outs, "param_defaults": {},
                  "outputs": outs})
assert len(ITEMS) == 164, len(ITEMS)


def feat(item, fid="f", **extra):
    return {"id": fid, "type": "feature", "key": item["key"], "timeframe": TF,
            **({"params": item["params"]} if item["params"] else {}), **extra}


def raw_stats(item):
    try:
        v = np.asarray(compute_feature(item["key"], DF, item["params"]), dtype=float)
        v = v.reshape(-1, 1) if v.ndim == 1 else v
        return {"raw_cols": v.shape[1], "raw_nan_rows": int(np.isnan(v).any(axis=1).sum()),
                "raw_inf": int(np.isinf(v).sum()), "raw_all_nan": bool(np.isnan(v).all())}
    except Exception as e:  # noqa: BLE001
        return {"raw_error": f"{type(e).__name__}: {e}"}


results = []
for it in ITEMS:
    r = {k: it[k] for k in ("name", "group", "inputs", "outputs")}
    r["params"] = it["param_defaults"]
    r["n_outputs"] = len(it["outputs"])
    try:
        sug = wu.suggest(it["key"], it["params"])
        r["memory_type"], r["base_bars"], r["needs_researcher"] = sug["memory_type"], sug["base_bars"], sug["needs_researcher"]
    except Exception as e:  # noqa: BLE001
        r["memory_type"], r["suggest_error"] = None, f"{type(e).__name__}: {e}"
    r.update(raw_stats(it))
    # 1) 預設設定直接送
    try:
        m = build_matrix(graph([feat(it)]))
        r["default"] = m
        r["setting"] = None
    except Exception as e:  # noqa: BLE001
        r["default_error"] = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        # 2) 依驗證訊息補設定再送
        extra = {}
        if r.get("memory_type") == "cumulative":
            extra = {"transform": {"type": "diff"}}
        elif r.get("needs_researcher"):
            try:
                eb = empirical_bars(feat(it))
            except Exception as e2:  # noqa: BLE001
                eb = None
                r["empirical_error"] = f"{type(e2).__name__}: {e2}"
            if eb is not None:
                extra = {"warmup": {"bars": int(max(eb, r.get("base_bars") or 0))}}
        if extra:
            r["setting"] = extra
            try:
                r["configured"] = build_matrix(graph([feat(it, **extra)]))
            except Exception as e3:  # noqa: BLE001
                r["configured_error"] = f"{type(e3).__name__}: {e3}".splitlines()[0][:300]
    results.append(r)


def ok(m):
    return m and m.get("shape") and m["n_train"] > 0 and m["n_val"] > 0 and m["nan"] == 0 and m["inf"] == 0 \
        and m["base_inf"] == 0


for r in results:
    if ok(r.get("default")):
        r["verdict"], r["used"] = "可直接使用", r["default"]
    elif ok(r.get("configured")):
        r["verdict"], r["used"] = "設定後可用", r["configured"]
    else:
        r["verdict"] = "需修正"
        r["used"] = r.get("configured") or r.get("default")


# ── 組合測試 ─────────────────────────────────────────────────────────────
def node_for(r, fid):
    it = next(i for i in ITEMS if i["name"] == r["name"])
    return feat(it, fid, **(r.get("setting") or {}))


usable = [r for r in results if r["verdict"] != "需修正"]
combos = {}


def run_combo(name, rs):
    nodes = [node_for(r, f"f{i}") for i, r in enumerate(rs)]
    try:
        m = build_matrix(graph(nodes))
        expect_cols = sum(r["n_outputs"] for r in rs)
        inter = set.intersection(*[r["used"]["valid_points"] for r in rs])
        combos[name] = {"members": [r["name"] for r in rs], "shape": m["shape"], "nan": m.get("nan"),
                        "inf": m.get("inf"), "n_train": m["n_train"], "n_val": m["n_val"],
                        "expected_cols": expect_cols, "cols_match": m["shape"] is not None and m["shape"][0][2] == expect_cols,
                        "valid": len(m["valid_points"]), "intersection_of_singles": len(inter),
                        "intersection_match": m["valid_points"] == inter,
                        "excluded": m["excluded"], "const_cols": m.get("const_cols")}
    except Exception as e:  # noqa: BLE001
        combos[name] = {"members": [r["name"] for r in rs], "error": f"{type(e).__name__}: {e}"[:400]}


for g in list(talib.get_function_groups()) + ["Custom"]:
    run_combo(f"group:{g}", [r for r in usable if r["group"] == g])
pick = ["RSI", "MACD", "BBANDS", "OBV", "ADX", "STOCH", "CDLDOJI", "HT_PHASOR", "ATR", "ohlcv", "LINEARREG_SLOPE",
        "AROON", "MINMAXINDEX"]
run_combo("cross:free_mix", [r for r in usable if r["name"] in pick])
run_combo("cross:all_usable", usable)
multi = [r for r in usable if r["n_outputs"] > 1]
run_combo("multi_output:all", multi)
cust = {r["name"]: r for r in usable if r["group"] == "Custom"}
run_combo("custom:dup_ohlcv+raw_price+raw_volume", [cust["ohlcv"], cust["raw_price"], cust["raw_volume"]])
rsi = next(r for r in usable if r["name"] == "RSI")
run_combo("dup:same_RSI_twice", [rsi, rsi])

# Custom 重複欄位：確認矩陣內 open/high/low/close/volume 兩份完全相同
try:
    g = graph([node_for(cust["ohlcv"], "a"), node_for(cust["raw_price"], "b"), node_for(cust["raw_volume"], "c")])
    validate_graph_spec(g)
    ctx = ev.Ctx(g, {TF: DF})
    asm = ev.Trainer(ctx, phase=1, split_strategy="random", start=START, end=END).assemble("m")
    b = asm.bases["main"]
    combos["custom:dup_ohlcv+raw_price+raw_volume"]["duplicate_columns_identical"] = bool(
        np.array_equal(b[:, 0:4], b[:, 5:9], equal_nan=True) and np.array_equal(b[:, 4], b[:, 9], equal_nan=True))
except Exception as e:  # noqa: BLE001
    combos["custom:dup_ohlcv+raw_price+raw_volume"]["dup_check_error"] = str(e)


def clean(o):
    if isinstance(o, set):
        return len(o)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k != "layout"}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


meta = {"talib": talib.__version__, "numpy": np.__version__, "pandas": pd.__version__, "bars": len(DF),
        "price_range": [float(DF.low.min()), float(DF.high.max())], "edge_rows": EDGE, "start": START, "end": END,
        "window": WINDOW, "timeframe": TF}
json.dump({"meta": meta, "items": clean(results), "combos": clean(combos)},
          open(f"{OUT}/result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
from collections import Counter

print(meta)
print(Counter(r["verdict"] for r in results))
for name, c in combos.items():
    print(name, {k: v for k, v in clean(c).items() if k not in ("members", "excluded")})
