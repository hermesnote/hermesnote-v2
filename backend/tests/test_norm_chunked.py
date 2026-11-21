"""正規化分批（2026-11-18 OOM 修正 A1／A2）與特徵快取釋放（A3）：新舊流程逐位元比對。

舊版函式原樣保留在這裡當對照組；比對用 uint64／uint32 位元檢視（NaN 位置、±0 都會被抓到），
逐欄回報差異筆數、最大絕對差、是否只是 ±0。端到端部分把整個 run_train_v2 換回舊函式、關閉快取釋放再跑一次，
比對正規化參數、權重、驗證逐筆預測與上游樣本外預測。

執行：python -m unittest tests.test_norm_chunked
"""

import os
import sys
import unittest
import warnings
from unittest import mock

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

import tests  # noqa: E402,F401 — 正式 hermesnote DB 防護（tests/__init__.py），必須在 training／services 之前
from training import engine_v2 as ev  # noqa: E402
from training import v2_inputs as vi  # noqa: E402


# ── 舊版（2026-11-16 部署版本，原樣） ──────────────────────────────────────
def legacy_fit_normalization(asm, train_points):
    state = {"version": 2, "branches": []}
    for b in asm.branches:
        base, e, w = asm.bases[b["id"]], asm.end_pos[b["id"]][train_points], b["window"]
        cover = np.zeros(len(base) + 1, dtype=np.int64)
        np.add.at(cover, e - w + 1, 1)
        np.add.at(cover, e + 1, -1)
        mask = np.cumsum(cover[:-1]) > 0
        rows = base[mask]
        steps = []
        for step in b["normalization"]:
            if step["method"] == "global_minmax":
                mn, mx = np.nanmin(rows, axis=0), np.nanmax(rows, axis=0)
                steps.append({"method": "global_minmax", "min": mn.tolist(),
                              "scale": np.where(mx > mn, mx - mn, 1.0).tolist()})
        state["branches"].append({"id": b["id"], "steps": steps})
    return state


def legacy_apply_normalization(asm, state):
    out = {}
    by_id = {s["id"]: s for s in state["branches"]}
    for b in asm.branches:
        x = asm.bases[b["id"]]
        for step in by_id[b["id"]]["steps"]:
            if step["method"] == "global_minmax":
                x = (x - np.asarray(step["min"])) / np.asarray(step["scale"])
        out[b["id"]] = np.asarray(x, dtype=np.float32)
    return out


# ── 比對工具 ─────────────────────────────────────────────────────────────
def bits(a):
    a = np.ascontiguousarray(np.asarray(a))
    return a.view(np.uint64 if a.dtype.itemsize == 8 else np.uint32)


def compare(old, new) -> dict:
    """逐欄位元比對。回傳 {identical, n_diff, signed_zero_only, max_abs, cols}。"""
    old, new = np.asarray(old), np.asarray(new)
    assert old.shape == new.shape and old.dtype == new.dtype, (old.shape, new.shape, old.dtype, new.dtype)
    o2, n2 = old.reshape(len(old), -1) if old.ndim > 1 else old.reshape(-1, 1), \
        new.reshape(len(new), -1) if new.ndim > 1 else new.reshape(-1, 1)
    diff = bits(o2) != bits(n2)
    out = {"identical": not diff.any(), "n_diff": int(diff.sum()), "cols": sorted(set(np.nonzero(diff)[1].tolist()))}
    if diff.any():
        a, b = o2[diff].astype(float), n2[diff].astype(float)
        out["signed_zero_only"] = bool(np.all((a == 0) & (b == 0)))
        with np.errstate(invalid="ignore"):
            out["max_abs"] = float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else None
    return out


class FakeAsm:
    def __init__(self, base, window=3, norm=None):
        n = len(base)
        self.branches = [{"id": "main", "window": window, "normalization": norm or [{"method": "global_minmax"}]}]
        self.bases = {"main": base}
        self.end_pos = {"main": np.arange(n)}


def edge_bases():
    rng = np.random.default_rng(0)
    n = 1003
    cases = {}
    cases["normal"] = rng.normal(0, 1, (n, 7))
    tx = rng.normal(22000, 300, (n, 3))
    cases["tx_scale"] = tx
    b = rng.normal(0, 1, (n, 6))
    b[:, 0] = np.nan                       # 整欄 NaN
    b[:, 1] = 5.0                          # 整欄常數
    b[rng.random(n) < 0.3, 2] = np.nan     # 部分 NaN
    b[:, 3] = 0.0
    b[::2, 3] = -0.0                       # ±0 交錯
    b[:5, 4] = [np.inf, -np.inf, 1e308, -1e308, 5e-324]  # Inf、極值、次正規數
    b[:, 5] = np.where(rng.random(n) < 0.5, 1e-310, -1e-310)  # 次正規數
    cases["edge"] = b
    z = np.zeros((n, 2))
    z[:, 1] = -0.0
    z[500:, 0] = -0.0
    cases["signed_zero"] = z
    cases["one_col"] = rng.normal(0, 1, (n, 1))
    return cases


class ChunkedNormalizationTests(unittest.TestCase):
    def check(self, base, train_points, norm=None, window=3):
        asm = FakeAsm(base, window, norm)
        ref_state = legacy_fit_normalization(asm, train_points)
        ref = legacy_apply_normalization(asm, ref_state)["main"]
        for chunk_rows in (1, 7, 64, 1000, 10 ** 6):  # 每批列數：1、不整除、整除附近、大於 n
            with mock.patch.object(vi, "NORM_CHUNK_BYTES", chunk_rows * base.shape[1] * 8):
                st = vi.fit_normalization(asm, train_points)
                for s_old, s_new in zip(ref_state["branches"][0]["steps"], st["branches"][0]["steps"]):
                    for k in ("min", "scale"):
                        r = compare(np.array(s_old[k]), np.array(s_new[k]))
                        self.assertTrue(r["identical"], (chunk_rows, k, r))
                got = vi.apply_normalization(asm, st)["main"]
                r = compare(ref, got)
                self.assertTrue(r["identical"], (chunk_rows, r))
                # 對照組用舊參數套新函式：確認差異不是來自 fit
                r2 = compare(ref, vi.apply_normalization(asm, ref_state)["main"])
                self.assertTrue(r2["identical"], (chunk_rows, r2))

    def test_edge_cases_bitwise(self):
        rng = np.random.default_rng(1)
        for name, base in edge_bases().items():
            n = len(base)
            # 訓練點至少要有完整窗格（組裝已排除歷史不足的點，end_pos < window−1 不會出現）
            for tp in (np.arange(2, n), np.sort(rng.choice(np.arange(2, n), n // 3, replace=False)),
                       np.array([2, 3, 500]), np.array([n - 1])):
                with self.subTest(case=name, n_train=len(tp)):
                    self.check(base, tp)

    def test_mask_with_gaps_and_windows(self):
        base = np.random.default_rng(2).normal(0, 1, (999, 4))
        tp = np.r_[np.arange(10, 50), np.arange(400, 420), np.arange(990, 999)]
        for w in (1, 5, 60):
            self.check(base, tp, window=w)

    def test_two_steps_and_no_steps(self):
        base = np.random.default_rng(3).normal(5, 2, (300, 3))
        self.check(base, np.arange(300), norm=[{"method": "global_minmax"}, {"method": "global_minmax"}])
        self.check(base, np.arange(300), norm=[])

    def test_empty_train_points_same_error(self):
        base = np.random.default_rng(4).normal(0, 1, (50, 2))
        asm = FakeAsm(base)
        with self.assertRaises(ValueError) as old:
            legacy_fit_normalization(asm, np.array([], dtype=np.int64))
        with self.assertRaises(ValueError) as new:
            vi.fit_normalization(asm, np.array([], dtype=np.int64))
        self.assertEqual(type(old.exception), type(new.exception))

    def test_empty_base(self):
        asm = FakeAsm(np.zeros((0, 3)))
        st = {"version": 2, "branches": [{"id": "main", "steps": [{"method": "global_minmax", "min": [0, 0, 0],
                                                                    "scale": [1, 1, 1]}]}]}
        self.assertEqual(vi.apply_normalization(asm, st)["main"].shape, (0, 3))


# ── 端到端：多模型、串接、多分支、推論 ─────────────────────────────────────
from tests.v2_synth import market  # noqa: E402

MK = market(days=15)


def _xgb(**over):
    p = {"n_estimators": 4, "max_depth": 2, "learning_rate": 0.3, "early_stopping": {"patience": 0}, "seed": 0}
    p.update(over)
    return p


def _lstm():
    return {"lstm_layers": [{"units": 4, "dropout": 0.0}], "heads": "single", "shared": {"dense": 4, "dropout": 0.0},
            "optimizer": {"type": "adam", "lr": 3e-3, "weight_decay": 0.0}, "batch_size": 32, "epochs": 2,
            "early_stopping": {"monitor": "val_loss", "patience": 0}, "seed": 0}


def graph(strategy, model="xgboost", multi_branch=False):
    lab = {"id": "l", "type": "label", "outcome": "log_return", "labeling_rule": "fixed_threshold",
           "timeframe": "tx_5m", "params": {"horizon": 1, "n_classes": 2}}
    feats = [{"id": "f", "type": "feature", "key": "ohlcv", "timeframe": "tx_5m"},
             {"id": "r", "type": "feature", "key": "talib_indicator", "timeframe": "tx_5m",
              "params": {"name": "RSI", "timeperiod": 5}},
             {"id": "b", "type": "feature", "key": "talib_indicator", "timeframe": "tx_5m",
              "params": {"name": "BBANDS", "timeperiod": 5}},
             # 累積型＋轉換、複合型＋研究者暖機（暖機與轉換路徑）
             {"id": "o", "type": "feature", "key": "talib_indicator", "timeframe": "tx_5m",
              "params": {"name": "OBV"}, "transform": {"type": "diff"}},
             {"id": "mc", "type": "feature", "key": "talib_indicator", "timeframe": "tx_5m",
              "params": {"name": "MACD"}, "warmup": {"bars": 40}}]
    params = _xgb() if model == "xgboost" else _lstm()
    up = {"id": "up", "type": "model", "key": model, "inputs": ["f", "r"], "label": "l", "window": 6, "params": params,
          "output_mode": "probability"}
    up2 = {"id": "up2", "type": "model", "key": model, "inputs": ["r", "b"], "label": "l", "window": 4,
           "params": params, "output_mode": "probability"}
    down = {"id": "m", "type": "model", "key": model, "inputs": ["f", "b", "o", "mc", "up", "up2"], "label": "l",
            "window": 6, "params": params}
    nodes = feats + [lab, up, up2, down]
    if multi_branch:
        feats.append({"id": "h", "type": "feature", "key": "talib_indicator", "timeframe": "tx_60m",
                      "params": {"name": "SMA", "timeperiod": 3}})
        feats.append({"id": "v1", "type": "feature", "key": "raw_volume", "timeframe": "tx_1m"})  # 較細→子 K 棒彙總
        down["inputs"] = ["f", "b", "o", "mc", "up", "up2", "h", "v1"]
        enc = "flatten" if model == "xgboost" else "lstm"
        down["branches"] = [{"id": "fast", "inputs": ["f", "o", "mc", "up", "up2", "v1"], "window": 6,
                             "encoder": enc, "aggregate": {"v1": "sum"}},
                            {"id": "slow", "inputs": ["h", "b"], "timeframe": "tx_60m", "window": 3, "encoder": enc,
                             "aggregate": {"b": "last"}}]
        nodes = feats + [lab, up, up2, down]
    settings = {"split_seed": 42, "val_ratio": 0.2}
    if strategy:
        settings["cascade"] = {"strategy": strategy, "k": 3}
    return {"mode": "v2", "start": "2024-01-03", "end": "2024-01-15", "settings": settings, "nodes": nodes}


BASES: dict = {}


def _recording(fn, tag):
    """包住組裝函式，記下每次組裝出來的合併矩陣（逐位元比對用）。"""
    def wrapped(model_node, *a, **k):
        asm = fn(model_node, *a, **k)
        BASES.setdefault(tag, []).append({bid: b.copy() for bid, b in asm.bases.items()})
        return asm
    return wrapped


def run(g, legacy: bool):
    """舊流程＝舊組裝（np.concatenate、不釋放快取）＋舊正規化；新流程＝目前程式，並讓引擎清空原始資料 dict。"""
    from tests._legacy_v2_assemble import legacy_assemble

    tag = "old" if legacy else "new"
    BASES[tag] = []
    dfs = {"tx_1m": MK["tx_1m"], "tx_5m": MK["tx_5m"], "tx_60m": MK["tx_60m"]}
    if legacy:
        old = _recording(lambda *a, release=None, **k: legacy_assemble(*a, **k), tag)
        with mock.patch.object(vi, "fit_normalization", legacy_fit_normalization), \
                mock.patch.object(vi, "apply_normalization", legacy_apply_normalization), \
                mock.patch.object(vi, "assemble", old):
            return ev.run_train_v2(g, dfs, phase=1)
    with mock.patch.object(vi, "assemble", _recording(vi.assemble, tag)):
        out = ev.run_train_v2(g, dfs, phase=1, consume_dfs=True)
    assert dfs == {}, "consume_dfs 應清空原始資料"
    return out


def _arrays(results, extras):
    out = {}
    for nid, r in results.items():
        for b in r["preprocessing_state"]["branches"]:
            for i, s in enumerate(b["steps"]):
                out[f"{nid}/{b['id']}/{i}/min"] = np.array(s["min"])
                out[f"{nid}/{b['id']}/{i}/scale"] = np.array(s["scale"])
        out[f"{nid}/weights"] = np.frombuffer(bytes(r["weights"]), dtype=np.uint8)
        if r.get("weights_last") is not None:
            out[f"{nid}/weights_last"] = np.frombuffer(bytes(r["weights_last"]), dtype=np.uint8)
    for nid, rec in extras["records"].items():
        for w, pred in rec["by_weights"].items():
            for k, v in pred.items():
                if isinstance(v, np.ndarray):
                    out[f"{nid}/val/{w}/{k}"] = v
    return out


class EndToEndEquivalenceTests(unittest.TestCase):
    def assert_same(self, g):
        r_old, f_old, x_old = run(g, legacy=True)
        r_new, f_new, x_new = run(g, legacy=False)
        self.assertEqual(f_old.hash, f_new.hash)
        a, b = _arrays(r_old, x_old), _arrays(r_new, x_new)
        self.assertEqual(sorted(a), sorted(b))
        for k in a:
            if a[k].dtype == np.uint8:
                self.assertTrue(np.array_equal(a[k], b[k]), k)
            else:
                r = compare(a[k], b[k])
                self.assertTrue(r["identical"], (k, r))
        for nid in x_old["folds"]:
            self.assertEqual([f.get("fingerprint") for f in x_old["folds"][nid]["folds"]],
                             [f.get("fingerprint") for f in x_new["folds"][nid]["folds"]], nid)
        # 每個模型、每個分支的合併矩陣逐位元相同（A4：預先配置＋逐塊複製 vs np.concatenate）
        self.assertEqual(len(BASES["old"]), len(BASES["new"]))
        for i, (bo, bn) in enumerate(zip(BASES["old"], BASES["new"])):
            self.assertEqual(sorted(bo), sorted(bn))
            for bid in bo:
                r = compare(bo[bid], bn[bid])
                self.assertTrue(r["identical"], (i, bid, r))
        # 每個特徵只計算一次（釋放快取沒有造成重算），且全部組裝完後快取清空
        ctx = x_new["ctx"]
        self.assertTrue(all(v == 1 for v in ctx.feature_computations.values()), ctx.feature_computations)
        self.assertEqual(ctx._feat, {})
        return r_new

    def test_cascade_strategies_xgboost(self):
        for strategy in ("forward_chaining", "purged_kfold", "blending", "in_sample"):
            with self.subTest(strategy=strategy):
                self.assert_same(graph(strategy))

    def test_lstm_cascade_and_multibranch(self):
        self.assert_same(graph("forward_chaining", model="lstm"))
        self.assert_same(graph("forward_chaining", model="lstm", multi_branch=True))
        self.assert_same(graph("purged_kfold", multi_branch=True))

    def test_inference_phase3_same(self):
        g = graph("forward_chaining")
        results, _f, _x = run(g, legacy=False)
        arts = {nid: {"architecture_key": next(n for n in g["nodes"] if n["id"] == nid)["key"],
                      "task_type": r["task_type"], "weights": r["weights"], "weights_last": r.get("weights_last"),
                      "model_config": r["model_config"], "preprocessing_state": r["preprocessing_state"]}
                for nid, r in results.items()}
        from tests._legacy_v2_assemble import legacy_assemble

        dfs = {"tx_1m": MK["tx_1m"], "tx_5m": MK["tx_5m"], "tx_60m": MK["tx_60m"]}
        with mock.patch.object(vi, "apply_normalization", legacy_apply_normalization), \
                mock.patch.object(vi, "assemble", lambda *a, release=None, **k: legacy_assemble(*a, **k)):
            old = ev.run_infer_v2(g, arts, dfs, "2024-01-10", "2024-01-16")
        dfs2 = dict(dfs)
        new = ev.run_infer_v2(g, arts, dfs2, "2024-01-10", "2024-01-16", consume_dfs=True)
        self.assertEqual(dfs2, {})
        for w in old["by_weights"]:
            for k, v in old["by_weights"][w].items():
                if isinstance(v, np.ndarray):
                    r = compare(v, new["by_weights"][w][k])
                    self.assertTrue(r["identical"], (w, k, r))


if __name__ == "__main__":
    unittest.main(verbosity=2)
