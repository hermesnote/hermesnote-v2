"""v2 引擎（M2～M6）驗收：樣本框架、暖機與訓練／推論一致、轉換元件、時間對齊、Phase 3 評估、串接防洩漏、多分支、
階段門檻、配對比較。全部用合成行情（tests/v2_synth.py），不連資料庫。"""

import copy
import os
import sys
import unittest
import warnings

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tests.contract_harness import lstm_params, xgb_params  # noqa: E402
from tests.v2_synth import market  # noqa: E402
from training import alignment as al  # noqa: E402
from training import cascade_audit as ca  # noqa: E402
from training import engine_v2 as ev  # noqa: E402
from training import frame as fr  # noqa: E402
from training import v2_inputs as vi  # noqa: E402
from training.graph_refs import GraphValidationError, validate_graph_spec  # noqa: E402
from training.holdout_eval import check_gates, evaluate_outputs  # noqa: E402
from training.registry import transforms as tr  # noqa: E402
from training.registry import warmup as wu  # noqa: E402

MK = market(days=15)
START, END = "2024-01-03", "2024-01-22"  # END 晚於合成資料最後一天（同 worker：期間之後沒有資料）


def feat(i, tf="tx_5m", key="ohlcv", **kw):
    return {"id": i, "type": "feature", "key": key, "timeframe": tf, **kw}


def rsi(i, tf="tx_5m", p=14, **kw):
    return feat(i, tf, "talib_indicator", params={"name": "RSI", "timeperiod": p}, **kw)


def lab(i="l", tf="tx_5m", h=2):
    return {"id": i, "type": "label", "outcome": "log_return", "labeling_rule": "fixed_threshold", "timeframe": tf,
            "params": {"horizon": h, "n_classes": 2}}


def model(i, inputs, label="l", key="xgboost", **kw):
    return {"id": i, "type": "model", "key": key, "inputs": inputs, "label": label, "window": kw.pop("window", 12),
            "params": kw.pop("params", xgb_params() if key == "xgboost" else lstm_params()), **kw}


def spec(*nodes, settings=None):
    return {"mode": "v2", "start": START, "end": END, "settings": {"split_seed": 42, "val_ratio": 0.2, **(settings or {})},
            "nodes": list(nodes)}


def truncate(dfs: dict, start: str, req: dict) -> dict:
    """模擬 worker 的載入：每個時間框架只從 start 之前 req 根開始。"""
    out = {}
    for tf, df in dfs.items():
        s = pd.to_datetime(df["bar_start_ts"])
        first = int(np.searchsorted(s.values, np.datetime64(pd.Timestamp(start))))
        out[tf] = df.iloc[max(0, first - req["bars"].get(tf, 2)):].reset_index(drop=True)
    return out


def artifacts_of(results: dict, graph: dict) -> dict:
    return {nid: {"architecture_key": next(n for n in graph["nodes"] if n["id"] == nid)["key"],
                  "task_type": r["task_type"], "weights": r["weights"], "weights_last": r.get("weights_last"),
                  "model_config": r["model_config"], "preprocessing_state": r["preprocessing_state"],
                  "train_class_counts": r["train_stats"]}
            for nid, r in results.items()}


class FrameTests(unittest.TestCase):
    def test_hash_independent_of_arm_and_stable(self):
        a = spec(feat("f"), lab(), model("m", ["f"]))
        b = spec(feat("f"), rsi("r"), lab(), model("m", ["f", "r"], window=30))
        _, fa, _ = ev.run_train_v2(a, MK, phase=1)
        _, fb, _ = ev.run_train_v2(b, MK, phase=1)
        self.assertEqual(fa.hash, fb.hash)
        c = copy.deepcopy(a)
        c["settings"]["split_seed"] = 7
        _, fc, _ = ev.run_train_v2(c, MK, phase=1)
        self.assertNotEqual(fa.hash, fc.hash)
        self.assertTrue(np.array_equal(fa.ts, fc.ts))

    def test_chronological_boundary_and_tail(self):
        g = spec(feat("f"), lab(h=3), model("m", ["f"], split_strategy="chronological"))
        _, f, _ = ev.run_train_v2(g, MK, phase=2)
        c = f.counts()
        self.assertEqual(c["excluded_reasons"].get("label_unavailable"), 3)
        self.assertEqual(c["excluded_reasons"].get("boundary"), 3)
        val0 = f.ts[f.index_of("val")[0]]
        self.assertTrue((f.target_ts[f.index_of("train")] < val0).all())

    def test_holdout_unmatured_equals_horizon(self):
        ctx = ev.Ctx(spec(feat("f"), lab(h=4), model("m", ["f"])), MK)
        y, _, h = ctx.labels("l")
        f = fr.build_frame(ctx.tfs["tx_5m"], y, h, START, END, phase=3, split_strategy="holdout", val_ratio=0, split_seed=0)
        self.assertEqual(f.counts()["excluded_reasons"], {"label_unmatured": 4})
        self.assertEqual(f.counts()["holdout"], len(f))


class WarmupTransformTests(unittest.TestCase):
    def test_memory_types_and_suggestions(self):
        self.assertEqual(wu.memory_type("talib_indicator", {"name": "SMA"}), "finite")
        self.assertEqual(wu.memory_type("talib_indicator", {"name": "OBV"}), "cumulative")
        self.assertEqual(wu.memory_type("talib_indicator", {"name": "MACD"}), "composite")
        self.assertEqual(wu.memory_type("talib_indicator", {"name": "BBANDS", "matype": 1}), "composite")
        s = wu.suggest("talib_indicator", {"name": "EMA", "timeperiod": 20}, 0.01)
        self.assertEqual(s["base_bars"], 19 + 47)  # lookback 19 ＋ ceil(ln .01 / ln(19/21)) = ceil(46.01) = 47
        self.assertFalse(wu.suggest("talib_indicator", {"name": "SMA", "timeperiod": 10})["needs_researcher"])
        self.assertTrue(wu.suggest("talib_indicator", {"name": "ADX", "timeperiod": 14})["needs_researcher"])

    def test_validation_rules(self):
        cases = [
            (feat("o", key="talib_indicator", params={"name": "OBV"}), "必須選擇 transform"),
            (feat("a", key="talib_indicator", params={"name": "ADX", "timeperiod": 14}), "warmup.bars"),
            (rsi("r", warmup={"bars": 3}), "只能加長"),
            (feat("o", key="talib_indicator", params={"name": "OBV"}, transform={"type": "log_return"}), "累積型指標不能用"),
        ]
        for f, msg in cases:
            with self.subTest(msg=msg), self.assertRaises(GraphValidationError) as cm:
                validate_graph_spec(spec(f, lab(), model("m", [f["id"]])))
            self.assertTrue(any(msg in i for i in cm.exception.issues), cm.exception.issues)
        validate_graph_spec(spec(feat("o", key="talib_indicator", params={"name": "OBV"}, transform={"type": "diff"}),
                                 lab(), model("m", ["o"])))

    def test_legacy_graph_with_v2_fields_rejected(self):
        g = spec(rsi("r", warmup={"bars": 40}), lab(), model("m", ["r"]))
        g.pop("mode")
        g.pop("settings")
        with self.assertRaises(GraphValidationError) as cm:
            validate_graph_spec(g)
        self.assertTrue(any("mode" in i for i in cm.exception.issues))

    def test_cumulative_transforms_independent_of_load_start(self):
        T_full = al.TfData("tx_5m", MK["tx_5m"])
        T_cut = al.TfData("tx_5m", MK["tx_5m"].iloc[200:].reset_index(drop=True))
        node = {"key": "talib_indicator", "params": {"name": "OBV"}}
        for t in ({"type": "diff"}, {"type": "rolling_sum", "window": 10}, {"type": "fixed_anchor", "anchor": "2024-01-10"}):
            with self.subTest(t=t["type"]):
                a = vi.compute_feature_v2({**node, "transform": t}, T_full, 0.01)[200:]
                b = vi.compute_feature_v2({**node, "transform": t}, T_cut, 0.01)
                ok = ~np.isnan(a) & ~np.isnan(b)
                self.assertGreater(ok.sum(), 100)
                np.testing.assert_allclose(a[ok], b[ok])
        a = vi.compute_feature_v2({**node, "transform": {"type": "legacy_cumulative"}}, T_full, 0.01)[200:]
        b = vi.compute_feature_v2({**node, "transform": {"type": "legacy_cumulative"}}, T_cut, 0.01)
        self.assertFalse(np.allclose(a, b))  # 維持舊行為＝與載入起點有關（只供對照）

    def test_rolling_sum_and_returns(self):
        x = np.array([[1.0], [2.0], [4.0], [7.0]])
        np.testing.assert_allclose(tr.apply(x, {"type": "diff"})[1:, 0], [1, 2, 3])
        np.testing.assert_allclose(tr.apply(x, {"type": "rolling_sum", "window": 2})[2:, 0], [3, 5])
        np.testing.assert_allclose(tr.apply(x, {"type": "pct_return"})[1:, 0], [1, 1, 0.75])


class ConsistencyTests(unittest.TestCase):
    """訓練與推論共用組裝：同一批時間點，推論（從 holdout 起點往前載入歷史需求根數）與訓練的特徵一致。"""

    def _infer_vs_train(self, features, tol):
        g = spec(*features, lab(), model("m", [f["id"] for f in features]))
        results, frame, extras = ev.run_train_v2(g, MK, phase=1)
        entry = extras["records"]["m"]
        req = vi.history_requirements(g)
        start = "2024-01-15"
        dfs = truncate(MK, start, req)
        out = ev.run_infer_v2(g, artifacts_of(results, g), dfs, start, END)
        self.assertEqual(len(out["skipped"]), 0)  # 往前取暖機後，測試期間第一個時間點就有預測
        # 同一時間點的預測：訓練時對 val 點的預測 vs 推論
        val_ts = entry["rows"].ts[entry["points"]]
        inf_ts = out["rows"].ts[out["points"]]
        common, ia, ib = np.intersect1d(val_ts, inf_ts, return_indices=True)
        self.assertGreater(len(common), 5)
        pa = entry["by_weights"]["best"]["proba"][ia]
        pb = out["by_weights"]["best"]["proba"][ib]
        np.testing.assert_allclose(pa, pb, atol=tol)
        return results

    def test_finite_features_identical(self):
        r = self._infer_vs_train([feat("f"), feat("s", key="talib_indicator", params={"name": "SMA", "timeperiod": 10})], 0)
        self.assertEqual(r["m"]["training_meta"]["warmup"]["s"]["bars"], 9)

    def test_recursive_within_threshold(self):
        r = self._infer_vs_train([feat("f"), rsi("r")], 0.05)
        w = r["m"]["training_meta"]["warmup"]["r"]
        self.assertEqual(w["bars"], wu.suggest("talib_indicator", {"name": "RSI", "timeperiod": 14})["base_bars"])

    def test_insufficient_history_skips_with_reason(self):
        g = spec(feat("f"), lab(), model("m", ["f"], window=12))
        dfs = {k: v.iloc[len(v) // 2:].reset_index(drop=True) for k, v in MK.items()}  # 資料起點晚於期間起點
        results, frame, _ = ev.run_train_v2(g, dfs, phase=1)
        ex = results["m"]["training_meta"]["exclusions"]
        self.assertGreater(sum(ex.get("train", {}).values()) + sum(ex.get("val", {}).values()), 0)
        self.assertIn("insufficient_history", {**ex.get("train", {}), **ex.get("val", {})})


class AlignmentTests(unittest.TestCase):
    def _tf(self, starts, ends, tf="tx_x", delay=0):
        s = pd.to_datetime(starts)
        return al.TfData(tf, pd.DataFrame({"bar_start_ts": s, "bar_end_ts": pd.to_datetime(ends), "open": 1.0,
                                           "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}), delay)

    def test_boundary_equal_close_usable_and_one_second(self):
        hour = self._tf(["2024-01-02 09:00"], ["2024-01-02 10:00"])
        t = np.array(["2024-01-02 09:59:59", "2024-01-02 10:00:00", "2024-01-02 10:00:01"], dtype="datetime64[ns]")
        self.assertEqual(al.asof_positions(hour, t).tolist(), [-1, 0, 0])
        for d, expect in ((0, [-1, 0, 0]), (1, [-1, -1, 0]), (60, [-1, -1, -1])):
            self.assertEqual(al.asof_positions(self._tf(["2024-01-02 09:00"], ["2024-01-02 10:00"], delay=d), t).tolist(),
                             expect)

    def test_daily_close_vs_last_1m_bar(self):
        T = {k: al.TfData(k, v) for k, v in MK.items()}
        m1 = T["tx_1m"]
        last = np.nonzero(m1.index.values == np.datetime64("2024-01-03 13:44"))[0][0]
        pos = al.asof_positions(T["tx_daily_day"], m1.available[[last - 1, last]])
        days = T["tx_daily_day"].index.values
        self.assertEqual(str(days[pos[1]])[:10], "2024-01-03")  # 13:44～13:45 收盤＝日線收盤，可用
        self.assertEqual(str(days[pos[0]])[:10], "2024-01-02")  # 前一根只能用前一天的日線

    def test_vectorized_matches_reference_loop(self):
        T = {k: al.TfData(k, v) for k, v in MK.items()}
        dec, src = T["tx_15m"], T["tx_60m"]
        vals = np.arange(len(src), dtype=float).reshape(-1, 1)
        got, pos = al.asof_values(src.available, vals, dec.available)
        for i, tau in enumerate(dec.available):
            ok = [j for j in range(len(src)) if src.available[j] <= tau]
            self.assertEqual(pos[i], ok[-1] if ok else -1)
        # 時間戳追蹤：取到的值所屬 K 棒的可用時間一律 ≤ 決策時刻
        used = pos >= 0
        self.assertTrue((src.available[pos[used]] <= dec.available[used]).all())

    def test_subbar_aggregation_matches_reference(self):
        T = {k: al.TfData(k, v) for k, v in MK.items()}
        fine, coarse = T["tx_1m"], T["tx_15m"]
        v = fine.ohlcv[["volume"]].to_numpy()
        got, present = al.aggregate_subbars(fine, v, coarse, "sum")
        for i in range(0, len(coarse), 37):
            m = (fine.index.values >= coarse.index.values[i]) & (fine.end <= coarse.end[i])
            self.assertAlmostEqual(got[i, 0], v[m, 0].sum())
        np.testing.assert_allclose(got[:, 0], coarse.ohlcv["volume"].to_numpy())

    def test_staleness_rules(self):
        T = {k: al.TfData(k, v) for k, v in MK.items()}
        dec, src = T["tx_15m"], T["tx_daily_day"]
        pos = al.asof_positions(src, dec.available)
        base = al.resolve_alignment({})
        s = al.staleness(src, pos, dec.available, base)
        self.assertEqual(int(s["stale"].sum()), 0)  # 預設：可用前一時段、最多 1 個時段
        strict = al.staleness(src, pos, dec.available, {**base, "allow_previous_session": False})
        self.assertGreater(int(strict["stale"].sum()), 0)
        aged = al.staleness(src, pos, dec.available, {**base, "max_age_sec": 3600})
        self.assertTrue((aged["age_sec"][aged["stale"]] > 3600).all())

    def test_stale_branch_excluded_and_recorded(self):
        g = spec(feat("f"), feat("d", "tx_daily_day"), lab(),
                 model("m", ["f", "d"], branches=[{"id": "main", "inputs": ["f"], "window": 6},
                                                   {"id": "day", "inputs": ["d"], "window": 1, "encoder": "last_value"}]),
                 settings={"alignment": {"allow_previous_session": False}})
        results, _, _ = ev.run_train_v2(g, MK, phase=1)
        ex = results["m"]["training_meta"]["exclusions"]
        self.assertGreater({**ex.get("train", {}), **ex.get("val", {})}.get("stale_branch", 0), 0)

    def test_cross_timeframe_rules(self):
        legacy = {"start": START, "end": END, "nodes": [feat("f"), feat("h", "tx_60m"), lab(), model("m", ["f", "h"])]}
        with self.assertRaises(GraphValidationError):
            validate_graph_spec(legacy)
        validate_graph_spec(spec(feat("f"), feat("h", "tx_60m"), lab(), model("m", ["f", "h"])))
        with self.assertRaises(GraphValidationError) as cm:  # 較細時間框架進較粗分支必須指定彙總
            validate_graph_spec(spec(feat("f"), feat("m1", "tx_1m"), lab(), model("m", ["f", "m1"])))
        self.assertTrue(any("aggregate" in i for i in cm.exception.issues))


class Phase3Tests(unittest.TestCase):
    def test_val_path_equals_holdout_evaluation_path(self):
        g = spec(feat("f"), lab(), model("m", ["f"]))
        results, _, extras = ev.run_train_v2(g, MK, phase=1)
        entry = extras["records"]["m"]
        r = results["m"]
        y = entry["rows"].y[entry["points"]].astype(int)
        for w in ("best", "last"):
            rep = evaluate_outputs("classification", r["model_config"], y, entry["by_weights"][w], r["train_stats"])
            self.assertEqual(rep["confusion_matrix"], r["evaluation"][w]["confusion_matrix"])
            self.assertAlmostEqual(rep["roc_auc"] or 0, r["evaluation"][w]["roc_auc"] or 0)
            self.assertEqual(rep["baseline_majority_class"], r["evaluation"][w]["baseline_majority_class"])

    def test_holdout_inference_best_last_and_unmatured(self):
        from training.v2_records import evaluate_entry, holdout_evaluations

        g = spec(feat("f"), lab(h=3), model("m", ["f"]))
        g["end"] = "2024-01-15"
        results, _, _ = ev.run_train_v2(g, MK, phase=1)
        out = ev.run_infer_v2(g, artifacts_of(results, g), MK, "2024-01-15", END)
        entry = {**out, "frame_hash": out["frame"].hash}
        raw, st = evaluate_entry(entry, task_type="classification", model_config=out["model_config"],
                                 stats=results["m"]["train_stats"])
        self.assertEqual(set(raw), {"best", "last"})
        self.assertEqual(st["n_unmatured"], 3)
        evs = holdout_evaluations("m", raw, specs=None, phase=3, period={"start": "2024-01-15", "end": END}, stats_out=st)
        self.assertEqual({e["point"]["kind"] for e in evs}, {"best", "last"})
        self.assertTrue(all(e["dataset"]["split"] == "holdout" for e in evs))
        # 已知答案：自行重算 accuracy
        rows = entry["rows"]
        inc = np.array([rows.frame_reason[p] is None for p in entry["points"]])
        acc = float(np.mean(out["by_weights"]["best"]["pred"][inc] == rows.y[entry["points"][inc]]))
        self.assertAlmostEqual(raw["best"]["balanced_accuracy"] >= 0 and acc, acc)


class CascadeTests(unittest.TestCase):
    def _graph(self, strategy):
        up = model("up", ["f"], params=xgb_params(n_estimators=60, max_depth=8, learning_rate=0.5), output_mode="probability",
                   upstream_policy={"strategy": strategy, "k": 3})
        dn = model("dn", ["up"], window=1, branches=[{"id": "u", "inputs": ["up"], "window": 1, "encoder": "last_value"}])
        return spec(feat("f"), lab(), up, dn)

    def test_leakage_detection_by_agreement(self):
        res = {}
        for strategy in ("in_sample", "forward_chaining"):
            g = self._graph(strategy)
            ctx = ev.Ctx(g, MK)
            main = ev.build_main_frame(ctx, phase=1, split_strategy="random", start=START, end=END)
            t = ev.Trainer(ctx, phase=1, split_strategy="random", start=START, end=END)
            t.run(main)
            s = t.series[("up", "default")]
            rows = t.records["dn"]["rows"]
            tr = np.nonzero(rows.status == "train")[0]
            p = s["values"][tr, 1]
            ok = ~np.isnan(p)
            res[strategy] = float(np.mean((p[ok] >= 0.5) == (rows.y[tr][ok] == 1)))
        self.assertGreater(res["in_sample"], 0.75)
        self.assertLess(res["forward_chaining"], 0.65)

    def test_audit_passes_and_counterexample_caught(self):
        for strategy in ("forward_chaining", "purged_kfold", "blending"):
            with self.subTest(strategy=strategy):
                _, frame, extras = ev.run_train_v2(self._graph(strategy), MK, phase=1)
                recs, folds = ca.records_from_engine(extras["records"], extras["folds"])
                rep = ca.audit(recs, folds, split_strategy="random")
                self.assertTrue(rep["passed"], rep["rules"])
                self.assertGreater(rep["rules"]["r2"]["checked"], 0)
                bad = copy.deepcopy(folds)
                for f in bad:  # 反例：把清除範圍拿掉（折的訓練點延伸到很晚）
                    if f["fold"] > 0 and f.get("before_max_le") is not None:
                        f["before_max_le"] = np.datetime64("2030-01-01")
                self.assertFalse(ca.audit(recs, bad, split_strategy="random")["passed"])

    def test_in_sample_flagged_and_audit_fails(self):
        _, _, extras = ev.run_train_v2(self._graph("in_sample"), MK, phase=1)
        self.assertIn("upstream_in_sample", [w["code"] for w in extras["warnings"]])
        recs, folds = ca.records_from_engine(extras["records"], extras["folds"])
        self.assertFalse(ca.audit(recs, folds)["passed"])


class MultiBranchTests(unittest.TestCase):
    def test_lstm_branches_save_reload_consistent(self):
        from training.registry.architectures import load_model

        h = feat("h", "tx_60m", "talib_indicator", params={"name": "SMA", "timeperiod": 3})
        bad = spec(feat("f"), h, lab(), model("m", ["f", "h"], key="lstm",
                                               branches=[{"id": "main", "inputs": ["f"], "window": 8}]))
        with self.assertRaises(GraphValidationError):  # branches 合起來必須等於 inputs
            validate_graph_spec(bad)
        g = spec(feat("f"), h, lab(), model("m", ["f", "h"], key="lstm", branches=[
            {"id": "main", "inputs": ["f"], "window": 8}, {"id": "hour", "inputs": ["h"], "window": 3}]))
        validate_graph_spec(g)
        results, _, extras = ev.run_train_v2(g, MK, phase=1)
        r = results["m"]
        self.assertEqual(len(r["model_config"]["input_layout"]["branches"]), 2)
        loaded = load_model("lstm", r["weights"], r["model_config"])
        entry = extras["records"]["m"]
        nz = vi.apply_normalization(entry["asm"], r["preprocessing_state"])
        X = vi.packed(entry["asm"], nz, entry["points"])
        np.testing.assert_allclose(loaded["predict_proba"](np.asarray(X)), entry["by_weights"]["best"]["proba"], atol=1e-5)

    def test_single_branch_uses_legacy_net(self):
        g = spec(feat("f"), lab(), model("m", ["f"], key="lstm"))
        results, _, _ = ev.run_train_v2(g, MK, phase=1)
        self.assertNotIn("input_layout", results["m"]["model_config"])

    def test_daily_branch_never_sees_unclosed_day(self):
        """日線分支的最後一根，可用時間（13:45 收盤）一律 ≤ 決策時刻：日盤進行中只能用前一天的日線。"""
        g = spec(feat("f"), feat("d", "tx_daily_day"), lab(),
                 model("m", ["f", "d"], branches=[{"id": "main", "inputs": ["f"], "window": 6},
                                                   {"id": "day", "inputs": ["d"], "window": 2, "encoder": "last_value"}]))
        _, _, extras = ev.run_train_v2(g, MK, phase=1)
        e = extras["records"]["m"]
        meta = e["asm"].branch_meta["day"]
        pts = e["points"]
        self.assertTrue((meta["available"][pts] <= e["rows"].tau[pts]).all())
        same_day = pd.DatetimeIndex(meta["last_ts"][pts]).normalize() == pd.DatetimeIndex(e["rows"].ts[pts]).normalize()
        before_close = pd.DatetimeIndex(e["rows"].tau[pts]).time < pd.Timestamp("13:45").time()
        self.assertFalse((same_day & before_close).any())

    def test_packed_windows_layout(self):
        base_a = np.arange(20, dtype=np.float32).reshape(10, 2)
        base_b = np.arange(5, dtype=np.float32).reshape(5, 1) + 100
        pw = vi.PackedWindows([(base_a, np.arange(10), 3, 0), (base_b, np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), 1, 2)],
                              3, 3, np.array([5, 9]))
        x = pw[np.arange(2)]
        np.testing.assert_array_equal(x[0, :, :2], base_a[3:6])
        np.testing.assert_array_equal(x[0, :2, 2], [0, 0])
        self.assertEqual(x[1, -1, 2], 104)


class GateCompareTests(unittest.TestCase):
    def test_gates(self):
        result = {"m": {"evaluations": [{"head": "default", "point": {"kind": "best"}, "metrics": {"macro_f1": 0.55}}]}}
        ok = check_gates([{"phase": 2, "metric": "macro_f1", "op": ">=", "value": 0.5}], result, "m", 2)
        self.assertTrue(ok["passed"] and ok["configured"])
        bad = check_gates([{"phase": 2, "metric": "macro_f1", "op": ">=", "value": 0.6}], result, "m", 2)
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["conditions"][0]["actual"], 0.55)
        none = check_gates([], result, "m", 2)
        self.assertFalse(none["configured"])

    def test_paired_compare(self):
        from routers.model_experiments import _paired

        ts = list(range(10))
        a = [{"sample_ts": t, "y_true": t % 2, "y_pred": t % 2} for t in ts]
        b = [{"sample_ts": t, "y_true": t % 2, "y_pred": 0} for t in ts[2:]]
        r = _paired(a, b, "classification")
        self.assertEqual(r["n_common"], 8)
        self.assertEqual(r["accuracy_a"], 1.0)
        self.assertEqual(r["only_a_correct"], 4)
        self.assertLess(r["mcnemar_exact_p"], 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
