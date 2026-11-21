"""2026-11-21 新目標／特徵／指標／模型驗收：首次越界、區間最高最低、缺格排除、K 棒與價格位置特徵、Brier／Log loss、
上游樣本外機率逐筆保存、GRU／CNN-LSTM／Transformer。全部用合成行情（tests/v2_synth.py），不連資料庫。"""

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
from tests.test_v2_pipeline import END, MK, START, artifacts_of, feat, spec  # noqa: E402
from training import engine_v2 as ev  # noqa: E402
from training.evaluation import probability_scores  # noqa: E402
from training.graph_refs import GraphValidationError, validate_graph_spec  # noqa: E402
from training.registry import bounds as bd  # noqa: E402
from training.registry import warmup as wu  # noqa: E402
from training.registry.features import compute_feature  # noqa: E402
from training.registry.outcomes import compute_outcome  # noqa: E402
from training.v2_records import upstream_oof_rows  # noqa: E402

BB = {"upper": {"name": "BBANDS", "timeperiod": 5, "nbdevup": 1.0, "nbdevdn": 1.0, "output": "upperband"},
      "lower": {"name": "BBANDS", "timeperiod": 5, "nbdevup": 1.0, "nbdevdn": 1.0, "output": "lowerband"}}


def fc_label(i="l", tf="tx_5m", h=4, **params):
    return {"id": i, "type": "label", "timeframe": tf, "outcome": "first_crossing", "labeling_rule": "class_index",
            "params": {"horizon": h, "n_classes": 2 * h + 1, "bounds": BB, **params}}


def reg_label(i="l", tf="tx_5m", h=4, outcome="future_high_offset"):
    return {"id": i, "type": "label", "timeframe": tf, "outcome": outcome, "labeling_rule": "identity",
            "params": {"horizon": h}}


def mdl(i, inputs, key, label="l", window=12, **kw):
    params = kw.pop("params", None) or (xgb_params() if key == "xgboost" else PARAMS[key])
    return {"id": i, "type": "model", "key": key, "inputs": inputs, "label": label, "window": window, "params": params, **kw}


_base = {k: v for k, v in lstm_params().items() if k != "lstm_layers"}
PARAMS = {
    "lstm": lstm_params(),
    "gru": {**_base, "gru_layers": [{"units": 8, "dropout": 0.0}]},
    "cnn_lstm": {**_base, "conv_layers": [{"filters": 32, "kernel_size": 1}], "pool_size": 3,
                 "lstm_layers": [{"units": 64, "dropout": 0.2}]},
    "transformer": {**_base, "transformer": {"d_model": 8, "n_heads": 2, "n_layers": 1, "ff_dim": 16, "dropout": 0.0}},
}


def ref_first_crossing(close, upper, lower, h):
    n = len(close)
    out = np.full(n, np.nan)
    for t in range(n - h):
        if np.isnan(upper[t]) or np.isnan(lower[t]):
            continue
        c = 2 * h
        for k in range(1, h + 1):
            if close[t + k] > upper[t]:
                c = k - 1
                break
            if close[t + k] < lower[t]:
                c = h + k - 1
                break
        out[t] = c
    return out


class OutcomeTests(unittest.TestCase):
    def test_first_crossing_matches_reference_and_touch_is_inside(self):
        df = pd.DataFrame(MK["tx_5m"][["open", "high", "low", "close", "volume"]])
        up, lo = bd.compute(df, BB)
        raw, vl = compute_outcome("first_crossing", df, {"horizon": 4, "bounds": BB})
        ref = ref_first_crossing(df["close"].to_numpy(), up, lo, 4)
        np.testing.assert_array_equal(np.isnan(raw), np.isnan(ref))
        ok = ~np.isnan(ref)
        np.testing.assert_array_equal(raw[ok], ref[ok])
        self.assertEqual(vl, len(df) - 4)
        self.assertEqual(set(np.unique(raw[ok]).astype(int)) - set(range(9)), set())
        # 恰好碰線：收盤等於上界不算上穿
        d2 = pd.DataFrame({"open": [1.0] * 6, "high": [1.0] * 6, "low": [1.0] * 6, "close": [1, 1, 1, 1, 1, 1.0],
                           "volume": [1.0] * 6})
        flat = {"upper": {"name": "MAX", "timeperiod": 2}, "lower": {"name": "MIN", "timeperiod": 2}}
        r2, _ = compute_outcome("first_crossing", d2, {"horizon": 2, "bounds": flat})
        self.assertTrue(np.all(r2[1:4] == 4))

    def test_high_low_offsets(self):
        df = pd.DataFrame(MK["tx_5m"][["open", "high", "low", "close", "volume"]])
        hi, _ = compute_outcome("future_high_offset", df, {"horizon": 3})
        lo, _ = compute_outcome("future_low_offset", df, {"horizon": 3})
        h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
        t = 10
        self.assertAlmostEqual(hi[t], 100 * (h[t + 1:t + 4].max() / c[t] - 1))
        self.assertAlmostEqual(lo[t], 100 * (l[t + 1:t + 4].min() / c[t] - 1))
        self.assertTrue(np.isnan(hi[-3:]).all() and np.isnan(lo[-3:]).all())


def _reasons(frame, code):
    return int((frame.reason == code).sum())


class GapTests(unittest.TestCase):
    def test_gap_cross_session_and_tail_are_separate(self):
        dfs = {k: v.copy() for k, v in MK.items()}
        df5 = dfs["tx_5m"]
        drop = 200  # 拿掉一根（同一時段內）：前 H 根的答案是 label_gap
        dfs["tx_5m"] = df5.drop(index=drop).reset_index(drop=True)
        g = spec(feat("f"), reg_label(h=3), mdl("m", ["f"], "xgboost"))
        _, frame, _ = ev.run_train_v2(g, dfs, phase=1)
        for t in df5["bar_start_ts"].iloc[drop - 3:drop].to_numpy():
            i = int(np.nonzero(frame.ts == np.datetime64(t))[0][0])
            self.assertEqual(frame.reason[i], "label_gap")
        # 日盤之間（合成資料沒有夜盤）：每天最後 3 根跨到下一個時段
        n_days = len(set(pd.to_datetime(frame.ts).date))
        self.assertGreaterEqual(_reasons(frame, "label_cross_session"), 3 * (n_days - 1))
        self.assertEqual(_reasons(frame, "label_gap"), 3)
        self.assertGreater(_reasons(frame, "label_unavailable"), 0)  # 期末不足 H 根，另外記

    def test_wrong_bar_length_is_gap(self):
        dfs = {k: v.copy() for k, v in MK.items()}
        d = dfs["tx_5m"]
        # 時間首尾相接但長度不對：第 100 根改成 4 分鐘、第 101 根提早 1 分鐘開始
        d.loc[100, "bar_end_ts"] = d.loc[100, "bar_end_ts"] - pd.Timedelta(minutes=1)
        d.loc[101, ["bar_start_ts", "datetime"]] = d.loc[101, "bar_start_ts"] - pd.Timedelta(minutes=1)
        g = spec(feat("f"), reg_label(h=2), mdl("m", ["f"], "xgboost"))
        ctx = ev.Ctx(g, dfs)
        r = ctx.label_reasons("l")
        T = ctx.tfs["tx_5m"]
        i = int(np.searchsorted(T.index.values, np.datetime64(d.loc[100, "bar_start_ts"])))
        self.assertEqual(r[i], "label_gap")
        self.assertEqual(r[i - 1], "label_gap")

    def test_cross_session_true_and_ignore(self):
        g = spec(feat("f"), reg_label(h=3), mdl("m", ["f"], "xgboost"))
        _, f0, _ = ev.run_train_v2(g, MK, phase=1)
        g["nodes"][1]["params"]["cross_session"] = True
        _, f1, _ = ev.run_train_v2(g, MK, phase=1)
        g["nodes"][1]["params"]["gaps"] = "ignore"
        _, f2, _ = ev.run_train_v2(g, MK, phase=1)
        self.assertGreater(_reasons(f0, "label_cross_session"), 0)
        self.assertEqual(_reasons(f1, "label_cross_session") + _reasons(f1, "label_gap"), 0)
        self.assertEqual(_reasons(f2, "label_cross_session") + _reasons(f2, "label_gap"), 0)

    def test_start_outside(self):
        g = spec(feat("f"), fc_label(), mdl("m", ["f"], "xgboost"))
        ctx = ev.Ctx(g, MK)
        r = ctx.label_reasons("l")
        T = ctx.tfs["tx_5m"]
        up, lo = bd.compute(T.ohlcv, BB)
        c = T.ohlcv["close"].to_numpy()
        with np.errstate(invalid="ignore"):
            out = (c > up) | (c < lo)
        cand = np.nonzero(out & (r != "label_gap") & (r != "label_cross_session") & (np.arange(len(c)) + 4 < len(c)))[0]
        self.assertGreater(len(cand), 0)
        self.assertTrue(all(r[i] == "label_start_outside" for i in cand))
        y, _, _ = ctx.labels("l")
        self.assertTrue(np.isnan(y[cand]).all())

    def test_existing_outcomes_unchanged(self):
        lab = {"id": "l", "type": "label", "outcome": "log_return", "labeling_rule": "fixed_threshold",
               "timeframe": "tx_5m", "params": {"horizon": 2, "n_classes": 2}}
        g = spec(feat("f"), lab, mdl("m", ["f"], "xgboost"))
        self.assertIsNone(ev.Ctx(g, MK).label_reasons("l"))
        _, f, _ = ev.run_train_v2(g, MK, phase=1)
        self.assertEqual(set(f.counts()["excluded_reasons"]) - {"label_unavailable", "boundary"}, set())


class ValidationTests(unittest.TestCase):
    def test_rules(self):
        def bad(label, mode="v2"):
            g = spec(feat("f"), label, mdl("m", ["f"], "xgboost"))
            g["mode"] = mode
            with self.assertRaises(GraphValidationError) as cm:
                validate_graph_spec(g)
            return "；".join(cm.exception.issues) if hasattr(cm.exception, "issues") else str(cm.exception)

        self.assertIn("n_classes 必須是 9", bad(fc_label(n_classes=3)))
        lab = fc_label()
        lab["labeling_rule"] = "fixed_threshold"
        self.assertIn("class_index", bad(lab))
        self.assertIn("mode=v2", bad(fc_label(), mode=None))
        self.assertIn("bounds", bad(fc_label(bounds={"upper": {"name": "BBANDS"}})))
        validate_graph_spec(spec(feat("f"), fc_label(), mdl("m", ["f"], "xgboost")))


class FeatureMetricTests(unittest.TestCase):
    def test_candle_parts_and_price_position(self):
        df = pd.DataFrame({"open": [10.0, 12], "high": [13.0, 12.5], "low": [9.0, 10], "close": [12.0, 11],
                           "volume": [1.0, 1]})
        np.testing.assert_array_equal(compute_feature("candle_parts", df), [[2, 1, 1], [1, 0.5, 1]])
        d5 = pd.DataFrame(MK["tx_5m"][["open", "high", "low", "close", "volume"]])
        pp = compute_feature("price_position", d5, {"bounds": BB})
        up, lo = bd.compute(d5, BB)
        np.testing.assert_allclose(pp[:, 0], up - d5["close"], equal_nan=True)
        np.testing.assert_allclose(pp[:, 1], d5["close"] - lo, equal_nan=True)
        s = wu.suggest("price_position", {"bounds": BB})
        self.assertEqual((s["memory_type"], s["base_bars"], s["needs_researcher"]), ("finite", 4, False))
        self.assertEqual(wu.suggest("candle_parts", {})["base_bars"], 0)

    def test_brier_log_loss(self):
        y = np.array([0, 1, 2])
        p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        r = probability_scores(y, p, [0, 1, 2])
        manual = np.mean([(0.3 ** 2 + 0.2 ** 2 + 0.1 ** 2), (0.1 ** 2 + 0.2 ** 2 + 0.1 ** 2), (0.3 ** 2 + 0.3 ** 2 + 0.6 ** 2)])
        self.assertAlmostEqual(r["brier"], manual)
        self.assertAlmostEqual(r["log_loss"], -np.mean(np.log([0.7, 0.8, 0.4])))
        rb = probability_scores(np.array([0, 1]), np.array([[0.8, 0.2], [0.4, 0.6]]), [0, 1])
        self.assertAlmostEqual(rb["brier"], (0.2 ** 2 + 0.4 ** 2) / 2)
        # 驗證集缺類別仍可算（labels 固定全部類別）
        self.assertIsNotNone(probability_scores(np.array([0, 0]), np.array([[0.5, 0.3, 0.2]] * 2), [0, 1, 2])["log_loss"])


class CascadeOofTests(unittest.TestCase):
    def test_nine_class_xgboost_upstream_oof_saved(self):
        up = mdl("up", ["f", "pp"], "xgboost", label="lfc", output_mode="probability",
                 upstream_policy={"strategy": "forward_chaining", "k": 3, "save_oof": True})
        dn = mdl("dn", ["f", "up"], "lstm", label="l",
                 branches=[{"id": "main", "inputs": ["f"], "window": 12},
                           {"id": "u", "inputs": ["up"], "window": 1, "encoder": "last_value"}])
        g = spec(feat("f"), feat("pp", key="price_position", params={"bounds": BB}), fc_label("lfc"),
                 {"id": "l", "type": "label", "outcome": "log_return", "labeling_rule": "fixed_threshold",
                  "timeframe": "tx_5m", "params": {"horizon": 2, "n_classes": 2}}, up, dn)
        validate_graph_spec(g)
        results, frame, extras = ev.run_train_v2(g, MK, phase=1)
        self.assertEqual(results["up"]["output_specs"]["default"]["classes"][:2], ["up_1", "up_2"])
        self.assertEqual(results["up"]["output_specs"]["default"]["columns"], 9)
        oof = extras["upstream_oof"]["up"]
        rows = list(upstream_oof_rows(oof, job_id="j", node_id="up", frame_hash=frame.hash))
        self.assertGreater(len(rows), 0)
        import json

        first = rows[0]
        self.assertEqual(first[6], "upstream_oof")
        self.assertEqual(len(json.loads(first[16])), 9)
        up_meta = json.loads(first[20])
        self.assertGreaterEqual(up_meta["fold"], 1)
        self.assertIsNotNone(up_meta["train_max_label_end"])
        self.assertLess(pd.Timestamp(up_meta["train_max_label_end"]), pd.Timestamp(first[10]))  # 訓練截止早於這筆的決策時刻
        ev_best = results["dn"]["evaluation"]["best"]
        self.assertIn("brier", ev_best)
        self.assertIn("log_loss", ev_best)
        self.assertNotIn("up", {k for k in extras["upstream_oof"] if k != "up"})


class SeqModelTests(unittest.TestCase):
    def _roundtrip(self, key, label, task):
        g = spec(feat("f"), label, mdl("m", ["f"], key))
        validate_graph_spec(g)
        g["end"] = "2024-01-15"
        results, _, extras = ev.run_train_v2(g, MK, phase=1)
        r = results["m"]
        self.assertEqual(r["model_config"].get("arch"), key)
        out = ev.run_infer_v2(g, artifacts_of(results, g), MK, "2024-01-10", "2024-01-15")
        # 重新載入的推論與訓練當下的驗證預測一致（同一時間點）
        ent = extras["records"]["m"]
        rows = ent["rows"]
        tr_pred = dict(zip(rows.ts[ent["points"]], ent["by_weights"]["best"]["pred"]))
        inf_pred = dict(zip(out["rows"].ts[out["points"]], out["by_weights"]["best"]["pred"]))
        common = [t for t in tr_pred if t in inf_pred]
        self.assertGreater(len(common), 0)
        for t in common[:50]:
            self.assertAlmostEqual(float(tr_pred[t]), float(inf_pred[t]), places=4)
        if task == "classification":
            self.assertIn("brier", r["evaluation"]["best"])
        return r

    def test_models_classification_and_regression(self):
        for key in ("gru", "cnn_lstm", "transformer"):
            with self.subTest(key=key):
                self._roundtrip(key, fc_label(), "classification")
                self._roundtrip(key, reg_label(outcome="future_low_offset"), "regression")

    def test_multi_branch_and_validation(self):
        g = spec(feat("f"), feat("r", key="candle_parts"), fc_label(),
                 mdl("m", ["f", "r"], "gru", branches=[{"id": "a", "inputs": ["f"], "window": 12},
                                                       {"id": "b", "inputs": ["r"], "window": 6}]))
        validate_graph_spec(g)
        ev.run_train_v2(g, MK, phase=1)
        bad = PARAMS["transformer"] | {"transformer": {**PARAMS["transformer"]["transformer"], "d_model": 7}}
        with self.assertRaises(GraphValidationError):
            validate_graph_spec(spec(feat("f"), fc_label(), mdl("m", ["f"], "transformer", params=bad)))

    def test_lstm_weights_format_unchanged(self):
        g = spec(feat("f"), fc_label(), mdl("m", ["f"], "lstm"))
        results, _, _ = ev.run_train_v2(g, MK, phase=1)
        self.assertNotIn("arch", results["m"]["model_config"])


def _with_without(mask: bool):
    """有／無上游機率的兩個組別：無機率組用 role=mask 的分支套用同一個上游的樣本可用性。"""
    up = mdl("up", ["pp"], "xgboost", label="lfc", output_mode="probability",
             upstream_policy={"strategy": "forward_chaining", "k": 3})
    base = [feat("f"), feat("pp", key="price_position", params={"bounds": BB}), fc_label("lfc"),
            {"id": "l", "type": "label", "outcome": "log_return", "labeling_rule": "fixed_threshold",
             "timeframe": "tx_5m", "params": {"horizon": 2, "n_classes": 2}}, up]
    with_arm = mdl("dn", ["f", "up"], "xgboost", branches=[{"id": "main", "inputs": ["f"], "window": 12},
                                                           {"id": "u", "inputs": ["up"], "window": 1, "encoder": "last_value"}])
    without = mdl("dn", ["f", "up"], "xgboost",
                  branches=[{"id": "main", "inputs": ["f"], "window": 12},
                            {"id": "u", "inputs": ["up"], "window": 1, "encoder": "last_value", "role": "mask"}])
    return spec(*base, without if mask else with_arm)


class SharedMaskTests(unittest.TestCase):
    def test_with_without_same_samples(self):
        ga, gb = _with_without(False), _with_without(True)
        validate_graph_spec(gb)
        ra, fa, _ = ev.run_train_v2(ga, MK, phase=1)
        rb, fb, _ = ev.run_train_v2(gb, MK, phase=1)
        sa, sb = ra["dn"]["training_meta"]["sample_sets"], rb["dn"]["training_meta"]["sample_sets"]
        self.assertEqual(fa.hash, fb.hash)
        self.assertEqual((sa["train"], sa["val"]), (sb["train"], sb["val"]))
        self.assertEqual(sb["mask_branches"], ["u"])
        # 無機率組的模型輸入確實少了上游的 9 欄
        la = ra["dn"]["v2"]["input_layout"]["f_total"]
        lb = rb["dn"]["v2"]["input_layout"]["f_total"]
        self.assertEqual(la - lb, 9)
        # 不加遮罩時樣本數較多（上游第一段沒有預測的樣本也會進來）
        gc = _with_without(True)
        gc["nodes"] = [n for n in gc["nodes"] if n["id"] not in ("up", "lfc", "pp")][:-1] + [mdl("dn", ["f"], "xgboost")]
        rc, _, _ = ev.run_train_v2(gc, MK, phase=1)
        self.assertGreater(rc["dn"]["training_meta"]["sample_sets"]["train"]["n"], sa["train"]["n"])
        # holdout：同一組樣本
        ga["end"] = gb["end"] = "2024-01-15"
        outs = []
        for g in (ga, gb):
            res, _, _ = ev.run_train_v2(g, MK, phase=1)
            from training.v2_records import evaluate_entry

            out = ev.run_infer_v2(g, artifacts_of(res, g), MK, "2024-01-10", "2024-01-15")
            _, st = evaluate_entry({**out, "frame_hash": out["frame"].hash}, task_type="classification",
                                   model_config=out["model_config"], stats=res["dn"]["train_stats"])
            outs.append(st["sample_set"])
        self.assertEqual(outs[0], outs[1])


def _ab(mask: bool, key="lstm"):
    """MCP 驗收 A／B 的縮小版：pp＋cp（5 欄）＋上游 9 類機率；B 的上游分支 role=mask。"""
    up = mdl("up", ["pp", "cp"], "xgboost", window=1, output_mode="probability",
             branches=[{"id": "main", "inputs": ["pp", "cp"], "window": 1, "encoder": "last_value"}],
             upstream_policy={"strategy": "forward_chaining", "k": 3})
    u = {"id": "u", "inputs": ["up"], "window": 1, "encoder": "last_value", **({"role": "mask"} if mask else {})}
    dn = mdl("dn", ["pp", "cp", "up"], key, branches=[{"id": "main", "inputs": ["pp", "cp"], "window": 12}, u])
    return spec(feat("pp", key="price_position", params={"bounds": BB}), feat("cp", key="candle_parts"),
                fc_label(), up, dn)


class V2SummaryTests(unittest.TestCase):
    """2026-11-21：job.result 保留 v2（input_layout 等）；舊任務由模型產物唯讀補回。"""

    @classmethod
    def setUpClass(cls):
        cls.runs = {}
        for name, mask in (("A", False), ("B", True)):
            g = _ab(mask)
            r, f, _ = ev.run_train_v2(g, MK, phase=1)
            cls.runs[name] = (g, r, f)

    @staticmethod
    def _art(g, r, nid):
        import hashlib

        return {"model_config": r[nid]["model_config"], "preprocessing_state": r[nid]["preprocessing_state"],
                "graph_spec_snapshot": g, "warmup_policy": r[nid]["v2"]["warmup_policy"],
                "history_requirements": r[nid]["v2"]["history_requirements"],
                "fingerprint": hashlib.sha256(bytes(r[nid]["weights"])).hexdigest()}

    def test_summary_keeps_v2(self):
        from training.worker import _summarize

        for name, f_total, fmt in (("A", 14, "input_layout"), ("B", 5, "legacy_single_branch")):
            with self.subTest(arm=name):
                g, r, _ = self.runs[name]
                s = _summarize(r, g, phase=1)
                v2 = s["dn"]["v2"]
                self.assertEqual(v2["input_layout"], r["dn"]["v2"]["input_layout"])
                self.assertEqual(v2["input_layout"]["f_total"], f_total)
                self.assertEqual(v2["model_input_format"], fmt)
                self.assertNotIn("weights", s["dn"])
                self.assertIn("v2", s["up"])
        la = self.runs["A"][1]["dn"]["v2"]["input_layout"]["branches"]
        self.assertEqual([b["cols"] for b in la], [[0, 5], [5, 14]])
        self.assertEqual(self.runs["B"][1]["dn"]["v2"]["mask_branches"], ["u"])

    def test_single_branch_model_format_unchanged(self):
        # B 是單分支：模型不收 input_layout（權重格式同舊版），只有摘要多了 v2
        _, rb, _ = self.runs["B"]
        self.assertNotIn("input_layout", rb["dn"]["model_config"])
        self.assertEqual(rb["dn"]["model_config"]["n_features"], 5)
        _, ra, _ = self.runs["A"]
        self.assertEqual(ra["dn"]["model_config"]["input_layout"]["f_total"], 14)

    def test_backfill_from_artifact(self):
        from services.model_artifacts import v2_from_artifact

        for name in ("A", "B"):
            g, r, f = self.runs[name]
            for nid in ("dn", "up"):
                with self.subTest(arm=name, node=nid):
                    out = v2_from_artifact(nid, self._art(g, r, nid), f.hash)
                    self.assertTrue(out["backfilled"])
                    self.assertEqual(out["input_layout"], r[nid]["v2"]["input_layout"])
                    self.assertEqual(out["model_input_format"], r[nid]["v2"]["model_input_format"])
                    self.assertEqual(out["mask_branches"], r[nid]["v2"]["mask_branches"])
                    self.assertEqual(out["weights_fingerprint"], r[nid]["v2"]["weights_fingerprint"])
                    self.assertEqual(out["frame_hash"], r[nid]["v2"]["frame_hash"])
                    self.assertIn("input_layout", out["source"])
                    self.assertEqual(out["unknown"], ["alignment", "upstream_policy"])
        g, r, _ = self.runs["B"]
        self.assertTrue(v2_from_artifact("dn", self._art(g, r, "dn"))["source"]["input_layout"]
                        .startswith("model_artifacts.model_config.n_features"))
        # 沒有產物、或來源不一致：維持未知
        self.assertIsNone(v2_from_artifact("dn", None)["input_layout"])
        bad = self._art(g, r, "dn")
        bad["model_config"] = {**bad["model_config"], "n_features": 6}
        self.assertIsNone(v2_from_artifact("dn", bad)["input_layout"])
        self.assertIn("input_layout", v2_from_artifact("dn", bad)["unknown"])

    def test_backfill_job_response(self):
        # get_job 的形狀（job_id 鍵）；DB 讀取換成本地產物；已有 v2 的節點不動
        import asyncio
        from unittest import mock

        from services import model_artifacts as ma

        g, r, f = self.runs["A"]
        arts = {nid: self._art(g, r, nid) for nid in ("up", "dn")}
        keep = {"input_layout": "原值"}
        job = {"job_id": "c0237875-f25e-4858-b56b-c3923e2c277c", "job_type": "train", "graph_spec": g,
               "result": {"dn": {"training_meta": {"frame": {"hash": f.hash}}}, "up": {"v2": keep}}}
        with mock.patch.object(ma, "v2_artifact_sources", mock.AsyncMock(return_value=arts)) as src:
            out = asyncio.run(ma.backfill_v2(job))
        src.assert_awaited_once_with("c0237875-f25e-4858-b56b-c3923e2c277c")
        self.assertEqual(out["result"]["dn"]["v2"]["input_layout"]["f_total"], 14)
        self.assertEqual(out["result"]["dn"]["v2"]["frame_hash"], f.hash)
        self.assertIs(out["result"]["up"]["v2"], keep)


class HoldoutCountTests(unittest.TestCase):
    """2026-11-21：Phase 3 n_unmatured 只算 label_unmatured；缺格／跨時段等另列 excluded_reasons。"""

    def test_counts_split_by_reason(self):
        from training.v2_records import evaluate_entry, holdout_description, holdout_evaluations

        g = spec(feat("f"), reg_label(), mdl("m", ["f"], "xgboost"))
        g["end"] = "2024-01-15"
        res, _, _ = ev.run_train_v2(g, MK, phase=1)
        out = ev.run_infer_v2(g, artifacts_of(res, g), MK, "2024-01-10", "2024-01-15")
        raw, st = evaluate_entry({**out, "frame_hash": out["frame"].hash}, task_type="regression",
                                 model_config=out["model_config"], stats=res["m"]["train_stats"])
        reasons = st["excluded_reasons"]
        self.assertGreater(reasons.get("label_cross_session", 0), 0)
        self.assertEqual(st["n_unmatured"], reasons.get("label_unmatured", 0))
        self.assertEqual(st["n_excluded"], sum(reasons.values()))
        self.assertEqual(st["n_included"] + st["n_excluded"], st["n_predicted"])
        self.assertLess(st["n_unmatured"], st["n_excluded"])
        ev_ = holdout_evaluations("m", raw, specs=None, phase=3, period={"start": "2024-01-10", "end": "2024-01-15"},
                                  stats_out=st)
        d = ev_[0]["dataset"]
        self.assertEqual((d["n_unmatured"], d["n_excluded"], d["excluded_reasons"]),
                         (st["n_unmatured"], st["n_excluded"], reasons))
        self.assertIn("跨時段", holdout_description(st, {}))

    def test_backfill_old_summary(self):
        from training.v2_records import backfill_holdout_counts

        def old():
            ds = {"split": "holdout", "n_samples": 2507, "n_unmatured": 65, "n_skipped": 0,
                  "task_data_range": {"start": "2024-02-02", "end": "2024-02-08"}, "description": "65 筆未成熟不評估"}
            return {"holdout": {"n_points": 2572, "n_predicted": 2572, "n_included": 2507, "n_unmatured": 65,
                                "n_skipped": 0, "cutoff_ts": "2024-02-06 05:00:00"},
                    "evaluations": [{"dataset": ds}, {"dataset": {"split": "val", "n_unmatured": 0}}]}

        r = old()
        reasons = {"label_gap": 40, "label_cross_session": 20, "label_unmatured": 5}
        self.assertTrue(backfill_holdout_counts(r, reasons))
        h = r["holdout"]
        self.assertEqual((h["n_unmatured"], h["n_excluded"], h["excluded_reasons"]), (5, 65, reasons))
        self.assertEqual(h["backfilled"]["n_unmatured_stored"], 65)
        d = r["evaluations"][0]["dataset"]
        self.assertEqual((d["n_unmatured"], d["n_excluded"]), (5, 65))
        self.assertIn("缺格 40", d["description"])
        self.assertNotIn("backfilled", r["evaluations"][1]["dataset"])
        self.assertFalse(backfill_holdout_counts(r, reasons))  # 已補正／新格式不再動
        r2 = old()
        self.assertFalse(backfill_holdout_counts(r2, {"label_gap": 1}))  # 總數不符：維持原值、標未知
        self.assertEqual(r2["holdout"]["n_unmatured"], 65)
        self.assertEqual(r2["holdout"]["backfilled"]["status"], "unknown")


class DotConfigTests(unittest.TestCase):
    """Dot 2026-11-21 本機驗收配置（window 60）。"""

    COMMON = {"heads": "single", "shared": {"dense": 64, "dropout": 0.0},
              "optimizer": {"type": "adam", "lr": 1e-3, "weight_decay": 0.0}, "batch_size": 64, "epochs": 2,
              "early_stopping": {"monitor": "val_loss", "patience": 0}, "seed": 42}
    CFG = {
        "cnn_lstm": {"conv_layers": [{"filters": 32, "kernel_size": 1, "activation": "relu"}], "pool_size": 3,
                     "lstm_layers": [{"units": 64, "dropout": 0.2}], "bidirectional": False},
        "gru": {"gru_layers": [{"units": 64, "dropout": 0.2}], "bidirectional": False},
        "transformer": {"transformer": {"d_model": 64, "n_heads": 4, "n_layers": 2, "ff_dim": 128, "dropout": 0.1,
                                        "positional": "sinusoidal", "pooling": "last"}},
    }

    def test_configs_train_and_cnn_steps(self):
        import torch

        from training.registry import seq_models as sm

        for key, cfg in self.CFG.items():
            with self.subTest(key=key):
                params = {**cfg, **self.COMMON}
                g = spec(feat("f"), fc_label(), mdl("m", ["f"], key, window=60, params=params))
                validate_graph_spec(g)
                r, _, _ = ev.run_train_v2(g, MK, phase=1)
                self.assertIn("brier", r["m"]["evaluation"]["best"])
        enc, dim = sm._encoder("cnn_lstm", {**self.CFG["cnn_lstm"]}, 5, 60)
        seen = {}
        enc.rnn.rnns[0].register_forward_hook(lambda m, i, o: seen.update(steps=i[0].shape[1]))
        enc(torch.zeros(2, 60, 5))
        self.assertEqual((seen["steps"], dim), (20, 64))
        self.assertEqual(enc.conv[0].padding, "same")


if __name__ == "__main__":
    unittest.main()
