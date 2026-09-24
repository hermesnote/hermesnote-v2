"""LSTM（key="lstm"）三軸獨立組合的驗收：方向（單／雙向）× Attention（無／有）× 輸出頭（單／雙）＝ 8 種組合。

每種組合都驗證：能訓練、輸出形狀符合宣告、best／last 兩組權重與評估都在、保存後重載推論逐值一致、
接法（sequence_summary）記進產物。另外直接證明雙向序列摘要取的是兩個方向各自的最終 hidden state，
以及單頭分類／回歸、class_weight、patience=0 跑滿長輪數。雙頭的細節與管線行為見 tests/test_lstm_dual.py。

執行：python -m unittest tests.test_lstm
"""

import itertools
import json
import unittest

from tests import contract_harness as H  # 會設定 CPU 與 sys.path

import numpy as np
import torch

from training import graph as graph_mod
from training.graph_refs import GraphValidationError, validate_graph_spec
from training.registry import architectures, lstm

KEY = "lstm"
ATTN = {"type": "additive", "dim": 4}


def data(task: str, n_train=120, n_val=60, seed=1):
    rng = np.random.default_rng(seed)

    def mk(n):
        X = rng.normal(size=(n, 8, 3)).astype(np.float32)
        y = 0.4 * X[:, -1, 0] + rng.normal(0, 0.05, n)
        return X, (y if task == "regression" else (y > 0).astype(int))

    return mk(n_train), mk(n_val)


def payload(bidirectional: bool, attention: bool, heads: str, **over) -> dict:
    p = {
        "lstm_layers": [{"units": 8, "dropout": 0.1}, {"units": 6, "dropout": 0.1}],
        "bidirectional": bidirectional, "attention": dict(ATTN) if attention else None, "heads": heads,
        "shared": {"dense": 6, "dropout": 0.1}, "optimizer": {"type": "adam", "lr": 3e-3, "weight_decay": 0.0}, "batch_size": 32,
        "epochs": 3, "early_stopping": {"monitor": "val_loss", "patience": 0}, "seed": 3,
    }
    if heads == "dual":
        p.update(direction_rule={"op": ">=", "threshold": 0.0}, loss_weights={"regression": 0.8, "direction": 0.2},
                 early_stopping={"monitor": "val_rmse", "patience": 0})
    p.update(over)
    return p


class EightCombinationTests(unittest.TestCase):
    def test_all_eight_combinations_train_reload_and_report_best_last(self):
        for bidirectional, attention, heads in itertools.product((False, True), (False, True), ("single", "dual")):
            with self.subTest(bidirectional=bidirectional, attention=attention, heads=heads):
                task = "regression" if heads == "dual" else "classification"
                (Xt, yt), (Xv, yv) = data(task)
                res = architectures.train_model(KEY, Xt, yt, Xv, yv, payload(bidirectional, attention, heads), None,
                                                task_type=task, n_classes=2)
                cfg = res["model_config"]
                json.dumps(cfg)
                self.assertEqual((cfg["config"]["bidirectional"], cfg["config"]["attention"] is not None, cfg["mode"]),
                                 (bidirectional, attention, "dual" if heads == "dual" else "cls"))
                self.assertIn("雙向" if bidirectional else "單向", cfg["sequence_summary"])
                self.assertEqual("Attention" in cfg["sequence_summary"], attention)
                self.assertEqual(sorted(res["evaluation"]), ["best", "last"])
                self.assertIn("last", res["final_metrics"])
                for weights in (res["weights"], res["weights_last"]):
                    loaded = architectures.load_model(KEY, weights, cfg)
                    if heads == "dual":
                        out = loaded["predict_outputs"](Xv)
                        self.assertEqual((out["regression"].shape, out["direction_probability"].shape), ((60,), (60,)))
                    else:
                        self.assertEqual(loaded["predict_proba"](Xv).shape, (60, 2))
                loaded_best = architectures.load_model(KEY, res["weights"], cfg)
                if heads == "dual":
                    np.testing.assert_allclose(res["predict_outputs"](Xv)["regression"],
                                               loaded_best["predict_outputs"](Xv)["regression"], atol=1e-6)
                else:
                    np.testing.assert_allclose(res["predict_proba"](Xv), loaded_best["predict_proba"](Xv), atol=1e-6)

    def test_combinations_through_graph_with_declared_outputs(self):
        """經過 run_graph（提交驗證 → 訓練 → 宣告輸出形狀檢查）；雙頭輸出可被下游具名引用。"""
        for bidirectional, attention in itertools.product((False, True), (False, True)):
            with self.subTest(bidirectional=bidirectional, attention=attention):
                down = H._model("down", "xgboost", [{"node": "m", "output": "direction_probability"}, "fa"], "lr",
                                params=H.xgb_params(n_estimators=3))
                spec = {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
                    H._feat("fa", "raw_price"), H._label("lr", "log_return", "identity"),
                    H._model("m", KEY, ["fa"], "lr", params=payload(bidirectional, attention, "dual")), down]}
                validate_graph_spec(spec)
                results = graph_mod.run_graph(spec, H.data_loader)
                self.assertEqual(sorted(results["m"]["output_specs"]), ["default", "direction_probability", "regression"])
                self.assertIn("final_metrics", results["down"])


class BidirectionalSummaryTests(unittest.TestCase):
    def test_backward_final_state_lives_at_sequence_start(self):
        """底層事實：反向方向「看完整條序列」的最終狀態在 out[:, 0]，不是 out[:, -1]。"""
        torch.manual_seed(0)
        layer = torch.nn.LSTM(4, 6, batch_first=True, bidirectional=True)
        out, (h_n, _) = layer(torch.randn(3, 10, 4))
        torch.testing.assert_close(out[:, -1, :6], h_n[0])
        torch.testing.assert_close(out[:, 0, 6:], h_n[1])
        self.assertFalse(torch.allclose(out[:, -1, 6:], h_n[1]))

    def test_network_uses_both_directions_final_hidden_state(self):
        cfg = architectures.validate(KEY, payload(True, False, "single"), {"task_type": "classification"})
        torch.manual_seed(0)
        net = lstm.build_net(4, cfg, 2).eval()
        x = torch.randn(3, 10, 4)
        with torch.no_grad():
            actual = net(x)
            seq = x
            for layer, drop in zip(net.lstms, net.drops):
                seq, (h_n, _) = layer(seq)
                seq = drop(seq)
            correct = net.head(net.shared_drop(torch.relu(net.shared(torch.cat([h_n[0], h_n[1]], dim=-1)))))
            naive = net.head(net.shared_drop(torch.relu(net.shared(seq[:, -1, :]))))
        torch.testing.assert_close(actual, correct)
        self.assertFalse(torch.allclose(actual, naive))


class SingleHeadTests(unittest.TestCase):
    def test_regression_single_head_with_target_scaling(self):
        (Xt, yt), (Xv, yv) = data("regression")
        p = payload(False, False, "single", target_scaling="standardize",
                    early_stopping={"monitor": "val_rmse", "patience": 0})
        res = architectures.train_model(KEY, Xt, yt, Xv, yv, p, None, task_type="regression")
        self.assertEqual(res["model_config"]["mode"], "reg")
        pred = res["predict"](Xv)
        self.assertAlmostEqual(float(np.sqrt(np.mean((pred - yv) ** 2))), res["final_metrics"]["val_rmse"], places=6)
        for key in ("mse", "rmse", "mae", "r2", "baseline_mean", "baseline_zero"):
            self.assertIn(key, res["evaluation"]["best"])

    def test_classification_three_classes_and_class_weight(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 8, 3)).astype(np.float32)
        y = np.digitize(X[:, -1, 0], [-0.5, 0.5])
        p = payload(False, True, "single", class_weight="balanced",
                    early_stopping={"monitor": "val_accuracy", "patience": 0})
        res = architectures.train_model(KEY, X[:150], y[:150], X[150:], y[150:], p, None,
                                        task_type="classification", n_classes=3)
        self.assertEqual(res["predict_proba"](X[150:]).shape, (50, 3))
        self.assertEqual(res["final_metrics"]["monitor_mode"], "max")
        self.assertEqual(len(res["evaluation"]["best"]["confusion_matrix"]), 3)

    def test_monitor_options_depend_on_heads_and_task(self):
        ok = lambda p, task: architectures.validate(KEY, p, {"task_type": task})
        ok(payload(False, False, "single", early_stopping={"monitor": "val_accuracy", "patience": 1}), "classification")
        ok(payload(False, False, "single", early_stopping={"monitor": "val_mae", "patience": 1}), "regression")
        with self.assertRaisesRegex(ValueError, "monitor"):
            ok(payload(False, False, "single", early_stopping={"monitor": "val_rmse", "patience": 1}), "classification")
        with self.assertRaisesRegex(ValueError, "monitor"):
            ok(payload(False, False, "dual", early_stopping={"monitor": "val_accuracy", "patience": 1}), "regression")

    def test_dual_only_fields_rejected_for_single_head(self):
        with self.assertRaisesRegex(ValueError, "loss_weights"):
            architectures.validate(KEY, payload(False, False, "single", loss_weights={"regression": 1, "direction": 1}),
                                   {"task_type": "classification"})
        with self.assertRaisesRegex(ValueError, "target_scaling"):  # 分類目標不適用目標縮放
            architectures.validate(KEY, payload(False, False, "single", target_scaling="standardize"),
                                   {"task_type": "classification"})

    def test_attention_list_rejected_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "不支援多元件組合"):
            architectures.validate(KEY, {**payload(False, False, "single"), "attention": [ATTN, ATTN]},
                                   {"task_type": "classification"})


class TrainingControlTests(unittest.TestCase):
    def test_patience_zero_runs_300_epochs(self):
        (Xt, yt), (Xv, yv) = data("classification", n_train=40, n_val=20)
        seen = []
        p = payload(False, False, "single", epochs=300, lstm_layers=[{"units": 2, "dropout": 0.0}],
                    shared={"dense": 2, "dropout": 0.0}, optimizer={"type": "adam", "lr": 1e-2, "weight_decay": 0.0}, batch_size=64)
        res = architectures.train_model(KEY, Xt, yt, Xv, yv, p, lambda e, m: seen.append(e),
                                        task_type="classification", n_classes=2)
        fm = res["final_metrics"]
        self.assertEqual((len(seen), fm["epochs_run"], fm["stopped_early"], fm["patience"]), (300, 300, False, 0))
        self.assertLessEqual(fm["best_epoch"], 299)

    def test_positive_patience_stops_and_keeps_best(self):
        (Xt, yt), (Xv, yv) = data("regression")
        p = payload(False, False, "dual", epochs=60, optimizer={"type": "adam", "lr": 3e-2, "weight_decay": 0.0}, batch_size=16,
                    early_stopping={"monitor": "val_rmse", "patience": 3})
        res = architectures.train_model(KEY, Xt, yt, Xv, yv, p, None, task_type="regression")
        fm = res["final_metrics"]
        if fm["stopped_early"]:
            self.assertEqual(fm["stopped_epoch"] - fm["best_epoch"], 3)
        self.assertLessEqual(fm["val_rmse"], fm["last"]["val_rmse"] + 1e-12)  # best 一定不比 last 差

    def test_submission_rejects_negative_patience(self):
        spec = {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
            H._feat("fa", "raw_price"), H._label("lc", "log_return", "fixed_threshold", n_classes=2),
            H._model("m", KEY, ["fa"], "lc", params=payload(False, False, "single",
                                                              early_stopping={"monitor": "val_loss", "patience": -1}))]}
        with self.assertRaises(GraphValidationError) as cm:
            validate_graph_spec(spec)
        self.assertTrue(any("patience" in i for i in cm.exception.issues))


if __name__ == "__main__":
    unittest.main(verbosity=2)
