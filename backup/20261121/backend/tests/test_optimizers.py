"""優化器元件（training/registry/optimizers.py）：Adam 參數預設與自訂值真的生效、補齊預設後的完整設定
存進產物、模型／優化器相容性由登記資訊實際檢查、新增優化器不用改模型程式。

執行：python -m unittest tests.test_optimizers
"""

import json
import unittest

from tests import contract_harness as H  # noqa: F401  （CPU＋sys.path）

import numpy as np

from training import graph as graph_mod
from training.registry import architectures, lstm, optimizers

CTX = {"task_type": "classification"}
ADAM_DEFAULTS = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}


def data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6, 3)).astype(np.float32)
    return X, (X[:, -1, 0] > 0).astype(int)


def train(optimizer: dict, spy: list | None = None):
    """訓練一個小 LSTM；spy 不是 None 時記下訓練實際建立的優化器（param_groups）。"""
    X, y = data()
    orig = lstm.build_optimizer
    if spy is not None:
        def recording(params, cfg):
            opt = orig(params, cfg)
            spy.append((type(opt).__name__, {k: v for k, v in opt.param_groups[0].items() if k != "params"}))
            return opt
        lstm.build_optimizer = recording
    try:
        return architectures.train_model("lstm", X[:60], y[:60], X[60:], y[60:],
                                         H.lstm_params(optimizer=optimizer, epochs=2), None,
                                         task_type="classification", n_classes=2)
    finally:
        lstm.build_optimizer = orig


class AdamParameterTests(unittest.TestCase):
    def test_defaults_are_filled_used_and_saved(self):
        spy = []
        res = train({"type": "adam", "lr": 3e-3, "weight_decay": 0.0}, spy)
        saved = res["model_config"]["config"]["optimizer"]
        self.assertEqual(saved, {"type": "adam", "lr": 3e-3, "weight_decay": 0.0, **ADAM_DEFAULTS})
        json.dumps(res["model_config"])
        name, group = spy[0]
        self.assertEqual((name, group["betas"], group["eps"], group["lr"], group["weight_decay"]),
                         ("Adam", (0.9, 0.999), 1e-8, 3e-3, 0.0))

    def test_custom_values_take_effect_and_are_saved(self):
        custom = {"type": "adam", "lr": 1e-2, "weight_decay": 1e-4, "beta1": 0.5, "beta2": 0.95, "eps": 1e-6}
        spy = []
        res = train(custom, spy)
        self.assertEqual(res["model_config"]["config"]["optimizer"], custom)
        _, group = spy[0]
        self.assertEqual((group["betas"], group["eps"], group["lr"], group["weight_decay"]),
                         ((0.5, 0.95), 1e-6, 1e-2, 1e-4))
        # 真的影響訓練：同 seed、只改 beta1，權重不同
        base = train({"type": "adam", "lr": 1e-2, "weight_decay": 1e-4})
        self.assertNotEqual(res["weights"], base["weights"])

    def test_invalid_values_rejected_with_field_path(self):
        cases = [({"beta1": 1.0}, "optimizer.beta1"), ({"beta2": -0.1}, "optimizer.beta2"),
                 ({"eps": 0}, "optimizer.eps"), ({"lr": 0}, "optimizer.lr"), ({"weight_decay": -1}, "optimizer.weight_decay"),
                 ({"momentum": 0.9}, "momentum")]
        for over, needle in cases:
            with self.subTest(over=over), self.assertRaisesRegex(ValueError, needle):
                architectures.validate("lstm", H.lstm_params(optimizer={"type": "adam", "lr": 1e-3, "weight_decay": 0.0, **over}), CTX)
        for bad, needle in ((None, "不能是 null"), ({"type": "rmsprop"}, "未登記")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, needle):
                architectures.validate("lstm", H.lstm_params(optimizer=bad), CTX)
        with self.assertRaisesRegex(ValueError, "optimizer 為必填"):
            architectures.validate("lstm", {k: v for k, v in H.lstm_params().items() if k != "optimizer"}, CTX)


class RegistryAndCompatibilityTests(unittest.TestCase):
    def test_capability_and_component_listing(self):
        by_key = {a["key"]: a for a in architectures.list_available()}
        slots = {s["slot_name"]: s for s in by_key["lstm"]["slots"]}
        self.assertEqual(slots["optimizer"]["component_registry"], "optimizer")
        self.assertEqual(by_key["xgboost"]["capabilities"]["optimizer"], "fixed_off")
        adam = {c["key"]: c for c in architectures.list_components("optimizer")}["adam"]
        self.assertEqual([f["name"] for f in adam["params_schema"]], ["lr", "weight_decay", "beta1", "beta2", "eps"])
        self.assertEqual(adam["slot_compatibility"], [["lstm", "optimizer"]])
        with self.assertRaises(KeyError):
            architectures.list_components("nope")

    def test_xgboost_does_not_accept_optimizer(self):
        with self.assertRaisesRegex(ValueError, "optimizer"):
            architectures.validate("xgboost", H.xgb_params(optimizer={"type": "adam", "lr": 1e-3, "weight_decay": 0}), CTX)

    def test_incompatible_optimizer_rejected(self):
        optimizers.OPTIMIZER_REGISTRY["gru_only"] = {**optimizers.OPTIMIZER_REGISTRY["adam"], "key": "gru_only",
                                                     "slot_compatibility": [("gru", "optimizer")]}
        try:
            with self.assertRaisesRegex(ValueError, "相容"):
                architectures.validate("lstm", H.lstm_params(optimizer={"type": "gru_only", "lr": 1e-3, "weight_decay": 0}), CTX)
        finally:
            optimizers.OPTIMIZER_REGISTRY.pop("gru_only")

    def test_new_optimizer_registers_without_touching_model_code(self):
        @optimizers.register("sgd_test", label="SGD（測試）", slot_compatibility=[("lstm", "optimizer")])
        class Sgd:
            params_schema = [{"name": "lr", "type": "float", "required": True, "exclusive_min": 0},
                             {"name": "momentum", "type": "float", "default": 0.0, "min": 0}]

            @staticmethod
            def build(parameters, cfg):
                import torch
                return torch.optim.SGD(parameters, lr=cfg["lr"], momentum=cfg["momentum"])
        try:
            spy = []
            res = train({"type": "sgd_test", "lr": 0.05}, spy)
            self.assertEqual(spy[0][0], "SGD")
            self.assertEqual(res["model_config"]["config"]["optimizer"], {"type": "sgd_test", "lr": 0.05, "momentum": 0.0})
        finally:
            optimizers.OPTIMIZER_REGISTRY.pop("sgd_test")


class TrainingPathTests(unittest.TestCase):
    def test_phase2_inherits_optimizer_and_graph_trains(self):
        """經 run_graph（Phase 1 random 與 chronological）訓練，產物設定帶完整優化器設定。"""
        opt = {"type": "adam", "lr": 3e-3, "weight_decay": 0.0, "beta2": 0.99}
        for split in ("random", "chronological"):
            with self.subTest(split=split):
                spec = {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
                    H._feat("fa", "raw_price"), H._label("lc", "log_return", "fixed_threshold", n_classes=2),
                    H._model("m", "lstm", ["fa"], "lc", split_strategy=split, params=H.lstm_params(optimizer=opt))]}
                res = graph_mod.run_graph(spec, H.data_loader)["m"]
                self.assertEqual(res["model_config"]["config"]["optimizer"],
                                 {"type": "adam", "lr": 3e-3, "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.99, "eps": 1e-8})
                self.assertEqual(sorted(res["evaluation"]), ["best", "last"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
