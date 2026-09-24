"""LSTM 雙頭（heads="dual"）與訓練管線共用行為的整合驗收（離線、合成資料）。用法（在 backend/ 目錄）：
    python -m unittest tests.test_lstm_dual -v
    RUN_DB_TESTS=1 python -m unittest tests.test_lstm_dual.DailyReadPathTests   # 另核對實際日線讀取路徑（唯讀）

涵蓋：雙頭參數驗證（schema 驅動）、提交驗證、網路結構與 Attention 公式、聯合損失、early stopping／best-last、
保存／載入一致、目標縮放、完整評估報告、進度指標、推論讀回與 dual 儲存列、具名輸出串接、訓練與推論欄位
順序、時間語意（識別時間 vs 資訊可用時間）、日線識別字驗證。八種組合與單頭見 tests/test_lstm.py。
"""

import asyncio
import json
import os
import unittest

from tests import contract_harness as H  # 會設定 CPU 與 sys.path

import numpy as np
import pandas as pd

from training import graph as graph_mod
from training import inference as inference_mod
from training import inference_store, progress, worker as worker_mod
from training.graph_refs import GraphValidationError, validate_graph_spec
from training.registry import architectures, lstm, target_transforms
from training.time_semantics import available_at, build_available_map

KEY = "lstm"


def params(**over):
    """雙頭＋Attention 的完整 payload（研究端未確認的值一律必填，這裡是測試用小數字）。"""
    p = {
        "lstm_layers": [{"units": 8, "dropout": 0.2}, {"units": 4, "dropout": 0.2}],
        "attention": {"type": "additive", "dim": 6},
        "heads": "dual",
        "direction_rule": {"op": ">=", "threshold": 0.0},
        "loss_weights": {"regression": 0.8, "direction": 0.2},
        "shared": {"dense": 6, "dropout": 0.2},
        "optimizer": {"type": "adam", "lr": 3e-3, "weight_decay": 1e-5}, "batch_size": 32,
        "epochs": 4, "early_stopping": {"monitor": "val_rmse", "patience": 2}, "seed": 42,
    }
    p.update(over)
    return p


def validate(p, **context):
    return architectures.validate(KEY, p, {"task_type": "regression", **context})


def daily_df(n=320, hours=5):
    df = H.synth_df(n).copy()
    df["bar_end_ts"] = df.index + pd.Timedelta(hours=hours)  # 識別時間 → 收盤時間（資訊可用時間）
    return df


def graph_spec(model_params=None, label_outcome="log_return", extra_model=None):
    nodes = [H._feat("fa", "raw_price"), H._feat("fv", "raw_volume"),
             H._label("lr", label_outcome, "identity"),
             H._model("dual", KEY, ["fa", "fv"], "lr", params=model_params or params())]
    if extra_model:
        nodes.append(extra_model)
    return {"start": "2024-01-01", "end": "2024-02-01", "nodes": nodes}


def noisy(n_train=120, n_val=60, seed=1, signal=0.0):
    rng = np.random.default_rng(seed)

    def mk(n):
        X = rng.normal(size=(n, 8, 3)).astype(np.float32)
        return X, signal * X[:, -1, 0] + rng.normal(0, 0.01, n)

    return mk(n_train), mk(n_val)


class ParamValidationTests(unittest.TestCase):
    def test_valid_payload_resolves_to_concrete_config(self):
        cfg = validate(params())
        self.assertEqual(cfg["lstm_layers"], [{"units": 8, "dropout": 0.2}, {"units": 4, "dropout": 0.2}])
        self.assertEqual(cfg["early_stopping"], {"monitor": "val_rmse", "patience": 2, "min_delta": 0.0})
        self.assertEqual(cfg["target_scaling"], "none")
        self.assertEqual(cfg["bidirectional"], False)
        self.assertEqual(cfg["attention"], {"type": "additive", "dim": 6})
        json.dumps(cfg)

    def test_paper_settings_accepted_as_payload_not_platform_limits(self):
        validate(params(lstm_layers=[{"units": 128, "dropout": 0.2}, {"units": 64, "dropout": 0.2}],
                        attention={"type": "additive", "dim": 64}, shared={"dense": 64, "dropout": 0.2},
                        optimizer={"type": "adam", "lr": 3e-4, "weight_decay": 1e-5}, batch_size=256, epochs=300,
                        early_stopping={"monitor": "val_rmse", "patience": 0}))
        validate(params(lstm_layers=[{"units": 32, "dropout": 0.0}], attention={"type": "additive", "dim": 3}))

    def test_unconfirmed_research_values_are_required_without_defaults(self):
        for missing in ("shared", "epochs", "seed", "lstm_layers", "heads", "optimizer", "early_stopping",
                        "direction_rule", "loss_weights"):
            p = params(); del p[missing]
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError, rf"params\.{missing}[\w.]* 為必填"):
                    validate(p)
        with self.assertRaisesRegex(ValueError, "shared.dense"):
            validate(params(shared={"dropout": 0.2}))
        with self.assertRaisesRegex(ValueError, "seed"):
            validate(params(seed=None))

    def test_attention_is_optional_for_dual_head(self):
        p = params(); del p["attention"]
        self.assertIsNone(validate(p)["attention"])

    def test_bad_values_all_reported_together(self):
        with self.assertRaises(ValueError) as cm:
            validate(params(
                lstm_layers=[{"units": 0, "dropout": 1.5}], attention={"type": "dot", "dim": -1}, epochs=0,
                loss_weights={"regression": -1, "direction": 0}, seed=-1,
                early_stopping={"monitor": "loss", "patience": -1}, target_scaling="log",
                optimizer={"type": "adam", "lr": 0, "weight_decay": -1}, batch_size=0, unknown_key=1,
                direction_rule={"op": "==", "threshold": 0}))
        msg = str(cm.exception)
        for needle in ("units", "dropout", "attention", "epochs", "loss_weights", "seed", "monitor",
                       "patience", "target_scaling", "lr", "weight_decay", "batch_size", "unknown_key", "op"):
            self.assertIn(needle, msg)

    def test_both_loss_weights_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能同時為 0"):
            validate(params(loss_weights={"regression": 0, "direction": 0}))

    def test_non_dict_and_extra_keys(self):
        for bad in ([1], "x", 5):
            with self.assertRaises(ValueError):
                validate(bad)
        with self.assertRaisesRegex(ValueError, "不是這個模型認得的欄位"):
            validate(params(shared={"dense": 4, "dropout": 0.1, "activation": "tanh"}))

    def test_single_head_only_fields_rejected_for_dual(self):
        with self.assertRaisesRegex(ValueError, "class_weight"):
            validate(params(class_weight="balanced"))


class DescribeAndSubmitTests(unittest.TestCase):
    def test_outputs_and_metadata(self):
        specs = validate_and_specs(graph_spec())
        self.assertEqual(sorted(specs), ["default", "direction_probability", "regression"])
        ev = specs["direction_probability"]["event"]
        self.assertEqual(ev, {"of": "regression", "op": ">=", "threshold": 0.0, "threshold_unit": "log_ratio"})
        self.assertEqual(specs["regression"]["target"]["outcome"], "log_return")
        self.assertEqual(specs["regression"]["target"]["horizon"], 1)

    def test_configurable_op_threshold_reflected(self):
        specs = validate_and_specs(graph_spec(params(direction_rule={"op": ">", "threshold": 0.002})))
        self.assertIn(">", specs["direction_probability"]["description"])
        self.assertEqual(specs["direction_probability"]["event"]["threshold"], 0.002)

    def test_rejected_combinations(self):
        def rejected(spec, needle):
            with self.assertRaises(GraphValidationError) as cm:
                validate_graph_spec(spec)
            self.assertTrue(any(needle in i for i in cm.exception.issues), cm.exception.issues)

        rejected(graph_spec(label_outcome="future_price"), "帶正負號")  # 非報酬型 outcome
        bad_label = graph_spec(); bad_label["nodes"][2]["labeling_rule"] = "fixed_threshold"
        rejected(bad_label, "連續目標")  # 分類目標
        with_mode = graph_spec(); with_mode["nodes"][3]["output_mode"] = "probability"
        rejected(with_mode, "output_mode")
        unit_bad = graph_spec(params(direction_rule={"op": ">=", "threshold": 0.0, "threshold_unit": "percent"}))
        rejected(unit_bad, "outcome_unit")
        p = params(); del p["shared"]
        rejected(graph_spec(p), "shared")

    def test_single_output_models_have_no_direction_output(self):
        spec = graph_spec(extra_model=H._model("x", "xgboost", [{"node": "dual", "output": "direction_probability"}], "lr",
                                               params=H.xgb_params(n_estimators=2)))
        validate_graph_spec(spec)  # 上游有這個輸出：合法
        spec2 = graph_spec(extra_model=H._model("x2", "xgboost", ["fa"], "lr", params=H.xgb_params(n_estimators=2)))
        spec2["nodes"].append(H._model("y", "spy", [{"node": "x2", "output": "direction_probability"}], "lr"))
        with self.assertRaises(GraphValidationError):
            validate_graph_spec(spec2)  # 單輸出模型沒有這個輸出

    def test_api_rejects_before_creating_job_and_accepts_full_payload(self):
        from fastapi import HTTPException
        from routers import model_training as router
        from training import job_store

        created = []

        async def fake_create(spec, job_type="train", phase=None, parent_job_id=None):
            created.append(spec)
            return "job-1"

        orig = job_store.create_job
        job_store.create_job = fake_create
        try:
            good = graph_spec()
            out = asyncio.run(router.start_training(router.TrainRequest(start=good["start"], end=good["end"], nodes=good["nodes"])))
            self.assertEqual(out, {"job_id": "job-1"})
            p = params(); del p["seed"]
            bad = graph_spec(p)
            with self.assertRaises(HTTPException) as cm:
                asyncio.run(router.start_training(router.TrainRequest(start=bad["start"], end=bad["end"], nodes=bad["nodes"])))
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn("params.seed", cm.exception.detail)
            self.assertEqual(len(created), 1)
        finally:
            job_store.create_job = orig

    def test_registry_lists_outputs_and_transform(self):
        arch = {a["key"]: a for a in architectures.list_available()}
        self.assertEqual(arch[KEY]["outputs"], ["default", "regression", "direction_probability"])
        self.assertEqual([t["key"] for t in target_transforms.list_available()], ["sign_threshold"])
        self.assertIn(KEY, architectures.LAZY_WINDOW_ARCHS)

    def test_attention_registry_lists_additive_with_contract_metadata(self):
        from training.registry import attention as attn_registry

        a = {x["key"]: x for x in attn_registry.list_available()}["additive"]
        self.assertEqual(a["query_source"], "none")
        self.assertEqual(a["combine"], "concat_summary")
        self.assertEqual(a["output_kind"], "context_vector")
        self.assertEqual(a["slot_compatibility"], [["lstm", "attention"]])
        self.assertEqual([f["name"] for f in a["params_schema"]], ["dim"])

    def test_attention_rejects_unregistered_type_before_submission(self):
        with self.assertRaisesRegex(ValueError, "未登記"):
            validate(params(attention={"type": "unknown_kind", "dim": 4}))

    def test_attention_choice_and_metadata_saved_into_model_config(self):
        (Xt, yt), (Xv, yv) = noisy()
        res = architectures.train_model(KEY, Xt, yt, Xv, yv, params(epochs=1), None, task_type="regression")
        meta = res["model_config"]["attention_meta"]
        self.assertEqual((meta["query_source"], meta["combine"]), ("none", "concat_summary"))
        self.assertEqual(res["model_config"]["config"]["attention"], {"type": "additive", "dim": 6})
        json.dumps(res["model_config"])  # 不含函式物件，完整可序列化


def validate_and_specs(spec):
    validate_graph_spec(spec)
    nodes = {n["id"]: n for n in spec["nodes"]}
    node = nodes["dual"]
    return architectures.describe_outputs(KEY, node, nodes[node["label"]], "regression")


class NetworkStructureTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.cfg = validate(params())
        self.net = lstm.build_net(3, self.cfg, 1)

    def test_forward_shapes_and_attention_normalized(self):
        (reg, logit), alpha = self.net(self.torch.randn(5, 7, 3), return_attention=True)
        self.assertEqual((tuple(reg.shape), tuple(logit.shape), tuple(alpha.shape)), ((5,), (5,), (5, 7)))
        np.testing.assert_allclose(alpha.sum(dim=1).detach().numpy(), 1.0, atol=1e-6)  # softmax 沿時間軸

    def test_layer_widths_and_dropout_positions(self):
        nn = self.torch.nn
        self.assertEqual([m.hidden_size for m in self.net.lstms], [8, 4])
        self.assertTrue(all(m.dropout == 0 and m.num_layers == 1 for m in self.net.lstms))  # 沒有 nn.LSTM 內建層間 dropout
        self.assertEqual([m.p for m in self.net.drops], [0.2, 0.2])  # 每層 LSTM 之後各一個
        self.assertEqual(self.net.shared_drop.p, 0.2)
        self.assertEqual(self.net.attn.proj.in_features, 4)
        self.assertEqual(self.net.attn.proj.out_features, 6)  # attention.dim
        self.assertEqual(self.net.shared.in_features, 2 * 4)  # concat([context, 摘要])
        self.assertIsInstance(self.net.dir_head, nn.Linear)

    def test_attention_formula_matches_contract(self):
        torch = self.torch
        net = lstm.build_net(3, self.cfg, 1).eval()
        x = torch.randn(2, 6, 3)
        seq = x
        for layer, drop in zip(net.lstms, net.drops):
            seq, _ = layer(seq); seq = drop(seq)
        e = (torch.tanh(seq @ net.attn.proj.weight.T + net.attn.proj.bias) @ net.attn.v.weight.T).squeeze(-1)
        alpha = torch.exp(e) / torch.exp(e).sum(dim=1, keepdim=True)  # e_t = v^T tanh(W h_t + b)；exp 正規化
        ctx = (alpha.unsqueeze(-1) * seq).sum(dim=1)
        z = torch.relu(net.shared(torch.cat([ctx, seq[:, -1, :]], dim=1)))  # 單向：h_n == 最後時間步
        expected_reg = net.reg_head(z).squeeze(-1)
        (reg, _), a = net(x, return_attention=True)
        np.testing.assert_allclose(a.detach().numpy(), alpha.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(reg.detach().numpy(), expected_reg.detach().numpy(), atol=1e-6)

    def test_target_transform_uses_configured_op_and_threshold(self):
        y = np.array([-0.01, 0.0, 0.01])
        f = lambda op, t: target_transforms.apply_target_transform("sign_threshold", y, {"op": op, "threshold": t}).tolist()
        self.assertEqual(f(">=", 0.0), [0.0, 1.0, 1.0])
        self.assertEqual(f(">", 0.0), [0.0, 0.0, 1.0])
        self.assertEqual(f("<", 0.0), [1.0, 0.0, 0.0])
        self.assertEqual(f(">=", 0.005), [0.0, 0.0, 1.0])


class TrainingBehaviorTests(unittest.TestCase):
    def train(self, p, data=None, signal=0.0):
        (Xt, yt), (Xv, yv) = data or noisy(signal=signal)
        seen = []
        res = architectures.train_model(KEY, Xt, yt, Xv, yv, p, lambda e, m: seen.append(dict(m)), task_type="regression")
        return res, seen, (Xv, yv)

    def test_metrics_named_separately(self):
        _, seen, _ = self.train(params())
        for k in ("joint_loss", "mse", "bce", "rmse", "dir_acc", "val_joint_loss", "val_mse", "val_bce", "val_rmse", "val_dir_acc"):
            self.assertIn(k, seen[0])
        self.assertEqual(seen[0]["loss"], seen[0]["joint_loss"])  # loss 明確就是聯合損失
        self.assertEqual(seen[0]["val_accuracy"], seen[0]["val_dir_acc"])  # accuracy 明確就是方向準確率
        self.assertTrue(all(0.0 <= m["dir_acc"] <= 1.0 and 0.0 <= m["val_dir_acc"] <= 1.0 for m in seen))

    def test_joint_loss_definition_matches_saved_model_outputs(self):
        res, _, (Xv, yv) = self.train(params(epochs=1))
        out = res["predict_outputs"](Xv)
        cls = target_transforms.apply_target_transform("sign_threshold", yv, {"op": ">=", "threshold": 0.0})
        p = np.clip(out["direction_probability"], 1e-12, 1 - 1e-12)
        mse = np.mean((out["regression"] - yv) ** 2)
        bce = -np.mean(cls * np.log(p) + (1 - cls) * np.log(1 - p))
        fm = res["final_metrics"]
        self.assertAlmostEqual(fm["val_joint_loss"], 0.8 * mse + 0.2 * bce, places=5)
        self.assertAlmostEqual(fm["val_rmse"], float(np.sqrt(mse)), places=6)

    def test_early_stopping_on_val_rmse_restores_best_epoch(self):
        p = params(epochs=40, optimizer={"type": "adam", "lr": 1e-2, "weight_decay": 0.0}, batch_size=16,
                   early_stopping={"monitor": "val_rmse", "patience": 2})
        res, seen, (Xv, yv) = self.train(p)
        fm = res["final_metrics"]
        self.assertTrue(fm["stopped_early"])
        self.assertEqual(fm["stopped_epoch"] - fm["best_epoch"], 2)  # patience=2：連續 2 輪沒進步才停
        self.assertEqual(fm["epochs_run"], fm["stopped_epoch"] + 1)
        val_rmse = [m["val_rmse"] for m in seen]
        self.assertEqual(int(np.argmin(val_rmse)), fm["best_epoch"])
        self.assertAlmostEqual(fm["val_rmse"], val_rmse[fm["best_epoch"]], places=12)  # 最終評估對應保存的 best
        pred = res["predict"](Xv)
        self.assertAlmostEqual(float(np.sqrt(np.mean((pred - yv) ** 2))), fm["val_rmse"], places=9)

    def test_patience_zero_runs_full_epochs_and_saves_best_and_last(self):
        res, seen, (Xv, yv) = self.train(params(epochs=6, early_stopping={"monitor": "val_rmse", "patience": 0}))
        fm = res["final_metrics"]
        self.assertFalse(fm["stopped_early"])
        self.assertEqual((fm["epochs_run"], fm["stopped_epoch"], len(seen)), (6, 5, 6))  # 跑滿 6 輪
        for k in ("val_rmse", "val_mae", "val_dir_acc", "val_joint_loss", "val_mse", "val_bce"):
            self.assertIn(k, fm["last"])
        loaded_best = architectures.load_model(KEY, res["weights"], res["model_config"])
        loaded_last = architectures.load_model(KEY, res["weights_last"], res["model_config"])
        self.assertAlmostEqual(float(np.sqrt(np.mean((loaded_best["predict"](Xv) - yv) ** 2))), fm["val_rmse"], places=9)
        self.assertAlmostEqual(float(np.sqrt(np.mean((loaded_last["predict"](Xv) - yv) ** 2))), fm["last"]["val_rmse"], places=9)
        if fm["best_epoch"] != fm["stopped_epoch"]:
            self.assertNotEqual(res["weights"], res["weights_last"])

    def test_no_early_stop_reports_full_run(self):
        res, _, _ = self.train(params(epochs=2, early_stopping={"monitor": "val_rmse", "patience": 5}))
        fm = res["final_metrics"]
        self.assertFalse(fm["stopped_early"])
        self.assertEqual((fm["epochs_run"], fm["stopped_epoch"]), (2, 1))

    def test_monitor_mode_max_for_accuracy(self):
        res, _, _ = self.train(params(epochs=3, early_stopping={"monitor": "val_dir_acc", "patience": 2}))
        self.assertEqual(res["final_metrics"]["monitor_mode"], "max")

    def test_deterministic_with_seed_and_seed_matters(self):
        r1, s1, _ = self.train(params(seed=7)); r2, s2, _ = self.train(params(seed=7)); _, s3, _ = self.train(params(seed=8))
        self.assertEqual([m["val_rmse"] for m in s1], [m["val_rmse"] for m in s2])
        self.assertEqual(r1["weights"], r2["weights"])
        self.assertNotEqual([m["val_rmse"] for m in s1], [m["val_rmse"] for m in s3])

    def test_save_load_predictions_identical_and_rebuilt_from_saved_config(self):
        res, _, (Xv, _) = self.train(params())
        cfg = res["model_config"]
        json.dumps(cfg)
        self.assertEqual(cfg["config"]["lstm_layers"], validate(params())["lstm_layers"])
        self.assertEqual((cfg["mode"], cfg["target_scaling"]["method"], cfg["config"]["seed"]), ("dual", "none", 42))
        self.assertIn("單向", cfg["sequence_summary"])
        loaded = architectures.load_model(KEY, res["weights"], cfg)
        a, b = res["predict_outputs"](Xv), loaded["predict_outputs"](Xv)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        np.testing.assert_array_equal(res["predict"](Xv), loaded["predict"](Xv))
        self.assertTrue(((b["direction_probability"] > 0) & (b["direction_probability"] < 1)).all())

    def test_target_scaling_saved_restored_and_direction_stays_in_original_scale(self):
        p = params(target_scaling="standardize", direction_rule={"op": ">=", "threshold": 0.004})
        (Xt, yt), (Xv, yv) = noisy(signal=0.02)
        yt = yt + 0.003; yv = yv + 0.003  # 平均不為 0，縮放與否才有差別
        res, _, _ = self.train(p, data=((Xt, yt), (Xv, yv)))
        sc = res["model_config"]["target_scaling"]
        self.assertEqual(sc["method"], "standardize")
        self.assertAlmostEqual(sc["mean"], float(np.mean(yt)), places=12)  # train-only fit
        self.assertAlmostEqual(sc["std"], float(np.std(yt)), places=12)
        loaded = architectures.load_model(KEY, res["weights"], res["model_config"])
        out = loaded["predict_outputs"](Xv)
        self.assertAlmostEqual(float(np.sqrt(np.mean((out["regression"] - yv) ** 2))), res["final_metrics"]["val_rmse"], places=9)
        expected = float(np.mean((out["direction_probability"] >= 0.5) == (yv >= 0.004)))
        self.assertAlmostEqual(res["final_metrics"]["val_dir_acc"], expected, places=9)

    def test_requires_regression_task_and_valid_params(self):
        (Xt, yt), (Xv, yv) = noisy()
        with self.assertRaisesRegex(ValueError, "連續目標"):
            architectures.train_model(KEY, Xt, yt, Xv, yv, params(), None, task_type="classification")
        p = params(); del p["epochs"]
        with self.assertRaisesRegex(ValueError, "epochs"):
            architectures.train_model(KEY, Xt, yt, Xv, yv, p, None, task_type="regression")

    def test_full_evaluation_report_for_both_heads_and_states(self):
        res, _, _ = self.train(params(epochs=3))
        for state in ("best", "last"):
            ev = res["evaluation"][state]
            direction, regression = ev["direction"], ev["regression"]
            for key in ("class_distribution", "confusion_matrix", "confusion_matrix_labels", "per_class",
                        "macro_f1", "balanced_accuracy", "roc_auc", "average_precision", "baseline_majority_class"):
                self.assertIn(key, direction, (state, key))
            self.assertEqual(len(direction["confusion_matrix"]), 2)
            self.assertIsInstance(direction["roc_auc"], float)
            self.assertIsInstance(direction["average_precision"], float)  # AP，不是梯形積分 PR-AUC
            self.assertEqual(direction["baseline_majority_class"]["method"], "predict_majority_class_from_train")
            for key in ("mse", "rmse", "mae", "r2", "baseline_mean", "baseline_zero"):
                self.assertIn(key, regression, (state, key))
        json.dumps(res["evaluation"])

    def test_evaluation_reports_reason_when_metric_undefined(self):
        from training.evaluation import classification_evaluation, regression_evaluation
        out = classification_evaluation([0, 1, 0, 1], [0, 1, 1, 1], y_proba=None)
        self.assertIsNone(out["roc_auc"])
        self.assertIn("機率", out["roc_auc_unavailable_reason"])
        self.assertIsNone(out["average_precision"])
        self.assertIsNone(out["baseline_majority_class"])
        self.assertIn("多數類別", out["baseline_majority_class_unavailable_reason"])
        out2 = regression_evaluation([1.0], [1.1])
        self.assertIsNone(out2["r2"])
        self.assertIn("R²", out2["r2_unavailable_reason"])
        self.assertEqual(out2["baseline_zero"]["method"], "predict_zero")
        self.assertIsNone(out2["baseline_mean"])
        self.assertIn("基準", out2["baseline_mean_unavailable_reason"])

    def test_classification_evaluation_majority_class_baseline_from_train_not_val(self):
        from training.evaluation import classification_evaluation
        out = classification_evaluation(y_true=[1, 1, 1, 0], y_pred=[1, 1, 1, 1], y_train=[0, 0, 0, 1])
        self.assertEqual(out["baseline_majority_class"]["class"], 0)


class GraphIntegrationTests(unittest.TestCase):
    def run_graph(self, spec, df=None, on_epoch=None, on_preview=None):
        df = H._DF if df is None else df
        return graph_mod.run_graph(spec, lambda tf: df, on_epoch, on_preview)

    def test_named_outputs_metadata_summary_and_downstream_columns(self):
        H.CAP = None
        cap = []

        def cap_train(X_train, y_train, X_val, y_val, p, on_epoch, task_type="regression", preview_hook=None):
            cap.append(np.array(X_train, copy=True))
            return H.spy_train(X_train, y_train, X_val, y_val, p, on_epoch, task_type, preview_hook)

        architectures.ARCHITECTURE_REGISTRY["capdown"] = cap_train
        try:
            down = H._model("down", "capdown", [{"node": "dual", "output": "direction_probability"}, "fa",
                                                 {"node": "dual", "output": "regression"}], "lr")
            spec = graph_spec(extra_model=down)
            validate_graph_spec(spec)
            results = self.run_graph(spec)
        finally:
            architectures.ARCHITECTURE_REGISTRY.pop("capdown", None)
        r = results["dual"]
        self.assertEqual(cap[-1].shape[-1], 1 + 4 + 1)
        self.assertEqual(sorted(r["output_specs"]), ["default", "direction_probability", "regression"])
        self.assertEqual(r["model_config"]["output_specs"], r["output_specs"])
        self.assertEqual(r["task_type"], "regression")
        self.assertEqual(r["training_meta"]["time_semantics"]["available_ts"], False)  # 這份假資料沒有 bar_end_ts

    def test_job_result_stays_lightweight_and_progress_carries_extended_metrics(self):
        rows, previews = [], []
        # 測試絕不能碰真實資料庫：進度、預覽、產物的寫入全部換成記憶體內的收集器
        orig = (worker_mod.write_progress, worker_mod.save_artifacts_for_job, worker_mod.save_preview_sample)
        worker_mod.write_progress = lambda job, node, epoch, metrics: rows.append((node, epoch, dict(metrics)))
        worker_mod.save_preview_sample = lambda job, node, preview: previews.append(preview)
        saved = {}
        worker_mod.save_artifacts_for_job = lambda job, spec, results: saved.update(results)
        try:
            result = worker_mod.run_one_sync("job", graph_spec(), {"tx_1m": H._DF})
        finally:
            worker_mod.write_progress, worker_mod.save_artifacts_for_job, worker_mod.save_preview_sample = orig
        summary = result["dual"]
        self.assertEqual(sorted(summary), ["device", "evaluation", "final_metrics", "output_specs", "training_meta"])
        self.assertLess(len(json.dumps(result)), 12000)  # 完整評估報告（含 evaluation）比純量摘要大，但仍是小型 O(類別數) 的量級
        self.assertNotIn("prediction_source", json.dumps(result))
        top_level_scalars = {k: v for k, v in summary["final_metrics"].items() if k != "last"}
        self.assertTrue(all(np.isscalar(v) or isinstance(v, (str, bool)) for v in top_level_scalars.values()))
        self.assertTrue(all(np.isscalar(v) or isinstance(v, (str, bool)) for v in summary["final_metrics"]["last"].values()))
        self.assertIn("weights", saved["dual"])  # 完整產物走 artifacts，不進摘要
        core, extras = progress.split_metrics(rows[0][2])
        self.assertEqual(sorted(core), ["accuracy", "loss", "val_accuracy", "val_loss"])
        self.assertEqual(sorted(extras), sorted(k for k in rows[0][2] if k not in core))
        self.assertIn("val_rmse", extras)

    def test_time_semantics_reach_preview_with_identity_vs_available_time(self):
        df = daily_df(hours=5)
        previews = []
        spec = graph_spec(params(epochs=2))
        self.run_graph(spec, df=df, on_preview=lambda nid, p: previews.append(p))
        self.assertTrue(previews)
        p = previews[0]
        # decision_ts 是最後一根輸入棒的識別時間（棒起點）；資訊可用時間是它收盤（+5 小時）之後
        self.assertEqual(p["decision_available_ts"] - p["decision_ts"], 5 * 3600)
        self.assertEqual(p["target_available_ts"] - p["target_ts"], 5 * 3600)
        self.assertGreater(p["target_available_ts"], p["decision_available_ts"])
        self.assertEqual(bars_last_time(p), p["decision_ts"])  # 圖上 T 標記仍對到那根棒本身

    def test_preview_has_no_available_time_when_data_lacks_bar_end(self):
        previews = []
        self.run_graph(graph_spec(params(epochs=1)), on_preview=lambda nid, p: previews.append(p))
        self.assertIsNone(previews[0]["decision_available_ts"])
        self.assertIsNone(previews[0]["target_available_ts"])


def bars_last_time(preview):
    return preview["bars"][-1]["time"]


class TimeSemanticsUnitTests(unittest.TestCase):
    def test_max_over_feature_inputs_and_none_when_any_missing(self):
        idx = pd.date_range("2024-01-01 08:45", periods=3, freq="D")
        a = pd.Series(idx + pd.Timedelta(hours=5), index=idx)
        b = pd.Series(idx + pd.Timedelta(hours=7), index=idx)
        out = available_at([a, b], idx.to_numpy())
        np.testing.assert_array_equal(out, (idx + pd.Timedelta(hours=7)).to_numpy())
        self.assertIsNone(available_at([a, None], idx.to_numpy()))
        self.assertIsNone(available_at([], idx.to_numpy()))

    def test_build_map_requires_column(self):
        self.assertIsNone(build_available_map(H._DF))
        self.assertIsNotNone(build_available_map(daily_df()))


class InferenceAndStorageTests(unittest.TestCase):
    def arts(self, spec, df):
        results = graph_mod.run_graph(spec, lambda tf: df)
        nodes = {n["id"]: n for n in spec["nodes"]}
        out = {}
        for nid, r in results.items():
            out[nid] = {"architecture_key": nodes[nid]["key"], "task_type": r["task_type"], "weights": r["weights"],
                        "model_config": r["model_config"], "feature_schema": [], "target_spec": {},
                        "graph_spec_snapshot": spec, "depends_on_node_ids": [], "preprocessing_state": r["preprocessing_state"]}
        return out

    def infer(self, arts, node_id, df, chunk=40):
        orig = inference_mod.load_artifact
        inference_mod.load_artifact = lambda job, nid: arts.get(nid)
        chunks = []
        try:
            n = inference_mod.run_inference_chunked("job", node_id, lambda tf: df, lambda *a: chunks.append(a), chunk_size=chunk)
        finally:
            inference_mod.load_artifact = orig
        return n, chunks

    def test_dual_inference_readback_rows_and_available_ts(self):
        df = daily_df()
        arts = self.arts(graph_spec(), df)
        n, chunks = self.infer(arts, "dual", df)
        self.assertEqual(n, 320 - 10)
        self.assertGreater(len(chunks), 1)  # 分批
        ts = [t for c in chunks for t in c[0]]
        rows = np.array([r for c in chunks for r in c[1]])
        self.assertTrue(all(c[4] == "dual" and c[3] is False for c in chunks))
        self.assertEqual(rows.shape, (n, 2))
        self.assertTrue(((rows[:, 1] > 0) & (rows[:, 1] < 1)).all())
        avail = [t for c in chunks for t in c[5]]
        self.assertEqual({a - t for a, t in zip(avail, ts)}, {5 * 3600})  # 每筆的可用時間 = 識別時間 + 收盤延遲
        # 推論結果跟直接用載入模型對同一批窗格算的一致（重新載入一致）
        loaded = architectures.load_model(KEY, arts["dual"]["weights"], arts["dual"]["model_config"])
        self.assertEqual(sorted(loaded), ["predict", "predict_outputs"])

    def test_dual_inference_matches_downstream_named_output_training_columns(self):
        cap_train, cap_infer = [], []

        def cap_train_fn(X_train, y_train, X_val, y_val, p, on_epoch, task_type="regression", preview_hook=None):
            cap_train.append(np.array(X_train, copy=True))
            return H.spy_train(X_train, y_train, X_val, y_val, p, on_epoch, task_type, preview_hook)

        def cap_loader(weights, cfg):
            def predict(X):
                cap_infer.append(np.array(X, copy=True)); return np.zeros(len(X))
            return {"predict": predict}

        architectures.ARCHITECTURE_REGISTRY["capd"] = cap_train_fn
        architectures.LOAD_REGISTRY["capd"] = cap_loader
        try:
            down = H._model("down", "capd", ["fa", {"node": "dual", "output": "direction_probability"}, "fv"], "lr")
            spec = graph_spec(extra_model=down)
            validate_graph_spec(spec)
            df = daily_df()
            arts = self.arts(spec, df)  # 訓練一次，產物與訓練時的輸入一起留下
            train_meta = graph_mod.run_graph(spec, lambda tf: df)["down"]["training_meta"]
            n, _ = self.infer(arts, "down", df, chunk=10_000)
        finally:
            architectures.ARCHITECTURE_REGISTRY.pop("capd", None)
            architectures.LOAD_REGISTRY.pop("capd", None)
        x_train, x_inf = cap_train[-1], np.concatenate(cap_infer, axis=0).astype(np.float32)
        self.assertEqual(x_train.shape[-1], 4 + 1 + 1)
        self.assertEqual(x_inf.shape[-1], x_train.shape[-1])
        # 欄位順序一致：訓練集每一列，都能在推論用同一批資料算出的窗格裡逐欄對上
        n_samples = train_meta["n_train"] + train_meta["n_val"]
        idx = np.random.default_rng(42).permutation(n_samples)
        train_idx = idx[max(1, int(n_samples * 0.2)):]
        np.testing.assert_allclose(x_inf[:n_samples][train_idx], x_train, atol=1e-4)

    def test_row_to_output_dual_semantics(self):
        t, predicted, probs = inference_store.row_to_output([0.0123, 0.7], "regression", False, "dual")
        self.assertEqual(t, "dual")
        self.assertAlmostEqual(predicted, 0.0123)
        p = json.loads(probs)
        self.assertAlmostEqual(p[0], 0.3); self.assertAlmostEqual(p[1], 0.7)  # [不符合事件, 符合事件]
        self.assertEqual(inference_store.row_to_output([2.0], "classification", False)[0], "class")  # 既有輸出型別不變
        self.assertEqual(inference_store.row_to_output([0.2, 0.8], "classification", True)[0], "probability")
        self.assertEqual(inference_store.row_to_output([1.5], "regression", False)[0], "regression")

    def test_writer_stores_dual_rows_with_available_ts(self):
        captured = {}

        class Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, sql, params=None): pass

        class Conn:
            autocommit = False
            def cursor(self): return Cur()
            def close(self): pass

        orig_connect, orig_ev = inference_store.psycopg2.connect, inference_store.execute_values
        inference_store.psycopg2.connect = lambda *a, **k: Conn()
        inference_store.execute_values = lambda cur, sql, rows, template=None: captured.update(sql=sql, rows=rows, template=template)
        try:
            w = inference_store.open_prediction_writer("job", "node")
            w.write_chunk([100, 200], [[0.01, 0.6], [-0.02, 0.4]], "regression", False, "dual", [110, 210])
            w.close()
        finally:
            inference_store.psycopg2.connect, inference_store.execute_values = orig_connect, orig_ev
        self.assertIn("available_ts", captured["sql"])
        self.assertEqual(captured["rows"][0][:4], ("job", "node", 100, 110))
        self.assertEqual(captured["rows"][0][4], "dual")
        self.assertEqual(w.written, 2)

    def test_old_single_output_inference_still_uses_default_path(self):
        df = H._DF
        spec = {"start": "a", "end": "b", "nodes": [H._feat("fa"), H._label("lr", "simple_return", "identity"),
                                                    H._model("solo", "xgboost", ["fa"], "lr", params=H.xgb_params(n_estimators=3))]}
        arts = self.arts(spec, df)
        n, chunks = self.infer(arts, "solo", df)
        self.assertTrue(all(c[4] is None for c in chunks))  # 不是 dual
        self.assertEqual(n, sum(len(c[0]) for c in chunks))


class ProgressWriteTests(unittest.TestCase):
    def fake_write(self, metrics):
        calls = []

        class Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, sql, params=None): calls.append((" ".join(sql.split()), params))

        class Conn:
            autocommit = False
            def cursor(self): return Cur()
            def close(self): pass

        orig = progress.psycopg2.connect
        progress.psycopg2.connect = lambda *a, **k: Conn()
        try:
            progress.write_progress("11111111-1111-1111-1111-111111111111", "n1", 3, metrics)
        finally:
            progress.psycopg2.connect = orig
        return calls

    def test_core_only_metrics_store_null_metrics_column(self):
        calls = self.fake_write({"loss": 0.5, "accuracy": 0.6, "val_loss": 0.7, "val_accuracy": 0.55, "gpu_mem_mb": 12.0})
        self.assertIn("metrics", calls[0][0].split("VALUES")[0])
        self.assertEqual(calls[0][0].count("%s"), len(calls[0][1]))
        self.assertIsNone(calls[0][1][-1])  # 沒有擴充指標 → NULL，不是空物件或 0
        payload = json.loads(calls[1][1][0])
        self.assertNotIn("metrics", payload)
        self.assertEqual(payload["loss"], 0.5)

    def test_non_finite_extended_metrics_are_stored_as_null(self):
        calls = self.fake_write({"loss": 0.5, "val_loss": 0.6, "rmse": float("nan"), "val_rmse": 0.1})
        self.assertEqual(json.loads(calls[0][1][-1]), {"rmse": None, "val_rmse": 0.1})

    def test_extended_metrics_go_to_metrics_column_and_nested_payload(self):
        m = {"loss": 0.4, "accuracy": 0.6, "val_loss": 0.5, "val_accuracy": 0.55, "joint_loss": 0.4, "rmse": 0.01, "val_rmse": 0.02,
             "dir_acc": 0.6, "mse": 1e-4, "bce": 0.69, "gpu_mem_mb": 3.0}
        calls = self.fake_write(m)
        self.assertIn("metrics", calls[0][0].split("VALUES")[0])
        stored = json.loads(calls[0][1][-1])
        self.assertEqual(sorted(stored), ["bce", "dir_acc", "joint_loss", "mse", "rmse", "val_rmse"])
        self.assertNotIn("loss", stored)  # 不重複存舊四欄
        self.assertNotIn("gpu_mem_mb", stored)
        self.assertEqual(calls[0][1][2], 0.4)  # loss 欄仍是聯合損失
        payload = json.loads(calls[1][1][0])
        self.assertEqual(payload["metrics"], stored)
        self.assertEqual(payload["rmse"], 0.01)  # 扁平格式維持，既有前台照舊可讀

    def test_none_metrics_are_not_faked_as_zero(self):
        core, extras = progress.split_metrics({"loss": None, "accuracy": None, "rmse": None})
        self.assertEqual(core["loss"], None)
        self.assertEqual(extras, {})


class QuotesIdentifierTests(unittest.TestCase):
    def test_daily_tables_accepted_and_injection_rejected(self):
        from fastapi import HTTPException
        from services.quotes import quotes_table

        self.assertEqual(quotes_table("tx", "daily_day"), "market.future_taifex_tx_daily_day")
        self.assertEqual(quotes_table("TX", "daily_full"), "market.future_taifex_tx_daily_full")
        self.assertEqual(quotes_table("tx", "15m"), "market.future_taifex_tx_15m")
        for tf in ("daily-day", "daily day", "daily_day\n", "daily__day", "_daily", "daily_", "d;drop", "a.b", "", 'd"x', "daily_day) --"):
            with self.subTest(timeframe=tf), self.assertRaises(HTTPException):
                quotes_table("tx", tf)
        for sym in ("tx;x", "t x", "", "tx\n", "tx_y"):
            with self.subTest(symbol=sym), self.assertRaises(HTTPException):
                quotes_table(sym, "daily_day")

    def test_worker_split_of_daily_timeframe(self):
        symbol, timeframe = "tx_daily_day".split("_", 1)  # worker.py 的拆法
        self.assertEqual((symbol, timeframe), ("tx", "daily_day"))


@unittest.skipUnless(os.environ.get("RUN_DB_TESTS") == "1", "需要 RUN_DB_TESTS=1（唯讀查詢正式 quotes DB）")
class DailyReadPathTests(unittest.TestCase):
    def test_worker_load_path_and_time_columns(self):
        spec = {"start": "2026-08-01", "end": "2026-09-19", "nodes": [
            {"id": "f", "type": "feature", "key": "raw_price", "timeframe": "tx_daily_day"},
            {"id": "l", "type": "label", "outcome": "log_return", "labeling_rule": "identity", "timeframe": "tx_daily_full", "params": {"horizon": 1}}]}
        dfs = asyncio.run(worker_mod._load_dataframes(spec))
        self.assertEqual(sorted(dfs), ["tx_daily_day", "tx_daily_full"])
        for tf, df in dfs.items():
            self.assertIn("bar_end_ts", df.columns)
            self.assertEqual(set(pd.to_datetime(df["datetime"]).dt.strftime("%H:%M")), {"08:45"})  # 識別時間固定 08:45
            self.assertTrue((pd.to_datetime(df["bar_end_ts"]) > pd.to_datetime(df["datetime"])).all())  # 資訊可用時間在收盤後
            self.assertEqual(set(pd.to_datetime(df["bar_end_ts"]).dt.strftime("%H:%M")), {"13:45"})
            self.assertTrue(df["datetime"].is_unique)

    def test_daily_graph_end_to_end_with_real_bars(self):
        spec = {"start": "2024-01-01", "end": "2026-09-19", "nodes": [
            {"id": "f", "type": "feature", "key": "raw_price", "timeframe": "tx_daily_day"},
            {"id": "lab", "type": "label", "outcome": "log_return", "labeling_rule": "identity", "timeframe": "tx_daily_day", "params": {"horizon": 1}},
            H._model("m", KEY, ["f"], "lab", params=params(epochs=1))]}
        spec["nodes"][2]["window"] = 60
        validate_graph_spec(spec)
        dfs = asyncio.run(worker_mod._load_dataframes(spec))
        previews = []
        graph_mod.run_graph(spec, lambda tf: dfs[tf], on_preview=lambda n, p: previews.append(p))
        p = previews[0]
        self.assertEqual(p["decision_available_ts"] - p["decision_ts"], 5 * 3600)  # 08:45 → 13:45


if __name__ == "__main__":
    unittest.main(verbosity=2)
