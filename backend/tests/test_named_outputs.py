"""具名輸出引用（平台契約）的驗收測試。用法（在 backend/ 目錄）：
    python -m unittest tests.test_named_outputs -v

不依賴 pytest。用測試專用註冊的確定性小模型（`test_dual`：一個 regression 輸出加一個
direction_probability 輸出）驗證平台契約，不必等真正的雙頭模型；測試結束會清除註冊。
真實的 lstm（單頭）／xgboost 只用在「宣告的輸出描述要跟實際行為一致」的契約測試。

對應驗收項：
  1 舊 graph／3 輸出模式：DefaultOutputContractTests、LegacyCompatTests（另有 contract_harness.py 的前後比對）
  4 Phase 2：Phase2Tests
  5 欄位順序：InferenceOrderTests
  6 具名引用：ValidationTests（6a 合法／6b 非法）、EntryPointTests（API 與 worker 都要擋、且不建 job）
  7 API 與摘要：ListAvailableTests、ResultSummaryTests
"""

import asyncio
import copy
import json
import sys
import unittest

from tests import contract_harness as H  # 會設定 CPU 與 sys.path

import numpy as np

from training import graph as graph_mod
from training import inference as inference_mod
from training import worker as worker_mod
from training.graph_refs import (
    GraphValidationError, InputRef, input_refs, normalize_input_ref, upstream_model_ids, validate_graph_spec,
)
from training.output_specs import (
    build_event_rule, default_output_specs, event_probability_spec, label_metadata,
)
from training.phases import build_phase2_graph_spec, final_model_node_id
from training.registry import architectures, labeling_rules, outcomes

_sigmoid = H._sigmoid
CAP: dict = {"train": [], "infer": []}


# ── 測試專用 architecture ─────────────────────────────────────────────────────────────
def _dual_describe(node, label_node, task_type):
    if task_type != "regression":
        raise ValueError("test_dual 需要連續（regression）目標")
    target = label_metadata(label_node, task_type)
    if not target["signed"]:
        raise ValueError("方向事件需要帶正負號的 outcome")
    rule = (node.get("params") or {}).get("event", {"op": ">=", "threshold": 0.0})
    reg = {"kind": "regression", "columns": 1, "target": target, "description": "test_dual 回歸輸出"}
    return {"default": reg, "regression": reg, "direction_probability": event_probability_spec(target, rule)}


def _dual_outputs(X):
    v = np.asarray(X)[:, -1, 0]
    return {"regression": v * 0.5, "direction_probability": _sigmoid(v)}


def _dual_train(X_train, y_train, X_val, y_val, params, on_epoch, task_type="regression", preview_hook=None):
    return {"final_metrics": {}, "predict": lambda X: np.asarray(X)[:, -1, 0] * 0.5,
            "predict_outputs": _dual_outputs, "device": "cpu", "weights": b"dual", "model_config": {"dual": True}}


def _cap_train(X_train, y_train, X_val, y_val, params, on_epoch, task_type="classification", preview_hook=None):
    CAP["train"].append(np.array(X_train, copy=True))
    return H.spy_train(X_train, y_train, X_val, y_val, params, on_epoch, task_type, preview_hook)


def _cap_loader(weights, model_config):
    def predict(X):
        CAP["infer"].append(np.array(X, copy=True))
        return np.zeros(len(X))

    return {"predict": predict, "predict_proba": _spy_loader(weights, model_config)["predict_proba"]}


def _spy_loader(weights, model_config):
    def predict_proba(X):
        p = _sigmoid(np.asarray(X)[:, -1, 0])
        return np.stack([1 - p, p], axis=1)

    return {"predict": lambda X: np.asarray(X)[:, -1, 0] * 0.5, "predict_proba": predict_proba}


def setUpModule():
    architectures.ARCHITECTURE_REGISTRY["spy"] = H.spy_train
    architectures.LOAD_REGISTRY["spy"] = _spy_loader
    # 測試替身不走 register()（沒有 params_schema），直接放進登記表：validate 對它們原樣放行
    architectures.ARCHITECTURE_REGISTRY["test_dual"] = _dual_train
    architectures.ARCHITECTURE_OUTPUTS["test_dual"] = ("default", "regression", "direction_probability")
    architectures.ARCHITECTURE_DESCRIBERS["test_dual"] = _dual_describe
    architectures.LOAD_REGISTRY["test_dual"] = lambda w, c: {
        "predict": lambda X: np.asarray(X)[:, -1, 0] * 0.5, "predict_outputs": _dual_outputs}
    architectures.ARCHITECTURE_REGISTRY["spycap"] = _cap_train
    architectures.LOAD_REGISTRY["spycap"] = _cap_loader


def tearDownModule():
    for key in ("spy", "test_dual", "spycap"):
        architectures.ARCHITECTURE_REGISTRY.pop(key, None)
        architectures.LOAD_REGISTRY.pop(key, None)
        architectures.ARCHITECTURE_OUTPUTS.pop(key, None)
        architectures.ARCHITECTURE_DESCRIBERS.pop(key, None)


# ── 建圖小工具 ───────────────────────────────────────────────────────────────────────
def base_nodes():
    return [
        H._feat("fa", "raw_price"), H._feat("fv", "raw_volume"),
        H._label("lr", "log_return", "identity"),
        H._label("lc", "log_return", "fixed_threshold", n_classes=2),
    ]


def graph(*model_nodes, extra_nodes=()):
    return {"start": "2024-01-01", "end": "2024-02-01", "nodes": base_nodes() + list(extra_nodes) + list(model_nodes)}


def dual(nid="up", **kw):
    return H._model(nid, "test_dual", ["fa"], "lr", **kw)


def down(inputs, nid="down", key="spy", label="lr", **kw):
    return H._model(nid, key, inputs, label, **kw)


def ref(node, output):
    return {"node": node, "output": output}


def issues_of(spec) -> list[str]:
    try:
        validate_graph_spec(spec)
    except GraphValidationError as e:
        return e.issues
    return []


class NormalizeRefTests(unittest.TestCase):
    def test_string_is_default_and_not_explicit(self):
        self.assertEqual(normalize_input_ref("m1"), InputRef("m1", "default", False))

    def test_object_form(self):
        self.assertEqual(normalize_input_ref(ref("m1", "regression")), InputRef("m1", "regression", True))

    def test_bad_forms_rejected(self):
        for bad in ["", {}, {"node": "a"}, {"output": "x"}, {"node": "a", "output": "x", "extra": 1},
                    {"node": 1, "output": "x"}, {"node": "a", "output": ""}, 5, None, ["a"]]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                normalize_input_ref(bad)

    def test_order_preserved_and_upstream_dedup(self):
        node = {"inputs": ["f", ref("m", "b"), "m", ref("m", "a")]}
        self.assertEqual([r.node for r in input_refs(node)], ["f", "m", "m", "m"])
        self.assertEqual(upstream_model_ids(node, {"m"}), ["m"])


class ValidationTests(unittest.TestCase):
    # 6a：合法的圖必須通過
    def test_6a_same_upstream_two_outputs_into_one_downstream(self):
        self.assertEqual(issues_of(graph(dual(), down([ref("up", "regression"), ref("up", "direction_probability"), "fa"]))), [])

    def test_6a_default_string_plus_named_output(self):
        self.assertEqual(issues_of(graph(dual(), down(["up", ref("up", "direction_probability")]))), [])

    def test_6a_different_downstreams_use_different_outputs(self):
        spec = graph(dual(), down([ref("up", "regression")], nid="d1"), down([ref("up", "direction_probability")], nid="d2"))
        self.assertEqual(issues_of(spec), [])

    def test_6a_legacy_all_string_graphs(self):
        self.assertEqual(issues_of(graph(down(["fa", "fv"]))), [])
        self.assertEqual(issues_of(graph(H._model("up", "spy", ["fa"], "lc"), down(["fa", "up", "fv"], label="lc"))), [])

    def test_6a_explicit_default_object_on_model(self):
        self.assertEqual(issues_of(graph(dual(), down([ref("up", "default")]))), [])

    # 6b：非法引用與成環必須拒絕
    def assertRejected(self, spec, needle):
        got = issues_of(spec)
        self.assertTrue(got, f"應該被拒絕但通過了（期待包含 {needle!r}）")
        self.assertTrue(any(needle in i for i in got), f"issues={got}，期待包含 {needle!r}")

    def test_6b_unknown_node(self):
        self.assertRejected(graph(down(["nope"])), "不存在的節點")

    def test_6b_unknown_output_name(self):
        self.assertRejected(graph(dual(), down([ref("up", "p_up")])), "沒有輸出")

    def test_6b_named_output_on_single_output_model(self):
        spec = graph(H._model("xg", "xgboost", ["fa"], "lr", params=H.xgb_params(n_estimators=2)), down([ref("xg", "direction_probability")]))
        self.assertRejected(spec, "沒有輸出")

    def test_6b_object_ref_to_feature(self):
        self.assertRejected(graph(down([ref("fa", "default")])), "只有 default 輸出")

    def test_6b_malformed_object_refs(self):
        self.assertRejected(graph(dual(), down([{"node": "up"}])), "缺少字串型的 output")
        self.assertRejected(graph(dual(), down([{"output": "regression"}])), "缺少字串型的 node")
        self.assertRejected(graph(dual(), down([{"node": "up", "output": "regression", "x": 1}])), "只允許 node、output")

    def test_6b_label_in_inputs(self):
        self.assertRejected(graph(down(["lr"])), "不能引用 Label")

    def test_6b_self_reference(self):
        self.assertRejected(graph(down(["fa", "down"])), "不能引用自己")

    def test_6b_cycle_two_nodes(self):
        spec = graph(H._model("a", "spy", ["fa", "b"], "lc"), H._model("b", "spy", ["fa", "a"], "lc"))
        self.assertRejected(spec, "循環依賴")

    def test_6b_cycle_three_nodes(self):
        spec = graph(H._model("a", "spy", ["fa", "c"], "lc"), H._model("b", "spy", ["a"], "lc"), H._model("c", "spy", ["b"], "lc"))
        self.assertRejected(spec, "循環依賴")

    def test_6b_cycle_through_named_refs(self):
        spec = graph(dual("a", ), dual("b"))
        spec["nodes"][-2]["inputs"] = ["fa", ref("b", "regression")]
        spec["nodes"][-1]["inputs"] = ["fa", ref("a", "direction_probability")]
        self.assertRejected(spec, "循環依賴")

    def test_6b_output_not_produced_by_this_config(self):
        # test_dual 的事件規則單位不符 → 描述失敗，不能默默放行
        bad = dual(params={"event": {"op": ">=", "threshold": 0.0, "threshold_unit": "percent"}})
        self.assertRejected(graph(bad, down([ref("up", "direction_probability")])), "模型設定不合法")

    def test_6b_dual_on_classification_label(self):
        self.assertRejected(graph(H._model("up", "test_dual", ["fa"], "lc")), "需要連續")

    def test_6b_dual_on_unsigned_outcome(self):
        nodes = [H._feat("fa", "raw_price"), H._label("lp", "future_price", "identity"), H._model("up", "test_dual", ["fa"], "lp")]
        self.assertRejected({"start": "a", "end": "b", "nodes": nodes}, "帶正負號")

    def test_6b_structural_problems(self):
        self.assertRejected(graph(H._model("m", "no_such_arch", ["fa"], "lr")), "未登記的 architecture")
        self.assertRejected(graph(H._model("m", "spy", ["fa"], "ghost")), "label 必須引用")
        self.assertRejected(graph(H._model("m", "spy", [], "lr")), "非空陣列")
        self.assertRejected(graph(H._model("m", "spy", ["fa"], "lr"), H._model("m", "spy", ["fa"], "lr")), "id 重複")

    def test_6b_illegal_outcome_rule_combination_still_caught(self):
        nodes = [H._feat("fa"), H._label("lp", "future_price", "fixed_threshold"), H._model("m", "spy", ["fa"], "lp")]
        self.assertRejected({"start": "a", "end": "b", "nodes": nodes}, "不合法的組合")

    # 回歸案例（GC 審查重現）：metadata 產生前必須先驗參數型別，不能丟未處理的 TypeError／AttributeError
    def _graph_with_label_params(self, params):
        spec = graph(H._model("m", "xgboost", ["fa"], "lr", params=H.xgb_params(n_estimators=2)))
        next(n for n in spec["nodes"] if n["id"] == "lr")["params"] = params
        return spec

    def test_regression_label_params_horizon_null(self):
        self.assertRejected(self._graph_with_label_params({"horizon": None}), "params.horizon")

    def test_regression_label_params_is_list(self):
        self.assertRejected(self._graph_with_label_params([1]), "params 必須是物件")

    def test_label_params_other_bad_types(self):
        for params in ({"horizon": 0}, {"horizon": -1}, {"horizon": "1"}, {"horizon": True}, {"horizon": 1.5},
                       {"horizon": [1]}, {"n_classes": None}, {"n_classes": 1}, {"threshold_pct": "0.1"},
                       {"threshold_pct": float("nan")}, "x", 5):
            with self.subTest(params=params):
                self.assertTrue(issues_of(self._graph_with_label_params(params)), "應該被拒絕")

    def test_label_params_valid_shapes_still_accepted(self):
        for params in (None, {}, {"horizon": 1}, {"horizon": 5.0}, {"horizon": 3, "n_classes": 3, "threshold_pct": 0.005}):
            with self.subTest(params=params):
                self.assertEqual(issues_of(self._graph_with_label_params(params)), [])

    def test_model_params_and_fields_bad_types(self):
        for mutate, needle in (
            (lambda n: n.__setitem__("params", [1]), "params 必須是物件"),
            (lambda n: n.__setitem__("params", {"n_classes": None}), "params.n_classes"),
            (lambda n: n.__setitem__("window", None), "window"),
            (lambda n: n.__setitem__("window", 0), "window"),
            (lambda n: n.__setitem__("val_ratio", 1.5), "val_ratio"),
            (lambda n: n.__setitem__("val_ratio", "0.2"), "val_ratio"),
            (lambda n: n.__setitem__("output_mode", 5), "output_mode"),
        ):
            with self.subTest(needle=needle):
                spec = graph(H._model("m", "xgboost", ["fa"], "lc", params=H.xgb_params(n_estimators=2)))
                mutate(spec["nodes"][-1])
                self.assertRejected(spec, needle)

    def test_describe_outputs_unexpected_error_reported_as_validation_issue(self):
        def boom(node, label_node, task_type):
            raise TypeError("describer bug")

        architectures.ARCHITECTURE_REGISTRY["test_boom"] = H.spy_train
        architectures.ARCHITECTURE_DESCRIBERS["test_boom"] = boom
        try:
            self.assertRejected(graph(H._model("m", "test_boom", ["fa"], "lr")), "輸出描述失敗（TypeError）")
        finally:
            architectures.ARCHITECTURE_REGISTRY.pop("test_boom", None)
            architectures.ARCHITECTURE_DESCRIBERS.pop("test_boom", None)
            architectures.ARCHITECTURE_OUTPUTS.pop("test_boom", None)

    def test_all_issues_reported_together(self):
        got = issues_of(graph(down(["nope", ref("fa", "default"), "lr"])))
        self.assertGreaterEqual(len(got), 3)


class EventRuleTests(unittest.TestCase):
    def test_description_follows_configured_op(self):
        t = label_metadata(H._label("l", "log_return", "identity"), "regression")
        for op in (">", ">=", "<", "<="):
            spec = event_probability_spec(t, {"op": op, "threshold": 0.01})
            self.assertIn(op, spec["description"])
            self.assertEqual(spec["event"]["op"], op)
            self.assertEqual(spec["event"]["threshold_unit"], "log_ratio")

    def test_unit_comes_from_outcome(self):
        pct = label_metadata(H._label("l", "simple_return", "identity"), "regression")
        self.assertEqual(pct["unit"], "percent")
        self.assertEqual(label_metadata(H._label("l", "log_return", "identity"), "regression")["unit"], "log_ratio")

    def test_rejects_bad_rules(self):
        for bad in [{"op": "==", "threshold": 0}, {"op": ">=", "threshold": True}, {"op": ">=", "threshold": float("nan")},
                    {"op": ">=", "threshold": float("inf")}, {"op": ">=", "threshold": "0"}, {"op": ">="},
                    {"op": ">=", "threshold": 0, "x": 1}, "≥0"]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                build_event_rule(bad, "log_ratio")

    def test_no_unit_conversion(self):
        with self.assertRaises(ValueError):
            build_event_rule({"op": ">=", "threshold": 1.0, "threshold_unit": "percent"}, "log_ratio")
        self.assertEqual(build_event_rule({"op": ">=", "threshold": 1.0, "threshold_unit": "log_ratio"}, "log_ratio")["threshold"], 1.0)


class DefaultOutputContractTests(unittest.TestCase):
    """宣告（describe_outputs）跟實際輸出必須一致：對真實的 lstm／xgboost、每種輸出模式逐一比對。"""

    CASES = [
        ("regression", "lr", None, 2), ("class", "lc", None, 2), ("probability", "lc", "probability", 2),
    ]

    def _run(self, arch, label_id, output_mode, n_classes, label_extra=None):
        # 類別數由 Label 決定，模型 params 不帶 n_classes
        params = H.xgb_params(n_estimators=2) if arch == "xgboost" else H.lstm_params(
            epochs=1, lstm_layers=[{"units": 4, "dropout": 0.0}], shared={"dense": 4, "dropout": 0.0},
            optimizer={"type": "adam", "lr": 3e-3, "weight_decay": 0.0}, batch_size=64)
        extra = {"output_mode": output_mode} if output_mode else {}
        nodes = base_nodes()
        if label_extra:
            nodes = [n for n in nodes if n["id"] != "lc"] + [label_extra]
            label_id = label_extra["id"]
        spec = {"start": "2024-01-01", "end": "2024-02-01", "nodes": nodes + [
            H._model("up", arch, ["fa"], label_id, params=params, **extra),
            H._model("down", "spycap", ["fa", "up"], label_id)]}
        validate_graph_spec(spec)
        CAP["train"].clear()
        results = graph_mod.run_graph(spec, H.data_loader)
        declared = results["up"]["output_specs"]["default"]
        # down 的輸入 = fa(4 欄) + up 的 default；實際欄數必須等於宣告
        self.assertEqual(CAP["train"][-1].shape[-1], 4 + declared["columns"], f"{arch}/{output_mode}")
        return declared

    def test_lstm_and_xgboost_all_modes(self):
        for arch in ("xgboost", "lstm"):
            for kind, label_id, mode, k in self.CASES:
                with self.subTest(arch=arch, kind=kind):
                    declared = self._run(arch, label_id, mode, k)
                    self.assertEqual(declared["kind"], kind)
                    self.assertEqual(declared["columns"], k if kind == "probability" else 1)

    def test_three_class_probability(self):
        lab = H._label("l3", "log_return", "fixed_threshold", n_classes=3, threshold_pct=0.005)
        for arch in ("xgboost", "lstm"):
            with self.subTest(arch=arch):
                declared = self._run(arch, "l3", "probability", 3, label_extra=lab)
                self.assertEqual((declared["kind"], declared["columns"]), ("probability", 3))

    def test_default_specs_carry_target_metadata(self):
        spec = default_output_specs({"params": {}}, H._label("lr", "simple_return", "identity", horizon=1), "regression")["default"]
        self.assertEqual((spec["target"]["outcome"], spec["target"]["unit"], spec["target"]["horizon"]),
                         ("simple_return", "percent", 1))


class NamedOutputFlowTests(unittest.TestCase):
    def _train(self, inputs):
        CAP["train"].clear()
        spec = graph(dual(), down(inputs, key="spycap"))
        validate_graph_spec(spec)
        return spec, graph_mod.run_graph(spec, H.data_loader)

    def test_named_outputs_reach_downstream_in_inputs_order(self):
        _, results = self._train([ref("up", "regression"), ref("up", "direction_probability"), "fa"])
        self.assertEqual(CAP["train"][-1].shape[-1], 1 + 1 + 4)
        specs = results["up"]["output_specs"]
        self.assertEqual(sorted(specs), ["default", "direction_probability", "regression"])
        self.assertEqual(results["up"]["model_config"]["output_specs"], specs)  # 跟產物一起保存
        self.assertEqual(specs["direction_probability"]["event"],
                         {"of": "regression", "op": ">=", "threshold": 0.0, "threshold_unit": "log_ratio"})
        json.dumps(specs)  # 必須可序列化

    def test_column_order_follows_inputs(self):
        self._train([ref("up", "regression"), ref("up", "direction_probability"), "fa"])
        a = CAP["train"][-1]
        self._train([ref("up", "direction_probability"), ref("up", "regression"), "fa"])
        b = CAP["train"][-1]
        # 每欄各自 minmax，互換兩欄輸入順序 → 結果的前兩欄剛好互換，其餘不變
        np.testing.assert_array_equal(a[..., [1, 0, 2, 3, 4, 5]], b)

    def test_default_ref_of_dual_equals_regression_output(self):
        self._train(["up"])
        a = CAP["train"][-1]
        self._train([ref("up", "regression")])
        np.testing.assert_array_equal(a, CAP["train"][-1])


class InferenceOrderTests(unittest.TestCase):
    """驗收 5：訓練與推論的欄位順序完全一致（含交錯排列的 inputs 與具名輸出）。"""

    def _compare(self, spec, node_id, tol=1e-4):
        CAP["train"].clear(), CAP["infer"].clear()
        results = graph_mod.run_graph(spec, H.data_loader)
        arts = H.build_artifacts(spec)
        orig = inference_mod.load_artifact
        inference_mod.load_artifact = lambda job, nid: arts.get(nid)
        try:
            CAP["infer"].clear()
            inference_mod.run_inference_chunked("job", node_id, H.data_loader, lambda *a: None, chunk_size=10_000)
        finally:
            inference_mod.load_artifact = orig
        x_train = CAP["train"][-1]
        x_infer = np.concatenate(CAP["infer"], axis=0).astype(np.float32)
        # 訓練端的樣本數（融合圖的下游因為要跟上游輸出的時間戳取交集，比單層少幾列）；
        # 推論端沒有 Label 的尾端裁切，多出來的尾巴不參與比對，起點相同所以前面的窗格逐列對應。
        meta = results[node_id]["training_meta"]
        n_samples = meta["n_train"] + meta["n_val"]
        rng = np.random.default_rng(42)
        idx = rng.permutation(n_samples)
        train_idx = idx[max(1, int(n_samples * 0.2)):]
        return x_train, x_infer[:n_samples][train_idx], results

    def test_interleaved_inputs_match_between_training_and_inference(self):
        spec = graph(H._model("up", "spy", ["fa"], "lc", output_mode="probability"),
                     H._model("down", "spycap", ["fa", "up", "fv"], "lc"))
        x_train, x_inf, _ = self._compare(spec, "down")
        np.testing.assert_allclose(x_inf, x_train, atol=1e-4)

    def test_control_wrong_order_would_be_detected(self):
        # 對照組：拿另一種欄位順序的圖的推論結果去比，必須對不上——證明上面那個測試真的分辨得出錯位
        spec_a = graph(H._model("up", "spy", ["fa"], "lc", output_mode="probability"),
                       H._model("down", "spycap", ["fa", "up", "fv"], "lc"))
        spec_b = copy.deepcopy(spec_a)
        spec_b["nodes"][-1]["inputs"] = ["fa", "fv", "up"]
        x_train, _, _ = self._compare(spec_a, "down")
        _, x_inf_b, _ = self._compare(spec_b, "down")
        self.assertFalse(np.allclose(x_inf_b, x_train, atol=1e-4))

    def test_named_output_matches_between_training_and_inference(self):
        spec = graph(dual(), down([ref("up", "direction_probability"), "fa", ref("up", "regression")], key="spycap"))
        x_train, x_inf, _ = self._compare(spec, "down")
        np.testing.assert_allclose(x_inf, x_train, atol=1e-4)

    def test_old_artifact_without_output_specs(self):
        spec = graph(dual(), down([ref("up", "direction_probability")], key="spycap"))
        arts = H.build_artifacts(spec)
        del arts["up"]["model_config"]["output_specs"]  # 模擬舊產物
        orig = inference_mod.load_artifact
        inference_mod.load_artifact = lambda job, nid: arts.get(nid)
        try:
            with self.assertRaisesRegex(ValueError, "沒有宣告輸出"):
                inference_mod.run_inference_chunked("job", "down", H.data_loader, lambda *a: None)
            # 但只用 default 引用的舊產物照常推論
            arts["down"]["graph_spec_snapshot"] = copy.deepcopy(arts["down"]["graph_spec_snapshot"])
            for n in arts["down"]["graph_spec_snapshot"]["nodes"]:
                if n["id"] == "down":
                    n["inputs"] = ["up"]
            self.assertGreater(inference_mod.run_inference_chunked("job", "down", H.data_loader, lambda *a: None), 0)
        finally:
            inference_mod.load_artifact = orig


class LegacyCompatTests(unittest.TestCase):
    def test_legacy_graph_specs_unchanged_by_refs(self):
        for name, spec in H.DATA_GRAPHS.items():
            with self.subTest(graph=name):
                validate_graph_spec(spec)
                for n in spec["nodes"]:
                    if n["type"] == "model":
                        self.assertTrue(all(isinstance(i, str) for i in n["inputs"]))
                        self.assertTrue(all(not r.explicit for r in input_refs(n)))

    def test_default_output_still_follows_output_mode(self):
        prob = {"params": {}, "output_mode": "probability"}  # 欄數跟 Label 的 n_classes
        cls = {"params": {}}
        lab = H._label("l", "log_return", "fixed_threshold", n_classes=3)
        self.assertEqual(default_output_specs(prob, lab, "classification")["default"]["columns"], 3)
        self.assertEqual(default_output_specs(cls, lab, "classification")["default"]["kind"], "class")
        self.assertEqual(default_output_specs(prob, H._label("l", "log_return", "identity"), "regression")["default"]["kind"], "regression")


class Phase2Tests(unittest.TestCase):
    def test_inheritance_preserves_named_refs_and_final_node(self):
        parent = graph(dual(), down([ref("up", "direction_probability"), "fa"]))
        child = build_phase2_graph_spec(parent)
        self.assertEqual(final_model_node_id(child["nodes"]), "down")
        self.assertEqual(final_model_node_id(parent["nodes"]), "down")
        down_node = next(n for n in child["nodes"] if n["id"] == "down")
        self.assertEqual(down_node["inputs"][0], ref("up", "direction_probability"))
        self.assertEqual(down_node["split_strategy"], "chronological")
        self.assertNotIn("split_strategy", next(n for n in parent["nodes"] if n["id"] == "down"))  # 深拷貝，母圖不被改
        validate_graph_spec(child)


class EntryPointTests(unittest.TestCase):
    """所有建立／繼承圖的入口共用驗證；非法時不建立 job 列。"""

    def setUp(self):
        from routers import model_training as router
        from training import job_store

        self.router, self.job_store = router, job_store
        self.created: list = []
        self._orig = (job_store.create_job, job_store.get_job)

        async def fake_create(graph_spec, job_type="train", phase=None, parent_job_id=None):
            self.created.append({"spec": graph_spec, "phase": phase, "parent": parent_job_id})
            return "fake-job-id"

        self.parents: dict = {}

        async def fake_get(job_id):
            return self.parents.get(job_id)

        job_store.create_job, job_store.get_job = fake_create, fake_get

    def tearDown(self):
        self.job_store.create_job, self.job_store.get_job = self._orig

    def _train(self, **kw):
        return asyncio.run(self.router.start_training(self.router.TrainRequest(**kw)))

    def test_api_accepts_legal_named_graph(self):
        spec = graph(dual(), down([ref("up", "regression"), ref("up", "direction_probability")]))
        out = self._train(start=spec["start"], end=spec["end"], nodes=spec["nodes"])
        self.assertEqual(out, {"job_id": "fake-job-id"})
        self.assertEqual(len(self.created), 1)

    def test_api_rejects_illegal_graphs_without_creating_job(self):
        from fastapi import HTTPException

        cases = {
            "unknown output": graph(dual(), down([ref("up", "p_up")])),
            "unknown node": graph(down(["nope"])),
            "cycle": graph(H._model("a", "spy", ["fa", "b"], "lc"), H._model("b", "spy", ["fa", "a"], "lc")),
            "object ref to feature": graph(down([ref("fa", "default")])),
        }
        for name, spec in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(HTTPException) as cm:
                    self._train(start=spec["start"], end=spec["end"], nodes=spec["nodes"])
                self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(self.created, [], "驗證失敗不應該建立 job 列")

    def test_phase2_inherits_and_revalidates(self):
        from fastapi import HTTPException

        good = graph(dual(), down([ref("up", "direction_probability"), "fa"]))
        self.parents["p1"] = {"phase": 1, "status": "done", "graph_spec": good}
        out = self._train(start="", end="", nodes=[], phase=2, parent_job_id="p1")
        self.assertEqual(out, {"job_id": "fake-job-id"})
        inherited = self.created[-1]["spec"]
        self.assertEqual(next(n for n in inherited["nodes"] if n["id"] == "down")["inputs"][0], ref("up", "direction_probability"))

        # 母任務的圖如果已經不合法（例如早於現行規則），Phase 2 繼承後仍會被擋
        bad = copy.deepcopy(good)
        next(n for n in bad["nodes"] if n["id"] == "down")["inputs"][0] = ref("up", "gone")
        self.parents["p2"] = {"phase": 1, "status": "done", "graph_spec": bad}
        before = len(self.created)
        with self.assertRaises(HTTPException) as cm:
            self._train(start="", end="", nodes=[], phase=2, parent_job_id="p2")
        self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(len(self.created), before)

    def test_api_rejects_bad_label_params_without_creating_job(self):
        from fastapi import HTTPException

        for params in ({"horizon": None}, [1]):
            with self.subTest(params=params):
                spec = graph(H._model("m", "xgboost", ["fa"], "lr", params=H.xgb_params(n_estimators=2)))
                next(n for n in spec["nodes"] if n["id"] == "lr")["params"] = params
                with self.assertRaises(HTTPException) as cm:
                    self._train(start=spec["start"], end=spec["end"], nodes=spec["nodes"])
                self.assertEqual(cm.exception.status_code, 400)
                self.assertIn("params", cm.exception.detail)
        self.assertEqual(self.created, [], "驗證失敗不應該建立 job 列")

    def test_phase2_rejects_parent_with_bad_label_params(self):
        from fastapi import HTTPException

        parent = graph(H._model("m", "xgboost", ["fa"], "lr", params=H.xgb_params(n_estimators=2)))
        next(n for n in parent["nodes"] if n["id"] == "lr")["params"] = {"horizon": None}
        self.parents["pb"] = {"phase": 1, "status": "done", "graph_spec": parent}
        with self.assertRaises(HTTPException) as cm:
            self._train(start="", end="", nodes=[], phase=2, parent_job_id="pb")
        self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(self.created, [])

    def test_worker_rejects_bad_label_params_before_loading_data(self):
        async def boom(_spec):
            raise AssertionError("驗證失敗時不該去讀資料")

        orig = worker_mod._load_dataframes
        worker_mod._load_dataframes = boom
        try:
            for params in ({"horizon": None}, [1]):
                with self.subTest(params=params):
                    spec = graph(H._model("m", "xgboost", ["fa"], "lr", params=H.xgb_params(n_estimators=2)))
                    next(n for n in spec["nodes"] if n["id"] == "lr")["params"] = params
                    with self.assertRaises(GraphValidationError):
                        asyncio.run(worker_mod.run_one("job", spec))
        finally:
            worker_mod._load_dataframes = orig

    def test_worker_validates_before_loading_data(self):
        async def boom(_spec):
            raise AssertionError("驗證失敗時不該去讀資料")

        orig = worker_mod._load_dataframes
        worker_mod._load_dataframes = boom
        try:
            spec = graph(dual(), down([ref("up", "p_up")]))
            with self.assertRaises(GraphValidationError):
                asyncio.run(worker_mod.run_one("job", spec))
        finally:
            worker_mod._load_dataframes = orig


class ListAvailableTests(unittest.TestCase):
    def test_architectures_list_outputs(self):
        by_key = {a["key"]: a for a in architectures.list_available()}
        self.assertEqual(by_key["lstm"]["outputs"], ["default", "regression", "direction_probability"])
        self.assertEqual(by_key["xgboost"]["outputs"], ["default"])
        self.assertEqual(by_key["test_dual"]["outputs"], ["default", "regression", "direction_probability"])

    def test_registries_expose_declarative_metadata(self):
        oc = {o["key"]: o for o in outcomes.list_available()}
        self.assertEqual((oc["log_return"]["unit"], oc["log_return"]["signed"]), ("log_ratio", True))
        self.assertEqual(oc["simple_return"]["unit"], "percent")
        self.assertFalse(oc["future_price"]["signed"])
        lr = {r["key"]: r for r in labeling_rules.list_available()}
        self.assertEqual((lr["identity"]["task_type"], lr["fixed_threshold"]["task_type"]), ("regression", "classification"))


class ResultSummaryTests(unittest.TestCase):
    def test_job_result_summary_has_output_specs_and_no_bulk(self):
        spec = graph(dual(), down([ref("up", "regression")], key="spy"))
        wm = worker_mod
        orig = (wm.write_progress, wm.save_artifacts_for_job)
        wm.write_progress = lambda *a, **k: None
        wm.save_artifacts_for_job = lambda *a, **k: None
        try:
            result = wm.run_one_sync("job", spec, {"tx_1m": H._DF})
        finally:
            wm.write_progress, wm.save_artifacts_for_job = orig
        self.assertEqual(sorted(result["up"]), ["device", "final_metrics", "output_specs", "training_meta"])
        self.assertNotIn("prediction_source", json.dumps(result))
        self.assertLess(len(json.dumps(result)), 5000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
