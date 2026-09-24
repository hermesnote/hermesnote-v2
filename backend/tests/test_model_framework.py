"""通用模型組裝框架：schema 驗證器、登記資訊（能力／slots／schema）、元件相容性的程式檢查、
訓練控制（BestTracker），以及「新增一個 architecture 只要登記就能被驗證／列出」的擴充性。

執行：python -m unittest tests.test_model_framework
"""

import unittest

from tests import contract_harness as H  # noqa: F401  （設定 CPU 與 sys.path）

from training.registry import architectures, attention
from training.registry.param_schema import condition_met, enum_options, validate_with_schema
from training.registry.training_control import BestTracker


class SchemaValidatorTests(unittest.TestCase):
    SCHEMA = [
        {"name": "mode", "type": "enum", "required": True, "enum_values": ["a", "b"]},
        {"name": "n", "type": "int", "required": True, "min": 1},
        {"name": "rate", "type": "float", "exclusive_min": 0, "max": 1, "default": 0.5},
        {"name": "sub.x", "type": "int", "required": {"field": "mode", "equals": "b"},
         "visible_if": {"field": "mode", "equals": "b"}},
        {"name": "only_reg", "type": "bool", "visible_if": {"field": "$task_type", "equals": "regression"}},
        {"name": "unit", "type": "string", "derived_from": "$unit"},
        {"name": "items", "type": "array_of_object", "min_items": 1, "max_items": 2,
         "item_schema": [{"name": "w", "type": "int", "required": True}]},
        {"name": "pick", "type": "enum", "enum_values": ["p"], "enum_cases": [
            {"when": {"field": "mode", "equals": "b"}, "values": ["q"]}]},
    ]

    def run_schema(self, params, **context):
        return validate_with_schema(self.SCHEMA, params, context)

    def test_defaults_and_conditional_fields(self):
        cfg, problems = self.run_schema({"mode": "a", "n": 2, "items": [{"w": 1}], "pick": "p"}, task_type="classification")
        self.assertEqual(problems, [])
        self.assertEqual(cfg["rate"], 0.5)
        self.assertNotIn("sub", cfg)
        _, problems = self.run_schema({"mode": "b", "n": 2, "items": [{"w": 1}], "pick": "q"})
        self.assertTrue(any("sub.x" in p and "必填" in p for p in problems))

    def test_hidden_field_present_is_an_error_not_ignored(self):
        _, problems = self.run_schema({"mode": "a", "n": 1, "sub": {"x": 1}, "items": [{"w": 1}]})
        self.assertTrue(any("sub.x" in p and "不適用" in p for p in problems))
        _, problems = self.run_schema({"mode": "a", "n": 1, "only_reg": True, "items": [{"w": 1}]}, task_type="classification")
        self.assertTrue(any("only_reg" in p for p in problems))

    def test_types_ranges_unknown_and_arrays(self):
        _, problems = self.run_schema({"mode": "c", "n": 0, "rate": 0, "zzz": 1, "items": [{"w": "x"}, {}, {}]})
        text = "；".join(problems)
        for needle in ("mode", "params.n", "rate", "zzz", "items"):
            self.assertIn(needle, text)
        _, problems = self.run_schema({"mode": "a", "n": 1, "items": [{"w": "x"}]})
        self.assertTrue(any("items[0].w" in p for p in problems))

    def test_enum_cases_and_derived_context(self):
        _, problems = self.run_schema({"mode": "b", "n": 1, "sub": {"x": 1}, "items": [{"w": 1}], "pick": "p"})
        self.assertTrue(any("pick" in p for p in problems))  # mode=b 時只能選 q
        _, problems = self.run_schema({"mode": "a", "n": 1, "items": [{"w": 1}], "unit": "pct"}, unit="log")
        self.assertTrue(any("unit" in p for p in problems))
        cfg, problems = self.run_schema({"mode": "a", "n": 1, "items": [{"w": 1}]}, unit="log")
        self.assertEqual((problems, cfg["unit"]), ([], "log"))  # 沒給就從情境補上

    def test_conditions_all_any_and_missing_context(self):
        lookup = {"a": 1, "$t": "reg"}.get
        self.assertTrue(condition_met({"all": [{"field": "a", "equals": 1}, {"field": "$t", "equals": "reg"}]},
                                      lambda k: lookup(k, None) if k != "$missing" else object()))
        self.assertEqual(enum_options({"enum_values": ["x"]}, lambda k: None), ["x"])


class RegistryTests(unittest.TestCase):
    def test_list_available_exposes_schema_capabilities_slots(self):
        by_key = {a["key"]: a for a in architectures.list_available()}
        # 正式登記（帶 params_schema）的架構只有這兩個；其他測試模組可能暫時塞了替身架構
        self.assertEqual(sorted(k for k, a in by_key.items() if a["params_schema"] is not None), ["lstm", "xgboost"])
        lstm, xgb = by_key["lstm"], by_key["xgboost"]
        self.assertEqual(set(lstm["capabilities"].values()), {"configurable"})
        self.assertEqual(set(xgb["capabilities"].values()), {"fixed_off"})
        self.assertEqual(lstm["slots"][0]["slot_name"], "attention")
        self.assertEqual({f["name"] for f in xgb["params_schema"]} & {"epochs"}, set())  # 輪數各自表達
        self.assertIn("epochs", {f["name"] for f in lstm["params_schema"]})
        self.assertIn("n_estimators", {f["name"] for f in xgb["params_schema"]})
        self.assertNotIn("extra_checks", lstm)

    def test_component_compatibility_is_checked_in_code(self):
        attention.ATTENTION_REGISTRY["seq_only"] = {
            **attention.ATTENTION_REGISTRY["additive"], "key": "seq_only", "output_kind": "sequence"}
        attention.ATTENTION_REGISTRY["other_family"] = {
            **attention.ATTENTION_REGISTRY["additive"], "key": "other_family", "slot_compatibility": [("gru", "attention")]}
        try:
            base = H.lstm_params(attention=None)
            for kind, needle in (("seq_only", "輸出形狀"), ("other_family", "相容")):
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, needle):
                    architectures.validate("lstm", {**base, "attention": {"type": kind, "dim": 2}},
                                           {"task_type": "classification"})
        finally:
            attention.ATTENTION_REGISTRY.pop("seq_only", None)
            attention.ATTENTION_REGISTRY.pop("other_family", None)

    def test_attention_interface_checked_when_building(self):
        import torch

        module, dim = attention.build_checked({"type": "additive", "dim": 3}, 5)
        ctx, weights = module(torch.zeros(4, 7, 5))
        self.assertEqual((tuple(ctx.shape), tuple(weights.shape), dim), ((4, 5), (4, 7), 5))

    def test_new_architecture_registers_without_touching_framework(self):
        """擴充性：新增一個架構＝呼叫 register()，驗證與列出自動生效。"""
        schema = [{"name": "width", "type": "int", "required": True, "min": 1}]

        @architectures.register("toy_mamba", family="mamba", label="Toy", params_schema=schema)
        def train_toy(*a, **k):
            return {}

        try:
            self.assertEqual(architectures.validate("toy_mamba", {"width": 4}, {"task_type": "regression"}), {"width": 4})
            with self.assertRaisesRegex(ValueError, "width"):
                architectures.validate("toy_mamba", {}, {"task_type": "regression"})
            self.assertIn("toy_mamba", {a["key"] for a in architectures.list_available()})
        finally:
            for reg in (architectures.ARCHITECTURE_REGISTRY, architectures.ARCHITECTURE_META, architectures.ARCHITECTURE_OUTPUTS):
                reg.pop("toy_mamba", None)


class BestTrackerTests(unittest.TestCase):
    def test_patience_zero_never_stops_but_tracks_best(self):
        t = BestTracker("min", 0)
        for epoch, v in enumerate([3, 2, 5, 6, 7, 8]):
            _, stop = t.update(epoch, v)
            self.assertFalse(stop)
        self.assertEqual((t.best_epoch, t.best_value), (1, 2))

    def test_patience_and_min_delta_and_max_mode(self):
        t = BestTracker("max", 2, min_delta=0.1)
        results = [t.update(e, v) for e, v in enumerate([0.5, 0.55, 0.58])]
        self.assertEqual(results[1], (False, False))  # 進步不到 min_delta
        self.assertEqual(results[2], (False, True))   # 連續兩輪沒進步 → 停
        self.assertEqual(t.best_epoch, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
