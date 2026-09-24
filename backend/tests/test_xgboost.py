"""XGBoost（key="xgboost"）在統一框架下的驗收：schema 驗證、n_estimators（boosting rounds）與
早停（patience=0 關閉）、best／last（截樹 vs 全部樹）保存與評估、二元／多元機率與重載一致、回歸。

執行：python -m unittest tests.test_xgboost
"""

import json
import unittest

from tests import contract_harness as H  # 會設定 CPU 與 sys.path

import numpy as np

from training.registry import architectures

KEY = "xgboost"


def data(kind: str, n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5, 3)).astype(np.float32)
    s = X[:, -1, 0]
    if kind == "bin":
        return X, (s > 0).astype(int), "classification", 2
    if kind == "tri":
        return X, np.digitize(s, [-0.4, 0.4]), "classification", 3
    return X, s + rng.normal(0, 0.1, n), "regression", None


class XgboostTests(unittest.TestCase):
    def train(self, kind, **over):
        X, y, task, k = data(kind)
        seen = []
        res = architectures.train_model(KEY, X[:200], y[:200], X[200:], y[200:],
                                        H.xgb_params(n_estimators=40, **over), lambda e, m: seen.append(m),
                                        task_type=task, n_classes=k)
        return res, seen, X[200:], y[200:]

    def test_patience_zero_grows_all_rounds_and_best_is_argmin(self):
        res, seen, Xv, yv = self.train("bin")
        fm = res["final_metrics"]
        self.assertEqual((fm["rounds_run"], len(seen), fm["stopped_early"]), (40, 40, False))
        self.assertEqual(fm["best_round"], int(np.argmin([m["val_loss"] for m in seen])))
        self.assertEqual(fm["monitor"], "val_logloss")

    def test_positive_patience_stops_after_best(self):
        res, seen, _, _ = self.train("bin", early_stopping={"patience": 3}, learning_rate=0.8)
        fm = res["final_metrics"]
        self.assertLessEqual(fm["rounds_run"], 40)
        if fm["stopped_early"]:
            self.assertEqual(fm["rounds_run"] - 1 - fm["best_round"], 3)

    def test_best_and_last_saved_evaluated_and_reload_identical(self):
        for kind in ("bin", "tri", "reg"):
            with self.subTest(kind=kind):
                res, _, Xv, yv = self.train(kind)
                self.assertEqual(sorted(res["evaluation"]), ["best", "last"])
                self.assertIn("last", res["final_metrics"])
                key = "predict" if kind == "reg" else "predict_proba"
                for weights in (res["weights"], res["weights_last"]):
                    loaded = architectures.load_model(KEY, weights, res["model_config"])
                    self.assertEqual(len(loaded[key](Xv)), len(Xv))
                loaded = architectures.load_model(KEY, res["weights"], res["model_config"])
                np.testing.assert_allclose(res[key](Xv), loaded[key](Xv), atol=1e-6)
                json.dumps(res["model_config"])

    def test_multiclass_probabilities_available_after_reload(self):
        res, _, Xv, _ = self.train("tri")
        proba = architectures.load_model(KEY, res["weights"], res["model_config"])["predict_proba"](Xv)
        self.assertEqual(proba.shape, (100, 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_class_count_comes_from_label_even_if_a_class_never_occurs(self):
        """3 類 Label、threshold_pct=0 時「平」幾乎不會出現：類別數依 Label 的 n_classes，不從 y 推。"""
        X, y, _, _ = data("tri")
        y = np.where(y == 1, 2, y)  # 只剩 0 與 2
        res = architectures.train_model(KEY, X[:200], y[:200], X[200:], y[200:], H.xgb_params(n_estimators=5), None,
                                        task_type="classification", n_classes=3)
        self.assertEqual(res["predict_proba"](X[200:]).shape, (100, 3))
        # 驗證集缺類別時 OvR ROC-AUC 沒有定義：回 None＋原因，不是 NaN（NaN 寫不進 JSONB）
        ev = res["evaluation"]["best"]
        self.assertIsNone(ev["roc_auc"])
        self.assertIn("沒有出現類別", ev["roc_auc_unavailable_reason"])

    def test_non_finite_values_are_stored_as_null(self):
        from training import json_safe

        text = json_safe.dumps({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2}})
        self.assertEqual(json.loads(text), {"a": None, "b": [1.0, None], "c": {"d": 2}})

    def test_evaluation_reports_classification_and_regression_metrics(self):
        res, _, _, _ = self.train("bin")
        for key in ("confusion_matrix", "macro_f1", "roc_auc", "average_precision", "baseline_majority_class"):
            self.assertIn(key, res["evaluation"]["best"])
        res, _, _, _ = self.train("reg")
        for key in ("rmse", "mae", "r2", "baseline_mean", "baseline_zero"):
            self.assertIn(key, res["evaluation"]["best"])

    def test_schema_validation(self):
        with self.assertRaisesRegex(ValueError, "n_estimators"):
            architectures.validate(KEY, {k: v for k, v in H.xgb_params().items() if k != "n_estimators"},
                                   {"task_type": "classification"})
        with self.assertRaisesRegex(ValueError, "不是這個模型認得的欄位"):
            architectures.validate(KEY, H.xgb_params(epochs=10), {"task_type": "classification"})
        with self.assertRaisesRegex(ValueError, "monitor"):  # XGBoost 監控指標固定，不開放選擇
            architectures.validate(KEY, H.xgb_params(early_stopping={"patience": 1, "monitor": "val_loss"}),
                                   {"task_type": "classification"})
        cfg = architectures.validate(KEY, H.xgb_params(), {"task_type": "regression"})
        self.assertEqual((cfg["subsample"], cfg["colsample_bytree"]), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
