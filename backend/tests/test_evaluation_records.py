"""通用指標視覺化的資料契約：metric_specs（模型引用的指標描述）與 evaluations（評估清單）。

- 模型宣告的逐輪序列鍵，必須真的出現在訓練迴圈回報的欄位裡（各模式都實際跑一次）
- 評估清單的數值與原始報告逐一相同（只重組＋由混淆矩陣推算 accuracy）
- 舊紀錄（evaluation.best／last）讀取時轉換，原欄位保留；新任務只存 evaluations＋metric_specs
- 驗證集說明：random 標明固定種子、分散整個期間；chronological 標明最後一段與邊界排除

執行：python -m unittest tests.test_evaluation_records
"""

import copy
import json
import unittest

from tests import contract_harness as H

from training import graph as graph_mod
from training import worker
from training.evaluation_records import build_evaluations, enrich_job
from training.registry import architectures, metrics

LSTM_DUAL = dict(heads="dual", direction_rule={"op": ">=", "threshold": 0.0}, loss_weights={"regression": 0.8, "direction": 0.2},
                 early_stopping={"monitor": "val_rmse", "patience": 0})
CASES = {
    "lstm_cls": ("lstm", H.lstm_params(epochs=2), ("log_return", "fixed_threshold", {"n_classes": 2})),
    "lstm_cls3": ("lstm", H.lstm_params(epochs=2), ("log_return", "fixed_threshold", {"n_classes": 3, "threshold_pct": 0.05})),
    "lstm_reg": ("lstm", H.lstm_params(epochs=2, early_stopping={"monitor": "val_rmse", "patience": 0}), ("log_return", "identity", {})),
    "lstm_dual": ("lstm", H.lstm_params(epochs=2, **LSTM_DUAL), ("log_return", "identity", {})),
    "xgb_cls": ("xgboost", H.xgb_params(n_estimators=3), ("log_return", "fixed_threshold", {"n_classes": 2})),
    "xgb_cls3": ("xgboost", H.xgb_params(n_estimators=3), ("log_return", "fixed_threshold", {"n_classes": 3, "threshold_pct": 0.05})),
    "xgb_reg": ("xgboost", H.xgb_params(n_estimators=3), ("log_return", "identity", {})),
}
_RUNS: dict = {}


def run(case: str, split: str = "random"):
    """實際訓練一次，回傳 (graph_spec, 完整結果, 逐輪回報)；同一 case 快取。"""
    key = (case, split)
    if key not in _RUNS:
        arch, params, (outcome, rule, lp) = CASES[case]
        spec = {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
            H._feat("fa", "raw_price"), H._label("l", outcome, rule, **lp),
            H._model("m", arch, ["fa"], "l", split_strategy=split, params=copy.deepcopy(params))]}
        progress = []
        results = graph_mod.run_graph(spec, H.data_loader, on_epoch=lambda n, e, m: progress.append(m))
        _RUNS[key] = (spec, results, progress)
    return _RUNS[key]


def old_style_job(case: str, phase=1, split="random"):
    """舊格式紀錄：result 只有 final_metrics／training_meta／output_specs／evaluation（正式庫現有的形狀）。"""
    spec, results, _ = run(case, split)
    r = results["m"]
    result = {"m": {k: json.loads(json.dumps(r[k])) for k in ("final_metrics", "training_meta", "output_specs", "evaluation")}}
    for k in ("n_samples", "val_ratio", "split_seed"):  # 舊紀錄沒有這幾個欄位
        result["m"]["training_meta"].pop(k, None)
    return {"job_id": "old", "job_type": "train", "phase": phase, "graph_spec": spec, "status": "done", "result": result}


class MetricSpecsTests(unittest.TestCase):
    def test_declared_series_keys_are_actually_reported(self):
        for case in CASES:
            with self.subTest(case=case):
                spec, _, progress = run(case)
                node = spec["nodes"][-1]
                ms = architectures.metric_specs(node["key"], node, spec["nodes"][1], "regression" if "reg" in case or "dual" in case else "classification")
                reported = set().union(*(p.keys() for p in progress))
                for s in ms["series"]:
                    for side in ("train", "val"):
                        if side in s:
                            self.assertIn(s[side], reported, f"{case} {s['id']}.{side}")
                    self.assertIn(s["metric"], ms["definitions"])
                heads = {h["key"] for h in ms["heads"]}
                self.assertTrue({s["head"] for s in ms["series"]} <= heads)
                self.assertTrue(set(ms["evaluation"]) <= heads)

    def test_round_unit_and_loss_meaning(self):
        spec, _, _ = run("xgb_cls3")
        node = spec["nodes"][-1]
        ms = architectures.metric_specs("xgboost", node, spec["nodes"][1], "classification")
        self.assertEqual((ms["round_unit"], ms["series"][0]["metric"]), ("boosting_round", "mlogloss"))
        spec, _, _ = run("lstm_reg")
        ms = architectures.metric_specs("lstm", spec["nodes"][-1], spec["nodes"][1], "regression")
        self.assertEqual((ms["series"][0]["metric"], ms["series"][0]["unit"]), ("mse", "loss_space"))
        self.assertEqual(ms["heads"][0]["target_unit"], "log_ratio")

    def test_every_referenced_metric_is_registered(self):
        with self.assertRaises(KeyError):
            metrics.definitions(["no_such_metric"])
        shapes = {m["key"]: m["shape"] for m in metrics.list_available()}
        self.assertEqual((shapes["confusion_matrix"], shapes["per_class"], shapes["class_distribution"], shapes["rmse"]),
                         ("matrix", "per_class", "distribution", "scalar"))


class EvaluationRecordsTests(unittest.TestCase):
    def _records(self, case, split="random"):
        job = enrich_job(old_style_job(case, split=split))
        return job, job["result"]["m"]["evaluations"]

    def test_values_identical_to_raw_reports(self):
        for case in CASES:
            with self.subTest(case=case):
                job, recs = self._records(case)
                raw = job["result"]["m"]["evaluation"]
                for rec in recs:
                    report = raw[rec["point"]["kind"]]
                    if rec["head"] != "default":
                        report = report[rec["head"]]
                    for k, v in rec["metrics"].items():
                        if k == "accuracy":
                            m = report["confusion_matrix"]
                            self.assertAlmostEqual(v, sum(m[i][i] for i in range(len(m))) / sum(map(sum, m)))
                        elif k == "confusion_matrix":
                            self.assertEqual(v["values"], report["confusion_matrix"])
                            self.assertEqual(v["labels"], report["confusion_matrix_labels"])
                        else:
                            self.assertEqual(v, report[k], f"{case} {k}")
                    for b in rec["baselines"]:
                        raw_key = {"majority_class": "baseline_majority_class", "train_mean": "baseline_mean",
                                   "zero": "baseline_zero"}[b["key"]]
                        for mk, mv in b["metrics"].items():
                            self.assertEqual(mv, report[raw_key][mk])
                    for k, reason in rec["unavailable"].items():
                        self.assertEqual(reason, report[f"{k}_unavailable_reason"])

    def test_identity_of_each_record(self):
        job, recs = self._records("lstm_dual")
        self.assertEqual(sorted(r["id"] for r in recs),
                         ["m/direction/val/best", "m/direction/val/last", "m/regression/val/best", "m/regression/val/last"])
        fm = job["result"]["m"]["final_metrics"]
        best = next(r for r in recs if r["point"]["kind"] == "best")
        last = next(r for r in recs if r["point"]["kind"] == "last")
        self.assertEqual((best["point"]["round"], last["point"]["round"]), (fm["best_epoch"], fm["epochs_run"] - 1))
        self.assertEqual(best["point"]["selected_by"]["monitor"], "val_rmse")
        ds = best["dataset"]
        self.assertEqual((ds["split"], ds["phase"], ds["task_data_range"]), ("val", 1, {"start": "2024-01-01", "end": "2024-02-01"}))
        _, recs = self._records("xgb_cls")
        fm = old_style_job("xgb_cls")["result"]["m"]["final_metrics"]
        self.assertEqual(recs[0]["point"]["round"], fm["best_round"])
        self.assertEqual(recs[0]["point"]["round_unit"], "boosting_round")

    def test_validation_set_description(self):
        _, recs = self._records("lstm_cls")
        ds = recs[0]["dataset"]
        self.assertEqual(ds["selection"], "random")
        self.assertEqual(ds["n_total"], ds["n_samples"] + ds["n_train"])  # 舊紀錄沒有 n_samples 也推得出來
        for needle in ("隨機打亂", "固定種子 42", "20%", "分散在整個期間"):
            self.assertIn(needle, ds["description"])
        _, recs = self._records("lstm_cls", split="chronological")
        ds = recs[0]["dataset"]
        self.assertEqual(ds["selection"], "chronological")
        self.assertIn("最後 20%", ds["description"])

    def test_old_record_keeps_raw_field_and_enrich_is_idempotent(self):
        job = old_style_job("lstm_cls")
        once = enrich_job(copy.deepcopy(job))
        self.assertIn("evaluation", once["result"]["m"])
        twice = enrich_job(copy.deepcopy(once))
        self.assertEqual(once["result"]["m"]["evaluations"], twice["result"]["m"]["evaluations"])
        self.assertIsNotNone(once["metric_specs"]["m"])

    def test_running_job_gets_metric_specs_without_result(self):
        spec, _, _ = run("lstm_cls")
        job = enrich_job({"job_type": "train", "phase": 1, "graph_spec": spec, "status": "running", "result": None})
        self.assertEqual(job["metric_specs"]["m"]["round_unit"], "epoch")

    def test_infer_job_untouched(self):
        job = {"job_type": "infer", "graph_spec": {"target_job_id": "x"}, "result": {"m": {"count": 3}}}
        self.assertEqual(enrich_job(copy.deepcopy(job)), job)


class NewJobSummaryTests(unittest.TestCase):
    def test_new_jobs_store_evaluations_not_raw_evaluation(self):
        spec, results, _ = run("lstm_dual")
        summary = worker._summarize(results, spec, phase=2)["m"]
        self.assertNotIn("evaluation", summary)
        self.assertEqual(len(summary["evaluations"]), 4)
        self.assertEqual(summary["evaluations"][0]["dataset"]["phase"], 2)
        self.assertEqual(summary["training_meta"]["split_seed"], 42)
        json.dumps(summary)
        # 讀取時不重算、不覆寫保存的內容
        job = {"job_type": "train", "phase": 2, "graph_spec": spec, "result": {"m": copy.deepcopy(summary)}}
        self.assertEqual(enrich_job(job)["result"]["m"]["evaluations"], summary["evaluations"])

    def test_same_builder_for_new_and_old(self):
        spec, results, _ = run("xgb_reg")
        r = results["m"]
        direct = build_evaluations("m", r["evaluation"], node=spec["nodes"][-1], final_metrics=r["final_metrics"],
                                   training_meta=r["training_meta"], phase=1,
                                   task_range={"start": spec["start"], "end": spec["end"]}, round_unit="boosting_round")
        self.assertEqual(direct, worker._summarize(results, spec, phase=1)["m"]["evaluations"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
