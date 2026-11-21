"""實驗層 API（routers/model_experiments.py）：設定驗證、組別圖組裝、Phase 1／2／3 建立任務、門檻 409 與放行紀錄、
修改限制。資料庫以 stub 取代。"""

import asyncio
import os
import sys
import unittest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from fastapi import HTTPException  # noqa: E402

import tests  # noqa: E402,F401 — 正式 hermesnote DB 防護（tests/__init__.py），必須在 training／services 之前
from routers import model_experiments as mx  # noqa: E402
from tests.contract_harness import xgb_params  # noqa: E402

SETTINGS = {"start": "2016-01-01", "end": "2024-01-01", "holdout": {"start": "2024-01-01", "end": "2026-01-01"},
            "label": {"timeframe": "tx_60m", "outcome": "log_return", "labeling_rule": "fixed_threshold",
                      "params": {"horizon": 1, "n_classes": 2}},
            "split_seed": 7, "val_ratio": 0.2,
            "gates": [{"phase": 2, "metric": "macro_f1", "op": ">=", "value": 0.6, "weights": "best"}]}
ARM = [{"id": "f", "type": "feature", "key": "ohlcv", "timeframe": "tx_60m"},
       {"id": "m", "type": "model", "key": "xgboost", "inputs": ["f"], "label": "label", "window": 10, "params": xgb_params()}]


class Stub:
    def __init__(self):
        self.exp = {"id": "e1", "settings": dict(SETTINGS), "jobs": [], "arms": [], "frames": []}
        self.arm = {"id": "a1", "experiment_id": "e1", "spec": {"nodes": ARM}, "model_seeds": [1, 2]}
        self.jobs, self.created, self.usage = {}, [], []

    def install(self, test):
        es, js = mx.es, mx.job_store
        saved = {n: getattr(es, n) for n in ("get_experiment", "get_arm", "latest_frame", "write_holdout_usage",
                                              "create_experiment", "update_settings")}
        saved_js = {n: getattr(js, n) for n in ("create_job", "get_job")}

        async def get_experiment(i):
            return self.exp if i == "e1" else None

        async def get_arm(i):
            return self.arm if i == "a1" else None

        async def latest_frame(e, phase, create=True):
            return {"id": f"fr{phase}", "phase": phase, "version": 1}

        async def write_holdout_usage(kind, **kw):
            self.usage.append(kind)

        async def create_experiment(name, desc, settings, created_by=None):
            return "e-new"

        async def update_settings(i, s):
            self.exp["settings"] = s

        async def create_job(spec, **kw):
            jid = f"j{len(self.created)}"
            self.created.append((spec, kw))
            return jid

        async def get_job(i):
            return self.jobs.get(i)

        for n, f in [("get_experiment", get_experiment), ("get_arm", get_arm), ("latest_frame", latest_frame),
                     ("write_holdout_usage", write_holdout_usage), ("create_experiment", create_experiment),
                     ("update_settings", update_settings)]:
            setattr(es, n, f)
        js.create_job, js.get_job = create_job, get_job

        def restore():
            for n, f in saved.items():
                setattr(es, n, f)
            for n, f in saved_js.items():
                setattr(js, n, f)
        test.addCleanup(restore)


def run(c):
    return asyncio.run(c)


class ExperimentApiTests(unittest.TestCase):
    def setUp(self):
        self.s = Stub()
        self.s.install(self)

    def test_settings_validation(self):
        bad = {**SETTINGS, "holdout": {"start": "2023-01-01", "end": "2026-01-01"}}
        with self.assertRaises(HTTPException) as cm:
            run(mx.create_experiment(mx.ExperimentRequest(name="x", settings=bad)))
        self.assertIn("holdout", cm.exception.detail)
        r = run(mx.create_experiment(mx.ExperimentRequest(name="x", settings=SETTINGS)))
        self.assertEqual(r["experiment_id"], "e-new")

    def test_arm_graph_injects_label_seed_and_frame(self):
        g = mx.arm_graph(self.s.exp, ARM, seed=5, arm_id="a1", frame={"id": "fr1", "phase": 1, "version": 1})
        self.assertEqual(g["mode"], "v2")
        self.assertEqual(g["settings"]["split_seed"], 7)
        self.assertEqual(next(n for n in g["nodes"] if n["id"] == "m")["params"]["seed"], 5)
        self.assertEqual(next(n for n in g["nodes"] if n["id"] == "label")["timeframe"], "tx_60m")
        self.assertEqual(g["frame_ref"]["frame_id"], "fr1")
        self.assertEqual(mx._arm_issues(self.s.exp, ARM, [1]), [])
        wrong = [ARM[0], {**ARM[1], "label": "own"}, {"id": "own", "type": "label", "timeframe": "tx_60m",
                                                       "outcome": "log_return", "params": {"horizon": 2}}]
        self.assertTrue(any("實驗的 Label" in i for i in mx._arm_issues(self.s.exp, wrong, [1])))

    def test_phase1_creates_one_job_per_seed(self):
        r = run(mx.run_arm("e1", "a1", mx.RunRequest(phase=1)))
        self.assertEqual([j["model_seed"] for j in r["jobs"]], [1, 2])
        spec, kw = self.s.created[0]
        self.assertEqual(kw["experiment_id"], "e1")
        self.assertEqual(kw["frame_id"], "fr1")
        self.assertTrue(all(n.get("split_strategy") == "random" for n in spec["nodes"] if n["type"] == "model"))

    def _parent(self, f1):
        g = mx.arm_graph(self.s.exp, ARM, seed=1, arm_id="a1")
        self.s.jobs["p1"] = {"job_id": "p1", "phase": 1, "status": "done", "arm_id": "a1", "model_seed": 1, "graph_spec": g,
                             "result": {"m": {"evaluations": [{"head": "default", "point": {"kind": "best"},
                                                               "metrics": {"macro_f1": f1}}]}}}
        self.s.exp["jobs"] = [{"id": "p1", "phase": 1, "status": "done", "arm_id": "a1", "job_type": "train"}]

    def test_gate_blocks_then_override_recorded(self):
        self._parent(0.55)
        with self.assertRaises(HTTPException) as cm:
            run(mx.run_arm("e1", "a1", mx.RunRequest(phase=2)))
        self.assertEqual(cm.exception.status_code, 409)
        self.assertEqual(cm.exception.detail["gate_check"]["conditions"][0]["actual"], 0.55)
        who = {"kind": "admin", "id": "admin:h@example.com", "scopes": []}
        r = run(mx.run_arm("e1", "a1", mx.RunRequest(phase=2, gate_override=mx.GateOverride(reason="探索", confirmed_by="H")),
                           principal=who))
        gc = r["jobs"][0]["gate_check"]
        self.assertEqual(gc["status"], "overridden")
        # 確認人取自驗證身分；呼叫端送的文字只當備註
        self.assertEqual(gc["override"]["confirmed_by"], "admin:h@example.com")
        self.assertEqual(gc["override"]["note_confirmed_by"], "H")
        self.assertEqual(self.s.created[-1][1]["requested_by"], "admin:h@example.com")
        spec, kw = self.s.created[-1]
        self.assertTrue(all(n.get("split_strategy") == "chronological" for n in spec["nodes"] if n["type"] == "model"))
        self.assertEqual(kw["parent_job_id"], "p1")

    def test_gate_pass_and_phase3_usage(self):
        self._parent(0.7)
        r = run(mx.run_arm("e1", "a1", mx.RunRequest(phase=2)))
        self.assertEqual(r["jobs"][0]["gate_check"]["status"], "passed")
        self.s.jobs["p2"] = {**self.s.jobs["p1"], "job_id": "p2", "phase": 2}
        self.s.exp["jobs"] = [{"id": "p2", "phase": 2, "status": "done", "arm_id": "a1", "job_type": "train"}]
        self.s.exp["settings"]["gates"] = []
        r = run(mx.run_arm("e1", "a1", mx.RunRequest(phase=3)))
        spec, kw = self.s.created[-1]
        self.assertEqual(kw["job_type"], "infer")
        self.assertEqual(spec["start"], "2024-01-01")
        self.assertEqual(spec["frame_ref"], {"frame_id": "fr3", "phase": 3, "version": 1})
        self.assertEqual(kw["frame_id"], "fr3")
        self.assertEqual(self.s.usage, ["inference"])
        self.assertEqual(r["jobs"][0]["gate_check"]["status"], "not_configured")

    def test_patch_only_mutable_keys(self):
        with self.assertRaises(HTTPException):
            run(mx.patch_settings("e1", mx.SettingsPatch(settings={"split_seed": 1})))
        r = run(mx.patch_settings("e1", mx.SettingsPatch(settings={"alignment": {"allow_previous_session": False}}, note="嚴格")))
        self.assertFalse(r["settings"]["alignment"]["allow_previous_session"])
        self.assertEqual(r["settings"]["history"][-1]["changed"], ["alignment"])
        with self.assertRaises(HTTPException):
            run(mx.patch_settings("e1", mx.SettingsPatch(settings={"alignment": {"max_sessions_back": 9}})))


if __name__ == "__main__":
    unittest.main(verbosity=2)
