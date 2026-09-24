"""LSTM 雙頭（key="lstm"、heads="dual"）走教授三階段協定（Phase 1 → 2 → 3）與「同設定重跑」所需的後端行為。

Phase 2 繼承 Phase 1 的圖並改成時間切分；Phase 3 載入 Phase 2 保存的權重對 holdout 期間推論。
路由層（血緣與 holdout 邊界檢查）用 monkeypatch 隔離資料庫，不寫正式庫。
執行：python -m unittest tests.test_phases_dual
"""

import asyncio
import copy
import os
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from routers import model_training as router  # noqa: E402
from tests import contract_harness as H  # noqa: E402
from tests.test_lstm_dual import KEY, daily_df, graph_spec, params  # noqa: E402
from training import graph as graph_mod  # noqa: E402
from training import inference as inference_mod  # noqa: E402
from training.graph_refs import validate_graph_spec  # noqa: E402
from training.phases import build_phase2_graph_spec, final_model_node_id  # noqa: E402


def phase1_spec():
    spec = graph_spec(params(epochs=3))
    spec["nodes"][-1]["split_strategy"] = "random"
    spec["end"] = "2024-02-01"
    return spec


class Phase2InheritanceTests(unittest.TestCase):
    def test_phase2_inherits_model_params_verbatim_including_early_stopping(self):
        # Phase 2 只覆寫 split_strategy，params（含 patience=0 的研究設定）原樣繼承、不代為補值。
        p1 = phase1_spec()
        p1["nodes"][-1]["params"]["early_stopping"] = {"monitor": "val_rmse", "patience": 0}
        validate_graph_spec(p1)
        p2 = build_phase2_graph_spec(p1)
        validate_graph_spec(p2)
        self.assertEqual(p2["nodes"][-1]["params"], p1["nodes"][-1]["params"])
        self.assertEqual(p2["nodes"][-1]["split_strategy"], "chronological")

    def test_phase2_spec_only_changes_split_strategy_and_stays_valid(self):
        p1 = phase1_spec()
        p2 = build_phase2_graph_spec(p1)
        validate_graph_spec(p2)
        self.assertEqual(p2["nodes"][-1]["split_strategy"], "chronological")
        expected = copy.deepcopy(p1)
        expected["nodes"][-1]["split_strategy"] = "chronological"
        self.assertEqual(p2, expected)  # 其餘（含雙頭的全部參數）逐項相同
        self.assertEqual(p1["nodes"][-1]["split_strategy"], "random")  # 母任務的圖不被改動

    def test_final_model_node_is_the_dual_node(self):
        self.assertEqual(final_model_node_id(phase1_spec()["nodes"]), "dual")

    def test_phase2_trains_chronologically_with_dual_and_boundary_exclusion(self):
        df = daily_df()
        r = graph_mod.run_graph(build_phase2_graph_spec(phase1_spec()), lambda tf: df)["dual"]
        meta = r["training_meta"]
        self.assertEqual(meta["split_strategy"], "chronological")
        self.assertIn("n_excluded_boundary", meta)
        self.assertIn("best_epoch", r["final_metrics"])
        self.assertIn("weights", r)

    def test_phase1_and_phase2_are_deterministic_reruns(self):
        # 「同設定重跑」：固定 seed 下，同一份圖在同一份資料上重跑，指標一致
        df = daily_df()
        spec = phase1_spec()
        a = graph_mod.run_graph(spec, lambda tf: df)["dual"]["final_metrics"]
        b = graph_mod.run_graph(copy.deepcopy(spec), lambda tf: df)["dual"]["final_metrics"]
        self.assertEqual({k: a[k] for k in ("val_rmse", "val_dir_acc", "best_epoch")},
                         {k: b[k] for k in ("val_rmse", "val_dir_acc", "best_epoch")})


class Phase3HoldoutTests(unittest.TestCase):
    def test_holdout_inference_from_phase2_artifact_returns_dual_rows(self):
        df = daily_df(400)
        spec = build_phase2_graph_spec(phase1_spec())
        results = graph_mod.run_graph(spec, lambda tf: df)
        node = spec["nodes"][-1]
        art = {"architecture_key": KEY, "task_type": results["dual"]["task_type"], "weights": results["dual"]["weights"],
               "model_config": results["dual"]["model_config"], "feature_schema": [], "target_spec": {},
               "graph_spec_snapshot": spec, "depends_on_node_ids": [], "preprocessing_state": results["dual"]["preprocessing_state"]}
        holdout = df.iloc[300:]  # 訓練期間之後的一段
        orig = inference_mod.load_artifact
        inference_mod.load_artifact = lambda job, nid: art if nid == node["id"] else None
        chunks = []
        try:
            n = inference_mod.run_inference_chunked("p2job", "dual", lambda tf: holdout, lambda *a: chunks.append(a), chunk_size=50)
        finally:
            inference_mod.load_artifact = orig
        self.assertEqual(n, len(holdout) - node["window"])
        self.assertTrue(all(c[4] == "dual" for c in chunks))  # output_kind
        rows = np.asarray([r for c in chunks for r in c[1]])
        self.assertEqual(rows.shape[1], 2)  # [回歸值, 事件機率]
        self.assertTrue(((rows[:, 1] > 0) & (rows[:, 1] < 1)).all())


class RouterLineageTests(unittest.TestCase):
    """路由層：Phase 2 繼承與 Phase 3 血緣／holdout 邊界，資料庫以 stub 取代。"""

    def setUp(self):
        self.created = []
        self.jobs = {}
        p1 = phase1_spec()
        self.jobs["p1"] = {"job_id": "p1", "phase": 1, "status": "done", "graph_spec": p1}
        self.jobs["p2"] = {"job_id": "p2", "phase": 2, "status": "done", "graph_spec": build_phase2_graph_spec(p1)}

        async def get_job(jid):
            return self.jobs.get(jid)

        async def create_job(spec, **kw):
            self.created.append((spec, kw))
            return "new-job"

        async def artifact_exists(jid, nid):
            return jid in self.jobs and nid == "dual"

        self._orig = (router.job_store.get_job, router.job_store.create_job, router.model_artifacts.artifact_exists)
        router.job_store.get_job, router.job_store.create_job = get_job, create_job
        router.model_artifacts.artifact_exists = artifact_exists
        self.addCleanup(self.restore)

    def restore(self):
        router.job_store.get_job, router.job_store.create_job, router.model_artifacts.artifact_exists = self._orig

    def call(self, coro):
        return asyncio.run(coro)

    def test_phase2_route_accepts_dual_parent(self):
        out = self.call(router.start_training(router.TrainRequest(start="", end="", nodes=[], phase=2, parent_job_id="p1")))
        self.assertEqual(out, {"job_id": "new-job"})
        spec, kw = self.created[0]
        self.assertEqual(kw, {"phase": 2, "parent_job_id": "p1"})
        self.assertEqual(spec["nodes"][-1]["split_strategy"], "chronological")

    def test_phase2_rerun_uses_same_parent_and_creates_equal_spec(self):
        # 後台「重跑 Phase 2」＝同一個 parent_job_id 再送一次：兩次建立的圖完全相同
        for _ in range(2):
            self.call(router.start_training(router.TrainRequest(start="", end="", nodes=[], phase=2, parent_job_id="p1")))
        self.assertEqual(self.created[0][0], self.created[1][0])

    def test_phase1_rerun_resubmits_recorded_graph(self):
        g = self.jobs["p1"]["graph_spec"]
        self.call(router.start_training(router.TrainRequest(start=g["start"], end=g["end"], nodes=g["nodes"], phase=1)))
        self.assertEqual(self.created[0][0], g)
        self.assertEqual(self.created[0][1], {"phase": 1})

    def test_phase3_route_creates_infer_job_and_rerun_payload_matches(self):
        req = router.InferRequest(target_job_id="p2", target_node_id="dual", start="2024-03-01", end="2024-04-01", phase=3)
        self.call(router.start_inference(req))
        spec, kw = self.created[0]
        self.assertEqual(spec, {"target_job_id": "p2", "target_node_id": "dual", "start": "2024-03-01", "end": "2024-04-01",
                                "use_weights": "best"})
        self.assertEqual(kw, {"job_type": "infer", "phase": 3, "parent_job_id": "p2"})
        # 後台重跑 Phase 3：用紀錄裡的 spec 原樣＋phase=3 再送，得到同樣的推論任務
        self.call(router.start_inference(router.InferRequest(**spec, phase=3)))
        self.assertEqual(self.created[1], self.created[0])

    def test_phase3_requires_phase2_parent_and_holdout_after_training_end(self):
        with self.assertRaises(HTTPException):  # 來源是 Phase 1，不是 Phase 2
            self.call(router.start_inference(router.InferRequest(target_job_id="p1", target_node_id="dual", start="2025-01-01", end="2025-02-01", phase=3)))
        with self.assertRaises(HTTPException) as ctx:  # holdout 與訓練期間重疊
            self.call(router.start_inference(router.InferRequest(target_job_id="p2", target_node_id="dual", start="2024-01-15", end="2024-03-01", phase=3)))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_infer_without_node_id_finds_the_dual_node(self):
        self.call(router.start_inference(router.InferRequest(target_job_id="p2", start="2024-03-01", end="2024-04-01", phase=3)))
        self.assertEqual(self.created[0][0]["target_node_id"], "dual")


if __name__ == "__main__":
    unittest.main(verbosity=2)
