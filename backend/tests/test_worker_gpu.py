"""training worker 的 GPU 顯存判斷與「同一 worker 連續接任務」回歸測試。

2026-09-29 事故：舊判斷是「總已用顯存 > 2000MB 就視為忙碌」，16GB 的 GPU 上唯一的佔用者是 worker
自己（約 2306MiB），GPU 使用率 0%，worker 卻反覆延後、新任務一直 pending。現在改成先釋放自身快取，
再依實際可用顯存對照 TRAINING_GPU_MIN_FREE_MB 判斷。

執行：python -m unittest tests.test_worker_gpu
"""

import asyncio
import unittest
from unittest import mock

from tests import contract_harness as H  # CPU＋合成資料

from training import worker

INCIDENT = {"total_mb": 16311, "used_mb": 2306, "free_mb": 14005}  # 事故當時：只有 worker 自己


class GpuCapacityTests(unittest.TestCase):
    def test_incident_numbers_no_longer_block(self):
        with mock.patch.object(worker, "query_gpu_memory", return_value=INCIDENT):
            ok, reason = worker.gpu_capacity()
        self.assertTrue(ok, reason)

    def test_insufficient_free_memory_blocks_with_reason(self):
        mem = {"total_mb": 16311, "used_mb": 13311, "free_mb": 3000}  # 例如外部 12B 模型佔用
        with mock.patch.object(worker, "query_gpu_memory", return_value=mem):
            ok, reason = worker.gpu_capacity()
        self.assertFalse(ok)
        for needle in ("total=16311MiB", "used=13311MiB", "free=3000MiB", f"free>={worker.GPU_MIN_FREE_MB}MiB",
                       "TRAINING_GPU_MIN_FREE_MB", "worker自身"):
            self.assertIn(needle, reason)

    def test_threshold_is_configurable(self):
        with mock.patch.object(worker, "query_gpu_memory", return_value=INCIDENT), \
                mock.patch.object(worker, "GPU_MIN_FREE_MB", 15000):
            self.assertFalse(worker.gpu_capacity()[0])

    def test_no_gpu_does_not_block(self):
        with mock.patch.object(worker, "query_gpu_memory", return_value=None):
            self.assertTrue(worker.gpu_capacity()[0])

    def test_capacity_check_releases_own_cache_first(self):
        calls = []
        with mock.patch.object(worker, "release_gpu_memory", side_effect=lambda: calls.append("release")), \
                mock.patch.object(worker, "query_gpu_memory", side_effect=lambda: calls.append("query") or INCIDENT):
            worker.gpu_capacity()
        self.assertEqual(calls, ["release", "query"])


class _Stop(Exception):
    pass


def _spec(seed):
    return {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
        H._feat("fa", "raw_price"), H._label("lc", "log_return", "fixed_threshold", n_classes=2),
        H._model("m", "lstm", ["fa"], "lc", params=H.lstm_params(seed=seed, epochs=2))]}


class SequentialJobsTests(unittest.TestCase):
    def test_same_worker_runs_next_job_after_finishing_one(self):
        """真的跑兩筆訓練（CPU、合成資料，DB 寫入換成記錄），第一筆完成後 GPU 顯示仍有 worker 佔用
        （事故數字），第二筆照常被接走並完成；每筆結束都釋放快取。"""
        queue = [{"id": "job-1", "job_type": "train", "graph_spec": _spec(1)},
                 {"id": "job-2", "job_type": "train", "graph_spec": _spec(2)}]
        events = []

        async def fetch_next_pending():
            if queue:
                return queue.pop(0)
            raise _Stop()

        async def mark(kind, job_id, *a):
            events.append((kind, job_id))

        async def load_dataframes(graph_spec):
            return {"tx_1m": H.data_loader("tx_1m")}

        async def no_sleep(_):
            events.append(("sleep", None))

        release = mock.MagicMock(wraps=worker.release_gpu_memory)
        with mock.patch.object(worker.job_store, "fetch_next_pending", fetch_next_pending), \
                mock.patch.object(worker.job_store, "mark_running", lambda j, d: mark("running", j)), \
                mock.patch.object(worker.job_store, "mark_done", lambda j, r: mark("done", j)), \
                mock.patch.object(worker.job_store, "mark_failed", lambda j, e: mark(f"failed:{e}", j)), \
                mock.patch.object(worker, "_load_dataframes", load_dataframes), \
                mock.patch.object(worker, "write_progress", lambda *a: None), \
                mock.patch.object(worker, "save_preview_sample", lambda *a: None), \
                mock.patch.object(worker, "save_artifacts_for_job", lambda *a: events.append(("artifacts", a[0]))), \
                mock.patch.object(worker, "query_gpu_memory", return_value=INCIDENT), \
                mock.patch.object(worker, "release_gpu_memory", release), \
                mock.patch.object(worker.asyncio, "sleep", no_sleep):
            with self.assertRaises(_Stop):
                asyncio.run(worker.main_loop())

        self.assertEqual([e for e in events if e[0] != "artifacts"],
                         [("running", "job-1"), ("done", "job-1"), ("running", "job-2"), ("done", "job-2")])
        self.assertEqual([e[1] for e in events if e[0] == "artifacts"], ["job-1", "job-2"])
        # 每筆：開始前判斷顯存時釋放一次＋結束後釋放一次
        self.assertEqual(release.call_count, 4)

    def test_failed_job_still_releases_and_next_job_runs(self):
        bad = _spec(1)
        bad["nodes"][-1]["params"] = {}  # 參數不合法 → 驗證失敗
        queue = [{"id": "bad", "job_type": "train", "graph_spec": bad},
                 {"id": "good", "job_type": "train", "graph_spec": _spec(2)}]
        events = []

        async def fetch_next_pending():
            if queue:
                return queue.pop(0)
            raise _Stop()

        async def mark(kind, job_id):
            events.append((kind, job_id))

        async def load_dataframes(graph_spec):
            return {"tx_1m": H.data_loader("tx_1m")}

        with mock.patch.object(worker.job_store, "fetch_next_pending", fetch_next_pending), \
                mock.patch.object(worker.job_store, "mark_running", lambda j, d: mark("running", j)), \
                mock.patch.object(worker.job_store, "mark_done", lambda j, r: mark("done", j)), \
                mock.patch.object(worker.job_store, "mark_failed", lambda j, e: mark("failed", j)), \
                mock.patch.object(worker, "_load_dataframes", load_dataframes), \
                mock.patch.object(worker, "write_progress", lambda *a: None), \
                mock.patch.object(worker, "save_preview_sample", lambda *a: None), \
                mock.patch.object(worker, "save_artifacts_for_job", lambda *a: None), \
                mock.patch.object(worker, "query_gpu_memory", return_value=INCIDENT):
            with self.assertRaises(_Stop):
                asyncio.run(worker.main_loop())
        self.assertEqual(events, [("running", "bad"), ("failed", "bad"), ("running", "good"), ("done", "good")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
