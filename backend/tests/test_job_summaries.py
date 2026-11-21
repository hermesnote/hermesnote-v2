"""GET /api/model/job-summaries：訓練頁下拉選單的輕量任務清單（只有 ID、類型、狀態、建立時間）。

執行：python -m unittest tests.test_job_summaries
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # noqa: E402,F401 — 正式 hermesnote DB 防護（tests/__init__.py），必須在 training／services 之前

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from training import job_store  # noqa: E402


class JobSummariesTests(unittest.TestCase):
    def setUp(self):
        self.seen = []
        rows = [{"job_id": f"00000000-0000-0000-0000-{i:012d}", "job_type": "train", "status": "done",
                 "created_at": f"2026-11-{i % 28 + 1:02d}T00:00:00+00:00"} for i in range(150)]

        async def summaries(limit=1000):
            self.seen.append(limit)
            return rows[:limit]

        p = mock.patch.object(job_store, "list_job_summaries", summaries)
        p.start()
        self.addCleanup(p.stop)
        self.client = TestClient(main.app)

    def test_returns_all_lightweight_fields(self):
        r = self.client.get("/api/model/job-summaries")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body), 150)  # 不再只有最新 10 筆
        self.assertEqual(set(body[0]), {"job_id", "job_type", "status", "created_at"})
        self.assertEqual(self.seen, [1000])

    def test_limit_is_clamped(self):
        self.client.get("/api/model/job-summaries?limit=0")
        self.client.get("/api/model/job-summaries?limit=999999")
        self.assertEqual(self.seen, [1, 5000])

    def test_path_does_not_collide_with_job_detail(self):
        # /jobs/{job_id} 的路由不會把 "job-summaries" 當成任務 ID
        self.assertEqual(self.client.get("/api/model/job-summaries").status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
