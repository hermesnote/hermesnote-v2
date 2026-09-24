"""`training/artifacts.py` 的 SQL 契約（假連線擷取送出的 SQL 與參數，不連資料庫）：
best（weights）與 last（weights_last）兩組權重都寫入、欄位數與參數數一致、讀回時兩組都在。
`weights_last` 欄位由 migrations/2026-09-23_weights_last_and_evaluation.sql 新增，部署前必須先套用。

執行：python -m unittest tests.test_artifacts_sql
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import artifacts  # noqa: E402

SPEC = {
    "start": "2024-01-01", "end": "2024-02-01",
    "nodes": [
        {"id": "f", "type": "feature", "key": "raw_price", "timeframe": "tx_1m"},
        {"id": "l", "type": "label", "outcome": "simple_return", "labeling_rule": "identity", "timeframe": "tx_1m",
         "params": {"horizon": 1}},
        {"id": "m", "type": "model", "key": "lstm", "inputs": ["f"], "label": "l", "window": 10},
    ],
}
RESULTS = {"m": {"weights": b"best-bytes", "weights_last": b"last-bytes", "model_config": {"foo": float("nan")},
                 "task_type": "regression"}}


class _FakeCursor:
    def __init__(self, captured, row=None):
        self.captured, self.row = captured, row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.captured.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.row


class _FakeConn:
    def __init__(self, captured, row=None):
        self.autocommit, self.captured, self.row = False, captured, row

    def cursor(self):
        return _FakeCursor(self.captured, self.row)

    def close(self):
        pass


def _bytes(p):
    return bytes(p.adapted) if hasattr(p, "adapted") else p


class ArtifactsSqlTests(unittest.TestCase):
    def _with_conn(self, fn, row=None):
        captured = []
        orig = artifacts.psycopg2.connect
        artifacts.psycopg2.connect = lambda *a, **k: _FakeConn(captured, row)
        try:
            return fn(), captured
        finally:
            artifacts.psycopg2.connect = orig

    def test_save_writes_best_and_last_weights(self):
        _, captured = self._with_conn(lambda: artifacts.save_artifacts_for_job("job1", SPEC, RESULTS))
        self.assertEqual(len(captured), 1)
        sql, params = captured[0]
        self.assertIn("weights_last", sql)
        self.assertEqual(sql.count("%s"), len(params))
        self.assertEqual([_bytes(p) for p in params[4:6]], [b"best-bytes", b"last-bytes"])
        self.assertIn('"foo": null', params[6])  # model_config 的非有限值存成 null（JSONB 不接受 NaN）

    def test_load_returns_both_weights(self):
        row = ("lstm", "regression", b"best", b"last", {}, [], {}, {}, [], {})
        loaded, captured = self._with_conn(lambda: artifacts.load_artifact("job1", "m"), row=row)
        self.assertIn("weights_last", captured[0][0])
        self.assertEqual((loaded["weights"], loaded["weights_last"]), (b"best", b"last"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
