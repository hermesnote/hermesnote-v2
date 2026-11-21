"""模型 API 寫入權限（auth.require_model_writer）：/api/model/* 的每個寫入端點都要求管理員 JWT 或服務金鑰；
失敗在建立任何資料前回 401／403；GET 不受影響；requested_by／確認人取自驗證身分。

執行：python -m unittest tests.test_model_auth
"""

import hashlib
import json
import os
import re
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from fastapi.testclient import TestClient
from jose import jwt

import tests  # noqa: E402,F401 — 正式 hermesnote DB 防護（tests/__init__.py），必須在 training／services 之前
import auth
import main
from training import job_store

SECRET = "unit-test-jwt-secret"
ADMIN = "admin@example.com"
KEY_TRAIN = "hnk_test_train_only_0123456789"
KEY_ALL = "hnk_test_all_scopes_0123456789"
KEYS = json.dumps([
    {"name": "mcp-test", "sha256": hashlib.sha256(KEY_TRAIN.encode()).hexdigest(), "scopes": ["model:train"]},
    {"name": "ops-test", "sha256": hashlib.sha256(KEY_ALL.encode()).hexdigest(), "scopes": sorted(auth.MODEL_SCOPES)},
])
GRAPH = {"start": "2024-01-02", "end": "2024-01-06", "nodes": [
    {"id": "f", "type": "feature", "key": "ohlcv", "timeframe": "tx_5m"},
    {"id": "l", "type": "label", "outcome": "log_return", "labeling_rule": "fixed_threshold", "timeframe": "tx_5m",
     "params": {"horizon": 1, "n_classes": 2}},
    {"id": "m", "type": "model", "key": "xgboost", "inputs": ["f"], "label": "l", "window": 6,
     "params": {"n_estimators": 2, "max_depth": 1, "learning_rate": 0.1, "early_stopping": {"patience": 0}, "seed": 0}}]}


def token(email=ADMIN, secret=SECRET, hours=1):
    exp = datetime.now(timezone.utc) + timedelta(hours=hours)
    return jwt.encode({"sub": email, "picture": "", "exp": exp}, secret, algorithm="HS256")


def write_routes():
    """兩個模型 router 上所有非 GET 路由（新增端點會自動納入檢查）。"""
    from routers import model_experiments, model_training

    out = []
    for r in model_training.router.routes + model_experiments.router.routes:
        path = getattr(r, "path", "")
        methods = getattr(r, "methods", None) or set()
        if path.startswith("/api/model") and methods - {"GET", "HEAD"}:
            for m in sorted(methods - {"GET", "HEAD"}):
                out.append((m, re.sub(r"\{[^}]+\}", "00000000-0000-0000-0000-000000000001", path)))
    return out


class ModelAuthTests(unittest.TestCase):
    def setUp(self):
        self.created = []

        async def create_job(spec, **kw):
            self.created.append((spec, kw))
            return "00000000-0000-0000-0000-0000000000aa"

        async def list_jobs(limit=20):
            return []

        patches = [mock.patch.object(auth, "JWT_SECRET", SECRET), mock.patch.object(auth, "ADMIN_EMAILS", {ADMIN}),
                   mock.patch.dict(os.environ, {"MODEL_API_KEYS": KEYS}),
                   mock.patch.object(job_store, "create_job", create_job),
                   mock.patch.object(job_store, "list_jobs", list_jobs)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app)

    def post_train(self, headers):
        return self.client.post("/api/model/train", json=GRAPH, headers=headers)

    def test_every_write_route_requires_credentials(self):
        routes = write_routes()
        # 舊入口 5 個＋新入口 7 個
        self.assertEqual(len(routes), 12, routes)
        for method, path in routes:
            for headers in ({}, {"X-Hermesnote-Key": "hnk_wrong"}, {"Authorization": "Bearer " + token(secret="forged")},
                            {"Authorization": "Bearer not-a-jwt"}):
                r = self.client.request(method, path, json={}, headers=headers)
                self.assertEqual(r.status_code, 401, (method, path, headers.keys(), r.text))
        self.assertEqual(self.created, [])

    def test_non_whitelisted_admin_forbidden(self):
        r = self.post_train({"Authorization": "Bearer " + token(email="someone@example.com")})
        self.assertEqual(r.status_code, 403)
        r = self.post_train({"Authorization": "Bearer " + token(hours=-1)})  # 過期
        self.assertEqual(r.status_code, 401)
        self.assertEqual(self.created, [])

    def test_empty_jwt_secret_disables_admin(self):
        with mock.patch.object(auth, "JWT_SECRET", ""):
            r = self.post_train({"Authorization": "Bearer " + token(secret="")})
        self.assertEqual(r.status_code, 401)
        self.assertEqual(self.created, [])

    def test_admin_jwt_passes_and_records_principal(self):
        r = self.post_train({"Authorization": "Bearer " + token()})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.created[0][1]["requested_by"], f"admin:{ADMIN}")

    def test_service_key_scope(self):
        r = self.post_train({"X-Hermesnote-Key": KEY_TRAIN})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.created[-1][1]["requested_by"], "key:mcp-test")
        n = len(self.created)
        for method, path in [("POST", "/api/model/infer"), ("POST", "/api/model/experiments"),
                             ("DELETE", "/api/model/jobs/00000000-0000-0000-0000-000000000001"),
                             ("POST", "/api/model/jobs/00000000-0000-0000-0000-000000000001/cancel"),
                             ("POST", "/api/model/registry/v2/feature_memory")]:
            r = self.client.request(method, path, json={}, headers={"X-Hermesnote-Key": KEY_TRAIN})
            self.assertEqual(r.status_code, 403, (method, path, r.text))
        self.assertEqual(len(self.created), n)
        # 權限範圍足夠時通過驗證，交給端點本身的檢查（這裡是純計算端點）
        r = self.client.post("/api/model/registry/v2/feature_memory", json={"key": "rsi", "params": {"period": 14}},
                             headers={"X-Hermesnote-Key": KEY_ALL})
        self.assertEqual(r.status_code, 200, r.text)

    def test_key_wins_over_bearer_and_bad_key_not_rescued(self):
        # 帶了錯的金鑰，即使同時帶有效 JWT 也拒絕（不會退回另一種身分）
        r = self.post_train({"X-Hermesnote-Key": "hnk_wrong", "Authorization": "Bearer " + token()})
        self.assertEqual(r.status_code, 401)

    def test_reads_stay_open(self):
        self.assertEqual(self.client.get("/api/model/registry/v2").status_code, 200)
        self.assertEqual(self.client.get("/api/model/jobs").status_code, 200)

    def test_malformed_key_config_is_ignored(self):
        bad = json.dumps([{"name": "x", "sha256": "short", "scopes": ["model:train"]},
                          {"name": "y", "sha256": hashlib.sha256(b"k").hexdigest(), "scopes": ["model:train", "bogus"]}])
        with mock.patch.dict(os.environ, {"MODEL_API_KEYS": bad}):
            keys = auth._service_keys()
            self.assertEqual([k["name"] for k in keys], ["y"])
            self.assertEqual(keys[0]["scopes"], ["model:train"])
        with mock.patch.dict(os.environ, {"MODEL_API_KEYS": "not json"}):
            self.assertEqual(auth._service_keys(), [])
            self.assertEqual(self.post_train({"X-Hermesnote-Key": KEY_TRAIN}).status_code, 401)


class GateConfirmedByTests(unittest.TestCase):
    def test_principal_id_handles_direct_calls(self):
        self.assertIsNone(auth.principal_id(object()))
        self.assertEqual(auth.principal_id({"id": "key:x"}), "key:x")
        with self.assertRaises(ValueError):
            auth.require_model_writer("model:unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
