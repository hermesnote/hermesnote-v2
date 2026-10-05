"""後台 Dashboard／RAG／MCP 整合 API：管理員限定、固定路徑代理、token 不外流、失聯時回報原因。

執行：python -m unittest tests.test_admin_api
"""

import json
import os
import unittest
from unittest import mock
import requests
from fastapi.testclient import TestClient

import main
from auth import get_current_user
from routers import admin as admin_router
from services import agent_services as svc

SECRET = "test-secret-token-should-never-leak"
SID = "3e49d22d-1056-5a47-a765-20661427d737"


class FakeResp:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return self._body


def envelope(data, next_cursor=None):
    return {"ok": True, "data": data, "error": None, "observed_at": "2026-10-05T00:00:00+00:00",
            "provenance": [], "next_cursor": next_cursor}


class FakeUpstream:
    """模擬 RAG（:30450）與 MCP（:30451）；記錄每次呼叫的 URL 與是否帶了正確 token。"""

    def __init__(self):
        self.calls = []
        self.down = set()

    def __call__(self, method, url, json=None, headers=None, timeout=None):
        self.calls.append((method, url, json, dict(headers or {})))
        service = "rag" if ":30450" in url else "mcp"
        if service in self.down:
            raise requests.ConnectionError("down")
        path = url.split(":3045", 1)[1][1:]
        if path == "/health":
            return FakeResp(200, {"service": service, "state": "running"})
        if (headers or {}).get("Authorization") != "Bearer " + SECRET:
            return FakeResp(401, {"error": "unauthorized"})
        if path == "/ready":
            return FakeResp(200, {"ready": True})
        if service == "rag":
            return self.rag(method, path, json)
        return self.mcp(method, path, json)

    def rag(self, method, path, body):
        if path.startswith("/api/research/v1/knowledge-bases"):
            return FakeResp(200, envelope([{"id": "research", "label": "Research", "state": "ready"}]))
        if path.startswith("/api/research/v1/summary"):
            return FakeResp(200, envelope({"documents": [{"source_type": "pdf", "status": "indexed", "count": 31}],
                                           "chunks": 1944}))
        if path.startswith("/api/research/v1/sources/") and "/content" in path:
            return FakeResp(200, envelope({"source_id": SID, "version": "v1", "chunks": [
                {"chunk_index": 3, "content": "x", "locator": {"page": 3}, "quality_warnings": [{"reason": "NUL"}]}]}))
        if path.startswith("/api/research/v1/sources"):
            if "status=needs_review" in path:
                return FakeResp(200, envelope([{"source_id": SID, "version": "v1", "title": "2021_Multi",
                                                "source_type": "pdf", "status": "needs_review",
                                                "metadata": {"warnings": [{"reason": "NUL", "locator": {"page": 3}},
                                                                          {"reason": "NUL", "locator": {"page": 5}}]},
                                                "updated_at": "2026-10-05T05:15:53+00:00"}]))
            return FakeResp(200, envelope([], next_cursor=None))
        if path.startswith("/api/research/v1/ingestion-jobs"):
            return FakeResp(200, envelope([{"job_id": "j1", "source_id": SID, "stage": "complete", "status": "done",
                                            "started_at": "t", "finished_at": "t"}]))
        if path == "/api/research/v1/search":
            return FakeResp(200, envelope([{"source_id": SID, "version": "v1", "chunk_index": 3, "content": body["query"],
                                            "locator": {"page": 3}, "score": 0.1, "quality_warnings": []}]))
        return FakeResp(404, {"detail": "Not Found"})

    def mcp(self, method, path, body):
        if path == "/api/mcp/v1/tools":
            return FakeResp(200, {"service": "mcp", "tools": [{"name": n, "inputSchema": {}} for n in
                                                              ["knowledge_bases", "search_evidence",
                                                               "source_content", "indexing_status"]]})
        if path == "/mcp/":
            m = body["method"]
            if m == "initialize":
                result = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "MCP"}}
            elif m == "notifications/initialized":
                return FakeResp(202, {})
            elif m == "tools/list":
                result = {"tools": [{"name": "knowledge_bases"}, {"name": "search_evidence"}]}
            elif m == "tools/call":
                # 以 text content 承載 JSON（沒有 structuredContent）
                result = {"content": [{"type": "text", "text": json.dumps(
                    envelope([{"id": "research", "state": "ready", "sources_scanned": 60}]))}], "isError": False}
            return FakeResp(200, {"jsonrpc": "2.0", "id": body.get("id"), "result": result})
        return FakeResp(404, {})




async def _model_jobs(limit):
    return [
        {"job_id": "m-run", "graph_spec": {"nodes": [{"type": "model", "key": "lstm", "params": {"epochs": 300}},
                                                     {"type": "feature", "timeframe": "tx_5m"}]},
         "status": "running", "error": None, "result": None, "job_type": "train", "phase": 1,
         "created_at": "c", "started_at": "s", "finished_at": None},
        {"job_id": "m-fail", "graph_spec": {"nodes": []}, "status": "failed", "error": "boom", "result": None,
         "job_type": "train", "phase": None, "created_at": "c", "started_at": "s", "finished_at": "f"},
        {"job_id": "m-done", "graph_spec": {"nodes": []}, "status": "done", "error": None, "result": {"m1": {}},
         "job_type": "train", "phase": 1, "created_at": "c", "started_at": "s", "finished_at": "f"},
    ]


async def _progress(job_id):
    return [{"epoch": 0}, {"epoch": 41}]


async def _backtests(limit):
    return [{"job_id": "b1", "symbol": "TX", "timeframe": "15m", "start_date": "2026-01-01",
             "end_date": "2026-02-01", "status": "pending", "error": None, "summary": None,
             "created_at": "c", "finished_at": None}]


ADMIN_ROUTES = [
    ("GET", "/api/admin/overview"), ("GET", "/api/admin/services"),
    ("GET", "/api/admin/rag/knowledge-bases"), ("GET", "/api/admin/rag/summary"),
    ("GET", "/api/admin/rag/sources"), ("GET", f"/api/admin/rag/sources/{SID}/content"),
    ("GET", "/api/admin/rag/ingestion-jobs"), ("POST", "/api/admin/rag/search"),
    ("GET", "/api/admin/mcp/status"), ("GET", "/api/admin/mcp/tools"), ("POST", "/api/admin/mcp/test"),
]


class AdminApiTest(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream()
        patches = [
            mock.patch.object(svc.requests, "request", self.upstream),
            mock.patch.dict(os.environ, {"RAG_BASE_URL": "http://192.168.0.44:30450",
                                         "MCP_BASE_URL": "http://192.168.0.44:30451",
                                         "RAG_SERVICE_TOKEN": SECRET, "MCP_SERVICE_TOKEN": SECRET}),
            mock.patch.object(svc, "SECRETS_FILE", "/nonexistent/settings.json"),
            mock.patch.object(admin_router.training_store, "list_jobs", _model_jobs),
            mock.patch.object(admin_router.training_store, "get_progress", _progress),
            mock.patch.object(admin_router.backtest_store, "list_jobs", _backtests),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        main.app.dependency_overrides[get_current_user] = lambda: {"email": "admin@example.com", "picture": ""}
        self.addCleanup(main.app.dependency_overrides.clear)
        self.client = TestClient(main.app)

    def test_anonymous_rejected(self):
        main.app.dependency_overrides.clear()
        anon = TestClient(main.app)
        for method, path in ADMIN_ROUTES:
            with self.subTest(path=path):
                self.assertEqual(anon.request(method, path, json={"query": "x"}).status_code, 401)
        self.assertEqual(self.upstream.calls, [], "未登入不得觸發任何上游呼叫")

    def test_non_admin_token_rejected(self):
        from jose import jwt
        import auth
        main.app.dependency_overrides.clear()
        with mock.patch.object(auth, "JWT_SECRET", "s"), mock.patch.object(auth, "ADMIN_EMAILS", {"admin@example.com"}):
            token = jwt.encode({"sub": "someone@example.com"}, "s", algorithm="HS256")
            resp = TestClient(main.app).get("/api/admin/overview", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(self.upstream.calls, [])

    def test_overview_counts_and_providers(self):
        data = self.client.get("/api/admin/overview").json()
        self.assertEqual(data["counts"], {"running": 1, "pending": 1, "attention": 2, "service_issues": 0})
        providers = {p["id"]: p for p in data["providers"]}
        self.assertEqual(providers["hermes_agent"]["state"], "not_connected")
        self.assertEqual(providers["hermes_agent"]["items"], [])
        self.assertEqual(providers["agent_sdk"]["state"], "not_connected")
        running = next(i for i in providers["model_training"]["items"] if i["id"] == "m-run")
        self.assertEqual(running["progress"], {"current": 42, "total": 300, "unit": "輪"})
        self.assertIn("tx_5m", running["title"])
        self.assertIn("Phase 1", running["title"])
        done = next(i for i in providers["model_training"]["items"] if i["id"] == "m-done")
        self.assertEqual(done["delivery"]["state"], "available")
        review = providers["rag"]["items"][0]
        self.assertEqual(review["status"], "needs_review")
        self.assertIn("3、5", review["error"])
        self.assertEqual({s["state"] for s in data["services"]}, {"ready"})
        self.assertNotIn(SECRET, json.dumps(data))

    def test_overview_service_down_keeps_other_providers(self):
        self.upstream.down.add("rag")
        data = self.client.get("/api/admin/overview").json()
        providers = {p["id"]: p for p in data["providers"]}
        self.assertEqual((providers["rag"]["state"], providers["rag"]["error"]), ("error", "無法連線"))
        self.assertEqual(providers["model_training"]["state"], "connected")
        self.assertNotIn("rag", data["counted_providers"])
        self.assertEqual(data["unavailable_providers"][0]["id"], "rag")
        rag = next(s for s in data["services"] if s["id"] == "rag")
        self.assertEqual((rag["state"], rag["alive"]["error"]), ("down", "無法連線"))
        self.assertEqual(data["counts"]["service_issues"], 1)

    def test_rag_proxy_fixed_paths_and_token(self):
        self.assertEqual(self.client.get("/api/admin/rag/summary").json()["data"]["chunks"], 1944)
        self.client.get("/api/admin/rag/sources", params={"offset": 20, "limit": 20, "source_type": "pdf"})
        content = self.client.get(f"/api/admin/rag/sources/{SID}/content", params={"chunk_index": 3}).json()
        self.assertEqual(content["data"]["chunks"][0]["locator"], {"page": 3})
        hit = self.client.post("/api/admin/rag/search", json={"query": "注意力", "limit": 5}).json()["data"][0]
        self.assertEqual((hit["source_id"], hit["chunk_index"]), (SID, 3))
        urls = [c[1] for c in self.upstream.calls]
        self.assertTrue(all(u.startswith("http://192.168.0.44:30450/api/research/v1/") for u in urls))
        self.assertTrue(any("offset=20" in u and "source_type=pdf" in u for u in urls))
        self.assertTrue(all(c[3]["Authorization"] == "Bearer " + SECRET for c in self.upstream.calls))
        self.assertEqual(self.upstream.calls[-1][2], {"query": "注意力", "limit": 5, "filters": {}})

    def test_rag_rejects_bad_source_id_without_calling(self):
        self.assertIn(self.client.get("/api/admin/rag/sources/..%2F..%2Fready/content").status_code, (404, 422))
        self.assertEqual(self.client.get("/api/admin/rag/sources/not-a-uuid/content").status_code, 422)
        self.assertEqual(self.upstream.calls, [])

    def test_rag_upstream_auth_failure_is_reported(self):
        with mock.patch.dict(os.environ, {"RAG_SERVICE_TOKEN": "wrong-token-value"}):
            resp = self.client.get("/api/admin/rag/summary")
        self.assertEqual(resp.status_code, 502)
        self.assertEqual(resp.json()["detail"]["reason"], "服務拒絕後端憑證（401）")
        self.assertNotIn("wrong-token-value", resp.text)

    def test_missing_credential(self):
        os.environ.pop("RAG_SERVICE_TOKEN")
        resp = self.client.get("/api/admin/rag/summary")
        self.assertEqual(resp.status_code, 502)
        self.assertEqual(resp.json()["detail"]["reason"], "後端未設定服務憑證")
        rag = next(s for s in self.client.get("/api/admin/services").json() if s["id"] == "rag")
        self.assertEqual((rag["state"], rag["credential_configured"]), ("unconfigured", False))

    def test_secrets_file_fallback(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "settings.json")
            with open(f, "w", encoding="utf-8") as fh:
                json.dump({"rag_url": "http://192.168.0.44:30450", "rag_token": SECRET, "mcp_token": SECRET}, fh)
            os.environ.pop("RAG_SERVICE_TOKEN")
            os.environ.pop("RAG_BASE_URL")
            with mock.patch.object(svc, "SECRETS_FILE", f):
                cfg = svc.service_config("rag")
        self.assertEqual((cfg["token_source"], cfg["base_url"]), ("secrets_file", "http://192.168.0.44:30450"))

    def test_mcp_status_tools_and_test(self):
        status = self.client.get("/api/admin/mcp/status").json()
        self.assertEqual((status["service"]["state"], status["upstream_rag"]["state"]), ("ready", "ready"))
        self.assertEqual(status["protocol_endpoint"], "http://192.168.0.44:30451/mcp/")
        self.assertEqual({a["state"] for a in status["agent_connections"]}, {"not_connected"})
        tools = self.client.get("/api/admin/mcp/tools").json()["tools"]
        self.assertEqual(len(tools), 4)
        test = self.client.post("/api/admin/mcp/test").json()
        self.assertTrue(test["ok"])
        self.assertEqual([s["step"] for s in test["steps"]], ["initialize", "tools/list", "tools/call knowledge_bases"])
        self.assertEqual(test["steps"][2]["detail"]["carrier"], "text")
        self.assertEqual(test["steps"][2]["detail"]["knowledge_bases"][0]["sources_scanned"], 60)
        self.assertNotIn(SECRET, json.dumps([status, tools, test]))

    def test_mcp_test_reports_failed_step(self):
        self.upstream.down.add("mcp")
        test = self.client.post("/api/admin/mcp/test").json()
        self.assertFalse(test["ok"])
        self.assertEqual(test["steps"][0]["error"], "無法連線")
        self.assertEqual(len(test["steps"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
