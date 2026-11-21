"""測試防護：任何測試都連不到正式 hermesnote DB（2026-11-18）。

  1. 靜態：每個 tests/test_*.py 都在第一次載入 training／services／main／auth／routers 之前 `import tests`
     （或 `from tests ...`），所以直接執行單一檔案也會套用 tests/__init__.py 的封鎖。
  2. 執行期：環境變數與已載入的 services.hermesnote_db 都是封鎖值；實際呼叫 get_conn 與 acquire_worker_lock
     會連線失敗（不會連到 .env 的正式位址）。

執行：python -m unittest tests.test_db_guard，或 python tests/test_db_guard.py
"""

import asyncio
import glob
import os
import re
import sys
import unittest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

import tests  # noqa: E402,F401 — 正式 hermesnote DB 防護（tests/__init__.py），必須在 training／services 之前

APP_IMPORT = re.compile(r"^\s*(from|import)\s+(training|services|main|auth|routers)\b")
GUARD_IMPORT = re.compile(r"^\s*(import tests\b|from tests(\.| )import|from tests\.)")


class DbGuardTests(unittest.TestCase):
    def test_every_test_module_imports_guard_first(self):
        bad = []
        for path in sorted(glob.glob(os.path.join(BACKEND_DIR, "tests", "test_*.py"))):
            guard_line = app_line = None
            for i, line in enumerate(open(path, encoding="utf-8"), 1):
                if guard_line is None and GUARD_IMPORT.match(line):
                    guard_line = i
                if app_line is None and APP_IMPORT.match(line):
                    app_line = i
            if app_line is not None and (guard_line is None or guard_line > app_line):
                bad.append(f"{os.path.basename(path)}（防護 {guard_line}、程式 {app_line}）")
        self.assertEqual(bad, [], "這些測試檔沒有在載入程式模組前 import tests")

    def test_runtime_dsn_blocked(self):
        if os.environ.get("HERMESNOTE_TESTS_ALLOW_REAL_DB") == "1":
            self.skipTest("已明確允許連正式資料庫")
        from services import hermesnote_db

        self.assertEqual(os.environ["HERMESNOTE_DATABASE_URL"], tests.BLOCKED_DSN)
        self.assertEqual(hermesnote_db.HERMESNOTE_DATABASE_URL, tests.BLOCKED_DSN)

    def test_all_captured_urls_blocked(self):
        """所有在載入時讀 HERMESNOTE_DATABASE_URL 的模組都列在 tests.CAPTURED，而且目前都是封鎖值。"""
        if os.environ.get("HERMESNOTE_TESTS_ALLOW_REAL_DB") == "1":
            self.skipTest("已明確允許連正式資料庫")
        import importlib

        readers = set()
        for path in glob.glob(os.path.join(BACKEND_DIR, "**", "*.py"), recursive=True):
            rel = os.path.relpath(path, BACKEND_DIR)
            if rel.startswith((".venv", "tests", "migrations")) or os.sep + "site-packages" + os.sep in path:
                continue
            for line in open(path, encoding="utf-8", errors="ignore"):
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=.*HERMESNOTE_DATABASE_URL", line):
                    readers.add(rel[:-3].replace(os.sep, "."))
        self.assertTrue(readers <= set(tests.CAPTURED), f"未列入 tests.CAPTURED：{sorted(readers - set(tests.CAPTURED))}")
        for name, (attr, value) in tests.CAPTURED.items():
            mod = importlib.import_module(name)
            self.assertTrue(str(getattr(mod, attr)).endswith("/blocked"), name)

    def test_real_connection_attempts_fail(self):
        if os.environ.get("HERMESNOTE_TESTS_ALLOW_REAL_DB") == "1":
            self.skipTest("已明確允許連正式資料庫")
        from services import hermesnote_db
        from training import job_store

        if os.environ.get("TEST_PG_DSN"):
            self.skipTest("設定了 TEST_PG_DSN 時，其他測試會暫時把連線換成測試資料庫；這裡只在未設定時檢查")
        for fn in (hermesnote_db.get_conn, job_store.acquire_worker_lock):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises((OSError, ConnectionError, asyncio.TimeoutError)):
                    asyncio.run(asyncio.wait_for(fn(), 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
