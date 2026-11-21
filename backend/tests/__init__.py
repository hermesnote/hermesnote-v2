"""單元測試一律不連正式 hermesnote DB（2026-11-18）。

worker 新增的單一實例鎖、原子認領、回收都會直接連資料庫；測試若漏掉替身，會打到 .env 指向的正式資料庫。
這裡把 HERMESNOTE_DATABASE_URL 指向不存在的位址：
  - 環境變數：之後才載入的 services.hermesnote_db 讀到的就是封鎖值（load_dotenv 不覆蓋既有環境變數）；
  - 已載入的 services.hermesnote_db：直接改模組上的值（get_conn／acquire_worker_lock 都是呼叫時才讀）。
每個 tests/test_*.py 都必須在載入 training／services／main 之前 `import tests`（tests/test_db_guard.py 會檢查），
所以 `python -m unittest discover`、`python -m unittest tests.xxx`、`python tests/xxx.py` 都受保護。
需要真正 PostgreSQL 的測試請用 TEST_PG_DSN（測試專用資料庫，見 tests/test_worker_claim.py）。
quotes 唯讀測試仍依既有規則以 RUN_DB_TESTS=1 開啟，不受影響。
"""

import os
import sys

BLOCKED_DSN = "postgresql://blocked-by-tests@127.0.0.1:9/blocked"
# 載入時就把位址存成模組變數的地方（之後才載入的模組會直接讀到封鎖值；已載入的在這裡改掉）。
# training/registry/features.py 是呼叫時才讀環境變數，不需要列。新增這類模組時要補進來（test_db_guard 會檢查）。
CAPTURED = {"services.hermesnote_db": ("HERMESNOTE_DATABASE_URL", BLOCKED_DSN),
            "training.artifacts": ("_SYNC_DATABASE_URL", BLOCKED_DSN),
            "training.inference_store": ("_SYNC_DATABASE_URL", BLOCKED_DSN),
            "training.preview_store": ("_SYNC_DATABASE_URL", BLOCKED_DSN),
            "training.progress": ("_SYNC_DATABASE_URL", BLOCKED_DSN)}

if os.environ.get("HERMESNOTE_TESTS_ALLOW_REAL_DB") != "1":
    os.environ["HERMESNOTE_DATABASE_URL"] = BLOCKED_DSN
    for _name, (_attr, _value) in CAPTURED.items():
        _mod = sys.modules.get(_name)
        if _mod is not None:
            setattr(_mod, _attr, _value)
