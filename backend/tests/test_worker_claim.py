"""訓練 worker 的排他、認領、取消、回收與失鎖後的寫入防護（2026-11-18）。

兩組測試：
  WorkerLoopTests      以替身取代資料庫：取得鎖→註冊中斷通知→等待接手緩衝→回收→才認領；認領失敗不執行；
                       失敗走有條件的寫入；watchdog 失鎖即結束；鎖查詢逾時視為失鎖。
  PostgresWorkerTests  需要真正的 PostgreSQL（環境變數 TEST_PG_DSN：只接受本機、名稱 hn_test_ 開頭的測試資料庫，
                       以 model_init.sql 建立完整資料表；未設定時跳過）。驗證：認領只在持有鎖時成功、每筆只認領一次、
                       取消、回收、只有自己能寫入、資源紀錄、鎖排他與斷線釋放、中斷通知、
                       **失鎖後舊 worker 的每一種寫入都被拒絕**、寫入進行中時回收會等待。

執行：python -m unittest tests.test_worker_claim
"""

import asyncio
import json
import os
import sys
import unittest
import uuid
from unittest import mock

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

import tests  # noqa: E402,F401 — 正式 hermesnote DB 防護（tests/__init__.py），必須在 training／services 之前
from training import fence, job_store, worker  # noqa: E402


class _Stop(Exception):
    pass


class FakeLock:
    def __init__(self, calls, claimed=None, alive=True):
        self.calls, self.claimed, self._alive = calls, claimed, alive
        self.conn = mock.MagicMock()

    async def alive(self):
        return self._alive

    async def reclaim_orphans(self, inst):
        self.calls.append(("reclaim", inst))
        return [{"id": "j-old", "instance": "dead"}]

    async def claim_next_pending(self, device, fp):
        self.calls.append(("claim", fp["worker"]["instance"]))
        return self.claimed


class WorkerLoopTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.addCleanup(setattr, fence, "INSTANCE", None)
        self.addCleanup(setattr, fence, "JOB", None)
        self.addCleanup(setattr, job_store, "RESOURCE_PROBE", None)

    def patch(self, obj, name, fn):
        p = mock.patch.object(obj, name, fn)
        p.start()
        self.addCleanup(p.stop)

    def test_lock_listener_grace_then_reclaim(self):
        lock = FakeLock(self.calls)
        attempts = iter([None, lock])

        async def acquire():
            self.calls.append("acquire")
            return next(attempts)

        async def sleep(sec):
            self.calls.append(("sleep", sec))

        self.patch(job_store, "acquire_worker_lock", acquire)
        self.patch(worker.asyncio, "sleep", sleep)
        ident = worker.worker_identity()
        self.assertIs(asyncio.run(worker.acquire_lock_and_reclaim(ident)), lock)
        self.assertEqual(self.calls, ["acquire", ("sleep", worker.LOCK_RETRY_SEC), "acquire",
                                      ("sleep", worker.TAKEOVER_GRACE_SEC), ("reclaim", ident["instance"])])
        lock.conn.add_termination_listener.assert_called_once()

    def run_loop(self, claimed, run_raises=None, alive=True):
        fetches = iter([{"id": "j1"}, _Stop])
        lock = FakeLock(self.calls, claimed, alive)

        async def fetch():
            v = next(fetches)
            if v is _Stop:
                raise _Stop
            return v

        async def run_one(jid, spec, phase):
            self.calls.append(("run", jid, fence.JOB, fence.INSTANCE))
            if run_raises:
                raise run_raises

        async def failed(jid, err):
            self.calls.append(("failed", jid, err))
            return True

        async def acquire_and_reclaim(ident):
            self.calls.append("lock")
            return lock

        async def watchdog(lk):
            await asyncio.Event().wait()

        def exit_(reason):
            self.calls.append(("exit", reason))
            raise _Stop

        self.patch(worker, "acquire_lock_and_reclaim", acquire_and_reclaim)
        self.patch(worker, "_lock_watchdog", watchdog)
        self.patch(worker, "_exit_lock_lost", exit_)
        self.patch(job_store, "fetch_next_pending", fetch)
        self.patch(job_store, "mark_failed_by_worker", failed)
        self.patch(worker, "run_one", run_one)
        self.patch(worker, "gpu_capacity", lambda: (True, "ok"))
        self.patch(worker, "read_peak_rss", lambda: 123)
        self.patch(worker, "reset_peak_rss", lambda: True)
        with self.assertRaises(_Stop):
            asyncio.run(worker.main_loop())

    def test_claim_lost_does_not_run(self):
        self.run_loop(claimed=None)
        self.assertEqual([c[0] if isinstance(c, tuple) else c for c in self.calls], ["lock", "claim"])

    def test_lock_lost_before_claim_exits_without_claim(self):
        self.run_loop(claimed={"id": "j1", "graph_spec": {}, "job_type": "train", "phase": None}, alive=False)
        self.assertEqual([c[0] if isinstance(c, tuple) else c for c in self.calls], ["lock", "exit"])

    def test_claimed_job_runs_with_fence_set(self):
        self.run_loop(claimed={"id": "j1", "graph_spec": {}, "job_type": "train", "phase": None})
        run = next(c for c in self.calls if c[0] == "run")
        claim = next(c for c in self.calls if c[0] == "claim")
        self.assertEqual(run[2], "j1")              # 執行期間 fence.JOB＝這筆
        self.assertEqual(run[3], claim[1])          # fence.INSTANCE＝寫進 env_fingerprint 的實例
        self.assertIsNone(fence.JOB)                # 結束後清空
        self.assertEqual(job_store.RESOURCE_PROBE()["peak_rss_bytes"], 123)

    def test_failure_written_through_guarded_path(self):
        self.run_loop(claimed={"id": "j1", "graph_spec": {}, "job_type": "train", "phase": None},
                      run_raises=ValueError("boom"))
        self.assertIn(("failed", "j1", "boom"), self.calls)

    def test_watchdog_exits_when_lock_lost(self):
        async def sleep(_):
            return None

        self.patch(worker.asyncio, "sleep", sleep)
        self.patch(worker, "_exit_lock_lost", lambda reason: (_ for _ in ()).throw(_Stop(reason)))
        with self.assertRaises(_Stop):
            asyncio.run(worker._lock_watchdog(FakeLock([], alive=False)))

    def test_lock_query_timeout_means_lost(self):
        class Hanging:
            def is_closed(self):
                return False

            async def fetchval(self, *a, **k):
                await asyncio.Event().wait()

        with mock.patch.object(job_store, "LOCK_QUERY_TIMEOUT_SEC", 0.5):
            self.assertFalse(asyncio.run(job_store.WorkerLock(Hanging()).alive()))

    def test_owned_clause_follows_fence_instance(self):
        fence.INSTANCE = None
        self.assertEqual(job_store._owned_clause(3), ("", []))
        fence.INSTANCE = "abc"
        clause, extra = job_store._owned_clause(3)
        self.assertIn("$3", clause)
        self.assertEqual(extra, ["abc"])

    def test_peak_rss_helpers_never_raise(self):
        worker.reset_peak_rss()
        v = worker.read_peak_rss()
        self.assertTrue(v is None or v > 0)


# ── 真正的 PostgreSQL ──────────────────────────────────────────────────────
DSN = os.getenv("TEST_PG_DSN")
SCHEMA = os.path.join(BACKEND_DIR, "model_init.sql")
TABLES = ["model_upstream_folds", "model_holdout_usage", "model_evaluations", "model_prediction_records",
          "model_sample_frame_points", "model_sample_frames", "model_experiment_arms", "model_experiments",
          "model_inference_predictions", "model_artifacts", "model_training_preview_samples", "model_training_progress",
          "model_training_jobs"]


def _check_test_dsn(dsn: str) -> None:
    """測試會建立／清空資料表：只允許本機、名稱 hn_test_ 開頭的資料庫，指錯就直接拒絕。"""
    from urllib.parse import urlsplit

    u = urlsplit(dsn)
    host, db = (u.hostname or ""), u.path.lstrip("/")
    if host not in ("127.0.0.1", "localhost", "::1") or not db.startswith("hn_test_"):
        raise RuntimeError(f"TEST_PG_DSN 必須指向本機、名稱以 hn_test_ 開頭的測試資料庫（收到 host={host!r} db={db!r}），拒絕執行")


GRAPH = {"nodes": [{"id": "f", "type": "feature", "key": "ohlcv", "timeframe": "tx_5m"},
                   {"id": "l", "type": "label", "outcome": "log_return", "labeling_rule": "fixed_threshold",
                    "timeframe": "tx_5m", "params": {}},
                   {"id": "m", "type": "model", "key": "xgboost", "inputs": ["f"], "label": "l"}]}
PREVIEW = {"epoch": 0, "source": "train", "decision_ts": 1700000000, "target_ts": 1700000300, "horizon": 1,
           "task_type": "classification", "n_classes": 2, "labeling_rule": "fixed_threshold", "bars": [],
           "actual": 1, "predicted": 0, "decision_available_ts": 1700000300, "target_available_ts": 1700000600}


class _Frame:
    """樣本框架替身：0 個時間點（只測所有權檢查與交易）。"""
    ts = tau = target_ts = target_avail = status = reason = []
    definition, hash = {}, "h"

    def __len__(self):
        return 0

    def counts(self):
        return {}


@unittest.skipUnless(DSN, "未設定 TEST_PG_DSN（測試用 PostgreSQL），跳過資料庫層測試")
class PostgresWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _check_test_dsn(DSN)
        import asyncpg

        async def init():
            c = await asyncpg.connect(DSN)
            await c.execute("DROP TABLE IF EXISTS " + ", ".join(TABLES) + " CASCADE")
            await c.execute(open(SCHEMA, encoding="utf-8").read())
            await c.close()

        asyncio.run(init())

    def setUp(self):
        import asyncpg

        self.asyncpg = asyncpg
        from services import hermesnote_db
        from training import artifacts, inference_store, preview_store, progress

        patches = [mock.patch.object(hermesnote_db, "HERMESNOTE_DATABASE_URL", DSN)]
        for m in (artifacts, inference_store, preview_store, progress):
            patches.append(mock.patch.object(m, "_SYNC_DATABASE_URL", DSN))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(setattr, fence, "INSTANCE", None)
        self.addCleanup(setattr, fence, "JOB", None)
        self.addCleanup(setattr, job_store, "RESOURCE_PROBE", None)
        self.q("TRUNCATE " + ", ".join(TABLES) + " CASCADE")

    # ── 工具 ──
    def q(self, sql, *args):
        async def go():
            c = await self.asyncpg.connect(DSN)
            try:
                return await c.fetch(sql, *args)
            finally:
                await c.close()
        return asyncio.run(go())

    def add(self, status="pending", instance=None, n=1):
        ids = []
        for _ in range(n):
            i = uuid.uuid4()
            fp = json.dumps({"worker": {"instance": instance}}) if instance else None
            self.q("INSERT INTO model_training_jobs (id, graph_spec, status, env_fingerprint) "
                   "VALUES ($1, '{}'::jsonb, $2, $3::jsonb)", i, status, fp)
            ids.append(str(i))
        return ids

    def row(self, jid):
        return self.q("SELECT status, error, data_report, env_fingerprint->'worker'->>'instance' AS inst "
                      "FROM model_training_jobs WHERE id = $1", uuid.UUID(jid))[0]

    def counts(self, jid):
        u = uuid.UUID(jid)
        out = {}
        for t in ("model_training_progress", "model_training_preview_samples", "model_artifacts",
                  "model_inference_predictions", "model_evaluations", "model_holdout_usage"):
            out[t] = self.q(f"SELECT count(*) AS n FROM {t} WHERE job_id = $1", u)[0]["n"]
        return out

    def all_writes(self, jid):
        """worker 的每一種寫入各做一次；回傳 {寫入: 'ok' 或例外類別名稱}。"""
        from training import artifacts, inference_store, preview_store, progress
        from training import experiment_store as es

        res = {}

        def attempt(name, fn):
            try:
                out = fn()
                if asyncio.iscoroutine(out):
                    out = asyncio.run(out)
                res[name] = "ok" if out is not False else "rejected(False)"
            except fence.LostOwnership:
                res[name] = "LostOwnership"
            except Exception as e:  # noqa: BLE001
                res[name] = type(e).__name__

        attempt("save_data_report", lambda: job_store.save_data_report(jid, {"x": 1}))
        attempt("write_progress", lambda: progress.write_progress(
            jid, "m", 0, {"loss": 1.0, "accuracy": 0.5, "val_loss": 1.0, "val_accuracy": 0.5}))
        attempt("preview_sample", lambda: preview_store._write_sample_sync(jid, "m", PREVIEW))
        attempt("save_artifacts", lambda: artifacts.save_artifacts_for_job(
            jid, GRAPH, {"m": {"weights": b"w", "model_config": {}, "task_type": "classification"}}))

        def infer():
            with inference_store.open_prediction_writer(jid, "m") as w:
                w.write_chunk([1700000000], [[0.4, 0.6]], "classification", True)
        attempt("inference_predictions", infer)
        attempt("save_frame", lambda: es.save_frame(str(uuid.uuid4()), _Frame()))
        attempt("prediction_records", lambda: es.write_prediction_records(iter([])))
        attempt("evaluation", lambda: es.write_evaluation(jid, "m", 3, "holdout", "best", [], {"n_included": 0}))
        attempt("holdout_usage", lambda: es.write_holdout_usage("evaluation", job_id=jid))
        attempt("upstream_folds", lambda: es.write_upstream_folds(
            jid, "m", {"folds": [], "consumers": [], "policy": {}, "purge_sec": 0}, "h"))
        attempt("mark_done", lambda: job_store.mark_done(jid, {"m": {}}))
        return res

    # ── 認領 ──
    def test_claim_requires_lock_and_each_job_once(self):
        ids = self.add(n=20)

        async def go():
            lock = await job_store.acquire_worker_lock()
            self.assertIsNone(await job_store.acquire_worker_lock())  # 第二個 worker 取不到鎖

            async def claimer():
                got = []
                while (j := await lock.claim_next_pending("cpu", {"worker": {"instance": "A"}})) is not None:
                    got.append(j["id"])
                return got

            results = await asyncio.gather(*[claimer() for _ in range(5)])
            await lock.close()
            return [j for r in results for j in r]

        flat = asyncio.run(go())
        self.assertEqual(sorted(flat), sorted(ids))
        self.assertEqual(len(flat), len(set(flat)))

    def test_claim_refused_after_lock_released(self):
        (p,) = self.add()

        async def go():
            lock = await job_store.acquire_worker_lock()
            await lock.conn.execute("SELECT pg_advisory_unlock($1)", job_store.WORKER_LOCK_KEY)  # 連線還在、鎖已不在
            self.assertFalse(await lock.alive())
            got = await lock.claim_next_pending("cpu", {"worker": {"instance": "A"}})
            await lock.close()
            return got

        self.assertIsNone(asyncio.run(go()))
        self.assertEqual(self.row(p)["status"], "pending")

    # ── 取消、回收、只有自己能寫入 ──
    def test_cancel_pending_and_running(self):
        (p,) = self.add()
        asyncio.run(job_store.mark_failed(p, "使用者手動中斷"))
        (r,) = self.add(status="running", instance="A")
        fence.INSTANCE = "A"
        fence.JOB = r
        asyncio.run(job_store.mark_failed(r, "使用者手動中斷"))  # 執行中被取消（web 端，不經 fence）
        res = self.all_writes(r)
        self.assertTrue(all(v in ("LostOwnership", "rejected(False)") for v in res.values()), res)
        self.assertEqual(self.row(r)["error"], "使用者手動中斷")
        self.assertEqual(sum(self.counts(r).values()), 0)

        async def claim():
            lock = await job_store.acquire_worker_lock()
            j = await lock.claim_next_pending("cpu", {"worker": {"instance": "A"}})
            await lock.close()
            return j
        self.assertIsNone(asyncio.run(claim()))  # 已取消的 pending 不會被認領

    def test_owner_writes_succeed_others_rejected(self):
        (r,) = self.add(status="running", instance="A")
        fence.INSTANCE, fence.JOB = "B", r
        res = self.all_writes(r)
        self.assertTrue(all(v in ("LostOwnership", "rejected(False)") for v in res.values()), res)
        self.assertEqual(self.row(r)["status"], "running")
        fence.INSTANCE, fence.JOB = "A", r
        job_store.RESOURCE_PROBE = lambda: {"peak_rss_bytes": 7}
        res = self.all_writes(r)
        self.assertEqual(set(res.values()), {"ok"}, res)
        st = self.row(r)
        self.assertEqual(st["status"], "done")
        self.assertEqual(json.loads(st["data_report"]), {"x": 1, "resources": {"peak_rss_bytes": 7}})
        c = self.counts(r)
        self.assertEqual(c["model_training_progress"], 1)
        self.assertEqual(c["model_artifacts"], 1)
        self.assertEqual(c["model_inference_predictions"], 1)

    def test_reclaim_only_identified_other_instances_once(self):
        a, = self.add(status="running", instance="dead-a")
        mine, = self.add(status="running", instance="me")
        legacy, = self.add(status="running")          # 沒有 worker 身分的舊任務：不動
        done, = self.add(status="done", instance="dead-a")

        async def go():
            lock = await job_store.acquire_worker_lock()
            first = await lock.reclaim_orphans("me")
            second = await lock.reclaim_orphans("me")
            await lock.close()
            return first, second

        first, second = asyncio.run(go())
        self.assertEqual([g["id"] for g in first], [a])
        self.assertEqual(second, [])
        self.assertTrue(self.row(a)["error"].startswith(job_store.INTERRUPTED_ERROR))
        self.assertEqual([self.row(j)["status"] for j in (mine, legacy, done)], ["running", "running", "done"])

    # ── 鎖、中斷通知、失鎖後的舊 worker ──
    def test_split_brain_old_worker_cannot_write_or_claim(self):
        j1, j2 = self.add(n=2)
        events = []

        async def phase1():
            lock_a = await job_store.acquire_worker_lock()
            lock_a.conn.add_termination_listener(lambda c: events.append("A 收到中斷通知"))
            ja = await lock_a.claim_next_pending("cuda", {"worker": {"instance": "A"}})
            admin = await self.asyncpg.connect(DSN)
            await admin.execute("SELECT pg_terminate_backend($1)", lock_a.conn.get_server_pid())  # 伺服器端切斷 A 的鎖
            await admin.close()
            for _ in range(50):
                if events:
                    break
                await asyncio.sleep(0.1)
            alive = await lock_a.alive()
            lock_b = await job_store.acquire_worker_lock()       # B 立即取得鎖
            reclaimed = await lock_b.reclaim_orphans("B")
            jb = await lock_b.claim_next_pending("cuda", {"worker": {"instance": "B"}})
            try:
                again = await lock_a.claim_next_pending("cuda", {"worker": {"instance": "A"}})
            except Exception as e:  # noqa: BLE001
                again = type(e).__name__
            return lock_b, ja, alive, reclaimed, jb, again

        lock_b, ja, alive, reclaimed, jb, again = asyncio.run(phase1())
        self.assertEqual(events, ["A 收到中斷通知"])
        self.assertFalse(alive)
        self.assertEqual([r["id"] for r in reclaimed], [ja["id"]])
        self.assertNotEqual(again if isinstance(again, str) else None, None)  # A 無法再認領（連線已斷）
        # A 尚未結束，對 J1 做每一種寫入：全部被拒絕，資料表沒有任何 A 寫入的列
        fence.INSTANCE, fence.JOB = "A", ja["id"]
        res = self.all_writes(ja["id"])
        self.assertTrue(all(v in ("LostOwnership", "rejected(False)") for v in res.values()), res)
        self.assertEqual(sum(self.counts(ja["id"]).values()), 0)
        st = self.row(ja["id"])
        self.assertEqual(st["status"], "failed")
        self.assertTrue(st["error"].startswith(job_store.INTERRUPTED_ERROR))
        # B 的寫入照常
        fence.INSTANCE, fence.JOB = "B", jb["id"]
        res_b = self.all_writes(jb["id"])
        self.assertEqual(set(res_b.values()), {"ok"}, res_b)
        # 連線屬於上面那個已結束的 event loop，改由伺服器端終止
        self.q("SELECT pg_terminate_backend($1)", lock_b.conn.get_server_pid())

    def test_in_flight_write_blocks_reclaim_until_commit(self):
        import psycopg2

        (r,) = self.add(status="running", instance="A")
        fence.INSTANCE = "A"
        conn = psycopg2.connect(DSN)
        cur = conn.cursor()
        fence.check_sync(cur, r)  # A 的寫入交易開始（FOR SHARE）

        async def reclaim():
            lock = await job_store.acquire_worker_lock()
            try:
                return await asyncio.wait_for(lock.reclaim_orphans("B"), 2)
            except asyncio.TimeoutError:
                return "blocked"
            finally:
                await lock.close()

        self.assertEqual(asyncio.run(reclaim()), "blocked")  # 回收等待 A 的交易
        cur.execute("INSERT INTO model_training_progress (job_id, epoch) VALUES (%s, 0)", (r,))
        conn.commit()                                          # A 在仍擁有時提交
        conn.close()

        async def reclaim2():
            lock = await job_store.acquire_worker_lock()
            out = await lock.reclaim_orphans("B")
            await lock.close()
            return out
        self.assertEqual([x["id"] for x in asyncio.run(reclaim2())], [r])
        self.assertEqual(self.counts(r)["model_training_progress"], 1)
        with self.assertRaises(fence.LostOwnership):            # 回收後 A 的下一筆寫入被拒絕
            from training import progress

            progress.write_progress(r, "m", 1, {"loss": 1.0})

    def test_real_process_exits_immediately_when_lock_connection_killed(self):
        """真實程序：worker 的鎖與中斷通知（acquire_lock_and_reclaim）＋ watchdog，伺服器端切斷鎖連線後立即以 70 結束。"""
        import subprocess
        import time

        script = (
            "import asyncio, os, sys\n"
            f"sys.path.insert(0, {BACKEND_DIR!r})\n"
            "import tests\n"
            "from services import hermesnote_db\n"
            f"hermesnote_db.HERMESNOTE_DATABASE_URL = {DSN!r}\n"
            "from training import worker\n"
            "worker.TAKEOVER_GRACE_SEC = 0\n"
            "async def main():\n"
            "    lock = await worker.acquire_lock_and_reclaim(worker.worker_identity())\n"
            "    print('PID', lock.conn.get_server_pid(), flush=True)\n"
            "    asyncio.create_task(worker._lock_watchdog(lock))\n"
            "    await asyncio.sleep(120)\n"
            "asyncio.run(main())\n")
        p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        pid = None
        for line in p.stdout:
            if line.startswith("PID"):
                pid = int(line.split()[1])
                break
        self.assertIsNotNone(pid)
        t0 = time.time()
        self.q("SELECT pg_terminate_backend($1)", pid)
        code = p.wait(timeout=30)
        self.assertEqual(code, worker.LOCK_LOST_EXIT_CODE)
        self.assertLess(time.time() - t0, 5)  # 中斷通知觸發，不必等 watchdog
        # 鎖已釋放，新 worker 可取得
        async def go():
            lk = await job_store.acquire_worker_lock()
            ok = lk is not None
            if lk:
                await lk.close()
            return ok
        self.assertTrue(asyncio.run(go()))

    def test_lock_exclusive_and_released_on_disconnect(self):
        async def go():
            first = await job_store.acquire_worker_lock()
            self.assertTrue(await first.alive())
            self.assertIsNone(await job_store.acquire_worker_lock())
            await first.close()
            self.assertFalse(await first.alive())
            second = await job_store.acquire_worker_lock()
            self.assertIsNotNone(second)
            await second.close()

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main(verbosity=2)
