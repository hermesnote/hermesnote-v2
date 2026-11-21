"""worker 寫入的所有權檢查（fencing，2026-11-18）。

問題：worker 失去單一實例鎖（資料庫連線被終止、網路中斷）到它自己結束之間，新 worker 可能已取得鎖、回收舊任務並開始
新任務。這段期間舊 worker 的任何寫入都不能生效。

做法：worker 程序的每一筆寫入，都在同一個交易內先執行
    SELECT 1 FROM model_training_jobs
    WHERE id = <這筆寫入的任務> AND status = 'running' AND env_fingerprint->'worker'->>'instance' = <本實例>
    FOR SHARE
查不到就拋出 LostOwnership、整個交易不寫入。FOR SHARE 讓回收（UPDATE 同一列）必須等這些交易結束：
寫入要嘛在仍擁有任務時完整提交，要嘛在回收之後被拒絕，不會有中間狀態。使用者手動取消（status 改成 failed）
也一樣會讓之後的寫入被拒絕，訓練在下一次寫入（例如下一個 epoch 的進度）時中止。

只在 worker 程序生效：INSTANCE 由 worker 啟動時設定；web 後端、測試、命令列工具沒有設定 → 不檢查（行為不變）。
"""

INSTANCE: str | None = None   # 本 worker 實例 ID（worker.main_loop 設定）
JOB: str | None = None        # 目前執行中的任務（沒有 job_id 參數的寫入，例如樣本框架，用這個）

_SQL = ("SELECT 1 FROM model_training_jobs WHERE id = {p1}::uuid AND status = 'running' "
        "AND (env_fingerprint->'worker'->>'instance') = {p2} FOR SHARE")
SQL_PSYCOPG = _SQL.format(p1="%s", p2="%s")
SQL_ASYNCPG = _SQL.format(p1="$1", p2="$2")


class LostOwnership(RuntimeError):
    """這個 worker 已不擁有該任務（已被回收、取消或改由其他實例執行），寫入被拒絕。"""


def enabled() -> bool:
    return INSTANCE is not None


def _job(job_id) -> str:
    jid = str(job_id) if job_id is not None else JOB
    if jid is None:
        raise LostOwnership("寫入沒有對應的任務（worker 目前沒有執行中的任務）")
    return jid


def check_sync(cur, job_id=None) -> None:
    """psycopg2：必須在寫入的同一個交易內（autocommit 關閉）呼叫。"""
    if not enabled():
        return
    jid = _job(job_id)
    cur.execute(SQL_PSYCOPG, (jid, INSTANCE))
    if cur.fetchone() is None:
        raise LostOwnership(f"任務 {jid} 已不屬於 worker 實例 {INSTANCE}（已被回收或取消），寫入被拒絕")


async def check_async(conn, job_id=None) -> None:
    """asyncpg：必須在 `async with conn.transaction():` 內呼叫。"""
    if not enabled():
        return
    jid = _job(job_id)
    if await conn.fetchval(SQL_ASYNCPG, jid, INSTANCE) is None:
        raise LostOwnership(f"任務 {jid} 已不屬於 worker 實例 {INSTANCE}（已被回收或取消），寫入被拒絕")
