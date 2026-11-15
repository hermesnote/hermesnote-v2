"""訓練讀取規則（流程版本 2026-11-r1）：排除 session='other'、依 bar_start_ts 排序並驗證唯一、讀取報告。

假連線測試檢查 SQL 與報錯；RUN_DB_TESTS=1 時另外對正式 quotes DB 做唯讀核對
（研究期間 5 個 other 日、30m／60m 日盤 K 棒保留且量與資料庫一致、各時間框架無重複）。
"""

import asyncio
import os
import sys
import unittest
from datetime import date

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

import pandas as pd  # noqa: E402

from services import quotes  # noqa: E402
from training.graph import _ensure_time_indexed  # noqa: E402

OTHER_DAYS = ["2012-03-03", "2012-12-22", "2014-12-27", "2016-01-30", "2016-09-10"]


class IntradayFakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.sqls = []
        self.closed = False

    async def fetch(self, sql, *args):
        self.sqls.append(sql)
        if "information_schema" in sql:
            return [{"column_name": c} for c in quotes.OPTIONAL_COLUMNS]
        if "session = 'other'" in sql:
            return [{"d": date(2016, 1, 30), "n": 1}]
        if "AS s," in sql:
            return [{"s": "day", "n": 2}]
        if "AS v," in sql:
            return [{"v": "taifex_front_outright_v1_2026-04-30", "n": 2}]
        return self.rows

    async def fetchrow(self, sql, *args):
        self.sqls.append(sql)
        return {"n_gaps": 1, "n_missing": 3.0}

    async def close(self):
        self.closed = True


def bar(t, start=None):
    t = pd.Timestamp(t)
    s = pd.Timestamp(start) if start else t
    return (t, 1.0, 2.0, 0.5, 1.5, 10, s + pd.Timedelta(minutes=30), s)


def fetch_with(rows, timeframe="30m"):
    conn = IntradayFakeConn(rows)

    async def fake_connect(url):
        return conn

    orig = quotes.asyncpg.connect
    quotes.asyncpg.connect = fake_connect
    try:
        df = asyncio.run(quotes.fetch_ohlcv_dataframe("tx", timeframe, date(2016, 1, 1), date(2016, 2, 1)))
    finally:
        quotes.asyncpg.connect = orig
    return conn, df


class ReaderRuleTests(unittest.TestCase):
    def test_excludes_other_and_orders_by_bar_start(self):
        conn, df = fetch_with([bar("2016-01-30 13:00"), bar("2016-01-30 13:30:02", "2016-01-30 13:30")])
        main = next(s for s in conn.sqls if "ORDER BY" in s and "count(" not in s)
        self.assertIn("session IS DISTINCT FROM 'other'", main)
        self.assertIn("ORDER BY bar_start_ts ASC, built_at ASC", main)
        self.assertIn("bar_start_ts", df.columns)
        self.assertTrue(conn.closed)

    def test_read_report(self):
        _, df = fetch_with([bar("2016-01-30 13:00"), bar("2016-01-30 13:30")])
        r = df.attrs["read_report"]
        self.assertEqual(r["n_excluded"], 1)
        self.assertEqual(r["excluded_trading_days"], ["2016-01-30"])
        self.assertEqual(r["excluded_sessions"], ["other"])
        self.assertEqual(r["order_key"], "bar_start_ts")
        self.assertEqual((r["n_gaps"], r["n_missing_bars"]), (1, 3))
        self.assertIn("taifex_front_outright_v1_2026-04-30", r["rule_versions"])

    def test_daily_table_has_no_gap_stats(self):
        conn, df = fetch_with([bar("2016-01-29 08:45")], timeframe="daily_day")
        self.assertNotIn("n_gaps", df.attrs["read_report"])
        self.assertFalse(any("lag(" in s for s in conn.sqls))

    def test_duplicate_bar_start_raises_with_details(self):
        rows = [bar("2016-01-30 13:30"), bar("2016-01-30 13:30:05", "2016-01-30 13:30")]
        with self.assertRaises(quotes.QuoteIntegrityError) as cm:
            fetch_with(rows)
        self.assertIn("bar_start_ts=2016-01-30 13:30:00 出現 2 次", str(cm.exception))

    def test_duplicate_datetime_raises(self):
        rows = [bar("2016-01-30 13:30", "2016-01-30 13:00"), bar("2016-01-30 13:30")]
        with self.assertRaises(quotes.QuoteIntegrityError) as cm:
            fetch_with(rows)
        self.assertIn("datetime=2016-01-30 13:30:00 出現 2 次", str(cm.exception))

    def test_graph_refuses_duplicate_index_instead_of_keeping_last(self):
        idx = pd.DatetimeIndex(["2024-01-01 09:00", "2024-01-01 09:00", "2024-01-01 09:01"])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
        with self.assertRaises(quotes.QuoteIntegrityError):
            _ensure_time_indexed(df)

    def test_graph_still_sorts_unique_index(self):
        idx = pd.DatetimeIndex(["2024-01-01 09:01", "2024-01-01 09:00"])
        df = pd.DataFrame({"close": [2.0, 1.0]}, index=idx)
        self.assertEqual(list(_ensure_time_indexed(df)["close"]), [1.0, 2.0])


@unittest.skipUnless(os.environ.get("RUN_DB_TESTS") == "1", "需要 RUN_DB_TESTS=1（唯讀查詢正式 quotes DB）")
class ResearchPeriodDbTests(unittest.TestCase):
    """研究期間（2026-01-01 以前）的唯讀核對。"""

    START, END = date(2011, 1, 1), date(2026, 1, 1)

    def read(self, tf, start=None, end=None):
        return asyncio.run(quotes.fetch_ohlcv_dataframe("tx", tf, start or self.START, end or self.END))

    def test_each_intraday_timeframe_excludes_five_other_rows_and_has_no_duplicates(self):
        for tf in ["1m", "5m", "15m", "30m", "60m"]:
            with self.subTest(tf=tf):
                df = self.read(tf)  # 有重複會直接丟 QuoteIntegrityError
                r = df.attrs["read_report"]
                self.assertEqual(r["n_excluded"], 5)
                self.assertEqual(r["excluded_trading_days"], OTHER_DAYS)
                self.assertNotIn("other", r["sessions"])
                self.assertEqual(r["order_key"], "bar_start_ts")

    def test_daily_tables_unaffected(self):
        for tf in ["daily_day", "daily_full"]:
            with self.subTest(tf=tf):
                r = self.read(tf).attrs["read_report"]
                self.assertEqual(r["n_excluded"], 0)

    def test_day_bars_on_other_days_kept_with_db_volume(self):
        import asyncpg

        async def db_volumes(tf):
            conn = await asyncpg.connect(quotes.QUOTES_DATABASE_URL)
            try:
                return {(str(r["trading_day"]), r["bar_start_ts"].strftime("%H:%M")): float(r["volume"]) for r in await conn.fetch(
                    f"SELECT trading_day, bar_start_ts, volume FROM market.future_taifex_tx_{tf} "
                    "WHERE session = 'day' AND trading_day = ANY($1::date[]) AND bar_start_ts::time = $2::time",
                    [date.fromisoformat(d) for d in OTHER_DAYS], pd.Timestamp("13:30" if tf == "30m" else "13:00").time())}
            finally:
                await conn.close()

        for tf in ["30m", "60m"]:
            with self.subTest(tf=tf):
                expected = asyncio.run(db_volumes(tf))
                self.assertEqual(len(expected), 5)
                for d in OTHER_DAYS:
                    df = self.read(tf, date.fromisoformat(d), date.fromisoformat(d) + pd.Timedelta(days=1))
                    hm = "13:30" if tf == "30m" else "13:00"
                    row = df[pd.to_datetime(df["bar_start_ts"]).dt.strftime("%H:%M") == hm]
                    self.assertEqual(len(row), 1, (tf, d))
                    self.assertEqual(float(row["volume"].iloc[0]), expected[(d, hm)])
                    self.assertGreater(float(row["volume"].iloc[0]), 1000)  # 真正的日盤 K 棒，不是 1～5 口那根
                    self.assertLess(pd.to_datetime(df["bar_start_ts"]).max().strftime("%H:%M"), "13:45")


if __name__ == "__main__":
    unittest.main(verbosity=2)
