"""資訊可用時間（available_ts）沿依賴鏈傳遞，以及缺 bar_end_ts 欄位資料表的讀取退路。

規則（training/time_semantics.py）：
  決策可用時間 = 全部輸入（Feature 與上游 Model）在樣本最後一根棒的可用時間的最晚者，任一未知就是未知；
  目標可用時間 = Label 節點自己資料在目標那根棒的 bar_end_ts（不借用 Feature 的時間）。

執行：python -m unittest tests.test_available_time
"""

import asyncio
import os
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from services import quotes  # noqa: E402
from tests import contract_harness as H  # noqa: E402
from tests.test_lstm_dual import KEY, params  # noqa: E402
from training import graph as graph_mod  # noqa: E402
from training import inference as inference_mod  # noqa: E402
from training.registry import architectures  # noqa: E402

HOUR = 3600


def df_with(hours):
    """hours=None 代表這份資料沒有 bar_end_ts 欄位。"""
    df = H.synth_df(320).copy()
    if hours is not None:
        df["bar_end_ts"] = df.index + pd.Timedelta(hours=hours)
    return df


def loader(mapping):
    return lambda tf: mapping[tf]


def node_feature(i, tf, key="raw_price"):
    return {"id": i, "type": "feature", "key": key, "timeframe": tf}


def node_label(i, tf):
    return {"id": i, "type": "label", "outcome": "log_return", "labeling_rule": "identity",
            "timeframe": tf, "params": {"horizon": 1}}


def install_probe_arch():
    """一個會呼叫 preview_hook 的假架構（spy 不會），下游節點用它才有預覽可看。"""
    def train(X_train, y_train, X_val, y_val, p, on_epoch, task_type="regression", preview_hook=None):
        if preview_hook is not None:
            preview_hook(0, 0, 0.0)
        return {"final_metrics": {"m": 1.0}, "predict": lambda X: np.zeros(len(X)), "device": "cpu",
                "weights": b"probe", "model_config": {"probe": True}}

    architectures.ARCHITECTURE_REGISTRY["probe"] = train
    architectures.LOAD_REGISTRY["probe"] = lambda w, c: {"predict": lambda X: np.zeros(len(X))}


def remove_probe_arch():
    architectures.ARCHITECTURE_REGISTRY.pop("probe", None)
    architectures.LOAD_REGISTRY.pop("probe", None)


def previews_of(spec, mapping, node_id):
    got = []
    graph_mod.run_graph(spec, loader(mapping), None, lambda nid, p: got.append(p) if nid == node_id else None)
    assert got, "沒有收到預覽"
    return got[0]


def probe_model(i, inputs, label, **extra):
    return H._model(i, "probe", inputs, label, **extra)


class TargetTimeComesFromLabelTests(unittest.TestCase):
    def setUp(self):
        install_probe_arch()
        self.addCleanup(remove_probe_arch)

    def spec(self):
        return {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
            node_feature("f", "tf_feat"), node_label("l", "tf_label"), probe_model("m", ["f"], "l")]}

    def test_feature_5h_label_7h_target_is_7h(self):
        # GC 重現案例：Feature 延後 5 小時、Label 延後 7 小時，目標可用時間必須是 7 小時，不是 5 小時
        p = previews_of(self.spec(), {"tf_feat": df_with(5), "tf_label": df_with(7)}, "m")
        self.assertEqual(p["decision_available_ts"] - p["decision_ts"], 5 * HOUR)
        self.assertEqual(p["target_available_ts"] - p["target_ts"], 7 * HOUR)

    def test_label_without_bar_end_target_unknown_decision_known(self):
        p = previews_of(self.spec(), {"tf_feat": df_with(5), "tf_label": df_with(None)}, "m")
        self.assertEqual(p["decision_available_ts"] - p["decision_ts"], 5 * HOUR)
        self.assertIsNone(p["target_available_ts"])  # 不用 Feature 的時間頂替

    def test_feature_without_bar_end_decision_unknown_target_known(self):
        p = previews_of(self.spec(), {"tf_feat": df_with(None), "tf_label": df_with(7)}, "m")
        self.assertIsNone(p["decision_available_ts"])
        self.assertEqual(p["target_available_ts"] - p["target_ts"], 7 * HOUR)

    def test_training_meta_reports_both_flags(self):
        r = graph_mod.run_graph(self.spec(), loader({"tf_feat": df_with(5), "tf_label": df_with(None)}))
        self.assertEqual(r["m"]["training_meta"]["time_semantics"]["available_ts"], True)
        self.assertEqual(r["m"]["training_meta"]["time_semantics"]["target_available_ts"], False)


class ChainPropagationTrainingTests(unittest.TestCase):
    def setUp(self):
        install_probe_arch()
        self.addCleanup(remove_probe_arch)

    def chain_spec(self, up_tf, direct_tf):
        return {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
            node_feature("fa", up_tf), node_feature("fv", direct_tf, "raw_volume"), node_label("l", "tf_label"),
            H._model("up", "spy", ["fa"], "l"),
            probe_model("down", ["up", "fv"], "l")]}

    def setUp2(self):
        architectures.ARCHITECTURE_REGISTRY["spy"] = H.spy_train

    def test_latest_of_upstream_chain_and_direct_feature(self):
        self.setUp2()
        # 上游經 Feature 延後 9 小時、下游直接 Feature 只延後 5 小時：決策可用時間取最晚的 9 小時
        p = previews_of(self.chain_spec("tf9", "tf5"), {"tf9": df_with(9), "tf5": df_with(5), "tf_label": df_with(7)}, "down")
        self.assertEqual(p["decision_available_ts"] - p["decision_ts"], 9 * HOUR)
        self.assertEqual(p["target_available_ts"] - p["target_ts"], 7 * HOUR)

    def test_direct_feature_later_than_upstream(self):
        self.setUp2()
        p = previews_of(self.chain_spec("tf5", "tf9"), {"tf5": df_with(5), "tf9": df_with(9), "tf_label": df_with(7)}, "down")
        self.assertEqual(p["decision_available_ts"] - p["decision_ts"], 9 * HOUR)

    def test_unknown_upstream_makes_downstream_unknown(self):
        self.setUp2()
        # 上游的 Feature 沒有 bar_end_ts → 上游未知 → 下游即使直接 Feature 已知也是未知（不猜）
        p = previews_of(self.chain_spec("tf_none", "tf5"), {"tf_none": df_with(None), "tf5": df_with(5), "tf_label": df_with(7)}, "down")
        self.assertIsNone(p["decision_available_ts"])
        self.assertEqual(p["target_available_ts"] - p["target_ts"], 7 * HOUR)

    def test_three_level_chain(self):
        self.setUp2()
        spec = {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
            node_feature("fa", "tf11"), node_feature("fv", "tf5", "raw_volume"), node_label("l", "tf_label"),
            H._model("a", "spy", ["fa"], "l"), H._model("b", "spy", ["a", "fv"], "l"), probe_model("c", ["b", "fv"], "l")]}
        p = previews_of(spec, {"tf11": df_with(11), "tf5": df_with(5), "tf_label": df_with(7)}, "c")
        self.assertEqual(p["decision_available_ts"] - p["decision_ts"], 11 * HOUR)  # 最上游的 11 小時一路傳到最下游


class ChainPropagationInferenceTests(unittest.TestCase):
    def setUp(self):
        install_probe_arch()
        architectures.ARCHITECTURE_REGISTRY["spy"] = H.spy_train
        architectures.LOAD_REGISTRY["spy"] = lambda w, c: {"predict": lambda X: np.asarray(X)[:, -1, 0] * 0.5}
        self.addCleanup(remove_probe_arch)
        self.addCleanup(lambda: architectures.LOAD_REGISTRY.pop("spy", None))

    def arts(self, spec, mapping):
        results = graph_mod.run_graph(spec, loader(mapping))
        nodes = {n["id"]: n for n in spec["nodes"]}
        return {nid: {"architecture_key": nodes[nid]["key"], "task_type": r["task_type"], "weights": r["weights"],
                      "model_config": r["model_config"], "feature_schema": [], "target_spec": {},
                      "graph_spec_snapshot": spec, "depends_on_node_ids": [], "preprocessing_state": r["preprocessing_state"]}
                for nid, r in results.items()}

    def infer_avail(self, arts, node_id, mapping):
        orig = inference_mod.load_artifact
        inference_mod.load_artifact = lambda job, nid: arts.get(nid)
        chunks = []
        try:
            inference_mod.run_inference_chunked("job", node_id, loader(mapping), lambda *a: chunks.append(a), chunk_size=10_000)
        finally:
            inference_mod.load_artifact = orig
        ts = [t for c in chunks for t in c[0]]
        avail = [c[5] for c in chunks]
        return ts, avail

    def spec(self, up_tf, direct_tf):
        return {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
            node_feature("fa", up_tf), node_feature("fv", direct_tf, "raw_volume"), node_label("l", "tf_label"),
            H._model("up", "spy", ["fa"], "l"), probe_model("down", ["up", "fv"], "l")]}

    def test_inference_takes_latest_over_whole_chain(self):
        mapping = {"tf9": df_with(9), "tf5": df_with(5), "tf_label": df_with(7)}
        spec = self.spec("tf9", "tf5")
        ts, avail = self.infer_avail(self.arts(spec, mapping), "down", mapping)
        flat = [a for chunk in avail for a in chunk]
        self.assertEqual({a - t for a, t in zip(flat, ts)}, {9 * HOUR})

    def test_inference_unknown_when_upstream_unknown(self):
        mapping = {"tf_none": df_with(None), "tf5": df_with(5), "tf_label": df_with(7)}
        spec = self.spec("tf_none", "tf5")
        _, avail = self.infer_avail(self.arts(spec, mapping), "down", mapping)
        self.assertEqual(avail, [None])

    def test_inference_feature_only_unchanged(self):
        mapping = {"tf5": df_with(5), "tf9": df_with(9), "tf_label": df_with(7)}
        spec = {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
            node_feature("fa", "tf5"), node_feature("fv", "tf9", "raw_volume"), node_label("l", "tf_label"),
            probe_model("m", ["fa", "fv"], "l")]}
        ts, avail = self.infer_avail(self.arts(spec, mapping), "m", mapping)
        self.assertEqual({a - t for a, t in zip([a for c in avail for a in c], ts)}, {9 * HOUR})


# ── 缺 bar_end_ts 欄位的資料表：明確的讀取退路 ─────────────────────────────────────────
class FakeConn:
    def __init__(self, has_bar_end):
        self.has_bar_end = has_bar_end
        self.fetch_sql = None
        self.fetchval_args = None
        self.closed = False

    async def fetchval(self, sql, *args):
        self.fetchval_args = args
        return self.has_bar_end

    async def fetch(self, sql, *args):
        self.fetch_sql = sql
        cols = ["datetime", "open", "high", "low", "close", "volume"] + (["bar_end_ts"] if self.has_bar_end else [])
        t = pd.Timestamp("2024-01-02 08:45")
        row = [t, 1.0, 2.0, 0.5, 1.5, 10] + ([t + pd.Timedelta(hours=5)] if self.has_bar_end else [])
        return [tuple(row)]

    async def close(self):
        self.closed = True


class MissingBarEndColumnTests(unittest.TestCase):
    def fetch(self, has_bar_end):
        conn = FakeConn(has_bar_end)

        async def fake_connect(url):
            return conn

        orig = quotes.asyncpg.connect
        quotes.asyncpg.connect = fake_connect
        try:
            df = asyncio.run(quotes.fetch_ohlcv_dataframe("tx", "daily_day", pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-02-01").date()))
        finally:
            quotes.asyncpg.connect = orig
        return conn, df

    def test_table_with_column_selects_it(self):
        conn, df = self.fetch(True)
        self.assertIn("bar_end_ts", conn.fetch_sql)
        self.assertIn("bar_end_ts", df.columns)
        self.assertEqual(conn.fetchval_args, ("market", "future_taifex_tx_daily_day"))  # schema／表名以參數綁定查詢

    def test_table_without_column_does_not_select_it_and_returns_six_columns(self):
        conn, df = self.fetch(False)
        self.assertNotIn("bar_end_ts", conn.fetch_sql)  # 不會 SQL 報錯
        self.assertEqual(list(df.columns), ["datetime", "open", "high", "low", "close", "volume"])
        self.assertTrue(conn.closed)

    def test_missing_column_means_unknown_available_time_end_to_end(self):
        _, df = self.fetch(False)
        from training.time_semantics import build_available_map
        self.assertIsNone(build_available_map(df.set_index("datetime")))

    def test_partial_null_column_is_unknown(self):
        from training.time_semantics import build_available_map
        d = df_with(5)
        d.loc[d.index[3], "bar_end_ts"] = pd.NaT
        self.assertIsNone(build_available_map(d))


@unittest.skipUnless(os.environ.get("RUN_DB_TESTS") == "1", "需要 RUN_DB_TESTS=1（唯讀查詢正式 quotes DB）")
class RealSchemaTests(unittest.TestCase):
    def test_only_spread_table_lacks_bar_end_ts_and_reader_works_on_real_tables(self):
        import asyncpg

        async def run():
            conn = await asyncpg.connect(quotes.QUOTES_DATABASE_URL)
            try:
                return [r["table_name"] for r in await conn.fetch(
                    "SELECT t.table_name FROM information_schema.tables t WHERE t.table_schema='market' "
                    "AND t.table_name LIKE 'future_taifex_%' AND NOT EXISTS (SELECT 1 FROM information_schema.columns c "
                    "WHERE c.table_schema=t.table_schema AND c.table_name=t.table_name AND c.column_name='bar_end_ts')")]
            finally:
                await conn.close()
        lacking = asyncio.run(run())
        # 缺欄位的只有 spread 系列（母表與它的年度分區），沒有任何 OHLCV 行情表缺這欄；
        # spread 沒有 open/high/low/close（欄位是 price），本來就不是 fetch_ohlcv_dataframe 的讀取對象
        self.assertIn("future_taifex_spread", lacking)
        self.assertTrue(all(t.startswith("future_taifex_spread") for t in lacking), lacking)
        df = asyncio.run(quotes.fetch_ohlcv_dataframe("tx", "daily_day", pd.Timestamp("2026-09-01").date(), pd.Timestamp("2026-09-19").date()))
        self.assertIn("bar_end_ts", df.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
