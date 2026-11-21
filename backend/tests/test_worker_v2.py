"""worker 的 v2 路徑（training/worker_v2.py）端到端：讀取（含往前載入參數）→ 框架核對／保存 → 訓練 → 產物 → 逐筆結果
（COPY 列格式）→ 串接折次 → 摘要；以及 v2 產物的 Phase 3（best／last 推論、評估、使用紀錄）。資料庫與行情以 stub 取代。"""

import asyncio
import os
import sys
import unittest
import warnings

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from tests.contract_harness import xgb_params  # noqa: E402
from tests.v2_synth import market  # noqa: E402
from training import experiment_store as es  # noqa: E402
from training import job_store, worker_v2  # noqa: E402
from training import artifacts as artifacts_mod  # noqa: E402
from training import inference_store  # noqa: E402
from training.v2_records import COLUMNS  # noqa: E402

MK = market(days=15)


def graph(strategy="forward_chaining"):
    return {"mode": "v2", "start": "2024-01-03", "end": "2024-01-15",
            "settings": {"split_seed": 42, "val_ratio": 0.2, "cascade": {"strategy": strategy, "k": 3}},
            "experiment": {"experiment_id": "00000000-0000-0000-0000-0000000000e1", "arm_id": None, "model_seed": 0},
            "frame_ref": {"frame_id": "00000000-0000-0000-0000-0000000000f1", "phase": 1, "version": 1},
            "nodes": [{"id": "f", "type": "feature", "key": "ohlcv", "timeframe": "tx_5m"},
                      {"id": "l", "type": "label", "outcome": "log_return", "labeling_rule": "fixed_threshold",
                       "timeframe": "tx_5m", "params": {"horizon": 1, "n_classes": 2}},
                      {"id": "up", "type": "model", "key": "xgboost", "inputs": ["f"], "label": "l", "window": 6,
                       "params": xgb_params(), "output_mode": "probability"},
                      {"id": "m", "type": "model", "key": "xgboost", "inputs": ["f", "up"], "label": "l", "window": 6,
                       "params": xgb_params()}]}


class Env:
    def __init__(self, test):
        self.calls = {"fetch": [], "records": [], "folds": [], "frames": [], "done": None, "reports": [], "evals": [],
                      "usage": [], "artifacts": None, "infer_rows": 0}
        self.stored_frame = {"status": "defined"}
        c = self.calls

        async def fetch(symbol, timeframe, start, end, **kw):
            c["fetch"].append((timeframe, kw))
            df = MK[f"{symbol}_{timeframe}"]
            s = pd.to_datetime(df["bar_start_ts"])
            return df[s < pd.Timestamp(end)].reset_index(drop=True)

        async def save_data_report(jid, r):
            c["reports"].append(r)

        async def mark_done(jid, r):
            c["done"] = r

        async def get_frame(fid):
            return self.stored_frame

        async def save_frame(fid, frame):
            c["frames"].append(frame.hash)

        async def write_records(it):
            rows = list(it)
            c["records"].extend(rows)
            return len(rows)

        async def write_folds(jid, nid, info, h):
            c["folds"].append((nid, len(info["folds"]), h))

        async def write_eval(*a):
            c["evals"].append(a)

        async def write_usage(kind, **kw):
            c["usage"].append(kind)

        def save_artifacts(jid, g, results):
            c["artifacts"] = results

        import services.quotes as q
        from training import preview_store, progress

        patches = [(q, "fetch_ohlcv_dataframe", fetch), (job_store, "save_data_report", save_data_report),
                   (job_store, "mark_done", mark_done), (es, "get_frame", get_frame), (es, "save_frame", save_frame),
                   (es, "write_prediction_records", write_records), (es, "write_upstream_folds", write_folds),
                   (es, "write_evaluation", write_eval), (es, "write_holdout_usage", write_usage),
                   (artifacts_mod, "save_artifacts_for_job", save_artifacts),
                   (progress, "write_progress", lambda *a, **k: None),
                   (preview_store, "save_preview_sample", lambda *a, **k: None)]
        for mod, name, fn in patches:
            orig = getattr(mod, name)
            setattr(mod, name, fn)
            test.addCleanup(setattr, mod, name, orig)


class WorkerV2Tests(unittest.TestCase):
    def test_train_job_end_to_end(self):
        env = Env(self)
        asyncio.run(worker_v2.run_train_job("00000000-0000-0000-0000-00000000000a", graph(), 1))
        c = env.calls
        self.assertTrue(all("lookback_bars" in kw and kw["with_session"] for _tf, kw in c["fetch"]))
        self.assertEqual(len(c["frames"]), 1)  # 第一次使用框架：保存
        self.assertTrue(c["records"])
        self.assertTrue(all(len(r) == len(COLUMNS) for r in c["records"]))
        datasets = {r[COLUMNS.index("dataset")] for r in c["records"]}
        self.assertEqual(datasets, {"val", "train"})  # 串接：下游訓練點的上游來源也寫入
        weights = {r[COLUMNS.index("weights")] for r in c["records"] if r[COLUMNS.index("dataset")] == "val"}
        self.assertTrue({"best", "last"} <= weights)
        self.assertEqual(c["folds"][0][0], "up")
        m = c["done"]["m"]
        self.assertEqual(m["training_meta"]["pipeline"], "v2")
        self.assertIn("train_stats", m)
        self.assertEqual(m["frame"]["hash"], c["frames"][0])
        self.assertIn("history_requirements", c["reports"][-1])

    def test_frame_hash_mismatch_fails(self):
        env = Env(self)
        env.stored_frame = {"status": "ready", "hash": "x" * 64}
        with self.assertRaises(ValueError) as cm:
            asyncio.run(worker_v2.run_train_job("00000000-0000-0000-0000-00000000000b", graph(), 1))
        self.assertIn("新版本框架", str(cm.exception))

    def test_phase3_v2_best_last_evaluated(self):
        env = Env(self)
        g = graph()
        asyncio.run(worker_v2.run_train_job("00000000-0000-0000-0000-00000000000c", g, 1))
        results = env.calls["artifacts"]
        arts = {nid: {"architecture_key": next(n for n in g["nodes"] if n["id"] == nid)["key"], "task_type": r["task_type"],
                      "weights": r["weights"], "weights_last": r.get("weights_last"), "model_config": r["model_config"],
                      "preprocessing_state": r["preprocessing_state"], "train_class_counts": r["train_stats"],
                      "graph_spec_snapshot": g} for nid, r in results.items()}

        async def chain(tj, tn):
            return g, arts

        class W:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def write_chunk(self, ts, preds, *a):
                env.calls["infer_rows"] += len(ts)

        orig_chain, orig_writer = worker_v2._artifact_chain, inference_store.open_prediction_writer
        worker_v2._artifact_chain = chain
        inference_store.open_prediction_writer = lambda jid, nid: W()
        self.addCleanup(setattr, worker_v2, "_artifact_chain", orig_chain)
        self.addCleanup(setattr, inference_store, "open_prediction_writer", orig_writer)
        env.calls["records"].clear()
        spec = {"target_job_id": "00000000-0000-0000-0000-00000000000c", "target_node_id": "m", "start": "2024-01-15",
                "end": "2024-01-22", "use_weights": "best"}
        asyncio.run(worker_v2._infer_v2("00000000-0000-0000-0000-00000000000d", spec, 3))
        done = env.calls["done"]["m"]
        self.assertEqual({e["point"]["kind"] for e in done["evaluations"]}, {"best", "last"})
        self.assertEqual(done["holdout"]["n_unmatured"], 1)
        self.assertGreater(env.calls["infer_rows"], 0)
        self.assertEqual(env.calls["usage"], ["evaluation"])
        self.assertEqual(len(env.calls["evals"]), 2)
        self.assertTrue(all(r[COLUMNS.index("dataset")] == "holdout" for r in env.calls["records"]))

        # 實驗層 Phase 3：第一次使用框架時保存；保存的雜湊不同就失敗、不寫結果
        frames_before = len(env.calls["frames"])
        env.stored_frame = {"status": "defined"}
        spec3 = {**spec, "frame_ref": {"frame_id": "00000000-0000-0000-0000-0000000000f3", "phase": 3, "version": 1}}
        asyncio.run(worker_v2._infer_v2("00000000-0000-0000-0000-00000000000e", spec3, 3))
        self.assertEqual(len(env.calls["frames"]), frames_before + 1)
        h3 = env.calls["frames"][-1]
        self.assertNotEqual(h3, env.calls["frames"][0])  # Phase 3 框架與 Phase 1 框架雜湊不同
        env.stored_frame = {"status": "ready", "hash": h3}
        asyncio.run(worker_v2._infer_v2("00000000-0000-0000-0000-00000000000f", spec3, 3))  # 相同 → 通過、不重寫
        self.assertEqual(len(env.calls["frames"]), frames_before + 1)
        env.stored_frame = {"status": "ready", "hash": "y" * 64}
        n_evals = len(env.calls["evals"])
        with self.assertRaises(ValueError):
            asyncio.run(worker_v2._infer_v2("00000000-0000-0000-0000-000000000010", spec3, 3))
        self.assertEqual(len(env.calls["evals"]), n_evals)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class LegacyPhase3Tests(unittest.TestCase):
    def test_legacy_phase3_evaluates_best_and_last(self):
        from tests import contract_harness as ch
        from training import inference as inference_mod
        from training import worker as worker_mod

        spec = {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
            ch._feat("fa"), ch._label("l1", "log_return", "fixed_threshold", n_classes=2),
            ch._model("solo", "xgboost", ["fa"], "l1", params=xgb_params(), output_mode="probability")]}
        arts = ch.build_artifacts(spec)
        for a in arts.values():
            a["weights_last"] = a["weights"]
        art = {**arts["solo"], "train_class_counts": {"task_type": "classification", "class_counts": [100, 120]}}
        env = Env(self)

        async def load_dfs(tj, tn, s, e):
            return {"tx_1m": ch._DF}

        orig = (worker_mod._load_dataframes_for_infer, worker_mod.run_infer_sync, inference_mod.load_artifact)
        worker_mod._load_dataframes_for_infer = load_dfs
        worker_mod.run_infer_sync = lambda *a, **k: {"count": 0, "use_weights": "best"}
        inference_mod.load_artifact = lambda jid, nid: arts.get(nid)
        self.addCleanup(lambda: (setattr(worker_mod, "_load_dataframes_for_infer", orig[0]),
                                 setattr(worker_mod, "run_infer_sync", orig[1]),
                                 setattr(inference_mod, "load_artifact", orig[2])))
        ispec = {"target_job_id": "00000000-0000-0000-0000-0000000000aa", "target_node_id": "solo",
                 "start": "2024-02-01", "end": "2024-03-01", "use_weights": "best"}
        asyncio.run(worker_v2._phase3_legacy("00000000-0000-0000-0000-0000000000ab", ispec, spec, art))
        done = env.calls["done"]["solo"]
        self.assertEqual({e["point"]["kind"] for e in done["evaluations"]}, {"best", "last"})
        # 舊推論路徑本來就不輸出最後一個窗格，horizon=1 時每個預測點都有答案
        self.assertEqual(done["holdout"]["n_unmatured"], 0)
        self.assertEqual(done["holdout"]["n_included"], done["holdout"]["n_predicted"])
        self.assertEqual(env.calls["usage"], ["evaluation"])
        rows = env.calls["records"]
        self.assertTrue(rows and all(len(r) == len(COLUMNS) for r in rows))
        maj = [b for b in done["evaluations"][0]["baselines"] if b["key"] == "majority_class"]
        self.assertTrue(maj and maj[0]["available"])
