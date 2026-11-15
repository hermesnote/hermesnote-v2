"""M1：流程版本、環境指紋、研究期間提示、讀取報告彙整、跨時間框架暫時拒絕。"""

import os
import sys
import unittest
from datetime import date

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

import pandas as pd  # noqa: E402

from tests.contract_harness import DATA_GRAPHS, xgb_params  # noqa: E402
from training import provenance  # noqa: E402
from training.graph_refs import GraphValidationError, validate_graph_spec  # noqa: E402


def feat(i, tf, key="ohlcv"):
    return {"id": i, "type": "feature", "key": key, "timeframe": tf}


def label(i, tf):
    return {"id": i, "type": "label", "outcome": "log_return", "labeling_rule": "fixed_threshold", "timeframe": tf,
            "params": {"horizon": 1, "n_classes": 2}}


def model(i, inputs, lab):
    return {"id": i, "type": "model", "key": "xgboost", "inputs": inputs, "label": lab, "window": 10,
            "val_ratio": 0.2, "params": xgb_params()}


def spec(*nodes):
    return {"start": "2024-01-01", "end": "2024-02-01", "nodes": list(nodes)}


class PeriodWarningTests(unittest.TestCase):
    def test_research_period_has_no_warning(self):
        self.assertEqual(provenance.period_warnings("2015-01-01", "2026-01-01"), [])  # end 為不含，等於上限不提示

    def test_after_research_end_warns_but_does_not_raise(self):
        w = provenance.period_warnings(date(2025, 1, 1), "2026-03-01")
        self.assertEqual([x["code"] for x in w], ["period_after_research_end"])
        self.assertIn("2026-01-01", w[0]["message"])


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_records_versions(self):
        fp = provenance.env_fingerprint()
        for k in ("pipeline_version", "python", "numpy", "pandas", "talib", "talib_c", "xgboost"):
            self.assertIn(k, fp)
        self.assertEqual(fp["pipeline_version"], provenance.PIPELINE_VERSION)

    def test_data_report_collects_reader_reports(self):
        df = pd.DataFrame({"close": [1.0]})
        df.attrs["read_report"] = {"n_excluded": 5}
        r = provenance.data_report({"tx_1m": df, "tx_5m": pd.DataFrame()}, {"start": "2020-01-01", "end": "2026-02-01"})
        self.assertEqual(r["timeframes"], {"tx_1m": {"n_excluded": 5}, "tx_5m": None})
        self.assertEqual(r["warnings"][0]["code"], "period_after_research_end")
        self.assertEqual(r["available_time"], {"source": "bar_end_ts", "publication_delay_sec": 0, "delay_status": "assumed"})


class CrossTimeframeRejectionTests(unittest.TestCase):
    def assert_rejected(self, s, node):
        with self.assertRaises(GraphValidationError) as cm:
            validate_graph_spec(s)
        self.assertTrue(any(f"Model 節點 {node}" in i and "多個時間框架" in i for i in cm.exception.issues), cm.exception.issues)

    def test_single_timeframe_accepted(self):
        validate_graph_spec(spec(feat("f", "tx_15m"), label("l", "tx_15m"), model("m", ["f"], "l")))

    def test_feature_mix_rejected(self):
        self.assert_rejected(spec(feat("a", "tx_15m"), feat("b", "tx_60m"), label("l", "tx_15m"), model("m", ["a", "b"], "l")), "m")

    def test_label_on_other_timeframe_rejected(self):
        self.assert_rejected(spec(feat("a", "tx_15m"), label("l", "tx_daily_day"), model("m", ["a"], "l")), "m")

    def test_mix_through_upstream_model_rejected(self):
        s = spec(feat("a", "tx_60m"), label("l60", "tx_60m"), model("up", ["a"], "l60"),
                 feat("b", "tx_15m"), label("l15", "tx_15m"), model("down", ["b", "up"], "l15"))
        self.assert_rejected(s, "down")

    def test_contract_graphs_still_valid(self):
        for name, g in DATA_GRAPHS.items():
            with self.subTest(name=name):
                g = {**g, "nodes": [{**n, "key": "xgboost", "params": xgb_params()} if n["type"] == "model" else n
                                    for n in g["nodes"]]}
                validate_graph_spec(g)


if __name__ == "__main__":
    unittest.main(verbosity=2)
