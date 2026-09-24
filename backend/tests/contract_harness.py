"""平台契約回歸測試的量測工具：在「改動前」擷取基準，改動後再比對。

刻意只用公開介面（`run_graph`、`ARCHITECTURE_REGISTRY`／`load_model`、`run_inference_chunked`、
`train_model`），同一支程式可以在改動前跑 `capture`、改動後跑 `compare`。其餘測試也共用這裡的
合成資料、節點建構與 payload 範本（`lstm_params`／`xgb_params`）。

三類量測：
  A 資料契約（逐值相同）：傳進 architecture 的 X/y 陣列、切分索引——用 spy architecture 擷取。
  B 載入一致（逐值相同）：本機合成產物走完整推論路徑的結果。
  C 重新訓練：固定 seed 後的逐輪指標，量「跑跑之間」的波動，改動後差異必須落在波動內。

用法：
    python tests/contract_harness.py capture <baseline.json>
    python tests/contract_harness.py compare <baseline.json>
"""

import hashlib
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 訓練測試固定用 CPU，避免 GPU 非決定性

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from training import graph as graph_mod  # noqa: E402
from training import inference as inference_mod  # noqa: E402
from training.registry import architectures  # noqa: E402

def lstm_params(**over) -> dict:
    """LSTM 單頭的最小合法 payload（研究值必填，這裡只是測試用的小數字）。"""
    p = {"lstm_layers": [{"units": 8, "dropout": 0.0}], "heads": "single",
         "shared": {"dense": 8, "dropout": 0.0}, "optimizer": {"type": "adam", "lr": 3e-3, "weight_decay": 0.0}, "batch_size": 32,
         "epochs": 3, "early_stopping": {"monitor": "val_loss", "patience": 0}, "seed": 0}
    p.update(over)
    return p


def xgb_params(**over) -> dict:
    p = {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.3, "early_stopping": {"patience": 0}, "seed": 0}
    p.update(over)
    return p


def _h(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes() + str(a.shape).encode() + str(a.dtype).encode()).hexdigest()[:16]


def synth_df(n: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.random(n)
    low = np.minimum(open_, close) - rng.random(n)
    vol = rng.integers(100, 1000, n).astype(float)
    idx = pd.date_range("2024-01-01 09:00", periods=n, freq="min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


_DF = synth_df()


def data_loader(_timeframe):
    return _DF


# ── spy architecture：記錄傳進來的陣列，回傳確定性的假預測 ─────────────────────────────
SPY_CALLS: list = []


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def spy_train(X_train, y_train, X_val, y_val, params, on_epoch, task_type="classification", preview_hook=None):
    SPY_CALLS.append({
        "X_train": _h(X_train), "y_train": _h(y_train), "X_val": _h(X_val), "y_val": _h(y_val),
        "X_train_shape": list(np.asarray(X_train).shape), "task_type": task_type,
    })

    def predict(X):
        v = np.asarray(X)[:, -1, 0]
        return (v > np.median(v)).astype(int) if task_type == "classification" else v * 0.5

    result = {"final_metrics": {"m": 1.0}, "predict": predict, "device": "cpu", "weights": b"spy",
              "model_config": {"spy": True}}
    if task_type == "classification":
        def predict_proba(X):
            p = _sigmoid(np.asarray(X)[:, -1, 0])
            return np.stack([1 - p, p], axis=1)
        result["predict_proba"] = predict_proba
    return result


def _feat(i, key="ohlcv"):
    return {"id": i, "type": "feature", "key": key, "timeframe": "tx_1m"}


def _label(i, outcome, rule, **params):
    return {"id": i, "type": "label", "outcome": outcome, "labeling_rule": rule, "timeframe": "tx_1m",
            "params": {"horizon": 1, **params}}


def _model(i, key, inputs, label, **extra):
    n = {"id": i, "type": "model", "key": key, "inputs": inputs, "label": label, "window": 10, "val_ratio": 0.2,
         "params": extra.pop("params", {})}
    n.update(extra)
    return {**n}


DATA_GRAPHS = {
    "g1_regression": {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
        _feat("f1"), _label("l1", "simple_return", "identity"), _model("m1", "spy", ["f1"], "l1")]},
    "g2_cls_chrono_prob": {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
        _feat("f1"), _feat("f2", "raw_volume"), _label("l1", "log_return", "fixed_threshold", n_classes=2),
        _model("m1", "spy", ["f1", "f2"], "l1", split_strategy="chronological", output_mode="probability")]},
    "g3_fusion_interleaved": {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
        _feat("fa", "raw_price"), _feat("fv", "raw_volume"), _label("l1", "log_return", "fixed_threshold", n_classes=2),
        _model("up", "spy", ["fa"], "l1", output_mode="probability"),
        _model("down", "spy", ["fa", "up", "fv"], "l1")]},
    "g4_fusion_regression_upstream": {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
        _feat("fa", "raw_price"), _label("lr", "simple_return", "identity"),
        _model("up", "spy", ["fa"], "lr"), _model("down", "spy", ["up", "fa"], "lr")]},
}


def capture_data_contract() -> dict:
    architectures.ARCHITECTURE_REGISTRY["spy"] = spy_train
    out = {}
    for name, spec in DATA_GRAPHS.items():
        SPY_CALLS.clear()
        results = graph_mod.run_graph(spec, data_loader)
        out[name] = {"calls": list(SPY_CALLS), "result_nodes": sorted(results.keys()),
                     "task_types": {k: v["task_type"] for k, v in results.items()}}
    return out


# ── B：本機合成產物的推論 ───────────────────────────────────────────────────────────
FUSION_REAL = {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
    _feat("fa", "raw_price"), _feat("fv", "raw_volume"), _label("l1", "log_return", "fixed_threshold", n_classes=2),
    _model("up", "xgboost", ["fa", "fv"], "l1", params=xgb_params(), output_mode="probability"),
    _model("down", "xgboost", ["fa", "fv", "up"], "l1", params=xgb_params()),
]}
SINGLE_REAL = {"start": "2024-01-01", "end": "2024-02-01", "nodes": [
    _feat("fa"), _label("lr", "simple_return", "identity"),
    _model("solo", "xgboost", ["fa"], "lr", params=xgb_params())]}


def build_artifacts(spec: dict) -> dict:
    results = graph_mod.run_graph(spec, data_loader)
    nodes = {n["id"]: n for n in spec["nodes"]}
    model_ids = {i for i, n in nodes.items() if n["type"] == "model"}
    arts = {}
    for nid, r in results.items():
        node = nodes[nid]
        arts[nid] = {
            "architecture_key": node["key"], "task_type": r["task_type"], "weights": r["weights"],
            "model_config": r["model_config"], "feature_schema": [], "target_spec": {},
            "graph_spec_snapshot": spec,
            "depends_on_node_ids": [i for i in (x if isinstance(x, str) else x["node"] for x in node["inputs"]) if i in model_ids],
            "preprocessing_state": r["preprocessing_state"],
        }
    return arts


def run_inference_with(arts: dict, node_id: str, chunk_size: int = 50) -> dict:
    """monkeypatch load_artifact，用合成產物走完整的推論路徑（串流版入口）。"""
    orig = inference_mod.load_artifact
    inference_mod.load_artifact = lambda job_id, nid: arts.get(nid)
    try:
        ts_all, pred_all = [], []

        def on_chunk(ts, preds, task_type, is_prob, *rest):
            ts_all.extend(ts)
            pred_all.extend(preds)

        n = inference_mod.run_inference_chunked("job", node_id, data_loader, on_chunk, chunk_size=chunk_size)
    finally:
        inference_mod.load_artifact = orig
    return {"n": n, "ts": ts_all, "pred": pred_all}


def capture_inference_local() -> dict:
    out = {}
    for name, spec, nid in [("fusion_real", FUSION_REAL, "down"), ("single_real", SINGLE_REAL, "solo")]:
        out[name] = run_inference_with(build_artifacts(spec), nid)
    return out


# ── C：重新訓練指標（固定隨機性） ────────────────────────────────────────────────────
def _train_metrics(kind: str) -> list:
    rng = np.random.default_rng(5)
    n_train, n_val, window, nf = 200, 60, 6, 3
    Xt = rng.normal(size=(n_train, window, nf)).astype(np.float32)
    Xv = rng.normal(size=(n_val, window, nf)).astype(np.float32)
    yt = (Xt[:, -1, 0] > 0).astype(np.float32)
    yv = (Xv[:, -1, 0] > 0).astype(np.float32)
    seen = []
    if kind == "xgboost":
        architectures.train_model("xgboost", Xt, yt.astype(int), Xv, yv.astype(int), xgb_params(n_estimators=8),
                                  lambda e, m: seen.append(dict(m)), n_classes=2)
    else:
        architectures.train_model("lstm", Xt, yt.astype(int), Xv, yv.astype(int), lstm_params(),
                                  lambda e, m: seen.append(dict(m)), n_classes=2)
    return seen


def capture_training_metrics(repeats: int = 3) -> dict:
    return {kind: [_train_metrics(kind) for _ in range(repeats)] for kind in ("xgboost", "lstm")}


def _spread(runs: list) -> float:
    """同一組設定重複跑之間，各指標的最大差異。"""
    worst = 0.0
    base = runs[0]
    for other in runs[1:]:
        for a, b in zip(base, other):
            for k in a:
                if a[k] is not None and b.get(k) is not None:
                    worst = max(worst, abs(a[k] - b[k]))
    return worst


# ── 入口 ──────────────────────────────────────────────────────────────────────────
def capture_all() -> dict:
    data = {
        "data_contract": capture_data_contract(),
        "inference_local": capture_inference_local(),
        "training_metrics": capture_training_metrics(),
    }
    data["training_spread"] = {k: _spread(v) for k, v in data["training_metrics"].items()}
    return data


def compare(base: dict, new: dict) -> list[str]:
    problems = []
    if base["data_contract"] != new["data_contract"]:
        for g in base["data_contract"]:
            if base["data_contract"][g] != new["data_contract"].get(g):
                problems.append(f"A 資料契約不一致：{g}")
    for k in base["inference_local"]:
        b, n = base["inference_local"][k], new["inference_local"][k]
        if b["n"] != n["n"] or b["ts"] != n["ts"] or not np.array_equal(np.asarray(b["pred"]), np.asarray(n["pred"])):
            problems.append(f"B 本機合成產物推論不一致：{k}")
    for kind in ("xgboost", "lstm"):
        tol = base["training_spread"][kind]
        new_run = new["training_metrics"][kind][0]
        base_run = base["training_metrics"][kind][0]
        worst = 0.0
        for a, b in zip(base_run, new_run):
            for key in a:
                if a[key] is not None and b.get(key) is not None:
                    worst = max(worst, abs(a[key] - b[key]))
        if len(base_run) != len(new_run):
            problems.append(f"C {kind} 輪數不同")
        elif worst > tol:
            problems.append(f"C {kind} 訓練指標差 {worst:.3e} 超出舊程式跑跑之間波動 {tol:.3e}")
    return problems


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "capture":
        data = capture_all()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print("captured:", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in data.items()})
        print("training run-to-run spread on baseline code:", data["training_spread"])
    elif mode == "compare":
        with open(path, encoding="utf-8") as f:
            base = json.load(f)
        new = json.loads(json.dumps(capture_all()))
        problems = compare(base, new)
        print("PROBLEMS:" if problems else "ALL MATCH")
        for p in problems:
            print(" -", p)
        sys.exit(1 if problems else 0)
