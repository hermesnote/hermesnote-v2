"""載入 `model_artifacts` 已保存的模型，對新資料做推論，不重新訓練。

跟 `graph.py` 共用時間戳對齊邏輯（直接沿用它的 `_ensure_time_indexed`），差別只在
Model Node 這一步不訓練，改成用 `architectures.load_model()` 載入保存的權重直接 predict。
一個 Model Node 如果依賴其他 Model Node（融合），這裡會遞迴載入整條依賴鏈，都用同一個
`job_id` 底下各自保存的產物重建，不會偷偷重新訓練上游。

推論可以用新資料（decision_time 之後已知的新 K 棒／歷史窗口算特徵，不算重新學習）；
凍結的是模型權重／已擬合的前處理狀態／決策設定——這裡目前 preprocessing 固定是 none，
之後接了正規化，要跟著從 artifact 的 preprocessing_state 讀出來套用，不是重新 fit。

窗格化改用 sliding_window_view（跟 training/graph.py 訓練時同一招）開 view，不是像舊版
那樣用 Python list 把每一個窗格逐一 append 再 np.asarray() 疊成一個 (n, window, features)
的大陣列——後者對全期間推論（可能百萬列）會直接把記憶體榨爆，這是這輪修的問題。
真正的預測也改成分批（chunk）呼叫 predict()/predict_proba()，不是把整段期間一次丟進去，
串流入口 run_inference_chunked() 更進一步：算完一批就交給呼叫端處理（存 DB），不會在
記憶體裡累積成一個跟全期間筆數一樣長的 predictions 陣列。
"""

from typing import Callable

import numpy as np

from training.artifacts import load_artifact
from training.graph import _ensure_time_indexed
from training.registry.architectures import load_model
from training.registry.features import compute_feature

CHUNK_SIZE = 5000


def _compute_windowed_features(job_id: str, node_id: str, data_loader: Callable[[str], "object"], cache: dict):
    """算出這個節點「已經正規化、還沒展開窗格」的原始特徵陣列 X_raw，再用
    sliding_window_view 開一個窗格化的 view——這一步幾乎不佔額外記憶體（view 不是複製），
    真正配置新記憶體只會發生在呼叫端對某個 chunk 呼叫 predict() 的那一刻。

    回傳 (X_view, window_ts, artifact, loaded, task_type, output_mode)。
    """
    artifact = load_artifact(job_id, node_id)
    if artifact is None:
        raise ValueError(f"找不到 job {job_id} 節點 {node_id} 的模型產物，無法推論（是不是訓練失敗或還沒完成？）")

    graph_spec = artifact["graph_spec_snapshot"]
    nodes = {n["id"]: n for n in graph_spec["nodes"]}
    node = nodes[node_id]

    feature_ids = [i for i in node["inputs"] if nodes[i]["type"] == "feature"]
    input_arrays: list[np.ndarray] = []
    input_ts: list[np.ndarray] = []
    for fid in feature_ids:
        fnode = nodes[fid]
        df = _ensure_time_indexed(data_loader(fnode["timeframe"]))
        input_arrays.append(compute_feature(fnode["key"], df, fnode.get("params")))
        input_ts.append(np.asarray(df.index.values))

    # 遞迴載入上游 Model Node 的推論結果，當額外特徵接進來——這就是「完整依賴鏈」，
    # 不是只存末端節點的權重，載入時才不會需要偷偷重訓上游。上游本身也是透過
    # _run_node_inference（下面）算出來的，一樣享有這裡的窗格化/分批預測優化。
    for upstream_id in artifact["depends_on_node_ids"]:
        pred, ts = _run_node_inference(job_id, upstream_id, data_loader, cache)
        input_arrays.append(pred)
        input_ts.append(ts)

    if not input_arrays:
        raise ValueError(f"節點 {node_id}：沒有可用的特徵輸入，無法推論")

    common_ts = input_ts[0]
    for ts in input_ts[1:]:
        common_ts = np.intersect1d(common_ts, ts)
    if len(common_ts) == 0:
        raise ValueError(f"節點 {node_id}：所有輸入在新資料上沒有共同的時間戳，無法對齊")

    def _select(arr: np.ndarray, ts: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(ts, common_ts)
        return arr[idx]

    aligned = [_select(a, ts) for a, ts in zip(input_arrays, input_ts)]
    X_raw = aligned[0] if len(aligned) == 1 else np.concatenate(aligned, axis=1)

    # 套用訓練當下存下來的正規化參數（train-only fit 出來的 min/scale），不是重新 fit——
    # 凍結的是已擬合的前處理狀態，這是「推論可以用新資料，但不能變成重新學習」的其中一環。
    # 在窗格化「之前」對 X_raw（形狀只有 n×特徵數）做，數學上跟窗格化後再做完全等價
    # （逐元素線性變換），但完全不用碰窗格化後的大陣列——跟 graph.py 訓練時同一個道理。
    preprocessing_state = artifact.get("preprocessing_state") or {"method": "none"}
    if preprocessing_state.get("method") == "minmax":
        feat_min = np.asarray(preprocessing_state["min"])
        feat_scale = np.asarray(preprocessing_state["scale"])
        X_raw = (X_raw - feat_min) / feat_scale

    window = int(node.get("window", 60))
    n = len(common_ts)
    if n <= window:
        raise ValueError(f"節點 {node_id}：新資料筆數不足以組出任何一個 window={window} 的樣本，無法推論")

    X_view_full = np.moveaxis(np.lib.stride_tricks.sliding_window_view(X_raw, window_shape=window, axis=0), -1, 1)
    X_view = X_view_full[:-1]  # 跟訓練時一樣，排除最後一個涵蓋到 common_ts 最後一列的窗格
    window_ts = common_ts[window - 1:n - 1]

    loaded = load_model(artifact["architecture_key"], bytes(artifact["weights"]), artifact["model_config"])
    return X_view, window_ts, artifact, loaded, node.get("output_mode")


def _predict_in_chunks(loaded: dict, X_view: np.ndarray, task_type: str, output_mode: str | None, chunk_size: int = CHUNK_SIZE):
    """依序把 X_view 切成 chunk_size 一批各自呼叫 predict()/predict_proba()，每次
    yield 這一批的結果，用完即丟——不會在呼叫端手上累積成一個跟全期間筆數一樣長的陣列。
    output_mode="probability" 才輸出機率分佈（回歸沒有 predict_proba，一律走 predict()）——
    跟 graph.py 訓練時、worker.py 存 artifact 時同一套規則，這裡對齊回去才會一致。
    """
    is_probability = task_type == "classification" and output_mode == "probability" and "predict_proba" in loaded
    n = len(X_view)
    for start in range(0, n, chunk_size):
        chunk = np.asarray(X_view[start:start + chunk_size])  # 這裡才真的配置這個 chunk 的記憶體
        raw = loaded["predict_proba"](chunk) if is_probability else loaded["predict"](chunk)
        preds = np.asarray(raw)
        preds = preds.reshape(-1, 1) if preds.ndim == 1 else preds
        yield start, preds, is_probability


def _run_node_inference(job_id: str, node_id: str, data_loader: Callable[[str], "object"], cache: dict):
    """給上游模型節點用：這裡還是得把上游的完整預測陣列算出來（下游每一列都需要對應
    的上游預測值當特徵），沒辦法真的串流，但窗格化跟預測本身都用了上面的省記憶體招式，
    只是最後還是 concatenate 成一個陣列——這是多模型融合架構本身的限制，不是這裡偷懶。
    """
    if node_id in cache:
        return cache[node_id]

    X_view, window_ts, artifact, loaded, output_mode = _compute_windowed_features(job_id, node_id, data_loader, cache)
    chunks = [preds for _, preds, _ in _predict_in_chunks(loaded, X_view, artifact["task_type"], output_mode)]
    predictions = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1))

    result = (predictions, np.asarray(window_ts))
    cache[node_id] = result
    return result


def run_inference(job_id: str, node_id: str, data_loader: Callable[[str], "object"]) -> dict:
    """對外入口（非串流版）：回傳 {"timestamps": [...], "predictions": [...], "prediction_source": [...]}，
    都是可以直接 json.dumps 的原生型別。prediction_source 逐列都是 "inference"（不是訓練時
    train/val 那種切分來源），跟 graph.py 訓練時逐列標記 train/val/unused 用同一個欄位名稱，
    下游/查看的人不用分兩套邏輯判斷這批預測是不是真的樣本外。

    這支會把整段期間的 predictions 攤開成一個 list 回傳，量級大的全期間推論請改用
    run_inference_chunked()，不要用這支再自己重新分批——那樣就白做了省記憶體的事。
    """
    cache: dict = {}
    predictions, timestamps = _run_node_inference(job_id, node_id, data_loader, cache)
    return {
        "timestamps": [int(np.datetime64(t, "s").astype("int64")) for t in timestamps],
        "predictions": predictions.tolist(),
        "prediction_source": ["inference"] * len(timestamps),
    }


def run_inference_chunked(
    job_id: str, node_id: str, data_loader: Callable[[str], "object"],
    on_chunk: Callable[[list[int], list[list[float]], str, bool], None],
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """串流版推論入口：目標節點（要存進 model_inference_predictions 的那個節點）用這支，
    不是先把整段期間的 predictions 全部算完存成一個大 list 才分批寫 DB——那樣「分批」只
    解決了寫入 DB 那一步，算預測本身那個大 list 早就把記憶體吃掉了。這裡每算完一批
    （chunk）就呼叫 on_chunk(timestamps_chunk, predictions_chunk, task_type, is_probability)，
    呼叫端（worker.py）可以馬上交給 inference_store.py 的 writer 寫進 DB 再丟掉，尖峰記憶體
    只跟 chunk_size 有關，不跟全期間筆數成正比。回傳總筆數。

    只用在「目標節點」，不是遞迴的上游節點——上游節點的完整預測陣列本來就得留在記憶體裡
    當下游的特徵輸入，沒辦法真的串流，那條路徑走 _run_node_inference()。
    """
    cache: dict = {}
    X_view, window_ts, artifact, loaded, output_mode = _compute_windowed_features(job_id, node_id, data_loader, cache)
    task_type = artifact["task_type"]
    total = 0
    for start, preds, is_probability in _predict_in_chunks(loaded, X_view, task_type, output_mode, chunk_size):
        ts_chunk = [int(np.datetime64(t, "s").astype("int64")) for t in window_ts[start:start + len(preds)]]
        on_chunk(ts_chunk, preds.tolist(), task_type, is_probability)
        total += len(preds)
    return total
