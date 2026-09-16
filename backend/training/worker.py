"""訓練容器的長駐 worker process（獨立於 web 容器，掛 GPU）。

職責：
  - 輪詢 model_training_jobs（status='pending'），同一時間只執行一筆
    （GPU 記憶體通常被單一訓練跑滿，不支援多工，天生序列化）
  - 執行前用 nvidia-smi 檢查 GPU 是否被其他 process（例如地端的 12B 模型）佔用，忙碌就延後重試
  - 真正執行 graph.py 的節點圖引擎（Feature Node / Label Node / Model Node），
    訓練過程中每個 epoch 透過 training/progress.py 寫進度回 DB + NOTIFY

web 容器只負責把任務寫成 status='pending'，完全不碰 GPU/torch；
兩邊只共用 model_training_jobs 這張表，不需要額外的容器間呼叫協定。

graph_spec 格式：{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "nodes": [...]}，
每個 feature/label 節點的 "timeframe" 是 "{symbol}_{timeframe}"（例如 "tx_15m"），
對應 services/quotes.py 的 quotes_table() 命名規則。

背景見 docs/record/archive/2026-09-08.md、2026-09-08-training-plan.md。
"""

import asyncio
import json
import subprocess
from datetime import date

from training import job_store
from training.artifacts import save_artifacts_for_job
from training.graph import run_graph
from training.inference import run_inference_chunked
from training.inference_store import open_prediction_writer
from training.preview_store import save_preview_sample
from training.progress import write_progress

POLL_INTERVAL_SEC      = 5
GPU_BUSY_RETRY_SEC     = 30
GPU_BUSY_THRESHOLD_MB  = 2000


def gpu_is_busy() -> bool:
    """判斷 GPU 是否被其他 process（如地端 12B 模型）佔用。
    查不到 nvidia-smi（環境本身沒有 GPU）時視為不忙碌，不擋訓練。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return False
        used_mb = int(out.stdout.strip().splitlines()[0])
        return used_mb > GPU_BUSY_THRESHOLD_MB
    except Exception:
        return False


async def _load_dataframes(graph_spec: dict) -> dict:
    """graph_spec 裡所有 feature/label 節點用到的 timeframe，在 async context 統一先撈好，
    撈完交給同步的 graph.run_graph 用一個純字典查表的 data_loader closure，
    這樣訓練那條同步/GPU 的執行緒完全不用碰 async DB 連線。
    """
    from services.quotes import fetch_ohlcv_dataframe

    start = date.fromisoformat(graph_spec["start"])
    end = date.fromisoformat(graph_spec["end"])

    timeframes = {
        n["timeframe"] for n in graph_spec["nodes"] if n["type"] in ("feature", "label")
    }
    dfs = {}
    for tf in timeframes:
        symbol, timeframe = tf.split("_", 1)
        dfs[tf] = await fetch_ohlcv_dataframe(symbol, timeframe, start, end)
    return dfs


def run_one_sync(job_id: str, graph_spec: dict, dfs: dict) -> dict:
    def data_loader(timeframe: str):
        return dfs[timeframe]

    # 每個 model node 的總 epochs/estimators 數，用來在 log 印進度百分比——只是給人在
    # `docker logs -f` 盯著看用的，不影響任何邏輯，也不透過資料庫。
    total_by_node = {
        n["id"]: (n.get("params") or {}).get("epochs") or (n.get("params") or {}).get("n_estimators")
        for n in graph_spec["nodes"] if n["type"] == "model"
    }

    def on_epoch(node_id: str, epoch: int, metrics: dict):
        write_progress(job_id, node_id, epoch, metrics)
        total = total_by_node.get(node_id)
        pct = f"{(epoch + 1) / total * 100:5.1f}%" if total else "  ?  "
        metric_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))
        print(f"[worker] job {job_id} [{node_id}] epoch {epoch + 1}/{total or '?'} ({pct}) {metric_str}", flush=True)

    def on_preview(node_id: str, preview: dict):
        # save_preview_sample 自己整段包 try/except、絕對不往外拋——這裡不用再防護一次，
        # 保留這行呼叫本身不包 try 是刻意的，用來提醒之後改動的人：例外處理的責任在
        # preview_store.py，不是在這裡重複做。
        save_preview_sample(job_id, node_id, preview)

    results = run_graph(graph_spec, data_loader, on_epoch, on_preview)
    # 訓練成功當下就落地保存模型產物（見 training/artifacts.py 的說明：不能等使用者
    # 事後按「儲存」才存，那時候模型物件已經從記憶體消失了）。用還沒被下面濾過的 results
    # （含 weights/model_config），存完才濾掉不可序列化的欄位存進 job 的 result 摘要。
    save_artifacts_for_job(job_id, graph_spec, results)
    # predict/predict_proba 是 closure（含模型本身）、weights 是原始 bytes——都不能序列化進
    # model_training_jobs.result 這個 JSONB 欄位，只留 final_metrics/device/training_meta 這些
    # 摘要；先前這裡漏了 predict_proba，任何跑到 output_mode="probability" 的分類任務存 DB
    # 時會直接因為 json.dumps 序列化不了 function 而失敗，順手修掉。
    _non_serializable = {"predict", "predict_proba", "weights"}
    return {
        node_id: {k: v for k, v in r.items() if k not in _non_serializable}
        for node_id, r in results.items()
    }


async def run_one(job_id: str, graph_spec: dict):
    dfs = await _load_dataframes(graph_spec)
    result = await asyncio.to_thread(run_one_sync, job_id, graph_spec, dfs)
    await job_store.mark_done(job_id, result)
    print(f"[worker] job {job_id} done: {result}", flush=True)


async def _load_dataframes_for_infer(target_job_id: str, target_node_id: str, start: date, end: date) -> dict:
    """推論用的資料，時間區間是使用者這次指定的新區間，不是訓練當時那段——
    需要哪些 timeframe，從已保存產物的 graph_spec_snapshot 裡的 feature node 定義反推
    （那份快照本來就含完整的圖，不用另外記錄一份「這個模型用哪些 timeframe」）。
    """
    from services.hermesnote_db import get_conn
    from services.quotes import fetch_ohlcv_dataframe

    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            "SELECT graph_spec_snapshot FROM model_artifacts WHERE job_id = $1 AND node_id = $2",
            target_job_id, target_node_id,
        )
    finally:
        await conn.close()
    if row is None:
        raise ValueError(f"找不到 job {target_job_id} 節點 {target_node_id} 的模型產物")

    # asyncpg 對 JSONB 欄位預設不會自動解碼成 dict（這條連線也沒有註冊 JSON type codec，
    # 見 services/hermesnote_db.py 的 get_conn()）——回來的是原始 JSON 字面文字，不是 dict，
    # 直接 ["nodes"] 會是「對字串做字典索引」而炸掉（TypeError: string indices must be
    # integers）。同一支檔案的 job_store.py 讀 graph_spec/result/window_meta 時都有明確
    # json.loads()，這裡是漏掉的那一處，補上跟既有慣例一致，不是這條連線特例的問題，
    # 不需要為此在 get_conn() 全域註冊 JSON codec。
    graph_spec = json.loads(row["graph_spec_snapshot"])
    timeframes = {n["timeframe"] for n in graph_spec["nodes"] if n["type"] == "feature"}
    dfs = {}
    for tf in timeframes:
        symbol, timeframe = tf.split("_", 1)
        dfs[tf] = await fetch_ohlcv_dataframe(symbol, timeframe, start, end)
    return dfs


def run_infer_sync(job_id: str, target_job_id: str, target_node_id: str, dfs: dict) -> dict:
    """真正的串流入口：每算完一批就馬上用同一條連線寫進 model_inference_predictions
    再丟掉，不會像舊版那樣先呼叫 run_inference() 拿到一個跟全期間筆數一樣長的
    predictions list，才回頭分批寫 DB——那樣「分批」只解決了寫入 DB 那一步，
    算預測本身那個大 list 早就把記憶體吃掉了。
    """
    def data_loader(timeframe: str):
        return dfs[timeframe]

    summary = {"count": 0, "start_ts": None, "end_ts": None}
    with open_prediction_writer(job_id, target_node_id) as writer:
        def on_chunk(ts_chunk, preds_chunk, task_type, is_probability):
            writer.write_chunk(ts_chunk, preds_chunk, task_type, is_probability)
            summary["count"] += len(ts_chunk)
            if summary["start_ts"] is None:
                summary["start_ts"] = ts_chunk[0]
            if ts_chunk:
                summary["end_ts"] = ts_chunk[-1]

        run_inference_chunked(target_job_id, target_node_id, data_loader, on_chunk)
    return summary


async def run_one_infer(job_id: str, infer_spec: dict):
    """infer_spec: {"target_job_id", "target_node_id", "start", "end"}——載入已保存的模型
    對新區間做推論，不重新訓練，跟 run_one() 共用同一個 job 佇列/狀態機制，只是執行內容不同。

    全期間的逐列預測（可能到百萬列）分批存進 model_inference_predictions，不是塞進
    model_training_jobs.result 這個 JSONB 欄位——那個欄位只留摘要（筆數、時間範圍），
    完整內容改用 GET /api/model/jobs/{job_id}/inference_predictions 依時間區間/游標分頁查。
    這是「最終模型推論」，跟訓練時的抽樣紀錄（model_training_preview_samples）是不同性質
    的資料，兩者分開儲存、分開查詢。
    """
    target_job_id = infer_spec["target_job_id"]
    target_node_id = infer_spec["target_node_id"]
    start = date.fromisoformat(infer_spec["start"])
    end = date.fromisoformat(infer_spec["end"])
    dfs = await _load_dataframes_for_infer(target_job_id, target_node_id, start, end)
    summary = await asyncio.to_thread(run_infer_sync, job_id, target_job_id, target_node_id, dfs)
    await job_store.mark_done(job_id, {target_node_id: summary})
    print(f"[worker] infer job {job_id} done（{summary['count']} 筆預測已分批存入 model_inference_predictions）", flush=True)


async def main_loop():
    print("[worker] training worker started, polling model_training_jobs(status='pending')", flush=True)
    while True:
        job = await job_store.fetch_next_pending()
        if job is None:
            await asyncio.sleep(POLL_INTERVAL_SEC)
            continue

        if gpu_is_busy():
            print(f"[worker] GPU busy, delaying job {job['id']}", flush=True)
            await asyncio.sleep(GPU_BUSY_RETRY_SEC)
            continue

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        await job_store.mark_running(job["id"], device)
        print(f"[worker] picked up job {job['id']} (job_type={job['job_type']})", flush=True)
        try:
            if job["job_type"] == "infer":
                await run_one_infer(job["id"], job["graph_spec"])
            else:
                await run_one(job["id"], job["graph_spec"])
        except Exception as e:
            print(f"[worker] job {job['id']} failed: {e}", flush=True)
            await job_store.mark_failed(job["id"], str(e))


if __name__ == "__main__":
    asyncio.run(main_loop())
