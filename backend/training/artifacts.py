"""模型產物落地保存 + 載入推論用的讀取，用同步 psycopg2（跟 progress.py 同一個理由：
呼叫端是訓練用的背景執行緒，沒有事件迴圈）。

保存時機：訓練 job 成功結束當下（`worker.py` 的 `run_one_sync`），不是等使用者按「儲存」
按鈕才存——那時候模型物件已經在記憶體裡，job 一結束就消失，等按鈕再存來源已經不在了。
UI 的「儲存模型」按鈕實際上是把 `kept` 標成 True，純粹是要不要在清單特別標記/顯示，
不影響這筆產物存不存在或能不能被載入推論（未標記的一樣可以拿去推論）。

權重直接存 bytea 在 `hermesnote` DB 的 `model_artifacts` 表，不是存檔案路徑——
training 容器跟 backend 容器目前沒有共用 volume（見 docker-compose.yml），存檔案兩邊
會互相讀不到，容器重建也會把檔案沖掉；兩個容器本來就都連得到這個 DB。
"""

import json
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

_SYNC_DATABASE_URL = os.getenv("HERMESNOTE_DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")


def save_artifacts_for_job(job_id: str, graph_spec: dict, results: dict) -> None:
    """graph_spec 裡每個 model 節點，只要訓練結果帶了 weights/model_config，就存一筆 artifact。
    `results` 是 `run_graph()` 的完整回傳（還沒被 worker.py 濾掉 predict/weights 之前的版本）。
    """
    nodes = {n["id"]: n for n in graph_spec["nodes"]}
    model_node_ids = {nid for nid, n in nodes.items() if n["type"] == "model"}

    conn = psycopg2.connect(_SYNC_DATABASE_URL)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for nid, result in results.items():
                if "weights" not in result or "model_config" not in result:
                    continue  # 這個架構這次沒回傳可保存的權重（理論上不會發生，防呆）
                node = nodes[nid]
                feature_ids = [i for i in node["inputs"] if nodes[i]["type"] == "feature"]
                depends_on_node_ids = [i for i in node["inputs"] if i in model_node_ids]
                feature_schema = [nodes[i] for i in feature_ids]
                label_node = nodes[node["label"]]
                target_spec = {
                    "outcome": label_node["outcome"],
                    "labeling_rule": label_node.get("labeling_rule", "fixed_threshold"),
                    "params": label_node.get("params", {}),
                }
                cur.execute(
                    """
                    INSERT INTO model_artifacts
                        (job_id, node_id, architecture_key, task_type, weights, model_config,
                         feature_schema, target_spec, graph_spec_snapshot, preprocessing_state,
                         depends_on_node_ids)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id, node_id) DO UPDATE SET
                        weights = EXCLUDED.weights, model_config = EXCLUDED.model_config,
                        feature_schema = EXCLUDED.feature_schema, target_spec = EXCLUDED.target_spec,
                        graph_spec_snapshot = EXCLUDED.graph_spec_snapshot,
                        preprocessing_state = EXCLUDED.preprocessing_state,
                        depends_on_node_ids = EXCLUDED.depends_on_node_ids
                    """,
                    (
                        job_id, nid, node["key"], result.get("task_type", "classification"),
                        psycopg2.Binary(result["weights"]), json.dumps(result["model_config"]),
                        json.dumps(feature_schema), json.dumps(target_spec), json.dumps(graph_spec),
                        json.dumps(result.get("preprocessing_state", {"method": "none"})),
                        json.dumps(depends_on_node_ids),
                    ),
                )
    finally:
        conn.close()


def load_artifact(job_id: str, node_id: str) -> dict | None:
    """同步讀取，給推論執行路徑用（在訓練容器的 sync 執行緒裡跑，跟 save_artifacts_for_job 同一套理由）。
    web 後端要查清單/切換 kept 用的是 `services/model_artifacts.py` 的 async 版本，不是這支——
    那邊是 FastAPI 的 async route handler，不該塞一個會擋住 event loop 的同步 DB 呼叫進去。
    """
    conn = psycopg2.connect(_SYNC_DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT architecture_key, task_type, weights, model_config, feature_schema,
                       target_spec, graph_spec_snapshot, depends_on_node_ids, preprocessing_state
                FROM model_artifacts WHERE job_id = %s AND node_id = %s
                """,
                (job_id, node_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            keys = ["architecture_key", "task_type", "weights", "model_config", "feature_schema",
                    "target_spec", "graph_spec_snapshot", "depends_on_node_ids", "preprocessing_state"]
            return dict(zip(keys, row))
    finally:
        conn.close()
