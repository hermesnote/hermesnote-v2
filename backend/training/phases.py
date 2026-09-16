"""教授的 Train/Validation/Test 三階段協定：任務血緣規則。

這是研究方法論的硬性規定（見 docs/temp/2026-09-10-three-party-design-alignment.md 的
I-001-H-002），不是設計選項：
  Phase 1：訓練期間內 random 切分。
  Phase 2：必須源自一個已完成的 Phase 1 job，完整繼承其 graph_spec（資料範圍/特徵/
           outcome/target/horizon/n_classes/threshold/模型架構，一律不可修改），
           僅覆寫 split_strategy=chronological，重新訓練一次。
  Phase 3：必須源自一個已完成且已有模型產物的 Phase 2 job，直接載入該產物做推論
           （不重新訓練），對 holdout 期間算預測準確度指標。

這支只負責血緣規則的驗證跟 Phase 2 的 graph_spec 繼承邏輯，job 佇列/狀態機制本身
不變（跟一般訓練/推論 job 共用 model_training_jobs 這張表）。
"""

import copy


def final_model_node_id(nodes: list[dict]) -> str:
    """找出這張圖裡「沒有被其他 model node 當上游依賴」的那個 model node——
    這是整張圖的最終輸出節點，Phase 協定操作的對象。單一模組訓練時，圖裡只有一個
    model node，自然就是它；多模組融合的圖，這是融合後的最終節點。
    """
    model_ids = {n["id"] for n in nodes if n["type"] == "model"}
    if not model_ids:
        raise ValueError("這張圖沒有任何 Model Node")
    referenced: set[str] = set()
    for n in nodes:
        if n["type"] == "model":
            referenced |= {i for i in n.get("inputs", []) if i in model_ids}
    finals = model_ids - referenced
    if len(finals) != 1:
        raise ValueError(
            f"無法唯一決定這張圖的最終模型節點（候選：{sorted(finals)}），"
            "Phase 協定目前只支援單一輸出節點的圖"
        )
    return next(iter(finals))


def build_phase2_graph_spec(parent_graph_spec: dict) -> dict:
    """完整繼承 Phase 1 的 graph_spec，只把每個 Model Node 的 split_strategy 覆寫成
    chronological，其餘一律不可修改（呼叫端不應該再讓使用者調整任何其他參數）。
    """
    graph_spec = copy.deepcopy(parent_graph_spec)
    for n in graph_spec["nodes"]:
        if n["type"] == "model":
            n["split_strategy"] = "chronological"
    return graph_spec


def validate_parent_for_phase(parent_job: dict | None, expected_parent_phase: int, this_phase: int) -> None:
    """`parent_job` 是 job_store.get_job() 的回傳；擋掉不合法的血緣。"""
    if parent_job is None:
        raise ValueError(f"Phase {this_phase} 必須指定一個已完成的 Phase {expected_parent_phase} job 當來源，找不到指定的來源 job")
    if parent_job.get("phase") != expected_parent_phase:
        raise ValueError(
            f"Phase {this_phase} 的來源 job 必須是 Phase {expected_parent_phase}，"
            f"指定的 job 卻是 Phase {parent_job.get('phase')!r}"
        )
    if parent_job.get("status") != "done":
        raise ValueError(f"Phase {this_phase} 的來源 job 必須是已完成（status=done）的任務，目前是 {parent_job.get('status')!r}")
