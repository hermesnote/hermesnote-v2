"""串接防洩漏的獨立核對（M5）：只讀資料庫裡的逐筆紀錄與上游折次清單重新計算，不使用訓練程式自己的判定。

規則（每一筆都必須成立，違反筆數必須是 0）：
  r1 可用時間     上游預測的可用時間 ≤ 下游的決策時刻
  r2 不重疊       下游訓練資料用到的每一折：該折訓練點區間（窗格起點～答案可用）不與這筆資料的區間重疊
                  （以折次清單的 before_max_le／after_min_ws 判定：資料區間必須落在兩者之間）
  r3 前進式       前進式／blending：該折訓練點最晚的答案可用時間 < 這筆資料的窗格起點
  r4 切分邊界     下游驗證資料用到的每一折，其訓練點不涵蓋這筆資料的決策時刻；時序切分時另需
                  該折訓練點最晚的答案可用時間 < 驗證集起點
  r5 框架一致     逐筆紀錄與折次清單的樣本框架雜湊全部相同
in_sample（舊行為）的折次（fold 0）一律計入 r2／r4 違反——那正是它有洩漏的原因。
"""

import csv
import io
import json
import random

import numpy as np
import pandas as pd


def _t(v):
    if v is None:
        return None
    t = pd.Timestamp(v)
    return None if pd.isna(t) else t


def audit(records: list[dict], folds: list[dict], *, split_strategy: str | None = None, sample_n: int = 5,
          seed: int = 0) -> dict:
    fold_by = {(f["node_id"], int(f["fold"])): f for f in folds}
    rules = {k: {"checked": 0, "violations": 0} for k in ("r1", "r2", "r3", "r4", "r5")}
    bad_rows = []
    used_recs = [r for r in records if r.get("dataset") == "train" or r.get("weights") == "best"]
    val_ts = [_t(r["sample_ts"]) for r in used_recs if r.get("dataset") == "val"]
    val_start = min(val_ts) if val_ts else None
    hashes = {r.get("frame_hash") for r in used_recs} | {f.get("frame_hash") for f in folds}
    rules["r5"]["checked"] = len(used_recs) + len(folds)
    if len({h for h in hashes}) > 1:
        rules["r5"]["violations"] = len(used_recs) + len(folds)

    def hit(rule, rec, u, f, why):
        rules[rule]["violations"] += 1
        bad_rows.append({"rule": rule, "sample_ts": str(rec["sample_ts"]), "dataset": rec.get("dataset"), "upstream": u,
                         "fold": f, "why": why})

    for rec in used_recs:
        ups = rec.get("upstream")
        ups = json.loads(ups) if isinstance(ups, str) else (ups or {})
        tau, ws, le = _t(rec.get("decision_available_ts")), _t(rec.get("window_start_ts")), _t(rec.get("target_available_ts"))
        for u, info in ups.items():
            pav = _t(info.get("pred_available_ts"))
            if pav is not None:
                rules["r1"]["checked"] += 1
                if tau is None or pav > tau:
                    hit("r1", rec, u, info.get("fold_at_decision"), f"上游可用 {pav} > 決策 {tau}")
            for f in info.get("folds_in_window") or []:
                f = int(f)
                if f == -2:
                    continue
                fb = fold_by.get((u, f))
                if fb is None:
                    rule = "r2" if rec["dataset"] == "train" else "r4"
                    rules[rule]["checked"] += 1
                    hit(rule, rec, u, f, "折次清單沒有這一折（含 in_sample 舊行為）")
                    continue
                b, a = _t(fb.get("before_max_le")), _t(fb.get("after_min_ws"))
                if rec["dataset"] == "train":
                    rules["r2"]["checked"] += 1
                    if (b is not None and not (ws is not None and b < ws)) or (a is not None and not (le is not None and a > le)):
                        hit("r2", rec, u, f, f"折訓練區間（before_max_le={b}, after_min_ws={a}）與資料區間 [{ws}, {le}] 重疊")
                    if fb.get("strategy") in ("forward_chaining", "blending"):
                        rules["r3"]["checked"] += 1
                        m = _t(fb.get("train_max_label_end"))
                        if m is not None and not (ws is not None and m < ws):
                            hit("r3", rec, u, f, f"折最晚答案可用 {m} ≥ 窗格起點 {ws}")
                else:
                    rules["r4"]["checked"] += 1
                    if (b is not None and not b < tau) or (a is not None and not a > tau):
                        hit("r4", rec, u, f, f"折訓練點涵蓋決策時刻 {tau}")
                    m = _t(fb.get("train_max_label_end"))
                    if split_strategy == "chronological" and val_start is not None and m is not None and not m < val_start:
                        hit("r4", rec, u, f, f"時序切分：折最晚答案可用 {m} ≥ 驗證起點 {val_start}")
    rng = random.Random(seed)
    sample = rng.sample(used_recs, min(sample_n, len(used_recs))) if used_recs else []
    chains = []
    for rec in sample:
        ups = rec.get("upstream")
        ups = json.loads(ups) if isinstance(ups, str) else (ups or {})
        chains.append({"sample_ts": str(rec["sample_ts"]), "dataset": rec.get("dataset"),
                       "decision_available_ts": str(rec.get("decision_available_ts")),
                       "window_start_ts": str(rec.get("window_start_ts")),
                       "upstream": {u: {**i, "folds": [{k: str(v) if v is not None else None
                                                         for k, v in fold_by.get((u, int(f)), {}).items()
                                                         if k in ("fold", "strategy", "train_start", "train_end",
                                                                  "train_max_label_end", "before_max_le", "after_min_ws")}
                                                        for f in i.get("folds_in_window") or [] if int(f) != -2]}
                                    for u, i in ups.items()}})
    return {"rules": rules, "passed": all(v["violations"] == 0 for v in rules.values()),
            "n_records": len(used_recs), "n_folds": len(folds), "frame_hashes": sorted(h for h in hashes if h),
            "violations": bad_rows[:200], "sample_chains": chains}


def violations_csv(report: dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["rule", "sample_ts", "dataset", "upstream", "fold", "why"])
    w.writeheader()
    for r in report["violations"]:
        w.writerow(r)
    return buf.getvalue()


def records_from_engine(job_records: dict, folds: dict) -> tuple[list, list]:
    """本機測試用：把 engine_v2 的輸出轉成與資料庫相同欄位的紀錄（經 v2_records 的列產生器，同一份格式）。"""
    from training.v2_records import COLUMNS, prediction_rows

    recs, out_folds = [], []
    for nid, entry in job_records.items():
        for row in prediction_rows(entry, job_id=None, node_id=nid, dataset="val"):
            d = dict(zip(COLUMNS, row))
            if d["upstream"] is not None:
                recs.append(d)
    for nid, info in folds.items():
        for f in info["folds"]:
            out_folds.append({**{k: (None if (isinstance(v, np.datetime64) and np.isnat(v)) else v) for k, v in f.items()},
                              "node_id": nid, "frame_hash": next(iter(job_records.values()))["frame_hash"]})
    return recs, out_folds


if __name__ == "__main__":
    # 獨立核對腳本：python -m training.cascade_audit <job_id> [--csv 檔名]。只讀 hermesnote DB 的逐筆紀錄與折次清單。
    import asyncio
    import sys

    from training import experiment_store

    async def _main(job_id: str, csv_path: str | None):
        data = await experiment_store.audit_inputs(job_id)
        rep = audit(data["records"], data["folds"])
        print(json.dumps({k: rep[k] for k in ("passed", "rules", "n_records", "n_folds", "frame_hashes")},
                         ensure_ascii=False, indent=2))
        for c in rep["sample_chains"]:
            print(json.dumps(c, ensure_ascii=False))
        if csv_path:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(violations_csv(rep))

    args = sys.argv[1:]
    asyncio.run(_main(args[0], args[args.index("--csv") + 1] if "--csv" in args else None))
