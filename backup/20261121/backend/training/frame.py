"""共同樣本框架（M2）：實驗層級的「比較時間點清單＋只做一次的切分」，所有組別、分支、上下游共用。

時間點＝決策時間框架（＝Label 的時間框架）在期間 [start, end) 內的每一根 K 棒（以 bar_start_ts 識別）。
每個時間點的狀態：
  train／val       Phase 1（random，切分種子在實驗層）、Phase 2（chronological，最後 val_ratio 為驗證集）
  holdout          Phase 3（測試期間）
  excluded＋原因   見 REASONS

框架與組別無關：組別各自的暖機往期間之前取資料；某組別在某時間點湊不到輸入（歷史不足、暖機未完成、
資料過舊…）時，那個組別在那個時間點不產生樣本並記錄原因（見 training/v2_inputs.py），框架本身不變。
雜湊＝定義（JSON）＋時間點＋狀態，用來確認各組別真的用同一份框架；不同雜湊的結果系統拒絕比較。
"""

import hashlib
import json

import numpy as np
import pandas as pd

REASONS = {
    "label_unavailable": "答案需要的未來 K 棒超出期間（期間尾端），不產生樣本",
    "label_unmatured": "答案尚未成熟（測試期間尾端）：照常預測與保存，但不納入評估",
    "boundary": "時序切分：訓練樣本的答案 K 棒跨進驗證集起點，排除（避免用到驗證期間的價格）",
    "outside_frame": "上游時間點對不到下游框架（在框架最後一個時間點之後）",
    "mapped_excluded": "上游時間點對應到的下游框架時間點本身被排除",
    "insufficient_history": "這個分支湊不到 window 根已可用的 K 棒（資料起點太晚或資料缺漏）",
    "warmup_not_ready": "窗格內有特徵還在暖機期（數值為 NaN）",
    "stale_branch": "分支最後一根可用 K 棒超過沿用前一時段／過舊上限的設定",
    "missing_availability": "對不到可用時間（缺 bar_end_ts）",
    "upstream_unavailable": "上游模型在這個時間點沒有樣本外預測（例如前進式第一段、上游輸入不足）",
    "upstream_purged": "上游預測所用模型的訓練區間與這筆樣本重疊（清除規則排除）",
}
INCLUDED = ("train", "val", "holdout")


class Frame:
    def __init__(self, definition: dict, ts, tau, target_ts, target_avail, status, reason, y=None):
        self.definition = definition
        self.ts = np.asarray(ts, dtype="datetime64[ns]")
        self.tau = np.asarray(tau, dtype="datetime64[ns]")
        self.target_ts = np.asarray(target_ts, dtype="datetime64[ns]")
        self.target_avail = np.asarray(target_avail, dtype="datetime64[ns]")
        self.status = np.asarray(status, dtype=object)
        self.reason = np.asarray(reason, dtype=object)
        self.y = None if y is None else np.asarray(y, dtype=float)
        self.hash = frame_hash(definition, self.ts, self.status, self.reason)

    def __len__(self):
        return len(self.ts)

    def counts(self) -> dict:
        out = {s: int((self.status == s).sum()) for s in ("train", "val", "holdout", "excluded")}
        reasons = pd.Series(self.reason[self.reason != None]).value_counts()  # noqa: E711
        out["excluded_reasons"] = {str(k): int(v) for k, v in reasons.items()}
        out["n_points"] = len(self)
        return out

    def summary(self) -> dict:
        return {"hash": self.hash, "definition": self.definition, "counts": self.counts(),
                "first_ts": str(self.ts[0]) if len(self) else None, "last_ts": str(self.ts[-1]) if len(self) else None}

    def index_of(self, status: str) -> np.ndarray:
        return np.nonzero(self.status == status)[0]


def frame_hash(definition: dict, ts, status, reason) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(definition, sort_keys=True, ensure_ascii=False, default=str).encode())
    h.update(np.asarray(ts, dtype="datetime64[ns]").astype("int64").tobytes())
    h.update("\n".join(f"{s}|{r or ''}" for s, r in zip(status, reason)).encode())
    return h.hexdigest()


def split_definition(phase: int | None, split_strategy: str, val_ratio: float, split_seed: int) -> dict:
    return {"phase": phase, "split_strategy": split_strategy, "val_ratio": float(val_ratio),
            "split_seed": int(split_seed) if split_strategy == "random" else None}


def build_frame(dec, y: np.ndarray, horizon: int, start, end, *, phase: int | None, split_strategy: str,
                val_ratio: float, split_seed: int, definition_extra: dict | None = None) -> Frame:
    """dec：決策時間框架的 TfData（已含往前載入的暖機資料）；y：dec 每一列的學習目標（尾端 NaN）。"""
    if split_strategy not in ("random", "chronological", "holdout"):
        raise ValueError(f"split_strategy 必須是 random／chronological／holdout，收到 {split_strategy!r}")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    idx = dec.index.values
    rows = np.nonzero((idx >= np.datetime64(start)) & (idx < np.datetime64(end)))[0]
    n = len(rows)
    tgt = rows + int(horizon)
    has_tgt = tgt < len(dec)
    tg = np.where(has_tgt, tgt, 0)
    target_ts = np.where(has_tgt, idx[tg], np.datetime64("NaT"))
    target_avail = np.where(has_tgt, dec.available[tg], np.datetime64("NaT"))
    y_pts = np.asarray(y, dtype=float)[rows] if n else np.zeros(0)
    labeled = ~np.isnan(y_pts) & has_tgt
    status = np.full(n, "excluded", dtype=object)
    reason = np.full(n, None, dtype=object)

    if split_strategy == "holdout":
        status[:] = "holdout"
        reason[~labeled] = "label_unmatured"
    else:
        reason[~labeled] = "label_unavailable"
        elig = np.nonzero(labeled)[0]
        val_size = max(1, int(len(elig) * float(val_ratio))) if len(elig) else 0
        if split_strategy == "random":
            perm = np.random.default_rng(int(split_seed)).permutation(len(elig))
            val_pos = elig[perm[:val_size]]
            train_pos = elig[perm[val_size:]]
            status[val_pos] = "val"
            status[train_pos] = "train"
        else:
            val_pos = elig[len(elig) - val_size:]
            cand = elig[:len(elig) - val_size]
            status[val_pos] = "val"
            if len(val_pos):
                val_start = idx[rows[val_pos[0]]]
                ok = target_ts[cand] < val_start
                status[cand[ok]] = "train"
                reason[cand[~ok]] = "boundary"
            else:
                status[cand] = "train"

    definition = {
        "decision_timeframe": dec.tf, "start": str(start.date()), "end": str(end.date()), "horizon": int(horizon),
        **split_definition(phase, "holdout" if split_strategy == "holdout" else split_strategy, val_ratio, split_seed),
        **(definition_extra or {}),
    }
    return Frame(definition, idx[rows], dec.available[rows], target_ts, target_avail, status, reason,
                 y=y_pts)


def derive_frame(base: Frame, dec, y: np.ndarray, horizon: int, start, end, *, definition_extra: dict | None = None) -> Frame:
    """上游模型（決策時間框架或答案定義跟下游不同）用的框架：時間點是上游自己的 K 棒，狀態依時間對到下游框架
    ——第一個決策時刻 ≥ 上游決策時刻的下游時間點是什麼狀態，上游就是什麼狀態；時序切分另外排除答案 K 棒跨進
    驗證起點的上游訓練點。這樣上游的訓練點永遠落在下游的訓練範圍，驗證集與 holdout 不會進上游訓練。"""
    strategy = base.definition.get("split_strategy")
    own = build_frame(dec, y, horizon, start, end, phase=base.definition.get("phase"),
                      split_strategy="holdout" if strategy == "holdout" else "chronological",
                      val_ratio=base.definition.get("val_ratio", 0.2), split_seed=0,
                      definition_extra=definition_extra)
    status = np.full(len(own), "excluded", dtype=object)
    reason = np.array(own.reason, dtype=object)
    j = np.searchsorted(base.tau, own.tau, side="left")
    inside = j < len(base)
    jj = np.where(inside, j, 0)
    mapped = np.where(inside, base.status[jj], "excluded")
    labeled = reason == None  # noqa: E711
    if strategy == "holdout":
        status[:] = np.where(inside, "holdout", "excluded")
        reason[~inside] = "outside_frame"
    else:
        ok = labeled & inside & np.isin(mapped, INCLUDED)
        status[ok] = mapped[ok]
        reason[labeled & ~inside] = "outside_frame"
        reason[labeled & inside & ~np.isin(mapped, INCLUDED)] = "mapped_excluded"
        val_idx = base.index_of("val")
        if strategy == "chronological" and len(val_idx):
            val_start = base.ts[val_idx[0]]
            bad = (status == "train") & ~(own.target_ts < val_start)
            status[bad] = "excluded"
            reason[bad] = "boundary"
    definition = {**own.definition, "derived_from": base.hash, "split_strategy": strategy}
    return Frame(definition, own.ts, own.tau, own.target_ts, own.target_avail, status, reason, y=own.y)
