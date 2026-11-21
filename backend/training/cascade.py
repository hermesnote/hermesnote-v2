"""模型串接防洩漏（M5）：上游模型給下游當輸入時，預測由「上游預測策略」產生，不再用樣本內預測。

`upstream_policy.save_oof`（預設 false，2026-11-21）：true 時把上游每一列的樣本外預測（各折模型的 best 權重）
寫進逐筆結果（node_id＝上游節點、dataset="upstream_oof"、upstream 欄位記折次），供研究端讀取與核對。

策略（上游 Model Node 的 `upstream_policy`，未設定時用 settings.cascade 的預設 forward_chaining、K=5）：
  in_sample         舊行為：上游用框架的 train／val 訓練一次，對全部列預測（有洩漏，只為相容保留，任務會帶提示）
  forward_chaining  可預測的列依時間切成 K+1 段；第 k 段由「只用答案在第 k 段開始前（再扣清除範圍）就成熟的
                    訓練點」訓練的模型預測；第 0 段沒有預測（下游那些樣本記為 upstream_unavailable）
  purged_kfold      切成 K 段；第 k 段由「區間（窗格起點～答案可用）與第 k 段±清除範圍不重疊的訓練點」訓練的模型預測
  blending          前後兩段：前段訓練、後段預測（等於 K=1 的前進式），最終模型預設沿用該模型

共同規則：
  - 上游只用框架的 train 點訓練（驗證集、holdout 永遠不進上游訓練）；每個折內模型的早停／best 選擇用「折內」
    從訓練點再切出的驗證段（Phase 1 隨機、Phase 2 時序），不用框架的驗證集。
  - 可預測的列＝上游輸入有效的全部決策 K 棒（含期間前的暖機列、框架的 val 列）——下游的訓練與驗證樣本
    都用樣本外的折內預測；最終上游模型（refit_full 用全部訓練點重訓，或 last_fold）只用在 Phase 3 與一般推論。
  - 清除範圍（時間）自動＝下游最長（窗格起點→答案可用）跨度＋上游最長窗格跨度，研究者只能往上加
    （extra_purge_sec）。另外對每筆下游樣本逐筆檢查（check_downstream），違反就排除（upstream_purged）。
"""

import numpy as np

POLICIES = {
    "in_sample": {"label": "樣本內預測（舊行為，有洩漏，只供相容與對照）", "default_final": "refit_full"},
    "forward_chaining": {"label": "前進式：每段只用更早成熟的訓練點", "default_final": "refit_full"},
    "purged_kfold": {"label": "分塊交叉＋清除：每段用不重疊的其餘訓練點", "default_final": "refit_full"},
    "blending": {"label": "前後兩段：前段訓練、後段預測", "default_final": "last_fold"},
}
FINALS = {"refit_full": "用全部訓練點重新訓練", "last_fold": "沿用最後一折的模型"}
DEFAULT_POLICY = {"strategy": "forward_chaining", "k": 5}
FOLD_FINAL = -1
FOLD_NONE = -2


def resolve_policy(node: dict, settings: dict) -> dict:
    base = {**DEFAULT_POLICY, **((settings or {}).get("cascade") or {}), **(node.get("upstream_policy") or {})}
    strategy = base.get("strategy", "forward_chaining")
    k = 1 if strategy == "blending" else int(base.get("k", 5))
    return {"strategy": strategy, "k": k, "final": base.get("final") or POLICIES.get(strategy, {}).get("default_final"),
            "extra_purge_sec": float(base.get("extra_purge_sec", 0) or 0),
            "save_oof": bool(base.get("save_oof", False))}


def policy_issues(nid: str, node: dict, settings: dict) -> list[str]:
    raw = {**((settings or {}).get("cascade") or {}), **(node.get("upstream_policy") or {})}
    out = []
    if raw.get("strategy", "forward_chaining") not in POLICIES:
        out.append(f"Model 節點 {nid}：upstream_policy.strategy 必須是 {sorted(POLICIES)} 之一")
    k = raw.get("k", 5)
    if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= 20:
        if raw.get("strategy") != "blending":
            out.append(f"Model 節點 {nid}：upstream_policy.k 必須是 2～20 的整數")
    if raw.get("final") is not None and raw["final"] not in FINALS:
        out.append(f"Model 節點 {nid}：upstream_policy.final 必須是 {sorted(FINALS)} 之一")
    if not isinstance(raw.get("save_oof", False), bool):
        out.append(f"Model 節點 {nid}：upstream_policy.save_oof 必須是 true／false")
    eps = raw.get("extra_purge_sec", 0)
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps < 0:
        out.append(f"Model 節點 {nid}：upstream_policy.extra_purge_sec 必須是 ≥ 0 的秒數")
    return out


def _blocks(rows: np.ndarray, tau: np.ndarray, n_blocks: int) -> list[np.ndarray]:
    order = rows[np.argsort(tau[rows], kind="stable")]
    return [b for b in np.array_split(order, n_blocks)]


def fold_plan(policy: dict, *, predict_rows: np.ndarray, train_rows: np.ndarray, ws: np.ndarray, tau: np.ndarray,
              le: np.ndarray, purge: np.timedelta64) -> list[dict]:
    """回傳每一折：{fold, train_rows, predict_rows, purge_start, purge_end}。時間單位 datetime64[ns]。
    ws／tau／le：每一列的窗格起點、決策時刻、答案可用時間（沒有答案的列 le 為 NaT，不會被拿來訓練）。"""
    s = policy["strategy"]
    if s == "in_sample":
        return []
    folds = []
    if s in ("forward_chaining", "blending"):
        blocks = _blocks(predict_rows, tau, policy["k"] + 1)
        for k in range(1, len(blocks)):
            blk = blocks[k]
            if len(blk) == 0:
                continue
            start = ws[blk].min()
            cut = start - purge
            tr = train_rows[le[train_rows] < cut]
            folds.append({"fold": k, "train_rows": tr, "predict_rows": blk, "purge_start": cut, "purge_end": start})
        return folds
    blocks = _blocks(predict_rows, tau, policy["k"])
    for k, blk in enumerate(blocks, start=1):
        if len(blk) == 0:
            continue
        g0, g1 = ws[blk].min() - purge, tau[blk].max() + purge
        tr = train_rows[(le[train_rows] < g0) | (ws[train_rows] > g1)]
        folds.append({"fold": k, "train_rows": tr, "predict_rows": blk, "purge_start": g0, "purge_end": g1})
    return folds


def fold_bounds(train_rows: np.ndarray, ws: np.ndarray, tau: np.ndarray, le: np.ndarray, gap: tuple | None) -> dict:
    """一折訓練點的範圍摘要（存進 model_upstream_folds，核對腳本據此重算）。
    before_max_le：清除區間之前的訓練點中最晚的答案可用時間；after_min_ws：清除區間之後的訓練點中最早的窗格起點。"""
    if len(train_rows) == 0:
        return {"n_train": 0, "train_start": None, "train_end": None, "train_max_label_end": None,
                "before_max_le": None, "after_min_ws": None}
    w, l, t = ws[train_rows], le[train_rows], tau[train_rows]
    if gap is None:
        before, after = np.ones(len(train_rows), bool), np.zeros(len(train_rows), bool)
    else:
        before, after = l < gap[0], w > gap[1]
    return {"n_train": int(len(train_rows)), "train_start": w.min(), "train_end": t.max(), "train_max_label_end": l.max(),
            "before_max_le": l[before].max() if before.any() else None,
            "after_min_ws": w[after].min() if after.any() else None}


def inner_split(rows: np.ndarray, start_ts: np.ndarray, target_ts: np.ndarray, strategy: str, val_ratio: float,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """折內（或最終上游模型）的訓練／驗證切分，只從框架的 train 點切。start_ts：每列 K 棒起點；
    時序切分時，答案 K 棒（target_ts）跨進驗證段起點的訓練點排除（同框架的 boundary 規則）。"""
    rows = rows[np.argsort(start_ts[rows], kind="stable")]
    n = len(rows)
    v = max(1, int(n * float(val_ratio))) if n > 1 else 0
    if strategy == "chronological":
        val, cand = rows[n - v:], rows[:n - v]
        if len(val):
            cand = cand[target_ts[cand] < start_ts[val[0]]]
        return cand, val
    perm = np.random.default_rng(int(seed)).permutation(n)
    return rows[perm[v:]], rows[perm[:v]]


def auto_purge(downstream_span: np.timedelta64, upstream_span: np.timedelta64, extra_sec: float) -> np.timedelta64:
    return downstream_span + upstream_span + np.timedelta64(int(extra_sec), "s")


def folds_in_window(fold_rows: np.ndarray, lo: np.ndarray, e: np.ndarray, fold_ids) -> dict:
    """每個時間點的窗格（分支列 lo..e）裡出現了哪些折：{fold_id: (n_points,) bool}。"""
    out = {}
    for f in fold_ids:
        cum = np.r_[0, np.cumsum(fold_rows == f)]
        out[f] = (cum[e + 1] - cum[lo]) > 0
    return out


def check_downstream(*, is_train: np.ndarray, ws: np.ndarray, tau: np.ndarray, le: np.ndarray, used: dict,
                     folds: dict, strategy: str) -> np.ndarray:
    """逐筆防洩漏檢查（規則 2、3、4），回傳違反的遮罩。
    used：{fold_id: (n,) bool}＝樣本窗格內用到該折的預測；folds：{fold_id: fold_bounds 摘要}。
      訓練樣本：用到的每一折，其訓練點區間不得與樣本區間 [ws, le] 重疊（前進式另需最晚答案可用 < ws）。
      驗證／holdout 樣本：用到的每一折，其訓練點不得涵蓋樣本的決策時刻 tau。"""
    bad = np.zeros(len(tau), dtype=bool)
    for f, mask in used.items():
        if f in (FOLD_NONE,) or not mask.any():
            continue
        fb = folds.get(f)
        if fb is None:
            bad |= mask
            continue
        bmax = fb.get("before_max_le")
        amin = fb.get("after_min_ws")
        b = np.datetime64(bmax) if bmax is not None else None
        a = np.datetime64(amin) if amin is not None else None
        tr_bad = np.zeros(len(tau), dtype=bool)
        if b is not None:
            tr_bad |= ~(b < ws)
        if a is not None:
            tr_bad |= ~(a > le)
        va_bad = np.zeros(len(tau), dtype=bool)
        if b is not None:
            va_bad |= ~(b < tau)
        if a is not None:
            va_bad |= ~(a > tau)
        bad |= mask & np.where(is_train, tr_bad, va_bad)
    return bad
