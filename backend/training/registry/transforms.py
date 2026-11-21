"""特徵轉換元件（M2）：套在 Feature Node 算出來的每一欄上，結果仍是 (n, k)。

  diff               x_t − x_{t−1}                         額外暖機 1
  pct_return         x_t / x_{t−1} − 1（分母 ≤ 0 時 NaN）    額外暖機 1
  log_return         ln(x_t / x_{t−1})（非正數時 NaN）       額外暖機 1
  rolling_sum        最近 N 根增量的加總＝x_t − x_{t−N}      額外暖機 N（對 OBV 等於最近 N 根的有號成交量和，與起點無關）
  fixed_anchor       x_t − x_a，a＝錨點日之前最後一根；錨點以前為 NaN。對 OBV／AD 等於「從錨點開始累積」，
                     只要資料載入起點不晚於錨點，訓練與推論完全一致（推論需載入錨點以後全部資料）
  legacy_cumulative  不轉換、從載入起點累積（舊行為，只供對照；結果與載入起點有關，任務會帶提示）
  none               不轉換

轉換在暖機遮罩之後套用：前一步是 NaN 的列，轉換後也是 NaN，不會把暖機期的數值帶進來。
"""

import numpy as np

TRANSFORMS = {
    "none": {"label": "不轉換", "extra_warmup": 0, "params": []},
    "diff": {"label": "差分 x_t − x_{t−1}", "extra_warmup": 1, "params": []},
    "pct_return": {"label": "報酬 x_t / x_{t−1} − 1", "extra_warmup": 1, "params": []},
    "log_return": {"label": "對數報酬 ln(x_t / x_{t−1})", "extra_warmup": 1, "params": []},
    "rolling_sum": {"label": "滾動 N 根增量加總 x_t − x_{t−N}", "extra_warmup": None,
                    "params": [{"name": "window", "type": "int", "min": 1, "required": True}]},
    "fixed_anchor": {"label": "固定錨點起算 x_t − x_錨點", "extra_warmup": 0,
                     "params": [{"name": "anchor", "type": "date", "required": True}]},
    "legacy_cumulative": {"label": "維持舊行為（從載入起點累積，結果與起點有關）", "extra_warmup": 0, "params": []},
}
# 累積型指標可接受的轉換（pct／log 報酬依賴絕對水位，對累積量沒有意義）
CUMULATIVE_OK = {"diff", "rolling_sum", "fixed_anchor", "legacy_cumulative"}


def list_available() -> list[dict]:
    return [{"key": k, **v, "cumulative_ok": k in CUMULATIVE_OK} for k, v in TRANSFORMS.items()]


def extra_warmup(spec: dict | None) -> int:
    spec = spec or {"type": "none"}
    t = spec.get("type", "none")
    if t == "rolling_sum":
        return int(spec["window"])
    return int(TRANSFORMS[t]["extra_warmup"])


def validate(spec, memory_type: str) -> list[str]:
    if spec is None:
        return ["累積型指標必須選擇 transform（diff／rolling_sum／fixed_anchor／legacy_cumulative）"] \
            if memory_type == "cumulative" else []
    if not isinstance(spec, dict) or spec.get("type") not in TRANSFORMS:
        return [f"transform.type 必須是 {sorted(TRANSFORMS)} 之一，收到 {spec!r}"]
    t = spec["type"]
    out = []
    if t == "rolling_sum":
        w = spec.get("window")
        if isinstance(w, bool) or not isinstance(w, int) or w < 1:
            out.append("transform rolling_sum 需要 window（≥ 1 的整數）")
    if t == "fixed_anchor":
        import datetime as _dt

        try:
            _dt.date.fromisoformat(str(spec.get("anchor")))
        except ValueError:
            out.append("transform fixed_anchor 需要 anchor（YYYY-MM-DD）")
    if memory_type == "cumulative" and t not in CUMULATIVE_OK:
        out.append(f"累積型指標不能用 transform {t!r}，可用：{sorted(CUMULATIVE_OK)}")
    return out


def apply(values: np.ndarray, spec: dict | None, index=None) -> np.ndarray:
    """values：(n, k) 浮點陣列（暖機遮罩已套用）；index：DatetimeIndex（fixed_anchor 需要）。"""
    spec = spec or {"type": "none"}
    t = spec.get("type", "none")
    x = np.asarray(values, dtype=float)
    if t in ("none", "legacy_cumulative"):
        return x
    out = np.full_like(x, np.nan)
    if t == "diff":
        out[1:] = x[1:] - x[:-1]
    elif t in ("pct_return", "log_return"):
        prev, cur = x[:-1], x[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            if t == "pct_return":
                r = np.where(prev > 0, cur / prev - 1.0, np.nan)
            else:
                r = np.where((prev > 0) & (cur > 0), np.log(cur / prev), np.nan)
        out[1:] = r
    elif t == "rolling_sum":
        w = int(spec["window"])
        if len(x) > w:
            out[w:] = x[w:] - x[:-w]
    elif t == "fixed_anchor":
        import pandas as pd

        if index is None:
            raise ValueError("fixed_anchor 需要時間索引")
        anchor = pd.Timestamp(spec["anchor"])
        pos = int(np.searchsorted(np.asarray(index.values), np.datetime64(anchor), side="left")) - 1
        if pos < 0:
            raise ValueError(f"fixed_anchor：資料載入起點 {index[0]} 晚於或等於錨點 {anchor.date()}，"
                             "無法從錨點起算（載入需涵蓋錨點前一根）")
        out[pos + 1:] = x[pos + 1:] - x[pos]
    else:
        raise ValueError(f"未登記的 transform {t!r}")
    return out
