"""v2 時間對齊（M4）：以 bar_start_ts 為識別時間、以 bar_end_ts＋發布延遲判定可用。

規則（所有時間框架同一條）：
  決策時刻 τ        ＝決策 K 棒的 bar_end_ts＋決策時間框架的發布延遲。
  其他 K 棒可用      ⇔ 它的 bar_end_ts＋它自己的發布延遲 ≤ τ（收盤時間等於決策時刻即可用）。
  as-of             取「可用」的最後一根；資料缺了就沿用最後一根已收完的，並記錄距離決策時刻多久。
  子 K 棒彙總        較細時間框架匯入較粗分支：只用「起點 ≥ 分支 K 棒起點、bar_end ≤ 分支 K 棒 bar_end、且
                    已可用」的子 K 棒，彙總方式 last／mean／sum／std／max／min；有任何一根是 NaN 結果就是 NaN。
  沿用前一時段       allow_previous_session（預設 true）／max_sessions_back（預設 1，0～5）／max_age_sec（預設不限）。
                    「往前幾個時段」用分支自己的交易時段曆計算：分支最後採用的 K 棒所屬時段之後，到決策時刻為止
                    又開始了幾個該分支的時段。超過上限 → 該時間點標成 stale_branch（不產生樣本、記錄原因）。
發布延遲預設 0 秒，狀態 assumed（未實測假設）；延遲依「資料來源×時間框架」設定。
"""

import numpy as np
import pandas as pd

INTRADAY_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60}
# 每個交易日的 K 棒數（換算不同時間框架的歷史根數用；min＝只有日盤、max＝日盤＋夜盤）
BARS_PER_DAY_MIN = {"1m": 300, "5m": 60, "15m": 20, "30m": 10, "60m": 5, "daily_day": 1, "daily_full": 1}
BARS_PER_DAY_MAX = {"1m": 1140, "5m": 228, "15m": 76, "30m": 38, "60m": 20, "daily_day": 1, "daily_full": 1}
AGGREGATIONS = ("last", "mean", "sum", "std", "max", "min")

DEFAULT_ALIGNMENT = {
    "availability_source": "bar_end_ts",
    "delays": {},  # {"tx_60m": {"sec": 0, "status": "assumed", "note": ""}}
    "allow_previous_session": True,
    "max_sessions_back": 1,
    "max_age_sec": None,
}


def tf_part(tf: str) -> str:
    """'tx_15m' → '15m'。"""
    return tf.split("_", 1)[1] if "_" in tf else tf


def resolve_alignment(settings: dict | None) -> dict:
    a = {**DEFAULT_ALIGNMENT, **((settings or {}).get("alignment") or {})}
    a["delays"] = dict(a.get("delays") or {})
    return a


def delay_sec(alignment: dict, tf: str) -> int:
    d = (alignment.get("delays") or {}).get(tf)
    if d is None:
        return 0
    return int(d.get("sec", 0) if isinstance(d, dict) else d)


def delay_report(alignment: dict, tfs) -> dict:
    out = {}
    for tf in sorted(tfs):
        d = (alignment.get("delays") or {}).get(tf)
        if d is None:
            out[tf] = {"sec": 0, "status": "assumed", "note": "預設 0 秒，未實測"}
        elif isinstance(d, dict):
            out[tf] = {"sec": int(d.get("sec", 0)), "status": d.get("status", "assumed"), "note": d.get("note", "")}
        else:
            out[tf] = {"sec": int(d), "status": "assumed", "note": ""}
    return out


def alignment_issues(alignment: dict) -> list[str]:
    out = []
    if not isinstance(alignment.get("allow_previous_session"), bool):
        out.append("alignment.allow_previous_session 必須是 true／false")
    m = alignment.get("max_sessions_back")
    if m is not None and (isinstance(m, bool) or not isinstance(m, int) or not 0 <= m <= 5):
        out.append("alignment.max_sessions_back 必須是 0～5 的整數或 null（不限）")
    age = alignment.get("max_age_sec")
    if age is not None and (isinstance(age, bool) or not isinstance(age, (int, float)) or age <= 0):
        out.append("alignment.max_age_sec 必須是正數（秒）或 null（不限）")
    if alignment.get("availability_source", "bar_end_ts") != "bar_end_ts":
        out.append("alignment.availability_source 目前只支援 bar_end_ts")
    for tf, d in (alignment.get("delays") or {}).items():
        sec = d.get("sec", 0) if isinstance(d, dict) else d
        if isinstance(sec, bool) or not isinstance(sec, (int, float)) or sec < 0:
            out.append(f"alignment.delays[{tf}].sec 必須是 ≥ 0 的秒數")
        if isinstance(d, dict) and d.get("status", "assumed") not in ("assumed", "measured"):
            out.append(f"alignment.delays[{tf}].status 必須是 assumed 或 measured")
    return out


class TfData:
    """一個時間框架的資料（v2）：依 bar_start_ts 排序、唯一的 DatetimeIndex，附可用時間與交易時段。"""

    def __init__(self, tf: str, df: "pd.DataFrame", delay: int = 0):
        self.tf = tf
        self.delay = int(delay)
        d = df
        if "bar_start_ts" in d.columns:
            start = pd.to_datetime(d["bar_start_ts"])
        elif "datetime" in d.columns:
            start = pd.to_datetime(d["datetime"])
        elif isinstance(d.index, pd.DatetimeIndex):
            start = pd.Series(d.index, index=d.index)
        else:
            raise ValueError(f"{tf}：資料沒有 bar_start_ts／datetime，無法做時間對齊")
        start = pd.DatetimeIndex(np.asarray(start))
        if start.has_duplicates:
            from services.quotes import QuoteIntegrityError, duplicate_details

            raise QuoteIntegrityError(f"{tf} 有重複 bar_start_ts：" + "；".join(duplicate_details(start, "bar_start_ts")))
        if "bar_end_ts" in d.columns and not pd.isna(d["bar_end_ts"]).any():
            end = pd.DatetimeIndex(np.asarray(pd.to_datetime(d["bar_end_ts"])))
            self.end_source = "bar_end_ts"
        else:
            end = start + _nominal_period(tf)
            self.end_source = "nominal_period"  # 測試資料或缺欄位：依名目週期推算（讀取報告會註明）
        order = np.argsort(start.values, kind="stable")
        self.index = start[order]
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in d.columns]
        self.ohlcv = pd.DataFrame({c: np.asarray(d[c], dtype=float)[order] for c in cols}, index=self.index)
        self.end = np.asarray(end.values)[order]
        self.available = self.end + np.timedelta64(self.delay, "s")
        if np.any(np.diff(self.available) < np.timedelta64(0, "s")):
            raise ValueError(f"{tf}：bar_end_ts 不是隨 bar_start_ts 遞增，無法做 as-of 對齊")
        self.session_start, self.session_ord = _sessions(tf, d, order, self.index)

    def __len__(self):
        return len(self.index)


def _nominal_period(tf: str) -> "pd.Timedelta":
    part = tf_part(tf)
    if part in INTRADAY_MINUTES:
        return pd.Timedelta(minutes=INTRADAY_MINUTES[part])
    return pd.Timedelta(hours=5)  # 日線：08:45 開、13:45 收


def _sessions(tf: str, d, order, index):
    """每根 K 棒所屬交易時段的序號，以及每個時段的開始時間（排序後）。資料有 trading_day／session 就用它，
    沒有（測試資料）就依時間推定：08:45～13:45 為日盤，其餘為夜盤，夜盤 15:00 以後歸下一個交易日。"""
    if "trading_day" in d.columns and "session" in d.columns:
        td = pd.to_datetime(np.asarray(d["trading_day"])[order]).values
        ss = np.asarray(d["session"]).astype(str)[order]
    else:
        t = index
        minutes = t.hour * 60 + t.minute
        is_day = (minutes >= 8 * 60 + 45) & (minutes < 13 * 60 + 45)
        ss = np.where(is_day, "day", "night")
        td = (t.normalize() + pd.to_timedelta(np.where((~is_day) & (minutes >= 15 * 60), 1, 0), unit="D")).values
    keys = pd.MultiIndex.from_arrays([td, ss])
    codes, uniques = pd.factorize(keys, sort=False)
    # 以時段內最早的 bar_start 排序時段
    first = pd.Series(index.values).groupby(codes).min()
    rank = first.sort_values(kind="stable")
    ord_of_code = pd.Series(np.arange(len(rank)), index=rank.index)
    session_ord = ord_of_code.reindex(codes).to_numpy()
    return rank.to_numpy(), session_ord


def asof_positions(branch: TfData, tau: np.ndarray) -> np.ndarray:
    """對每個決策時刻 τ，branch 上「可用時間 ≤ τ」的最後一根位置；沒有就是 -1。"""
    return np.searchsorted(branch.available, tau, side="right") - 1


def staleness(branch: TfData, pos: np.ndarray, tau: np.ndarray, alignment: dict) -> dict:
    """pos 為 asof_positions 的結果（-1 表示沒有）。回傳 age_sec、sessions_back、stale 遮罩。"""
    ok = pos >= 0
    p = np.where(ok, pos, 0)
    age = (tau - branch.available[p]) / np.timedelta64(1, "s")
    started = np.searchsorted(branch.session_start, tau, side="right") - 1
    back = started - branch.session_ord[p]
    stale = np.zeros(len(tau), dtype=bool)
    if not alignment.get("allow_previous_session", True):
        stale |= back > 0
    elif alignment.get("max_sessions_back") is not None:
        stale |= back > int(alignment["max_sessions_back"])
    if alignment.get("max_age_sec") is not None:
        stale |= age > float(alignment["max_age_sec"])
    stale &= ok
    return {"age_sec": np.where(ok, age, np.nan), "sessions_back": np.where(ok, back, -1), "stale": stale}


def asof_values(src_index_available: np.ndarray, src_values: np.ndarray, tau: np.ndarray) -> tuple:
    """把來源（可用時間遞增）的值 as-of 對到 τ；回傳 (values, positions)，沒有可用值的列是 NaN／-1。"""
    pos = np.searchsorted(src_index_available, tau, side="right") - 1
    out = np.full((len(tau), src_values.shape[1]), np.nan)
    ok = pos >= 0
    out[ok] = src_values[pos[ok]]
    return out, pos


def aggregate_subbars(fine: TfData, fine_values: np.ndarray, coarse: TfData, method: str) -> np.ndarray:
    """較細時間框架的值彙總到較粗分支的每一根 K 棒（見模組說明）。回傳 (values, present)：present＝有子 K 棒。"""
    if method not in AGGREGATIONS:
        raise ValueError(f"子 K 棒彙總方式必須是 {AGGREGATIONS} 之一，收到 {method!r}")
    k = fine_values.shape[1]
    out = np.full((len(coarse), k), np.nan)
    present = np.zeros(len(coarse), dtype=bool)
    if len(fine) == 0 or len(coarse) == 0:
        return out, present
    owner = np.searchsorted(coarse.index.values, fine.index.values, side="right") - 1
    valid = owner >= 0
    o = np.where(valid, owner, 0)
    valid &= fine.end <= coarse.end[o]
    valid &= fine.available <= coarse.available[o]
    if not valid.any():
        return out, present
    own, vals = owner[valid], np.asarray(fine_values[valid], dtype=float)
    starts = np.r_[0, np.nonzero(np.diff(own))[0] + 1]
    groups = own[starts]
    counts = np.diff(np.r_[starts, len(own)])[:, None].astype(float)
    nan_cnt = np.add.reduceat(np.isnan(vals).astype(np.int64), starts, axis=0)
    filled = np.nan_to_num(vals)
    if method == "last":
        r = vals[np.r_[starts[1:], len(own)] - 1]
    elif method in ("sum", "mean", "std"):
        total = np.add.reduceat(filled, starts, axis=0)
        r = total if method == "sum" else total / counts
        if method == "std":
            sq = np.add.reduceat(filled ** 2, starts, axis=0) / counts
            r = np.sqrt(np.clip(sq - r ** 2, 0, None))
    else:
        r = (np.maximum if method == "max" else np.minimum).reduceat(filled, starts, axis=0)
    r = np.where(nan_cnt > 0, np.nan, r)
    out[groups] = r
    present[groups] = True
    return out, present


def is_coarser(tf_a: str, tf_b: str) -> bool:
    """tf_a 的週期是否比 tf_b 長。"""
    order = ["1m", "5m", "15m", "30m", "60m", "daily_day", "daily_full"]
    a, b = tf_part(tf_a), tf_part(tf_b)
    return order.index(a) > order.index(b)


def bars_needed_in(target_tf: str, bars: int, of_tf: str) -> int:
    """of_tf 上 `bars` 根的時間跨度，換算成 target_tf 需要的根數（保守估計，另加 2 根餘裕）。"""
    a, b = tf_part(target_tf), tf_part(of_tf)
    ratio = BARS_PER_DAY_MAX[a] / BARS_PER_DAY_MIN[b]
    return int(np.ceil(bars * ratio)) + 2
