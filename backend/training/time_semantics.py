"""K 棒「識別時間」與「資訊可用時間」的區分。

行情資料表每根 K 棒有兩個時間：
  datetime    識別時間（棒起點）。對齊、切分、時間戳交集、樣本 `ts` 一律用它。
  bar_end_ts  資訊可用時間（棒終點）。這根棒完整的 OHLCV 最早只能在這個時間之後才用來決策。

日線最明顯：`daily_day`／`daily_full` 的 datetime 固定是 08:45，收盤 bar_end_ts 是 13:45——
完整日棒的資訊不能被當成 08:45 就已知。所以樣本的「決策時刻」不能只標識別時間，要另外帶
`available_ts`：這個樣本的輸入（最後一根棒）的資訊最早何時可用。

規則（訓練、預覽、推論共用這一份，避免各處定義不同）：
  一個樣本的 available_ts = 這個節點「全部輸入」，各自在「樣本最後一根棒」的可用時間的最大值（最晚者）。
  - Feature 輸入：該根棒的 bar_end_ts。
  - 上游 Model Node 輸入：沿依賴鏈傳遞——上游樣本自己的 available_ts（也就是它全部輸入的最晚可用時間，
    見 model_available_map），不是只看直接的 Feature。
  - 任何一個輸入的可用時間未知（資料沒有 bar_end_ts、上游未知、對不到時間點）→ 整個 available_ts 是 None，
    不猜、不用其餘輸入的時間頂替。
  - 目標的可用時間（target_available_ts）取自 **Label 節點自己資料**的目標那根棒的 bar_end_ts，
    跟 Feature 的時間無關（Label 的資料表可以比 Feature 晚才可用）。
這只是「標時間」用，不參與任何計算與切分，不會改變訓練資料。
"""

import numpy as np
import pandas as pd


def build_available_map(df: "pd.DataFrame"):
    """回傳 index 為識別時間、值為 bar_end_ts 的 Series；資料沒有 bar_end_ts 欄位就回 None。
    df 必須已經是 `_ensure_time_indexed` 處理過的（DatetimeIndex、去重、排序）。"""
    if "bar_end_ts" not in df.columns:
        return None
    end = pd.to_datetime(df["bar_end_ts"])
    if end.isna().any():
        return None  # 整欄只要有一列沒有值就當作未知，不猜、不補
    return pd.Series(end.to_numpy(), index=df.index)


def model_available_map(sample_ts, sample_available):
    """上游 Model Node 的輸出時間對應表：index 是該模型的樣本時間（輸入最後一根棒的識別時間），
    值是該樣本的資訊可用時間（模型自己所有輸入的最晚可用時間）。模型自己的可用時間未知就回 None，
    下游據此也視為未知。"""
    if sample_available is None:
        return None
    return pd.Series(np.asarray(sample_available), index=pd.DatetimeIndex(sample_ts))


def available_at(maps: list, common_ts: np.ndarray):
    """對 common_ts 每個時間點，取所有 Feature 輸入 bar_end_ts 的最大值；回傳 datetime64 陣列，
    任何一個輸入沒有 bar_end_ts、或沒有 Feature 輸入時回 None。"""
    if not maps or any(m is None for m in maps):
        return None
    idx = pd.DatetimeIndex(common_ts)
    stacked = np.stack([m.reindex(idx).to_numpy() for m in maps], axis=0)
    if np.isnat(stacked).any():
        return None  # 對不到 bar_end_ts 的列不猜，整組不標
    return stacked.max(axis=0)


def to_unix(value) -> int | None:
    """跟既有 decision_ts 相同的轉法（naive 時間戳當 UTC 秒），None 原樣回傳。"""
    if value is None:
        return None
    return int(pd.Timestamp(value).timestamp())
