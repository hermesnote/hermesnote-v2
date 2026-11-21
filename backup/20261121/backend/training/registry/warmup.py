"""特徵的記憶類型與暖機建議（M2，流程版本 2026-11-r2 起的 v2 圖使用）。

記憶類型決定「這個特徵在某一根 K 棒的值，跟資料從哪裡開始載入有沒有關係」：
  none        只用當根資料（ohlcv／raw_price／raw_volume）。暖機 0。
  finite      只回看固定根數（SMA、BBANDS（SMA 平均）、STOCH（SMA 平均）…）。暖機＝TA-Lib lookback，
              之後的值跟載入起點完全無關。
  recursive   單一 EMA／Wilder 平滑（EMA、RSI、ATR…），初始值的影響以 (1-α)^k 衰減。
              暖機＝lookback＋ceil(ln ε／ln(1-α))，ε 預設 0.01（殘留權重 < 1%）；之後的值只保證在門檻內一致。
  composite   多層遞迴或自適應（MACD、ADX、KAMA、T3、STOCHRSI、MA 類型不是 SMA 的 BBANDS／STOCH…）。
              程式無法可靠推算，**必須由研究者設定 warmup.bars**；可用 POST /api/model/warmup/suggest 以實際
              資料測試取得建議值。
  cumulative  從起點累加（OBV、AD），值跟載入起點永遠有關；**必須選擇 transform**（diff／rolling_sum／
              fixed_anchor，或 legacy_cumulative 維持舊行為並標示提示），延長暖機不是解法。
  unknown     程式不認得（quant_saved_strategy、未分類的 TA-Lib 函式）→ 必須由研究者設定 warmup.bars。

研究者可以把暖機設得更長（warmup.bars），不能比程式算出的基本值短。
"""

import math

DEFAULT_EPSILON = 0.01
EPSILON_RANGE = (1e-6, 0.1)

# ── TA-Lib 函式分類 ───────────────────────────────────────────────────────
_FINITE = {
    "SMA", "WMA", "TRIMA", "MIDPOINT", "MIDPRICE", "MIN", "MAX", "MININDEX", "MAXINDEX", "MINMAX", "MINMAXINDEX",
    "SUM", "ROC", "ROCP", "ROCR", "ROCR100", "MOM", "WILLR", "CCI", "AROON", "AROONOSC", "STDDEV", "VAR",
    "LINEARREG", "LINEARREG_ANGLE", "LINEARREG_INTERCEPT", "LINEARREG_SLOPE", "TSF", "CORREL", "BETA", "ULTOSC",
    "MFI", "BOP", "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE", "TRANGE", "ADD", "SUB", "MULT", "DIV",
    "BBANDS", "STOCH", "STOCHF", "APO", "PPO", "MA",
}
# 單一平滑：α（以 timeperiod 表示）
_RECURSIVE_ALPHA = {
    "EMA": lambda p: 2.0 / (p + 1.0),
    "RSI": lambda p: 1.0 / p, "ATR": lambda p: 1.0 / p, "NATR": lambda p: 1.0 / p, "CMO": lambda p: 1.0 / p,
    "PLUS_DI": lambda p: 1.0 / p, "MINUS_DI": lambda p: 1.0 / p, "PLUS_DM": lambda p: 1.0 / p,
    "MINUS_DM": lambda p: 1.0 / p, "DX": lambda p: 1.0 / p,
}
_COMPOSITE = {
    "MACD", "MACDEXT", "MACDFIX", "DEMA", "TEMA", "T3", "TRIX", "ADX", "ADXR", "STOCHRSI", "KAMA", "MAMA",
    "SAR", "SAREXT", "ADOSC", "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDLINE", "HT_TRENDMODE",
}
_CUMULATIVE = {"OBV", "AD"}
# 這些參數不是 0（SMA）時，原本的有限型指標內部改用遞迴平均 → 視為 composite
_MATYPE_PARAMS = ("matype", "slowk_matype", "slowd_matype", "fastd_matype")

MEMORY_TYPES = {
    "none": "只用當根資料，暖機 0",
    "finite": "固定回看根數，暖機＝lookback，之後與載入起點無關",
    "recursive": "單一 EMA／Wilder 平滑，暖機＝lookback＋ceil(ln ε / ln(1-α))，之後在門檻 ε 內一致",
    "composite": "多層遞迴或自適應，程式無法可靠推算，必須由研究者設定 warmup.bars（可先用實測建議）",
    "cumulative": "從起點累加，必須選擇 transform（diff／rolling_sum／fixed_anchor／legacy_cumulative）",
    "unknown": "程式不認得的特徵，必須由研究者設定 warmup.bars",
}


def _talib_function(params: dict):
    import talib.abstract as ta

    name = (params or {}).get("name") or ""
    fn = ta.Function(name)
    kwargs = {k: v for k, v in (params or {}).items() if k != "name"}
    if kwargs:
        expected = fn.parameters
        fn.set_parameters({k: (float(v) if isinstance(expected.get(k), float) else
                               int(v) if isinstance(expected.get(k), int) else v) for k, v in kwargs.items()})
    return fn


def memory_type(key: str, params: dict | None) -> str:
    params = params or {}
    if key in ("ohlcv", "raw_price", "raw_volume"):
        return "none"
    if key != "talib_indicator":
        return "unknown"
    name = str(params.get("name") or "").upper()
    if name in _CUMULATIVE:
        return "cumulative"
    if name in _COMPOSITE:
        return "composite"
    if name in _RECURSIVE_ALPHA:
        return "recursive"
    if name in _FINITE:
        if any(int(params.get(p, 0) or 0) != 0 for p in _MATYPE_PARAMS):
            return "composite"
        return "finite"
    if name.startswith("CDL"):
        return "finite"
    return "unknown"


def talib_lookback(params: dict) -> int:
    return int(_talib_function(params).lookback)


def recursive_extra(name: str, timeperiod: float, epsilon: float) -> int:
    alpha = _RECURSIVE_ALPHA[name](float(timeperiod))
    if alpha >= 1:
        return 0
    return int(math.ceil(math.log(epsilon) / math.log(1.0 - alpha)))


def suggest(key: str, params: dict | None, epsilon: float = DEFAULT_EPSILON) -> dict:
    """程式依記憶類型給的暖機（根數，從資料載入起點算）。

    回傳 {memory_type, base_bars, needs_researcher, method, detail}：
      base_bars          程式能保證的最低暖機（none／finite／recursive 有值；其他為 lookback，僅供參考）
      needs_researcher   True＝必須由研究者設定 warmup.bars（composite／unknown），或必須選 transform（cumulative）
    """
    params = params or {}
    mt = memory_type(key, params)
    if mt == "none":
        return {"memory_type": mt, "base_bars": 0, "needs_researcher": False, "method": "none", "detail": "只用當根資料"}
    if key != "talib_indicator":
        return {"memory_type": mt, "base_bars": 0, "needs_researcher": True, "method": "unknown",
                "detail": f"特徵 {key!r} 的記憶長度無法由程式判定，請設定 warmup.bars"}
    lookback = talib_lookback(params)
    name = str(params.get("name")).upper()
    if mt == "finite":
        return {"memory_type": mt, "base_bars": lookback, "needs_researcher": False, "method": "lookback",
                "detail": f"TA-Lib lookback {lookback} 根"}
    if mt == "recursive":
        period = params.get("timeperiod") or _talib_function(params).parameters.get("timeperiod")
        extra = recursive_extra(name, period, epsilon)
        return {"memory_type": mt, "base_bars": lookback + extra, "needs_researcher": False, "method": "epsilon",
                "epsilon": epsilon,
                "detail": f"lookback {lookback} ＋ ceil(ln {epsilon:g} / ln(1-α)) = {extra}（α 依 timeperiod {period}）"}
    if mt == "cumulative":
        return {"memory_type": mt, "base_bars": lookback, "needs_researcher": True, "method": "transform_required",
                "detail": "累積型指標與載入起點永遠有關，請選擇 transform（diff／rolling_sum／fixed_anchor／legacy_cumulative）"}
    return {"memory_type": mt, "base_bars": lookback, "needs_researcher": True, "method": "researcher",
            "detail": f"{name} 屬複合或自適應指標（lookback {lookback} 根僅供參考），請以實測建議值設定 warmup.bars"}


def empirical_suggestion(values_full, values_by_offset: dict, tol: float) -> dict:
    """實測暖機：`values_full` 是從最早載入點算出的特徵（n, k）；`values_by_offset[s]` 是從第 s 根才開始載入
    算出的同一段（對齊到同一列）。對每個 s，找最小的 k 使得第 s+k 列以後的差異都 ≤ tol。回傳最保守的 k。"""
    import numpy as np

    full = np.asarray(values_full, dtype=float)
    out = []
    for s, part in values_by_offset.items():
        part = np.asarray(part, dtype=float)
        diff = np.abs(part - full[s:]).max(axis=1) if part.ndim == 2 else np.abs(part - full[s:])
        bad = ~(diff <= tol)  # NaN 也視為不一致
        k = int(np.max(np.nonzero(bad)[0]) + 1) if bad.any() else 0
        out.append({"offset": int(s), "bars": k})
    worst = max((o["bars"] for o in out), default=0)
    return {"bars": worst, "trials": out, "tolerance": tol}
