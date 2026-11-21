"""上下界序列（2026-11-21）：首次越界標籤（outcomes.first_crossing）與價格位置特徵（features.price_position）共用。

上下界各自是一個 TA-Lib 指標的某一個輸出，在 T 那一根收完時就已知（只用 ≤ T 的資料），例如布林通道：

  {"upper": {"name": "BBANDS", "timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "output": "upperband"},
   "lower": {"name": "BBANDS", "timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "output": "lowerband"}}

`output` 是 TA-Lib 的輸出名稱（單一輸出的指標可省略）。標籤和特徵用同一份設定，就保證兩邊的上下界一致。
暖機期（指標還算不出來）上下界是 NaN：標籤在那些列沒有答案，特徵在那些列是 NaN（窗格檢查會排除）。
"""

import numpy as np

SIDES = ("upper", "lower")


def _function(spec: dict):
    import talib.abstract as ta

    name = spec.get("name") or ""
    fn = ta.Function(name)
    kwargs = {k: v for k, v in spec.items() if k not in ("name", "output")}
    if kwargs:
        expected = fn.parameters
        fn.set_parameters({k: (float(v) if isinstance(expected.get(k), float) else
                               int(v) if isinstance(expected.get(k), int) else v) for k, v in kwargs.items()})
    return fn


def validate(spec) -> list[str]:
    """回傳問題清單（空＝合法）。只檢查結構與 TA-Lib 是否認得，不需要資料。"""
    if not isinstance(spec, dict):
        return ["bounds 必須是物件 {\"upper\": {...}, \"lower\": {...}}"]
    out = []
    extra = set(spec) - set(SIDES)
    if extra:
        out.append(f"bounds 不認得這些欄位：{sorted(extra)}（只允許 upper、lower）")
    for side in SIDES:
        s = spec.get(side)
        if not isinstance(s, dict) or not s.get("name"):
            out.append(f"bounds.{side} 必須是 TA-Lib 指標設定，至少要有 name（例如 \"BBANDS\"）")
            continue
        try:
            fn = _function(s)
        except Exception as e:  # noqa: BLE001 — TA-Lib 的錯誤訊息直接回報
            out.append(f"bounds.{side}：TA-Lib 不認得 {s.get('name')!r} 或參數不合法（{e}）")
            continue
        names = list(fn.output_names)
        o = s.get("output")
        if o is None and len(names) > 1:
            out.append(f"bounds.{side}：{s['name']} 有多個輸出 {names}，請用 output 指定")
        elif o is not None and o not in names:
            out.append(f"bounds.{side}.output 必須是 {names} 之一，收到 {o!r}")
    return out


def compute(df, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (upper, lower)，長度與 df 相同；暖機期為 NaN。"""
    problems = validate(spec)
    if problems:
        raise ValueError("；".join(problems))
    inputs = {c: df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close", "volume")}
    res = []
    for side in SIDES:
        s = spec[side]
        fn = _function(s)
        fn.input_arrays = inputs
        out = fn.run()
        if isinstance(out, list):
            out = out[list(fn.output_names).index(s.get("output") or fn.output_names[0])]
        res.append(np.asarray(out, dtype=float))
    return res[0], res[1]


def talib_specs(spec: dict) -> list[dict]:
    """暖機計算用：上下界各自的 TA-Lib 參數（去掉 output）。"""
    return [{k: v for k, v in spec[side].items() if k != "output"} for side in SIDES]
