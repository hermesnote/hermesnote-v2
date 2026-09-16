"""TA-Lib 指標登記表：從 TA-Lib 官方分組機械式產生完整指標清單（含參數 schema）。

只有 signal_supported() 判定為 True 的指標，才有對應的進出場觸發規則（見 indicators/signals.py），
可以當 A 組（trigger）；其餘指標沒有觸發模板，但仍然可以當 B 組（filter，見 indicators/filters.py）
的輔助條件——filter 是機械式的通用機制，161 個全部都適用，不需要另外標記。
"""

import talib
from talib import abstract

from indicators.descriptions import TA_DESCRIPTIONS
from indicators.signals import (
    OSCILLATOR_DEFAULTS,
    is_crossover,
    is_oscillator,
    is_pattern,
    signal_supported,
)

GROUP_LABELS_ZH = {
    "Overlap Studies": "重疊研究",
    "Momentum Indicators": "動量指標",
    "Volume Indicators": "成交量指標",
    "Volatility Indicators": "波動度指標",
    "Price Transform": "價格轉換",
    "Cycle Indicators": "週期指標",
    "Pattern Recognition": "型態辨識",
    "Statistic Functions": "統計函數",
    "Math Transform": "數學轉換",
    "Math Operators": "數學運算",
}


def _signal_type(key: str) -> str | None:
    if is_pattern(key):
        return "pattern"
    if is_oscillator(key):
        return "oscillator"
    if is_crossover(key):
        return "crossover"
    return None


def build_registry() -> list[dict]:
    """回傳依官方分組排列的指標清單，每組底下是該組所有指標的參數 schema。"""
    groups = talib.get_function_groups()
    result = []
    for group_name, fn_names in groups.items():
        indicators = []
        for fn_name in sorted(fn_names):
            fn = abstract.Function(fn_name)
            info = fn.info
            entry = {
                "key": fn_name,
                "display_name": info.get("display_name", fn_name),
                "parameters": dict(fn.parameters),
                "output_names": fn.output_names,
                "signal_supported": signal_supported(fn_name),
                "signal_type": _signal_type(fn_name),
                "description": TA_DESCRIPTIONS.get(fn_name, ""),
            }
            if is_oscillator(fn_name):
                entry["signal_defaults"] = OSCILLATOR_DEFAULTS[fn_name]
            indicators.append(entry)
        result.append(
            {
                "group": group_name,
                "group_zh": GROUP_LABELS_ZH.get(group_name, group_name),
                "indicators": indicators,
            }
        )
    return result
