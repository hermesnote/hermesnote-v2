"""通用條件樹：進場/出場各自是一棵可以巢狀的 AND/OR/加權運算樹。

取代原本寫死的「A 組（觸發、AND/OR 鏈）+ B 組（過濾、整組單一模式）」兩層結構——
那個結構把「誰負責決定時機」跟「誰負責當條件」綁死在 A/B 分組上，沒辦法自由混搭分子群組。

樹節點兩種型別（用 dict 的 "type" 欄位分辨，不用嚴格的 Pydantic model，畢竟是內部後台工具）：

- 條件節點：{"type": "condition", "kind": "long_entry"|"long_exit"|"short_entry"|"short_exit"|"filter",
  "key": "RSI", "params": {...}, (filter 專用) "output_name": "real", "operator": ">",
  "threshold_value": 25, "weight": 1.0}
  - long_entry/short_exit：只有 signal_supported 的指標能用，取該指標訊號模板算出來的「上升方向」
    事件（震盪指標跌破下界、型態看漲、快線上穿慢線）
  - long_exit/short_entry：取「下降方向」事件（震盪指標突破上界、型態看跌、快線下穿慢線）
    ——期貨可以直接放空，做多出場跟做空進場本來就是同一個下降事件，只是位置不同
  - filter：161 個任何指標都能用，是「現在符不符合」的通用門檻比較，沒有方向性，四個位置都能放

- 群組節點：{"type": "group", "mode": "and"|"or"|"weighted", "threshold": float(weighted 用，
  可省略，預設是所有子節點權重總和的一半), "weight": 1.0, "children": [node, ...]}
  - children 可以是條件節點，也可以是另一個群組節點（巢狀），沒有層數限制
"""

import pandas as pd

from indicators.filters import compute_filter_condition
from indicators.signals import (
    compute_crossover_signals,
    compute_oscillator_signals,
    compute_pattern_signals,
    is_crossover,
    is_oscillator,
    is_pattern,
)


def _trigger_entries_exits(df: pd.DataFrame, key: str, params: dict):
    if is_pattern(key):
        return compute_pattern_signals(df, key)
    if is_oscillator(key):
        return compute_oscillator_signals(
            df, key, params.get("timeperiod", 14), params.get("oversold"), params.get("overbought")
        )
    if is_crossover(key):
        return compute_crossover_signals(df, key, params)
    raise ValueError(f"{key} 不是可觸發指標（沒有訊號模板），不能當 long_entry/long_exit/short_entry/short_exit 用")


# kind -> 要取訊號模板算出來的哪個方向（"up" = 跌破下界/看漲/上穿，"down" = 突破上界/看跌/下穿）
_DIRECTION_BY_KIND = {
    "long_entry": "up",
    "short_exit": "up",
    "long_exit": "down",
    "short_entry": "down",
}


def evaluate_node(df: pd.DataFrame, node: dict) -> tuple[pd.Series, str]:
    """回傳 (布林序列, 這個節點的白話文敘述)。"""
    node_type = node.get("type")

    if node_type == "condition":
        kind = node["kind"]
        key = node["key"]
        params = node.get("params") or {}

        if kind in _DIRECTION_BY_KIND:
            up, down, up_reason, down_reason = _trigger_entries_exits(df, key, params)
            if _DIRECTION_BY_KIND[kind] == "up":
                return up.fillna(False), up_reason
            return down.fillna(False), down_reason

        if kind == "filter":
            output_name = node.get("output_name")
            operator_symbol = node["operator"]
            threshold_value = node["threshold_value"]
            series = compute_filter_condition(df, key, output_name, params, operator_symbol, threshold_value)
            desc = f"{key}.{output_name or 'real'} {operator_symbol} {threshold_value:g}"
            return series, desc

        raise ValueError(f"不支援的條件種類：{kind}")

    if node_type == "group":
        children = node.get("children") or []
        if not children:
            raise ValueError("群組不能是空的")
        mode = node.get("mode", "and")

        results, descs, weights = [], [], []
        for child in children:
            series, desc = evaluate_node(df, child)
            results.append(series)
            descs.append(desc)
            weights.append(child.get("weight", 1.0))

        if mode == "and":
            combined = results[0]
            for s in results[1:]:
                combined = combined & s
            joiner = " 且 "
        elif mode == "or":
            combined = results[0]
            for s in results[1:]:
                combined = combined | s
            joiner = " 或 "
        elif mode == "weighted":
            score = sum(s.astype(int) * w for s, w in zip(results, weights))
            threshold = node.get("threshold")
            if threshold is None:
                threshold = sum(weights) / 2
            combined = score >= threshold
            joiner = "、"
        else:
            raise ValueError(f"不支援的組合模式：{mode}")

        desc = "（" + joiner.join(descs) + "）"
        return combined.fillna(False), desc

    raise ValueError(f"不支援的節點型別：{node_type}")


def collect_indicator_keys(node: dict) -> list[str]:
    if node.get("type") == "condition":
        return [node["key"]]
    keys = []
    for child in node.get("children") or []:
        keys.extend(collect_indicator_keys(child))
    return keys
