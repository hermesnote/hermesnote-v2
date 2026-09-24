"""寫進 PostgreSQL JSONB 前的數值清理。

JSONB 不接受 NaN／Infinity，而 json.dumps 預設會寫出 NaN 字面值，INSERT／UPDATE 會直接失敗
（整筆訓練結果或整輪進度寫不進去）。非有限值一律存成 null＝「沒有這個數值」；評估指標本身另有
`*_unavailable_reason` 說明原因（見 training/evaluation.py）。
"""

import json
import math


def json_safe(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    return v


def dumps(v) -> str:
    return json.dumps(json_safe(v), allow_nan=False)
