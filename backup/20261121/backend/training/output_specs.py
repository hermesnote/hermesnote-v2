"""具名輸出（named outputs）的 metadata：產生、驗證、單位／事件規則。

平台契約（見 docs/agent-api/training/api.md）：一個 Model Node 可以有多個具名輸出，下游用
`inputs` 裡的 `{"node": ..., "output": ...}` 選用；字串形式的引用一律等於 `default`。
輸出名稱是通用語意（`default`／`regression`／`direction_probability`……），不含 outcome 名稱——
輸出實際是什麼（哪個 Outcome、單位、horizon、事件規則）由這裡產生的 metadata 描述，
避免放開目標設定之後，名稱還是誤導成 log_return／p_up。

metadata 格式（每個輸出一份）：
  kind        "regression"（連續值）／"class"（類別索引）／"probability"（機率）
  columns     輸出欄數（形狀是 (n, columns)）
  target      這個輸出對應的目標：{label_node, outcome, unit, signed, horizon, labeling_rule, task_type}
  description 人類可讀的語意說明
  event       只有「事件機率」才有：{of, op, threshold, threshold_unit}——見 build_event_rule
"""

DEFAULT_OUTPUT = "default"
OUTPUT_KINDS = ("regression", "class", "probability")
EVENT_OPS = (">", ">=", "<", "<=")


def label_metadata(label_node: dict, task_type: str | None = None) -> dict:
    """從 Label Node 抽出「這個目標是什麼」：Outcome、單位、horizon、labeling_rule。"""
    from training.registry import outcomes

    params = label_node.get("params") or {}
    outcome = label_node.get("outcome")
    meta = outcomes.OUTCOME_META.get(outcome, {})
    return {
        "label_node": label_node.get("id"),
        "outcome": outcome,
        "unit": meta.get("unit"),
        "signed": meta.get("signed"),
        "horizon": int(params.get("horizon", 1)),
        "labeling_rule": label_node.get("labeling_rule", "fixed_threshold"),
        "task_type": task_type,
    }


def default_output_specs(node: dict, label_node: dict, task_type: str) -> dict:
    """單一輸出頭的 `default` 輸出描述，要跟 graph.py 實際行為一致：

      回歸                              → 連續預測值，(n, 1)
      分類 + output_mode="probability"  → 各類別機率，(n, k)，k = Label 的 n_classes（預設 2；類別數由 Label 決定，不是模型參數）
      分類 + 其他                        → 類別索引，(n, 1)

    這份宣告不能只靠人寫——graph.py 會在訓練後檢查「宣告的欄數」跟「實際輸出的欄數」一致，
    tests/test_named_outputs.py 也有契約測試逐一比對，避免描述跟行為漂移。
    """
    target = label_metadata(label_node, task_type)
    if task_type == "regression":
        spec = {"kind": "regression", "columns": 1, "target": target,
                "description": "連續預測值（單位同 Label 的 outcome）"}
    elif node.get("output_mode") == "probability":
        k = int((label_node.get("params") or {}).get("n_classes", 2))
        spec = {"kind": "probability", "columns": k, "target": target,
                "description": f"各類別機率，{k} 欄，欄位順序即類別索引"}
    else:
        spec = {"kind": "class", "columns": 1, "target": target, "description": "預測類別索引"}
    return {DEFAULT_OUTPUT: spec}


def build_event_rule(rule: dict, target_unit: str | None) -> dict:
    """驗證並正規化「事件規則」（例如 目標 >= 0.0）。

    - op 只允許 > >= < <=。
    - threshold 必須是有限數值（不接受布林、NaN、無限大）。
    - 門檻單位固定等於目標的單位；payload 若另外給 threshold_unit 且不同就拒絕。這裡**不做單位
      換算**——若要換算（例如百分比對 log 比值）需要另外定義有明確公式的轉換規則，不能只改
      metadata 的單位文字。
    """
    import math

    if not isinstance(rule, dict):
        raise ValueError("事件規則必須是物件，例如 {\"op\": \">=\", \"threshold\": 0.0}")
    extra = set(rule) - {"op", "threshold", "threshold_unit"}
    if extra:
        raise ValueError(f"事件規則不認得這些欄位：{sorted(extra)}（只允許 op、threshold、threshold_unit）")
    op = rule.get("op")
    if op not in EVENT_OPS:
        raise ValueError(f"事件規則的 op 只能是 {list(EVENT_OPS)}，收到 {op!r}")
    threshold = rule.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError(f"事件規則的 threshold 必須是有限數值，收到 {threshold!r}")
    unit = rule.get("threshold_unit", target_unit)
    if unit != target_unit:
        raise ValueError(
            f"事件規則的門檻單位（{unit!r}）必須等於目標的單位（{target_unit!r}）；"
            "目前不支援單位換算，請直接用目標的單位表示門檻"
        )
    return {"op": op, "threshold": float(threshold), "threshold_unit": target_unit}


def event_probability_spec(target: dict, rule: dict, of: str = "regression") -> dict:
    """「符合設定規則的機率」的 metadata。描述文字由規則產生，不寫死成 P(目標 >= 門檻)。"""
    event_rule = build_event_rule(rule, target.get("unit"))
    unit_text = f" {event_rule['threshold_unit']}" if event_rule["threshold_unit"] else ""
    return {
        "kind": "probability",
        "columns": 1,
        "derived_from": of,
        "target": target,
        "event": {"of": of, **event_rule},
        "classes": ["not_event", "event"],
        "description": f"P(目標 {event_rule['op']} {event_rule['threshold']}{unit_text})，即符合設定事件規則的機率",
    }


def validate_output_specs(key: str, specs) -> dict:
    """architecture 的 describe_outputs 回傳值必須符合這份格式，不符就是 architecture 自己的 bug。"""
    if not isinstance(specs, dict) or DEFAULT_OUTPUT not in specs:
        raise ValueError(f"architecture {key!r} 的 describe_outputs 必須回傳含 {DEFAULT_OUTPUT!r} 的字典")
    for name, spec in specs.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"architecture {key!r} 的輸出名稱必須是非空字串，收到 {name!r}")
        if not isinstance(spec, dict) or spec.get("kind") not in OUTPUT_KINDS:
            raise ValueError(f"architecture {key!r} 的輸出 {name!r} 缺少合法的 kind（{list(OUTPUT_KINDS)}）")
        cols = spec.get("columns")
        if isinstance(cols, bool) or not isinstance(cols, int) or cols < 1:
            raise ValueError(f"architecture {key!r} 的輸出 {name!r} 的 columns 必須是正整數")
    return specs


def check_output_shape(node_id: str, name: str, spec: dict, arr, n_rows: int) -> None:
    """訓練後／推論時檢查實際輸出跟宣告一致。"""
    if getattr(arr, "ndim", None) != 2 or arr.shape[0] != n_rows or arr.shape[1] != spec["columns"]:
        raise ValueError(
            f"節點 {node_id} 的輸出 {name!r} 形狀 {getattr(arr, 'shape', None)} 跟宣告的 "
            f"({n_rows}, {spec['columns']}) 不一致"
        )
