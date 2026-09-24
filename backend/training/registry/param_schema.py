"""模型參數 schema（`params_schema`）：UI 表單渲染與後端驗證共用的單一來源。

每個 architecture（以及每個可插拔元件，例如 Attention 型態）用一份 FieldSpec 陣列描述自己的
參數；後端 `validate_with_schema()` 依同一份描述驗證 payload，前端 `SchemaForm` 依同一份描述
產生表單——兩邊讀的是同一份資料，不會各自表述。模型專屬、schema 表達不了的規則（例如方向門檻
op 的語意、兩個損失權重不能同時為 0）由該模型的 `extra_checks` 補，不塞進 schema 語言本身。

FieldSpec（dict）：
  name            dot path，對應巢狀 params 的位置（`shared.dense` → params["shared"]["dense"]）
  type            int／float／bool／string／enum／component_ref／array_of_object
  label／description   顯示用
  required        True／False 或條件（見下），預設 False
  default         選填欄位沒給時的值；None 代表不補（該欄位直接不出現在解析結果）
  min／max／exclusive_min／exclusive_max   數值範圍（int／float）
  enum_values     enum 的預設可選值
  enum_cases      [{"when": 條件, "values": [...]}]，依序比對，第一個成立的取代 enum_values
  component_registry   component_ref 的來源（"attention"／"optimizer"，見 architectures.component_registries）；
                       值是 {"type": key, ...該型態參數}；只有 default 為 null 的欄位才可以是 null（＝不用）
  item_schema／min_items／max_items   array_of_object 每個元素的欄位與個數限制
  visible_if      條件；不成立時這個欄位「不適用」——payload 帶了這個欄位就是錯誤（不是靜默忽略）
  derived_from    "$context_key"：值由情境決定（例如 `$outcome_unit`），使用者不能另填不同的值

條件（condition）：
  True／False
  {"field": 另一個欄位 name 或 "$情境鍵", "equals": 值} ／ {"field": ..., "not_equals": 值}
  {"all": [條件, ...]} ／ {"any": [條件, ...]}
情境鍵（context）以 `$` 開頭，由呼叫端提供：`$task_type`（Label 決定的 classification／
regression）、`$outcome_unit`（Label outcome 的單位）。提交驗證時兩者都有；訓練當下只有
`$task_type`——缺少的情境鍵對應的檢查一律略過（不猜），不影響其他檢查。
"""

import math

_MISSING = object()


def _get(params, path: str):
    node = params
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _set(out: dict, path: str, value) -> None:
    parts = path.split(".")
    node = out
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _is_int(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and float(v).is_integer()


def _is_num(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def condition_met(cond, lookup) -> bool | None:
    """lookup(name) → 值（找不到回 _MISSING）。回傳 None 代表「情境鍵缺少，無法判斷」。"""
    if cond is None or cond is True:
        return True
    if cond is False:
        return False
    if "all" in cond:
        results = [condition_met(c, lookup) for c in cond["all"]]
        if any(r is False for r in results):
            return False
        return None if any(r is None for r in results) else True
    if "any" in cond:
        results = [condition_met(c, lookup) for c in cond["any"]]
        if any(r is True for r in results):
            return True
        return None if any(r is None for r in results) else False
    value = lookup(cond["field"])
    if value is _MISSING and cond["field"].startswith("$"):
        return None
    if value is _MISSING:
        value = None
    if "equals" in cond:
        return value == cond["equals"]
    if "not_equals" in cond:
        return value != cond["not_equals"]
    return True


def enum_options(spec: dict, lookup) -> list:
    for case in spec.get("enum_cases", []):
        if condition_met(case["when"], lookup) is True:
            return list(case["values"])
    return list(spec.get("enum_values", []))


def _allowed_tree(schema: list) -> dict:
    """schema 的 dot path 攤成巢狀樹，用來找出 payload 裡「schema 沒描述」的欄位。"""
    tree: dict = {}
    for spec in schema:
        node = tree
        parts = spec["name"].split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault(parts[-1], {})["__leaf__"] = True
    return tree


def _unknown_fields(params: dict, tree: dict, prefix: str, problems: list) -> None:
    for key, value in params.items():
        path = f"{prefix}{key}"
        if key not in tree:
            problems.append(f"params.{path} 不是這個模型認得的欄位")
            continue
        sub = tree[key]
        if sub.get("__leaf__"):
            continue
        if not isinstance(value, dict):
            problems.append(f"params.{path} 必須是物件，收到 {type(value).__name__}")
            continue
        _unknown_fields(value, sub, f"{path}.", problems)


def _check_scalar(spec: dict, value, path: str, lookup, problems: list) -> bool:
    t = spec["type"]
    if t == "int" and not _is_int(value):
        problems.append(f"params.{path} 必須是整數，收到 {value!r}")
        return False
    if t == "float" and not _is_num(value):
        problems.append(f"params.{path} 必須是有限數值，收到 {value!r}")
        return False
    if t == "bool" and not isinstance(value, bool):
        problems.append(f"params.{path} 必須是布林值，收到 {value!r}")
        return False
    if t == "string" and not isinstance(value, str):
        problems.append(f"params.{path} 必須是字串，收到 {value!r}")
        return False
    if t == "enum":
        options = enum_options(spec, lookup)
        if value not in options:
            # 可選值依情境縮小時（enum_cases），附上欄位說明，讓錯誤訊息講得出「為什麼」
            why = f"（{spec['description']}）" if spec.get("enum_cases") and spec.get("description") else ""
            problems.append(f"params.{path} 必須是 {options} 之一，收到 {value!r}{why}")
            return False
    if t in ("int", "float"):
        bounds = [("min", lambda v, b: v >= b, ">="), ("max", lambda v, b: v <= b, "<="),
                  ("exclusive_min", lambda v, b: v > b, ">"), ("exclusive_max", lambda v, b: v < b, "<")]
        for key, ok, sym in bounds:
            if key in spec and not ok(value, spec[key]):
                problems.append(f"params.{path} 必須 {sym} {spec[key]}，收到 {value!r}")
                return False
    return True


def validate_with_schema(schema: list, params, context: dict | None = None,
                         component_registries: dict | None = None) -> tuple[dict, list]:
    """回傳 (解析後的完整設定, 問題清單)。問題清單非空就代表 payload 不合法。

    component_registries：{"attention": {key: {"params_schema": [...], "validate_params": fn?, ...}}, ...}，
    component_ref 欄位的型態參數依該型態自己的 params_schema 驗證（並補預設值），有 validate_params
    的型態再交給它做最後把關。
    """
    context = context or {}
    component_registries = component_registries or {}
    problems: list = []
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return {}, [f"params 必須是物件，收到 {type(params).__name__}"]

    def lookup(name: str):
        if name.startswith("$"):
            return context.get(name[1:], _MISSING)
        return _get(params, name)

    _unknown_fields(params, _allowed_tree(schema), "", problems)
    cfg: dict = {}

    for spec in schema:
        path, t = spec["name"], spec["type"]
        value = _get(params, path)
        visible = condition_met(spec.get("visible_if"), lookup)
        if visible is False:
            if value is not _MISSING:
                problems.append(f"params.{path} 在目前的設定下不適用（{spec.get('description') or '條件不成立'}），請移除")
            continue

        if "derived_from" in spec:
            expected = lookup(spec["derived_from"])
            if expected is not _MISSING and value is not _MISSING and value != expected:
                problems.append(f"params.{path} 必須等於 {spec['derived_from'][1:]}（{expected!r}），收到 {value!r}")
                continue
            if value is _MISSING and expected is not _MISSING:
                value = expected

        if value is _MISSING:
            required = condition_met(spec.get("required", False), lookup)
            if required is True:
                problems.append(f"params.{path} 為必填（沒有預設值）")
            elif spec.get("default") is not None:
                _set(cfg, path, spec["default"])
            elif t == "component_ref" and "default" in spec:
                _set(cfg, path, None)
            continue

        if t == "array_of_object":
            items_spec = spec.get("item_schema", [])
            if not isinstance(value, list):
                problems.append(f"params.{path} 必須是陣列")
                continue
            lo, hi = spec.get("min_items", 0), spec.get("max_items")
            if len(value) < lo or (hi is not None and len(value) > hi):
                problems.append(f"params.{path} 的項目數必須在 {lo}～{hi} 之間，收到 {len(value)}")
                continue
            resolved = []
            for i, item in enumerate(value):
                sub_cfg, sub_problems = validate_with_schema(items_spec, item, context, component_registries)
                problems.extend(p.replace("params.", f"params.{path}[{i}].", 1) for p in sub_problems)
                resolved.append(sub_cfg)
            _set(cfg, path, resolved)
            continue

        if t == "component_ref":
            if value is None:
                if "default" in spec and spec["default"] is None:
                    _set(cfg, path, None)
                else:
                    problems.append(f"params.{path} 必須選一個元件（type＋參數的物件），不能是 null")
                continue
            registry = component_registries.get(spec.get("component_registry"), {})
            if isinstance(value, list):
                problems.append(f"params.{path} 本版本只支援單一元件（零或一個），不支援多元件組合，必須是物件或 null")
                continue
            if not isinstance(value, dict) or not isinstance(value.get("type"), str):
                problems.append(f"params.{path} 必須是 null 或 {{\"type\": ..., ...}} 物件")
                continue
            entry = registry.get(value["type"])
            if entry is None:
                problems.append(f"params.{path}.type 未登記：{value['type']!r}，目前有：{sorted(registry)}")
                continue
            sub_params = {k: v for k, v in value.items() if k != "type"}
            sub_cfg, sub_problems = validate_with_schema(entry.get("params_schema", []), sub_params, context)
            problems.extend(p.replace("params.", f"params.{path}.", 1) for p in sub_problems)
            if not sub_problems and entry.get("validate_params"):
                try:
                    sub_cfg = entry["validate_params"](sub_params)
                except ValueError as e:
                    problems.append(f"params.{path}：{e}")
            _set(cfg, path, {"type": value["type"], **sub_cfg})
            continue

        if _check_scalar(spec, value, path, lookup, problems):
            _set(cfg, path, int(value) if t == "int" else float(value) if t == "float" else value)

    return cfg, problems
