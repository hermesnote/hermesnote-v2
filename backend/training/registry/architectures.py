"""Model Node 登記表（通用模型組裝框架的核心）。

每個 architecture 用 `register()` 登記：
  train 函式      `(X_train, y_train, X_val, y_val, params, on_epoch, task_type, *, preview_hook, n_classes) -> dict`
                   params 已經是 validate() 解析後的完整設定（含預設值），不是原始 payload
  params_schema    FieldSpec 陣列（見 training/registry/param_schema.py）：UI 表單與後端驗證共用
  extra_checks     `(cfg, context) -> [問題...]`：schema 表達不了的模型專屬規則
  capabilities     三態能力宣告：{軸: "fixed_on"|"fixed_off"|"configurable"}，只描述這個 key
                   「允許設定什麼」，不代表某次提交實際選了什麼（那要看提交的 params／產物 model_config）
  slots            可插拔位置：[{slot_name, component_registry, cardinality, accepts_output_kind?}]
                   （accepts_output_kind 只有「元件輸出會接進網路」的位置才宣告，例如 attention；optimizer 沒有）
  outputs／describe_outputs   具名輸出（見 training/output_specs.py）
  lazy_windows     訓練迴圈是否逐 batch 取樣（可用 graph.py 的 LazyWindowed 省記憶體）

train 回傳 dict 至少含 final_metrics／predict／weights／model_config／evaluation；有 best／last
之分的一律同時回傳 weights_last 與 evaluation={"best":…, "last":…}。LOAD_REGISTRY[key] 用
(weights, model_config) 重建可推論的物件，不重新訓練。

新增一個 architecture（例如 Transformer／Mamba）＝新增一個模組、呼叫 register()，
graph.py／inference.py／API／前端 SchemaForm 都不需要修改（見 docs/agent-api/training/api.md 擴充指南）。
"""

from training.output_specs import DEFAULT_OUTPUT, default_output_specs, validate_output_specs
from training.registry.param_schema import validate_with_schema

ARCHITECTURE_REGISTRY: dict = {}
LOAD_REGISTRY: dict = {}
ARCHITECTURE_OUTPUTS: dict = {}
ARCHITECTURE_DESCRIBERS: dict = {}
ARCHITECTURE_META: dict = {}
LAZY_WINDOW_ARCHS: set = set()


def register(key: str, *, family: str, label: str, params_schema: list, extra_checks=None,
             capabilities: dict | None = None, slots: list | None = None,
             outputs: tuple = (DEFAULT_OUTPUT,), describe_outputs=None, lazy_windows: bool = False):
    def deco(fn):
        ARCHITECTURE_REGISTRY[key] = fn
        ARCHITECTURE_OUTPUTS[key] = tuple(outputs)
        ARCHITECTURE_META[key] = {
            "family": family, "label": label, "params_schema": params_schema,
            "extra_checks": extra_checks, "capabilities": capabilities or {}, "slots": slots or [],
        }
        if describe_outputs is not None:
            ARCHITECTURE_DESCRIBERS[key] = describe_outputs
        if lazy_windows:
            LAZY_WINDOW_ARCHS.add(key)
        return fn
    return deco


def register_loader(key: str):
    def deco(fn):
        LOAD_REGISTRY[key] = fn
        return fn
    return deco


def _component_modules() -> dict:
    from training.registry import attention, optimizers

    return {"attention": attention, "optimizer": optimizers}


def component_registries() -> dict:
    """component_ref 欄位可引用的元件登記表：{登記表名稱: {key: 登記資訊}}。"""
    return {"attention": _component_modules()["attention"].ATTENTION_REGISTRY,
            "optimizer": _component_modules()["optimizer"].OPTIMIZER_REGISTRY}


def list_components(registry: str) -> list[dict]:
    """GET /api/model/registry/components/{registry}：該登記表每個元件的 params_schema 與相容性宣告。"""
    modules = _component_modules()
    if registry not in modules:
        raise KeyError(registry)
    return modules[registry].list_available()


def validate(key: str, params, context: dict) -> dict:
    """依登記的 params_schema＋extra_checks 驗證，回傳解析後的完整設定；不合法丟 ValueError
    （一次列出全部問題）。context：{"task_type", "outcome_unit"（可省略）, "family"}。"""
    if key not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"未登記的 architecture key: {key!r}，目前有：{sorted(ARCHITECTURE_REGISTRY)}")
    if key not in ARCHITECTURE_META:
        return params or {}  # 測試用、直接塞進 ARCHITECTURE_REGISTRY 的替身架構：沒有 schema 可驗
    meta = ARCHITECTURE_META[key]
    cfg, problems = validate_with_schema(meta["params_schema"], params, context, component_registries())
    if not problems:
        problems.extend(_slot_compatibility_problems(key, cfg))
    if not problems and meta["extra_checks"]:
        problems.extend(meta["extra_checks"](cfg, context))
    if problems:
        raise ValueError("；".join(problems))
    return cfg


def _slot_compatibility_problems(key: str, cfg: dict) -> list:
    """選用的元件是否真的相容：元件登記的 slot_compatibility 要包含 (這個 family, 這個 slot)，
    輸出形狀（output_kind）要是這個 slot 接受的形狀——程式實際檢查，不是只看描述文字。"""
    from training.registry.param_schema import _MISSING, _get

    meta = ARCHITECTURE_META[key]
    registries = component_registries()
    problems = []
    for slot in meta["slots"]:
        value = _get(cfg, slot["slot_name"])
        if value is _MISSING or value is None:
            continue
        entry = registries[slot["component_registry"]][value["type"]]
        if (meta["family"], slot["slot_name"]) not in {tuple(p) for p in entry["slot_compatibility"]}:
            problems.append(f"params.{slot['slot_name']}：元件 {value['type']!r} 沒有登記與 "
                            f"{meta['family']}／{slot['slot_name']} 相容")
        elif "accepts_output_kind" in slot and entry.get("output_kind") not in slot["accepts_output_kind"]:
            problems.append(f"params.{slot['slot_name']}：元件 {value['type']!r} 的輸出形狀是 "
                            f"{entry['output_kind']}，這個位置只接受 {slot['accepts_output_kind']}")
    return problems


def list_available() -> list[dict]:
    """GET /api/model/registry/architectures：能力、可插拔位置、參數 schema 一次給齊，
    UI 與 Agent 依此組表單／payload。extra_checks 是程式邏輯，不對外輸出。"""
    out = []
    for k, fn in ARCHITECTURE_REGISTRY.items():
        meta = ARCHITECTURE_META.get(k) or {"family": k, "label": k, "capabilities": {}, "slots": [], "params_schema": None}
        out.append({
            "key": k, "family": meta["family"], "label": meta["label"],
            "description": (fn.__doc__ or "").strip(),
            "outputs": list(ARCHITECTURE_OUTPUTS.get(k, (DEFAULT_OUTPUT,))),
            "capabilities": meta["capabilities"], "slots": meta["slots"], "params_schema": meta["params_schema"],
            "lazy_windows": k in LAZY_WINDOW_ARCHS,
        })
    return out


def describe_outputs(key: str, node: dict, label_node: dict, task_type: str) -> dict:
    """提交階段：先依 schema 驗證 params（含 outcome 單位等情境），再描述這次設定會產生的輸出。"""
    from training.output_specs import label_metadata

    target = label_metadata(label_node, task_type)
    validate(key, node.get("params"), {"task_type": task_type, "outcome_unit": target["unit"],
                                        "signed": target["signed"]})
    fn = ARCHITECTURE_DESCRIBERS.get(key, default_output_specs)
    return validate_output_specs(key, fn(node, label_node, task_type))


def load_model(key: str, weights: bytes, model_config: dict) -> dict:
    if key not in LOAD_REGISTRY:
        raise ValueError(f"未登記推論載入函式的 architecture key: {key!r}，目前有：{sorted(LOAD_REGISTRY)}")
    return LOAD_REGISTRY[key](weights, model_config)


def train_model(key: str, X_train, y_train, X_val, y_val, params: dict, on_epoch=None,
                task_type: str = "classification", preview_hook=None, n_classes: int | None = None) -> dict:
    """訓練當下再驗證一次（涵蓋繞過 API 的 job）；n_classes 由 Label 決定（分類時），不是模型參數。"""
    if key not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"未登記的 architecture key: {key!r}，目前有：{sorted(ARCHITECTURE_REGISTRY)}")
    fn = ARCHITECTURE_REGISTRY[key]
    if key not in ARCHITECTURE_META:  # 測試用替身架構：沿用最簡簽名
        return fn(X_train, y_train, X_val, y_val, params, on_epoch, task_type, preview_hook=preview_hook)
    cfg = validate(key, params, {"task_type": task_type})
    return fn(X_train, y_train, X_val, y_val, cfg, on_epoch, task_type, preview_hook=preview_hook,
              n_classes=n_classes if task_type == "classification" else None)


# 放在檔案最後：各 architecture 模組用上面的 register／register_loader 登記自己。
# （torch／xgboost 只在函式內才匯入，API 容器匯入這個模組不需要它們。）
from training.registry import lstm  # noqa: E402,F401
from training.registry import xgboost_model  # noqa: E402,F401
