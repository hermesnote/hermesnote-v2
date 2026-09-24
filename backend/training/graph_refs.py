"""節點圖的輸入引用解析與提交前驗證——所有讀取 `inputs` 的地方都經過這裡，不再各自解析。

輸入引用（`inputs` 陣列的每一項）有兩種寫法：
  "節點id"                            字串。目標可以是 Feature Node 或 Model Node，取 `default` 輸出。
                                      （舊 graph_spec 全部是這種，行為完全不變。）
  {"node": "節點id", "output": "名稱"}  具名引用，目標只能是 Model Node，`output` 必須是該節點
                                      （依實際設定）宣告的輸出名稱，見 training/output_specs.py。

用物件而不是 "節點id:名稱" 字串，是避免節點 id 本身含分隔符號時歧義，也讓舊格式的解析完全不受影響。

`validate_graph_spec()` 是所有「建立／繼承圖」入口共用的驗證：POST /train（一般與 Phase 2 繼承後）、
worker 撿到任務之後（讀資料之前），涵蓋繞過 API 直接寫入資料庫的任務。純函式，不做 I/O、不匯入 torch，
一次回報全部問題（不是遇到第一個就停）。
"""

from typing import NamedTuple

from training.output_specs import DEFAULT_OUTPUT


class InputRef(NamedTuple):
    node: str
    output: str
    explicit: bool = False  # True = 物件形式（具名引用）；False = 字串形式（等於 default）


class GraphValidationError(ValueError):
    """驗證失敗；issues 是全部問題的清單，str(e) 是用「；」串起來的單行訊息（給 HTTP detail／job error 用）。"""

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        super().__init__("；".join(self.issues))


def normalize_input_ref(item) -> InputRef:
    """把 inputs 的單一項統一成 InputRef；格式不對丟 ValueError。"""
    if isinstance(item, str):
        if not item:
            raise ValueError("inputs 項目不能是空字串")
        return InputRef(item, DEFAULT_OUTPUT, False)
    if isinstance(item, dict):
        extra = set(item) - {"node", "output"}
        if extra:
            raise ValueError(f"具名引用只允許 node、output 兩個鍵，多出：{sorted(extra)}")
        node, output = item.get("node"), item.get("output")
        if not isinstance(node, str) or not node:
            raise ValueError("具名引用缺少字串型的 node")
        if not isinstance(output, str) or not output:
            raise ValueError("具名引用缺少字串型的 output")
        return InputRef(node, output, True)
    raise ValueError(f"inputs 項目必須是節點 id 字串或 {{node, output}} 物件，收到 {type(item).__name__}")


def input_refs(node: dict) -> list[InputRef]:
    """依原始順序解析這個節點的全部輸入引用（順序就是特徵欄位串接的順序，訓練與推論必須一致）。"""
    return [normalize_input_ref(i) for i in (node.get("inputs") or [])]


def upstream_model_ids(node: dict, model_ids: set) -> list[str]:
    """這個節點依賴的上游 Model Node id（依出現順序、去重）。"""
    seen: dict = {}
    for ref in input_refs(node):
        if ref.node in model_ids:
            seen.setdefault(ref.node, None)
    return list(seen)


def _int_at_least(value, minimum: int) -> bool:
    """整數（不接受布林；接受 1.0 這種整數值的浮點，JSON 數字可能這樣來）且 >= minimum。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value).is_integer() and value >= minimum


def _finite_number(value) -> bool:
    import math

    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _param_issues(nid: str, node: dict) -> list[str]:
    """metadata 產生（describe_outputs／label_metadata）之前的參數型別驗證。

    這些函式會直接對 params 取值並轉型（例如 int(params["horizon"])）；params 型別不對
    （null、陣列、字串……）不能讓它們丟出未處理的 TypeError／AttributeError，要在這裡統一
    回報成驗證問題，API 才能回 400、worker 才能在讀資料前就拒絕。只檢查「有給的欄位」的型別與範圍，
    沒給的欄位維持原本的預設值行為，不新增必填。
    """
    kind = {"label": "Label", "model": "Model"}[node["type"]]
    params = node.get("params")
    if params is None:
        return []
    if not isinstance(params, dict):
        return [f"{kind} 節點 {nid}：params 必須是物件，收到 {type(params).__name__}"]
    out: list[str] = []

    def bad(field: str, expect: str):
        out.append(f"{kind} 節點 {nid}：params.{field} 必須是{expect}，收到 {params[field]!r}")

    if node["type"] == "label":
        if "horizon" in params and not _int_at_least(params["horizon"], 1):
            bad("horizon", "大於等於 1 的整數")
        if "n_classes" in params and not _int_at_least(params["n_classes"], 2):
            bad("n_classes", "大於等於 2 的整數")
        if "threshold_pct" in params and not _finite_number(params["threshold_pct"]):
            bad("threshold_pct", "有限數值")
    # Model 節點的 params 由各 architecture 的 params_schema 完整驗證（見 architectures.describe_outputs）
    return out


def _node_field_issues(nid: str, node: dict) -> list[str]:
    """Model Node 本身的欄位型別（window／val_ratio／output_mode），graph.py 會直接轉型使用。"""
    out: list[str] = []
    if "window" in node and not _int_at_least(node["window"], 1):
        out.append(f"Model 節點 {nid}：window 必須是大於等於 1 的整數，收到 {node['window']!r}")
    if "val_ratio" in node and not (_finite_number(node["val_ratio"]) and 0 < node["val_ratio"] < 1):
        out.append(f"Model 節點 {nid}：val_ratio 必須是 0 到 1 之間（不含）的數值，收到 {node['val_ratio']!r}")
    if node.get("output_mode") is not None and not isinstance(node["output_mode"], str):
        out.append(f"Model 節點 {nid}：output_mode 必須是字串，收到 {node['output_mode']!r}")
    return out


def validate_graph_spec(graph_spec: dict) -> None:
    """提交前／執行前的完整驗證，失敗丟 GraphValidationError（含全部問題）。

    涵蓋：節點結構、architecture 與 outcome／labeling_rule 已登記且組合合法、Model Node 的 Label 引用、
    輸入引用規則（見模組說明）、具名輸出是否存在（依 describe_outputs 對實際設定的描述）、成環。
    """
    from training.registry import architectures, labeling_rules, outcomes

    issues: list[str] = []
    nodes_list = graph_spec.get("nodes") if isinstance(graph_spec, dict) else None
    if not isinstance(nodes_list, list) or not nodes_list:
        raise GraphValidationError(["graph_spec.nodes 必須是非空陣列"])

    nodes: dict = {}
    for n in nodes_list:
        if not isinstance(n, dict) or not isinstance(n.get("id"), str) or "type" not in n:
            issues.append(f"節點格式不對（需要字串 id 與 type）：{n!r}"[:200])
            continue
        if n["id"] in nodes:
            issues.append(f"節點 id 重複：{n['id']}")
            continue
        nodes[n["id"]] = n
        if n["type"] not in ("feature", "label", "model"):
            issues.append(f"節點 {n['id']}：未知的節點類型 {n['type']!r}")

    label_task_types: dict = {}
    for nid, n in nodes.items():
        if n["type"] != "label":
            continue
        outcome, rule = n.get("outcome"), n.get("labeling_rule", "fixed_threshold")
        ok = True
        param_problems = _param_issues(nid, n)
        if param_problems:
            issues.extend(param_problems)
            ok = False
        if not isinstance(outcome, str) or outcome not in outcomes.OUTCOME_REGISTRY:
            issues.append(f"Label 節點 {nid}：未登記的 outcome {outcome!r}，目前有：{sorted(outcomes.OUTCOME_REGISTRY)}")
            ok = False
        if not isinstance(rule, str) or rule not in labeling_rules.LABELING_RULE_REGISTRY:
            issues.append(
                f"Label 節點 {nid}：未登記的 labeling_rule {rule!r}，目前有：{sorted(labeling_rules.LABELING_RULE_REGISTRY)}"
            )
            ok = False
        if ok:
            try:
                labeling_rules.validate_combination(outcome, rule)
            except ValueError as e:
                issues.append(f"Label 節點 {nid}：{e}")
                ok = False
        if ok:
            label_task_types[nid] = labeling_rules.LABELING_RULE_TASK_TYPE.get(rule)

    # 每個 Model Node 的輸出描述（供具名引用檢查）；描述失敗本身就是一個問題，該節點的具名引用就不再重複報錯。
    model_specs: dict = {}
    for nid, n in nodes.items():
        if n["type"] != "model":
            continue
        key = n.get("key")
        if not isinstance(key, str) or key not in architectures.ARCHITECTURE_REGISTRY:
            issues.append(f"Model 節點 {nid}：未登記的 architecture {key!r}，目前有：{sorted(architectures.ARCHITECTURE_REGISTRY)}")
            continue
        label_id = n.get("label")
        label_node = nodes.get(label_id) if isinstance(label_id, str) else None
        model_problems = _param_issues(nid, n) + _node_field_issues(nid, n)
        issues.extend(model_problems)
        if label_node is None or label_node["type"] != "label":
            issues.append(f"Model 節點 {nid}：label 必須引用一個存在的 Label 節點，收到 {label_id!r}")
        elif not model_problems and label_id in label_task_types and label_task_types[label_id]:
            try:
                model_specs[nid] = architectures.describe_outputs(key, n, label_node, label_task_types[label_id])
            except ValueError as e:
                issues.append(f"Model 節點 {nid} 的模型設定不合法：{e}")
            except Exception as e:  # noqa: BLE001 — 描述函式（含各 architecture 自己的）任何意外都要回報成驗證問題，不能讓提交流程 500／worker 崩潰
                issues.append(f"Model 節點 {nid} 的輸出描述失敗（{type(e).__name__}）：{e}")

    edges: dict = {nid: [] for nid in nodes}
    for nid, n in nodes.items():
        if n["type"] != "model":
            continue
        raw_inputs = n.get("inputs")
        if not isinstance(raw_inputs, list) or not raw_inputs:
            issues.append(f"Model 節點 {nid}：inputs 必須是非空陣列")
            raw_inputs = []
        if isinstance(n.get("label"), str) and n["label"] in nodes:
            edges[nid].append(n["label"])
        for item in raw_inputs:
            try:
                ref = normalize_input_ref(item)
            except ValueError as e:
                issues.append(f"Model 節點 {nid}：{e}")
                continue
            target = nodes.get(ref.node)
            if target is None:
                issues.append(f"Model 節點 {nid}：引用了不存在的節點 {ref.node!r}")
                continue
            edges[nid].append(ref.node)
            if ref.node == nid:
                issues.append(f"Model 節點 {nid}：不能引用自己")
            elif target["type"] == "label":
                issues.append(f"Model 節點 {nid}：inputs 不能引用 Label 節點 {ref.node!r}（Label 只能用 label 欄位）")
            elif target["type"] == "feature" and ref.explicit:
                issues.append(
                    f"Model 節點 {nid}：Feature 節點 {ref.node!r} 只有 default 輸出，請直接用字串 id 引用，不能用 {{node, output}} 物件"
                )
            elif target["type"] == "model" and ref.node in model_specs and ref.output not in model_specs[ref.node]:
                issues.append(
                    f"Model 節點 {nid}：節點 {ref.node!r} 沒有輸出 {ref.output!r}，可用的輸出：{sorted(model_specs[ref.node])}"
                )

    issues.extend(_cycle_issues(edges))
    if issues:
        raise GraphValidationError(issues)


def _cycle_issues(edges: dict) -> list[str]:
    """DFS 找環；每個環只報一次。"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in edges}
    found: list[str] = []
    seen_cycles: set = set()

    def visit(start: str):
        stack = [(start, iter(edges[start]))]
        path = [start]
        color[start] = GRAY
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[node] = BLACK
                stack.pop()
                path.pop()
                continue
            if nxt not in color:
                continue
            if color[nxt] == GRAY:
                cycle = path[path.index(nxt):] + [nxt]
                sig = frozenset(cycle)
                if sig not in seen_cycles:
                    seen_cycles.add(sig)
                    found.append("循環依賴：" + " → ".join(cycle))
            elif color[nxt] == WHITE:
                color[nxt] = GRAY
                path.append(nxt)
                stack.append((nxt, iter(edges[nxt])))

    for n in list(edges):
        if color[n] == WHITE:
            visit(n)
    return found
