import { useEffect, useRef, useState } from "react";
import "./TreeBuilder.css";

export type SignalType = "pattern" | "oscillator" | "crossover" | null;
export type RegistryIndicator = {
  key: string;
  display_name: string;
  description: string;
  parameters: Record<string, number>;
  output_names: string[];
  signal_supported: boolean;
  signal_type: SignalType;
  signal_defaults?: { oversold: number; overbought: number };
};
export type RegistryGroup = { group: string; group_zh: string; indicators: RegistryIndicator[] };

export type TriggerKind = "long_entry" | "long_exit" | "short_entry" | "short_exit";
export type ConditionNode = {
  id: string;
  type: "condition";
  kind: TriggerKind | "filter";
  key: string;
  params: Record<string, number>;
  outputName?: string;
  operator?: ">" | ">=" | "<" | "<=";
  thresholdValue?: number;
  weight: number;
};
export type GroupNode = {
  id: string;
  type: "group";
  mode: "and" | "or" | "weighted";
  threshold?: number;
  weight: number;
  children: TreeNode[];
};
export type TreeNode = ConditionNode | GroupNode;

let uidCounter = 0;
export function nextUid() {
  uidCounter += 1;
  return `n${uidCounter}`;
}

export function emptyGroup(mode: GroupNode["mode"] = "and"): GroupNode {
  return { id: nextUid(), type: "group", mode, weight: 1, children: [] };
}

export const TRIGGER_LABELS: Record<TriggerKind, string> = {
  long_entry: "多方進場",
  long_exit: "多方出場",
  short_entry: "空方進場",
  short_exit: "空方出場",
};

export function findIndicator(registry: RegistryGroup[], key: string): RegistryIndicator | undefined {
  for (const g of registry) {
    const found = g.indicators.find((i) => i.key === key);
    if (found) return found;
  }
  return undefined;
}

export function describeNode(node: TreeNode): string {
  if (node.type === "condition") {
    if (node.kind === "filter") {
      return `${node.key}.${node.outputName || "real"} ${node.operator} ${node.thresholdValue ?? 0}`;
    }
    return `${node.key}(${TRIGGER_LABELS[node.kind as TriggerKind]})`;
  }
  const joiner = node.mode === "and" ? " 且 " : node.mode === "or" ? " 或 " : "、";
  const inner = node.children.map(describeNode).join(joiner);
  const prefix = node.mode === "weighted" ? `加權(門檻${node.threshold ?? "總權重一半"})：` : "";
  return `（${prefix}${inner}）`;
}

function updateChild(group: GroupNode, id: string, updater: (n: TreeNode) => TreeNode): GroupNode {
  return {
    ...group,
    children: group.children.map((c) => {
      if (c.id === id) return updater(c);
      if (c.type === "group") return updateChild(c, id, updater);
      return c;
    }),
  };
}
function removeChild(group: GroupNode, id: string): GroupNode {
  return {
    ...group,
    children: group.children
      .filter((c) => c.id !== id)
      .map((c) => (c.type === "group" ? removeChild(c, id) : c)),
  };
}

export function StrategyGroupEditor({
  group,
  onChange,
  registry,
  triggerKind,
  depth = 0,
}: {
  group: GroupNode;
  onChange: (g: GroupNode) => void;
  registry: RegistryGroup[];
  triggerKind: TriggerKind;
  depth?: number;
}) {
  const isWeighted = group.mode === "weighted";

  function patchSelf(patch: Partial<GroupNode>) {
    onChange({ ...group, ...patch });
  }
  function patchChild(id: string, patch: Partial<TreeNode>) {
    onChange(updateChild(group, id, (n) => ({ ...n, ...patch }) as TreeNode));
  }
  function removeById(id: string) {
    onChange(removeChild(group, id));
  }
  function addTriggerCondition(key: string) {
    const meta = findIndicator(registry, key);
    if (!meta || !meta.signal_supported) return;
    const cond: ConditionNode = {
      id: nextUid(),
      type: "condition",
      kind: triggerKind,
      key: meta.key,
      params:
        meta.signal_type === "oscillator"
          ? { ...meta.parameters, oversold: meta.signal_defaults?.oversold ?? 30, overbought: meta.signal_defaults?.overbought ?? 70 }
          : { ...meta.parameters },
      weight: 1,
    };
    onChange({ ...group, children: [...group.children, cond] });
  }
  function addFilterCondition(key: string) {
    const meta = findIndicator(registry, key);
    if (!meta) return;
    const cond: ConditionNode = {
      id: nextUid(),
      type: "condition",
      kind: "filter",
      key: meta.key,
      params: { ...meta.parameters },
      outputName: meta.output_names[0],
      operator: ">",
      thresholdValue: 0,
      weight: 1,
    };
    onChange({ ...group, children: [...group.children, cond] });
  }
  function addSubgroup() {
    onChange({ ...group, children: [...group.children, emptyGroup("and")] });
  }

  return (
    <div className="tree-group" style={{ marginLeft: depth > 0 ? 16 : 0 }}>
      <div className="tree-group-head">
        <select value={group.mode} onChange={(e) => patchSelf({ mode: e.target.value as GroupNode["mode"] })}>
          <option value="and">AND（全部符合）</option>
          <option value="or">OR（任一符合）</option>
          <option value="weighted">加權（總分過門檻）</option>
        </select>
        {isWeighted && (
          <label className="tree-threshold">
            門檻分數
            <input
              type="number"
              step="0.5"
              value={group.threshold ?? ""}
              placeholder="預設總權重一半"
              onChange={(e) => patchSelf({ threshold: e.target.value === "" ? undefined : Number(e.target.value) })}
            />
          </label>
        )}
        {depth > 0 && (
          <label className="tree-weight">
            此群組權重
            <input type="number" step="0.5" value={group.weight} onChange={(e) => patchSelf({ weight: Number(e.target.value) })} />
          </label>
        )}
      </div>

      {group.children.length === 0 && <div className="tree-empty">還沒有任何條件</div>}

      {group.children.map((child) =>
        child.type === "group" ? (
          <div className="tree-child-group" key={child.id}>
            <StrategyGroupEditor
              group={child}
              onChange={(g) => patchChild(child.id, g)}
              registry={registry}
              triggerKind={triggerKind}
              depth={depth + 1}
            />
            <button type="button" className="tree-remove-btn" onClick={() => removeById(child.id)}>移除這個子群組</button>
          </div>
        ) : (
          <ConditionRow
            key={child.id}
            node={child}
            registry={registry}
            showWeight={isWeighted}
            onChange={(patch) => patchChild(child.id, patch)}
            onRemove={() => removeById(child.id)}
          />
        )
      )}

      <div className="tree-add-row">
        <IndicatorAdder registry={registry} onlySupported label="＋ 加觸發條件" onAdd={addTriggerCondition} />
        <IndicatorAdder registry={registry} label="＋ 加過濾條件" onAdd={addFilterCondition} />
        <button type="button" className="tree-add-btn" onClick={addSubgroup}>＋ 加子群組</button>
      </div>
    </div>
  );
}

export function IndicatorAdder({
  registry,
  onAdd,
  label,
  onlySupported,
  excludeKeys,
}: {
  registry: RegistryGroup[];
  onAdd: (key: string) => void;
  label: string;
  onlySupported?: boolean;
  // 已經被用掉、不能再選第二次的 key（例如同一個模組裡別的特徵列已經選過的指標）。
  // 選填——沒傳就跟原本行為一樣，不擋重複，backtest 那邊的用法不受影響。
  excludeKeys?: string[];
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const flat = registry
    .flatMap((g) => g.indicators.map((ind) => ({ ...ind, group_zh: g.group_zh })))
    .filter((ind) => !onlySupported || ind.signal_supported)
    .filter((ind) => !excludeKeys?.includes(ind.key));
  const q = query.trim().toLowerCase();
  const shown = q
    ? flat.filter(
        (ind) =>
          ind.key.toLowerCase().includes(q) ||
          ind.display_name.toLowerCase().includes(q) ||
          ind.description.toLowerCase().includes(q)
      )
    : flat;

  function pick(key: string) {
    onAdd(key);
    setQuery("");
    setOpen(false);
  }

  return (
    <div className="tree-adder" ref={boxRef}>
      <input
        type="text"
        className="tree-adder-input"
        placeholder={label}
        value={query}
        onFocus={() => setOpen(true)}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
      />
      {open && (
        <div className="tree-adder-list">
          {shown.length === 0 && <div className="tree-adder-empty">找不到符合的指標</div>}
          {shown.map((ind) => (
            <button type="button" key={ind.key} className="tree-adder-option" onClick={() => pick(ind.key)}>
              {/* TA-Lib 指標的 display_name 預設就是它的 key（例如 RSI），秀哪個看起來一樣；
                  但 Custom 那幾個（ohlcv/raw_price/raw_volume）的 key 是給後端配對用的
                  程式識別字串，display_name 才是給人看的名稱（OHLCV/Raw Price/Raw Volume），
                  兩者不一定相同，這裡一律秀 display_name 才不會兩邊都要遷就。 */}
              <span className="tree-adder-option-key">{ind.display_name}</span>
              <span className="tree-adder-option-group">{ind.group_zh}</span>
              <span className="tree-adder-option-desc">{ind.description || ind.display_name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function ConditionRow({
  node,
  registry,
  showWeight,
  onChange,
  onRemove,
}: {
  node: ConditionNode;
  registry: RegistryGroup[];
  showWeight: boolean;
  onChange: (patch: Partial<ConditionNode>) => void;
  onRemove: () => void;
}) {
  const meta = findIndicator(registry, node.key);
  const kindLabel = node.kind === "filter" ? "過濾條件" : TRIGGER_LABELS[node.kind];

  return (
    <div className="tree-condition">
      <div className="tree-condition-main">
        <span className="tree-condition-key">{node.key}</span>
        <span className="tree-condition-kind">{kindLabel}</span>
        {meta?.description && <span className="tree-condition-desc">{meta.description}</span>}

        {node.kind === "filter" && meta && meta.output_names.length > 1 && (
          <label>
            輸出線
            <select value={node.outputName} onChange={(e) => onChange({ outputName: e.target.value })}>
              {meta.output_names.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          </label>
        )}

        {Object.entries(node.params).map(([pname, pval]) => (
          <label key={pname}>
            {pname}
            <input type="number" value={pval} onChange={(e) => onChange({ params: { ...node.params, [pname]: Number(e.target.value) } })} />
          </label>
        ))}

        {node.kind === "filter" ? (
          <>
            <label>
              條件
              <select value={node.operator} onChange={(e) => onChange({ operator: e.target.value as ConditionNode["operator"] })}>
                <option value=">">大於</option>
                <option value=">=">大於等於</option>
                <option value="<">小於</option>
                <option value="<=">小於等於</option>
              </select>
            </label>
            <label>
              門檻值
              <input type="number" value={node.thresholdValue} onChange={(e) => onChange({ thresholdValue: Number(e.target.value) })} />
            </label>
          </>
        ) : null}

        {showWeight && (
          <label>
            權重
            <input type="number" step="0.5" value={node.weight} onChange={(e) => onChange({ weight: Number(e.target.value) })} />
          </label>
        )}
      </div>
      <button type="button" className="tree-remove-btn" onClick={onRemove} aria-label="移除">×</button>
    </div>
  );
}
