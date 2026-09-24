// Schema 驅動的通用模型參數表單：讀 GET /api/model/registry/architectures 每個 architecture 的
// params_schema（語意見 schemaFormLogic.ts／後端 param_schema.py），LSTM、XGBoost、未來新架構共用
// 同一支元件，不為個別架構手刻表單。每個架構只顯示自己 schema 裡、目前條件成立的欄位。
import {
  compatibleComponents, componentNullable, conditionMet, enumOptions, initialItem, isRequired, reconcileEnums,
  withComponentValues,
  type Component, type FieldSpec, type FormContext, type FormValues,
} from "./schemaFormLogic";

function title(f: FieldSpec, required: boolean) {
  return `${f.label ?? f.name}${required ? "" : "（選填）"}`;
}

function NumberInput({ f, value, onChange, required }: {
  f: FieldSpec; value: unknown; onChange: (v: number | undefined) => void; required: boolean;
}) {
  return (
    <label title={f.description}>
      <span>{title(f, required)}</span>
      <input
        type="number"
        min={f.min ?? f.exclusive_min} max={f.max ?? f.exclusive_max}
        step={f.type === "float" ? "any" : 1}
        value={value === undefined || value === null ? "" : String(value)}
        placeholder={required ? "必填" : ""}
        onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))}
      />
    </label>
  );
}

function LeafField({ f, values, context, setValue }: {
  f: FieldSpec; values: FormValues; context: FormContext; setValue: (name: string, v: unknown) => void;
}) {
  const required = isRequired(f, values, context);
  const val = values[f.name];
  if (f.derived_from) {
    // 由情境（例如 Label 的 outcome 單位）決定，送出時直接帶情境值，這裡只顯示
    return (
      <label title={f.description}>
        <span>{f.label ?? f.name}（跟隨 Label）</span>
        <input type="text" value={String(context[f.derived_from.slice(1)] ?? "")} readOnly disabled />
      </label>
    );
  }
  if (f.type === "bool") {
    return (
      <label title={f.description}>
        <span>{f.label ?? f.name}</span>
        <input type="checkbox" checked={Boolean(val)} onChange={(e) => setValue(f.name, e.target.checked)} />
      </label>
    );
  }
  if (f.type === "enum") {
    return (
      <label title={f.description}>
        <span>{f.label ?? f.name}</span>
        <select value={String(val ?? "")} onChange={(e) => setValue(f.name, e.target.value)}>
          {enumOptions(f, values, context).map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      </label>
    );
  }
  if (f.type === "int" || f.type === "float") {
    return <NumberInput f={f} value={val} required={required} onChange={(v) => setValue(f.name, v)} />;
  }
  return (
    <label title={f.description}>
      <span>{title(f, required)}</span>
      <input type="text" value={String(val ?? "")} onChange={(e) => setValue(f.name, e.target.value)} />
    </label>
  );
}

export default function SchemaForm({ heading, family, schema, values, context, registries, setValues }: {
  heading: string;
  family: string; // 架構的 family：元件依 slot_compatibility 只列跟 (family, 欄位名稱) 相容的
  schema: FieldSpec[];
  values: FormValues;
  context: FormContext;
  registries: Record<string, Component[]>;
  setValues: (v: FormValues) => void;
}) {
  const setValue = (name: string, v: unknown) => setValues(reconcileEnums(schema, { ...values, [name]: v }, context));

  return (
    <div className="model-settings-indicators" data-testid="schema-form">
      <div className="model-settings-indicators-head">
        <span>{heading}</span>
        <span className="model-settings-hint">欄位由後端登記的 params_schema 產生；送出時後端會再完整驗證一次。</span>
      </div>

      {schema.map((f) => {
        if (!conditionMet(f.visible_if, values, context)) return null;

        if (f.type === "array_of_object") {
          const items = (values[f.name] as Record<string, unknown>[] | undefined) ?? [];
          const max = f.max_items ?? Infinity;
          const min = f.min_items ?? 0;
          return (
            <div key={f.name}>
              <div className="model-settings-indicator-row">
                <div className="model-settings-indicator-params">
                  <span className="model-settings-hint">
                    {f.label ?? f.name}（{min}～{max === Infinity ? "不限" : max} 個）{f.description ? `：${f.description}` : ""}
                  </span>
                </div>
              </div>
              {items.map((item, i) => (
                <div className="model-settings-indicator-row" key={i}>
                  <div className="model-settings-indicator-params">
                    {(f.item_schema ?? []).map((sub) => (
                      <NumberInput
                        key={sub.name}
                        f={{ ...sub, label: `#${i + 1} ${sub.label ?? sub.name}` }}
                        value={item[sub.name]}
                        required={isRequired(sub, item, context)}
                        onChange={(v) => {
                          const next = items.slice();
                          next[i] = { ...item, [sub.name]: v };
                          setValue(f.name, next);
                        }}
                      />
                    ))}
                    {items.length > min && (
                      <button type="button" className="model-settings-remove-btn"
                        onClick={() => setValue(f.name, items.filter((_, j) => j !== i))}>×</button>
                    )}
                  </div>
                </div>
              ))}
              {items.length < max && (
                <button type="button" className="model-settings-add-btn"
                  onClick={() => setValue(f.name, [...items, initialItem(f.item_schema ?? [])])}>
                  ＋加一個 {f.label ?? f.name}
                </button>
              )}
            </div>
          );
        }

        if (f.type === "component_ref") {
          const registry = compatibleComponents(f, registries, family);
          const selectedKey = String(values[f.name] ?? "");
          const selected = registry.find((c) => c.key === selectedKey);
          return (
            <div className="model-settings-indicator-row" key={f.name}>
              <div className="model-settings-indicator-params">
                <label title={f.description}>
                  <span>{f.label ?? f.name}</span>
                  <select value={selectedKey} onChange={(e) => {
                    const key = e.target.value;
                    const comp = registry.find((c) => c.key === key);
                    setValues(reconcileEnums(schema, withComponentValues(f.name, comp, { ...values, [f.name]: key }), context));
                  }}>
                    {(componentNullable(f) || !selectedKey) && (
                      <option value="">{componentNullable(f) ? "（不使用）" : "（請選擇）"}</option>
                    )}
                    {registry.map((c) => <option key={c.key} value={c.key}>{c.label ?? c.key}</option>)}
                  </select>
                </label>
                {selected?.params_schema?.map((sub) => (
                  <LeafField key={sub.name} f={{ ...sub, name: `${f.name}.${sub.name}` }}
                    values={values} context={context} setValue={setValue} />
                ))}
              </div>
            </div>
          );
        }

        return (
          <div className="model-settings-indicator-row" key={f.name}>
            <div className="model-settings-indicator-params">
              <LeafField f={f} values={values} context={context} setValue={setValue} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
