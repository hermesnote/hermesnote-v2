// Schema 驅動表單的純邏輯（不含 React）：跟後端 backend/training/registry/param_schema.py
// 同一套 FieldSpec／條件語意。後台表單與驗證用的 harness 都呼叫這裡，後端驗證永遠是最終把關。

export type Condition =
  | boolean
  | { field: string; equals?: unknown; not_equals?: unknown }
  | { all: Condition[] }
  | { any: Condition[] };

export type FieldSpec = {
  name: string;
  type: "int" | "float" | "bool" | "string" | "enum" | "component_ref" | "array_of_object";
  label?: string;
  description?: string;
  required?: Condition;
  default?: unknown;
  min?: number; max?: number; exclusive_min?: number; exclusive_max?: number;
  enum_values?: string[];
  enum_cases?: { when: Condition; values: string[] }[];
  component_registry?: string;
  item_schema?: FieldSpec[];
  min_items?: number; max_items?: number;
  visible_if?: Condition;
  derived_from?: string; // "$outcome_unit" 之類的情境鍵
};

export type Component = {
  key: string; label?: string; description?: string; params_schema?: FieldSpec[];
  slot_compatibility?: string[][]; // [[family, slot_name], ...]
};

// 表單狀態：key 是 FieldSpec.name（dot path）；component_ref 存選定的 key（"" = 不用／尚未選），
// 元件自己的參數存在 `${name}.${子欄位}`；array_of_object 存物件陣列。
export type FormValues = Record<string, unknown>;
// 情境：`$task_type`、`$outcome_unit` 由 Label 設定推得，不是使用者填的欄位
export type FormContext = Record<string, unknown>;

export function conditionMet(cond: Condition | undefined, values: FormValues, context: FormContext): boolean {
  if (cond === undefined || cond === true) return true;
  if (cond === false) return false;
  if ("all" in cond) return cond.all.every((c) => conditionMet(c, values, context));
  if ("any" in cond) return cond.any.some((c) => conditionMet(c, values, context));
  const v = cond.field.startsWith("$") ? context[cond.field.slice(1)] : values[cond.field];
  if ("equals" in cond) return v === cond.equals;
  if ("not_equals" in cond) return v !== cond.not_equals;
  return true;
}

export function enumOptions(f: FieldSpec, values: FormValues, context: FormContext): string[] {
  for (const c of f.enum_cases ?? []) if (conditionMet(c.when, values, context)) return c.values;
  return f.enum_values ?? [];
}

export function isRequired(f: FieldSpec, values: FormValues, context: FormContext): boolean {
  return conditionMet(f.required ?? false, values, context);
}

function initialFor(f: FieldSpec): unknown {
  if (f.default !== undefined) return f.default ?? (f.type === "component_ref" ? "" : undefined);
  if (f.type === "array_of_object") return [initialItem(f.item_schema ?? [])];
  if (f.type === "bool") return false;
  if (f.type === "enum") return f.enum_values?.[0] ?? "";
  if (f.type === "component_ref") return "";
  return undefined; // 研究值必填、沒有預設：留空讓使用者填
}

export function initialItem(itemSchema: FieldSpec[]): Record<string, unknown> {
  const item: Record<string, unknown> = {};
  for (const sub of itemSchema) item[sub.name] = initialFor(sub);
  return item;
}

/** 補上尚未出現過的欄位的初始值；已有的值不覆蓋（使用者填到一半不會被重設）。 */
export function withInitialValues(schema: FieldSpec[], values: FormValues): FormValues {
  const next = { ...values };
  for (const f of schema) if (!(f.name in next)) next[f.name] = initialFor(f);
  return next;
}

/** 選定元件後，補上該元件自己的參數初始值。 */
export function withComponentValues(name: string, component: Component | undefined, values: FormValues): FormValues {
  const next = { ...values };
  for (const sub of component?.params_schema ?? []) {
    const k = `${name}.${sub.name}`;
    if (!(k in next)) next[k] = initialFor(sub);
  }
  return next;
}

/** 這個架構（family）的這個位置（slot＝component_ref 欄位名稱）可以用的元件：依元件登記的 slot_compatibility 過濾，
 * 跟後端 architectures._slot_compatibility_problems 同一條規則。 */
export function compatibleComponents(f: FieldSpec, registries: Record<string, Component[]>, family: string): Component[] {
  return (registries[f.component_registry ?? ""] ?? []).filter((c) =>
    !c.slot_compatibility || c.slot_compatibility.some(([fam, slot]) => fam === family && slot === f.name));
}

/** component_ref 能不能「不選」：只有預設為 null 的可以（例如 attention）；必填的（例如 optimizer）不行。 */
export function componentNullable(f: FieldSpec): boolean {
  return f.default === null;
}

/** 必填的 component_ref 還沒選時，預選第一個相容元件並補上它的參數初始值（研究值沒有預設的仍留空）。 */
export function withRequiredComponents(
  schema: FieldSpec[], values: FormValues, registries: Record<string, Component[]>, family: string,
): FormValues {
  let next = values;
  for (const f of schema) {
    if (f.type !== "component_ref" || componentNullable(f) || next[f.name]) continue;
    const first = compatibleComponents(f, registries, family)[0];
    if (first) next = withComponentValues(f.name, first, { ...next, [f.name]: first.key });
  }
  return next;
}

/** enum 目前的值不在可選範圍時（例如切換 heads 之後 monitor 清單變了），換成第一個合法值。 */
export function reconcileEnums(schema: FieldSpec[], values: FormValues, context: FormContext): FormValues {
  let next = values;
  for (const f of schema) {
    if (f.type !== "enum") continue;
    const opts = enumOptions(f, next, context);
    if (opts.length && !opts.includes(next[f.name] as string)) next = { ...next, [f.name]: opts[0] };
  }
  return next;
}

function setPath(out: Record<string, unknown>, path: string, value: unknown) {
  const parts = path.split(".");
  let node = out;
  for (const part of parts.slice(0, -1)) {
    if (typeof node[part] !== "object" || node[part] === null) node[part] = {};
    node = node[part] as Record<string, unknown>;
  }
  node[parts[parts.length - 1]] = value;
}

/** 表單狀態 → API params：只送「目前可見」的欄位；空值（沒填）不送，讓後端回報必填問題；
 * derived_from 欄位直接用情境值；component_ref 送 null 或 {type, ...該元件參數}。 */
export function buildParams(
  schema: FieldSpec[], values: FormValues, context: FormContext, registries: Record<string, Component[]>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of schema) {
    if (!conditionMet(f.visible_if, values, context)) continue;
    if (f.derived_from) {
      const v = context[f.derived_from.slice(1)];
      if (v !== undefined && v !== null && v !== "") setPath(out, f.name, v);
      continue;
    }
    const v = values[f.name];
    if (f.type === "component_ref") {
      if (!v) { setPath(out, f.name, null); continue; }
      const comp = (registries[f.component_registry ?? ""] ?? []).find((c) => c.key === v);
      const obj: Record<string, unknown> = { type: v };
      for (const sub of comp?.params_schema ?? []) {
        const sv = values[`${f.name}.${sub.name}`];
        if (sv !== undefined && sv !== "") obj[sub.name] = sv;
      }
      setPath(out, f.name, obj);
      continue;
    }
    if (v === undefined || v === "" || (typeof v === "number" && Number.isNaN(v))) continue;
    setPath(out, f.name, v);
  }
  return out;
}
