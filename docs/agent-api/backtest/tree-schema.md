# 條件樹 Schema

> 對應原始碼：`backend/indicators/tree.py`（求值邏輯）、`backend/indicators/filters.py`（filter 節點的計算）。
> 一次回測要送 4 棵樹：`long_entry_tree` / `long_exit_tree` / `short_entry_tree` / `short_exit_tree`（後兩棵可以是 `null`，等於純多方策略，不做空）。每棵樹是一個節點，節點可以巢狀。

## 兩種節點型別

節點只有兩種，用 `"type"` 欄位分辨：

### 1. 條件節點（`"type": "condition"`）

```json
{
  "type": "condition",
  "kind": "long_entry",
  "key": "RSI",
  "params": { "timeperiod": 14, "oversold": 30, "overbought": 70 },
  "weight": 1.0
}
```

| 欄位 | 必填 | 說明 |
|------|------|------|
| `type` | 是 | 固定 `"condition"` |
| `kind` | 是 | `"long_entry"` / `"long_exit"` / `"short_entry"` / `"short_exit"` / `"filter"`，決定這個節點的角色跟怎麼求值（見下） |
| `key` | 是 | 指標代號，對應 `GET /api/indicators` 回傳的 `key`（見 `../indicators.md`） |
| `params` | 否 | 覆蓋這個指標的參數，沒給的部分沿用該指標 `parameters` 的預設值 |
| `weight` | 否 | 只有在**父節點是 `"weighted"` 群組**時才有意義，預設 `1.0`（見下方群組節點說明） |

#### `kind = long_entry` / `long_exit` / `short_entry` / `short_exit`

**只有 `signal_supported = true` 的指標能用**（見 `../indicators.md`）。系統會取該指標訊號模板算出來的「上升方向」或「下降方向」事件，你不用自己選方向，**系統依 `kind` 放在哪個位置自動決定要哪個方向**：

| `kind` | 取哪個方向 | 白話 |
|--------|-----------|------|
| `long_entry` | 上升方向 | 開多 |
| `short_exit` | 上升方向 | 平空（回補）——跟開多是同一個事件，只是意義不同 |
| `long_exit` | 下降方向 | 平多 |
| `short_entry` | 下降方向 | 開空——跟平多是同一個事件，只是意義不同 |

這樣設計的原因：期貨可以直接放空，不需要先持有部位，所以「多方出場」跟「空方進場」在訊號層面本來就是同一個下降事件，只是策略上的意義不同，放的位置不同而已。

額外欄位：
- **`oscillator` 型**（RSI/CCI/WILLR/MFI/ULTOSC/CMO/STOCH）：`params` 裡可以加 `oversold`/`overbought` 覆蓋門檻，不給就用 `signal_defaults`
- **`crossover` 型**（MACD）：`params` 裡可以加 `fast_output`/`slow_output` 指定要比較哪兩條輸出線，不給有預設（`macd` vs `macdsignal`）
- **`pattern` 型**：沒有額外欄位，`params` 只影響指標本身算法（大部分型態指標沒有可調參數）

#### `kind = "filter"`

**161 個指標都能用**，不分方向，四個位置都可以放。是「現在符不符合這個數值門檻」的通用比較，**只會擋進場，不會擋出場**（避免部位因為過濾條件變化就卡住出不來）。

```json
{
  "type": "condition",
  "kind": "filter",
  "key": "ADX",
  "params": { "timeperiod": 14 },
  "output_name": "real",
  "operator": ">",
  "threshold_value": 25
}
```

額外必填欄位：

| 欄位 | 說明 |
|------|------|
| `output_name` | 要比較指標的哪一條輸出線（見 `../indicators.md` 的 `output_names`），不給預設用第一條 |
| `operator` | 比較符號，只支援 `">"` / `">="` / `"<"` / `"<="`（沒有 `"=="`/`"!="`） |
| `threshold_value` | 門檻數值 |

### 2. 群組節點（`"type": "group"`）

```json
{
  "type": "group",
  "mode": "and",
  "children": [ ...節點, ...節點 ]
}
```

| 欄位 | 必填 | 說明 |
|------|------|------|
| `type` | 是 | 固定 `"group"` |
| `mode` | 是 | `"and"` / `"or"` / `"weighted"` |
| `children` | 是 | 節點陣列，**不能是空陣列**；每個元素可以是條件節點，也可以是另一個群組節點——巢狀沒有層數限制 |
| `threshold` | 否，只有 `weighted` 用 | 加權總分要 ≥ 這個值才算成立，不給預設是「所有子節點 `weight` 加總的一半」 |

- **`and`**：全部子節點同時成立才算成立
- **`or`**：任一子節點成立就算成立
- **`weighted`**：每個子節點依它的 `weight`（預設 1.0）加總計分，成立的子節點算 `weight` 分、不成立算 0 分，總分 ≥ `threshold` 就算成立——適合「不用每個條件都滿足，但要湊到一定強度」的情境

## 完整範例

MACD 黃金交叉且 RSI 未過熱才進多方、RSI 突破 75 或 MACD 死叉出場、純多方（不做空）：

```json
{
  "long_entry_tree": {
    "type": "group",
    "mode": "and",
    "children": [
      { "type": "condition", "kind": "long_entry", "key": "MACD", "params": {} },
      { "type": "condition", "kind": "filter", "key": "RSI", "params": { "timeperiod": 14 },
        "output_name": "real", "operator": "<", "threshold_value": 60 }
    ]
  },
  "long_exit_tree": {
    "type": "group",
    "mode": "or",
    "children": [
      { "type": "condition", "kind": "long_exit", "key": "RSI", "params": { "overbought": 75 } },
      { "type": "condition", "kind": "long_exit", "key": "MACD", "params": {} }
    ]
  },
  "short_entry_tree": null,
  "short_exit_tree": null
}
```

## 常見錯誤

- 把 `signal_supported = false` 的指標放進 `long_entry`/`long_exit`/`short_entry`/`short_exit`——只能放 `filter`
- `weighted` 群組忘記給 `threshold` 又希望門檻低於「一半權重」——不給就是預設一半，要別的值要自己填
- `short_entry_tree`/`short_exit_tree` 只給一個、另一個留 `null`——兩個要嘛都給、要嘛都不給（純多方）
