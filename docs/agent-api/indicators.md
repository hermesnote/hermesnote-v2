# 指標參考

> 這份文件解釋「怎麼讀懂一個指標」，不是把 161 個指標的完整內容複製貼上——那份清單會變動（尤其之後加自訂指標），
> 複製一份靜態文字很快就會跟系統對不上。**永遠用 `GET /api/indicators` 拿當下最新、最準的清單**，這份文件教你怎麼解讀那份清單的每個欄位。

## 目前規模

161 個指標，全部來自 TA-Lib 官方函式，分成 10 大類（`group`）；之後會加入本研究自訂、TA-Lib 沒有的指標（見 `../architecture.md` 提到的 Custom 分類），到時候一樣透過 `GET /api/indicators` 查得到，不用另外找文件。

## `GET /api/indicators` 回傳格式

```json
{
  "groups": [
    {
      "group": "Momentum Indicators",
      "group_zh": "動量指標",
      "indicators": [
        {
          "key": "RSI",
          "display_name": "Relative Strength Index",
          "parameters": { "timeperiod": 14 },
          "output_names": ["real"],
          "signal_supported": true,
          "signal_type": "oscillator",
          "description": "相對強弱指標，衡量近期漲跌力道比例，判斷超買超賣",
          "signal_defaults": { "oversold": 30, "overbought": 70 }
        }
      ]
    }
  ]
}
```

### 欄位解讀

| 欄位 | 意思 |
|------|------|
| `key` | 組條件樹時要填的指標代號（`indicators/tree.py` 的 `key` 欄位就是這個） |
| `parameters` | 這個指標**可以調的參數跟預設值**——key 是參數名稱，value 是預設值。組條件樹時可以整包沿用，也可以覆蓋任一個 |
| `output_names` | 這個指標算出來會有幾條輸出線、各自叫什麼名字（例如 MACD 有 `macd`/`macdsignal`/`macdhist` 三條） |
| `signal_supported` | 這個指標能不能拿來當 `long_entry`/`long_exit`/`short_entry`/`short_exit`（觸發用）。`false` 的指標**只能當 `filter`**（見 `backtest/tree-schema.md`），不能放進出場位置 |
| `signal_type` | `signal_supported=true` 才有值：`"oscillator"`（震盪指標）／`"pattern"`（型態辨識）／`"crossover"`（交叉型，目前只有 MACD） |
| `signal_defaults` | 只有 `oscillator` 型有這個欄位：超賣/超買的預設門檻，`long_entry`（用「上升方向」）會在數值跌破 `oversold` 時觸發，`long_exit`/`short_entry`（下降方向）會在數值突破 `overbought` 時觸發 |

## 三種 `signal_type` 各自怎麼判斷方向

- **`oscillator`**（例：RSI、CCI、WILLR、MFI、ULTOSC、CMO、STOCH，共 7 個）：數值從高處跌破 `oversold` 門檻＝上升方向事件（通常對應「超賣後準備反彈」）；數值從低處突破 `overbought` 門檻＝下降方向事件。門檻可以用 `signal_defaults`，也可以自己指定不同數值——**不建議照抄預設值不加思考**，同一組門檻套用在不同商品/週期上，靈敏度可能完全不合理，這個判斷本身沒有標準答案，是策略設計的一部分。
- **`pattern`**（例：K 線型態辨識，共 61 個）：TA-Lib 原生輸出已經是標準化的 ±100/0（+100=看漲型態成立，-100=看跌型態成立，0=沒有型態），上升方向＝看漲，下降方向＝看跌，沒有額外參數可調。
- **`crossover`**（目前只有 MACD）：比較兩條輸出線的交叉，快線上穿慢線＝上升方向，快線下穿慢線＝下降方向。

## 常見參數名稱的通用意思

TA-Lib 的參數命名有慣例，不用每個指標分開記：

| 參數名稱 | 意思 |
|----------|------|
| `timeperiod` | 計算窗口的 K 棒根數（回看幾根算一次），數字越大越平滑、反應越慢 |
| `fastperiod` / `slowperiod` | 快線／慢線各自的天期（MACD 這類雙均線指標） |
| `signalperiod` | 訊號線的平滑天期 |
| `fastk_period` / `slowk_period` / `slowd_period` | KD 系列指標的 K 值／D 值計算與平滑天期 |
| `nbdev`（如 BBANDS） | 標準差倍數，通道開多寬 |

## 沒有 `signal_supported` 的 92 個指標怎麼用

這些不能放進出場位置，但**全部都能當 `filter` 用**——`filter` 是通用門檻比較（例如 ADX 大於 25 才允許進場，避免盤整期被雜訊洗），不分方向，四個位置（`long_entry`/`long_exit`/`short_entry`/`short_exit`）都可以放，用來擋不想要的情境，不會單獨觸發進出場。細節看 `backtest/tree-schema.md` 的 `filter` 節點格式。
