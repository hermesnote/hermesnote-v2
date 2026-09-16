# Agent API 文件

> 這個資料夾是寫給**呼叫 API 的 Agent** 看的操作手冊，不是給人看的架構文件（架構文件在 `../architecture.md`）。
> 如果你是 Agent，準備要送一個回測或訓練任務，先讀這份索引，再點進對應的檔案。

## 這裡有什麼

| 檔案 | 內容 | 誰會用到 |
|------|------|----------|
| [indicators.md](indicators.md) | 指標參考：TA-Lib 161 個 + 未來自訂指標，`signal_supported`／`params`／`signal_defaults` 怎麼解讀 | 回測、以後的訓練都會用到——共用資源，不屬於任何單一能力 |
| [backtest/api.md](backtest/api.md) | 量化回測怎麼送：`POST /api/backtest/strategy` 請求/回應格式、輪詢、範例 | 送回測任務 |
| [backtest/tree-schema.md](backtest/tree-schema.md) | 條件樹 schema：條件節點／群組節點、`kind` 種類、filter 專用欄位 | 組出合法的進出場條件 |

## 現在有哪些能力

- ✅ **量化回測**（`backtest/`）——已可用，規則穩定
- ⏳ **模型訓練**——尚未整理成文件，之後會開 `training/`
- ⏳ **實驗重現／自訂實驗設計**——要不要分開兩種文件還沒定案，等三方（Hermes / Hermes Agent / Claude）討論後再建

## 連線位址

後端目前用什麼位址對 Agent 開放（container 內部網路 / 宿主機 IP / 檔案系統直接存取）還沒確定，這份文件先不假設，Agent 自己判斷連得到哪個位址就用哪個。本機開發時後端跑在 `http://localhost:8000`（僅供同機測試參考，不保證是 Agent 實際能用的位址）。

## 基本流程（以回測為例）

1. 讀 `indicators.md`，決定要用哪些指標
2. 讀 `backtest/tree-schema.md`，把指標組成進出場條件樹
3. 讀 `backtest/api.md`，送出請求、輪詢結果
4. 回測完成後，結果可以在前台 `/quant?job={job_id}` 用瀏覽器看到圖表與交易明細
