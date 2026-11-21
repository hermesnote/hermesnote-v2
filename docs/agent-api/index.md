# Agent API 文件

> 這個資料夾是寫給**呼叫 API 的 Agent** 看的操作手冊，不是給人看的架構文件（架構文件在 `../architecture.md`）。
> 如果你是 Agent，準備要送一個回測或訓練任務，先讀這份索引，再點進對應的檔案。

## 這份文件在 NAS 上的位置

`deploy.ps1` 每次部署會把這整個資料夾鏡像同步到 `/mnt/Hermesnote/web/hermes/docs/agent-api`（跟本機這份 repo 同一個相對路徑），純粹是複製檔案，不需要重建/重啟任何容器就會生效——2026-09-17 之前這步驟不存在（`deploy.ps1` 只同步 `frontend/dist`／`backend`），是這輪才補上的。查文件是不是最新版，看這個路徑底下檔案的更新時間，或直接比對這裡（git repo）跟那邊的內容。

## 這裡有什麼

| 檔案 | 內容 | 誰會用到 |
|------|------|----------|
| [indicators.md](indicators.md) | 指標參考：TA-Lib 161 個 + 未來自訂指標，`signal_supported`／`params`／`signal_defaults` 怎麼解讀 | 回測、訓練都會用到——共用資源，不屬於任何單一能力 |
| [backtest/api.md](backtest/api.md) | 量化回測怎麼送：`POST /api/backtest/strategy` 請求/回應格式、輪詢、收藏策略、範例 | 送回測任務 |
| [backtest/tree-schema.md](backtest/tree-schema.md) | 條件樹 schema：條件節點／群組節點、`kind` 種類、filter 專用欄位 | 組出合法的進出場條件 |
| [training/api.md](training/api.md) | 模型訓練／推論怎麼送：graph_spec 節點格式、能力查詢（`registry/architectures`／`registry/components/{attention,optimizer}`）與 `params_schema` 解讀、LSTM（方向／Attention／單雙頭／優化器元件）與 XGBoost 完整 payload 範例、提交、結果讀取（best／last、`evaluation`、`use_weights`）、研究操作方針（`patience: 0`）、已知限制 | 送訓練/推論任務 |
| [training/extending.md](training/extending.md) | 擴充指南：新增架構（Transformer／Mamba…）、Attention 型態、優化器、可插拔位置的步驟與契約；多元件組合的未來接點 | 開發者擴充模型 |

## 現在有哪些能力

- ✅ **量化回測**（`backtest/`）——已可用，規則穩定
- ✅ **模型訓練／推論**（`training/`）——2026-09-23 統一模型入口後改寫；落差以 `GET /api/model/registry/*` 的即時清單為準
- ⏳ **實驗重現／自訂實驗設計**——要不要分開兩種文件還沒定案，等三方（Hermes / Hermes Agent / Claude）討論後再建

## 連線位址

後端目前用什麼位址對 Agent 開放（container 內部網路 / 宿主機 IP / 檔案系統直接存取）還沒確定，這份文件先不假設，Agent 自己判斷連得到哪個位址就用哪個。本機開發時後端跑在 `http://localhost:8000`（僅供同機測試參考，不保證是 Agent 實際能用的位址）。

## 認證方式（2026-11-16 更新，對照程式碼）

- **模型 API 的寫入端點要求認證**（`/api/model/*` 的 POST／PATCH／DELETE，共 12 個；細節、權限範圍與錯誤碼見 [training/api.md](training/api.md) 的「認證」一節）。Agent／MCP 用服務金鑰標頭 `X-Hermesnote-Key`；後台網頁用管理員登入的 session JWT。
- **讀取端點仍不需要認證**：所有 GET（`health`、`registry`、`jobs`、結果、逐筆紀錄…）與 WebSocket 進度串流。
- **回測（`/api/backtest/*`）與指標端點目前仍沒有認證**，包含 `POST /api/backtest/strategy` 這類會建立任務的端點；它們的存取控制仍只靠網路層，另案處理。
- 金鑰明文只存在呼叫端的執行環境；網站後端 `.env` 只保存 SHA-256 雜湊。

## 基本流程（以回測為例）

1. 讀 `indicators.md`，決定要用哪些指標
2. 讀 `backtest/tree-schema.md`，把指標組成進出場條件樹
3. 讀 `backtest/api.md`，送出請求、輪詢結果
4. 回測完成後，結果可以在前台 `/quant?job={job_id}` 用瀏覽器看到圖表與交易明細
