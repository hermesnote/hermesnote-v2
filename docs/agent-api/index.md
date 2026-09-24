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

## 認證方式（2026-09-17 對照程式碼確認）

**這個資料夾底下涵蓋的所有端點——`indicators.md`／`backtest/`／`training/` 三份文件列的每一支，讀跟寫（含 `POST /api/model/train`、`/infer`、`POST /api/backtest/strategy` 這類會建立真實任務的端點）——目前都沒有掛任何認證檢查**，查證方式：`grep` 整個 `backend/routers/`，只有 `GET /api/auth/me`（後台網頁自己登入狀態查詢用，跟這份文件無關）有 `Depends(get_current_user)`，其餘全部端點完全沒有 auth dependency。

也就是說：
- 不是「唯讀 API 不帶 token、寫入 API 需要專用憑證還沒驗證」——**唯讀跟送任務的端點是同一套（沒有）認證狀態**，程式碼裡目前根本不存在「Agent 專用憑證」這個機制，不是還沒驗證，是還沒有。
- 現在的存取控制**完全依賴網路層**（哪些網段/容器連得到 backend 這個位址），不是應用層驗證身分。Hermes Agent 容器能連進來查 `health`／`registry`／`openapi.json`，用同一個網路路徑理論上就能直接打 `POST /api/model/train`，程式面不會擋。
- 如果之後要開放給更廣泛的來源、或提交端點要留稽核軌跡（誰送的、什麼時候送的），需要另外設計一層憑證機制（例如 API key 或沿用現有的 JWT），這是要不要做、什麼時候做的產品決策，不是這份文件能自己定案的——先誠實記在這裡，等三方對齊後再決定。

## 基本流程（以回測為例）

1. 讀 `indicators.md`，決定要用哪些指標
2. 讀 `backtest/tree-schema.md`，把指標組成進出場條件樹
3. 讀 `backtest/api.md`，送出請求、輪詢結果
4. 回測完成後，結果可以在前台 `/quant?job={job_id}` 用瀏覽器看到圖表與交易明細
