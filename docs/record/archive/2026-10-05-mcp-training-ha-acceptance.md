# MCP 訓練工具與 HA 接線驗收

2026-10-05，Human 核准先完成接線條件、訓練 MCP、HA 知識及一次輕量訓練驗收。研究方向、文獻 Search/Reading 封裝及 Dashboard 改版另議。

## 部署

- NAS 維護來源：`/mnt/Hermesnote/hermes-agent/services/rag-mcp/mcp_service.py`；Custom App `mcp`，30451，RAG 仍為 30450。
- 新增 `training_registry`、`submit_training`、`training_status`、`training_result`、`recent_tool_calls`。原四個知識工具保留，共九個。
- 訓練工具轉接現有 `http://192.168.0.44:8000/api/model/*`，設定沿用網站 TrainRequest 與 registry；本輪未修改網站或訓練 API。
- MCP 入口沿用 Bearer 認證。既有訓練 LAN API 本輪未新增認證；MCP 接入不改變網站既有端點的存取政策。
- `submit_training` 使用持久化 request_id/payload hash 記錄：同 ID 同設定回傳已接受 job，同 ID 不同設定拒絕；提交結果不明保留待核實，禁止同 ID 盲目重送。不同 ID 是新請求，呼叫者仍須遵守核准次數。
- SQLite `/runtime/mcp-calls.sqlite3` 保存呼叫時間、工具、成功/失敗、耗時及 job ID 摘要；不記錄憑證、完整文獻內容，也不臆測呼叫者身分。提供唯讀 `/api/mcp/v1/calls`；前端尚未加入紀錄區。
- 修改前備份：NAS `backup/training-mcp-20261005T071028Z/`。

## HA

- `research-orchestrator`：四個知識工具及呼叫紀錄。
- `research-reproducer`：上述工具加訓練 registry、提交、進度、結果。
- 兩個 Profile 的 workspace/runtime `config.yaml` 新增 hermesnote MCP 與 toolset；runtime `.env` 保存 token，設定引用環境變數。其他 Skill/runtime 候選規則不變。
- Profile 備份各在 `backups/mcp-connect-20261005T071120Z/`。
- HA CLI 的 MCP discovery filter 使用伺服器名 `hermesnote`；第一次只傳 `mcp-hermesnote` 未載入工具。修正為伺服器名後實際成功。日常新會話使用已更新 config；既有長駐會話須重新載入工具。
- 原生 MCP test 成功，實際 Agent 測試以限定 MCP 工具的 CLI 會話執行，無 Kanban 拆卡、Telegram 通知或其他 Profile 派工。

## 實際驗收

1. 10 項單元測試通過：防重送、payload 衝突、逾時不確定狀態、後端拒絕、路徑限制、進度查詢、紀錄摘要及原知識工具回歸。
2. Orchestrator session `20261005_071543_f39ffc` 實際完成 knowledge_bases → search_evidence → source_content → indexing_status；四次服務端紀錄皆 success。
3. 查回來源為 literature-index 第 5–34 行，source/version/chunk 一致。這次搜尋結果未提供方向標籤公式，Agent 明確回報不足；工具接線通過不代表查詢語意品質已全面驗收。
4. Reproducer session `20261005_071717_0bd2bf` 實際呼叫 registry、提交、進度、結果。
5. 唯一工程測試 job：`ce8ee9bf-3d6c-40a5-919e-2de46bfb8a93`；request_id `a5202610-0500-4000-8000-000000000005`。
6. 沿用已完成 5m job `d2af34ef-30c7-495f-a267-af3eb581c1c9` graph，日期縮至 `[2024-01-02,2024-01-06)`、epochs=2；模型節點改為 `mcp_smoke_20261005`，其餘特徵、window、雙頭/Attention/模型設定保留。phase=1。
7. 任務 done、error=null、CUDA；台灣時間 15:18:24–15:18:30，約 5.4 秒，逐輪紀錄兩筆（epoch 0/1）。保存 graph 與預定 payload 完全一致。
8. HA 經 MCP 讀回四組 evaluations（regression/direction × best/last）及 metric_specs。結果屬工程連通驗收，不作研究效能結論。

HA 證據目錄：`/opt/data/workspace/temp/mcp-acceptance-20261005/`，內含 prompts、training-payload.json、training-result.json。原始呼叫內容另可在上述兩個 HA session 回查。

正式結果頁：`https://hermesnote.com/model?job=ce8ee9bf-3d6c-40a5-919e-2de46bfb8a93`。
