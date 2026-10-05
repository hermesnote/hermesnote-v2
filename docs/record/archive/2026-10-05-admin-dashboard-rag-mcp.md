# 2026-10-05 後台 Dashboard＋知識庫（RAG）＋MCP 服務頁（第一版）

> 狀態：**已部署（2026-10-05，Hermes 核准）並完成正式驗收**；版面待 Hermes 登入確認後調整。不需 DB migration。
> 依據：GC 工作包 `tmp/rag-split/CC-DASHBOARD-HANDOFF.md`。NAS RAG（:30450）／MCP（:30451）由 GC 部署，本輪只做網站端。

## 範圍

**後端**（全部限管理員：`APIRouter(dependencies=[Depends(get_current_user)])`）
- `services/agent_services.py`（新）：RAG／MCP 客戶端。
  - 服務 token 只在後端讀取：先讀 `RAG_SERVICE_TOKEN`／`MCP_SERVICE_TOKEN` 環境變數，沒有就讀 `AGENT_SERVICES_SECRETS_FILE`（預設 `/run/secrets/rag-mcp/settings.json`）。
  - 只呼叫寫死的路徑；`source_id` 先驗證是 UUID，不提供可指定任意 URL／path 的代理。
  - 錯誤只回原因，不含 token。
  - 存活（`/health`）和就緒（`/ready`）分開檢查。
  - MCP 連線測試走協定本身：initialize → tools/list → tools/call `knowledge_bases`。工具回應會同時讀 `structuredContent` 和 text content。
- `routers/admin.py`（新），`main.py` 已掛上：
  - `GET /api/admin/overview`：工作總覽。
  - `GET /api/admin/services`：服務狀態。
  - `/api/admin/rag/{knowledge-bases,summary,sources,sources/{id}/content,ingestion-jobs}`（GET）與 `/api/admin/rag/search`（POST）。
  - `/api/admin/mcp/{status,tools}`（GET）與 `/api/admin/mcp/test`（POST）。
- 工作來源 provider：
  - 已接入：模型訓練／推論（`model_training_jobs`，執行中顯示輪數進度）、量化回測（`quant_backtest_jobs`）、知識庫索引（目前版本 `needs_review` 列為待核對；執行中的索引工作列為進行中）。
  - 尚未接入：HA Profile、Claude／Codex SDK，標 `not_connected`，不提供任何工作項目。
  - 統計只算已接入且這次查得到的來源；查不到的來源列為「無法確認（未計入）」並附原因。
  - 工作狀態和交付狀態分開記錄。
- `docker-compose.yml`：backend 唯讀掛載 `/mnt/Hermesnote/hermes-agent/services/rag-mcp/mcp-config` 到 `/run/secrets/rag-mcp`。token 不進 repo、不進 `.env`，也不經過本機。
- `backend/.env`：新增 `RAG_BASE_URL`、`MCP_BASE_URL`（不是秘密；`.env` 已在 gitignore，部署時由 robocopy 同步）。

**前端**
- `pages/admin/dashboard/`：
  - `adminApi.ts`：帶管理員 session 的 fetch；低頻輪詢，失敗時保留上次成功的資料並顯示最後成功時間。
  - `parts.tsx`：狀態標籤、服務卡、錯誤提示、分頁。
  - `dashboard.css`：沿用 index.css 的 token。
- `AdminDashboard.tsx`（取代原本的 placeholder）：
  - 頂端是執行中、待處理、需要處理、服務異常四項，以及統計來源和未接入來源的說明。
  - 主區是正在進行的工作；右區是需要處理、服務狀態（存活、就緒、Agent 已連線三者分開）。
  - 下方是最近完成與交付、快速入口、工作來源。
  - 每 30 秒輪詢一次，分頁不可見時暫停。
- `sections/RagKnowledge.tsx`（`/admin/rag`）：
  - 知識庫選擇，加上總覽／文件／搜尋／索引工作四個分頁。
  - 文件可依類型、狀態篩選並分頁。點開會讀完整索引片段，顯示 version、頁碼或行號、品質警告；警告可直接跳到該頁的片段。
  - 搜尋結果可「回查原片段」（用同一組 source／version 開啟詳細並標出命中片段），也可「複製引用」。
  - PDF 原檔預覽註明需要先補受控來源 API，沒有做失效連結。
  - 沒有重新索引或刪除按鈕，因為服務沒有這類 API。
- `sections/McpService.tsx`（`/admin/mcp`）：MCP 與上游 RAG 狀態、後端憑證是否設定（只顯示來源，不顯示值）、端點、4 個工具的輸入定義（參數表＋完整 JSON Schema）與唯讀標記、連線測試步驟結果、Agent 連線（HA／SDK 尚未接入）。
- `AdminLayout.tsx`：左側選單最前面加上 Dashboard、知識庫、MCP 服務，既有項目不變。`App.tsx` 加上 `rag`、`mcp` 路由。

## 驗證

- **後端**：
  - 新增 `tests/test_admin_api.py`，共 11 項，用 backend `.venv` 執行。conda 訓練環境沒有 google-auth，所以在那個環境無法載入這個模組。涵蓋：
    - 11 個端點匿名都回 401，且不會觸發任何上游呼叫。
    - 非白名單帳號回 403。
    - 總覽統計與 provider 狀態（含 not_connected）。
    - RAG 失聯時其他來源照常、該來源不計入、服務卡顯示「無法連線」。
    - 代理只走固定路徑並帶 token；不合法的 source_id 不會呼叫上游。
    - 上游 401、未設定憑證時的呈現。
    - secrets 檔備援讀取。
    - MCP 狀態、工具、三步驟測試（含 text content 解析）與失敗步驟。
    - 所有回應都不含 token。
  - 既有 201 項照常通過（3 skipped）。
- **前端**：`tsc -b`、`npm run build` 都通過。
- **本機畫面**（Browser pane）：
  - 環境：獨立的檢查用後端（:18000，測試用 JWT 密鑰與白名單，不影響 :8000），模型／回測工作讀正式 hermesnote DB（唯讀查詢），RAG／MCP 用 NAS 實際回應的快照模擬（從 MCP 容器內取得，token 沒有離開容器）。
  - Dashboard：統計 0 執行中、0 待處理、3 需要處理（1 份待核對文件、2 筆手動中斷的訓練）、0 服務異常；最近完成 4 筆訓練、3 筆回測；HA／SDK 顯示尚未接入。
  - 知識庫：
    - 總覽顯示 60 份來源、1944 片段（PDF 30 已索引＋1 待核對、研究卡 28、索引 1）。
    - 篩選「待核對」後只剩 2021_Multi-Relational PDF，點開共 33 片段、2 則警告；點第 3 頁警告後列出 #7–#11 片段，其中 #9、#10 含 U+FFFD。
    - 中文「注意力機制 LSTM」與英文「attention mechanism stock prediction」各 10 筆，第一筆回查後是同一個 source/version/chunk，內容完全一致。
    - 索引工作分頁顯示階段、狀態、片段數、耗時。
  - MCP：4 個工具（knowledge_bases、search_evidence、source_content、indexing_status），都有唯讀與非破壞性標記；連線測試三步驟都通過。
  - 失聯模擬：RAG 改指向不存在的位址後，Dashboard 的「知識庫索引」標為無法確認（未計入），服務卡顯示無法連線，模型與回測區塊照常；知識庫頁顯示「RAG：無法連線」與時間。
  - console 只有預期中的連線失敗與 502，沒有 JS 錯誤。
- **部署前的 NAS 唯讀確認**：
  - 沒有 pending／running 任務（訓練 done 4、failed 2；回測 done 3）。
  - backend 容器連 `192.168.0.44:30450/30451` 的 `/health` 都回 200。
  - 要掛載的 `settings.json` 存在，包含 `mcp_token`、`rag_token`、`rag_url` 三個欄位（只看欄位名稱，沒讀值）。

## 部署與正式驗收（2026-10-05，Hermes 核准）

1. **部署前**：再次唯讀確認沒有 pending／running 任務（訓練 done 4、failed 2；回測 done 3）。
2. **部署**：執行 `deploy.ps1`（exit 0）。backend 與 training 重建，frontend 重啟。backend 掛載為 `/mnt/Hermesnote/hermes-agent/services/rag-mcp/mcp-config -> /run/secrets/rag-mcp`，唯讀（rw=false）。
3. **在 backend 容器內驗收（即時資料）**：
   - RAG、MCP 的 token 來源都是 `secrets_file`。
   - summary 為 60 來源、1944 片段（PDF 30＋待核對 1、研究卡 28、索引 1）。
   - 待核對的 2021_Multi-Relational PDF 有第 3 頁 #7–#11、第 5 頁 #15–#18 片段，都帶品質警告。
   - 中文與英文搜尋的第一筆回查，source/version/chunk 與內容都一致。
   - 索引工作分頁有下一頁（歷史共 63 筆）。
   - MCP 4 個工具；協定測試 initialize、tools/list、tools/call 都通過，回應用 text content 承載。
   - 總覽：0 執行中、0 待處理、3 需要處理、0 服務異常；三個來源都已接入，HA／SDK 為 not_connected。
   - 回應中沒有 token。
4. **公開網址的管理員權限**：`https://hermesnote.com/api/admin/*` 的 11 個端點匿名都回 401，帶偽造 JWT 也是 401。非白名單帳號回 403 由單元測試涵蓋，因為正式環境無法用簽章正確的非管理員 token 實測。`/admin/rag` 未登入會導向 `/admin/login`。
5. **憑證外洩檢查**：NAS 上 `frontend/dist` 所有檔案和 `backend/.env` 都不含兩個 token（在 NAS 端比對，只輸出是否命中）。
6. **既有頁面回歸**：
   - API：`/api/model/jobs`、`/api/backtest/jobs`、`/api/model/registry/metrics`、`/registry/components/optimizer` 都回 200；`/api/auth/me` 匿名回 401。
   - 頁面：`/`、`/model`、`/quant`、`/admin/*` 都回 200。`/model` 實際載入 6 筆訓練紀錄，console 沒有錯誤。
   - 日誌：backend 沒有錯誤；training worker 已啟動並在輪詢。
7. **待 Hermes 確認**：登入後看 `/admin`、`/admin/rag`、`/admin/mcp` 的版面，再依意見調整。

## 測試依賴

新增 `backend/requirements-dev.txt`，內容是 `-r requirements.txt` 加上 `httpx2==2.13.1`（starlette 1.x 的 TestClient 需要它）。這個檔案不進正式映像，Dockerfile 只裝 `requirements.txt`。重跑方式：`pip install -r requirements-dev.txt`，再執行 `python -m unittest discover -s tests -t .`。

## 限制

- 工作總覽一次讀最近 50 筆訓練、30 筆回測。
- 索引工作沒有 source 標題欄位，用目前版本清單（前 100 筆）對照，對不到時顯示 id。
- HA、SDK 工作來源只保留 provider 位置，資料契約由 GC 與 HA 整理。

## 提交

Hermes 核准提交（2026-10-05）。只提交本輪相關檔案：後端 `routers/admin.py`、`services/agent_services.py`、`main.py`、`tests/test_admin_api.py`、`requirements-dev.txt`；`docker-compose.yml`（backend 唯讀掛載）；前端 `pages/admin/dashboard/`、`AdminDashboard.tsx`、`AdminLayout.tsx`、`sections/RagKnowledge.tsx`、`sections/McpService.tsx`、`App.tsx`；本紀錄與 `docs/record/index.md` 的本輪那一列。`backend/.env` 在 gitignore 內，新增的兩個服務位址不進 repo。以下不在本次提交：同日的 HA 盤點紀錄（`2026-10-05-agent-platform-inventory.md` 及其索引列）、他人的修改（`docs/index.md`、`frontend/public/interview-resume.html`、`docs/agent-workflow/` 等）、`tmp/`。
