# 2026-10-05 HA 串接盤點與 Agent 平台設計（Dashboard／MCP／RAG／Agent 執行）

> 狀態：**盤點與設計，未實作、未部署、未改 DB**。NAS 盤點只做唯讀檢查，加上一次最小的 Claude 實呼叫測試（見 §2.3，該測試改寫了一個憑證檔，已如實記錄）。
> 範圍：julie 容器完全排除；HA runtime 的設定、auth、Kanban 只看檔案是否存在，不讀內容、不修改。
> 標記：🟢 已能沿用｜🟡 需要開發｜🔴 訂閱接法尚未通過｜⚪ 查不到或超出能力範圍（只標註）

## 0. 結論

| 類別 | 項目 |
|------|------|
| 🟢 已能沿用 | V2 訓練／推論／回測 HTTP API 與 Agent API 手冊（NAS 已同步）；HA 容器經 LAN `:8000` 連得到 V2（9/17、9/24 HA 回報曾實際送任務成功）；Postgres（hermesnote DB）＋可用 `pg_trgm`；後台 `/admin` Google 登入＋`ADMIN_EMAILS` 白名單；`deploy.ps1` 部署流程；claude-ssh 容器（Claude Code 2.1.216，`claude -p --output-format json` 執行介面，HA 經 `claude-bridge` 以 SSH 呼叫）；NAS 上已有 ollama 容器 |
| 🟡 需要開發 | **P0：API 認證**（公開網域 `hermesnote.com/api/*` 對外 200，寫入端點無驗證）；Agent 執行服務（任務表、佇列、log）；MCP 伺服器；RAG 索引與檢索 API；後台 Agent Dashboard |
| 🔴 訂閱接法尚未通過 | **Claude**：`claude auth status` 回報 `loggedIn: true`、`subscriptionType: pro`，但實際呼叫失敗：`Failed to authenticate: OAuth session expired and could not be refreshed`，所以現在 HA 的 `claude-bridge` 路徑也不能用。**Codex**：NAS 主機與所有非 julie 容器都沒有安裝 Codex CLI／SDK，訂閱接法還沒開始。兩者都要等 Hermes 登入、再通過 §6 的實呼叫驗收，才算可用 |
| ⚪ 查不到／超出範圍 | HA 各 profile 的 openai-codex（ChatGPT 訂閱）目前 token 是否有效（屬 HA auth，本輪不碰）；HA Kanban 內部資料（SQLite 在 HA runtime）；`/mnt/Hermesnote/hermes-agent/{claude,claude-image}` 主機目錄（truenas_admin 沒有讀取權限）；研究文獻庫能否作為 RAG 來源（ACL 未授權 V2 讀取）；個人訂閱用於自動化代理的條款限制（需 Hermes 查核官方條款，這裡不下定論）；「HA 完整規格」：repo 內沒有單一份檔案，本文以 `docs/agent-api/`＋`docs/agent-workflow/`（GC 維護，唯讀引用）作為現行規格來源 |

## 1. 盤點證據

### 1.1 網站／API／DB

- **routers**：`auth`、`indicators`、`strategy`、`model_training`。只有 `GET /api/auth/me` 掛了 `get_current_user`，其餘端點（含 `POST /api/model/train`、`/infer`、`POST /api/backtest/strategy`）都沒有驗證。`docs/agent-api/index.md` 已記載這一點。
- **公開曝露**（本輪實測，只用 GET）：`https://hermesnote.com/api/model/registry/metrics` 回 200，`https://hermesnote.com/openapi.json` 回 200。NPM 把 `/api` 轉發到 8000，所以網際網路上任何人都能用同一路徑打寫入端點。**POST 沒有測試**，因為會建立真實任務。
- **檢索能力**：backend 沒有 embedding、全文或搜尋相關程式。Postgres 容器 `ix-postgres-postgres-1`（PG17）只裝了 `pg_trgm`，沒有 `vector`／`vchord`。immich 那個 pgvecto 屬於 immich，不沿用。
- **後台**：`/admin` 的區塊有 交易系統、模型訓練、量化回測、家庭理財，加上 `AdminDashboard`。可以新增區塊。

### 1.2 部署環境（NAS 容器，julie 除外）

| 容器 | 相關內容 |
|------|----------|
| hermesnote-frontend／backend／training | V2，running。三個容器都沒有 node、claude、codex |
| ix-hermes-agent-hermes-hermes-agent-1 | `nousresearch/hermes-agent:v2026.9.14`（Hermes Agent v0.21.3），node v26.5.1；`/opt/data/bin/claude-bridge`（893B，2026-07-22）；`/opt/data/auth.json`、`profiles/research-orchestrator/auth.json` 存在（未讀內容） |
| ix-claude-ssh-hermes-claude-ssh-hermes-1 | `claude-code-ssh-hermes:1.0`，Claude Code 2.1.216，node v22.23.1；sshd 對外 `2222`；使用者 `claude`；掛載 `/mnt/Hermesnote/hermes-agent/claude → /home/claude`（rw）、`research → /workspace/research`（rw）、`/mnt/Hermesnote/web → /workspace/web`（rw） |
| ix-ollama-ollama-1 | 存在，可作為 embedding 候選（未測模型清單） |

HA profiles 的 provider（只讀 `provider`／`default` 兩個欄位）：六個 research-* profile 都是 `openai-codex`，模型有 gpt-6-astra／gpt-5.6-sol／gpt-5.6-luna。所有 profile 的 `config.yaml` 都沒有 `mcp_servers`。

### 1.3 `claude-bridge` 現行接法

HA 執行 `claude-bridge new|resume <id>`，經 SSH（金鑰登入、known_hosts）連到 `claude@192.168.0.44:2222`，在 `/workspace/web` 下執行 `claude -p --output-format json --dangerously-skip-permissions`，stdin 是 prompt，回傳 JSON（result、session_id、is_error、permission_denials）。

## 2. 個人訂閱執行接法

### 2.1 Claude（建議沿用 claude-ssh 容器，不另裝）

- **接法**：以 claude-ssh 容器作為 Claude 執行器，沿用 `claude -p` headless 介面與訂閱 OAuth（`authMethod: claude.ai`、`apiProvider: firstParty`），不另申請 API key。
- **V2 呼叫方式**：Agent 執行服務比照 `claude-bridge`，用專用 SSH 金鑰呼叫，並加上 `--no-session-persistence`（一次性任務）或 `--resume`（延續任務）、`--max-turns`、`--allowedTools`、`--mcp-config`（接 §3.2 的 MCP）。**V2 的呼叫不使用 `--dangerously-skip-permissions`**，改用工具白名單。
- **保留既有環境**：不改映像，也不改 HA 的 bridge。V2 只新增自己的 SSH 公鑰到 `/home/claude/.ssh/authorized_keys`，這屬於環境修改，需 Hermes 核准。

### 2.2 Codex（尚未安裝，提案）

- NAS 上沒有任何 Codex CLI。提案新建容器 `hermesnote-codex`：node 映像加上 `@openai/codex` CLI，執行 `codex exec --json`。Hermes 用 `codex login`（ChatGPT 訂閱登入，裝置碼流程）登入一次，憑證放在該容器自己的 volume。
- 不共用 HA 的 openai-codex auth：那是 HA runtime 的資產，共用會讓兩邊互相刷新 token。9/17 的紀錄已引用過多用戶端刷新衝突的 issue。
- 可用性待驗證，有兩點：一是 `codex exec` 在 ChatGPT 訂閱登入下的無人值守 headless 呼叫；二是 token 自動刷新能撐多久。

### 2.3 本輪實測結果（如實記錄）

1. `claude auth status`（claude 使用者）：`loggedIn: true`、`subscriptionType: pro`。**這只代表憑證檔存在，不代表可用。**
2. 最小實呼叫（`claude -p "只回覆 OK 兩個字" --no-session-persistence --max-turns 1`，`-w /tmp`，停用 Bash、Edit、Write、Read）：`is_error: true`，錯誤為 `OAuth session expired and could not be refreshed`，`total_cost_usd: 0`。
3. **對環境的影響**：這次呼叫讓 CLI 在刷新失敗後改寫了憑證檔。
   - `/home/claude/.claude/.credentials.json`：504B（mtime 2026-08-17）變成 276B（mtime 2026-10-05 10:04）。
   - `.claude.json` 的 mtime 同時更新。
   - 測試前沒有備份這個檔案。原憑證本來就已過期、無法刷新，功能面沒有差別，但檔案內容已經不是原樣。需要 Hermes 重新登入時會一併覆寫。
4. 推論：憑證檔最後一次正常寫入是 08-17，HA 的 `claude-bridge` 很可能從那之後就無法使用。這點沒有向 HA 求證。

### 2.4 Hermes 需要做的登入步驟（Code 不代為登入）

1. Claude：`ssh -p 2222 claude@192.168.0.44`（或 `docker exec -it -u claude …`），執行 `claude`，再執行 `/login`，用 claude.ai 帳號登入。
2. Codex：等 §5 的容器建好後，執行 `docker exec -it hermesnote-codex codex login`。
3. 登入後交給 Code 跑 §6 的 A1／A2 驗收，**以實呼叫成功為準**。

## 3. 設計

### 3.1 Agent 執行服務（新）

- **位置**：新建 `backend/agents/` 模組，加上獨立 worker 容器 `hermesnote-agent-runner`（Python，與 training worker 同一模式：輪詢 DB、單工或限流）。Web API 只負責寫任務表，不直接跑代理。
- **執行器介面** `Executor.run(spec) -> RunResult`，實作兩種：
  - `ClaudeSshExecutor`：SSH 到 claude-ssh 容器執行 `claude -p --output-format stream-json`。
  - `CodexExecutor`：`docker`／SSH 到 `hermesnote-codex` 執行 `codex exec --json`。
  - 兩者把 stdout 事件逐行寫入事件表，供 Dashboard 即時顯示。
- **資料表**（hermesnote DB，`agent_` 前綴，SQL migration）：
  - `agent_runs`：id、executor、project、title、prompt、cwd、allowed_tools、max_turns、status（pending／running／done／failed／cancelled）、session_id、result、is_error、usage、cost、error、requested_by（admin email 或 API key 名稱）、建立／開始／結束時間。
  - `agent_run_events`：run_id、seq、ts、type、payload。
  - `agent_executor_checks`：executor、checked_at、auth_status、ping_ok、ping_error。這張表讓 Dashboard 把「已登入」和「實呼叫可用」分開顯示。
- **健康檢查**：worker 每 N 小時對每個執行器做一次最小實呼叫，結果寫入 `agent_executor_checks`。是否開啟、頻率多少，需 Hermes 決定（會消耗訂閱額度）。

### 3.2 MCP 伺服器（新）

- **形式**：backend 內掛載官方 `mcp` Python SDK 的 streamable HTTP，路徑 `/mcp`；工具內部直接呼叫既有 service 函式，不走 HTTP 自我呼叫。
- **工具（第一批）**：
  - `registry_list`（模型／元件／指標）
  - `training_submit`、`training_get`、`training_list`
  - `backtest_submit`、`backtest_get`
  - `rag_search`、`rag_get`
  - `agent_run_submit`、`agent_run_get`

  參數 schema 直接沿用 `params_schema`，與 HTTP API 共用同一套驗證。
- **認證**：用 Bearer API key（見 §3.5）。
- **接入**：
  - HA：要在 profile `config.yaml` 加 `mcp_servers`。這屬於 HA／Orchestrator 的範圍，由 Hermes 與 HA 處理，Code 只提供設定範例。
  - Claude 執行器：用 `--mcp-config` 帶入。
  - Codex：用 `config.toml` 的 `mcp_servers` 帶入。

### 3.3 RAG 服務（新）

- **第一階段**（不需要裝新的 DB extension）：
  - 資料表：`rag_documents`（source、path、title、sha256、indexed_at）和 `rag_chunks`（doc_id、ord、heading、text、`tsvector`，外加 `pg_trgm` GIN 索引）。
  - 檢索：`pg_trgm` 相似度與 `simple` 全文混合排序。中文以 trigram 為主。
- **第二階段**（選配）：用 ollama 產生 embedding，向量相似度合併排序。前提是 Postgres 要裝 `pgvector`，這會改動 `ix-postgres` 環境，需 Hermes 核准；不裝的話就只能用應用層向量檔，不建議。
- **來源**：
  - 第一批：`docs/agent-api/`、`docs/spec.md`、`architecture.md`、`decisions.md`、`docs/record/`。
  - 研究文獻 `/mnt/Hermesnote/hermes-agent/research/library/extracted`：要先決定 V2 容器的掛載與 ACL（⚪，牽涉 9/17 的 ACL 決策）。
- **API**：
  - `POST /api/rag/reindex`（admin）
  - `GET /api/rag/search?q=&source=&k=`
  - `GET /api/rag/documents/{id}`

  回傳內容都附 path 和段落位置，方便代理引用來源。

### 3.4 後台 Dashboard（新區塊 `/admin/agents`）

- **執行器卡片**：每個執行器顯示 CLI 版本、登入狀態、最近一次實呼叫結果與時間。登入但呼叫失敗時，紅字顯示「已登入但不可用」。
- **任務列表**：依狀態、執行器、專案篩選；詳情頁顯示事件串流（輪詢）、結果、usage 與 session_id，提供「延續」（resume）與取消。
- **送出表單**：執行器、專案／cwd 白名單、prompt、工具白名單、max_turns。
- **V2 任務來源**：`model_training_jobs` 與回測任務新增 `requested_by` 欄位，區分網站和 HA（API key 名稱）。
- **RAG 狀態**：各來源的文件數、最後索引時間；提供重新索引按鈕。
- **HA Kanban／profile 狀態**：⚪ 不納入第一版。資料在 HA runtime 內，要讀就得碰 HA，需 Hermes 另行決定。

### 3.5 認證（P0，建議最先做）

- 新增 `agent_api_keys` 表（name、hash、scopes、created_at、last_used_at、revoked_at）和 `require_scope()` dependency：
  - 讀取端點：`read`
  - 送任務端點：`submit`
  - agent 執行：`agent`
  - 後台管理：admin JWT
- **相容性**：HA 現有的 HTTP 呼叫沒有帶憑證。上線順序是：先發 key 給 HA，HA 更新呼叫並確認，再開啟強制驗證。在 HA 完成更新前，可以先用 NPM 阻擋外網的寫入方法。HA 走 LAN `:8000`，不經 NPM，不受影響。
- 公開網站讀取（前台 `/model`、`/quant` 用到的 GET）是否保持免驗證，需 Hermes 決定。

## 4. 修改範圍

| 層 | 檔案 |
|----|------|
| DB | `backend/migrations/2026-10-xx_agent_platform.sql`：`agent_api_keys`、`agent_runs`、`agent_run_events`、`agent_executor_checks`、`rag_documents`、`rag_chunks`；訓練與回測任務加 `requested_by` |
| 後端 | `auth.py`（API key 驗證、scope）；所有 routers 的寫入端點掛上 dependency；新增 `routers/agents.py`、`routers/rag.py`；`agents/`（executors、worker、store）；`rag/`（chunker、indexer、search）；`mcp_server.py`（掛載在 `main.py`）；`requirements.txt` 加 `mcp` |
| 容器 | `docker-compose`：新增 `hermesnote-agent-runner`（沿用 backend 映像，換 command）、`hermesnote-codex`（新 Dockerfile） |
| 前端 | `pages/admin/sections/Agents.tsx`（及 css）、`AdminLayout` 導覽加入「Agent」、`App.tsx` 路由 |
| 文件 | `docs/agent-api/`：新增 `auth.md`、`mcp.md`、`rag.md`、`agents.md`；`index.md` 更新能力與認證段落；`spec.md`、`architecture.md`、`decisions.md`、本紀錄。`docs/agent-workflow/` 由 GC 維護，Code 不修改 |
| 不動 | julie；HA runtime、profiles、auth、Kanban、`claude-bridge`；claude-ssh 映像；訓練 Phase 1/2/3 與防洩漏流程 |

## 5. 部署方式（分四批，每批各自核准）

1. **P0 認證**：
   - 套用 migration，部署 backend；先設 `AUTH_ENFORCE=false`（只記錄、不擋）。
   - 發 key 給 HA，HA 改呼叫並確認。
   - 再切換成 `true`，並在 NPM 對外網的 `/api` 寫入方法加阻擋。
2. **Agent 執行（Claude）**：
   - Hermes 重新登入 claude-ssh。
   - 新增 V2 專用 SSH 公鑰（環境修改，需核准）。
   - 部署 agent-runner 與後台 Agent 區塊。
3. **MCP＋RAG 第一階段**：
   - 部署 `/mcp`、`/api/rag/*`，跑首次索引。
   - 提供 HA `mcp_servers` 設定範例，交由 Hermes／HA 決定是否接入。
4. **Codex**：
   - 建 `hermesnote-codex` 容器，Hermes 執行 `codex login`。
   - 接上 `CodexExecutor`。
   - RAG 第二階段（pgvector＋ollama）另案核准。

每批都照既有規則：先確認沒有 pending／running 的訓練任務，再執行 `deploy.ps1` 重建；新容器以 compose 增量加入，不重建既有的非 V2 容器。

## 6. 驗收方案（以實際可用為準）

| ID | 項目 | 通過條件 |
|----|------|----------|
| A1 | Claude 訂閱 | V2 runner 經 SSH 執行最小 prompt，回傳 `is_error=false` 且結果符合預期；24 小時後再跑一次仍成功（驗證 token 刷新）。`auth status` 為 true 不算通過 |
| A2 | Codex 訂閱 | `codex exec --json` 最小 prompt 成功；24 小時後再跑一次成功 |
| A3 | 條款 | Hermes 確認個人訂閱用於自建自動化執行符合 Anthropic／OpenAI 現行條款（Code 不代為判定） |
| B1 | 認證 | 沒帶 key 的外網 POST 回 401（`AUTH_ENFORCE=true` 後）；key 的 scope 不足時回 403；HA 用 key 送訓練成功；前台公開頁正常 |
| B2 | 稽核 | 新任務的 `requested_by` 正確區分網站與 HA |
| C1 | Agent 執行 | 後台送出一筆唯讀任務（例如「列出 docs/agent-api 檔名」）→ 狀態從 pending、running 到 done，事件串流可見，結果與 usage 有保存；取消可中止執行；resume 能延續同一個 session |
| C2 | 失敗呈現 | 故意讓憑證失效（或用測試替身）時，任務標為 failed，原始錯誤可見；執行器卡片顯示「已登入但不可用」 |
| D1 | MCP | 用 MCP Inspector 或 Claude `--mcp-config` 列出工具；`registry_list`、`training_get` 結果與 HTTP API 相同；`training_submit` 的驗證錯誤與 HTTP 一致；沒帶 key 會被拒絕 |
| E1 | RAG | 索引後，用已知詞查詢（例如 `evaluations`、`patience`）的前 3 筆結果包含正確檔案與段落；文件改動後重新索引會依 sha256 更新；中文查詢有結果 |
| F1 | Dashboard | `/admin/agents` 各卡片與列表在正式網域登入後正常；console 無錯誤 |
| G1 | 不回歸 | 後端測試全過；contract harness ALL MATCH；既有訓練與回測 API 行為不變（除了新增認證） |

## 7. 待 Hermes 決定

1. P0 認證的順序與公開 GET 是否保持免驗證；HA 何時改帶 key。
2. Claude 重新登入的時間；是否核准 V2 專用 SSH 公鑰。
3. Codex 容器新建與登入。
4. 健康檢查實呼叫的頻率（會消耗訂閱額度）。
5. RAG 是否納入研究文獻庫（掛載與 ACL），以及是否安裝 pgvector。
6. HA 是否接 MCP（profile 設定由 HA 處理）。
7. 個人訂閱用於自動化的條款確認（A3）。
