# Hermes Agent × V2 工作流程

> 新對話先讀本頁，再讀目前待辦與相關紀錄。這裡由 Codex 維護需求討論與串接規劃；網站程式實作由 Claude 負責。

> 2026-09-21 統一簡稱：**HA = Hermes Agent；GC = GPT Codex；CC = Claude Code**。使用者最新指示：暫不修改網站，優先推進 HA 研究工作流程。本頁完成狀態依既有回報，不代表今日已重新查詢 NAS。

> **職責更正（2026-09-21，優先於下文早期提案）**：GC協助HA具備自主推進研究所需的工具與執行能力，處理API、Kanban派工／回報、Profile工具與Skill部署等技術障礙；網站程式缺口交CC。研究卡選擇、是否重現、研究方法與結果判斷由使用者和HA Orchestrator決定，GC不充當額外研究審查或核准關卡。候選Skill的GC檢閱限於工具可用性、授權範圍與部署品質，不藉此接管研究判斷。

## 目的與文件邊界

**使用者確認的建設目標（2026-09-21）**：先讓HA的工具、Profile派工與研究執行流程可運作，再建置MCP整合；MCP是明確後續目標，不再只視為可選支線。更長期由獨立於HA的NAS服務整合Codex SDK、Claude Agent SDK與HA接口，並在個人網站後台呈現跨專案任務與代理狀態。SDK是執行整合元件，Dashboard及任務管理需另外建置；不假定安裝SDK就能同步桌面全部對話。GC負責系統規劃、串接排障與驗收，CC負責網站程式，研究決策由使用者與HA處理。此為方向對齊，未授權本輪立即安裝或部署。

串起「文獻搜尋／閱讀 → 研究卡 → 實驗決策 → 提交訓練／回測 → 結果回寫與審閱」。既有代理的實際角色、名稱與能力，須以 Orchestrator 的交接資料為準。

- 本目錄：需求、角色與權責、串接規劃、決策理由、待確認事項、驗收與討論歷程。
- [`../agent-api/index.md`](../agent-api/index.md)：給執行代理使用的正式 API 操作文件；這裡的提案在實作與驗證前，不視為 API 契約。
- [`../record/index.md`](../record/index.md)：既有 Claude 開發紀錄；本工作流程的討論另記於本目錄的 `record/`。
- `../temp/`：不作為本工作流程的正式紀錄位置。
- [HA 本地官方手冊](../Hermes-Agent-Guide/docs/index.mdx)：實際根目錄 `D:\hermesnote-v2\docs\Hermes-Agent-Guide\docs`；Codex／Claude 查 HA 用法優先使用。使用者於 2026-09-17 提供，來源 tag／commit 待確認，不直接假定與 NAS 安裝版本一致。

## 已確認的工作原則

| 項目 | 內容 | 依據 |
|------|------|------|
| 系統現況 | 使用者表示 V2 已上線，TA-Lib 指標已入庫，訓練與量化回測目前可運作；尚未由本輪獨立驗收 NAS 現況 | 2026-09-17 使用者交接 |
| 需求與實作分工 | Hermes 與 Codex 討論需求；網站前後端程式、缺少的模型與參數交給 Claude | 2026-09-17 使用者指示 |
| 目前串接測試限制 | 由使用者指定的 Orchestrator 4 自己執行；已獲准的單筆訓練及修復後單筆重試均完成。不得自行增加提交，不派發其他 Agent／Profile、不建立 Kanban 派工、不新增或修改 Skill／記錄檔，只在目前對話回報；V2 正常保存任務與產物不在禁止範圍 | 2026-09-17 使用者指示與 HA 回報 |
| 決策與執行分工 | Orchestrator 判斷是否進行實驗；既有 Reproducer／Experimenter 負責重現／實驗，但 V2 提交者及 Critic 必經與否仍待對齊 | 使用者指示＋2026-09-17 Orchestrator 交接 |
| 下一步 | 使用者要求先釐清workspace來源與runtime關係，暫停精確命令白名單提案。唯讀確認Reproducer workspace已有baseline三份流程Skill（references/scripts等為空），runtime自身目錄未有這三份；7/21 Orchestrator紀錄表明當時只建workspace框架，不同步runtime。先讓Orchestrator依既有框架釐清預定工具與部署差異，再討論實作／審批，不新增平行Skill或覆蓋runtime。t_391b803d整體技術驗收未通過；錯誤寫回只是部分機制可用 | 2026-09-21 使用者糾正＋掛載、檔案、歷史工具紀錄唯讀核對 |
| 文件留存 | 整個討論與串接過程需要留存文件，供後續對話追蹤 | 2026-09-17 使用者指示 |

## 目前階段與下一步

**最新驗收與任務範圍（2026-09-21）**：HA回報LSTM `f58245f9-207f-4a76-a69a-86a57f4870a7` 為done、error=null，30輪（0–29）完整、每輪四項指標皆有限值，1255訓練／314驗證／排除1，模型cuda且產物存在。phase與parent_job_id皆null；前台畫面尚未驗收。這補足下文歷史「HA摘要待補」狀態。上一版包含補摘要、選篇、Kanban草案的提示尚未由使用者提交，已撤換為「只推薦一張既有研究卡」單一任務；後續規格、Skill、派工分輪確認。

**研究工作流程對齊中**：使用者要求每個研究決策節點先回報並取得確認，依Scout／Reproducer／Experimenter／Critic／Recorder既有角色分工推進研究循環。未來Skill草案先在workspace審查，再核准同步指定runtime；MCP僅封裝共用操作工具。本輪先討論，不建立Skill／MCP或派工。既有小型訓練請求phase=null，不等於正式Phase1→2→3已驗收。

**2026-09-21 補充定義**：使用者確認決策→Orchestrator透過HA既有Kanban外掛派給Profile→Profile回報Orchestrator→Orchestrator回報使用者→下一步核准。Orchestrator負責撰寫／更新其他研究角色的候選Skill，只寫workspace；使用者與GC檢閱並核准具體版本後，才同步該版本到指定runtime。此為後續執行規則，不是立即派工或寫Skill授權。

**網站修正狀態更新**：已收到CC完成回覆，GC本地核對worker摘要白名單與job_store列表／詳情SQL投影確實存在。CC報告對正式DB的唯讀測試由約22MB降至1696bytes；回覆與紀錄明示尚未部署，未收到後續上線驗收。此次針對大型舊LSTM紀錄載入，不是XGBoost本身訓練造成。網站暫不改動／部署，依使用者最新範圍推進HA討論。

**使用者後續操作驗證**：使用者回報前台切換訓練紀錄已不再造成NAS大量資源占用，歷史推論正常；網站先維持現狀，GC不再追查或改動。XGBoost指標顯示修正狀態仍未核對。本輪開始準備給Orchestrator的第一個研究決策提示：補LSTM摘要、推薦一張既有研究卡及重現理由、提出Kanban/Reproducer任務草案；由使用者轉貼，核准前不實際派工或執行研究。

**HUD 已停用（2026-09-18 更新）**：使用者授權後，Codex 停用 TrueNAS 排程 ID5「Hermes HUD - hermes」，並只對已驗證的 HUD 程序正常終止；保留檔案，HA 主服務與30433監聽正常，30001已關閉。停止後約20秒觀測溫度36.1–38.3°C，未捕捉到V2後端高負載。先前採樣曾81.9–90°C，但兩次間隔一段時間，尚不能確認HUD是唯一根因。未改V2／太太服務，未提交推論或回測。詳見當日紀錄。

**00:12 再次尖峰**：HUD仍不存在，V2 backend原兩個worker再度各接近一核心滿載，溫度84.1–89.9°C；training當下CPU0.05%、GPU0%。需繼續由CC定位後端負載來源，不能以停HUD作為問題已解決的證據。使用者已轉交LSTM測試及CC修正指示；LSTM正式完成回報尚未收到。

目前已完成「單筆訓練流程驗收」：HA 首次提交因 training 映像缺少 scikit-learn 而失敗；CC 修正依賴，使用者回報部署完成後，HA 親自重試一筆並回報成功。Codex 本輪僅整理轉貼證據，未另查 NAS 或送任務。推論、回測及研究卡到正式實驗的完整流程仍待驗收。

已知流程：Scout 搜尋／精讀 → 研究卡 → Kanban 相依摘要卡 → Orchestrator 整合並回報 Telegram，已有 done 任務證據。研究卡 → V2 payload → 提交／監控／結果回寫尚未跑通。各 profile、模型、任務 ID 與證據限制見當日紀錄。

workspace 存取修復已執行：使用者同意先試改後，Codex 保持 App 568／568，透過 SSH 備份並增補指定路徑 ACL，共 285 個路徑。研究 library／progress 增補讀寫、六個研究 profiles 增補讀取、workspace／research／profiles 父層增補列出與通過；保留原 owner、mode、mask 與其他 ACL。UID 568 已實際開啟 148 個檔案，三個研究目錄寫入／讀回／刪除成功，UID 3001 建檔後 568 讀寫的繼承測試成功。未重啟 App、未同步 profiles 或送出實驗。復原備份與驗收限制見當日紀錄。

1. Orchestrator 已透過 read_file 成功讀取 2025 Analysis 文獻研究卡與 workspace STATUS 的檔案開頭，未再發生權限錯誤；這兩條工具路徑已驗收，不等於整套任務流程或全部共享目錄驗收。note、temp、rounds、features、database 等未增補權限。六個共享研究 profiles 根目錄原 owner 為 568:10000，因此不是強制唯讀。五份研究 SOUL 與 runtime 不同的原因仍未查，不覆蓋同步；Orchestrator STATUS 兩份一致。
2. 與 Hermes 確認 V2 提交責任、Human 核准範圍與 Critic 審查時機；目前文件有 Claude 提交及 Critic 按需／必經的衝突。
3. HTTP 連線、Orchestrator GET 與單筆訓練已通過；先 HTTP、暫不建 MCP。手冊已發布且 HA 可讀。Agent API 讀寫端點現皆無應用層認證；Phase 2 schema 尚未修，文件採空字串／空陣列佔位，本次一般流程測試 phase=null 不驗收 Phase 2。
4. 使用者已部署並另複製 workspace/document/agent-api。Codex 驗證本地、Web、workspace 三份各 5 檔 SHA-256 全部一致；HA UID 568 已成功開啟 Web `/mnt/Hermesnote/web/hermes/docs/agent-api` 全部 5 檔，workspace 副本仍 Permission denied。本階段直接使用 Web 手冊固定路徑，不因副本再擴大 ACL；副本留存，未刪改。Experimenter 目前未找到專用實驗操作 Skill。
5. 對齊後以一張適合重現的研究卡完成一次小規模實驗及結果回写；Survey 抽取完成不等於具備可執行規格。

版本／模型：Orchestrator 回報 v0.21.3、Docker、commit `345cd2b0`，現行 `openai-codex`／`gpt-6-astra` 有成功 API 日誌。歷史有過載、429、401，未找到自動退回 Sol 的證據。第二輪回報 30432 由 family-fp API Server 使用；Kanban 核心派工不依賴 API／Webhook adapter 或 Recorder Telegram，但依賴承載 dispatcher 的 Gateway（此次是 Experimenter）。Webhook 斷線的完整原因、TUI／Dashboard 模型顯示差異仍未查明。壓縮門檻建議先保留 85%。本地手冊路徑已確認，來源版本仍待確認。

瀏覽器交接：技術上已驗證能從 Dashboard 歷史頁讀取文字，但使用者考量操作與 token 成本，決定後續由使用者貼回覆、Codex 整理，不再預設直接操作 HA 瀏覽器或讀整段歷史。若未來另有明確要求再使用。Copy last response 本次剪貼簿讀取為空；直接送訊息尚未測試。

## 討論紀錄

| 日期 | 主題 | 文件 |
|------|------|------|
| 2026-09-21 | 統一HA／GC／CC簡稱；核對兩模型測試狀態，釐清研究卡選篇尚未執行；網站暫不修改、優先推HA流程 | [record/2026-09-21.md](record/2026-09-21.md) |
| 2026-09-18 | 使用者授權停用 HUD 守護排程及程序；HA 主服務正常，短時間觀測溫度36.1–38.3°C，尚不判定唯一根因 | [record/2026-09-18.md](record/2026-09-18.md) |
| 2026-09-17 | workspace ACL 修復、手冊發布與 API 連通；首筆訓練失敗後 CC 補 scikit-learn，使用者部署，HA 重試成功。推論／回測與 MCP 尚未驗收或建立 | [record/2026-09-17-kickoff.md](record/2026-09-17-kickoff.md) |

## 維護方式

- 本頁維持最新摘要與下一步；每次有實質進展，補寫 `record/` 當日紀錄並更新上表。同一天優先沿用同一份文件，不因新增議題另建 MD；交接清單也沿用既有檔案。
- 區分「使用者確認」「文件／程式觀察」「提案」「待驗證」，不要把提案寫成已上線功能。
- 紀錄保留決策、理由、未決問題、交接與驗收結果；只保存必要的對話摘要，不逐字堆疊聊天內容。
- 決策改變時，更新本頁並在新紀錄說明取代哪個決策；保留舊紀錄供追溯。
- 正式 API 規格與 Claude 開發紀錄以連結引用，避免複製成多份相互矛盾的文件。
- 不記錄 API key、token、密碼或其他登入秘密。
