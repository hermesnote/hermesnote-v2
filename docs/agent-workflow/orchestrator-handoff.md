# GC／HA／CC 現況交接（更新：2026-09-24）

本節為新對話接手入口。下方 2026-09-21 舊內容僅供追溯，其未部署、卡片封鎖、流程未通等描述已過時。

## 當前結論

V2 單一模型入口、可組合 LSTM、共用 Adam、schema UI、完整評估與 best/last 已部署。依 HA 最終回報，第一份文獻方法的 Phase 1 移植已跑完 300 epochs，Reproducer、Orchestrator、Recorder 三張卡已完成。後續研究由 Hermes 與 HA Orchestrator 決定；GC 處理卡住的工程／流程問題，不自行安排 Experimenter。

使用者可能在新對話討論本篇文獻的特徵新增，尚未指定新增項目或授權實作。先看正式研究報告、必要原文及現行 V2 能力，不重做連通驗收。

## 分工與使用者要求

- GC 先給精簡摘要與必要方案，使用者同意後才生成 CC／HA 施工提示。集中收斂、一次交付、必要輕量驗證，避免反覆追加與過度測試。
- CC 修改網站；HA 管理 profiles、派工與研究流程；使用者通常代轉訊息，HA 最近使用 TUI。NAS 唯讀已授權，修改、部署、DB 操作依具體授權，不由本文件擴大。
- Reproducer 移植可用的方法到我們的資料及條件，僅既有 Phase 1，不追求完整複製論文；缺項記錄影響，不自動變成平台改造。Phase 1/2/3 流程與切分契約保持不變。
- 模型、Attention、optimizer 以共用元件及相容性規則擴充；UI／API 單一入口，舊測試資料與 payload 不要求相容。
- 本輪神經網路訓練 300 epochs、patience=0；平台保留早停選項。XGBoost 使用 n_estimators，不稱 epochs。
- 預設訓練截至 2024 年底，保留 2025；使用 2025 訓練另議，最晚至 2025 年底。2026 測試先看完整性。起點依研究目標、最早 2011。
- 週日 00:00–08:00 NAS scrub 避開活動，由 HA 排程管理；曾討論週六晚提前停止至週日中午，週六時間未定。V2 scrub／checkpoint／硬中斷恢復新增功能已撤回，勿重新加入。
- 週報簡報使用者主動來要求（最晚週二 22:00），未建立排程。

## 已部署的工程狀態

CC 回報 weights_last migration 已套用；舊測試 jobs12/artifacts10/progress420/preview368/inference0 已清零，model_references 保留；backend/training 重建、frontend 重啟；Agent API 文件同步 NAS。程式尚未 commit（最後回報）。正式庫無 checkpoint 表，不需 DROP。

GC 已直接唯讀確認 NAS：三容器 running、restarts=0；backend/training 近期35行日誌無錯誤，backend 啟動完成、worker polling。health、architecture registry、optimizer registry 均 HTTP200。

- 模型只有 lstm／xgboost；LSTM 支援單雙向×零或一個 Attention×單雙頭。
- Adam 獨立元件；lr、weight_decay 必填，beta1=.9、beta2=.999、eps=1e-8 可調、實際值完整保存。batch_size 位於 params.batch_size，与 params.optimizer 並列。
- 元件端點 GET /api/model/registry/components/{registry}，目前 attention、optimizer；舊 /registry/attention_types 已移除。
- early_stopping.patience=0 關閉早停；best/last 都保存與評估；infer 明列 use_weights。
- API 已可用 inputs:[{"node":"m1","output":"direction_probability"}] 選具名輸出；UI 畫布目前只接 default，使用者接受可後補。
- 混合目前是節點輸出串接、分別訓練，非跨模型聯合訓練；上游仍是樣本內輸出、非 OOF。多 Attention 同時使用尚未實作。

## 本次正式 Reproducer：HA 回報，GC 尚未獨立讀完整研究報告

Job：193c8921-ff5b-4ef4-b120-01d550a1b13b，done/error=null，epoch0–299完整，CUDA，約28.96秒，train/val2668/667，單次 train。歷史 blocked 已解除，不需重授權或重送。

設定：TX 全日日線 [2011-01-03,2025-01-01)，Phase1 random/val .2；window60、log_return horizon1/identity；單向LSTM128/64、additive Attention64、Dense32、雙頭方向>=0、seed42、batch256；Adam lr=.0003、weight_decay=.00001、beta1=.9、beta2=.999、eps=1e-8。完整損失、dropout、特徵參數以保存 payload／報告為準。

特徵背景：OHLCV、SMA/EMA(5/10/20)、RSI、MACD、BBANDS、STOCH、OBV；省略 GARCH/HMM、成交量截尾等，未授權自動擴充。

結果：best epoch126 RMSE .01122352、last epoch299 .01130682、均值基準 .01122737；方向 accuracy55.47%/53.67%、多數類55.47%；balanced accuracy .5000/.5054；ROC-AUC .4988/.4867。best667筆全判class1。HA判斷有訓練擬合、尚無預測優勢；random重疊視窗與驗證選模非獨立時間樣本外結果。逐樣本回歸／機率分布與獨立核對binary仍缺證據，未追加infer。

NAS正式報告：
`/opt/data/workspace/research/topics/attention-dual-head-tx-daily-full/reproduction-v001.md`

詳細報告：
`/opt/data/workspace/temp/0660/attention-dual-head-phase1-v010-reproduction-report.md`

曲線：
`/opt/data/workspace/temp/0660/job-193c8921-ff5b-4ef4-b120-01d550a1b13b-learning-curves.svg`

## 優先閱讀文件

- `docs/agent-api/training/api.md`：現行API與payload、結果讀取。
- `docs/agent-api/training/extending.md`：擴充指南。
- `docs/record/archive/2026-09-24-optimizer-component.md`：最新實作與部署。
- `docs/record/archive/2026-09-23-unified-model-entry.md`：單一入口整合。
- `docs/record/index.md`：歷史索引。
- NAS `/opt/data/workspace/research/records-contract.md`：Markdown研究保存契約。
- HA手冊：`/mnt/Hermesnote/web/hermes/docs/agent-api/`。

Scout 目前是來源整理、與 V2 TA-Lib／Custom 比對去重及用法查詢，不負責自行發明量化定義。舊 research/features 不作新版能力依據。新特徵需先討論定義／來源、現有等價能力及缺口，再決定工程範圍。

目前沒有已核准的新特徵、新訓練或 Experimenter 任務。研究與取捨由使用者和 HA 討論，GC 工程協助按需介入。

---

## 以下為 2026-09-21 歷史內容（非當前狀態）

> **當前工程階段更正（2026-09-21）**：Kanban health驗收t_391b803d未通過，不能以錯誤回寫成功代表整體成功。使用者要求先對齊workspace設計來源、runtime實際內容、Skill的references/scripts與同步範圍。精確curl白名單提案未獲核准且暫停，不重跑或直接改runtime。Reproducer workspace已有baseline/specification、execution、comparison三份流程Skill，來自7/21 Orchestrator只建workspace框架的工作；runtime自身skills目錄未有這三份。先釐清既有設計與技術缺口，不另起重複架構。

> **職責更正（2026-09-21）**：GC負責協助打通HA的工具、API、Kanban派工／回報與Skill部署能力，網站程式缺口交CC。選篇、是否重現、研究方法與研究結果由使用者和HA Orchestrator判斷，GC不增加研究審查關卡，也不替Orchestrator安排逐項研究查核。下文早期選篇與原文審查提案不再作為GC推進主線。既有使用者核准節點與workspace先審後同步runtime原則仍有效；本次職責澄清不等於立即授權派工、修改或部署。

> 最新限制（2026-09-21）：XGBoost與LSTM均已收到HA完整API流程成功摘要，不需補查或重跑。使用者要求一次一項工作；上一版多項提示尚未提交，改為由 **Orchestrator 自己**唯讀推薦一張既有研究卡及理由。不得自行追加任務、派發其他Agent／Profile、建Kanban派工或新增HA Skill／記錄檔。本輪不擬完整重現規格或派工草案；研究決策、Skill審查及實際派工留待逐步確認。

最新對齊（2026-09-21）：使用者與Orchestrator確認研究決策後，由Orchestrator透過HA既有Kanban外掛建立核准範圍內的派工，交給對應Profile；Profile完成回報Orchestrator，再由Orchestrator整合回報使用者，取得下一步確認。Kanban是既有派工機制，MCP不取代它；本輪整理不等於授權立即建卡／派工。

Skill製作責任：Orchestrator負責將已驗證可執行流程整理成Reproducer／Experimenter等角色的候選Skill，新增／修改先且只能寫workspace對應來源。使用者與GC檢閱具體版本並確認後，才能把該版本選擇性同步到對應runtime。未審查不得直接寫runtime，禁止以核准Skill為由整棵覆蓋profile或runtime狀態。此原則取代GC先前「讓各角色自行擬寫Skill」的混合提案。Recorder承接必要成果、決策與歷程，Critic按需提供第二意見。

最新驗收（2026-09-21依HA回報）：research-orchestrator已親自讀取手冊並GET成功，後續已依核准提交XGBoost（修復依賴後成功）及LSTM，均已有完整API驗收摘要。LSTM完整ID為f58245f9-207f-4a76-a69a-86a57f4870a7，30輪完整、cuda、產物存在，前台畫面尚未驗收。兩筆核准規格均phase=null；不等於正式三階段或推論／回測已完成。研究卡選篇與專責角色重現尚無執行回覆；下一步僅唯讀提出一張卡的選篇建議，不視為新提交／派工／Skill部署授權。

狀態：2026-09-17 已收到使用者轉貼的第一輪交接，摘要見 [當日紀錄](record/2026-09-17-kickoff.md)。本文件保留固定交接清單並追蹤未答項目，不是已定案的工作流程規格。

## 本輪待補查

以下為第一輪後提出的補查範圍。使用者已發送 workspace／Gateway 診斷請求；第二輪回覆已由 Codex 在 Dashboard 歷史頁讀取，摘要见當日紀錄。未授權修改 NAS 設定。

1. `/opt/data/workspace` 存取失敗的原始錯誤、使用的工具、程序身分與相關目錄／掛載狀態；區分 OS 權限、掛載不可見與操作核准限制。確認是否只影響 Orchestrator，或也影響 Scout／Reproducer／Experimenter／Recorder；無法確認者明列。
2. 列出要求 Claude 提交實驗、以及 Critic 按需／必經的現有文件精確位置與相關原文。回報執行代理是否能讀到這些規範，先不改寫職責。
3. 前次未回答的 Gateway 問題：30432 的 listener／程序與各 profile 啟動方式，API Server／Webhook／Recorder Telegram 是否被現有派工或結果回報依賴。只回報設定名稱、必要非敏感值與證據，不提供金鑰。

回覆進度：第 1 項已修復指定範圍 ACL；Codex 容器 UID 568 讀取／寫入測試成功，使用者再轉貼 Orchestrator read_file 成功讀取研究卡與 workspace STATUS 開頭，兩條工具路徑驗收通過。其他共享目錄未增補，不等於全部工作流程驗收。第 3 項已回報 family-fp 占用 30432，以及 Kanban 使用本地 DB／dispatcher、Orchestrator 自行回報 Telegram。第 2 項職責衝突仍待對齊，接下來回到 V2 串接。

後續更正：使用者要求先查更新前後差異，Codex 已透過 SSH 只讀比對 TrueNAS 保存的 1.0.9／1.0.21 設定，確認新版新增 runtime UID/GID 568。優先處理 App 執行身分，不先授權 568 或遞迴修改 ACL。詳細證據及原服務帳號 UID 10000／主要 GID 3002 見當日紀錄。

網站側 V2 接口盤點另由 Codex 與 Claude 依使用者分工處理，不能要求 Orchestrator 在未知權限與位址下自行提交測試任務。

後續接口盤點：Codex 已由 HA 容器 UID 568 成功 GET V2 health／registry／OpenAPI；LAN health 位址 `http://192.168.0.44:8000/health` 已驗證。採先 HTTP、後評估 MCP；機器認證、NAS 手冊發布與 Phase 2 必填欄位落差交由 Claude 確認。尚未驗證執行代理工具的 HTTP 呼叫或提交任務，詳細證據見當日紀錄。

## 可交給 Orchestrator 的訊息

我們正在規劃將 Hermes Agent 的研究工作流程接到 Hermes Note V2。V2 目前已能由網站操作訓練與量化回測，接下來要讓負責實驗的代理提交任務、收回結果，並與研究卡對應。Codex 負責與 Hermes 討論需求，Claude 負責網站前後端實作。

請根據目前實際設定與已有文件做一次現況交接；未知部分請明列，不要為了填齊而新增角色或假設能力。這次先回報現況，不更新服務、不修改設定，也不提交實驗。

1. **Profiles 與分工**：列出名稱、角色、實際使用的模型／provider、可用工具，以及各自的讀寫或執行權限。誰找文獻、讀文獻、維護研究卡、設計實驗、提交實驗與審閱結果？尚未有負責者的環節請標明。
2. **研究卡**：儲存位置、格式或 schema、唯一識別方式、狀態如何流轉、誰負責更新。請提供一張去除敏感資訊的既有範例，區分文獻事實、研究推論與待驗證假設。
3. **現有工作流程**：搜尋與閱讀已做到哪裡？代理之間如何派工與交接？哪些步驟已實際跑過，哪些仍只是設計？請附相關文件位置。
4. **實驗決策與執行**：Orchestrator 如何判斷要重現或延伸研究？如何把決策交给執行代理？目前有沒有實驗規格、狀態追蹤、失敗處理與結果回寫約定？
5. **後台串接現況**：是否已能存取 V2 的 Agent API 文件？是否已有後台連線位址、機器認證方式、HTTP 工具或 MCP 工具？只列設定名稱與方式，不提供任何秘密值。
6. **版本與模型問題**：目前 TrueNAS 套件版本、Hermes 本體版本／image tag；Astra 是選單中不存在或實際呼叫失敗？請提供已去除敏感資訊的錯誤內容、時間、實際 model ID、provider／登入方式，以及是否自動退回其他模型。

請最後整理「已可用」「部分完成」「尚未建立」「需要 Hermes 決定」四類事項，讓我們依既有架構制定下一步。

## 收到回覆後

將交接摘要及來源記入 `record/`，更新 [索引](index.md) 的現況與下一步。實際角色名稱及職責以交接資料為基礎，再與 Hermes 確認有衝突或缺漏的部分。
