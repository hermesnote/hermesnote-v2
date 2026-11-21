# Search／Reading MCP 封裝與驗收

2026-10-05，依 Human 指示，在 HA 整理研究內容期間完成 Scout Search／Reading 的 MCP 封裝。未新增研究派工、訓練或正式研究結論；未修改 HA 正在使用的 Profile 設定與 Skill。

## 已部署

TrueNAS Custom App `mcp`，入口仍為 `http://192.168.0.44:30451/mcp/`。新增八個工具，連同既有知識、訓練與紀錄工具共十七個。RAG 仍使用 30450。

| 流程 | 工具 | 用途 |
|---|---|---|
| Search | `literature_search` | IEEE 搜尋，保存 query、purpose、頁次及候選；回傳訂閱標示、年份、作者與歷史下載提示 |
| Search | `literature_article` | 讀取搜尋候選的摘要、DOI、作者及頁面內容，供 Agent 判斷相關性 |
| Search | `literature_download` | 單篇下載已讀摘要且符合既有訂閱條件的候選；要求選用理由，核對 PDF 標題並保存搜尋紀錄 |
| Reading | `reading_library` | 列出現有 PDF 及對應研究卡 |
| Reading | `reading_open` | 建立閱讀 session，回傳原 Scout 模板、頁數、PDF hash、既有卡與版本 |
| Reading | `reading_pages` | 逐頁取得全文、頁碼及行號；標記抽取品質問題 |
| Reading | `reading_page_image` | 回傳原頁 PNG，供 Agent 核對圖表與公式 |
| Reading | `reading_save_card` | 保存 Agent 撰寫的研究卡，檢查版本、必要章節及既有平台對應，修改前備份 |

Search 沿用既有 NCU 登入及 IEEE CDP browser，使用獨立分頁與服務端鎖；不提供任意 URL、JavaScript 或檔案路徑執行入口。現行 Scholar/其他網站搜尋不在這個 IEEE 封裝內。

Reading 原本只有 Skill 與模板，沒有可直接搬用的閱讀腳本。新增的是全文、圖片與保存介面；文獻理解仍由 Agent 完成。已提供全部頁面不等於已理解全文，模板要求的原文依據、缺漏標記與完整閱讀聲明仍須如實填寫。

## 維護與資料

- NAS 維護來源：`/mnt/Hermesnote/hermes-agent/services/rag-mcp/` 下的 `mcp_service.py`、`literature_tools.py`、`reading-template.md`、`start-mcp.sh`、`compose-mcp.json`。
- 本機工程副本與測試：`tmp/rag-split/`。
- 備份：NAS `backup/literature-mcp-20261005T082129Z/`；可還原舊 compose、gateway 與啟動腳本，再透過 Custom App update 重建。
- 容器新增 papers、extracted、Scout 搜尋歷史目錄掛載；補上與 RAG 一致的群組 10000／3001，記憶體上限由 512 MiB 調整為 1 GiB，供 PDF 抽取和圖像渲染使用。
- 新增固定依賴：websocket-client 1.8.0、pypdf 6.1.1、pypdfium2 4.30.0、Pillow 11.3.0。wheel 保存在持久化 runtime，重新建立容器可離線安裝。
- 校園登入從既有 Scout 憑證於 NAS 內複製到獨立 `mcp-config/ieee-settings.json`，0600；未輸出內容，未放進 repo，也未加入網站 backend 掛載的 settings.json。
- 閱讀 sessions、搜尋 contexts、卡片備份與下载 receipts 在 `/runtime/literature/`。原文及研究卡仍使用既有共用 library；新資料由現行 RAG scanner 後續收錄。
- Search history 保持原五欄及 UTC+8、新紀錄置頂。已存在的 PDF 不覆寫；下載成功但歷史保存失敗時保留 receipt，重試會標示待核對，不冒稱完成。
- 研究卡採 hash 版本檢查、修改前備份、0660；Agent 撰寫新內容前須取得全頁全文。原有特徵對應表中的已填映射行不得悄悄移除。

## 驗收

1. 十一項新增測試與十項既有訓練 MCP 回歸測試全部通過。涵蓋路徑穿越／symlink、PDF 和卡片版本變更、未讀全頁、圖像回傳、抽取警告、平台映射保護、備份、重複保存及搜尋／下載條件。
2. 正式 MCP initialize → tools/list 成功，列出十七個工具。
3. 經正式 MCP 搜尋既有 Attention 雙頭 LSTM 文獻、讀摘要、確認已存在的 PDF 並沿用，未重複加入來源。
4. 經正式 MCP 取得該篇五頁全文、第一頁原圖（原生 image content），並以相同內容保存既有研究卡，回傳 unchanged。
5. 31 份 PDF、28 張研究卡，共 59 個原始檔案 hash 全部未變。驗收沒有新增正式研究卡。
6. 隔離目錄額外實測單篇 PDF 下載、五頁解析、標題核對、歷史記錄一次、重複請求沿用、研究卡實際寫入及0660。測試副本沒有送入正式 library 或 RAG。
7. knowledge_bases 及 training_registry 仍正常。匿名工具目錄回401，帶既有服務憑證 /ready 回200；Custom App 健康正常。
8. 原 Scout 分頁保留；MCP 使用另一個分頁。未重啟 HA、未增加 Kanban 卡片或研究任務。

NAS 證據：`mcp-runtime/literature-acceptance.json`、`mcp-runtime/literature-write-acceptance.json`；既有呼叫紀錄可查看各工具成功／失敗。

## 接續方式

用戶端可透過現有 MCP 入口使用新增工具。HA Orchestrator／Reproducer 目前的 include 清單沿用上一輪設定，Scout 此輪也沒有自動切換；本輪未將新增工具注入正在整理研究的會話。待 HA 回覆後，再依實際分工調整工具清單與 Skill 入口。

Search：`literature_search(query, purpose, limit, page)` → 選候選 → `literature_article(context_id, article_number)` → 確認符合研究需求後 `literature_download(context_id, article_number, selection_reason)`。

Reading：`reading_library` → `reading_open(filename)` → 分批 `reading_pages`，遇圖表公式用 `reading_page_image` → Agent 按回傳模板撰寫 → `reading_save_card(session_id, markdown, expected_version)`。已有卡須帶 reading_open 回傳的 card_version，新卡帶 null。

服務封裝及工程驗收完成，接續等待 HA 的研究整理，不以本次工具測試產生研究結論。

## 同日接續：Orchestrator 接線

Human 隨後授權完成接線與 Agent 實測，再交付「同一對話、不建立看板」的研究選題任務。

- Orchestrator 的 workspace/runtime `config.yaml` 同步新增上述八個文獻工具，原五個知識／紀錄工具保留，共十三個允許工具。與備份做語意比較，唯一變更是 MCP include 清單；模型、研究 Skill 與既有看板設定未改。
- 兩份設定備份：各 Profile 的 `backups/literature-mcp-connect-20261005T083909Z/config.yaml`。
- 原生 MCP 連線測試成功。原生 discovery 列出 server 的十七個工具，Profile include 則限定十三個；兩者不是同一個數量。
- Orchestrator 實際驗收 session：`20261005_084001_973d98`。限定 `hermesnote` toolset，不提供看板或其他 Profile 派工工具。
- 初次驗收發現 IEEE session 過期後搜尋導回 NCU 登入頁，舊工具錯把它回傳成 `no_results_or_page_not_ready`。HA 如實停止、未繞過。已修正為辨識登入重導、依既有 Scout 規則只重登一次；未載入結果與認證失敗明確報錯，不與真正零結果混用。
- 修正後共二十三項單元／回歸測試通過。HA 實際完成 literature_search → literature_article → reading_library → reading_open → reading_pages → reading_page_image，六筆服務端呼叫皆 success，取得既有文獻標題、DOI 與五頁頁數。
- HA MCP adapter 將 image content 轉存為 MEDIA 路徑，因此 Agent 尚須呼叫 `vision_analyze` 才能目視原圖。此步曾被自動審查拒絕；只讀路由檢查確認為既有 OpenAI Codex gpt-6-astra 原生圖片輸入（native_vision=True），但審查仍要求 Human 明確授權該頁影像送至 OpenAI。Human 隨後明確要求完成全部測試，才繼續執行。
- 最後的圖片驗收已通過：同一 session 的 `vision_analyze` 工具紀錄（message id 24891）顯示圖片已原生載入模型上下文；HA 從第1頁原圖讀出完整題名與第一作者 Jay Patel。`vision-acceptance.txt` 保存此次回報。搜尋、摘要、全文與原圖閱讀接線全部完成；沒有把單純圖片傳回成功誤當成模型已看圖。
- 證據目錄：`/opt/data/workspace/temp/literature-mcp-connect-20261005/` 的 `acceptance.txt` 與 `acceptance-after-fix.txt`。本次未新增訓練、研究卡或研究派工。
