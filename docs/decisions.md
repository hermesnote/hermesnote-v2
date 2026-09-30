# Hermesnote V2 — Decisions

> 讀者：接手實作的 AI。記錄「為什麼這樣設計」，讓 AI 不把刻意的設計當成 bug 修掉，也不重提已否決或已撤回的方案。
> `spec.md`／`architecture.md` 以 `D-xxx` 引用本檔。狀態：**採用中**／**已取代**（指向新決策）／**已撤回**。
> 新增決策：編號遞增，不改寫舊條目；被取代時只更新狀態欄。
> 來源連結指向 `docs/record/`（逐輪詳細紀錄）。最後整理：2026-09-29。

## 索引

| ID | 決策 | 狀態 | 日期 |
|---|---|---|---|
| D-001 | 重做 V2：研究敘事型網站，保留 FastAPI／React | 採用中 | 2026-08-17 |
| D-002 | 沿用 V1 原則：教授三階段 train／val／test 協定 | 採用中 | 2026-09-08 |
| D-003 | 沿用 V1 原則：data leakage 防範 | 採用中 | 2026-09-08 |
| D-004 | V1 封存，只參考 D-002／D-003 | 採用中 | 2026-09-08 |
| D-005 | 回測計算用 vectorbt | 採用中 | 2026-09-02 |
| D-006 | 圖表用 lightweight-charts | 採用中 | 2026-09-02 |
| D-007 | 後台：Google OAuth＋email 白名單，不開放註冊 | 採用中 | 2026-09-02 |
| D-008 | 策略＝四棵條件樹，可巢狀 AND／OR／加權 | 採用中 | 2026-09-02 |
| D-009 | 回測改非同步 job＋DB 持久化＋有界分頁 | 採用中 | 2026-09-06 |
| D-010 | 實驗由 Agent 經 API 送出，`docs/agent-api/` 為 Agent 手冊 | 採用中 | 2026-09-06 |
| D-011 | 三容器部署；`deploy.ps1` 只更新程式、不建容器 | 採用中 | 2026-09-08 |
| D-012 | 訓練改為 Feature／Label／Model 節點 DAG | 採用中 | 2026-09-08 |
| D-013 | 訓練獨立容器：job 表＋worker 輪詢＋NOTIFY／WebSocket | 採用中 | 2026-09-08 |
| D-014 | Label 拆成 outcome＋labeling_rule；Phase 血緣用 `phase`＋`parent_job_id` | 採用中 | 2026-09-10 |
| D-015 | 產物在訓練完成當下存 DB（bytea），不存檔案 | 採用中 | 2026-09-10 |
| D-016 | 訓練抽樣預覽與正式推論分開；推論分批串流寫入 | 採用中 | 2026-09-16 |
| D-017 | `job.result` 只存摘要 | 採用中 | 2026-09-18 |
| D-018 | 具名輸出引用＋共用提交驗證 | 採用中 | 2026-09-21 |
| D-019 | 識別時間與資訊可用時間分開 | 採用中 | 2026-09-21 |
| D-020 | 研究值不由平台代填預設 | 採用中 | 2026-09-21 |
| D-021 | scrub 維護閘門／暫停／checkpoint／硬中斷恢復 | 已撤回 | 2026-09-23 |
| D-022 | 多個 LSTM key（`lstm_attention_dual`、`lstm_bidirectional`、`lstm_custom`） | 已取代 → D-023 | 2026-09-21～23 |
| D-023 | 單一模型入口＋schema 驅動 UI／API | 採用中 | 2026-09-23 |
| D-024 | 訓練控制：輪數各自表達、`patience=0` 關閉、best／last 全保存 | 採用中 | 2026-09-23 |
| D-025 | 分類類別數由 Label 決定 | 採用中 | 2026-09-23 |
| D-026 | XGBoost 用原生 `xgb.train` | 採用中 | 2026-09-23 |
| D-027 | JSONB 非有限數值存 null | 採用中 | 2026-09-23 |
| D-028 | 移除舊版／未遷移相容分支，migration 先於部署 | 採用中 | 2026-09-23 |
| D-029 | 優化器改為可共用元件 | 採用中 | 2026-09-24 |
| D-030 | 元件查詢改通用端點；`batch_size` 移到頂層 | 採用中 | 2026-09-24 |
| D-031 | worker 依實際可用顯存判斷 GPU 忙碌 | 採用中 | 2026-09-29 |
| D-032 | Reproducer 操作方針：`patience: 0`、`epochs: 300` | 採用中 | 2026-09-23 |
| D-033 | DB 權限：hermesnote 讀寫、quotes 唯讀 | 採用中 | 2026-09-03 |
| D-034 | 報價讀取依 `built_at` 去重 | 採用中 | 2026-09-02 |
| D-035 | 版本鎖：Vite ^6.3.1、plotly 5.24.1 | 採用中 | 2026-08-18／09-02 |
| D-036 | 通用指標視覺化：指標登記表＋`metric_specs`＋`evaluations` 清單 | 採用中 | 2026-09-30 |

---

## V1 → V2

### D-001 重做 V2：研究敘事型網站，保留 FastAPI／React
- **決定**：新 repo `D:\hermesnote-v2` 重做 hermesnote.com，定位為研究敘事型個人網站。後端維持 FastAPI、前端維持 React，換分層不換框架；不保留 V1 的 API 設計、訓練紀錄、會議紀錄資料。
- **理由**：V1 缺少分層，feature library 與資料來源寫死在程式（例：V1 `backend/routers/tasks.py` 572 行）；重做的首要目的是刻意練習後端架構分層，其次才是對外展示。
- **替代方案**：在 V1 上重構（否決：技術債重，且研究資料與網站程式混在一起）；換框架（否決：練習重點是分層，不是新框架）。
- **來源**：`record/archive/2026-08-17.md`

### D-002 沿用 V1 原則：教授三階段 train／val／test 協定
- **決定**：Phase 1 訓練期間內 random 切分；Phase 2 繼承 Phase 1 設定，只改 chronological 切分；Phase 3 對 Phase 2 模型做 holdout 推論，holdout 不得與訓練期間重疊。此協定不可變動（INV-01）。
- **理由**：教授要求的研究流程。V2 只延續原則，程式重新實作。
- **替代方案**：只做 train／val 兩段（否決：無法做乾淨的未見資料測試）。
- **來源**：`record/archive/2026-09-08.md`、`record/archive/2026-09-10-training-v2-redesign-plan.md`

### D-003 沿用 V1 原則：data leakage 防範
- **決定**：正規化等前處理只用訓練集 fit，驗證／測試一律沿用訓練集參數；時間對齊與切分不得使用未來資訊（INV-02）。
- **理由**：教授要求；避免評估結果虛高。
- **來源**：`record/archive/2026-09-08.md`

### D-004 V1 封存，只參考 D-002／D-003
- **決定**：V1（`D:\hermesnote`）停用封存，不維護、不沿用程式；只有 D-002、D-003 兩項原則可參考。
- **理由**：2026-09-08 曾把整段 V2 訓練系統設計誤在 V1 進行，Hermes 明確劃定此邊界。
- **來源**：`record/archive/2026-09-08.md`

## 網站與量化回測

### D-005 回測計算用 vectorbt
- **決定**：回測用 vectorbt（`Portfolio.from_signals`）。
- **理由**：向量化運算，大量參數掃描快，適合 Agent 連續送多組參數；「逐根重播」是前端呈現的事，後端一次算完即可。
- **替代方案**：backtrader（開發趨緩、新專案少推薦）、zipline-reloaded（環境設定重）。
- **限制**：不適合逐筆即時處理；未來接券商即時交易需另一條路。
- **來源**：commit `395e357`、`record/archive/2026-09-02.md`

### D-006 圖表用 lightweight-charts
- **決定**：前端圖表用 TradingView 開源的 `lightweight-charts`；回測重播與未來即時報價共用同一套元件。
- **替代方案**：TradingView Advanced Chart Widget（只能顯示報價，不能畫自己的回測結果）、MultiChart（桌面軟體，無法嵌入網頁）。
- **來源**：`record/archive/2026-09-02.md`

### D-007 後台：Google OAuth＋email 白名單，不開放註冊
- **決定**：前端取 Google ID token，後端驗證並比對 `ADMIN_EMAILS` 白名單，簽發 12 小時 session JWT。公開頁只呈現，操作都在後台。
- **理由**：只有 Hermes 使用；不自建帳密系統。
- **替代方案**：自建帳密（否決：維護成本與安全風險）；開放註冊（否決：不是多人平台，SCOPE-04）。
- **來源**：`record/archive/2026-09-02.md`

### D-008 策略＝四棵條件樹，可巢狀 AND／OR／加權
- **決定**：多進／多出／空進／空出各一棵樹，對應 vectorbt 的四條訊號；節點為條件或群組，群組可無限巢狀；觸發方向依所在位置自動決定；`filter` 只擋進場。
- **理由**：直接對應 `from_signals` 的四個輸入；使用者不必自己選方向。
- **替代方案**：兩層 A 組／B 組設計（`indicators/combo.py`，試過後放棄並刪除：表達力不足）。介面參考 TrendSpider（遞迴群組）、TEJ Pro（加權共存）、Zerodha Streak（白話預覽）。
- **來源**：`record/archive/2026-09-02.md`

### D-009 回測改非同步 job＋DB 持久化＋有界分頁
- **決定**：送出即回 `job_id`，背景執行，結果存 `quant_*` 表，查詢全部有界分頁。
- **理由**：全範圍（TX 1 分鐘線 15 年、約 2.9M 根）壓力測試下，同步版本撐爆後端記憶體、瀏覽器收到巨大 JSON 卡死。
- **限制**：背景執行用 `asyncio.create_task`，不是正式佇列，後端重啟會遺失執行中任務（個人專案接受）。
- **來源**：`record/archive/2026-09-06.md`

### D-010 實驗由 Agent 經 API 送出，`docs/agent-api/` 為 Agent 手冊
- **決定**：HA 透過 HTTP API 送回測／訓練／推論；`docs/agent-api/` 是寫給 Agent 的操作手冊，部署時同步到 NAS 供 HA 讀取。先用 HTTP，暫不建 MCP。
- **理由**：網站不必是操作台；Agent 需要一份與程式一致、可直接照做的契約。
- **來源**：`record/archive/2026-09-06.md`、`docs/agent-workflow/index.md`（2026-09-17）

### D-011 三容器部署；`deploy.ps1` 只更新程式、不建容器
- **決定**：frontend／backend／training 三容器（docker compose）；`deploy.ps1` 先確認三容器都在跑才部署，否則中止；build 與 up 分開執行；每次 `restart frontend`。
- **理由**：容器層問題要單獨排查，不應被部署腳本「順手建一個」掩蓋；build 與 up 合在一行時 build 失敗不會讓 SSH 回傳非 0（2026-09-08 實際發生）。
- **來源**：`record/archive/2026-09-08-deploy.md`

## 訓練系統

### D-012 訓練改為 Feature／Label／Model 節點 DAG
- **決定**：實驗＝節點圖（JSON），Feature Node 產特徵、Label Node 產目標、Model Node 接任意上游陣列；執行引擎依拓樸排序執行，新增組合方式不改引擎。
- **理由**：取代 V1 以 mode 字串分流（單／多時間框架 × 單／混合模型的固定矩陣），可表達任意融合結構，訓練與推論共用同一張圖。
- **來源**：`record/archive/2026-09-08.md`

### D-013 訓練獨立容器：job 表＋worker 輪詢＋NOTIFY／WebSocket
- **決定**：web 容器只寫 pending job，不碰 GPU／torch；training 容器的 worker 輪詢執行；逐輪進度寫表並 NOTIFY，web 端 LISTEN 轉 WebSocket。
- **理由**：GPU 工作與 API 服務隔離；兩者只共用 DB，不需另外的容器間協定。
- **來源**：`record/archive/2026-09-08-training-plan.md`

### D-014 Label 拆成 outcome＋labeling_rule；Phase 血緣用 `phase`＋`parent_job_id`
- **決定**：outcome 算原始數值，labeling_rule 轉學習目標（分類或回歸），兩者可自由組合（不合法組合由後端擋）；Phase 2／3 以 `parent_job_id` 指向上一階段。
- **理由**：三方（Hermes／Codex／Claude）對齊的組件契約（A 時間戳～G 回測修正）。
- **來源**：`record/archive/2026-09-10-training-v2-redesign-plan.md`

### D-015 產物在訓練完成當下存 DB（bytea），不存檔案
- **決定**：訓練成功當下把權重與設定存進 `model_artifacts`；「收藏」只是標記。
- **理由**：模型物件在 job 結束後就消失，不能等使用者按儲存；training 與 backend 容器沒有共用 volume，存檔案兩邊互相讀不到。
- **來源**：`record/archive/2026-09-10-training-v2-redesign-plan.md`、`backend/training/artifacts.py`

### D-016 訓練抽樣預覽與正式推論分開；推論分批串流寫入
- **決定**：訓練中抽樣預覽存 `model_training_preview_samples`（展示用）；正式推論用 `/infer`，以 `sliding_window_view` 分批算、分批寫 `model_inference_predictions`；保留完整機率向量（`output_type`）。
- **理由**：兩者性質不同；全期間推論可達百萬列，一次算完會耗盡記憶體。
- **來源**：`record/archive/2026-09-16-preview-inference-hardening.md`

### D-017 `job.result` 只存摘要
- **決定**：worker 以白名單只存 `final_metrics`／`device`／`training_meta`／`output_specs`／`evaluation`；查詢端 SQL 再投影掉歷史大欄位。
- **理由**：曾把逐樣本陣列存進 result，單筆回應 22MB，造成 NAS 高負載。
- **來源**：`record/archive/2026-09-18-job-result-bloat-fix.md`

### D-018 具名輸出引用＋共用提交驗證
- **決定**：`inputs` 可用字串（取 default）或 `{node, output}`；輸出名稱是通用語意，實際意義看 `output_specs`；API、Phase 2 繼承、worker 共用 `validate_graph_spec`，失敗不建 job。
- **理由**：多頭模型需要讓下游選擇輸出；驗證集中一處，錯誤一次列出。
- **來源**：`record/archive/2026-09-21-named-outputs-contract.md`

### D-019 識別時間與資訊可用時間分開
- **決定**：`datetime`＝K 棒識別時間（對齊、切分用）；`bar_end_ts`＝資訊可用時間（日線 13:45）；樣本另帶 `available_ts`，只標時間不改計算。
- **理由**：日線 08:45 的棒要到收盤才有完整資訊，混用會造成洩漏或誤判。
- **來源**：`record/archive/2026-09-21-attention-dual-head-capability.md`

### D-020 研究值不由平台代填預設
- **決定**：研究端尚未確認的超參數一律必填、無預設。
- **理由**：避免平台預設值悄悄變成研究設定。
- **來源**：`record/archive/2026-09-21-attention-dual-head-capability.md`、`record/archive/2026-09-23-unified-model-entry.md`

### D-021 scrub 維護閘門／暫停／checkpoint／硬中斷恢復【已撤回】
- **原決定**：週日 00:00–08:00 維護窗口暫停訓練，checkpoint 後接續，搭配硬性截止復原。
- **撤回**：2026-09-23 Hermes 要求撤回整批功能（排程、暫停、硬中斷恢復）；相關程式、測試、建表語句已移除，`model_training_checkpoints` 從未套用到正式庫。不得重新引入（INV-09）。
- **來源**：`record/archive/2026-09-23-scrub-withdrawal-and-lstm-custom.md`、`record/archive/2026-09-23-lstm-custom-independence-schema-ui-checkpoint-audit.md`

### D-022 多個 LSTM key【已取代 → D-023】
- **原決定**：`lstm`、`lstm_bidirectional`、`lstm_attention_dual`、`lstm_custom` 平行並存，各有專用表單。
- **取代原因**：組合數增加後 key 與表單重複；雙向摘要取法不一致。
- **來源**：`record/archive/2026-09-21-attention-dual-head-capability.md`、`record/archive/2026-09-23-attention-family-eval-expansion.md`、`record/archive/2026-09-23-scrub-withdrawal-and-lstm-custom.md`

### D-023 單一模型入口＋schema 驅動 UI／API
- **決定**：每個架構一個 key（`lstm`、`xgboost`），以 `register()` 登記 `params_schema`／`capabilities`／`slots`；同一份 schema 驅動後台表單與後端驗證；LSTM 方向／Attention／輸出頭三軸獨立；雙向摘要用兩方向各自最終 hidden state；舊 key 與舊表單刪除（舊紀錄已確認可捨棄）。
- **理由**：新模型與元件只要登記即可，不改引擎、API、表單；UI 與 API 規則不會分歧。
- **來源**：`record/archive/2026-09-23-unified-model-entry.md`

### D-024 訓練控制：輪數各自表達、`patience=0` 關閉、best／last 全保存
- **決定**：神經網路用 `epochs`、boosting 用 `n_estimators`；`early_stopping.patience=0`＝關閉（不另設 `enabled`）；所有模型保存 best 與 last 兩組權重並各自完整評估；XGBoost 監控指標固定。
- **理由**：研究需要比較跑滿與最佳輪；兩種輪數語意不同，混用會誤導。
- **來源**：`record/archive/2026-09-23-unified-model-entry.md`

### D-025 分類類別數由 Label 決定
- **決定**：`n_classes` 只在 Label 的 params，不是模型參數。
- **理由**：類別數是目標定義的一部分；放兩處會不一致。
- **來源**：`record/archive/2026-09-23-unified-model-entry.md`

### D-026 XGBoost 用原生 `xgb.train`
- **決定**：不用 sklearn 的 `XGBClassifier`；`num_class` 取自 Label；多分類用 `multi:softprob`；best＝截到 best round 的樹。
- **理由**：sklearn 包裝會從 y 推類別，3 類、門檻 0 時「平」不出現就報錯；原生 API 數值與改動前一致。
- **來源**：`record/archive/2026-09-23-unified-model-entry.md`

### D-027 JSONB 非有限數值存 null
- **決定**：寫 JSONB 前經 `training/json_safe.py` 把 NaN／Infinity 轉 `null`；評估指標另給 `*_unavailable_reason`。
- **理由**：多分類驗證集缺類別時 ROC-AUC 為 NaN，PostgreSQL JSONB 不接受，訓練跑完會寫不進結果。
- **來源**：`record/archive/2026-09-23-unified-model-entry.md`

### D-028 移除舊版／未遷移相容分支，migration 先於部署
- **決定**：不再為舊資料或未遷移的 DB 保留程式分支（例：`weights_last` 欄位存在檢查、metrics 欄可能不存在）；需要的 migration 在部署前套用。
- **理由**：舊紀錄已確認可捨棄；相容分支會讓程式路徑變多、測試不完整。
- **來源**：`record/archive/2026-09-23-unified-model-entry.md`、`record/archive/2026-09-24-optimizer-component.md`（執行紀錄）

### D-029 優化器改為可共用元件
- **決定**：`optimizers.py` 登記優化器（目前 Adam：lr、weight_decay 必填；beta1 0.9、beta2 0.999、eps 1e-8）；模型以 optimizer slot 引用，訓練經 `build_optimizer()` 建立；`model_config.config.optimizer` 保存名稱與補齊預設後的完整設定。`weight_decay` 維持 Adam 的 L2 形式，AdamW 若需要另登記新元件。
- **理由**：D-023 曾移除 betas／eps 且未保存實際值，不利重現；元件化後其他模型可共用、新優化器不改模型程式。
- **來源**：`record/archive/2026-09-24-optimizer-component.md`

### D-030 元件查詢改通用端點；`batch_size` 移到頂層
- **決定**：`GET /api/model/registry/components/{registry}` 取代 `/registry/attention_types`（不保留舊端點）；`batch_size` 不屬於優化器，移到 params 頂層。
- **理由**：新增元件登記表時 API 與前端不用改；參數歸屬清楚。
- **來源**：`record/archive/2026-09-24-optimizer-component.md`

### D-031 worker 依實際可用顯存判斷 GPU 忙碌
- **決定**：開始任務前先釋放 worker 自身的模型引用與 CUDA 快取，再以 `nvidia-smi` 的 `memory.free` 對照 `TRAINING_GPU_MIN_FREE_MB`（預設 4096）；延後時記錄原因與數值；每筆任務結束也釋放。
- **理由**：舊判斷「總已用 > 2000MB」把 worker 自己留下的約 2.3GB 當外部佔用，16GB GPU 使用率 0% 時新任務仍一直 pending。
- **替代方案**：只調高固定門檻（否決：仍會把自身快取誤判為外部佔用）。
- **來源**：`record/archive/2026-09-29-worker-gpu-capacity-fix.md`

### D-032 Reproducer 操作方針：`patience: 0`、`epochs: 300`
- **決定**：Reproducer／Experimenter 送出的訓練明確關閉早停、跑滿 300 epochs，比較 best／last；重新啟用早停需 Orchestrator 說明理由並經確認。早停仍是平台選項。
- **來源**：`record/archive/2026-09-23-attention-family-eval-expansion.md`、`docs/agent-api/training/api.md`

## 資料與環境

### D-033 DB 權限：hermesnote 讀寫、quotes 唯讀
- **決定**：`hermesnote` DB 可讀寫（含 DDL），新資料表以用途前綴（`quant_`、`model_`），不加版本後綴；`quotes` DB 只能 SELECT；超出範圍需 Hermes 同意。
- **來源**：Hermes 工作規則（2026-09-03，`D:\hermesnote\CLAUDE.md` 規則 1）

### D-034 報價讀取依 `built_at` 去重
- **決定**：讀 OHLCV 時 `ORDER BY datetime, built_at`，同一 datetime 保留最新 `built_at`。
- **理由**：資料管線重跑會對同一根 K 棒寫入多筆，重複時間戳會讓前端圖表拒畫、訓練樣本重複。
- **來源**：`record/archive/2026-09-02.md`、`record/archive/2026-09-08-training-plan.md`

### D-035 版本鎖：Vite ^6.3.1、plotly 5.24.1
- **決定**：前端鎖 Vite ^6.3.1；後端鎖 plotly 5.24.1。
- **理由**：Vite 8 的 rolldown 原生綁定在開發機安裝失敗；vectorbt 1.1.0 與新版 plotly 屬性名稱不相容，import 即失敗。
- **來源**：`record/archive/2026-08-18.md`（Vite）、`record/archive/2026-09-02.md`（plotly）

### D-036 通用指標視覺化：指標登記表＋`metric_specs`＋`evaluations` 清單
- **決定**：指標定義集中登記（`training/registry/metrics.py`），模型以 `metric_specs` 引用並宣告逐輪序列與評估指標；評估結果改為 `evaluations` 清單，每筆標明模型節點、輸出頭、資料集（切分、Phase、任務資料範圍、樣本數）、評估時點與基準；前後台共用 `components/metrics/` 依資料描述選元件。新任務只保存 `evaluations`＋`metric_specs`；舊紀錄的 `evaluation.best／last` 保留原樣、API 讀取時轉換，不改寫資料庫。
- **理由**：原本前端依模型種類與指標名稱寫死（`DUAL_METRICS`、依架構猜 loss 種類、逐欄對應評估報告），新增指標或評估方式都要改頁面；`{best, last}` 無法容納其他資料集與評估時點，也沒有標明比較條件。
- **替代方案**：回填改寫舊紀錄（否決：正式研究結果要保留原樣）；新舊格式並存保存（否決：兩種格式長期並行，讀取端要分支）。
- **後續**：Phase 3 holdout 評估、中間 checkpoint 評估（格式已預留 `split: "holdout"`、`point.kind: "checkpoint"`）。
- **來源**：`record/archive/2026-09-30-generic-metric-visualization.md`

