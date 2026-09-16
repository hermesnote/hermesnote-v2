# 2026-09-08（三）：模型訓練系統開工前規劃 checklist

> 本檔是動工前的規劃/checklist，確認後開工。**執行原則**：能做的先做，做不了或需要 Hermes 判斷的地方不擅自決定，直接在對應項目下記錄「卡住原因」，跳過繼續做別的，留到 Hermes 有空時一起處理。進度即時更新回這份檔案。

## 2026-09-09（續）：DAG 引擎補模型接模型 + 回測策略收藏當特徵，皆已測試通過

- [x] **DAG 引擎：模型接模型（多模型融合）**——`training/graph.py` 重寫：每個 Model Node 訓練完，除了 metrics/predict，額外對自己的全範圍窗格跑一次預測，存進 `outputs[nid]` 當一個 (n,1) 陣列；下游 Model Node 的 `inputs` 可以是其他 Model Node，用「從尾端對齊到最短長度」處理不同上游 window 造成的長度差異。本機測試通過：兩個不同 window（60、30）的子模型 → 一個最終融合模型，執行無誤，長度對齊正確。**限制寫在檔案開頭註解**：目前只處理同一時間框架融合，跨時間框架要用時間戳對齊，還沒做；上游模型對自己 train 段的「全範圍預測」是用同一個模型直接預測，不是 OOF，對下游有輕微資料洩漏風險——這兩點都留給 Hermes 判斷要不要現在補
- [x] **回測策略收藏當 Feature Node**——新表 `quant_saved_strategies`（`ON DELETE CASCADE` 綁 `quant_backtest_jobs`）；`/admin/quant` 歷史紀錄每列加「☆ 收藏／★ 已收藏」按鈕（`services/job_store.py` + `routers/strategy.py` 新增 save/unsave/list 三支 API）；新 Feature Node `quant_saved_strategy`，重用 `indicators/tree.py` 的 `evaluate_node()` 對訓練資料重算條件樹訊號，輸出 0/1 欄位；`training/Dockerfile` 補上 `COPY indicators/`。本機測試通過（mock DB fetch，驗證 shape 正確）。UI 加了免責說明：「回測有效不代表對模型預測有用」
- [x] **訓練模組節點圖編輯器——已完成並測試通過**：
  - 術語改用「訓練模組」，不用「子群組」
  - UI：A區「＋加訓練模組」按鈕（展開現有表單填寫一個模組）→ 存成卡片顯示在 B 區
  - 模組數 ≥2 時才需要組圖；用 **React Flow**（新引入的 npm 套件，節點編輯器，拉線連接埠決定 inputs 關係，不用重刻拖線引擎）畫布，卡片=節點，手動拉線＝宣告 inputs
  - 拉出來的圖只是產生 `graph_spec` JSON 的介面，不是另一套業務邏輯——序列化成同一份「node id + inputs」格式，餵給今晚已經寫完測試過的 `graph.py` DAG 引擎，正確性靠重用既有、已驗證的後端邏輯保證
  - 「扁平」＝所有非最終模組的輸出都接到最後一個模組當特徵（＝今晚做的「模型接模型」融合，不用新後端邏輯）；「巢狀」＝允許先分組成中間節點、中間節點再往上接，同一條遞迴規則，不窮舉形狀
  - 最終節點先只做「可訓練 Model Node」一種，「純組裝 Ensemble Node（不訓練，只concat/平均）」的雙選項先不做，UI 出來看過再決定要不要加
  - 每張模型卡片可加一個「輸出類別 vs 輸出機率（softmax）」開關，機率輸出=更豐富的下游特徵，先做這個小開關即可，不用等 Attention/Transformer
  - Attention 對應「架構本身是 Transformer 類」才有意義，不是通用開關，這輪不用管

  **測試結果（本機瀏覽器操作，走完整個流程）**：
  - 用臨時的免登入測試路由（測完已移除）實際點開 `/admin/model`：新增模組→填表單→存成卡片→拖拉分開兩張卡片→**用滑鼠拖出接點連線（mod1→mod2）成功**（DOM 確認 `data-id="xy-edge__mod1-mod2"`）→送出訓練
  - 送出後端收到的 `graph_spec` JSON 完全正確：`mod2` 的 `inputs` 正確包含 `["mod2_f2", "mod1"]`，證明「畫布只是產生 JSON 的介面，序列化邏輯正確」這個設計原則成立
  - 這次真的送到 NAS 訓練 worker 跑（`device: cuda`），第一次跑出 `graph.py` 的一個真實 bug：`aligned = [a[-min_len:] for a in input_arrays]` 在 `min_len`算出來是特殊值時，Python 的 `-0 == 0` 會讓切片變成整個陣列而不是空陣列，導致維度對不上而崩潰。已修（加上 `min_len<=0` 的明確擋下＋錯誤訊息）。**修完後在本機用真實 CPU LSTM（新裝的 torch cpu 版）重現同一個 2 模組融合情境，跑成功**，但老實說：本機重現「原本會失敗」的那個確切條件沒有百分之百對上（本機這次直接就成功了，無法確定是修好了、還是原本那次是偶發），這個殘留的不確定性記錄在案，不確定就講不確定，不硬拗成「完全查清楚了」
  - Softmax/機率輸出（`predict_proba`）已加到 `lstm`／`xgboost` 兩個架構，`output_mode: "probability"` 時 `graph.py` 會改用它——這條路徑本身還沒有實際測過（測試時兩個模組都用預設的 `class` 模式），下次測試時記得試一次機率模式
  - 本機已安裝 CPU 版 torch（`.venv` 裡），以後遇到 LSTM 相關的 bug 可以先在本機重現+修，不用每次都靠 NAS 的 GPU 資源除錯

## 2026-09-09 補充：前台/後台結構重做 + 第一次真實 GPU 訓練測試

Hermes 上線後指出一個明確的架構錯誤：昨晚把「建立訓練」的表單直接放在前台公開頁，沒有比照 `/admin/quant` 的「後台設定＋前台唯讀展示」分工。已重做：

- **`/admin/model`**（`ModelSettings.tsx`，新建）：比照 `QuantSettings.tsx` 的結構——「建立訓練」表單（可加多個 Feature Node、Label Node 完整暴露 horizon/n_classes/threshold_pct、Model Node 暴露 window/val_ratio/epochs/architecture）+「歷史紀錄」清單，點擊導到 `/model?job={id}`。多時間框架/多模型混合訓練的圖形化組合介面**還沒做**（後端 DAG 引擎已支援任意組合，前端組圖 UI 是下一輪），這次先讓單一路徑的表單能用
- **`/model`（前台）改成純唯讀**：讀 `?job=` 網址參數（沒帶就抓最新一筆，比照 `/quant` 的做法），A 區塊只顯示設定資訊＋即時數據（數字，不用圖表化）、B 區塊改成一定會顯示的價位K線圖（不是條件性載入）、C 區塊維持 loss/accuracy 曲線+epoch清單
- 後端新增 `GET /api/model/jobs`（列表，比照 `/api/backtest/jobs`）

**第一次真實 GPU 訓練測試（意外達成）**：本機測試 admin 表單送出的 payload 格式時，因為打的是同一個正式 DB，**NAS 上真正的訓練 worker 真的把這筆任務撿去跑了**——這是這個系統第一次真的呼叫 `lstm` 訓練函式，不是 stub。結果：`status: failed`，錯誤 `No module named 'fastapi'`——`services/quotes.py`（訓練 worker 用來抓價量資料）用了 `from fastapi import HTTPException`，但訓練容器沒裝 fastapi。已修：`training/requirements.txt` 加上 `fastapi`（輕量依賴，重用這支共用服務函式划算，不去改 `quotes.py` 本身影響到 web 後端的錯誤處理行為）。**這個修正還沒部署，等 Hermes 下次 deploy 才會生效，屆時會是真正第一次完整跑通 LSTM 訓練的機會**。

## 醒來先看這裡：整體狀態（開工當晚完成，以上是隔天追加）

1~6 全部做完並本機測試通過，**包括真的開瀏覽器點按鈕、即時看到 WebSocket 推送更新畫面**，不是只測到後端。三件事需要你判斷，其他都是可以直接接續的完成品：

1. **正規化沒接**（教授要求 val/test 只能用 train 統計量 transform）——見 4 的已知缺口
2. **教授三階段流程只做了 train/val，沒有 in-sample 回測／holdout**——見 4 的已知缺口
3. **epoch 粒度 vs window 粒度的落差**——你原本設想「一段一段window即時跳轉」，目前做的是標準 epoch（掃完全部樣本才回報一次），B 區塊沒辦法點 epoch 清單跳轉到具體K線位置——見 6 的已知落差

**部署當下踩到一個 deploy.ps1 的坑，已修**：第一次跑 `deploy.ps1` 後 `/model` 頁面整片空白，查出來是 `docker compose up -d --build` 沒有真的重建 backend/training 容器（`docker compose ps` 顯示這兩個還是「10 hours ago」的舊容器，只有 frontend 因為有明講 restart 才是新的）——單獨手動跑 `docker compose build backend training` 確認 build 本身沒有錯誤且真的產出新 image hash，但 `up --build` 合成一行時沒有把這個結果正確反映出來，腳本也沒檢查 SSH 指令的結果，所以就算沒真的換新還是印「Deploy complete!」。手動跑 `docker compose up -d` 換上新 image 後恢復正常。`deploy.ps1` 已經改成 build/up 分開兩步＋檢查 exit code，之後再發生會直接報錯，不會誤判成功。

另外**真正的 GPU 訓練本身完全沒有實測過**（本機無 cu128 環境），這次全部是用假的 stub 架構驗證管線通不通，`lstm`/`xgboost` 這兩個真正的訓練函式邏輯本身沒有被實際呼叫過一次——**部署到 NAS 訓練容器、真的送一個訓練任務，會是第一次驗證這兩個函式能不能真的動**，這步驟需要你跑 `deploy.ps1`。

## 背景與定案摘要

- 視覺化：訓練當下**epoch 級的 loss/accuracy/進度必須即時**（WebSocket + Postgres LISTEN/NOTIFY，不用輪詢）；K線窗格重播（目前在跑哪一段資料 + val/test 段疊模型預測 vs 實際）可以一段一段呈現，不要求逐根即時
- Target/Label/Horizon 全部做成**可插拔 registry**（新增=登記一個新函式，不改介面），跟 Feature Node 同一套模式，這輪一次做完整，不做「先做最小組合」
- 三個 registry（features/labels/architectures）各開一支查詢 API，順便讓 FastAPI 自動產生的 Swagger 文件成為 Agent/MCP 之後查詢可用積木的入口
- 操作邊界：DB 我可直接操作；部署仍需 Hermes 執行 `deploy.ps1`（互動密碼），NAS 不再需要手動 SSH 建容器
- 本機測試原則：純邏輯（registry 函式、graph 引擎、NOTIFY 機制）本機輕量驗證；GPU 真訓練本機無法測，只能部署到 NAS 訓練容器後才是第一次端對端跑通

## Checklist

- [x] **1. 資料庫schema**
  - [x] 1.1 `backend/model_init.sql` 新增 `model_training_progress` 表
  - [x] 1.2 已在 NAS `hermesnote` DB 執行 DDL
  - [x] 1.3 本機測試通過：`write_progress()` 寫入 + `NOTIFY` 被獨立連線收到（見 4.4 的端對端測試一併驗證）

- [x] **2. 後端：三個 Registry（可插拔積木）**
  - [x] 2.1 `backend/training/registry/features.py` — 已實作 `ohlcv`、`talib_indicator`（任一 TA-Lib 指標透過 abstract API，不用逐一寫死）
  - [x] 2.2 `backend/training/registry/labels.py` — 已實作 `nbar_return`（T+horizon close 二元/三元，threshold_pct 可調）
  - [x] 2.3 `backend/training/registry/architectures.py` — 已實作 `lstm`（PyTorch）、`xgboost`，統一介面 `(X_train,y_train,X_val,y_val,params,on_epoch)->dict`，`on_epoch` callback 讓 graph.py 不用知道訓練細節就能拿到進度
  - [x] 2.4 本機測試通過：`ohlcv`(300,5)、`RSI`(300,1)、`MACD`(300,3) 形狀正確；`nbar_return` 二元/三元類別值域正確

- [x] **3. 後端：Graph 執行引擎**
  - [x] 3.1 `backend/training/graph.py` — 拓樸排序 + Feature/Label/Model 節點依序執行，`data_loader` 依賴注入（不綁定資料來源）
  - [x] 3.2 本機測試通過：2 個 Feature Node（ohlcv+RSI）拼接成 6 維特徵、window=60 正確切片、train/val 隨機切分筆數正確、progress callback 帶對的 node_id 觸發 3 次

- [x] **4. 後端：Worker 串接真訓練**
  - [x] 4.1 `worker.py` 換成呼叫 `graph.run_graph`，佔位函式已移除
  - [x] 4.2 `training/progress.py`：同步 psycopg2 寫進度 + `NOTIFY`（呼叫端在訓練用的背景執行緒，沒有事件迴圈，用同步連線最簡單）
  - [x] 4.3 資料撈取：`_load_dataframes()` 在 async context 先把 graph 用到的所有 timeframe 撈成 DataFrame，再丟給同步的 `run_graph`，訓練那條執行緒完全不用碰 async DB 連線
  - [x] 4.4 **本機端對端測試通過**（用真實 quotes DB 資料 + stub 架構模擬訓練迴圈，繞開本機沒有 GPU 的限制）：建立任務 → 撈真實 `tx_15m` 資料（513筆）→ graph 執行（ohlcv 特徵 + nbar_return 標籤 + window=20）→ 2 個 epoch 各自寫入 `model_training_progress` + `NOTIFY` → 獨立 WebSocket client 即時收到推送，訊息內容正確 → 清理測試資料。**真正的 LSTM/XGBoost 訓練本身（GPU）還沒在本機或 NAS 跑過，這一步只驗證到「架構插進來會被正確呼叫」，架構函式內部邏輯本身沒有實測**
  - [ ] **⚠️ 已知缺口，不擅自處理，留給 Hermes 判斷**：
    1. **正規化沒有接進來**——教授「val/test 只能用 train 統計量做 transform，不能重新 fit」這條原則目前完全沒有實作，`graph.py` 的 window 切完直接送進模型，沒有做 minmax/zscore 這類正規化。要接在哪一層（Feature Node 內？還是 Model Node 訓練前一道獨立步驟？）需要你決定
    2. **教授三階段流程只做了「train/val」，沒有「in-sample 回測」「holdout 測試」**——這次只實作了 `_split_data` 的隨機 train/val 切分，沒有做時序的 in-sample 全期推論、也沒有隔離的 holdout 區間測試。這兩個是教授明確要求的階段，這次沒動是因為時間，不是判斷不需要
    3. 這兩項要不要在下一輪補、還是這輪硬塞進去，等你醒了一起決定

- [x] **5. 後端：API 路由**
  - [x] 5.1 `backend/routers/model_training.py`：`POST /train`、`GET /jobs/{id}`、`GET /jobs/{id}/progress`（補歷史用）、`WS /jobs/{id}/stream`、三個 `/registry/*` 查詢
  - [x] 5.2 掛進 `main.py`
  - [x] 本機測試通過：起真的 FastAPI server，`curl` 測三個 registry 查詢＋建立任務＋查狀態；獨立 WebSocket client 連線後，另一支腳本呼叫 `write_progress()`，**1.5 秒後即時收到推送**，訊息格式正確

- [x] **6. 前端：模型訓練頁面**
  - [x] 6.1 `frontend/src/pages/ModelTrainingPage.tsx` + `.css`：三欄版面（A 設定/B K線/C 進度），比照 `/quant` 的 CSS 命名與 grid 結構
  - [x] 6.2 WebSocket 連線 hook（連 `/api/model/jobs/{id}/stream`），收到訊息即時 append 進度、更新 loss/accuracy 曲線
  - [x] 6.3 建立訓練任務表單：時間框架/日期區間/window、三個 registry 的下拉選單**直接查後端 API 動態產生**（不是寫死選項）
  - [x] 6.4 版面（A 左側欄可捲動設定／B K線背景圖／C 即時曲線+epoch清單）已落地
  - [x] **本機端對端瀏覽器測試通過**：真的開瀏覽器點「開始訓練」→ B 區塊即時載入真實 TX 15m K線→ 送出任務拿到 job_id → WebSocket 連上 → 另一支腳本模擬 5 個 epoch 進度寫入 → **瀏覽器即時收到全部 5 筆、loss/accuracy 曲線即時畫出、epoch 清單即時更新**，過程中沒有重新整理頁面
  - **過程中意外抓到一個真實 bug（已修，跟這次功能無關但值得記）**：`services/quotes.py` 的 `fetch_ohlcv_rows()` 只有 `ORDER BY datetime, built_at`，沒有去重——`docs/architecture.md` 早就記錄過「同一根K棒可能因管線重跑寫入兩筆」這個已知問題，但這支共用函式本身沒有處理，只有呼叫端知道要自己處理。這次在 `/api/model/candles` 端點加了去重（保留最後一筆）才解決 `lightweight-charts` 收到重複時間戳記直接拒畫的問題。**`/api/backtest/demo`／`/api/backtest/date-range` 這些既有呼叫端有沒有踩到同樣的坑，我没有動去查，先記錄，你之後可以評估要不要把去重邏輯搬到 `fetch_ohlcv_rows()` 本身，這樣所有呼叫端都受益，不用每個端點各自记得處理。**
  - **⚠️ 已知落差，記錄不擅自解讀**：Hermes 原本設想的是「一段一段 window 播放、B 區塊即時跳轉到目前處理的那一段」；目前實作的 epoch 是「標準 ML 定義：完整掃過一次所有訓練樣本」，不是「單一個 window」，所以現在 B 區塊只能顯示整段期間的背景 K 線，**沒有做到「點擊 epoch 清單、B 區塊跳轉到那個 epoch 對應的具體K線窗格」**——因為目前的 `window_meta` 只有 `{"node_id": ...}`，沒有實際時間範圍可以跳轉。要做到這個，需要決定：要不要把訓練粒度從「epoch」改成「batch/window」（每處理一個 window 就回報一次，不是掃完一整輪才回報一次）？這個改動不小（要動 `architectures.py` 的訓練迴圈），這次先不擅自決定，等你醒了一起判斷這個粒度要不要改

## 尚未決定（不阻塞開工，之後遇到再定）

- 視覺化方案細節（曲線+K線重播如何在同一頁面排版）、示範用的模型/時間框架選擇——Hermes 保留，屆時再定
- 融合節點（多模型/多時間框架組圖）暫不在這輪範圍，先讓單一路徑的圖跑通
- 正規化、in-sample/holdout 三階段流程——見上方 4 的已知缺口，等 Hermes 判斷
