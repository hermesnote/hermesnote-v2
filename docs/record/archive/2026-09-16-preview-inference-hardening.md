# 2026-09-13 ～ 2026-09-16：即時預覽／最終推論分批儲存 全面整改 + 上線

延續 2026-09-10 訓練系統 V2 三方對齊之後的下一輪，Codex（研究端）針對 `/model` 頁面與
訓練/推論後端提出一連串收斂需求，跨好幾輪逐步落地，最後完成正式環境部署與驗收。

## Batch 2：即時預覽重新設計

- 即時視窗預覽從「固定樣本看到底」改成「跟著目前正在處理的 batch 持續輪替抽樣」，
  節流至約 1.5 秒一次、不拖慢訓練，真實時間戳（隨機切分下日期跳動是正常現象）。
- 把預覽併入主要 K 線區域，移除舊的、跟核心功能無關的全期間背景圖，新增
  「訓練即時預覽／完成後歷史瀏覽」分頁切換，兩者共用同一個 lightweight-charts instance。
- 分類預測改顯示真實意義（漲/不漲、跌/平/漲，僅在 `labeling_rule=fixed_threshold` 下成立），
  不再是「類別 0」。
- 修掉 PostgreSQL NOTIFY payload 8000 bytes 上限問題（官方文件已確認）：`bars` 等大內容
  改存 `model_training_preview_samples` 表，NOTIFY 只帶 `{type, node_id, sample_id}`，
  WebSocket 端點收到通知才查表轉發完整內容。
- Loss/Accuracy 圖表可讀性：固定圖例（不用 lightweight-charts 原生會擋內容的 title 標籤，
  改用自製 HTML/CSS 圖例疊層）、train/val 實線虛線區分、Loss 4 位小數／Accuracy 百分比
  2 位小數、hover 同時顯示兩者數值、已完成輪數預設貼齊滿版寬度但保留使用者手動縮放。
- 修掉窄視窗（≤900px）左側設定欄不會撐滿寬度的真實 CSS bug（`flex-direction: column` 切換
  時 `align-items: flex-start` 沒有跟著改成 `stretch`）。
- 特徵清單加上「全部展開／全部收合」，個別指標仍可單獨展開。

## 收尾輪（Round F）

- `preview_store.py` 的 `save_preview_sample()` 原本是同步、無逾時的 DB 寫入，直接卡在訓練
  熱路徑裡；改成有界佇列（maxsize=50，滿了就丟棄不擋訓練）+ 背景執行緒異步寫入，
  DB 連線加 `connect_timeout`／`statement_timeout` 兩層逾時。本機用 monkeypatch 模擬
  DB 永久 hang 驗證：200 次呼叫仍在 0.0012 秒內全部回傳。
- 歷史瀏覽換模型節點時查詢不會重新觸發的真實 React bug（`useEffect` 同一輪 commit 讀到
  尚未反映的舊 `historySamples.length`）——改用 `historyLoadedKeyRef` 記錄「這個
  job+節點組合是否已觸發過載入」，不再靠 state 長度判斷；用兩個獨立世代編號
  （`loadGenRef` 管 job、`historyLoadGenRef` 管 job+節點）擋過期回應。
- 最終模型推論（`job_type='infer'`）後端儲存/查詢初版：新增 `model_inference_predictions`
  表，分批寫入、依時間區間＋游標分頁查。
- 產出 `docs/temp/2026-09-15-preview-inference-migration-checklist.md`，明列兩張新表的
  部署步驟；LSTM 真實跑通這輪只完成 code review，未實際執行（明確揭露不能用 XGBoost
  本機測試取代）。

## 後續修正：完整機率保留 + 真正串流 + 前台瀏覽接完

Codex 抓到三個問題並要求修正：

1. `inference_store.py` 把機率向量 argmax 成單一類別後只存類別，原始機率遺失——改成
   `output_type`（`class`／`probability`／`regression`）discriminator + `probabilities`
   JSONB 欄位完整保留向量，`predicted` 只是額外方便查詢用的 argmax。
2. `ModelSettings.tsx` 讀 `r.predictions?.length` 顯示推論筆數，但新版 API 已經改回傳
   `{count, start_ts, end_ts}` 摘要，會一直顯示 0——改讀 `r.count`，向下相容舊格式。
3. 「分批寫 DB」其實只解決寫入這一步，`inference.py` 仍先把整段期間的 predictions
   算成一個大 list——改用 `numpy.lib.stride_tricks.sliding_window_view` 開窗格化 view
   （沿用 `graph.py` 訓練時的既有技巧），新增 `run_inference_chunked()` 真正串流：
   算一批、寫一批（`inference_store.open_prediction_writer()`）、丟一批，尖峰記憶體只跟
   chunk size 有關。本機測試：200 列、window=10、chunk=7 → 190 筆預測分 28 批寫入，
   `predict_proba` 只呼叫 28 次而非逐列呼叫，證明真的有分批不是繞過。
4. 前端補上 `job_type='infer'` 的完整瀏覽 UI（`ModelTrainingPage.tsx`）：偵測 infer job
   的 `graph_spec` 形狀跟 train job 完全不同（`{target_job_id, target_node_id, start, end}`，
   沒有 `nodes`），設定欄改顯示來源 job/節點/期間/筆數摘要，主要區域改成全期間推論結果
   清單（游標分頁），沿用既有版面元件（`.model-primary-wrap`／`.model-history-list`），
   不再開新美化工作。過程中修掉一個真的 TDZ bug（`isInferJob` 在宣告之前被
   effect 依賴陣列引用）。

DB 故障測試全程用 monkeypatch 隔離進行，未曾讓正式 NAS DB 不可用；`psycopg2.connect`
失敗會確實往外拋例外（不像 preview 寫入那樣吞掉），對應 `worker.py` 的
`except Exception → mark_failed`，確保推論失敗不會被誤報成功。

## 部署前置修正

- `npm run build`（`tsc -b`）在 `ModelTrainingPage.tsx:536` 炸出 TS2339——
  `job_type !== 'infer'` 只窄化了 `job_type` 本身，`graph_spec: TrainGraphSpec |
  InferGraphSpec` 不是真正的 discriminated union，TS 不會連帶窄化。改成明確
  `as TrainGraphSpec` 轉型後建置通過。
- `model_init.sql` 補上防呆：`model_inference_predictions` 除了
  `CREATE TABLE IF NOT EXISTS`，額外加 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
  （`output_type`／`probabilities`），確保不管正式環境是全新套用還是曾套用過舊版，
  重跑同一支腳本都不會缺欄位。

## 正式環境部署與驗收

- 確認無 pending/running 訓練後，在正式 NAS `hermesnote` DB 執行 `model_init.sql`（透過
  backend 容器內建的 psycopg2，整份 SQL 包成單一 transaction，失敗即 rollback、不留半套）。
  驗證：兩張新表＋所有欄位齊全，既有三張表列數完全不變（job 4/progress 310/artifacts 4），
  `/preview_samples`、`/inference_predictions` 兩支端點皆回 200。全程未重啟容器、未送
  新訓練。
- 部署後 Codex 實測真實推論 job（`334438fb-...`）失敗：`string indices must be integers,
  not 'str'`。查證 `worker.py:_load_dataframes_for_infer()` 用 asyncpg 讀
  `model_artifacts.graph_spec_snapshot`（JSONB），這條連線（`services/hermesnote_db.py`
  的 `get_conn()`）沒有註冊 JSON type codec，asyncpg 預設不會自動解碼，回來的是原始
  JSON 字串，直接 `["nodes"]` 索引字串就炸掉。修法：補一行 `json.loads()`，跟同檔案
  `job_store.py`／`preview_store.py`／`inference_store.py` 讀 JSONB 欄位的既有慣例一致，
  不改動 `get_conn()` 全域行為。**修正後尚待 Hermes 重建 `hermesnote-training` 容器生效，
  重測沿用已保存模型（`3e4d15c1-.../mod1`），不需要重新訓練。**

## 個人履歷頁更新

`/hermes` 路由（`HermesResumePage.tsx`）自 2026-09-08 上線後就是純 iframe 包
`frontend/public/interview-resume.html`（Hermes 提供的自包含互動式「三區履歷時間軸」
靜態頁）。這次 Hermes 提供更新版（標題改成正式的「楊馥瑋｜AI 應用與系統整合專案履歷」，
內容/圖片更完整，2.45MB→4.76MB），直接覆蓋取代，未保留備份，route/元件都不用改。
