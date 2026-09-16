# 即時視窗預覽／最終推論分批儲存——資料庫遷移與部署檢查清單

> 2026-09-16 更新：`model_inference_predictions` 加了 `output_type`／`probabilities`
> 兩個欄位（保留完整機率向量，不再 argmax 之後就丟掉），`model_init.sql` 已同步補上
> `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`，不管正式環境是「全新套用」還是「已經套用過
> 舊版」，跑同一支腳本結果都一樣、不會缺欄位。這次以可部署為優先，其餘功能/美化未再追加。

這輪（即時視窗預覽 + 完成後歷史瀏覽 + 最終推論分批儲存）新增了兩張表，`model_init.sql`
本身沒有自動套用到正式 DB 的機制，需要人工執行一次。以下步驟可重複執行（`CREATE TABLE
IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` / `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
三種寫法都是重複跑不出錯、也不會動到既有資料的寫法），但**目前一次都還沒有在正式環境
跑過**，需要 Hermes 執行並確認。

## 新增的兩張表

- `model_training_preview_samples`：訓練途中「正在處理哪個 window」的抽樣紀錄（含真實
  OHLC bars），給即時預覽跟完成後歷史瀏覽用。
- `model_inference_predictions`：最終模型推論（`job_type='infer'`）的逐列結果，分批寫入，
  跟上面那張是不同性質的資料。欄位裡 `output_type` 決定怎麼解讀 `predicted`／
  `probabilities`：
  - `'class'`／`'regression'` — `predicted` 是唯一的數值，`probabilities` 是 `NULL`。
  - `'probability'` — `predicted` 是 argmax 類別索引（方便排序/篩選），`probabilities`
    是完整機率向量（JSONB array）——原始機率不能只留 argmax 就丟掉，之後機率門檻策略
    要用完整分布。

兩張表都用 `ON DELETE CASCADE` 掛在 `model_training_jobs(id)` 上，刪 job 會自動連帶清掉，
不會變成孤兒資料。

## 部署步驟

1. **套用資料庫 schema**（可重複執行、對既有資料無破壞性）：
   ```bash
   psql "$HERMESNOTE_DATABASE_URL" -f backend/model_init.sql
   ```
   這一步可以在**現有訓練還在跑的時候**執行——`CREATE TABLE IF NOT EXISTS` 不會鎖到
   `model_training_jobs`/`model_training_progress` 這些既有表，也不會影響正在寫入的資料。

2. **確認新表／新欄位確實存在**（不管是全新建立還是舊表補欄位，跑完都要看到這些）：
   ```sql
   \dt model_training_preview_samples
   \dt model_inference_predictions
   \d model_inference_predictions
   ```
   `\d model_inference_predictions` 的欄位清單裡要能看到 `output_type`（TEXT NOT NULL）
   跟 `probabilities`（JSONB，可為 NULL）——這兩個是這次新加的，如果環境是套用過
   2026-09-15 那版 `model_init.sql`（沒有這兩欄）又再套用這次的版本，這一步就是驗證
   `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 真的補上了，不是缺欄位又沒被發現。

3. **部署新的 backend / training 映像檔**——這一步才需要重建/重啟容器，時機由 Hermes 決定
   （見下方「跟正在跑的訓練的關係」）。

4. **部署後的健康檢查**（不需要真的送訓練）：
   - `GET /api/model/jobs?limit=1` 正常回應（確認 backend 容器正常啟動，import 新模組沒有炸掉）。
   - 對任一個 job 打 `GET /api/model/jobs/{job_id}/preview_samples?node_id=xxx&limit=1`，
     應該回傳 `[]`（空陣列，不是 500）——確認新端點能連得到新表。
   - 對任一個 job 打 `GET /api/model/jobs/{job_id}/inference_predictions?node_id=xxx&limit=1`，
     同樣應該回傳 `[]`（不是 500）——這支如果欄位沒補上會直接因為 SQL 查詢欄位不存在而
     500，是最快能抓到「缺欄位」的健康檢查點。
   - 前端 `/model` 頁面打開任一筆 `job_type='infer'` 的舊任務（如果有的話），確認設定欄
     跟「全期間推論結果」清單不會因為缺資料而整頁白畫面／console 報錯——沒有資料時應該
     顯示「此任務尚無推論結果」，不是 500 或空白。

## 跟「目前正在跑的訓練」的關係

- **步驟 1（套用 schema）不需要重啟任何容器**，訓練繼續跑不受影響。
- **步驟 3（部署新程式碼）需要重建 `hermesnote-training` 容器**，這會中斷正在執行的訓練
  ——這件事的時機必須由 Hermes 決定，我不會自己動手重建/重啟。
- 新程式碼部署之後，**新啟動**的訓練才會開始寫入這兩張表；部署前就已經在跑、或部署前
  已經完成的舊任務，不會回溯補上這些資料——這是下一節「舊任務沒有預覽資料」要處理的情況。

## 舊任務沒有預覽資料時的畫面行為

已經在前端修正：
- 「訓練即時預覽」分頁，如果任務已經是 done/failed 狀態、且這個 session 沒收到過任何一筆
  即時預覽，畫面會顯示「訓練已結束，若曾記錄過抽樣視窗，請切換『完成後歷史瀏覽』查看」，
  不會一直顯示「等待訓練回報」這種暗示還在等待的文字（因為已結束的任務根本不會再建立
  WebSocket 連線，不可能等到）。
- 「完成後歷史瀏覽」分頁，查詢成功但回傳是空陣列時，畫面明確顯示「此任務未記錄預覽」；
  查詢本身失敗（網路/伺服器錯誤）則顯示「載入失敗」＋重試按鈕，兩種情況不會混在一起。
- 這兩種文案都只在**這個功能上線之前建立的舊任務**、或**推論任務**（`job_type='infer'`
  不會有 preview_samples，這是預期行為，不是缺陷）身上看到。

## 尚未做、需要 Hermes 排程的部分

- **實際執行遷移**：以上 SQL 目前只存在檔案裡，還沒有對正式 DB 跑過。
- **部署時機**：新程式碼需要重建 `hermesnote-training`／`hermesnote-backend`，會中斷正在跑
  的訓練，這個時機要 Hermes 決定。
- **部署後的真正驗收**：見同一輪回報裡「最後驗收要包含真正的 LSTM」那一節，需要在部署
  完成之後，用一筆真的 LSTM 短訓練實測——這件事我沒辦法在部署之前先做完。
