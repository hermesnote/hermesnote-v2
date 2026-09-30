# 2026-09-30 通用指標視覺化

> 狀態：**已部署**（2026-09-30，`92b4a24`＋`39d5114`）；後台畫面需登入確認。不需要 DB migration，不改寫正式資料。
> 決策：`decisions.md` D-036；規範：`spec.md` REQ-TR-09／REQ-TR-15／REQ-UI-05。

## 目標

模型／評估流程產生的資料，前台依資料描述用對應元件呈現：逐輪序列畫曲線、評估結果用數值比較與表格、矩陣用熱圖；評估組織可容納 best、last、未來的 checkpoint、不同資料集；標明模型、輸出頭、資料集與評估時點，適用的基準一起呈現；新指標透過資料描述接入。

## 實作

**後端**
- `training/registry/metrics.py`（新）：指標定義登記表（label、shape、direction、unit、format、description）、`build_specs()`、`GET /api/model/registry/metrics`。
- `architectures.register(..., metric_specs=)`：LSTM（單頭分類、單頭回歸、雙頭）與 XGBoost（二元、多元、回歸）宣告逐輪序列（對應訓練迴圈實際欄位與指標意義，例如同一個 `loss` 在 LSTM 分類是 cross_entropy、XGBoost 分類是 logloss）與評估指標、輸出頭、目標單位。
- `training/evaluation_records.py`（新）：`build_evaluations()` 把評估報告轉成 `evaluations` 清單（id、model_node、head、dataset、point、metrics、unavailable、baselines），只重組不重算，accuracy 由混淆矩陣推算；`summarize_node()` 供 worker 保存新任務；`enrich_job()` 供 API 讀取時補 `metric_specs`（進行中也有）並轉換舊紀錄。
- `dataset` 標明：任務資料範圍（graph_spec 的 start／end）、切分方式、驗證比例、樣本數、Phase，以及驗證集怎麼來的說明——random：固定種子 42 打亂、驗證樣本分散在整個期間；chronological：時間最後一段、邊界排除筆數。
- `training_meta` 新增 `n_samples`、`val_ratio`、`split_seed`（新任務）。
- worker：新任務只保存 `evaluations`＋`metric_specs`（不再另存 `evaluation`）；`fetch_next_pending` 帶出 `phase`。
- API：`GET /api/model/jobs`、`/jobs/{id}` 經 `enrich_job`；舊紀錄 `evaluation` 原樣保留並回傳。

**前端**（`frontend/src/components/metrics/`，前台 `/model` 與後台歷史紀錄共用）
- `MetricsDashboard`：依 `metric_specs.series` 每個序列一張 `MetricSeriesChart`（train／val、best 標記、同輸出頭同單位的基準參考線），逐輪明細表欄位也由描述產生。
- `EvaluationView`：輸出頭／資料集分頁、評估時點勾選；比較條件（資料集、任務資料範圍、切分方式、樣本數、驗證集說明、best 挑選依據、目標單位）；`MetricCompare`（best／last／基準並排、依 direction 標較佳者、相對條、無法計算附原因）、`PerClassTable`、`DistributionBars`、`MatrixHeatmap`（依列比例上色）。後台用精簡模式。
- 移除：`DUAL_METRICS`、依架構名稱猜 loss 種類、`FinalModelPanel`、`EvaluationPanel`、`pages/evaluationRows.ts`、固定的 Loss／Accuracy 兩張圖與其同步邏輯。

**文件**：`docs/agent-api/training/api.md`（讀取訓練結果改寫：`evaluations`、`metric_specs` 格式、HA 從 `evaluation.best／last` 改讀的對照表與範例）、`extending.md`（宣告指標、新增指標）、`spec.md`、`architecture.md`、`decisions.md`（D-036）。

## 驗證

- 後端 198 項測試通過（3 skipped）；contract harness ALL MATCH。新增 `tests/test_evaluation_records.py`（11 項）：7 種模式實際訓練後，宣告的逐輪欄位都真的有回報；轉換後數值與原報告逐一相同；紀錄身分（輸出頭、評估時點輪次、挑選依據、Phase、任務資料範圍）；random／chronological 的驗證集說明；舊欄位保留、轉換可重複；進行中任務也有 `metric_specs`；推論任務不受影響；新任務只存新格式；新舊共用同一個轉換函式。
- **正式資料（唯讀）**：3 筆已完成的雙頭 LSTM（Phase 1）讀取轉換後，138 個數值與原始 `evaluation` 逐一相同，原欄位未改動。
- **實際畫面**（Browser pane 暫時 harness，掛真正的 `ModelTrainingPage`／`ModelSettings`；資料用正式 3 筆＋合成新格式 3 類 LSTM、XGBoost Phase 2 回歸；不打本機 API；驗證後已移除）：雙頭的聯合損失／RMSE／MSE／方向準確率／BCE 五張曲線、best 標記、RMSE 基準參考線；評估的比較條件、best／last／基準比較、逐類別表、類別分布、兩張混淆矩陣熱圖；XGBoost 顯示 Boosting 輪次與時間切分說明；3 類的熱圖與「無法計算」原因；後台歷史紀錄精簡模式。瀏覽器 console 無錯誤。
- 前端 `tsc -b`、`npm run build` 通過；api.md 範例 payload 經後端驗證；HA 改讀範例在正式資料轉換結果上實跑通過。

## 限制

- 評估只有驗證集的 best／last；Phase 3 holdout 評估、中間 checkpoint、訓練集評估列為後續需求（格式已預留）。
- 舊紀錄沒有 `n_samples`／`split_seed`，由 `n_train＋n_val＋邊界排除` 推算總數、種子依程式固定值 42 說明。
- 基準參考線只畫在單位一致的曲線上（損失空間的曲線不畫原單位基準）。
- 同一任務多個模型節點時，頁面沿用既有的節點選單逐一顯示。

## 部署步驟（待確認，未執行）

1. 不需要 DB migration，也不清理資料；部署前確認沒有 pending／running 任務（worker 重建會中斷執行中任務）。
2. 本機 `npm run build`（frontend）→ `deploy.ps1`：同步 `frontend/dist`、`backend`、`docs/agent-api`，重建 backend、training，重啟 frontend。
3. HA 文件：`docs/agent-api/training/api.md`、`extending.md`、`index.md` 隨部署同步到 NAS `/mnt/Hermesnote/web/hermes/docs/agent-api`；HA 依 api.md「從 `evaluation.best／last` 改讀 `evaluations`」更新解析腳本（舊紀錄的 `evaluation` 仍會回傳，但新任務不再有）。
4. 部署後確認：`GET /api/model/registry/metrics` 200；正式 3 筆紀錄的 `/api/model/jobs/{id}` 回傳 `evaluations`（每筆 4 筆紀錄）與 `metric_specs`；`/model?job=` 與後台歷史紀錄畫面正常。

## 提交

Hermes 核准本輪與相關文件一起提交：後端（`registry/metrics.py`、`evaluation_records.py`、architectures／lstm／xgboost_model 的 `metric_specs`、worker、job_store、graph 的 `training_meta`、router）、前端（`components/metrics/`、`ModelTrainingPage.tsx`、`ModelSettings.tsx`，刪除 `pages/evaluationRows.ts`）、測試、`docs/agent-api/training/`、`docs/spec.md`／`architecture.md`／`decisions.md`（2026-09-29 整理的三份，含本輪 D-036 等更新）、`docs/index.md`（只提交三份文件的索引列與閱讀順序；他人新增的 agent-workflow／HA 手冊列未納入）、`docs/record/`。部署待目前 5 分 K 訓練完成、確認無 pending／running 任務後再安排，HA 解析更新同步銜接。

## GC 檢閱修正（追加提交）

GC 指出：前端非數值元件依固定 key（`per_class`／`class_distribution`／`confusion_matrix`）分派，評估轉換也用固定指標清單、固定基準表、固定雙頭 key。修正為由指標登記的 shape 與資料契約分派：

- 後端 `registry/metrics.py`：`register()` 加 `axes`（矩陣軸名稱）與 `derive`（衍生指標，`DERIVATIONS` 方法表；accuracy 改為登記 `derive={"from": "confusion_matrix", "method": "matrix_diagonal_ratio"}`）；新增 `BASELINE_REGISTRY`／`register_baseline()`；未知 shape 直接拒絕。
- `evaluation_records.py`：報告裡凡是已登記的指標都依 shape 轉成資料契約（matrix＝`<key>`＋`<key>_labels`＋定義的 `axes`）；衍生指標依登記推算；`baseline_*` 依基準登記命名，基準的 `metrics` 只取已登記指標、其他欄位放 `detail`；輸出頭依 `metric_specs.heads` 拆分（不再寫死 `direction`／`regression`）；`build_evaluations` 改收 `specs`。
- 前端：`PerClassTable`／`DistributionBars`／`MatrixHeatmap` 改以 `metricKey` 參數讀資料，per_class 欄位由資料決定；`EvaluationView` 依 `definitions[key].shape` 分派（沒有定義時由資料推斷 shape）。
- 驗證：新增 3 項測試——`evaluation.py` 產生的鍵都已登記；以新名稱指標（`regime_transition_matrix` 矩陣＋自訂軸、`regime_stay_ratio` 衍生、`per_regime`、`regime_counts`、`hit_rate`、新基準 `baseline_coin_flip`）只登記不改程式即進入評估清單、未登記鍵不納入；未知 shape 拒絕。後端 201 項通過、harness ALL MATCH、正式 3 筆 138 個數值仍逐一相同。畫面：同一批新名稱指標在評估區分別以熱圖（自訂軸名）、逐類別表（欄位 hit／n）、分布、數值比較（含新基準）呈現；正式雙頭紀錄呈現與修正前相同。
- 文件：api.md（資料契約、依 shape 解析）、extending.md（shape 契約表、衍生指標、新基準）。

## 部署紀錄（2026-09-30，Hermes 核准）

1. 部署前唯讀確認：無 pending／running 任務；既有結果 日線 `193c8921`、15m `f6a3095c`、1m `5028249c`、5m `d2af34ef`（皆 Phase 1 雙頭 LSTM，done），另 2 筆 failed；資料全部保留。
2. `deploy.ps1`（本機 HEAD `39d5114`）：同步 frontend／backend／`docs/agent-api`，重建 backend、training，重啟 frontend。
3. 正式 API（backend 容器內）：`/registry/metrics` 200（18 個指標、4 種 shape）、`/registry/components/optimizer` 200；4 筆結果皆有 `metric_specs`（5 條序列）與 `evaluations`（每筆 4 筆紀錄），每筆 24 個數值與原 `evaluation` 相同，原欄位保留；backend log 無錯誤，worker 已啟動。
4. 正式前台 `https://hermesnote.com/model?job=`：4 筆皆顯示 5 張曲線、比較條件（任務資料範圍 2011-01-03～2025-01-01、隨機切分 20%、樣本數與驗證集說明、best 挑選依據）、回歸頭 4 項／方向頭 5 項比較、逐類別表、2 張混淆矩陣熱圖；console 無錯誤。（直接連 `:8082` 沒有 `/api` 轉發，屬部署架構，須經 hermesnote.com。）
5. HA 手冊：NAS `/mnt/Hermesnote/web/hermes/docs/agent-api/training/api.md`、`extending.md` 與本機雜湊一致。
6. 未做：後台 `/admin/model` 需 Google 登入，由 Hermes 確認歷史紀錄的「展開完整評估」。

