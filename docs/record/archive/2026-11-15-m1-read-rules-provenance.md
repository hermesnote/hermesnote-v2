# 2026-11-15　M1：讀取修正、流程版本、資料紀錄（已遷移、已部署）

依三方對齊的交付清單第五版（加上 Hermes 2026-11-15 補回的兩項：跨時間框架「沿用前一時段／過舊上限」可設定、後台 UI，排在 M4／M7）。

## 範圍與實作

| 項目 | 內容 | 檔案 |
|---|---|---|
| 讀取修正 | 排除 `session='other'`（不寫成 `IN ('day','night')`：`daily_full` 的 session 是 `full`）；依 `bar_start_ts` 排序；`bar_start_ts` 與 `datetime` 都驗證唯一，重複就報錯並列明細，graph 不再 `keep="last"` 靜默去重 | `services/quotes.py`、`training/graph.py` |
| 讀取報告 | 每個時間框架：列數、排除筆數與日期、session／rule_version 分布、時段內缺格數（只統計不補棒）、範圍；統計在 SQL 端算，不把字串欄位撈進記憶體 | `services/quotes.py`（`DataFrame.attrs["read_report"]`）、`training/provenance.py` |
| 流程版本、環境指紋 | `PIPELINE_VERSION = "2026-11-r1"`；指紋含 python／numpy／pandas／torch／CUDA／GPU／xgboost／sklearn／TA-Lib py 與 C 版本 | `training/provenance.py` |
| 研究期間提示 | 迄日超過 2026-01-01 只提示、不擋；寫進任務紀錄並回給呼叫端（train、Phase 2、infer） | `training/provenance.py`、`routers/model_training.py` |
| 任務紀錄 | `model_training_jobs` 新增 `pipeline_version`、`warnings`、`env_fingerprint`、`data_report`；worker 讀完資料、訓練前寫讀取報告 | `training/job_store.py`、`training/worker.py` |
| 跨時間框架暫時拒絕 | 同一 Model Node（含上游鏈與 Label）用到多個 timeframe 時驗證失敗；M4 完成後把 `CROSS_TIMEFRAME_ENABLED` 改成 True | `training/graph_refs.py` |
| Dockerfile | 移除另外編譯的 TA-Lib 0.4.0（NAS 容器實際載入的是 wheel 內附 0.7.1）；已核對兩個映像全部依賴都有 Linux cp312 wheel，不需編譯器 | `Dockerfile`、`training/Dockerfile` |
| 前端 | 訓練頁「資料紀錄」區塊（流程版本、提示、各時間框架讀取報告、可用時間假設、執行環境）；後台送出後顯示提示、歷史列顯示流程版本與提示 | `ModelTrainingPage.tsx`、`admin/sections/ModelSettings.tsx` |
| 遷移 | 只新增 4 個 nullable 欄位；**先遷移、後部署** | `migrations/2026-11-15_m1_pipeline_provenance.sql`、`model_init.sql` |

調整：實驗與樣本框架的資料表原列在 M1，改到 M2 和樣本框架一起設計與遷移（避免先定 schema 再改）。

## 驗收

- 單元測試 231 項通過（新增 `test_quotes_read_rules.py`、`test_m1_provenance.py`；既有測試只改測試替身的介面）。
- `contract_harness` 對改動前基準 ALL MATCH。
- 正式 quotes DB 唯讀測試（`RUN_DB_TESTS=1`）通過：研究期間 1m～60m 各排除 5 筆 other、日期正確、無重複；日線排除 0 筆；那 5 天 30m 13:30、60m 13:00 的日盤 K 棒保留且量與資料庫一致。
- 前端 `tsc -b`、`vite build` 通過。畫面未在瀏覽器實際操作（需要本機後端與後台登入）。

## 讀取修正對特徵、答案、樣本的影響（實測，過往任務設定：11 個 TA-Lib 指標＋OHLCV、window 60、2011-01-03～2025-01-01）

舊讀取＝舊 SQL（`ORDER BY datetime, built_at`）＋舊 graph 去重（同 datetime 保留最後一筆）。

| 時間框架 | 原始列變化 | 特徵值不同的列 | 答案不同 | 窗格受影響的樣本 |
|---|---|---|---|---|
| 1m | 少 5 根（13:45） | 2,125,177／2,544,388 | 5 | 2,125,415 |
| 5m | 少 5 根 | 434,575／517,663 | 5 | 434,846 |
| 15m | 少 5 根 | 145,633／172,694 | 5 | 145,897 |
| 30m | 少 1 根；2016-01-30 13:30 換回真正的日盤 K 棒 | 78,917／89,740 | 0 | 78,917 |
| 60m | 少 3 根；2016-01-30 13:00 換回真正的日盤 K 棒 | 44,985／46,694 | 1 | 44,985 |

- 差異幾乎全部來自 **OBV**（累積型）：少一根就永久偏移一個常數（1m 最大 3 口；60m 最大 22,421），一路延續到期末。
- 其餘 10 個指標的差異在事件後幾天到幾週內消失（最長：60m MACD 到 2016-11-22、RSI 到 2016-11-09）。
- 舊讀取在 2016-01-30 的 30m、60m **確實保留了 other 那根（1 口）而丟掉真正的日盤 K 棒**——不是理論風險。
- 累積型指標的處理（差分、滾動加總、固定錨點）屬於 M2 的暖機與轉換元件。

分析腳本（唯讀）：`scratchpad/m1/impact.py`，結果 `impact.json`（暫存，未進 repo）。

## 遷移與部署（2026-11-15，Hermes 同意後執行）

- 部署前確認：`model_training_jobs` 沒有 pending／running（done 5、failed 2）。
- 遷移 `2026-11-15_m1_pipeline_provenance.sql` 已套用：4 欄存在（`pipeline_version` text、`warnings`／`env_fingerprint`／`data_report` jsonb）；既有 7 筆維持 NULL。
- `deploy.ps1`（robocopy dry run 先確認只有 M1 變更、無刪除、`.env` 與 NAS 相同）：backend、training 重建，frontend 重啟；三個容器 running、restarts=0；worker 正常輪詢、uvicorn 兩個 worker 啟動完成。
- 容器內驗證：系統 `/usr/lib/libta_lib*` 已不存在；backend 與 training 的 TA-Lib 皆 0.7.1（C 0.7.1）；training 指紋 torch 2.10.0+cu128、CUDA 12.8、cuDNN 91002、RTX 5060 Ti。
- 容器內唯讀讀取 1m／60m／daily_day（2011-01-03～2025-01-01）的讀取報告與本機逐欄相同。
- API：`GET /api/model/jobs`、`/jobs/{id}` 回傳新欄位（舊任務 `pipeline_version: null`、`warnings: []`）；前端 200，新版 bundle 含「資料紀錄」區塊。
- `docs/agent-api/training/api.md` 補上版本表、錯誤與失敗對照、新欄位與 Phase 3 範例，已同步到 NAS。

## 部署後驗證任務（Hermes 同意後送出）

共同設定：XGBoost、tx_60m、OHLCV＋RSI(14)、log_return／fixed_threshold 2 類門檻 0、window 20、Phase 1 random、20 棵樹、seed 42。

| 任務 | 期間 | 結果 |
|---|---|---|
| A `3a9f02fb` | 2024-01-02～2024-03-01 | done（cuda，約 2 秒）；回應 `warnings: []`；`pipeline_version` 2026-11-r1；指紋 torch 2.10.0+cu128／CUDA 12.8／RTX 5060 Ti／TA-Lib C 0.7.1；讀取報告 696 根、排除 0、缺格 0、全部舊版 rule_version；樣本 661（train 529／val 132） |
| B `3ccfb173` | 2025-12-01～2026-02-01 | done；回應與任務紀錄都有 `period_after_research_end` 提示、任務照常執行；讀取報告 428 根，**最後一根 2025-12-31 13:00**——2026 年 1 月在已知的 1～4 月缺口內，實際沒有讀到 2026 年的 K 棒 |

## 待辦

- M2：暖機、轉換元件、共同樣本框架（含實驗與樣本框架資料表的遷移，需另經同意）。
