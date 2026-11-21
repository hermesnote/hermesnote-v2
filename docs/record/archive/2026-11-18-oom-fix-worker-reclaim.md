# 2026-11-18　全期間訓練記憶體修正與 worker 中斷回收（本機實作與測試，未部署）

> 狀態：本機實作與測試完成。**未部署、未修改正式資料庫、未重送訓練。** 需要真正 PostgreSQL 的 6 項測試待 Hermes 同意安裝本機測試用 PostgreSQL 後執行。

起因：Dot 的 2011～2025 tx_1m、155 個特徵（178 欄）Phase 1 兩次在開始約 45 秒時被 NAS 主機 global OOM 終止（交接紀錄 `D:\dot\research\docs\record\archive\2026-11-18-phase1-fixed-library\full-period-oom-diagnostic-handoff.md`），兩筆任務殘留 `running`。決定依 GC 帶話（Hermes 轉交）。

## 修改範圍

| 檔案 | 內容 |
|---|---|
| `training/v2_inputs.py` | A1／A2：`fit_normalization` 分批以 `fmin`／`fmax` 歸約求 min／max，不再複製整份 `base[mask]`；`apply_normalization` 分批以 float64 計算 `(x − min) / scale` 後寫入預先配置的 float32。每批 `NORM_CHUNK_BYTES`（64 MiB）。運算與轉型順序不變；訓練點為 0 時沿用原本的錯誤 |
| `training/engine_v2.py` | A3：`Ctx` 記每個特徵還有幾個 Model Node 未組裝，`assembled(nid)` 在最後一個使用者組裝後釋放快取（訓練與推論路徑都呼叫）；`feature_computations` 記實際計算次數。驗證逐筆結果重算 `nz` 前、串接每一折結束後先釋放舊 `nz` |
| `training/job_store.py` | 單一實例鎖（`acquire_worker_lock`／`lock_alive`，PostgreSQL session advisory lock，伺服器端 keepalive 30／10／3）；原子認領 `claim_next_pending`（`FOR UPDATE SKIP LOCKED`，同時寫 `env_fingerprint.worker`）；`reclaim_orphans`；`mark_done`／`mark_failed_by_worker` 只在任務仍是 `running` 且為本實例認領時寫入；`record_resources` 併入 `data_report.resources` |
| `training/worker.py` | 啟動先取得鎖（取不到每 15 秒重試，不認領）→ 回收 → 每 30 秒檢查鎖，失去鎖立即結束程序（不寫結果）；改用原子認領；任務開始時歸零 `VmHWM`（`/proc/self/clear_refs`），結束時寫入峰值 |
| `tests/test_norm_chunked.py`（新） | 新舊正規化逐位元比對、端到端等價 |
| `tests/test_worker_claim.py`（新） | worker 流程（替身）7 項；PostgreSQL 6 項（需 `TEST_PG_DSN`，未設定時跳過） |
| `tests/test_worker_gpu.py` | 補上新增資料庫步驟的替身 |
| `tests/__init__.py` | 測試一律把 `HERMESNOTE_DATABASE_URL` 指向不存在的位址（除非 `HERMESNOTE_TESTS_ALLOW_REAL_DB=1`） |
| `docs/agent-api/training/api.md` | `env_fingerprint.worker`、`data_report.resources`、取消與回收的狀態規則 |

未改：特徵、期間、切分、訓練條件、資料庫結構（`env_fingerprint` 為 JSONB，整個程式只有開始執行時寫入一次，前端只顯示白名單欄位）、docker-compose。

## 數值比對：全部逐位元相同

- 邊界案例（uint64／uint32 位元比對，NaN 位置與 ±0 都會被抓到）：一般、台指尺度、整欄 NaN、整欄常數、部分 NaN、±0 交錯、Inf、±1e308、次正規數、單欄、遮罩缺口、window 1／5／60、兩個正規化步驟、無步驟、空矩陣；每批 1／7／64／1000／大於 n 列。訓練點為 0 時新舊拋出相同錯誤。
- 真實資料（quotes 唯讀）：2024 tx_1m 274,790 列 × 178 欄（Dot 的 155 個特徵）：`min`、`scale`、正規化矩陣 178／178 欄逐位元相同（48,912,620 值、259,172 個 NaN 位置相同）。此資料沒有 −0。
- 端到端（舊函式＋不釋放快取 vs 新流程）：forward_chaining、purged_kfold、blending、in_sample（XGBoost）；LSTM 串接；多分支跨時間框架（LSTM、XGBoost）；Phase 3 推論。框架雜湊、正規化參數、best／last 權重位元組、驗證逐筆預測、上游各折指紋全部相同；每個特徵只計算 1 次；結束時快取為空。
- 全套 300 項通過（12 項跳過：既有 quotes 唯讀 6 項、新 PostgreSQL 6 項）；`contract_harness` 對 M1 基準 ALL MATCH。
- 不需要提升流程版本（沒有任何數值差異）。

## 記憶體：估算與實測

本機 Windows、合成 tx_1m（日盤 300 根／日）、Dot 的 155 個特徵，各規模各自一個程序，`peak_wset`；含驗證逐筆結果的第二次套用；不含 torch 訓練階段。

| 列數 | 修改前峰值 | 修改後峰值 |
|---|---|---|
| 0.3M | 1.97 GiB | 0.98 GiB |
| 0.6M | 3.85 GiB | 1.87 GiB |
| 1.2M | 7.63 GiB | 3.68 GiB |
| 2.82M（全期間） | 未實測（線性外推 ≈ 17.8 GiB；到 fit 為止 ≈ 12.4、到正規化為止 ≈ 16.0） | **8.54 GiB（實測）** |

- 修改前的外推與核心日誌被終止時的 14.5 GiB（介於 fit 與正規化之間）一致。
- 修改後峰值出現在組裝（特徵快取＋合併矩陣同時存在）；之後常駐約 6.6 GiB（合併矩陣 float64 3.74 GiB＋float32 1.87 GiB＋資料）。
- 尚未實測：NAS（Linux、容器 numpy 2.4.4）、真實全期間資料、torch／CUDA 主機端記憶體與訓練階段。

## 過程中的事件：測試誤連正式資料庫（無資料變更）

第一次全套測試時，既有的 `tests/test_worker_gpu.py` 直接執行 `worker.main_loop`，沒有替新加的「取得鎖、回收、認領」換替身，因此經本機 `.env` 連到正式 hermesnote DB：取得一次 advisory lock、執行回收 `UPDATE`（0 列）、執行認領 `UPDATE`（0 列，正式資料庫沒有 pending），第二個測試因鎖被第一個測試持有而卡住。處理：
- 唯讀核對：`model_training_jobs` 為 failed 2、done 141、running 2，沒有任何任務帶 `env_fingerprint.worker`、3 小時內沒有任務開始；兩筆中斷任務仍為 running；資料庫沒有殘留鎖連線或 advisory lock。**沒有資料變更。**
- 修正：`test_worker_gpu.py` 補替身；`tests/__init__.py` 讓所有測試預設連不到正式 hermesnote DB。

## 未解風險

- 修改後峰值仍有 8.54 GiB 來自組裝時「快取＋合併矩陣」並存；若要再降約 3.7 GiB，可讓組裝直接寫入預先配置的合併矩陣、每複製一個特徵就釋放（A4，未做）。
- 失去鎖即結束程序：資料庫短暫斷線也會中斷進行中的任務（換取不會有兩個 worker 同時寫入）。
- 舊程序連線半開時，新 worker 會等到資料庫偵測斷線（keepalive 約 1 分鐘）才取得鎖。
- 執行中被取消的任務：狀態不再被改回 done（行為變更），但已寫出的模型產物、逐筆結果仍保留。
- 6 項 PostgreSQL 層測試（併發認領、取消、回收不重複、只有自己能寫入、resources 合併、鎖排他與斷線釋放）尚未執行。

## 待確認

1. 下載可攜式 PostgreSQL 到本機測試專用位置，執行 6 項資料庫層測試。
2. NAS 隔離容器全期間記憶體驗證（計畫見回報）。
3. 容器記憶體上限設定值。
4. 部署；兩筆舊任務的資料庫更新；全期間重送。

## 補充（同日）：資料庫層測試、測試防護、鎖失效時的重疊

- 本機測試用 PostgreSQL 16（conda-forge，安裝在 scratchpad 獨立環境，只監聽 127.0.0.1:54329，資料庫 `hn_test_worker`；測完已停止）。6 項資料庫層測試全部通過；全套 304 項：無測試庫時 OK（skipped 12）、有測試庫時 OK（skipped 7）。
- 測試防護補強：`tests/__init__.py` 除環境變數外，已載入的 `services.hermesnote_db` 與 `training.artifacts／inference_store／preview_store／progress`（載入時就存下位址）一併改成封鎖值；所有 `tests/test_*.py` 都在載入程式模組前 `import tests`；`tests/test_db_guard.py` 靜態檢查順序、檢查所有存下位址的模組都已列入、實際連線必定失敗。`unittest discover`、`-m unittest tests.xxx`、直接執行檔案、從其他目錄執行都驗證有效。`TEST_PG_DSN` 只接受本機且名稱 `hn_test_` 開頭的資料庫，指向正式位址時在連線前拒絕。
- 鎖失效重疊（本機測試庫模擬，伺服器端終止舊 worker 的鎖連線）：新 worker 立即取得鎖、回收舊任務並認領下一筆；舊 worker 在 watchdog 下次檢查（最多 30 秒）前：`mark_done` 被擋（✓），但 `save_data_report`、`record_resources` 仍寫入，且**仍能認領下一筆任務**，兩個 worker 同時執行。網路中斷（連線未關閉）時 `lock_alive` 沒有逾時，watchdog 會一直卡住，舊 worker 不會結束；伺服器約 60 秒後釋放鎖，新 worker 開始，重疊不會自行結束。修正建議見回報（未實作）。

## 兩筆中斷任務已結束（Hermes 2026-11-18 指示）

先確認 training 容器 worker 自 05:36:05Z 重啟後閒置（RSS 71 MB、之後沒有認領任何任務），再以單一交易將 `8e6ab8d3…`、`3bb253ed…` 從 `running` 改為 `failed`（條件 `status='running'`，各 1 列），`finished_at` 2026-11-18 09:34:53Z，`error` 各自寫入 OOM 證據（kernel global OOM 時間、PID、anon-rss、容器重啟時間）。只改 `status`／`finished_at`／`error`，其他欄位與資料未刪除。之後任務統計：done 141、failed 4，沒有 running／pending。

## 失敗任務已全部刪除（Hermes 2026-11-18 指示：刪失敗的、完成的都要保留）

刪除前確認 9 張以外鍵關聯的資料表（進度、產物、預覽、推論、逐筆結果、評估、holdout 使用紀錄、串接折次、子任務）對這些任務都是 0 列。以單一交易刪除 `status='failed'` 的 4 筆：`2011e1f0…`、`dac08d4d…`（2026-09-29 使用者手動中斷；`2026-09-29-worker-gpu-capacity-fix.md` 原寫「保持原狀」）、`8e6ab8d3…`、`3bb253ed…`（2026-11-18 OOM 中斷）。刪除後只剩 done 141 筆。OOM 證據保留在本紀錄與 Dot 的交接紀錄；實驗 `995c953d…` 的組別與樣本框架 `410fb7eb…`（defined）未動。

## 一次修復：失鎖後的寫入防護（F1～F6）、A4、原始資料精簡、NAS 驗證、部署（Hermes 2026-11-18 晚間指示）

**修正內容**
- F6 寫入所有權檢查 `training/fence.py`：worker 程序的每一筆寫入在同一個交易內先 `SELECT … FOR SHARE` 確認任務仍是 running 且屬於本實例，否則拋出 `LostOwnership` 不寫入。涵蓋 `progress`、`preview_store`、`artifacts`（同一任務產物改為單一交易）、`inference_store`（每批一個交易）、`experiment_store` 的樣本框架／逐筆結果（整批一個交易）／評估／holdout 使用紀錄／串接折次；任務列的讀取報告、完成、失敗改在 UPDATE 的 WHERE 帶同樣條件；峰值記憶體併入完成／失敗那一次寫入（`RESOURCE_PROBE`）。web 後端與測試不設定實例 → 不檢查。取消也因此在下一次寫入時中止訓練。
- `job_store.WorkerLock`：認領與回收在持有鎖的連線上執行，同一條 SQL 確認仍持有鎖（F5）；鎖查詢 5 秒逾時（F2）。worker：鎖連線中斷通知立即結束程序（F1）、watchdog 每 10 秒（F3）、取得鎖後等待 30 秒才回收（F4，輔助，可用 `WORKER_TAKEOVER_GRACE_SEC` 調整）、認領前再確認鎖。
- A4：`v2_inputs.assemble` 先取得各輸入、每次引用後通知引擎（`Ctx.feature_used`，按所有模型所有分支的引用次數釋放快取），再配置合併矩陣逐塊複製、複製完即釋放。
- 原始資料精簡：`run_train_v2`／`run_infer_v2` 的 `consume_dfs`（worker 開啟）在時間軸建好後清空原始 DataFrame（讀取報告已先算完）。
- 檔案寫入：worker 沒有寫檔（`cascade_audit` 的 CSV 只在命令列工具）。
- 新增 `GET /api/model/job-summaries`（訓練頁下拉選單改列全部紀錄）。

**測試**
- 本機測試用 PostgreSQL（conda-forge 16，以 `model_init.sql` 建完整 13 張表）：worker 層 18 項，含只有持鎖才能認領、每筆只認領一次、取消後所有寫入被拒、非擁有者寫入被拒／擁有者全部成功、回收只動有身分的其他實例且不重複、寫入交易進行中回收會等待提交、**伺服器端切斷舊 worker 鎖後：舊 worker 收到中斷通知、無法再認領、11 種寫入全部被拒且資料表 0 列，新 worker 回收並正常寫入**、真實子程序在鎖連線被切斷後 1 秒內以 70 結束且鎖釋放、鎖查詢逾時視為失鎖。
- 新舊逐位元比對：A4 合併矩陣（各模型各分支）、正規化、權重、驗證逐筆預測、上游各折指紋、Phase 3 推論全部相同；涵蓋四種串接策略、LSTM／XGBoost、多分支（較粗 as-of、較細子 K 棒加總）、累積型＋diff、複合型＋研究者暖機、同一特徵被多模型多分支引用；每個特徵只計算 1 次。2024 真實資料：A4 合併矩陣與舊版 `np.concatenate` 逐位元相同、排除原因相同、正規化 178／178 欄相同。
- 全套 312 項：無測試庫 OK（skipped 15）、有測試庫 OK（skipped 7）；`contract_harness` ALL MATCH；前端 `tsc -b`、`vite build` 通過。
- 測試防護：所有 `tests/test_*.py` 在載入程式前 `import tests`；`tests/__init__.py` 封鎖環境變數與已載入模組（`hermesnote_db`、`artifacts`、`inference_store`、`preview_store`、`progress`）的位址；`TEST_PG_DSN` 只接受本機 `hn_test_*`。

**NAS 隔離全期間驗證**（現有 `hermes-training` 映像唯讀掛載新程式、quotes 唯讀、hermesnote DB 位址封鎖、主機監看 MemAvailable < 2 GiB 即中止）
- 第一次（只有 A1～A3、上限 9 GiB）：組裝時容器被 cgroup 終止（anon-rss 9.41 GB），主機最低可用 3.4 GiB、沒有主機層級 OOM。原因：真實資料讀取後原始 DataFrame 常駐約 3 GiB（本機合成資料沒有這些欄位）。
- 第二次（加 A4＋原始資料精簡、上限 10 GiB、GPU 實際訓練 1 epoch）：**通過**。程序峰值 VmHWM 8.09 GiB（出現在特徵計算＋組裝）、清空原始資料後常駐 0.45 GiB、組裝後 4.56 GiB、訓練中 7.57 GiB；cgroup memory.peak 8.49 GiB、oom_kill 0；主機最低可用 4.25 GiB；1 epoch 約 6.3 分鐘。事前估計 5～6 GiB 低估，實測為準。
- 驗證容器與 NAS 上的驗證檔（含連線設定）已移除。

**部署**：確認沒有 pending／running 後執行 `deploy.ps1`，backend／training 重建。容器內 5 個核心檔 md5 與本機相同；worker 取得鎖（`hermesnote-training-worker-lock`）、等待 30 秒、回收 0 筆；`job-summaries` 回 141 筆；train／infer／experiments 不帶憑證 401；正式前端 bundle 已含新清單。任務統計 done 141。

**兩筆 OOM 中斷任務**：晚間指示要求保留證據改為 failed，但這兩筆已在同日下午依 Hermes 指示（刪除所有失敗任務）刪除，無法照辦；OOM 證據保留於本紀錄與 Dot 交接紀錄。

**仍有限制**
- 全期間峰值 8.09 GiB＋其他服務：NAS 平日可用 7～14 GiB（Immich 處理照片時 immich-ml 會升到 5 GiB），可用量低時全期間訓練仍可能觸發主機 OOM。容器記憶體上限（建議 12 GiB）未設定，待確認。
- 峰值出現在特徵計算＋組裝過程（比組裝後常駐多約 3.5 GiB），推測為記憶體配置器保留，未再細分。
- 失鎖即結束程序：資料庫短暫斷線也會中斷進行中的訓練。
- 不重送正式訓練；四筆全期間訓練由 Dot 接手。
