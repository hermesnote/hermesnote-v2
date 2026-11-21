# 2026-11-21 新目標、特徵、指標、上游機率保存、三種序列模型（本機完成，尚未部署）

## 來源
Dot 的評估請求（首次越界、區間高低點、價格位置／K 棒形態、越界機率串接、CNN-LSTM／GRU／Transformer）。CC 回覆評估並提出 7 個問題；依 Hermes 的原則「能做的先做、先升一版、有缺再補」，研究端尚未回覆的部分先用下列預設實作，Dot 回覆後再調整。

## 預設值（待 Dot 確認）
| 問題 | 本版預設 |
|---|---|
| 上下界來源 | `params.bounds`：上下界各為一個 TA-Lib 指標的某個輸出（例 BBANDS 上下軌）；標籤與 `price_position` 特徵共用同格式 |
| 9 類順序 | 0～H-1＝上穿第 1～H 根、H～2H-1＝下穿第 1～H 根、2H＝未越界；名稱 `up_k`／`down_k`／`none` 寫進 output_specs.classes |
| 缺格／跨時段 | 新目標預設 `gaps=exclude`、`cross_session=false`（日線預設 true）；不連續＝前一根 bar_end_ts ≠ 下一根 bar_start_ts；排除原因 `label_gap` |
| 最高／最低單位 | 相對 T 收盤的點數，範圍 T+1～T+H |
| 有加／沒加機率的樣本 | 未處理（沿用各自的框架與排除）；配對比較本來就只比共同時間點 |
| 多分類 Brier | 各類加總（Brier 1950）；二元用 mean((p₁−y)²)；log_loss labels 固定全部類別 |
| CNN-LSTM | stride 1、padding same、activation 預設 relu；MaxPool 步長＝pool_size |

## 改動
- `training/registry/bounds.py`（新）：上下界設定驗證與計算。
- `training/registry/outcomes.py`：`first_crossing`、`future_high_offset`、`future_low_offset`；meta `gap_check`、`v2_only`、`OUTCOME_CLASSES`。
- `training/registry/labeling_rules.py`：`class_index`；類別型目標與規則的組合檢查。
- `training/graph_refs.py`：`_outcome_issues`（v2 限定、bounds、n_classes＝2H+1、gaps／cross_session 型別）。
- `training/engine_v2.py`：`label_gaps()`、`Ctx.label_gap`／`gap_mask()`；框架建立傳 `gap`；上游 `save_oof` 保存；`run_train_v2` extras 多 `upstream_oof`。
- `training/frame.py`：`label_gap` 原因；`build_frame`／`derive_frame` 多 `gap` 參數（None 時與原本完全相同）。
- `training/output_specs.py`：分類輸出加 `classes`（類別型目標）。
- `training/registry/features.py`：`candle_parts`、`price_position`；`warmup.py` 對應記憶類型。
- `training/evaluation.py`、`registry/metrics.py`：`brier`、`log_loss`（加入 CLASSIFICATION_EVAL）。
- `training/cascade.py`：`upstream_policy.save_oof`；`v2_records.upstream_oof_rows`；`worker_v2.py` 寫入 `dataset=upstream_oof`、`result[上游].n_upstream_oof_records`。不需要 DB 遷移（dataset 欄位是 TEXT）。
- `training/registry/seq_models.py`（新）：`gru`、`cnn_lstm`、`transformer`，共用 `lstm.train_lstm`（新增 `_builder`／`_arch`／`_summary` 參數，LSTM 不傳，行為與權重格式不變）；`v2_inputs.ENCODERS`、`optimizers.adam` 相容清單加入三個家族。
- 文件：`docs/agent-api/training/api.md` 新增「新目標、特徵、模型（2026-11-21）」、版本表增補列、已知限制更新。

## 測試
- `tests/test_targets_models.py`（原名 test_targets_models_20261121.py）12 項：首次越界對照逐筆參考迴圈、碰線算區間內；高低點；拿掉一根 K 棒後前 H 根為 label_gap、跨日預設 label_gap、cross_session／ignore；既有目標無 label_gap；驗證規則；兩個特徵與暖機；Brier／Log loss 手算；9 類 XGBoost 上游＋save_oof＋LSTM 下游（類別名稱、9 欄機率、折次、評估含 brier／log_loss）；三種模型分類與回歸訓練後重新載入推論與訓練當下預測一致、多分支、d_model 驗證；LSTM 產物不含 arch。
- 全部 324 項測試通過（skipped 15，需本機測試庫）；`contract_harness compare`（m1 基準）ALL MATCH。
- 過程中抓到自己的錯：框架缺格區塊一開始插錯位置，把切分程式吞進 if 裡（gap=None 時全部樣本變 excluded），測試抓到後已修正。
- 未測：上游 OOF 寫入真的 DB（欄位數與既有逐筆寫入相同，走同一個 COPY）；真實資料全期間的筆數與記憶體。

## 未做
- 網站舊版模型畫布未加入新目標／模型（用實驗頁 JSON 或 API）。
- 部署：需 Hermes 同意。

## 第二輪：依 Dot 規格收斂（同日）
| Dot 項目 | 改動 |
|---|---|
| 1 上下界 | 首版 BBANDS(20, SMA, ±2σ) 收盤（格式不變，文件給範例）；新增 `label_start_outside`：T 收盤在界外排除；越線維持嚴格 |
| 2 類別順序 | 不變；測試驗證 `output_specs.default.classes` 與 9 欄機率、逐筆 `proba` 欄序 |
| 3 缺格／跨盤 | `label_exclusions()` 取代 `label_gaps()`：`label_gap`（同時段內首尾不相接、K 棒長度≠週期、起點不在時間格上）、`label_cross_session`（新）、尾端仍是 `label_unavailable`／`label_unmatured`；日線例外；框架參數 `gap`→`label_reasons` |
| 4 高低點 | 改百分比 `100×(極值/Close[T]−1)`，unit=percent |
| 5 有／無機率 | 分支 `role: "mask"`：只套用可用性不輸入模型；`training_meta.sample_sets`（train／val 筆數＋ID 雜湊）、Phase 3 `holdout.sample_set`；OOF 逐筆 `upstream` 多 `train_end`、`train_max_label_end` |
| 6 Brier／Log loss | 算法不變；文件補尺度差異、零機率（列和正規化＋sklearn 1.9 截到 [eps,1−eps]）、驗證缺類、訓練折缺類 |
| 7 CNN-LSTM | 測試確認 window 60 → LSTM 20 步、padding same、摘要寬 64 |
| 8 GRU／Transformer | Dot 配置可直接送；差異：`shared` Dense 為必填（不能直接接輸出層），共同欄位必填 |

真實資料（quotes 唯讀，2024-01～06，first_crossing H=4、BBANDS 20）：tx_1m 132,771 點，排除 start_outside 15,378、gap 1,742、cross_session 932、unavailable 4；tx_15m 8,890 點（start_outside 1,108、cross_session 932）；tx_60m 2,344 點（cross_session 932、start_outside 133）。cross_session 932＝233 個時段×每段最後 4 根，符合預期；未越界類別約佔 60～65%。

測試：本檔 16 項（新增：跨時段與缺格與尾端分開、K 棒長度錯誤＝label_gap、起點在界外、有／無機率 train／val／holdout 樣本雜湊相同且輸入少 9 欄、不加遮罩時樣本較多、Dot 三種模型配置訓練、CNN-LSTM 20 步）；全部 328 項通過（skipped 15）；harness ALL MATCH。

## 版本管理整理（同日，Hermes 指示）
- `backup/`（Hermes 建立）：`backup/index.md`＋`backup/20261121/`＝部署前 NAS 線上版中今天改過的 20 個檔與舊 `deploy.ps1`（從 NAS 唯讀複製）。退版方式見 `backup/index.md`。
- `deploy.ps1`：robocopy backend 時排除 `backend\tests`（只在本機執行；完整路徑排除）。下次部署時 NAS 上既有的 tests 會被同步移除。
- 根目錄 `.gitignore`：`tmp/`、`output/`。
- 測試檔改名 `test_targets_models.py`。
- 一次 commit 全部未提交的變更（含 11-16 M2～M7、11-18 OOM 修正、今天）並 push；`backend/tmp_ind_full.json` 不收。

## 部署（2026-11-21 14:35，Hermes 同意）
- 部署前：pending／running 0 筆（done 145）；版本＝commit `618f43c`（已 push）；退版＝`backup/20261121/`（見 `backup/index.md`）。
- `deploy.ps1` 完成，backend／training 重建。容器內 `engine_v2.py`、`seq_models.py`、`frame.py` md5 與本機相同。
- 輕量檢查：architectures 清單含 gru／cnn_lstm／transformer；outcomes 含三個新目標；`job-summaries` 200；`POST /train` 不帶憑證 401；worker 取得鎖、等待 30 秒、回收 0 筆、開始輪詢。
- 已知：robocopy 的 `/XD` 排除的資料夾不會被 `/MIR` 刪除，NAS 上仍留著舊的 `backend/tests`（不再更新，不影響訓練容器）。
- 未做：MCP 小資料驗收（交由 MCP 端執行）。

- 部署後（Hermes 同意）：透過 SMB 刪除 NAS 上舊的 `backend/tests`（26 個舊測試檔，皆在 Git 內）。後端容器映像內仍有一份舊副本，下次部署重建時自然消失。
