# 2026-11-18　特徵庫 164 項實際可送訓練能力驗收（第一階段：只查核與測試）

> 狀態：測試完成，未修改程式、未部署、未送任務、未寫入任何資料庫。修正方案待 Hermes 確認。

## 範圍與環境

- **164 項**＝TA-Lib 161 個函式（`talib.get_function_groups()`，後台與 `indicators/registry.py` 的清單來源）＋ Custom 3 項（`ohlcv`、`raw_price`、`raw_volume`）。`quant_saved_strategy` 要讀資料庫，不在 164 項內。
- **NAS 版本核對**（唯讀 `docker exec`）：backend、training 兩個容器的 9 個核心檔（features、warmup、transforms、v2_inputs、engine_v2、alignment、graph_refs、frame、indicators/registry）md5 與本機完全相同；TA-Lib 0.7.1（C 函式庫 0.7.1）、函式 161 個，三處相同。差異只有 training 容器 numpy 2.4.4／pandas 3.0.2，本機 2.4.6／3.0.5。
- **測試環境**：本機 `backend/.venv`，純計算，不連資料庫、不碰 NAS。
- **路徑**：`graph_refs.validate_graph_spec`（送訓練時同一個驗證，失敗即 API 400）→ `engine_v2.Ctx`／`build_main_frame`／`Rows` → `Trainer.assemble`（`compute_feature` → 暖機遮罩 → 轉換 → 分支組裝與原因代碼）→ `fit_normalization`（框架 train 點）→ `apply_normalization` → `packed` → 訓練輸入矩陣。只有最後的 `train_model` 不呼叫。v2 圖、單一分支、LSTM、window 10。舊格式（`training/graph.py`）不在這次範圍。
- **合成資料**（固定種子，可重現）：`tx_5m`，2026-01-02～03-31 共 3780 根（每日日盤 60 根），價格 21218～22315（台指原始尺度、整數跳動），含連續平盤 48 根（O=H=L=C）、零量 40 根、平盤＋零量 24 根、單根長 K、日間跳空 600 點；邊界條件同時放在暖機區與樣本期內。樣本期 02-02～04-01（框架 2519 點，期前 1260 根歷史）；Label 為 log_return、horizon 1、2 類。
- **判定規則**：
  - 可直接使用：預設參數、不加任何設定即通過驗證，訓練與驗證樣本都 > 0，特徵與矩陣都沒有 NaN／Inf。
  - 設定後可用：驗證要求研究者設定（`warmup.bars` 或 `transform`），依要求設定後同上條件成立。`warmup.bars` 用比照 `POST /warmup/suggest` 的實測建議值（ε 0.01），`transform` 用 `diff`。
  - 需修正：設定後仍無法計算、有效樣本 0、矩陣出現 NaN／Inf，或輸出恆為常數。
- 程式與原始結果：`archive/2026-11-18-feature-acceptance/feature_acceptance.py`、`result.json`。重跑：在 `backend` 下 `PYTHONPATH=. .venv/Scripts/python.exe <script> <輸出資料夾>`。

## 結果

**164 項中，實測可直接使用 122 項、設定後可用 35 項、需修正 7 項、尚未驗證 0 項。**

| 判定 | 特徵選項數 | 輸出欄位數 |
|---|---|---|
| 可直接使用 | 122 | 136 |
| 設定後可用 | 35 | 47 |
| 需修正 | 7 | 7 |
| 合計 | 164 | 190 |

靜態核對對照：122 項自動暖機 ✔；「34 項需指定暖機＋AD／OBV」實測為 33 項需設定 `warmup.bars`＋AD／OBV 需 `transform`（TANH 改列需修正）；MAVP、ACOS／ASIN、EXP／COSH／SINH 實測確認。

## 組合測試

| 組合 | 欄位（預期＝實際） | 矩陣 shape（train／val） | NaN／Inf | 有效樣本＝各單項有效點交集 |
|---|---|---|---|---|
| 10 個 TA-Lib 分組＋Custom 各自整組加入（只含非「需修正」項） | 全部相符 | 例：型態辨識 (2016,10,61)／(503,10,61) | 0／0 | 全部相符 |
| 跨分類自由組合（RSI、MACD、BBANDS、OBV(diff)、ADX、STOCH、CDLDOJI、HT_PHASOR、ATR、ohlcv、LINEARREG_SLOPE、AROON、MINMAXINDEX） | 25＝25 | (2016,10,25)／(503,10,25) | 0／0 | 2519＝2519 |
| 全部可用項一起（158 項，含 TANH） | 184＝184 | (1835,10,184)／(459,10,184) | 0／0 | 2294＝2294 |
| 全部多輸出指標（16 項） | 42＝42 | (2016,10,42)／(503,10,42) | 0／0 | 2519＝2519 |
| Custom 重複欄位（ohlcv＋raw_price＋raw_volume） | 10＝10 | (2016,10,10)／(503,10,10) | 0／0 | 2519＝2519；重複的 OHLCV 欄逐值相同 |
| 同一指標兩個節點（RSI×2） | 2＝2 | (2016,10,2)／(503,10,2) | 0／0 | 2519＝2519 |

- 共同暖機：組合後的有效點集合與各單項有效點的交集**逐點相同**（不只筆數）。動量指標組與全部可用項少 225 點，來自 IMI 的實測暖機建議 1456 根超過合成資料的期前歷史（見下方 IMI）；正式任務會依 `history_requirements` 往前載入。
- 重複欄位與同一指標重複加入都被接受、沒有提示，矩陣會有完全相同的欄位。

## 需要處理的項目

| 項目 | 問題 | 影響 |
|---|---|---|
| EXP、COSH、SINH | 台指價格尺度溢位，全部 +Inf。**組裝只用 `np.isnan` 判斷無效，Inf 不會被排除**；`warmup.bars=0` 時通過驗證、2519 點全部「可用」，正規化後該欄全為 NaN（與 RSI 合用時矩陣 25190 個 NaN）。 | 任務會帶著 NaN 輸入開始訓練 |
| ACOS、ASIN | 輸入定義域 [-1, 1]，台指價格全部超出，全部 NaN | 有效樣本 0，任務在「訓練點 0」失敗 |
| TANH | 台指尺度飽和，輸出恆為 1.0 | 矩陣有效但整欄常數，沒有資訊 |
| MAVP | 需要第二個輸入序列 `periods`，Feature Node 只提供 OHLCV；計算即失敗：`input_arrays parameter missing required data key: periods` | 無法使用 |
| IMI（設定後可用，但有問題） | 平盤（高＝低）時輸出 NaN，樣本期內 22 列，被記成 `warmup_not_ready`（原因代碼不正確）；實測暖機建議把「兩邊都是 NaN」當成不一致，建議 1456 根（lookback 13 即可，設 13 時有效樣本 2499、矩陣無 NaN） | 樣本被排除的原因誤導；暖機建議過長 |
| 分類缺漏（不影響數值） | 逐點數學轉換 ATAN、CEIL、COS、FLOOR、LN、LOG10、SIN、SQRT、TAN、TANH 與 IMI、ACCBANDS、AVGDEV 落在 `unknown`，要求研究者設定 `warmup.bars`（0 即可）；若補進分類，可直接使用會增加 | 研究者多一步設定 |
| 合成資料特性（不是缺陷） | 36 個 K 線型態在合成資料中未觸發，只驗到全 0 的路徑；訓練段整欄常數時正規化 scale＝1，驗證／推論段若出現 ±100 會以原值進矩陣（仍是有限值）。SIN／COS／TAN 對價格數值有效但無經濟意義 | 需要時用真實資料再看 |

## 逐項結果

| 項目 | 群組 | 所需輸入 | 參數（預設） | 輸出欄位數 | 暖機／轉換要求 | 有效樣本（train+val） | NaN／Inf（矩陣） | 矩陣 shape（train／val） | 判定 | 原因／備註 |
|---|---|---|---|---|---|---|---|---|---|---|
| HT_DCPERIOD | 週期指標 | close | — | 1 | 必設 warmup.bars（程式基本 32；測試用實測建議 172） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| HT_DCPHASE | 週期指標 | close | — | 1 | 必設 warmup.bars（程式基本 63；測試用實測建議 174） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| HT_PHASOR | 週期指標 | close | — | 2 | 必設 warmup.bars（程式基本 32；測試用實測建議 123） | 2519 | 0 / 0 | (2016, 10, 2) / (503, 10, 2) | 設定後可用 | 通過；需依「要求」欄設定 |
| HT_SINE | 週期指標 | close | — | 2 | 必設 warmup.bars（程式基本 63；測試用實測建議 175） | 2519 | 0 / 0 | (2016, 10, 2) / (503, 10, 2) | 設定後可用 | 通過；需依「要求」欄設定 |
| HT_TRENDMODE | 週期指標 | close | — | 1 | 必設 warmup.bars（程式基本 63；測試用實測建議 156） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 合成資料中整段常數（資料特性） |
| ADD | 數學運算 | high, low | — | 1 | 自動 0 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| DIV | 數學運算 | high, low | — | 1 | 自動 0 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MAX | 數學運算 | close | timeperiod=30 | 1 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MAXINDEX | 數學運算 | close | timeperiod=30 | 1 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MIN | 數學運算 | close | timeperiod=30 | 1 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MININDEX | 數學運算 | close | timeperiod=30 | 1 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MINMAX | 數學運算 | close | timeperiod=30 | 2 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 2) / (503, 10, 2) | 可直接使用 |  |
| MINMAXINDEX | 數學運算 | close | timeperiod=30 | 2 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 2) / (503, 10, 2) | 可直接使用 |  |
| MULT | 數學運算 | high, low | — | 1 | 自動 0 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| SUB | 數學運算 | high, low | — | 1 | 自動 0 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| SUM | 數學運算 | close | timeperiod=30 | 1 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| ACOS | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 3402） | 0 | — | — | 需修正 | 輸入定義域 [-1,1]，台指價格全部超出 → 全部 NaN，有效樣本 0（任務會在「訓練點 0」失敗） |
| ASIN | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 3402） | 0 | — | — | 需修正 | 同 ACOS：全部 NaN，有效樣本 0 |
| ATAN | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 台指尺度接近 π/2（變動 1.8e-6），等同價格的單調轉換 |
| CEIL | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| COS | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 同 SIN |
| COSH | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 3402） | 368 | 3680 / 0（特徵 Inf 378） | (290, 10, 1) / (78, 10, 1) | 需修正 | 同 EXP：全部 Inf，流入矩陣成 NaN |
| EXP | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 3402） | 368 | 3680 / 0（特徵 Inf 378） | (290, 10, 1) / (78, 10, 1) | 需修正 | 台指價格尺度溢位，全部 +Inf；**Inf 不會被判為無效**，warmup.bars=0 時通過驗證、樣本全數「可用」，矩陣該欄全為 NaN（與其他特徵合用也一樣） |
| FLOOR | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| LN | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| LOG10 | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| SIN | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 對價格取 sin，數值有效但無經濟意義 |
| SINH | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 3402） | 368 | 3680 / 0（特徵 Inf 378） | (290, 10, 1) / (78, 10, 1) | 需修正 | 同 EXP：全部 Inf，流入矩陣成 NaN |
| SQRT | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| TAN | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 同 SIN，範圍 −160～391 |
| TANH | 數學轉換 | close | — | 1 | 必設 warmup.bars（程式基本 0；測試用實測建議 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 需修正 | 台指價格尺度飽和，輸出恆為 1.0；矩陣數值有效但整欄常數（正規化後全 0），沒有資訊 |
| ADX | 動量指標 | high, low, close | timeperiod=14 | 1 | 必設 warmup.bars（程式基本 27；測試用實測建議 103） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| ADXR | 動量指標 | high, low, close | timeperiod=14 | 1 | 必設 warmup.bars（程式基本 40；測試用實測建議 113） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| APO | 動量指標 | close | fastperiod=12, slowperiod=26, matype=0 | 1 | 自動 25 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| AROON | 動量指標 | high, low | timeperiod=14 | 2 | 自動 14 根（finite） | 2519 | 0 / 0 | (2016, 10, 2) / (503, 10, 2) | 可直接使用 |  |
| AROONOSC | 動量指標 | high, low | timeperiod=14 | 1 | 自動 14 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| BOP | 動量指標 | open, high, low, close | — | 1 | 自動 0 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CCI | 動量指標 | high, low, close | timeperiod=14 | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CMO | 動量指標 | close | timeperiod=14 | 1 | 自動 77 根（recursive） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| DX | 動量指標 | high, low, close | timeperiod=14 | 1 | 自動 77 根（recursive） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| IMI | 動量指標 | open, close | timeperiod=14 | 1 | 必設 warmup.bars（程式基本 13；測試用實測建議 1456） | 2294 | 0 / 0 | (1835, 10, 1) / (459, 10, 1) | 設定後可用 | 平盤（高＝低）時輸出 NaN：樣本期內 22 列 NaN，被記成 warmup_not_ready（原因代碼不正確）；實測暖機建議把兩邊都是 NaN 當成不一致，建議出 1456 根（實際 lookback 13 即可） |
| MACD | 動量指標 | close | fastperiod=12, slowperiod=26, signalperiod=9 | 3 | 必設 warmup.bars（程式基本 33；測試用實測建議 88） | 2519 | 0 / 0 | (2016, 10, 3) / (503, 10, 3) | 設定後可用 | 通過；需依「要求」欄設定 |
| MACDEXT | 動量指標 | close | fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0 | 3 | 必設 warmup.bars（程式基本 33；測試用實測建議 33） | 2519 | 0 / 0 | (2016, 10, 3) / (503, 10, 3) | 設定後可用 | 通過；需依「要求」欄設定 |
| MACDFIX | 動量指標 | close | signalperiod=9 | 3 | 必設 warmup.bars（程式基本 33；測試用實測建議 88） | 2519 | 0 / 0 | (2016, 10, 3) / (503, 10, 3) | 設定後可用 | 通過；需依「要求」欄設定 |
| MFI | 動量指標 | high, low, close, volume | timeperiod=14 | 1 | 自動 14 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MINUS_DI | 動量指標 | high, low, close | timeperiod=14 | 1 | 自動 77 根（recursive） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MINUS_DM | 動量指標 | high, low | timeperiod=14 | 1 | 自動 76 根（recursive） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MOM | 動量指標 | close | timeperiod=10 | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| PLUS_DI | 動量指標 | high, low, close | timeperiod=14 | 1 | 自動 77 根（recursive） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| PLUS_DM | 動量指標 | high, low | timeperiod=14 | 1 | 自動 76 根（recursive） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| PPO | 動量指標 | close | fastperiod=12, slowperiod=26, matype=0 | 1 | 自動 25 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| ROC | 動量指標 | close | timeperiod=10 | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| ROCP | 動量指標 | close | timeperiod=10 | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| ROCR | 動量指標 | close | timeperiod=10 | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| ROCR100 | 動量指標 | close | timeperiod=10 | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| RSI | 動量指標 | close | timeperiod=14 | 1 | 自動 77 根（recursive） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| STOCH | 動量指標 | high, low, close | fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0 | 2 | 自動 8 根（finite） | 2519 | 0 / 0 | (2016, 10, 2) / (503, 10, 2) | 可直接使用 |  |
| STOCHF | 動量指標 | high, low, close | fastk_period=5, fastd_period=3, fastd_matype=0 | 2 | 自動 6 根（finite） | 2519 | 0 / 0 | (2016, 10, 2) / (503, 10, 2) | 可直接使用 |  |
| STOCHRSI | 動量指標 | close | timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0 | 2 | 必設 warmup.bars（程式基本 20；測試用實測建議 450） | 2519 | 0 / 0 | (2016, 10, 2) / (503, 10, 2) | 設定後可用 | 通過；需依「要求」欄設定 |
| TRIX | 動量指標 | close | timeperiod=30 | 1 | 必設 warmup.bars（程式基本 88；測試用實測建議 154） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| ULTOSC | 動量指標 | high, low, close | timeperiod1=7, timeperiod2=14, timeperiod3=28 | 1 | 自動 28 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| WILLR | 動量指標 | high, low, close | timeperiod=14 | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| ACCBANDS | 重疊研究 | high, low, close | timeperiod=20 | 3 | 必設 warmup.bars（程式基本 19；測試用實測建議 19） | 2519 | 0 / 0 | (2016, 10, 3) / (503, 10, 3) | 設定後可用 | 通過；需依「要求」欄設定 |
| BBANDS | 重疊研究 | close | timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=0 | 3 | 自動 4 根（finite） | 2519 | 0 / 0 | (2016, 10, 3) / (503, 10, 3) | 可直接使用 |  |
| DEMA | 重疊研究 | close | timeperiod=30 | 1 | 必設 warmup.bars（程式基本 58；測試用實測建議 100） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| EMA | 重疊研究 | close | timeperiod=30 | 1 | 自動 99 根（recursive） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| HT_TRENDLINE | 重疊研究 | close | — | 1 | 必設 warmup.bars（程式基本 63；測試用實測建議 176） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| KAMA | 重疊研究 | close | timeperiod=30 | 1 | 必設 warmup.bars（程式基本 30；測試用實測建議 140） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| MA | 重疊研究 | close | timeperiod=30, matype=0 | 1 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MAMA | 重疊研究 | close | fastlimit=0.5, slowlimit=0.05 | 2 | 必設 warmup.bars（程式基本 32；測試用實測建議 106） | 2519 | 0 / 0 | (2016, 10, 2) / (503, 10, 2) | 設定後可用 | 通過；需依「要求」欄設定 |
| MAVP | 重疊研究 | close, periods | minperiod=2, maxperiod=30, matype=0 | 1 | 自動 29 根（unknown） | 0 | — | — | 需修正 | 需要第二個輸入序列 periods，Feature Node 只提供 OHLCV → 計算即失敗：input_arrays parameter missing required data key: periods |
| MIDPOINT | 重疊研究 | close | timeperiod=14 | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MIDPRICE | 重疊研究 | high, low | timeperiod=14 | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| SAR | 重疊研究 | high, low | acceleration=0.02, maximum=0.2 | 1 | 必設 warmup.bars（程式基本 1；測試用實測建議 51） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| SAREXT | 重疊研究 | high, low | startvalue=0.0, offsetonreverse=0.0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2 | 1 | 必設 warmup.bars（程式基本 1；測試用實測建議 38） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| SMA | 重疊研究 | close | timeperiod=30 | 1 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| T3 | 重疊研究 | close | timeperiod=5, vfactor=0.7 | 1 | 必設 warmup.bars（程式基本 24；測試用實測建議 24） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| TEMA | 重疊研究 | close | timeperiod=30 | 1 | 必設 warmup.bars（程式基本 87；測試用實測建議 117） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| TRIMA | 重疊研究 | close | timeperiod=30 | 1 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| WMA | 重疊研究 | close | timeperiod=30 | 1 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDL2CROWS | 型態辨識 | open, high, low, close | — | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDL3BLACKCROWS | 型態辨識 | open, high, low, close | — | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDL3INSIDE | 型態辨識 | open, high, low, close | — | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDL3LINESTRIKE | 型態辨識 | open, high, low, close | — | 1 | 自動 8 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDL3OUTSIDE | 型態辨識 | open, high, low, close | — | 1 | 自動 3 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDL3STARSINSOUTH | 型態辨識 | open, high, low, close | — | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDL3WHITESOLDIERS | 型態辨識 | open, high, low, close | — | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLABANDONEDBABY | 型態辨識 | open, high, low, close | penetration=0.3 | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLADVANCEBLOCK | 型態辨識 | open, high, low, close | — | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLBELTHOLD | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLBREAKAWAY | 型態辨識 | open, high, low, close | — | 1 | 自動 14 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLCLOSINGMARUBOZU | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLCONCEALBABYSWALL | 型態辨識 | open, high, low, close | — | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLCOUNTERATTACK | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLDARKCLOUDCOVER | 型態辨識 | open, high, low, close | penetration=0.5 | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLDOJI | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLDOJISTAR | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLDRAGONFLYDOJI | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLENGULFING | 型態辨識 | open, high, low, close | — | 1 | 自動 2 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLEVENINGDOJISTAR | 型態辨識 | open, high, low, close | penetration=0.3 | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLEVENINGSTAR | 型態辨識 | open, high, low, close | penetration=0.3 | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLGAPSIDESIDEWHITE | 型態辨識 | open, high, low, close | — | 1 | 自動 7 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLGRAVESTONEDOJI | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLHAMMER | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLHANGINGMAN | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLHARAMI | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLHARAMICROSS | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLHIGHWAVE | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLHIKKAKE | 型態辨識 | open, high, low, close | — | 1 | 自動 5 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLHIKKAKEMOD | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLHOMINGPIGEON | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLIDENTICAL3CROWS | 型態辨識 | open, high, low, close | — | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLINNECK | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLINVERTEDHAMMER | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLKICKING | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLKICKINGBYLENGTH | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLLADDERBOTTOM | 型態辨識 | open, high, low, close | — | 1 | 自動 14 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLLONGLEGGEDDOJI | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLLONGLINE | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLMARUBOZU | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLMATCHINGLOW | 型態辨識 | open, high, low, close | — | 1 | 自動 6 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLMATHOLD | 型態辨識 | open, high, low, close | penetration=0.5 | 1 | 自動 14 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLMORNINGDOJISTAR | 型態辨識 | open, high, low, close | penetration=0.3 | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLMORNINGSTAR | 型態辨識 | open, high, low, close | penetration=0.3 | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLONNECK | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLPIERCING | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLRICKSHAWMAN | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLRISEFALL3METHODS | 型態辨識 | open, high, low, close | — | 1 | 自動 14 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLSEPARATINGLINES | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLSHOOTINGSTAR | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLSHORTLINE | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLSPINNINGTOP | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLSTALLEDPATTERN | 型態辨識 | open, high, low, close | — | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLSTICKSANDWICH | 型態辨識 | open, high, low, close | — | 1 | 自動 7 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLTAKURI | 型態辨識 | open, high, low, close | — | 1 | 自動 10 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CDLTASUKIGAP | 型態辨識 | open, high, low, close | — | 1 | 自動 7 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLTHRUSTING | 型態辨識 | open, high, low, close | — | 1 | 自動 11 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLTRISTAR | 型態辨識 | open, high, low, close | — | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLUNIQUE3RIVER | 型態辨識 | open, high, low, close | — | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLUPSIDEGAP2CROWS | 型態辨識 | open, high, low, close | — | 1 | 自動 12 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| CDLXSIDEGAP3METHODS | 型態辨識 | open, high, low, close | — | 1 | 自動 2 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 | 合成資料未觸發此型態（只驗到全 0） |
| AVGDEV | 價格轉換 | close | timeperiod=14 | 1 | 必設 warmup.bars（程式基本 13；測試用實測建議 13） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| AVGPRICE | 價格轉換 | open, high, low, close | — | 1 | 自動 0 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| MEDPRICE | 價格轉換 | high, low | — | 1 | 自動 0 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| TYPPRICE | 價格轉換 | high, low, close | — | 1 | 自動 0 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| WCLPRICE | 價格轉換 | high, low, close | — | 1 | 自動 0 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| BETA | 統計函數 | high, low | timeperiod=5 | 1 | 自動 5 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| CORREL | 統計函數 | high, low | timeperiod=30 | 1 | 自動 29 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| LINEARREG | 統計函數 | close | timeperiod=14 | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| LINEARREG_ANGLE | 統計函數 | close | timeperiod=14 | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| LINEARREG_INTERCEPT | 統計函數 | close | timeperiod=14 | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| LINEARREG_SLOPE | 統計函數 | close | timeperiod=14 | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| STDDEV | 統計函數 | close | timeperiod=5, nbdev=1.0 | 1 | 自動 4 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| TSF | 統計函數 | close | timeperiod=14 | 1 | 自動 13 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| VAR | 統計函數 | close | timeperiod=5, nbdev=1.0 | 1 | 自動 4 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| ATR | 波動度指標 | high, low, close | timeperiod=14 | 1 | 自動 77 根（recursive） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| NATR | 波動度指標 | high, low, close | timeperiod=14 | 1 | 自動 77 根（recursive） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| TRANGE | 波動度指標 | high, low, close | — | 1 | 自動 1 根（finite） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
| AD | 成交量指標 | high, low, close, volume | — | 1 | 必選 transform（測試用 diff） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| ADOSC | 成交量指標 | high, low, close, volume | fastperiod=3, slowperiod=10 | 1 | 必設 warmup.bars（程式基本 9；測試用實測建議 27） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| OBV | 成交量指標 | close, volume | — | 1 | 必選 transform（測試用 diff） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 設定後可用 | 通過；需依「要求」欄設定 |
| ohlcv | Custom | open, high, low, close, volume | — | 5 | 無（暖機 0） | 2519 | 0 / 0 | (2016, 10, 5) / (503, 10, 5) | 可直接使用 |  |
| raw_price | Custom | open, high, low, close | — | 4 | 無（暖機 0） | 2519 | 0 / 0 | (2016, 10, 4) / (503, 10, 4) | 可直接使用 |  |
| raw_volume | Custom | volume | — | 1 | 無（暖機 0） | 2519 | 0 / 0 | (2016, 10, 1) / (503, 10, 1) | 可直接使用 |  |
## 修正方案（待 Hermes 確認，未動手）

1. **非有限值防護**：`v2_inputs.assemble` 改用 `~np.isfinite` 判斷，暖機後出現的 NaN／Inf 用新的原因代碼（例如 `invalid_feature_value`）排除並記錄，不再算成 `warmup_not_ready`。同時解決 EXP 類流入矩陣與 IMI 原因代碼誤導。
2. **定義域與飽和**：驗證階段對 ACOS、ASIN、EXP、COSH、SINH、TANH 直接拒絕並說明原因（目前 Feature Node 不能先轉換輸入再套函式）；或從後台清單移除。
3. **MAVP**：從清單移除或驗證時拒絕；若要支援，需設計 `periods` 來源（另一個特徵節點的輸出），屬新功能。
4. **記憶類型分類**：逐點數學轉換歸為 `none`，IMI、ACCBANDS、AVGDEV 歸為 `finite`。
5. **實測暖機建議**：兩邊都是 NaN 視為一致。
6. （選做）摘要提示訓練段常數欄與重複欄位。

1、4、5 改 `training/` 程式，影響流程結果（排除原因、暖機根數），需要提升流程版本並重新部署 backend／training；2、3 只影響驗證與清單。
