# 訓練系統 V2 重新設計：三方對齊結論 + 實作排程

三方（Hermes／CLAUDE／Codex 研究端）對齊討論的完整過程與逐項理由，見
[docs/temp/2026-09-10-three-party-design-alignment.md](../../temp/2026-09-10-three-party-design-alignment.md)。
這份文件只記錄**定案範圍**與**實作排程**，不重複討論過程；要查「為什麼這樣設計」回去看對齊文件。

## 背景：這次為什麼要重新設計

R1 那輪（`2026-09-08-training-plan.md`）做出來的訓練系統，Hermes 實際打開 `/admin/model` 表單後，發現一連串設計問題：特徵清單沒有共用回測的 TA-Lib 指標庫、Label/Target/Horizon 焊死在一個 function 裡無法擴充、Architecture 超參數表單過於陽春、模型融合有洩漏風險。這些問題透過跟 Hermes 逐輪討論收斂，後來因為牽涉到教授的研究方法論（Train/Val/Test 三階段協定），進一步拉 Codex（研究端）三方對齊，最終產出一份完整的組件契約（A～G）+ 三階段任務血緣機制。這份文件就是那次對齊的實作落地清單。

## 定案的架構總覽

```
A 時間化資料/特徵 → B Target(outcome+task_type+labeling_rule) → C 模型訓練(切分策略+洩漏防範)
                                                                        ↓
                                                          D 模型產物(自動存檔+載入推論)
                                                                        ↓
                                    E 多模型對齊(時間戳+prediction_source) ← Phase 1/2/3 任務血緣機制
                                                                        ↓
                                              F 決策規則契約(→回測)      G 回測執行修正(獨立/隨時可做)
```

依賴順序：A 是地基，B/C/D 依序疊上去，E 依賴 D，F 依賴 D，G 完全獨立可以隨時做。

## Checklist

- [ ] **0. 獨立小修正（不依賴其他項目，可隨時做）**
  - [x] 0.1 `services/backtest_demo.py:62` 的 `freq="15min"` 改成依傳入的 `timeframe` 動態帶入（新增 `_TIMEFRAME_TO_FREQ` 對照表＋`_resolve_freq()`，`compute_indicator_backtest`/`compute_strategy_backtest`/`run_backtest` 都改吃 `timeframe` 參數，`main.py`／`job_runner.py` 呼叫端同步更新）
  - [x] 0.2 `quant_saved_strategies` 的 `unsave_strategy` 改成軟刪除（NAS DB 已用 `ALTER TABLE ADD COLUMN IF NOT EXISTS deleted_at` 加欄位＋`quant_init.sql` 同步更新給未來全新環境用；`unsave_strategy` 改成 `UPDATE ... SET deleted_at = now()`，`list_saved_strategies`／`get_saved_strategy_id_for_job` 都加 `WHERE deleted_at IS NULL`；`quant_saved_strategy` Feature Node 讀取端沒有濾這個欄位，舊 job 仍能解析已軟刪除的策略，行為正確）
  - [x] 0.3 `services/backtest_demo.py` 的 `fees=0.0` 改成可傳入參數（`run_backtest`/`compute_indicator_backtest`/`compute_strategy_backtest` 都加 `fees: float = 0.0`，預設行為不變，之後要接 UI 或 Agent 傳非零值不用再改函式簽名）
  - [x] 0.4 模型訓練頁的「特徵」清單改成共用回測頁的 TA-Lib 161 指標 registry：`TreeBuilder.tsx` 的 `IndicatorAdder`／`findIndicator` 改成 export，`ModelSettings.tsx` 新增 fetch `/api/indicators`，`talib_indicator` 的「指標名稱」從純文字輸入改成可搜尋選單，選定後自動帶入該指標的預設 `timeperiod`（若有）
  - [x] 0.5 表單排版修正：`.model-editor-panel` 寬度從 `min(900px,100%)` 調成 `min(960px,100%)`，5 個欄位＋間距需要 864px，900px 扣掉左右 padding 後不夠塞下第 5 欄才會被 wrap 擠到下一行
  - 順便修的兩個小 bug（不在原清單但屬同性質）：模組編號用全域計數器不會歸零導致「模組4」這種跳號（改成用畫面上實際模組數量算）；模組卡片沒有直接可見的刪除按鈕，只能雙擊進編輯才找得到（卡片右上角加了 × 直接刪除）
  - 驗證：`tsc --noEmit` 通過、`python -m py_compile` 通過所有改到的後端檔案、前端 dev server 起得來且 `/admin/model` 頁面沒有跟這次改動相關的 console 錯誤（無法用真帳號登入測完整互動，之後 Hermes 登入後台可肉眼確認指標搜尋選單跟排版）

- [x] **1. A：時間戳貫穿三個 registry + graph.py 對齊重構（地基工程，優先做）**
  - [x] 1.1 **設計調整（跟原清單寫的不同，理由如下）**：沒有改 `features.py`/`labels.py` 每個函式的回傳簽名，因為所有 Feature/Label Node 都是對「同一個 df」計算、輸出長度永遠等於 `len(df)`，時間戳本來就等於那個 df 的索引，不需要每個 function 各自回傳一份——改成在 `graph.py` 內部維護一個跟 `outputs` 平行的 `timestamps` 字典，`feature`/`label` 節點的時間戳直接記錄 `df.index.values`。這樣改動範圍小很多，也不用動任何已登記的 function
  - [x] 1.2 `graph.py` 新增 `_ensure_time_indexed()`：把 `df` 轉成排序、去重的 `DatetimeIndex`（訓練這條路徑原本沒有做這個去重，回測那邊 `prepare_ohlcv` 早就有做，這次補齊，屬於順便修的既有缺口）；`model` 節點的對齊邏輯從「`a[-min_len:]` 裁到最短長度」改成用 `np.intersect1d` 算所有輸入＋Label 的時間戳交集，再用 `np.searchsorted` 各自取出交集對應的列——不再假設「長度一樣就是對齊的」，中間缺一根也能對到正確位置。舊版那個 `-0==0` 切片陷阱的防呆註解跟著這次重構一起移除（整條路徑已經不再用負數長度切片，這個問題類別直接消失，不是繞過）
  - [x] 1.3 本機測試（`hermesnote-backend` conda env，stub 架構，不用真的訓練）：① 基本情境（1 個 Feature+Label+Model）維持正確；② 兩個 Feature Node 同時間框架串接，維度跟樣本數精確驗證（10 維特徵、279 筆窗格，跟手算公式完全吻合）；③ **刻意在上游資料中間拿掉一根K棒模擬缺棒**，驗證改成時間戳交集後，融合節點還是能對到正確的共同時間點而不是憑陣列長度矇混——交集長度精確等於 199（跟預期完全吻合），不會因為兩邊陣列長度不同就出錯，也不會因為長度剛好一樣就悄悄對錯位置

- [ ] **2. B：Target 拆分 + 回歸訓練分支（範圍已由 Hermes 確認要做完整，不做半套）**
  - [x] 2.1 新增 `training/registry/outcomes.py`（`OUTCOME_REGISTRY`）：`simple_return`（原 `nbar_return` 內部邏輯搬出來）、`log_return`（新）、`future_price`（新）
  - [x] 2.2 新增 `training/registry/labeling_rules.py`（`LABELING_RULE_REGISTRY`）：`fixed_threshold`（分類，原切類別邏輯搬出來）、`identity`（回歸，直接輸出 raw 數值）；順便加了 `validate_combination()`＋`ILLEGAL_COMBINATIONS`（目前只有 `future_price + fixed_threshold` 不合法），後端在 `POST /api/model/train` 直接擋，不只靠前端擋——Agent 可能繞過前端直接打 API
  - [x] 2.3 `graph.py` 的 `label` 節點改成先 `compute_outcome()` 再 `apply_labeling_rule()`，拿到 `task_type` 存進 `label_task_types` 字典；`model` 節點依 `task_type` 決定 `y` 的 dtype（分類 int / 回歸 float），並把 `task_type` 傳給 `train_model()`；`output_mode="probability"` 現在只在分類時生效（回歸沒有機率的概念）；舊的 `training/registry/labels.py` 整支刪除（不是保留兩套）
  - [x] 2.4 `train_lstm`／`train_xgboost` 都加了 `task_type` 參數＋回歸分支：LSTM 回歸換單一輸出神經元＋`MSELoss`，指標用 MAE；XGBoost 回歸換成 `XGBRegressor`＋`reg:squarederror`，指標用 MAE。**順便修了兩個連帶發現的問題**：① `train_lstm` 的驗證段原本整包一次丟 GPU（`Xv = torch.tensor(X_val, ..., device=device)`），改成跟訓練段一樣分批次搬（新增 `_batched_eval()`），這正是之前跟 Hermes 討論過的 VRAM 風險，等於順手做掉了原本排在第 3 步的 3.4；② `worker.py` 的 `run_one_sync` 之前只濾掉 `"predict"`，沒有濾掉 `"predict_proba"`——任何 `output_mode="probability"` 的分類任務結果要存進 DB 時，會因為 `json.dumps` 序列化不了殘留的 function 物件而整個失敗，這是本輪順手發現並修掉的既有 bug，跟這次改動同一批 code review 抓到的
  - [x] 2.5 前端 `ModelSettings.tsx`：outcome 下拉選單切換時，若目前的 `labeling_rule` 對新 outcome 不合法（`ILLEGAL_LABELING_RULES` 前端端也存了一份跟後端一致的對照），自動切換成第一個合法選項；labeling_rule 選單本身也會過濾掉不合法選項，不會讓使用者選出無效組合；`n_classes`/`threshold_pct`/「輸出」欄位只在 `labeling_rule === "fixed_threshold"` 時顯示，回歸模式下這些欄位沒有意義所以隱藏
  - [x] 2.6 Architecture 超參數表單：新增 `ARCHITECTURE_PARAM_FIELDS` 對照表，依 `architectureKey` 動態渲染對應欄位——LSTM（units/layers/dropout/batch_size/learning_rate）、XGBoost（n_estimators/max_depth/learning_rate），純數字輸入框，`TrainingModule` 型別新增對應欄位，`buildGraphSpec()` 依架構組出對的 `params`
  - [x] 2.7 本機測試（`hermesnote-backend` conda env，真的 `lstm`/`xgboost`，本機有 CUDA）：① 分類（`simple_return`+`fixed_threshold`）維持原本行為；② 回歸走 3 種 outcome（simple_return/log_return/future_price）× 2 種架構全部跑過，數值量級符合預期（例如 log_return 的 MAE 是 0.03 這種小數量級、future_price 的 MAE 是 98 這種價格量級，沒有搞混單位）；③ 不合法組合（`future_price`+`fixed_threshold`）在 `validate_combination()` 直接呼叫、以及整個 `run_graph()` 執行流程兩個層次都正確擋下；④ `worker.py` 濾掉 `predict`/`predict_proba` 的邏輯獨立驗證過 `json.dumps` 不會再炸掉。**誠實說明測試範圍限制**：本機沒有裝 TA-Lib（C 函式庫只在 training 容器裡編過），沒辦法把整個 `main.py`／FastAPI app 在本機啟動起來做端到端測試，改成直接 `import routers.model_training` 驗證新的 `/api/model/registry/outcomes`／`/api/model/registry/labeling_rules` 兩個端點確實掛上、回傳內容正確；`tsc --noEmit` 前端型別檢查通過，dev server 起得來沒有 console 錯誤，但 `/admin/model` 需要 Google 登入沒辦法用真帳號測完整互動

- [ ] **3. C：訓練切分策略 + 標籤洩漏防範 + VRAM 修正**
  - [x] 3.1 新增 `split_strategy` 參數（`random`／`chronological`），`graph.py` 的 `model` 節點依此切換，前端 `ModelSettings.tsx` 加了對應下拉選單（標註 Phase 1／Phase 2 對照，方便理解，跟 Phase 血緣機制本身是兩件事，這輪只做參數本身，血緣機制排在第 5 步）
  - [x] 3.2 這一步依賴 Phase 1/2/3 任務血緣才有「holdout」概念可以框，已經跟第 5 步一起做完：`POST /api/model/infer` 在 `phase=3` 時檢查 `body.start` 必須晚於來源 Phase 1/2 job 的訓練期間結束日期，holdout 起始日期落在訓練期間內會被直接擋下（400），不是只靠使用者自律不要選重疊的區間
  - [x] 3.3 `chronological` 模式下，train→validation 邊界排除標籤成熟時間跨越切分時刻的樣本：用 `horizon`（後 h 根實際K棒的位置語意，不是固定時間長度）算出每個樣本的標籤成熟時間，跟 val 起始時刻比較，跨界的從 train 排除；`random` 模式維持原行為不做這個排除（現有實驗結果的重現基準，見 3.1 的設計理由）；排除數量記錄在 `model_result["training_meta"]["n_excluded_boundary"]`
  - [x] 3.4 **VRAM 修正**：已在做 2.4（回歸分支）時順手一起改掉，`architectures.py` 新增 `_batched_eval()`，訓練/驗證都分批次搬 GPU，不再整包一次丟
  - [x] 3.5 前處理狀態明確存 `preprocessing: none`，跟 3.3 的排除資訊一起放進 `model_result["training_meta"]`（`split_strategy`／`n_train`／`n_val`／`n_excluded_boundary`／`preprocessing`），不用等第 4 步的 `model_artifacts` 表就能先查得到
  - [x] 3.6 本機測試（stub 架構）：① `random` 切分行為不變，`n_excluded_boundary` 恆為 0；② `chronological` 切分確實排除邊界樣本；③ **精確驗證排除數量跟 horizon 成正比**——horizon=1 排除 1 筆、horizon=20 排除 20 筆，在這組等間隔測試資料上精確吻合，不是大概對而已；④ `preprocessing: "none"` 有被正確記錄

- [x] **4. D：模型產物自動保存 + 載入推論（範圍已由 Hermes 確認要完整，不只做保存）**
  - [x] 4.1 新表 `model_artifacts`：**設計調整（跟原清單「權重存放位置」寫法不同，理由如下）**——training 容器跟 backend 容器目前沒有共用 docker volume，存檔案兩邊會互相讀不到、容器重建也會沖掉，改成 `weights` 用 `BYTEA` 直接存進 Postgres（兩個容器本來就都連得到 DB），不用改 docker-compose、不影響「container 歸 container」原則；其餘欄位（`feature_schema`／`target_spec`／`graph_spec_snapshot`／`preprocessing_state`／`depends_on_node_ids`／`kept`）都照原規劃。`model_training_jobs` 順便加了 `job_type`／`phase`／`parent_job_id` 三個欄位（`phase`/`parent_job_id` 是第 5 步要用的，這輪順手一起加，欄位新增成本低，不用之後再改一次表結構）。已在 NAS `hermesnote` DB 執行
  - [x] 4.2 **自動保存時機**：`training/artifacts.py` 的 `save_artifacts_for_job()` 在 `worker.py` 的 `run_one_sync()` 裡、`run_graph()` 訓練成功結束當下就呼叫，不是等使用者按按鈕；後台「儲存模型」按鈕呼叫 `POST /api/model/artifacts/{job_id}/{node_id}/keep` 只是切換 `kept` 標記，不影響產物存不存在
  - [x] 4.3 完整依賴鏈：`depends_on_node_ids` 記錄這個 model node 依賴的上游「其他 model node」id；推論時 `training/inference.py` 遞迴載入整條依賴鏈的產物，各自重建模型再組合，不會偷偷重訓上游——**過程中用真實訓練+推論的端對端測試抓到一個真的 bug 並修掉**：推論階段一開始沒有尊重上游節點的 `output_mode`（一律用 `predict()` 拿類別），導致融合情境下餵給下游的特徵維度跟訓練時對不上直接炸掉（訓練時 `output_mode=probability` 是 2 維，推論時 `predict()` 只給 1 維），修成推論也依 `output_mode` 決定用 `predict()` 還是 `predict_proba()`，跟 `graph.py` 訓練時的規則對稱
  - [x] 4.4 權重格式：`architectures.py` 新增 `LSTMNet.build()` 讓訓練/載入共用同一份結構定義（避免訓練用一個 class 定義、載入時重新定義另一個導致 `state_dict` 對不起來）；LSTM 存 `state_dict`（`torch.save` 序列化成 bytes）＋`model_config`（n_features/units/layers/dropout/out_dim/is_cls）；XGBoost 用 `Booster.save_raw()` 原生格式（不是文字 dump）＋`model_config`（is_cls/n_classes）。**誠實記錄一個已知限制**：多分類 XGBoost 用原生 `Booster`（不是訓練時的 sklearn wrapper）載入後，`predict()` 能正確給類別，但 `predict_proba()` 目前做不到（`multi:softmax` 訓練出來的原生 booster 吐不出機率分佈，這是 sklearn wrapper 內部另外處理的邏輯），二元分類沒有這個問題；程式碼裡明確擋掉並說明原因，不是給錯誤數字冒充
  - [x] 4.5 新增推論 job 類型：`model_training_jobs` 加 `job_type`（'train'/'infer'）沿用同一個 job 佇列/`status` 狀態機，不是另建一套系統；`job_type='infer'` 時 `graph_spec` 欄位改存 `{target_job_id, target_node_id, start, end}`；`worker.py` 的 `main_loop` 依 `job_type` 分流到 `run_one`（訓練）或 `run_one_infer`（載入推論，新資料的 timeframe 從已保存的 `graph_spec_snapshot` 反推，不用使用者重新指定 symbol/timeframe）
  - [x] 4.6 API：`POST /api/model/artifacts/{job_id}/{node_id}/keep`（決定 kept 標記，不是原規劃的 `/save`，因為保存本身是自動的，這個端點語意上是「切換收藏標記」更準確）、`GET /api/model/artifacts`、`POST /api/model/infer`；查詢類用 `services/model_artifacts.py`（async/asyncpg，給 FastAPI route handler 用，不會擋住 event loop），保存/載入用 `training/artifacts.py`（sync/psycopg2，跟 `progress.py` 同一套理由，在訓練容器的同步執行緒裡跑）
  - [x] 4.7 後台 UI：`ModelSettings.tsx` 歷史紀錄每個已完成的模型節點旁加「☆收藏模型/★已收藏」切換按鈕（比照 `/admin/quant` 的收藏模式）＋「用這個模型推論」按鈕，點了跳出小表單選新的日期區間送出；`job_type='infer'` 的歷史紀錄項目顯示「推論筆數」而不是訓練指標
  - [x] 4.8 本機測試（打真的 NAS `hermesnote` DB，真的 CUDA LSTM/XGBoost，測完清理測試資料）：① 單模型完整跑一次「訓練→自動存 `model_artifacts`（驗證 bytea 權重長度 > 0）→用接續在後的新資料做推論→拿到跟樣本數對得上的預測+時間戳」，全程通過；② **模型融合（XGBoost→LSTM）完整跑一次同樣流程**，驗證 `depends_on_node_ids` 正確記錄、推論最終節點會遞迴載入上游產物、也能單獨只推論上游節點——這一輪測試就是抓到 4.3 那個 `output_mode` bug 的地方，修完後兩種情境都驗證通過；`tsc --noEmit` 前端型別檢查通過，dev server 起得來沒有相關 console 錯誤

- [x] **5. Phase 1/2/3 任務血緣機制（教授三階段協定，硬性規定，跟 D 共用同一套機制）**
  - [x] 5.1 `model_training_jobs` 加 `phase`（1/2/3，可為 NULL＝不屬於三階段協定的一般訓練）＋`parent_job_id`（自我參照 FK）——這兩個欄位在第 4 步做 `model_artifacts` 表時就順手一起加了，這步是把邏輯接上
  - [x] 5.2 新增 `training/phases.py`：`validate_parent_for_phase()` 擋不合法血緣（來源 job 的 phase 不對／status 不是 done）；`POST /api/model/train` 的 `phase=2` 分支、`POST /api/model/infer` 的 `phase=3` 分支都會呼叫，不合法回 400 附清楚的錯誤訊息
  - [x] 5.3 `training/phases.py` 的 `build_phase2_graph_spec()`：深拷貝 Phase 1 的完整 `graph_spec`，只把每個 Model Node 的 `split_strategy` 覆寫成 `chronological`，其餘一律不接受呼叫端覆寫——`POST /train` 在 `phase=2` 時完全不理會 `body.nodes/start/end`，圖是從父 job 直接繼承出來的，不是使用者重新填一份長得很像的表單
  - [x] 5.4 Phase 3 是 `job_type="infer"` 的 job（沿用第 4 步做好的推論機制），`phase=3` 時額外要求來源 job 必須是 Phase 2＋status=done，且**新增了 holdout 邊界檢查**（3.2 那項延後到這裡一起做的部分）：holdout 起始日期必須晚於 Phase 1/2 訓練期間的結束日期，重疊會被 400 擋下，不是只靠使用者自己記得不要選錯區間；`target_node_id` 沒指定時用 `training/phases.py` 的 `final_model_node_id()` 自動找出圖裡「沒有被其他 Model Node 依賴」的最終輸出節點，不用使用者自己去猜 node id
  - [x] 5.5 UI 雙向觸發：`ModelSettings.tsx` 主表單送出一律標記 `phase=1`（不需要使用者自己選，送出訓練就是走三階段協定的起點）；歷史紀錄裡 Phase 1 的已完成項目旁有「進到 Phase 2」按鈕（一鍵，不用重填表單）、Phase 2 的已完成項目（且已有模型產物）旁有「進到 Phase 3（holdout）」按鈕（跳出小表單只需要填 holdout 日期區間）；每筆歷史紀錄前面會顯示 `Phase N` 標籤
  - [x] 5.6 本機測試（打真的 NAS DB，真的 CUDA LSTM，測完清理）：完整跑一次 Phase1（random）→Phase2（自動繼承+chronological，驗證跟父 job 除了 split_strategy 外其餘設定完全一致）→Phase3（infer，驗證不重新訓練、自動解析 target_node_id）的血緣鏈，全部通過；另外驗證三種邊界情境會被正確擋下：Phase2 來源指向另一個 Phase2（擋）、Phase3 來源指向 Phase1（擋）、Phase3 的 holdout 起始日落在訓練期間內（擋）

- [x] **6. E：多模型對齊語意修正**
  - [x] 6.1 上游模型輸出帶時間戳——這個在第 1 步（A）做時間戳地基工程時就已經做了（`timestamps[nid]` 對每個節點型別一視同仁，Model Node 也不例外），不用重做
  - [x] 6.2 `graph.py` 的 `model` 節點新增 `prediction_source`（逐列標記 `train`/`val`/`unused`，`unused` 是因為 chronological 邊界排除而沒被用在訓練或驗證任一邊的樣本，不是漏掉，是誠實標記），存進 `model_result["prediction_source"]`；`training/inference.py` 的推論結果也對應加了同一個欄位，逐列都標 `"inference"`（跟訓練時 train/val 是同一套欄位語意，不是另外一套判斷邏輯）——已用真實訓練資料驗證過陣列長度精確對得上 `n_train+n_val`

- [x] **7. F：決策規則契約（只定義介面 + 最小驗證，不做完整 UI）**
  - [x] 7.1 新增 `training/registry/decision_rules.py`（`DECISION_RULE_REGISTRY`，跟其他 registry 同一套查表模式）：簽名 `(predictions, timestamps, task_type, params) -> dict`，回傳 key 是 `long_entry`/`long_exit`/`short_entry`/`short_exit`（跟 `indicators/tree.py` 的 `TRIGGER_LABELS` 語意完全對應），value 是布林陣列——這輪刻意不碰 `indicators/tree.py` 本身，它現有的 entries/exits 格式已經很成熟，只做「模型輸出→這個格式」的轉換層
  - [x] 7.2 兩個最小驗證 function：`probability_threshold`（分類機率門檻，`class_index` 指定看哪一類、`entry_threshold`/`exit_threshold` 兩個門檻，只給 long_entry/long_exit，不擅自幫多空各配一組；輸入不是 2D 機率會直接擋，不會誤把類別輸出當機率算）、`regression_sign`（回歸值正負號當方向，`params.outcome_key == "future_price"` 會直接擋掉——價格絕對值域沒有方向意義，這個檢查是純防呆，計算本身不依賴這個參數）
  - [x] 7.3 API：`GET /api/model/registry/decision_rules`（只有查詢端點，沒有對應的送出端點——模型訓練完→選決策規則→送進回測 job 這條完整路徑刻意留到下一輪，這裡只交付契約）
  - [x] 7.4 本機測試：兩個 function 都用形狀正確的假輸入驗證轉換結果正確（`regression_sign` 逐筆核對正負號轉換的布林值完全吻合）；三種不合法用法都驗證會被擋下——分類規則收到 1D（非機率）輸入、分類規則用在回歸模型、`regression_sign` 用在 `outcome=future_price`

## 明確排除在這輪範圍外（不是忘記，是討論後決定延後）

- 完整 rolling/expanding walk-forward 多折機制、真正 OOF 樣本外預測——模型融合時上游模型的「全範圍預測」仍是用同一個模型對自己訓練過的資料重新預測一次，`prediction_source` 逐列標記的是 `train`/`val`/`unused`（訓練時）或 `inference`（推論時），這是誠實標記實際來源，不是假裝已經是樣本外
- 正規化（train-only fit）、early stopping、gradient clipping、optimizer 選項（AdamW/weight_decay）、類別不平衡處理——`model_result["training_meta"]["preprocessing"]` 固定是 `"none"`，明確可查
- 跨時間框架（不同 bar_interval）的重採樣對齊
- 波動度門檻／分位數切分／成本導向／triple barrier 這些進階 labeling_rule
- 決策規則→回測的完整後台 UI 串接（F 只做契約＋查詢端點，沒有送出端點）
- 交易執行層面（即時報價、滑價模型細節）——研究端明確定調這是下一階段、跟本輪訓練系統重做完全分開的問題

## 誠實聲明（給之後任何人看這份 checklist 時的提醒，第二輪 UX/參數修正後更新）

A～G 七個組件、Phase 1/2/3 任務血緣機制全部做完並用真實 NAS DB + 真實 CUDA 訓練測試過（見各項目下的測試記錄），架構上把契約打對了。**第二輪已經把正規化（train-only fit MinMax）跟 early stopping 補上**（見下方「第二輪：UX 修正＋模型參數擴充」），這兩項不再是缺口；仍然缺的是**真正 OOF 樣本外預測**——多模型融合的訓練結果**還不能當研究證據使用**，只能證明架構可以運行、Phase 1/2/3 血緣機制跟 holdout 邊界防護正確生效。這點需要反映在 UI／文件上，避免之後誤用當研究結論。

---

## 第二輪：UX 修正 ＋ 模型參數擴充（Hermes 實際打開後台操作後的回饋）

第一輪部署後，Hermes 實際登入 `/admin/model` 操作，抓到幾個設計問題跟參數不足，收斂討論後這輪修正：

- [x] **特徵分組**：`ModelSettings.tsx` 新增「＋整組加入」按鈕，可以一次把某個 TA-Lib 分類（例如「動量指標」）整組加入，各自帶預設 `timeperiod`；加進來的特徵會標記 `group`，畫面上同一組會收在一起顯示、可以一鍵「移除整組」（消融實驗時把某一組指標整組拿掉重跑 Phase 1 用）。不是巢狀分組，是平的分組標記，純前端 UI 概念，不影響 `graph_spec` 的節點格式。
- [x] **split_strategy 不再讓使用者自選（修真的 bug）**：上一輪發現的設計矛盾——主表單讓人自由選 random/chronological，會讓人誤以為不用先跑過 Phase 1 就能建 Phase 2。改成主表單一律鎖定送出 `split_strategy: "random"`（`phase: 1`），UI 上完全拿掉這個下拉選單；chronological 只能透過歷史紀錄「進到 Phase 2」按鈕自動帶入（`training/phases.py` 的 `build_phase2_graph_spec()` 本來就是這樣做，這輪只是把前端「應該鎖但沒鎖」的漏洞補上）。
- [x] **模型參數大幅擴充**（比對 V1 完整清單＋教授課程教材 `H_時序列模型理論與實作.pdf` 的 Keras 範例架構 LSTM→Dropout→Dense→Dropout→輸出）：
  - LSTM 新增：`dense`（Dense 隱藏層寬度）、`head_dropout`（Dense 後面獨立的 dropout，不是跟 LSTM 內部 dropout 共用同一個數字）、`bidirectional`、`l2_lambda`（對應 Adam 的 `weight_decay`）、`patience`（early stopping，0=不啟用）、`class_weight`（none/balanced，三分類「平」樣本數過多時可用）。`LSTMNet.build()` 結構同步改成 LSTM→Dropout→Dense(relu)→Dropout→輸出，訓練/載入推論共用同一份結構定義。
  - XGBoost 新增：`subsample`、`colsample_bytree`、`patience`（`early_stopping_rounds`）。
  - **過程中測試抓到一個真的 bug**：XGBoost 開 early stopping 後，載入推論階段用原生 `Booster.predict()` 沒有限制只用到 `best_iteration` 那棵樹為止，會用到 early stopping 之後「已經開始過擬合」的多餘樹，導致載入後的預測跟訓練當下對不起來——已修（`model_config` 存 `best_iteration`，載入時用 `iteration_range` 限制）。
  - LSTM early stopping 也測試過：確實會提早停止、還原的是驗證集 loss 最好那次的權重，不是最後一個 epoch 的。
- [x] **資料標準化（MinMax，train-only fit）**：這是教授明確要求、V1 也有的功能，這輪補上——`graph.py` 只用 `train_idx` 那些列算 min/max，套用到全部樣本（含 val），fit 出來的 min/scale 存進 `model_artifacts.preprocessing_state`；`training/inference.py` 載入舊模型推論新資料時，套用同一組保存的 min/scale，不是重新 fit，符合「推論凍結前處理狀態」的要求。`training_meta.preprocessing` 從 `"none"` 改成 `"minmax"`。
- [x] **卡片精簡**：`ModuleCardNode` 拿掉「雙擊編輯」提示文字，第二行改成「商品 時間框架 架構」（例如 `TX 15m lstm`），第三行維持「預測目標」（例如 `simple_return 2類`），卡片 `min-width`/padding 也一併縮小。
- [x] **畫布連線可以刪除**：`<ReactFlow>` 加 `onEdgeClick`，點擊連線直接移除，不用再找鍵盤快捷鍵。
- [x] **左下角縮放控制列深色主題**：React Flow 內建 `<Controls />` 預設亮色系，跟畫布深色背景對比度不夠，加了 CSS override（`.model-flow-canvas .react-flow__controls*`）。
- 縮小畫面後拖不動卡片（容易誤觸刪除鈕）：Hermes 回報目前看起來不明顯（可以先放大再拖），先不處理。

**測試**：後端（模型參數＋正規化）用真實 NAS DB + 真實 CUDA 訓練完整跑過端對端測試（訓練→存產物→推論、模型融合依賴鏈、Phase 1/2/3 血緣鏈全部重新驗證一次，含新抓到的 XGBoost early stopping bug 修復後的驗證），測完清理乾淨，資料庫確認無殘留。前端 `tsc --noEmit`、`npm run build`（生產環境建置）都乾淨通過；`/admin/model` 需要 Google 登入，沒辦法用真帳號測完整互動，下次 Hermes 登入後台可以肉眼確認這些 UI 修正。
