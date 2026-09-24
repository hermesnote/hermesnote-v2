# 2026-09-23（五續）：撤回 scrub 排程／暫停／硬中斷恢復；交付 `lstm_custom` 可組合框架

Hermes 明確要求撤回本日稍早交付的 scrub 維護閘門排程、暫停機制、硬中斷恢復新增功能，保留研究功能；同時確認實作「通用模型組裝框架」設計（本輪以 LSTM 落地），**一次收斂交付**。**本批尚未部署。**

## 1. 精準撤回：scrub／暫停／硬中斷恢復

撤回本日稍早新增的全部程式與測試，保留研究功能不動：

| 檔案 | 動作 |
|---|---|
| `training/scrub_window.py`／`training/pause.py`／`training/checkpoints.py` | 整個檔案刪除 |
| `tests/test_scrub_window.py`／`tests/test_hard_kill_recovery.py`／`tests/_hard_kill_*_harness.py` | 整批刪除 |
| `training/artifacts.py` | 移除 `list_completed_node_ids()` |
| `training/inference_store.py` | 移除 `count_predictions()` |
| `training/job_store.py` | 移除 `fetch_resumable_paused`／`mark_paused`／`fetch_orphaned_running` |
| `training/graph.py` | `run_graph()` 移除 `should_pause`／`resume_checkpoints`／`job_id`／`completed_node_ids`／`on_node_complete`，還原成 `run_graph(graph_spec, data_loader, on_epoch, on_preview)` |
| `training/inference.py` | `run_inference_chunked()` 移除 `should_pause`／`resume_from_index`／`InferencePaused` |
| `training/worker.py` | 移除 `_reap_orphaned_running`、暫停處理、`should_pause_for_scrub` 檢查、`on_node_complete`，還原成單純跑到底、`mark_done`／`mark_failed` 兩種結局 |
| `training/registry/architectures.py`／`attention_dual.py` | 移除 `resume_from`／`should_pause` 參數與 checkpoint 打包／還原邏輯（含先前的 CUDA RNG 修正，因為整個機制都撤了） |
| `training/registry/architectures.py`（XGBoost） | **還原成原生 `early_stopping_rounds=`**（`patience>0` 時傳 `patience`，否則 `None`），移除手動追蹤（`_es_state`／`_check_early_stop`／`_pause_state`）——這是當初為了 resume 續跑才做的替換，撤回 resume 後沒有存在理由；`patience=0` 關閉、正整數表示容忍輪數的既有語意不變 |
| `routers/model_training.py` | `/cancel` 只接受 `pending`／`running`（撤回 `paused`） |
| 前端 `ModelTrainingPage.tsx`／`ModelSettings.tsx` | `status` 型別、`STATUS_LABEL`、所有條件判斷移除 `paused` |
| `docs/agent-api/training/api.md` | 移除「週日 scrub 維護閘門」整節，補一段撤回說明 |

**保留（研究功能，逐項確認未受影響）**：`training/registry/attention.py`；`architectures.py` 的 family/variant metadata、`lstm_bidirectional`；`attention_dual.py` 的雙頭 LSTM、`early_stopping.enabled` 開關、best/last 雙權重（`weights_last`）；`training/evaluation.py`；`model_artifacts.weights_last` migration；`/infer` 的 `use_weights`。

**未處理、待 Hermes 決定**：`model_training_checkpoints` 表（`2026-09-23_scrub_window_checkpoints.sql` 建立，已套用到正式庫）——撤回程式碼後沒有任何程式在用，是否要另下 `DROP TABLE` migration，這次沒有擅自決定，`model_init.sql` 裡的定義也維持原樣未動（避免正式庫遷移的擅自決策）。

## 2. `lstm_custom`：真正可組合的 LSTM 入口

新增 `training/registry/lstm_custom.py`（family=`lstm`、variant=`custom`），方向（單向／雙向）、Attention（無／單一登記型態）、輸出頭（單頭／雙頭）三個獨立軸同一個 payload 就能組合，不需要每個組合各開一個 key。既有四個 key（`lstm`／`lstm_bidirectional`／`lstm_attention_dual`／`xgboost`）**完全凍結**，payload 形狀、驗證訊息、產物格式逐位不變。

- 單頭路徑：新的 `_build_single_net`／`_train_single`，結構跟既有 `lstm` 一致（早停/best-last/評估邏輯對齊），差異只在方向與 Attention 都可設定。
- 雙頭路徑：直接沿用 `attention_dual.py` 的 `build_net`／`_evaluate`／`_make_predictors`——`build_net` 補上 `bidirectional` 支援（`cfg` 沒有這個 key 時預設 `False`，`lstm_attention_dual` 的既有 cfg 形狀完全不受影響），不重寫一份幾乎一樣的邏輯。
- **Attention 組合限制程式化落地**：`validate_params` 明確檢查 `attention` 是不是清單，是的話 400 並說明「本版本尚未支援 Attention 組合」；還檢查選用型態的 `output_kind` 是不是 `context_vector`（目前唯一登記的 `additive` 是），不是的話一樣拒絕——不是只有文件宣稱，是真的在提交前擋下來。
- 雙頭時 `attention` 不能是 `null`（400，沿用 `lstm_attention_dual` 的既有設計假設，這個組合本輪不開放，不是技術做不到）。
- `capabilities`（三態：`fixed_on`／`fixed_off`／`configurable`）、`slots`、`params_schema` 掛在 `ARCHITECTURE_META` 底下，四個既有 key 也補上對應的三態能力宣告（皆非 configurable 或部分固定）——純描述性 metadata，不驅動任何驗證邏輯，後端 `validate_params` 永遠是最終把關者。

### 共用訓練控制

新增 `training/registry/training_control.py`：`parse_legacy_patience`（`lstm`／`xgboost` 的扁平 `patience`，`0`＝關閉的既有語意逐位不變）／`parse_structured`（`lstm_attention_dual`／`lstm_custom` 的結構化 `early_stopping`，純搬移既有驗證邏輯，規則不變）。`lstm`／`attention_dual.py` 的既有驗證邏輯改呼叫共用模組，payload 契約完全不變，只是內部有唯一的規則來源。

### Attention registry 擴充

`attention.py` 的 `register()` 新增 `output_kind`（`"context_vector"`／`"sequence"`，宣告這個元件是「序列聚合成向量」還是「序列進序列出」——兩種本質不同的介面形狀，只有同一種才有明確定義的組合方式）與 `slot_compatibility`（`[family, slot_name]` 配對，比舊有的 `compatible_families` 更精確）；`compatible_families` 保留不動，舊消費者不受影響。

## 3. 相容驗收

- **有舊版本等效組態**（單向單頭無 attention ≈ `lstm`；雙向單頭無 attention ≈ `lstm_bidirectional`）：新增測試對同一組資料、同一個隨機種子分別跑兩個 key，**逐位元組比對**最終權重（用「按參數建立順序逐一比對 tensor 數值」而不是比較序列化 bytes，因為兩邊 state_dict 的 key 命名習慣不同——`lstm` 是單一 `nn.LSTM`，`lstm_custom` 是 `ModuleList`，命名不同不代表數值不同）與 `final_metrics`，結果完全一致。
- **沒有舊版本的新組合**（單向+attention+單頭；雙向+attention+雙頭）：驗證形狀（`output_specs`／`model_config` 正確）、訓練完成、保存重載一致性（存進產物再讀回，對同批輸入預測結果一致）、推論一致性——不跟舊版比數字。
- 額外新增一項雙頭（無方向）+attention 對 `lstm_attention_dual` 的逐位元組比對，確認 `lstm_custom` 的雙頭路徑本身沒有引入偏差，不是只靠 `attention_dual` 自己的既有測試把關。
- Phase 1/2/3 契約：`lstm_custom` 走同一套 `validate_graph_spec`，不開特例（既有測試涵蓋的路徑不受影響）。

## 4. UI／API 一致性

`/registry/architectures`／`/registry/attention_types` 已經是動態 `list_available()`，新增欄位自動透過既有端點回傳，**router 本身不需要改**。前端 `ModelSettings.tsx`：`groupArchitecturesByFamily` 納入 `lstm_custom`；既有 `DualHeadEditor` 擴充成同時服務 `lstm_attention_dual`（介面完全不變）與 `lstm_custom`（新增方向開關、輸出頭切換、Attention 可選「不使用」、依單頭/雙頭切換監控指標選項與必要欄位顯示）——重用同一組已驗證過的表單元件（lstm_layers 陣列編輯、結構化 early stopping、optimizer 欄位），**沒有另外做一套通用 JSON-schema 驅動的表單引擎**：這是對「新增家族免改前端」承諾範圍的如實限定——`params_schema` 這份 metadata 已經在 API 裡備妥給未來的通用渲染器用，但這輪前端仍是針對 `lstm_custom` 手刻的專用表單，不是自動生成的。

## 測試與驗證

新增 `tests/test_lstm_custom.py`（12 項：驗證規則 5 項、逐位元組等價 2 項、新組合形狀/保存重載 2 項＋額外雙頭等價 1 項、`run_graph` 端到端 2 項）、`tests/test_training_control.py`（7 項）。既有 `test_architecture_families.py` 一項更新（LSTM family 從三個 variant 改四個）。全部測試（含既有 `test_named_outputs`／`test_attention_dual`／`test_available_time`／`test_phases_dual`／`test_architecture_families`／`test_evaluation_and_epochs`）共 **181 項全過**；`contract_harness.py capture` 確認訓練數值路徑仍是零 run-to-run spread（決定性不變）。前端 `npx tsc --noEmit` 與 `npm run build` 皆通過。

## 已知限制（如實保留）

- `model_training_checkpoints` 表是否要下 `DROP TABLE` migration，留給 Hermes 決定，這輪沒有處理。
- Attention 本版本只支援零或一個，多個 Attention 的組合介面（`combine` 的 `sequential_refine`／`parallel_concat`）只定義了列舉值，沒有實作執行迴圈。
- `lstm_custom` 雙頭必須有 Attention（`attention=null` 400），這是刻意的範圍限制，不是技術限制——沿用 `lstm_attention_dual` 的既有設計假設，避免在完全沒有請求的情況下擴大驗證面。
- 混合模型仍僅支援節點輸出串接（凍結輸出當下游特徵），不支援聯合訓練或其他融合結構。
- 前端表單是針對 `lstm_custom` 手刻的專用表單，不是讀 `params_schema` 自動生成的通用渲染器——這份 metadata 目前只有 API 層備妥，等真的有第二個需要動態表單的新家族出現時再考慮做成通用引擎。
