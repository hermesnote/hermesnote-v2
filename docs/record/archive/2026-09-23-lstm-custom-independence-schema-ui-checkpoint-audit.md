# 2026-09-23（六續）：LSTM 雙頭/Attention 獨立選擇修正、schema 驅動 UI、Checkpoint 表唯讀查證

補齊上一輪（五續）交付落差：LSTM 三軸真正獨立可組合＋雙向摘要正確性修正、完成 schema 驅動的通用表單、Agent API 手冊同步、唯讀查證 `model_training_checkpoints` 並更正先前的錯誤記錄。**本批尚未部署，正式庫變更另待核准。**

## 1. LSTM 雙頭與 Attention 獨立選擇

### 1a. 雙頭不再強制要求 Attention

`lstm_custom` 的方向、Attention、輸出頭三軸這輪起**完全獨立**——之前雙頭刻意要求 `attention` 必填的限制已移除。`training/registry/attention_dual.py::build_net()`（雙頭共用的網路建構函式，`lstm_attention_dual` 跟 `lstm_custom` 的雙頭路徑都靠它）擴充成 `attention=None` 時直接用序列摘要接共享層，不建立 attention 子模組；`lstm_custom.py::_train_dual()` 移除了原本的 400 擋。`lstm_attention_dual` 這個既有 key 的 payload 本來就強制必填 `attention`，這個擴充對它沒有行為影響。

### 1b. 雙向序列摘要正確性修正（真實 bug，不是風格選擇）

發現並修正一個真實的正確性問題：`build_net()`／`_build_single_net()`（`lstm_custom` 單頭路徑）原本用 `seq[:, -1, :]`（最後一個時間步的完整輸出向量）當序列摘要——單向時這剛好等於 LSTM 真正的最終 hidden state，**但雙向時不是**：`nn.LSTM` 反向方向是從序列尾端往回跑，`out[:, -1, :]` 這個位置對反向來說只處理過一個時間步，不是它看完整條序列後的真正摘要（那個在 `out[:, 0, :]` 或直接用 `h_n[1]`）。修正成用 `nn.LSTM` 回傳的 `h_n`（真正最終狀態）：單向直接用 `h_n[0]`，雙向用 `torch.cat([h_n[0], h_n[1]])`。

- 單向情況下 `h_n[0]` 在數學上就等於 `seq[:, -1, :]`（同一個值），所以這個修正對既有 `lstm_attention_dual`（固定單向）**逐位元組不變**——`test_attention_dual.py` 全部 37 項測試維持通過驗證了這一點。
- 雙向情況下，這個修正讓 `lstm_custom` 的雙向摘要跟既有、凍結的 `lstm_bidirectional`（維持原本較粗略的取法，沒有動它）**不再逐位元組相同**——這是刻意的分歧，不是回歸，已經用一個明確斷言「不相等」的測試釘住這個已知分歧，避免以後不小心又改回一樣。
- 用真正的 `nn.LSTM` 模組直接驗證了「反向方向的最終狀態在序列開頭，不是結尾」這個底層事實，也用 `lstm_custom` 實際的網路模組驗證了 forward() 算出來的結果真的對應 `h_n` 串接的版本、不對應 naive 取法的版本——不是只驗證「訓練還能跑」，是直接證明接法本身是對的。
- 「接法」存進 `model_config.sequence_summary`（人類可讀字串，例如「雙向（正反兩個方向的最終 hidden state 串接），再與 Attention context 向量串接」）；重載時實際靠 `bidirectional`／`attention` 兩個既有欄位重建網路，這個字串只是方便事後查閱，不是重建邏輯依賴的欄位。

### 測試

新增 6 項測試（雙頭無 attention 單向／雙向的形狀與保存重載一致性、單頭雙向無 attention 保存重載、雙向序列摘要正確性 2 項、既有分歧的明確斷言 1 項），移除 1 項不再成立的舊測試（原本斷言雙向跟 `lstm_bidirectional` 逐位元組相同）。`tests/test_lstm_custom.py` 共 16 項全過。

## 2. 完成 schema 驅動的通用 UI

前端新增真正通用的 `SchemaForm` 元件（`frontend/src/pages/admin/sections/ModelSettings.tsx`），讀 `GET /registry/architectures` 的 `params_schema`（`FieldSpec` 陣列：`name`/`type`/`required`/`default`/`min`/`max`/`enum_values`/`enum_values_if`/`component_registry`/`item_schema`/`visible_if`/`derived_from`）動態渲染表單，取代原本手刻、只服務 `lstm_custom` 一個 key 的表單擴充：

- 巢狀設定：`name` 用 dot path（例如 `heads.direction.rule.op`），`buildParamsFromSchema()` 依 path 組回巢狀 JSON，不需要為每種巢狀形狀另外寫解讀規則。
- 層清單：`array_of_object` 型別（`lstm_layers`）搭配 `item_schema` 描述每個陣列元素的欄位，通用的加一層／刪一層 UI。
- 條件顯示：`visible_if`／`required` 接受 `{"field","equals"}`／`{"field","not_equals"}` 簡單等式條件；`enum_values_if` 讓 `early_stopping.monitor` 依 `heads` 是不是 `"dual"` 切換合法值清單，前端不再另外硬寫一份 monitor 清單。
- 元件選擇：`component_ref` 型別（`attention`）選定型態後，動態讀**該型態自己的** `params_schema`（新增：`training/registry/attention.py` 的 `register()` 也補上 `params_schema`，`additive` 宣告了自己的 `dim` 欄位）渲染對應子欄位——新增一種 Attention 型態完全不用碰前端。
- 已知、如實標記的例外：`derived_from: "outcome_unit"` 這一種欄位（雙頭方向規則的 `threshold_unit`）值由外層根據選定 Label outcome 算好寫入，不是使用者自由填的一般 schema 欄位；這是目前唯一超出「純 schema 表達範圍」的特例。

`DualHeadEditor`（服務既有 `lstm_attention_dual`）**還原成上一輪之前的樣子**，移除了五續交付時加進去的 `custom` 模式分支——那個手刻擴充現在由更好的 `SchemaForm` 取代，`lstm_attention_dual` 的介面維持完全不變、不受這輪任何改動影響。

### 驗證

`npx tsc --noEmit`／`npm run build` 皆通過。**實際瀏覽器操作驗證需要人工登入**（本機開發環境沒有 Google 帳密，Code 無法自行完成登入）——集中列出的驗收步驟見 `docs/agent-api/training/api.md` 的「Schema 驅動 UI 的驗收步驟」一節。這輪額外做了一個可驗證的替代檢查：用一支獨立腳本重現前端 `condMet`／`buildParamsFromSchema` 的邏輯，對兩個代表組合（單頭+attention；雙向+無attention+雙頭）手動組出使用者會填的欄位值，產生的 JSON 直接餵給後端 `lstm_custom.validate_params()`，兩組都通過驗證——證明「schema 驅動表單會送出的 payload 形狀」跟「後端實際期待的形狀」一致，但這不是瀏覽器裡真的點過表單，兩者不能混為一談。

## 3. Agent API 手冊同步

`docs/agent-api/training/api.md`「LSTM 自訂組合」專節更新：八個組合的完整對照表（含哪些有舊版本可比對、哪些沒有）、無 Attention 雙頭 payload 範例、`params_schema` 的 `FieldSpec` 形狀說明、`sequence_summary` 產物欄位說明、Schema 驅動 UI 的驗收步驟。**本地文件（這份 `api.md`）這輪已更新完成；HA 讀取的是同一份檔案透過 NAS SMB 同步（既有既定流程，不是新機制）——這個同步動作本身尚未執行，等實際部署時一併處理，不是這輪工作範圍**，這裡明確區分「本地文件完成」與「HA 同步完成」是兩件事，避免之後有人誤以為文件更新＝HA 已經讀得到新版本。

## 4. 唯讀查證 Checkpoint 表——發現並更正前後矛盾，順帶抓到一個真實 bug

用唯讀查詢直接連正式 `hermesnote` DB（確認連到真的有資料的正式庫：`model_artifacts` 10 筆、`model_training_jobs` 12 筆，不是空庫或測試庫）查證：

- **`model_training_checkpoints` 表不存在**——先前記錄「這張表已套用到正式庫」是錯的。
- **`model_artifacts.weights_last` 欄位也不存在**——先前記錄「這個遷移已核准並套用」同樣是錯的。

**這是真正的前後矛盾，已更正**：`docs/agent-api/training/api.md` 補一段「六續更正」明確寫出查證結果與更正內容，不是含糊帶過。

**順帶抓到一個真實 bug**：`training/artifacts.py::save_artifacts_for_job()` 原本無條件把 `weights_last` 寫進 `INSERT` 陳述式——欄位不存在的情況下（也就是正式庫目前的實際狀態），**任何**架構（不限雙頭）呼叫這支函式都會直接因為 SQL 引用不存在的欄位整個失敗，代表現在部署現在的程式碼，第一次訓練完成、要保存產物的那一刻就會炸掉。修正成跟既有 `load_artifact()` 同一套防呆：先查 `information_schema.columns` 有沒有這個欄位，沒有就把 `weights_last` 從 `INSERT` 的欄位清單與 `ON CONFLICT DO UPDATE` 子句整個拿掉；遷移套用後自動接上，不需要再改程式碼。新增 `tests/test_artifacts_weights_last_compat.py`（2 項），驗證欄位存在／不存在兩種情況都正確運作。

**處理範圍界定**：
- `model_training_checkpoints` 的建表語句因為屬於已撤回的 scrub 功能、且從未套用過，這輪已把它從 `model_init.sql` 與 `migrations/2026-09-23_scrub_window_checkpoints.sql`（整檔刪除）移除——這是移除從未生效的 SQL 檔案本身，不是操作正式資料庫。
- `weights_last` 的 `ALTER TABLE` 語句**保留**在 `model_init.sql`（這是要保留的研究功能，只是遷移尚未實際套用），連同前述程式碼防呆一起，讓「遷移套用前」與「遷移套用後」兩種正式庫狀態都能正確運作。
- 是否要／何時要真的對正式庫執行 `weights_last` 遷移，**留給 Hermes 決定與核准**，這輪沒有觸碰正式資料庫本身。

## 測試與驗證彙總

`tests/test_lstm_custom.py`（16 項，+6/-1）、`tests/test_artifacts_weights_last_compat.py`（新增 2 項）。全部測試（含既有 `test_attention_dual`／`test_named_outputs`／`test_available_time`／`test_phases_dual`／`test_architecture_families`／`test_evaluation_and_epochs`／`test_training_control`）共 **187 項全過**；`contract_harness.py capture` 確認訓練數值路徑仍是零 run-to-run spread。前端 `tsc --noEmit`／`npm run build` 通過。正式庫唯讀查詢（`SELECT` 與 `information_schema` 查詢，未寫入）確認上述兩項欄位/表狀態。

## 已知限制（如實保留）

- Schema 驅動 UI 沒有經過瀏覽器實際登入點擊驗證，只驗證到型別檢查、建置，以及 payload 形狀跟後端驗證規則的邏輯交叉核對——集中列出的驗收步驟已交給 Hermes，需要人工登入才能執行。
- `derived_from: "outcome_unit"` 是目前唯一超出純 schema 表達範圍、需要外層特殊處理的欄位類型；之後如果出現更多這類需要外部情境資料的欄位，這個機制可能需要擴充。
- `model_artifacts.weights_last`／`model_training_checkpoints`（已移除）兩個遷移的正式庫狀態這輪只查證、不變更；`weights_last` 何時套用留給 Hermes 決定。
- Attention 仍只支援零或一個，多個 Attention 的組合介面（`combine` 的 `sequential_refine`／`parallel_concat`）只定義了列舉值，沒有實作執行迴圈——跟五續交付時的範圍一致，這輪沒有再擴大。
- 混合模型仍僅支援節點輸出串接（凍結輸出當下游特徵），不支援聯合訓練或其他融合結構。
