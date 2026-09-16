# 2026-09-11 NAS 訓練異常查核紀錄

## 目的與操作範圍

- 目的：確認 Hermes Note V2 訓練管線在壓力下能否完整運行，釐清本機成功、NAS 停滯或被 OOM kill 的原因；本次不評估模型準確度，也不接手開發。
- 操作者：Codex。時間均為 Asia/Taipei（UTC+8）。
- 本機：專案程式碼唯讀；使用者另行授權新增、持續更新本紀錄。
- NAS 與網站：只讀取、查詢，不修改設定、不新增或刪除資料、不部署、不重啟容器、不送出訓練、不儲存模組。
- 瀏覽器僅限使用者指定的三個分頁：TrueNAS 命令列、`https://hermesnote.com/model`、`https://hermesnote.com/admin/model`。其他分頁不操作。
- 13:04 起，使用者授權在指定 NAS Shell 執行唯讀查詢，並試操作後台設定表單；可填未提交草稿，不儲存模組、不建立訓練任務。先前「先看不動作」階段已結束。
- 若 TrueNAS session 登出，使用者已授權使用瀏覽器預存帳密重新登入同一 NAS；不把帳密記錄到本文。
- 若後續在 NAS 查詢，依使用者建議，在需要權限的唯讀命令前使用 `sudo`。
- 先前操作沒有逐項記錄精確時間，下表明確標成「補記」；時間欄是補記時間，不冒充原始執行時間。

## 使用者提供的測試參考設定

以下為對話中兩張截圖的原始設定；後續口述覆蓋值見下節。資料日期、商品／時間框架後來已在草稿確認，phase 與 split_strategy 仍未確認。未提交或建立任何實驗。

| 類別 | 項目 | 截圖設定 |
| --- | --- | --- |
| Feature | OHLCV | `ohlcv`，無參數 |
| Feature | RSI | timeperiod=14 |
| Feature | MACD | fastperiod=12、slowperiod=26、signalperiod=9 |
| Feature | STOCH | fastk_period=5、slowk_period=3、slowk_matype=0、slowd_period=3、slowd_matype=0 |
| Feature | ATR | timeperiod=14 |
| Feature | BBANDS | timeperiod=5、nbdevup=2、nbdevdn=2、matype=0 |
| Feature | ADX | timeperiod=14 |
| Target / Label | outcome / horizon | simple_return / 1 |
| Target / Label | labeling_rule | fixed_threshold |
| Target / Label | n_classes / threshold_pct | 2（漲／不漲）/ 0 |
| Model | architecture / output | lstm / 類別 |
| Model | window / val_ratio | 60 / 0.2 |
| Model | epochs / patience | 300 / 30 |
| Model | units / layers / dense | 64 / 2 / 64 |
| Model | bidirectional | 未勾選（false） |
| Model | dropout / head_dropout | 0.2 / 0.2 |
| Model | l2_lambda / class_weight | 0 / none |
| Model | batch_size / learning_rate | 64 / 0.001 |

七項特徵不等於七個數值欄位。依目前 `backend/training/registry/features.py` 的輸出方式與這組指標，預期為 OHLCV 5 + RSI 1 + MACD 3 + STOCH 2 + ATR 1 + BBANDS 3 + ADX 1 = **16 欄**；仍需用實際任務的輸入 shape 核實。

### 13:04 使用者更新的測試設定

- 商品 TX、時間框架 1 分鐘；開始日期保留表單原值 **2011-01-03**，結束日期固定 **2025-12-31**，兩者已在 UI 核實。
- epochs=100、units=128、dense=128、batch_size=128、patience=0（不做 early stopping）。其餘特徵、Target／Label 及模型設定依截圖；上述值覆蓋截圖中的舊值。
- 本輪只試填草稿與查核權限，未授權建立或執行壓力測試。

## 查核流水帳

| 時間（台北） | 動作 | 目的 | 結果／證據 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-09-11 12:59（補記） | 唯讀檢查專案結構、compose、Dockerfile、部署腳本、訓練程式與歷史文件 | 確認網站與訓練的架構及本機／NAS 差異 | compose 定義 frontend、backend、training 三個服務；training 為單一長駐 worker，以 DB 任務佇列取得工作。沒有執行部署腳本或專案服務 | 核對 NAS 實際執行的 image／程式版本與容器配置 |
| 2026-09-11 12:59（補記） | 閱讀 worker 與 job_store 狀態流程 | 判斷畫面 running 能否證明訓練仍在執行 | `mark_running` 在資料載入前呼叫；worker 只撿 pending；未見重啟後自動處理遺留 running 任務的機制；compose 為 restart: always | 對照失敗任務與容器重啟／OOM 的時間，確認是否為遺留狀態 |
| 2026-09-11 12:59（補記） | 閱讀 GPU 接單門檻 | 找出 CPU／GPU 不運算時的等待條件 | `gpu_is_busy()` 檢查第一張 GPU 的 memory.used；大於 2000 MB 就延後 30 秒，與 GPU 使用率無關；此條件在 mark_running 前執行 | 對照任務是否 pending，以及 log 是否有 GPU busy |
| 2026-09-11 12:59（補記） | 閱讀資料查詢與視窗／切分流程 | 檢查 GPU 開始訓練之前的 RAM 配置 | quotes 仍以 `conn.fetch` 全量取得資料後建 DataFrame；graph 改用 sliding_window_view，但傳入模型時 `X[train_idx]`、`X[val_idx]` 仍會產生完整子集副本 | 取得實際資料筆數、特徵維度、RAM 峰值與容器限制，避免只憑總 RAM 判斷 |
| 2026-09-11 12:59（補記） | 在本機用 NumPy 小型假陣列檢查記憶體共享；查閱 NumPy 官方文件 | 驗證切分是否複製資料，不連資料庫、不跑訓練 | 使用現有 .venv Python 並加 `-B`；本機 NumPy 2.4.6。切分陣列不與視窗共享記憶體；該小例子已為 C-contiguous，ascontiguousarray 沒有再次複製。因此不能把額外一整份 contiguous 副本當成已證實的開銷 | 以真實資料 layout／版本核對；目前只確認切分副本存在 |
| 2026-09-11 12:59（補記） | 檢視 training Dockerfile 與歷史測試敘述 | 評估「GPU 架構不相容」是否已有證據 | 本機檔案釘住 torch 2.10.0、cu128，但不能證明 NAS 執行中的 image 已更新。註解中的 GPU 歸因尚缺 NAS 實際錯誤證據；本機成功不足以排除程式記憶體配置問題 | 先取得實際版本、退出原因與 kernel 紀錄，再判斷 CUDA 相容性 |
| 2026-09-11 12:59（補記） | 嘗試唯讀取得 Git status／近期 log | 建立本機版本基準 | Git 因 dubious ownership 拒絕執行；未修改 safe.directory 或其他 Git 設定，未取得 commit 基準 | 如需要版本比較，可另行唯讀核對檔案內容／雜湊 |
| 2026-09-11 12:59（補記） | 在使用者提出瀏覽器途徑後，確認可用分頁並讀取已登入的 TrueNAS 看板 | 驗證能否直接取得 NAS 狀態 | 成功讀取 TrueNAS 25.10.4 看板；CPU 型號為 AMD Ryzen 7 8845HS；當時約 30.6 GiB 總記憶體、14.2 GiB free、1.9 GiB ZFS cache、14.5 GiB 服務用量。只讀取看板，尚未下 NAS 指令；未操作其他網站分頁 | 此為當下快照，需查失敗當時的紀錄，不能當成 OOM 根因 |
| 2026-09-11 12:59（補記） | 接收「僅限三個分頁、唯讀、先看不動作」限制與第一張截圖 | 固定操作範圍與壓力測試目的 | 記下七項特徵；沒有點加訓練模組、沒有填表、沒有儲存或送訓練 | 接收剩餘參數並記錄，不自行猜測未提供欄位 |
| 2026-09-11 12:59 | 接收第二張截圖；確認 docs/temp；新增並讀回本紀錄 | 保存 Target／Model 參數與可追溯查核歷程 | 已確認文件存在、UTF-8 中文可讀，包含兩張截圖設定、已查證事實、未驗證假說及操作限制；新增前確認目標資料夾存在，未見該資料夾或 docs 的 AGENTS.md | 後續每項查核同步填寫動作、目的、結果、下一步 |
| 2026-09-11 13:04 | 在指定 TrueNAS Shell 執行 `sudo -n docker ps -a --filter name=hermesnote`（附格式化輸出） | 確認網頁 Shell 與 sudo 查詢可用 | 成功，無密碼或權限阻擋；frontend／backend／training 皆 Up 約 17 分鐘。另有舊 hermesnote 容器 Exited (127) 約四個月，未操作它。終端輸入的 tab 分隔符未保留，欄位黏在一起，後續改用普通空格 | 查 training 容器的完整狀態與上限 |
| 2026-09-11 13:05 | `sudo -n docker inspect hermesnote-training --format ...` | 核對當前容器的退出／重啟／OOM 狀態及 RAM 限制 | running=true、OOMKilled=false、ExitCode=0、RestartCount=0、Memory=0、MemorySwap=0；StartedAt=2026-09-11T04:47:17.28532568Z（台北 12:47:17）；image sha256:ae2f46abc4aa6f27a1a0a003ff00a88427ae3061df24e370b501c23cc7164e5a。Memory=0 表示 Docker 未設此容器硬上限，不代表主機可用 RAM 無限 | 當前容器狀態不代表較早已替換的容器；查 logs／kernel 歷史 |
| 2026-09-11 13:05 | 讀取指定訓練後台頁面 | 確認可操作及是否有現成失敗任務 | 已登入；0 個模組、建立訓練按鈕停用，頁面顯示「尚無訓練紀錄」 | 打開設定表單，僅試填草稿 |
| 2026-09-11 13:06–13:07 | 讀取 training 最近 25 行 timestamped logs；以 journalctl 查今日 kernel 的 OOM／Xid 訊息 | 取得真正的退出證據，避免只憑目前容器狀態判斷 | 當前 training log 只有 12:47:27 worker started/polling 訊息。kernel 則有 11:12:00 與 12:29:44 兩次 global_oom、CONSTRAINT_NONE，均殺掉容器內 Python。11:12 PID=3350232、anon-rss=14996124 kB；12:29 PID=3598470、anon-rss=14991924 kB，約 14.3 GiB 匿名 RAM。兩次 task_memcg 容器 ID 前綴分別 b5b7586a6a95、991f69233a0b。postgres／hermes 是觸發 OOM 配置請求的程序名稱，不能直接視為耗盡 RAM 的唯一元凶 | 查這兩個 ID 能否映射回容器名稱；仍需對上特定訓練任務 |
| 2026-09-11 13:06–13:07 | 試填未提交模組草稿 | 確認後台 UI 可操作及保留日期原值 | TX、1m、開始 2011-01-03、結束 2025-12-31 已在 UI 確認；epochs=100、units/dense/batch_size=128、patience=0。Target 維持預設；逐一選擇截圖特徵中 | 完成七項特徵檢查；不按儲存模組或建立訓練 |
| 2026-09-11 13:07 | 列出容器 ID／名稱，並對 OOM 紀錄兩個前綴執行 docker inspect | 建立 kernel 紀錄到容器名稱的對應 | `991f69233a0b` 與 `b5b7586a6a95` 均回覆 No such object；目前無法以 inspect 還原名稱。這是舊物件已不可查的證據缺口，不是 sudo 權限問題 | 查保留的 Docker OOM 事件作補充 |
| 2026-09-11 13:07–13:08 | 查 10:30–13:00 的 Docker OOM 事件 | 嘗試補回已不存在容器的事件屬性 | 查詢成功但無輸出；不能以此否定 kernel 已記錄的 OOM，也不能據此斷言事件為何未保留 | 保留缺口，不宣稱兩次 Python OOM 已精確對應訓練 job |
| 2026-09-11 13:08 | `sudo -n nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv` 及 `sudo -n free -h` | 核實 GPU、RAM、swap 當下狀態與查詢權限 | RTX 5060 Ti、driver 570.172.08、VRAM 16311 MiB，當時 used=0 MiB、utilization=0%；free 顯示 Mem total=30Gi、used=16Gi、free=13Gi、available=14Gi，Swap total/used/free 均 0B。皆查詢成功 | 目前 GPU 不忙；這些是空閒時快照，不能當作失敗時的負載分布 |
| 2026-09-11 13:08 | 讀取指定 `/model` 前台 | 查看即時頁面是否有可供觀察的現成任務 | 顯示「找不到這筆訓練紀錄」，URL 未帶 job 參數，各指標為空；未更改網址或啟動任務 | 沒有任務時不能驗證即時推送是否正常 |
| 2026-09-11 13:08 | 完成七項特徵草稿並讀取完整 UI 核對 | 確認可按使用者指定設定操作表單 | OHLCV、RSI14、MACD12/26/9、STOCH5/3/0/3/0、ATR14、BBANDS5/2/2/0、ADX14 全部選定；Target 與其餘模型欄位符合截圖，口述覆蓋值均正確。仍為 0 個已儲存模組、建立訓練按鈕停用；保留未提交草稿供使用者查看 | 本轮操作驗證完成；不儲存、不送任務，更新 MD |

### 17:19 後續核對：回應本機與 NAS 差異的質疑

| 時間（台北） | 動作 | 目的 | 結果／證據 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-09-11 17:19 | 重讀目前本機資料讀取、視窗切分與模型 tensor 路徑，讀取修改時間 | 確認 Claude 後續版本是否已改掉先前指出的配置 | `graph.py:260` 仍以 `X[train_idx]`／`X[val_idx]` 傳入模型；architectures.py 仍用 from_numpy/ascontiguousarray。graph 最後修改 04:20:14、architectures 為前一日 22:25:36、worker 12:41:22、quotes 12:37:55。僅讀取，未修改程式 | 核對容器內實際檔案，而不是只相信本機部署設定 |
| 2026-09-11 17:19 | 本機 `Get-CimInstance Win32_PageFileUsage`、`Win32_OperatingSystem` | 查 Windows 分頁檔是否可能解釋同為 32 GB 的差異 | 兩項皆拒絕存取，沒有取得分頁檔／當前虛擬記憶體數值；依使用者「先記錄權限問題、一輪後再討論」要求記下，未改權限。不能把分頁檔當成已確認原因 | 若後續需要，可由使用者提供工作管理員的已認可記憶體與分頁檔資訊，或授權本機查詢 |
| 2026-09-11 17:19–17:20 | 本機 Get-FileHash 與 NAS `sudo -n docker exec hermesnote-training sha256sum ...` | 排除關鍵訓練檔案未部署或版本不同 | graph、architectures、worker、quotes 四個檔案 SHA256 全部一致，見下表；NAS 查詢成功。這證明目前磁碟上四個檔案相同，未證明早上失敗時也是同版本，也未比較所有套件或資料量 | 下一輪以當前版本、同一筆設定取得實際執行與資源時間軸 |

| 檔案（本機 backend 與 NAS /app 相對路徑） | 兩端相同的 SHA256 |
| --- | --- |
| training/graph.py | `85a172fda1be07b7205ef1701aac11e188393af0e7e35390736faf92f5a73533` |
| training/registry/architectures.py | `746d4ebe4e2043eb6f59d71be2ac0b6d093eb2d69a7a3e1f525bc9493e50148b` |
| training/worker.py | `8e75330aed827ba5519a5d794f6e2562823e64536255f4821ac06b3e08b661ab` |
| services/quotes.py | `74acdcbea4f7a689104059576c52fc153db948a65bf95e09e6d20a215201d5ad` |

**釐清證據強度：** 使用者質疑是合理的。10.41 GiB 只是特定假設下兩個輸入陣列的容量，不是程序的實測峰值，更不足以解釋為何本機成功、NAS 失敗。14.3 GiB 是兩次被殺 Python 的 anon-rss，不是當時整台主機總用量。空閒看板的服務／free／cache 也不是失敗當時的數值。需要量到同一次執行中 Python、資料庫／其他程序與主機的變化，才能判斷差異；不可假定 NAS 謊報容量，也不可直接歸因 ZFS 或 GPU。

**建議下一個實驗（尚未執行）：** 使用已核對的目前版本與使用者指定草稿，經授權建立一筆 NAS 測試任務，同時記錄 job ID、容器 ID、啟動時間、容器 RAM／CPU、主機 available RAM、GPU 與 worker logs；首次中斷即核對 kernel 和重啟狀態，不連續重送。現有 log 只到接單／epoch 粒度，若再次在第一個 epoch 前中斷，資源記錄能縮小範圍但未必能定位到某一行，屆時再提出必要且最小的觀測程式修改。此處只是建議，沒有送任務或改程式。

### 17:25 新增觀察：曾到 epoch 6，但 GPU／VRAM 幾乎未使用

| 時間（台北） | 動作 | 目的 | 結果／證據 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-09-11 17:25 | 記錄使用者回憶：前一次訓練 epoch 曾跑到 6，當時持續查看 GPU status，GPU 幾乎閒置、VRAM 幾乎未佔用 | 修正只聚焦訓練前配置／OOM 的排查方向 | 此為使用者現場觀察，Codex 先前未掌握，尚無同一 job 的 log／裝置紀錄可獨立核實。不能將這次觀察直接與 11:12／12:29 的 OOM 合併為同一次事故 | 下一次由使用者啟動測試時，優先對齊同一 job 的 epoch、實際訓練裝置、GPU 監測來源與時間 |
| 2026-09-11 17:25 | 唯讀重查目前 LSTM 裝置選擇與進度 callback | 判斷是否可能 GPU 沒動但 epoch 照樣增加 | architectures.py:106 在 torch.cuda.is_available() 為 false 時自動選 CPU；每個 epoch 完成訓練與驗證才呼叫 on_epoch。因此 CPU 路徑可以正常產生 epoch 進度，epoch 增加本身不證明 GPU 有執行。若是同一 job 真正產生的 1–6 進度，該次執行已通過初始資料準備與模型建立，不能描述成從未進入訓練 | 驗證 CPU 路徑、監測是否對到同一時段／裝置，以及畫面是否為當次任務進度；不預先斷言原因 |

使用者已決定先與 Claude 確認修改，必要時自行部署與送一筆訓練；如仍有問題，再由 Codex 接手指定三個分頁唯讀查核。本次沒有代送任務、部署、改程式或操作 NAS；只重讀本機程式並更新此紀錄。

## 截至 17:19 的判斷與限制（歷史快照）

**18:02 更新：** 下列容量推算與容器狀態保留作為當時證據。最新本機 LSTM 已於 17:56／17:57 改成 LazyWindowed 與 batch tensor，不能再用舊版全量視窗副本描述其目前行為；NAS 是否部署尚未核實。最新查核與擴充提案見本文後段。

1. **已獨立查到兩次主機 global OOM 殺掉容器內 Python，尚未精確對上訓練 job。** 11:12、12:29 的 kernel 原始紀錄證明 RAM OOM 確實發生；舊容器已不可 inspect、事件查詢無輸出、後台無歷史紀錄，因此不能宣稱根因已完整定位到特定程式行。
2. **RAM 與 VRAM 要分開看。** 視窗切分的 RAM 副本在呼叫模型前就配置，縮小 batch_size 不會縮小這一塊。
3. **修正先前示例的適用範圍：** 對話曾以「291 萬筆、60 視窗、7 欄 float32」示例估算約 4.55 GiB，並非實測值。使用者截圖預期為 16 欄；若仍假設約 291 萬個視窗樣本，train+val 特徵副本合計約 **10.41 GiB**（樣本數 × 60 × 16 × 4 bytes），尚未包含原始查詢結果、DataFrame、特徵／對齊中間陣列、模型或其他服務。草稿日期已確認，但實際样本數與失敗任務設定仍未確認，這個估算不能作為特定程式行觸發 OOM 的證明。
4. **沒有進行壓力測試。** 僅完成未提交的後台草稿；未儲存模組、未建立任務、未執行 GPU 計算。這輪證明介面與查詢權限可用，尚未證明訓練可成功完成。
5. **當前容器健康不會抹掉舊事故。** training 於 12:47 啟動、重啟 0 次，晚於兩次 OOM；目前 OOMKilled=false 不能排除它的舊版本曾被殺。
6. **NAS 唯讀權限目前足夠；本機 CIM 查詢有權限缺口。** NAS sudo 的 Docker、journalctl、nvidia-smi、free 查詢皆成功，沒有使用或讀取預存密碼。本機分頁檔／OS 記憶體查詢於 17:19 被拒絕。主要證據缺口仍是歷史容器／任務對照，以及新壓力測試尚未執行。

## 後續查核狀態

| 順序 | 查核 | 目的與預期產出 |
| --- | --- | --- |
| 1 | NAS Shell、前後台、表單操作 | 已完成；舊訓練紀錄目前無法由前後台取得；草稿留在後台 |
| 2 | Docker 當前狀態、上限、最後 logs | 已完成，無權限阻擋；舊容器不存在 |
| 3 | kernel OOM、GPU、RAM、swap | 已完成第一輪；兩次 global OOM，当前約 14Gi available、無 swap、GPU 空閒 |
| 4 | 實際套件／程式版本、資料量與 shape | 已完成四個核心檔案 SHA256 比對，兩端相同；套件、實際 shape 與成功／失敗當時版本仍待核對 |
| 5 | 新壓力測試及階段資源觀測 | 尚未執行；必須先有使用者授權建立任務。若後續執行，應保留同一 job／容器／時間軸，避免再次失去對照證據；程式修正與部署仍未授權 |

## 本輪關鍵 NAS 查詢（均已執行，僅供回溯）

```sh
sudo -n docker logs --timestamps --tail 25 hermesnote-training
sudo -n journalctl -k --since '2026-09-11 00:00:00' --no-pager --grep='oom-kill|Out of memory|Killed process|NVRM: Xid' -n 20
sudo -n docker inspect --format '{{.Id}} {{.Name}}' 991f69233a0b b5b7586a6a95
sudo -n docker events --since '2026-09-11T10:30:00+08:00' --until '2026-09-11T13:00:00+08:00' --filter event=oom --format '{{json .}}'
sudo -n nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
sudo -n free -h
```

## 主要證據位置

- `backend/training/worker.py`：GPU busy 門檻、接單、mark_running、資料載入、epoch log。
- `backend/training/job_store.py`：pending 取件、任務狀態更新。
- `backend/training/graph.py`：時間戳對齊、視窗 view、train／val 索引副本。
- `backend/training/registry/architectures.py`：LSTM tensor 建立、訓練／驗證批次、推論載入。
- `backend/training/registry/features.py`：OHLCV 與多輸出指標的特徵欄位。
- `backend/services/quotes.py`：全量 SQL 查詢與 DataFrame 建立。
- `docker-compose.yml`、`backend/training/Dockerfile`、`deploy.ps1`：本機部署設定快照，非 NAS 實際狀態證明。
- [NumPy 官方：Advanced indexing 會產生副本](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)。
- 使用者在本對話提供的兩張截圖：特徵設定、Target／Model 設定。

## 18:02 提案查核補記：最新程式與 V1 參考

| 時間（台北） | 動作 | 目的 | 結果／證據 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-09-11 18:02（本輪彙整） | 唯讀檢查目前 V2 訓練程式與既有 Phase 協定 | 讓擴充提案符合 Claude 最新修改，而非沿用舊版結論 | graph.py 修改時間 17:56:49，已新增 LazyWindowed，LSTM 使用索引按需組窗；architectures.py 修改時間 17:57:21，按 batch 轉 tensor。graph SHA256=BF29E84CCDF5BC17FDC2953AA30C03E0BDA1156F45F3E33DE861B866D9E8C985；architectures SHA256=B1B659830B3C12AC21A7009F6743CA51EF3DCBBBCD150E1AEA958BDED8BDAD25。兩者已不同於 17:19 的快照，本輪未連 NAS，不能宣稱 NAS 已部署。MinMax、Dropout、Adam weight_decay 路徑已存在 | 以最新 lazy 方向為起點討論磁碟特徵管線；早先「LSTM 全量展開」只適用於當時所讀版本 |
| 2026-09-11 18:02（本輪彙整） | 唯讀參考本機 V1 `D:/hermesnote/backend/ml/train.py` 及 utils.py | 找到可借用的設計與查核重疊視窗疑慮 | V1 某些 LSTM／Transformer 分類路徑已有 LazyWindowDataset（train.py:806 起、utils.py:723 起），但並非所有架構／target 都走 lazy；所讀 random 分支仍逐視窗隨機分配。其 normalizer 用 train 索引的最小／最大範圍，並非精確訓練輸入聯集，不應直接當作防漏範本。這只是本機 V1 檔案，未核實過去 NAS 實際版本 | 借用按需組窗概念，不背書「V1 已完全消除洩漏」 |
| 2026-09-11 18:02（本輪彙整） | 套用 hermes-spec-check；閱讀官方 Parquet、NumPy、PyTorch、scikit-learn、TA-Lib、XGBoost 文件 | 整理可討論的儲存／批次／切分方案 | 技能原 SPEC_TEMPLATE.md 路徑已失效，已在本機 research/codex/archive 找到並讀取；既有文件明確要求 Phase1=random、Phase2=chronological、Phase3=載入 Phase2 產物。以下附完整提案，不更改此協定 | 請 Claude 評估實作成本、記憶體驗收及隨機切分單位；提案非部署或測試授權 |

補充：使用者另貼出 17:45:08 kernel log，顯示 global OOM 殺掉 Python PID 359673，anon-rss=14892044 kB，task_memcg 容器前綴 e33f231d2eab。這一筆來源為使用者貼文；當時依「只看不動作」要求未操作。它仍不足以證明最新 17:56／17:57 的 lazy 修改已失敗。

## 給 Claude 討論的提案：大量特徵的磁碟資料管線與時間序列切分

狀態：**設計提案，尚未實作、未跑測試、未部署。** 目標是數百萬筆 OHLCV、數百個特徵、window=60 及可調 units／batch_size 下，避免整包展開耗盡 RAM，並使實驗的時間與資料邊界可驗證。容量設計與評估有效性分開驗收，省 RAM 不等於消除 leakage。

### A. 目前起點與不可混淆的名詞

- 最新本機 LSTM 已 lazy 組窗；應沿用此方向，補足磁碟資料層與資源預算。不能再把舊版完整視窗副本當成此刻 LSTM 的既定行為。
- 二維原始特徵、資料查詢結果、特徵對齊、正規化及索引仍會消耗 RAM；即使 lazy，整個工作峰值也不會「只跟 batch_size 有關」。目前 XGBoost 分支仍會完整展開，需獨立處理。
- 正規化：由 train 擬合 min/max 或均值／標準差，套用到 train／val／test。正則化：Dropout、權重懲罰等模型訓練機制，不是另一份要預先算完的資料表，也不會修復資料洩漏。
- 一個 sample 是一個按時間排列的視窗。window=60 表示一個 sample 內有 60 根；batch_size=128 表示一次訓練 128 個 sample。每個 sample 內部不得打亂。Dataset 可以逐 sample 讀，DataLoader 把它們組成 batch。[PyTorch 官方資料載入說明](https://docs.pytorch.org/docs/2.10/data.html)

### B. 儲存選型：Parquet 快照＋二維 NPY 訓練快取

| 用途 | 建議格式 | 形狀／內容 | 使用方式 |
| --- | --- | --- | --- |
| 固定市場資料快照 | 分片 Parquet（初版可用 ZSTD 壓縮） | datetime、OHLCV、必要資料版本欄位，按商品／時間框架／時間排序 | 分批寫入與讀取，保留明確 schema、時間型別、空值及版本 |
| 訓練用特徵快取 | 未壓縮 `.npy` 分片，以 read-only mmap 開啟 | 每片 `(rows, feature_columns)`、C-order、float32；永遠不存成全部 `(N,W,F)` | 每次只取必要列；完整 row_id 與 manifest 讓 window 能跨分片讀取 |
| 樣本與切分 | `.npy` 數值陣列，或等價二進位索引表 | sample_id、window_start/end、decision_time、label_end_time、y、train/val/test 索引、valid mask | 儲存編號與時間，不複製特徵實體；train/val membership 固定 |
| 可稽核設定 | 小型 JSON | feature names／順序、dtype、資料版本、參數、scaler、split seed／version、檔案校驗值 | JSON 適合中繼資料，不拿來存數百萬筆特徵數值 |

Parquet 提供分區、壓縮及分批讀取；NumPy `open_memmap` 可建立／開啟 memory-mapped NPY。以上是本案的選型建議，不代表 Parquet 與 NPY 是唯一可行格式。[Arrow Parquet](https://arrow.apache.org/docs/python/parquet.html)、[ParquetFile 分批讀取](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html)、[NumPy open_memmap](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html)

不用先把所有資料讀進 RAM 再寫檔。也不做一個 window 一個檔案：那會產生數百萬個小檔。先用 row shards，單片大小由解壓後 byte 預算決定，例如以 64 MiB 為初始目標，再實測吞吐；不是固定「幾萬列對所有 feature 數都適用」。若總列數已由固定快照得知，也可使用單一預配置 NPY，由批次填入。

快取設於 NAS 主機上的持久化資料目錄，掛入 training 容器；優先評估現有 SSD 儲存位置，不在本提案指定購置設備。網頁 backend 不需要獲得整份訓練快取。此為未來部署設計，現在不新增 volume 或路徑。

### C. 完整資料流

```text
固定來源快照與未來 holdout 邊界
  → DB 有界批次讀取／去重／排序 → 原始 Parquet 分片
  → 因果特徵計算 → 二維未正規化 feature.npy 分片
  → 依原始時間軸產生 sample 索引、target 與標籤成熟時間
  → 固定本 Phase 的 split manifest（只切索引）
  → 只掃 train 實際輸入資料擬合 scaler，完成後凍結
  → 從 train 索引取 batch → 從 NPY 組 B×W×F → 套 scaler → GPU
  → validation／inference 同樣分批 → 指標累計、預測分批落地
```

1. **DB 讀取：** 使用 server-side cursor 或穩定排序鍵的 keyset 分頁；避免 fetchall／完整 Record list／list-of-dict。同一 job 使用可重現快照，不能一頁讀舊資料、一頁讀後來回補的新版本。用一致快照或明確資料版本完成查詢；快照成本、SQL 排序／去重的 DB 端 RAM 另行量測，client 分批並不自動限制 PostgreSQL 的排序記憶體。
2. **特徵：** 只計算截至 observation time 已知的資訊。rolling 使用過去值；不能 centered rolling、向後取未來值補缺、用全期間排名或全資料擬合 PCA。固定公式的因果特徵可預算全期再依索引取用；需要 fit 的轉換、補值統計、特徵選擇只能在分割後用 train fit。
3. **遞迴指標：** 每塊不能重新初始化 RSI／EMA／MACD 後當作同一條指標。有限 lookback 指標保留必要歷史；遞迴指標需延續精確狀態或採經驗證的等價演算法。TA-Lib 官方指出遞迴函式依賴起算歷史；不能一律補 60 根便宣稱等價。若現有 binding 無法保持等價，可先在明確 RAM 預算內「逐指標計算完整歷史、立即寫磁碟並釋放」，不得同時保留幾百個完整輸出。這個過渡方案仍隨 N 增長，要明確標示上限。[TA-Lib 數值穩定性](https://ta-lib.org/api/#42-numerical-stability)
4. **原始時間軸：** 使用唯一、單調時間鍵；暖機 NaN 和缺值以 valid mask 標記。不能把中間缺值列刪掉後，悄悄把原本不連續的列當成連續窗口或改變 horizon。horizon 是後 h 根實際 K 棒；應先在來源時間軸確定 label_end，再映射有效 sample，不能用 clip 把未成熟標籤壓回已知區間。
5. **資料與模型隔離：** Dataset 回傳的 X 只含 schema 明列的 feature，不把 y、future_close、split flag 等拼進輸入。檔案存在哪個目錄不是防漏機制；模型究竟收到哪些欄位與擬合過哪些統計量才是。
6. **快取重用：** 原始／因果特徵 cache key 包含資料內容版本、商品／時間框架、特徵參數與順序、指標庫與程式版本、缺值政策。units／batch_size 的改動可重用相同特徵；scaler 另以 split manifest 與方法為 key，Phase 2 不誤用 Phase 1 的 fitted scaler。新特徵未完成前使用獨立建置狀態，完整且校驗通過才供訓練讀；不要讀半寫入的 cache。
7. **生命週期：** 原始／特徵快取是可重建衍生物，但必須有磁碟配額、有效使用者／job 引用與明確保留規則。不能在仍有訓練讀取時清理檔案；不得用刪正式資料來騰空間。原子完成標記、重用及日後清理皆需另行實作授權。

### D. 正規化規則：切分固定後，再 fit，再開始訓練

- 本案先保留目前 MinMax 語意，避免一次變更資料架構及實驗方法。以分塊 min/max 累積即可，不需要把 train 全陣列讀進 RAM；官方 MinMaxScaler 也提供 partial_fit。[MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- train fit 集合必須是「所有 train 視窗真正使用的二維特徵列的聯集」。只取 train 視窗終點不夠；用最小 train index 到最大 index 的整段，也可能納入沒有任何 train window 使用的列。先建 bitmap／區間聯集，再分塊掃描。
- MinMax 的 min/max 不受同一列重複出現在多個 train 視窗影響。若日後採 StandardScaler，要另定「每個原始列一次」或「按視窗出現次数加權」；兩者的均值／變異數不同，不能把改成串流當成偷偷換掉正規化語意的理由。
- 必須先掃完 train 並凍結統計量，再開始第一個訓練 epoch；不能一邊 partial_fit 更新 scaler、一邊訓練，讓前後 batch 的尺度不同。
- validation/test 只 transform；可能超出 [0,1]，不能為了漂亮而重新 fit。clip、常數欄位、NaN／inf、float32 精度政策應入 manifest。float64 計算／統計，轉 float32 前後做數值等價檢查；不要先在低精度價格上計算微小 target 報酬。
- Phase 2 用它自己的 train 重新 fit；Phase 3 載入 Phase 2 模型、feature schema、scaler，全部凍結。資料範圍包含 holdout 並不授權使用其分布／標籤調參。[scikit-learn 防漏原則](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

### E. 時間語意與隨機切分的界線

以「已收完的第 t 根 K 棒」為 observation time T 的示例：

```text
X_t = feature[t-W+1 : t+1]   # W 根，順序固定
y_t = rule((close[t+h] / close[t]) - 1)
decision_time = 第 t 根資料真正可取得的時間
label_end_time = 第 t+h 根資料真正可取得的時間
```

此處是提案的明確語意，不假設 DB datetime 已是收盤時刻；若目前 datetime 表示開盤，必須在 dataset 契約中轉成 availability time。二元 threshold=0 時 y=1 表示報酬 >0，y=0 表示未上漲；y=0 不自動等於交易做空。此提案不變更 BT 策略，也不以準確率作本輪容量驗收。

**不要打散原始 K 棒後再組 window。** 應先決定合法視窗的 sample 索引，再分配 train／val。儲存 chunk、window、batch、split block 是不同概念：磁碟分片邊界不應改變样本；batch 是計算單位；train/val membership 一旦建立就固定，不能每個 epoch 重新 8:2 分組。每個 epoch 只可重新排列既定 train 索引的讀取順序。

**重疊與 leakage 的區分：** 相鄰 W=60 視窗共享 59 根歷史，會造成高度相依；這本身不等同於每一種评估都有未來洩漏。例如 chronological 驗證第一個視窗使用切分前已知歷史，可以符合真實預測。問題在 random split：較晚的 train sample 可能包含較早 validation 的結果期間，而且模型本身用未來樣本訓練再評估過去。因此 train-only scaler 不能讓它變成未來外推測試。[scikit-learn TimeSeriesSplit 的適用原因](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

**既定協定維持：** Phase 1 random、Phase 2 chronological 並重訓、Phase 3 載入 Phase 2 產物。不能在此次系統改造中直接把 Phase 1 換成 chronological。對 Phase 1 提供下列兩條明確路徑供討論，初版預設保留 A：

| 路徑 | 切分方式 | 能回答什麼／限制 |
| --- | --- | --- |
| A：相容既有 Phase 1 | 固定 seed，對合法 window sample 索引隨機 8:2 | 維持舊實驗定義與比較基準；可用於訓練診斷／容量測試，不能宣稱 train/val 原始資料完全獨立或當成時間外推成績。記錄 overlap 與時間相依限制，封住 Phase 3 邊界 |
| B：另立版本的 block-random，待確認 | 先形成遠大於 W＋h 的連續時間區塊，再按區塊隨機約 8:2；依資料資訊區間丟棄跨 train/val 區塊的邊界 sample | 能減少相鄰 window 分落不同集合的問題；但改變了 random 的單位，必須先確認符合教授協定，不能無聲取代 A。仍不能宣稱未來外推，也不自動保證金融序列統計獨立 |

**B 的保護條件不是固定丟 60 根：** 每個 sample 記錄 feature 的最早資訊來源與 label_end，依其真實資訊区間排除跨界／互相污染的樣本。有限指標 lookback 會讓區間比 W 更長；EMA 等遞迴指標的精確歷史依賴可能很長。若要求「完全零原始資料共享」，須另定遞迴狀態／reset／近似誤差政策，不能只加 gap 就承諾做到。reset 會改指標值，必須作為另一個明確版本。區塊長度、抽樣 seed、保護區间與實際保留比例全部記錄。

若先把每個相鄰 window 隨機 8:2，再刪掉所有與 validation 重疊的 train window，可能幾乎刪光 train。應先做長區塊分配再處理邊界；保護後比例可能不再精確 80/20，要回報实际數量，不能為了湊比例把邊界樣本塞回去。

**Phase 2／3 的硬邊界：** train 標籤成熟時間必須早於 validation 的第一個決策可用時刻（或明確定義相同時刻的先後事件順序），Phase 1／2 的任何訓練與選模標籤不得跨入 Phase 3 holdout。較早、已知的歷史可供 validation／test 組視窗；對 holdout 期間觀察到的最新已收盤資料可用於當時特徵計算，但不能用未來資料重算、更新 fitted scaler 或回頭調參。不能在刪 NaN 後用重新壓縮的位置加 h 或 clip 來冒充成熟時間。

若加入 stacking：上游對 train 的 fitted 預測不能冒充 OOF；下游訓練必須遵循同一 split／時間約束。這是另一個防漏點，磁碟快取或新增正則化都不會自動修好。

### F. RAM／VRAM 與吞吐控制

例：N=3,000,000、F=300、W=60、float32；以下是十進位容量，只計數值陣列，不是實測峰值：

| 項目 | 容量 |
| --- | --- |
| 二維特徵 N×F×4 | 3.6 GB 磁碟數值資料 |
| 全量視窗 N×W×F×4 | 約 216 GB，应避免物化 |
| 單 batch B=128 | 9.216 MB 輸入 |
| 單 batch B=256 | 18.432 MB 輸入 |
| 300 萬個 int64 索引 | 24 MB |

- 以上不代表「GPU 只需 9 MB」：LSTM 激活、反向傳播、參數、optimizer state、workspace 另計。units 變大會增加模型容量與計算成本，部分參數項含 units 的平方；batch 變大增加輸入及激活。磁碟方案解決資料總量，不會讓任意大模型都塞得進 16 GB VRAM。
- 用明確 budget 控制 DB／feature chunk、同時駐留的 feature blocks、預取隊列與 pinned memory；不把整台 free RAM 當成訓練可獨占額度。初版單訓練 worker、DataLoader num_workers=0、無大量預取，確定基準後才逐步增加。多 worker 可複製父程序的 Python 物件而放大 RAM。[PyTorch 多程序注意事項](https://docs.pytorch.org/docs/2.10/data.html#multi-process-data-loading)
- mmap 不等於不吃 RAM：讀入頁面與 page cache 仍需記憶體，容器 memory.current 與主機記憶體也要觀察。除了匿名 RSS，還要區分可回收 file cache、鎖頁及 DB／其他程序用量。
- 隨機 split 不必改成磁碟隨機亂掃：sample membership 固定，取得一個 batch 的編號後可先依 shard／row 排序讀取，再還原既定 batch 順序；使用有上限的 read cache。若改用按大區塊打亂 batch 順序，要標記 sampler 變更，不能假裝完全相同的訓練順序。
- 正規化只在 batch 做成本較簡單；若實測 CPU transform 成為瓶頸，可在 scaler 凍結後分批產生本 split 專用的 normalized NPY 快取（再多一份 N×F 磁碟容量）。不一次在 RAM 製造完整 normalized copy。
- 驗證／完整推論同樣 batch 化：逐批累加 loss、accuracy、confusion matrix 等統計；需要逐列預測時寫磁碟。不可把所有 GPU batch output 放 list 後再 cat。精確 AUC 等需要全體排序的指標另設磁碟／記憶體預算，不冒充常數空間。
- 若 batch／units 超過預算，明確告知可用／要求容量；不偷偷縮 batch、改 precision 或退 CPU。gradient accumulation 可另開設定增加有效 batch，但 micro-batch、optimizer step 次數与隨機層使它不能未驗證就冒充完全相同的 batch 行為。
- GPU 壓力測試應記錄並核實模型與首個 batch 的實際 device；要求 GPU 的任務可採 fail-fast，避免 CPU 路徑跑出 epoch 卻誤以為 GPU 已驗證。這是待實作設定，現在未改。
- XGBoost 不能直接套 LSTM Dataset。它有官方 external-memory 機制，但需獨立 adapter、磁碟／host memory預算及版本驗證，不能把小 batch 重複 fit 當成相同 boosting。初版可先對 XGBoost 做容量預檢，超限清楚拒絕。[XGBoost external memory](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html)

### G. 分階段實作與驗收（待授權）

| 階段 | 範圍 | 必須驗收 |
| --- | --- | --- |
| 1：凍結目前語意 | 固定小資料快照、features、T／h、split seed／membership、scaler 方法；保留最新 lazy 作基準 | 同一 sample 的時間、X、y 與 split 可完全對照；不得以此次工作偷偷修其他 target／切分定義 |
| 2：磁碟資料層 | DB 分批、Parquet snapshot、二維 NPY／索引、train-only scaler、batch reader | 多個 chunk 大小／分片邊界結果等價；訓練、驗證、推論皆不完整展開 N×W×F |
| 3：容量與執行證據 | 數百萬筆、F=16/100/300/500、B=32/64/128/256 等代表配置，units 分開遞增 | 記錄 RAM／VRAM 峰值、資料准备時間、rows/s、batch 延遲、first-batch device、epoch 完成；不得只看單一瞬間 nvidia-smi。容量不允許時可預檢拒絕，不以主機 OOM 作正常容量控制 |
| 4：切分有效性 | 先保留 Phase1 相容路徑；B block-random 另版本另行確認 | membership 固定；所有跨界排除可解释；Phase3 完全隔離於 fit／選模；block 路徑的有限 lookback 保護與遞迴指標限制都可驗證 |

防漏驗收至少包含：

1. **未來擾動測試：** 修改 t 之後的輸入，feature[t] 及更早特徵不得改變（在相同歷史起點／版本下）；label 只有依其定義依賴到的未來區間可改變。
2. **scaler 集合測試：** 確認統計量只由精確 train 輸入聯集計算。random 視窗若共享原始列，不能錯把共享列稱為 validation-only；可擾動真正不屬於 train 依賴的列來檢查。
3. **時間邊界測試：** label_end 不越 Phase2／3 邊界；缺棒、暖機 NaN、跨交易日、夜盤、window 橫跨磁碟 shard 時仍保持原始時間語意。
4. **分塊等價測試：** 舊基準與新 reader 對同一 sample 比較 X、y、normalization，float32 轉換允許的誤差先定義；遞迴指標不能只比較遠離分塊邊界的幾個點。
5. **資料重用測試：** 換 batch／units 不重算未改變的因果特徵；換資料版本／feature 參數會使對應 cache 失效；換 split 必須重新 fit scaler；恢復模型後推論使用保存的 schema／scaler。
6. **資源與失敗可觀測性：** cache 未完成不能開跑；每階段有 job／container／PID／device／RSS／GPU 與時間戳；中止後狀態可反映失敗，不留下永久 running。測試與任何 container 限額調整需另授權。

### H. 有效性評估與交給 Claude 的確認點

【有效性評估】

- **結論：** Parquet 來源快照＋二維 NPY mmap＋索引切分＋batch 組窗是可行方向。它改善大 N／F 的資料容量；不能單靠這個宣稱 Phase1 random 無時間相依或完全無 leakage。
- **證據狀態：** 已靜態讀到最新本機 lazy 路徑及現有 MinMax／正則化，格式與載入機制有官方文件支持；未在 NAS 驗證此提案的吞吐、RAM、數值等價或無洩漏，也未核實最新部署。
- **主要風險：** 全期特徵與正規化中間副本、遞迴指標分塊不等價、錯用 scaler cache、random 重疊與後見性、標籤成熟時間錯位、驗證輸出累積、XGBoost 不同資料介面。
- **待 Code 確認：** DB datetime 的可用時刻與區間 end 是否包含當日；sample 的最後一窗與 label_end 索引；最新 lazy 修改部署／測試狀態；feature 精確依賴；可用 SSD 路徑及预算；V1 实際成功路徑是否為所讀 lazy 分支。
- **方法論待確認：** Phase1 的「random」是否允許以長區塊作隨機單位。確認前保留既定逐視窗 random；若要求嚴格無未來污染，不能同時承諾保留任意前後混雜的單窗 random 外推評估。
- **是否建議進入實作：** 建議先討論並同意階段 1／2 的最小範圍與驗收，再授權實作；此文件不構成改程式、部署或建立訓練任務的授權。

## 22:52：500 個特徵下，Lazy Window 是否必要

| 時間（台北） | 動作 | 目的 | 結果 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-09-11 22:52 | 唯讀重查 V1 utils.py:723、train.py:806，以及 V2 graph.py:56、:279、:298 與 architectures.py:80、:166 | 回答是否已有 Lazy Window，以及 500 特徵是否還需要請 Claude 補強 | V1 確有 LazyWindowDataset，但受架構／分類／target／feature_selection 條件限制；目前本機 V2 LSTM 確有 LazyWindowed 且每 batch 才組窗。V1、目前 V2 都仍保留全量二維特徵於 RAM，不能據此宣稱容量與 N 無關。未連 NAS，未執行訓練 | 沿用 V2 現有 lazy；向 Claude 提出有界特徵建置、mmap 及分批正規化的擴充需求，不重複新增同樣 class |

### 估算假設與決策

500 指 **模型輸入的 500 個數值欄位**，不是 500 個可能各自多輸出的指標。W=60、float32=4 bytes，下表為十進位容量，只計陣列；視窗總數近似原始列數，忽略頭尾排除：

| 原始資料列數 | 二維特徵 N×500×4 | 若完整物化 N×60×500×4 |
| --- | --- | --- |
| 100 萬 | 2 GB | 約 120 GB |
| 300 萬 | 6 GB | 約 360 GB |
| 500 萬 | 10 GB | 約 600 GB |

一個 B=128 的視窗 batch 輸入為 15.36 MB；B=256 為 30.72 MB。這不是 RAM／VRAM 峰值，尚有組窗／tensor 搬運、模型、反向傳播、預取及原始資料。units 變大不改變二維特徵表大小，但會增加模型與激活成本。

**決策：** 對「數百萬列、500 欄、W=60、32 GB RAM」需求，避免全量物化視窗是必要設計要求，V1 的 lazy 概念值得保留；V2 已有，毋須重複新增。僅複製 V1 的 RAM-backed lazy 不足以完成擴充目標。

目前可定位的剩餘風險：

1. V2 quotes.py:63 仍以 conn.fetch 取得完整 Record 集合再建 DataFrame；graph.py:191 對 aligned 特徵 concatenate、:203 可能依 valid_mask 複製，來源／對齊／組合陣列可能同時駐留。
2. graph.py:279、:280 的 X_raw[train_row_mask] 會建立選取副本；:286 對全量 X_raw 做減法／除法並轉 float32，仍有全量中間陣列。300 萬×500 的一份 float32 為 6 GB，若 dtype 是 float64 則為 12 GB；這是條件估算，未量測當前任務 dtype 或峰值。[NumPy advanced indexing 副本語意](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)
3. LazyWindowed 類別註解「不管資料筆數多大都只有幾十～幾百 MB」「尖峰只跟 batch_size 有關」不適用整條管線；應請 Claude 修正說明，不能拿註解當驗收證據。
4. architectures.py:80 的驗證輸入雖按批讀，但 outs 仍累積全部 GPU 輸出再 cat；它不是全量窗口等級的放大，仍需在完整容量驗收中記錄並評估。XGBoost 仍另走全量窗口路徑，這次 LSTM 判斷不能套用。

### 可轉交 Claude 的需求

請保留目前 V2 的 LazyWindowed 與 batch 訓練介面，不必另複製一套 V1 Dataset。將擴充目標明定為 300～500 萬列、500 個實際輸入欄位、window=60、batch=128／256；units=128 起另外測量模型容量。補上 DB／特徵分批建置、二維 NPY mmap、train-only scaler 分批 fit 與 batch transform，避免在準備階段仍產生完整特徵副本。NPY 可用 read-only memory mapping；僅換成 mmap 而仍全量運算／選取，不能算完成。[NumPy open_memmap](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html)

驗收先固定樣本、特徵、target、split 與 scaler 語意，確認新舊 X／y 等價，再量測資料準備、訓練、驗證的 RAM／VRAM 峰值。需包含實際特徵建置壓力，不能只用預先生成的隨機矩陣證明整條管線可用。增加 N 時不得出現 N×W×F 的實體配置；DB buffers、計算 chunks、預取與 page cache 仍須納入預算。Lazy Window 本身不解決 random split 的時間相依／leakage，切分方法另案討論，不在此容量修改中暗改。

本輪僅靜態查核、容量算式與提案更新，未證明 NAS 上新版可成功跑完，也未授權 Claude 或任何工具實作／部署。

## 23:30：閱讀三方對齊 I-002-CL-001 並擬回覆

| 時間（台北） | 動作 | 目的 | 結果 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-09-11 23:30 | 唯讀 docs/temp/2026-09-10-three-party-design-alignment.md 的背景、既有協定與 I-002-CL-001；查 TA-Lib 官方文件及 RSI／EMA 原始碼 | 回答 Claude 對 500 欄規模、暖機重疊、快取失效的三個問題 | Claude 新增本機 run_graph 峰值 5,067.9 MB、逐 epoch 數值一致的測量報告，本端未獨立重現。三個問題合理；max lookback 不能保證遞迴指標分塊等價，磁碟工作集即使不跨 job 重用仍有減少 RAM 的價值 | 回覆文字交由 Hermes 轉貼；未修改三方對齊文件，也未操作 NAS 或程式 |

### I-002-CX-001 回覆草案要點（由 Hermes 轉交）

1. **需求規模：** 依 Hermes 本輪口述，500 個實際輸入欄位是未來擴充的容量設計／壓力測試目標，不是已提供完整特徵清單、已排定日期的近期實驗。本端無依據承諾近期只用 30～50 欄，也不把未提供清單視為取消 500 欄目標。保留已完成 LazyWindowed；允許分階段做，16／50 欄可作中途檢查點，500 欄達標須另有實測，不能以此刻 16 欄成功代稱。
2. **數值正確性：** 接受分批暖機有風險，但不接受一律 max(lookback) 即宣稱精確等價。TA-Lib lookback 指第一個可輸出值前需要的輸入數，不等於遞迴函式全部歷史依賴；RSI 使用累積 gain/loss 的 Wilder 平滑，EMA／MACD 也依賴遞迴狀態。有限依賴的指標可用經驗證的重疊段，遞迴指標需正確續接狀態或明確近似誤差政策。本輪不替 Hermes 授權近似／reset。
3. **降低首階段成本的替代：** 不強求第一版把所有 TA-Lib 指標改成按時間分塊。可在固定原始時間軸及相同起算歷史下，逐指標或小組計算完整歷史，立即寫磁碟並釋放其輸出；避免全體特徵累積、全量對齊 concatenate 及完整正規化副本。DB 仍分批取，scaler 分批 fit／transform，訓練沿用 lazy。500 萬列的一個 float64 數值欄約 40 MB；OHLCV 五個純數值欄約 200 MB，未含索引、DataFrame、指標內部 buffer。這是受 N 限制的過渡路徑，須實測單指標峰值與磁碟 I/O，不宣稱任意 N 常數 RAM。磁碟欄位組裝成訓練所需 row-major 分片也必須分塊。
4. **磁碟功能的最低範圍：** 第一版可每 job 建立獨立工作集、不跨 job 命中重用，省 RAM 的價值依然成立。工作集仍需 schema、資料來源／快照身分、特徵參數與順序、程式／指標庫版本、完整完成標記及 job／Phase 對應。不能讀半寫入或其他 job 的檔案；磁碟容量與清理生命週期要清楚。跨 job 去重、命中率最佳化及自動增量更新可延後；未做時明確寫「不支援重用」，不能盲目沿用舊檔。
5. **既定研究契約保留：** 容量改造不改 Phase 1 random／Phase 2 chronological／Phase 3 凍結推論；每 Phase 的 train-only scaler 與資料快照繼承仍要成立。每 job 重算不是每次任意查一份最新資料，Phase 1→2 必須能還原同一資料版本。
6. **證據措辭修正：** Claude 所報 5,067.9 MB 是本機繞過 queue 的 run_graph 測量，能支持 lazy 降低該測試負載的 RAM；未證明 NAS 完整 job 流程修復。另一台 GPU 成功也不能單獨排除 NAS 的 CUDA／依賴差異；現有 kernel global OOM 確認的是該次主機 RAM 不足，不代替獨立 GPU 路徑驗證。不把各次 kernel anon-rss 與不同測量工具的 MB 當作完全同一單位／樣本。

參考：[TA-Lib lookback 與 numerical stability](https://ta-lib.org/api/)、[TA-Lib RSI 原始碼](https://raw.githubusercontent.com/TA-Lib/ta-lib/main/src/ta_func/ta_RSI.c)、[TA-Lib EMA 原始碼](https://raw.githubusercontent.com/TA-Lib/ta-lib/main/src/ta_func/ta_EMA.c)。官方 main 原始碼用於確認演算法性質，不代表 NAS 安裝版本已核實。

結論為建議的開發範圍與正確性條件；本輪未授權實作，三方原文件 H-004 的先對齊再排程要求維持。
