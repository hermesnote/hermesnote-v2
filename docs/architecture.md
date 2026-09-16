# Hermesnote v2 — 技術架構

> 狀態：持續更新中，記錄目前定案的技術決策，不是一次寫死的規格

## 技術棧

| 層級 | 技術 | 備註 |
|------|------|------|
| 前端 | React + TypeScript + Vite | 版本鎖定 `vite ^6.3.1`（vite 8 的 rolldown 原生綁定在開發機上會安裝失敗，改用舊站已驗證過的版本組合） |
| 後端 | Python + FastAPI | 分層改為 router / service / repository，尚未動工 |
| ORM | SQLAlchemy 2（async）+ asyncpg | async 指非同步：伺服器可以同時處理多個資料庫查詢，不用一個查完才能做下一個，不是「同步」的意思 |
| Migration | Alembic | 搭配 SQLAlchemy 管理資料庫 schema 版本變更 |
| 資料庫 | PostgreSQL | 不變，沿用舊站 NAS 192.168.0.44:5432 |

**選型理由**：這次重做的刻意練習重點放在後端架構分層，不是重新學一套框架，所以後端/資料庫/ORM 維持跟舊站相同的組合，只改分層方式；前端同理維持 React。

## 前端專案結構

```
frontend/
├── index.html
├── src/
│   ├── main.tsx        ← 進入點，把 App 掛進 index.html 的 #root
│   ├── App.tsx          ← 最上層畫面元件
│   ├── index.css        ← 全站共用樣式 token（顏色、字體）
│   └── components/       ← 一元件一 .tsx 配一 .css
```

## 後端 Python 環境

- `backend/.venv`（Python 3.11）——**專屬於 hermesnote-v2 backend 的虛擬環境**，跟系統全域 Python、`hermes-agent` 的 venv、舊站的 conda env 都無關，互不影響
- **TA-Lib 已安裝並實測**（`TA-Lib` 套件 0.7.1，Windows 有預編譯 wheel，`pip install TA-Lib` 直接成功，沒有遇到 C 函式庫編譯問題）
- `talib.get_function_groups()` 會直接吐出官方分組，實測結果跟 `\\TRUENAS\hermes-agent\profiles\research-orchestrator\skills\research\references\features\library.md` 這份分類文件的 TA-Lib 十大類完全一致：

  | 分組 | 函式數 |
  |---|---|
  | Overlap Studies | 18 |
  | Momentum Indicators | 31 |
  | Volume Indicators | 3 |
  | Volatility Indicators | 3 |
  | Price Transform | 5 |
  | Cycle Indicators | 5 |
  | Pattern Recognition | 61 |
  | Statistic Functions | 9 |
  | Math Transform | 15 |
  | Math Operators | 11 |

  （共 161 個函式）
- 每個 TA-Lib 函式可個別單獨呼叫（例如只呼叫 `talib.SMA(...)`），不需要整組一起叫；`get_function_groups()` 只是查詢/列出用途
- **結論**：TA-Lib 最上層的十大類不用自己手動抽取重分類，呼叫官方方法就有；`library.md` 裡真正需要我們自己維護的，是中間的「次類」（研究用細分）跟 `Custom` 分支（TA-Lib 沒有的：Raw 原始價量/K棒幾何、Structure 價格結構）

## 特徵庫分類

分類文件：`\\TRUENAS\hermes-agent\profiles\research-orchestrator\skills\research\references\features\library.md`（跟 Hermes Agent 討論定案，樹狀結構：Library → 大類 → 次類 → 具體特徵）

**落地方式（設計方向，尚未實作）**：
- 每個具體特徵（例如 SMA）包裝成一個小 Python 函式，統一介面（輸入價格資料 + 參數，輸出特徵數值）
- 用一個「註冊表（Registry）」把特徵 key（對應 `library.md` 的路徑，例如 `talib.overlap.sma`）對應到實際函式 + 參數規格，不用寫死 if/else
- API 提供一支「特徵目錄」端點（例如 `GET /api/features`），把整棵分類樹 + 每個特徵的參數規格吐成 JSON——**這份 JSON 就是給 Agent（Codex 或未來的 Claude 對話）讀的「使用手冊」**，不是給人看的文件
- Agent 送實驗時打「送實驗」API，帶特徵 key + 參數清單，後端查註冊表呼叫對應函式，不用改程式碼就能組合不同特徵
- 分類文件、程式碼組織、Agent 看到的 API 規格，三者是同一套東西的三種呈現，設計上要保持一致對應

## 回測套件：vectorbt（決定，理由見下）

- 2026 現況查證：backtrader 新專案不太被推薦（開發趨緩）；zipline-reloaded 適合因子研究但環境設定重；**vectorbt 用向量化運算，大量參數掃描速度快**，適合「Agent 連續送很多組參數做實驗」的情境
- **重要限制**：vectorbt 是「整段資料一次算完」（post-processing），不是「一根一根即時往前推進」，官方 GitHub 上有人問過一樣的問題，證實它不是為 tick-by-tick 即時處理設計的
- **這個限制不影響我們**：回測的「即時視覺化重播」是**前端的事**，不是後端運算方式的事——後端一次算完整段結果（含每個時間點的進出場標記），前端用圖表函式庫把這份已算好的資料畫成可以拖拉縮放、甚至模擬播放的畫面
- 真正需要「後端也一根一根處理」的情境，是未來接上券商即時報價、做真實交易邏輯判斷的時候，那是另一條路，不是現在要解決的問題

## 圖表 / 即時渲染

- **TradingView**：有免費的 Advanced Chart Widget，可直接嵌入即時報價圖表，可行
- **MultiChart**：桌面軟體，無公開網頁嵌入方式，無法直接嵌入
- **TA-Lib**：技術指標計算函式庫，屬於後端計算層，不是顯示層（詳見上方「後端 Python 環境」）
- **vectorbt**（回測套件，已決定）：同樣是後端計算層，算出來的結果（權益曲線、進出場點）才透過 API 送到前端用圖表函式庫畫出來
- 前端圖表函式庫：`lightweight-charts`（TradingView 出的開源輕量圖表庫），回測重播跟未來的即時報價共用同一套元件，差別只在資料怎麼餵進去（一個是重播批次算好的資料，一個是接即時串流）
- 資料流向：後端（TA-Lib + vectorbt 算數據）→ API / websocket → 前端 `lightweight-charts` 渲染

## 前端路由與版面（2026-09-02 改版）

- 改用 **React Router**（`react-router-dom` v7）做真的頁面路由，不再是單頁滾動敘事
- `Layout.tsx`：固定外層（logo lockup 靠左縮小、深色版 + 導覽列「模型訓練」「量化回測」「Hermes」），`<Outlet/>` 渲染下方頁面內容
- 整站改深色主題（`index.css` 的 `:root` token 全部換成深色版），logo 用品牌系統內建的 reverse 變體（`#F6F4F0` 米白 + `#E2624F` 深底紅），不是自己發明的顏色
- 舊的 Hero 三分頁面板、過去專案輪播已從頁面上移除（程式碼保留在 `components/`，未刪除，之後可能重用）
- 公開頁面（`/quant`、`/model`）**只呈現，不提供操作**——送實驗的設定區（選特徵/參數/送出按鈕）規劃放在需要登入的後台，尚未實作，登入機制討論延後

## 部署架構（2026-09-08 起，V2 已正式取代 V1 上線）

- NAS（192.168.0.44）`/mnt/Hermesnote/web/hermes/` 底下，跟本機開發資料夾對稱：`frontend/`（build 產物）、`backend/`（含 `training/`）、`docker-compose.yml`、`nginx.conf`
- 三個容器，統一用 `hermesnote-` 前綴、docker compose 管理：`hermesnote-frontend`（nginx:alpine，不掛 GPU）、`hermesnote-backend`（web API，不掛 GPU）、`hermesnote-training`（訓練 worker，掛 GPU，`deploy.resources.reservations.devices` 語法，`runtime: nvidia` 那種舊寫法在這台機器上測過是無效的）
- Nginx Proxy Manager：`hermesnote.com` → `192.168.0.44:8082`（frontend），Custom Location `/api` → `192.168.0.44:8000`（backend）；用的是主機 port 不是容器內部 bridge IP，容器砍掉重建不影響這個轉發規則
- `deploy.ps1`（repo 根目錄）：本機一鍵部署，**只負責更新程式碼，不負責容器的建立/刪除**——部署前會先確認三個容器都存在且在跑，對不上就直接中止，不會嘗試建立新容器；`frontend` 服務因為沒有 build 步驟，光靠 bind mount 換檔案不會讓 nginx 重讀設定，所以每次都會額外 `docker compose restart frontend`
- V1（`D:\hermesnote`）已完全停用封存，不再維護、不再參考程式碼，僅三階段流程與 data leakage 防範兩項設計原則列為參考

## 量化回測 Demo（第一個打通的垂直切片，2026-09-02）

**驗證完成：資料庫 → TA-Lib → 策略層 → vectorbt → FastAPI → 前端圖表，整條路真的通了。**

- 資料來源：`quotes` DB（唯讀）`market.future_taifex_tx_15m`，2011 年至今，197,274 筆
- **重要資料品質問題（已修正，未來要記得）**：這張表(以及推測其他 `future_taifex_*` 表)偶爾同一根 K 棒因為資料管線重跑被寫入兩筆，`rule_version` 相同但 `built_at` 不同（差幾秒，像是同一批次跑了兩次沒做 upsert）。查詢時要 `ORDER BY datetime ASC, built_at ASC`，再依 datetime 去重保留最後一筆（= 最新 `built_at`），否則會有重複時間戳記，前端圖表庫會直接報錯拒畫
- 三段式管線（誰負責什麼，之前釐清過的分工）：
  1. **TA-Lib 算指標數值**（demo 用 RSI，`timeperiod=14`）——只是數字，沒有「該不該進場」的意思
  2. **策略/規則層（我們自己寫的程式碼，TA-Lib 和 vectorbt 都不提供）**——把指標數值轉成進出場布林訊號（demo 規則：RSI 由上往下跌破 30 進場、由下往上突破 70 出場）。這層對應 `library.md` 分類裡 `Custom → Structure → Events`
  3. **vectorbt 模擬**——`Portfolio.from_signals(close, entries, exits, ...)`，吐出完整 Portfolio 物件（逐筆交易、權益曲線、統計數字）
- 後端：`backend/services/backtest_demo.py`（管線邏輯）+ `backend/main.py`（`GET /api/backtest/demo`，asyncpg 查資料庫，TA-Lib/vectorbt 是同步運算丟到 `asyncio.to_thread` 避免卡住 event loop）
- 前端：`frontend/src/pages/QuantBacktestPage.tsx`，`lightweight-charts` v5（`addSeries(CandlestickSeries, ...)` + `createSeriesMarkers()` 標進出場箭頭），下方交易明細表 + 統計數字列
- **套件版本注意**：`plotly` 要鎖 `5.24.1`——開源版 `vectorbt`（1.1.0）跟 pip 預設裝的最新版 plotly（改了屬性名稱）不相容，import 就直接掛掉，這個要記進 `requirements.txt`（已凍結在 `backend/requirements.txt`）
- **這只是 demo 管線，不是最終策略**：RSI 門檻策略本身是賠錢的（示範用），重點是驗證整條技術路徑通不通，不是策略好壞

## 後台登入：Google OAuth + email 白名單（2026-09-02）

- 只有 Hermes（可能+太太）用後台，不做 DIY 帳密系統：前端用 Google Identity Services 拿 ID token，後端驗證後檢查 email 是否在白名單（`.env` 的 `ADMIN_EMAILS`），過了就簽發自己的 session JWT（12 小時效期，`python-jose`）
- 後端：`backend/auth.py`（驗證、簽發/解析 JWT、`get_current_user` dependency）、`backend/routers/auth.py`（`POST /api/auth/google`、`GET /api/auth/me`）
- 前端：`src/auth/AuthContext.tsx`（token 存 localStorage，開頁自動打 `/api/auth/me` 驗證）、`src/auth/RequireAuth.tsx`（路由保護殼）、`src/pages/admin/LoginPage.tsx`
- **重要 gotcha**：`.env` 相關的模組（`auth.py`、`services/quotes.py`）要自己在檔案開頭呼叫 `load_dotenv()`，不能依賴 `main.py` 的呼叫順序——如果某個模組在 `main.py` 呼叫 `load_dotenv()` 之前就被 import，模組層級讀環境變數會讀到空字串並被快取住

## 後台整體外殼（`/admin/*`）

- `AdminLayout.tsx`：頂部列（`LogoMark.tsx` 共用元件 + 頭像選單）+ 左側欄（交易系統／模型訓練／量化回測／家庭理財，單層不做樹狀）+ `<Outlet/>` 內容區
- 側欄跟前台的橫向導覽刻意做成不同語彙（側欄 vs 頂部橫排），一眼能分辨自己在前台還是後台
- 「家庭理財」＝記帳 Agent 專案在後台的入口（不是「記帳」，避免名字顯得太小，之後會長出理財規劃功能）

## 指標登記表 + 訊號模板 + 策略條件樹（量化回測核心引擎，2026-09-02）

### 指標登記表：TA-Lib 161 個機械式產生

- `backend/indicators/registry.py` 用 TA-Lib 的 `abstract.Function` API 機械式列出全部 161 個函式的參數 schema、輸出線名稱，不是手key的，跟官方分組數字逐一核對一致
- `backend/indicators/descriptions.py`：161 個函式的中文一句話說明（人工撰寫，TA-Lib 本身沒有說明文字），透過 registry 的 `description` 欄位吐給前端；`docs/library.md` 同步維護同一份內容
- `GET /api/indicators` 把整份登記表（含 `signal_supported`、`signal_type`、`description`）吐給前端

### 訊號模板（A/B 兩種角色的由來）

`indicators/signals.py` 三套通用模板，把「有沒有自己的進出場時機」跟「能不能算出數值」分開看：
- **型態辨識**（61 個）：TA-Lib 輸出已標準化成看漲(>0)/看跌(<0)/無(0)，直接當訊號
- **震盪指標門檻穿越**（RSI/CCI/WILLR/MFI/ULTOSC/CMO/STOCH，7 個）：人工查證各自合理的超買超賣門檻，共用「跌破下界/突破上界」模板
- **交叉型**（MACD，1 個）：兩條輸出線互相穿越

以上 69 個叫「觸發型」，能決定進出場的那一刻。其餘 92 個沒有觸發模板，但透過 `indicators/filters.py` 的通用門檻比較（任一輸出線跟門檻比大小），161 個全部都能當「過濾型」條件——過濾型不決定時機，只回答「現在符不符合」。

### 策略條件樹：進場/出場各自四棵樹，支援多空雙向與無限巢狀 AND/OR/加權

`indicators/tree.py` 是目前的核心引擎（取代了中間試過又放棄的 `indicators/combo.py` 兩層 A組/B組設計，設計演進過程見 `docs/record/archive/2026-09-02.md` 下半場）：

- 節點只有兩種：**條件節點**（`kind` 是 `long_entry`/`long_exit`/`short_entry`/`short_exit`/`filter`）跟**群組節點**（`mode` 是 `and`/`or`/`weighted`，`children` 可以是條件節點或另一個群組節點，無限巢狀）
- 每個觸發型指標算出「上升方向」「下降方向」兩個事件（震盪指標的跌破/突破、型態的看漲/看跌、交叉型的上穿/下穿）；放進 `long_entry`/`short_exit` 用上升方向，放進 `long_exit`/`short_entry` 用下降方向——使用者不用自己選方向，系統依放的位置自動接
- 過濾型條件（`kind: "filter"`）不分方向，四個位置都能放，只當閘門用（只擋進場，不擋出場，避免部位因為條件變化就卡住出不來）
- 一次回測是四棵樹：`long_entry_tree`／`long_exit_tree`／`short_entry_tree`／`short_exit_tree`（後兩個可以是 `null`，等於純多方）——這剛好對應 vectorbt `Portfolio.from_signals` 需要的四條訊號（`entries`/`exits`/`short_entries`/`short_exits`），不是巧合是設計對了
- `services/backtest_demo.py` 的 `compute_strategy_backtest` 負責跑四棵樹求值 + 呼叫 vectorbt

### 回測執行：非同步 Job + DB 持久化（取代掉最早的同步阻塞版本）

全資料範圍（TX 1 分鐘線 15 年、2.9M 根K棒）壓力測試把同步版本的後端記憶體衝爆、瀏覽器收到巨大 JSON 卡死，才改成這個架構，細節見 `docs/record/archive/2026-09-06.md`：

- `POST /api/backtest/strategy`（`routers/strategy.py`）收到四棵樹的 JSON 後立刻回 `job_id`，實際運算丟給 `services/job_runner.py`（`asyncio.create_task` 背景執行，不是正式 job queue，重開伺服器會遺失還在跑的任務——個人專案接受這個限制）
- 結果存進 `hermesnote` DB 的三張表：`quant_backtest_jobs`（任務狀態/摘要統計）、`quant_backtest_candles`（PK `(job_id, ts)`）、`quant_backtest_trades`，都以 `job_id` 關聯、`ON DELETE CASCADE`
- 查詢端點都做成有界分頁，不管總資料量多大，單次回應大小都有上限：`GET /jobs/{id}/candles`（`before`/`around`/不給三種模式，搭配前端拖曳自動載入下一段）、`/trades`（`offset`/`limit`）、`/trades/range`（時間窗，圖表標記用）
- `DELETE /jobs/{id}` 刪任務，candles/trades 靠外鍵 cascade 自動一起清掉
- 前台 `/quant?job={id}` 用這個 job_id 讀資料庫顯示結果；後台建立回測時輪詢 `GET /jobs/{id}` 直到 `status` 變成 `done`/`failed`

### 前端：遞迴群組編輯器

- `frontend/src/pages/admin/sections/TreeBuilder.tsx`：遞迴的 `StrategyGroupEditor` 元件，每個群組上面選 AND/OR/加權，底下「＋加觸發條件」「＋加過濾條件」「＋加子群組」，子群組本身又是完整的一份可以一直往下疊
- 版面設計參考真的 survey 過幾個回測平台（TradingView/QuantConnect/MetaTrader/FinLab/TEJ/Portfolio Visualizer/BigQuant/TrendSpider/Zerodha Streak）的介面做法，最後借用 TrendSpider 的遞迴群組樹、TEJ Pro 的加權模式共存、Zerodha Streak 的白話文即時預覽
- `/admin/quant` 版面：左半 LONG／右半 SHORT，各自進場+出場兩棵樹，SHORT 有「啟用」開關（不強迫每次都要設空方策略）

## 已完成

- Hero 區塊（`src/components/Hero.tsx` / `Hero.css`，**目前未掛載在頁面上**）：品牌識別、三分頁即時渲染面板（交易系統／模型訓練／量化回測）、standby 誠實狀態、RWD 用 `clamp()` 讓寬高同時隨視窗縮放
- `/quant` 頁面：真的資料、真的圖表、真的交易明細，撐得住全資料範圍（2.9M 根K棒）；圖表含 K線+成交量+MA疊圖+交易定位，XQ 式時間軸滾動軸
- `docs/agent-api/`：給外部 Agent 呼叫 API 用的操作手冊（回測已可用，訓練待建）
- 後台登入（Google OAuth）+ `/admin/*` 整體外殼
- 量化回測後台設定頁（`/admin/quant`）：TA-Lib 161 個指標登記表、觸發/過濾兩種角色、多空四棵條件樹、無限巢狀 AND/OR/加權——`/quant`（前台結果頁）跟這頁共用同一套後端引擎，差別只在前台是唯讀展示、後台能設定送出
