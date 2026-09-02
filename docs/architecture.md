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

## 已完成

- Hero 區塊（`src/components/Hero.tsx` / `Hero.css`）：品牌識別、三分頁即時渲染面板（交易系統／模型訓練／量化回測）、standby 誠實狀態、RWD 用 `clamp()` 讓寬高同時隨視窗縮放
