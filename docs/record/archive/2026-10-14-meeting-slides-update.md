# 2026-10-14 研究進度簡報更新，路由改為 /20261014

> 狀態：已部署（Hermes 指示）。取代 [2026-09-30 簡報路由](2026-09-30-meeting-slides-route.md) 的網址。

- **內容**：`output/meeting-2026-09-30-slides.html` 在 9/30 之後有修改（第 4 頁加了文獻表 `literature-table`，列出 Levy 1967、Cornell & Dietrich 1978、Campbell 等 1993、Xue 等 2023，並補上對應 CSS），但網站讀的是 `public` 裡的舊複本。這次把定稿重新原樣複製到 `frontend/public/meeting-2026-09-30-slides.html`，檔名不變，兩份 md5 相同（`e343c562…`）。
- **路由**：`App.tsx` 由 `/20260930` 改為 `/20261014`。依 Hermes 指示不保留舊網址，也不做轉址。頁面元件 `Meeting20260930Page` 和 iframe 來源都不變。
- **部署**：
  - 部署前確認沒有 pending／running 任務：訓練 done 5、failed 2；回測 done 3。
  - 執行 `deploy.ps1`，exit 0。
  - 另外確認工作區中未提交的 `interview-resume.html` 與正式站上的版本相同，部署不會帶出其他變更。
- **驗證**：
  - 正式站 `/meeting-2026-09-30-slides.html` 與 `public` 的 md5 相同，含 `literature-table`。
  - 開啟 `https://hermesnote.com/20261014`：iframe 載入 4 頁，第 4 頁文獻表首列是「Levy（1967）」。
  - 正式 bundle 的路由只剩 `/20261014`；`/20260930` 已沒有簡報頁。
- **正式網址**：https://hermesnote.com/20261014
- **提交**：`frontend/public/meeting-2026-09-30-slides.html`、`frontend/src/App.tsx`、本紀錄，以及 `docs/record/index.md` 的本列。其他人的修改沒有納入：`docs/index.md`、`interview-resume.html`，以及 `docs/record/index.md` 中 10/05 的 HA 盤點列。
