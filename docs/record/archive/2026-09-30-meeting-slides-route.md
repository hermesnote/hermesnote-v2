# 2026-09-30 研究進度簡報路由 /20260930

> 狀態：已部署（Hermes 指示）。

- 定稿 HTML `output/meeting-2026-09-30-slides.html` 原樣複製為 `frontend/public/meeting-2026-09-30-slides.html`（內容未改）。
- 比照 `/hermes`：新增 `frontend/src/pages/Meeting20260930Page.tsx`（iframe 嵌入 `?embedded=1`，隱藏簡報自己的預覽導覽列；載入後把焦點交給簡報，方向鍵可換頁），`App.tsx` 在 `Layout` 下新增 `/20260930`，沿用網站導覽列（未加入導覽列選項）。
- 版面：iframe 填滿導覽列下方空間，簡報依可用空間等比縮放（1600×900 畫布）。
- 驗證：本機 1280×720、1024×600 與正式網站皆無頁面捲動、畫布不超出可用區；四頁各頁 0 個元素超出畫布；上一頁／下一頁、圓點、方向鍵／Home／End 換頁正常；正式網址直接開啟與重新整理（navigation type=reload）皆 200。
- 部署：部署前確認無 pending／running 任務；`deploy.ps1` 只更新 frontend（backend／training 程式未變，維持執行）。
- 正式網址：https://hermesnote.com/20260930
- 提交（Hermes 指示）：`frontend/public/meeting-2026-09-30-slides.html`、`frontend/src/pages/Meeting20260930Page.tsx`、`frontend/src/App.tsx` 與本紀錄；同一個 commit 一併收入稍早未提交的通用指標視覺化部署紀錄（`2026-09-30-generic-metric-visualization.md` 部署紀錄段落與索引狀態）。`docs/index.md` 他人新增列、`frontend/public/interview-resume.html` 等無關變更未納入。

