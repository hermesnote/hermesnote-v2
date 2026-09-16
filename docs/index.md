# Hermes Note v2 — 文件目錄

> 新對話 / 回來繼續開發，先讀這份，再依需要點進對應文件。

## 目前狀態

規格與架構文件尚在討論階段，本檔案先佔位，後續文件定案一份補一條連結。

## 文件

| 文件 | 內容 | 狀態 |
|------|------|------|
| （待建）spec.md | 主規格：範圍、目的、非目標 | 討論中 |
| [architecture.md](architecture.md) | 技術棧、前端結構、後台/量化回測策略引擎架構 | 使用中 |
| [library.md](library.md) | 特徵庫分類樹（TA-Lib 161 個 + Custom），含一句話中文說明 | 使用中 |
| （待建）decisions.md | 決策紀錄（決定＋理由＋替代方案） | 未開始 |
| [record/index.md](record/index.md) | 開發記錄摘要索引 | 使用中 |
| [agent-api/index.md](agent-api/index.md) | 給 Agent 呼叫 API 用的操作手冊（回測已可用，訓練/實驗重現待建） | 使用中 |

## 專案根目錄結構

```
hermesnote-v2/
├── docs/       ← 本目錄
├── frontend/
├── backend/
└── data/
```
