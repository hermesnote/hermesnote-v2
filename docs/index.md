# Hermes Note v2 — 文件目錄

> 新對話 / 回來繼續開發，先讀這份，再依需要點進對應文件。

## 目前狀態

給 AI 接手實作時依序讀：`spec.md`（必須遵守什麼）→ `architecture.md`（東西在哪、怎麼擴充）→ `decisions.md`（為什麼這樣設計）→ `record/index.md`（近期交付）。

## 文件

| 文件 | 內容 | 狀態 |
|------|------|------|
| [spec.md](spec.md) | 給 AI 的實作規範：定位與範圍、不變條件（INV）、功能需求逐條摘要（REQ）、AI 操作規則（OPS）、非目標、待確認（OPEN） | 使用中 |
| [architecture.md](architecture.md) | 給 AI 的架構說明：拓樸、技術棧與版本鎖、模組職責、資料庫、資料流程、模型組裝框架、擴充點與禁區、已知陷阱 | 使用中 |
| [library.md](library.md) | 特徵庫分類樹（TA-Lib 161 個 + Custom），含一句話中文說明 | 使用中 |
| [decisions.md](decisions.md) | 決策紀錄（ADR 式，D-xxx：決定＋理由＋替代方案＋狀態＋來源 record），含沿用自 V1 的兩項原則 | 使用中 |
| [record/index.md](record/index.md) | 開發記錄摘要索引 | 使用中 |
| [agent-api/index.md](agent-api/index.md) | 給 Agent 呼叫 API 用的操作手冊（回測、訓練／推論、擴充指南） | 使用中 |

## 專案根目錄結構

```
hermesnote-v2/
├── docs/       ← 本目錄
├── frontend/
├── backend/
└── data/
```
