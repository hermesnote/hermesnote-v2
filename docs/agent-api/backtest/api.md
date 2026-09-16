# 量化回測 API

> 對應原始碼：`backend/routers/strategy.py`。連線位址見 `../index.md`。

## 流程總覽

1. `POST /api/backtest/strategy` → 立刻回 `job_id`（回測在背景跑，不會卡住這個請求）
2. 輪詢 `GET /api/backtest/jobs/{job_id}` → 看 `status`，`done` 才有結果
3. 需要的話，`GET /api/backtest/jobs/{job_id}/candles` / `/trades` / `/trades/range` 拿細節資料
4. 人要看圖表的話，瀏覽器開 `{前台網址}/quant?job={job_id}`

## `POST /api/backtest/strategy`

送出一次回測任務。

**請求 body：**

```json
{
  "symbol": "tx",
  "timeframe": "1m",
  "start": "2011-01-03",
  "end": "2026-09-02",
  "long_entry_tree": { ... },
  "long_exit_tree": { ... },
  "short_entry_tree": null,
  "short_exit_tree": null
}
```

| 欄位 | 型別 | 說明 |
|------|------|------|
| `symbol` | string | 目前有資料的：`tx`（大台）、`mtx`（小台）、`tmf`（微台），小寫 |
| `timeframe` | string | 目前有資料的：`1s`／`1m`／`5m`／`15m`／`30m`／`60m` |
| `start` / `end` | string，`YYYY-MM-DD` | 回測資料範圍，`end` 是**不含**的上界（`< end`，不是 `<= end`） |
| `long_entry_tree` | object | 必填，見 `tree-schema.md` |
| `long_exit_tree` | object | 必填，見 `tree-schema.md` |
| `short_entry_tree` | object 或 `null` | 不做空就給 `null`；要做空要跟 `short_exit_tree` 一起給 |
| `short_exit_tree` | object 或 `null` | 同上 |

**回應：**

```json
{ "job_id": "84b675dd-d4b5-4d92-bef2-bbac0e40fb51" }
```

**注意**：如果 `symbol`/`timeframe` 這個組合實際上沒有資料（資料庫裡沒有這張表，或這段 `start`/`end` 期間查無資料），任務會直接失敗，`status` 變成 `failed`，`error` 欄位會有訊息——這個失敗是**任務送出之後**才發生的，`POST` 當下拿到 `job_id` 不代表資料一定存在，要靠輪詢才會知道。

## `GET /api/backtest/jobs/{job_id}`

查一個任務的狀態跟結果摘要（不含逐筆交易明細，那個資料量可能很大，見下面的分頁端點）。

```json
{
  "job_id": "84b675dd-d4b5-4d92-bef2-bbac0e40fb51",
  "symbol": "tx",
  "timeframe": "1m",
  "start_date": "2011-01-03",
  "end_date": "2026-09-02",
  "status": "done",
  "error": null,
  "summary": {
    "total_trades": 186304,
    "win_rate_pct": 31.0,
    "total_return_pct": -98.75,
    "max_drawdown_pct": 98.81,
    "sharpe_ratio": -0.78,
    "profit_factor": 0.95,
    "avg_win_pct": 0.11,
    "avg_loss_pct": -0.06,
    "best_trade_pct": 5.44,
    "worst_trade_pct": -8.14
  },
  "indicators": [ { "key": "RSI", "label": "RSI", "params": {} } ],
  "rules": [ { "action": "entry", "text": "多方進場：..." } ],
  "bar_count": 2913981,
  "trade_count": 186304,
  "created_at": "2026-09-03T10:00:00+00:00",
  "finished_at": "2026-09-03T10:03:12+00:00"
}
```

`status` 的可能值：`pending`（還沒開始跑）／`running`（跑中）／`done`（完成，`summary` 有值）／`failed`（失敗，看 `error`）。

輪詢間隔：資料量小（幾天/幾週）通常幾秒內完成；全資料範圍（15 年 1 分鐘線，百萬根等級）曾經跑到數分鐘，**不要每秒打一次**，建議每 3-5 秒查一次，或先從小範圍測資料通不通，再跑大範圍。

## `GET /api/backtest/jobs?limit=20`

列出最近的任務（預設 20 筆，依建立時間新到舊），欄位跟上面單筆查詢差不多，但不含 `indicators`/`rules`/`trade_count`。找不到 `job_id` 或忘記存的話可以從這裡回頭找。

## 交易明細與 K 棒（除錯／進一步分析用，不是每次都需要）

- `GET /api/backtest/jobs/{job_id}/trades?offset=0&limit=200` — 分頁查逐筆交易（`total`/`offset`/`limit`/`trades`）
- `GET /api/backtest/jobs/{job_id}/trades/range?start={unix}&end={unix}` — 查某個時間窗內的交易（`unix` 是秒）
- `GET /api/backtest/jobs/{job_id}/candles?limit=2000` — 查 K 棒，`before={unix}` 往回查、`around={unix}` 從某時間點查起，三選一或都不給（給最新一段）

一般驗證回測邏輯對不對，看 `summary` 的統計數字通常就夠，不需要把幾十萬筆交易明細全部撈回來自己算一次。

## 完整範例（curl）

```bash
curl -X POST http://{後端位址}/api/backtest/strategy \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "tx",
    "timeframe": "1m",
    "start": "2024-01-01",
    "end": "2024-02-01",
    "long_entry_tree": {
      "type": "condition", "kind": "long_entry", "key": "MACD", "params": {}
    },
    "long_exit_tree": {
      "type": "condition", "kind": "long_exit", "key": "MACD", "params": {}
    },
    "short_entry_tree": null,
    "short_exit_tree": null
  }'
# → {"job_id": "..."}

curl http://{後端位址}/api/backtest/jobs/{job_id}
# → 輪詢直到 status == "done" 或 "failed"
```
