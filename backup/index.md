# 本機退版備份

每個日期資料夾＝該日部署「之前」線上版（NAS）的檔案，只收當日改過的檔，路徑與專案根目錄相同。
退版：把該日資料夾的檔案蓋回專案根目錄、刪掉「當日新增」的檔案，再執行 `deploy.ps1`。

| 日期 | 退回到 | 備份內容 | 當日新增（退版時刪除） |
|---|---|---|---|
| 20261121 | 2026-11-18 OOM 修正版（新目標／特徵／GRU／CNN-LSTM／Transformer 之前） | 訓練程式 17 檔、測試 2 檔、`docs/agent-api/training/api.md`、`deploy.ps1`（排除 tests 之前） | `backend/training/registry/bounds.py`、`backend/training/registry/seq_models.py`、`backend/tests/test_targets_models.py`、根目錄 `.gitignore`（可留） |
