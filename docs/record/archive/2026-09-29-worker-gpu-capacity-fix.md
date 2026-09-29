# 2026-09-29 training worker GPU 忙碌誤判修正

> 狀態：已修正、驗證並部署（Hermes 授權修正與部署）。

## 原因（已查證）

`worker.gpu_is_busy()` 用「GPU 總已用顯存 > 2000MB」判定忙碌。NAS 是 16GB RTX 5060 Ti，當時 GPU 使用率 0%，唯一佔用者是 training worker 自己（約 2306MiB，前一筆任務留下的 CUDA context 與 PyTorch 快取），worker 因此反覆延後，新任務一直 pending。受影響的 job `2011e1f0-72c8-4355-90d6-04d547a83f22` 已由 HA 取消（failed、started_at=null），保持原狀。

## 修改（`backend/training/worker.py`）

- 移除「總顯存 > 2GB 就阻擋」。改為 `gpu_capacity()`：先 `release_gpu_memory()`（`gc.collect()`＋`torch.cuda.empty_cache()`／`ipc_collect()`，torch 未載入時不動作、不初始化 CUDA），再用 nvidia-smi 的 `memory.free` 對照 `TRAINING_GPU_MIN_FREE_MB`（環境變數，`backend/.env` 可設，預設 4096）。查不到 nvidia-smi 視為無 GPU、不擋。
- 每筆任務結束（成功或失敗，`finally`）都釋放模型引用與 CUDA 快取。
- 延後時記錄：原因、total／used／free、worker 自身 allocated／reserved、門檻與變數名；接任務時也記錄當下顯存狀態。
- 文件：`docs/agent-api/training/api.md`（pending 的意義與原因查詢位置）、`docs/architecture.md`（部署架構）。

## 驗證

- `tests/test_worker_gpu.py`（7 項）：事故數字（total 16311／used 2306／free 14005）不再阻擋；可用不足時阻擋且訊息含各數值與門檻；門檻可設定；無 GPU 不擋；判斷前先釋放快取；**同一 worker 真的跑完一筆 LSTM 訓練後接續下一筆並完成**（CPU、合成資料，DB 寫入以記錄替代）；前一筆失敗後仍釋放並接續下一筆。
- 全部 187 項測試通過（3 skipped）；contract harness ALL MATCH。
- 本機 GPU（RTX 3070）確認釋放生效：訓練後 reserved 412MiB → 釋放後 88MiB，nvidia-smi 可用顯存回升約 300MiB（只驗證釋放行為，不代表 NAS 訓練結果）。

## 部署與正式服務確認

- 部署前確認無 pending／running 任務；`deploy.ps1` 重建 backend、training 並重啟 frontend。
- 正式環境：training 容器內 worker 已啟動、程式為新版；容器內獨立行程實測 `gpu_capacity()` → 可開始（total 16311MiB、used 0、free 15859MiB、門檻 4096MiB），GPU 使用率 0%；`/api/model/registry/architectures`、`/registry/components/optimizer`、`/api/model/jobs` 與前端皆 200。

## 提交

Hermes 指示後提交到 `main`：`backend/training/worker.py`、`backend/tests/test_worker_gpu.py`、`docs/agent-api/training/api.md`、`docs/architecture.md`、`docs/record/`（本紀錄與索引）。工作目錄中其他非本次修正的變更未納入。
