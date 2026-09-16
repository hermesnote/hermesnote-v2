# 訓練容器（跟 web API 容器分離，見 docs/record/archive/2026-09-06.md 的 B 決策 + 2026-09-08 的落地）
# Build context 需為 backend/ 目錄（跟 web 容器的 backend/Dockerfile 一致）：
#   docker build -f training/Dockerfile -t hermesnote-v2-training .
#
# GPU/CUDA 版本說明：NAS 實體卡是 RTX 5060 Ti（Blackwell 架構），
# 已確認的主機驅動是 CUDA 12.8（見舊站 docs 稽核記錄），故用 PyTorch 官方 cu128 wheel。
# 2026-09-11 排查一連串「job 標記 running 後容器整個靜默死掉、log 連 traceback 都沒有」的事故：
# 本機（RTX 3070，Ampere 架構）用同一份程式碼、同樣的高負載設定（7 個 TA-Lib 特徵、
# units/batch=128、也試過崩潰當下那組 units/batch=64）跑了兩輪完整訓練都正常成功，
# VRAM/CPU/DRAM 都在合理範圍——排除了程式邏輯本身跟資料量的問題。兩邊唯一無法在本機
# 複現的差異就是 GPU 架構本身：這裡原本用不指定版本的 `pip install torch`，等於「build image
# 當下 PyPI 上最新的是什麼就裝什麼」，完全沒有記錄過哪個版本真的在這張 Blackwell（sm_120）
# 卡上驗證過、也可能在某次重建時不小心裝到一個還沒收錄 sm_120 kernel 的舊版本——
# CUDA 底層的 kernel 不相容通常就是這種「Python 完全來不及印出例外、process 直接被
# CUDA driver 中斷」的表現，跟這次事故的症狀吻合。改成明確釘住一個版本，之後每次重建
# 都是同一顆已知版本，不會因為重建時間點不同悄悄換掉；如果這個版本在 5060 Ti 上還是
# 一樣死掉，至少能確定問題不是「剛好裝到某個爛版本」，可以更有把握地往別的方向查。
# 下面這個版本號是本機（`hermesnote-backend` conda env）已經裝起來、且用另一個環境
# （`train-gpu`）確認過同一支 cu128 wheel 系列在這台機器上能正常初始化 CUDA 的版本，
# 但 Blackwell 這張卡本身的相容性**還是沒有在真正的 5060 Ti 上實測過**，deploy 後
# 務必看 worker 印出的 device/cuda_device_name 確認，不要假設一定成功。
FROM python:3.12-slim

WORKDIR /app

# TA-Lib 的 Python 套件只是綁定，底層這顆 C library 要自己編譯裝（跟 web 容器的 Dockerfile 一致）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget \
    && wget https://github.com/ta-lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get purge -y wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# 只裝 CPU 版以外的 torch（GPU 版），獨立一層方便 Docker cache
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 只複製訓練 worker 需要的程式碼：跟 DB 連線共用 services/hermesnote_db.py（hermesnote DB）、
# services/quotes.py（quotes DB，抓價量資料）、indicators/（收藏策略當 Feature Node 要重算條件樹）
# + training/ 本身
COPY services/hermesnote_db.py ./services/hermesnote_db.py
COPY services/quotes.py ./services/quotes.py
COPY services/__init__.py ./services/__init__.py
COPY indicators/ ./indicators
COPY training/ ./training

CMD ["python", "-m", "training.worker"]
