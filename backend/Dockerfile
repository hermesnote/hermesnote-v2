FROM python:3.12-slim

WORKDIR /app

# TA-Lib：PyPI 的 TA-Lib==0.7.1 wheel 已內附 C 函式庫（ta_lib.libs/libta-lib-*.so.0.7.1，2026-11 以 ldd
# 在 NAS 容器核對過）。以前在這裡另外編譯的 0.4.0 從來沒被載入，只會造成「到底用哪一版」的混淆，
# 2026-11（M1）移除；訓練任務的環境指紋（training/provenance.py）會記錄實際載入的 talib_c 版本。

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
