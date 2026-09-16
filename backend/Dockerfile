FROM python:3.12-slim

WORKDIR /app

# TA-Lib 的 Python 套件只是綁定，底層這顆 C library 要自己編譯裝——
# pip 那個 TA-Lib==0.7.1 沒有這個會直接 import 失敗
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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
