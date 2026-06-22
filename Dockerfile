# ── 构建阶段 ──
FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --target=/install -r /tmp/requirements.txt


# ── 运行阶段 ──
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV ONLINE_HISTORY_DIR=/app/data/online
ENV TABPFN_MODEL_CACHE_DIR=/app/models/tabpfn_cache
ENV MODEL_DEVICE=cpu
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

WORKDIR /app

COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

COPY . /app

RUN mkdir -p /app/models /app/data/raw /app/data/online /app/logs /app/logs/results

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
