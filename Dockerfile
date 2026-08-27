# syntax=docker/dockerfile:1
# ---- Inference service image (M2 task 3) ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.pt

WORKDIR /app

# System deps for pillow / torch runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching). torch from the CPU index to
# keep the image small (no CUDA).
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 && \
    pip install -r requirements.txt

# App code
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY models ./models
RUN pip install --no-deps -e .

# Run as non-root
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
