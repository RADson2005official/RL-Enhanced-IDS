# ── Build stage ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="RL-IDS Team"
LABEL description="RL-Enhanced Anomaly-Based IDS — Production Image"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application
COPY . .

# Create non-root user
RUN groupadd -r rlids && useradd --no-log-init -r -g rlids rlids && \
    mkdir -p /app/logs /app/models /app/reports && \
    chown -R rlids:rlids /app

USER rlids

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RL_IDS_LOG_LEVEL=INFO \
    RL_IDS_DASHBOARD_HOST=0.0.0.0 \
    RL_IDS_DASHBOARD_PORT=8050

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8050/health')" || exit 1

# Default: run training
CMD ["python", "scripts/train.py"]
