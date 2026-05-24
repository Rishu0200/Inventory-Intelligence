# ── Stage 1: install dependencies ────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app
 
# System deps for chromadb + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
 
# ── Stage 2: lean runtime image ──────────────────────────────
FROM python:3.11-slim
WORKDIR /app
 
# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
 
# Copy application code
COPY config.py        ./config.py
COPY api/             ./api/
COPY agents/          ./agents/
COPY orchestrator/    ./orchestrator/
COPY knowledge/       ./knowledge/
# Pre-built data (chroma_store + trained models)
COPY data/processed/  ./data/processed/
COPY data/raw/        ./data/raw/
 
# Render injects $PORT at runtime
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
