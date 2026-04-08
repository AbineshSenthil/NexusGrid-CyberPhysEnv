# NexusGrid-CyberPhysEnv Dockerfile
# Base: python:3.11-slim (~130MB) — fast cold start, <400MB final image
# HEALTHCHECK on /health endpoint — HF Space shows "Running" after health passes

# ============================================================================
# Build stage
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for Docker layer caching)
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy source code
COPY . /app/src

# ============================================================================
# Runtime stage
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY --from=builder /app/src /app

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app"

# Environment variables (overridden at runtime via docker run -e)
# NOTE: Do NOT set sensitive values here — pass via -e at runtime
ENV API_BASE_URL=""
ENV MODEL_NAME=""

# Health check — /health endpoint returns immediately
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Run the FastAPI server with timeout-keep-alive for HF Space
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "30"]
