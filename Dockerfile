# ==========================================
# Stage 1: Build & Dependency Resolution
# ==========================================
FROM --platform=linux/amd64 python:3.11-slim AS builder

WORKDIR /build

# Install compilation tools for binary wheel resolution
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-prod.txt

# ==========================================
# Stage 2: Lean Production Runtime Image
# ==========================================
FROM --platform=linux/amd64 python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root application user prior to copying files
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application, model artifacts (rf_regressor.pkl, scaler.pkl, feature_columns.pkl), and README
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser README.md .

USER appuser

EXPOSE 8000

# Zero-dependency health check using standard-library urllib (no curl package overhead)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch production server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
