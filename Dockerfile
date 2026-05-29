# ==========================================================================
#  Multi-stage Dockerfile — simple-model-api
#
#  Stage 1 (builder):  Install all Python dependencies into a virtual-env
#                      so we can copy only the finished artefact forward.
#  Stage 2 (runtime):  Lean python:3.10-slim image with just the venv and
#                      application code.  Runs as non-root.
#
#  Layer-caching strategy:
#      requirements.txt is COPY'd and pip-installed BEFORE any application
#      source code.  A change to main.py / inference.py only invalidates
#      the final COPY layer — PyTorch and friends are never re-downloaded.
# ==========================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Prevent .pyc clutter and ensure pip output is unbuffered for build logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# --- Layer-cache pivot: dependencies first, code second -----------------
#
# This COPY + install runs ONLY when requirements.txt changes.
# Everything below this layer is invalidated only by code changes.
COPY requirements.txt .

# Install CPU-only PyTorch from the official index to avoid pulling the
# ~2 GB CUDA bundle.  All packages land in a virtual-env we can copy
# cleanly into the runtime stage.
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --default-timeout=1000 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Cache-busting bypass: Install python-multipart in a separate layer
# so we don't invalidate the massive PyTorch installation above.
RUN /opt/venv/bin/pip install python-multipart


# ── Stage 2: Runtime ─────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Deterministic, crash-friendly Python behaviour in containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy the pre-built virtual-env from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user to run the application.
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid appuser --no-create-home appuser

WORKDIR /app

# --- Application source (invalidated by ANY code change) ----------------
COPY . .

# Drop privileges.
USER appuser

# Health-check: hit the /health endpoint the app already exposes (FR-2.2).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", \
         "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

EXPOSE 8000

# Default entrypoint — overridden by docker-compose command.
CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--preload"]
