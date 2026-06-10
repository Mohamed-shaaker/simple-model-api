# ==========================================================================
#  Multi-stage Dockerfile — simple-model-api  (v2.0.0 — baked weights)
#
#  Stage 1 (builder):  Install all Python dependencies into a virtual-env
#                      so we can copy only the finished artefact forward.
#  Stage 2 (runtime):  Lean python:3.10-slim image with just the venv,
#                      pre-downloaded ResNet-50 weights, and app code.
#                      Runs as non-root.  Fully offline-ready.
#
#  Layer-caching strategy (top → bottom = most stable → most volatile):
#      1. OS base + venv copy            (changes only on dependency bump)
#      2. ResNet-50 weight download      (changes only on model version bump)
#      3. Application source code        (changes on every code commit)
#
#  This ordering ensures a code change never re-downloads the ~100 MB model.
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
# ~2 GB CUDA bundle.  Index configuration (CPU-only primary index) is
# declared in requirements.txt itself — single source of truth.
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --default-timeout=1000 \
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

# ── Layer 2: Bake ResNet-50 weights into the image ───────────────────────
# Pre-download at BUILD time so the container never hits the network at
# runtime.  This layer sits between the venv and application code so that:
#   • A code change does NOT invalidate the ~100 MB weights layer.
#   • A requirements.txt change DOES rebuild everything below it (correct).
#
# TORCH_HOME is set for both build and runtime so torchvision.models finds
# the cached weights at /opt/torch_cache/hub/checkpoints/.
ENV TORCH_HOME=/opt/torch_cache

RUN python -c "\
import torchvision; \
from torchvision.models import ResNet50_Weights; \
torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)" && \
    chown -R appuser:appuser /opt/torch_cache

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
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
