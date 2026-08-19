"""
main.py - FastAPI application for ResNet-50 image classification.

Implements:
  FR-1.1:  Model Selection and Loading (lifespan-managed ResNet-50)
  FR-2.1:  POST /predict (multipart image upload, top_k query parameter)
  FR-2.2:  GET  /health  (model-aware readiness probe)
  FR-2.4:  Input validation (non-empty file, 10 MB size cap)
  FR-3.1:  Structured JSON error responses with correlation IDs
  FR-3.2:  Defensive RuntimeError handling (OOM / dimension mismatch)
  Meta:    GET  /info, X-Correlation-ID, X-Response-Time headers
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, File, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from inference import preprocess_image
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from metrics import INFERENCE_LATENCY, REQUEST_COUNT

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB (FR-2.4)
APP_VERSION: str = "1.0.0"
MODEL_ARCHITECTURE: str = "ResNet-50"


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    correlation_id: str,
) -> JSONResponse:
    """Return a standardised JSON error envelope.

    Every error response shares the same shape so clients can parse
    failures uniformly:

        {
            "success": false,
            "error": {
                "code":    "VALIDATION_ERROR",
                "message": "Human-readable explanation."
            }
        }

    The correlation ID is also set as a response header so load-balancer
    logs can join request <-> response without parsing the body.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": message,
            },
        },
        headers={"X-Correlation-ID": correlation_id},
    )


_IMAGENET_LABELS: list[str] | None = None


def _get_imagenet_labels() -> list[str]:
    """Return the 1 000-class ImageNet label list, cached after first call."""
    global _IMAGENET_LABELS
    if _IMAGENET_LABELS is None:
        from torchvision.models import ResNet50_Weights
        meta = ResNet50_Weights.IMAGENET1K_V2.meta
        _IMAGENET_LABELS = meta["categories"]
    return _IMAGENET_LABELS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the ResNet-50 model lifecycle.

    On startup:
      - Downloads (or loads from cache) ResNet-50 with ImageNet V2 weights.
      - Switches to eval mode (disables dropout / batch-norm running stats).
      - Stores the model on ``app.state.model`` for endpoint access.
      - Pre-warms dispatch tables with a dummy forward pass.

    On shutdown:
      - Deletes the model reference so garbage collection can reclaim RAM
        (important when running under Gunicorn --preload + fork).
    """
    from torchvision.models import resnet50, ResNet50_Weights

    logger.info("Loading %s with IMAGENET1K_V2 weights ...", MODEL_ARCHITECTURE)
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()

    # Pre-warm: the first forward pass through PyTorch compiles internal
    # dispatch tables. Doing it here keeps the latency out of the first
    # real request.
    with torch.no_grad():
        _warmup = model(torch.randn(1, 3, 224, 224))
    del _warmup

    app.state.model = model
    logger.info("%s loaded and cached in app.state.", MODEL_ARCHITECTURE)

    _get_imagenet_labels()

    yield

    logger.info("Releasing %s from memory.", MODEL_ARCHITECTURE)
    del app.state.model
    torch.cuda.empty_cache()  # no-op on CPU, safe to call unconditionally


class PerformanceHeadersMiddleware(BaseHTTPMiddleware):
    """Attach tracing and timing headers to every response.

    - X-Correlation-ID: unique per request; enables distributed tracing.
    - X-Response-Time: wall-clock ms for the full request lifecycle.
    """

    async def dispatch(self, request: Request, call_next):
        correlation_id = uuid.uuid4().hex
        request.state.correlation_id = correlation_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"
        return response


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Increment ``http_requests_total`` for every completed response.

    Labels: status_code, method, endpoint.
    Requests to ``/metrics`` are excluded to prevent recursive self-counting
    during Prometheus scrapes.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path != "/metrics":
            REQUEST_COUNT.labels(
                status_code=str(response.status_code),
                method=request.method,
                endpoint=request.url.path,
            ).inc()
        return response


app = FastAPI(
    title="Simple Model API",
    description="Production-grade ResNet-50 image classification service.",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(PerformanceHeadersMiddleware)
app.add_middleware(PrometheusMiddleware)

# /metrics is unauthenticated by design; access control is enforced at the
# network/namespace level in Kubernetes. Uses a native route instead of
# app.mount(make_asgi_app()) because BaseHTTPMiddleware strips path context
# from mounted ASGI sub-apps.
@app.get("/metrics", tags=["operations"])
async def metrics() -> Response:
    """Prometheus scrape endpoint. Returns all registered metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", tags=["operations"])
async def health(request: Request) -> JSONResponse:
    """Return 200 if the model is loaded and ready, 503 otherwise.

    Consumed by: Dockerfile HEALTHCHECK, docker-compose healthcheck,
    and Kubernetes readiness/liveness probes.
    """
    model = getattr(request.app.state, "model", None)
    correlation_id: str = getattr(request.state, "correlation_id", "")

    if model is None:
        return _error_response(
            status_code=503,
            code="MODEL_NOT_LOADED",
            message="The model has not been loaded yet. The service is not ready.",
            correlation_id=correlation_id,
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "status": "healthy",
            "model": MODEL_ARCHITECTURE,
            "version": APP_VERSION,
        },
        headers={"X-Correlation-ID": correlation_id},
    )


@app.get("/info", tags=["operations"])
async def info(request: Request) -> JSONResponse:
    """Return static service metadata for operational dashboards."""
    correlation_id: str = getattr(request.state, "correlation_id", "")

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "data": {
                "app_name": "simple-model-api",
                "version": APP_VERSION,
                "model": {
                    "architecture": MODEL_ARCHITECTURE,
                    "weights": "IMAGENET1K_V2",
                    "num_classes": 1000,
                    "input_size": [3, 224, 224],
                },
                "preprocessing": {
                    "resize": 256,
                    "center_crop": 224,
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
                "constraints": {
                    "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
                    "accepted_formats": ["JPEG", "PNG", "BMP", "TIFF", "WEBP"],
                    "top_k_range": [1, 10],
                },
            },
        },
        headers={"X-Correlation-ID": correlation_id},
    )


def _run_inference(
    model: torch.nn.Module,
    image_bytes: bytes,
    top_k: int,
    correlation_id: str,
    filename: str | None,
) -> JSONResponse:
    """Execute preprocess -> forward pass -> postprocess, return JSONResponse.

    Extracted from the predict endpoint so the caller can wrap the single
    call-site in try/finally for guaranteed Prometheus histogram observation
    on every exit path (success, preprocessing failure, or OOM).
    """
    result = preprocess_image(image_bytes)

    if not result.ok:
        return _error_response(
            status_code=422,
            code="PREPROCESSING_FAILED",
            message=result.error or "Image preprocessing failed.",
            correlation_id=correlation_id,
        )

    try:
        with torch.no_grad():
            logits: torch.Tensor = model(result.tensor)
    except RuntimeError as exc:
        # FR-3.2: OOM or dimension mismatch during inference.
        logger.critical("RuntimeError during model forward pass: %s", exc)
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
        return _error_response(
            status_code=500,
            code="INFERENCE_RUNTIME_ERROR",
            message=(
                "The model encountered a runtime error during inference. "
                "This is typically caused by resource exhaustion. "
                "Please retry or contact support."
            ),
            correlation_id=correlation_id,
        )

    try:
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        labels = _get_imagenet_labels()
        predictions: list[dict[str, Any]] = [
            {
                "rank": rank + 1,
                "class_index": idx.item(),
                "class_name": labels[idx.item()],
                "confidence": round(prob.item(), 6),
            }
            for rank, (idx, prob) in enumerate(
                zip(top_indices.squeeze(0), top_probs.squeeze(0))
            )
        ]
    except RuntimeError as exc:
        logger.critical("RuntimeError during post-processing: %s", exc)
        return _error_response(
            status_code=500,
            code="POSTPROCESSING_RUNTIME_ERROR",
            message="Failed to compute prediction probabilities.",
            correlation_id=correlation_id,
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "predictions": predictions,
            "meta": {
                "model": MODEL_ARCHITECTURE,
                "top_k": top_k,
                "filename": filename,
            },
        },
        headers={"X-Correlation-ID": correlation_id},
    )


@app.post("/predict", tags=["inference"])
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, TIFF, WEBP)"),
    top_k: int = Query(
        default=5,
        ge=1,
        le=10,
        description="Number of top predictions to return (1-10).",
    ),
) -> JSONResponse:
    """Classify an uploaded image and return the top-k predictions.

    Pipeline:
        multipart upload -> validation -> preprocess_image() -> model forward
        -> softmax -> top-k sort -> structured JSON response.
    """
    correlation_id: str = getattr(request.state, "correlation_id", "")

    model = getattr(request.app.state, "model", None)
    if model is None:
        return _error_response(
            status_code=503,
            code="MODEL_NOT_LOADED",
            message="The model is not available. Please retry after startup completes.",
            correlation_id=correlation_id,
        )

    image_bytes: bytes = await file.read()

    if not image_bytes:
        return _error_response(
            status_code=400,
            code="EMPTY_FILE",
            message="The uploaded file is empty (0 bytes received).",
            correlation_id=correlation_id,
        )

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(image_bytes) / (1024 * 1024)
        return _error_response(
            status_code=413,
            code="FILE_TOO_LARGE",
            message=(
                f"File size {size_mb:.1f} MB exceeds the 10 MB limit. "
                "Please resize or compress the image."
            ),
            correlation_id=correlation_id,
        )

    # try/finally guarantees the histogram sample is recorded on every
    # exit path: success, preprocessing rejection, or forward-pass OOM.
    _t0 = time.perf_counter()
    try:
        return _run_inference(model, image_bytes, top_k, correlation_id, file.filename)
    finally:
        INFERENCE_LATENCY.observe(time.perf_counter() - _t0)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert any uncaught exception into a structured JSON error.

    Last line of defence: if a bug slips past per-endpoint handlers,
    the client still receives a parseable JSON body instead of a raw
    stack trace.
    """
    correlation_id: str = getattr(request.state, "correlation_id", uuid.uuid4().hex)
    logger.exception(
        "Unhandled %s [correlation_id=%s]",
        type(exc).__name__,
        correlation_id,
    )
    return _error_response(
        status_code=500,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected internal error occurred. Please retry or contact support.",
        correlation_id=correlation_id,
    )
