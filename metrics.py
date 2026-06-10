"""
metrics.py — Prometheus metric definitions for the ResNet-50 inference service.

Defines:
  INFERENCE_LATENCY  — Histogram tracking end-to-end inference pipeline duration
                       (preprocess → forward pass → postprocess).
  REQUEST_COUNT      — Counter tracking total HTTP requests, partitioned by
                       status code, method, and endpoint path.

Design notes:
  • Histogram buckets are tuned for CPU-bound ResNet-50 on a single Gunicorn
    worker.  Typical inference lands in the 50ms–2s range; the tails cover
    cold-start and resource-throttled edge cases.
  • With --workers 1, the default in-process collector is sufficient.
    If scaling to multiple Gunicorn workers, set PROMETHEUS_MULTIPROC_DIR
    and switch to prometheus_client.multiprocess.MultiProcessCollector.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ── Inference Latency ────────────────────────────────────────────────────
# Buckets (seconds): 10ms → 10s, log-spaced to capture the full CPU-bound
# inference distribution without wasting bucket space on sub-ms noise.
INFERENCE_LATENCY: Histogram = Histogram(
    "model_inference_duration_seconds",
    "End-to-end latency of the ResNet-50 inference pipeline "
    "(preprocess + forward pass + postprocess).",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ── HTTP Request Counter ────────────────────────────────────────────────
# Labels kept to three low-cardinality dimensions to avoid metric explosion.
REQUEST_COUNT: Counter = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the application.",
    ["status_code", "method", "endpoint"],
)
