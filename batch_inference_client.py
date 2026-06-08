"""
batch_inference_client.py — Async load simulator for ResNet-50 microservice.

====================================================================================
Scout Project · Production Load Simulation Engine
====================================================================================

Architecture Notes
------------------
Load Simulation Mode
    Unlike a simple batch processor that walks a directory, this tool fires a
    configurable number of concurrent requests using the SAME verified image
    file to simulate realistic multi-user traffic hitting the inference
    endpoint simultaneously.  This is the primary mode for stress-testing the
    HPA and measuring autoscaler response under load.

Connection Pooling
    A single ``httpx.AsyncClient`` is instantiated once and shared across ALL
    concurrent tasks.  httpx maintains a connection-pool (HTTP/1.1 keep-alive)
    keyed by (scheme, host, port).  Pool limits mirror the semaphore cap so
    every task draws from the same pool — no per-task TCP handshake, TLS
    negotiation, or DNS lookup overhead.

Task Throttling
    An ``asyncio.Semaphore(max_concurrency)`` gates how many coroutines are
    inside the HTTP-call critical section simultaneously.  Excess tasks suspend
    at ``async with semaphore:`` until a slot opens.  This prevents socket
    exhaustion and keeps memory bounded regardless of total request count.

Ingress Routing
    Minikube's ingress controller listens on 127.0.0.1.  The service is
    routed via the ``Host: model-api.local`` header.  This script sends
    requests to http://127.0.0.1/predict with that host header by default,
    matching the verified working configuration.

Usage
-----
    python batch_inference_client.py                           # 50 reqs, 10 concurrency
    python batch_inference_client.py --requests 200            # heavier load
    python batch_inference_client.py --concurrency 25          # wider parallelism
    python batch_inference_client.py --waves 3                 # 3 successive bursts
    python batch_inference_client.py --image dog.jpg           # different image
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("load_sim")

# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_URL: str = "http://127.0.0.1/predict"
DEFAULT_HOST_HEADER: str = "model-api.local"
DEFAULT_IMAGE: str = "test.jpg.jfif"
DEFAULT_TOTAL_REQUESTS: int = 50
DEFAULT_CONCURRENCY: int = 10
DEFAULT_WAVES: int = 1
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_OUTPUT_FILE: str = "inference_results.json"


# ═══════════════════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class RequestResult:
    """Outcome of a single inference request."""

    request_id: int
    status: str                          # "success" | "error"
    status_code: int | None = None
    latency_ms: float = 0.0
    predictions: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None


@dataclass(slots=True)
class WaveMetrics:
    """Aggregated telemetry for a single wave of requests."""

    wave_number: int = 0
    total_requests: int = 0
    successes: int = 0
    failures: int = 0
    total_runtime_s: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    requests_per_second: float = 0.0
    error_code_distribution: dict[int | str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Image Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_image(filepath: Path) -> tuple[bytes, str]:
    """Load the image file into memory once.  Returns (bytes, mime_type).

    Exits if the file is missing or unreadable.
    """
    if not filepath.is_file():
        logger.error("Image file not found: %s", filepath.resolve())
        sys.exit(1)

    ext = filepath.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".jfif": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")

    data = filepath.read_bytes()
    logger.info(
        "Loaded image: %s (%d bytes, %s)",
        filepath.name, len(data), mime,
    )
    return data, mime


# ═══════════════════════════════════════════════════════════════════════════
#  Single-Request Worker
# ═══════════════════════════════════════════════════════════════════════════

async def send_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    request_id: int,
    image_bytes: bytes,
    image_name: str,
    mime_type: str,
    endpoint: str,
    timeout: float,
) -> RequestResult:
    """Fire a single inference request through the semaphore gate.

    The image bytes are already in memory (loaded once at startup),
    so no file I/O happens inside the hot path.
    """
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.post(
                endpoint,
                files={"file": (image_name, image_bytes, mime_type)},
                timeout=timeout,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return RequestResult(
                request_id=request_id,
                status="error",
                latency_ms=latency_ms,
                error_message=f"Timeout after {timeout}s",
            )

        except httpx.HTTPError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return RequestResult(
                request_id=request_id,
                status="error",
                latency_ms=latency_ms,
                error_message=f"{type(exc).__name__}: {exc}",
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return RequestResult(
                request_id=request_id,
                status="error",
                latency_ms=latency_ms,
                error_message=f"Unexpected: {type(exc).__name__}: {exc}",
            )

    # ── Process response ─────────────────────────────────────────────
    if response.status_code == 200:
        try:
            body = response.json()
            predictions = body.get("predictions", [])
        except (json.JSONDecodeError, KeyError):
            predictions = []

        return RequestResult(
            request_id=request_id,
            status="success",
            status_code=200,
            latency_ms=latency_ms,
            predictions=predictions,
        )

    # Non-200 → failure with status code
    error_detail = ""
    try:
        error_body = response.json()
        error_detail = error_body.get("error", {}).get(
            "message", response.text[:200],
        )
    except Exception:
        error_detail = response.text[:200]

    return RequestResult(
        request_id=request_id,
        status="error",
        status_code=response.status_code,
        latency_ms=latency_ms,
        error_message=f"HTTP {response.status_code}: {error_detail}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    results: list[RequestResult],
    runtime_s: float,
    wave_number: int,
) -> WaveMetrics:
    """Derive aggregate telemetry from individual request outcomes."""
    m = WaveMetrics(wave_number=wave_number, total_requests=len(results))

    latencies: list[float] = []
    for r in results:
        if r.status == "success":
            m.successes += 1
        else:
            m.failures += 1
            key: int | str = r.status_code if r.status_code is not None else "N/A"
            m.error_code_distribution[key] = (
                m.error_code_distribution.get(key, 0) + 1
            )
        if r.latency_ms > 0:
            latencies.append(r.latency_ms)

    m.total_runtime_s = runtime_s

    if latencies:
        latencies.sort()
        m.avg_latency_ms = sum(latencies) / len(latencies)
        m.min_latency_ms = latencies[0]
        m.max_latency_ms = latencies[-1]
        m.p50_latency_ms = _percentile(latencies, 50)
        m.p95_latency_ms = _percentile(latencies, 95)
        m.p99_latency_ms = _percentile(latencies, 99)

    if runtime_s > 0:
        m.requests_per_second = len(results) / runtime_s

    return m


def _percentile(sorted_data: list[float], pct: int) -> float:
    """Nearest-rank percentile from pre-sorted data."""
    if not sorted_data:
        return 0.0
    k = max(0, min(len(sorted_data) - 1, int(len(sorted_data) * pct / 100)))
    return sorted_data[k]


# ═══════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════

_BOX_W = 62
_DIV = "─" * _BOX_W


def print_wave_summary(m: WaveMetrics) -> None:
    """Render a clean, scannable report for a single wave."""
    rate = (m.successes / m.total_requests * 100) if m.total_requests else 0.0

    print()
    print(f"╔{'═' * _BOX_W}╗")
    title = f"WAVE {m.wave_number} RESULTS"
    print(f"║{title:^{_BOX_W}}║")
    print(f"╠{'═' * _BOX_W}╣")

    rows = [
        ("Total Requests", str(m.total_requests)),
        ("Successes", f"{m.successes}"),
        ("Failures", f"{m.failures}"),
        ("Success Rate", f"{rate:.1f}%"),
        ("", ""),
        ("Total Runtime", f"{m.total_runtime_s:.2f} s"),
        ("Throughput (RPS)", f"{m.requests_per_second:.2f}"),
        ("", ""),
        ("Avg Latency", f"{m.avg_latency_ms:.1f} ms"),
        ("Min Latency", f"{m.min_latency_ms:.1f} ms"),
        ("Max Latency", f"{m.max_latency_ms:.1f} ms"),
        ("P50 Latency", f"{m.p50_latency_ms:.1f} ms"),
        ("P95 Latency", f"{m.p95_latency_ms:.1f} ms"),
        ("P99 Latency", f"{m.p99_latency_ms:.1f} ms"),
    ]

    for label, value in rows:
        if label == "":
            print(f"║  {_DIV[:_BOX_W - 4]}  ║")
        else:
            line = f"  {label:.<40} {value}"
            print(f"║{line:<{_BOX_W}}║")

    if m.error_code_distribution:
        print(f"║  {_DIV[:_BOX_W - 4]}  ║")
        header = "  Error Code Breakdown:"
        print(f"║{header:<{_BOX_W}}║")
        for code, count in sorted(m.error_code_distribution.items(), key=str):
            err_line = f"    HTTP {code}: {count} occurrence(s)"
            print(f"║{err_line:<{_BOX_W}}║")

    print(f"╚{'═' * _BOX_W}╝")
    print()


def print_grand_summary(all_metrics: list[WaveMetrics]) -> None:
    """Render a combined summary across all waves."""
    total_req = sum(m.total_requests for m in all_metrics)
    total_ok = sum(m.successes for m in all_metrics)
    total_fail = sum(m.failures for m in all_metrics)
    total_time = sum(m.total_runtime_s for m in all_metrics)
    rate = (total_ok / total_req * 100) if total_req else 0.0

    # Collect all latencies for cross-wave percentiles
    all_latencies: list[float] = []
    combined_errors: dict[int | str, int] = {}
    for m in all_metrics:
        # We don't have raw latencies here, so use per-wave averages
        for code, cnt in m.error_code_distribution.items():
            combined_errors[code] = combined_errors.get(code, 0) + cnt

    avg_lat = (
        sum(m.avg_latency_ms * m.total_requests for m in all_metrics) / total_req
        if total_req else 0.0
    )
    rps = total_req / total_time if total_time > 0 else 0.0

    print()
    print(f"╔{'═' * _BOX_W}╗")
    title = f"GRAND TOTAL — {len(all_metrics)} WAVE(S)"
    print(f"║{title:^{_BOX_W}}║")
    print(f"╠{'═' * _BOX_W}╣")

    rows = [
        ("Total Requests", str(total_req)),
        ("Total Successes", str(total_ok)),
        ("Total Failures", str(total_fail)),
        ("Overall Success Rate", f"{rate:.1f}%"),
        ("", ""),
        ("Combined Runtime", f"{total_time:.2f} s"),
        ("Effective RPS", f"{rps:.2f}"),
        ("Weighted Avg Latency", f"{avg_lat:.1f} ms"),
    ]

    for label, value in rows:
        if label == "":
            print(f"║  {_DIV[:_BOX_W - 4]}  ║")
        else:
            line = f"  {label:.<40} {value}"
            print(f"║{line:<{_BOX_W}}║")

    if combined_errors:
        print(f"║  {_DIV[:_BOX_W - 4]}  ║")
        header = "  Combined Error Breakdown:"
        print(f"║{header:<{_BOX_W}}║")
        for code, count in sorted(combined_errors.items(), key=str):
            err_line = f"    HTTP {code}: {count} occurrence(s)"
            print(f"║{err_line:<{_BOX_W}}║")

    print(f"╚{'═' * _BOX_W}╝")
    print()


def export_results(
    all_results: list[list[RequestResult]],
    all_metrics: list[WaveMetrics],
    output_path: Path,
) -> None:
    """Persist structured results + metrics to a JSON file."""
    waves_payload = []
    for wave_results, metrics in zip(all_results, all_metrics):
        waves_payload.append({
            "wave": metrics.wave_number,
            "summary": {
                "total_requests": metrics.total_requests,
                "successes": metrics.successes,
                "failures": metrics.failures,
                "success_rate_pct": round(
                    metrics.successes / metrics.total_requests * 100, 2
                ) if metrics.total_requests else 0.0,
                "total_runtime_s": round(metrics.total_runtime_s, 3),
                "requests_per_second": round(metrics.requests_per_second, 2),
                "latency_ms": {
                    "avg": round(metrics.avg_latency_ms, 2),
                    "min": round(metrics.min_latency_ms, 2),
                    "max": round(metrics.max_latency_ms, 2),
                    "p50": round(metrics.p50_latency_ms, 2),
                    "p95": round(metrics.p95_latency_ms, 2),
                    "p99": round(metrics.p99_latency_ms, 2),
                },
                "error_code_distribution": {
                    str(k): v
                    for k, v in metrics.error_code_distribution.items()
                },
            },
            "results": [
                {
                    "request_id": r.request_id,
                    "status": r.status,
                    "status_code": r.status_code,
                    "latency_ms": round(r.latency_ms, 2),
                    "predictions": r.predictions,
                    "error_message": r.error_message,
                }
                for r in wave_results
            ],
        })

    output_path.write_text(
        json.dumps({"waves": waves_payload}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Results exported → %s", output_path.resolve())


# ═══════════════════════════════════════════════════════════════════════════
#  Live Progress Tracker
# ═══════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """Thread-safe counter that prints a progress line every N completions."""

    def __init__(self, total: int, report_every: int = 10) -> None:
        self.total = total
        self.report_every = report_every
        self._completed = 0
        self._successes = 0
        self._lock = asyncio.Lock()

    async def record(self, result: RequestResult) -> None:
        async with self._lock:
            self._completed += 1
            if result.status == "success":
                self._successes += 1

            if (
                self._completed % self.report_every == 0
                or self._completed == self.total
            ):
                pct = self._completed / self.total * 100
                logger.info(
                    "Progress: %d/%d (%.0f%%) — %d ok, %d err",
                    self._completed,
                    self.total,
                    pct,
                    self._successes,
                    self._completed - self._successes,
                )


# ═══════════════════════════════════════════════════════════════════════════
#  Wave Runner
# ═══════════════════════════════════════════════════════════════════════════

async def run_wave(
    wave_num: int,
    total_requests: int,
    concurrency: int,
    image_bytes: bytes,
    image_name: str,
    mime_type: str,
    endpoint: str,
    host_header: str,
    timeout: float,
) -> tuple[list[RequestResult], WaveMetrics]:
    """Fire a single wave of concurrent requests and return results + metrics."""

    semaphore = asyncio.Semaphore(concurrency)
    tracker = ProgressTracker(total_requests, report_every=max(1, total_requests // 10))

    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    logger.info(
        "── Wave %d: %d requests, concurrency=%d ──",
        wave_num, total_requests, concurrency,
    )

    async def tracked_request(req_id: int) -> RequestResult:
        result = await send_request(
            client, semaphore, req_id,
            image_bytes, image_name, mime_type,
            endpoint, timeout,
        )
        await tracker.record(result)
        return result

    wave_start = time.perf_counter()

    async with httpx.AsyncClient(
        limits=limits,
        headers={"Host": host_header},
        follow_redirects=True,
        http2=False,
    ) as client:
        tasks = [tracked_request(i + 1) for i in range(total_requests)]
        results: list[RequestResult] = await asyncio.gather(*tasks)

    wave_elapsed = time.perf_counter() - wave_start
    metrics = compute_metrics(results, wave_elapsed, wave_num)
    return results, metrics


# ═══════════════════════════════════════════════════════════════════════════
#  Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

async def run_load_test(
    image_path: Path,
    endpoint: str,
    host_header: str,
    total_requests: int,
    concurrency: int,
    waves: int,
    timeout: float,
    output_file: Path,
) -> None:
    """Load image, run all waves, print reports, export JSON."""

    image_bytes, mime_type = load_image(image_path)
    image_name = image_path.name

    all_results: list[list[RequestResult]] = []
    all_metrics: list[WaveMetrics] = []

    for w in range(1, waves + 1):
        results, metrics = await run_wave(
            wave_num=w,
            total_requests=total_requests,
            concurrency=concurrency,
            image_bytes=image_bytes,
            image_name=image_name,
            mime_type=mime_type,
            endpoint=endpoint,
            host_header=host_header,
            timeout=timeout,
        )
        all_results.append(results)
        all_metrics.append(metrics)
        print_wave_summary(metrics)

        # Brief pause between waves to let HPA react
        if w < waves:
            logger.info("Cooling down 5s before wave %d…", w + 1)
            await asyncio.sleep(5)

    if waves > 1:
        print_grand_summary(all_metrics)

    export_results(all_results, all_metrics, output_file)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Async load simulator for ResNet-50 inference microservice.\n"
            "Fires concurrent image uploads to stress-test the HPA and\n"
            "measure autoscaler response under realistic traffic patterns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python batch_inference_client.py\n"
            "  python batch_inference_client.py --requests 200 --concurrency 20\n"
            "  python batch_inference_client.py --waves 3 --requests 100\n"
            "  python batch_inference_client.py --url http://10.0.0.5/predict "
            "--host model-api.local\n"
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(DEFAULT_IMAGE),
        help=f"Image file to upload (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"Prediction endpoint URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST_HEADER,
        help=f"Host header for ingress routing (default: {DEFAULT_HOST_HEADER})",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_TOTAL_REQUESTS,
        help=f"Total requests per wave (default: {DEFAULT_TOTAL_REQUESTS})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max simultaneous in-flight requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--waves",
        type=int,
        default=DEFAULT_WAVES,
        help=f"Number of successive load waves (default: {DEFAULT_WAVES})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print()
    logger.info("═" * _BOX_W)
    logger.info("  Scout · Load Simulation Engine")
    logger.info("═" * _BOX_W)
    logger.info("  Endpoint    : %s", args.url)
    logger.info("  Host Header : %s", args.host)
    logger.info("  Image       : %s", args.image.resolve())
    logger.info("  Requests    : %d per wave", args.requests)
    logger.info("  Concurrency : %d", args.concurrency)
    logger.info("  Waves       : %d", args.waves)
    logger.info("  Timeout     : %.0fs", args.timeout)
    logger.info("  Output      : %s", args.output.resolve())
    logger.info("═" * _BOX_W)
    print()

    asyncio.run(
        run_load_test(
            image_path=args.image,
            endpoint=args.url,
            host_header=args.host,
            total_requests=args.requests,
            concurrency=args.concurrency,
            waves=args.waves,
            timeout=args.timeout,
            output_file=args.output,
        )
    )


if __name__ == "__main__":
    main()
