# Simple Model API

Image classification service built on ResNet-50 and FastAPI, running as a horizontally-scalable microservice inside Kubernetes.

The service accepts an image upload, runs it through a pre-trained ResNet-50 (ImageNet V2 weights), and returns the top-k predicted classes with confidence scores. Responses are structured JSON with correlation-ID headers; metrics are exposed natively to Prometheus.

## Architecture

```
Client (curl / browser / load sim)
    │
    │  POST /predict   (multipart image)
    │  GET  /health    (readiness probe)
    ▼
┌─────────────────────────────────┐
│  NGINX Ingress Controller       │  host: model-api.local
└──────────────┬──────────────────┘
               │  ClusterIP :80
               ▼
┌─────────────────────────────────┐
│  simple-model-api-service       │  kube-proxy DNAT → :8000
└──────────────┬──────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
  Pod 1      Pod 2      Pod 3       ← HPA scales 3–10 replicas
  Gunicorn + UvicornWorker
  ResNet-50 (baked weights)
```

Each pod runs a single Gunicorn worker with a Uvicorn async event loop. Model weights are baked into the Docker image at build time, so containers have no network calls on startup.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | /predict | Classify an uploaded image. Returns top-k predictions with confidence scores. |
| GET | /health | Model-aware readiness probe. Returns 200 when the model is loaded, 503 otherwise. |
| GET | /info | Static service metadata: model architecture, preprocessing config, constraints. |
| GET | /metrics | Prometheus scrape endpoint. Exposes `model_inference_duration_seconds` histogram and `http_requests_total` counter. |

### POST /predict

Request:

```bash
curl -X POST http://model-api.local/predict \
  -F "file=@dog.jpg" \
  -H "Host: model-api.local"
```

Query parameters:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| top_k | int | 5 | 1–10 | Number of top predictions to return. |

Response (200):

```json
{
  "success": true,
  "predictions": [
    {
      "rank": 1,
      "class_index": 258,
      "class_name": "Samoyed",
      "confidence": 0.891247
    }
  ],
  "meta": {
    "model": "ResNet-50",
    "top_k": 5,
    "filename": "dog.jpg"
  }
}
```

Constraints:

- Max file size: 10 MB
- Accepted formats: JPEG, PNG, BMP, TIFF, WEBP

Every response includes `X-Correlation-ID` and `X-Response-Time` headers for distributed tracing.

## Project structure

```
simple-model-api/
├── main.py                    # FastAPI app, middleware, route handlers
├── inference.py               # Image preprocessing pipeline (PIL → tensor)
├── metrics.py                 # Prometheus metric definitions
├── batch_inference_client.py  # Async load simulator for stress testing
├── requirements.txt           # Pinned dependencies (CPU-only PyTorch)
├── Dockerfile                 # Multi-stage build with baked model weights
├── docker-compose.yml         # Local dev with Gunicorn, cgroup limits, health checks
└── kubernetes/
    ├── deployment.yaml        # 3-replica Deployment, rolling updates, health probes
    ├── service.yaml           # ClusterIP Service (port 80 → 8000)
    ├── ingress.yaml           # NGINX Ingress routing /predict and /health
    ├── hpa.yaml               # HPA: 3–10 replicas, CPU 70% + memory 80% targets
    ├── configmap.yaml         # Externalised config (log level, env, model metadata)
    └── servicemonitor.yaml    # Prometheus Operator auto-discovery CRD
```

## Quick start

### Docker Compose (local)

```bash
docker compose up --build
```

The service starts on `http://localhost:8000`. The first build downloads ResNet-50 weights (~100 MB) and bakes them into the image. Subsequent rebuilds skip this layer unless dependencies change.

```bash
# Health check
curl http://localhost:8000/health

# Classify an image
curl -X POST http://localhost:8000/predict -F "file=@test.jpg.jfif"
```

### Minikube (Kubernetes)

```bash
# Start cluster with adequate resources
minikube start --cpus=4 --memory=8192

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server

# Build the image inside Minikube's Docker daemon
minikube image build -t simple-model-api:latest -f Dockerfile .

# Deploy
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
kubectl apply -f kubernetes/hpa.yaml

# Verify pods are running
kubectl get pods -l app=simple-model-api
```

Add the Minikube IP to your hosts file for ingress routing:

```bash
# Linux / macOS
echo "$(minikube ip) model-api.local" | sudo tee -a /etc/hosts

# Windows (PowerShell as Admin)
Add-Content C:\Windows\System32\drivers\etc\hosts "$(minikube ip) model-api.local"
```

Then send requests through the ingress:

```bash
curl -X POST http://model-api.local/predict \
  -F "file=@test.jpg.jfif" \
  -H "Host: model-api.local"
```

## Monitoring

The service exposes Prometheus metrics on `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| model_inference_duration_seconds | Histogram | End-to-end latency of the inference pipeline (preprocess + forward pass + postprocess). |
| http_requests_total | Counter | Total HTTP requests, labelled by status_code, method, endpoint. |

### Prometheus Operator

A ServiceMonitor CRD is included for automatic scrape target discovery. If running kube-prometheus-stack, apply it after the stack is installed:

```bash
kubectl apply -f kubernetes/servicemonitor.yaml
```

The ServiceMonitor scrapes every 15 seconds on the named port `http`.

## Load testing

`batch_inference_client.py` is an async load simulator for stress-testing the HPA autoscaler:

```bash
# Default: 50 requests, 10 concurrent
python batch_inference_client.py

# Heavy load
python batch_inference_client.py --requests 200 --concurrency 20

# Multi-wave burst
python batch_inference_client.py --waves 3 --requests 100

# Custom endpoint
python batch_inference_client.py --url http://127.0.0.1/predict --host model-api.local
```

Results are written to `inference_results.json` with per-request latency, status codes, and aggregate metrics (p50/p95/p99, RPS, success rate).

## Resource budget

| Resource | Request (guaranteed) | Limit (ceiling) |
|----------|----------------------|-----------------|
| CPU | 250m | 1000m |
| Memory | 512 Mi | 1536 Mi (1.5 Gi) |

The HPA scales between 3 and 10 replicas. Scale-up triggers at 70% CPU or 80% memory utilisation. Scale-down uses a 5-minute stabilisation window, removing at most one pod every 3 minutes.

## Key design decisions

**Baked model weights.** ResNet-50 weights are downloaded at Docker build time and cached in `/opt/torch_cache`. Containers have no network calls at runtime.

**CPU-only PyTorch.** `requirements.txt` uses the CPU-only PyTorch index, avoiding the ~2 GB CUDA bundle.

**Single Gunicorn worker per pod.** The service scales horizontally via HPA rather than vertically via process count. This keeps memory footprint predictable and avoids GIL contention in the PyTorch forward pass.

**Three-probe health strategy.** `startupProbe` gates the boot sequence (up to 5 minutes), `readinessProbe` controls Service traffic, and `livenessProbe` auto-restarts frozen processes, each with tuned delays to prevent false kills.

**Structured error responses.** Every error returns a uniform JSON envelope with a `code`, `message`, and `X-Correlation-ID` header. Clients parse failures the same way regardless of error source.

**Defensive OOM handling.** All PyTorch `RuntimeError` exceptions (CUDA OOM, dimension mismatch) are caught, logged at `CRITICAL`, and converted to clean HTTP responses. The worker process does not crash.

## Requirements

- Python 3.10+
- Docker (or Minikube with Docker driver)
- For Kubernetes: `kubectl`, Minikube, and optionally `kube-prometheus-stack` for monitoring

## License

This project is unlicensed. All rights reserved.
