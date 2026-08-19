"""
tests/test_main.py - API integration tests for the FastAPI application.

Uses TestClient (Starlette) with a mock model injected into app.state and
a mocked lifespan so tests run instantly without downloading 100MB model weights.
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import main
from main import app, MAX_FILE_SIZE_BYTES


# -- Helpers & Mocks ----------------------------------------------------------

def _make_png_bytes(width: int = 300, height: int = 300) -> bytes:
    """Minimal valid RGB PNG image as bytes."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_model(num_classes: int = 1000) -> MagicMock:
    """Return a callable mock that mimics the ResNet-50 forward pass output."""
    mock = MagicMock()
    mock.return_value = torch.zeros(1, num_classes)
    mock.return_value[0][281] = 10.0  # tabby cat gets highest logit
    return mock


@asynccontextmanager
async def _noop_lifespan(app: FastAPI):
    """No-op lifespan for tests - skips model download."""
    yield


@pytest.fixture(autouse=True)
def setup_labels():
    """Ensure ImageNet labels are populated with dummy strings for fast testing."""
    main._IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]
    main._IMAGENET_LABELS[281] = "tabby, tabby cat"


@pytest.fixture()
def client_with_model():
    """TestClient with a fake model loaded in app.state."""
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app, raise_server_exceptions=True) as client:
        app.state.model = _make_mock_model()
        yield client
    if hasattr(app.state, "model"):
        del app.state.model


@pytest.fixture()
def client_no_model():
    """TestClient with NO model loaded (simulates pre-startup or failed load)."""
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app, raise_server_exceptions=True) as client:
        if hasattr(app.state, "model"):
            del app.state.model
        yield client


# -- GET /health --------------------------------------------------------------

class TestHealth:
    def test_healthy_when_model_loaded(self, client_with_model):
        r = client_with_model.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["status"] == "healthy"
        assert "model" in body
        assert "version" in body

    def test_503_when_model_not_loaded(self, client_no_model):
        r = client_no_model.get("/health")
        assert r.status_code == 503
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "MODEL_NOT_LOADED"

    def test_response_has_correlation_id_header(self, client_with_model):
        r = client_with_model.get("/health")
        assert "x-correlation-id" in r.headers

    def test_response_has_response_time_header(self, client_with_model):
        r = client_with_model.get("/health")
        assert "x-response-time" in r.headers


# -- GET /info ----------------------------------------------------------------

class TestInfo:
    def test_returns_200(self, client_with_model):
        r = client_with_model.get("/info")
        assert r.status_code == 200

    def test_response_schema(self, client_with_model):
        body = client_with_model.get("/info").json()
        assert body["success"] is True
        data = body["data"]
        assert "app_name" in data
        assert "version" in data
        assert "model" in data
        assert "preprocessing" in data
        assert "constraints" in data

    def test_model_metadata(self, client_with_model):
        data = client_with_model.get("/info").json()["data"]
        model_meta = data["model"]
        assert model_meta["architecture"] == "ResNet-50"
        assert model_meta["num_classes"] == 1000
        assert model_meta["input_size"] == [3, 224, 224]

    def test_constraints_metadata(self, client_with_model):
        data = client_with_model.get("/info").json()["data"]
        constraints = data["constraints"]
        assert constraints["max_file_size_bytes"] == MAX_FILE_SIZE_BYTES
        assert "JPEG" in constraints["accepted_formats"]


# -- GET /metrics -------------------------------------------------------------

class TestMetrics:
    def test_returns_200(self, client_with_model):
        r = client_with_model.get("/metrics")
        assert r.status_code == 200

    def test_content_type_is_prometheus(self, client_with_model):
        r = client_with_model.get("/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_contains_inference_latency_metric(self, client_with_model):
        img_bytes = _make_png_bytes()
        client_with_model.post("/predict", files={"file": ("img.png", img_bytes, "image/png")})
        r = client_with_model.get("/metrics")
        assert b"model_inference_duration_seconds" in r.content

    def test_contains_request_count_metric(self, client_with_model):
        client_with_model.get("/health")
        r = client_with_model.get("/metrics")
        assert b"http_requests_total" in r.content


# -- POST /predict ------------------------------------------------------------

class TestPredict:
    def test_valid_image_returns_200(self, client_with_model):
        img_bytes = _make_png_bytes()
        r = client_with_model.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert isinstance(body["predictions"], list)
        assert len(body["predictions"]) == 5

    def test_top_k_parameter_respected(self, client_with_model):
        img_bytes = _make_png_bytes()
        r = client_with_model.post(
            "/predict?top_k=3",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 3

    def test_prediction_schema(self, client_with_model):
        img_bytes = _make_png_bytes()
        r = client_with_model.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        pred = r.json()["predictions"][0]
        assert "rank" in pred
        assert "class_index" in pred
        assert "class_name" in pred
        assert "confidence" in pred
        assert pred["rank"] == 1

    def test_meta_in_response(self, client_with_model):
        img_bytes = _make_png_bytes()
        r = client_with_model.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        meta = r.json()["meta"]
        assert meta["model"] == "ResNet-50"
        assert meta["top_k"] == 5
        assert meta["filename"] == "test.png"

    def test_empty_file_returns_400(self, client_with_model):
        r = client_with_model.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert r.status_code == 400
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "EMPTY_FILE"

    def test_oversized_file_returns_413(self, client_with_model):
        oversized = b"X" * (MAX_FILE_SIZE_BYTES + 1)
        r = client_with_model.post(
            "/predict",
            files={"file": ("big.png", oversized, "image/png")},
        )
        assert r.status_code == 413
        body = r.json()
        assert body["error"]["code"] == "FILE_TOO_LARGE"

    def test_corrupt_image_returns_422(self, client_with_model):
        r = client_with_model.post(
            "/predict",
            files={"file": ("corrupt.png", b"\x00\x01\x02\x03NOTANIMAGE", "image/png")},
        )
        assert r.status_code == 422
        body = r.json()
        assert body["error"]["code"] == "PREPROCESSING_FAILED"

    def test_503_when_model_not_loaded(self, client_no_model):
        img_bytes = _make_png_bytes()
        r = client_no_model.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "MODEL_NOT_LOADED"

    def test_top_k_out_of_range_returns_422(self, client_with_model):
        img_bytes = _make_png_bytes()
        r = client_with_model.post(
            "/predict?top_k=99",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 422

    def test_top_k_zero_returns_422(self, client_with_model):
        img_bytes = _make_png_bytes()
        r = client_with_model.post(
            "/predict?top_k=0",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 422

    def test_response_has_correlation_id_header(self, client_with_model):
        img_bytes = _make_png_bytes()
        r = client_with_model.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert "x-correlation-id" in r.headers

    def test_correlation_id_is_hex_string(self, client_with_model):
        img_bytes = _make_png_bytes()
        r = client_with_model.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        cid = r.headers["x-correlation-id"]
        assert len(cid) == 32
        int(cid, 16)


# -- Correlation ID uniqueness ------------------------------------------------

class TestCorrelationID:
    def test_each_request_gets_unique_correlation_id(self, client_with_model):
        r1 = client_with_model.get("/health")
        r2 = client_with_model.get("/health")
        cid1 = r1.headers["x-correlation-id"]
        cid2 = r2.headers["x-correlation-id"]
        assert cid1 != cid2
