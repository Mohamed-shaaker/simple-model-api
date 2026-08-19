"""
tests/test_inference.py - Unit tests for inference.py preprocessing pipeline.

Tests exercise every codepath in preprocess_image() without loading
the ResNet-50 model, so the suite runs fast on any CPU and in CI.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from inference import PreprocessResult, preprocess_image


# -- Helpers ------------------------------------------------------------------

def _make_image_bytes(width: int = 300, height: int = 300, mode: str = "RGB") -> bytes:
    """Return in-memory PNG bytes for a synthetic solid-colour image."""
    img = Image.new(mode, (width, height), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 300, height: int = 300) -> bytes:
    img = Image.new("RGB", (width, height), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_rgba_bytes() -> bytes:
    img = Image.new("RGBA", (300, 300), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_grayscale_bytes() -> bytes:
    img = Image.new("L", (300, 300), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# -- Happy-path tests ---------------------------------------------------------

class TestPreprocessImageSuccess:
    """Valid inputs should produce a (1, 3, 224, 224) float32 tensor."""

    def test_rgb_png_succeeds(self):
        result = preprocess_image(_make_image_bytes(mode="RGB"))
        assert result.ok
        assert result.error is None
        assert result.tensor is not None
        assert tuple(result.tensor.shape) == (1, 3, 224, 224)

    def test_jpeg_succeeds(self):
        result = preprocess_image(_make_jpeg_bytes())
        assert result.ok
        assert tuple(result.tensor.shape) == (1, 3, 224, 224)

    def test_rgba_converted_to_rgb(self):
        """RGBA image must be converted to RGB and succeed."""
        result = preprocess_image(_make_rgba_bytes())
        assert result.ok, f"Expected ok but got error: {result.error}"
        assert tuple(result.tensor.shape) == (1, 3, 224, 224)

    def test_grayscale_converted_to_rgb(self):
        """Grayscale (L mode) image must be converted to RGB and succeed."""
        result = preprocess_image(_make_grayscale_bytes())
        assert result.ok, f"Expected ok but got error: {result.error}"
        assert tuple(result.tensor.shape) == (1, 3, 224, 224)

    def test_small_image_still_succeeds(self):
        """Images smaller than 224px must still succeed via Resize(256)."""
        result = preprocess_image(_make_image_bytes(width=50, height=50))
        assert result.ok
        assert tuple(result.tensor.shape) == (1, 3, 224, 224)

    def test_large_image_succeeds(self):
        """Large images must be downsampled without error."""
        result = preprocess_image(_make_image_bytes(width=1024, height=768))
        assert result.ok
        assert tuple(result.tensor.shape) == (1, 3, 224, 224)

    def test_output_dtype_is_float32(self):
        import torch
        result = preprocess_image(_make_image_bytes())
        assert result.tensor.dtype == torch.float32

    def test_result_is_immutable(self):
        """PreprocessResult is a frozen dataclass - no attribute mutation."""
        result = preprocess_image(_make_image_bytes())
        with pytest.raises((AttributeError, TypeError)):
            result.tensor = None  # type: ignore[misc]


# -- Failure-path tests -------------------------------------------------------

class TestPreprocessImageFailure:
    """Invalid inputs should return a result with ok=False and an error string."""

    def test_empty_bytes_returns_error(self):
        result = preprocess_image(b"")
        assert not result.ok
        assert result.tensor is None
        assert result.error is not None
        assert len(result.error) > 0

    def test_corrupt_bytes_returns_error(self):
        """Random bytes that are not a valid image should return an error."""
        result = preprocess_image(b"\x00\x01\x02\x03NOT_AN_IMAGE")
        assert not result.ok
        assert result.error is not None

    def test_truncated_png_returns_error(self):
        """A valid PNG header followed by truncated data should fail cleanly."""
        valid_png = _make_image_bytes()
        truncated = valid_png[:50]
        result = preprocess_image(truncated)
        assert not result.ok
        assert result.error is not None

    def test_text_bytes_returns_error(self):
        """Plain UTF-8 text is not an image."""
        result = preprocess_image(b"Hello, this is not an image at all.")
        assert not result.ok
        assert result.error is not None

    def test_error_result_has_no_tensor(self):
        """On failure, tensor must be None."""
        result = preprocess_image(b"")
        assert result.tensor is None

    def test_ok_property_false_on_error(self):
        result = preprocess_image(b"\xff\xfe\x00\x01")
        assert result.ok is False


# -- PreprocessResult contract tests ------------------------------------------

class TestPreprocessResultContract:
    """The PreprocessResult dataclass must obey its documented invariants."""

    def test_ok_true_when_tensor_present_no_error(self):
        result = preprocess_image(_make_image_bytes())
        assert result.ok is True
        assert result.tensor is not None
        assert result.error is None

    def test_ok_false_when_error_present(self):
        result = preprocess_image(b"bad")
        assert result.ok is False
