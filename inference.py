"""
inference.py — Image preprocessing and safe tensor operations.

Implements:
  FR-1.3: Image Preprocessing (RGB conversion, 224x224 resize/crop, ImageNet normalization)
  FR-3.2: Out-of-Memory Handling (catch RuntimeError from PyTorch, return clean error state)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── FR-1.3: Exact ImageNet normalization values per specification ────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = 224

# Pre-built transform pipeline — constructed once at module import,
# not per-request.  This is a pure function chain with no learnable
# parameters, so it is safe to share across threads.
_preprocess_pipeline = transforms.Compose([
    transforms.Resize(256),                          # Scale shortest edge to 256
    transforms.CenterCrop(TARGET_SIZE),              # Deterministic 224x224 center crop
    transforms.ToTensor(),                           # HWC uint8 [0,255] → CHW float [0,1]
    transforms.Normalize(mean=IMAGENET_MEAN,         # Channel-wise ImageNet normalization
                         std=IMAGENET_STD),
])


# ── Result container ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PreprocessResult:
    """Immutable result of the preprocessing pipeline.

    On success:  tensor is a (1, 3, 224, 224) float32 batch, error is None.
    On failure:  tensor is None, error is a user-safe diagnostic string.

    Endpoints inspect `.ok` and branch — the worker never crashes.
    """
    tensor: Optional[torch.Tensor]
    error: Optional[str]

    @property
    def ok(self) -> bool:
        return self.tensor is not None and self.error is None


# ── FR-1.3 + FR-3.2: Core preprocessing function ───────────────────────

def preprocess_image(image_bytes: bytes) -> PreprocessResult:
    """Convert raw binary image bytes into a normalized, batched tensor.

    Pipeline:
        raw bytes
          → PIL.Image (any mode)
          → RGB conversion (strips alpha, converts grayscale/palette)
          → Resize shortest edge to 256
          → Center crop to 224×224
          → float32 tensor in [0, 1]
          → ImageNet channel normalization
          → unsqueeze to batch dim (1, 3, 224, 224)

    Error handling strategy (FR-3.2):
        Each stage that can fail is wrapped individually so the error
        message pinpoints the failure.  All PyTorch RuntimeErrors
        (OOM, dimension mismatch, CUDA errors) are caught and converted
        to a clean PreprocessResult with error context.  The worker
        process is never killed.

    Args:
        image_bytes: Raw file bytes from the HTTP request body.

    Returns:
        PreprocessResult with either a ready-to-infer tensor or an error.
    """

    # ── Guard: reject empty / missing payload ────────────────────────
    if not image_bytes:
        return PreprocessResult(tensor=None, error="Empty image payload — no bytes received.")

    # ── Stage 1: Decode bytes → PIL Image ────────────────────────────
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Force full decode now so corrupt trailing chunks are caught
        # here, not inside the transform pipeline.
        image.load()
    except UnidentifiedImageError:
        logger.warning("Received bytes that are not a recognized image format.")
        return PreprocessResult(
            tensor=None,
            error="Unrecognized image format. Supported: JPEG, PNG, BMP, TIFF, WEBP.",
        )
    except (OSError, SyntaxError) as exc:
        # OSError  — truncated file, disk read failure
        # SyntaxError — malformed header (Pillow raises this for some formats)
        logger.warning("Image decode failed: %s", exc)
        return PreprocessResult(
            tensor=None,
            error=f"Corrupt or truncated image file: {type(exc).__name__}.",
        )
    except Exception as exc:
        # Defensive catch — Pillow plugin ecosystem can raise unexpected types
        logger.exception("Unexpected error during image decode.")
        return PreprocessResult(
            tensor=None,
            error=f"Image decode error: {type(exc).__name__}.",
        )

    # ── Stage 2: Convert to RGB ──────────────────────────────────────
    #   - RGBA → drops alpha channel
    #   - L (grayscale) → replicates to 3 channels
    #   - P (palette) → expands to full RGB
    #   - CMYK → converts color space
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
    except (OSError, ValueError) as exc:
        logger.warning("RGB conversion failed for mode '%s': %s", image.mode, exc)
        return PreprocessResult(
            tensor=None,
            error=f"Cannot convert image mode '{image.mode}' to RGB.",
        )

    # ── Stage 3: Resize, crop, normalize → tensor ───────────────────
    #   This is the PyTorch-heavy stage where OOM and dimension errors
    #   can surface.
    try:
        tensor: torch.Tensor = _preprocess_pipeline(image)
    except RuntimeError as exc:
        # FR-3.2: PyTorch memory allocation failures and dimension
        # mismatches surface as RuntimeError.  Catch, log, return
        # a clean error instead of letting the exception propagate
        # and kill the uvicorn worker.
        _handle_runtime_error(exc)
        return PreprocessResult(
            tensor=None,
            error=f"Preprocessing failed (PyTorch RuntimeError): {exc}",
        )
    except (TypeError, ValueError) as exc:
        # E.g., image with 0-pixel dimension after crop
        logger.warning("Transform pipeline value/type error: %s", exc)
        return PreprocessResult(
            tensor=None,
            error=f"Image dimensions incompatible with preprocessing pipeline: {exc}",
        )

    # ── Stage 4: Add batch dimension ─────────────────────────────────
    #   Shape goes from (3, 224, 224) → (1, 3, 224, 224).
    try:
        tensor = tensor.unsqueeze(0)
    except RuntimeError as exc:
        _handle_runtime_error(exc)
        return PreprocessResult(
            tensor=None,
            error=f"Failed to add batch dimension (OOM or shape error): {exc}",
        )

    # ── Stage 5: Validate output shape ───────────────────────────────
    expected_shape = (1, 3, TARGET_SIZE, TARGET_SIZE)
    if tensor.shape != expected_shape:
        logger.error(
            "Shape mismatch after preprocessing: got %s, expected %s",
            tensor.shape, expected_shape,
        )
        return PreprocessResult(
            tensor=None,
            error=f"Internal error: unexpected tensor shape {tuple(tensor.shape)}.",
        )

    return PreprocessResult(tensor=tensor, error=None)


# ── FR-3.2: Centralized RuntimeError handler ────────────────────────────

def _handle_runtime_error(exc: RuntimeError) -> None:
    """Log and triage a PyTorch RuntimeError without re-raising.

    Distinguishes between:
      • CUDA OOM  — logged at CRITICAL (operator may need to scale down batch)
      • CPU OOM   — logged at CRITICAL
      • Other     — logged at ERROR (dimension mismatch, dtype conflict, etc.)

    The caller is responsible for returning a PreprocessResult with the
    error message.  The worker process stays alive.
    """
    msg = str(exc).lower()

    if "cuda" in msg and "out of memory" in msg:
        logger.critical(
            "CUDA out-of-memory during preprocessing. "
            "Consider reducing max concurrent requests or switching to CPU. "
            "Original error: %s", exc,
        )
        # Attempt to reclaim leaked GPU memory so subsequent requests
        # have a chance of succeeding.
        torch.cuda.empty_cache()

    elif "out of memory" in msg or "alloc" in msg:
        logger.critical(
            "CPU memory allocation failure during preprocessing. "
            "Original error: %s", exc,
        )

    else:
        # Dimension mismatch, dtype errors, etc.
        logger.error("PyTorch RuntimeError during preprocessing: %s", exc)
