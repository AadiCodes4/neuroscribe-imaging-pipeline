"""
Small image I/O and visualization helpers shared by the API layer.

Deliberately implemented with just numpy + Pillow (no matplotlib dependency
at request time) so the API stays fast and light to import.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def load_grayscale(image_bytes: bytes, size: int) -> np.ndarray:
    """Decode arbitrary image bytes into a (size, size) float32 array in [0, 1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def array_to_png_base64(arr_uint8_rgb: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 array as a base64-encoded PNG string."""
    img = Image.fromarray(arr_uint8_rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def grayscale_to_png_base64(arr_float01: np.ndarray) -> str:
    """Encode a single-channel float [0,1] array as a base64-encoded grayscale PNG."""
    arr_uint8 = np.clip(arr_float01 * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _hot_colormap(v: np.ndarray) -> np.ndarray:
    """A small dependency-free approximation of the 'hot' colormap.

    v: float array in [0, 1] -> returns (..., 3) float array in [0, 1] (RGB).
    Ramps black -> red -> yellow -> white.
    """
    v = np.clip(v, 0.0, 1.0)
    r = np.clip(v * 3.0, 0.0, 1.0)
    g = np.clip(v * 3.0 - 1.0, 0.0, 1.0)
    b = np.clip(v * 3.0 - 2.0, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def heatmap_overlay_png_base64(base_gray_float01: np.ndarray, heat_float01: np.ndarray, alpha: float = 0.55) -> str:
    """Blend a 'hot' colormap heatmap over a grayscale base image, return base64 PNG."""
    base_rgb = np.stack([base_gray_float01] * 3, axis=-1)
    heat_rgb = _hot_colormap(heat_float01)
    blended = (1 - alpha) * base_rgb + alpha * heat_rgb
    blended_uint8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return array_to_png_base64(blended_uint8)


def mask_overlay_png_base64(base_gray_float01: np.ndarray, mask_float01: np.ndarray, alpha: float = 0.5) -> str:
    """Overlay a binary segmentation mask (translucent red/cyan) over the base image."""
    base_rgb = np.stack([base_gray_float01] * 3, axis=-1)
    overlay_color = np.array([1.0, 0.15, 0.15], dtype=np.float32)  # red
    mask3 = mask_float01[..., None]
    blended = base_rgb * (1 - alpha * mask3) + overlay_color * (alpha * mask3)
    blended_uint8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return array_to_png_base64(blended_uint8)
