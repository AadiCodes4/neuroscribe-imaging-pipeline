"""
API-level tests for the FastAPI app using FastAPI's TestClient (httpx under
the hood). Exercises /health and /segment against a real, synthetically
generated image (no real MRI/patient data anywhere).
"""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.data import generate_sample
from app.main import app


@pytest.fixture()
def client():
    # Using TestClient as a context manager ensures FastAPI's startup event
    # (which loads/trains the model) actually runs before requests are sent.
    with TestClient(app) as c:
        yield c


def _synthetic_png_bytes() -> bytes:
    image, _ = generate_sample(size=64, seed=123)
    arr_uint8 = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(arr_uint8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["image_size"] == 64


def test_segment_endpoint_returns_valid_pngs(client):
    png_bytes = _synthetic_png_bytes()

    response = client.post(
        "/segment",
        files={"file": ("synthetic_scan.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()

    for field in (
        "original_png_base64",
        "mask_png_base64",
        "mask_overlay_png_base64",
        "saliency_overlay_png_base64",
    ):
        assert field in body
        decoded = base64.b64decode(body[field])
        # Should be a valid, decodable PNG.
        img = Image.open(io.BytesIO(decoded))
        img.verify()

    assert "saliency_method" in body
    # Must be honestly labeled as a gradient-based approximation, never claimed
    # to literally be SHAP (it may legitimately mention SHAP only to disclaim it).
    assert "gradient-based approximation" in body["saliency_method"]
    assert "not SHAP" in body["saliency_method"]
    assert 0.0 <= body["predicted_lesion_fraction"] <= 1.0
    assert "disclaimer" in body
    assert "synthetic" in body["disclaimer"].lower()


def test_segment_endpoint_vanilla_gradient_method(client):
    png_bytes = _synthetic_png_bytes()
    response = client.post(
        "/segment?saliency_method=vanilla_gradient",
        files={"file": ("synthetic_scan.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "vanilla input-gradient" in body["saliency_method"]


def test_segment_endpoint_rejects_bad_method(client):
    png_bytes = _synthetic_png_bytes()
    response = client.post(
        "/segment?saliency_method=shap",
        files={"file": ("synthetic_scan.png", png_bytes, "image/png")},
    )
    assert response.status_code == 400


def test_segment_endpoint_rejects_empty_file(client):
    response = client.post(
        "/segment",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
