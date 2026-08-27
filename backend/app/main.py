"""
NeuroScribe FastAPI backend.

Endpoints:
    GET  /health   -> basic liveness + model-loaded status
    POST /segment  -> accepts an uploaded image, returns a synthetic-lesion
                       segmentation mask + a gradient-based saliency heatmap,
                       each as base64-encoded PNGs, plus summary stats.

IMPORTANT / ETHICS NOTE: this model is trained ONLY on synthetically
generated data (see app/data.py) meant to loosely resemble a 2D scan slice
with lesion-like blobs. No real patient data or real MRI/CT scans are used
anywhere in this project. This is an educational/portfolio demo, NOT a
clinical or diagnostic tool, and must never be used for real medical
decisions.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.imaging import (  # noqa: E402
    grayscale_to_png_base64,
    heatmap_overlay_png_base64,
    load_grayscale,
    mask_overlay_png_base64,
)
from app.interpretability import compute_saliency  # noqa: E402
from app.model import build_model  # noqa: E402

IMAGE_SIZE = 64
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "unet_synthetic.pt")

_state: dict = {"model": None, "model_loaded_from": None}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _state["model"] = _load_or_train_model()
    yield


app = FastAPI(
    title="NeuroScribe API",
    description=(
        "Educational demo API for synthetic medical-image-style segmentation "
        "and gradient-based saliency interpretability. Synthetic data only — "
        "not a clinical or diagnostic tool."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _load_or_train_model():
    """Load trained weights if present; otherwise train briefly in-process.

    This means the API works out of the box even if `python train.py` was
    never run manually, though running the standalone training script first
    (and letting it finish its full schedule) gives a better-fit model.
    """
    model = build_model()
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _state["model_loaded_from"] = "checkpoint"
        return model

    # Fallback: quick in-process training so a fresh checkout still works.
    from app.data import generate_batch
    from app.model import segmentation_loss

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(15):
        images, masks = generate_batch(16, size=IMAGE_SIZE)
        optimizer.zero_grad()
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        loss.backward()
        optimizer.step()
    model.eval()
    _state["model_loaded_from"] = "fallback_inline_training"
    return model


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_source: str | None = None
    image_size: int


class SegmentResponse(BaseModel):
    original_png_base64: str
    mask_png_base64: str
    mask_overlay_png_base64: str
    saliency_overlay_png_base64: str
    saliency_method: str
    predicted_lesion_fraction: float
    disclaimer: str


DISCLAIMER = (
    "Synthetic demo only. Trained purely on procedurally generated data — "
    "no real patient data or real MRI/CT scans were used. Not a clinical or "
    "diagnostic tool; never use for real medical decisions."
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = _state.get("model")
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_source=_state.get("model_loaded_from"),
        image_size=IMAGE_SIZE,
    )


@app.post("/segment", response_model=SegmentResponse)
async def segment(
    file: UploadFile = File(...),
    saliency_method: str = Query(
        "grad_cam",
        description="Which gradient-based saliency approximation to use: 'grad_cam' or 'vanilla_gradient'.",
    ),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Sigmoid probability threshold for the binary mask."),
):
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if saliency_method not in ("grad_cam", "vanilla_gradient"):
        raise HTTPException(status_code=400, detail="saliency_method must be 'grad_cam' or 'vanilla_gradient'.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        gray = load_grayscale(contents, size=IMAGE_SIZE)
    except Exception as exc:  # pragma: no cover - depends on Pillow's error types
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    input_tensor = torch.from_numpy(gray[None, None, :, :].astype(np.float32))

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (probs >= threshold).astype(np.float32)
    saliency_map, method_desc = compute_saliency(model, input_tensor, method=saliency_method)

    original_b64 = grayscale_to_png_base64(gray)
    mask_b64 = grayscale_to_png_base64(mask)
    mask_overlay_b64 = mask_overlay_png_base64(gray, mask)
    saliency_overlay_b64 = heatmap_overlay_png_base64(gray, saliency_map)

    return SegmentResponse(
        original_png_base64=original_b64,
        mask_png_base64=mask_b64,
        mask_overlay_png_base64=mask_overlay_b64,
        saliency_overlay_png_base64=saliency_overlay_b64,
        saliency_method=method_desc,
        predicted_lesion_fraction=float(mask.mean()),
        disclaimer=DISCLAIMER,
    )
