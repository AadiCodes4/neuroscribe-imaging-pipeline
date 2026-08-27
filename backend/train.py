#!/usr/bin/env python3
"""
Train the NeuroScribe SmallUNet on synthetic scan/lesion data and save a
checkpoint that the FastAPI app can load at startup.

Run from the `backend/` directory:

    python3 train.py

This trains briefly (a few hundred steps of synthetic data are generated
on the fly, so there is no fixed dataset to download) and prints the loss
at each epoch so you can see it actually decrease. It then writes
`app/weights/unet_synthetic.pt`.

NOTE: All training data is synthetic and generated in-process by
`app/data.py`. No real patient data or real MRI/CT scans are used anywhere
in this project.
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.data import generate_batch  # noqa: E402
from app.model import build_model, segmentation_loss  # noqa: E402

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "unet_synthetic.pt")


def train(
    epochs: int = 20,
    steps_per_epoch: int = 20,
    batch_size: int = 16,
    image_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    save: bool = True,
) -> list[float]:
    torch.manual_seed(seed)

    model = build_model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: list[float] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step in range(steps_per_epoch):
            images, masks = generate_batch(batch_size, size=image_size)
            optimizer.zero_grad()
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / steps_per_epoch
        epoch_losses.append(avg_loss)
        print(f"epoch {epoch:3d}/{epochs}  avg_loss={avg_loss:.4f}")

    elapsed = time.time() - t0
    print(f"training finished in {elapsed:.1f}s")

    if save:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "image_size": image_size,
                "epoch_losses": epoch_losses,
            },
            WEIGHTS_PATH,
        )
        print(f"saved weights to {WEIGHTS_PATH}")

    return epoch_losses


if __name__ == "__main__":
    train()
