#!/usr/bin/env python3
"""
Evaluate the trained SmallUNet on a held-out synthetic test set.

Run from the `backend/` directory after `python3 train.py`:

    python3 evaluate.py

The test set is generated with a fixed seed so results are reproducible.
Training batches are drawn from the same synthetic distribution, so these
numbers measure in-distribution generalization only, not real-scan accuracy.

Reports Dice, IoU, precision and recall at a 0.5 threshold, compares against
a simple intensity-threshold baseline, and times CPU inference.
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.data import generate_batch  # noqa: E402
from app.model import build_model  # noqa: E402

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "weights", "unet_synthetic.pt")
TEST_SEED = 2026
N_TEST = 500
EPS = 1e-7


def scores(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Mean per-image Dice/IoU plus pooled precision/recall for binary masks."""
    dims = (1, 2, 3)
    tp = (pred * target).sum(dims)
    fp = (pred * (1 - target)).sum(dims)
    fn = ((1 - pred) * target).sum(dims)
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": (tp.sum() / (tp.sum() + fp.sum() + EPS)).item(),
        "recall": (tp.sum() / (tp.sum() + fn.sum() + EPS)).item(),
    }


def best_threshold_baseline(images: torch.Tensor, masks: torch.Tensor) -> tuple[float, dict[str, float]]:
    """Strongest simple baseline: a single global brightness threshold, tuned on the test set itself."""
    best = (0.0, {"dice": -1.0})
    for t in [i / 100 for i in range(30, 96)]:
        s = scores((images > t).float(), masks)
        if s["dice"] > best[1]["dice"]:
            best = (t, s)
    return best


def main() -> None:
    if not os.path.exists(WEIGHTS_PATH):
        sys.exit("No weights found. Run `python3 train.py` first.")
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    size = ckpt.get("image_size", 64)

    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    images, masks = generate_batch(N_TEST, size=size, seed=TEST_SEED)

    with torch.no_grad():
        model(images[:8])  # warm-up
        t0 = time.perf_counter()
        probs = torch.sigmoid(model(images))
        batch_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        for i in range(50):
            model(images[i : i + 1])
        single_ms = (time.perf_counter() - t0) * 1000 / 50

    unet = scores((probs > 0.5).float(), masks)
    thr, base = best_threshold_baseline(images, masks)

    print(f"Held-out synthetic test set: {N_TEST} images, {size}x{size}, seed {TEST_SEED}\n")
    print(f"{'Method':<34}{'Dice':>8}{'IoU':>8}{'Prec':>8}{'Recall':>8}")
    print(f"{'SmallUNet (threshold 0.5)':<34}{unet['dice']:>8.3f}{unet['iou']:>8.3f}{unet['precision']:>8.3f}{unet['recall']:>8.3f}")
    print(f"{f'Intensity threshold baseline ({thr:.2f})':<34}{base['dice']:>8.3f}{base['iou']:>8.3f}{base['precision']:>8.3f}{base['recall']:>8.3f}")
    print(f"\nCPU inference: {single_ms:.1f} ms per image (batch size 1), "
          f"{batch_ms / N_TEST:.2f} ms per image (batch of {N_TEST})")


if __name__ == "__main__":
    main()
