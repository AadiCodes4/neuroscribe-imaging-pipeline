"""
Tests for the SmallUNet model: shape correctness and a real learning-signal
check (loss decreases over a handful of training steps on synthetic data).
"""

import torch

from app.data import generate_batch
from app.model import build_model, segmentation_loss


def test_forward_shape():
    model = build_model()
    images, _ = generate_batch(batch_size=2, size=64, seed=0)
    logits = model(images)
    assert logits.shape == (2, 1, 64, 64)


def test_loss_decreases_with_training():
    torch.manual_seed(0)
    model = build_model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step in range(25):
        images, masks = generate_batch(batch_size=8, size=64, seed=step)
        optimizer.zero_grad()
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first_avg = sum(losses[:3]) / 3
    last_avg = sum(losses[-3:]) / 3

    assert last_avg < first_avg, (
        f"expected loss to decrease with training, got first_avg={first_avg:.4f} "
        f"last_avg={last_avg:.4f}"
    )


def test_dice_loss_zero_for_perfect_prediction():
    from app.model import dice_loss

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:5, 2:5] = 1.0
    # Very confident, correct logits (large positive where target=1, large negative elsewhere)
    logits = (target * 2 - 1) * 20.0
    loss = dice_loss(logits, target)
    assert loss.item() < 0.01
