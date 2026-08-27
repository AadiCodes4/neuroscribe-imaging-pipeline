"""
Synthetic "medical scan" data generator.

IMPORTANT: This project uses NO real patient data and NO real MRI/CT scans.
Everything here is procedurally generated with numpy to *look* roughly like a
2D cross-sectional scan (a smooth, noisy "tissue" background) with one or
more brighter blob-like regions standing in for lesions. This is purely for
demonstrating a working segmentation + interpretability pipeline, not for
any clinical purpose.

Each sample is a single-channel 2D image (H, W) plus a binary mask (H, W)
marking the synthetic "lesion" pixels. A 2D slice-based approach was chosen
over a full 3D volume so the whole pipeline (data gen -> training -> API ->
frontend) trains and runs quickly on CPU, while still exercising a real
U-Net-style encoder/decoder architecture with skip connections.
"""

from __future__ import annotations

import numpy as np
import torch


def _smooth_noise(size: int, rng: np.random.Generator, octaves: int = 3) -> np.ndarray:
    """Cheap multi-octave smooth noise used as a synthetic 'tissue' background.

    Built by upsampling low-resolution random noise with bilinear-like
    interpolation (via repeated block upsampling + averaging), which is much
    cheaper than a real Perlin noise implementation but gives a similar
    smooth, blobby texture.
    """
    field = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    res = 4
    for _ in range(octaves):
        small = rng.random((res, res)).astype(np.float32)
        # Nearest-neighbour then simple box-blur upsample to `size`.
        rep = max(1, size // res)
        up = np.kron(small, np.ones((rep, rep), dtype=np.float32))
        up = up[:size, :size]
        if up.shape != (size, size):
            pad = np.zeros((size, size), dtype=np.float32)
            pad[: up.shape[0], : up.shape[1]] = up
            up = pad
        field += amplitude * up
        total_amp += amplitude
        amplitude *= 0.5
        res *= 2
    field /= total_amp
    # Normalize to [0, 1]
    field -= field.min()
    if field.max() > 1e-8:
        field /= field.max()
    return field


def _gaussian_blob(size: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-d2 / (2 * sigma ** 2))


def generate_sample(
    size: int = 64,
    n_blobs_range: tuple[int, int] = (1, 3),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic (image, mask) pair.

    image: float32 array in [0, 1], shape (size, size) — synthetic scan.
    mask:  float32 array in {0, 1}, shape (size, size) — synthetic lesion mask.
    """
    rng = np.random.default_rng(seed)

    background = _smooth_noise(size, rng) * 0.5 + 0.25  # roughly mid-gray tissue
    background += rng.normal(0, 0.02, size=(size, size)).astype(np.float32)  # sensor noise

    mask = np.zeros((size, size), dtype=np.float32)
    image = background.copy()

    n_blobs = rng.integers(n_blobs_range[0], n_blobs_range[1] + 1)
    margin = size * 0.2
    for _ in range(n_blobs):
        cy = rng.uniform(margin, size - margin)
        cx = rng.uniform(margin, size - margin)
        sigma = rng.uniform(size * 0.05, size * 0.11)
        intensity = rng.uniform(0.35, 0.55) * rng.choice([1, -1], p=[0.85, 0.15])

        blob = _gaussian_blob(size, cy, cx, sigma)
        image += intensity * blob

        # Lesion mask = region where the gaussian blob is "strong enough",
        # i.e. within roughly one sigma of the blob center.
        mask = np.maximum(mask, (blob > 0.6).astype(np.float32))

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image, mask


def generate_batch(
    batch_size: int,
    size: int = 64,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of synthetic samples as torch tensors.

    Returns:
        images: FloatTensor (B, 1, size, size)
        masks:  FloatTensor (B, 1, size, size)
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((batch_size, 1, size, size), dtype=np.float32)
    masks = np.zeros((batch_size, 1, size, size), dtype=np.float32)
    for i in range(batch_size):
        sample_seed = int(rng.integers(0, 2**31 - 1))
        img, msk = generate_sample(size=size, seed=sample_seed)
        images[i, 0] = img
        masks[i, 0] = msk
    return torch.from_numpy(images), torch.from_numpy(masks)
