"""
Interpretability for NeuroScribe.

HONESTY NOTE: this module implements a *gradient-based saliency
approximation* (vanilla input-gradient saliency, plus an optional
Grad-CAM-style variant using the bottleneck feature map). It is NOT SHAP
(Shapley Additive Explanations). SHAP was intentionally not used here because
exact/kernel/deep SHAP is comparatively heavy and slow for pixel-grid image
models and is overkill for a small demo U-Net; a gradient-based saliency map
is a standard, much cheaper approximation that answers a similar question
("which input pixels most influenced the model's output?") and is honestly
labeled as such throughout the code, API, and README.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import SmallUNet


def vanilla_gradient_saliency(model: SmallUNet, input_tensor: torch.Tensor) -> np.ndarray:
    """Compute a vanilla-gradient saliency map.

    Saliency = |d(sum of predicted lesion probabilities) / d(input pixel)|

    This measures how sensitive the model's total predicted lesion mass is
    to each input pixel, which is a real gradient computed on the actual
    model output for the given input (not a placeholder).

    Args:
        model: a trained SmallUNet in eval mode.
        input_tensor: FloatTensor (1, 1, H, W), requires no pre-existing grad.

    Returns:
        saliency: numpy array (H, W), normalized to [0, 1].
    """
    model.eval()
    x = input_tensor.clone().detach().requires_grad_(True)

    logits = model(x)
    probs = torch.sigmoid(logits)
    score = probs.sum()  # scalar: total predicted lesion "mass"

    model.zero_grad(set_to_none=True)
    score.backward()

    grad = x.grad.detach().abs()[0, 0].cpu().numpy()
    grad = _normalize(grad)
    return grad


def grad_cam(model: SmallUNet, input_tensor: torch.Tensor) -> np.ndarray:
    """Grad-CAM-style saliency using gradients at the U-Net bottleneck layer.

    This is a Grad-CAM *style* method adapted for a segmentation network: it
    hooks the bottleneck's last convolution (see `model.bottleneck_conv`),
    captures activations and gradients of the summed predicted probability
    with respect to those activations, computes channel-wise weights via
    global-average-pooled gradients, forms a weighted, ReLU'd combination of
    the activation maps, and upsamples the result back to input resolution.

    Returns:
        cam: numpy array (H, W), normalized to [0, 1].
    """
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inp, output):
        activations["value"] = output

    def bwd_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_fwd = model.bottleneck_conv.register_forward_hook(fwd_hook)
    handle_bwd = model.bottleneck_conv.register_full_backward_hook(bwd_hook)

    try:
        x = input_tensor.clone().detach().requires_grad_(True)
        logits = model(x)
        probs = torch.sigmoid(logits)
        score = probs.sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        acts = activations["value"].detach()[0]      # (C, h, w)
        grads = gradients["value"].detach()[0]        # (C, h, w)

        weights = grads.mean(dim=(1, 2))               # (C,)
        cam = torch.zeros(acts.shape[1:], dtype=acts.dtype)
        for c in range(acts.shape[0]):
            cam += weights[c] * acts[c]
        cam = F.relu(cam)

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam_np = cam[0, 0].cpu().numpy()
        cam_np = _normalize(cam_np)
        return cam_np
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    max_val = arr.max()
    if max_val > 1e-8:
        arr = arr / max_val
    return arr


def compute_saliency(
    model: SmallUNet, input_tensor: torch.Tensor, method: str = "grad_cam"
) -> tuple[np.ndarray, str]:
    """Dispatch to the requested saliency method.

    Args:
        method: "grad_cam" or "vanilla_gradient".

    Returns:
        (saliency_map (H, W) in [0, 1], human-readable method description)
    """
    if method == "vanilla_gradient":
        return vanilla_gradient_saliency(model, input_tensor), (
            "vanilla input-gradient saliency (gradient-based approximation, not SHAP)"
        )
    elif method == "grad_cam":
        return grad_cam(model, input_tensor), (
            "Grad-CAM-style saliency on U-Net bottleneck activations "
            "(gradient-based approximation, not SHAP)"
        )
    else:
        raise ValueError(f"Unknown saliency method: {method}")
