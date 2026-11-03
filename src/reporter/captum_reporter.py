# src/reporter/captum_reporter.py

"""
Captum Interpretability Reporter (ULTIMATE EDITION)

Provides Grad-CAM and Integrated Gradients visualizations using Captum.

✅ ZERO impact on model accuracy — only runs after training
✅ Device-aware (CPU/CUDA)
✅ Robust error handling
✅ Returns paths for JSON integration

Author: Babak Yousefian
Version: 2.0.0
"""

import os
import warnings
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ================================================================
# ✅ IMPORT CAPTUM
# ================================================================

try:
    from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam

    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("⚠️ Captum not installed. Install with: pip install captum")


# ================================================================
# ✅ HELPERS
# ================================================================

def _find_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    """Find the last Conv2d layer in the model (for Grad-CAM target)."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def _get_model_device(model: nn.Module) -> torch.device:
    """Get device of model parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _denormalize(
        tensor: torch.Tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    """Reverse ImageNet normalization. Returns (H, W, C) in [0, 1]."""
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    img = tensor.detach().cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))  # (C, H, W) → (H, W, C)
    return img


def _overlay_heatmap(orig: np.ndarray, attr: np.ndarray, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """Overlay an attention heatmap on the original image."""
    # Normalize attr to [0, 255]
    attr_norm = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    attr_uint8 = (attr_norm * 255).astype(np.uint8)

    # Resize to match original if needed
    if attr_uint8.shape[:2] != orig.shape[:2]:
        attr_uint8 = cv2.resize(attr_uint8, (orig.shape[1], orig.shape[0]))

    heatmap = cv2.applyColorMap(attr_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * (orig * 255) + (1 - alpha) * heatmap).astype(np.uint8)
    return overlay


# ================================================================
# ✅ GRAD-CAM
# ================================================================

def generate_gradcam(
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        target_class: int,
) -> torch.Tensor:
    """
    Generate Grad-CAM attribution map.

    Args:
        model: PyTorch model
        input_tensor: Input (B, C, H, W), must be on same device as model
        target_layer: Target conv layer
        target_class: Target class index

    Returns:
        Attribution map (B, 1, H, W)
    """
    model.eval()

    if target_layer is None:
        target_layer = _find_last_conv_layer(model)
        if target_layer is None:
            raise ValueError("No Conv2d layer found for Grad-CAM")

    grad_cam = LayerGradCam(model, target_layer)

    # Ensure input requires grad
    if not input_tensor.requires_grad:
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

    attributions = grad_cam.attribute(input_tensor, target=target_class)
    attribution = LayerAttribution.interpolate(attributions, (224, 224))

    return attribution


def _save_gradcam_grid(
        model: nn.Module,
        dataloader,
        output_dir: str,
        class_names: List[str],
        device: str = "cpu",
        num_samples: int = 8,
) -> Optional[str]:
    """Generate Grad-CAM grid visualization."""
    os.makedirs(output_dir, exist_ok=True)

    target_layer = _find_last_conv_layer(model)
    if target_layer is None:
        print("⚠️ No Conv2d layer found, skipping Grad-CAM")
        return None

    model.eval()
    model.to(device)

    saved = []  # (orig, attr, label)

    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        # ✅ Handle different batch formats
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            images = batch
            labels = torch.zeros(images.size(0), dtype=torch.long)

        img = images[0:1].to(device).requires_grad_(True)
        label = int(labels[0].item()) if hasattr(labels[0], "item") else int(labels[0])

        try:
            attribution = generate_gradcam(model, img, target_layer, label)
            attr_np = attribution.squeeze().detach().cpu().numpy()
            orig = _denormalize(images[0])
            overlay = _overlay_heatmap(orig, attr_np, cv2.COLORMAP_JET, alpha=0.55)
            saved.append((orig, attr_np, overlay, label))
        except Exception as e:
            print(f"⚠️ Grad-CAM failed for sample {i}: {e}")
            continue

    if not saved:
        return None

    # Build grid
    n = len(saved)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = np.asarray(axes).flatten()

    for i, (orig, attr, overlay, label) in enumerate(saved):
        axes[i].imshow(overlay)
        title = class_names[label] if 0 <= label < len(class_names) else f"Class {label}"
        axes[i].set_title(f"Grad-CAM — {title}", fontsize=10)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Grad-CAM Visualizations", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "gradcam.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Grad-CAM saved: {path}")
    return path


# ================================================================
# ✅ INTEGRATED GRADIENTS
# ================================================================

def generate_integrated_gradients(
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
        n_steps: int = 30,
) -> torch.Tensor:
    """Generate Integrated Gradients attribution."""
    model.eval()

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor)

    if not input_tensor.requires_grad:
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
    )

    return attributions


def _save_ig_grid(
        model: nn.Module,
        dataloader,
        output_dir: str,
        class_names: List[str],
        device: str = "cpu",
        num_samples: int = 8,
) -> Optional[str]:
    """Generate Integrated Gradients grid visualization."""
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    model.to(device)

    saved = []

    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            images = batch
            labels = torch.zeros(images.size(0), dtype=torch.long)

        img = images[0:1].to(device).requires_grad_(True)
        label = int(labels[0].item()) if hasattr(labels[0], "item") else int(labels[0])

        try:
            attributions = generate_integrated_gradients(model, img, label)
            # Aggregate channels via abs-sum
            attr_np = attributions.squeeze().detach().cpu().numpy()
            attr_np = np.abs(attr_np).sum(axis=0)
            orig = _denormalize(images[0])
            overlay = _overlay_heatmap(orig, attr_np, cv2.COLORMAP_HOT, alpha=0.55)
            saved.append((orig, attr_np, overlay, label))
        except Exception as e:
            print(f"⚠️ IG failed for sample {i}: {e}")
            continue

    if not saved:
        return None

    n = len(saved)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = np.asarray(axes).flatten()

    for i, (orig, attr, overlay, label) in enumerate(saved):
        axes[i].imshow(overlay)
        title = class_names[label] if 0 <= label < len(class_names) else f"Class {label}"
        axes[i].set_title(f"Int. Gradients — {title}", fontsize=10)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Integrated Gradients Visualizations", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "integrated_gradients.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Integrated Gradients saved: {path}")
    return path


# ================================================================
# ✅ MAIN ENTRY
# ================================================================

def generate_interpretability_report(
        model: nn.Module,
        dataloader,
        output_dir: str,
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
        num_samples: int = 8,
) -> Dict[str, Any]:
    """
    Generate complete Captum interpretability report.

    Args:
        model: PyTorch model (must output class logits for `model(x)`)
        dataloader: DataLoader with (images, labels)
        output_dir: Directory to save visualizations
        device: 'cpu' or 'cuda'
        class_names: List of class names
        num_samples: Number of samples to visualize

    Returns:
        dict with keys: gradcam_path, ig_path, status
    """
    if not HAS_CAPTUM:
        return {
            "gradcam_path": None,
            "ig_path": None,
            "status": "captum_unavailable",
        }

    if class_names is None:
        class_names = ["NL", "G1", "G2", "G3"]

    os.makedirs(output_dir, exist_ok=True)

    print("\n🔬 Generating Captum interpretability report...")

    gradcam_path = None
    ig_path = None

    try:
        gradcam_path = _save_gradcam_grid(
            model, dataloader, output_dir, class_names, device, num_samples
        )
    except Exception as e:
        print(f"⚠️ Grad-CAM generation failed: {e}")

    try:
        ig_path = _save_ig_grid(
            model, dataloader, output_dir, class_names, device, num_samples
        )
    except Exception as e:
        print(f"⚠️ Integrated Gradients generation failed: {e}")

    status = "success" if (gradcam_path or ig_path) else "failed"

    return {
        "gradcam_path": gradcam_path,
        "ig_path": ig_path,
        "status": status,
    }


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Captum Reporter")
    print("=" * 80)

    if not HAS_CAPTUM:
        print("⚠️ Captum not installed. Skipping test.")
    else:
        # Dummy model + data
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 4)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x).flatten(1)
                return self.fc(x)


        from torch.utils.data import DataLoader, TensorDataset

        model = Dummy()
        imgs = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 4, (8,))
        loader = DataLoader(TensorDataset(imgs, labels), batch_size=1)

        result = generate_interpretability_report(
            model, loader, "test_captum", device="cpu"
        )
        print(f"\n📊 Result: {result}")