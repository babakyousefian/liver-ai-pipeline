# src/reporter/shap_reporter.py

"""
SHAP Interpretability Reporter (ULTIMATE EDITION)

Uses GradientExplainer for PyTorch models — more robust than DeepExplainer.

✅ ZERO impact on model accuracy
✅ Handles device mismatch
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

warnings.filterwarnings("ignore")

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ SHAP not installed. Install with: pip install shap")


# ================================================================
# ✅ MAIN ENTRY
# ================================================================

def generate_shap_explanation(
    model: torch.nn.Module,
    dataloader,
    output_dir: str,
    device: str = "cpu",
    num_background: int = 50,
    num_explain: int = 10,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate SHAP explanations using GradientExplainer.

    Args:
        model: PyTorch model
        dataloader: DataLoader with (images, labels)
        output_dir: Directory to save visualizations
        device: 'cpu' or 'cuda'
        num_background: Number of background samples
        num_explain: Number of samples to explain
        class_names: Class names

    Returns:
        dict with keys: shap_path, status, error (optional)
    """
    if not HAS_SHAP:
        return {"shap_path": None, "status": "shap_unavailable"}

    if class_names is None:
        class_names = ["NL", "G1", "G2", "G3"]

    os.makedirs(output_dir, exist_ok=True)

    print("\n🔬 Generating SHAP explanation...")

    model.eval()
    model.to(device)

    # ✅ Collect background and explain samples
    background_list = []
    explain_list = []
    labels_list = []

    total_bg = 0
    for i, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            images = batch
            labels = torch.zeros(images.size(0), dtype=torch.long)

        if total_bg < num_background:
            background_list.append(images)
            total_bg += images.size(0)

    # Second pass for explain samples
    total_ex = 0
    for i, batch in enumerate(dataloader):
        if total_ex >= num_explain:
            break
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            images = batch
            labels = torch.zeros(images.size(0), dtype=torch.long)

        explain_list.append(images)
        labels_list.append(labels)
        total_ex += images.size(0)

    if not background_list or not explain_list:
        return {"shap_path": None, "status": "no_data"}

    background = torch.cat(background_list, dim=0)[:num_background].to(device)
    explain_samples = torch.cat(explain_list, dim=0)[:num_explain].to(device)

    try:
        # ✅ Use GradientExplainer — more robust than DeepExplainer
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(explain_samples, nsamples=50)

        # shap_values is a list (one per class) of arrays with shape (N, C, H, W)
        if not isinstance(shap_values, list):
            shap_values = [shap_values]

        shap_path = _save_shap_summary(
            shap_values, explain_samples, output_dir, class_names
        )

        print(f"✅ SHAP saved: {shap_path}")
        return {
            "shap_path": shap_path,
            "status": "success",
            "num_classes": len(shap_values),
        }

    except Exception as e:
        print(f"⚠️ SHAP generation failed: {e}")
        return {
            "shap_path": None,
            "status": "failed",
            "error": str(e),
        }


def _save_shap_summary(
    shap_values: List[np.ndarray],
    samples: torch.Tensor,
    output_dir: str,
    class_names: List[str],
) -> Optional[str]:
    """Save SHAP summary visualization (one panel per class)."""
    n_classes = len(shap_values)
    cols = min(4, n_classes)
    rows = (n_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = np.asarray(axes).flatten()

    for c in range(n_classes):
        sv = shap_values[c]  # (N, C, H, W) or (N, H, W)

        # Handle different shapes
        if sv.ndim == 4:  # (N, C, H, W)
            sv_agg = np.abs(sv).mean(axis=(0, 1))  # (H, W)
        elif sv.ndim == 3:  # (N, H, W)
            sv_agg = np.abs(sv).mean(axis=0)  # (H, W)
        else:
            sv_agg = np.abs(sv)

        # ✅ FIX: اگر sv_agg بیش از 2 بُعد دارد، میانگین بگیر
        if sv_agg.ndim > 2:
            sv_agg = np.mean(sv_agg, axis=-1) if sv_agg.ndim == 3 else sv_agg.reshape(sv_agg.shape[0], -1)

        # ✅ FIX: اگر هنوز بیش از 2 بُعد دارد، reshape کن
        if sv_agg.ndim > 2:
            sv_agg = sv_agg.reshape(sv_agg.shape[0], -1)

        # Normalize
        sv_agg = (sv_agg - sv_agg.min()) / (sv_agg.max() - sv_agg.min() + 1e-8)

        # Resize
        if sv_agg.shape[:2] != (224, 224):
            sv_agg = cv2.resize(sv_agg.astype(np.float32), (224, 224))

        axes[c].imshow(sv_agg, cmap="hot")
        name = class_names[c] if c < len(class_names) else f"Class {c}"
        axes[c].set_title(f"SHAP — {name}", fontsize=11)
        axes[c].axis("off")

    for i in range(n_classes, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        "SHAP Feature Importance (Mean |SHAP| per Class)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    return path


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing SHAP Reporter")
    print("=" * 80)

    if not HAS_SHAP:
        print("⚠️ SHAP not installed. Skipping test.")
    else:
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

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

        model = Dummy()
        imgs = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 4, (20,))
        loader = DataLoader(TensorDataset(imgs, labels), batch_size=4)

        result = generate_shap_explanation(
            model, loader, "test_shap", device="cpu",
            num_background=8, num_explain=4,
        )
        print(f"\n📊 Result: {result}")