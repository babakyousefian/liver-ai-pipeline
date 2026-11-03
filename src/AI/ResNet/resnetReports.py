# src/AI/ResNet/resnetReports.py

"""
ResNet Reports Generator - Professional Version (ULTIMATE EDITION)
This module is designed to generate visual and PDF reports of ResNet model results
Each graph is placed on a separate page for best display
Reports are stored in: finalReports/AI_Comparisons/ResNet/reports/

Features:
- Professional color palettes
- High-resolution figures (300 DPI)
- Comprehensive metrics tables
- Model configuration display
- Page numbers and headers
- Multiple chart types
- Grad-CAM visualization
- Training history plots
- Comparison reports
- Cross-Validation results
- JSON export
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import json
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# =====================================================
# ✅ Try importing PyMuPDF with fallback
# =====================================================
try:
    import fitz

    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("⚠️ PyMuPDF (fitz) not installed. PDF generation will use JSON fallback.")
    print("   Install with: pip install PyMuPDF")

# =====================================================
# ✅ Try importing Grad-CAM
# =====================================================
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("⚠️ pytorch_grad_cam not installed. Grad-CAM will be skipped.")
    print("   Install with: pip install pytorch-grad-cam")

# =====================================================
# ✅ PROFESSIONAL COLOR PALETTES 🔥
# =====================================================
CLASS_NAMES = ["NL", "G1", "G2", "G3"]

# Modern, professional color palette
COLORS = {
    "primary": "#2c3e50",
    "secondary": "#3498db",
    "success": "#2ecc71",
    "danger": "#e74c3c",
    "warning": "#f39c12",
    "info": "#9b59b6",
    "light": "#ecf0f1",
    "dark": "#2c3e50",
}

CLASS_COLORS = [
    "#2ecc71",  # NL - Green
    "#3498db",  # G1 - Blue
    "#9b59b6",  # G2 - Purple
    "#e74c3c",  # G3 - Red
]

# ✅ Normalization stats for denormalization
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]

# ✅ Professional style settings
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)

# =====================================================
# ✅ Report storage path based on tree structure
# =====================================================
from .utils import OUTPUT_DIR
BASE_DIR = OUTPUT_DIR
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# =====================================================
# ✅ DENORMALIZE FUNCTION
# =====================================================
def denormalize_image(tensor, mean=RESNET_MEAN, std=RESNET_STD):
    """
    Denormalize image tensor for visualization

    Args:
        tensor: (C, H, W) tensor
        mean: mean values for denormalization
        std: std values for denormalization

    Returns:
        numpy array: (H, W, C) in [0, 1] range
    """
    img = tensor.clone().detach()

    # Handle single channel
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    # Denormalize
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]

    # Clamp to [0, 1]
    img = torch.clamp(img, 0, 1)

    # Convert to numpy and rearrange
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))

    return img


# =====================================================
# ✅ CONFUSION MATRIX
# =====================================================
def plot_confusion_matrix(
    cm,
    save_path="cm.png",
    class_names=None,
    title="Confusion Matrix",
    figsize=(10, 8),
    cmap="Blues",
):
    """Draw professional confusion matrix"""
    if class_names is None:
        class_names = CLASS_NAMES

    plt.figure(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8},
    )

    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold", labelpad=10)
    plt.ylabel("True Label", fontsize=14, fontweight="bold", labelpad=10)
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Confusion matrix saved: {save_path}")
    return save_path


# =====================================================
# ✅ CROSS-VALIDATION RESULTS PLOT 🔥 NEW
# =====================================================
def plot_cv_results(
    cv_scores: Dict[str, List[float]],
    save_path: str = "cv_results.png",
    title: str = "Cross-Validation Results",
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot cross-validation results as boxplot

    Args:
        cv_scores: Dictionary with metric names and lists of scores
        save_path: Output path for the plot
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    metric_names = list(cv_scores.keys())
    data = [cv_scores[name] for name in metric_names]

    # Create boxplot
    bp = plt.boxplot(data, patch_artist=True, widths=0.6)

    # Color the boxes
    colors_box = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors_box[: len(metric_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points (jittered)
    for i, scores in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(scores))
        plt.plot(x, scores, "o", color="#2c3e50", alpha=0.5, markersize=6)

    plt.xticks(
        range(1, len(metric_names) + 1),
        [m.replace("_", " ").title() for m in metric_names],
        fontsize=12,
    )
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(0, 1.05)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ CV results plot saved: {save_path}")
    return save_path


# =====================================================
# ✅ CROSS-VALIDATION TABLE 🔥 NEW
# =====================================================
def create_cv_table(cv_scores: Dict[str, List[float]]) -> str:
    """
    Create a formatted CV results table

    Args:
        cv_scores: Dictionary with metric names and lists of scores

    Returns:
        Formatted table string
    """
    if not cv_scores:
        return "No Cross-Validation results available."

    n_folds = len(next(iter(cv_scores.values())))

    table = "╔" + "═" * 80 + "╗\n"
    table += "║" + " " * 25 + "CROSS-VALIDATION RESULTS (5-Fold)" + " " * 22 + "║\n"
    table += "╠" + "═" * 80 + "╣\n"

    # Header
    header = "║  Fold   "
    for metric in cv_scores.keys():
        header += f"│ {metric.replace('_', ' ').title():>12} "
    header += "║\n"
    table += header
    table += "╠" + "═" * 80 + "╣\n"

    # Data rows
    for fold in range(n_folds):
        row = f"║  {fold+1:>4}   "
        for metric, scores in cv_scores.items():
            row += f"│ {scores[fold]:>12.4f} "
        row += "║\n"
        table += row

    table += "╠" + "═" * 80 + "╣\n"

    # Mean and Std
    mean_row = f"║  Mean   "
    std_row = f"║  Std    "
    for metric, scores in cv_scores.items():
        mean_row += f"│ {np.mean(scores):>12.4f} "
        std_row += f"│ {np.std(scores):>12.4f} "
    mean_row += "║\n"
    std_row += "║\n"
    table += mean_row
    table += std_row

    table += "╚" + "═" * 80 + "╝"

    return table


# =====================================================
# ✅ METRICS BAR CHART
# =====================================================
def plot_metrics_bar(metrics, save_path="bar.png", title="Performance Metrics", figsize=(10, 6)):
    """Draw professional metrics bar chart"""
    precision = metrics.get("weighted avg", {}).get("precision", 0)
    recall = metrics.get("weighted avg", {}).get("recall", 0)
    f1_score = metrics.get("weighted avg", {}).get("f1-score", 0)
    accuracy = metrics.get("accuracy", 0)

    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [accuracy, precision, recall, f1_score]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12"]

    plt.figure(figsize=figsize)

    bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=1.5, alpha=0.85)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    plt.ylim(0, 1.1)
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Metrics bar chart saved: {save_path}")
    return save_path


# =====================================================
# ✅ PER-CLASS METRICS
# =====================================================
def plot_per_class_metrics(
    metrics, save_path="per_class.png", title="Per-Class Metrics", figsize=(12, 7)
):
    """Draw professional per-class metrics bar chart"""
    class_names = CLASS_NAMES
    precision_list = []
    recall_list = []
    f1_list = []

    for cls in class_names:
        if cls in metrics:
            precision_list.append(metrics[cls]["precision"])
            recall_list.append(metrics[cls]["recall"])
            f1_list.append(metrics[cls]["f1-score"])

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=figsize)

    bars1 = plt.bar(
        x - width,
        precision_list,
        width,
        label="Precision",
        color="#3498db",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
    )
    bars2 = plt.bar(
        x,
        recall_list,
        width,
        label="Recall",
        color="#2ecc71",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
    )
    bars3 = plt.bar(
        x + width,
        f1_list,
        width,
        label="F1-Score",
        color="#9b59b6",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
    )

    for bars, values in [
        (bars1, precision_list),
        (bars2, recall_list),
        (bars3, f1_list),
    ]:
        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.xlabel("Classes", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xticks(x, class_names, fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Per-class metrics saved: {save_path}")
    return save_path


# =====================================================
# ✅ PIE CHART
# =====================================================
def plot_pie(metrics, save_path="pie.png", title="Performance Distribution", figsize=(8, 8)):
    """Draw professional pie chart"""
    precision = metrics.get("weighted avg", {}).get("precision", 0)
    recall = metrics.get("weighted avg", {}).get("recall", 0)
    f1_score = metrics.get("weighted avg", {}).get("f1-score", 0)

    values = [precision, recall, f1_score]
    labels = ["Precision", "Recall", "F1-Score"]
    colors = ["#3498db", "#2ecc71", "#9b59b6"]
    explode = (0.05, 0.05, 0.05)

    plt.figure(figsize=figsize)

    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({pct/100:.3f})",
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(12)

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Pie chart saved: {save_path}")
    return save_path


# =====================================================
# ✅ ROC CURVE
# =====================================================
def plot_roc(
    y_true,
    y_prob,
    save_path="roc.png",
    class_names=None,
    title="ROC Curves",
    figsize=(10, 8),
):
    """Draw professional ROC curves"""
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)

    try:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        plt.figure(figsize=figsize)

        roc_auc_total = {}

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            roc_auc_total[i] = roc_auc

            plt.plot(
                fpr,
                tpr,
                color=CLASS_COLORS[i % len(CLASS_COLORS)],
                label=f"{class_names[i]} (AUC = {roc_auc:.4f})",
                linewidth=2.5,
            )

        plt.plot(
            [0, 1],
            [0, 1],
            "k--",
            label="Random (AUC = 0.5)",
            linewidth=2,
            alpha=0.7,
            color="#7f8c8d",
        )

        plt.xlabel("False Positive Rate", fontsize=14, fontweight="bold")
        plt.ylabel("True Positive Rate", fontsize=14, fontweight="bold")
        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        auc_score = np.mean(list(roc_auc_total.values())) if roc_auc_total else 0.0

        print(f"✅ ROC curves saved: {save_path}")
        print(f"   Average AUC: {auc_score:.4f}")

        return auc_score, save_path

    except Exception as e:
        print(f"⚠️ ROC plot error: {e}")

        plt.figure(figsize=(10, 8))
        plt.text(
            0.5,
            0.5,
            f"ROC Curve Error\n{str(e)}",
            ha="center",
            va="center",
            fontsize=14,
            color="red",
        )
        plt.title("ROC Curve - Error", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        return 0.0, save_path


# =====================================================
# ✅ GRAD-CAM VISUALIZATION - FIXED 🔥
# =====================================================
def generate_gradcam(model, image_tensor, save_path="gradcam.png", target_layer=None):
    """
    Generate Grad-CAM visualization with proper denormalization

    Args:
        model: PyTorch model
        image_tensor: (C, H, W) tensor (normalized or raw)
        save_path: output path
        target_layer: target layer for Grad-CAM

    Returns:
        str: save path
    """

    # ✅ First, denormalize image for visualization
    try:
        img_vis = denormalize_image(image_tensor)
    except:
        img = image_tensor.clone().detach().cpu().numpy()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        img_vis = np.clip(img, 0, 1)

    # ✅ If model is None or Grad-CAM not available
    if model is None or not HAS_GRADCAM:
        print("⚠️ Grad-CAM not available. Saving sample image...")

        plt.figure(figsize=(10, 10))
        if img_vis.ndim == 2:
            plt.imshow(img_vis, cmap="gray")
        else:
            plt.imshow(img_vis)
        plt.axis("off")
        plt.title("Sample Image (Grad-CAM unavailable)", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Sample image saved: {save_path}")
        return save_path

    # ✅ Find target layer for ResNet
    if target_layer is None:
        target_layer = find_resnet_target_layer(model)

    if target_layer is None:
        print("⚠️ Could not find target layer for Grad-CAM. Saving sample image...")

        plt.figure(figsize=(10, 10))
        if img_vis.ndim == 2:
            plt.imshow(img_vis, cmap="gray")
        else:
            plt.imshow(img_vis)
        plt.axis("off")
        plt.title("Sample Image (No target layer)", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        return save_path

    try:
        # ✅ Prepare input tensor
        img_tensor = image_tensor.clone().detach()
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(next(model.parameters()).device)

        # ✅ Generate Grad-CAM
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=img_tensor)[0]

        # ✅ Create visualization
        vis = show_cam_on_image(img_vis, grayscale_cam, use_rgb=True)

        plt.figure(figsize=(10, 10))
        plt.imshow(vis)
        plt.axis("off")
        plt.title("Grad-CAM Visualization", fontsize=16, fontweight="bold")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Grad-CAM saved: {save_path}")
        return save_path

    except Exception as e:
        print(f"⚠️ Grad-CAM error: {e}")

        # ✅ Fallback: save sample image
        plt.figure(figsize=(10, 10))
        if img_vis.ndim == 2:
            plt.imshow(img_vis, cmap="gray")
        else:
            plt.imshow(img_vis)
        plt.axis("off")
        plt.title("Sample Image (Grad-CAM failed)", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        return save_path


def find_resnet_target_layer(model):
    """✅ Find appropriate target layer for ResNet"""

    target_layer = None

    # ✅ Try to find layer4 (last ResNet block)
    if hasattr(model, "layer4"):
        try:
            # Try to get the last module of layer4
            if hasattr(model.layer4, "[-1]"):
                target_layer = model.layer4[-1]
            else:
                # Find the last conv layer in layer4
                for name, module in reversed(list(model.layer4.named_modules())):
                    if isinstance(module, torch.nn.modules.conv.Conv2d):
                        target_layer = module
                        break
                if target_layer is None:
                    target_layer = model.layer4
        except:
            target_layer = model.layer4

    # ✅ If not found, try the whole model
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                target_layer = module
                break

    # ✅ If still None, try to find any layer with 'layer4' in name
    if target_layer is None:
        for name, module in model.named_modules():
            if "layer4" in name and isinstance(module, torch.nn.Module):
                target_layer = module
                break

    return target_layer


# =====================================================
# ✅ TRAINING HISTORY PLOT
# =====================================================
def plot_training_history(
    history, save_path="history.png", title="Training History", figsize=(12, 6)
):
    """Draw professional training history plots"""
    plt.figure(figsize=figsize)

    # Loss plot
    plt.subplot(1, 2, 1)

    if isinstance(history, dict):
        if "train_loss" in history and history["train_loss"]:
            plt.plot(
                history["train_loss"],
                label="Train Loss",
                color="#e74c3c",
                linewidth=2.5,
            )
        if "val_loss" in history and history["val_loss"]:
            plt.plot(
                history["val_loss"],
                label="Validation Loss",
                color="#3498db",
                linewidth=2.5,
            )
    elif isinstance(history, list) and len(history) > 0:
        # If history is a list of dicts (from trainer)
        train_loss = [h.get("loss", 0) for h in history]
        val_loss = [h.get("val_loss", 0) for h in history]
        if train_loss:
            plt.plot(train_loss, label="Train Loss", color="#e74c3c", linewidth=2.5)
        if val_loss:
            plt.plot(val_loss, label="Validation Loss", color="#3498db", linewidth=2.5)

    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Loss", fontsize=12, fontweight="bold")
    plt.title("Loss History", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle="--")

    # Accuracy plot
    plt.subplot(1, 2, 2)

    if isinstance(history, dict):
        if "val_acc" in history and history["val_acc"]:
            plt.plot(
                history["val_acc"],
                label="Validation Accuracy",
                color="#2ecc71",
                linewidth=2.5,
            )
    elif isinstance(history, list) and len(history) > 0:
        val_acc = [h.get("val_accuracy", 0) for h in history]
        if val_acc:
            plt.plot(val_acc, label="Validation Accuracy", color="#2ecc71", linewidth=2.5)

    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
    plt.title("Accuracy History", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Training history saved: {save_path}")
    return save_path


# =====================================================
# ✅ METRICS TABLE
# =====================================================
def create_metrics_table(metrics, class_names=None) -> str:
    """
    Create a formatted metrics table

    Args:
        metrics: Classification metrics dict
        class_names: List of class names

    Returns:
        Formatted table string
    """
    if class_names is None:
        class_names = CLASS_NAMES

    table = "╔" + "═" * 70 + "╗\n"
    table += "║" + " " * 25 + "PERFORMANCE METRICS" + " " * 25 + "║\n"
    table += "╠" + "═" * 70 + "╣\n"

    # Overall metrics
    table += f"║ {'Accuracy':<20} : {metrics.get('accuracy', 0):.4f} {' ' * 38}║\n"

    weighted_avg = metrics.get("weighted avg", {})
    table += (
        f"║ {'Precision (Weighted)':<20} : {weighted_avg.get('precision', 0):.4f} {' ' * 33}║\n"
    )
    table += f"║ {'Recall (Weighted)':<20} : {weighted_avg.get('recall', 0):.4f} {' ' * 34}║\n"
    table += f"║ {'F1-Score (Weighted)':<20} : {weighted_avg.get('f1-score', 0):.4f} {' ' * 33}║\n"

    table += "╠" + "═" * 70 + "╣\n"
    table += "║" + " " * 22 + "PER-CLASS METRICS" + " " * 22 + "║\n"
    table += "╠" + "═" * 70 + "╣\n"

    # Per-class metrics
    header = f"║ {'Class':<8} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Support':>8} ║\n"
    table += header
    table += "║" + "─" * 70 + "║\n"

    for cls in class_names:
        if cls in metrics:
            m = metrics[cls]
            table += f"║ {cls:<8} | {m['precision']:>10.4f} | {m['recall']:>10.4f} | {m['f1-score']:>10.4f} | {m['support']:>8} ║\n"

    table += "╚" + "═" * 70 + "╝"

    return table


# =====================================================
# ✅ PDF REPORT GENERATION - با پشتیبانی از CV 🔥
# =====================================================
def generate_full_pdf(
    metrics: Dict[str, Any],
    cm: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model: Optional[torch.nn.Module],
    sample_image_tensor: torch.Tensor,
    output_path: Optional[str] = None,
    model_name: str = "ResNet",
    history: Optional[Dict[str, List[float]]] = None,
    class_names: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    cv_scores: Optional[Dict[str, List[float]]] = None,  # ✅ NEW
    best_accuracy: float = None,  # ✅ NEW: بهترین دقت Validation
    best_accuracy_label: str = "Validation", # ✅ NEW
    test_accuracy: float = None,  # ✅ NEW
) -> str:
    """
    Generate full PDF report with all visualizations (ULTIMATE EDITION)

    Args:
        metrics: classification metrics dict
        cm: confusion matrix
        y_true: true labels
        y_prob: prediction probabilities
        model: PyTorch model (for Grad-CAM)
        sample_image_tensor: sample image for Grad-CAM
        output_path: output path for PDF
        model_name: name of the model
        history: training history
        class_names: list of class names
        config: model configuration
        cv_scores: cross-validation results (NEW)

    Returns:
        str: output path
    """

    if class_names is None:
        class_names = CLASS_NAMES

    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, f"{model_name}_Report.pdf")

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    # ✅ Create temp directory for plots
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_plots")
    os.makedirs(temp_dir, exist_ok=True)

    # Plot paths
    cm_path = os.path.join(temp_dir, "cm.png")
    bar_path = os.path.join(temp_dir, "bar.png")
    per_class_path = os.path.join(temp_dir, "per_class.png")
    pie_path = os.path.join(temp_dir, "pie.png")
    roc_path = os.path.join(temp_dir, "roc.png")
    gradcam_path = os.path.join(temp_dir, "gradcam.png")
    history_path = os.path.join(temp_dir, "history.png")
    cv_path = os.path.join(temp_dir, "cv_results.png")  # ✅ NEW

    # ✅ Generate plots
    plot_confusion_matrix(cm, cm_path, class_names, f"{model_name} - Confusion Matrix")
    plot_metrics_bar(metrics, bar_path, f"{model_name} - Performance Metrics")
    plot_per_class_metrics(metrics, per_class_path, f"{model_name} - Per-Class Metrics")
    plot_pie(metrics, pie_path, f"{model_name} - Performance Distribution")
    auc_score, _ = plot_roc(y_true, y_prob, roc_path, class_names, f"{model_name} - ROC Curves")

    # ✅ Generate Grad-CAM
    print(f"\n📸 Generating Grad-CAM for {model_name}...")
    generate_gradcam(model, sample_image_tensor, gradcam_path)

    if history:
        plot_training_history(history, history_path, f"{model_name} - Training History")

    # ✅ Generate CV results plot if available
    cv_plot_path = None
    if cv_scores and len(cv_scores) > 0:
        print(f"\n📊 Generating Cross-Validation results plot...")
        cv_plot_path = plot_cv_results(
            cv_scores, cv_path, f"{model_name} - Cross-Validation Results"
        )

    # ✅ Extract metrics
    precision = metrics.get("weighted avg", {}).get("precision", 0)
    recall = metrics.get("weighted avg", {}).get("recall", 0)
    f1_score = metrics.get("weighted avg", {}).get("f1-score", 0)
    accuracy = metrics.get("accuracy", 0)

    # ✅ Create metrics table
    metrics_table = create_metrics_table(metrics, class_names)

    # ✅ Create CV table if available
    cv_table = ""
    if cv_scores and len(cv_scores) > 0:
        cv_table = create_cv_table(cv_scores)

    # ================================================
    # ✅ PDF Generation
    # ================================================

    if HAS_FITZ:
        doc = fitz.open()

        # ============================================
        # Page 1: Title and Summary
        # ============================================
        page = doc.new_page(width=595, height=842)

        # Title
        page.insert_text(
            (50, 80),
            f"{model_name} - Liver AI Report",
            fontsize=24,
            fontname="helvetica-bold",
        )

        # Date
        page.insert_text(
            (50, 120),
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=13,
            fontname="helvetica",
        )

        # Model info
        if config:
            config_text = json.dumps(config, indent=2)
            page.insert_text(
                (50, 150),
                f"Model Configuration:",
                fontsize=11,
                fontname="helvetica-bold",
            )
            page.insert_text((50, 175), config_text[:500], fontsize=9, fontname="helvetica")

        # ✅ Executive Summary
        accuracy = metrics.get('accuracy', 0)
        f1_score = metrics.get('weighted avg', {}).get('f1-score', 0)
    
        # ✅ استفاده از test_accuracy اگر ارسال شده باشه
        test_acc = test_accuracy if test_accuracy is not None else accuracy
    
        if best_accuracy is not None:
            summary_text = f"""
    ====================================================================
                            EXECUTIVE SUMMARY
    ====================================================================

    ✅ Model: {model_name}
    ✅ Best {best_accuracy_label} Accuracy:  {best_accuracy:.4f} ({best_accuracy*100:.2f}%)
    ✅ Test Accuracy:             {test_acc:.4f} ({test_acc*100:.2f}%)
    ✅ Best F1-Score:             {f1_score:.4f} ({f1_score*100:.2f}%)
    ✅ Best AUC Score:            {auc_score:.4f} ({auc_score*100:.2f}%)

    📌 Note: {best_accuracy_label} accuracy is the best achieved during training,
            while test accuracy is the true performance on unseen data.
    """
        else:
            summary_text = f"""
    ====================================================================
                            EXECUTIVE SUMMARY
    ====================================================================

    ✅ Model: {model_name}
    ✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
    ✅ F1-Score:  {f1_score:.4f} ({f1_score*100:.2f}%)
    ✅ AUC Score: {auc_score:.4f} ({auc_score*100:.2f}%)
    """        
            
        page.insert_text((50, 230), summary_text, fontsize=11)

        # Metrics table
        page.insert_text((50, 380), metrics_table, fontsize=9)

        # Footer
        page.insert_text(
            (50, 820),
            f"Page 1 • Report Generated by Liver AI Pipeline • {datetime.now().strftime('%Y-%m-%d')}",
            fontsize=8,
            fontname="helvetica",
        )

        # ============================================
        # Pages 2-8: Charts
        # ============================================
        chart_pages = [
            ("CONFUSION MATRIX", cm_path),
            ("PERFORMANCE METRICS", bar_path),
            ("PER-CLASS METRICS", per_class_path),
            ("ROC CURVES", roc_path),
            ("PERFORMANCE DISTRIBUTION", pie_path),
            ("GRAD-CAM VISUALIZATION", gradcam_path),
        ]

        page_num = 2
        total_pages = (
            1
            + len(chart_pages)
            + (1 if history and os.path.exists(history_path) else 0)
            + (1 if cv_plot_path and os.path.exists(cv_plot_path) else 0)
            + 1
        )

        for title, img_path in chart_pages:
            if os.path.exists(img_path):
                page = doc.new_page(width=595, height=842)
                page.insert_text((50, 50), title, fontsize=16, fontname="helvetica-bold")
                page.insert_image(fitz.Rect(50, 80, 545, 750), filename=img_path)
                page.insert_text(
                    (50, 820),
                    f"Page {page_num}/{total_pages} • Report Generated by Liver AI Pipeline • {datetime.now().strftime('%Y-%m-%d')}",
                    fontsize=8,
                    fontname="helvetica",
                )
                page_num += 1

        # ============================================
        # Training History Page
        # ============================================
        if history and os.path.exists(history_path):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), "TRAINING HISTORY", fontsize=16, fontname="helvetica-bold")
            page.insert_image(fitz.Rect(50, 80, 545, 750), filename=history_path)
            page.insert_text(
                (50, 820),
                f"Page {page_num}/{total_pages} • Report Generated by Liver AI Pipeline • {datetime.now().strftime('%Y-%m-%d')}",
                fontsize=8,
                fontname="helvetica",
            )
            page_num += 1

        # ============================================
        # Cross-Validation Results Page (NEW)
        # ============================================
        if cv_plot_path and os.path.exists(cv_plot_path):
            page = doc.new_page(width=595, height=842)
            page.insert_text(
                (50, 50),
                "CROSS-VALIDATION RESULTS",
                fontsize=16,
                fontname="helvetica-bold",
            )
            page.insert_image(fitz.Rect(50, 80, 545, 750), filename=cv_plot_path)

            # Add CV table
            if cv_table:
                page.insert_text((50, 780), cv_table, fontsize=8)

            page.insert_text(
                (50, 820),
                f"Page {page_num}/{total_pages} • Report Generated by Liver AI Pipeline • {datetime.now().strftime('%Y-%m-%d')}",
                fontsize=8,
                fontname="helvetica",
            )
            page_num += 1

        # ============================================
        # Last Page: Detailed Results
        # ============================================
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), "DETAILED RESULTS", fontsize=16, fontname="helvetica-bold")

        detailed_text = f"""
====================================================================
MODEL: {model_name}
====================================================================

✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
✅ Precision: {precision:.4f} ({precision*100:.2f}%)
✅ Recall:    {recall:.4f} ({recall*100:.2f}%)
✅ F1 Score:  {f1_score:.4f} ({f1_score*100:.2f}%)
✅ AUC Score: {auc_score:.4f} ({auc_score*100:.2f}%)

====================================================================
CLASSIFICATION REPORT
====================================================================
"""

        for cls in class_names:
            if cls in metrics:
                detailed_text += f"""
{cls}:
  Precision: {metrics[cls]['precision']:.4f} ({metrics[cls]['precision']*100:.2f}%)
  Recall:    {metrics[cls]['recall']:.4f} ({metrics[cls]['recall']*100:.2f}%)
  F1-Score:  {metrics[cls]['f1-score']:.4f} ({metrics[cls]['f1-score']*100:.2f}%)
  Support:   {metrics[cls]['support']}
"""

        detailed_text += f"""
====================================================================
CONFUSION MATRIX
====================================================================
"""

        for i, cls in enumerate(class_names):
            detailed_text += f"\n    {cls:>6}: "
            for j in range(len(class_names)):
                detailed_text += f"{cm[i][j]:5d} "

        # ✅ Add CV results to detailed page
        if cv_scores and len(cv_scores) > 0:
            detailed_text += f"""

====================================================================
CROSS-VALIDATION RESULTS
====================================================================
"""
            for metric, scores in cv_scores.items():
                detailed_text += f"\n{metric.replace('_', ' ').title()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}"
                detailed_text += f"\n  Folds: {', '.join([f'{s:.4f}' for s in scores])}"

        if config:
            detailed_text += f"""

====================================================================
MODEL CONFIGURATION
====================================================================
{json.dumps(config, indent=2)}
"""

        detailed_text += """

====================================================================
                  Report Generated by Liver AI Pipeline
====================================================================
"""

        page.insert_text((50, 100), detailed_text, fontsize=10)
        page.insert_text(
            (50, 820),
            f"Page {page_num}/{total_pages} • Report Generated by Liver AI Pipeline • {datetime.now().strftime('%Y-%m-%d')}",
            fontsize=8,
            fontname="helvetica",
        )

        # ✅ Save PDF
        doc.save(output_path)
        doc.close()

        print(f"✅ PDF report saved: {output_path}")
        print(f"   Total pages: {total_pages}")

    else:
        # ✅ Fallback: Save as JSON
        print("⚠️ PyMuPDF not available. Saving report as JSON...")

        json_path = output_path.replace(".pdf", ".json")
        report = {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "auc_score": auc_score,
            "metrics": metrics,
            "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
            "class_names": class_names,
            "config": config,
            "cv_scores": cv_scores,  # ✅ NEW
            "generated_at": datetime.now().isoformat(),
        }

        with open(json_path, "w") as f:
            json.dump(report, f, indent=4, default=str)

        print(f"✅ Report saved as JSON: {json_path}")

    # ✅ Cleanup temp directory
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except:
        pass

    return output_path


# =====================================================
# ✅ COMPARISON REPORT
# =====================================================
def generate_comparison_pdf(
    comparison_data: Dict[str, Any],
    output_path: Optional[str] = None,
    title: str = "Model Comparison Report",
) -> str:
    """
    Generate comparison report between models (IMPROVED)

    Args:
        comparison_data: dict with model comparison data
        output_path: output path
        title: report title

    Returns:
        str: output path
    """

    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, "Comparison_Report.pdf")

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    # ✅ Validate comparison data
    if "model_comparison" not in comparison_data:
        print("⚠️ Invalid comparison data structure. Creating default...")
        comparison_data = {
            "model_comparison": {
                "ResNet18": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "auc_score": 0.0,
                },
                "ResNet50": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "auc_score": 0.0,
                },
            },
            "conclusion": {
                "better_model": "ResNet18",
                "accuracy_improvement": 0.0,
                "recommendation": "ResNet18",
            },
        }

    if HAS_FITZ:
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)

        # Title
        page.insert_text((50, 50), title, fontsize=24, fontname="helvetica-bold")
        page.insert_text(
            (50, 90),
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=13,
            fontname="helvetica",
        )

        y_pos = 150
        page.insert_text(
            (50, y_pos),
            "MODEL COMPARISON RESULTS",
            fontsize=16,
            fontname="helvetica-bold",
        )
        y_pos += 35

        # Table header
        page.insert_text((50, y_pos), "Metric", fontsize=13, fontname="helvetica-bold")
        page.insert_text((200, y_pos), "ResNet18", fontsize=13, fontname="helvetica-bold")
        page.insert_text((350, y_pos), "ResNet50", fontsize=13, fontname="helvetica-bold")
        page.insert_text((500, y_pos), "Difference", fontsize=13, fontname="helvetica-bold")
        page.insert_text((620, y_pos), "Status", fontsize=13, fontname="helvetica-bold")

        y_pos += 15
        page.insert_text((50, y_pos), "-" * 90, fontsize=10)
        y_pos += 25

        # Table rows
        metric_keys = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]

        for label, key in zip(metric_labels, metric_keys):
            val18 = comparison_data["model_comparison"]["ResNet18"].get(key, 0)
            val50 = comparison_data["model_comparison"]["ResNet50"].get(key, 0)
            diff = val50 - val18

            status = "✅ Better" if diff > 0.01 else ("⚠️ Worse" if diff < -0.01 else "➖ Equal")

            page.insert_text((50, y_pos), label, fontsize=11)
            page.insert_text((200, y_pos), f"{val18:.4f}", fontsize=11)
            page.insert_text((350, y_pos), f"{val50:.4f}", fontsize=11)
            page.insert_text((500, y_pos), f"{diff:+.4f}", fontsize=11)
            page.insert_text((620, y_pos), status, fontsize=11)

            y_pos += 28

        # Conclusion
        y_pos += 20
        page.insert_text((50, y_pos), "=" * 90, fontsize=10)
        y_pos += 25

        better_model = comparison_data.get("conclusion", {}).get("recommendation", "ResNet50")
        improvement = comparison_data.get("conclusion", {}).get("accuracy_improvement", 0)

        conclusion_text = f"""
====================================================================
CONCLUSION
====================================================================

Based on the comparison results, {better_model} performs better.

✅ Accuracy Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)

🎯 Recommendation: Use {better_model} for production deployment.

====================================================================
                  Report Generated by Liver AI Pipeline
====================================================================
"""

        page.insert_text((50, y_pos), conclusion_text, fontsize=12)

        # Footer
        page.insert_text(
            (50, 820),
            f"Page 1/1 • Report Generated by Liver AI Pipeline • {datetime.now().strftime('%Y-%m-%d')}",
            fontsize=8,
            fontname="helvetica",
        )

        doc.save(output_path)
        doc.close()

        print(f"✅ Comparison report saved: {output_path}")
        print(f"   Total pages: {len(doc)}")

    else:
        # ✅ Fallback: Save as JSON
        json_path = output_path.replace(".pdf", ".json")
        with open(json_path, "w") as f:
            json.dump(comparison_data, f, indent=4, default=str)
        print(f"✅ Comparison saved as JSON: {json_path}")

    return output_path


# =====================================================
# ✅ EXPORT METRICS TO JSON
# =====================================================
def export_metrics_to_json(
    metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    model_name: str = "ResNet",
) -> Optional[str]:
    """Export metrics to JSON file"""
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, f"{model_name.lower()}_metrics.json")

    try:
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value

        with open(output_path, "w") as f:
            json.dump(metrics_serializable, f, indent=4, default=str)

        print(f"✅ Metrics exported to JSON: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error exporting metrics: {e}")
        return None


# =====================================================
# ✅ EXPORT COMPARISON TO JSON
# =====================================================
def export_comparison_to_json(
    comparison_data: Dict[str, Any], output_path: Optional[str] = None
) -> Optional[str]:
    """Export comparison data to JSON file"""
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, "model_comparison.json")

    try:
        with open(output_path, "w") as f:
            json.dump(comparison_data, f, indent=4, default=str)

        print(f"✅ Comparison data exported to JSON: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error exporting comparison data: {e}")
        return None


# =====================================================
# ✅ MAIN - FOR TESTING
# =====================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Testing ResNet Reports - ULTIMATE EDITION")
    print("=" * 70)

    try:
        # Test data
        cm = np.array([[100, 10, 5, 2], [8, 90, 12, 3], [5, 15, 85, 8], [2, 5, 10, 70]])

        y_true = np.random.randint(0, 4, 100)
        y_prob = np.random.rand(100, 4)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        metrics = {
            "accuracy": 0.85,
            "weighted avg": {"precision": 0.84, "recall": 0.85, "f1-score": 0.84},
            "NL": {"precision": 0.86, "recall": 0.85, "f1-score": 0.85, "support": 118},
            "G1": {"precision": 0.84, "recall": 0.85, "f1-score": 0.84, "support": 115},
            "G2": {"precision": 0.82, "recall": 0.83, "f1-score": 0.82, "support": 112},
            "G3": {"precision": 0.80, "recall": 0.81, "f1-score": 0.80, "support": 90},
        }

        history = {
            "train_loss": [0.5, 0.4, 0.35, 0.3, 0.25],
            "val_loss": [0.6, 0.5, 0.45, 0.4, 0.35],
            "val_acc": [0.7, 0.75, 0.8, 0.82, 0.85],
        }

        config = {
            "batch_size": 32,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "dropout": 0.15,
            "label_smoothing": 0.1,
            "epochs": 70,
            "optimizer": "AdamW",
            "scheduler": "WarmupCosine",
        }

        # ✅ Test CV data
        cv_scores = {
            "accuracy": [0.82, 0.84, 0.85, 0.83, 0.86],
            "precision": [0.81, 0.83, 0.84, 0.82, 0.85],
            "recall": [0.82, 0.84, 0.85, 0.83, 0.86],
            "f1_score": [0.815, 0.835, 0.845, 0.825, 0.855],
        }

        print("\n📌 Testing PDF report generation with CV...")

        sample_img = torch.randn(3, 256, 256)

        output_path = generate_full_pdf(
            metrics=metrics,
            cm=cm,
            y_true=y_true,
            y_prob=y_prob,
            model=None,
            sample_image_tensor=sample_img,
            model_name="ResNet18",
            history=history,
            config=config,
            cv_scores=cv_scores,  # ✅ NEW
        )

        print(f"\n✅ Report generated: {output_path}")

        print("\n📌 Testing comparison report...")

        comparison_data = {
            "model_comparison": {
                "ResNet18": {
                    "accuracy": 0.85,
                    "precision": 0.84,
                    "recall": 0.85,
                    "f1_score": 0.84,
                    "auc_score": 0.92,
                },
                "ResNet50": {
                    "accuracy": 0.88,
                    "precision": 0.87,
                    "recall": 0.88,
                    "f1_score": 0.87,
                    "auc_score": 0.94,
                },
            },
            "conclusion": {
                "better_model": "ResNet50",
                "accuracy_improvement": 0.03,
                "recommendation": "ResNet50",
            },
        }

        comparison_path = generate_comparison_pdf(comparison_data)
        print(f"✅ Comparison report generated: {comparison_path}")

        print("\n📌 Testing metrics export...")
        export_metrics_to_json(metrics, model_name="ResNet18")

        print("\n📌 Testing comparison export...")
        export_comparison_to_json(comparison_data)

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
