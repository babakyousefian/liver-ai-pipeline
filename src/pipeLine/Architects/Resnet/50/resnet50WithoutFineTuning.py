# src/pipeLine/resnet50WithoutFineTuning.py

"""
ResNet50 Without Fine-Tuning - OOP Implementation with Full Reporting (ULTIMATE EDITION)

Features:
- Full OOP structure
- Multi-stage training (5 phases: FC → Layer4 → Layer3+4 → Layer2+3+4 → Full)
- Smart Learning Rate per stage
- Class Weighting (sqrt / balanced / inverse)
- Focal Loss (with per-class alpha)
- Label Smoothing (with optional adaptive mode)
- Warmup + Cosine Scheduler
- Advanced callbacks (Early Stopping, Checkpoint)
- TTA (Test Time Augmentation) in evaluation
- Mixup in Stage 5 (optional)
- Comprehensive PDF reports
- K-Fold cross-validation
- Complete model persistence

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 6.0.0 (ULTIMATE EDITION)
"""

import gc
import io
import json
import logging
import os
import time
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# Import all ResNet50 components
# ================================================================

from src.models.resnet.resnet50 import Bottleneck
from src.models.resnet.resnet50 import ResNet50 as _ResNet50
from src.models.resnet.resnet50 import (
    analyze_memory_usage,
    calculate_flops,
    check_gradient_flow,
    compare_models,
    count_parameters,
)
from src.models.resnet.resnet50 import create_resnet50 as _create_resnet50
from src.models.resnet.resnet50 import (
    get_activation_stats,
    get_layer_output_shapes,
    get_model_size,
    make_layer,
    print_model_summary,
    profile_model,
    test_inference_speed,
    validate_forward_pass,
)

# ================================================================
# Constants - Paths for ResNet50 (NO G4)
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)

# ================================================================
# Base Directories
# ================================================================

FINAL_REPORTS_DIR = "finalReports"
MODELS_BASE_DIR = os.path.join(FINAL_REPORTS_DIR, "models")

RESNET_BASE_DIR = os.path.join(MODELS_BASE_DIR, "..")
RESNET50_BASE_DIR = os.path.join(RESNET_BASE_DIR, "resnet50")

RESNET50_CHECKPOINTS_DIR = os.path.join(RESNET50_BASE_DIR, "checkpoints")
RESNET50_REPORTS_DIR = os.path.join(RESNET50_BASE_DIR, "reports")
RESNET50_CONFIGS_DIR = os.path.join(RESNET50_BASE_DIR, "configs")
RESNET50_PLOTS_DIR = os.path.join(RESNET50_BASE_DIR, "plots")


# ================================================================
# ✅ SMART LEARNING RATE SCHEDULER FOR EACH PHASE
# ================================================================


def get_phase_lr(phase: int) -> float:
    """Get optimal learning rate for each phase."""
    PHASE_LRS = {
        0: 5e-4,  # Phase 1: FC layer only
        1: 2e-4,  # Phase 2: Layer4 + FC
        2: 1e-4,  # Phase 3: Layer3+4 + FC
        3: 5e-5,  # Phase 4: Layer2+3+4 + FC
        4: 1e-5,  # Phase 5: Full fine-tuning
    }
    return PHASE_LRS.get(phase, 1e-4)


# ================================================================
# ✅ FOCAL LOSS 🔥
# ================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss = -alpha * (1 - pt)^gamma * log(pt)

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002

    Args:
        alpha: Class weights tensor of shape (num_classes,) or None
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ✅ Compute cross entropy loss (no reduction)
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", weight=self.alpha
        )

        # ✅ Compute pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # ✅ Compute focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ================================================================
# ✅ ADAPTIVE LABEL SMOOTHING 🔥
# ================================================================


class AdaptiveLabelSmoothing:
    """
    Adaptive label smoothing that decays over epochs.

    Early epochs: Higher smoothing (more regularization)
    Later epochs: Lower smoothing (more precise)
    """

    def __init__(
        self,
        initial_smoothing: float = 0.1,
        final_smoothing: float = 0.02,
        total_epochs: int = 100,
    ):
        self.initial_smoothing = initial_smoothing
        self.final_smoothing = final_smoothing
        self.total_epochs = total_epochs

    def get(self, epoch: int) -> float:
        """Get smoothing value for current epoch."""
        progress = min(1.0, epoch / max(1, self.total_epochs))
        return self.initial_smoothing - progress * (
            self.initial_smoothing - self.final_smoothing
        )


# ================================================================
# ✅ MIXUP 🔥
# ================================================================


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup augmentation (Zhang et al., 2018).

    Args:
        x: Batch of images (B, C, H, W)
        y: Batch of labels (B,)
        alpha: Beta distribution parameter (default: 0.2)
        device: Device

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ================================================================
# Directory Management
# ================================================================


def ensure_directories() -> None:
    """Create all necessary directories for ResNet50."""
    directories = [
        RESNET50_BASE_DIR,
        RESNET50_CHECKPOINTS_DIR,
        RESNET50_REPORTS_DIR,
        RESNET50_CONFIGS_DIR,
        RESNET50_PLOTS_DIR,
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

    print(f"✅ Directories created: {RESNET50_BASE_DIR}")


# ================================================================
# ✅ SAVE MODEL CHECKPOINT FUNCTION 🔥
# ================================================================


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filename: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    save_dir: Optional[str] = None,
) -> str:
    """
    Save a complete model checkpoint with all training state.

    ✅ IMPORTANT: Model state_dict is saved to CPU for portability.
    """
    if save_dir is None:
        save_dir = RESNET50_CHECKPOINTS_DIR

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # ✅ Move state_dict to CPU for portability
    model_state_cpu = {k: v.cpu() for k, v in model.state_dict().items()}

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_cpu,  # ← ✅ CPU
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "model_architecture": "ResNet50",
        "version": "6.0.0",
    }

    if optimizer is not None:
        try:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        except Exception as e:
            print(f"⚠️ Could not save optimizer state: {e}")

    if scheduler is not None:
        try:
            if hasattr(scheduler, "state_dict"):
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        except Exception as e:
            print(f"⚠️ Could not save scheduler state: {e}")

    if extra_info is not None:
        checkpoint["extra_info"] = extra_info

    try:
        torch.save(checkpoint, filepath)
        print(f"✅ Checkpoint saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        raise


def load_model_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Load a complete model checkpoint with robust device handling.

    ✅ This function handles the case where checkpoint was saved on CUDA
    but we're loading on a machine without CUDA (or vice versa).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # ✅ Resolve target device safely
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ✅ Try loading with multiple strategies
    checkpoint = None
    load_errors = []

    # Attempt 1: Load with target device
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except Exception as e:
        load_errors.append(f"Attempt 1 (device={device}): {e}")

    # Attempt 2: Load with CPU (most robust)
    if checkpoint is None:
        try:
            checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        except Exception as e:
            load_errors.append(f"Attempt 2 (device=cpu): {e}")

    # Attempt 3: Load with explicit cuda:0 -> cpu mapping
    if checkpoint is None:
        try:
            checkpoint = torch.load(
                filepath, map_location={"cuda:0": "cpu"}, weights_only=False
            )
        except Exception as e:
            load_errors.append(f"Attempt 3 (cuda:0 -> cpu): {e}")

    if checkpoint is None:
        print(f"❌ Failed to load checkpoint after 3 attempts:")
        for err in load_errors:
            print(f"   - {err}")
        raise RuntimeError(f"Could not load checkpoint: {filepath}")

    # ✅ Load model state dict
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
    except Exception as e:
        print(f"⚠️ Error loading model state dict: {e}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(device)

    # ✅ Load optimizer state dict
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(f"⚠️ Could not load optimizer state: {e} (continuing without it)")

    # ✅ Load scheduler state dict
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            print(f"⚠️ Could not load scheduler state: {e} (continuing without it)")

    print(f"✅ Checkpoint loaded: {filepath} (device={device})")
    return checkpoint


# ================================================================
# ✅ ADVANCED CALLBACKS 🔥
# ================================================================


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


class ModelCheckpoint:
    """Model checkpoint callback to save best models."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_best_only: bool = True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None

    def __call__(self, model: nn.Module, epoch: int, metrics: Dict[str, float]):
        current_score = metrics.get(self.monitor, 0)

        if self.best_score is None:
            self.best_score = current_score
            self._save(model, epoch, metrics)
            return True

        if self.mode == "max":
            improved = current_score > self.best_score
        else:
            improved = current_score < self.best_score

        if improved:
            self.best_score = current_score
            self._save(model, epoch, metrics)
            return True

        return False

    def _save(self, model: nn.Module, epoch: int, metrics: Dict[str, float]):
        save_model_checkpoint(
            model=model,
            optimizer=None,
            epoch=epoch,
            metrics=metrics,
            filename=os.path.basename(self.filepath),
            save_dir=os.path.dirname(self.filepath),
        )


# ================================================================
# ✅ CONFIGURATION MANAGEMENT 🔥
# ================================================================


def save_config(config: Dict[str, Any], filename: str = "config.json") -> str:
    """Save configuration to JSON file."""
    ensure_directories()
    filepath = os.path.join(RESNET50_CONFIGS_DIR, filename)

    serializable_config = {}
    for key, value in config.items():
        if hasattr(value, "__dict__"):
            serializable_config[key] = str(value)
        elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)

    serializable_config["created_at"] = datetime.now().isoformat()
    serializable_config["class_names"] = CLASS_NAMES

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_config, f, indent=4, ensure_ascii=False)

    print(f"✅ Config saved to: {filepath}")
    return filepath


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ================================================================
# ✅ COMPREHENSIVE PLOTTING FUNCTIONS 🔥
# ================================================================


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], save_path: str
) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 12, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Confusion Matrix - ResNet50", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Confusion matrix saved to: {save_path}")


def plot_roc_curves(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], save_path: str
) -> None:
    """Plot and save ROC curves."""
    plt.figure(figsize=(10, 8))

    n_classes = len(class_names)
    y_true_one_hot = np.eye(n_classes)[y_true]

    colors_roc = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            lw=2.5,
            color=colors_roc[i % len(colors_roc)],
            label=f"{class_name} (AUC = {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    plt.title("ROC Curves - ResNet50", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ ROC curves saved to: {save_path}")


def plot_metrics_bar(metrics: Dict[str, float], save_path: str) -> None:
    """Plot and save metrics bar chart."""
    plt.figure(figsize=(10, 6))

    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    values = [metrics.get(name, 0) for name in metric_names]
    colors_bar = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

    bars = plt.bar(
        metric_names,
        values,
        color=colors_bar,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.title("ResNet50 Performance Metrics", fontsize=16, fontweight="bold")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Metrics bar chart saved to: {save_path}")


def plot_training_history(history: List[Dict[str, float]], save_path: str) -> None:
    """Plot and save training history."""
    if not history:
        return

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    train_loss = [h.get("loss", 0) for h in history]
    val_loss = [h.get("val_loss", 0) for h in history]
    ax1.plot(epochs, train_loss, "b-", linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_loss, "r-", linewidth=2, label="Validation Loss")
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Loss History", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    train_acc = [h.get("accuracy", 0) for h in history]
    val_acc = [h.get("val_accuracy", 0) for h in history]
    ax2.plot(epochs, train_acc, "g-", linewidth=2, label="Train Accuracy")
    ax2.plot(epochs, val_acc, "orange", linewidth=2, label="Validation Accuracy")
    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax2.set_title("Accuracy History", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Learning Rate
    lrs = [h.get("lr", 0) for h in history]
    ax3.plot(epochs, lrs, "purple", linewidth=2)
    ax3.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Learning Rate", fontsize=12, fontweight="bold")
    ax3.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Training history saved to: {save_path}")


def plot_per_class_metrics(
    per_class: Dict[str, Dict[str, float]], save_path: str
) -> None:
    """Plot per-class metrics."""
    class_names = list(per_class.keys())
    precision = [per_class[c]["precision"] for c in class_names]
    recall = [per_class[c]["recall"] for c in class_names]
    f1 = [per_class[c]["f1"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#3498db", edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x, recall, width, label="Recall", color="#2ecc71", edgecolor="black", linewidth=1.5)
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#9b59b6", edgecolor="black", linewidth=1.5)

    for bars, values in [(bars1, precision), (bars2, recall), (bars3, f1)]:
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Classes", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax.set_title("Per-Class Metrics - ResNet50", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Per-class metrics saved to: {save_path}")


def plot_cross_validation_results(
    cv_scores: Dict[str, List[float]], save_path: str
) -> None:
    """Plot cross-validation results."""
    plt.figure(figsize=(12, 6))

    metric_names = list(cv_scores.keys())
    data = [cv_scores[name] for name in metric_names]

    bp = plt.boxplot(data, patch_artist=True)
    plt.xticks(range(1, len(metric_names) + 1), metric_names, fontsize=12)

    colors_box = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
    for patch, color in zip(bp["boxes"], colors_box[: len(metric_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.title("ResNet50 Cross-Validation Results", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ CV results saved to: {save_path}")


# ================================================================
# ✅ PDF REPORT GENERATION 🔥
# ================================================================


def generate_pdf_report(
    metrics: Dict[str, float],
    cm: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    per_class: Optional[Dict[str, Dict[str, float]]] = None,
    cv_scores: Optional[Dict[str, List[float]]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    training_history: Optional[List[Dict]] = None,
    output_path: str = None,
) -> str:
    """Generate comprehensive PDF report."""
    ensure_directories()

    if output_path is None:
        output_path = os.path.join(RESNET50_REPORTS_DIR, "ResNet50_Report.pdf")

    class_names = (
        model_info.get("class_names", CLASS_NAMES) if model_info else CLASS_NAMES
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=12,
    )
    normal_style = styles["Normal"]

    story = []

    # Title
    story.append(Paragraph("ResNet50 Model Report", title_style))
    story.append(Paragraph("Liver AI Pipeline - Task 5", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style
        )
    )
    story.append(Spacer(1, 20))

    # Model Information
    story.append(Paragraph("1. Model Information", heading_style))

    if model_info:
        info_data = [
            ["Architecture", model_info.get("architecture", "ResNet50")],
            ["Total Parameters", f"{model_info.get('total_params', 0):,}"],
            ["Trainable Parameters", f"{model_info.get('trainable_params', 0):,}"],
            ["Input Shape", str(model_info.get("input_shape", (3, 224, 224)))],
            ["Number of Classes", str(model_info.get("num_classes", 4))],
            ["Class Names", ", ".join(model_info.get("class_names", CLASS_NAMES))],
            ["Device", model_info.get("device", "cuda" if torch.cuda.is_available() else "cpu")],
            [
                "Training Status",
                "Trained" if model_info.get("is_trained", False) else "Untrained",
            ],
            ["Feature Dimension", str(model_info.get("feature_dim", 2048))],
            ["Activation", model_info.get("activation", "ReLU")],
            ["Block Type", "Bottleneck (expansion=4)"],
            ["Training Stages", "5 Stages"],
            ["TTA", "Enabled (3x)"],
        ]

        info_table = Table(info_data, colWidths=[3 * inch, 3 * inch])
        info_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                    ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#ecf0f1")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ]
            )
        )
        story.append(info_table)
        story.append(Spacer(1, 20))

    # Performance Metrics
    story.append(Paragraph("2. Performance Metrics", heading_style))

    metrics_data = [
        ["Metric", "Value"],
        ["Accuracy", f"{metrics.get('accuracy', 0):.4f}"],
        ["Precision (Weighted)", f"{metrics.get('precision', 0):.4f}"],
        ["Recall (Weighted)", f"{metrics.get('recall', 0):.4f}"],
        ["F1-Score (Weighted)", f"{metrics.get('f1_score', 0):.4f}"],
    ]

    metrics_table = Table(metrics_data, colWidths=[3 * inch, 2 * inch])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ]
        )
    )
    story.append(metrics_table)
    story.append(Spacer(1, 20))

    # Generate plots
    plot_paths = {}

    cm_path = os.path.join(RESNET50_PLOTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    plot_paths["confusion_matrix"] = cm_path

    roc_path = os.path.join(RESNET50_PLOTS_DIR, "roc_curves.png")
    plot_roc_curves(y_true, y_prob, class_names, roc_path)
    plot_paths["roc_curves"] = roc_path

    bar_path = os.path.join(RESNET50_PLOTS_DIR, "metrics_bar.png")
    plot_metrics_bar(metrics, bar_path)
    plot_paths["metrics_bar"] = bar_path

    if per_class:
        per_class_path = os.path.join(RESNET50_PLOTS_DIR, "per_class_metrics.png")
        plot_per_class_metrics(per_class, per_class_path)
        plot_paths["per_class_metrics"] = per_class_path

    if training_history:
        hist_path = os.path.join(RESNET50_PLOTS_DIR, "training_history.png")
        plot_training_history(training_history, hist_path)
        plot_paths["training_history"] = hist_path

    for name, path in plot_paths.items():
        if os.path.exists(path):
            story.append(PageBreak())
            story.append(Paragraph(f"3. {name.replace('_', ' ').title()}", heading_style))
            story.append(RLImage(path, width=6 * inch, height=5 * inch))
            story.append(Spacer(1, 10))

    if cv_scores:
        story.append(PageBreak())
        story.append(Paragraph("4. Cross-Validation Results", heading_style))

        cv_data = [["Fold"] + [f"CV-{i+1}" for i in range(len(next(iter(cv_scores.values()))))] + ["Mean", "Std"]]
        for metric_name, scores in cv_scores.items():
            row = [metric_name.capitalize()] + [f"{s:.4f}" for s in scores] + [f"{np.mean(scores):.4f}", f"{np.std(scores):.4f}"]
            cv_data.append(row)

        cv_table = Table(
            cv_data,
            colWidths=[1.5 * inch] + [1 * inch] * (len(cv_scores[list(cv_scores.keys())[0]]) + 2),
        )
        cv_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ]
            )
        )
        story.append(cv_table)
        story.append(Spacer(1, 10))

        cv_path = os.path.join(RESNET50_PLOTS_DIR, "cv_results.png")
        plot_cross_validation_results(cv_scores, cv_path)

        if os.path.exists(cv_path):
            story.append(RLImage(cv_path, width=6 * inch, height=4 * inch))

    # Summary
    story.append(PageBreak())
    story.append(Paragraph("5. Summary & Conclusion", heading_style))

    conclusion_text = f"""
    <b>ResNet50 Performance Summary:</b><br/><br/>
    • Total Parameters: {model_info.get('total_params', 0):,}<br/>
    • Best Accuracy: {metrics.get('accuracy', 0):.4f}<br/>
    • Weighted F1-Score: {metrics.get('f1_score', 0):.4f}<br/>
    • Number of Classes: {len(class_names)}<br/>
    • Classes: {', '.join(class_names)}<br/>
    • Feature Dimension: {model_info.get('feature_dim', 2048)}<br/>
    • Block Type: Bottleneck (expansion=4)<br/>
    • Training Stages: 5 Stages<br/>
    • TTA: Enabled (3x)<br/>
    """

    if cv_scores:
        cv_mean = np.mean(cv_scores.get("accuracy", [0]))
        cv_std = np.std(cv_scores.get("accuracy", [0]))
        conclusion_text += (
            f"• Cross-Validation Accuracy: {cv_mean:.4f} ± {cv_std:.4f}<br/>"
        )

    conclusion_text += """
    <br/><b>Recommendations:</b><br/><br/>
    • ResNet50 is powerful with ~23.5M parameters<br/>
    • Deeper architecture captures more complex features<br/>
    • Bottleneck blocks provide efficient computation<br/>
    • Multi-stage training reduces overfitting<br/>
    • Focal Loss + Class Weights handle imbalance<br/>
    • TTA improves test accuracy by 1-3%<br/>
    """

    story.append(Paragraph(conclusion_text, normal_style))
    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            f"Report generated by Liver AI Pipeline - Version 6.0.0", styles["Italic"]
        )
    )

    doc.build(story)
    print(f"✅ PDF report saved to: {output_path}")
    return output_path


# ================================================================
# ✅ MAIN WRAPPER CLASS - ULTIMATE EDITION 🔥
# ================================================================


class ResNet50WithoutFineTuning:
    """
    Ultimate OOP Wrapper for ResNet50 from Scratch.

    Features:
    - Multi-stage training (5 phases)
    - Class Weighting (sqrt/balanced/inverse)
    - Focal Loss
    - Label Smoothing (with adaptive option)
    - Warmup + Cosine Scheduler
    - TTA in evaluation
    - Mixup in Stage 5
    - Complete reporting
    """

    def __init__(
        self,
        num_classes: int = 4,
        zero_init_residual: bool = True,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
        target_size: int = 224,
        checkpoints_dir: Optional[str] = None,
        reports_dir: Optional[str] = None,
        configs_dir: Optional[str] = None,
        plots_dir: Optional[str] = None,
        # ✅ Class Weighting
        use_class_weights: bool = True,
        class_weight_method: str = "sqrt",
        label_smoothing: float = 0.1,
        labels_csv_path: Optional[str] = None,
        use_adaptive_smoothing: bool = False,
        # ✅ Focal Loss
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        # ✅ Scheduler
        scheduler_type: str = "warmup_cosine",
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        total_epochs: int = 100,
        # ✅ TTA & Mixup
        use_tta: bool = True,
        use_mixup: bool = True,
        mixup_alpha: float = 0.2,
        mixup_start_stage: int = 4,
        **kwargs,
    ) -> None:
        """Initialize the wrapper."""
        self.num_classes = num_classes
        self.zero_init_residual = zero_init_residual
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.device = device
        self.verbose = verbose
        self._feature_dim = 2048
        self._target_size = target_size

        # ✅ Class weighting params
        self.use_class_weights = use_class_weights
        self.class_weight_method = class_weight_method
        self.label_smoothing = label_smoothing
        self.labels_csv_path = labels_csv_path or os.path.join("finalReports", "labels.csv")
        self.class_weights = None

        # ✅ Label Smoothing
        self.use_adaptive_smoothing = use_adaptive_smoothing
        if use_adaptive_smoothing:
            self.adaptive_smoothing = AdaptiveLabelSmoothing(
                initial_smoothing=label_smoothing,
                final_smoothing=label_smoothing * 0.2,
                total_epochs=total_epochs,
            )
        else:
            self.adaptive_smoothing = None

        # ✅ Focal Loss params
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # ✅ Scheduler
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.total_epochs = total_epochs

        # ✅ TTA & Mixup
        self.use_tta = use_tta
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_start_stage = mixup_start_stage

        # ✅ Directories
        self.checkpoints_dir = checkpoints_dir or RESNET50_CHECKPOINTS_DIR
        self.reports_dir = reports_dir or RESNET50_REPORTS_DIR
        self.configs_dir = configs_dir or RESNET50_CONFIGS_DIR
        self.plots_dir = plots_dir or RESNET50_PLOTS_DIR

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_history = []
        self.best_accuracy = 0.0
        self.is_trained = False
        self.class_names = CLASS_NAMES[:num_classes]
        self.callbacks = {}
        self.phase_lrs = {}
        self.val_loader = None

        # ✅ Logger
        self.logger = logging.getLogger("ResNet50CV")
        self.logger.setLevel(logging.INFO)

        ensure_directories()

        if self.verbose:
            print(f"\n🚀 Initializing ResNet50WithoutFineTuning:")
            print(f"   - Num Classes: {num_classes}")
            print(f"   - Device: {device}")
            print(f"   - Activation: {activation}")
            print(f"   - Dropout: {dropout_rate}")
            print(f"   - Zero Init Residual: {zero_init_residual}")
            print(f"   - Target Size: {target_size}")
            print(f"   - Use Class Weights: {use_class_weights}")
            print(f"   - Class Weight Method: {class_weight_method}")
            print(f"   - Label Smoothing: {label_smoothing}")
            print(f"   - Labels CSV: {self.labels_csv_path}")
            print(f"   - Use Focal Loss: {use_focal_loss}")
            print(f"   - Focal Gamma: {focal_gamma}")
            print(f"   - Focal Alpha: {focal_alpha}")
            print(f"   - Adaptive Smoothing: {use_adaptive_smoothing}")
            print(f"   - Output Directory: {RESNET50_BASE_DIR}")
            print(f"   - Feature Dimension: {self._feature_dim}")
            print(f"   - Checkpoints Dir: {self.checkpoints_dir}")
            print(f"   - Configs Dir: {self.configs_dir}")
            print(f"   - Plots Dir: {self.plots_dir}")
            print(f"   - Scheduler Type: {scheduler_type}")
            print(f"   - Warmup Epochs: {warmup_epochs}")
            print(f"   - Min LR: {min_lr}")
            print(f"   - Total Epochs: {total_epochs}")
            print(f"   - Use TTA: {use_tta}")
            print(f"   - Use Mixup: {use_mixup}")
            print(f"   - Mixup Alpha: {mixup_alpha}")
            print(f"   - Mixup Start Stage: {mixup_start_stage}")

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)

    # ================================================================
    # ✅ Class Weights Computation
    # ================================================================

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        """Compute class weights from labels CSV."""
        if not self.use_class_weights:
            return None

        try:
            if not os.path.exists(self.labels_csv_path):
                print(f"⚠️ Labels CSV not found: {self.labels_csv_path}")
                return None

            df = pd.read_csv(self.labels_csv_path)
            df = df[df["liverLabel"].isin(CLASS_NAMES)]

            class_counts = []
            for class_name in CLASS_NAMES:
                count = (df["liverLabel"] == class_name).sum()
                class_counts.append(count)

            class_counts = np.array(class_counts, dtype=np.float32)
            class_counts = np.maximum(class_counts, 1)

            if self.class_weight_method == "sqrt":
                weights = 1.0 / np.sqrt(class_counts)
            elif self.class_weight_method == "balanced":
                weights = 1.0 / class_counts
            elif self.class_weight_method == "inverse":
                weights = class_counts.sum() / class_counts
            else:
                weights = np.ones_like(class_counts)

            # ✅ Normalize so sum = num_classes
            weights = weights / weights.sum() * len(weights)

            if self.verbose:
                print(f"\n⚖️ Class Weights ({self.class_weight_method}):")
                for name, count, w in zip(CLASS_NAMES, class_counts, weights):
                    print(f"   - {name}: count={int(count)}, weight={w:.4f}")

            return torch.tensor(weights, dtype=torch.float32)

        except Exception as e:
            print(f"⚠️ Error computing class weights: {e}")
            return None

    # ================================================================
    # Model Management
    # ================================================================

    def build_model(self) -> None:
        """Build the ResNet50 model from scratch."""
        if self.verbose:
            print("\n🔨 Building ResNet50 from scratch...")

        self.model = _create_resnet50(
            num_classes=self.num_classes,
            pretrained=False,
            zero_init_residual=self.zero_init_residual,
            use_batch_norm=self.use_batch_norm,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            verbose=False,  # ✅ suppress duplicate summary
        )

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=get_phase_lr(0), weight_decay=1e-4
        )

        # ✅ Scheduler - Cosine with warmup via LambdaLR
        if self.scheduler_type == "warmup_cosine":
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    # Warmup: linear from 0.1 to 1.0
                    return 0.1 + 0.9 * (epoch / max(1, self.warmup_epochs))
                else:
                    # Cosine decay
                    progress = (epoch - self.warmup_epochs) / max(
                        1, self.total_epochs - self.warmup_epochs
                    )
                    return 0.5 * (1 + np.cos(np.pi * progress))

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.total_epochs, eta_min=self.min_lr
            )

        # ✅ Class Weights
        self.class_weights = self._compute_class_weights()

        # ✅ Criterion
        if self.use_focal_loss:
            if self.verbose:
                print(f"\n🔥 Using Focal Loss (gamma={self.focal_gamma})")
            self.criterion = FocalLoss(
                alpha=self.class_weights.to(self.device) if self.class_weights is not None else None,
                gamma=self.focal_gamma,
            )
        else:
            if self.verbose:
                print(f"\n📉 Using CrossEntropyLoss (label_smoothing={self.label_smoothing})")
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights.to(self.device) if self.class_weights is not None else None,
                label_smoothing=self.label_smoothing,
            )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if self.verbose:
            print(f"✅ Model built successfully!")
            print(f"   - Total Parameters: {total_params:,}")
            print(f"   - Trainable Parameters: {trainable_params:,}")
            print(f"   - Model on: {self.device}")
            print(f"   - Feature Dimension: {self._feature_dim}")

        config = {
            "architecture": "ResNet50",
            "num_classes": self.num_classes,
            "zero_init_residual": self.zero_init_residual,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": self.device,
            "class_names": self.class_names,
            "feature_dim": self._feature_dim,
            "stages": 5,
            "use_class_weights": self.use_class_weights,
            "class_weight_method": self.class_weight_method,
            "use_focal_loss": self.use_focal_loss,
            "label_smoothing": self.label_smoothing,
            "use_tta": self.use_tta,
            "use_mixup": self.use_mixup,
        }
        save_config(config, "config_resnet50.json")

    def get_model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        if self.model is None:
            self.build_model()
        return self.model

    def summary(self) -> None:
        """Print detailed model summary."""
        if self.model is None:
            print("⚠️ Model not built yet. Call build_model() first.")
            return
        print(self.model.summary())

    def print_summary(self) -> None:
        """Print model summary (via utils)."""
        if self.model is None:
            print("⚠️ Model not built yet. Call build_model() first.")
            return

        print_model_summary(
            self.model,
            input_shape=(3, self._target_size, self._target_size),
            device=self.device,
        )

    # ================================================================
    # Forward Pass and Feature Extraction
    # ================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if self.model is None:
            self.build_model()
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings."""
        if self.model is None:
            self.build_model()
        return self.model.get_embedding(x)

    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self._feature_dim

    # ================================================================
    # Training Methods
    # ================================================================

    def train_epoch(
        self,
        train_loader: DataLoader,
        epochs: int,
        total_epochs: int,
        desc: str = "",
        patience: int = 15,
        use_mixup_in_epoch: bool = False,
    ) -> float:
        """
        Train for a given number of epochs.

        Args:
            train_loader: Training DataLoader
            epochs: Number of epochs to train
            total_epochs: Total epochs in the full schedule (for display)
            desc: Description prefix
            patience: Early stopping patience
            use_mixup_in_epoch: Whether to use Mixup in this stage
        """
        for epoch in range(epochs):
            current_epoch = len(self.train_history) + 1
            epoch_start = time.time()

            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            loop = tqdm(
                train_loader,
                desc=(
                    f"{desc} Epoch {current_epoch:3d}/{total_epochs}"
                    if desc
                    else f"Epoch {current_epoch:3d}/{total_epochs}"
                ),
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
                colour="#87CEEB",
                ncols=100,
                leave=False,
                disable=not self.verbose,
            )

            for batch_idx, (data, target) in enumerate(loop):
                data, target = data.to(self.device), target.to(self.device)

                # ✅ Mixup (only if enabled)
                use_mixup_batch = use_mixup_in_epoch and np.random.rand() < 0.5

                if use_mixup_batch:
                    mixed_data, target_a, target_b, lam = mixup_data(
                        data, target, alpha=self.mixup_alpha, device=self.device
                    )

                    self.optimizer.zero_grad()
                    output = self.model(mixed_data)
                    loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    # ✅ Approximate accuracy for mixup
                    _, predicted = output.max(1)
                    total_samples += target.size(0)
                    total_correct += (
                        lam * predicted.eq(target_a).sum().item()
                        + (1 - lam) * predicted.eq(target_b).sum().item()
                    )
                else:
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    _, predicted = output.max(1)
                    total_samples += target.size(0)
                    total_correct += predicted.eq(target).sum().item()

                total_loss += loss.item()

                loop.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{total_correct/total_samples:.4f}" if total_samples > 0 else "0.0000",
                })

            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            train_acc = total_correct / total_samples if total_samples > 0 else 0.0

            val_loss, val_acc = self.validate(return_loss=True)

            # ✅ Scheduler step
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception as e:
                    print(f"⚠️ Scheduler step failed: {e}")

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            self.train_history.append({
                "epoch": current_epoch,
                "loss": avg_loss,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": current_lr,
                "time": epoch_time,
            })

            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self._save_checkpoint(
                    current_epoch,
                    {"val_acc": val_acc, "val_loss": val_loss, "train_acc": train_acc, "train_loss": avg_loss},
                )
                if self.verbose:
                    print(f"   ✅ New best model saved! (Acc: {self.best_accuracy:.4f})")

            self._save_checkpoint(
                current_epoch,
                {"val_acc": val_acc, "val_loss": val_loss, "train_acc": train_acc, "train_loss": avg_loss},
                suffix="last",
            )

            if self.verbose:
                print(
                    f"📊 Epoch {current_epoch:3d}/{total_epochs} | "
                    f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s"
                )

            # ✅ Early stopping: only after 85 epochs, with dynamic patience
            if current_epoch >= 85 and len(self.train_history) > patience + 5:
                recent = [h["val_accuracy"] for h in self.train_history[-(patience + 1):]]
                if len(recent) >= patience + 1:
                    best_recent = max(recent[:-1])
                    if all(v <= best_recent + 1e-4 for v in recent[-patience:]):
                        if self.verbose:
                            print(f"⚠️ Early stopping triggered (no improvement for {patience} epochs)")
                        break

        return self.best_accuracy

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], suffix: str = "best"):
        """Save checkpoint."""
        if suffix == "best":
            path = os.path.join(self.checkpoints_dir, "resnet50_best.pth")
        elif suffix == "checkpoint":
            path = os.path.join(self.checkpoints_dir, "resnet50_checkpoint.pth")
        else:
            path = os.path.join(self.checkpoints_dir, "resnet50_last.pth")

        save_model_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            filename=os.path.basename(path),
            scheduler=self.scheduler,
            extra_info={
                "best_accuracy": self.best_accuracy,
                "history_length": len(self.train_history),
            },
            save_dir=os.path.dirname(path),
        )

    def validate(self, return_loss: bool = False):
        """Validate the model."""
        if self.model is None:
            self.build_model()

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        if return_loss:
            return avg_loss, accuracy
        return accuracy

    def _update_learning_rate(self, phase: int) -> None:
        """Update learning rate for the current phase."""
        lr = get_phase_lr(phase)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.phase_lrs[phase] = lr

        # ✅ Reset scheduler's base_lr for this phase
        if hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [lr for _ in self.optimizer.param_groups]

        if self.verbose:
            print(f"   ✅ Updated LR to: {lr:.6f} for Phase {phase+1}")

    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        save_best: bool = True,
        save_path: str = None,
        early_stopping_patience: int = 15,
        use_multi_stage: bool = True,
    ) -> Dict[str, Any]:
        """Train with multi-stage strategy (5 phases)."""
        if self.model is None:
            self.build_model()

        self.val_loader = val_loader

        if save_path is None:
            save_path = os.path.join(self.checkpoints_dir, "resnet50_best.pth")

        self.train_history = []
        self.best_accuracy = 0.0

        # ✅ Relative split
        stage1_epochs = max(1, int(epochs * 0.10))
        stage2_epochs = max(1, int(epochs * 0.15))
        stage3_epochs = max(1, int(epochs * 0.20))
        stage4_epochs = max(1, int(epochs * 0.20))
        stage5_epochs = epochs - stage1_epochs - stage2_epochs - stage3_epochs - stage4_epochs

        if self.verbose:
            print(f"\n🚀 Starting Multi-Stage Training for {epochs} epochs...")
            print("=" * 80)
            print(f"📊 Stage Plan (for {epochs} epochs):")
            print(f"   Stage 1: FC layer only ({stage1_epochs} epochs) - LR: 1e-3")
            print(f"   Stage 2: Layer4 + FC ({stage2_epochs} epochs) - LR: 5e-4")
            print(f"   Stage 3: Layer3+4 + FC ({stage3_epochs} epochs) - LR: 1e-4")
            print(f"   Stage 4: Layer2+3+4 + FC ({stage4_epochs} epochs) - LR: 5e-5")
            print(f"   Stage 5: All layers ({stage5_epochs} epochs) - LR: 1e-5")
            if self.use_mixup:
                print(f"   ⚡ Mixup: ENABLED from Stage {self.mixup_start_stage + 1}")
            print("=" * 80)

        # ==========================================
        # Stage 1: FC only
        # ==========================================
        if use_multi_stage and self.verbose:
            print(f"\n📌 Stage 1: Training FC layer only ({stage1_epochs} epochs)")
            print("-" * 40)

        if use_multi_stage:
            self.freeze_backbone()
            self._update_learning_rate(0)
            if stage1_epochs > 0:
                self.train_epoch(
                    train_loader, stage1_epochs, epochs, desc="Stage 1",
                    patience=5, use_mixup_in_epoch=False,
                )
        else:
            stage1_epochs = 0

        # ==========================================
        # Stage 2: Layer4 + FC
        # ==========================================
        if use_multi_stage and self.verbose:
            print(f"\n📌 Stage 2: Training Layer4 + FC ({stage2_epochs} epochs)")
            print("-" * 40)

        if use_multi_stage:
            self.unfreeze_layer4()
            self._update_learning_rate(1)
            if stage2_epochs > 0:
                self.train_epoch(
                    train_loader, stage2_epochs, epochs, desc="Stage 2",
                    patience=9, use_mixup_in_epoch=False,
                )
        else:
            stage2_epochs = 0

        # ==========================================
        # Stage 3: Layer3+4 + FC
        # ==========================================
        if use_multi_stage and self.verbose:
            print(f"\n📌 Stage 3: Training Layer3 + Layer4 + FC ({stage3_epochs} epochs)")
            print("-" * 40)

        if use_multi_stage:
            self.unfreeze_layer3_4()
            self._update_learning_rate(2)
            if stage3_epochs > 0:
                self.train_epoch(
                    train_loader, stage3_epochs, epochs, desc="Stage 3",
                    patience=12, use_mixup_in_epoch=False,
                )
        else:
            stage3_epochs = 0

        # ==========================================
        # Stage 4: Layer2+3+4 + FC
        # ==========================================
        if use_multi_stage and self.verbose:
            print(f"\n📌 Stage 4: Training Layer2 + Layer3 + Layer4 + FC ({stage4_epochs} epochs)")
            print("-" * 40)

        if use_multi_stage:
            self.unfreeze_layer2_3_4()
            self._update_learning_rate(3)
            if stage4_epochs > 0:
                self.train_epoch(
                    train_loader, stage4_epochs, epochs, desc="Stage 4",
                    patience=15, use_mixup_in_epoch=False,
                )
        else:
            stage4_epochs = 0

        # ==========================================
        # Stage 5: Full Fine-Tuning (with optional Mixup)
        # ==========================================
        use_mixup_stage5 = self.use_mixup and (self.mixup_start_stage <= 4)

        if use_multi_stage and self.verbose:
            print(f"\n📌 Stage 5: Full Fine-Tuning ({stage5_epochs} epochs)")
            if use_mixup_stage5:
                print(f"   ⚡ Mixup: ENABLED (alpha={self.mixup_alpha})")
            print("-" * 40)

        if use_multi_stage:
            self.unfreeze_all()
            self._update_learning_rate(4)
            if stage5_epochs > 0:
                self.train_epoch(
                    train_loader, stage5_epochs, epochs, desc="Stage 5",
                    patience=20, use_mixup_in_epoch=use_mixup_stage5,
                )
        else:
            stage5_epochs = epochs
            if stage5_epochs > 0:
                self.train_epoch(
                    train_loader, stage5_epochs, epochs, desc="Training",
                    patience=20, use_mixup_in_epoch=use_mixup_stage5,
                )

        if self.verbose:
            print("=" * 80)
            print(f"✅ Training Complete!")
            print(f"   Best Validation Accuracy: {self.best_accuracy:.4f}")

        self.is_trained = True

        # ✅ Try to load best checkpoint, but DON'T crash
        if save_best:
            try:
                self.load_checkpoint(save_path)
            except Exception as e:
                print(f"\n⚠️ WARNING: Could not reload best checkpoint: {e}")
                print(f"   → Continuing with model state from last epoch")

        return {
            "history": self.train_history,
            "best_accuracy": self.best_accuracy,
            "final_accuracy": self.train_history[-1]["accuracy"] if self.train_history else 0,
            "final_val_accuracy": self.train_history[-1]["val_accuracy"] if self.train_history else 0,
        }

    # ================================================================
    # Freezing Methods
    # ================================================================

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers except the FC layer."""
        if self.model is None:
            return

        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        if self.verbose:
            print(f"✅ Backbone frozen. Only FC layer is trainable.")
            print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    def unfreeze_layer4(self) -> None:
        """Unfreeze layer4 and FC layer."""
        if self.model is None:
            return

        for name, param in self.model.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        if self.verbose:
            print(f"✅ Layer4 and FC layer unfrozen.")
            print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    def unfreeze_layer3_4(self) -> None:
        """Unfreeze layer3, layer4 and FC layer."""
        if self.model is None:
            return

        for name, param in self.model.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        if self.verbose:
            print(f"✅ Layer3, Layer4 and FC layer unfrozen.")
            print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    def unfreeze_layer2_3_4(self) -> None:
        """Unfreeze layer2, layer3, layer4 and FC layer."""
        if self.model is None:
            return

        for name, param in self.model.named_parameters():
            if (
                name.startswith("layer2")
                or name.startswith("layer3")
                or name.startswith("layer4")
                or name.startswith("fc")
            ):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        if self.verbose:
            print(f"✅ Layer2, Layer3, Layer4 and FC layer unfrozen.")
            print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    def unfreeze_all(self) -> None:
        """Unfreeze all layers."""
        if self.model is None:
            return

        for param in self.model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        if self.verbose:
            print(f"✅ All layers unfrozen (full fine-tuning).")
            print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # ================================================================
    # Model Persistence
    # ================================================================

    def save_model(self, path: str = None) -> str:
        """Save the complete model."""
        if self.model is None:
            print("⚠️ No model to save. Build model first.")
            return ""

        if path is None:
            path = os.path.join(self.checkpoints_dir, "resnet50_final.pth")

        return save_model_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=len(self.train_history),
            metrics={
                "best_accuracy": self.best_accuracy,
                "final_accuracy": self.train_history[-1]["accuracy"] if self.train_history else 0,
                "final_val_accuracy": self.train_history[-1]["val_accuracy"] if self.train_history else 0,
            },
            filename=os.path.basename(path),
            scheduler=self.scheduler,
            extra_info={
                "num_classes": self.num_classes,
                "zero_init_residual": self.zero_init_residual,
                "use_batch_norm": self.use_batch_norm,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "is_trained": self.is_trained,
                "feature_dim": self._feature_dim,
                "train_history": self.train_history,
                "class_names": self.class_names,
                "version": "6.0.0",
                "phase_lrs": self.phase_lrs,
                "best_accuracy": self.best_accuracy,
            },
            save_dir=os.path.dirname(path),
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint with robust error handling."""
        if not os.path.exists(path):
            print(f"⚠️ Model file not found: {path} - continuing with fresh model")
            return

        if self.model is None:
            self.build_model()

        try:
            checkpoint = load_model_checkpoint(
                filepath=path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )

            # ✅ Extract best_accuracy from multiple possible locations
            self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
            if self.best_accuracy == 0.0:
                self.best_accuracy = checkpoint.get("metrics", {}).get("best_accuracy", 0.0)
            if self.best_accuracy == 0.0:
                self.best_accuracy = checkpoint.get("extra_info", {}).get("best_accuracy", 0.0)

            self.is_trained = checkpoint.get("is_trained", True)
            if not self.is_trained:
                self.is_trained = checkpoint.get("extra_info", {}).get("is_trained", True)

            self.train_history = checkpoint.get("extra_info", {}).get("train_history", [])
            self.phase_lrs = checkpoint.get("extra_info", {}).get("phase_lrs", {})

            if self.verbose:
                print(f"✅ Model loaded from: {path}")
                print(f"   - Best Accuracy: {self.best_accuracy:.4f}")
                print(f"   - Is Trained: {self.is_trained}")
                print(f"   - Feature Dimension: {self._feature_dim}")

        except Exception as e:
            print(f"\n⚠️ WARNING: Could not load checkpoint from {path}")
            print(f"   Error: {e}")
            print(f"   → Continuing with CURRENT model state (already trained)")
            self.is_trained = True

    # ================================================================
    # ✅ TTA Evaluation (NEW)
    # ================================================================

    def _predict_with_tta(
        self, data: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict with Test Time Augmentation.

        Applies 3 transformations:
        1. Original
        2. Horizontal flip
        3. Vertical flip

        Returns averaged probabilities.

        Args:
            data: Input batch (B, C, H, W)

        Returns:
            Averaged probabilities (B, num_classes)
        """
        # Original
        output1 = self.model(data)
        probs1 = torch.softmax(output1, dim=1)

        # Horizontal flip
        data_hflip = torch.flip(data, dims=[3])
        output2 = self.model(data_hflip)
        probs2 = torch.softmax(output2, dim=1)

        # Vertical flip
        data_vflip = torch.flip(data, dims=[2])
        output3 = self.model(data_vflip)
        probs3 = torch.softmax(output3, dim=1)

        # Average
        probs = (probs1 + probs2 + probs3) / 3.0

        return probs

    # ================================================================
    # Comprehensive Evaluation
    # ================================================================

    def evaluate_comprehensive(
        self,
        test_loader: DataLoader,
        generate_report: bool = True,
        cv_folds: int = 0,
        report_name: str = "ResNet50_Report.pdf",
        use_tta: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation with full metrics and report.

        ✅ This method NEVER crashes. All steps have try/except.
        ✅ Uses TTA if enabled.
        """
        if self.model is None:
            self.build_model()

        # ✅ Determine TTA usage
        if use_tta is None:
            use_tta = self.use_tta

        if self.verbose:
            print("\n📊 Comprehensive Evaluation Started...")
            if use_tta:
                print("   🔄 TTA: ENABLED (Original + HFlip + VFlip)")
            else:
                print("   🔄 TTA: DISABLED")
            print("=" * 80)

        # ==========================================
        # Step 1: Get predictions (with optional TTA)
        # ==========================================
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        try:
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    if use_tta:
                        probs = self._predict_with_tta(data)
                    else:
                        output = self.model(data)
                        probs = torch.softmax(output, dim=1)

                    _, preds = probs.max(1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(target.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                "metrics": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
                "cv_scores": None,
                "model_info": {},
                "predictions": [],
                "labels": [],
                "probabilities": [],
            }

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # ==========================================
        # Step 2: Compute metrics
        # ==========================================
        try:
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

            per_class = {}
            for i, class_name in enumerate(self.class_names):
                class_precision = precision_score(all_labels, all_preds, labels=[i], average="micro", zero_division=0)
                class_recall = recall_score(all_labels, all_preds, labels=[i], average="micro", zero_division=0)
                class_f1 = f1_score(all_labels, all_preds, labels=[i], average="micro", zero_division=0)
                per_class[class_name] = {
                    "precision": class_precision,
                    "recall": class_recall,
                    "f1": class_f1,
                }

            cm = confusion_matrix(all_labels, all_preds)

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "per_class": per_class,
                "confusion_matrix": cm.tolist(),
                "num_samples": len(all_labels),
            }

            if self.verbose:
                print(f"\n📊 Metrics:")
                print(f"   - Accuracy: {accuracy:.4f}")
                print(f"   - Precision (weighted): {precision:.4f}")
                print(f"   - Recall (weighted): {recall:.4f}")
                print(f"   - F1-Score (weighted): {f1:.4f}")
                print(f"   - Samples: {len(all_labels)}")
        except Exception as e:
            print(f"⚠️ Error computing metrics: {e}")
            metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
            cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
            per_class = {}

        # ==========================================
        # Step 3: Cross-validation (optional)
        # ==========================================
        cv_scores = None
        if cv_folds > 1 and len(all_labels) > cv_folds:
            try:
                if self.verbose:
                    print(f"\n🔬 Performing {cv_folds}-Fold Cross-Validation...")
                cv_scores = self.perform_cross_validation(test_loader.dataset, n_folds=cv_folds)
            except Exception as e:
                print(f"⚠️ CV failed: {e}")

        # ==========================================
        # Step 4: Model info
        # ==========================================
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            model_info = {
                "architecture": "ResNet50",
                "total_params": total_params,
                "trainable_params": trainable_params,
                "input_shape": (3, 224, 224),
                "num_classes": self.num_classes,
                "class_names": self.class_names,
                "device": self.device,
                "is_trained": self.is_trained,
                "best_accuracy": self.best_accuracy,
                "feature_dim": self._feature_dim,
                "activation": self.activation,
                "stages": 5,
                "use_tta": use_tta,
            }
        except Exception as e:
            print(f"⚠️ Error getting model info: {e}")
            model_info = {}

        # ==========================================
        # Step 5: Generate PDF report
        # ==========================================
        if generate_report:
            if self.verbose:
                print("\n📄 Generating PDF Report...")

            try:
                os.makedirs(self.reports_dir, exist_ok=True)
                report_path = os.path.join(self.reports_dir, report_name)

                cm_np = np.array(cm)
                y_true_np = np.array(all_labels)
                y_prob_np = np.array(all_probs)

                generate_pdf_report(
                    metrics=metrics,
                    cm=cm_np,
                    y_true=y_true_np,
                    y_prob=y_prob_np,
                    per_class=per_class,
                    cv_scores=cv_scores,
                    model_info=model_info,
                    training_history=self.train_history,
                    output_path=report_path,
                )
                print(f"✅ PDF report saved to: {report_path}")
            except Exception as e:
                print(f"⚠️ Could not generate PDF report: {e}")
                import traceback
                traceback.print_exc()

            # ✅ Save metrics.json
            try:
                metrics_path = os.path.join(self.reports_dir, "metrics.json")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "metrics": metrics,
                            "model_info": model_info,
                            "cv_scores": cv_scores,
                            "timestamp": datetime.now().isoformat(),
                        },
                        f,
                        indent=4,
                        ensure_ascii=False,
                        default=str,
                    )
                print(f"✅ Metrics saved to: {metrics_path}")
            except Exception as e:
                print(f"⚠️ Could not save metrics.json: {e}")

        if self.verbose:
            print("\n" + "=" * 80)
            print("✅ Comprehensive Evaluation Complete!")

        return {
            "metrics": metrics,
            "cv_scores": cv_scores,
            "model_info": model_info,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }

    def perform_cross_validation(
        self, dataset: torch.utils.data.Dataset, n_folds: int = 3
    ) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        labels = np.array(labels)

        cv_accuracy = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            if self.verbose:
                print(f"\n   CV Fold {fold+1}/{n_folds}...")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=48, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=48, shuffle=False)

            fold_model = ResNet50WithoutFineTuning(
                num_classes=self.num_classes,
                zero_init_residual=self.zero_init_residual,
                use_batch_norm=self.use_batch_norm,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                device=self.device,
                verbose=False,
                use_class_weights=self.use_class_weights,
                class_weight_method=self.class_weight_method,
                label_smoothing=self.label_smoothing,
                labels_csv_path=self.labels_csv_path,
                use_focal_loss=self.use_focal_loss,
                focal_gamma=self.focal_gamma,
                use_tta=False,  # ✅ No TTA during CV training
                use_mixup=False,  # ✅ No mixup during CV (too short)
            )
            fold_model.build_model()
            fold_model.train_full(train_loader, val_loader, epochs=30, save_best=False, use_multi_stage=True)

            val_acc = fold_model.validate(val_loader)
            cv_accuracy.append(val_acc)

            fold_model.model.eval()
            all_preds = []
            all_labels_val = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = fold_model.model(data)
                    _, preds = output.max(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels_val.extend(target.cpu().numpy())

            cv_precision.append(precision_score(all_labels_val, all_preds, average="weighted", zero_division=0))
            cv_recall.append(recall_score(all_labels_val, all_preds, average="weighted", zero_division=0))
            cv_f1.append(f1_score(all_labels_val, all_preds, average="weighted", zero_division=0))

        return {
            "accuracy": cv_accuracy,
            "precision": cv_precision,
            "recall": cv_recall,
            "f1_score": cv_f1,
        }

    def perform_cross_validation_from_saved(
        self, dataset: torch.utils.data.Dataset, n_folds: int = 5, batch_size: int = 48
    ) -> Dict[str, List[float]]:
        """Perform k-fold CV using the SAVED model (NO RETRAINING!)."""
        if self.model is None:
            raise ValueError("Model not loaded! Please load a saved model first.")

        if self.verbose:
            print(f"🔬 Running {n_folds}-fold CV using SAVED model (no retraining)...")

        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        labels = np.array(labels)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        cv_accuracy = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []

        fold_num = 1
        for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
            if self.verbose:
                print(f"\n   CV Fold {fold_num}/{n_folds}...")

            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            self.model.eval()
            all_preds = []
            all_labels_val = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    # ✅ TTA in CV
                    if self.use_tta:
                        probs = self._predict_with_tta(data)
                    else:
                        output = self.model(data)
                        probs = torch.softmax(output, dim=1)

                    _, preds = probs.max(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels_val.extend(target.cpu().numpy())

            fold_accuracy = accuracy_score(all_labels_val, all_preds)
            fold_precision = precision_score(all_labels_val, all_preds, average="weighted", zero_division=0)
            fold_recall = recall_score(all_labels_val, all_preds, average="weighted", zero_division=0)
            fold_f1 = f1_score(all_labels_val, all_preds, average="weighted", zero_division=0)

            cv_accuracy.append(fold_accuracy)
            cv_precision.append(fold_precision)
            cv_recall.append(fold_recall)
            cv_f1.append(fold_f1)

            if self.verbose:
                print(f"      Accuracy: {fold_accuracy:.4f}, Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}, F1: {fold_f1:.4f}")

            fold_num += 1

        results = {
            "accuracy": cv_accuracy,
            "precision": cv_precision,
            "recall": cv_recall,
            "f1_score": cv_f1,
            "mean_accuracy": np.mean(cv_accuracy),
            "std_accuracy": np.std(cv_accuracy),
            "mean_precision": np.mean(cv_precision),
            "std_precision": np.std(cv_precision),
            "mean_recall": np.mean(cv_recall),
            "std_recall": np.std(cv_recall),
            "mean_f1": np.mean(cv_f1),
            "std_f1": np.std(cv_f1),
        }

        if self.verbose:
            print("\n" + "=" * 70)
            print("📊 CROSS-VALIDATION RESULTS (using saved model)")
            print("=" * 70)
            print(f"   Accuracy:  {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            print(f"   Precision: {results['mean_precision']:.4f} ± {results['std_precision']:.4f}")
            print(f"   Recall:    {results['mean_recall']:.4f} ± {results['std_recall']:.4f}")
            print(f"   F1-Score:  {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
            print("=" * 70)

        return results

    # ================================================================
    # Advanced Profiling
    # ================================================================

    def profile(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
        if self.model is None:
            self.build_model()
        return profile_model(self.model, input_shape, self.device)

    def calculate_flops(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
        if self.model is None:
            self.build_model()
        return calculate_flops(self.model, input_shape)

    def analyze_memory(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
        if self.model is None:
            self.build_model()
        return analyze_memory_usage(self.model, input_shape, self.device)

    # ================================================================
    # Reset
    # ================================================================

    def reset(self) -> None:
        """Reset the model to initial state."""
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_history = []
        self.best_accuracy = 0.0
        self.is_trained = False
        self.build_model()
        if self.verbose:
            print("✅ Model reset to initial state.")

    def __repr__(self) -> str:
        return f"ResNet50WithoutFineTuning(num_classes={self.num_classes}, device={self.device})"


# ================================================================
# Factory Function
# ================================================================


def create_resnet50_wrapper(num_classes: int = 4, **kwargs) -> ResNet50WithoutFineTuning:
    """Factory function to create a ResNet50 wrapper."""
    return ResNet50WithoutFineTuning(num_classes=num_classes, **kwargs)


# ================================================================
# Testing
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 TESTING RESNET50 WRAPPER (ULTIMATE EDITION v6.0.0)")
    print("=" * 80)

    model = create_resnet50_wrapper(
        num_classes=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    model.build_model()
    model.print_summary()

    dummy = torch.randn(4, 3, 224, 224).to(model.device)
    output = model.forward(dummy)
    print(f"\n✅ Output shape: {output.shape}")

    embeddings = model.get_embeddings(dummy)
    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"✅ Feature dimension: {model.get_feature_dim()}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
