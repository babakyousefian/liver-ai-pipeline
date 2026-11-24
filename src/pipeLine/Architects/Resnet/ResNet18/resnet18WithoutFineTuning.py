# src/pipeLine/resnet18WithoutFineTuning.py

"""
ResNet18 Without Fine-Tuning - OOP Implementation with Full Reporting (ULTIMATE EDITION)

This file implements an OOP wrapper around ResNet18 that encapsulates
all functionality for training, evaluation, feature extraction, and
comprehensive reporting.

Features:
- Full OOP structure with comprehensive methods
- Multi-stage training (FC → Layer4 → Layer3+4 → Full) - 4 Stages
- Smart Learning Rate per stage (1e-3 → 1e-4 → 5e-5 → 1e-5)
- Advanced callbacks (Early Stopping, Checkpoint, LR Scheduler)
- Comprehensive PDF reports with all visualizations
- K-Fold cross-validation
- Feature extraction for transfer learning
- Grad-CAM visualization support
- Complete model persistence (save/load)
- Model profiling and FLOPs calculation

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 4.0.0 (ULTIMATE EDITION)
"""

import gc
import io
import json
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
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,       # ← ✅ NEW
    StratifiedGroupKFold,          # ← ✅ NEW
)

from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from collections import defaultdict  # ← ✅ NEW

warnings.filterwarnings("ignore")

# ================================================================
# Import all ResNet18 components from the modular structure
# ================================================================

import logging

from src.models.resnet.resnet18 import BasicBlock, Bottleneck
from src.models.resnet.resnet18 import ResNet18 as _ResNet18
from src.models.resnet.resnet18 import (
    analyze_memory_usage,
    calculate_flops,
    check_gradient_flow,
    compare_models,
    count_parameters,
)
from src.models.resnet.resnet18 import create_resnet18 as _create_resnet18
from src.models.resnet.resnet18 import (
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
# Constants - Paths for ResNet18 (NO G4)
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)

# ================================================================
# Base Directories - ResNet18
# ================================================================

FINAL_REPORTS_DIR = "finalReports"
MODELS_BASE_DIR = os.path.join(FINAL_REPORTS_DIR, "models")

RESNET_BASE_DIR = os.path.join(MODELS_BASE_DIR, "..")
RESNET18_BASE_DIR = os.path.join(RESNET_BASE_DIR, "resnet18")

RESNET18_CHECKPOINTS_DIR = os.path.join(RESNET18_BASE_DIR, "checkpoints")
RESNET18_REPORTS_DIR = os.path.join(RESNET18_BASE_DIR, "reports")
RESNET18_CONFIGS_DIR = os.path.join(RESNET18_BASE_DIR, "configs")
RESNET18_PLOTS_DIR = os.path.join(RESNET18_BASE_DIR, "plots")

# ================================================================
# ✅ ADVANCED LEARNING RATE SCHEDULERS WITH WARMUP
# ================================================================

from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    LinearLR,
    SequentialLR,
)

import math

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Linear Warmup + Cosine Annealing Learning Rate Scheduler.

    This is the BEST scheduler for training from scratch.

    Schedule:
        Phase 1 - Warmup (warmup_epochs):
            LR linearly increases from warmup_start_lr to base_lr
        Phase 2 - Cosine Annealing (remaining epochs):
            LR follows cosine curve from base_lr to eta_min

    Formula:
        Warmup:  LR(t) = warmup_start_lr + (base_lr - warmup_start_lr) * (t / warmup_epochs)
        Cosine:  LR(t) = eta_min + (base_lr - eta_min) * (1 + cos(π * t / T_max)) / 2

    Args:
        optimizer: PyTorch optimizer
        T_max: Maximum number of iterations (total epochs for cosine phase)
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Starting LR for warmup (default: base_lr / 100)
        eta_min: Minimum LR (default: 0)
        last_epoch: Last epoch (-1 for initial)

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        >>> scheduler = LinearWarmupCosineAnnealingLR(
        ...     optimizer,
        ...     T_max=80,
        ...     warmup_epochs=5,
        ...     warmup_start_lr=2e-5,
        ...     eta_min=1e-6,
        ... )
        >>> for epoch in range(80):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            T_max: int,
            warmup_epochs: int = 5,
            warmup_start_lr: float = None,
            eta_min: float = 0.0,
            last_epoch: int = -1,
    ) -> None:
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min

        # ✅ Get base LR from optimizer
        self.base_lr = optimizer.param_groups[0]["lr"]

        # ✅ Default warmup_start_lr = base_lr / 100
        if warmup_start_lr is None:
            self.warmup_start_lr = self.base_lr / 100
        else:
            self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute LR for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # ==========================================
            # ✅ Warmup Phase
            # ==========================================
            # LR linearly increases
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * alpha

        else:
            # ==========================================
            # ✅ Cosine Annealing Phase
            # ==========================================
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.T_max - self.warmup_epochs, 1
            )
            # Clamp progress to [0, 1]
            progress = min(max(progress, 0.0), 1.0)

            # Cosine annealing formula
            cosine_factor = (1 + math.cos(math.pi * progress)) / 2
            lr = self.eta_min + (self.base_lr - self.eta_min) * cosine_factor

        # ✅ Return LR for each param group
        return [lr for _ in self.base_lrs]

    def __repr__(self) -> str:
        return (
            f"LinearWarmupCosineAnnealingLR("
            f"T_max={self.T_max}, "
            f"warmup_epochs={self.warmup_epochs}, "
            f"base_lr={self.base_lr:.2e}, "
            f"warmup_start_lr={self.warmup_start_lr:.2e}, "
            f"eta_min={self.eta_min:.2e}"
            f")"
        )


class LinearWarmupScheduler:
    """
    Simple Linear Warmup + any scheduler.

    Uses PyTorch's native `LinearLR` + `SequentialLR`.

    This is cleaner and uses official PyTorch components.
    """

    @staticmethod
    def create(
            optimizer: torch.optim.Optimizer,
            main_scheduler_type: str = "cosine",
            total_epochs: int = 100,
            warmup_epochs: int = 5,
            min_lr: float = 1e-6,
    ):
        """
        Create a warmup + main scheduler using SequentialLR.

        Args:
            optimizer: PyTorch optimizer
            main_scheduler_type: "cosine", "onecycle", "restart"
            total_epochs: Total training epochs
            warmup_epochs: Number of warmup epochs
            min_lr: Minimum LR

        Returns:
            SequentialLR scheduler
        """
        # ✅ Get base LR
        base_lr = optimizer.param_groups[0]["lr"]

        # ==========================================
        # Warmup Scheduler
        # ==========================================
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,  # Start at 1% of base_lr
            end_factor=1.0,  # End at base_lr
            total_iters=warmup_epochs,
        )

        # ==========================================
        # Main Scheduler
        # ==========================================
        if main_scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=min_lr,
            )

        elif main_scheduler_type == "restart":
            main_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=20,  # First restart at epoch 20
                T_mult=2,  # Double period after each restart
                eta_min=min_lr,
            )

        elif main_scheduler_type == "onecycle":
            # ✅ OneCycleLR needs total_steps
            # We'll use steps_per_epoch = total_epochs (approximation)
            main_scheduler = OneCycleLR(
                optimizer,
                max_lr=base_lr * 3,  # Peak LR
                total_steps=total_epochs - warmup_epochs,
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1000.0,
            )

        else:
            raise ValueError(f"Unknown scheduler type: {main_scheduler_type}")

        # ==========================================
        # Combine with SequentialLR
        # ==========================================
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

        return scheduler


# ================================================================
# ✅ SCHEDULER FACTORY
# ================================================================

def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "warmup_cosine",
        total_epochs: int = 100,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        **kwargs,
):
    """
    Factory function to create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
            - "cosine": Standard Cosine Annealing
            - "warmup_cosine": Linear Warmup + Cosine ⭐ (recommended)
            - "warmup_restart": Linear Warmup + Cosine Warm Restarts
            - "onecycle": OneCycleLR (without explicit warmup)
            - "warmup_onecycle": Warmup + OneCycle
            - "plateau": ReduceLROnPlateau
            - "step": StepLR
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum LR

    Returns:
        Scheduler instance (or tuple for plateau)
    """
    base_lr = optimizer.param_groups[0]["lr"]

    # ==========================================
    # 1. Standard Cosine
    # ==========================================
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_lr,
        )

    # ==========================================
    # 2. Warmup + Cosine ⭐ (RECOMMENDED)
    # ==========================================
    elif scheduler_type == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=base_lr / 100,
            eta_min=min_lr,
        )

    # ==========================================
    # 3. Warmup + Warm Restarts
    # ==========================================
    elif scheduler_type == "warmup_restart":
        scheduler = LinearWarmupScheduler.create(
            optimizer=optimizer,
            main_scheduler_type="restart",
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
        )

    # ==========================================
    # 4. OneCycleLR
    # ==========================================
    elif scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=base_lr * 3,
            total_steps=total_epochs,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )

    # ==========================================
    # 5. Warmup + OneCycle
    # ==========================================
    elif scheduler_type == "warmup_onecycle":
        scheduler = LinearWarmupScheduler.create(
            optimizer=optimizer,
            main_scheduler_type="onecycle",
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
        )

    # ==========================================
    # 6. ReduceLROnPlateau
    # ==========================================
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=True,
        )

    # ==========================================
    # 7. StepLR
    # ==========================================
    elif scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=20,
            gamma=0.5,
        )

    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    return scheduler


# ================================================================
# ✅ ADVANCED CROSS-VALIDATION
# ================================================================

class AdvancedCrossValidator:
    """
    Advanced Cross-Validation with multiple strategies and comprehensive reporting.

    Features:
        - Stratified K-Fold (standard)
        - Repeated Stratified K-Fold (more reliable)
        - Stratified Group K-Fold (if patient IDs available)
        - Per-fold detailed metrics
        - Confusion matrix per fold
        - ROC/AUC per fold
        - Statistical analysis (mean, std, CI)
        - CSV/JSON export
        - Visualization

    Args:
        n_folds: Number of folds (default: 5)
        n_repeats: Number of repeats for Repeated K-Fold (default: 1)
        strategy: "stratified", "repeated", "group"
        random_state: Random seed
        verbose: Print progress
    """

    def __init__(
            self,
            n_folds: int = 5,
            n_repeats: int = 1,
            strategy: str = "stratified",
            random_state: int = 42,
            verbose: bool = True,
    ) -> None:
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.strategy = strategy
        self.random_state = random_state
        self.verbose = verbose

        # ✅ Results storage
        self.fold_results = []
        self.summary = {}

    def _get_splitter(self, X, y, groups=None):
        """Get the appropriate splitter based on strategy."""
        if self.strategy == "stratified":
            return StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            ).split(X, y)

        elif self.strategy == "repeated":
            return RepeatedStratifiedKFold(
                n_splits=self.n_folds,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            ).split(X, y)

        elif self.strategy == "group":
            if groups is None:
                raise ValueError("groups required for 'group' strategy")
            return StratifiedGroupKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            ).split(X, y, groups)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def evaluate_model(
            self,
            model: nn.Module,
            dataset: torch.utils.data.Dataset,
            labels: np.ndarray,
            device: str = "cuda",
            batch_size: int = 32,
            num_workers: int = 0,
            groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model using cross-validation.

        Args:
            model: Trained PyTorch model
            dataset: Dataset to evaluate
            labels: Array of labels (N,)
            device: Device
            batch_size: Batch size
            num_workers: Num workers
            groups: Optional group labels (for group strategy)

        Returns:
            Dictionary with CV results
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"🔬 ADVANCED CROSS-VALIDATION ({self.strategy})")
            print("=" * 80)
            print(f"   Strategy: {self.strategy}")
            print(f"   Folds: {self.n_folds}")
            if self.strategy == "repeated":
                print(f"   Repeats: {self.n_repeats}")
            print(f"   Total evaluations: {self.n_folds * (self.n_repeats if self.strategy == 'repeated' else 1)}")
            print(f"   Samples: {len(dataset)}")
            print(f"   Classes: {np.unique(labels)}")
            print("=" * 80)

        # ✅ Get splitter
        X = np.zeros(len(dataset))  # dummy X
        splitter = self._get_splitter(X, labels, groups)

        # ✅ Storage
        self.fold_results = []
        fold_num = 0

        model.eval()

        # ✅ Iterate over folds
        for train_idx, val_idx in splitter:
            fold_num += 1

            if self.verbose:
                print(f"\n📌 Fold {fold_num}/{self.n_folds * (self.n_repeats if self.strategy == 'repeated' else 1)}")
                print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}")

            # ✅ Create validation subset
            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if device == "cuda" else False,
            )

            # ✅ Evaluate
            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    probs = torch.softmax(output, dim=1)
                    _, preds = output.max(1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(target.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)

            # ✅ Compute metrics
            fold_metrics = self._compute_fold_metrics(
                all_labels, all_preds, all_probs
            )

            # ✅ Store results
            self.fold_results.append({
                "fold": fold_num,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                **fold_metrics,
            })

            if self.verbose:
                print(f"   ✅ Accuracy:  {fold_metrics['accuracy']:.4f}")
                print(f"   ✅ Precision: {fold_metrics['precision']:.4f}")
                print(f"   ✅ Recall:    {fold_metrics['recall']:.4f}")
                print(f"   ✅ F1-Score:  {fold_metrics['f1']:.4f}")
                if fold_metrics['auc_macro'] > 0:
                    print(f"   ✅ AUC (macro): {fold_metrics['auc_macro']:.4f}")

        # ✅ Compute summary
        self.summary = self._compute_summary()

        if self.verbose:
            self._print_summary()

        return {
            "fold_results": self.fold_results,
            "summary": self.summary,
            "strategy": self.strategy,
            "n_folds": self.n_folds,
            "n_repeats": self.n_repeats if self.strategy == "repeated" else 1,
        }

    def _compute_fold_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_prob: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all metrics for a fold."""
        metrics = {}

        # ✅ Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ))
        metrics["recall"] = float(recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ))
        metrics["f1"] = float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ))

        # ✅ Per-class metrics
        for class_idx in range(NUM_CLASSES):
            class_name = CLASS_NAMES[class_idx]
            metrics[f"precision_{class_name}"] = float(precision_score(
                y_true, y_pred, labels=[class_idx], average="micro", zero_division=0
            ))
            metrics[f"recall_{class_name}"] = float(recall_score(
                y_true, y_pred, labels=[class_idx], average="micro", zero_division=0
            ))
            metrics[f"f1_{class_name}"] = float(f1_score(
                y_true, y_pred, labels=[class_idx], average="micro", zero_division=0
            ))

        # ✅ AUC (macro)
        try:
            n_classes = len(np.unique(y_true))
            if n_classes > 1:
                metrics["auc_macro"] = float(roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                ))
            else:
                metrics["auc_macro"] = 0.0
        except Exception as e:
            metrics["auc_macro"] = 0.0

        # ✅ Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics across folds."""
        if not self.fold_results:
            return {}

        # ✅ Metrics to summarize
        metric_names = [
            "accuracy", "precision", "recall", "f1", "auc_macro",
            "precision_NL", "recall_NL", "f1_NL",
            "precision_G1", "recall_G1", "f1_G1",
            "precision_G2", "recall_G2", "f1_G2",
            "precision_G3", "recall_G3", "f1_G3",
        ]

        summary = {}

        for metric_name in metric_names:
            values = [r[metric_name] for r in self.fold_results if metric_name in r]
            if values:
                values = np.array(values)
                summary[metric_name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(np.median(values)),
                }

                # ✅ 95% Confidence Interval
                if len(values) > 1:
                    from scipy import stats
                    ci = stats.t.interval(
                        0.95, len(values) - 1,
                        loc=values.mean(),
                        scale=stats.sem(values),
                    )
                    summary[metric_name]["ci_95_lower"] = float(ci[0])
                    summary[metric_name]["ci_95_upper"] = float(ci[1])

        return summary

    def _print_summary(self) -> None:
        """Print CV summary."""
        print("\n" + "=" * 80)
        print(f"📊 CROSS-VALIDATION SUMMARY ({self.strategy})")
        print("=" * 80)

        print(f"\n   {'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print(f"   {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

        # ✅ Main metrics
        main_metrics = ["accuracy", "precision", "recall", "f1", "auc_macro"]
        for metric in main_metrics:
            if metric in self.summary:
                s = self.summary[metric]
                print(
                    f"   {metric:<20} "
                    f"{s['mean']:<12.4f} "
                    f"{s['std']:<12.4f} "
                    f"{s['min']:<12.4f} "
                    f"{s['max']:<12.4f}"
                )

        print(f"\n   📊 Per-Class F1-Score:")
        for class_name in CLASS_NAMES:
            metric = f"f1_{class_name}"
            if metric in self.summary:
                s = self.summary[metric]
                print(f"      {class_name:<6} {s['mean']:.4f} ± {s['std']:.4f}")

        print("\n" + "=" * 80)

    def save_results(
            self,
            output_dir: str,
            prefix: str = "cv_results",
    ) -> Dict[str, str]:
        """
        Save CV results to CSV and JSON.

        Args:
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            Dictionary with saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        # ✅ Save fold results as CSV
        csv_path = os.path.join(output_dir, f"{prefix}_folds.csv")
        df = pd.DataFrame(self.fold_results)
        # Convert confusion matrix to string for CSV
        if "confusion_matrix" in df.columns:
            df["confusion_matrix"] = df["confusion_matrix"].apply(
                lambda x: str(x) if isinstance(x, list) else "[]"
            )
        df.to_csv(csv_path, index=False)

        # ✅ Save summary as JSON
        json_path = os.path.join(output_dir, f"{prefix}_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "strategy": self.strategy,
                "n_folds": self.n_folds,
                "n_repeats": self.n_repeats if self.strategy == "repeated" else 1,
                "summary": self.summary,
                "fold_results": self.fold_results,
            }, f, indent=4, default=str)

        return {
            "csv": csv_path,
            "json": json_path,
        }

    def plot_results(self, output_dir: str, prefix: str = "cv_results") -> List[str]:
        """
        Plot CV results as box plots and bar charts.

        Returns:
            List of saved plot paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_plots = []

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # ✅ Extract per-fold values
            fold_data = defaultdict(list)
            for r in self.fold_results:
                for key in ["accuracy", "precision", "recall", "f1"]:
                    fold_data[key].append(r[key])

            # ✅ Plot 1: Box plot of main metrics
            plt.figure(figsize=(12, 6))
            metric_names = ["accuracy", "precision", "recall", "f1"]
            data = [fold_data[m] for m in metric_names]

            bp = plt.boxplot(data, patch_artist=True, labels=metric_names)
            colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            plt.ylabel("Score", fontsize=12)
            plt.title(f"Cross-Validation Results ({self.strategy})", fontsize=14)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(output_dir, f"{prefix}_boxplot.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            saved_plots.append(plot_path)

            # ✅ Plot 2: Per-fold bar chart
            plt.figure(figsize=(14, 6))
            folds = [r["fold"] for r in self.fold_results]
            x = np.arange(len(folds))
            width = 0.2

            for i, (metric_name, color) in enumerate(zip(
                    metric_names,
                    ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
            )):
                values = [r[metric_name] for r in self.fold_results]
                plt.bar(x + i * width, values, width, label=metric_name, color=color)

            plt.xlabel("Fold", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.title(f"Per-Fold Metrics ({self.strategy})", fontsize=14)
            plt.xticks(x + 1.5 * width, folds)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(output_dir, f"{prefix}_per_fold.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            saved_plots.append(plot_path)

            # ✅ Plot 3: Per-class F1
            plt.figure(figsize=(10, 6))
            class_f1_mean = []
            class_f1_std = []
            for class_name in CLASS_NAMES:
                metric = f"f1_{class_name}"
                if metric in self.summary:
                    class_f1_mean.append(self.summary[metric]["mean"])
                    class_f1_std.append(self.summary[metric]["std"])

            x = np.arange(len(CLASS_NAMES))
            bars = plt.bar(x, class_f1_mean, yerr=class_f1_std, capsize=10,
                           color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])

            for bar, val in zip(bars, class_f1_mean):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

            plt.xlabel("Class", fontsize=12)
            plt.ylabel("F1-Score", fontsize=12)
            plt.title(f"Per-Class F1-Score ({self.strategy})", fontsize=14)
            plt.xticks(x, CLASS_NAMES)
            plt.ylim(0, 1.05)
            plt.grid(alpha=0.3, axis="y")
            plt.tight_layout()

            plot_path = os.path.join(output_dir, f"{prefix}_per_class.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            saved_plots.append(plot_path)

        except Exception as e:
            print(f"⚠️ Plot error: {e}")

        return saved_plots


# ================================================================
# ✅ CROSS-VALIDATION HELPER
# ================================================================

def run_cross_validation(
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        labels: np.ndarray,
        n_folds: int = 5,
        n_repeats: int = 1,
        strategy: str = "stratified",
        batch_size: int = 32,
        device: str = "cuda",
        output_dir: str = None,
        prefix: str = "cv_results",
        save_plots: bool = True,
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run cross-validation.

    Args:
        model: Trained PyTorch model
        dataset: Dataset
        labels: Labels array
        n_folds: Number of folds
        n_repeats: Repeats (for 'repeated' strategy)
        strategy: "stratified", "repeated", "group"
        batch_size: Batch size
        device: Device
        output_dir: Where to save results
        prefix: Filename prefix
        save_plots: Save plots
        verbose: Print progress

    Returns:
        CV results dictionary
    """
    cv = AdvancedCrossValidator(
        n_folds=n_folds,
        n_repeats=n_repeats,
        strategy=strategy,
        verbose=verbose,
    )

    results = cv.evaluate_model(
        model=model,
        dataset=dataset,
        labels=labels,
        device=device,
        batch_size=batch_size,
    )

    # ✅ Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_result = cv.save_results(output_dir, prefix)
        results["saved_files"] = save_result

        if save_plots:
            plot_paths = cv.plot_results(output_dir, prefix)
            results["saved_plots"] = plot_paths

    return results


# ================================================================
# ✅ FOCAL LOSS (Lin et al., 2017)
# ================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection (Lin et al., 2017).

    Focal Loss is a variant of CrossEntropyLoss that focuses on hard examples
    by down-weighting the loss assigned to well-classified examples.

    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
        - p_t: probability of the correct class
        - α_t: class balancing weight (can be class weights)
        - γ (gamma): focus parameter (higher = more focus on hard examples)

    Advantages over CrossEntropyLoss:
        - Handles class imbalance better
        - Focuses on hard examples (like G3 in our dataset)
        - Works well with noisy labels
        - Combined with class weights = very powerful for imbalanced datasets

    Args:
        alpha: Class balancing weights (tensor) or scalar (float)
        gamma: Focus parameter (default: 2.0 from original paper)
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Label smoothing factor (default: 0.0)

    Example:
        >>> # Basic usage
        >>> criterion = FocalLoss(gamma=2.0)
        >>> loss = criterion(logits, targets)
        >>>
        >>> # With class weights
        >>> class_weights = torch.tensor([0.39, 0.65, 0.87, 2.09])
        >>> criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
            self,
            alpha: Optional[torch.Tensor] = None,
            gamma: float = 2.0,
            reduction: str = "mean",
            label_smoothing: float = 0.0,
    ) -> None:
        super(FocalLoss, self).__init__()

        # ✅ Store parameters
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # ✅ Validation
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.

        Args:
            inputs: (batch_size, num_classes) - raw logits from model
            targets: (batch_size,) - ground truth labels

        Returns:
            Focal loss value
        """
        # ==========================================
        # Step 1: Compute CrossEntropy Loss (no reduction)
        # ==========================================
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",  # ← important: we need per-sample loss
            label_smoothing=self.label_smoothing,
        )

        # ==========================================
        # Step 2: Compute p_t (probability of correct class)
        # ==========================================
        # p_t = exp(-ce_loss) for the correct class (mathematically equivalent)
        # But we compute it directly for numerical stability:

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)

        # Get probability of the correct class
        # p_t = p[target]
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # ✅ Clamp to avoid numerical issues
        p_t = torch.clamp(p_t, min=1e-8, max=1.0)

        # ==========================================
        # Step 3: Compute Focal Loss
        # ==========================================
        # FL = (1 - p_t)^γ * CE_loss
        focal_weight = (1.0 - p_t) ** self.gamma

        # ✅ Apply focal weight to CE loss
        focal_loss = focal_weight * ce_loss

        # ==========================================
        # Step 4: Reduction
        # ==========================================
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # none
            return focal_loss

    def __repr__(self) -> str:
        return (
            f"FocalLoss(gamma={self.gamma}, "
            f"alpha={'None' if self.alpha is None else 'tensor'}, "
            f"label_smoothing={self.label_smoothing}, "
            f"reduction={self.reduction})"
        )


# ================================================================
# ✅ FOCAL LOSS VARIANTS
# ================================================================

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for 2-class problems.
    Uses sigmoid + binary cross entropy formulation.
    """

    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = "mean",
    ) -> None:
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE loss per sample
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # p_t
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)

        # alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    def __repr__(self) -> str:
        return f"BinaryFocalLoss(alpha={self.alpha}, gamma={self.gamma})"


# ================================================================
# ✅ COMBINED LOSS (Focal + Label Smoothing + Class Weights)
# ================================================================
class CombinedLoss(nn.Module):
    """
    Ultimate Combined Loss for imbalanced datasets.

    Combines:
        - Focal Loss (hard example mining)
        - Class Weights (class balancing)
        - Label Smoothing (calibration)
        - Adaptive Smoothing (optional, changes over epochs)

    This is the most comprehensive loss function for imbalanced medical image classification.
    """

    def __init__(
            self,
            class_weights: Optional[torch.Tensor] = None,
            gamma: float = 2.0,
            label_smoothing: float = 0.1,
            use_focal: bool = True,
            use_adaptive_smoothing: bool = False,
            total_epochs: int = 100,
            reduction: str = "mean",
    ) -> None:
        super(CombinedLoss, self).__init__()

        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.use_adaptive_smoothing = use_adaptive_smoothing
        self.total_epochs = total_epochs
        self.reduction = reduction

        # ✅ Build internal criterion
        if use_focal:
            # Focal Loss with everything
            self.criterion = FocalLossWithSmoothing(
                alpha=class_weights,
                gamma=gamma,
                label_smoothing=label_smoothing,
                use_adaptive_smoothing=use_adaptive_smoothing,
                total_epochs=total_epochs,
                reduction=reduction,
            )
        else:
            # CrossEntropyLoss with class weights + label smoothing
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
                reduction=reduction,
            )

    def step(self, epoch: int) -> None:
        """Update adaptive parameters based on epoch."""
        if hasattr(self.criterion, "step"):
            self.criterion.step(epoch)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(inputs, targets)

    def __repr__(self) -> str:
        if self.use_focal:
            return (
                f"CombinedLoss("
                f"Focal, gamma={self.gamma}, "
                f"label_smoothing={self.label_smoothing}, "
                f"adaptive={self.use_adaptive_smoothing}, "
                f"has_class_weights={self.class_weights is not None}"
                f")"
            )
        else:
            return (
                f"CombinedLoss("
                f"CE, "
                f"label_smoothing={self.label_smoothing}, "
                f"has_class_weights={self.class_weights is not None}"
                f")"
            )


# ================================================================
# ✅ ADVANCED LABEL SMOOTHING
# ================================================================

class AdaptiveLabelSmoothing(nn.Module):
    """
    Adaptive Label Smoothing that adjusts the smoothing factor
    based on training progress or class imbalance.

    This is more sophisticated than standard label smoothing:
        - Higher smoothing for minority classes
        - Reduces smoothing as training progresses
        - Can be combined with Focal Loss

    Formula:
        For each sample with class c:
            epsilon_c = epsilon_base * imbalance_factor_c
            y_smooth = (1 - epsilon_c) * y_one_hot + epsilon_c / K

    Args:
        num_classes: Number of classes
        epsilon: Base smoothing factor (default: 0.1)
        class_weights: Optional class weights to adjust per-class smoothing
        use_adaptive: If True, reduces smoothing over epochs
        total_epochs: Total training epochs (for adaptive schedule)
        min_epsilon: Minimum epsilon during adaptive schedule
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
            self,
            num_classes: int = 4,
            epsilon: float = 0.1,
            class_weights: Optional[torch.Tensor] = None,
            use_adaptive: bool = False,
            total_epochs: int = 100,
            min_epsilon: float = 0.01,
            reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.base_epsilon = epsilon
        self.current_epsilon = epsilon
        self.class_weights = class_weights
        self.use_adaptive = use_adaptive
        self.total_epochs = total_epochs
        self.min_epsilon = min_epsilon
        self.reduction = reduction

        # ✅ Normalize class weights to [0.5, 1.5] range
        if class_weights is not None:
            w = class_weights / class_weights.mean()
            # Scale to [0.5, 1.5]
            self.class_epsilon_multiplier = 0.5 + w * 0.5
        else:
            self.class_epsilon_multiplier = None

    def step(self, epoch: int) -> None:
        """Update epsilon based on epoch (for adaptive schedule)."""
        if not self.use_adaptive:
            return

        # Linear decay from base_epsilon to min_epsilon
        progress = min(epoch / self.total_epochs, 1.0)
        self.current_epsilon = (
                self.base_epsilon * (1 - progress) +
                self.min_epsilon * progress
        )

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with adaptive label smoothing.

        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - labels

        Returns:
            Loss value
        """
        # ✅ Compute standard CE with smoothing
        if self.reduction == "none":
            # ✅ Per-sample smoothing for adaptive
            if self.class_epsilon_multiplier is not None:
                # Different epsilon per class
                per_sample_epsilon = self.current_epsilon * self.class_epsilon_multiplier[targets]

                # Get per-sample CE loss
                ce_loss = F.cross_entropy(
                    inputs, targets,
                    reduction="none",
                    label_smoothing=0.0  # We'll handle smoothing manually
                )

                # ✅ Apply per-sample smoothing
                # Compute smoothed target distribution
                log_softmax = F.log_softmax(inputs, dim=-1)

                # Manual label smoothing
                n_classes = inputs.size(-1)
                smooth_targets = torch.zeros_like(inputs).scatter_(
                    1, targets.unsqueeze(1), 1.0
                )
                smooth_targets = (
                        smooth_targets * (1 - per_sample_epsilon.unsqueeze(1)) +
                        per_sample_epsilon.unsqueeze(1) / n_classes
                )

                loss = -(smooth_targets * log_softmax).sum(dim=-1)
            else:
                # ✅ Standard smoothing
                loss = F.cross_entropy(
                    inputs, targets,
                    reduction="none",
                    label_smoothing=self.current_epsilon,
                )

            return loss
        else:
            # ✅ Standard reduction
            return F.cross_entropy(
                inputs, targets,
                reduction=self.reduction,
                label_smoothing=self.current_epsilon,
            )

    def __repr__(self) -> str:
        return (
            f"AdaptiveLabelSmoothing("
            f"epsilon={self.current_epsilon:.4f}, "
            f"adaptive={self.use_adaptive}, "
            f"min_epsilon={self.min_epsilon}"
            f")"
        )


class FocalLossWithSmoothing(nn.Module):
    """
    Complete Focal Loss with Adaptive Label Smoothing and Class Weights.

    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t^smooth)

    Where p_t^smooth is computed with label smoothing.

    This combines 3 techniques:
        - Focal Loss: Focus on hard examples
        - Class Weights: Balance classes
        - Label Smoothing: Prevent overconfidence

    Args:
        alpha: Class weights (tensor)
        gamma: Focus parameter (default: 2.0)
        label_smoothing: Smoothing factor (default: 0.1)
        use_adaptive_smoothing: Enable adaptive smoothing
        reduction: 'mean', 'sum', 'none'
    """

    def __init__(
            self,
            alpha: Optional[torch.Tensor] = None,
            gamma: float = 2.0,
            label_smoothing: float = 0.1,
            use_adaptive_smoothing: bool = False,
            total_epochs: int = 100,
            reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.base_label_smoothing = label_smoothing
        self.current_label_smoothing = label_smoothing
        self.use_adaptive_smoothing = use_adaptive_smoothing
        self.total_epochs = total_epochs
        self.reduction = reduction

    def step(self, epoch: int) -> None:
        """Update label smoothing based on epoch."""
        if not self.use_adaptive_smoothing:
            return

        # Reduce smoothing over time
        progress = min(epoch / self.total_epochs, 1.0)
        self.current_label_smoothing = self.base_label_smoothing * (1 - progress * 0.8)

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Focal Loss with label smoothing."""

        # ✅ Step 1: Standard CE with label smoothing (per-sample)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.current_label_smoothing,
        )

        # ✅ Step 2: Compute p_t
        with torch.no_grad():
            p = F.softmax(inputs, dim=1)
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            p_t = torch.clamp(p_t, min=1e-8, max=1.0)

        # ✅ Step 3: Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma

        # ✅ Step 4: Focal loss
        focal_loss = focal_weight * ce_loss

        # ✅ Step 5: Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    def __repr__(self) -> str:
        return (
            f"FocalLossWithSmoothing("
            f"gamma={self.gamma}, "
            f"label_smoothing={self.current_label_smoothing:.4f}, "
            f"adaptive={self.use_adaptive_smoothing}, "
            f"has_alpha={self.alpha is not None}"
            f")"
        )

# ================================================================
# ✅ CLASS WEIGHTING / WEIGHTED LOSS
# ================================================================

def compute_class_weights(
        labels: Optional[List[int]] = None,
        labels_csv_path: Optional[str] = None,
        num_classes: int = 4,
        method: str = "balanced",
        device: str = "cpu",
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    This function computes weights inversely proportional to class frequencies,
    which helps the model pay more attention to minority classes.

    Args:
        labels: List of integer labels (0 to num_classes-1). If None, reads from CSV.
        labels_csv_path: Path to labels.csv. Used if labels is None.
        num_classes: Number of classes (default: 4 for NL, G1, G2, G3)
        method: Weighting method:
            - "balanced": n_samples / (n_classes * class_count) - sklearn style
            - "inverse": 1 / class_count - simple inverse
            - "sqrt": 1 / sqrt(class_count) - softer weighting
            - "effective": (1 - beta) / (1 - beta^n) - effective number of samples
        device: Device to put weights on ("cpu" or "cuda")

    Returns:
        torch.Tensor of shape (num_classes,) with class weights

    Example:
        >>> weights = compute_class_weights(labels_csv_path="finalReports/labels.csv")
        >>> print(weights)
        tensor([0.3958, 1.1211, 2.0205, 11.5213])  # NL, G1, G2, G3
    """
    import numpy as np
    import pandas as pd
    from collections import Counter

    # ==========================================
    # Step 1: Get labels
    # ==========================================
    if labels is None:
        if labels_csv_path is None:
            raise ValueError("Either 'labels' or 'labels_csv_path' must be provided")

        # ✅ Read labels from CSV
        if not os.path.exists(labels_csv_path):
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv_path}")

        df = pd.read_csv(labels_csv_path)

        # ✅ Filter valid labels (NL, G1, G2, G3 only)
        class_names = ["NL", "G1", "G2", "G3"]
        df = df[df["liverLabel"].isin(class_names)]

        # ✅ Map labels to integers
        label_map = {name: idx for idx, name in enumerate(class_names)}
        labels = df["liverLabel"].map(label_map).dropna().astype(int).tolist()

        print(f"📊 Read {len(labels)} labels from CSV")

    # ==========================================
    # Step 2: Count classes
    # ==========================================
    labels_array = np.array(labels)
    class_counts = np.bincount(labels_array, minlength=num_classes)

    # ✅ Handle zero-count classes (avoid division by zero)
    class_counts = np.maximum(class_counts, 1)

    print(f"\n📊 Class Distribution:")
    class_names = ["NL", "G1", "G2", "G3"][:num_classes]
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"   {name}: {count} samples ({count / len(labels) * 100:.2f}%)")

    # ==========================================
    # Step 3: Compute weights
    # ==========================================
    n_samples = len(labels)

    if method == "balanced":
        # ✅ sklearn style: n_samples / (n_classes * class_count)
        # Weights are normalized so that the mean weight is ~1
        weights = n_samples / (num_classes * class_counts)

    elif method == "inverse":
        # ✅ Simple inverse frequency
        weights = 1.0 / class_counts
        # Normalize to sum to num_classes
        weights = weights / weights.sum() * num_classes

    elif method == "sqrt":
        # ✅ Inverse square root (softer)
        weights = 1.0 / np.sqrt(class_counts)
        weights = weights / weights.sum() * num_classes

    elif method == "effective":
        # ✅ Effective number of samples (Cui et al., 2019)
        # Best for very imbalanced datasets
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes

    else:
        raise ValueError(f"Unknown method: {method}")

    # ==========================================
    # Step 4: Convert to tensor
    # ==========================================
    weights = np.array(weights, dtype=np.float32)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"\n✅ Class Weights (method={method}):")
    for i, (name, weight) in enumerate(zip(class_names, weights)):
        print(f"   {name}: {weight:.4f}")
    print(f"   Sum: {weights.sum():.4f}")
    print(f"   Max/Min ratio: {weights.max() / weights.min():.2f}x")

    return weights_tensor

# ================================================================
# ✅ SMART LEARNING RATE SCHEDULER FOR EACH PHASE
# ================================================================


def get_phase_lr(phase: int) -> float:
    """
    Get optimal learning rate for each phase based on global standards.

    Args:
        phase: Phase number (0-indexed)

    Returns:
        float: Optimal learning rate for the phase
    """
    PHASE_LRS = {
        0: 1e-3,  # Phase 1: FC layer only (higher LR for new layer)
        1: 5e-4,  # Phase 2: Layer4 + FC
        2: 1e-4,  # Phase 3: Layer3+4 + FC (lower LR)
        3: 5e-5,  # Phase 4: Full fine-tuning (very low LR)
    }
    return PHASE_LRS.get(phase, 1e-4)


# ================================================================
# Directory Management
# ================================================================


def ensure_directories() -> None:
    """Create all necessary directories for ResNet18."""
    directories = [
        RESNET18_BASE_DIR,
        RESNET18_CHECKPOINTS_DIR,
        RESNET18_REPORTS_DIR,
        RESNET18_CONFIGS_DIR,
        RESNET18_PLOTS_DIR,
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

    print(f"✅ Directories created: {RESNET18_BASE_DIR}")


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
    """
    if save_dir is None:
        save_dir = RESNET18_CHECKPOINTS_DIR

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # ✅ ذخیره best_accuracy در checkpoint
    best_accuracy = extra_info.get("best_accuracy", 0.0) if extra_info else 0.0

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "best_accuracy": best_accuracy,  # ✅ اضافه شد
        "timestamp": datetime.now().isoformat(),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "model_architecture": "ResNet18",
        "version": "4.0.0",
    }

    if scheduler is not None:
        try:
            if hasattr(scheduler, "state_dict"):
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        except:
            pass

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
    Load a complete model checkpoint.

    ✅ FIXED v4: NEVER falls back to CPU if CUDA was working
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    try:
        # ==========================================
        # ✅ FIX: Check CUDA state FRESH (not cached)
        # ==========================================
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0

        print(f"🔍 CUDA Check at load time:")
        print(f"   torch.cuda.is_available(): {cuda_available}")
        print(f"   torch.cuda.device_count(): {cuda_count}")

        # ==========================================
        # ✅ Load checkpoint ALWAYS on CPU first
        # ==========================================
        # This is the SAFEST approach
        checkpoint = torch.load(
            filepath,
            map_location="cpu",  # ← ALWAYS load to CPU first
            weights_only=False,
        )

        # ==========================================
        # ✅ Load state dict
        # ==========================================
        model.load_state_dict(checkpoint["model_state_dict"])

        # ==========================================
        # ✅ Move model to requested device
        # ==========================================
        if isinstance(device, str) and device.startswith("cuda"):
            if cuda_available and cuda_count > 0:
                model = model.to(device)
                print(f"✅ Model moved to: {device}")
            else:
                # ⚠️ CUDA not available - but DON'T change device, keep it as-is
                # The caller should handle this
                print(f"⚠️ CUDA requested but not available!")
                print(f"   → Loading on CPU. WARNING: caller must handle device mismatch")
                model = model.to("cpu")
                device = "cpu"
        else:
            model = model.to(device)
            print(f"✅ Model moved to: {device}")

        # ==========================================
        # ✅ Load optimizer state
        # ==========================================
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # Move optimizer state to device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except Exception as e:
                print(f"⚠️ Could not load optimizer state: {e}")

        # ==========================================
        # ✅ Load scheduler state
        # ==========================================
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"⚠️ Could not load scheduler state: {e}")

        print(f"✅ Checkpoint loaded: {filepath} (device: {device})")
        return checkpoint

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        raise


# ================================================================
# ✅ ADVANCED CALLBACKS 🔥
# ================================================================


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min"):
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
    filepath = os.path.join(RESNET18_CONFIGS_DIR, filename)

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
    """Plot and save confusion matrix with professional style."""
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
    plt.title("Confusion Matrix - ResNet18", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")


def plot_roc_curves(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], save_path: str
) -> None:
    """Plot and save ROC curves for all classes."""
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
    plt.title("ROC Curves - ResNet18", fontsize=16, fontweight="bold")
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
    plt.title("ResNet18 Performance Metrics", fontsize=16, fontweight="bold")

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
    print(f"✅ Metrics bar chart saved: {save_path}")


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

    bars1 = ax.bar(
        x - width,
        precision,
        width,
        label="Precision",
        color="#3498db",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x,
        recall,
        width,
        label="Recall",
        color="#2ecc71",
        edgecolor="black",
        linewidth=1.5,
    )
    bars3 = ax.bar(
        x + width,
        f1,
        width,
        label="F1-Score",
        color="#9b59b6",
        edgecolor="black",
        linewidth=1.5,
    )

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
    ax.set_title("Per-Class Metrics - ResNet18", fontsize=16, fontweight="bold")
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
    plt.title("ResNet18 Cross-Validation Results", fontsize=16, fontweight="bold")
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
        output_path = os.path.join(RESNET18_REPORTS_DIR, "ResNet18_Report.pdf")

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
    story.append(Paragraph("ResNet18 Model Report", title_style))
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
            ["Architecture", model_info.get("architecture", "ResNet18")],
            ["Total Parameters", f"{model_info.get('total_params', 0):,}"],
            ["Trainable Parameters", f"{model_info.get('trainable_params', 0):,}"],
            ["Input Shape", str(model_info.get("input_shape", (3, 224, 224)))],
            ["Number of Classes", str(model_info.get("num_classes", 4))],
            ["Class Names", ", ".join(model_info.get("class_names", CLASS_NAMES))],
            ["Device", model_info.get("device", "CUDA")],
            [
                "Training Status",
                "Trained" if model_info.get("is_trained", False) else "Untrained",
            ],
            ["Feature Dimension", str(model_info.get("feature_dim", 512))],
            ["Activation", model_info.get("activation", "ReLU")],
            ["Training Stages", "4 Stages (FC → Layer4 → Layer3+4 → Full)"],
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

    # Generate and add plots
    plot_paths = {}

    # Confusion Matrix
    cm_path = os.path.join(RESNET18_PLOTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    plot_paths["confusion_matrix"] = cm_path

    # ROC Curves
    roc_path = os.path.join(RESNET18_PLOTS_DIR, "roc_curves.png")
    plot_roc_curves(y_true, y_prob, class_names, roc_path)
    plot_paths["roc_curves"] = roc_path

    # Metrics Bar
    bar_path = os.path.join(RESNET18_PLOTS_DIR, "metrics_bar.png")
    plot_metrics_bar(metrics, bar_path)
    plot_paths["metrics_bar"] = bar_path

    # Per-class metrics
    if per_class:
        per_class_path = os.path.join(RESNET18_PLOTS_DIR, "per_class_metrics.png")
        plot_per_class_metrics(per_class, per_class_path)
        plot_paths["per_class_metrics"] = per_class_path

    # Training History
    if training_history:
        hist_path = os.path.join(RESNET18_PLOTS_DIR, "training_history.png")
        plot_training_history(training_history, hist_path)
        plot_paths["training_history"] = hist_path

    # Add plots to PDF
    for name, path in plot_paths.items():
        if os.path.exists(path):
            story.append(PageBreak())
            story.append(
                Paragraph(f"3. {name.replace('_', ' ').title()}", heading_style)
            )
            story.append(RLImage(path, width=6 * inch, height=5 * inch))
            story.append(Spacer(1, 10))

    # Cross-Validation Results
    if cv_scores:
        story.append(PageBreak())
        story.append(Paragraph("4. Cross-Validation Results", heading_style))

        cv_data = [
            ["Fold"]
            + [f"CV-{i+1}" for i in range(len(next(iter(cv_scores.values()))))]
            + ["Mean", "Std"]
        ]

        for metric_name, scores in cv_scores.items():
            row = (
                [metric_name.capitalize()]
                + [f"{s:.4f}" for s in scores]
                + [f"{np.mean(scores):.4f}", f"{np.std(scores):.4f}"]
            )
            cv_data.append(row)

        cv_table = Table(
            cv_data,
            colWidths=[1.5 * inch]
            + [1 * inch] * (len(cv_scores[list(cv_scores.keys())[0]]) + 2),
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

        cv_path = os.path.join(RESNET18_PLOTS_DIR, "cv_results.png")
        plot_cross_validation_results(cv_scores, cv_path)

        if os.path.exists(cv_path):
            story.append(RLImage(cv_path, width=6 * inch, height=4 * inch))

    # Summary
    story.append(PageBreak())
    story.append(Paragraph("5. Summary & Conclusion", heading_style))

    conclusion_text = f"""
    <b>ResNet18 Performance Summary:</b><br/><br/>
    • Total Parameters: {model_info.get('total_params', 0):,}<br/>
    • Best Accuracy: {metrics.get('accuracy', 0):.4f}<br/>
    • Weighted F1-Score: {metrics.get('f1_score', 0):.4f}<br/>
    • Number of Classes: {len(class_names)}<br/>
    • Classes: {', '.join(class_names)}<br/>
    • Feature Dimension: {model_info.get('feature_dim', 512)}<br/>
    • Activation: {model_info.get('activation', 'ReLU')}<br/>
    • Training Stages: 4 Stages<br/>
    """

    if cv_scores:
        cv_mean = np.mean(cv_scores.get("accuracy", [0]))
        cv_std = np.std(cv_scores.get("accuracy", [0]))
        conclusion_text += (
            f"• Cross-Validation Accuracy: {cv_mean:.4f} ± {cv_std:.4f}<br/>"
        )

    conclusion_text += """
    <br/><b>Recommendations:</b><br/><br/>
    • ResNet18 is efficient with only ~11.7M parameters<br/>
    • BasicBlock architecture provides strong feature extraction<br/>
    • ReLU activation and batch normalization improve training<br/>
    • Multi-stage training reduces overfitting<br/>
    • Smart LR scheduling improves convergence<br/>
    """

    story.append(Paragraph(conclusion_text, normal_style))
    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            f"Report generated by Liver AI Pipeline - Version 5.2.0", styles["Italic"]
        )
    )

    doc.build(story)
    print(f"✅ PDF report saved to: {output_path}")
    return output_path


# ================================================================
# ✅ MAIN WRAPPER CLASS - ULTIMATE EDITION 🔥
# ================================================================


class ResNet18WithoutFineTuning:
    """
    Ultimate OOP Wrapper for ResNet18 from Scratch.

    Features:
    - Full OOP structure with comprehensive methods
    - Multi-stage training (4 stages: FC → Layer4 → Layer3+4 → Full)
    - Smart Learning Rate per stage
    - Advanced callbacks (Early Stopping, Checkpoint, LR Scheduler)
    - Comprehensive PDF reports with all visualizations
    - K-Fold cross-validation
    - Feature extraction for transfer learning
    - Grad-CAM visualization support
    - Complete model persistence (save/load)
    - Model profiling and FLOPs calculation
    """

    def __init__(
        self,
        num_classes: int = 4,
        zero_init_residual: bool = True,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
        target_size: int = 224,
        checkpoints_dir: str = None,
        configs_dir: str = None,
        plots_dir: str = None,
        use_class_weights: bool = True,
        class_weight_method: str = "sqrt",
        label_smoothing: float = 0.1,
        labels_csv_path: str = None,
        use_adaptive_smoothing: bool = False,  # ← ✅ NEW
        # ✅ NEW: Focal Loss
        use_focal_loss: bool = True,  # ← ✅ فعال‌سازی Focal Loss
        focal_gamma: float = 2.0,  # ← ✅ Focus parameter
        focal_alpha: Optional[float] = None,  # ← ✅ (اختیاری) scalar alpha
        # ✅ NEW: Scheduler params
        scheduler_type: str = "warmup_cosine",  # ← ✅ نوع scheduler
        warmup_epochs: int = 5,  # ← ✅ warmup epochs
        min_lr: float = 1e-6,  # ← ✅ حداقل LR
        total_epochs: int = 100,  # ← ✅ NEW
        **kwargs
    ) -> None:
        """Initialize the wrapper."""
        self.num_classes = num_classes
        self.zero_init_residual = zero_init_residual
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.device = device
        self.verbose = verbose
        self._feature_dim = 512
        self._target_size = target_size

        # ✅ NEW: Store class weighting params
        self.use_class_weights = use_class_weights
        self.class_weight_method = class_weight_method
        self.label_smoothing = label_smoothing
        self.labels_csv_path = labels_csv_path or os.path.join(
            "finalReports", "labels.csv"
        )
        self.class_weights = None  # ← محاسبه میشه در build_model

        # ✅ Label Smoothing params
        self.label_smoothing = label_smoothing
        self.use_adaptive_smoothing = use_adaptive_smoothing  # ← ✅ NEW

        # ✅ NEW: Focal Loss params
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # ✅ NEW: Scheduler
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

        # ✅ Total epochs for adaptive schedule
        self.total_epochs = total_epochs  # ← ✅ NEW

        # ✅ ذخیره مسیرهای جدید
        self.checkpoints_dir = checkpoints_dir or RESNET18_CHECKPOINTS_DIR
        self.configs_dir = configs_dir or RESNET18_CONFIGS_DIR
        self.plots_dir = plots_dir or RESNET18_PLOTS_DIR

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_history = []
        self.best_accuracy = 0.0
        self.is_trained = False
        self.class_names = CLASS_NAMES[:num_classes]
        self.callbacks = {}
        self.phase_lrs = {}  # ✅ ذخیره نرخ یادگیری هر فاز
        self.val_loader = None  # ✅ ذخیره val_loader برای validation

        self.logger = logging.getLogger("ResNet18CV")
        self.logger.setLevel(logging.INFO)

        ensure_directories()

        if self.verbose:
            print(f"\n🚀 Initializing ResNet18WithoutFineTuning:")
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
            print(f"   - Label Smoothing: {label_smoothing}")
            print(f"   - Adaptive Smoothing: {use_adaptive_smoothing}")  # ← ✅ NEW
            print(f"   - Output Directory: {RESNET18_BASE_DIR}")
            print(f"   - Feature Dimension: {self._feature_dim}")
            print(f"   - Checkpoints Dir: {self.checkpoints_dir}")
            print(f"   - Configs Dir: {self.configs_dir}")
            print(f"   - Plots Dir: {self.plots_dir}")
            print(f"   - Scheduler Type: {scheduler_type}")  # ← ✅ NEW
            print(f"   - Warmup Epochs: {warmup_epochs}")  # ← ✅ NEW
            print(f"   - Min LR: {min_lr}")  # ← ✅ NEW
            print(f"   - Total Epochs: {total_epochs}")  # ← ✅ NEW

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(handler)

    # ================================================================
    # Model Management
    # ================================================================

    def build_model(self) -> None:
        """Build the ResNet18 model from scratch with class weighting + focal loss."""
        if self.verbose:
            print("\n🔨 Building ResNet18 from scratch...")

        self.model = _create_resnet18(
            num_classes=self.num_classes,
            pretrained=False,
            zero_init_residual=self.zero_init_residual,
            use_batch_norm=self.use_batch_norm,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
        )

        self.model = self.model.to(self.device)

        # ==========================================
        # ✅ Optimizer
        # ==========================================
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=get_phase_lr(0),
            weight_decay=5e-5,
        )

        # ==========================================
        # ✅ Scheduler
        # ==========================================
        # ✅ NEW: Advanced Scheduler with Warmup
        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.scheduler_type,
            total_epochs=self.total_epochs,
            warmup_epochs=self.warmup_epochs,
            min_lr=self.min_lr,
        )

        if self.verbose:
            print(f"\n✅ Scheduler: {self.scheduler}")
            print(f"   - Type: {self.scheduler_type}")
            if hasattr(self.scheduler, "warmup_epochs"):
                print(f"   - Warmup: {self.scheduler.warmup_epochs} epochs")
            if hasattr(self.scheduler, "base_lr"):
                print(f"   - Base LR: {self.scheduler.base_lr:.2e}")
            if hasattr(self.scheduler, "warmup_start_lr"):
                print(f"   - Warmup Start LR: {self.scheduler.warmup_start_lr:.2e}")
            if hasattr(self.scheduler, "eta_min"):
                print(f"   - Min LR: {self.scheduler.eta_min:.2e}")

        # ==========================================
        # ✅ Class Weights (if enabled)
        # ==========================================
        if self.use_class_weights:
            if self.verbose:
                print("\n" + "=" * 60)
                print("⚖️ COMPUTING CLASS WEIGHTS")
                print("=" * 60)

            try:
                self.class_weights = compute_class_weights(
                    labels=None,
                    labels_csv_path=self.labels_csv_path,
                    num_classes=self.num_classes,
                    method=self.class_weight_method,
                    device=self.device,
                )
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Failed to compute class weights: {e}")
                self.class_weights = None

        # ==========================================
        # ✅ Criterion: Focal Loss + Class Weights + Label Smoothing
        # ==========================================
        if self.use_focal_loss:
            # ✅ Use Combined Loss (Focal + Class Weights + Label Smoothing)
            self.criterion = CombinedLoss(
                class_weights=self.class_weights,
                gamma=self.focal_gamma,
                label_smoothing=self.label_smoothing,
                use_focal=self.use_focal_loss,
                use_adaptive_smoothing=self.use_adaptive_smoothing,  # ← ✅ NEW
                total_epochs=self.total_epochs,  # ← ✅ NEW
                reduction="mean",
            )

            if self.verbose:
                print(f"\n✅ Criterion: {self.criterion}")
                print(f"   - Label Smoothing: {self.label_smoothing}")
                print(f"   - Adaptive Smoothing: {self.use_adaptive_smoothing}")
                if self.class_weights is not None:
                    print(f"   - Class Weights: {self.class_weights.cpu().numpy()}")
                if self.use_focal_loss:
                    print(f"   - Focal Gamma: {self.focal_gamma}")
                print("=" * 60)

        else:
            # ✅ Standard CrossEntropyLoss with Class Weights + Label Smoothing
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )

            if self.verbose:
                print(f"\n✅ Criterion: CrossEntropyLoss")
                print(f"   - Class Weights: {'✅ Yes' if self.class_weights is not None else '❌ No'}")
                if self.class_weights is not None:
                    print(f"      Values: {self.class_weights.cpu().numpy()}")
                print(f"   - Label Smoothing: {self.label_smoothing}")

        # ==========================================
        # ✅ Model Info
        # ==========================================
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        if self.verbose:
            print(f"\n✅ Model built successfully!")
            print(f"   - Total Parameters: {total_params:,}")
            print(f"   - Trainable Parameters: {trainable_params:,}")
            print(f"   - Model on: {self.device}")
            print(f"   - Feature Dimension: {self._feature_dim}")

        # ==========================================
        # ✅ Save Config
        # ==========================================
        config = {
            "architecture": "ResNet18",
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
            "stages": 4,
            "best_accuracy": self.best_accuracy,
            "is_trained": self.is_trained,
            # ✅ Class weighting
            "use_class_weights": self.use_class_weights,
            "class_weight_method": self.class_weight_method,
            "label_smoothing": self.label_smoothing,
            "use_adaptive_smoothing": self.use_adaptive_smoothing,
            "scheduler_type": self.scheduler_type,
            "warmup_epochs": self.warmup_epochs,
            "min_lr": self.min_lr,
            "class_weights": (
                self.class_weights.cpu().numpy().tolist()
                if self.class_weights is not None else None
            ),
            # ✅ Focal loss
            "use_focal_loss": self.use_focal_loss,
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
            "total_epochs": self.total_epochs,
        }
        save_config(config, "config_resnet18.json")

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
        """Print comprehensive model summary."""
        if self.model is None:
            print("⚠️ Model not built yet. Call build_model() first.")
            return

        # ✅ Ensure model is on the correct device
        self.model = self.model.to(self.device)

        print_model_summary(
            self.model, input_shape=(3, self._target_size, self._target_size)
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
    # Validation Method
    # ================================================================

    def validate(self, val_loader: DataLoader = None, return_loss: bool = False):
        """
        Validate the model on validation set.

        Args:
            val_loader: DataLoader for validation (uses self.val_loader if None)
            return_loss: Whether to return loss along with accuracy

        Returns:
            tuple: (loss, accuracy) if return_loss=True, else accuracy
        """
        if self.model is None:
            self.build_model()

        if val_loader is None:
            val_loader = self.val_loader

        if val_loader is None:
            raise ValueError(
                "No validation loader provided. Please pass val_loader or set self.val_loader."
            )

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        if return_loss:
            return avg_loss, accuracy
        return accuracy

    # ================================================================
    # Training Methods with Multi-Stage (4 Stages)
    # ================================================================

    def train_epoch(
        self, train_loader: DataLoader, epochs: int, total_epochs: int, desc: str = ""
    ) -> float:
        """
        Train for a given number of epochs with optional description.

        Args:
            train_loader: DataLoader for training
            epochs: Number of epochs to train
            total_epochs: Total epochs (for display)
            desc: Description for progress bar

        Returns:
            float: Best accuracy achieved
        """
        best_acc = self.best_accuracy

        for epoch in range(epochs):
            # Update current epoch counter
            current_epoch = len(self.train_history) + 1

            # ===== UPDATE ADAPTIVE PARAMETERS =====
            if hasattr(self.criterion, "step"):
                self.criterion.step(current_epoch)

                # ✅ Print adaptive info every 10 epochs
                if self.verbose and current_epoch % 10 == 0:
                    if hasattr(self.criterion.criterion, "current_label_smoothing"):
                        print(f"   📊 Adaptive Label Smoothing: {self.criterion.criterion.current_label_smoothing:.4f}")

            epoch_start = time.time()

            # ===== TRAINING PHASE =====
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # Reset gradients
            self.optimizer.zero_grad()

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

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total_samples += target.size(0)
                total_correct += predicted.eq(target).sum().item()

                loop.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": (
                            f"{total_correct/total_samples:.4f}"
                            if total_samples > 0
                            else "0.0000"
                        ),
                    }
                )

            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            train_acc = total_correct / total_samples if total_samples > 0 else 0.0

            # ===== VALIDATION PHASE =====
            val_loss, val_acc = self.validate(return_loss=True)

            # ===== UPDATE SCHEDULER (FIXED) =====
            # ✅ Handle different scheduler types
            if isinstance(self.scheduler, ReduceLROnPlateau):
                # Plateau needs val_loss
                self.scheduler.step(val_loss)
            else:
                # Other schedulers don't need metrics
                self.scheduler.step()

            # ✅ گرفتن LR از optimizer
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ===== SAVE HISTORY =====
            epoch_time = time.time() - epoch_start
            self.train_history.append(
                {
                    "epoch": current_epoch,
                    "loss": avg_loss,
                    "accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "lr": current_lr,
                    "time": epoch_time,
                }
            )

            # ===== SAVE BEST MODEL =====
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self._save_checkpoint(
                    current_epoch,
                    {
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "train_loss": avg_loss,
                        "best_accuracy": self.best_accuracy,
                    },
                )
                if self.verbose:
                    print(f"   ✅ New best model saved! (Acc: {self.best_accuracy:.4f})")

            # ===== SAVE LAST MODEL =====
            self._save_checkpoint(
                current_epoch,
                {
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "train_loss": avg_loss,
                    "best_accuracy": self.best_accuracy,
                },
                suffix="last",
            )

            # ===== SAVE CHECKPOINT EVERY 5 EPOCHS =====
            if current_epoch % 5 == 0:
                self._save_checkpoint(
                    current_epoch,
                    {
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "train_loss": avg_loss,
                        "best_accuracy": self.best_accuracy,
                    },
                    suffix="checkpoint",
                )

            # ===== PRINT RESULTS =====
            if self.verbose:
                print(
                    f"📊 Epoch {current_epoch:3d}/{total_epochs} | "
                    f"Train Loss: {avg_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.2f}s"
                )

            # ===== EARLY STOPPING =====
            if len(self.train_history) > 15:
                last_10 = [h["val_accuracy"] for h in self.train_history[-10:]]
                if len(last_10) >= 15 and all(v <= last_10[0] for v in last_10[1:]):
                    if self.verbose:
                        print(
                            f"⚠️ Early stopping triggered (no improvement for 15 epochs)"
                        )
                    break

        return self.best_accuracy

    def _save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], suffix: str = "best"
    ):
        """Save checkpoint with best_accuracy included."""
        if suffix == "best":
            path = os.path.join(RESNET18_CHECKPOINTS_DIR, "resnet18_best.pth")
        elif suffix == "checkpoint":
            path = os.path.join(RESNET18_CHECKPOINTS_DIR, "resnet18_checkpoint.pth")
        else:
            path = os.path.join(RESNET18_CHECKPOINTS_DIR, "resnet18_last.pth")

        # ✅ استخراج best_accuracy از metrics
        best_acc = metrics.get("best_accuracy", self.best_accuracy)

        save_model_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            filename=os.path.basename(path),
            scheduler=self.scheduler,
            extra_info={
                "best_accuracy": best_acc,
                "history_length": len(self.train_history),
                "num_classes": self.num_classes,
                "zero_init_residual": self.zero_init_residual,
                "use_batch_norm": self.use_batch_norm,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "is_trained": self.is_trained,
                "feature_dim": self._feature_dim,
                "train_history": self.train_history,
                "class_names": self.class_names,
                "version": "4.0.0",
                "phase_lrs": self.phase_lrs,
            },
            save_dir=os.path.dirname(path),
        )

    def _update_learning_rate(self, phase: int) -> None:
        """Update learning rate for the current phase."""
        lr = get_phase_lr(phase)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.phase_lrs[phase] = lr

        if self.verbose:
            print(f"   ✅ Updated LR to: {lr:.6f} for Phase {phase+1}")

    def _get_current_lr(self) -> float:
        """Get current LR from scheduler."""
        return self.optimizer.param_groups[0]["lr"]

    def _apply_stage_lr(self, phase: int) -> None:
        """Apply stage-specific LR adjustment to scheduler base."""
        # ✅ Get phase-specific LR
        phase_lr = get_phase_lr(phase)

        # ✅ If scheduler supports base_lr adjustment, update it
        if hasattr(self.scheduler, "base_lr"):
            self.scheduler.base_lr = phase_lr
            # Also update warmup_start_lr
            if hasattr(self.scheduler, "warmup_start_lr"):
                self.scheduler.warmup_start_lr = phase_lr / 100
            print(f"   ✅ Scheduler base_lr updated to: {phase_lr:.2e}")

        # ✅ For other schedulers, just set LR directly
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = phase_lr

    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_best: bool = True,
        save_path: str = None,
        early_stopping_patience: int = 15,
        use_multi_stage: bool = True,
    ) -> Dict[str, Any]:
        """Train with multi-stage strategy (4 phases)."""
        if self.model is None:
            self.build_model()

        # Store val_loader for validation
        self.val_loader = val_loader

        if save_path is None:
            save_path = os.path.join(RESNET18_CHECKPOINTS_DIR, "resnet18_best.pth")

        if self.verbose:
            print(f"\n🚀 Starting Multi-Stage Training for {epochs} epochs...")
            print("=" * 80)
            print("📊 Stage Plan:")
            print(f"   Stage 1: FC layer only - LR: {get_phase_lr(0):.2e}")
            print(f"   Stage 2: Layer4 + FC - LR: {get_phase_lr(1):.2e}")
            print(f"   Stage 3: Layer3+4 + FC - LR: {get_phase_lr(2):.2e}")
            print(f"   Stage 4: All layers - LR: {get_phase_lr(3):.2e}")
            print()
            print(f"📊 Scheduler: {self.scheduler_type}")
            print(f"   - Warmup: {self.warmup_epochs} epochs")
            print(f"   - Total: {epochs} epochs")
            print(f"   - Min LR: {self.min_lr:.2e}")
            print("=" * 80)

        self.train_history = []
        self.best_accuracy = 0.0
        self.is_trained = False

        # ==========================================
        # Stage 1: Train only FC layer (5 epochs) - LR: 1e-3
        # ==========================================
        if use_multi_stage and self.verbose:
            print("\n📌 Stage 1: Training FC layer only (5 epochs)")
            print("-" * 40)

        if use_multi_stage:
            self.freeze_backbone()
            self._apply_stage_lr(0)  # ✅ LR: 1e-3
            stage1_epochs = min(8, epochs // 4)
            if stage1_epochs > 0:
                self.train_epoch(train_loader, stage1_epochs, epochs, desc="Stage 1")
        else:
            stage1_epochs = 0

        # ==========================================
        # Stage 2: Train FC + Layer4 (10 epochs) - LR: 1e-4
        # ==========================================
        if use_multi_stage and self.verbose:
            print("\n📌 Stage 2: Training Layer4 + FC (10 epochs)")
            print("-" * 40)

        if use_multi_stage:
            self.unfreeze_layer4()
            self._apply_stage_lr(1)  # ✅ LR: 1e-4
            stage2_epochs = min(15, epochs // 3)
            if stage2_epochs > 0:
                self.train_epoch(train_loader, stage2_epochs, epochs, desc="Stage 2")
        else:
            stage2_epochs = 0

        # ==========================================
        # Stage 3: Train FC + Layer3 + Layer4 (10 epochs) - LR: 5e-5
        # ==========================================
        if use_multi_stage and self.verbose:
            print("\n📌 Stage 3: Training Layer3 + Layer4 + FC (10 epochs)")
            print("-" * 40)

        if use_multi_stage:
            self.unfreeze_layer3_4()
            self._apply_stage_lr(2)  # ✅ LR: 5e-5
            stage3_epochs = min(15, epochs // 3)
            if stage3_epochs > 0:
                self.train_epoch(train_loader, stage3_epochs, epochs, desc="Stage 3")
        else:
            stage3_epochs = 0

        # ==========================================
        # Stage 4: Full Fine-Tuning (remaining epochs) - LR: 1e-5
        # ==========================================
        if use_multi_stage and self.verbose:
            print("\n📌 Stage 4: Full Fine-Tuning (remaining epochs)")
            print("-" * 40)

        if use_multi_stage:
            self.unfreeze_all()
            self._apply_stage_lr(3)  # ✅ LR: 1e-5
            stage4_epochs = epochs - stage1_epochs - stage2_epochs - stage3_epochs
            if stage4_epochs > 0:
                self.train_epoch(train_loader, stage4_epochs, epochs, desc="Stage 4")
        else:
            stage4_epochs = epochs
            if stage4_epochs > 0:
                self.train_epoch(train_loader, stage4_epochs, epochs, desc="Training")

        self.is_trained = True

        if self.verbose:
            print("=" * 80)
            print(f"✅ Training Complete!")
            print(f"   Best Validation Accuracy: {self.best_accuracy:.4f}")

        if save_best:
            self.load_checkpoint(save_path)

        return {
            "history": self.train_history,
            "best_accuracy": self.best_accuracy,
            "final_accuracy": (
                self.train_history[-1]["accuracy"] if self.train_history else 0
            ),
            "final_val_accuracy": (
                self.train_history[-1]["val_accuracy"] if self.train_history else 0
            ),
        }

    # ================================================================
    # Freezing Methods (4 Stages)
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
            print(
                f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)"
            )

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
            print(
                f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)"
            )

    def unfreeze_layer3_4(self) -> None:
        """Unfreeze layer3, layer4 and FC layer."""
        if self.model is None:
            return

        for name, param in self.model.named_parameters():
            if (
                name.startswith("layer3")
                or name.startswith("layer4")
                or name.startswith("fc")
            ):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        if self.verbose:
            print(f"✅ Layer3, Layer4 and FC layer unfrozen.")
            print(
                f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)"
            )

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
            print(
                f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)"
            )

    # ================================================================
    # Model Persistence
    # ================================================================

    def save_model(self, path: str = None) -> str:
        """Save the complete model."""
        if self.model is None:
            print("⚠️ No model to save. Build model first.")
            return ""

        if path is None:
            path = os.path.join(RESNET18_CHECKPOINTS_DIR, "resnet18_final.pth")

        return save_model_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=len(self.train_history),
            metrics={
                "best_accuracy": self.best_accuracy,
                "final_accuracy": (
                    self.train_history[-1]["accuracy"] if self.train_history else 0
                ),
                "final_val_accuracy": (
                    self.train_history[-1]["val_accuracy"] if self.train_history else 0
                ),
            },
            filename=os.path.basename(path),
            scheduler=self.scheduler,
            extra_info={
                "best_accuracy": self.best_accuracy,
                "num_classes": self.num_classes,
                "zero_init_residual": self.zero_init_residual,
                "use_batch_norm": self.use_batch_norm,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "is_trained": self.is_trained,
                "feature_dim": self._feature_dim,
                "train_history": self.train_history,
                "class_names": self.class_names,
                "version": "4.0.0",
                "phase_lrs": self.phase_lrs,
            },
            save_dir=os.path.dirname(path),
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint and restore best_accuracy."""
        if not os.path.exists(path):
            print(f"❌ Model file not found: {path}")
            return

        if self.model is None:
            self.build_model()

        # ==========================================
        # ✅ FIX: Force CUDA state check
        # ==========================================
        print(f"\n🔍 Loading checkpoint...")
        print(f"   self.device: {self.device}")
        print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")

        # ✅ If self.device is cuda but CUDA isn't available
        if "cuda" in str(self.device) and not torch.cuda.is_available():
            print(f"   ⚠️ CUDA not available but self.device=cuda")
            print(f"   → Keeping self.device as cuda (model will stay on GPU if it was there)")

        checkpoint = load_model_checkpoint(
            filepath=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        # ==========================================
        # ✅ CRITICAL: Ensure model is on self.device AFTER loading
        # ==========================================
        if "cuda" in str(self.device) and torch.cuda.is_available():
            self.model = self.model.to(self.device)
            print(f"✅ Model moved back to: {self.device}")
        else:
            # If CUDA is not available, update self.device to match reality
            actual_device = next(self.model.parameters()).device
            self.device = str(actual_device)
            print(f"⚠️ Updated self.device to: {self.device}")

        # Restore best_accuracy
        if "extra_info" in checkpoint and "best_accuracy" in checkpoint["extra_info"]:
            self.best_accuracy = checkpoint["extra_info"]["best_accuracy"]
        elif "best_accuracy" in checkpoint:
            self.best_accuracy = checkpoint["best_accuracy"]
        elif "metrics" in checkpoint and "best_accuracy" in checkpoint["metrics"]:
            self.best_accuracy = checkpoint["metrics"]["best_accuracy"]
        else:
            self.best_accuracy = 0.0

        self.is_trained = checkpoint.get("is_trained", False)

        if "extra_info" in checkpoint:
            self.train_history = checkpoint["extra_info"].get("train_history", [])
            self.phase_lrs = checkpoint["extra_info"].get("phase_lrs", {})

        if self.verbose:
            print(f"✅ Model loaded from: {path}")
            print(f"   - Best Accuracy: {self.best_accuracy:.4f}")
            print(f"   - Is Trained: {self.is_trained}")
            print(f"   - Model device: {next(self.model.parameters()).device}")

    # ================================================================
    # Comprehensive Evaluation
    # ================================================================

    def evaluate_comprehensive(
        self,
        test_loader: DataLoader,
        generate_report: bool = True,
        cv_folds: int = 0,
        report_name: str = "ResNet18_Report.pdf",
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation with full metrics and report.

        Args:
            test_loader: DataLoader for test data
            generate_report: Whether to generate PDF report
            cv_folds: Number of CV folds (0 = skip CV)
            report_name: Name of the report file

        Returns:
            Dict with metrics, cv_scores, model_info, etc.
        """
        if self.model is None:
            self.build_model()

        # ==========================================
        # ✅ CRITICAL FIX: Determine ACTUAL device from model
        # ==========================================
        actual_device = next(self.model.parameters()).device
        print(f"\n🔍 Evaluation starting...")
        print(f"   Model is on: {actual_device}")
        print(f"   self.device: {self.device}")
        print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")

        # ✅ Use ACTUAL model device, not self.device
        # This prevents device mismatch
        if actual_device.type == "cuda":
            print(f"✅ Using CUDA for evaluation")
        else:
            print(f"⚠️ Model is on CPU - evaluation will be slower")

        if self.verbose:
            print("\n📊 Comprehensive Evaluation Started...")
            print("=" * 80)

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, target in test_loader:
                # ✅ Use actual_device (not self.device)
                data = data.to(actual_device)
                target = target.to(actual_device)

                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                _, preds = output.max(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())


        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        recall = recall_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        per_class = {}
        for i, class_name in enumerate(self.class_names):
            class_precision = precision_score(
                all_labels, all_preds, labels=[i], average="micro", zero_division=0
            )
            class_recall = recall_score(
                all_labels, all_preds, labels=[i], average="micro", zero_division=0
            )
            class_f1 = f1_score(
                all_labels, all_preds, labels=[i], average="micro", zero_division=0
            )
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

        # ✅ CV فقط در صورت درخواست انجام میشه
        cv_scores = None
        if cv_folds > 1 and len(all_labels) > cv_folds:
            if self.verbose:
                print(f"\n🔬 Performing {cv_folds}-Fold Cross-Validation...")
            try:
                cv_scores = self.perform_cross_validation(
                    test_loader.dataset, n_folds=cv_folds
                )
                if cv_scores and "accuracy" in cv_scores and cv_scores["accuracy"]:
                    if self.verbose:
                        print(
                            f"   CV Accuracy: {np.mean(cv_scores['accuracy']):.4f} ± {np.std(cv_scores['accuracy']):.4f}"
                        )
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ CV failed: {e}")
                cv_scores = None

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        model_info = {
            "architecture": "ResNet18",
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
            "stages": 4,
        }

        if generate_report:
            if self.verbose:
                print("\n📄 Generating PDF Report...")

            report_path = os.path.join(RESNET18_REPORTS_DIR, report_name)

            generate_pdf_report(
                metrics=metrics,
                cm=cm,
                y_true=all_labels,
                y_prob=all_probs,
                per_class=per_class,
                cv_scores=cv_scores,
                model_info=model_info,
                training_history=self.train_history,
                output_path=report_path,
            )

            metrics_path = os.path.join(RESNET18_REPORTS_DIR, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metrics": metrics,
                        "model_info": model_info,
                        "cv_scores": cv_scores,
                        "best_accuracy": self.best_accuracy,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            if self.verbose:
                print(f"✅ Metrics saved to: {metrics_path}")

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

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(labels)), labels)
        ):
            if self.verbose:
                print(f"\n   CV Fold {fold+1}/{n_folds}...")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

            fold_model = ResNet18WithoutFineTuning(
                num_classes=self.num_classes,
                zero_init_residual=self.zero_init_residual,
                use_batch_norm=self.use_batch_norm,
                activation=self.activation,
                dropout_rate=0.1,
                device=self.device,
                verbose=False,
            )
            fold_model.build_model()
            fold_model.train_full(
                train_loader,
                val_loader,
                epochs=50,
                save_best=True,
                use_multi_stage=True,
            )

            val_metrics = fold_model.validate(val_loader, return_loss=True)

            # ✅ چون val_metrics یک tuple هستش (loss, accuracy)
            if isinstance(val_metrics, tuple):
                cv_accuracy.append(val_metrics[1])  # index 1 = accuracy
            else:
                cv_accuracy.append(val_metrics.get("val_accuracy", 0))

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

            cv_precision.append(
                precision_score(
                    all_labels_val, all_preds, average="weighted", zero_division=0
                )
            )
            cv_recall.append(
                recall_score(
                    all_labels_val, all_preds, average="weighted", zero_division=0
                )
            )
            cv_f1.append(
                f1_score(all_labels_val, all_preds, average="weighted", zero_division=0)
            )

            # ✅ آزادسازی حافظه
            del fold_model
            gc.collect()

        return {
            "accuracy": cv_accuracy,
            "precision": cv_precision,
            "recall": cv_recall,
            "f1_score": cv_f1,
        }

    # ================================================================
    # Advanced Profiling
    # ================================================================

    def profile(
        self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ) -> Dict[str, Any]:
        """Profile model performance."""
        if self.model is None:
            self.build_model()

        return profile_model(self.model, input_shape, self.device)

    def calculate_flops(
        self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ) -> Dict[str, Any]:
        """Calculate model FLOPs."""
        if self.model is None:
            self.build_model()

        return calculate_flops(self.model, input_shape)

    def analyze_memory(
        self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ) -> Dict[str, Any]:
        """Analyze memory usage."""
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
        return (
            f"ResNet18WithoutFineTuning("
            f"num_classes={self.num_classes}, "
            f"device={self.device}, "
            f"use_class_weights={self.use_class_weights}, "
            f"use_focal_loss={self.use_focal_loss}, "
            f"label_smoothing={self.label_smoothing}, "
            f"scheduler_type='{self.scheduler_type}', "
            f"warmup_epochs={self.warmup_epochs}"
            f")"
        )

    def perform_cross_validation_from_saved(
        self, dataset: torch.utils.data.Dataset, n_folds: int = 5, batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation using the SAVED model (NO RETRAINING!)

        Args:
            dataset: Dataset to perform CV on
            n_folds: Number of folds
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with CV results
        """
        if self.model is None:
            raise ValueError("Model not loaded! Please load a saved model first.")

        self.logger.info(
            f"🔬 Running {n_folds}-fold CV using SAVED model (no retraining)..."
        )

        # Get labels for stratification
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

            # Create validation subset
            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # ✅ ONLY EVALUATE - NO TRAINING!
            self.model.eval()

            all_preds = []
            all_labels_val = []
            all_probs = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    probs = torch.softmax(output, dim=1)
                    _, preds = output.max(1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels_val.extend(target.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            # Calculate metrics for this fold
            fold_accuracy = accuracy_score(all_labels_val, all_preds)
            fold_precision = precision_score(
                all_labels_val, all_preds, average="weighted", zero_division=0
            )
            fold_recall = recall_score(
                all_labels_val, all_preds, average="weighted", zero_division=0
            )
            fold_f1 = f1_score(
                all_labels_val, all_preds, average="weighted", zero_division=0
            )

            cv_accuracy.append(fold_accuracy)
            cv_precision.append(fold_precision)
            cv_recall.append(fold_recall)
            cv_f1.append(fold_f1)

            if self.verbose:
                print(
                    f"      Accuracy: {fold_accuracy:.4f}, Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}, F1: {fold_f1:.4f}"
                )

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
            print(
                f"   Accuracy:  {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}"
            )
            print(
                f"   Precision: {results['mean_precision']:.4f} ± {results['std_precision']:.4f}"
            )
            print(
                f"   Recall:    {results['mean_recall']:.4f} ± {results['std_recall']:.4f}"
            )
            print(f"   F1-Score:  {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
            print("=" * 70)

        return results

    # ================================================================
    # ✅ ADVANCED CROSS-VALIDATION METHODS
    # ================================================================

    def run_advanced_cv(
            self,
            train_loader: DataLoader,
            n_folds: int = 5,
            n_repeats: int = 1,
            strategy: str = "stratified",
            retrain_each_fold: bool = False,
            output_dir: str = None,
            prefix: str = "cv_results",
            save_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Run advanced cross-validation.

        Two modes:
            1. **Evaluate-only** (retrain_each_fold=False):
               Uses the current trained model and evaluates on each fold's validation set.
               Fast, but not a true CV.

            2. **Full CV** (retrain_each_fold=True):
               Trains a new model for each fold.
               Slow, but provides true CV estimate.

        Args:
            train_loader: DataLoader with training data
            n_folds: Number of folds
            n_repeats: Repeats (for 'repeated' strategy)
            strategy: "stratified", "repeated", "group"
            retrain_each_fold: If True, retrain model per fold (full CV)
            output_dir: Directory to save results
            prefix: Filename prefix
            save_plots: Save plots

        Returns:
            CV results
        """
        if self.model is None:
            raise ValueError("Model not built! Call build_model() first.")

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"🔬 ADVANCED CROSS-VALIDATION")
            print("=" * 80)
            print(f"   Mode: {'Full CV (retrain)' if retrain_each_fold else 'Evaluate-only'}")
            print(f"   Strategy: {strategy}")
            print(f"   Folds: {n_folds}")
            print(f"   Repeats: {n_repeats if strategy == 'repeated' else 1}")
            print("=" * 80)

        # ✅ Get dataset and labels
        dataset = train_loader.dataset

        # ✅ Extract labels from dataset
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(int(label) if isinstance(label, torch.Tensor) else label)
        labels = np.array(labels)

        if self.verbose:
            print(f"\n   Dataset size: {len(dataset)}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Unique labels: {np.unique(labels)}")
            print(f"   Class distribution: {np.bincount(labels, minlength=self.num_classes)}")

        # ==========================================
        # Mode 1: Evaluate-only (fast)
        # ==========================================
        if not retrain_each_fold:
            if self.verbose:
                print("\n   📌 Mode: Evaluate-only (using current trained model)")

            cv = AdvancedCrossValidator(
                n_folds=n_folds,
                n_repeats=n_repeats,
                strategy=strategy,
                verbose=self.verbose,
            )

            results = cv.evaluate_model(
                model=self.model,
                dataset=dataset,
                labels=labels,
                device=self.device,
                batch_size=32,
            )

        # ==========================================
        # Mode 2: Full CV (retrain per fold)
        # ==========================================
        else:
            if self.verbose:
                print("\n   📌 Mode: Full CV (retraining per fold)")
                print("   ⚠️ This will take a LONG time!")

            results = self._run_full_cv(
                dataset=dataset,
                labels=labels,
                n_folds=n_folds,
                strategy=strategy,
            )

            cv = None  # We'll handle results differently

        # ==========================================
        # Save results
        # ==========================================
        if output_dir and not retrain_each_fold:
            os.makedirs(output_dir, exist_ok=True)
            save_result = cv.save_results(output_dir, prefix)
            results["saved_files"] = save_result

            if save_plots:
                plot_paths = cv.plot_results(output_dir, prefix)
                results["saved_plots"] = plot_paths

        return results

    def _run_full_cv(
            self,
            dataset: torch.utils.data.Dataset,
            labels: np.ndarray,
            n_folds: int = 5,
            strategy: str = "stratified",
    ) -> Dict[str, Any]:
        """
        Run full cross-validation (retrain per fold).

        ⚠️ WARNING: This is VERY slow!
        """
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=42,
        )

        fold_results = []
        X = np.zeros(len(dataset))

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, labels), 1):
            if self.verbose:
                print(f"\n📌 FOLD {fold_num}/{n_folds}")
                print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}")

            # ✅ Create subsets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=32,
                shuffle=True,
                num_workers=0,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=32,
                shuffle=False,
                num_workers=0,
            )

            # ✅ Create fresh model
            fold_model = ResNet18WithoutFineTuning(
                num_classes=self.num_classes,
                zero_init_residual=self.zero_init_residual,
                use_batch_norm=self.use_batch_norm,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                device=self.device,
                verbose=False,
                # Same regularization
                use_class_weights=self.use_class_weights,
                class_weight_method=self.class_weight_method,
                labels_csv_path=self.labels_csv_path,
                label_smoothing=self.label_smoothing,
                use_focal_loss=self.use_focal_loss,
                focal_gamma=self.focal_gamma,
                scheduler_type=self.scheduler_type,
                warmup_epochs=self.warmup_epochs,
                min_lr=self.min_lr,
                total_epochs=self.total_epochs,
            )

            # ✅ Build and train
            fold_model.build_model()
            fold_model.train_full(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.total_epochs,
                save_best=True,
                use_multi_stage=False,  # Simple training for CV
            )

            # ✅ Evaluate
            fold_model.model.eval()

            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = fold_model.model(data)
                    probs = torch.softmax(output, dim=1)
                    _, preds = output.max(1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(target.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)

            # ✅ Compute metrics
            metrics = {
                "fold": fold_num,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "accuracy": float(accuracy_score(all_labels, all_preds)),
                "precision": float(precision_score(all_labels, all_preds, average="weighted", zero_division=0)),
                "recall": float(recall_score(all_labels, all_preds, average="weighted", zero_division=0)),
                "f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
            }

            # ✅ Per-class
            for class_idx, class_name in enumerate(CLASS_NAMES):
                metrics[f"f1_{class_name}"] = float(f1_score(
                    all_labels, all_preds, labels=[class_idx], average="micro", zero_division=0
                ))

            # ✅ AUC
            try:
                metrics["auc_macro"] = float(roc_auc_score(
                    all_labels, all_probs, multi_class="ovr", average="macro"
                ))
            except:
                metrics["auc_macro"] = 0.0

            # ✅ Confusion matrix
            metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()

            fold_results.append(metrics)

            if self.verbose:
                print(f"   ✅ Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

            # ✅ Clean up
            del fold_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ✅ Compute summary
        summary = {}
        for metric in ["accuracy", "precision", "recall", "f1", "auc_macro"]:
            values = [r[metric] for r in fold_results]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        for class_name in CLASS_NAMES:
            metric = f"f1_{class_name}"
            values = [r[metric] for r in fold_results]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        # ✅ Print summary
        if self.verbose:
            print("\n" + "=" * 80)
            print("📊 FULL CROSS-VALIDATION SUMMARY")
            print("=" * 80)
            print(f"\n   {'Metric':<15} {'Mean':<12} {'Std':<12}")
            print(f"   {'-' * 15} {'-' * 12} {'-' * 12}")
            for metric in ["accuracy", "precision", "recall", "f1", "auc_macro"]:
                print(f"   {metric:<15} {summary[metric]['mean']:<12.4f} {summary[metric]['std']:<12.4f}")

            print(f"\n   Per-Class F1:")
            for class_name in CLASS_NAMES:
                m = summary[f"f1_{class_name}"]
                print(f"      {class_name:<6} {m['mean']:.4f} ± {m['std']:.4f}")
            print("=" * 80)

        return {
            "fold_results": fold_results,
            "summary": summary,
            "strategy": strategy,
            "n_folds": n_folds,
            "mode": "full_cv",
        }



# ================================================================
# Factory Function
# ================================================================


def create_resnet18_wrapper(
    num_classes: int = 4,
    checkpoints_dir: str = None,
    configs_dir: str = None,
    plots_dir: str = None,
    use_class_weights: bool = True,
    class_weight_method: str = "sqrt",
    label_smoothing: float = 0.1,
    use_adaptive_smoothing: bool = False,  # ← ✅ NEW
    labels_csv_path: str = None,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
    scheduler_type: str = "warmup_cosine",
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    total_epochs: int = 100,
    **kwargs
) -> ResNet18WithoutFineTuning:
    """
    Factory function to create a ResNet18 wrapper.

    Args:
        num_classes: Number of output classes
        checkpoints_dir: Directory for checkpoints
        configs_dir: Directory for configs
        plots_dir: Directory for plots
        use_class_weights: Enable class weighting for imbalanced datasets
        class_weight_method: Method for computing weights
            ("balanced", "inverse", "sqrt", "effective")
        label_smoothing: Label smoothing factor (0.0 to disable)
        labels_csv_path: Path to labels.csv for computing class weights
        use_focal_loss: Enable Focal Loss (recommended for imbalanced data)
        focal_gamma: Focus parameter (default: 2.0)
            - 0.0: same as CrossEntropyLoss
            - 1.0: moderate focus
            - 2.0: strong focus (recommended)
            - 3.0+: very strong focus
        focal_alpha: Optional scalar alpha (if None, uses class weights)
        **kwargs: Additional arguments for ResNet18WithoutFineTuning

    Returns:
        ResNet18WithoutFineTuning instance

     Factory function to create a ResNet18 wrapper.

    Args:
        # ... (بقیه پارامترها مثل قبل)

        scheduler_type: Type of LR scheduler
            - "cosine": Standard Cosine Annealing
            - "warmup_cosine": Linear Warmup + Cosine ⭐ (recommended)
            - "warmup_restart": Linear Warmup + Cosine Warm Restarts
            - "onecycle": OneCycleLR
            - "warmup_onecycle": Warmup + OneCycle
            - "plateau": ReduceLROnPlateau
            - "step": StepLR

        warmup_epochs: Number of warmup epochs (default: 5)
        min_lr: Minimum learning rate (default: 1e-6)
        total_epochs: Total training epochs

    Returns:
        ResNet18WithoutFineTuning instance

    """
    return ResNet18WithoutFineTuning(
        num_classes=num_classes,
        checkpoints_dir=checkpoints_dir,
        configs_dir=configs_dir,
        plots_dir=plots_dir,
        # Class weighting
        use_class_weights=use_class_weights,
        class_weight_method=class_weight_method,
        label_smoothing=label_smoothing,
        labels_csv_path=labels_csv_path,
        use_adaptive_smoothing=use_adaptive_smoothing,
        # Focal loss
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        scheduler_type=scheduler_type,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        total_epochs=total_epochs,
        **kwargs
    )


# ================================================================
# Testing
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 TESTING RESNET18 WRAPPER (ULTIMATE EDITION)")
    print("=" * 80)

    model = create_resnet18_wrapper(num_classes=4, device="cuda", verbose=True)

    model.build_model()
    model.print_summary()

    dummy = torch.randn(4, 3, 224, 224)
    output = model.forward(dummy)
    print(f"\n✅ Output shape: {output.shape}")

    embeddings = model.get_embeddings(dummy)
    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"✅ Feature dimension: {model.get_feature_dim()}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
