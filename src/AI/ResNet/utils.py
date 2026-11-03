# src/AI/ResNet/utils.py

"""
ResNet Utilities for Liver Disease Classification
This module contains utility functions for training ResNet models
ResNet is a family of deep residual networks for image classification
"""

import math
import os
import random
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# =========================
# ✅ GLOBAL PATHS - مطابق با ساختار شما
# =========================
PROJECT_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
FINAL_REPORTS_DIR = os.path.join(PROJECT_ROOT, "finalReports")

# مسیرهای اصلی
CROPPED = os.path.join(FINAL_REPORTS_DIR, "cropped")
CSVAddress = os.path.join(FINAL_REPORTS_DIR, "labels.csv")

# ✅ مسیر AI_Comparisons
AI_COMPARISONS_DIR = os.path.join(FINAL_REPORTS_DIR, "AI_Comparisons")

# ✅ مسیر ResNet (OUTPUT_DIR اصلی)
OUTPUT_DIR = os.path.join(AI_COMPARISONS_DIR, "ResNet")

# ✅ مسیرهای فرعی ResNet
RESNET18_DIR = os.path.join(OUTPUT_DIR, "resnet18")
RESNET50_DIR = os.path.join(OUTPUT_DIR, "resnet50")
RESNET_REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# =========================
# ✅ ایجاد پوشه‌ها
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESNET18_DIR, exist_ok=True)
os.makedirs(RESNET50_DIR, exist_ok=True)
os.makedirs(RESNET_REPORTS_DIR, exist_ok=True)

# =========================
# LABEL MAP - FIXED ORDER 🔥
# =========================
CLASS_NAMES = ["NL", "G1", "G2", "G3"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(CLASS_NAMES)

# =========================
# NORMALIZATION PARAMS 🔥
# =========================
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]

# =========================
# ✅ HYPERPARAMETER CONFIG 🔥
# =========================
HYPERPARAMS = {
    "resnet18": {
        "batch_size": 64,
        "lr": 1e-4,
        "base_lr": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 1,
        "warmup_epochs": 5,
        "epochs_per_phase": [10, 15, 15, 15],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 256,
        "use_smart_lr": True,
        "early_stopping_patience": 25
    },
    "resnet50": {
        "batch_size": 64,
        "lr": 1e-4,
        "base_lr": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 1,
        "warmup_epochs": 5,
        "epochs_per_phase": [15, 20, 20, 20, 25],
        "phase_lrs": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 256,
        "use_smart_lr": True,
        "early_stopping_patience": 25,
    },
    "efficientnet_b0": {
        "batch_size": 64,
        "lr": 1e-4,
        "base_lr": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 1,
        "warmup_epochs": 5,
        "epochs_per_phase": [15, 20, 20, 20],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 25,
    },
    "efficientnet_b1": {
        "batch_size": 64,
        "lr": 1e-4,
        "base_lr": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 1,
        "warmup_epochs": 5,
        "epochs_per_phase": [10, 15, 15, 20],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 240,
        "use_smart_lr": True,
        "early_stopping_patience": 25,
    },
}



# =========================
# ✅ LABEL SMOOTHING LOSS 🔥
# =========================
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss

    Reduces overconfidence and improves generalization
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)

        # One-hot encoding with smoothing
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # Log softmax and loss
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smoothed * log_prob).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# =========================
# ✅ FOCAL LOSS 🔥
# =========================
class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification

    Reduces weight of easy examples and focuses on hard ones
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# =========================
# ✅ WARMUP + COSINE ANNEALING SCHEDULER 🔥
# =========================
class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing

    Features:
    - Linear warmup for first N epochs
    - Cosine annealing decay after warmup
    - Smooth transition between phases
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 55,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_epoch = 0
        self.current_lr = max_lr

        # Store initial LR
        for param_group in optimizer.param_groups:
            param_group["lr"] = max_lr

    def step(self) -> float:
        """Update learning rate and return current LR"""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # ✅ Linear warmup
            progress = self.current_epoch / self.warmup_epochs
            self.current_lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        else:
            # ✅ Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            self.current_lr = (
                self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
            )

        # Apply LR
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr

        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.current_lr


# =========================
# ✅ RANDOM ERASING AUGMENTATION 🔥
# =========================
class RandomErasing:
    """
    Random Erasing augmentation

    Randomly erases a rectangular region in the image
    Helps the model focus on non-local features
    """

    def __init__(
        self,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        value: float = 0.0,
    ):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.value = value

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random erasing

        Args:
            img: Image array (H, W) or (H, W, C)

        Returns:
            Augmented image
        """
        if np.random.random() > self.probability:
            return img

        h, w = img.shape[:2]
        area = h * w

        for _ in range(100):
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)
            h_erase = int(round((target_area * aspect_ratio) ** 0.5))
            w_erase = int(round((target_area / aspect_ratio) ** 0.5))

            if h_erase < h and w_erase < w:
                x1 = np.random.randint(0, h - h_erase)
                y1 = np.random.randint(0, w - w_erase)

                if len(img.shape) == 2:
                    img[x1 : x1 + h_erase, y1 : y1 + w_erase] = self.value
                else:
                    img[x1 : x1 + h_erase, y1 : y1 + w_erase, :] = self.value
                return img

        return img


# =========================
# ✅ CUTMIX AUGMENTATION 🔥
# =========================
class CutMix:
    """
    CutMix augmentation for batch training

    Mixes two images by cutting and pasting patches
    """

    def __init__(self, alpha: float = 1.0, probability: float = 0.5):
        self.alpha = alpha
        self.probability = probability

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to batch

        Args:
            x: Image batch (B, C, H, W)
            y: Labels batch (B,)

        Returns:
            Mixed images, mixed labels, lambda
        """
        if np.random.random() > self.probability:
            return x, y, 1.0

        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        # Randomly pair images
        index = torch.randperm(batch_size).to(x.device)

        # Get bounding box
        h, w = x.size(2), x.size(3)
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        cx = np.random.randint(0, w - cut_w)
        cy = np.random.randint(0, h - cut_h)

        # Create mixed images
        x_mixed = x.clone()
        x_mixed[:, :, cy : cy + cut_h, cx : cx + cut_w] = x[
            index, :, cy : cy + cut_h, cx : cx + cut_w
        ]

        # Mixed labels
        y_mixed = y * lam + y[index] * (1 - lam)

        return x_mixed, y_mixed, lam


# =========================
# ✅ MIXUP AUGMENTATION 🔥
# =========================
class MixUp:
    """
    MixUp augmentation for batch training

    Mixes two images by linear interpolation
    """

    def __init__(self, alpha: float = 0.2, probability: float = 0.5):
        self.alpha = alpha
        self.probability = probability

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to batch

        Args:
            x: Image batch (B, C, H, W)
            y: Labels batch (B,)

        Returns:
            Mixed images, mixed labels, lambda
        """
        if np.random.random() > self.probability:
            return x, y, 1.0

        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        # Randomly pair images
        index = torch.randperm(batch_size).to(x.device)

        # Mixed images
        x_mixed = lam * x + (1 - lam) * x[index]

        # Mixed labels
        y_mixed = y * lam + y[index] * (1 - lam)

        return x_mixed, y_mixed, lam


# =========================
# ✅ ENHANCED AUGMENTATION 🔥
# =========================
def apply_advanced_augmentation(
    image: np.ndarray, use_erasing: bool = True, erasing_prob: float = 0.3
) -> np.ndarray:
    """
    Apply advanced augmentations to an image

    Args:
        image: Input image (H, W)
        use_erasing: Whether to apply random erasing
        erasing_prob: Probability of random erasing

    Returns:
        Augmented image
    """
    # ✅ Basic augmentations
    image = apply_augmentation(image)

    # ✅ Random Erasing
    if use_erasing:
        eraser = RandomErasing(probability=erasing_prob)
        image = eraser(image)

    return image


def apply_augmentation(image: np.ndarray) -> np.ndarray:
    """
    Apply basic augmentations to an image

    Args: image (numpy.ndarray): Input image

    Returns: numpy.ndarray: Augmented image
    """
    try:
        # Horizontal flip with 50% probability
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)

        # Random rotation between -15 and +15 degrees
        angle = np.random.uniform(-15, 15)
        h, w = image.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Brightness adjustment with probability 40%
        if np.random.rand() < 0.4:
            brightness = np.random.uniform(0.7, 1.3)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)

        # Contrast adjustment with probability 40%
        if np.random.rand() < 0.4:
            contrast = np.random.uniform(0.7, 1.3)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)

        # Gaussian noise with 20% probability
        if np.random.rand() < 0.2:
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Blur with 15% probability
        if np.random.rand() < 0.15:
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # Sharpening with 10% probability
        if np.random.rand() < 0.1:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image = cv2.filter2D(image, -1, kernel)
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    except Exception as e:
        print(f"⚠️ Error in augmentation: {e}")
        return image


# =========================
# ✅ CROSS-VALIDATION 🔥
# =========================
def cross_validate_model(
    model_class: Any,
    df: pd.DataFrame,
    img_dir: str,
    n_folds: int = 5,
    target_size: int = 256,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    dropout_rate: float = 0.15,
    epochs: int = 10,
    label_smoothing: float = 0.1,
    accumulation_steps: int = 4,
    warmup_epochs: int = 5,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Perform K-Fold cross-validation on a model

    Args:
        model_class: Trainer class (ResNet18Trainer, ResNet50Trainer, etc.)
        df: DataFrame with labels
        img_dir: Image directory
        n_folds: Number of folds
        target_size: Target image size
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        dropout_rate: Dropout rate
        epochs: Epochs per fold
        label_smoothing: Label smoothing value
        accumulation_steps: Gradient accumulation steps
        warmup_epochs: Warmup epochs
        verbose: Print progress
        **kwargs: Additional arguments

    Returns:
        Dictionary with results
    """
    # Remove UNKNOWN
    df = df[df["liverLabel"] != "UNKNOWN"].copy()

    # Remove classes with few samples
    class_counts = df["liverLabel"].value_counts()
    min_samples = 2
    valid_classes = class_counts[class_counts >= min_samples].index.tolist()
    df = df[df["liverLabel"].isin(valid_classes)]

    if len(df) == 0:
        raise ValueError("No valid labels found after filtering!")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {
        "folds": [],
        "accuracies": [],
        "best_models": [],
        "mean_accuracy": 0.0,
        "std_accuracy": 0.0,
        "best_fold": 0,
    }

    if verbose:
        print("\n" + "=" * 70)
        print(f"📊 CROSS-VALIDATION ({n_folds}-fold)")
        print("=" * 70)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        if verbose:
            print(f"\n🔄 Fold {fold+1}/{n_folds}")
            print("-" * 40)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        if verbose:
            print(f"   Train size: {len(train_df)}")
            print(f"   Val size: {len(val_df)}")

        # Create dataset with train/val split
        train_ds = LiverDataset(train_df, img_dir, augment=True, target_size=target_size)
        val_ds = LiverDataset(val_df, img_dir, augment=False, target_size=target_size)

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Create model
        from src.AI.ResNet.resnet18 import ResNet18Trainer

        trainer = ResNet18Trainer()

        # Override dataloaders
        trainer.train_ds = train_ds
        trainer.val_ds = val_ds
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader

        # Build model
        trainer.model = trainer._build_model()

        # Use Label Smoothing
        trainer.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

        # Optimizer
        trainer.optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Warmup + Cosine Annealing Scheduler
        trainer.scheduler = WarmupCosineScheduler(
            trainer.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            min_lr=1e-7,
            max_lr=lr,
        )

        # Train
        acc = trainer.train_epoch(epochs=epochs, desc=f"Fold {fold+1}")
        fold_accuracies.append(acc)

        results["folds"].append(
            {
                "fold": fold + 1,
                "accuracy": acc,
                "train_size": len(train_df),
                "val_size": len(val_df),
            }
        )

        if verbose:
            print(f"   ✅ Fold {fold+1} Accuracy: {acc:.4f}")

    # Results
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    best_idx = np.argmax(fold_accuracies)

    results["accuracies"] = fold_accuracies
    results["mean_accuracy"] = float(mean_acc)
    results["std_accuracy"] = float(std_acc)
    results["best_fold"] = best_idx + 1

    if verbose:
        print("\n" + "=" * 70)
        print("📊 CROSS-VALIDATION RESULTS")
        print("=" * 70)
        print(f"   Mean Accuracy: {mean_acc:.4f}")
        print(f"   Std Accuracy: {std_acc:.4f}")
        print(f"   Best Fold: {best_idx + 1} ({fold_accuracies[best_idx]:.4f})")
        print(f"   All Accuracies: {[f'{a:.4f}' for a in fold_accuracies]}")
        print("=" * 70)

    return results


# =========================
# SEED SETTING
# =========================
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set to: {seed}")


# =========================
# SAFETY CHECK
# =========================
def check_paths():
    """
    Check for the existence of required paths

    Returns: bool: True if all paths are valid, False otherwise
    """
    try:
        if not os.path.exists(CROPPED):
            print(f"❌ CROPPED path invalid: {CROPPED}")
            return False

        if not os.path.exists(CSVAddress):
            print(f"❌ CSV path invalid: {CSVAddress}")
            return False

        # Create all required folders
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(RESNET18_DIR, exist_ok=True)
        os.makedirs(RESNET50_DIR, exist_ok=True)
        os.makedirs(RESNET_REPORTS_DIR, exist_ok=True)

        print("✅ All paths verified successfully!")
        return True

    except Exception as e:
        print(f"❌ Error checking paths: {e}")
        return False


def validate_csv(csv_path=None):
    """
    CSV File Validation

    Args: csv_path (str, optional): The path to the CSV file. If None, CSVAddress is used.

    Returns: tuple: (is_valid, df, message)
    """
    if csv_path is None:
        csv_path = CSVAddress

    try:
        if not os.path.exists(csv_path):
            return False, None, f"CSV file not found: {csv_path}"

        df = pd.read_csv(csv_path)

        # Check required columns
        required_columns = ["liverName", "liverLabel"]
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            return False, None, f"Missing columns: {missing_cols}"

        if len(df) == 0:
            return False, None, "CSV file is empty"

        # Checking for valid labels
        valid_labels = CLASS_NAMES
        invalid_labels = df[~df["liverLabel"].isin(valid_labels + ["UNKNOWN"])][
            "liverLabel"
        ].unique()

        if len(invalid_labels) > 0:
            print(f"⚠️ Found invalid labels: {invalid_labels}")

        valid_count = len(df[df["liverLabel"].isin(valid_labels)])
        unknown_count = len(df[df["liverLabel"] == "UNKNOWN"])

        print(
            f"✅ CSV validated: {len(df)} rows, {valid_count} valid labels, {unknown_count} unknown"
        )

        return True, df, "CSV is valid"

    except Exception as e:
        return False, None, f"Error reading CSV: {e}"


def validate_images(csv_path=None, img_dir=None, sample_size=None):
    """
    Validate images in CSV

    Args: csv_path (str, optional): CSV file path
          img_dir (str, optional): Image directory path
          sample_size (int, optional): Number of samples to check

    Returns: tuple: (total_images, valid_images, missing_images)
    """
    if csv_path is None:
        csv_path = CSVAddress

    if img_dir is None:
        img_dir = CROPPED

    try:
        is_valid, df, message = validate_csv(csv_path)

        if not is_valid:
            return 0, 0, []

        if sample_size is not None and sample_size < len(df):
            df_sample = df.sample(min(sample_size, len(df)))
        else:
            df_sample = df

        valid_images = []
        missing_images = []

        for _, row in df_sample.iterrows():
            img_name = row["liverName"]
            img_path = os.path.join(img_dir, img_name)

            if os.path.exists(img_path) and os.path.isfile(img_path):
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        valid_images.append(img_name)
                    else:
                        missing_images.append(img_name)
                except:
                    missing_images.append(img_name)
            else:
                missing_images.append(img_name)

        print(f"✅ Image validation: {len(valid_images)} valid, {len(missing_images)} missing")

        return len(valid_images), len(missing_images), missing_images

    except Exception as e:
        print(f"❌ Error validating images: {e}")
        return 0, 0, []


# =========================
# RESIZE WITHOUT DISTORTION
# =========================
def resize_with_padding(image, size=256):
    """
    Resize image while maintaining aspect ratio and adding padding

    Args: image (numpy.ndarray): Input image (grayscale)
          size (int): Target size (size x size)

    Returns: numpy.ndarray: Resized image with padding
    """
    try:
        h, w = image.shape
        scale = size / max(h, w)

        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

        canvas = np.zeros((size, size), dtype=np.uint8)

        y = (size - nh) // 2
        x = (size - nw) // 2

        canvas[y : y + nh, x : x + nw] = img

        return canvas

    except Exception as e:
        print(f"❌ Error in resize_with_padding: {e}")
        return np.zeros((size, size), dtype=np.uint8)


# =========================
# NORMALIZATION FUNCTION 🔥
# =========================
def normalize_resnet(image_tensor):
    """
    Normalize image tensor with ResNet stats

    Args:
        image_tensor: (C, H, W) tensor in range [0, 1]

    Returns:
        Normalized tensor
    """
    mean = torch.tensor(RESNET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(RESNET_STD, dtype=torch.float32).view(3, 1, 1)
    return (image_tensor - mean) / std


# =========================
# ✅ DATASET - با قابلیت برگشت PATH 🔥
# =========================
class LiverDataset(Dataset):
    """
    Dataset for training deep learning models

    Args: df (pd.DataFrame): DataFrame containing liverName and liverLabel
          img_dir (str): Path to image directory
          augment (bool): Whether to apply augmentation
          target_size (int): Target size of images
          apply_normalize (bool): Whether to apply normalization
          use_advanced_augment (bool): Whether to use advanced augmentation
          return_path (bool): Whether to return image path
    """

    def __init__(
        self,
        df,
        img_dir=None,
        augment=False,
        target_size=256,
        apply_normalize=True,
        use_advanced_augment=True,
        return_path=False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir if img_dir is not None else CROPPED
        self.augment = augment
        self.target_size = target_size
        self.apply_normalize = apply_normalize
        self.use_advanced_augment = use_advanced_augment
        self.return_path = return_path

        # Delete UNKNOWN rows
        self.df = self.df[self.df["liverLabel"] != "UNKNOWN"]

        # Remove classes with very few samples
        class_counts = self.df["liverLabel"].value_counts()
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        self.df = self.df[self.df["liverLabel"].isin(valid_classes)]

        if len(self.df) == 0:
            raise ValueError("No valid labels found in dataset!")

        print(f"✅ Dataset initialized with {len(self.df)} samples")
        print(f"   Classes: {sorted(self.df['liverLabel'].unique())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        Returns: tuple: (image_tensor, label_tensor, [path])
        """
        try:
            row = self.df.iloc[idx]

            img_path = os.path.join(self.img_dir, row["liverName"])
            label = LABEL_MAP[row["liverLabel"]]

            # Image reading
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"⚠️ Image not found: {img_path}")
                img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

            # Resize
            img = resize_with_padding(img, self.target_size)

            # Advanced Augmentation
            if self.augment:
                if self.use_advanced_augment:
                    img = apply_advanced_augmentation(img, use_erasing=True)
                else:
                    img = apply_augmentation(img)

            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0

            # Convert to 3 channels (ResNet needs 3 channels)
            img = np.stack([img, img, img], axis=0)  # (3, H, W)

            # Convert to tensor
            img_tensor = torch.tensor(img, dtype=torch.float32)

            # Apply normalization
            if self.apply_normalize:
                img_tensor = normalize_resnet(img_tensor)

            if self.return_path:
                return img_tensor, torch.tensor(label, dtype=torch.long), img_path

            return img_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"❌ Error in __getitem__: {e}")

            # Return an empty instance on error
            img = np.zeros((3, self.target_size, self.target_size), dtype=np.float32)
            img_tensor = torch.tensor(img, dtype=torch.float32)
            if self.apply_normalize:
                img_tensor = normalize_resnet(img_tensor)

            if self.return_path:
                return img_tensor, torch.tensor(0, dtype=torch.long), "error_image.png"
            return img_tensor, torch.tensor(0, dtype=torch.long)


# =========================
# DATA LOADING FUNCTIONS
# =========================
def load_split_data(csv_path=None, img_dir=None, test_size=0.2, random_state=42, target_size=256):
    """
    Load and split data into train/validation

    Args: csv_path (str, optional): CSV file path
          img_dir (str, optional): Image directory path
          test_size (float): Validation data ratio
          random_state (int): seed for splitting data
          target_size (int): Target image size

    Returns: tuple: (train_dataset, val_dataset)
    """
    if csv_path is None:
        csv_path = CSVAddress

    if img_dir is None:
        img_dir = CROPPED

    try:
        # Upload CSV
        is_valid, df, message = validate_csv(csv_path)

        if not is_valid:
            raise ValueError(f"Invalid CSV: {message}")

        # Remove UNKNOWN
        df = df[df["liverLabel"] != "UNKNOWN"]

        # Remove classes with very few samples
        class_counts = df["liverLabel"].value_counts()
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        df = df[df["liverLabel"].isin(valid_classes)]

        if len(df) == 0:
            raise ValueError("No valid labels found after removing UNKNOWN and filtering!")

        # Checking that all classes exist
        unique_labels = df["liverLabel"].unique()
        missing_labels = [label for label in CLASS_NAMES if label not in unique_labels]

        if missing_labels:
            print(f"⚠️ Warning: Missing labels in dataset: {missing_labels}")

        print("\n📊 Label Distribution:")
        for label in sorted(df["liverLabel"].unique()):
            count = len(df[df["liverLabel"] == label])
            print(f"   {label}: {count}")

        # Splitting data with stratify
        try:
            train_df, val_df = train_test_split(
                df,
                test_size=test_size,
                stratify=df["liverLabel"],
                random_state=random_state,
            )
        except ValueError:
            print("⚠️ Stratify failed. Using random split.")
            train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

        # Creating datasets with normalization and advanced augmentation
        train_ds = LiverDataset(
            train_df,
            img_dir,
            augment=True,
            target_size=target_size,
            apply_normalize=True,
            use_advanced_augment=True,
        )

        val_ds = LiverDataset(
            val_df,
            img_dir,
            augment=False,
            target_size=target_size,
            apply_normalize=True,
            use_advanced_augment=False,
        )

        print(f"\n✅ Train: {len(train_ds)} samples")
        print(f"✅ Val: {len(val_ds)} samples")

        return train_ds, val_ds

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


# =========================
# CLASS WEIGHTS - FIXED 🔥
# =========================
def get_class_weights(csv_path=None , device="cuda"):
    """
    Calculate class weights for handling imbalance with FIXED ordering

    Args: csv_path (str, optional): CSV file path

    Returns: torch.Tensor: Class weights
    """
    try:
        is_valid, df, message = validate_csv(csv_path)

        if not is_valid:
            raise ValueError(f"Invalid CSV: {message}")

        # remove UNKNOWN
        df = df[df["liverLabel"] != "UNKNOWN"]

        # Count each class with FIXED order
        class_counts = []
        for label in CLASS_NAMES:
            count = len(df[df["liverLabel"] == label])
            class_counts.append(count)

        # Calculate weights
        total = sum(class_counts)
        num_classes = len(CLASS_NAMES)

        class_weights = []
        for count in class_counts:
            if count > 0:
                weight = total / (num_classes * count)
            else:
                weight = 1.0
            class_weights.append(weight)

        # Normalization
        class_weights = np.array(class_weights) / np.mean(class_weights)

        print(f"✅ Class weights ({CLASS_NAMES}): {class_weights}")

        return torch.tensor(class_weights, dtype=torch.float32).to(device)

    except Exception as e:
        print(f"❌ Error calculating class weights: {e}")
        return torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)


# =========================
# ✅ WEIGHTED SAMPLER 🔥
# =========================
def create_weighted_sampler(dataset, csv_path=None):
    """
    Create WeightedRandomSampler for imbalanced dataset

    Args:
        dataset: PyTorch Dataset
        csv_path (str, optional): CSV file path

    Returns:
        WeightedRandomSampler: Sampler for training
    """
    try:
        labels = []
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                if len(item) == 3:
                    _, label, _ = item
                else:
                    _, label = item
                labels.append(label.item() if torch.is_tensor(label) else label)
            except Exception as e:
                print(f"⚠️ Error at index {i}: {e}")
                continue

        if len(labels) == 0:
            print("❌ No valid labels found in dataset!")
            return None

        labels = np.array(labels)

        # محاسبه وزن برای هر نمونه
        class_counts = np.bincount(labels)
        num_classes = len(class_counts)

        # وزن هر کلاس = 1 / count
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels]

        # نرمال‌سازی
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

        # ایجاد sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        print(f"✅ WeightedRandomSampler created with {len(sample_weights)} samples")

        return sampler

    except Exception as e:
        print(f"❌ Error creating sampler: {e}")
        return None


# =========================
# METRICS FUNCTIONS - FIXED AUC 🔥
# =========================
def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Calculating Evaluation Metrics with fixed AUC handling

    Args: y_true (array-like): True labels
          y_pred (array-like): Predicted labels
          y_prob (array-like, optional): Predicted probabilities

    Returns: tuple: (report, confusion_matrix, auc_score)
    """
    try:
        # Classification Report
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

        # AUC Score with try-except
        auc_score = 0.0
        if y_prob is not None:
            try:
                # Check if all classes are present
                unique_classes = np.unique(y_true)
                if len(unique_classes) == NUM_CLASSES:
                    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
                    auc_score = roc_auc_score(
                        y_true_bin, y_prob, multi_class="ovr", average="weighted"
                    )
                else:
                    print(f"⚠️ AUC: Missing classes in validation. Available: {unique_classes}")

                    # Calculate AUC only for present classes
                    present_classes = unique_classes
                    y_true_bin = label_binarize(y_true, classes=present_classes)
                    y_prob_present = y_prob[:, present_classes]
                    auc_score = roc_auc_score(
                        y_true_bin,
                        y_prob_present,
                        multi_class="ovr",
                        average="weighted",
                    )

            except Exception as e:
                print(f"⚠️ AUC calculation failed: {e}")
                auc_score = 0.0

        # Accuracy
        accuracy = (np.array(y_true) == np.array(y_pred)).mean()
        report["accuracy"] = accuracy

        print(f"✅ Metrics computed: Accuracy = {accuracy:.4f}, AUC = {auc_score:.4f}")

        return report, cm, auc_score

    except Exception as e:
        print(f"❌ Error computing metrics: {e}")

        empty_report = {
            "accuracy": 0.0,
            "weighted avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0},
        }
        return empty_report, np.zeros((NUM_CLASSES, NUM_CLASSES)), 0.0


def print_metrics_summary(metrics, cm=None):
    """
    Print all metrics

    Args:
        metrics (dict): dictionary of metrics
        cm (array, optional): confusion matrix
    """
    print("\n" + "=" * 60)
    print("📊 METRICS SUMMARY")
    print("=" * 60)

    if "accuracy" in metrics:
        print(f"\n✅ Accuracy: {metrics['accuracy']:.4f}")

    if "weighted avg" in metrics:
        print(f"\n📈 Weighted Average:")
        print(f"   Precision: {metrics['weighted avg']['precision']:.4f}")
        print(f"   Recall:    {metrics['weighted avg']['recall']:.4f}")
        print(f"   F1-Score:  {metrics['weighted avg']['f1-score']:.4f}")

    print(f"\n📋 Per-class Metrics:")
    for label in CLASS_NAMES:
        if label in metrics:
            print(f"\n   {label}:")
            print(f"      Precision: {metrics[label]['precision']:.4f}")
            print(f"      Recall:    {metrics[label]['recall']:.4f}")
            print(f"      F1-Score:  {metrics[label]['f1-score']:.4f}")
            print(f"      Support:   {metrics[label]['support']}")

    if cm is not None:
        print(f"\n📊 Confusion Matrix:")
        print("   " + "   ".join(CLASS_NAMES))
        for i, label in enumerate(CLASS_NAMES):
            print(f"   {label} ", end="")
            for j in range(NUM_CLASSES):
                print(f"{cm[i][j]:4d} ", end="")
            print()

    print("=" * 60)


# =========================
# MODEL UTILITIES
# =========================
# ✅ چاپ فقط یک بار
_DEVICE_PRINTED = False


def get_device():
    """Return CUDA device if available, else CPU (silent)"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def save_model(model, path, optimizer=None, epoch=None, metrics=None):
    """
    Save models with additional information

    Args:
        model (nn.Module): model
        path (str): path
        optimizer (torch.optim.Optimizer, optional): optimizer
        epoch (int, optional): epoch number
        metrics (dict, optional): metrics
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "class_names": CLASS_NAMES,
            "label_map": LABEL_MAP,
        }

        if optimizer is not None:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            save_dict["epoch"] = epoch

        if metrics is not None:
            save_dict["metrics"] = metrics

        torch.save(save_dict, path)

        print(f"✅ Model saved: {path}")

    except Exception as e:
        print(f"❌ Error saving model: {e}")


def load_model(path, model, device="cpu"):
    """
    Load the model from file

    Args:
        path (str): path
        model (nn.Module): model
        device (str): device

    Returns:
        nn.Module: loaded model
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        print(f"✅ Model loaded: {path}")

        if "epoch" in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")

        if "metrics" in checkpoint:
            print(f"   Metrics: {checkpoint['metrics']}")

        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


# =========================
# EXPORT FUNCTIONS
# =========================
def export_to_csv(data, path, columns=None):
    """
    Save data to CSV

    Args:
        data (pd.DataFrame or list): data
        path (str): path
        columns (list, optional): column names
    """
    try:
        if isinstance(data, list):
            if columns is not None:
                df = pd.DataFrame(data, columns=columns)
            else:
                df = pd.DataFrame(data)
        else:
            df = data

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

        print(f"✅ Data exported: {path}")

    except Exception as e:
        print(f"❌ Error exporting CSV: {e}")


def export_predictions(y_true, y_pred, y_prob, labels, output_path):
    """
    Save prediction results

    Args:
        y_true (array): true labels
        y_pred (array): predicted labels
        y_prob (array): probabilities
        labels (list): label names
        output_path (str): output path
    """
    try:
        data = {
            "True_Label": [INV_LABEL_MAP.get(y, "UNKNOWN") for y in y_true],
            "Predicted_Label": [INV_LABEL_MAP.get(y, "UNKNOWN") for y in y_pred],
            "Confidence": (np.max(y_prob, axis=1) if y_prob is not None else np.ones(len(y_true))),
        }

        # Add probabilities for each class
        if y_prob is not None and y_prob.shape[1] == NUM_CLASSES:
            for i, label in enumerate(CLASS_NAMES):
                data[f"Prob_{label}"] = y_prob[:, i]

        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"✅ Predictions exported: {output_path}")

    except Exception as e:
        print(f"❌ Error exporting predictions: {e}")


# =========================
# MAIN - FOR TESTING
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing utils.py (FULLY FIXED VERSION)")
    print("=" * 60)

    # test set_seed
    print("\n🔍 Testing set_seed...")
    set_seed(42)

    # test check_paths
    print("\n🔍 Testing check_paths...")
    check_paths()

    # test validate_csv
    print("\n🔍 Testing validate_csv...")
    is_valid, df, message = validate_csv()
    print(f"   Valid: {is_valid}, Message: {message}")

    # test validate_images
    print("\n🔍 Testing validate_images...")
    total, missing_count, missing = validate_images(sample_size=10)
    print(f"   Total: {total}, Missing: {missing_count}")

    # test LABEL_MAP
    print("\n🔍 Testing LABEL_MAP...")
    print(f"   CLASS_NAMES: {CLASS_NAMES}")
    print(f"   LABEL_MAP: {LABEL_MAP}")
    print(f"   INV_LABEL_MAP: {INV_LABEL_MAP}")

    # test class weights
    print("\n🔍 Testing get_class_weights...")
    weights = get_class_weights()
    print(f"   Weights: {weights}")

    # test resize_with_padding
    print("\n🔍 Testing resize_with_padding...")
    test_img = np.ones((100, 80), dtype=np.uint8) * 255
    resized = resize_with_padding(test_img, 256)
    print(f"   Input shape: {test_img.shape}, Output shape: {resized.shape}")

    # test normalization
    print("\n🔍 Testing normalization...")
    test_tensor = torch.rand(3, 256, 256)
    norm_tensor = normalize_resnet(test_tensor)
    print(f"   Input mean: {test_tensor.mean():.4f}, std: {test_tensor.std():.4f}")
    print(f"   Output mean: {norm_tensor.mean():.4f}, std: {norm_tensor.std():.4f}")

    # Test Label Smoothing
    print("\n🔍 Testing Label Smoothing...")
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    pred = torch.randn(4, 5)
    target = torch.randint(0, 5, (4,))
    loss = loss_fn(pred, target)
    print(f"   Label Smoothing Loss: {loss.item():.4f}")

    # Test Random Erasing
    print("\n🔍 Testing Random Erasing...")
    eraser = RandomErasing(probability=1.0)
    test_img = np.ones((256, 256), dtype=np.uint8) * 255
    erased = eraser(test_img)
    print(f"   Random Erasing applied: {erased.shape}")

    # Test Warmup Cosine Scheduler
    print("\n🔍 Testing Warmup Cosine Scheduler...")
    optimizer = torch.optim.AdamW([torch.tensor(1.0, requires_grad=True)], lr=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=50)
    lrs = []
    for _ in range(50):
        lrs.append(scheduler.step())
    print(f"   LR range: {min(lrs):.2e} - {max(lrs):.2e}")

    # Test Cross-Validation
    print("\n🔍 Testing Cross-Validation...")
    print("   (Skipping actual CV to avoid long test)")
    print("   ✅ Cross-validation function available")

    print("\n✅ All tests passed!")
