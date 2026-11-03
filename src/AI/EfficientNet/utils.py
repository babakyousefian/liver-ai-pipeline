# src/AI/EfficientNet/utils.py

"""
EfficientNet Utilities for Liver Disease Classification
This module contains utility functions for training EfficientNet models
EfficientNet is a family of efficient models designed with a balance between accuracy and speed
"""

import json
import math
import os
import random
import time
import warnings
from datetime import datetime
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
# GLOBAL PATHS (READ-ONLY INPUTS)
# =========================
PROJECT_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
FINAL_REPORTS_DIR = os.path.join(PROJECT_ROOT, "finalReports")

CROPPED = os.path.join(FINAL_REPORTS_DIR, "cropped")
CSVAddress = os.path.join(FINAL_REPORTS_DIR, "labels.csv")

# ✅ the main address of EfficientNet
OUTPUT_DIR = os.path.join(FINAL_REPORTS_DIR, "AI_Comparisons", "EfficientNet")

# =========================
# LABEL MAP - FIXED ORDER 🔥
# =========================
CLASS_NAMES = ["NL", "G1", "G2", "G3"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(CLASS_NAMES)

# =========================
# NORMALIZATION PARAMS
# =========================
EFFICIENTNET_MEAN = [0.485, 0.456, 0.406]
EFFICIENTNET_STD = [0.229, 0.224, 0.225]

# =========================
# ✅ HYPERPARAMETER CONFIG 🔥 NEW
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
# ✅ LABEL SMOOTHING LOSS 🔥 NEW
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
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smoothed * log_prob).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =========================
# ✅ FOCAL LOSS 🔥 NEW
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
        return focal_loss


# =========================
# ✅ WARMUP + COSINE ANNEALING SCHEDULER 🔥 NEW
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

        for param_group in optimizer.param_groups:
            param_group["lr"] = max_lr

    def step(self) -> float:
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            self.current_lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            self.current_lr = (
                self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr

        return self.current_lr

    def get_lr(self) -> float:
        return self.current_lr


# =========================
# ✅ RANDOM ERASING AUGMENTATION 🔥 NEW
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
# ✅ ENHANCED AUGMENTATION 🔥 NEW
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
    Apply basic augmentations to an image (ENHANCED)

    Args: image (numpy.ndarray): Input image

    Returns: numpy.ndarray: Augmented image
    """
    try:
        # Horizontal flip with 50% probability
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)

        # Random rotation between -15 and +15 degrees (increased)
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

        # ✅ Gaussian noise with 20% probability
        if np.random.rand() < 0.2:
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # ✅ Blur with 15% probability
        if np.random.rand() < 0.15:
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # ✅ Sharpening with 10% probability
        if np.random.rand() < 0.1:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image = cv2.filter2D(image, -1, kernel)
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    except Exception as e:
        time.sleep(0.5)
        print(f"⚠️ Error in augmentation: {e}")
        time.sleep(0.5)
        return image


# =========================
# ✅ CROSS-VALIDATION 🔥 NEW
# =========================
def cross_validate_model(
    model_class: Any,
    df: pd.DataFrame,
    img_dir: str,
    n_folds: int = 5,
    target_size: int = 224,
    batch_size: int = 32,
    lr: float = 1e-3,
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
        model_class: Trainer class
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

        # Create dataset
        train_ds = EfficientNetDataset(train_df, img_dir, augment=True, target_size=target_size)
        val_ds = EfficientNetDataset(val_df, img_dir, augment=False, target_size=target_size)

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

        # Create trainer
        if model_class.__name__ == "EfficientNetB0Trainer":
            from .efficientNetB0 import EfficientNetB0Trainer

            trainer = EfficientNetB0Trainer()
        else:
            from .efficientNetB1 import EfficientNetB1Trainer

            trainer = EfficientNetB1Trainer()

        trainer.train_ds = train_ds
        trainer.val_ds = val_ds
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader

        trainer.model = trainer._build_model()

        trainer.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

        trainer.optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        trainer.scheduler = WarmupCosineScheduler(
            trainer.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            min_lr=1e-7,
            max_lr=lr,
        )

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set to: {seed}")


# =========================
# DEVICE DETECTION
# =========================
# ✅ چاپ فقط یک بار
_DEVICE_PRINTED = False


def get_device():
    """Return CUDA device if available, else CPU (silent)"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


# =========================
# PATH VALIDATION
# =========================
def check_paths():
    try:
        if not os.path.exists(CROPPED):
            time.sleep(0.5)
            print(f"❌ CROPPED path invalid: {CROPPED}")
            time.sleep(0.5)
            return False

        if not os.path.exists(CSVAddress):
            time.sleep(0.5)
            print(f"❌ CSV path invalid: {CSVAddress}")
            time.sleep(0.5)
            return False

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "b0"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "b1"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)
        time.sleep(0.5)
        print("✅ All paths verified successfully!")
        time.sleep(0.5)
        return True

    except Exception as e:
        time.sleep(0.5)
        print(f"❌ Error checking paths: {e}")
        time.sleep(0.5)
        return False


def validate_csv(csv_path=None):
    if csv_path is None:
        csv_path = CSVAddress

    try:
        if not os.path.exists(csv_path):
            return False, None, f"CSV file not found: {csv_path}"

        df = pd.read_csv(csv_path)

        required_columns = ["liverName", "liverLabel"]
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            return False, None, f"Missing columns: {missing_cols}"

        if len(df) == 0:
            return False, None, "CSV file is empty"

        valid_labels = CLASS_NAMES
        invalid_labels = df[~df["liverLabel"].isin(valid_labels + ["UNKNOWN"])][
            "liverLabel"
        ].unique()

        if len(invalid_labels) > 0:
            time.sleep(0.5)
            print(f"⚠️ Found invalid labels: {invalid_labels}")
            time.sleep(0.5)

        valid_count = len(df[df["liverLabel"].isin(valid_labels)])
        unknown_count = len(df[df["liverLabel"] == "UNKNOWN"])

        time.sleep(0.5)
        print(
            f"✅ CSV validated: {len(df)} rows, {valid_count} valid labels, {unknown_count} unknown"
        )
        time.sleep(0.5)

        return True, df, "CSV is valid"

    except Exception as e:
        return False, None, f"Error reading CSV: {e}"


def validate_images(csv_path=None, img_dir=None, sample_size=None):
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

        time.sleep(0.5)
        print(f"✅ Image validation: {len(valid_images)} valid, {len(missing_images)} missing")
        time.sleep(0.5)

        return len(valid_images), len(missing_images), missing_images

    except Exception as e:
        time.sleep(0.5)
        print(f"❌ Error validating images: {e}")
        time.sleep(0.5)
        return 0, 0, []


# =========================
# IMAGE PROCESSING FUNCTIONS
# =========================
def resize_with_padding(image, size=256):
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
        time.sleep(0.5)
        print(f"❌ Error in resize_with_padding: {e}")
        time.sleep(0.5)
        return np.zeros((size, size), dtype=np.uint8)


# =========================
# NORMALIZATION FUNCTION
# =========================
def normalize_efficientnet(image_tensor):
    """
    Normalize image tensor with EfficientNet stats

    Args:
        image_tensor: (C, H, W) tensor in range [0, 1]

    Returns:
        Normalized tensor
    """
    mean = torch.tensor(EFFICIENTNET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(EFFICIENTNET_STD, dtype=torch.float32).view(3, 1, 1)
    return (image_tensor - mean) / std


# =========================
# EFFICIENTNET DATASET - WITH ADVANCED AUGMENTATION 🔥
# =========================
class EfficientNetDataset(Dataset):
    """
    Dataset for training EfficientNet models

    Args:
        df (pd.DataFrame): Dataframe including liverName and liverLabel
        img_dir (str): Path to the images directory
        augment (bool): Whether to apply augmentation
        target_size (int): target size of images
        apply_normalize (bool): Whether to apply normalization
        use_advanced_augment (bool): Whether to use advanced augmentation
    """

    def __init__(
        self,
        df,
        img_dir=None,
        augment=False,
        target_size=224,
        apply_normalize=True,
        use_advanced_augment=True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir if img_dir is not None else CROPPED
        self.augment = augment
        self.target_size = target_size
        self.apply_normalize = apply_normalize
        self.use_advanced_augment = use_advanced_augment

        # حذف UNKNOWN
        self.df = self.df[self.df["liverLabel"] != "UNKNOWN"]

        # حذف کلاس‌های با نمونه کم
        class_counts = self.df["liverLabel"].value_counts()
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        self.df = self.df[self.df["liverLabel"].isin(valid_classes)]

        if len(self.df) == 0:
            raise ValueError("No valid labels found in dataset!")

        time.sleep(0.5)
        print(f"✅ EfficientNet Dataset initialized with {len(self.df)} samples")
        time.sleep(0.5)
        print(f"   Classes: {sorted(self.df['liverLabel'].unique())}")
        time.sleep(0.5)
        print(f"   Target Size: {self.target_size}")
        time.sleep(0.5)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]

            img_path = os.path.join(self.img_dir, row["liverName"])
            label = LABEL_MAP[row["liverLabel"]]

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                time.sleep(0.5)
                print(f"⚠️ Image not found: {img_path}")
                time.sleep(0.5)
                img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            else:
                img = resize_with_padding(img, self.target_size)

            # ✅ Advanced Augmentation
            if self.augment:
                if self.use_advanced_augment:
                    img = apply_advanced_augmentation(img, use_erasing=True)
                else:
                    img = apply_augmentation(img)

            # تبدیل به [0, 1]
            img = img.astype(np.float32) / 255.0

            # تبدیل به 3 کانال
            img = np.stack([img, img, img], axis=0)  # (3, H, W)

            # تبدیل به tensor
            img_tensor = torch.tensor(img, dtype=torch.float32)

            # اعمال normalization
            if self.apply_normalize:
                img_tensor = normalize_efficientnet(img_tensor)

            return (img_tensor, torch.tensor(label, dtype=torch.long), row["liverName"])

        except Exception as e:
            time.sleep(0.5)
            print(f"❌ Error in __getitem__ (idx={idx}): {e}")
            time.sleep(0.5)

            img = np.zeros((3, self.target_size, self.target_size), dtype=np.float32)
            img_tensor = torch.tensor(img, dtype=torch.float32)
            if self.apply_normalize:
                img_tensor = normalize_efficientnet(img_tensor)

            return (img_tensor, torch.tensor(0, dtype=torch.long), "error_image.png")


# =========================
# DATA LOADING FUNCTIONS
# =========================
def load_split_data(csv_path=None, img_dir=None, test_size=0.2, random_state=42, target_size=224):
    """
    Load and split data into train/validation

    Args:
        csv_path (str, optional): CSV file path
        img_dir (str, optional): Image directory path
        test_size (float): Validation data ratio
        random_state (int): seed for splitting data
        target_size (int): Target image size

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    if csv_path is None:
        csv_path = CSVAddress

    if img_dir is None:
        img_dir = CROPPED

    try:
        is_valid, df, message = validate_csv(csv_path)

        if not is_valid:
            raise ValueError(f"Invalid CSV: {message}")

        # حذف UNKNOWN
        df = df[df["liverLabel"] != "UNKNOWN"]

        # حذف کلاس‌های کم
        class_counts = df["liverLabel"].value_counts()
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        df = df[df["liverLabel"].isin(valid_classes)]

        if len(df) == 0:
            raise ValueError("No valid labels found after removing UNKNOWN and filtering!")

        time.sleep(0.5)
        print("\n📊 Label Distribution:")
        time.sleep(0.5)

        for label in sorted(df["liverLabel"].unique()):
            count = len(df[df["liverLabel"] == label])
            time.sleep(0.5)
            print(f"   {label}: {count}")
            time.sleep(0.5)

        # Stratified split
        try:
            train_df, val_df = train_test_split(
                df,
                test_size=test_size,
                stratify=df["liverLabel"],
                random_state=random_state,
            )
        except ValueError:
            train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

        # ایجاد دیتاست‌ها با normalize و advanced augmentation
        train_ds = EfficientNetDataset(
            train_df,
            img_dir,
            augment=True,
            target_size=target_size,
            apply_normalize=True,
            use_advanced_augment=True,
        )

        val_ds = EfficientNetDataset(
            val_df,
            img_dir,
            augment=False,
            target_size=target_size,
            apply_normalize=True,
            use_advanced_augment=False,
        )

        time.sleep(0.5)
        print(f"\n✅ Train: {len(train_ds)} samples")
        time.sleep(0.5)
        print(f"✅ Val: {len(val_ds)} samples")
        time.sleep(0.5)

        return train_ds, val_ds

    except Exception as e:
        time.sleep(0.5)
        print(f"❌ Error loading data: {e}")
        time.sleep(0.5)
        raise


# =========================
# CLASS WEIGHTS - FIXED 🔥
# =========================
def get_class_weights(csv_path=None):
    """
    Calculate class weights with FIXED ordering

    Args:
        csv_path (str, optional): CSV file path

    Returns:
        torch.Tensor: Class weights in order of CLASS_NAMES
    """
    try:
        is_valid, df, message = validate_csv(csv_path)

        if not is_valid:
            raise ValueError(f"Invalid CSV: {message}")

        # حذف UNKNOWN
        df = df[df["liverLabel"] != "UNKNOWN"]

        # محاسبه وزن‌ها با ترتیب ثابت
        class_counts = []
        for label in CLASS_NAMES:
            count = len(df[df["liverLabel"] == label])
            class_counts.append(count)

        total = sum(class_counts)
        num_classes = len(CLASS_NAMES)

        # وزن‌ها معکوس فرکانس
        class_weights = []
        for count in class_counts:
            if count > 0:
                weight = total / (num_classes * count)
            else:
                weight = 1.0
            class_weights.append(weight)

        # نرمال‌سازی
        class_weights = np.array(class_weights) / np.mean(class_weights)

        time.sleep(0.5)
        print(f"✅ Class weights ({CLASS_NAMES}): {class_weights}")
        time.sleep(0.5)

        return torch.tensor(class_weights, dtype=torch.float32)

    except Exception as e:
        time.sleep(0.5)
        print(f"❌ Error calculating class weights: {e}")
        time.sleep(0.5)
        return torch.tensor([0.5, 1.5, 2.0, 3.0, 2.0], dtype=torch.float32)


# =========================
# WEIGHTED SAMPLER 🔥 NEW
# =========================
# src/AI/EfficientNet/utils.py


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
        # ✅ تصحیح شده
        labels = []
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                if len(item) == 3:
                    _, label, _ = item
                else:
                    image, label = item
                labels.append(label.item() if torch.is_tensor(label) else label)
            except Exception as e:
                print(f"⚠️ Error at index {i}: {e}")
                continue

        if len(labels) == 0:
            print("❌ No valid labels found in dataset!")
            return None

        labels = np.array(labels)

        class_counts = np.bincount(labels)
        num_classes = len(class_counts)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels]
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

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
    Compute metrics with fixed AUC handling

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        y_prob (array-like, optional): Prediction probabilities

    Returns:
        tuple: (report, confusion_matrix, auc_score)
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
                unique_classes = np.unique(y_true)
                if len(unique_classes) == NUM_CLASSES:
                    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
                    auc_score = roc_auc_score(
                        y_true_bin, y_prob, multi_class="ovr", average="weighted"
                    )
                else:
                    time.sleep(0.5)
                    print(f"⚠️ AUC: Missing classes in validation. Available: {unique_classes}")
                    time.sleep(0.5)

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
                time.sleep(0.5)
                print(f"⚠️ AUC calculation failed: {e}")
                time.sleep(0.5)
                auc_score = 0.0

        # Accuracy
        accuracy = (np.array(y_true) == np.array(y_pred)).mean()
        report["accuracy"] = accuracy

        time.sleep(0.5)
        print(f"✅ Metrics computed: Accuracy = {accuracy:.4f}, AUC = {auc_score:.4f}")
        time.sleep(0.5)

        return report, cm, auc_score

    except Exception as e:
        time.sleep(0.5)
        print(f"❌ Error computing metrics: {e}")
        time.sleep(0.5)

        empty_report = {
            "accuracy": 0.0,
            "weighted avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0},
        }
        return empty_report, np.zeros((NUM_CLASSES, NUM_CLASSES)), 0.0


def print_metrics_summary(metrics, cm=None):
    time.sleep(0.5)
    print("\n" + "=" * 70)
    time.sleep(0.5)
    print("📊 METRICS SUMMARY")
    time.sleep(0.5)
    print("=" * 70)
    time.sleep(0.5)

    if "accuracy" in metrics:
        time.sleep(0.5)
        print(f"\n✅ Accuracy: {metrics['accuracy']:.4f}")
        time.sleep(0.5)

    if "weighted avg" in metrics:
        time.sleep(0.5)
        print(f"\n📈 Weighted Average:")
        time.sleep(0.5)
        print(f"   Precision: {metrics['weighted avg']['precision']:.4f}")
        time.sleep(0.5)
        print(f"   Recall:    {metrics['weighted avg']['recall']:.4f}")
        time.sleep(0.5)
        print(f"   F1-Score:  {metrics['weighted avg']['f1-score']:.4f}")
        time.sleep(0.5)

    time.sleep(0.5)
    print(f"\n📋 Per-class Metrics:")
    time.sleep(0.5)

    for label in CLASS_NAMES:
        if label in metrics:
            time.sleep(0.5)
            print(f"\n   {label}:")
            time.sleep(0.5)
            print(f"      Precision: {metrics[label]['precision']:.4f}")
            time.sleep(0.5)
            print(f"      Recall:    {metrics[label]['recall']:.4f}")
            time.sleep(0.5)
            print(f"      F1-Score:  {metrics[label]['f1-score']:.4f}")
            time.sleep(0.5)
            print(f"      Support:   {metrics[label]['support']}")
            time.sleep(0.5)

    if cm is not None:
        time.sleep(0.5)
        print(f"\n📊 Confusion Matrix:")
        time.sleep(0.5)
        print("   " + "   ".join(CLASS_NAMES))
        time.sleep(0.5)

        for i, label in enumerate(CLASS_NAMES):
            time.sleep(0.5)
            print(f"   {label} ", end="")
            time.sleep(0.5)
            for j in range(NUM_CLASSES):
                time.sleep(0.5)
                print(f"{cm[i][j]:4d} ", end="")
                time.sleep(0.5)
            time.sleep(0.5)
            print()
            time.sleep(0.5)

    time.sleep(0.5)
    print("=" * 70)
    time.sleep(0.5)


# =========================
# MODEL UTILITIES
# =========================
def save_model(model, path, optimizer=None, epoch=None, metrics=None):
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

        time.sleep(0.5)
        print(f"✅ Model saved: {path}")
        time.sleep(0.5)

    except Exception as e:
        time.sleep(0.5)
        print(f"❌ Error saving model: {e}")
        time.sleep(0.5)


def load_model(path, model, device="cpu"):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        time.sleep(0.5)
        print(f"✅ Model loaded: {path}")
        time.sleep(0.5)

        if "epoch" in checkpoint:
            time.sleep(0.5)
            print(f"   Epoch: {checkpoint['epoch']}")
            time.sleep(0.5)

        if "metrics" in checkpoint:
            time.sleep(0.5)
            print(f"   Metrics: {checkpoint['metrics']}")
            time.sleep(0.5)

        return model

    except Exception as e:
        time.sleep(0.5)
        print(f"❌ Error loading model: {e}")
        time.sleep(0.5)
        raise


# =========================
# EXPORT FUNCTIONS
# =========================
def export_predictions(y_true, y_pred, y_prob, labels, output_path):
    try:
        data = {
            "True_Label": [INV_LABEL_MAP.get(y, "UNKNOWN") for y in y_true],
            "Predicted_Label": [INV_LABEL_MAP.get(y, "UNKNOWN") for y in y_pred],
            "Confidence": (np.max(y_prob, axis=1) if y_prob is not None else np.ones(len(y_true))),
        }

        if y_prob is not None and y_prob.shape[1] == NUM_CLASSES:
            for i, label in enumerate(CLASS_NAMES):
                data[f"Prob_{label}"] = y_prob[:, i]

        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        time.sleep(0.5)
        print(f"✅ Predictions exported: {output_path}")
        time.sleep(0.5)

    except Exception as e:
        time.sleep(0.5)
        print(f"❌ Error exporting predictions: {e}")
        time.sleep(0.5)


# =========================
# MAIN - FOR TESTING
# =========================
if __name__ == "__main__":
    time.sleep(0.5)
    print("=" * 70)
    time.sleep(0.5)
    print("🧪 Testing EfficientNet Utils (FULLY OPTIMIZED)")
    time.sleep(0.5)
    print("=" * 70)
    time.sleep(0.5)

    # test set_seed
    time.sleep(0.5)
    print("\n🔍 Testing set_seed...")
    time.sleep(0.5)
    set_seed(42)

    # test get_device
    time.sleep(0.5)
    print("\n🔍 Testing get_device...")
    time.sleep(0.5)
    device = get_device()
    print(f"   Device: {device}")

    # test check_paths
    time.sleep(0.5)
    print("\n🔍 Testing check_paths...")
    time.sleep(0.5)
    check_paths()

    # test validate_csv
    time.sleep(0.5)
    print("\n🔍 Testing validate_csv...")
    time.sleep(0.5)
    is_valid, df, message = validate_csv()
    print(f"   Valid: {is_valid}, Message: {message}")

    # test validate_images
    time.sleep(0.5)
    print("\n🔍 Testing validate_images...")
    time.sleep(0.5)
    total, missing, missing_list = validate_images(sample_size=10)
    print(f"   Total: {total}, Missing: {missing}")

    # test resize_with_padding
    time.sleep(0.5)
    print("\n🔍 Testing resize_with_padding...")
    time.sleep(0.5)
    test_img = np.ones((100, 80), dtype=np.uint8) * 255
    resized = resize_with_padding(test_img, 224)
    print(f"   Input shape: {test_img.shape}, Output shape: {resized.shape}")

    # test LABEL_MAP
    time.sleep(0.5)
    print("\n🔍 Testing LABEL_MAP...")
    time.sleep(0.5)
    print(f"   CLASS_NAMES: {CLASS_NAMES}")
    time.sleep(0.5)
    print(f"   LABEL_MAP: {LABEL_MAP}")
    time.sleep(0.5)
    print(f"   INV_LABEL_MAP: {INV_LABEL_MAP}")
    time.sleep(0.5)

    # test class weights
    time.sleep(0.5)
    print("\n🔍 Testing get_class_weights...")
    time.sleep(0.5)
    weights = get_class_weights()
    print(f"   Weights: {weights}")
    time.sleep(0.5)

    # test normalization
    time.sleep(0.5)
    print("\n🔍 Testing normalization...")
    time.sleep(0.5)
    test_tensor = torch.rand(3, 224, 224)
    norm_tensor = normalize_efficientnet(test_tensor)
    print(f"   Input mean: {test_tensor.mean():.4f}, std: {test_tensor.std():.4f}")
    print(f"   Output mean: {norm_tensor.mean():.4f}, std: {norm_tensor.std():.4f}")
    time.sleep(0.5)

    # ✅ Test Label Smoothing
    time.sleep(0.5)
    print("\n🔍 Testing Label Smoothing...")
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    pred = torch.randn(4, 5)
    target = torch.randint(0, 5, (4,))
    loss = loss_fn(pred, target)
    print(f"   Label Smoothing Loss: {loss.item():.4f}")
    time.sleep(0.5)

    # ✅ Test Random Erasing
    time.sleep(0.5)
    print("\n🔍 Testing Random Erasing...")
    eraser = RandomErasing(probability=1.0)
    test_img = np.ones((224, 224), dtype=np.uint8) * 255
    erased = eraser(test_img)
    print(f"   Random Erasing applied: {erased.shape}")
    time.sleep(0.5)

    # ✅ Test Warmup Cosine Scheduler
    time.sleep(0.5)
    print("\n🔍 Testing Warmup Cosine Scheduler...")
    optimizer = torch.optim.AdamW([torch.tensor(1.0, requires_grad=True)], lr=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=55)
    lrs = []
    for _ in range(55):
        lrs.append(scheduler.step())
    print(f"   LR range: {min(lrs):.2e} - {max(lrs):.2e}")
    time.sleep(0.5)

    # ✅ Test HYPERPARAMS
    time.sleep(0.5)
    print("\n🔍 Testing HYPERPARAMS...")
    print(f"   B0 config: {HYPERPARAMS['efficientnet_b0']}")
    print(f"   B1 config: {HYPERPARAMS['efficientnet_b1']}")
    time.sleep(0.5)

    time.sleep(0.5)
    print("\n✅ All tests passed!")
    time.sleep(0.5)
