# src/AI/VGG16/utils.py
# -*- coding: utf-8 -*-

"""
VGG16 Utilities
===============
Dataset, transforms, metrics, plotting, and training helpers.
Designed for liver ultrasound 4-class classification (NL, G1, G2, G3).

Author: Babak Yousefian
"""

import os
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, auc,
    cohen_kappa_score, matthews_corrcoef,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DEFAULT_CLASSES = ["NL", "G1", "G2", "G3"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ================================================================
#  REPRODUCIBILITY & DEVICE
# ================================================================

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return best available device."""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[Device] CUDA: {p.name} | VRAM: {p.total_memory/1e9:.2f} GB")
        return torch.device("cuda")
    print("[Device] CPU")
    return torch.device("cpu")


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ================================================================
#  AVERAGE METER
# ================================================================

class AverageMeter:
    """Track running average of a metric."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


# ================================================================
#  EARLY STOPPING
# ================================================================

class EarlyStopping:
    """Early stopping when monitored metric stops improving."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = "max", verbose: bool = True):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] Triggered. Best epoch: {self.best_epoch}")
                return True
        return False


# ================================================================
#  DATASET (labels.csv + cropped images)
# ================================================================

class LiverDataset(Dataset):
    """
    Custom dataset for liver ultrasound.

    Loads images from `cropped/` folder and labels from `labels.csv`.
    labels.csv must have columns: filename, label
    """

    def __init__(self, images_dir: str, labels_csv: str,
                 transform: Optional[Callable] = None,
                 class_to_idx: Optional[Dict[str, int]] = None,
                 samples: Optional[List[Tuple[str, int]]] = None):
        self.images_dir = Path(images_dir)
        self.labels_csv = Path(labels_csv)
        self.transform = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Labels CSV not found: {self.labels_csv}")

        df = pd.read_csv(self.labels_csv)
        df.columns = [c.strip().lower() for c in df.columns]

        # Find columns
        fname_col = next((c for c in ["filename", "file_name", "image", "image_name", "name", "path"] if c in df.columns), df.columns[0])
        label_col = next((c for c in ["label", "class", "grade", "target", "category"] if c in df.columns), df.columns[1])

        self.df = df[[fname_col, label_col]].rename(columns={fname_col: "filename", label_col: "label"})

        # Class mapping
        if class_to_idx is None:
            unique = sorted(self.df["label"].astype(str).str.strip().unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(unique)}
        else:
            self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        # Build samples
        if samples is None:
            self.samples = []
            missing = 0
            for _, row in self.df.iterrows():
                fname = str(row["filename"]).strip()
                label = str(row["label"]).strip()
                if label not in self.class_to_idx:
                    continue
                img_path = self._find_image(fname)
                if img_path is None:
                    missing += 1
                    continue
                self.samples.append((str(img_path), self.class_to_idx[label]))
            if missing > 0:
                print(f"[LiverDataset] WARNING: {missing} images not found")
        else:
            self.samples = samples

        self.targets = [s[1] for s in self.samples]
        print(f"[LiverDataset] Loaded {len(self.samples)} samples")
        print(f"[LiverDataset] Classes: {self.classes}")

    def _find_image(self, filename: str) -> Optional[Path]:
        """Locate image file by trying multiple paths/extensions."""
        p = Path(filename)
        if p.is_absolute() and p.exists():
            return p
        candidate = self.images_dir / filename
        if candidate.exists():
            return candidate
        stem = p.stem
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            c = self.images_dir / f"{stem}{ext}"
            if c.exists():
                return c
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.targets, minlength=len(self.classes))
        counts = np.maximum(counts, 1)
        w = 1.0 / counts
        w = w / w.sum() * len(self.classes)
        return torch.tensor(w, dtype=torch.float32)

    def get_sample_weights(self) -> np.ndarray:
        cw = self.get_class_weights().numpy()
        return np.array([cw[t] for t in self.targets])

    def get_labels(self) -> np.ndarray:
        return np.array(self.targets)


# ================================================================
#  TRANSFORMS
# ================================================================

def get_train_transforms(input_size: int = 224, strong: bool = True) -> transforms.Compose:
    """Training transforms with augmentation (VGG16 overfits easily)."""
    if strong:
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.15, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1), shear=5),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(input_size: int = 224) -> transforms.Compose:
    """Deterministic transforms for validation/test."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ================================================================
#  DATALOADERS
# ================================================================

def build_dataloaders(images_dir: str, labels_csv: str,
                      batch_size: int = 16, num_workers: int = 4,
                      input_size: int = 224,
                      val_split: float = 0.15, test_split: float = 0.15,
                      seed: int = 42, use_weighted_sampler: bool = True,
                      class_to_idx: Optional[Dict[str, int]] = None,
                      train_samples=None, val_samples=None, test_samples=None
                      ) -> Dict[str, Any]:
    """Build train/val/test DataLoaders."""

    # Class mapping
    if class_to_idx is None:
        df = pd.read_csv(labels_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        label_col = next((c for c in ["label", "class", "grade", "target", "category"] if c in df.columns), df.columns[1])
        unique = sorted(df[label_col].astype(str).str.strip().unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(unique)}

    # CV mode or split mode
    if train_samples is not None and val_samples is not None:
        train_ds = LiverDataset(images_dir, labels_csv, get_train_transforms(input_size, True), class_to_idx, train_samples)
        val_ds = LiverDataset(images_dir, labels_csv, get_val_transforms(input_size), class_to_idx, val_samples)
        test_ds = None
        if test_samples is not None:
            test_ds = LiverDataset(images_dir, labels_csv, get_val_transforms(input_size), class_to_idx, test_samples)
    else:
        full = LiverDataset(images_dir, labels_csv, None, class_to_idx)
        labels = full.get_labels()
        indices = np.arange(len(full))
        tv_idx, test_idx = train_test_split(indices, test_size=test_split, stratify=labels, random_state=seed)
        val_ratio = val_split / (1.0 - test_split)
        tr_idx, val_idx = train_test_split(tv_idx, test_size=val_ratio, stratify=labels[tv_idx], random_state=seed)

        train_samples = [full.samples[i] for i in tr_idx]
        val_samples = [full.samples[i] for i in val_idx]
        test_samples = [full.samples[i] for i in test_idx]

        train_ds = LiverDataset(images_dir, labels_csv, get_train_transforms(input_size, True), class_to_idx, train_samples)
        val_ds = LiverDataset(images_dir, labels_csv, get_val_transforms(input_size), class_to_idx, val_samples)
        test_ds = LiverDataset(images_dir, labels_csv, get_val_transforms(input_size), class_to_idx, test_samples)

    # Sampler
    if use_weighted_sampler:
        sw = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds) if test_ds else 0}")
    print(f"[Data] Class weights: {train_ds.get_class_weights().numpy()}")

    return {
        "train": train_loader, "val": val_loader, "test": test_loader,
        "class_to_idx": class_to_idx, "class_names": train_ds.classes,
        "train_samples": train_ds.samples, "val_samples": val_ds.samples,
        "test_samples": test_ds.samples if test_ds else None,
        "num_classes": len(class_to_idx),
    }


# ================================================================
#  METRICS
# ================================================================

@torch.no_grad()
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None,
                    num_classes: int = 4,
                    class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    try: kappa = float(cohen_kappa_score(y_true, y_pred))
    except: kappa = 0.0
    try: mcc = float(matthews_corrcoef(y_true, y_pred))
    except: mcc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    auc_macro, auc_per_class = None, None
    if y_prob is not None:
        try:
            auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            auc_per_class = []
            for i in range(num_classes):
                y_bin = (y_true == i).astype(int)
                auc_per_class.append(float(roc_auc_score(y_bin, y_prob[:, i])) if len(np.unique(y_bin)) > 1 else float("nan"))
        except Exception as e:
            print(f"[Metrics] AUC failed: {e}")

    report = classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                   target_names=class_names, digits=4, zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(f1m),
        "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(f1w),
        "precision_per_class": p.tolist(), "recall_per_class": r.tolist(),
        "f1_per_class": f1.tolist(), "support_per_class": s.tolist(),
        "confusion_matrix": cm.tolist(),
        "auc_macro": auc_macro, "auc_per_class": auc_per_class,
        "kappa": kappa, "mcc": mcc,
        "classification_report": report, "class_names": class_names,
    }


# ================================================================
#  PLOTTING
# ================================================================

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix",
                           normalize=True, figsize=(8, 6), dpi=150):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cm = np.asarray(cm)
    if normalize:
        disp = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fmt, cmap = ".2%", "Blues"
    else:
        disp, fmt, cmap = cm, "d", "Blues"
    sns.heatmap(disp, annot=True, fmt=fmt, cmap=cmap, cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, square=True, linewidths=0.5, annot_kws={"size": 12})
    ax.set_title(title + (" (Normalized)" if normalize else ""), fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(save_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_roc_curves(y_true, y_prob, class_names, save_path,
                    title="ROC Curves (One-vs-Rest)", figsize=(8, 6), dpi=150):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2: continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
    ax.set_xlabel("FPR", fontsize=12); ax.set_ylabel("TPR", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_training_curves(history, save_path, title="Training History",
                         figsize=(14, 5), dpi=150):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    if "train_loss" in history and "val_loss" in history:
        ep = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(ep, history["train_loss"], label="Train", linewidth=2, marker="o", markersize=3)
        axes[0].plot(ep, history["val_loss"], label="Val", linewidth=2, marker="s", markersize=3)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss", fontweight="bold"); axes[0].legend(); axes[0].grid(alpha=0.3)
    if "train_acc" in history and "val_acc" in history:
        ep = range(1, len(history["train_acc"]) + 1)
        axes[1].plot(ep, history["train_acc"], label="Train", linewidth=2, marker="o", markersize=3)
        axes[1].plot(ep, history["val_acc"], label="Val", linewidth=2, marker="s", markersize=3)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy", fontweight="bold"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(save_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_metrics_bar(metrics, save_path, title="Metrics Summary",
                     figsize=(10, 6), dpi=150):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics.get(k, 0.0) for k in keys]
    if metrics.get("auc_macro") is not None:
        keys.append("auc_macro"); labels.append("AUC"); values.append(metrics["auc_macro"])
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = sns.color_palette("viridis", len(values))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); plt.savefig(save_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_per_class_metrics(metrics, save_path, title="Per-Class Metrics",
                           figsize=(10, 6), dpi=150):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    class_names = metrics.get("class_names", [])
    n = len(class_names)
    p = np.array(metrics["precision_per_class"])
    r = np.array(metrics["recall_per_class"])
    f1 = np.array(metrics["f1_per_class"])
    x = np.arange(n); w = 0.25
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.bar(x - w, p, w, label="Precision", color="#2E86AB", edgecolor="black")
    ax.bar(x, r, w, label="Recall", color="#A23B72", edgecolor="black")
    ax.bar(x + w, f1, w, label="F1", color="#F18F01", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


# ================================================================
#  CHECKPOINTS
# ================================================================

def save_checkpoint(state: Dict[str, Any], save_path: str) -> None:
    """Save checkpoint with metadata."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state["timestamp"] = pd.Timestamp.now().isoformat()
    torch.save(state, save_path)


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                    optimizer=None, scheduler=None, device=None) -> Dict[str, Any]:
    """Load checkpoint and restore states."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"[Checkpoint] Loaded: {checkpoint_path}")
    return ckpt