# src/pipeLine/densenetFineTune.py
# -*- coding: utf-8 -*-

"""
DenseNet Fine-Tune Pipeline (ULTIMATE LIGHT EDITION)
======================================================
Mirrors vgg16FineTune.py / mobilenetv2FineTune.py exactly, adapted for DenseNet.

⚠️ LIGHT CONFIG for RTX 2050 (4GB VRAM):
   - batch_size = 8
   - gradient_accumulation = 4 (effective batch = 32)
   - image_size = 224
   - epochs = 40

Flow:
1. Load best SSL checkpoint from weightTransformer/
2. Fine-Tune on labeled data (80/10/10 split)
3. Focal Loss + Class Weights
4. 5-Fold CV
5. Bootstrap CI 95%
6. TTA (16 augmentations)
7. 24-page PDF report

Author: Babak Yousefian
Version: 1.0.0 (LIGHT)
"""

import copy
import gc
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, auc, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# ✅ Project Imports
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================================================================
# ✅ Paths (ABSOLUTE)
# ================================================================

SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")
DENSENET_ROOT = os.path.join(SELF_SUPERVISED_ROOT, "", "")
DENSENET_SSL_DIR = os.path.join(DENSENET_ROOT, "SSL_checkpoints")
DENSENET_ARCHITECTURE_DIR = os.path.join(DENSENET_SSL_DIR, "architecture")

# ✅ FIX: weightTransformer is at DenseNet/ level (not DenseNet/densenet/)
DENSENET_WEIGHT_TRANSFORMER_DIR = os.path.join(
    SELF_SUPERVISED_ROOT, "", "weightTransformer"
)

# ✅ Reporter root
REPORTER_ROOT = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "densenet_finetune")
FINAL_MODEL_PATH = os.path.join(REPORTER_ROOT, "densenet_finetuned_final.pth")
CV_DIR = os.path.join(REPORTER_ROOT, "../../CV")
INTERP_DIR = os.path.join(REPORTER_ROOT, "interpretability")
PLOTS_DIR = os.path.join(REPORTER_ROOT, "plots")
ARCH_DIR = os.path.join(REPORTER_ROOT, "architecture")

# ✅ Labels
LABELS_CSV = os.path.join(PROJECT_ROOT, "finalReports", "labels.csv")
CROPPED_DIR = os.path.join(PROJECT_ROOT, "finalReports", "cropped")

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

FEATURE_DIM = 1024  # DenseNet121

# ✅ LIGHT CONFIG defaults
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 8      # ✅ DenseNet121 LIGHT
DEFAULT_GRAD_ACC = 4        # ✅ effective batch = 32
DEFAULT_EPOCHS = 30
DEFAULT_EPOCHS_CV = 15
DEFAULT_TTA = 16
DEFAULT_BOOTSTRAP = 1000


# ================================================================
# ✅ Directory Management
# ================================================================

def ensure_directories():
    for d in [REPORTER_ROOT, CV_DIR, INTERP_DIR, PLOTS_DIR, ARCH_DIR]:
        os.makedirs(d, exist_ok=True)


# ================================================================
# ✅ Check transferred checkpoint
# ================================================================

def check_transferred_model() -> Tuple[bool, Optional[str]]:
    """Find most recent transferred DenseNet checkpoint in weightTransformer/."""
    import glob

    pattern = os.path.join(
        DENSENET_WEIGHT_TRANSFORMER_DIR,
        "densenet_transferred_from_*.pth",
    )
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    if not candidates:
        return False, None
    return True, candidates[0]


def get_transferred_metadata(path: str) -> Dict[str, Any]:
    """Read metadata from a transferred DenseNet checkpoint."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        def _safe_int(v, default=0):
            try:
                return int(v)
            except (ValueError, TypeError):
                return default

        def _safe_float(v, default=0.0):
            try:
                return float(v)
            except (ValueError, TypeError):
                return default

        return {
            "source_method": ckpt.get("source_method", "unknown"),
            "best_lp_accuracy": _safe_float(ckpt.get("best_lp_accuracy", 0.0)),
            "best_lp_f1": _safe_float(ckpt.get("best_lp_f1", 0.0)),
            "transferred_layers": _safe_int(ckpt.get("transferred_layers", 0)),
            "total_layers": _safe_int(ckpt.get("total_layers", 0)),
            "timestamp": ckpt.get("timestamp", "N/A"),
            "all_methods": ckpt.get("all_methods_lp", {}),
        }
    except Exception as e:
        return {"error": str(e)}


# ================================================================
# ✅ SSL Methods Comparison
# ================================================================

def get_ssl_methods_comparison() -> Dict[str, Any]:
    """Load SSL comparison data for the report."""
    summary_path = os.path.join(DENSENET_WEIGHT_TRANSFORMER_DIR, "best_method_summary.json")

    if not os.path.exists(summary_path):
        return {
            "available": False,
            "best_method": None,
            "all_methods": {},
            "summary_path": summary_path,
        }

    try:
        with open(summary_path, "r") as f:
            data = json.load(f)

        return {
            "available": True,
            "best_method": data.get("best_method"),
            "all_methods": data.get("all_methods", {}),
            "summary_path": summary_path,
        }
    except Exception as e:
        return {
            "available": False,
            "best_method": None,
            "all_methods": {},
            "summary_path": summary_path,
            "error": str(e),
        }


# ================================================================
# ✅ Focal Loss
# ================================================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance with label smoothing."""
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        ce = -(true_dist * log_probs).sum(dim=-1)
        if self.weight is not None:
            ce = ce * self.weight[targets]
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


# ================================================================
# ✅ Labeled Dataset
# ================================================================

class DenseNetLabeledDataset(Dataset):
    """Labeled dataset for DenseNet Fine-Tune."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        target_size: int = 224,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size))

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


# ================================================================
# ✅ Load Labeled Data
# ================================================================

def load_labeled_data(
    labels_csv: str = None,
    image_dir: str = None,
    max_per_class: int = None,
) -> Tuple[List[str], List[int]]:
    """Load labeled data from CSV + image directory."""
    if labels_csv is None:
        labels_csv = LABELS_CSV
    if image_dir is None:
        image_dir = CROPPED_DIR

    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    df = pd.read_csv(labels_csv)
    df = df[df["liverLabel"].isin(CLASS_NAMES)]

    image_paths = []
    labels = []
    class_counts = {c: 0 for c in CLASS_NAMES}

    for _, row in df.iterrows():
        cls = row["liverLabel"]
        if max_per_class is not None and class_counts[cls] >= max_per_class:
            continue

        img_name = row["liverName"]
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            base = os.path.splitext(img_name)[0]
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                test = os.path.join(image_dir, base + ext)
                if os.path.exists(test):
                    img_path = test
                    break
            else:
                continue

        image_paths.append(img_path)
        labels.append(LABEL_MAP[cls])
        class_counts[cls] += 1

    print(f"✅ Loaded {len(image_paths)} images")
    for c in CLASS_NAMES:
        print(f"   {c}: {class_counts[c]}")

    return image_paths, labels


# ================================================================
# ✅ TTA
# ================================================================

def predict_with_tta(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_aug: int = 16,  # ✅ LIGHT
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with Test-Time Augmentation."""
    import torchvision.transforms.functional as TF

    model.eval()
    all_probs = []
    all_labels = []

    def apply_tta(img_tensor, aug_idx):
        if aug_idx == 0:
            return img_tensor
        elif aug_idx == 1:
            return TF.hflip(img_tensor)
        elif aug_idx == 2:
            return TF.vflip(img_tensor)
        elif aug_idx == 3:
            return TF.rotate(img_tensor, 90)
        elif aug_idx == 4:
            return TF.rotate(img_tensor, 180)
        elif aug_idx == 5:
            return TF.rotate(img_tensor, 270)
        elif aug_idx == 6:
            return TF.adjust_brightness(img_tensor, 1.1)
        elif aug_idx == 7:
            return TF.adjust_brightness(img_tensor, 0.9)
        elif aug_idx == 8:
            return TF.adjust_contrast(img_tensor, 1.1)
        elif aug_idx == 9:
            return TF.adjust_contrast(img_tensor, 0.9)
        elif aug_idx == 10:
            return torch.flip(img_tensor, dims=[2])
        elif aug_idx == 11:
            return torch.rot90(img_tensor, 1, dims=[1, 2])
        else:
            return img_tensor

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="TTA", leave=False):
            images = images.to(device)
            batch_size = images.size(0)
            batch_probs = torch.zeros(batch_size, NUM_CLASSES, device=device)

            for aug_idx in range(num_aug):
                aug_batch = torch.stack([
                    apply_tta(images[i], aug_idx % 12)
                    for i in range(batch_size)
                ]).to(device)

                logits = model(aug_batch)
                probs = F.softmax(logits.float(), dim=1)
                batch_probs += probs

            batch_probs /= num_aug
            all_probs.append(batch_probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


# ================================================================
# ✅ Metrics
# ================================================================

def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Compute comprehensive metrics."""
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced = float(balanced_accuracy_score(y_true, y_pred))
    precision_w = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_w = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        mask = y_true == i
        if mask.sum() > 0:
            per_class[cname] = {
                "recall": float((y_pred[mask] == i).mean()),
                "support": int(mask.sum()),
            }

    roc_auc = {}
    try:
        y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
        for i in range(NUM_CLASSES):
            if y_bin[:, i].sum() > 0 and (1 - y_bin[:, i]).sum() > 0:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc[i] = float(auc(fpr, tpr))
    except Exception:
        pass

    roc_auc_mean = float(np.mean(list(roc_auc.values()))) if roc_auc else 0.0

    class_report = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision_w,
        "recall": recall_w,
        "f1_score": f1_w,
        "f1_macro": f1_m,
        "roc_auc": roc_auc,
        "roc_auc_mean": roc_auc_mean,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "per_class": per_class,
        "num_samples": int(len(y_true)),
    }


# ================================================================
# ✅ Bootstrap CI
# ================================================================

def bootstrap_ci95(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Compute 95% confidence intervals via bootstrap."""
    np.random.seed(42)
    n_samples = len(y_true)

    accs, f1s, bal_accs = [], [], []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]

        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="weighted", zero_division=0))
        bal_accs.append(balanced_accuracy_score(yt, yp))

    alpha = (1 - confidence) / 2 * 100

    def ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci95_low": float(np.percentile(arr, alpha)),
            "ci95_high": float(np.percentile(arr, 100 - alpha)),
        }

    return {
        "accuracy": ci(accs),
        "f1_score": ci(f1s),
        "balanced_accuracy": ci(bal_accs),
    }


# ================================================================
# ✅ Train One Fold
# ================================================================

def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    patience: int = 10,
    verbose: bool = True,
    train_labels: Optional[List[int]] = None,
    grad_accum: int = 4,
) -> Dict[str, Any]:
    """Train one fold with early stopping + Focal Loss + class weights + grad accum."""
    model.train()

    # ✅ Backbone: lower LR, classifier: full LR
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": classifier_params, "lr": lr},
        ],
        weight_decay=5e-3,   # ✅ افزایش از 2e-3 به 5e-3
    )

    # ✅ BALANCED class weights — بدون ×4
    if train_labels is not None:
        train_class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        # استفاده از inverse sqrt (ملایم‌تر از inverse)
        class_weights_loss = 1.0 / np.sqrt(train_class_counts + 1e-8)
        class_weights_loss = class_weights_loss / class_weights_loss.sum() * NUM_CLASSES
        class_weights_tensor = torch.tensor(
            class_weights_loss, dtype=torch.float32
        ).to(device)
        if verbose:
            print(f"   📊 Loss class weights (sqrt, balanced): {class_weights_tensor.cpu().numpy().round(3)}")
    else:
        class_weights_tensor = None

    # ✅ Focal Loss با gamma=1.0 (نه 2.0)
    criterion = FocalLoss(alpha=1.0, gamma=0.5, weight=class_weights_tensor, label_smoothing=0.1)

    # ✅ Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels) / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            n_batches += 1

        if n_batches % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss = total_loss / max(n_batches, 1)

        # ✅ Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        per_class_correct = np.zeros(NUM_CLASSES)
        per_class_total = np.zeros(NUM_CLASSES)

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for c in range(NUM_CLASSES):
                    mask = labels == c
                    per_class_correct[c] += (preds[mask] == c).sum().item()
                    per_class_total[c] += mask.sum().item()

        val_loss /= max(len(val_loader), 1)
        val_acc = correct / max(total, 1)
        per_class_recall = per_class_correct / np.maximum(per_class_total, 1)
        balanced_val_acc = float(np.mean(per_class_recall))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"   Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Bal Acc: {balanced_val_acc:.4f}")

        # ✅ انتخاب بهترین بر اساس balanced accuracy (نه accuracy)
        if balanced_val_acc > best_val_acc:
            best_val_acc = balanced_val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"   ⏹️ Early stopping at epoch {epoch+1} (best bal acc = {best_val_acc:.4f})")
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_acc": best_val_acc,
        "history": history,
        "best_state": best_state,
    }

# ================================================================
# ✅ MAIN: DenseNet Fine-Tune Pipeline
# ================================================================

def run_densenet_finetune_pipeline(
    ssl_checkpoint_path: str,
    ssl_method: str = "byol",
    epochs_final: int = DEFAULT_EPOCHS,
    epochs_cv: int = DEFAULT_EPOCHS_CV,
    lr_final: float = 1e-4,
    lr_cv: float = 1e-4,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience_final: int = 15,
    patience_cv: int = 8,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    n_tta: int = DEFAULT_TTA,
    cv_folds: int = 5,
    input_size: int = DEFAULT_IMAGE_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRAD_ACC,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete DenseNet Fine-Tune Pipeline — LIGHT EDITION.
    """
    ensure_directories()

    if verbose:
        print("\n" + "=" * 80)
        print("🚀 DENSENET FINE-TUNE PIPELINE (LIGHT EDITION)")
        print("=" * 80)
        print(f"   SSL Checkpoint: {ssl_checkpoint_path}")
        print(f"   SSL Method: {ssl_method.upper()}")
        print(f"   FT Epochs: {epochs_final} (final) / {epochs_cv} (CV)")
        print(f"   Batch Size: {batch_size}")
        print(f"   Grad Accum: {gradient_accumulation_steps}")
        print(f"   Effective Batch: {batch_size * gradient_accumulation_steps}")
        print(f"   Image Size: {input_size}")
        print(f"   CV Folds: {cv_folds}")
        print(f"   TTA: {n_tta}")
        print(f"   Bootstrap: {n_bootstrap}")
        print(f"   Device: {device}")
        print("=" * 80)

    # ============================================================
    # ✅ Load Data
    # ============================================================
    if verbose:
        print("\n📥 Loading labeled data...")

    image_paths, labels = load_labeled_data()

    # ✅ 80/10/10 split
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=0.10, random_state=42, stratify=labels,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=1/9, random_state=42, stratify=train_val_labels,
    )

    if verbose:
        print(f"   Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # ============================================================
    # ✅ Build Model and Load SSL weights
    # ============================================================
    if verbose:
        print("\n🔧 Building DenseNet model...")

    # ✅ Build DenseNet with growth_rate=24 to match SSL checkpoint architecture
    from src.models.DenseNet.DenseNet import DenseNet
    model = DenseNet(
        num_classes=NUM_CLASSES,
        input_channels=3,
        input_size=input_size,
        layers=[6, 12, 24, 16],
        growth_rate=24,
        bn_size=4,
        compression=0.5,
        init_channels=32,
        dropout=0.2,
        verbose=False,
    )

    ckpt = torch.load(ssl_checkpoint_path, map_location="cpu", weights_only=False)
    if "encoder_state_dict" in ckpt:
        state = ckpt["encoder_state_dict"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    # ✅ Shape-aware loading
    model_state = model.state_dict()
    new_state = {}
    loaded = 0
    shape_mismatch = 0
    missing = 0

    for name, param in model_state.items():
        if name in state and state[name].shape == param.shape:
            new_state[name] = state[name]
            loaded += 1
        elif name in state:
            new_state[name] = param
            shape_mismatch += 1
        else:
            new_state[name] = param
            missing += 1

    model.load_state_dict(new_state, strict=False)
    if verbose:
        print(f"   ✅ Loaded SSL weights: {loaded} loaded, {shape_mismatch} shape mismatch, {missing} missing")
        print(f"   Total model params: {len(model_state)}")

    model = model.to(device)

    # ============================================================
    # ✅ Transforms
    # ============================================================
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ============================================================
    # ✅ Phase 1: Fine-Tune on Train/Val
    # ============================================================
    if verbose:
        print("\n" + "=" * 80)
        print("🧠 Phase 1: Fine-Tune on train/val split")
        print("=" * 80)

    train_dataset = DenseNetLabeledDataset(train_paths, train_labels, train_transform, input_size)
    val_dataset = DenseNetLabeledDataset(val_paths, val_labels, val_transform, input_size)

    train_class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.sqrt(train_class_counts + 1e-8)
    sample_weights = np.array([class_weights[y] for y in train_labels])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    ft_result = train_one_fold(
        model, train_loader, val_loader,
        epochs=epochs_final, lr=lr_final, device=device,
        patience=patience_final, verbose=verbose,
        train_labels=train_labels,
        grad_accum=gradient_accumulation_steps,
    )

    if verbose:
        print(f"\n✅ Fine-Tune best val acc: {ft_result['best_val_acc']:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_acc": ft_result["best_val_acc"],
            "ssl_method": ssl_method,
            "ssl_checkpoint": ssl_checkpoint_path,
            "history": ft_result["history"],
            "timestamp": datetime.now().isoformat(),
        },
        FINAL_MODEL_PATH,
    )
    if verbose:
        print(f"✅ Final model saved: {FINAL_MODEL_PATH}")

    # ============================================================
    # ✅ Phase 2: Test Set Evaluation + TTA
    # ============================================================
    if verbose:
        print("\n" + "=" * 80)
        print("📊 Phase 2: Test set evaluation with TTA")
        print("=" * 80)

    test_dataset = DenseNetLabeledDataset(test_paths, test_labels, val_transform, input_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    test_probs, test_labels_arr = predict_with_tta(model, test_loader, device, n_tta)
    test_preds = test_probs.argmax(axis=1)

    test_metrics = compute_full_metrics(test_labels_arr, test_preds, test_probs)

    if verbose:
        print(f"\n📊 Test Metrics (with TTA):")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"   F1 (weighted): {test_metrics['f1_score']:.4f}")
        print(f"   F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"   AUC: {test_metrics['roc_auc_mean']:.4f}")
        for cname, m in test_metrics["per_class"].items():
            print(f"   {cname} recall: {m['recall']:.4f}")

    # ============================================================
    # ✅ Phase 3: 5-Fold CV
    # ============================================================
    if verbose:
        print("\n" + "=" * 80)
        print(f"🔬 Phase 3: {cv_folds}-Fold Cross-Validation")
        print("=" * 80)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(image_paths, labels)):
        if verbose:
            print(f"\n📌 Fold {fold_idx + 1}/{cv_folds}")

        tr_paths = [image_paths[i] for i in tr_idx]
        tr_lbls = [labels[i] for i in tr_idx]
        te_paths = [image_paths[i] for i in te_idx]
        te_lbls = [labels[i] for i in te_idx]

        # ✅ Build fresh DenseNet with growth_rate=24 for each fold
        from src.models.DenseNet.DenseNet import DenseNet
        fold_model = DenseNet(
            num_classes=NUM_CLASSES,
            input_channels=3,
            input_size=input_size,
            layers=[6, 12, 24, 16],
            growth_rate=24,
            bn_size=4,
            compression=0.5,
            init_channels=32,
            dropout=0.2,
            verbose=False,
        )

        # Shape-aware load
        fold_state = fold_model.state_dict()
        fold_new_state = {}
        for name, param in fold_state.items():
            if name in state and state[name].shape == param.shape:
                fold_new_state[name] = state[name]
            else:
                fold_new_state[name] = param
        fold_model.load_state_dict(fold_new_state, strict=False)
        fold_model = fold_model.to(device)

        tr_ds = DenseNetLabeledDataset(tr_paths, tr_lbls, train_transform, input_size)
        te_ds = DenseNetLabeledDataset(te_paths, te_lbls, val_transform, input_size)

        tr_counts = np.bincount(tr_lbls, minlength=NUM_CLASSES)
        cw = 1.0 / np.sqrt(tr_counts + 1e-8)
        sw = np.array([cw[y] for y in tr_lbls])
        fold_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sw).float(),
            num_samples=len(sw), replacement=True,
        )

        tr_ld = DataLoader(tr_ds, batch_size=batch_size, sampler=fold_sampler,
                           num_workers=0, pin_memory=True, drop_last=True)
        te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)

        fold_result = train_one_fold(
            fold_model, tr_ld, te_ld,
            epochs=epochs_cv, lr=lr_cv, device=device,
            patience=patience_cv, verbose=False,
            train_labels=tr_lbls,
            grad_accum=gradient_accumulation_steps,
        )

        fold_probs, fold_lbls = predict_with_tta(fold_model, te_ld, device, n_tta)
        fold_preds = fold_probs.argmax(axis=1)
        fold_acc = accuracy_score(fold_lbls, fold_preds)
        fold_f1 = f1_score(fold_lbls, fold_preds, average="weighted", zero_division=0)
        fold_bal = balanced_accuracy_score(fold_lbls, fold_preds)

        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": float(fold_acc),
            "f1_weighted": float(fold_f1),
            "balanced_accuracy": float(fold_bal),
            "best_val_acc": float(fold_result["best_val_acc"]),
            "history": fold_result["history"],
        })

        if verbose:
            print(f"   ✅ Fold {fold_idx+1}: Acc={fold_acc:.4f} | F1={fold_f1:.4f} | BalAcc={fold_bal:.4f}")

    cv_summary = {
        "mean_accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
        "std_accuracy": float(np.std([r["accuracy"] for r in fold_results])),
        "mean_f1": float(np.mean([r["f1_weighted"] for r in fold_results])),
        "std_f1": float(np.std([r["f1_weighted"] for r in fold_results])),
        "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in fold_results])),
        "std_balanced_accuracy": float(np.std([r["balanced_accuracy"] for r in fold_results])),
        "fold_results": fold_results,
    }

    if verbose:
        print(f"\n📊 CV Summary:")
        print(f"   Mean Accuracy: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        print(f"   Mean F1: {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        print(f"   Mean Bal Acc: {cv_summary['mean_balanced_accuracy']:.4f} ± {cv_summary['std_balanced_accuracy']:.4f}")

    # ============================================================
    # ✅ Phase 4: Bootstrap CI 95%
    # ============================================================
    if verbose:
        print("\n" + "=" * 80)
        print(f"🎲 Phase 4: Bootstrap 95% CI (n={n_bootstrap})")
        print("=" * 80)

    ci95 = bootstrap_ci95(test_labels_arr, test_preds, n_bootstrap=n_bootstrap)

    if verbose:
        print(f"\n📊 CI 95%:")
        print(f"   Accuracy: [{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]")
        print(f"   F1: [{ci95['f1_score']['ci95_low']:.4f}, {ci95['f1_score']['ci95_high']:.4f}]")
        print(f"   Bal Acc: [{ci95['balanced_accuracy']['ci95_low']:.4f}, {ci95['balanced_accuracy']['ci95_high']:.4f}]")

    # ============================================================
    # ✅ Save metrics
    # ============================================================
    metrics_json = {
        "ssl_method": ssl_method,
        "ssl_checkpoint": ssl_checkpoint_path,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs_final": epochs_final,
            "epochs_cv": epochs_cv,
            "lr_final": lr_final,
            "lr_cv": lr_cv,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "n_tta": n_tta,
            "cv_folds": cv_folds,
            "n_bootstrap": n_bootstrap,
            "input_size": input_size,
        },
        "data_split": {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
            "total": len(image_paths),
        },
        "final_metrics": test_metrics,
        "cv_summary": cv_summary,
        "ci95": ci95,
        "history": ft_result["history"],
    }

    metrics_path = os.path.join(REPORTER_ROOT, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=4, default=str)
    if verbose:
        print(f"\n✅ Metrics saved: {metrics_path}")

    return {
        "status": "success",
        "final_model": FINAL_MODEL_PATH,
        "final_model_path": FINAL_MODEL_PATH,
        "metrics": test_metrics,
        "final_metrics": test_metrics,
        "cv_summary": cv_summary,
        "ci95": ci95,
        "history": ft_result["history"],
        "ssl_method": ssl_method,
        "ssl_checkpoint": ssl_checkpoint_path,
        "split_info": {
            "test_paths": test_paths,
            "test_labels": test_labels,
            "train_paths": train_paths,
            "val_paths": val_paths,
        },
        "test_paths": test_paths,
        "test_labels": test_labels,
        "metrics_json": metrics_path,
        "num_train": len(train_paths),
        "num_val": len(val_paths),
        "num_test": len(test_paths),
    }