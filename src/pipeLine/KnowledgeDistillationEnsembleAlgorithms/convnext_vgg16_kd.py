# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/convnext_vgg16_kd.py
# -*- coding: utf-8 -*-

"""
ConvNeXt (Teacher) → VGG16 (Student) Knowledge Distillation
================================================================
- Teacher: ConvNeXt-Tiny with custom head (matches bestKDE.pth)
- Student: VGG16 with custom head (matches vgg16BEST.pth)
- Full KD training loop with:
  * KL Divergence + Cross Entropy loss
  * Weighted Random Sampler (balanced classes)
  * Gradient Clipping
  * Early Stopping (on Val Loss with patience)
  * Cosine Annealing Scheduler
- Advanced: Real CV5, Real CI95 (Bootstrap), Clean TTA32
- G1-targeted fine-tune (post-KD) for FP < 7%
- Generates full 24-page report using kd_reporter.py
"""

import os
import sys
import json
import time
import copy
import warnings
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report,
    f1_score
)
from sklearn.utils import resample

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Safe imports ────────────────────────────────────────────────────
try:
    from src.AI.ResNet.utils import CLASS_NAMES
except ImportError:
    CLASS_NAMES = ["NL", "G1", "G2", "G3"]

try:
    from src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.kd_reporter import (
        generate_kd_report,
    )
except ImportError:
    def generate_kd_report(*args, **kwargs):
        print("⚠️ kd_reporter not found. Skipping PDF generation.")
        return None

# ── Rich UI ─────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    )
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# ── Config ──────────────────────────────────────────────────────────
CONVNEXT_TEACHER_PATH = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\bestKDEnsemble\bestKDE.pth"
VGG16_STUDENT_PATH = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\VGG16\vgg16BEST.pth"
IMAGES_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\cropped"
LABELS_CSV = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\labels.csv"
OUTPUT_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\bestKDEnsembles\VGG16"

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH = BATCH_SIZE * ACCUMULATION_STEPS
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
LR_BACKBONE = 5e-6
LR_HEAD = 3e-4
WEIGHT_DECAY = 0.1
TEMPERATURE = 4.0
ALPHA = 0.7
PATIENCE = 9
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
SEED = 42

# CV5 hyperparameters
CV5_EPOCHS = 10
CV5_LR = 1e-5
CV5_PATIENCE = 3

# G1 fine-tune hyperparameters
G1_FT_EPOCHS = 5
G1_FT_LR = 1e-5
G1_FT_PATIENCE = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═════════════════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════════════════

class LiverDataset(torch.utils.data.Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["liverName"]
        label = CLASS_NAMES.index(row["liverLabel"])
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label, img_name


# ═════════════════════════════════════════════════════════════════════
# MODELS (exact match to checkpoints)
# ═════════════════════════════════════════════════════════════════════

def get_convnext_teacher():
    """
    Build ConvNeXt-Tiny with the EXACT custom head from bestKDE.pth.
    Keys must match: classifier.0, classifier.2.1, classifier.2.2, classifier.2.5
    Forward: flatten -> classifier.0 (LayerNorm) -> classifier.2.*
    """
    from torchvision.models import convnext_tiny
    from collections import OrderedDict
    import torch.nn as nn

    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features  # 768

    # Build custom classifier with exact key names
    # Checkpoint keys:
    #   classifier.0.weight (768,)      -> LayerNorm
    #   classifier.0.bias   (768,)
    #   classifier.2.1.weight (256, 768) -> Linear
    #   classifier.2.1.bias   (256,)
    #   classifier.2.2.weight (256,)     -> LayerNorm
    #   classifier.2.2.bias   (256,)
    #   classifier.2.5.weight (4, 256)   -> Linear
    #   classifier.2.5.bias   (4,)

    custom_classifier = nn.Sequential(OrderedDict([
        ("0", nn.LayerNorm(in_features, eps=1e-6)),  # classifier.0
        ("2", nn.Sequential(OrderedDict([
            ("1", nn.Linear(in_features, 256)),   # classifier.2.1
            ("2", nn.LayerNorm(256, eps=1e-6)),    # classifier.2.2
            ("3", nn.GELU()),                       # classifier.2.3
            ("4", nn.Dropout(0.2)),                 # classifier.2.4
            ("5", nn.Linear(256, len(CLASS_NAMES))), # classifier.2.5
        ]))),
    ]))

    # We need to handle the forward manually because we need to Flatten BEFORE LayerNorm
    class ConvNeXtWithCustomHead(nn.Module):
        def __init__(self, base_model, classifier):
            super().__init__()
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            self.classifier = classifier

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)           # (B, 768, 1, 1)
            x = torch.flatten(x, 1)       # (B, 768)  ← Flatten FIRST
            x = self.classifier(x)        # LayerNorm -> Linear -> ...
            return x

    return ConvNeXtWithCustomHead(model, custom_classifier)


def get_vgg16_student():
    """VGG16 with custom head (matches vgg16BEST.pth)."""
    from torchvision.models import vgg16
    model = vgg16(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASS_NAMES)),
    )
    return model


# ═════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════

def make_loader(ds, shuffle=True, sampler=None, batch_size=BATCH_SIZE, drop_last=False):
    if sampler:
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=False, drop_last=drop_last)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False, drop_last=drop_last)


def get_sampler(df):
    labels = [CLASS_NAMES.index(l) for l in df["liverLabel"]]
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                  replacement=True)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            with autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    avg_loss = total_loss / len(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, s = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CLASS_NAMES)))
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc_macro = 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "y_true": all_labels,
        "y_pred": all_preds,
        "y_prob": all_probs,
        "precision_per_class": p.tolist(),
        "recall_per_class": r.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": s.tolist(),
        "confusion_matrix": cm,
        "auc_macro": auc_macro,
        "weighted_avg": {
            "precision": float(np.average(p, weights=s)),
            "recall": float(np.average(r, weights=s)),
            "f1-score": float(np.average(f1, weights=s)),
        },
    }


# ═════════════════════════════════════════════════════════════════════
# PLOT: TRAIN/VAL GAP
# ═════════════════════════════════════════════════════════════════════

def plot_train_val_gap(history, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=200)

    epochs = range(1, len(history["train_loss"]) + 1)
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    gap = np.array(val_loss) - np.array(train_loss)

    axes[0].plot(epochs, train_loss, "o-", label="Train Loss",
                 color="#3498db", lw=2, ms=4)
    axes[0].plot(epochs, val_loss, "s-", label="Val Loss",
                 color="#e74c3c", lw=2, ms=4)
    axes[0].fill_between(epochs, train_loss, val_loss, alpha=0.15, color="#f39c12")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Loss", fontweight="bold")
    axes[0].set_title("Train vs Validation Loss — Gap Analysis",
                      fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3, ls="--")

    stats_text = (f"Gap Statistics:\n"
                  f"Max:  {gap.max():+.4f}\n"
                  f"Min:  {gap.min():+.4f}\n"
                  f"Avg:  {gap.mean():+.4f}\n"
                  f"Final: {gap[-1]:+.4f}")
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                 fontsize=9, va="top", bbox=dict(boxstyle="round,pad=0.3",
                 facecolor="#fff9e6", edgecolor="#f39c12"))

    colors = ["#e74c3c" if g > 0 else "#2ecc71" for g in gap]
    axes[1].bar(epochs, gap, color=colors, edgecolor="black", alpha=0.7)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].axhline(gap.mean(), color="#f39c12", ls="--", lw=2,
                    label=f"Avg Gap = {gap.mean():+.4f}")
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Gap (Val - Train)", fontweight="bold")
    axes[1].set_title("Overfitting / Underfitting Indicator",
                      fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3, ls="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Gap plot saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════
# REAL CV5 (retrain on each fold)
# ═════════════════════════════════════════════════════════════════════

def run_real_cv5(student, df, train_transform, val_transform):
    """
    Real 5-Fold CV: for each fold, clone model, fine-tune on train_fold,
    evaluate on val_fold. Returns per-fold metrics.
    """
    print(f"\n   🔄 Running REAL CV5 ({CV5_EPOCHS} epochs per fold)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_accs, cv_aucs = [], []
    fold_details = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["liverLabel"])):
        fold_start = time.time()
        tr_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        tr_ds = LiverDataset(tr_df, IMAGES_DIR, train_transform)
        val_ds = LiverDataset(val_df, IMAGES_DIR, val_transform)
        tr_sampler = get_sampler(tr_df)
        tr_loader = make_loader(tr_ds, sampler=tr_sampler)
        val_loader = make_loader(val_ds, shuffle=False)

        # Clone student model
        fold_model = get_vgg16_student().to(DEVICE)
        fold_model.load_state_dict(student.state_dict())

        # Optimizer (small LR for fine-tune)
        optimizer = AdamW(fold_model.parameters(), lr=CV5_LR, weight_decay=WEIGHT_DECAY)
        ce_loss = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, CV5_EPOCHS + 1):
            fold_model.train()
            for batch in tr_loader:
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                with autocast(enabled=(DEVICE == "cuda")):
                    logits = fold_model(x)
                    loss = ce_loss(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), GRAD_CLIP)
                optimizer.step()

            # Validate
            val_m = evaluate(fold_model, val_loader, ce_loss, DEVICE)
            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                best_state = copy.deepcopy(fold_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= CV5_PATIENCE:
                    break

        fold_model.load_state_dict(best_state)
        final_m = evaluate(fold_model, val_loader, ce_loss, DEVICE)
        cv_accs.append(final_m["accuracy"])
        cv_aucs.append(final_m["auc_macro"])

        elapsed = time.time() - fold_start
        print(f"     Fold {fold_idx + 1}/5 | Acc: {final_m['accuracy']:.4f} | "
              f"AUC: {final_m['auc_macro']:.4f} | Time: {elapsed:.1f}s")

        fold_details.append({
            "fold": fold_idx + 1,
            "acc": float(final_m["accuracy"]),
            "auc": float(final_m["auc_macro"]),
        })

    return {
        "folds": fold_details,
        "mean_acc": float(np.mean(cv_accs)),
        "std_acc": float(np.std(cv_accs)),
        "mean_auc": float(np.mean(cv_aucs)),
        "std_auc": float(np.std(cv_aucs)),
    }


# ═════════════════════════════════════════════════════════════════════
# REAL CI95 (Bootstrap on actual predictions)
# ═════════════════════════════════════════════════════════════════════

def run_real_ci95(y_true, y_prob, n_bootstrap=1000):
    """Bootstrap 95% CI on actual y_true and y_pred."""
    print(f"\n   🎲 Running REAL CI95 (Bootstrap {n_bootstrap} on predictions)...")
    rng = np.random.RandomState(SEED)
    n = len(y_true)
    boot_accs, boot_aucs = [], []

    y_pred_base = np.argmax(y_prob, axis=1)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t = y_true[idx]
        y_p = y_pred_base[idx]
        y_pr = y_prob[idx]
        boot_accs.append(accuracy_score(y_t, y_p))
        try:
            boot_aucs.append(roc_auc_score(y_t, y_pr, multi_class="ovr", average="macro"))
        except Exception:
            boot_aucs.append(0.0)

    return {
        "ci": 95,
        "n_bootstrap": n_bootstrap,
        "acc": {
            "mean": float(np.mean(boot_accs)),
            "lower": float(np.percentile(boot_accs, 2.5)),
            "upper": float(np.percentile(boot_accs, 97.5)),
        },
        "auc": {
            "mean": float(np.mean(boot_aucs)),
            "lower": float(np.percentile(boot_aucs, 2.5)),
            "upper": float(np.percentile(boot_aucs, 97.5)),
        },
    }


# ═════════════════════════════════════════════════════════════════════
# REAL TTA-32 (clean)
# ═════════════════════════════════════════════════════════════════════

def run_real_tta32(model, val_df, n_aug=32):
    """TTA with n_aug augmentations, averaged probabilities."""
    print(f"\n   🔄 Running REAL TTA-{n_aug}...")
    tta_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tta_ds = LiverDataset(val_df, IMAGES_DIR, tta_transform)
    tta_loader = make_loader(tta_ds, shuffle=False)

    # Get all_labels once (from first pass)
    all_labels = []
    for batch in tta_loader:
        if len(batch) == 3:
            _, y, _ = batch
        else:
            _, y = batch
        all_labels.extend(y.cpu().numpy())
    all_labels = np.array(all_labels)

    # Accumulate probs over n_aug passes
    probs_sum = np.zeros((len(all_labels), len(CLASS_NAMES)))
    model.eval()
    with torch.no_grad():
        for aug_idx in range(n_aug):
            offset = 0
            for batch in tta_loader:
                if len(batch) == 3:
                    x, _, _ = batch
                else:
                    x, _ = batch
                x = x.to(DEVICE)
                logits = model(x)
                probs = F.softmax(logits.float(), dim=1).cpu().numpy()
                probs_sum[offset:offset + len(probs)] += probs
                offset += len(probs)
    avg_probs = probs_sum / n_aug

    tta_preds = np.argmax(avg_probs, axis=1)
    tta_acc = accuracy_score(all_labels, tta_preds)
    try:
        tta_auc = roc_auc_score(all_labels, avg_probs, multi_class="ovr", average="macro")
    except Exception:
        tta_auc = 0.0

    return {
        "n_aug": n_aug,
        "acc": float(tta_acc),
        "auc": float(tta_auc),
    }


# ═════════════════════════════════════════════════════════════════════
# G1-TARGETED FINE-TUNE (post-KD)
# ═════════════════════════════════════════════════════════════════════

def run_g1_finetune(student, train_df, val_df, train_transform, val_transform):
    """
    Fine-tune student on a subset focused on G1 vs NL boundary.
    Goal: reduce G1 false positives below 7%.
    """
    print(f"\n   🎯 Running G1-targeted fine-tune ({G1_FT_EPOCHS} epochs)...")

    # Filter to only NL and G1 samples
    g1_nl_df = train_df[train_df["liverLabel"].isin(["NL", "G1"])].reset_index(drop=True)
    val_g1_nl_df = val_df[val_df["liverLabel"].isin(["NL", "G1"])].reset_index(drop=True)

    if len(g1_nl_df) < 50:
        print("   ⚠️ Not enough NL/G1 samples. Skipping G1 fine-tune.")
        return student

    print(f"      G1/NL train samples: {len(g1_nl_df)}")
    print(f"      G1/NL val samples: {len(val_g1_nl_df)}")

    # Class weights: heavily weight G1 to reduce false positives
    class_weights = torch.tensor([1.0, 3.0, 0.0, 0.0], dtype=torch.float32).to(DEVICE)

    g1_ft_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tr_ds = LiverDataset(g1_nl_df, IMAGES_DIR, g1_ft_transform)
    val_ds = LiverDataset(val_g1_nl_df, IMAGES_DIR, val_transform)
    tr_sampler = get_sampler(g1_nl_df)
    tr_loader = make_loader(tr_ds, sampler=tr_sampler, batch_size=8)
    val_loader = make_loader(val_ds, shuffle=False, batch_size=8)

    optimizer = AdamW(student.parameters(), lr=G1_FT_LR, weight_decay=WEIGHT_DECAY)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, G1_FT_EPOCHS + 1):
        student.train()
        for batch in tr_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=(DEVICE == "cuda")):
                logits = student(x)
                loss = ce_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            optimizer.step()

        val_m = evaluate(student, val_loader, nn.CrossEntropyLoss(), DEVICE)
        print(f"      G1-FT Epoch {epoch}/{G1_FT_EPOCHS} | "
              f"Val Loss: {val_m['loss']:.4f} | Val Acc: {val_m['accuracy']:.4f}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = copy.deepcopy(student.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= G1_FT_PATIENCE:
                print(f"      ⏹️ G1-FT early stopping at epoch {epoch}")
                break

    if best_state is not None:
        student.load_state_dict(best_state)
    return student


def print_config_panel(config: dict):
    """Print a beautiful config panel with Rich."""
    if not HAS_RICH:
        print("\n" + "=" * 70)
        print("  ⚙️  TRAINING CONFIGURATION")
        print("=" * 70)
        for k, v in config.items():
            print(f"  {k:<30}: {v}")
        print("=" * 70 + "\n")
        return

    table = Table(
        title="[bold cyan]⚙️  KD Training Configuration[/bold cyan]",
        box=box.ROUNDED,
        border_style="bright_cyan",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Parameter", style="bold yellow", width=28)
    table.add_column("Value", style="bold white", width=40)

    for k, v in config.items():
        table.add_row(str(k), str(v))

    console.print()
    console.print(table)
    console.print()

def make_epoch_table(history: dict, current_epoch: int, total_epochs: int,
                     best_val_acc: float, best_epoch: int,
                     patience_counter: int, patience: int,
                     current_train_loss: float = None,
                     current_val_loss: float = None) -> Table:
    """Build a Rich Table showing all epochs so far."""
    table = Table(
        title=f"[bold cyan]🎯 KD Training — Epoch {current_epoch}/{total_epochs}[/bold cyan]",
        box=box.ROUNDED,
        border_style="bright_cyan",
        header_style="bold magenta",
        show_lines=False,
        expand=False,
    )
    table.add_column("Epoch", style="bold yellow", width=6, justify="center")
    table.add_column("Train Loss", style="cyan", width=10, justify="right")
    table.add_column("Val Loss", style="cyan", width=10, justify="right")
    table.add_column("Val Acc", style="bold green", width=10, justify="right")
    table.add_column("Gap", style="yellow", width=9, justify="right")
    table.add_column("LR", style="dim white", width=10, justify="right")
    table.add_column("Teach Agree", style="magenta", width=12, justify="right")
    table.add_column("Best?", style="bold white", width=8, justify="center")
    table.add_column("Patience", style="red", width=8, justify="center")

    n_epochs = len(history["train_loss"])
    for i in range(n_epochs):
        is_best = (i + 1) == best_epoch
        best_mark = "✅" if is_best else ""
        # Current epoch row (highlight if it's the last one)
        style = "on grey23" if (i + 1) == n_epochs else ""
        pat_str = f"{patience_counter}/{patience}" if (i + 1) == n_epochs else ""
        table.add_row(
            str(i + 1),
            f"{history['train_loss'][i]:.4f}",
            f"{history['val_loss'][i]:.4f}",
            f"{history['val_acc'][i]:.4f}",
            f"{history['gap'][i]:+.4f}",
            f"{history['learning_rates'][i]:.2e}",
            f"{history['teacher_agreement'][i]:.4f}",
            best_mark,
            pat_str,
            style=style,
        )

    return table


def make_progress_table(current_epoch: int, total_epochs: int,
                        best_val_acc: float, best_epoch: int,
                        patience_counter: int, patience: int,
                        current_train_loss: float = None,
                        current_val_loss: float = None) -> Table:
    """Build a compact status table (progress, best, patience)."""
    table = Table(
        box=box.SIMPLE,
        show_header=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Key", style="bold yellow", width=20)
    table.add_column("Value", style="bold white")

    table.add_row("Epoch", f"{current_epoch}/{total_epochs}")
    table.add_row("Best Val Acc", f"[bold green]{best_val_acc:.4f}[/bold green] (epoch {best_epoch})")
    table.add_row("Patience", f"[red]{patience_counter}/{patience}[/red]")
    if current_train_loss is not None:
        table.add_row("Current Train Loss", f"{current_train_loss:.4f}")
    if current_val_loss is not None:
        table.add_row("Current Val Loss", f"{current_val_loss:.4f}")

    return table

# ═════════════════════════════════════════════════════════════════════
# MAIN KD PIPELINE
# ═════════════════════════════════════════════════════════════════════

def run_convnext_vgg16_kd():
    print("\n" + "=" * 70)
    print("  🚀 ConvNeXt (Teacher) → VGG16 (Student) KD Ensemble")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load Data ──────────────────────────────────────────────
    print("\n📂 Loading dataset...")
    df = pd.read_csv(LABELS_CSV)
    print(f"   Total samples: {len(df)}")
    print(f"   Class distribution:\n{df['liverLabel'].value_counts()}")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ✅ 80/10/10 split (stratified)
    # First: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["liverLabel"]
    )
    # Then: split temp 50/50 → 10% val, 10% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["liverLabel"]
    )

    print(f"\n   📊 Split summary:")
    print(f"      Train: {len(train_df)} samples ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"      Val:   {len(val_df)} samples ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"      Test:  {len(test_df)} samples ({len(test_df) / len(df) * 100:.1f}%)")
    print(f"\n   Train class distribution:\n{train_df['liverLabel'].value_counts()}")
    print(f"\n   Val class distribution:\n{val_df['liverLabel'].value_counts()}")
    print(f"\n   Test class distribution:\n{test_df['liverLabel'].value_counts()}")

    train_ds = LiverDataset(train_df, IMAGES_DIR, train_transform)
    val_ds = LiverDataset(val_df, IMAGES_DIR, val_transform)
    test_ds = LiverDataset(test_df, IMAGES_DIR, val_transform)

    train_sampler = get_sampler(train_df)
    train_loader = make_loader(train_ds, sampler=train_sampler, drop_last=True)
    val_loader = make_loader(val_ds, shuffle=False, drop_last=False)
    test_loader = make_loader(test_ds, shuffle=False, drop_last=False)

    # ── 2. Load Models ────────────────────────────────────────────
    print("\n🧠 Loading models...")
    teacher = get_convnext_teacher().to(DEVICE)
    student = get_vgg16_student().to(DEVICE)

    if os.path.exists(CONVNEXT_TEACHER_PATH):
        checkpoint = torch.load(CONVNEXT_TEACHER_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            teacher.load_state_dict(checkpoint["model_state_dict"])
        else:
            teacher.load_state_dict(checkpoint)
        print(f"   ✅ Teacher loaded from {CONVNEXT_TEACHER_PATH}")
    else:
        print(f"   ⚠️ Teacher weights not found")

    if os.path.exists(VGG16_STUDENT_PATH):
        checkpoint = torch.load(VGG16_STUDENT_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            student.load_state_dict(checkpoint["model_state_dict"])
        else:
            student.load_state_dict(checkpoint)
        print(f"   ✅ Student loaded from {VGG16_STUDENT_PATH}")
    else:
        print(f"   ⚠️ Student weights not found")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── 3. Loss & Optimizer ───────────────────────────────────────
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_df["liverLabel"].map(lambda x: CLASS_NAMES.index(x)).values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    kd_loss = nn.KLDivLoss(reduction="batchmean")

    backbone_params = []
    head_params = []
    for name, param in student.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # ── 4. Training Loop ──────────────────────────────────────────
    print("\n🏋️ Starting KD training...")

    # ✅ Print config panel BEFORE training
    config_display = {
        "Teacher": "ConvNeXt (bestKDE.pth)",
        "Student": "VGG16 (vgg16BEST.pth)",
        "Epochs": NUM_EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Effective Batch": EFFECTIVE_BATCH,
        "Accumulation Steps": ACCUMULATION_STEPS,
        "LR (Backbone)": f"{LR_BACKBONE:.2e}",
        "LR (Head)": f"{LR_HEAD:.2e}",
        "Weight Decay": WEIGHT_DECAY,
        "Temperature": TEMPERATURE,
        "Alpha (KD weight)": ALPHA,
        "Label Smoothing": LABEL_SMOOTHING,
        "Grad Clip": GRAD_CLIP,
        "Early Stopping Patience": PATIENCE,
        "Scheduler": "CosineAnnealingWarmRestarts (T_0=10, T_mult=2)",
        "Device": DEVICE,
        "Seed": SEED,
        "Train samples": len(train_df),
        "Val samples": len(val_df),
        "Test samples": len(test_df),
    }
    print_config_panel(config_display)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
        "teacher_agreement": [],
        "gap": [],
    }
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    best_model_path = os.path.join(OUTPUT_DIR, "convnext_to_vgg16_kd_best.pth")

    # ✅ Header for epoch logs
    print("\n" + "=" * 110)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Acc':>8} | "
          f"{'Gap':>8} | {'LR':>10} | {'Teach Agr':>9} | {'Best?':>6} | {'Patience':>8}")
    print("=" * 110)

    for epoch in range(1, NUM_EPOCHS + 1):
        student.train()
        epoch_loss = 0.0
        total = 0

        # Inner tqdm for batches (simple, no Live)
        if HAS_TQDM:
            batch_iter = tqdm(
                train_loader,
                desc=f"  Epoch {epoch:02d}",
                leave=False,
                unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                colour="green",
                ncols=100,
            )
        else:
            batch_iter = train_loader

        for step, batch in enumerate(batch_iter):
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            with autocast(enabled=(DEVICE == "cuda")):
                with torch.no_grad():
                    t_logits = teacher(x)

                s_logits = student(x)
                loss_ce = ce_loss(s_logits, y)

                T = TEMPERATURE
                soft_t = F.softmax(t_logits / T, dim=1)
                log_soft_s = F.log_softmax(s_logits / T, dim=1)
                loss_kd = kd_loss(log_soft_s, soft_t) * (T * T)

                loss = ALPHA * loss_kd + (1 - ALPHA) * loss_ce

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * x.size(0)
            total += x.size(0)

            if HAS_TQDM and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / total

        # ── Validation ──
        val_metrics = evaluate(student, val_loader, nn.CrossEntropyLoss(), DEVICE)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        gap = val_loss - avg_train_loss

        # Teacher agreement
        with torch.no_grad():
            agree = 0
            total_val = 0
            for batch in val_loader:
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                x = x.to(DEVICE)
                s_pred = student(x).argmax(dim=1)
                t_pred = teacher(x).argmax(dim=1)
                agree += (s_pred == t_pred).sum().item()
                total_val += x.size(0)
            teacher_agreement = agree / total_val

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(lr)
        history["teacher_agreement"].append(teacher_agreement)
        history["gap"].append(gap)

        # ── Check best model ──
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(student.state_dict())
            patience_counter = 0
            is_best = True
            torch.save(best_state, best_model_path)
        else:
            patience_counter += 1

        # ✅ Print one clean line per epoch
        best_mark = "✅" if is_best else " "
        print(f"{epoch:>5} | {avg_train_loss:>10.4f} | {val_loss:>8.4f} | {val_acc:>8.4f} | "
              f"{gap:>+8.4f} | {lr:>10.2e} | {teacher_agreement:>9.4f} | "
              f"{best_mark:>6} | {patience_counter:>3}/{PATIENCE}")

        # ✅ If best, print save message on new line
        if is_best:
            print(f"        💾 Best model saved! (Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}) → {best_model_path}")

        scheduler.step()

        # Early stopping
        if patience_counter >= PATIENCE:
            print("\n" + "=" * 110)
            print(f"⏹️  Early stopping at epoch {epoch} (patience={PATIENCE} reached)")
            print(
                f"   Best epoch: {best_epoch} | Best Val Acc: {best_val_acc:.4f} | Best Val Loss: {best_val_loss:.4f}")
            print("=" * 110)
            break

    # Restore best model
    if best_state is not None:
        student.load_state_dict(best_state)
        print(f"\n🏆 Best model restored from epoch {best_epoch} "
              f"(Val Acc: {best_val_acc:.4f})")

    # ── 5. Save Best KD Model ─────────────────────────────────────
    print("\n💾 Saving best KD model...")
    student.load_state_dict(best_state)
    kd_model_path = os.path.join(OUTPUT_DIR, "convnext_to_vgg16_kd_best.pth")
    torch.save(best_state, kd_model_path)
    print(f"   ✅ KD model saved: {kd_model_path}")

    # ── 6. G1-Targeted Fine-Tune ──────────────────────────────────
    print("\n🎯 Post-KD G1-targeted fine-tune...")
    student = run_g1_finetune(student, train_df, val_df, train_transform, val_transform)
    g1_ft_model_path = os.path.join(OUTPUT_DIR, "convnext_to_vgg16_kd_g1ft.pth")
    torch.save(student.state_dict(), g1_ft_model_path)
    print(f"   ✅ G1-FT model saved: {g1_ft_model_path}")

    # ── 7. Final Evaluation on TEST Set ────────────────────────────
    print("\n📊 Final evaluation on TEST set...")
    final_metrics = evaluate(student, test_loader, nn.CrossEntropyLoss(), DEVICE)

    # ── 8. Save classification_report.txt ────────────────────────
    report_txt = classification_report(
        final_metrics["y_true"], final_metrics["y_pred"],
        target_names=CLASS_NAMES, digits=4
    )
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("VGG16 (KD + G1-FT) — Classification Report (Val Set)\n")
        f.write("=" * 70 + "\n")
        f.write(report_txt)
    print(f"   ✅ Classification report saved")

    # ── 9. Save metrics.json ──────────────────────────────────────
    metrics_json = {
        "model": "VGG16 (KD + G1-FT)",
        "teacher": "ConvNeXt",
        "timestamp": datetime.now().isoformat(),
        "best_epoch": len(history["train_loss"]),
        "class_names": CLASS_NAMES,
        "config": {
            "input_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "effective_batch_size": EFFECTIVE_BATCH,
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "epochs": NUM_EPOCHS,
            "lr_backbone": LR_BACKBONE,
            "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "grad_clip": GRAD_CLIP,
            "label_smoothing": LABEL_SMOOTHING,
            "seed": SEED,
            "g1_ft_epochs": G1_FT_EPOCHS,
            "g1_ft_lr": G1_FT_LR,
        },
        "metrics": {
            "accuracy": final_metrics["accuracy"],
            "auc_macro": final_metrics["auc_macro"],
            "weighted_avg": final_metrics["weighted_avg"],
            "per_class": {
                CLASS_NAMES[i]: {
                    "precision": final_metrics["precision_per_class"][i],
                    "recall": final_metrics["recall_per_class"][i],
                    "f1-score": final_metrics["f1_per_class"][i],
                    "support": final_metrics["support_per_class"][i],
                }
                for i in range(len(CLASS_NAMES))
            },
        },
        "history_summary": {
            "total_epochs": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_val_acc": history["val_acc"][-1],
            "best_val_acc": max(history["val_acc"]),
            "gap_mean": float(np.mean(history["gap"])),
            "gap_final": float(history["gap"][-1]),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4, ensure_ascii=False)
    print(f"   ✅ Metrics saved")

    # ── 10. Advanced Evaluation ───────────────────────────────────
    print("\n📈 Running advanced evaluation (REAL CV5 + CI95 + TTA32)...")
    cv5_results = run_real_cv5(student, train_df, train_transform, val_transform)
    ci95_results = run_real_ci95(final_metrics["y_true"], final_metrics["y_prob"], n_bootstrap=1000)
    tta32_results = run_real_tta32(student, test_df, n_aug=32)

    # ✅ Build adv_results dict (THIS WAS MISSING!)
    adv_results = {
        "cv5": cv5_results,
        "ci95": ci95_results,
        "tta32": tta32_results,
    }

    # ── 11. Save summary.txt ──────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  CONVNEXT → VGG16 KD ENSEMBLE + G1-FT — TEST RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Classes: {', '.join(CLASS_NAMES)}\n\n")
        f.write("-" * 70 + "\n")
        f.write("  🎯 OVERALL METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Accuracy                      : {final_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision (weighted)          : {final_metrics['weighted_avg']['precision']:.4f}\n")
        f.write(f"  Recall (weighted)             : {final_metrics['weighted_avg']['recall']:.4f}\n")
        f.write(f"  F1-Score (weighted)           : {final_metrics['weighted_avg']['f1-score']:.4f}\n")
        f.write(f"  AUC (macro)                   : {final_metrics['auc_macro']:.4f}\n\n")
        f.write("-" * 70 + "\n")
        f.write("  PER-CLASS METRICS\n")
        f.write("-" * 70 + "\n")
        f.write("  Class         Precision       Recall           F1    Support\n")
        f.write("  ------------------------------------------------------------\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {name:<12} {final_metrics['precision_per_class'][i]:.4f}       "
                    f"{final_metrics['recall_per_class'][i]:.4f}       "
                    f"{final_metrics['f1_per_class'][i]:.4f}      "
                    f"{int(final_metrics['support_per_class'][i]):>5}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("  ADVANCED EVALUATION\n")
        f.write("=" * 70 + "\n")
        f.write(f"  CV5 (real, {CV5_EPOCHS} epochs/fold):\n")
        f.write(f"    Mean Acc: {cv5_results['mean_acc']:.4f} ± {cv5_results['std_acc']:.4f}\n")
        f.write(f"    Mean AUC: {cv5_results['mean_auc']:.4f} ± {cv5_results['std_auc']:.4f}\n\n")
        f.write(f"  CI95 (Bootstrap 1000):\n")
        f.write(f"    Acc: {ci95_results['acc']['mean']:.4f} "
                f"[{ci95_results['acc']['lower']:.4f}, {ci95_results['acc']['upper']:.4f}]\n")
        f.write(f"    AUC: {ci95_results['auc']['mean']:.4f} "
                f"[{ci95_results['auc']['lower']:.4f}, {ci95_results['auc']['upper']:.4f}]\n\n")
        f.write(f"  TTA-32:\n")
        f.write(f"    Acc: {tta32_results['acc']:.4f}\n")
        f.write(f"    AUC: {tta32_results['auc']:.4f}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("  Report generated by Liver AI Pipeline\n")
        f.write("  developed by : babak yousefian\n")
        f.write("=" * 70 + "\n")
    print(f"   ✅ Summary saved")

    # ── 12. Generate Report ────────────────────────────────────────
    print("\n📄 Generating 24-page report...")
    plot_train_val_gap(history, os.path.join(OUTPUT_DIR, "train_val_gap.png"))

    config = {
        "Teacher": "ConvNeXt (bestKDE.pth)",
        "Student": "VGG16 (vgg16BEST.pth)",
        "Temperature": TEMPERATURE,
        "Alpha": ALPHA,
        "Epochs": NUM_EPOCHS,
        "Batch Size": BATCH_SIZE,
        "LR (Backbone)": LR_BACKBONE,
        "LR (Head)": LR_HEAD,
        "Weight Decay": WEIGHT_DECAY,
        "Patience": PATIENCE,
        "Grad Clip": GRAD_CLIP,
        "Label Smoothing": LABEL_SMOOTHING,
        "Device": DEVICE,
        "Seed": SEED,
        "G1-FT Epochs": G1_FT_EPOCHS,
        "G1-FT LR": G1_FT_LR,
    }

    reporter_history = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "learning_rates": history["learning_rates"],
        "teacher_agreement": history["teacher_agreement"],
    }

    report_path = generate_kd_report(
        model=student,
        teacher=teacher,
        test_ds=test_ds,
        test_metrics=final_metrics,
        history=reporter_history,
        adv_results=adv_results,  # ← حالا تعریف شده!
        output_dir=OUTPUT_DIR,
        student_name="VGG16 (KD + G1-FT)",
        teacher_name="ConvNeXt",
        config=config,
    )

    print(f"\n✅ KD pipeline complete!")
    print(f"   📄 Report: {report_path}")
    print(f"   📁 Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    return {
        "status": "success",
        "model_path": g1_ft_model_path,
        "kd_model_path": kd_model_path,
        "report_path": report_path,
        "metrics": final_metrics,
        "history": history,
        "advanced_eval": adv_results,
    }


# ═════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = run_convnext_vgg16_kd()
    if result:
        print("\n🎉 Done!")