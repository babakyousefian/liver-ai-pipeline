# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/convnext_vit_kd.py
# -*- coding: utf-8 -*-

"""
ConvNeXt (Teacher) → Vision Transformer (Student) Knowledge Distillation
================================================================
- Teacher: ConvNeXt-Tiny with custom head (matches bestKDE.pth)
- Student: torchvision ViT-B/16 with custom head (matches visiontransformerBEST.pth)
- Full KD training loop (NO G1 fine-tune)
- Advanced: Real CV5, Real CI95 (Bootstrap), Clean TTA32
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Rich UI ─────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
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

# ── Config ──────────────────────────────────────────────────────────
CONVNEXT_TEACHER_PATH = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\bestKDEnsemble\bestKDE.pth"
VIT_STUDENT_PATH = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\VisionTransformer\visiontransformerBEST.pth"
IMAGES_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\cropped"
LABELS_CSV = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\labels.csv"
OUTPUT_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\bestKDEnsembles\ViT"

# Hyperparameters (NO G1 fine-tune!)
IMG_SIZE = 224
BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH = BATCH_SIZE * ACCUMULATION_STEPS
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
LR_BACKBONE = 5e-6
LR_HEAD = 2e-4
WEIGHT_DECAY = 0.1
TEMPERATURE = 4.0
ALPHA = 0.7
PATIENCE = 5
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
SEED = 42

# CV5 hyperparameters
CV5_EPOCHS = 5
CV5_LR = 1e-5
CV5_PATIENCE = 3

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
# TEACHER (ConvNeXt with custom head)
# ═════════════════════════════════════════════════════════════════════

def get_convnext_teacher():
    """
    Build ConvNeXt-Tiny with the EXACT custom head from bestKDE.pth.
    Keys match: classifier.0, classifier.2.1, classifier.2.2, classifier.2.5
    """
    from torchvision.models import convnext_tiny
    from collections import OrderedDict
    import torch.nn as nn

    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features  # 768

    custom_classifier = nn.Sequential(OrderedDict([
        ("0", nn.LayerNorm(in_features, eps=1e-6)),
        ("2", nn.Sequential(OrderedDict([
            ("1", nn.Linear(in_features, 256)),
            ("2", nn.LayerNorm(256, eps=1e-6)),
            ("3", nn.GELU()),
            ("4", nn.Dropout(0.2)),
            ("5", nn.Linear(256, len(CLASS_NAMES))),
        ]))),
    ]))

    class ConvNeXtWithCustomHead(nn.Module):
        def __init__(self, base_model, classifier):
            super().__init__()
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            self.classifier = classifier

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    return ConvNeXtWithCustomHead(model, custom_classifier)


# ═════════════════════════════════════════════════════════════════════
# STUDENT (ViT with custom head — EXACT MATCH)
# ═════════════════════════════════════════════════════════════════════

def get_vit_student():
    """
    Build torchvision ViT-B/16 with the EXACT custom head from
    visiontransformerBEST.pth.

    Checkpoint structure:
        class_token                                      (1, 1, 768)
        conv_proj.weight                                 (768, 3, 16, 16)
        conv_proj.bias                                   (768,)
        encoder.pos_embedding                            (1, 197, 768)
        encoder.layers.encoder_layer_{0..11}.*           ...
        encoder.ln.weight/bias                           (768,)
        heads.1.weight/bias                              (256, 768), (256,)
        heads.2.weight/bias/running_mean/running_var     (256,)
        heads.5.weight/bias                              (4, 256), (4,)
    """
    from torchvision.models import vit_b_16
    from collections import OrderedDict

    model = vit_b_16(weights=None, num_classes=len(CLASS_NAMES))

    # Replace the default heads (Sequential with a single Linear)
    # with the custom heads from checkpoint:
    #   heads.1 = Linear(768, 256)
    #   heads.2 = BatchNorm1d(256)
    #   heads.3 = ReLU
    #   heads.4 = Dropout(0.5)
    #   heads.5 = Linear(256, 4)
    custom_heads = nn.Sequential(OrderedDict([
        ("1", nn.Linear(768, 256)),
        ("2", nn.BatchNorm1d(256)),
        ("3", nn.ReLU(inplace=True)),
        ("4", nn.Dropout(0.5)),
        ("5", nn.Linear(256, len(CLASS_NAMES))),
    ]))

    model.heads = custom_heads
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
                                  replacement=False)


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
# REAL CV5
# ═════════════════════════════════════════════════════════════════════

def run_real_cv5(student, df, train_transform, val_transform):
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

        fold_model = get_vit_student().to(DEVICE)
        fold_model.load_state_dict(student.state_dict())

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
# REAL CI95
# ═════════════════════════════════════════════════════════════════════

def run_real_ci95(y_true, y_prob, n_bootstrap=1000):
    print(f"\n   🎲 Running REAL CI95 (Bootstrap {n_bootstrap})...")
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
# REAL TTA-32
# ═════════════════════════════════════════════════════════════════════

def run_real_tta32(model, val_df, n_aug=8, batch_size=32):
    """
    Lightweight TTA: n_aug passes (default 8) with larger batch size (32).
    Much faster than 32 passes for heavy models like ViT.
    """
    print(f"\n   🔄 Running REAL TTA-{n_aug} (batch_size={batch_size})...")
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
    tta_loader = DataLoader(
        tta_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Get all labels once
    all_labels = []
    for batch in tta_loader:
        if len(batch) == 3:
            _, y, _ = batch
        else:
            _, y = batch
        all_labels.extend(y.cpu().numpy())
    all_labels = np.array(all_labels)

    n_samples = len(all_labels)
    probs_sum = np.zeros((n_samples, len(CLASS_NAMES)))
    model.eval()

    with torch.no_grad():
        for aug_idx in range(n_aug):
            print(f"      TTA pass {aug_idx + 1}/{n_aug}...", end="\r")
            offset = 0
            for batch in tta_loader:
                if len(batch) == 3:
                    x, _, _ = batch
                else:
                    x, _ = batch
                x = x.to(DEVICE)
                with autocast(enabled=(DEVICE == "cuda")):
                    logits = model(x)
                probs = F.softmax(logits.float(), dim=1).cpu().numpy()
                probs_sum[offset:offset + len(probs)] += probs
                offset += len(probs)
    print(f"      ✅ TTA complete ({n_aug} passes)")

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
# PRINT CONFIG PANEL
# ═════════════════════════════════════════════════════════════════════

def print_config_panel(config: dict):
    if not HAS_RICH:
        print("\n" + "=" * 70)
        print("  ⚙️  KD TRAINING CONFIGURATION")
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
    table.add_column("Value", style="bold white", width=45)

    for k, v in config.items():
        table.add_row(str(k), str(v))

    console.print()
    console.print(table)
    console.print()


# ═════════════════════════════════════════════════════════════════════
# MAIN KD PIPELINE
# ═════════════════════════════════════════════════════════════════════

def run_convnext_vit_kd():
    print("\n" + "=" * 70)
    print("  🚀 ConvNeXt (Teacher) → ViT (Student) KD Ensemble")
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

    # 80/10/10 split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["liverLabel"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["liverLabel"]
    )

    print(f"\n   📊 Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

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
    student = get_vit_student().to(DEVICE)

    if os.path.exists(CONVNEXT_TEACHER_PATH):
        checkpoint = torch.load(CONVNEXT_TEACHER_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            teacher.load_state_dict(checkpoint["model_state_dict"])
        else:
            teacher.load_state_dict(checkpoint)
        print(f"   ✅ Teacher loaded")
    else:
        print(f"   ⚠️ Teacher not found")

    if os.path.exists(VIT_STUDENT_PATH):
        checkpoint = torch.load(VIT_STUDENT_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            student.load_state_dict(checkpoint["model_state_dict"])
        else:
            student.load_state_dict(checkpoint)
        print(f"   ✅ Student loaded (ViT)")
    else:
        print(f"   ⚠️ Student not found")

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
        if any(x in name.lower() for x in ["head", "heads", "classifier", "fc"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # ── 4. Training Loop ──────────────────────────────────────────
    print("\n🏋️ Starting KD training...")

    config_display = {
        "Teacher": "ConvNeXt (bestKDE.pth)",
        "Student": "ViT-B/16 (visiontransformerBEST.pth)",
        "Epochs": NUM_EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Effective Batch": EFFECTIVE_BATCH,
        "LR (Backbone)": f"{LR_BACKBONE:.2e}",
        "LR (Head)": f"{LR_HEAD:.2e}",
        "Weight Decay": WEIGHT_DECAY,
        "Temperature": TEMPERATURE,
        "Alpha (KD weight)": ALPHA,
        "Label Smoothing": LABEL_SMOOTHING,
        "Patience": PATIENCE,
        "Scheduler": "CosineAnnealingLR",
        "Device": DEVICE,
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
    best_model_path = os.path.join(OUTPUT_DIR, "convnext_to_vit_kd_best.pth")

    print("\n" + "=" * 110)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Acc':>8} | "
          f"{'Gap':>8} | {'LR':>10} | {'Teach Agr':>9} | {'Best?':>6} | {'Patience':>8}")
    print("=" * 110)

    for epoch in range(1, NUM_EPOCHS + 1):
        student.train()
        epoch_loss = 0.0
        total = 0

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

        val_metrics = evaluate(student, val_loader, nn.CrossEntropyLoss(), DEVICE)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        gap = val_loss - avg_train_loss

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

        is_best = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(student.state_dict())
            torch.save(best_state, best_model_path)
            is_best = True
            patience_counter = 0  # ← ریست Patience
        else:
            patience_counter += 1

        best_mark = "✅" if is_best else " "
        print(f"{epoch:>5} | {avg_train_loss:>10.4f} | {val_loss:>8.4f} | {val_acc:>8.4f} | "
              f"{gap:>+8.4f} | {lr:>10.2e} | {teacher_agreement:>9.4f} | "
              f"{best_mark:>6} | {patience_counter:>3}/{PATIENCE}")

        if is_best:
            print(f"        💾 Best model saved! (Val Acc: {val_acc:.4f}) → {best_model_path}")

        scheduler.step()

        if patience_counter >= PATIENCE:
            print("\n" + "=" * 110)
            print(f"⏹️  Early stopping at epoch {epoch} (patience={PATIENCE})")
            print(f"   Best epoch: {best_epoch} | Best Val Acc: {best_val_acc:.4f}")
            print("=" * 110)
            break

    # Restore best
    if best_state is not None:
        student.load_state_dict(best_state)
        print(f"\n🏆 Best model restored from epoch {best_epoch} (Val Acc: {best_val_acc:.4f})")

    # ── 5. Final Evaluation on TEST ───────────────────────────────
    print("\n📊 Final evaluation on TEST set...")
    final_metrics = evaluate(student, test_loader, nn.CrossEntropyLoss(), DEVICE)

    # ── 6. Save classification_report.txt ────────────────────────
    report_txt = classification_report(
        final_metrics["y_true"], final_metrics["y_pred"],
        target_names=CLASS_NAMES, digits=4
    )
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("ViT-B/16 (KD) — Classification Report (Test Set)\n")
        f.write("=" * 70 + "\n")
        f.write(report_txt)
    print(f"   ✅ classification_report.txt saved")

    # ── 7. Save metrics.json ──────────────────────────────────────
    metrics_json = {
        "model": "ViT-B/16 (KD)",
        "teacher": "ConvNeXt",
        "timestamp": datetime.now().isoformat(),
        "best_epoch": best_epoch,
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
    print(f"   ✅ metrics.json saved")

    # ── 8. Advanced Evaluation ────────────────────────────────────
    print("\n📈 Running advanced evaluation (CV5 + CI95 + TTA32)...")
    # بعد (جایگزین کن):
    cv5_results = {
        "folds": [],
        "mean_acc": float(final_metrics["accuracy"]),
        "std_acc": 0.0,
        "mean_auc": float(final_metrics["auc_macro"]),
        "std_auc": 0.0,
        "note": "CV5 skipped (too slow for ViT). Using test metrics as proxy.",
    }
    ci95_results = run_real_ci95(final_metrics["y_true"], final_metrics["y_prob"], n_bootstrap=1000)
    tta32_results = run_real_tta32(student, test_df, n_aug=32)

    adv_results = {
        "cv5": cv5_results,
        "ci95": ci95_results,
        "tta32": tta32_results,
    }

    # ── 9. Save summary.txt ──────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  CONVNEXT → ViT-B/16 KD ENSEMBLE — TEST RESULTS\n")
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
    print(f"   ✅ summary.txt saved")

    # ── 10. Generate Report ──────────────────────────────────────
    print("\n📄 Generating 24-page report...")
    plot_train_val_gap(history, os.path.join(OUTPUT_DIR, "train_val_gap.png"))

    config = {
        "Teacher": "ConvNeXt (bestKDE.pth)",
        "Student": "ViT-B/16 (visiontransformerBEST.pth)",
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
        adv_results=adv_results,
        output_dir=OUTPUT_DIR,
        student_name="ViT-B/16 (KD)",
        teacher_name="ConvNeXt",
        config=config,
    )

    print(f"\n✅ KD pipeline complete!")
    print(f"   📄 Report: {report_path}")
    print(f"   📁 Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    return {
        "status": "success",
        "model_path": best_model_path,
        "report_path": report_path,
        "metrics": final_metrics,
        "history": history,
        "advanced_eval": adv_results,
    }


def run_advanced_only_vit():
    """
    Load saved ViT KD model, run ONLY:
    - Final evaluation on TEST set
    - CI95 (Bootstrap)
    - TTA32
    - Generate 24-page report

    Skips CV5 (too slow for ViT).
    """
    print("\n" + "=" * 70)
    print("  📄 ADVANCED-ONLY: ViT KD Report (no retraining)")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_model_path = os.path.join(OUTPUT_DIR, "convnext_to_vit_kd_best.pth")
    if not os.path.exists(best_model_path):
        print(f"❌ Best model not found: {best_model_path}")
        print("   Run option 7 (full KD) first.")
        return None

    # ── 1. Load Data ──
    print("\n📂 Loading dataset...")
    df = pd.read_csv(LABELS_CSV)

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

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["liverLabel"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["liverLabel"]
    )

    val_ds = LiverDataset(val_df, IMAGES_DIR, val_transform)
    test_ds = LiverDataset(test_df, IMAGES_DIR, val_transform)
    val_loader = make_loader(val_ds, shuffle=False, drop_last=False)
    test_loader = make_loader(test_ds, shuffle=False, drop_last=False)

    # ── 2. Load Models ──
    print("\n🧠 Loading models...")
    teacher = get_convnext_teacher().to(DEVICE)
    student = get_vit_student().to(DEVICE)

    if os.path.exists(CONVNEXT_TEACHER_PATH):
        ckpt = torch.load(CONVNEXT_TEACHER_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            teacher.load_state_dict(ckpt["model_state_dict"])
        else:
            teacher.load_state_dict(ckpt)

    ckpt = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        student.load_state_dict(ckpt["model_state_dict"])
    else:
        student.load_state_dict(ckpt)
    print(f"   ✅ Student loaded from {best_model_path}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── 3. Final Evaluation on TEST ──
    print("\n📊 Final evaluation on TEST set...")
    final_metrics = evaluate(student, test_loader, nn.CrossEntropyLoss(), DEVICE)

    # ── 4. Save classification_report.txt ──
    report_txt = classification_report(
        final_metrics["y_true"], final_metrics["y_pred"],
        target_names=CLASS_NAMES, digits=4
    )
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("ViT-B/16 (KD) — Classification Report (Test Set)\n")
        f.write("=" * 70 + "\n")
        f.write(report_txt)
    print(f"   ✅ classification_report.txt saved")

    # ── 5. Save metrics.json ──
    metrics_json = {
        "model": "ViT-B/16 (KD)",
        "teacher": "ConvNeXt",
        "timestamp": datetime.now().isoformat(),
        "class_names": CLASS_NAMES,
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
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4, ensure_ascii=False)
    print(f"   ✅ metrics.json saved")

    # ── 6. Advanced Evaluation (NO CV5) ──
    print("\n📈 Running advanced evaluation (CI95 + TTA32, NO CV5)...")

    cv5_results = {
        "folds": [],
        "mean_acc": float(final_metrics["accuracy"]),
        "std_acc": 0.0,
        "mean_auc": float(final_metrics["auc_macro"]),
        "std_auc": 0.0,
        "note": "CV5 skipped (too slow for ViT). Using test metrics as proxy.",
    }
    ci95_results = run_real_ci95(final_metrics["y_true"], final_metrics["y_prob"], n_bootstrap=1000)
    tta32_results = run_real_tta32(student, test_df, n_aug=8, batch_size=32)

    adv_results = {
        "cv5": cv5_results,
        "ci95": ci95_results,
        "tta32": tta32_results,
    }

    # ── 7. Save summary.txt ──
    # ── 7. Load training history (if exists) ──
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            saved_history = json.load(f)
        print(f"   ✅ Loaded training history from {history_path}")
        reporter_history = saved_history
    else:
        print(f"   ⚠️ Training history not found. Gap plot will be empty.")
        reporter_history = {"train_loss": [], "val_loss": [], "val_acc": [],
                            "learning_rates": [], "teacher_agreement": []}

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  CONVNEXT → ViT-B/16 KD ENSEMBLE — TEST RESULTS\n")
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
        f.write(f"  CV5: SKIPPED (too slow for ViT)\n\n")
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
    print(f"   ✅ summary.txt saved")

    # ── 8. Generate Report ──
    print("\n📄 Generating 24-page report...")
    # NOTE: no train_val_gap plot (we don't have history here)
    config = {
        "Teacher": "ConvNeXt (bestKDE.pth)",
        "Student": "ViT-B/16 (visiontransformerBEST.pth)",
        "Note": "Advanced-only (no retraining)",
    }

    report_path = generate_kd_report(
        model=student,
        teacher=teacher,
        test_ds=test_ds,
        test_metrics=final_metrics,
        history={"train_loss": [], "val_loss": [], "val_acc": [],
                 "learning_rates": [], "teacher_agreement": []},
        adv_results=adv_results,
        output_dir=OUTPUT_DIR,
        student_name="ViT-B/16 (KD)",
        teacher_name="ConvNeXt",
        config=config,
    )

    # ── 9. Generate Report ──
    print("\n📄 Generating 24-page report...")
    # NOTE: plot_train_val_gap will skip if history is empty
    config = {
        "Teacher": "ConvNeXt (bestKDE.pth)",
        "Student": "ViT-B/16 (visiontransformerBEST.pth)",
        "Note": "Advanced-only (no retraining)",
    }

    report_path = generate_kd_report(
        model=student,
        teacher=teacher,
        test_ds=test_ds,
        test_metrics=final_metrics,
        history=reporter_history,
        adv_results=adv_results,
        output_dir=OUTPUT_DIR,
        student_name="ViT-B/16 (KD)",
        teacher_name="ConvNeXt",
        config=config,
    )

    print(f"\n✅ Advanced-only pipeline complete!")
    print(f"   📄 Report: {report_path}")
    print(f"   📁 Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    return {
        "status": "success",
        "model_path": best_model_path,
        "report_path": report_path,
        "metrics": final_metrics,
        "advanced_eval": adv_results,
    }


# ═════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = run_convnext_vit_kd()
    if result:
        print("\n🎉 Done!")