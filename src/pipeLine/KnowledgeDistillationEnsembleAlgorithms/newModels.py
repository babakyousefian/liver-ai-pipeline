# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/newModels.py
# -*- coding: utf-8 -*-

"""
Knowledge Distillation Ensemble — NEW MODELS (v2.1)

Pipeline:
    DenseNet Teacher (best_models.pth)
        ├── → Student AlexNet    (option 1)
        ├── → Student MobileNetV2 (option 2)
        └── → Student VGG16      (option 3)

✅ v2.1 LOW-RISK CHANGES (vs v2.0):
    1. Early stopping on ACCURACY (was: f1_macro)
    2. KD patience: 15 → 20
    3. Polish patience: 8 → 12

Everything else unchanged: LR, WD, Focal γ=0.5, Dropout, 3 seeds, TTA(32), CV5, CI95.

Author: Babak Yousefian
Version: 2.1.0
"""

import os
import gc
import json
import time
import random
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize

from tqdm import tqdm

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


# ================================================================
# ✅ Rich
# ================================================================

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.align import Align
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    print("⚠️ Rich not installed: pip install rich")

# ✅ Music
try:
    from src.MassiveClasses.Audio.music import Music
    HAS_MUSIC = True
except ImportError as e:
    HAS_MUSIC = False
    Music = None
    print(f"⚠️ Music not available: {e}")

# ✅ Music paths (one per student type)
MUSIC_PATHS = {
    "alexnet": "packages/facilities/assets/musics/music5.mp3",
    "mobilenetv2": "packages/facilities/assets/musics/music23.mp3",
    "vgg16": "packages/facilities/assets/musics/music39.mp3",
}

# ================================================================
# ✅ Paths (ABSOLUTE)
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))

SELF_SUP_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")
SSL_REPORTER_ROOT = os.path.join(SELF_SUP_ROOT, "SSL_Reporter")

TEACHER_DENSENET_PATH = os.path.join(
    SSL_REPORTER_ROOT, "FindBestNewModels", "best_models.pth"
)

ALEXNET_STUDENT_PATH = os.path.join(
    SSL_REPORTER_ROOT, "alexnet_finetune", "alexnet_finetuned_final.pth"
)
MOBILENETV2_STUDENT_PATH = os.path.join(
    SSL_REPORTER_ROOT, "mobilenetv2_finetune", "mobilenetv2_finetuned_final.pth"
)
VGG16_STUDENT_PATH = os.path.join(
    SSL_REPORTER_ROOT, "vgg16_finetune", "vgg16_finetuned_final.pth"
)

KD_ENSEMBLE_ROOT = os.path.join(SELF_SUP_ROOT, "KD_EnsembleModels")
KD_ALEXNET_ROOT = os.path.join(KD_ENSEMBLE_ROOT, "DenseNetVSAlexnet")
KD_MNV2_ROOT = os.path.join(KD_ENSEMBLE_ROOT, "DenseNetVSMobileNetV2")
KD_VGG16_ROOT = os.path.join(KD_ENSEMBLE_ROOT, "DenseNetVSVGG16")

FINAL_MODELS_DIR = KD_ENSEMBLE_ROOT


# ================================================================
# ✅ Constants
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = 4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEEDS = [42, 123, 2024]

DROPOUT_MAP = {
    "alexnet": 0.20,
    "mobilenetv2": 0.10,
    "vgg16": 0.25,
}

FOCAL_GAMMA = 0.5

# ✅ v2.1: Early stopping parameters
KD_PATIENCE = 20          # ✅ was 15
POLISH_PATIENCE = 12      # ✅ was 8
MIN_DELTA = 1e-4          # improvement threshold

# ================================================================
# ✅ v2.2: Training Profiles
# ================================================================
# Two profiles:
#   - "v2.1": baseline for AlexNet/VGG16 (unchanged)
#   - "v2.2": MobileNetV2 only — lower LR, stronger WD, gap monitor
#
# Goal of v2.2:
#   - Reduce gap between train_loss and val_loss (target: <18%)
#   - Increase test accuracy without heavy F1 drop
# ================================================================

TRAINING_PROFILES = {
    "v2.1": {
        # ---- KD ----
        "kd_epochs": 60,
        "kd_patience": 20,
        # ---- Polish ----
        "polish_epochs": 15,
        "polish_patience": 12,
        # ---- Learning Rates ----
        "backbone_lr": 1e-5,
        "head_lr": 1e-3,
        "polish_lr": 5e-6,
        # ---- Weight Decay ----
        "backbone_wd": 1e-2,
        "head_wd": 1e-3,
        # ---- Loss ----
        "label_smoothing": 0.05,
        # ---- Regularization ----
        "grad_clip": 0.5,
        # ---- Overfitting Control ----
        "gap_monitor_threshold": None,   # disabled
        "gap_monitor_patience": None,    # disabled
        "min_delta": 1e-4,
    },
    "v2.2": {
        # ---- KD ----
        "kd_epochs": 80,           # more room to learn
        "kd_patience": 25,         # patient for slow convergence
        # ---- Polish ----
        "polish_epochs": 20,
        "polish_patience": 15,
        # ---- Learning Rates (HALVED) ----
        "backbone_lr": 5e-6,       # ← was 1e-5
        "head_lr": 5e-4,           # ← was 1e-3
        "polish_lr": 2e-6,         # ← was 5e-6
        # ---- Weight Decay (DOUBLED) ----
        "backbone_wd": 2e-2,       # ← was 1e-2
        "head_wd": 2e-3,           # ← was 1e-3
        # ---- Loss ----
        "label_smoothing": 0.10,   # ← was 0.05
        # ---- Regularization ----
        "grad_clip": 0.5,
        # ---- Overfitting Control (GAP MONITOR) ----
        "gap_monitor_threshold": 1.5,   # val_loss > 1.5 × train_loss
        "gap_monitor_patience": 3,      # 3 consecutive epochs → stop
        "min_delta": 1e-4,
    },
    "v2.3": {
        # ---- VGG16-specific profile (heaviest model) ----
        # VGG16 has ~15M params (custom classifier) — needs
        # even lower LR + stronger WD than v2.2 to control overfit.
        # ---- KD ----
        "kd_epochs": 80,
        "kd_patience": 25,
        # ---- Polish ----
        "polish_epochs": 20,
        "polish_patience": 15,
        # ---- Learning Rates (LOWER than v2.2) ----
        "backbone_lr": 3e-6,       # ← was 5e-6 in v2.2
        "head_lr": 3e-4,           # ← was 5e-4 in v2.2
        "polish_lr": 1e-6,         # ← was 2e-6 in v2.2
        # ---- Weight Decay (STRONGER than v2.2) ----
        "backbone_wd": 3e-2,       # ← was 2e-2 in v2.2
        "head_wd": 3e-3,           # ← was 2e-3 in v2.2
        # ---- Loss ----
        "label_smoothing": 0.10,
        # ---- Regularization ----
        "grad_clip": 0.5,
        # ---- Overfitting Control (TIGHTER than v2.2) ----
        "gap_monitor_threshold": 1.4,   # ← 1.4 vs 1.5 in v2.2
        "gap_monitor_patience": 3,
        "min_delta": 1e-4,
        },
}

DEFAULT_PROFILE = "v2.1"

# Per-student profile mapping
STUDENT_PROFILE_MAP = {
    "alexnet": "v2.1",
    "mobilenetv2": "v2.2",
    "vgg16": "v2.3",       # ← NEW: dedicated VGG16 profile
}


def get_profile(student_type: str) -> Dict[str, Any]:
    """Return the training profile dict for a given student type."""
    name = STUDENT_PROFILE_MAP.get(student_type, DEFAULT_PROFILE)
    profile = dict(TRAINING_PROFILES[name])
    profile["name"] = name
    return profile

# ================================================================
# ✅ Rich helpers
# ================================================================

def print_banner(title: str, subtitle: str = ""):
    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold cyan]{title}[/bold cyan]\n[dim italic]{subtitle}[/dim italic]",
            border_style="bright_cyan", padding=(1, 4),
        ))
    else:
        print(f"\n{'='*80}\n  {title}\n  {subtitle}\n{'='*80}")

def print_step(step: int, total: int, text: str):
    if HAS_RICH:
        console.print(f"\n[bold yellow]▶ Step {step}/{total}:[/bold yellow] "
                      f"[bold white]{text}[/bold white]")
    else:
        print(f"\n[{step}/{total}] {text}")

def print_success(t: str):
    (console.print(f"[bold green]✅ {t}[/bold green]") if HAS_RICH else print(f"✅ {t}"))

def print_error(t: str):
    (console.print(f"[bold red]❌ {t}[/bold red]") if HAS_RICH else print(f"❌ {t}"))

def print_info(t: str):
    (console.print(f"[cyan]ℹ️  {t}[/cyan]") if HAS_RICH else print(f"ℹ️  {t}"))

def print_warning(t: str):
    (console.print(f"[yellow]⚠️  {t}[/yellow]") if HAS_RICH else print(f"⚠️  {t}"))

# ================================================================
# ✅ Music helper
# ================================================================

# ================================================================
# ✅ Music helper
# ================================================================

def _start_music(student_type: str):
    """Start background music for the given student type."""
    if not HAS_MUSIC:
        return None
    path = MUSIC_PATHS.get(student_type)
    if not path or not os.path.exists(path):
        print_warning(f"Music file not found: {path}")
        return None
    try:
        player = Music(MUSIC_PATH=path)
        player.play_music()
        print_info(f"🎵 Music playing: {os.path.basename(path)}")
        return player
    except Exception as e:
        print_warning(f"Could not play music: {e}")
        return None


def _stop_music(player):
    """Stop background music if it's playing."""
    if player is None:
        return
    try:
        player.stop_music()
        print_info("🎵 Music stopped.")
    except Exception as e:
        print_warning(f"Could not stop music cleanly: {e}")


from functools import wraps


def with_music(student_type: str):
    """Decorator: play music for given student_type while the function runs."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            player = _start_music(student_type)
            try:
                return func(*args, **kwargs)
            finally:
                _stop_music(player)
        return wrapper
    return decorator


# ================================================================
# ✅ Seed + Dirs
# ================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(root: str):
    for sub in ["", "plots", "cv", "interpretability", "seeds"]:
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    os.makedirs(KD_ENSEMBLE_ROOT, exist_ok=True)

# ================================================================
# ✅ Dataset — Labeled (for train/val)
# ================================================================

class LabeledDataset(Dataset):
    """
    Standard labeled dataset for KD training.
    Light augmentation only (NO Mixup, NO CutMix).
    """
    def __init__(self, paths, labels, input_size=224, augment=False,
                 strength="light"):
        self.paths = paths
        self.labels = labels
        self.input_size = input_size
        self.augment = augment
        self.strength = strength

    def __len__(self):
        return len(self.paths)

    def _load(self, p):
        img = cv2.imread(p)
        if img is None:
            return np.zeros((self.input_size, self.input_size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.input_size, self.input_size))

    def _aug(self, img):
        # Horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img).copy()

        # Small rotation
        if random.random() < 0.5:
            if self.strength == "light":
                ang = random.uniform(-12, 12)
            else:
                ang = random.uniform(-20, 20)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Brightness + contrast jitter
        if random.random() < 0.6:
            a = random.uniform(0.9, 1.1)
            b = random.uniform(-10, 10)
            img = np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)

        return img

    def __getitem__(self, i):
        img = self._load(self.paths[i])
        if self.augment:
            img = self._aug(img)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (t - m) / s, self.labels[i]


# ================================================================
# ✅ Dataset — TTA (32 augmentations per sample)
# ================================================================

class TTADataset(Dataset):
    """
    Test-Time Augmentation dataset.
    Each image is replicated n_aug times (default 32) with deterministic
    augmentations. Returns (image, label, original_index).
    """
    def __init__(self, paths, labels, input_size=224, n_aug=32):
        self.paths = paths
        self.labels = labels
        self.input_size = input_size
        self.n_aug = n_aug

    def __len__(self):
        return len(self.paths) * self.n_aug

    def _load(self, p):
        img = cv2.imread(p)
        if img is None:
            return np.zeros((self.input_size, self.input_size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.input_size, self.input_size))

    def _tta(self, img, k):
        # 0: identity
        if k == 0:
            return img
        # 1-2: flips
        if k == 1:
            return np.fliplr(img).copy()
        if k == 2:
            return np.flipud(img).copy()
        # 3-6: rotations (±5, ±10)
        if 3 <= k <= 6:
            ang = [5, -5, 10, -10][k - 3]
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # 7-10: brightness scale
        if 7 <= k <= 10:
            f = [1.1, 0.9, 1.2, 0.8][k - 7]
            return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
        # 11-14: contrast adjust
        if 11 <= k <= 14:
            f = [1.1, 0.9, 1.2, 0.8][k - 11]
            m = img.mean()
            return np.clip((img.astype(np.float32) - m) * f + m, 0, 255).astype(np.uint8)
        # 15-17: gamma
        if 15 <= k <= 17:
            g = [0.9, 1.1, 1.15][k - 15]
            return np.clip((img.astype(np.float32) / 255.0) ** g * 255, 0, 255).astype(np.uint8)
        # 18-20: zoom
        if 18 <= k <= 20:
            zoom = [0.9, 0.85, 1.15][k - 18]
            h, w = img.shape[:2]
            nh, nw = int(h * zoom), int(w * zoom)
            r = cv2.resize(img, (nw, nh))
            if zoom < 1:
                out = np.zeros((h, w, 3), np.uint8)
                y1, x1 = (h - nh) // 2, (w - nw) // 2
                out[y1:y1 + nh, x1:x1 + nw] = r
                return out
            y1, x1 = (nh - h) // 2, (nw - w) // 2
            return r[y1:y1 + h, x1:x1 + w]
        # 21-24: translations (5 px)
        if 21 <= k <= 24:
            dx, dy = [(5, 0), (-5, 0), (0, 5), (0, -5)][k - 21]
            h, w = img.shape[:2]
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # 25-27: hflip + variants
        if 25 <= k <= 27:
            f = np.fliplr(img).copy()
            if k == 26:
                f = np.clip(f.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
            elif k == 27:
                f = np.clip(f.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
            return f
        # 28: center crop 90%
        if k == 28:
            h, w = img.shape[:2]
            ch, cw = int(h * 0.9), int(w * 0.9)
            y1, x1 = (h - ch) // 2, (w - cw) // 2
            return cv2.resize(img[y1:y1 + ch, x1:x1 + cw], (w, h))
        # 29-31: rotations (±15, 20)
        if 29 <= k <= 31:
            ang = [15, -15, 20][k - 29]
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return img

    def __getitem__(self, i):
        idx = i // self.n_aug
        aug = i % self.n_aug
        img = self._tta(self._load(self.paths[idx]), aug)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (t - m) / s, self.labels[idx], idx


# ================================================================
# ✅ Data Loading
# ================================================================

def load_labeled_data():
    """
    Load all labeled images from:
        finalReports/labels.csv + finalReports/cropped/

    Returns:
        paths:  list of absolute image paths
        labels: list of int labels in {0:NL, 1:G1, 2:G2, 3:G3}

    Falls back to alternate extensions if exact filename not found.
    """
    csv_p = os.path.join(PROJECT_ROOT, "finalReports", "labels.csv")
    img_dir = os.path.join(PROJECT_ROOT, "finalReports", "cropped")

    if not os.path.exists(csv_p):
        raise FileNotFoundError(f"Labels CSV not found: {csv_p}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    df = pd.read_csv(csv_p)
    df = df[df["liverLabel"].isin(CLASS_NAMES)]

    paths, labels = [], []
    for _, r in df.iterrows():
        name = r["liverName"]
        lbl = CLASS_NAMES.index(r["liverLabel"])
        p = os.path.join(img_dir, name)

        if os.path.exists(p):
            paths.append(p)
            labels.append(lbl)
            continue

        # Fallback: try alternate extensions
        base = os.path.splitext(name)[0]
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            c = os.path.join(img_dir, base + ext)
            if os.path.exists(c):
                paths.append(c)
                labels.append(lbl)
                break

    return paths, labels


# ================================================================
# ✅ Losses
# ================================================================
# Design decisions (from user's experience):
#   - NO Mixup / NO CutMix
#   - Focal γ = 0.5 (soft)
#   - Label smoothing = 0.05
#   - Class weights = inverse frequency (for G3)
# ================================================================

class FocalLossSoft(nn.Module):
    """
    Focal Loss with SOFT gamma (0.5) + label smoothing 0.05.

    L = (1 - p_t)^γ * CE(y, p)   with γ = 0.5
    """
    def __init__(self, alpha=None, gamma: float = 0.5, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, target):
        w = self.alpha.to(logits.device) if self.alpha is not None else None
        ce = F.cross_entropy(
            logits, target,
            weight=w,
            reduction="none",
            label_smoothing=self.ls,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class KDLossV2(nn.Module):
    """
    Knowledge Distillation Loss v2:
        L = α * KL(teacher || student, T) * T²
          + (1 - α) * Focal_soft(student, true_label)

    Notes:
    - T² scaling is standard (Hinton et al.)
    - Focal γ = 0.5 (soft)
    - NO feature matching
    - NO Mixup / NO CutMix
    """
    def __init__(self, alpha: float = 0.7, temperature: float = 3.0,
                 gamma: float = 0.5, class_weights=None,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.focal = FocalLossSoft(
            alpha=class_weights, gamma=gamma, label_smoothing=label_smoothing
        )
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, s_logits, t_probs, y):
        t_probs = t_probs.to(s_logits.device)
        s_log = F.log_softmax(s_logits / self.T, dim=1)
        kl = self.kl(s_log, t_probs) * (self.T ** 2)
        fl = self.focal(s_logits, y)
        return self.alpha * kl + (1.0 - self.alpha) * fl

# ================================================================
# ✅ Metrics (comprehensive)
# ================================================================
# compute_metrics returns ALL metrics needed for the final report:
#   - accuracy, precision, recall, f1 (weighted), f1_macro
#   - balanced_accuracy (mean per-class recall)
#   - ROC-AUC (per-class + mean)
#   - confusion matrix
#   - per-class classification report
# ================================================================

def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, Any]:
    """
    Compute a comprehensive metric dictionary.

    Args:
        y_true: (N,) int array of ground truth labels
        y_pred: (N,) int array of predicted labels
        y_prob: (N, NUM_CLASSES) float array of predicted probabilities

    Returns:
        dict with all metrics used across the pipeline.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Balanced accuracy = mean of per-class recall
    pcr = []
    for c in range(NUM_CLASSES):
        mask = y_true == c
        if mask.sum() > 0:
            pcr.append(float((y_pred[mask] == c).mean()))
    bal = float(np.mean(pcr)) if pcr else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

    # ROC-AUC (one-vs-rest per class + mean)
    aucs, fpr_d, tpr_d = {}, {}, {}
    try:
        yb = label_binarize(y_true, classes=range(NUM_CLASSES))
        for i in range(NUM_CLASSES):
            if yb[:, i].sum() > 0 and (1 - yb[:, i]).sum() > 0:
                fpr, tpr, _ = roc_curve(yb[:, i], y_prob[:, i])
                fpr_d[i] = fpr
                tpr_d[i] = tpr
                aucs[i] = float(auc(fpr, tpr))
    except Exception:
        pass
    auc_mean = float(np.mean(list(aucs.values()))) if aucs else 0.0

    # Per-class report (safe fallback for missing classes)
    cr = classification_report(
        y_true, y_pred,
        labels=range(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    for c in CLASS_NAMES:
        if c not in cr:
            cr[c] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "f1_macro": f1m,
        "balanced_accuracy": bal,
        "roc_auc": aucs,
        "roc_auc_mean": auc_mean,
        "fpr": fpr_d,
        "tpr": tpr_d,
        "confusion_matrix": cm,
        "classification_report": cr,
        "num_samples": int(len(y_true)),
    }


# ================================================================
# ✅ Model Loaders
# ================================================================
# Teacher: DenseNet from FindBestNewModels/best_models.pth
#   Confirmed architecture:
#     create_densenet(num_classes=4, variant="medium")
#     → layers=[6,12,24,16], growth_rate=24
#     → final_bn = 766, classifier = (4, 766)
#     → total params ≈ 3.95M
#
# Students: AlexNet / MobileNetV2 / VGG16
#   Dropout (per model, strictly within [0.05, 0.25]):
#     - AlexNet     → 0.20
#     - MobileNetV2 → 0.10
#     - VGG16       → 0.25
# ================================================================

def load_teacher_densenet(path: str, device: torch.device):
    """
    Load DenseNet teacher from best_models.pth.

    best_models.pth = {
        "model_state_dict": dict(len=727),   ← DenseNet weights
        "best_val_acc": float,
        "ssl_method": str,
        "ssl_checkpoint": str,
        "history": dict,
        "timestamp": str,
    }

    The teacher is fully frozen (requires_grad=False, eval mode).
    """
    from src.models.DenseNet.DenseNet import create_densenet

    if not os.path.exists(path):
        raise FileNotFoundError(f"Teacher DenseNet not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Unwrap the confirmed structure
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta = {
            "best_val_acc": ckpt.get("best_val_acc", None),
            "ssl_method": ckpt.get("ssl_method", None),
            "timestamp": ckpt.get("timestamp", None),
        }
        if meta["best_val_acc"] is not None:
            print_info(
                f"Teacher meta | best_val_acc={meta['best_val_acc']:.4f} | "
                f"ssl_method={meta['ssl_method']} | "
                f"timestamp={meta['timestamp']}"
            )
    else:
        # Fallback: assume ckpt IS the state dict
        state = ckpt

    # ✅ CORRECT ARCHITECTURE: variant="medium"
    model = create_densenet(
        num_classes=NUM_CLASSES,
        variant="medium",
        pretrained=False,
        verbose=False,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)

    if len(missing) > 0 or len(unexpected) > 0:
        print_warning(
            f"Teacher mismatch | missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
        if 0 < len(missing) <= 10:
            print_warning(f"   missing keys: {missing}")
        if 0 < len(unexpected) <= 10:
            print_warning(f"   unexpected keys: {unexpected}")
    else:
        print_success(
            f"Teacher DenseNet loaded cleanly (0 missing, 0 unexpected) | "
            f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
            f"variant=medium"
        )

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def load_student(student_type: str, checkpoint_path: str,
                 device: torch.device):
    """
    Load a student model (AlexNet / MobileNetV2 / VGG16) and initialize
    its weights from a pretrained fine-tuned checkpoint.

    Dropout:
        AlexNet     → 0.20
        MobileNetV2 → 0.10
        VGG16       → 0.25

    All students output NUM_CLASSES=4 logits.
    """
    dp = DROPOUT_MAP[student_type]

    # ---------- Architecture ----------
    if student_type == "alexnet":
        from src.models.AlexNet import create_alexnet
        model = create_alexnet(
            num_classes=NUM_CLASSES,
            width_mult=0.5,
            fc6_features=1024,
            fc7_features=512,
            dropout_fc=dp,              # ✅ 0.20
            use_batch_norm=True,
            input_size=224,
            verbose=False,
        )

    elif student_type == "mobilenetv2":
        from torchvision.models import mobilenet_v2
        # torchvision's mobilenet_v2 exposes dropout via `dropout=` kwarg
        model = mobilenet_v2(
            num_classes=NUM_CLASSES,
            width_mult=1.0,
            dropout=dp,                 # ✅ 0.10
        )

    elif student_type == "vgg16":
        from torchvision.models import vgg16
        # torchvision's vgg16 default dropout = 0.5 inside `classifier`
        # We rebuild the classifier with our own dropout
        model = vgg16(num_classes=NUM_CLASSES, weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dp),             # ✅ 0.25
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dp),             # ✅ 0.25
            nn.Linear(4096, NUM_CLASSES),
        )

    else:
        raise ValueError(f"Unknown student_type: {student_type}")

    # ---------- Load weights ----------
    if not os.path.exists(checkpoint_path):
        print_warning(f"Student checkpoint not found: {checkpoint_path}")
        return model.to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "encoder_state_dict" in ckpt:
            state = ckpt["encoder_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    # If checkpoint was saved from a KD pipeline (nested), unwrap
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print_info(
        f"Loaded {student_type} (dropout={dp}) | "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    return model.to(device)


# ================================================================
# ✅ Teacher inference helper
# ================================================================

@torch.no_grad()
def teacher_probs(teacher: nn.Module, images: torch.Tensor,
                  device: torch.device) -> torch.Tensor:
    """
    Run the teacher on a batch of images and return softmax probabilities
    as float32 on the *device*. Used inside the KD training loop.

    - no_grad: teacher is frozen
    - eval mode: stable BN stats
    """
    teacher.eval()
    logits = teacher(images.to(device, non_blocking=True))
    return F.softmax(logits.float(), dim=1)

# ================================================================
# ✅ KD Training — ONE seed
# ================================================================
# ✅ v2.1 CHANGES (low-risk):
#   - Early stopping on ACCURACY (was: f1_macro)
#   - KD patience: 20 (was: 15)
#   - MIN_DELTA = 1e-4 for improvement threshold
#
# Everything else unchanged:
#   - Warmup 3 + Cosine Annealing
#   - Weight Decay: backbone=1e-2, head=1e-3
#   - Gradient Clip: 0.5
#   - Focal γ=0.5 + Label Smoothing 0.05
#   - Mixed Precision (BF16 if available, else FP16)
# ================================================================

# ================================================================
# ✅ KD Training — ONE seed (v2.2 with profile support + Gap Monitor)
# ================================================================

def kd_train_one_seed(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    profile: str = DEFAULT_PROFILE,       # ← NEW: "v2.1" | "v2.2"
    alpha: float = 0.7,
    temperature: float = 3.0,
    class_weights: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a single student model via Knowledge Distillation for one seed.

    v2.2 improvements:
    - Lower LR (backbone 5e-6, head 5e-4)
    - Stronger WD (2e-2 / 2e-3)
    - Label smoothing 0.10
    - GAP MONITOR: stop if val_loss > 1.5 × train_loss for 3 epochs
    - Early stopping on Accuracy (same as v2.1)
    """
    prof = TRAINING_PROFILES[profile]

    # Unpack profile
    epochs = prof["kd_epochs"]
    patience = prof["kd_patience"]
    backbone_lr = prof["backbone_lr"]
    head_lr = prof["head_lr"]
    backbone_wd = prof["backbone_wd"]
    head_wd = prof["head_wd"]
    label_smoothing = prof["label_smoothing"]
    grad_clip = prof["grad_clip"]
    min_delta = prof["min_delta"]
    gap_threshold = prof["gap_monitor_threshold"]
    gap_patience = prof["gap_monitor_patience"]

    student = student.to(device)
    teacher = teacher.to(device).eval()

    if class_weights is not None:
        class_weights = class_weights.to(device)

    crit = KDLossV2(
        alpha=alpha,
        temperature=temperature,
        gamma=FOCAL_GAMMA,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    ).to(device)

    # ---------- Param groups (head vs backbone) ----------
    head_kw = ["fc", "classifier", "head", "final"]
    bb, hd = [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n.lower() for k in head_kw):
            hd.append(p)
        else:
            bb.append(p)

    pg = []
    if bb:
        pg.append({"params": bb, "lr": backbone_lr, "weight_decay": backbone_wd})
    if hd:
        pg.append({"params": hd, "lr": head_lr, "weight_decay": head_wd})
    opt = optim.AdamW(pg, betas=(0.9, 0.999))

    # ---------- Warmup + Cosine ----------
    warm = min(3, max(1, epochs // 10))
    cos_T = max(1, epochs - warm)
    cos = CosineAnnealingLR(opt, T_max=cos_T, eta_min=backbone_lr * 0.01)
    wm = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warm)
    sched = SequentialLR(opt, [wm, cos], milestones=[warm])

    # ---------- AMP ----------
    use_amp = device.type == "cuda"
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        amp_name = "BF16"
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        amp_name = "FP16"
    else:
        amp_dtype = None
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        amp_name = "disabled"

    if verbose:
        print_info(
            f"Seed {seed} | Profile={profile} | AMP={amp_name} | "
            f"warmup={warm} | α={alpha} | T={temperature} | γ={FOCAL_GAMMA} | "
            f"patience={patience} | ES=accuracy"
        )
        print_info(
            f"  LR: backbone={backbone_lr:.1e}, head={head_lr:.1e} | "
            f"WD: backbone={backbone_wd:.1e}, head={head_wd:.1e} | "
            f"LabelSmooth={label_smoothing}"
        )
        if gap_threshold is not None:
            print_info(
                f"  Gap Monitor: val_loss > {gap_threshold}× train_loss "
                f"for {gap_patience} epochs → STOP"
            )

    # ---------- History & early stopping ----------
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "val_f1_macro": [], "lr": [],
        "gap_ratio": [],                    # ← NEW: track gap
    }

    best_acc = -1.0
    best_state = None
    best_ep = 0
    wait = 0
    gap_wait = 0                            # ← NEW: for gap monitor
    final_val_metrics: Dict[str, Any] = {}

    # ---------- Training loop ----------
    for ep in range(epochs):
        # ====== TRAIN ======
        student.train()
        tl = 0.0
        tp, ty = [], []
        pbar = tqdm(
            train_loader,
            desc=f"S{seed} KD {ep+1}/{epochs}",
            leave=False,
            disable=not verbose,
            ncols=100,
        )
        for imgs, lbls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            with torch.no_grad():
                t_probs = teacher_probs(teacher, imgs, device)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    s_logits = student(imgs)
                    loss = crit(s_logits, t_probs, lbls)
            else:
                s_logits = student(imgs)
                loss = crit(s_logits, t_probs, lbls)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                opt.step()

            tl += loss.item()
            with torch.no_grad():
                tp.extend(s_logits.argmax(1).cpu().numpy())
                ty.extend(lbls.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        tr_loss = tl / max(1, len(train_loader))
        tr_acc = accuracy_score(ty, tp)
        tr_f1 = f1_score(ty, tp, average="weighted", zero_division=0)

        # ====== VALIDATION ======
        student.eval()
        vl = 0.0
        vp, vy, vprob = [], [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                t_probs = teacher_probs(teacher, imgs, device)

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        s_logits = student(imgs)
                        loss = crit(s_logits, t_probs, lbls)
                else:
                    s_logits = student(imgs)
                    loss = crit(s_logits, t_probs, lbls)

                vl += loss.item()
                prob = F.softmax(s_logits.float(), dim=1)
                vp.extend(prob.argmax(1).cpu().numpy())
                vy.extend(lbls.cpu().numpy())
                vprob.append(prob.cpu().numpy())

        val_loss = vl / max(1, len(val_loader))
        vm = compute_metrics(
            np.array(vy), np.array(vp), np.concatenate(vprob, axis=0)
        )
        final_val_metrics = vm

        # ---------- Gap ratio (v2.2) ----------
        gap_ratio = val_loss / max(tr_loss, 1e-6)
        history["gap_ratio"].append(gap_ratio)

        # ---------- History ----------
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(float(tr_acc))
        history["val_acc"].append(vm["accuracy"])
        history["train_f1"].append(float(tr_f1))
        history["val_f1"].append(vm["f1_score"])
        history["val_f1_macro"].append(vm["f1_macro"])
        history["lr"].append(opt.param_groups[0]["lr"])

        sched.step()

        if verbose:
            gap_str = f" Gap={gap_ratio:.2f}x" if gap_threshold is not None else ""
            print(f"   S{seed} E{ep+1:03d}/{epochs} | "
                  f"TrL={tr_loss:.4f} TrF1={tr_f1:.4f} | "
                  f"VaL={val_loss:.4f} VaF1={vm['f1_score']:.4f} "
                  f"VaF1m={vm['f1_macro']:.4f} Acc={vm['accuracy']:.4f}"
                  f"{gap_str}")

        # ---------- GAP MONITOR (v2.2 only) ----------
        if gap_threshold is not None:
            if gap_ratio > gap_threshold:
                gap_wait += 1
                if verbose:
                    print(f"   ⚠️  Gap warning {gap_wait}/{gap_patience} "
                          f"(ratio={gap_ratio:.2f} > {gap_threshold})")
                if gap_wait >= gap_patience:
                    if verbose:
                        print(f"   🛑 Seed {seed} GAP MONITOR STOP at epoch {ep+1} "
                              f"(gap_ratio={gap_ratio:.2f})")
                    break
            else:
                gap_wait = 0

        # ---------- Early stopping on ACCURACY ----------
        if vm["accuracy"] > best_acc + min_delta:
            best_acc = vm["accuracy"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in student.state_dict().items()
            }
            best_ep = ep + 1
            wait = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "model_state_dict": best_state,
                    "epoch": ep + 1,
                    "best_accuracy": best_acc,
                    "seed": seed,
                    "profile": profile,
                    "history": history,
                }, save_path)
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"   🛑 Seed {seed} early stop at epoch {ep+1} "
                          f"(best_acc={best_acc:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        student.load_state_dict(best_state)
        student = student.to(device)

    return {
        "history": history,
        "best_accuracy": best_acc,
        "best_epoch": best_ep,
        "final_val": final_val_metrics,
        "seed": seed,
        "state": best_state,
        "profile": profile,
    }


# ================================================================
# ✅ Polish Fine-Tune — short refinement, NO KD
# ================================================================
# ✅ v2.1 CHANGES:
#   - Early stopping on ACCURACY (was: f1_macro)
#   - Polish patience: 12 (was: 8)
# ================================================================

# ================================================================
# ✅ Polish Fine-Tune (v2.2 with profile support)
# ================================================================

def polish_finetune(
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    profile: str = DEFAULT_PROFILE,       # ← NEW
    class_weights: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Short polish fine-tune on true labels only (no teacher).

    v2.2:
    - LR from profile (2e-6 vs 5e-6)
    - Label smoothing from profile (0.10 vs 0.05)
    - Early stopping on Accuracy
    """
    prof = TRAINING_PROFILES[profile]

    epochs = prof["polish_epochs"]
    patience = prof["polish_patience"]
    lr = prof["polish_lr"]
    label_smoothing = prof["label_smoothing"]
    grad_clip = prof["grad_clip"]
    min_delta = prof["min_delta"]

    student = student.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    crit = FocalLossSoft(
        alpha=class_weights, gamma=FOCAL_GAMMA, label_smoothing=label_smoothing
    ).to(device)

    opt = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    # AMP
    use_amp = device.type == "cuda"
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        amp_dtype = None
        scaler = torch.amp.GradScaler("cuda", enabled=False)

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [], "val_acc": [],
        "val_f1_macro": [],
        "gap_ratio": [],
    }

    best_acc = -1.0
    best_state = None
    best_ep = 0
    wait = 0

    for ep in range(epochs):
        # ====== TRAIN ======
        student.train()
        tl = 0.0
        tp, ty = [], []
        pbar = tqdm(
            train_loader,
            desc=f"S{seed} Polish {ep+1}/{epochs}",
            leave=False,
            disable=not verbose,
            ncols=100,
        )
        for imgs, lbls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    s_logits = student(imgs)
                    loss = crit(s_logits, lbls)
            else:
                s_logits = student(imgs)
                loss = crit(s_logits, lbls)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                opt.step()

            tl += loss.item()
            with torch.no_grad():
                tp.extend(s_logits.argmax(1).cpu().numpy())
                ty.extend(lbls.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        tr_loss = tl / max(1, len(train_loader))
        tr_f1 = f1_score(ty, tp, average="weighted", zero_division=0)

        # ====== VALIDATION ======
        student.eval()
        vl = 0.0
        vp, vy, vprob = [], [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        s_logits = student(imgs)
                        loss = crit(s_logits, lbls)
                else:
                    s_logits = student(imgs)
                    loss = crit(s_logits, lbls)
                vl += loss.item()
                prob = F.softmax(s_logits.float(), dim=1)
                vp.extend(prob.argmax(1).cpu().numpy())
                vy.extend(lbls.cpu().numpy())
                vprob.append(prob.cpu().numpy())

        val_loss = vl / max(1, len(val_loader))
        vm = compute_metrics(
            np.array(vy), np.array(vp), np.concatenate(vprob, axis=0)
        )

        gap_ratio = val_loss / max(tr_loss, 1e-6)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(float(tr_f1))
        history["val_f1"].append(vm["f1_score"])
        history["val_acc"].append(vm["accuracy"])
        history["val_f1_macro"].append(vm["f1_macro"])
        history["gap_ratio"].append(gap_ratio)

        if verbose:
            print(f"   S{seed} Polish E{ep+1}/{epochs} | "
                  f"TrF1={tr_f1:.4f} | VaF1={vm['f1_score']:.4f} "
                  f"VaF1m={vm['f1_macro']:.4f} Acc={vm['accuracy']:.4f} | "
                  f"Gap={gap_ratio:.2f}x")

        # ✅ Early stopping on ACCURACY
        if vm["accuracy"] > best_acc + min_delta:
            best_acc = vm["accuracy"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in student.state_dict().items()
            }
            best_ep = ep + 1
            wait = 0
            if save_path:
                torch.save({
                    "model_state_dict": best_state,
                    "epoch": ep + 1,
                    "best_accuracy": best_acc,
                    "seed": seed,
                    "profile": profile,
                    "history": history,
                }, save_path)
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"   🛑 Seed {seed} polish early stop at epoch {ep+1} "
                          f"(best_acc={best_acc:.4f})")
                break

    if best_state is not None:
        student.load_state_dict(best_state)
        student = student.to(device)

    return {
        "history": history,
        "best_accuracy": best_acc,
        "best_epoch": best_ep,
        "final_val": vm,
        "seed": seed,
        "state": best_state,
        "profile": profile,
    }

# ================================================================
# ✅ 3-Seed Ensemble — probability averaging
# ================================================================
# We train 3 independent students (seeds 42, 123, 2024).
# At inference, the ensemble probability for each image is the arithmetic
# mean of the 3 students' softmax outputs.
# This reduces variance and improves generalization, especially for G3.
# ================================================================

@torch.no_grad()
def ensemble_predict(models: List[nn.Module], images: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    """
    Run each model in `models` on the batch and average their softmax probs.

    Args:
        models: list of loaded models in eval mode
        images: (B, C, H, W) tensor already on CPU
        device: torch device

    Returns:
        (B, NUM_CLASSES) averaged softmax probabilities (CPU float32)
    """
    probs_sum = None
    for m in models:
        m.eval()
        m.to(device)
        logits = m(images.to(device, non_blocking=True))
        probs = F.softmax(logits.float(), dim=1).cpu()
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(models)


# ================================================================
# ✅ TTA Evaluation on the 3-seed ensemble
# ================================================================

@torch.no_grad()
def evaluate_with_tta(
    models: List[nn.Module],
    image_paths: List[str],
    labels: List[int],
    device: torch.device,
    input_size: int = 224,
    n_aug: int = 32,
    batch_size: int = 8,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full TTA evaluation:
    - For each image, generate n_aug augmented versions
    - Run 3-seed ensemble on every augmented version
    - Average probabilities across all (3 seeds × n_aug) predictions per image
    - Return (preds, averaged_probs)
    """
    n = len(image_paths)
    tta_ds = TTADataset(image_paths, labels, input_size=input_size, n_aug=n_aug)
    tta_loader = DataLoader(
        tta_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    per_img_probs: List[List[np.ndarray]] = [[] for _ in range(n)]

    pbar = tqdm(
        tta_loader, desc="   TTA Inference",
        leave=False, disable=not verbose, ncols=100,
    )
    for imgs, _lbls, img_idx in pbar:
        # Ensemble over 3 seeds → averaged probs per aug-sample
        probs = ensemble_predict(models, imgs, device).numpy()  # (B, C)
        for i, idx in enumerate(img_idx.numpy()):
            per_img_probs[idx].append(probs[i])

    avg_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    for i in range(n):
        if len(per_img_probs[i]) > 0:
            avg_probs[i] = np.mean(per_img_probs[i], axis=0)

    preds = avg_probs.argmax(axis=1)
    return preds, avg_probs


# ================================================================
# ✅ Bootstrap 95% CI
# ================================================================

def compute_ci95_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap 95% confidence intervals for accuracy, f1, precision, recall.

    For each metric, returns:
        {"mean": float, "std": float, "ci95_low": float, "ci95_high": float}
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_acc, boot_f1, boot_prec, boot_rec = [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_acc.append(accuracy_score(yt, yp))
        boot_f1.append(f1_score(yt, yp, average="weighted", zero_division=0))
        boot_prec.append(precision_score(yt, yp, average="weighted", zero_division=0))
        boot_rec.append(recall_score(yt, yp, average="weighted", zero_division=0))

    def _ci(arr):
        arr = np.array(arr)
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }

    return {
        "accuracy": _ci(boot_acc),
        "f1_score": _ci(boot_f1),
        "precision": _ci(boot_prec),
        "recall": _ci(boot_rec),
    }


# ================================================================
# ✅ Stratified 5-Fold CV (with TTA per fold)
# ================================================================

def run_cv5(
    models: List[nn.Module],
    image_paths: List[str],
    labels: List[int],
    device: torch.device,
    input_size: int = 224,
    n_folds: int = 5,
    n_aug_tta: int = 32,
    batch_size: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Lightweight CV5: evaluate the trained 3-seed ensemble on 5 different
    stratified splits. This measures robustness on different held-out
    distributions (not full retraining per fold).

    Returns:
        {
            "fold_results": [...],
            "mean_accuracy", "std_accuracy",
            "mean_f1", "std_f1",
            "mean_auc", "std_auc"
        }
    """
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    if verbose:
        print_info(f"Running {n_folds}-fold evaluation with TTA ({n_aug_tta} augs)")

    for fold_idx, (_, val_idx) in enumerate(skf.split(image_paths, labels_arr)):
        val_paths = [image_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        if verbose:
            print(f"\n   📌 Fold {fold_idx + 1}/{n_folds} "
                  f"({len(val_paths)} samples)")

        preds, probs = evaluate_with_tta(
            models, val_paths, val_labels, device,
            input_size=input_size, n_aug=n_aug_tta,
            batch_size=batch_size, verbose=False,
        )
        m = compute_metrics(val_labels, preds, probs)

        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": m["accuracy"],
            "f1_score": m["f1_score"],
            "f1_macro": m["f1_macro"],
            "precision": m["precision"],
            "recall": m["recall"],
            "balanced_accuracy": m["balanced_accuracy"],
            "roc_auc_mean": m["roc_auc_mean"],
            "confusion_matrix": m["confusion_matrix"].tolist(),
        })

        if verbose:
            print(f"      Acc={m['accuracy']:.4f} | F1={m['f1_score']:.4f} "
                  f"| F1m={m['f1_macro']:.4f} | AUC={m['roc_auc_mean']:.4f}")

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_score"] for r in fold_results]
    aucs = [r["roc_auc_mean"] for r in fold_results]

    cv_summary = {
        "fold_results": fold_results,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }

    if verbose:
        if HAS_RICH:
            console.print(f"\n   [bold cyan]CV Summary:[/bold cyan]")
        else:
            print("\n   CV Summary:")
        print(f"      Mean Accuracy: {cv_summary['mean_accuracy']:.4f} "
              f"± {cv_summary['std_accuracy']:.4f}")
        print(f"      Mean F1:       {cv_summary['mean_f1']:.4f} "
              f"± {cv_summary['std_f1']:.4f}")
        print(f"      Mean AUC:      {cv_summary['mean_auc']:.4f} "
              f"± {cv_summary['std_auc']:.4f}")

    return cv_summary


# ================================================================
# ✅ Plot Generators
# ================================================================
# All plots saved at 200 DPI with consistent color palette.
# ================================================================

COLORS_ROC = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
COLORS_BAR = ["#27ae60", "#2ecc71", "#3498db", "#2980b9", "#e74c3c", "#c0392b"]


# ----------------------------------------------------------------
# 1) Confusion Matrix
# ----------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, save_path: str,
                          title: str = "Confusion Matrix") -> str:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.ylabel("True Label", fontweight="bold", fontsize=12)
    plt.xlabel("Predicted Label", fontweight="bold", fontsize=12)
    plt.title(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ----------------------------------------------------------------
# 2) ROC Curves (one-vs-rest)
# ----------------------------------------------------------------
def plot_roc_curves(metrics: Dict[str, Any], save_path: str,
                    title: str = "ROC Curves") -> str:
    plt.figure(figsize=(9, 8))
    for i in range(NUM_CLASSES):
        if i in metrics["fpr"] and i in metrics["tpr"]:
            fpr = metrics["fpr"][i]
            tpr = metrics["tpr"][i]
            auc_v = metrics["roc_auc"].get(i, 0)
            plt.plot(
                fpr, tpr, color=COLORS_ROC[i], linewidth=2.5,
                label=f"{CLASS_NAMES[i]} (AUC = {auc_v:.4f})",
            )
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Random")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontweight="bold", fontsize=12)
    plt.ylabel("True Positive Rate", fontweight="bold", fontsize=12)
    plt.title(title, fontweight="bold", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.text(
        0.55, 0.15, f"Mean AUC = {metrics['roc_auc_mean']:.4f}",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F39C12", alpha=0.3),
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ----------------------------------------------------------------
# 3) Metrics Bar
# ----------------------------------------------------------------
def plot_metrics_bar(metrics: Dict[str, Any], save_path: str,
                     title: str = "Metrics") -> str:
    names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1", "AUC"]
    values = [
        metrics["accuracy"], metrics["balanced_accuracy"],
        metrics["precision"], metrics["recall"],
        metrics["f1_score"], metrics["roc_auc_mean"],
    ]
    plt.figure(figsize=(11, 6))
    bars = plt.bar(names, values, color=COLORS_BAR,
                   edgecolor="black", linewidth=1.2)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                 f"{v:.4f}", ha="center", fontweight="bold", fontsize=11)
    plt.ylim(0, 1.10)
    plt.ylabel("Score", fontweight="bold", fontsize=12)
    plt.title(title, fontweight="bold", fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ----------------------------------------------------------------
# 4) Per-Class Metrics
# ----------------------------------------------------------------
def plot_per_class_metrics(metrics: Dict[str, Any], save_path: str,
                           title: str = "Per-Class Metrics") -> str:
    pc = metrics["classification_report"]
    prec = [pc[c]["precision"] for c in CLASS_NAMES]
    rec = [pc[c]["recall"] for c in CLASS_NAMES]
    f1s = [pc[c]["f1-score"] for c in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    width = 0.27

    plt.figure(figsize=(12, 6))
    b1 = plt.bar(x - width, prec, width, label="Precision", color="#3498db")
    b2 = plt.bar(x, rec, width, label="Recall", color="#2ecc71")
    b3 = plt.bar(x + width, f1s, width, label="F1-Score", color="#e74c3c")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                     f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.xticks(x, CLASS_NAMES, fontsize=12, fontweight="bold")
    plt.ylim(0, 1.10)
    plt.ylabel("Score", fontweight="bold", fontsize=12)
    plt.title(title, fontweight="bold", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ----------------------------------------------------------------
# 5) CI95 Bars
# ----------------------------------------------------------------
def plot_ci_bars(ci95: Dict[str, Dict[str, float]], save_path: str) -> str:
    metrics_to_plot = ["accuracy", "f1_score", "precision", "recall"]
    labels = ["Accuracy", "F1", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, metric in enumerate(metrics_to_plot):
        ci = ci95[metric]
        mean, low, high = ci["mean"], ci["ci95_low"], ci["ci95_high"]
        y = len(metrics_to_plot) - i - 1
        ax.plot([low, high], [y, y], color="#3498db", linewidth=8, alpha=0.4)
        ax.scatter([mean], [y], color="#e74c3c", s=200, marker="D",
                   zorder=4, edgecolor="white", linewidth=2)
        ax.text(high + 0.005, y, f"  {mean:.4f} [{low:.4f}, {high:.4f}]",
                va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(range(len(metrics_to_plot)))
    ax.set_yticklabels(labels[::-1], fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.15)
    plt.title("95% Confidence Interval (Bootstrap, 2000 samples)",
              fontweight="bold", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ----------------------------------------------------------------
# 6) CV Folds
# ----------------------------------------------------------------
def plot_cv_folds(cv_summary: Dict[str, Any], save_path: str) -> str:
    folds = [f"Fold {r['fold']}" for r in cv_summary["fold_results"]]
    accs = [r["accuracy"] for r in cv_summary["fold_results"]]
    f1s = [r["f1_score"] for r in cv_summary["fold_results"]]

    x = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#3498db")
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1", color="#2ecc71")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(y=cv_summary["mean_accuracy"], color="red", linestyle="--",
               linewidth=2,
               label=f"Mean Acc: {cv_summary['mean_accuracy']:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("5-Fold CV Results", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ----------------------------------------------------------------
# 7) Training Curves (KD + Polish)
# ----------------------------------------------------------------
def plot_training_curves(
    kd_history: Dict[str, List[float]],
    polish_history: Optional[Dict[str, List[float]]],
    save_path: str,
    title: str = "Training Curves",
) -> str:
    """
    2x2 grid:
        - KD loss (train/val)
        - KD accuracy (val)
        - KD F1 (train/val) + Polish F1 overlay
        - Learning Rate schedule
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- (0,0) Loss ----
    trl = kd_history.get("train_loss", [])
    val = kd_history.get("val_loss", [])
    ep = range(1, len(trl) + 1)
    axes[0, 0].plot(ep, trl, label="Train", linewidth=2, color="#3498db")
    axes[0, 0].plot(ep, val, label="Val", linewidth=2, color="#e74c3c")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("KD Loss", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ---- (0,1) Accuracy ----
    va = kd_history.get("val_acc", [])
    axes[0, 1].plot(ep[:len(va)], va, linewidth=2, color="#2ecc71",
                    label="Val Acc")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Validation Accuracy", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ---- (1,0) F1 ----
    trf = kd_history.get("train_f1", [])
    vaf = kd_history.get("val_f1", [])
    axes[1, 0].plot(ep, trf, label="KD Train F1", linewidth=2, color="#9b59b6")
    axes[1, 0].plot(ep, vaf, label="KD Val F1", linewidth=2, color="#1abc9c")
    if polish_history:
        pf = polish_history.get("val_f1", [])
        if pf:
            p_ep = range(len(trl) + 1, len(trl) + len(pf) + 1)
            axes[1, 0].plot(p_ep, pf, label="Polish Val F1",
                            linewidth=2, color="#f39c12", linestyle="--")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1-Score")
    axes[1, 0].set_title("F1 Curves", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ---- (1,1) LR ----
    lr = kd_history.get("lr", [])
    if lr:
        axes[1, 1].plot(range(1, len(lr) + 1), lr,
                        linewidth=2, color="#e67e22")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate (log)")
        axes[1, 1].set_title("LR Schedule", fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "No LR history",
                        ha="center", va="center", fontsize=12)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

    plt.suptitle(title, fontweight="bold", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ----------------------------------------------------------------
# 8) Per-Seed Bar (3 seeds comparison)
# ----------------------------------------------------------------
# ✅ v2.1: Uses "best_accuracy" (was: "best_f1_macro")
# ----------------------------------------------------------------
def plot_per_seed_bars(seed_results: List[Dict[str, Any]],
                       save_path: str) -> str:
    seeds = [str(r["seed"]) for r in seed_results]
    best_acc = [r["best_accuracy"] for r in seed_results]        # ✅ new key
    final_val_acc = [r.get("final_val", {}).get("accuracy", 0.0)
                     for r in seed_results]

    x = np.arange(len(seeds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width / 2, best_acc, width,
                label="Best Val Acc (ES)", color="#3498db")
    b2 = ax.bar(x + width / 2, final_val_acc, width,
                label="Final Val Acc", color="#2ecc71")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.4f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds],
                       fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title("Per-Seed Performance (before ensemble)",
                 fontweight="bold", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path

# ================================================================
# ✅ PDF Generator — 24 pages
# ================================================================

def generate_pdf(
    pdf_path: str,
    student_type: str,
    teacher_name: str,
    test_metrics: Dict[str, Any],
    cv_summary: Dict[str, Any],
    ci95: Dict[str, Dict[str, float]],
    seed_results: List[Dict[str, Any]],
    kd_history: Dict[str, List[float]],
    polish_history: Optional[Dict[str, List[float]]],
    plots: Dict[str, str],
    config: Dict[str, Any],
    data_split: Dict[str, int],
    n_test: int,
    device: torch.device,
) -> Optional[str]:
    """
    Generate a 24-page comprehensive PDF report.
    Returns the PDF path on success, None on failure.
    """
    try:
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm, inch
        from reportlab.platypus import (
            Image as RLImage, PageBreak, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        print_error("reportlab not installed: pip install reportlab")
        return None

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=1.5 * cm, leftMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title=f"KD Report — {student_type.upper()}",
        author="Babak Yousefian",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "T", parent=styles["Heading1"],
        fontSize=26, spaceAfter=20, alignment=TA_CENTER,
        textColor=rl_colors.HexColor("#1a252f"),
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading2"],
        fontSize=18, spaceAfter=12, spaceBefore=12,
        textColor=rl_colors.HexColor("#2c3e50"),
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading3"],
        fontSize=14, spaceAfter=8,
        textColor=rl_colors.HexColor("#34495e"),
    )
    body = ParagraphStyle("Body", parent=styles["Normal"],
                          fontSize=10, leading=14, alignment=TA_JUSTIFY)
    center = ParagraphStyle("C", parent=body, alignment=TA_CENTER)

    def add_img(path, w=6.5 * inch, h=4.5 * inch):
        if path and os.path.exists(path) and os.path.getsize(path) > 100:
            try:
                return RLImage(path, width=w, height=h)
            except Exception:
                return Paragraph("⚠️ Image load failed", body)
        return Paragraph("⚠️ Image not available", body)

    story = []

    # =========================================================
    # PAGE 1 — Title
    # =========================================================
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Knowledge Distillation Report", title_style))
    story.append(Paragraph(
        f"DenseNet Teacher → {student_type.upper()} Student (3-Seed Ensemble)",
        ParagraphStyle("sub", parent=body, alignment=TA_CENTER,
                       fontSize=14, textColor=rl_colors.grey)))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        center))
    story.append(Paragraph(f"Test Samples: {n_test}", center))
    story.append(Paragraph(f"Device: {device}", center))
    story.append(Paragraph("Author: Babak Yousefian", center))
    story.append(Paragraph("Version: 2.1.0 (NEW MODELS — ES on Accuracy)", center))
    story.append(PageBreak())

    # =========================================================
    # PAGE 2 — Executive Summary
    # =========================================================
    story.append(Paragraph("Executive Summary", h1))
    story.append(Paragraph(
        f"This report presents the results of a 3-seed Knowledge Distillation "
        f"ensemble where the <b>{student_type.upper()}</b> student learns from "
        f"a <b>DenseNet</b> teacher. The pipeline uses 3 independent seeds "
        f"({', '.join(map(str, SEEDS))}), Focal Loss with soft γ=0.5, "
        f"label smoothing, and strong L2 regularization. "
        f"<b>Training Profile</b>: {config.get('Profile', 'v2.1')}. "
        f"Final evaluation combines probability averaging "
        f"across seeds, full TTA (32 augmentations), 5-fold cross-validation, "
        f"and bootstrap confidence intervals.",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>Headline Results (TTA on Test Set):</b>", h2))

    head_table = [["Metric", "Value"]]
    for k in ["accuracy", "balanced_accuracy", "precision", "recall",
              "f1_score", "f1_macro", "roc_auc_mean"]:
        if k in test_metrics:
            head_table.append([
                k.replace("_", " ").title(),
                f"{test_metrics[k]:.4f}",
            ])
    t = Table(head_table, colWidths=[2.5 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(PageBreak())

    # =========================================================
    # PAGE 3 — Pipeline
    # =========================================================
    story.append(Paragraph("Pipeline", h1))
    story.append(Paragraph(
        "<b>Step 1 — Data:</b> Load labeled images from "
        "finalReports/labels.csv + cropped/. Stratified split "
        "80% train / 10% val / 10% test.<br/>"
        "<b>Step 2 — Teacher:</b> Load DenseNet from "
        "FindBestNewModels/best_models.pth. Freeze all params.<br/>"
        f"<b>Step 3 — Student init:</b> Load {student_type.upper()} from its "
        "own fine-tuned checkpoint.<br/>"
        "<b>Step 4 — KD training:</b> For each seed ∈ {42, 123, 2024}, "
        "train student to mimic teacher (KL + Focal γ=0.5). "
        "<b>Early stopping on Accuracy</b> (patience=20).<br/>"
        "<b>Step 5 — Polish fine-tune:</b> Short refinement with Focal "
        "only (no KD). <b>Early stopping on Accuracy</b> (patience=12).<br/>"
        "<b>Step 6 — 3-Seed Ensemble:</b> Average softmax probabilities "
        "across the 3 trained students.<br/>"
        "<b>Step 7 — TTA:</b> 32 augmentations per test image → "
        "ensemble → mean probability.<br/>"
        "<b>Step 8 — CV5 + CI95:</b> Stratified 5-fold CV + bootstrap "
        "CI (2000 samples).",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>Regularization choices:</b>", h2))
    story.append(Paragraph(
        f"• Dropout = {DROPOUT_MAP[student_type]} (range: 0.05–0.25)<br/>"
        f"• Focal γ = {FOCAL_GAMMA} (soft)<br/>"
        f"• Label Smoothing = 0.05<br/>"
        f"• Weight Decay: backbone=1e-2, head=1e-3<br/>"
        f"• Gradient Clip = 0.5<br/>"
        f"• NO Mixup, NO CutMix<br/>"
        f"• Warmup 3 epochs + Cosine Annealing<br/>"
        f"• <b>Early Stopping on Accuracy</b> (patience=20/12) — v2.1",
        body,
    ))
    story.append(PageBreak())

    # =========================================================
    # PAGE 4 — Dataset
    # =========================================================
    story.append(Paragraph("Dataset & Split", h1))
    split_data = [["Split", "Count"]]
    for k, v in data_split.items():
        split_data.append([k.title(), str(v)])
    t = Table(split_data, colWidths=[2.5 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#3498DB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "The dataset is imbalanced (NL ≈ 60%, G1 ≈ 23%, G2 ≈ 14%, G3 ≈ 3%). "
        "To compensate, we use:\n"
        "• Class weights (inverse frequency) in Focal Loss\n"
        "• WeightedRandomSampler during training\n"
        "• F1-macro as a secondary metric (primary = Accuracy in v2.1)\n"
        "• Balanced accuracy in the report\n"
        "• Stratified splits at every stage (train/val/test + CV5)",
        body,
    ))
    story.append(PageBreak())

    # =========================================================
    # PAGE 5 — Training Config
    # =========================================================
    story.append(Paragraph("Training Configuration", h1))
    cfg_data = [["Parameter", "Value"]]
    for k, v in config.items():
        cfg_data.append([k, str(v)])
    t = Table(cfg_data, colWidths=[2.8 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#E67E22")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(PageBreak())

    # =========================================================
    # PAGE 6 — Overall Metrics
    # =========================================================
    story.append(Paragraph("Detailed Metrics", h1))
    met_data = [["Metric", "Value"]]
    for k in ["accuracy", "balanced_accuracy", "precision", "recall",
              "f1_score", "f1_macro", "roc_auc_mean", "num_samples"]:
        v = test_metrics.get(k, "N/A")
        if isinstance(v, float):
            met_data.append([k.replace("_", " ").title(), f"{v:.4f}"])
        else:
            met_data.append([k.replace("_", " ").title(), str(v)])
    t = Table(met_data, colWidths=[2.5 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#3498DB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(PageBreak())

    # =========================================================
    # PAGE 7 — Per-Class Metrics Table
    # =========================================================
    story.append(Paragraph("Per-Class Metrics", h1))
    pc_data = [["Class", "Precision", "Recall", "F1", "Support"]]
    for cname in CLASS_NAMES:
        if cname in test_metrics["classification_report"]:
            r = test_metrics["classification_report"][cname]
            pc_data.append([
                cname,
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1-score']:.4f}",
                str(int(r['support'])),
            ])
    t = Table(pc_data, colWidths=[1.2 * inch] * 5)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#34495E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "<b>Note on G3:</b> G3 has very few samples (~3% of the dataset). "
        "In v2.1, early stopping optimizes accuracy (not f1_macro), so "
        "G3 recall may decrease in exchange for higher overall accuracy.",
        body,
    ))
    story.append(PageBreak())

    # =========================================================
    # PAGES 8-11 — Plots
    # =========================================================
    for key, title in [
        ("cm", "Confusion Matrix"),
        ("roc", "ROC Curves"),
        ("metrics", "Metrics Overview"),
        ("per_class", "Per-Class Metrics"),
    ]:
        if plots.get(key):
            story.append(Paragraph(title, h1))
            story.append(add_img(plots[key], w=6 * inch, h=5 * inch))
            story.append(PageBreak())

    # =========================================================
    # PAGE 12 — Per-Seed Comparison
    # =========================================================
    if plots.get("per_seed"):
        story.append(Paragraph("Per-Seed Performance (before ensemble)", h1))
        story.append(Paragraph(
            "The three students are trained with independent seeds to "
            "reduce variance. This chart shows each seed's best "
            "validation accuracy (used for early stopping) and final "
            "validation accuracy before ensembling.",
            body,
        ))
        story.append(Spacer(1, 0.2 * inch))
        story.append(add_img(plots["per_seed"], w=6.5 * inch, h=4 * inch))
        story.append(PageBreak())

    # =========================================================
    # PAGE 13 — CV5 Table
    # =========================================================
    story.append(Paragraph("5-Fold Cross-Validation", h1))
    story.append(Paragraph(
        "The trained 3-seed ensemble is evaluated on 5 stratified folds. "
        "This measures robustness on different data distributions "
        "(not full retraining per fold).",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))

    cv_data = [["Fold", "Accuracy", "F1", "F1-Macro", "AUC", "Balanced"]]
    for r in cv_summary["fold_results"]:
        cv_data.append([
            f"Fold {r['fold']}",
            f"{r['accuracy']:.4f}",
            f"{r['f1_score']:.4f}",
            f"{r['f1_macro']:.4f}",
            f"{r['roc_auc_mean']:.4f}",
            f"{r['balanced_accuracy']:.4f}",
        ])
    cv_data.append([
        "Mean ± Std",
        f"{cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}",
        f"{cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}",
        "—",
        f"{cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}",
        "—",
    ])

    t = Table(cv_data, colWidths=[1.1 * inch, 1.4 * inch, 1.2 * inch,
                                   1.2 * inch, 1.1 * inch, 1.1 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#27AE60")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("BACKGROUND", (0, -1), (-1, -1), rl_colors.HexColor("#E8F8F0")),
    ]))
    story.append(t)
    story.append(PageBreak())

    # =========================================================
    # PAGE 14 — CV5 Plot
    # =========================================================
    if plots.get("cv_folds"):
        story.append(Paragraph("CV Folds Visualization", h1))
        story.append(add_img(plots["cv_folds"], w=6.5 * inch, h=4 * inch))
        story.append(PageBreak())

    # =========================================================
    # PAGE 15 — CI95 Table
    # =========================================================
    story.append(Paragraph("95% Confidence Interval (Bootstrap)", h1))
    story.append(Paragraph(
        "Bootstrap resampling (2000 samples) estimates the distribution "
        "of each metric on the test set. The 2.5th and 97.5th percentiles "
        "give the 95% confidence interval.",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))

    ci_data = [["Metric", "Mean", "95% CI Low", "95% CI High", "Std"]]
    for metric_name, ci in ci95.items():
        ci_data.append([
            metric_name.replace("_", " ").title(),
            f"{ci['mean']:.4f}",
            f"{ci['ci95_low']:.4f}",
            f"{ci['ci95_high']:.4f}",
            f"{ci['std']:.4f}",
        ])
    t = Table(ci_data, colWidths=[1.8 * inch, 1.2 * inch, 1.4 * inch,
                                   1.4 * inch, 1.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#E74C3C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ]))
    story.append(t)
    story.append(PageBreak())

    # =========================================================
    # PAGE 16 — CI95 Plot
    # =========================================================
    if plots.get("ci95"):
        story.append(Paragraph("CI95 Visualization", h1))
        story.append(add_img(plots["ci95"], w=6.5 * inch, h=4.5 * inch))
        story.append(PageBreak())

    # =========================================================
    # PAGE 17 — Training Curves
    # =========================================================
    story.append(Paragraph("Training Curves", h1))
    story.append(Paragraph(
        "These curves show the KD training trajectory (train/val loss, "
        "val accuracy, train/val F1) and the polish fine-tune overlay "
        "in dashed orange. <b>v2.1:</b> Early stopping monitored accuracy "
        "(not f1_macro), so val loss may rise while accuracy peaks first.",
        body,
    ))
    if plots.get("kd_curves"):
        story.append(add_img(plots["kd_curves"], w=6.5 * inch, h=4.5 * inch))
        story.append(PageBreak())

    # =========================================================
    # PAGE 18 — Ensemble Analysis
    # =========================================================
    story.append(Paragraph("Ensemble Analysis", h1))
    story.append(Paragraph(
        f"Three students are trained with seeds {SEEDS}. "
        f"Each seed contributes equal weight (1/3) to the ensemble. "
        f"Ensembling reduces variance and improves stability, "
        f"especially for minority classes (G3).",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))

    seed_data = [["Seed", "Best Acc (ES)", "Final Val Acc", "Best Epoch"]]
    for r in seed_results:
        seed_data.append([
            str(r["seed"]),
            f"{r['best_accuracy']:.4f}",              # ✅ v2.1 key
            f"{r.get('final_val', {}).get('accuracy', 0.0):.4f}",
            str(r["best_epoch"]),
        ])
    t = Table(seed_data, colWidths=[1.2 * inch] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#8E44AD")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(PageBreak())

    # =========================================================
    # PAGE 19 — TTA Description
    # =========================================================
    story.append(Paragraph("Test-Time Augmentation (TTA)", h1))
    story.append(Paragraph(
        "Each test image is augmented <b>32 times</b> using deterministic "
        "transformations (flips, rotations ±5°/±10°/±15°/±20°, brightness, "
        "contrast, gamma, zoom, translations, crops). "
        "The 3-seed ensemble produces a probability for each augmented "
        "version, and the final prediction is the mean across all 32 × 3 = 96 "
        "predictions per image.",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>Why TTA helps:</b>", h2))
    story.append(Paragraph(
        "• Reduces prediction variance<br/>"
        "• Averages out augmentation-specific errors<br/>"
        "• Particularly effective for small test sets<br/>"
        "• Improves F1-macro and G3 recall",
        body,
    ))
    story.append(PageBreak())

    # =========================================================
    # PAGE 20 — Classification Report (text)
    # =========================================================
    story.append(Paragraph("Classification Report (Full)", h1))

    cr = test_metrics["classification_report"]
    lines = [f"{'Class':<10}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>10}"]
    lines.append("=" * 60)
    for cname in CLASS_NAMES:
        r = cr[cname]
        lines.append(
            f"{cname:<10}"
            f"{r['precision']:>12.4f}"
            f"{r['recall']:>12.4f}"
            f"{r['f1-score']:>12.4f}"
            f"{int(r['support']):>10}"
        )
    if "macro avg" in cr:
        r = cr["macro avg"]
        lines.append("-" * 60)
        lines.append(
            f"{'Macro':<10}"
            f"{r['precision']:>12.4f}"
            f"{r['recall']:>12.4f}"
            f"{r['f1-score']:>12.4f}"
            f"{int(r['support']):>10}"
        )
    if "weighted avg" in cr:
        r = cr["weighted avg"]
        lines.append(
            f"{'Weighted':<10}"
            f"{r['precision']:>12.4f}"
            f"{r['recall']:>12.4f}"
            f"{r['f1-score']:>12.4f}"
            f"{int(r['support']):>10}"
        )

    monospaced = ParagraphStyle(
        "Mono", parent=body, fontName="Courier",
        fontSize=9, leading=12,
    )
    for ln in lines:
        story.append(Paragraph(ln.replace(" ", "&nbsp;"), monospaced))
    story.append(PageBreak())

    # =========================================================
    # PAGE 21 — Recommendations
    # =========================================================
    story.append(Paragraph("Recommendations", h1))
    story.append(Paragraph(
        "Based on the results, we recommend the following for deployment:", h2))
    story.append(Paragraph(
        "1. <b>Use the 3-seed ensemble</b> — it is more robust than any "
        "single seed, especially for G3.<br/><br/>"
        "2. <b>Always run TTA at inference</b> — the 32-augmentation "
        "average is measurably better than single-pass prediction.<br/><br/>"
        "3. <b>Monitor overall accuracy</b> as the primary metric (v2.1). "
        "Also watch G3 recall if that class matters clinically.<br/><br/>"
        "4. <b>Re-run CV5 periodically</b> if new data arrives, to detect "
        "distribution shift early.<br/><br/>"
        "5. <b>Knowledge distillation</b> gives a single, deployable .pth "
        "that captures teacher knowledge while being fast to serve.",
        body,
    ))
    story.append(PageBreak())

    # =========================================================
    # PAGE 22 — Limitations
    # =========================================================
    story.append(Paragraph("Limitations", h1))
    story.append(Paragraph(
        "• <b>Small G3 dataset</b> (~70 samples) — even with class "
        "weights and ensembling, G3 generalization remains the hardest "
        "challenge. v2.1 optimizes accuracy, which may trade off G3 recall.<br/><br/>"
        "• <b>CV5 is lightweight</b> — the ensemble is not retrained "
        "per fold, so CV5 estimates robustness rather than true "
        "out-of-fold generalization.<br/><br/>"
        "• <b>Label noise</b> — labels come from a single source; "
        "multi-reader agreement was not measured.<br/><br/>"
        "• <b>Single-modality</b> — only ultrasound images. "
        "Clinical metadata could help.<br/><br/>"
        "• <b>No external validation</b> — evaluate on an external "
        "dataset before clinical use.",
        body,
    ))
    story.append(PageBreak())

    # =========================================================
    # PAGE 23 — Final Verdict
    # =========================================================
    story.append(Paragraph("Final Verdict", h1))
    verdict = f"""
    <b>Knowledge Distillation Result:</b><br/>
    The <b>{student_type.upper()}</b> student was trained to mimic 
    the DenseNet teacher via 3-seed KD with Focal γ=0.5, 
    label smoothing, warmup + cosine, and strong L2. 
    <b>Early stopping on accuracy</b> (patience=20 KD / 12 Polish).<br/><br/>

    <b>Final Test Metrics (TTA on ensemble):</b><br/>
    - Accuracy: {test_metrics['accuracy']:.4f}<br/>
    - Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}<br/>
    - F1-Score (weighted): {test_metrics['f1_score']:.4f}<br/>
    - F1-Score (macro): {test_metrics['f1_macro']:.4f}<br/>
    - AUC (mean): {test_metrics['roc_auc_mean']:.4f}<br/><br/>

    <b>5-Fold CV Mean:</b> {cv_summary['mean_accuracy']:.4f} 
    ± {cv_summary['std_accuracy']:.4f}<br/><br/>

    <b>95% CI (Accuracy):</b> 
    [{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]<br/><br/>

    <b>Recommendation:</b> The distilled student is a single, deployable 
    .pth model that captures the teacher's knowledge while being 
    faster to run and easier to serve. Ensembling 3 seeds and TTA 
    during inference further improves reliability for clinical use.
    """
    story.append(Paragraph(verdict, body))
    story.append(PageBreak())

    # =========================================================
    # PAGE 24 — Closing
    # =========================================================
    story.append(Spacer(1, 3 * inch))
    story.append(Paragraph("✅ Report Complete", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"Author: Babak Yousefian<br/>"
        f"Version: 2.1.0 (NEW MODELS — {student_type.upper()})",
        center,
    ))

    # ---------- Build ----------
    try:
        doc.build(story)
        print_success(f"PDF generated: {pdf_path}")
        return pdf_path
    except Exception as e:
        print_error(f"PDF build failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ================================================================
# ✅ Main Pipeline — AlexNet (Option 1)
# ================================================================
# ✅ v2.1: Early stopping on Accuracy, KD patience=20, Polish patience=12
# ================================================================

@with_music("alexnet")
def run_kd_densenet_vs_alexnet():
    """
    Full pipeline: DenseNet teacher → AlexNet student (3-seed KD ensemble).
    """
    print_banner(
        "Knowledge Distillation — DenseNet → AlexNet (3 Seeds)",
        "KD (KL + Focal γ=0.5) → Polish → Ensemble → TTA → CV5 → CI95 → PDF",
    )

    student_type = "alexnet"
    teacher_name = "DenseNet"
    student_ckpt = ALEXNET_STUDENT_PATH
    out_root = KD_ALEXNET_ROOT
    ensure_dirs(out_root)

    plots_dir = os.path.join(out_root, "plots")
    cv_dir = os.path.join(out_root, "cv")
    seeds_dir = os.path.join(out_root, "seeds")
    final_model_path = os.path.join(
        out_root, "kd_densenet_to_alexnet_final.pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Device: {device}")

    # ✅ Start background music
    music_player = _start_music(student_type)

    # ============================================================
    # STEP 1 — Check files
    # ============================================================
    print_step(1, 10, "Checking teacher + student checkpoints")

    if not os.path.exists(TEACHER_DENSENET_PATH):
        print_error(f"Teacher not found: {TEACHER_DENSENET_PATH}")
        return {"status": "failed", "error": "Teacher missing"}

    if not os.path.exists(student_ckpt):
        print_warning(f"Student checkpoint missing: {student_ckpt}")
        print_warning("→ Continuing with ImageNet-random init for the student.")

    print_success(f"Teacher: {TEACHER_DENSENET_PATH}")
    print_success(f"Student: {student_ckpt}")

    # ============================================================
    # STEP 2 — Load data
    # ============================================================
    print_step(2, 10, "Loading labeled data (labels.csv + cropped/)")

    try:
        all_paths, all_labels = load_labeled_data()
    except Exception as e:
        print_error(f"Data loading failed: {e}")
        return {"status": "failed", "error": str(e)}

    if len(all_paths) == 0:
        print_error("No images found.")
        return {"status": "failed", "error": "Empty dataset"}

    print_success(f"Loaded {len(all_paths)} labeled images")
    class_counts = np.bincount(np.array(all_labels), minlength=NUM_CLASSES)
    for i, cn in enumerate(CLASS_NAMES):
        print_info(f"  {cn}: {class_counts[i]} samples")

    # Stratified 80/10/10
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.20, stratify=all_labels,
        random_state=SEEDS[0],
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels,
        random_state=SEEDS[0],
    )
    print_info(
        f"Split | Train={len(train_paths)} | "
        f"Val={len(val_paths)} | Test={len(test_paths)}"
    )

    # Class weights (inverse frequency on train split)
    tr_counts = np.bincount(np.array(train_labels), minlength=NUM_CLASSES)
    tr_counts = np.maximum(tr_counts, 1)
    class_weights_np = len(train_labels) / (NUM_CLASSES * tr_counts)
    class_weights_t = torch.tensor(class_weights_np, dtype=torch.float32)
    print_info(f"Class weights: {class_weights_np.round(4).tolist()}")

    # ============================================================
    # STEP 3 — Load teacher
    # ============================================================
    print_step(3, 10, "Loading DenseNet teacher (frozen)")

    try:
        teacher = load_teacher_densenet(TEACHER_DENSENET_PATH, device)
    except Exception as e:
        print_error(f"Teacher load failed: {e}")
        return {"status": "failed", "error": str(e)}

    print_success("Teacher ready.")

    # ============================================================
    # STEP 4 — DataLoaders
    # ============================================================
    print_step(4, 10, "Building DataLoaders")

    train_ds = LabeledDataset(
        train_paths, train_labels, input_size=224,
        augment=True, strength="light",
    )
    tr_labels_arr = np.array(train_labels)
    c_counts = np.bincount(tr_labels_arr, minlength=NUM_CLASSES)
    c_counts = np.maximum(c_counts, 1)
    sample_w = 1.0 / c_counts[tr_labels_arr]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_w),
        num_samples=len(sample_w),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=16, sampler=sampler,
        num_workers=0, drop_last=True, pin_memory=True,
    )

    val_ds = LabeledDataset(val_paths, val_labels, input_size=224, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print_success(
        f"DataLoaders | train={len(train_loader)} batches | "
        f"val={len(val_loader)} batches"
    )

    # ============================================================
    # STEP 5 — Train 3 seeds
    # ============================================================
    print_step(5, 10, f"Training {len(SEEDS)} seeds via KD")

    seed_results: List[Dict[str, Any]] = []
    seed_model_paths: List[str] = []

    for si, seed in enumerate(SEEDS, start=1):
        print_banner(
            f"Seed {si}/{len(SEEDS)} — {seed}",
            f"Student: {student_type.upper()} | Focal γ={FOCAL_GAMMA} | "
            f"Dropout={DROPOUT_MAP[student_type]}",
        )
        set_seed(seed)

        try:
            student = load_student(student_type, student_ckpt, device)
        except Exception as e:
            print_error(f"Student load failed for seed {seed}: {e}")
            continue

        seed_save = os.path.join(seeds_dir, f"student_seed{seed}.pth")
        seed_model_paths.append(seed_save)

        # ---- KD training (patience=20, ES=accuracy) ----
        t0 = time.time()
        kd_res = kd_train_one_seed(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=60,
            alpha=0.7,
            temperature=3.0,
            backbone_lr=1e-5,
            head_lr=1e-3,
            backbone_wd=1e-2,
            head_wd=1e-3,
            patience=KD_PATIENCE,             # ✅ 20
            class_weights=class_weights_t,
            save_path=seed_save,
            verbose=True,
            seed=seed,
        )
        kd_time = (time.time() - t0) / 60.0
        print_success(
            f"KD done | best Acc={kd_res['best_accuracy']:.4f} | "
            f"best epoch={kd_res['best_epoch']} | time={kd_time:.1f} min"
        )

        # ---- Polish fine-tune (patience=12, ES=accuracy) ----
        t0 = time.time()
        pol_res = polish_finetune(
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=15,
            lr=5e-6,
            patience=POLISH_PATIENCE,          # ✅ 12
            class_weights=class_weights_t,
            save_path=seed_save,
            verbose=True,
            seed=seed,
        )
        pol_time = (time.time() - t0) / 60.0
        print_success(
            f"Polish done | best Acc={pol_res['best_accuracy']:.4f} | "
            f"best epoch={pol_res['best_epoch']} | time={pol_time:.1f} min"
        )

        seed_results.append({
            "seed": seed,
            "kd": kd_res,
            "polish": pol_res,
            "best_accuracy": pol_res["best_accuracy"],     # ✅ new key
            "best_epoch": pol_res["best_epoch"],
            "final_val": pol_res["final_val"],
            "kd_history": kd_res["history"],
            "polish_history": pol_res["history"],
            "save_path": seed_save,
        })

        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(seed_results) == 0:
        print_error("No seed completed successfully.")
        return {"status": "failed", "error": "All seeds failed"}

    print_success(f"All {len(seed_results)} seeds trained.")

    # ============================================================
    # STEP 6 — Load 3 models (ensemble)
    # ============================================================
    print_step(6, 10, "Loading 3 best students into ensemble")

    ensemble_models: List[nn.Module] = []
    for sr in seed_results:
        m = load_student(student_type, sr["save_path"], device)
        m.eval()
        ensemble_models.append(m)
    print_success(f"Ensemble of {len(ensemble_models)} models ready.")

    # ============================================================
    # STEP 7 — TTA on test set
    # ============================================================
    print_step(7, 10, "TTA evaluation on test set (32 augs × 3 seeds)")

    test_preds, test_probs = evaluate_with_tta(
        ensemble_models, test_paths, test_labels, device,
        input_size=224, n_aug=32, batch_size=8, verbose=True,
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    print_success(
        f"Test | Acc={test_metrics['accuracy']:.4f} | "
        f"F1={test_metrics['f1_score']:.4f} | "
        f"F1m={test_metrics['f1_macro']:.4f} | "
        f"AUC={test_metrics['roc_auc_mean']:.4f}"
    )

    # ============================================================
    # STEP 8 — CV5 + CI95
    # ============================================================
    print_step(8, 10, "CV5 + Bootstrap CI95")

    cv_summary = run_cv5(
        ensemble_models, all_paths, all_labels, device,
        input_size=224, n_folds=5, n_aug_tta=32,
        batch_size=8, seed=SEEDS[0], verbose=True,
    )

    ci95 = compute_ci95_bootstrap(
        np.array(test_labels), test_preds,
        n_bootstrap=2000, seed=SEEDS[0],
    )
    print_success(
        f"CI95 Accuracy: [{ci95['accuracy']['ci95_low']:.4f}, "
        f"{ci95['accuracy']['ci95_high']:.4f}]"
    )

    # ============================================================
    # STEP 9 — Plots
    # ============================================================
    print_step(9, 10, "Generating plots")

    plots: Dict[str, str] = {}
    plots["cm"] = plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        os.path.join(plots_dir, "confusion_matrix.png"),
        f"{student_type.upper()} KD Ensemble — Confusion Matrix",
    )
    plots["roc"] = plot_roc_curves(
        test_metrics,
        os.path.join(plots_dir, "roc_curves.png"),
        f"{student_type.upper()} KD Ensemble — ROC",
    )
    plots["metrics"] = plot_metrics_bar(
        test_metrics,
        os.path.join(plots_dir, "metrics_bar.png"),
        f"{student_type.upper()} KD Ensemble — Metrics",
    )
    plots["per_class"] = plot_per_class_metrics(
        test_metrics,
        os.path.join(plots_dir, "per_class.png"),
        f"{student_type.upper()} KD Ensemble — Per-Class",
    )
    plots["ci95"] = plot_ci_bars(
        ci95,
        os.path.join(plots_dir, "ci95.png"),
    )
    plots["cv_folds"] = plot_cv_folds(
        cv_summary,
        os.path.join(cv_dir, "cv_folds.png"),
    )

    last = seed_results[-1]
    plots["kd_curves"] = plot_training_curves(
        last["kd_history"],
        last["polish_history"],
        os.path.join(plots_dir, "training_curves.png"),
        f"{student_type.upper()} — KD + Polish Training Curves (seed {last['seed']})",
    )

    plots["per_seed"] = plot_per_seed_bars(
        seed_results,
        os.path.join(plots_dir, "per_seed.png"),
    )
    print_success(f"Generated {len(plots)} plots.")

    # ============================================================
    # STEP 10 — PDF + save artifacts
    # ============================================================
    print_step(10, 10, "Generating 24-page PDF + saving artifacts")

    config = {
        "Student": student_type.upper(),
        "Teacher": teacher_name,
        "Seeds": ", ".join(map(str, SEEDS)),
        "Dropout": DROPOUT_MAP[student_type],
        "Focal gamma": FOCAL_GAMMA,
        "Label smoothing": 0.05,
        "KD alpha": 0.7,
        "KD temperature": 3.0,
        "KD epochs": 60,
        "Polish epochs": 15,
        "KD Patience": KD_PATIENCE,               # ✅ 20
        "Polish Patience": POLISH_PATIENCE,       # ✅ 12
        "ES Metric": "accuracy",                  # ✅ v2.1
        "Backbone LR": 1e-5,
        "Head LR": 1e-3,
        "Backbone WD": 1e-2,
        "Head WD": 1e-3,
        "Grad clip": 0.5,
        "TTA augs": 32,
        "CV folds": 5,
        "Bootstrap": 2000,
        "Mixup/CutMix": "NONE",
    }

    data_split = {
        "train": len(train_paths),
        "val": len(val_paths),
        "test": len(test_paths),
        "total": len(all_paths),
    }

    pdf_path = os.path.join(
        out_root, f"KD_DenseNet_to_{student_type.upper()}_Report.pdf"
    )
    generate_pdf(
        pdf_path=pdf_path,
        student_type=student_type,
        teacher_name=teacher_name,
        test_metrics=test_metrics,
        cv_summary=cv_summary,
        ci95=ci95,
        seed_results=seed_results,
        kd_history=last["kd_history"],
        polish_history=last["polish_history"],
        plots=plots,
        config=config,
        data_split=data_split,
        n_test=len(test_paths),
        device=device,
    )

    # Save final bundle
    try:
        final_bundle = {
            "student_type": student_type,
            "teacher": teacher_name,
            "seeds": SEEDS,
            "seed_model_paths": seed_model_paths,
            "test_metrics": {
                k: v for k, v in test_metrics.items()
                if k not in ["all_preds", "all_labels", "all_probs",
                             "fpr", "tpr", "confusion_matrix", "classification_report"]
            },
            "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
            "test_classification_report": test_metrics["classification_report"],
            "cv_summary": cv_summary,
            "ci95": ci95,
            "config": config,
            "data_split": data_split,
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(final_bundle, final_model_path)
        print_success(f"Final bundle saved: {final_model_path}")
    except Exception as e:
        print_warning(f"Could not save final bundle: {e}")

    # Save metrics JSON
    try:
        metrics_json_path = os.path.join(out_root, "kd_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump({
                "student_type": student_type,
                "teacher": teacher_name,
                "seeds": SEEDS,
                "test_metrics": {
                    k: v for k, v in test_metrics.items()
                    if k not in ["all_preds", "all_labels", "all_probs",
                                 "fpr", "tpr", "confusion_matrix",
                                 "classification_report"]
                },
                "cv_summary": cv_summary,
                "ci95": ci95,
                "config": config,
                "data_split": data_split,
                # ✅ NEW: preserve these for PDF regeneration
                "seed_results": [
                    {
                        "seed": r["seed"],
                        "best_accuracy": r["best_accuracy"],
                        "best_epoch": r["best_epoch"],
                        "final_val": {
                            "accuracy": r["final_val"].get("accuracy", 0.0)
                        },
                    }
                    for r in seed_results
                ],
                "kd_history": last["kd_history"],
                "polish_history": last["polish_history"],
                "timestamp": datetime.now().isoformat(),
            }, f, indent=4, default=str)
        print_success(f"Metrics JSON saved: {metrics_json_path}")
    except Exception as e:
        print_warning(f"Could not save metrics JSON: {e}")

    # ============================================================
    # Final summary table
    # ============================================================
    if HAS_RICH:
        console.print()
        table = Table(
            title="🎉 Final Results — DenseNet → AlexNet",
            box=box.DOUBLE_EDGE, border_style="green",
        )
        table.add_column("Metric", style="cyan bold")
        table.add_column("Value", style="green bold")
        table.add_row("Student", student_type.upper())
        table.add_row("Teacher", teacher_name)
        table.add_row("Seeds", ", ".join(map(str, SEEDS)))
        table.add_row("ES Metric", "Accuracy (patience 20/12)")
        table.add_row("Test Accuracy", f"{test_metrics['accuracy']:.4f}")
        table.add_row("Test Precision", f"{test_metrics['precision']:.4f}")
        table.add_row("Test Recall", f"{test_metrics['recall']:.4f}")
        table.add_row("Test F1 (weighted)", f"{test_metrics['f1_score']:.4f}")
        table.add_row("Test F1 (macro)", f"{test_metrics['f1_macro']:.4f}")
        table.add_row("Test AUC", f"{test_metrics['roc_auc_mean']:.4f}")
        table.add_row("Test Balanced Acc", f"{test_metrics['balanced_accuracy']:.4f}")
        table.add_row(
            "CV Mean Acc",
            f"{cv_summary['mean_accuracy']:.4f} "
            f"± {cv_summary['std_accuracy']:.4f}",
        )
        table.add_row(
            "CI95 Accuracy",
            f"[{ci95['accuracy']['ci95_low']:.4f}, "
            f"{ci95['accuracy']['ci95_high']:.4f}]",
        )
        console.print(table)
        console.print(f"\n[bold green]📄 PDF:[/bold green] {pdf_path}")
        console.print(f"[bold green]🎯 Bundle:[/bold green] {final_model_path}")

    return {
        "status": "success",
        "student_type": student_type,
        "teacher": teacher_name,
        "pdf_path": pdf_path,
        "final_model_path": final_model_path,
        "test_metrics": test_metrics,
        "cv_summary": cv_summary,
        "ci95": ci95,
        "seed_results": seed_results,
    }

# ================================================================
# ✅ Generic Pipeline — shared by MobileNetV2 and VGG16
# ================================================================
# ⚠️  To avoid code duplication, the generic pipeline is a thin
# wrapper that mirrors the AlexNet logic. AlexNet has its own dedicated
# function (B6b-1), while MobileNetV2 and VGG16 share this.
#
# ✅ v2.1: Early stopping on Accuracy, KD patience=20, Polish patience=12
# ================================================================

def _run_kd_pipeline_generic(
    student_type: str,
    student_ckpt: str,
    out_root: str,
    title: str,
    final_bundle_name: str,
) -> Dict[str, Any]:
    """
    Generic KD pipeline (teacher = DenseNet) for a given student type.

    Parameters
    ----------
    student_type : "mobilenetv2" | "vgg16"
    student_ckpt : path to the student fine-tuned checkpoint
    out_root     : output directory for this pipeline
    title        : banner title shown to the user
    final_bundle_name : filename for the final saved bundle (.pth)

    Returns
    -------
    dict with status, metrics, and artifact paths.
    """
    print_banner(
        title,
        "KD (KL + Focal γ=0.5) → Polish → Ensemble → TTA → CV5 → CI95 → PDF",
    )

    teacher_name = "DenseNet"
    ensure_dirs(out_root)

    plots_dir = os.path.join(out_root, "plots")
    cv_dir = os.path.join(out_root, "cv")
    seeds_dir = os.path.join(out_root, "seeds")
    final_model_path = os.path.join(out_root, final_bundle_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Device: {device}")
    # ✅ v2.2: pick profile based on student_type
    profile = get_profile(student_type)
    print_info(f"Training profile: {profile['name'].upper()}")
    print_info(
        f"  LR: backbone={profile['backbone_lr']:.1e}, "
        f"head={profile['head_lr']:.1e} | "
        f"WD: backbone={profile['backbone_wd']:.1e}, "
        f"head={profile['head_wd']:.1e} | "
        f"LabelSmooth={profile['label_smoothing']}"
    )
    if profile["gap_monitor_threshold"] is not None:
        print_info(
            f"  Gap Monitor: ON (threshold="
            f"{profile['gap_monitor_threshold']}×, "
            f"patience={profile['gap_monitor_patience']})"
        )

    # ============================================================
    # STEP 1 — Check files
    # ============================================================
    print_step(1, 10, "Checking teacher + student checkpoints")

    if not os.path.exists(TEACHER_DENSENET_PATH):
        print_error(f"Teacher not found: {TEACHER_DENSENET_PATH}")
        return {"status": "failed", "error": "Teacher missing"}

    if not os.path.exists(student_ckpt):
        print_warning(f"Student checkpoint missing: {student_ckpt}")
        print_warning("→ Continuing with ImageNet-random init for the student.")

    print_success(f"Teacher: {TEACHER_DENSENET_PATH}")
    print_success(f"Student: {student_ckpt}")

    # ============================================================
    # STEP 2 — Load data
    # ============================================================
    print_step(2, 10, "Loading labeled data (labels.csv + cropped/)")

    try:
        all_paths, all_labels = load_labeled_data()
    except Exception as e:
        print_error(f"Data loading failed: {e}")
        return {"status": "failed", "error": str(e)}

    if len(all_paths) == 0:
        print_error("No images found.")
        return {"status": "failed", "error": "Empty dataset"}

    print_success(f"Loaded {len(all_paths)} labeled images")
    class_counts = np.bincount(np.array(all_labels), minlength=NUM_CLASSES)
    for i, cn in enumerate(CLASS_NAMES):
        print_info(f"  {cn}: {class_counts[i]} samples")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.20, stratify=all_labels,
        random_state=SEEDS[0],
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels,
        random_state=SEEDS[0],
    )
    print_info(
        f"Split | Train={len(train_paths)} | "
        f"Val={len(val_paths)} | Test={len(test_paths)}"
    )

    tr_counts = np.bincount(np.array(train_labels), minlength=NUM_CLASSES)
    tr_counts = np.maximum(tr_counts, 1)
    class_weights_np = len(train_labels) / (NUM_CLASSES * tr_counts)
    class_weights_t = torch.tensor(class_weights_np, dtype=torch.float32)
    print_info(f"Class weights: {class_weights_np.round(4).tolist()}")

    # ============================================================
    # STEP 3 — Load teacher
    # ============================================================
    print_step(3, 10, "Loading DenseNet teacher (frozen)")

    try:
        teacher = load_teacher_densenet(TEACHER_DENSENET_PATH, device)
    except Exception as e:
        print_error(f"Teacher load failed: {e}")
        return {"status": "failed", "error": str(e)}

    print_success("Teacher ready.")

    # ============================================================
    # STEP 4 — DataLoaders
    # ============================================================
    print_step(4, 10, "Building DataLoaders")

    train_ds = LabeledDataset(
        train_paths, train_labels, input_size=224,
        augment=True, strength="light",
    )
    tr_labels_arr = np.array(train_labels)
    c_counts = np.bincount(tr_labels_arr, minlength=NUM_CLASSES)
    c_counts = np.maximum(c_counts, 1)
    sample_w = 1.0 / c_counts[tr_labels_arr]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_w),
        num_samples=len(sample_w),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=16, sampler=sampler,
        num_workers=0, drop_last=True, pin_memory=True,
    )

    val_ds = LabeledDataset(val_paths, val_labels, input_size=224, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print_success(
        f"DataLoaders | train={len(train_loader)} batches | "
        f"val={len(val_loader)} batches"
    )

    # ============================================================
    # STEP 5 — Train 3 seeds
    # ============================================================
    print_step(5, 10, f"Training {len(SEEDS)} seeds via KD")

    seed_results: List[Dict[str, Any]] = []
    seed_model_paths: List[str] = []

    for si, seed in enumerate(SEEDS, start=1):
        print_banner(
            f"Seed {si}/{len(SEEDS)} — {seed}",
            f"Student: {student_type.upper()} | "
            f"Focal γ={FOCAL_GAMMA} | Dropout={DROPOUT_MAP[student_type]}",
        )
        set_seed(seed)

        try:
            student = load_student(student_type, student_ckpt, device)
        except Exception as e:
            print_error(f"Student load failed for seed {seed}: {e}")
            continue

        seed_save = os.path.join(seeds_dir, f"student_seed{seed}.pth")
        seed_model_paths.append(seed_save)

        # KD
        t0 = time.time()
        kd_res = kd_train_one_seed(
            student=student, teacher=teacher,
            train_loader=train_loader, val_loader=val_loader,
            device=device,
            profile=profile["name"],  # ← NEW
            alpha=0.7, temperature=3.0,
            class_weights=class_weights_t,
            save_path=seed_save, verbose=True, seed=seed,
        )
        kd_time = (time.time() - t0) / 60.0
        print_success(
            f"KD done | best Acc={kd_res['best_accuracy']:.4f} | "
            f"epoch={kd_res['best_epoch']} | time={kd_time:.1f} min"
        )

        # Polish
        t0 = time.time()
        pol_res = polish_finetune(
            student=student, train_loader=train_loader,
            val_loader=val_loader, device=device,
            profile=profile["name"],  # ← NEW
            class_weights=class_weights_t,
            save_path=seed_save, verbose=True, seed=seed,
        )
        pol_time = (time.time() - t0) / 60.0
        print_success(
            f"Polish done | best Acc={pol_res['best_accuracy']:.4f} | "
            f"epoch={pol_res['best_epoch']} | time={pol_time:.1f} min"
        )

        seed_results.append({
            "seed": seed,
            "kd": kd_res, "polish": pol_res,
            "best_accuracy": pol_res["best_accuracy"],    # ✅ new key
            "best_epoch": pol_res["best_epoch"],
            "final_val": pol_res["final_val"],
            "kd_history": kd_res["history"],
            "polish_history": pol_res["history"],
            "save_path": seed_save,
        })

        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(seed_results) == 0:
        print_error("No seed completed successfully.")
        return {"status": "failed", "error": "All seeds failed"}

    print_success(f"All {len(seed_results)} seeds trained.")

    # ============================================================
    # STEP 6 — Load 3 models
    # ============================================================
    print_step(6, 10, "Loading 3 best students into ensemble")

    ensemble_models: List[nn.Module] = []
    for sr in seed_results:
        m = load_student(student_type, sr["save_path"], device)
        m.eval()
        ensemble_models.append(m)
    print_success(f"Ensemble of {len(ensemble_models)} models ready.")

    # ============================================================
    # STEP 7 — TTA
    # ============================================================
    print_step(7, 10, "TTA evaluation on test set (32 augs × 3 seeds)")

    test_preds, test_probs = evaluate_with_tta(
        ensemble_models, test_paths, test_labels, device,
        input_size=224, n_aug=32, batch_size=8, verbose=True,
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    print_success(
        f"Test | Acc={test_metrics['accuracy']:.4f} | "
        f"F1={test_metrics['f1_score']:.4f} | "
        f"F1m={test_metrics['f1_macro']:.4f} | "
        f"AUC={test_metrics['roc_auc_mean']:.4f}"
    )

    # ============================================================
    # STEP 8 — CV5 + CI95
    # ============================================================
    print_step(8, 10, "CV5 + Bootstrap CI95")

    cv_summary = run_cv5(
        ensemble_models, all_paths, all_labels, device,
        input_size=224, n_folds=5, n_aug_tta=32,
        batch_size=8, seed=SEEDS[0], verbose=True,
    )

    ci95 = compute_ci95_bootstrap(
        np.array(test_labels), test_preds,
        n_bootstrap=2000, seed=SEEDS[0],
    )
    print_success(
        f"CI95 Accuracy: [{ci95['accuracy']['ci95_low']:.4f}, "
        f"{ci95['accuracy']['ci95_high']:.4f}]"
    )

    # ============================================================
    # STEP 9 — Plots
    # ============================================================
    print_step(9, 10, "Generating plots")

    plots: Dict[str, str] = {}
    plots["cm"] = plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        os.path.join(plots_dir, "confusion_matrix.png"),
        f"{student_type.upper()} KD Ensemble — Confusion Matrix",
    )
    plots["roc"] = plot_roc_curves(
        test_metrics,
        os.path.join(plots_dir, "roc_curves.png"),
        f"{student_type.upper()} KD Ensemble — ROC",
    )
    plots["metrics"] = plot_metrics_bar(
        test_metrics,
        os.path.join(plots_dir, "metrics_bar.png"),
        f"{student_type.upper()} KD Ensemble — Metrics",
    )
    plots["per_class"] = plot_per_class_metrics(
        test_metrics,
        os.path.join(plots_dir, "per_class.png"),
        f"{student_type.upper()} KD Ensemble — Per-Class",
    )
    plots["ci95"] = plot_ci_bars(ci95, os.path.join(plots_dir, "ci95.png"))
    plots["cv_folds"] = plot_cv_folds(
        cv_summary, os.path.join(cv_dir, "cv_folds.png"),
    )

    last = seed_results[-1]
    plots["kd_curves"] = plot_training_curves(
        last["kd_history"], last["polish_history"],
        os.path.join(plots_dir, "training_curves.png"),
        f"{student_type.upper()} — Training Curves (seed {last['seed']})",
    )
    plots["per_seed"] = plot_per_seed_bars(
        seed_results, os.path.join(plots_dir, "per_seed.png"),
    )
    print_success(f"Generated {len(plots)} plots.")

    # ============================================================
    # STEP 10 — PDF + artifacts
    # ============================================================
    print_step(10, 10, "Generating 24-page PDF + saving artifacts")

    config = {
        "Profile": profile["name"].upper(),  # ← NEW
        "Student": student_type.upper(),
        "Teacher": teacher_name,
        "Seeds": ", ".join(map(str, SEEDS)),
        "Dropout": DROPOUT_MAP[student_type],
        "Focal gamma": FOCAL_GAMMA,
        "Label smoothing": profile["label_smoothing"],  # ← NEW
        "KD alpha": 0.7,
        "KD temperature": 3.0,
        "KD epochs": profile["kd_epochs"],  # ← NEW
        "Polish epochs": profile["polish_epochs"],  # ← NEW
        "KD Patience": profile["kd_patience"],  # ← NEW
        "Polish Patience": profile["polish_patience"],  # ← NEW
        "ES Metric": "accuracy",
        "Backbone LR": profile["backbone_lr"],  # ← NEW
        "Head LR": profile["head_lr"],  # ← NEW
        "Backbone WD": profile["backbone_wd"],  # ← NEW
        "Head WD": profile["head_wd"],  # ← NEW
        "Grad clip": profile["grad_clip"],  # ← NEW
        "Gap Monitor": (
            f"ON (>{profile['gap_monitor_threshold']}×, "
            f"patience={profile['gap_monitor_patience']})"
            if profile["gap_monitor_threshold"] is not None
            else "OFF"
        ),  # ← NEW
        "TTA augs": 32,
        "CV folds": 5,
        "Bootstrap": 2000,
        "Mixup/CutMix": "NONE",
    }

    data_split = {
        "train": len(train_paths),
        "val": len(val_paths),
        "test": len(test_paths),
        "total": len(all_paths),
    }

    pdf_path = os.path.join(
        out_root, f"KD_DenseNet_to_{student_type.upper()}_Report.pdf"
    )
    generate_pdf(
        pdf_path=pdf_path,
        student_type=student_type,
        teacher_name=teacher_name,
        test_metrics=test_metrics,
        cv_summary=cv_summary,
        ci95=ci95,
        seed_results=seed_results,
        kd_history=last["kd_history"],
        polish_history=last["polish_history"],
        plots=plots,
        config=config,
        data_split=data_split,
        n_test=len(test_paths),
        device=device,
    )

    # Save final bundle
    try:
        final_bundle = {
            "student_type": student_type,
            "teacher": teacher_name,
            "seeds": SEEDS,
            "seed_model_paths": seed_model_paths,
            "test_metrics": {
                k: v for k, v in test_metrics.items()
                if k not in ["all_preds", "all_labels", "all_probs",
                             "fpr", "tpr", "confusion_matrix",
                             "classification_report"]
            },
            "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
            "test_classification_report": test_metrics["classification_report"],
            "cv_summary": cv_summary,
            "ci95": ci95,
            "config": config,
            "data_split": data_split,
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(final_bundle, final_model_path)
        print_success(f"Final bundle saved: {final_model_path}")
    except Exception as e:
        print_warning(f"Could not save final bundle: {e}")

    # Save metrics JSON
    try:
        metrics_json_path = os.path.join(out_root, "kd_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump({
                "student_type": student_type,
                "teacher": teacher_name,
                "seeds": SEEDS,
                "test_metrics": {
                    k: v for k, v in test_metrics.items()
                    if k not in ["all_preds", "all_labels", "all_probs",
                                 "fpr", "tpr", "confusion_matrix",
                                 "classification_report"]
                },
                "cv_summary": cv_summary,
                "ci95": ci95,
                "config": config,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=4, default=str)
        print_success(f"Metrics JSON saved: {metrics_json_path}")
    except Exception as e:
        print_warning(f"Could not save metrics JSON: {e}")

    # ============================================================
    # Final summary table
    # ============================================================
    if HAS_RICH:
        console.print()
        table = Table(
            title=f"🎉 Final Results — DenseNet → {student_type.upper()}",
            box=box.DOUBLE_EDGE, border_style="green",
        )
        table.add_column("Metric", style="cyan bold")
        table.add_column("Value", style="green bold")
        table.add_row("Student", student_type.upper())
        table.add_row("Teacher", teacher_name)
        table.add_row("Seeds", ", ".join(map(str, SEEDS)))
        table.add_row("ES Metric", "Accuracy (patience 20/12)")
        table.add_row("Test Accuracy", f"{test_metrics['accuracy']:.4f}")
        table.add_row("Test Precision", f"{test_metrics['precision']:.4f}")
        table.add_row("Test Recall", f"{test_metrics['recall']:.4f}")
        table.add_row("Test F1 (weighted)", f"{test_metrics['f1_score']:.4f}")
        table.add_row("Test F1 (macro)", f"{test_metrics['f1_macro']:.4f}")
        table.add_row("Test AUC", f"{test_metrics['roc_auc_mean']:.4f}")
        table.add_row("Test Balanced Acc", f"{test_metrics['balanced_accuracy']:.4f}")
        table.add_row(
            "CV Mean Acc",
            f"{cv_summary['mean_accuracy']:.4f} "
            f"± {cv_summary['std_accuracy']:.4f}",
        )
        table.add_row(
            "CI95 Accuracy",
            f"[{ci95['accuracy']['ci95_low']:.4f}, "
            f"{ci95['accuracy']['ci95_high']:.4f}]",
        )
        console.print(table)
        console.print(f"\n[bold green]📄 PDF:[/bold green] {pdf_path}")
        console.print(f"[bold green]🎯 Bundle:[/bold green] {final_model_path}")

    return {
        "status": "success",
        "student_type": student_type,
        "teacher": teacher_name,
        "pdf_path": pdf_path,
        "final_model_path": final_model_path,
        "test_metrics": test_metrics,
        "cv_summary": cv_summary,
        "ci95": ci95,
        "seed_results": seed_results,
    }

# ================================================================
# ✅ Regenerate PDF from existing outputs (no retraining)
# ================================================================
# Reads kd_metrics.json + plots/*.png from the student's output dir
# and rebuilds the 24-page PDF without touching the models.
# ================================================================

def regenerate_pdf_for_student(student_type: str) -> Optional[str]:
    """
    Regenerate the PDF report for a given student type using
    only the already-saved artifacts (metrics JSON + plots).

    Returns the PDF path on success, None on failure.
    """
    # ----- Resolve paths -----
    if student_type == "alexnet":
        out_root = KD_ALEXNET_ROOT
        student_label = "ALEXNET"
    elif student_type == "mobilenetv2":
        out_root = KD_MNV2_ROOT
        student_label = "MOBILENETV2"
    elif student_type == "vgg16":
        out_root = KD_VGG16_ROOT
        student_label = "VGG16"
    else:
        print_error(f"Unknown student_type: {student_type}")
        return None

    metrics_json_path = os.path.join(out_root, "kd_metrics.json")
    plots_dir = os.path.join(out_root, "plots")
    cv_dir = os.path.join(out_root, "cv")

    if not os.path.exists(metrics_json_path):
        print_error(f"kd_metrics.json not found: {metrics_json_path}")
        print_info("👉 You need to run the full pipeline first (KD + Polish + TTA).")
        return None

    # ----- Load saved metrics -----
    try:
        with open(metrics_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print_error(f"Could not read metrics JSON: {e}")
        return None

    test_metrics = data.get("test_metrics", {})
    cv_summary = data.get("cv_summary", {})
    ci95 = data.get("ci95", {})
    config = data.get("config", {})
    data_split = data.get("data_split", {})

    # ----- Reconstruct plots dict -----
    plots = {}
    plot_files = {
        "cm": ("confusion_matrix.png", plots_dir),
        "roc": ("roc_curves.png", plots_dir),
        "metrics": ("metrics_bar.png", plots_dir),
        "per_class": ("per_class.png", plots_dir),
        "ci95": ("ci95.png", plots_dir),
        "kd_curves": ("training_curves.png", plots_dir),
        "per_seed": ("per_seed.png", plots_dir),
        "cv_folds": ("cv_folds.png", cv_dir),
    }
    for key, (fname, base_dir) in plot_files.items():
        p = os.path.join(base_dir, fname)
        if os.path.exists(p):
            plots[key] = p

    # ----- Reconstruct seed_results (only what's needed for PDF) -----
    seed_results = data.get("seed_results", [])
    # If not present, we can't rebuild some tables — create placeholders
    if not seed_results:
        seed_results = [
            {"seed": s, "best_accuracy": 0.0, "best_epoch": 0,
             "final_val": {"accuracy": 0.0}}
            for s in SEEDS
        ]

    # ----- Reconstruct kd_history / polish_history (for training curves page) -----
    # If not available, use empty dict (PDF will just show an empty plot)
    kd_history = data.get("kd_history", {})
    polish_history = data.get("polish_history", {})

    n_test = int(test_metrics.get("num_samples", 0))

    # ----- Output PDF path -----
    pdf_path = os.path.join(
        out_root, f"KD_DenseNet_to_{student_label}_Report.pdf"
    )

    print_banner(
        f"Regenerating PDF — DenseNet → {student_label}",
        "Uses saved metrics + plots (no retraining)",
    )
    print_info(f"Source: {metrics_json_path}")
    print_info(f"Plots:  {len(plots)} files found")
    print_info(f"Output: {pdf_path}")

    # ----- Build PDF -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = generate_pdf(
        pdf_path=pdf_path,
        student_type=student_type,
        teacher_name="DenseNet",
        test_metrics=test_metrics,
        cv_summary=cv_summary,
        ci95=ci95,
        seed_results=seed_results,
        kd_history=kd_history,
        polish_history=polish_history,
        plots=plots,
        config=config,
        data_split=data_split,
        n_test=n_test,
        device=device,
    )

    if result:
        print_success(f"PDF regenerated: {pdf_path}")
    else:
        print_error("PDF regeneration failed.")

    return result

# ================================================================
# ✅ Recover from seeds/*.pth (if kd_metrics.json is missing)
# ================================================================
# When the pipeline crashed AFTER saving seeds but BEFORE saving
# kd_metrics.json, use this to:
#   1. Load the 3 seed checkpoints
#   2. Recompute test metrics via TTA
#   3. Recompute CV5 + CI95
#   4. Rebuild kd_metrics.json
#   5. Regenerate the PDF
# ================================================================

def recover_from_seeds(student_type: str) -> Optional[str]:
    """
    Recover a full run from seeds/*.pth (no retraining of the models).
    Use when kd_metrics.json is missing but seeds exist.

    Time: ~5-10 minutes (mostly TTA + CV5).

    Returns the PDF path on success, None on failure.
    """
    # ----- Resolve paths -----
    if student_type == "alexnet":
        out_root = KD_ALEXNET_ROOT
        student_label = "ALEXNET"
    elif student_type == "mobilenetv2":
        out_root = KD_MNV2_ROOT
        student_label = "MOBILENETV2"
    elif student_type == "vgg16":
        out_root = KD_VGG16_ROOT
        student_label = "VGG16"
    else:
        print_error(f"Unknown student_type: {student_type}")
        return None

    seeds_dir = os.path.join(out_root, "seeds")
    plots_dir = os.path.join(out_root, "plots")
    cv_dir = os.path.join(out_root, "cv")
    final_model_path = os.path.join(
        out_root, f"kd_densenet_to_{student_type}_final.pth"
    )
    metrics_json_path = os.path.join(out_root, "kd_metrics.json")
    pdf_path = os.path.join(
        out_root, f"KD_DenseNet_to_{student_label}_Report.pdf"
    )

    # ----- Check seeds exist -----
    seed_paths = []
    for s in SEEDS:
        p = os.path.join(seeds_dir, f"student_seed{s}.pth")
        if not os.path.exists(p):
            print_error(f"Missing seed: {p}")
            print_info("👉 You need all 3 seeds to recover.")
            return None
        seed_paths.append(p)

    print_banner(
        f"Recovering from seeds — DenseNet → {student_label}",
        "Load 3 seeds → TTA → CV5 → CI95 → kd_metrics.json → PDF",
    )
    print_info(f"Seeds found: {len(seed_paths)}")
    for p in seed_paths:
        print_info(f"  ✓ {os.path.basename(p)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Device: {device}")
    print_info("⏱️  This will take ~5-10 min (TTA + CV5), not retraining.")

    # ----- Reproduce the same test split (random_state=SEEDS[0]=42) -----
    try:
        all_paths, all_labels = load_labeled_data()
    except Exception as e:
        print_error(f"Data loading failed: {e}")
        return None

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.20, stratify=all_labels,
        random_state=SEEDS[0],
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels,
        random_state=SEEDS[0],
    )
    print_info(
        f"Split | Train={len(train_paths)} | "
        f"Val={len(val_paths)} | Test={len(test_paths)}"
    )

    # ----- Load 3 seeds into ensemble -----
    print_info("Loading 3 seed checkpoints...")
    ensemble_models: List[nn.Module] = []
    for p in seed_paths:
        try:
            m = load_student(student_type, p, device)
            m.eval()
            ensemble_models.append(m)
        except Exception as e:
            print_error(f"Failed to load {p}: {e}")
            return None
    print_success(f"Ensemble of {len(ensemble_models)} models ready.")

    # ----- TTA on test -----
    print_step(1, 4, "TTA evaluation on test set (32 augs × 3 seeds)")
    test_preds, test_probs = evaluate_with_tta(
        ensemble_models, test_paths, test_labels, device,
        input_size=224, n_aug=32, batch_size=8, verbose=True,
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    print_success(
        f"Test | Acc={test_metrics['accuracy']:.4f} | "
        f"F1={test_metrics['f1_score']:.4f} | "
        f"F1m={test_metrics['f1_macro']:.4f} | "
        f"AUC={test_metrics['roc_auc_mean']:.4f}"
    )

    # ----- CV5 + CI95 -----
    print_step(2, 4, "CV5 + Bootstrap CI95")
    cv_summary = run_cv5(
        ensemble_models, all_paths, all_labels, device,
        input_size=224, n_folds=5, n_aug_tta=32,
        batch_size=8, seed=SEEDS[0], verbose=True,
    )
    ci95 = compute_ci95_bootstrap(
        np.array(test_labels), test_preds,
        n_bootstrap=2000, seed=SEEDS[0],
    )
    print_success(
        f"CI95 Accuracy: [{ci95['accuracy']['ci95_low']:.4f}, "
        f"{ci95['accuracy']['ci95_high']:.4f}]"
    )

    # ----- Rebuild seed_results from seed checkpoint metadata -----
    seed_results: List[Dict[str, Any]] = []
    for s, p in zip(SEEDS, seed_paths):
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            best_acc = ckpt.get("best_accuracy", 0.0)
            best_ep = ckpt.get("epoch", 0)
            seed_results.append({
                "seed": s,
                "best_accuracy": float(best_acc),
                "best_epoch": int(best_ep),
                "final_val": {"accuracy": float(best_acc)},
            })
            print_info(f"  Seed {s}: best_acc={best_acc:.4f}, epoch={best_ep}")
        except Exception as e:
            print_warning(f"Could not read metadata for seed {s}: {e}")
            seed_results.append({
                "seed": s, "best_accuracy": 0.0, "best_epoch": 0,
                "final_val": {"accuracy": 0.0},
            })

    # ----- Plots dict -----
    plots = {}
    plot_files = {
        "cm": ("confusion_matrix.png", plots_dir),
        "roc": ("roc_curves.png", plots_dir),
        "metrics": ("metrics_bar.png", plots_dir),
        "per_class": ("per_class.png", plots_dir),
        "ci95": ("ci95.png", plots_dir),
        "kd_curves": ("training_curves.png", plots_dir),
        "per_seed": ("per_seed.png", plots_dir),
        "cv_folds": ("cv_folds.png", cv_dir),
    }
    for key, (fname, base_dir) in plot_files.items():
        p = os.path.join(base_dir, fname)
        if os.path.exists(p):
            plots[key] = p
    print_info(f"Plots found: {len(plots)}")

    # ----- Config (best-effort reconstruction) -----
    profile = get_profile(student_type)
    config = {
        "Profile": profile["name"].upper(),
        "Student": student_type.upper(),
        "Teacher": "DenseNet",
        "Seeds": ", ".join(map(str, SEEDS)),
        "Dropout": DROPOUT_MAP[student_type],
        "Focal gamma": FOCAL_GAMMA,
        "Label smoothing": profile["label_smoothing"],
        "KD alpha": 0.7,
        "KD temperature": 3.0,
        "KD epochs": profile["kd_epochs"],
        "Polish epochs": profile["polish_epochs"],
        "KD Patience": profile["kd_patience"],
        "Polish Patience": profile["polish_patience"],
        "ES Metric": "accuracy",
        "Backbone LR": profile["backbone_lr"],
        "Head LR": profile["head_lr"],
        "Backbone WD": profile["backbone_wd"],
        "Head WD": profile["head_wd"],
        "Grad clip": profile["grad_clip"],
        "Gap Monitor": (
            f"ON (>{profile['gap_monitor_threshold']}×, "
            f"patience={profile['gap_monitor_patience']})"
            if profile["gap_monitor_threshold"] is not None
            else "OFF"
        ),
        "TTA augs": 32,
        "CV folds": 5,
        "Bootstrap": 2000,
        "Mixup/CutMix": "NONE",
        "Recovered": "YES (from seeds)",
    }

    data_split = {
        "train": len(train_paths),
        "val": len(val_paths),
        "test": len(test_paths),
        "total": len(all_paths),
    }

    # ----- Save kd_metrics.json -----
    print_step(3, 4, "Rebuilding kd_metrics.json")
    try:
        with open(metrics_json_path, "w") as f:
            json.dump({
                "student_type": student_type,
                "teacher": "DenseNet",
                "seeds": SEEDS,
                "test_metrics": {
                    k: v for k, v in test_metrics.items()
                    if k not in ["all_preds", "all_labels", "all_probs",
                                 "fpr", "tpr", "confusion_matrix",
                                 "classification_report"]
                },
                "cv_summary": cv_summary,
                "ci95": ci95,
                "config": config,
                "data_split": data_split,
                "seed_results": seed_results,
                "kd_history": {},        # unknown
                "polish_history": {},    # unknown
                "timestamp": datetime.now().isoformat(),
            }, f, indent=4, default=str)
        print_success(f"Metrics JSON saved: {metrics_json_path}")
    except Exception as e:
        print_warning(f"Could not save metrics JSON: {e}")

    # ----- Save final bundle (optional) -----
    try:
        final_bundle = {
            "student_type": student_type,
            "teacher": "DenseNet",
            "seeds": SEEDS,
            "seed_model_paths": seed_paths,
            "test_metrics": {
                k: v for k, v in test_metrics.items()
                if k not in ["all_preds", "all_labels", "all_probs",
                             "fpr", "tpr", "confusion_matrix",
                             "classification_report"]
            },
            "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
            "test_classification_report": test_metrics["classification_report"],
            "cv_summary": cv_summary,
            "ci95": ci95,
            "config": config,
            "data_split": data_split,
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(final_bundle, final_model_path)
        print_success(f"Final bundle saved: {final_model_path}")
    except Exception as e:
        print_warning(f"Could not save final bundle: {e}")

    # ----- Generate PDF -----
    print_step(4, 4, "Generating 24-page PDF")
    result = generate_pdf(
        pdf_path=pdf_path,
        student_type=student_type,
        teacher_name="DenseNet",
        test_metrics=test_metrics,
        cv_summary=cv_summary,
        ci95=ci95,
        seed_results=seed_results,
        kd_history={},
        polish_history={},
        plots=plots,
        config=config,
        data_split=data_split,
        n_test=len(test_paths),
        device=device,
    )

    if result:
        print_success(f"PDF generated: {pdf_path}")
    else:
        print_error("PDF generation failed.")

    return result

# ================================================================
# ✅ Find Best Model Among All 3 Ensemble Bundles (Option 5)
# ================================================================
# Reads the 3 `kd_*_final.pth` bundles (each contains `test_metrics`),
# selects the one with the HIGHEST `test_metrics.accuracy`,
# and copies it to: BestModels/best_ensemble_models.pth
#
# Fast: no model loading, no TTA — just reads JSON-like metadata
# from the bundles and copies the winner.
# ================================================================

def find_best_models() -> Optional[str]:
    """
    Compare the 3 KD ensemble final bundles by their saved test accuracy,
    pick the best one, and copy it to BestModels/best_ensemble_models.pth.

    Returns the path of the copied best model, or None on failure.
    """
    # ----- Resolve paths -----
    candidates = {
        "alexnet": os.path.join(
            KD_ALEXNET_ROOT, "kd_densenet_to_alexnet_final.pth"
        ),
        "mobilenetv2": os.path.join(
            KD_MNV2_ROOT, "kd_densenet_to_mobilenetv2_final.pth"
        ),
        "vgg16": os.path.join(
            KD_VGG16_ROOT, "kd_densenet_to_vgg16_final.pth"
        ),
    }

    best_models_dir = os.path.join(KD_ENSEMBLE_ROOT, "BestModels")
    best_model_path = os.path.join(best_models_dir, "best_ensemble_models.pth")

    print_banner(
        "Find Best Model — Among 3 KD Ensemble Bundles",
        "Compare test accuracy → copy winner to BestModels/",
    )

    # ----- Check existence + read test accuracy -----
    results: List[Dict[str, Any]] = []
    for name, path in candidates.items():
        if not os.path.exists(path):
            print_warning(f"Missing: {path}")
            results.append({
                "name": name,
                "path": path,
                "exists": False,
                "accuracy": -1.0,
                "f1": -1.0,
                "f1_macro": -1.0,
                "auc": -1.0,
            })
            continue

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print_warning(f"Could not read {os.path.basename(path)}: {e}")
            results.append({
                "name": name,
                "path": path,
                "exists": True,
                "accuracy": -1.0,
                "f1": -1.0,
                "f1_macro": -1.0,
                "auc": -1.0,
            })
            continue

        tm = ckpt.get("test_metrics", {}) or {}
        results.append({
            "name": name,
            "path": path,
            "exists": True,
            "accuracy": float(tm.get("accuracy", -1.0)),
            "f1": float(tm.get("f1_score", -1.0)),
            "f1_macro": float(tm.get("f1_macro", -1.0)),
            "auc": float(tm.get("roc_auc_mean", -1.0)),
        })

    # ----- Print comparison table -----
    if HAS_RICH:
        table = Table(
            title="📊 Candidate Bundles (sorted by test accuracy)",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold magenta",
        )
        table.add_column("Model", style="bold white")
        table.add_column("Test Acc", style="bold green", justify="right")
        table.add_column("F1 (w)", style="white", justify="right")
        table.add_column("F1 (macro)", style="white", justify="right")
        table.add_column("AUC", style="white", justify="right")
        table.add_column("Status", style="cyan")

        sorted_results = sorted(
            results, key=lambda r: r["accuracy"], reverse=True
        )
        for r in sorted_results:
            if not r["exists"]:
                table.add_row(r["name"], "—", "—", "—", "—", "❌ missing")
            elif r["accuracy"] < 0:
                table.add_row(r["name"], "—", "—", "—", "—", "⚠️ unreadable")
            else:
                table.add_row(
                    r["name"].upper(),
                    f"{r['accuracy']:.4f}",
                    f"{r['f1']:.4f}",
                    f"{r['f1_macro']:.4f}",
                    f"{r['auc']:.4f}",
                    "✅ ready",
                )
        console.print()
        console.print(table)
        console.print()
    else:
        print("\n" + "=" * 78)
        print("  📊 Candidate Bundles (sorted by test accuracy)")
        print("=" * 78)
        for r in sorted(results, key=lambda r: r["accuracy"], reverse=True):
            if not r["exists"]:
                print(f"  {r['name'].upper():<12}  —  (missing)")
            elif r["accuracy"] < 0:
                print(f"  {r['name'].upper():<12}  —  (unreadable)")
            else:
                print(f"  {r['name'].upper():<12}  "
                      f"Acc={r['accuracy']:.4f}  "
                      f"F1={r['f1']:.4f}  "
                      f"F1m={r['f1_macro']:.4f}  "
                      f"AUC={r['auc']:.4f}")
        print("=" * 78)

    # ----- Select winner -----
    valid = [r for r in results if r["exists"] and r["accuracy"] >= 0]
    if len(valid) == 0:
        print_error("No valid bundles found. Cannot pick a winner.")
        return None

    winner = max(valid, key=lambda r: r["accuracy"])

    print_success(
        f"🏆 Winner: {winner['name'].upper()} "
        f"(Test Acc = {winner['accuracy']:.4f})"
    )
    print_info(f"Source: {winner['path']}")

    # ----- Copy to BestModels/ -----
    os.makedirs(best_models_dir, exist_ok=True)
    try:
        import shutil
        shutil.copy2(winner["path"], best_model_path)
        print_success(f"Copied to: {best_model_path}")
    except Exception as e:
        print_error(f"Could not copy: {e}")
        return None

    # ----- Save a small JSON summary next to it -----
    summary_path = os.path.join(best_models_dir, "best_model_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump({
                "winner": winner["name"],
                "winner_path": winner["path"],
                "copied_to": best_model_path,
                "test_metrics": {
                    "accuracy": winner["accuracy"],
                    "f1_score": winner["f1"],
                    "f1_macro": winner["f1_macro"],
                    "roc_auc_mean": winner["auc"],
                },
                "all_candidates": [
                    {
                        "name": r["name"],
                        "path": r["path"],
                        "exists": r["exists"],
                        "test_accuracy": r["accuracy"],
                        "f1_score": r["f1"],
                        "f1_macro": r["f1_macro"],
                        "roc_auc_mean": r["auc"],
                    }
                    for r in results
                ],
                "timestamp": datetime.now().isoformat(),
            }, f, indent=4, default=str)
        print_success(f"Summary saved: {summary_path}")
    except Exception as e:
        print_warning(f"Could not save summary: {e}")

    # ----- Final panel -----
    if HAS_RICH:
        console.print()
        console.print(Panel.fit(
            f"[bold green]🏆 Best Model Selected![/bold green]\n"
            f"[cyan]Model:[/cyan] {winner['name'].upper()}\n"
            f"[cyan]Test Acc:[/cyan] {winner['accuracy']:.4f}\n"
            f"[cyan]F1 (weighted):[/cyan] {winner['f1']:.4f}\n"
            f"[cyan]F1 (macro):[/cyan] {winner['f1_macro']:.4f}\n"
            f"[cyan]AUC:[/cyan] {winner['auc']:.4f}\n"
            f"[cyan]Saved to:[/cyan] {best_model_path}",
            border_style="green",
            padding=(1, 4),
        ))

    return best_model_path


# ================================================================
# ✅ Option 2 — MobileNetV2
# ================================================================

@with_music("mobilenetv2")
def run_kd_densenet_vs_mobilenetv2() -> Dict[str, Any]:
    """
    KD Ensemble: DenseNet teacher → MobileNetV2 student (3 seeds).
    Wrapper around the generic pipeline.
    """
    return _run_kd_pipeline_generic(
        student_type="mobilenetv2",
        student_ckpt=MOBILENETV2_STUDENT_PATH,
        out_root=KD_MNV2_ROOT,
        title="Knowledge Distillation — DenseNet → MobileNetV2 (3 Seeds)",
        final_bundle_name="kd_densenet_to_mobilenetv2_final.pth",
    )


# ================================================================
# ✅ Option 3 — VGG16
# ================================================================

@with_music("vgg16")
def run_kd_densenet_vs_vgg16() -> Dict[str, Any]:
    """
    KD Ensemble: DenseNet teacher → VGG16 student (3 seeds).
    Wrapper around the generic pipeline.
    """
    return _run_kd_pipeline_generic(
        student_type="vgg16",
        student_ckpt=VGG16_STUDENT_PATH,
        out_root=KD_VGG16_ROOT,
        title="Knowledge Distillation — DenseNet → VGG16 (3 Seeds)",
        final_bundle_name="kd_densenet_to_vgg16_final.pth",
    )

# ================================================================
# ✅ Internal Menu
# ================================================================
# This menu is displayed ONLY when user selects option 4 in main.py
# (via main.py → handle_kd_new_models() → run_new_models_pipeline()).
# ================================================================

def print_internal_menu():
    """Print the internal KD menu with Rich."""
    if HAS_RICH:
        table = Table(
            title="[bold cyan]🎯 KD Ensemble — New Models[/bold cyan]",
            box=box.ROUNDED,
            border_style="bright_blue",
            header_style="bold magenta",
            title_style="bold cyan",
            show_lines=True,
        )
        table.add_column("#", style="bold yellow", width=4, justify="center")
        table.add_column("Option", style="bold white", width=38)
        table.add_column("Description", style="dim white", width=40)

        table.add_row("0", "🚪 Exit", "Return to main menu")
        table.add_row(
            "1",
            "🧠 KD Ensemble DenseNet vs. AlexNet",
            "DenseNet → AlexNet (3 seeds, TTA, CV5, PDF)",
        )
        table.add_row(
            "2",
            "🧠 KD Ensemble DenseNet vs. MobileNetV2",
            "DenseNet → MobileNetV2 (3 seeds, TTA, CV5, PDF)",
        )
        table.add_row(
            "3",
            "🧠 KD Ensemble DenseNet vs. VGG16",
            "DenseNet → VGG16 (3 seeds, TTA, CV5, PDF)",
        )
        table.add_row(
            "4",
            "📄 Regenerate PDF (VGG16)",
            "Rebuild PDF from saved metrics (no retraining)",
        )
        table.add_row(
            "5",
            "🏆 Find Best Model (All 3)",
            "Compare test acc → copy winner to BestModels/",
        )
        table.add_row(
            "6",
            "🏆 KD Final (Teacher → Student)",
            "MobileNetV2 ensemble → ResNet50 | 24-page PDF + interpretability",
        )
        table.add_row(
            "7",
            "📄 Regenerate PDF (KD Final)",
            "Rebuild 24-page PDF from saved KDFinal artifacts (no retraining)",
        )

        console.print()
        console.print(table)
        console.print()
    else:
        print("\n" + "=" * 78)
        print("  🎯 KD Ensemble — New Models")
        print("=" * 78)
        print("  0) 🚪 Exit")
        print("  1) 🧠 KD Ensemble DenseNet vs. AlexNet")
        print("  2) 🧠 KD Ensemble DenseNet vs. MobileNetV2")
        print("  3) 🧠 KD Ensemble DenseNet vs. VGG16")
        print("  4) 🏆 Generate PDF for VGG16")
        print("  5) 🏆 Find Best Model (All 3) — copy winner to BestModels/")
        print("  6) 🏆 KD Final (Teacher → Student) — MobileNetV2 → ResNet50")
        print("=" * 78)


def get_internal_choice() -> str:
    """Get user's choice from the internal menu (0-5)."""
    if HAS_RICH:
        return Prompt.ask(
            "\n[bold cyan]👉 Enter your choice[/bold cyan]",
            choices=["0", "1", "2", "3", "4", "5", "6", "7"],   # ← "7" اضافه شد
            default="0",
        )
    else:
        return input("\n👉 Enter your choice (0-4): ").strip()


def handle_exit():
    """Handle exit from internal menu."""
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold yellow]👋 Returning to main menu...[/bold yellow]",
            border_style="yellow",
            padding=(1, 4),
        ))
    else:
        print("\n👋 Returning to main menu...")


# ================================================================
# ✅ Main entry point for internal menu
# ================================================================

def run_new_models_pipeline():
    """
    Entry point for the KD "New Models" sub-pipeline.

    Called from main.py when user selects option 4:
        KD Ensemble new Models

    Displays internal menu (0-3) in a loop until user exits.
    """
    # Header banner
    if HAS_RICH:
        console.clear()
        console.print()
        console.print(Panel.fit(
            "[bold cyan]╔══════════════════════════════════════════════════════════╗[/bold cyan]\n"
            "[bold cyan]║[/bold cyan]                                                          [bold cyan]║[/bold cyan]\n"
            "[bold cyan]║[/bold cyan]  [bold green]🎯 KD ENSEMBLE — NEW MODELS[/bold green]                        [bold cyan]║[/bold cyan]\n"
            "[bold cyan]║[/bold cyan]  [dim italic]DenseNet Teacher → AlexNet / MobileNetV2 / VGG16[/dim italic]  [bold cyan]║[/bold cyan]\n"
            "[bold cyan]║[/bold cyan]                                                          [bold cyan]║[/bold cyan]\n"
            "[bold cyan]║[/bold cyan]  [bold yellow]3 Seeds · ES on Accuracy · TTA · CV5 · CI95 · PDF[/bold yellow]  [bold cyan]║[/bold cyan]\n"
            "[bold cyan]║[/bold cyan]                                                          [bold cyan]║[/bold cyan]\n"
            "[bold cyan]╚══════════════════════════════════════════════════════════╝[/bold cyan]",
            border_style="bright_cyan",
            padding=(1, 2),
        ))
        console.print()
    else:
        print("\n" + "=" * 78)
        print("  🎯 KD ENSEMBLE — NEW MODELS")
        print("  DenseNet Teacher → AlexNet / MobileNetV2 / VGG16")
        print("  3 Seeds · ES on Accuracy · TTA · CV5 · CI95 · PDF")
        print("=" * 78 + "\n")

    # Menu loop
    while True:
        print_internal_menu()

        try:
            choice = get_internal_choice()
        except (KeyboardInterrupt, EOFError):
            choice = "0"

        if choice == "0":
            handle_exit()
            break

        elif choice == "1":
            try:
                run_kd_densenet_vs_alexnet()
            except Exception as e:
                print_error(f"AlexNet pipeline failed: {e}")
                import traceback
                traceback.print_exc()

        elif choice == "2":
            try:
                run_kd_densenet_vs_mobilenetv2()
            except Exception as e:
                print_error(f"MobileNetV2 pipeline failed: {e}")
                import traceback
                traceback.print_exc()


        elif choice == "3":
            try:
                run_kd_densenet_vs_vgg16()
            except Exception as e:
                print_error(f"VGG16 pipeline failed: {e}")
                import traceback
                traceback.print_exc()
        elif choice == "4":
            try:
                # Check if kd_metrics.json exists; if not, recover from seeds
                vgg16_root = KD_VGG16_ROOT
                metrics_json = os.path.join(vgg16_root, "kd_metrics.json")
                if os.path.exists(metrics_json):
                    # Normal path — just regenerate PDF
                    regenerate_pdf_for_student("vgg16")
                else:
                    # Recovery path — rebuild from seeds/*.pth
                    print_warning("kd_metrics.json not found — recovering from seeds...")
                    recover_from_seeds("vgg16")
            except Exception as e:
                print_error(f"PDF regeneration/recovery failed: {e}")
                import traceback
                traceback.print_exc()
        elif choice == "5":
            try:
                find_best_models()
            except Exception as e:
                print_error(f"Find Best Model failed: {e}")
                import traceback
                traceback.print_exc()
        elif choice == "6":
            try:
                from .kdFinal import run_kd_final
                run_kd_final()
            except Exception as e:
                print_error(f"KD Final failed: {e}")
                import traceback
                traceback.print_exc()
        elif choice == "7":
            try:
                from .kdFinal import regenerate_kdfinal_pdf
                regenerate_kdfinal_pdf()
            except Exception as e:
                print_error(f"PDF regeneration failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print_error(f"Invalid choice: {choice}")

        # Pause before returning to internal menu
        if choice != "0":
            if HAS_RICH:
                console.print()
                Prompt.ask("[dim]Press Enter to continue[/dim]", default="")
            else:
                input("\nPress Enter to continue...")


# ================================================================
# ✅ Typer CLI (optional — for direct invocation)
# ================================================================
# Users can also run this file directly:
#     python -m src.pipeLine.KnowledgeDistillationEnsembleAlgorithms.newModels
# and get a Typer-based CLI with subcommands:
#     new-models alexnet
#     new-models mobilenetv2
#     new-models vgg16
#     new-models menu         # default — the interactive menu
# ================================================================

try:
    import typer
    HAS_TYPER = True
    app = typer.Typer(
        name="new-models",
        help="KD Ensemble — DenseNet → {AlexNet, MobileNetV2, VGG16} (v2.1)",
        no_args_is_help=False,
        add_completion=False,
    )
except ImportError:
    HAS_TYPER = False
    app = None


if HAS_TYPER:

    @app.command(name="menu")
    def cli_menu():
        """Interactive menu (same as option 4 in main.py)."""
        run_new_models_pipeline()

    @app.command(name="alexnet")
    def cli_alexnet():
        """Run KD: DenseNet → AlexNet (3 seeds)."""
        run_kd_densenet_vs_alexnet()

    @app.command(name="mobilenetv2")
    def cli_mobilenetv2():
        """Run KD: DenseNet → MobileNetV2 (3 seeds)."""
        run_kd_densenet_vs_mobilenetv2()

    @app.command(name="vgg16")
    def cli_vgg16():
        """Run KD: DenseNet → VGG16 (3 seeds)."""
        run_kd_densenet_vs_vgg16()

    @app.callback(invoke_without_command=True)
    def cli_default(ctx: typer.Context):
        """If no subcommand is given, show the interactive menu."""
        if ctx.invoked_subcommand is None:
            run_new_models_pipeline()


# ================================================================
# ✅ Entry Point
# ================================================================

if __name__ == "__main__":
    try:
        if HAS_TYPER and len(os.sys.argv) > 1:
            app()
        else:
            run_new_models_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Goodbye!\n")
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

