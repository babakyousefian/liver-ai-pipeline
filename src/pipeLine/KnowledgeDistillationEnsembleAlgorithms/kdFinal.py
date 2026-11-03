# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/kdFinal.py
# -*- coding: utf-8 -*-

"""
KD Final — Teacher (MobileNetV2 ensemble) → Student (ResNet50)

This module performs the FINAL Knowledge Distillation step:
  Teacher: best_ensemble_models.pth  (MobileNetV2 3-seed ensemble, acc=0.8264)
  Student: final_model.pth           (ResNet50, acc=0.7736)

Pipeline:
  1. Load both models (teacher = 3 MobileNetV2 seeds, student = ResNet50)
  2. Confirm teacher accuracy > student accuracy (auto-detect)
  3. Run KD training (3 seeds of student) with Profile v2.2
  4. Polish fine-tune each seed
  5. Ensemble + TTA + CV5 + CI95
  6. Generate interpretability (Captum, LIME, SHAP, Grad-CAM)
  7. Build 24-page PDF (with all plots + tables + interpretability embedded)
  8. Save final_best_model.pth + PDF + all artifacts to KD_EnsembleModels/KDFinal/

Author: Babak Yousefian
Version: 1.0.0
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
# ✅ Reuse from newModels.py (avoid duplicating 3000 lines)
# ================================================================

from .newModels import (
    # --- Constants ---
    CLASS_NAMES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD,
    SEEDS, DROPOUT_MAP, FOCAL_GAMMA,
    KD_PATIENCE, POLISH_PATIENCE, MIN_DELTA,
    TRAINING_PROFILES, DEFAULT_PROFILE, STUDENT_PROFILE_MAP, get_profile,
    # --- Paths ---
    PROJECT_ROOT, SELF_SUP_ROOT, SSL_REPORTER_ROOT,
    KD_ENSEMBLE_ROOT,
    # --- Datasets ---
    LabeledDataset, TTADataset, load_labeled_data,
    # --- Losses ---
    FocalLossSoft, KDLossV2,
    # --- Metrics ---
    compute_metrics,
    # --- Model helpers ---
    teacher_probs,
    # --- Evaluation ---
    ensemble_predict, evaluate_with_tta,
    compute_ci95_bootstrap, run_cv5,
    # --- Plots ---
    plot_confusion_matrix, plot_roc_curves, plot_metrics_bar,
    plot_per_class_metrics, plot_ci_bars, plot_cv_folds,
    plot_training_curves, plot_per_seed_bars,
    # --- Rich helpers ---
    HAS_RICH, console,
    print_banner, print_step, print_success, print_error,
    print_info, print_warning,
    # --- NEW: seed + constants ---
    set_seed,                                   # ← اضافه شد
)

# ✅ Rich (Table, box) — import directly
try:
    from rich import box
    from rich.table import Table
    HAS_TABLE = True
except ImportError:
    box = None
    Table = None
    HAS_TABLE = False
    print("⚠️ rich.table not available")

# ✅ Music (KDFinal-specific)
try:
    from src.MassiveClasses.Audio.music import Music
    HAS_MUSIC = True
except ImportError:
    HAS_MUSIC = False
    Music = None

# Custom music path for KDFinal
KDFINAL_MUSIC_PATH = "packages/facilities/assets/musics/music41.mp3"

def _start_music_kdfinal():
    if not HAS_MUSIC:
        return None
    if not os.path.exists(KDFINAL_MUSIC_PATH):
        print_warning(f"Music file not found: {KDFINAL_MUSIC_PATH}")
        return None
    try:
        player = Music(MUSIC_PATH=KDFINAL_MUSIC_PATH)
        player.play_music()
        print_info(f"🎵 Music playing: {os.path.basename(KDFINAL_MUSIC_PATH)}")
        return player
    except Exception as e:
        print_warning(f"Could not play music: {e}")
        return None

def _stop_music_kdfinal(player):
    if player is None:
        return
    try:
        player.stop_music()
        print_info("🎵 Music stopped.")
    except Exception as e:
        print_warning(f"Could not stop music cleanly: {e}")

# ================================================================
# ✅ Paths (KDFinal-specific)
# ================================================================

# Source models (from previous runs)
FINAL_MODEL_PATH = os.path.join(
    SELF_SUP_ROOT, "final_models", "final_model.pth"
)
BEST_ENSEMBLE_PATH = os.path.join(
    KD_ENSEMBLE_ROOT, "BestModels", "best_ensemble_models.pth"
)

# Output root
KD_FINAL_ROOT = os.path.join(KD_ENSEMBLE_ROOT, "KDFinal")
KD_FINAL_PLOTS_DIR = os.path.join(KD_FINAL_ROOT, "plots")
KD_FINAL_CV_DIR = os.path.join(KD_FINAL_ROOT, "cv")
KD_FINAL_CI_DIR = os.path.join(KD_FINAL_ROOT, "ci")
KD_FINAL_INTERP_DIR = os.path.join(KD_FINAL_ROOT, "interpretability")
KD_FINAL_SEEDS_DIR = os.path.join(KD_FINAL_ROOT, "seeds")

KD_FINAL_METRICS_JSON = os.path.join(KD_FINAL_ROOT, "kd_metrics.json")
KD_FINAL_MODEL_PATH = os.path.join(KD_FINAL_ROOT, "final_best_model.pth")
KD_FINAL_PDF_PATH = os.path.join(KD_FINAL_ROOT, "final_best_model_reports.pdf")


# ================================================================
# ✅ Constants (KDFinal-specific)
# ================================================================

# Use v2.2 profile (proven to work well with 0 gap + high accuracy)
KDFINAL_PROFILE = "v2.2"

# KDFinal-specific overrides (on top of v2.2 profile)
# Goal: reduce train/val gap to < 1.0, get val_loss < 2.0
KDFINAL_PROFILE_OVERRIDES = {
    "backbone_lr": 5e-6,           # ← کاهش: یادگیری کندتر برای کاهش gap
    "head_lr": 3e-4,               # ← کاهش از 5e-4
    "backbone_wd": 3e-2,           # ← افزایش WD: منظم‌سازی بیشتر
    "head_wd": 3e-3,               # ← افزایش WD
    "kd_patience": 12,             # ← کاهش از 15 (سریع‌تر stop)
    "gap_monitor_threshold": 1.15,  # ← سخت‌گیرانه‌تر از 1.6
    "gap_monitor_patience": 2,     # ← سریع‌تر stop
    "label_smoothing": 0.10,
    "kd_epochs": 80,
}


def get_kdfinal_profile() -> Dict[str, Any]:
    """Return v2.2 profile with KDFinal overrides applied."""
    prof = dict(TRAINING_PROFILES[KDFINAL_PROFILE])
    prof.update(KDFINAL_PROFILE_OVERRIDES)
    prof["name"] = KDFINAL_PROFILE
    return prof

# Student uses ResNet50 architecture
STUDENT_ARCH = "resnet50"

# Teacher is a 3-seed ensemble of MobileNetV2
TEACHER_ARCH = "mobilenetv2"
TEACHER_N_SEEDS = 3


def ensure_kdfinal_dirs():
    """Create all KDFinal output directories."""
    for d in [
        KD_FINAL_ROOT,
        KD_FINAL_PLOTS_DIR,
        KD_FINAL_CV_DIR,
        KD_FINAL_CI_DIR,
        KD_FINAL_INTERP_DIR,
        KD_FINAL_SEEDS_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

# ================================================================
# ✅ Resolve Teacher / Student
# ================================================================
# Reads both bundles, compares test_metrics.accuracy, and decides:
#   - Higher accuracy  → Teacher
#   - Lower accuracy   → Student (will be retrained via KD)
#
# In the expected case:
#   - best_ensemble_models.pth (MobileNetV2) → Teacher (0.8264)
#   - final_model.pth           (ResNet50)   → Student (0.7736)
# ================================================================

# ================================================================
# ✅ TTADataset64 — 64 truly distinct augmentations
# ================================================================
# Extends the original 32-aug TTA (from newModels.py) with 32 more
# genuinely different augmentations:
#   32-35: shear (±5°, ±10°)
#   36-39: combined flip + rotation
#   40-43: perspective-style warp (small)
#   44-47: RGB channel-wise gains
#   48-51: slight blur (Gaussian, 3x3)
#   52-55: small shifts (3 px, diagonal)
#   56-59: contrast + brightness together
#   60-63: crop + resize variants
# ================================================================

class TTADataset64(Dataset):
    """
    Test-Time Augmentation dataset with 64 distinct augmentations.
    0-31: same as TTADataset (from newModels.py)
    32-63: 32 new augmentations (shear, combined, warp, blur, ...)
    """
    def __init__(self, paths, labels, input_size=224, n_aug=64):
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

    def _tta64(self, img, k):
        h, w = img.shape[:2]

        # ---- 0-31: reuse the original 32 augmentations ----
        if 0 <= k <= 31:
            if k == 0:
                return img
            if k == 1:
                return np.fliplr(img).copy()
            if k == 2:
                return np.flipud(img).copy()
            if 3 <= k <= 6:
                ang = [5, -5, 10, -10][k - 3]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
                return cv2.warpAffine(img, M, (w, h),
                                      borderMode=cv2.BORDER_REFLECT)
            if 7 <= k <= 10:
                f = [1.1, 0.9, 1.2, 0.8][k - 7]
                return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
            if 11 <= k <= 14:
                f = [1.1, 0.9, 1.2, 0.8][k - 11]
                m = img.mean()
                return np.clip((img.astype(np.float32) - m) * f + m,
                               0, 255).astype(np.uint8)
            if 15 <= k <= 17:
                g = [0.9, 1.1, 1.15][k - 15]
                return np.clip((img.astype(np.float32) / 255.0) ** g * 255,
                               0, 255).astype(np.uint8)
            if 18 <= k <= 20:
                zoom = [0.9, 0.85, 1.15][k - 18]
                nh, nw = int(h * zoom), int(w * zoom)
                r = cv2.resize(img, (nw, nh))
                if zoom < 1:
                    out = np.zeros((h, w, 3), np.uint8)
                    y1, x1 = (h - nh) // 2, (w - nw) // 2
                    out[y1:y1 + nh, x1:x1 + nw] = r
                    return out
                y1, x1 = (nh - h) // 2, (nw - w) // 2
                return r[y1:y1 + h, x1:x1 + w]
            if 21 <= k <= 24:
                dx, dy = [(5, 0), (-5, 0), (0, 5), (0, -5)][k - 21]
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                return cv2.warpAffine(img, M, (w, h),
                                      borderMode=cv2.BORDER_REFLECT)
            if 25 <= k <= 27:
                f = np.fliplr(img).copy()
                if k == 26:
                    f = np.clip(f.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
                elif k == 27:
                    f = np.clip(f.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
                return f
            if k == 28:
                ch, cw = int(h * 0.9), int(w * 0.9)
                y1, x1 = (h - ch) // 2, (w - cw) // 2
                return cv2.resize(img[y1:y1 + ch, x1:x1 + cw], (w, h))
            if 29 <= k <= 31:
                ang = [15, -15, 20][k - 29]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
                return cv2.warpAffine(img, M, (w, h),
                                      borderMode=cv2.BORDER_REFLECT)

        # ---- 32-63: NEW 32 augmentations ----

        # 32-35: shear (±5°, ±10°)
        if 32 <= k <= 35:
            shear = [0.05, -0.05, 0.10, -0.10][k - 32]
            M = np.float32([[1, shear, 0], [0, 1, 0]])
            return cv2.warpAffine(img, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT)

        # 36-39: combined flip + rotation
        if 36 <= k <= 39:
            f = np.fliplr(img).copy()
            ang = [5, -5, 10, -10][k - 36]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
            return cv2.warpAffine(f, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT)

        # 40-43: perspective-style warp (small)
        if 40 <= k <= 43:
            d = [3, 5, 7, 9][k - 40]
            src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            offsets = [
                [[d, 0], [w - d, d], [0, h - d], [w, h]],
                [[0, d], [w, 0], [d, h], [w - d, h]],
                [[d, 0], [w, 0], [0, h], [w - d, h - d]],
                [[0, 0], [w - d, 0], [d, h], [w, h - d]],
            ][k - 40]
            dst = np.float32(offsets)
            M = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT)

        # 44-47: RGB channel-wise gains
        if 44 <= k <= 47:
            gains = [(1.1, 0.95, 0.95), (0.95, 1.1, 0.95),
                     (0.95, 0.95, 1.1), (1.05, 1.0, 0.95)][k - 44]
            out = img.astype(np.float32)
            for c in range(3):
                out[:, :, c] *= gains[c]
            return np.clip(out, 0, 255).astype(np.uint8)

        # 48-51: slight Gaussian blur
        if 48 <= k <= 51:
            ksizes = [3, 3, 5, 5][k - 48]
            return cv2.GaussianBlur(img, (ksizes, ksizes), 0)

        # 52-55: small diagonal shifts
        if 52 <= k <= 55:
            dx, dy = [(3, 3), (-3, 3), (3, -3), (-3, -3)][k - 52]
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(img, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT)

        # 56-59: contrast + brightness combined
        if 56 <= k <= 59:
            params = [(1.1, 5), (1.1, -5), (0.9, 5), (0.9, -5)][k - 56]
            a, b = params
            return np.clip(a * img.astype(np.float32) + b,
                           0, 255).astype(np.uint8)

        # 60-63: crop + resize variants
        if 60 <= k <= 63:
            crops = [(0.85, 0.85), (0.9, 0.9), (0.85, 0.9), (0.9, 0.85)][k - 60]
            ch, cw = int(h * crops[0]), int(w * crops[1])
            y1, x1 = (h - ch) // 2, (w - cw) // 2
            return cv2.resize(img[y1:y1 + ch, x1:x1 + cw], (w, h))

        return img

    def __getitem__(self, i):
        idx = i // self.n_aug
        aug = i % self.n_aug
        img = self._tta64(self._load(self.paths[idx]), aug)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (t - m) / s, self.labels[idx], idx


# ================================================================
# ✅ evaluate_with_tta64 — TTA evaluation with 64 augs
# ================================================================

@torch.no_grad()
def evaluate_with_tta64(
    models: List[nn.Module],
    image_paths: List[str],
    labels: List[int],
    device: torch.device,
    input_size: int = 224,
    n_aug: int = 64,
    batch_size: int = 8,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as evaluate_with_tta but uses TTADataset64 (64 augs).
    """
    n = len(image_paths)
    tta_ds = TTADataset64(image_paths, labels, input_size=input_size, n_aug=n_aug)
    tta_loader = DataLoader(
        tta_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    per_img_probs: List[List[np.ndarray]] = [[] for _ in range(n)]

    pbar = tqdm(
        tta_loader, desc="   TTA(64) Inference",
        leave=False, disable=not verbose, ncols=100,
    )
    for imgs, _lbls, img_idx in pbar:
        probs = ensemble_predict(models, imgs, device).numpy()
        for i, idx in enumerate(img_idx.numpy()):
            per_img_probs[idx].append(probs[i])

    avg_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    for i in range(n):
        if len(per_img_probs[i]) > 0:
            avg_probs[i] = np.mean(per_img_probs[i], axis=0)

    preds = avg_probs.argmax(axis=1)
    return preds, avg_probs


# ================================================================
# ✅ run_cv5_tta64 — 5-fold CV with TTA(64)
# ================================================================

def run_cv5_tta64(
    models: List[nn.Module],
    image_paths: List[str],
    labels: List[int],
    device: torch.device,
    input_size: int = 224,
    n_folds: int = 5,
    n_aug_tta: int = 64,
    batch_size: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Same as run_cv5 but uses evaluate_with_tta64.
    """
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    if verbose:
        print_info(f"Running {n_folds}-fold CV with TTA(64) ({n_aug_tta} augs)")

    for fold_idx, (_, val_idx) in enumerate(skf.split(image_paths, labels_arr)):
        val_paths = [image_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        if verbose:
            print(f"\n   📌 Fold {fold_idx + 1}/{n_folds} "
                  f"({len(val_paths)} samples)")

        preds, probs = evaluate_with_tta64(
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


def resolve_teacher_student() -> Dict[str, Any]:
    """
    Compare accuracies of the two source models and decide which is
    teacher and which is student.

    Returns a dict:
        {
            "teacher_path": str,
            "teacher_arch": "mobilenetv2" | "resnet50",
            "teacher_acc": float,
            "teacher_type": "ensemble" | "single",
            "student_path": str,
            "student_arch": "resnet50" | "mobilenetv2",
            "student_acc": float,
        }
    """
    print_step(0, 1, "Resolving Teacher / Student")

    if not os.path.exists(FINAL_MODEL_PATH):
        raise FileNotFoundError(f"final_model.pth not found: {FINAL_MODEL_PATH}")
    if not os.path.exists(BEST_ENSEMBLE_PATH):
        raise FileNotFoundError(f"best_ensemble_models.pth not found: {BEST_ENSEMBLE_PATH}")

    # ----- Load metadata from both -----
    ck_final = torch.load(FINAL_MODEL_PATH, map_location="cpu", weights_only=False)
    ck_best = torch.load(BEST_ENSEMBLE_PATH, map_location="cpu", weights_only=False)

    final_arch = ck_final.get("architecture", "resnet50")
    final_acc = float(ck_final.get("test_metrics_tta", {}).get("accuracy", 0.0))

    best_student_type = ck_best.get("student_type", "mobilenetv2")
    best_acc = float(ck_best.get("test_metrics", {}).get("accuracy", 0.0))

    print_info(f"final_model.pth  : arch={final_arch}, acc={final_acc:.4f}")
    print_info(f"best_ensemble.pth: arch={best_student_type}, acc={best_acc:.4f}")

    # ----- Decide who is teacher -----
    if best_acc >= final_acc:
        # Expected: best_ensemble (MobileNetV2) is teacher, final (ResNet50) is student
        teacher_path = BEST_ENSEMBLE_PATH
        teacher_arch = best_student_type  # "mobilenetv2"
        teacher_acc = best_acc
        teacher_type = "ensemble"  # 3-seed ensemble

        student_path = FINAL_MODEL_PATH
        student_arch = final_arch  # "resnet50"
        student_acc = final_acc

        print_success(
            f"Teacher: {teacher_arch.upper()} (ensemble) — {teacher_acc:.4f}"
        )
        print_success(
            f"Student: {student_arch.upper()} — {student_acc:.4f} (will be retrained)"
        )
    else:
        # Rare case: final_model (ResNet50) is better than best_ensemble
        teacher_path = FINAL_MODEL_PATH
        teacher_arch = final_arch  # "resnet50"
        teacher_acc = final_acc
        teacher_type = "single"

        student_path = BEST_ENSEMBLE_PATH
        student_arch = best_student_type
        student_acc = best_acc

        print_success(
            f"Teacher: {teacher_arch.upper()} (single) — {teacher_acc:.4f}"
        )
        print_success(
            f"Student: {student_arch.upper()} — {student_acc:.4f} (will be retrained)"
        )

    return {
        "teacher_path": teacher_path,
        "teacher_arch": teacher_arch,
        "teacher_acc": teacher_acc,
        "teacher_type": teacher_type,
        "student_path": student_path,
        "student_arch": student_arch,
        "student_acc": student_acc,
    }

# ================================================================
# ✅ Model Loaders
# ================================================================
# Teacher (MobileNetV2 ensemble):
#   - best_ensemble_models.pth is a BUNDLE containing seed_model_paths
#   - We load all 3 MobileNetV2 seeds → ensemble_predict averages them
#
# Student (ResNet50):
#   - final_model.pth contains model_state_dict for ResNet50
#   - We init ResNet50 from these weights (NOT from scratch)
#   - KD will continue training from this starting point
# ================================================================

def load_teacher_ensemble(teacher_info: Dict[str, Any], device: torch.device) -> List[nn.Module]:
    """
    Load the teacher as a list of models (for ensemble_predict).

    If teacher is MobileNetV2 ensemble:
        → Load all 3 seeds from seed_model_paths
    If teacher is a single model (rare case):
        → Load just one
    """
    from .newModels import load_student as load_any_student

    path = teacher_info["teacher_path"]
    arch = teacher_info["teacher_arch"]
    ttype = teacher_info["teacher_type"]

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if ttype == "ensemble":
        # MobileNetV2 3-seed ensemble
        seed_paths = ckpt.get("seed_model_paths", [])
        if not seed_paths:
            raise ValueError(f"Bundle has no seed_model_paths: {path}")

        models = []
        for sp in seed_paths:
            if not os.path.exists(sp):
                print_warning(f"Teacher seed missing: {sp}")
                continue
            m = load_any_student(arch, sp, device)
            m.eval()
            models.append(m)

        if len(models) == 0:
            raise RuntimeError("No teacher seeds could be loaded.")

        print_success(f"Teacher ensemble loaded: {len(models)} models ({arch})")
        return models

    else:
        # Single teacher (e.g., ResNet50)
        m = load_any_student(arch, path, device)
        m.eval()
        print_success(f"Teacher single loaded: {arch}")
        return [m]


def load_student_resnet50(student_info: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Load ResNet50 student.

    - If checkpoint path contains 'seeds/' → it's a trained seed → load weights DIRECTLY
      into a fresh ResNet50 (no dropout wrapper, because checkpoints were saved
      AFTER the dropout wrapper — but state_dict keys are fc.weight / fc.bias,
      so we need to build the model WITHOUT the dropout wrapper for loading,
      then re-add it. Actually: we build WITH dropout wrapper, but we map keys.)
    - If checkpoint path is final_model.pth → init weights from it, then add dropout.
    """
    from src.models.resnet.resnet50 import create_resnet50

    path = student_info["student_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Student weights not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    is_seed = "seeds" in path.replace("\\", "/")
    if is_seed:
        # ============================================================
        # CASE 1: Loading a TRAINED SEED checkpoint (from Step 5)
        # ============================================================
        # Build ResNet50 with dropout wrapper — but we need to remap
        # fc.weight → fc.1.weight, fc.bias → fc.1.bias
        model = create_resnet50(
            num_classes=NUM_CLASSES,
            pretrained=False,
            verbose=False,
        )
        in_features = model.fc.in_features
        new_fc = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(in_features, NUM_CLASSES),
        )
        model.fc = new_fc

        # Remap keys
        remapped_state = {}
        for k, v in state.items():
            if k == "fc.weight":
                remapped_state["fc.1.weight"] = v
            elif k == "fc.bias":
                remapped_state["fc.1.bias"] = v
            else:
                remapped_state[k] = v

        missing, unexpected = model.load_state_dict(remapped_state, strict=False)
        if missing or unexpected:
            print_warning(
                f"Seed load mismatch | missing={missing} unexpected={unexpected}"
            )
        else:
            print_success(f"Seed loaded cleanly from {os.path.basename(path)}")

        return model.to(device)

    else:
        # ============================================================
        # CASE 2: Initial load from final_model.pth (Step 4 start of seed)
        # ============================================================
        model = create_resnet50(
            num_classes=NUM_CLASSES,
            pretrained=False,
            verbose=False,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        print_info(
            f"Student ResNet50 init from final_model.pth | "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

        # Add dropout wrapper, preserving pretrained fc weights
        original_fc_weight = model.fc.weight.detach().clone()
        original_fc_bias = model.fc.bias.detach().clone()
        in_features = model.fc.in_features

        new_fc = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(in_features, NUM_CLASSES),
        )
        with torch.no_grad():
            new_fc[1].weight.copy_(original_fc_weight)
            new_fc[1].bias.copy_(original_fc_bias)
        model.fc = new_fc

        print_info(
            f"Dropout(0.15) added before fc; fc weights preserved "
            f"({NUM_CLASSES}x{in_features})"
        )
        return model.to(device)

# ================================================================
# ✅ KD Training — one seed (Teacher ensemble → ResNet50 student)
# ================================================================
# Uses Profile v2.2 (proven on MobileNetV2):
#   - LR: backbone 5e-6, head 5e-4
#   - WD: backbone 2e-2, head 2e-3
#   - Label smoothing 0.10
#   - Gap Monitor: val_loss > 1.5× train_loss for 3 epochs → STOP
#   - Early stopping on ACCURACY
#   - KD patience 25, Polish patience 15
# ================================================================

def kd_train_kdfinal_seed(
    student: nn.Module,
    teacher_models: List[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    profile: str = KDFINAL_PROFILE,
    alpha: float = 0.7,
    temperature: float = 3.0,
    class_weights: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train ResNet50 student via KD from MobileNetV2 ensemble teacher.
    Same profile, early stopping, and gap monitor as newModels.py v2.2.
    """
    prof = get_kdfinal_profile()   # ← این
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
    teacher_models = [t.to(device).eval() for t in teacher_models]
    if class_weights is not None:
        class_weights = class_weights.to(device)

    crit = KDLossV2(
        alpha=alpha,
        temperature=temperature,
        gamma=FOCAL_GAMMA,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    ).to(device)

    # ---------- Param groups ----------
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
            f"  Teacher models: {len(teacher_models)} ({teacher_models[0].__class__.__name__})"
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

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "val_f1_macro": [], "lr": [],
        "gap_ratio": [],
    }

    best_acc = -1.0
    best_state = None
    best_ep = 0
    wait = 0
    gap_wait = 0
    final_val_metrics: Dict[str, Any] = {}

    for ep in range(epochs):
        # ====== TRAIN ======
        student.train()
        tl = 0.0
        tp, ty = [], []
        pbar = tqdm(
            train_loader, desc=f"S{seed} KD {ep+1}/{epochs}",
            leave=False, disable=not verbose, ncols=100,
        )
        for imgs, lbls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            # Teacher = average of 3 MobileNetV2 models
            with torch.no_grad():
                t_probs = ensemble_predict(teacher_models, imgs, device)

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
                t_probs = ensemble_predict(teacher_models, imgs, device)

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

        gap_ratio = val_loss / max(tr_loss, 1e-6)
        history["gap_ratio"].append(gap_ratio)

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

        # ---------- GAP MONITOR ----------
        if gap_threshold is not None:
            if gap_ratio > gap_threshold:
                gap_wait += 1
                if verbose:
                    print(f"   ⚠️  Gap warning {gap_wait}/{gap_patience} "
                          f"(ratio={gap_ratio:.2f} > {gap_threshold})")
                if gap_wait >= gap_patience:
                    if verbose:
                        print(f"   🛑 Seed {seed} GAP MONITOR STOP at epoch {ep+1}")
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
# ✅ Polish Fine-Tune (same as newModels.py)
# ================================================================

def polish_finetune_kdfinal(
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    profile: str = KDFINAL_PROFILE,
    class_weights: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Short polish fine-tune on true labels only (no teacher)."""
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
        "val_f1_macro": [], "gap_ratio": [],
    }

    best_acc = -1.0
    best_state = None
    best_ep = 0
    wait = 0

    for ep in range(epochs):
        student.train()
        tl = 0.0
        tp, ty = [], []
        pbar = tqdm(
            train_loader, desc=f"S{seed} Polish {ep+1}/{epochs}",
            leave=False, disable=not verbose, ncols=100,
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
                    print(f"   🛑 Seed {seed} polish early stop at epoch {ep+1}")
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
# ✅ Short Polish (max 5 epochs, very low LR)
# ================================================================
# The full polish (20 epochs, lr=2e-6) OVERFIT and dropped val accuracy.
# This short polish:
#   - Max 5 epochs (hard limit)
#   - LR = 1e-7 (100x lower than before)
#   - Patience = 2 (stops immediately if no improvement)
#   - Only used if it beats KD; otherwise KD wins.
# ================================================================

def polish_finetune_kdfinal_short(
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = 5,
    lr: float = 1e-7,
    patience: int = 2,
    class_weights: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Very short polish: max 5 epochs, LR=1e-7, patience=2.
    Designed to gently refine without overfitting.
    """
    student = student.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Use standard label smoothing (0.05) — NOT 0.10 (that was too aggressive)
    crit = FocalLossSoft(
        alpha=class_weights, gamma=FOCAL_GAMMA, label_smoothing=0.05
    ).to(device)

    opt = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

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
        "val_f1_macro": [], "gap_ratio": [],
    }

    best_acc = -1.0
    best_state = None
    best_ep = 0
    wait = 0

    for ep in range(max_epochs):
        # ====== TRAIN ======
        student.train()
        tl = 0.0
        tp, ty = [], []
        pbar = tqdm(
            train_loader, desc=f"S{seed} Polish(short) {ep+1}/{max_epochs}",
            leave=False, disable=not verbose, ncols=100,
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
                torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
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
            print(f"   S{seed} Polish(short) E{ep+1}/{max_epochs} | "
                  f"TrF1={tr_f1:.4f} | VaF1={vm['f1_score']:.4f} "
                  f"VaF1m={vm['f1_macro']:.4f} Acc={vm['accuracy']:.4f} | "
                  f"Gap={gap_ratio:.2f}x")

        # ----- Early stopping on accuracy (patience=2) -----
        if vm["accuracy"] > best_acc + 1e-4:
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
                    "profile": KDFINAL_PROFILE,
                    "source": "polish_short",
                    "history": history,
                }, save_path)
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"   🛑 Seed {seed} polish(short) early stop at epoch {ep+1}")
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
        "profile": KDFINAL_PROFILE,
    }

# ================================================================
# ✅ Evaluation: 3-seed Ensemble + TTA + CV5 + CI95
# ================================================================
# After training 3 ResNet50 students (seeds 42, 123, 2024), we evaluate:
#   1. Ensemble prediction (mean of 3 students' softmax)
#   2. Full TTA (32 augmentations per image)
#   3. Stratified 5-Fold CV (on full dataset)
#   4. Bootstrap 95% CI (2000 resamples)
# ================================================================

def evaluate_kdfinal_pipeline(
    student_models: List[nn.Module],
    test_paths: List[str],
    test_labels: List[int],
    all_paths: List[str],
    all_labels: List[int],
    device: torch.device,
    n_aug_tta: int = 32,          # ← 32 aug
    n_folds: int = 5,
    n_bootstrap: int = 2000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Full evaluation with TTA(32) + CV5(32) + CI95."""
    print_step(7, 10, "TTA(32) evaluation on test set (32 augs × 3 seeds)")

    # ----- 1. TTA(32) on test set (from newModels.py) -----
    test_preds, test_probs = evaluate_with_tta(    # ← evaluate_with_tta (32)
        student_models, test_paths, test_labels, device,
        input_size=224, n_aug=n_aug_tta, batch_size=8, verbose=verbose,
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    print_success(
        f"Test | Acc={test_metrics['accuracy']:.4f} | "
        f"F1={test_metrics['f1_score']:.4f} | "
        f"F1m={test_metrics['f1_macro']:.4f} | "
        f"AUC={test_metrics['roc_auc_mean']:.4f}"
    )

    # ----- 2. CV5(32) on full dataset (from newModels.py) -----
    print_step(8, 10, "CV5(32) + Bootstrap CI95")
    cv_summary = run_cv5(    # ← run_cv5 (32)
        student_models, all_paths, all_labels, device,
        input_size=224, n_folds=n_folds, n_aug_tta=n_aug_tta,
        batch_size=8, seed=SEEDS[0], verbose=verbose,
    )

    # ----- 3. Bootstrap CI95 -----
    ci95 = compute_ci95_bootstrap(
        np.array(test_labels), test_preds,
        n_bootstrap=n_bootstrap, seed=SEEDS[0],
    )
    print_success(
        f"CI95 Accuracy: [{ci95['accuracy']['ci95_low']:.4f}, "
        f"{ci95['accuracy']['ci95_high']:.4f}]"
    )

    return {
        "test_metrics": test_metrics,
        "cv_summary": cv_summary,
        "ci95": ci95,
        "test_preds": test_preds,
        "test_probs": test_probs,
    }


# ================================================================
# ✅ Generate all standard plots (confusion, ROC, metrics, ...)
# ================================================================

def generate_kdfinal_plots(
    eval_results: Dict[str, Any],
    seed_results: List[Dict[str, Any]],
    student_label: str = "RESNET50",
) -> Dict[str, str]:
    """
    Generate all 8 standard plots for KDFinal.

    Saves to KD_FINAL_PLOTS_DIR and KD_FINAL_CV_DIR.
    Returns dict of plot paths.
    """
    print_step(9, 10, "Generating standard plots")

    tm = eval_results["test_metrics"]
    cv = eval_results["cv_summary"]
    ci = eval_results["ci95"]

    plots: Dict[str, str] = {}

    # 1) Confusion matrix
    plots["cm"] = plot_confusion_matrix(
        tm["confusion_matrix"],
        os.path.join(KD_FINAL_PLOTS_DIR, "confusion_matrix.png"),
        f"{student_label} (KD Final) — Confusion Matrix",
    )

    # 2) ROC curves
    plots["roc"] = plot_roc_curves(
        tm,
        os.path.join(KD_FINAL_PLOTS_DIR, "roc_curves.png"),
        f"{student_label} (KD Final) — ROC",
    )

    # 3) Metrics bar
    plots["metrics"] = plot_metrics_bar(
        tm,
        os.path.join(KD_FINAL_PLOTS_DIR, "metrics_bar.png"),
        f"{student_label} (KD Final) — Metrics",
    )

    # 4) Per-class metrics
    plots["per_class"] = plot_per_class_metrics(
        tm,
        os.path.join(KD_FINAL_PLOTS_DIR, "per_class.png"),
        f"{student_label} (KD Final) — Per-Class",
    )

    # 5) CI95 bars
    plots["ci95"] = plot_ci_bars(
        ci,
        os.path.join(KD_FINAL_CI_DIR, "ci95.png"),
    )

    # 6) CV folds
    plots["cv_folds"] = plot_cv_folds(
        cv,
        os.path.join(KD_FINAL_CV_DIR, "cv_folds.png"),
    )

    # 7) Training curves (last seed KD + Polish)
    last = seed_results[-1]
    plots["kd_curves"] = plot_training_curves(
        last["kd_history"],
        last["polish_history"],
        os.path.join(KD_FINAL_PLOTS_DIR, "training_curves.png"),
        f"{student_label} (KD Final) — Training Curves (seed {last['seed']})",
    )

    # 8) Per-seed bars
    plots["per_seed"] = plot_per_seed_bars(
        seed_results,
        os.path.join(KD_FINAL_PLOTS_DIR, "per_seed.png"),
    )

    print_success(f"Generated {len(plots)} plots.")
    return plots


# ================================================================
# ✅ Save final model bundle
# ================================================================

def save_kdfinal_bundle(
    student_models: List[nn.Module],
    seed_results: List[Dict[str, Any]],
    seed_model_paths: List[str],
    eval_results: Dict[str, Any],
    teacher_info: Dict[str, Any],
    student_info: Dict[str, Any],  # ← اضافه کن
    config: Dict[str, Any],
    data_split: Dict[str, int],
) -> str:
    """
    Save the final ResNet50 model + full metadata to final_best_model.pth.
    Also saves kd_metrics.json.
    """
    tm = eval_results["test_metrics"]
    cv = eval_results["cv_summary"]
    ci = eval_results["ci95"]

    # Use the best seed as the "final" model (highest val acc)
    best_seed_idx = max(
        range(len(seed_results)),
        key=lambda i: seed_results[i]["best_accuracy"],
    )
    best_seed = seed_results[best_seed_idx]
    final_model = student_models[best_seed_idx]

    print_info(
        f"Saving final model: seed {best_seed['seed']} "
        f"(best_val_acc={best_seed['best_accuracy']:.4f})"
    )

    # ----- Save final_best_model.pth -----
    bundle = {
        "model_state_dict": final_model.state_dict(),
        "architecture": "resnet50",
        "pipeline": "kd_final",
        "teacher_info": {
            "path": teacher_info["teacher_path"],
            "arch": teacher_info["teacher_arch"],
            "acc": teacher_info["teacher_acc"],
            "type": teacher_info["teacher_type"],
        },
        "student_info": {
        "init_from": student_info["student_path"],   # ← درست
        "init_acc": student_info["student_acc"],
        },
        "best_seed": best_seed["seed"],
        "seed_model_paths": seed_model_paths,
        "test_metrics": {
            k: v for k, v in tm.items()
            if k not in ["all_preds", "all_labels", "all_probs",
                         "fpr", "tpr", "confusion_matrix",
                         "classification_report"]
        },
        "test_confusion_matrix": tm["confusion_matrix"].tolist(),
        "test_classification_report": tm["classification_report"],
        "cv_summary": cv,
        "ci95": ci,
        "config": config,
        "data_split": data_split,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(bundle, KD_FINAL_MODEL_PATH)
    print_success(f"Final model saved: {KD_FINAL_MODEL_PATH}")

    # ----- Save kd_metrics.json -----
    try:
        with open(KD_FINAL_METRICS_JSON, "w") as f:
            json.dump({
                "student_type": "resnet50",
                "teacher": teacher_info["teacher_arch"],
                "seeds": SEEDS,
                "seed_model_paths": seed_model_paths,
                "test_metrics": {
                    k: v for k, v in tm.items()
                    if k not in ["all_preds", "all_labels", "all_probs",
                                 "fpr", "tpr", "confusion_matrix",
                                 "classification_report"]
                },
                "cv_summary": cv,
                "ci95": ci,
                "config": config,
                "data_split": data_split,
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
                "kd_history": seed_results[-1]["kd_history"],
                "polish_history": seed_results[-1]["polish_history"],
                "timestamp": datetime.now().isoformat(),
            }, f, indent=4, default=str)
        print_success(f"Metrics JSON saved: {KD_FINAL_METRICS_JSON}")
    except Exception as e:
        print_warning(f"Could not save metrics JSON: {e}")

    return KD_FINAL_MODEL_PATH

# ================================================================
# ✅ Interpretability — Grad-CAM, LIME, SHAP, Integrated Gradients
# ================================================================
# For the final ensemble, we generate interpretability visualizations
# and save them to KD_FINAL_INTERP_DIR. These will be embedded in the PDF.
# ================================================================

def _select_samples_for_interpretability(
    test_paths: List[str],
    test_labels: List[int],
    test_preds: np.ndarray,
    n_per_class: int = 1,
) -> List[Dict[str, Any]]:
    """
    Pick n_per_class samples per class that were CORRECTLY predicted.
    Fallback: any sample per class.
    """
    test_preds = np.asarray(test_preds)
    picks: List[Dict[str, Any]] = []

    for c in range(NUM_CLASSES):
        # Try correctly-predicted first
        correct = [
            i for i in range(len(test_paths))
            if int(test_labels[i]) == c and int(test_preds[i]) == c
        ]
        # Fallback: any sample of this class
        if not correct:
            correct = [i for i in range(len(test_paths)) if int(test_labels[i]) == c]
        if correct:
            for idx in correct[:n_per_class]:
                picks.append({
                    "index": int(idx),
                    "path": test_paths[idx],
                    "true_label": int(test_labels[idx]),
                    "pred_label": int(test_preds[idx]),
                })
    return picks


# ---------- Grad-CAM ----------
def _gradcam_for_model(model: nn.Module, image_tensor: torch.Tensor,
                       target_layer: nn.Module, device: torch.device,
                       target_class: int) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for the given layer and target class.
    Returns a (H, W) float array in [0, 1] (same size as input H, W).
    """
    model.eval()
    activations = {}
    gradients = {}

    def _fwd_hook(module, inp, out):
        activations["value"] = out.detach()

    def _bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(_fwd_hook)
    h2 = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        img = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
        out = model(img)
        model.zero_grad()
        score = out[0, target_class]
        score.backward()

        act = activations["value"][0]       # (C, h, w)
        grad = gradients["value"][0]        # (C, h, w)

        # Global average pooling of gradients
        weights = grad.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
        cam = (weights * act).sum(dim=0)                # (h, w)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Resize to input size
        h_in, w_in = image_tensor.shape[1], image_tensor.shape[2]
        cam = cv2.resize(cam, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        return cam
    finally:
        h1.remove()
        h2.remove()


def _get_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    """Find the last Conv2d layer in the model."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def _denormalize_for_display(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized (C,H,W) tensor to uint8 (H,W,3) for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0, 1)
    img = (img * 255).byte().permute(1, 2, 0).numpy()
    return img


def _save_gradcam_grid(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    device: torch.device,
    save_path: str,
    ncols: int = 4,
):
    """Overlay Grad-CAM heatmaps on images in a grid, save PNG."""
    target_layer = _get_last_conv_layer(model)
    if target_layer is None:
        print_warning("No Conv2d layer found for Grad-CAM; skipping.")
        return

    n = len(samples)
    if n == 0:
        return
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(-1)

    for i, s in enumerate(samples):
        ax = axes[i]
        # Load and preprocess
        img = cv2.imread(s["path"])
        if img is None:
            ax.axis("off"); ax.set_title("load failed")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        sd = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_n = (t - m) / sd

        try:
            cam = _gradcam_for_model(
                model, img_n, target_layer, device,
                target_class=int(s["pred_label"]),
            )
        except Exception as e:
            print_warning(f"Grad-CAM failed for {s['path']}: {e}")
            ax.axis("off"); continue

        # Overlay
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.axis("off")
        true_c = CLASS_NAMES[s["true_label"]]
        pred_c = CLASS_NAMES[s["pred_label"]]
        ax.set_title(f"T:{true_c} P:{pred_c}", fontsize=10)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Grad-CAM — KD Final (ResNet50)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


# ---------- LIME ----------
def _lime_for_model(
    model: nn.Module,
    img_path: str,
    device: torch.device,
    save_path: str,
):
    """
    Generate a LIME explanation for a single image, save as PNG.
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        print_warning("LIME not installed (pip install lime scikit-image). Skipping.")
        return

    img = cv2.imread(img_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    def _predict(batch_np):
        # batch_np: list of (H, W, 3) uint8 arrays
        model.eval()
        out_probs = []
        with torch.no_grad():
            for x in batch_np:
                t = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
                m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                sd = torch.tensor(IMAGENET_STD).view(3, 1, 1)
                t = ((t - m) / sd).unsqueeze(0).to(device)
                logits = model(t)
                out_probs.append(
                    F.softmax(logits, dim=1)[0].cpu().numpy()
                )
        return np.array(out_probs)

    explainer = lime_image.LimeImageExplainer()
    try:
        explanation = explainer.explain_instance(
            img,
            _predict,
            top_labels=4,
            hide_color=0,
            num_samples=200,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=8,
            hide_rest=False,
        )
        vis = mark_boundaries(temp / 255.0, mask)
        plt.figure(figsize=(6, 6))
        plt.imshow(vis)
        plt.axis("off")
        plt.title(f"LIME — pred={CLASS_NAMES[explanation.top_labels[0]]}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print_warning(f"LIME failed: {e}")


# ---------- SHAP ----------
def _shap_for_model(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    device: torch.device,
    save_path: str,
):
    """
    SHAP summary using GradientExplainer on the given samples.
    """
    try:
        import shap
    except ImportError:
        print_warning("SHAP not installed (pip install shap). Skipping.")
        return

    if len(samples) == 0:
        return

    def _load_prep(p):
        img = cv2.imread(p)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        sd = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return ((t - m) / sd)

    batch = []
    for s in samples:
        t = _load_prep(s["path"])
        if t is not None:
            batch.append(t)
    if len(batch) == 0:
        return
    x = torch.stack(batch).to(device)

    try:
        model.eval()
        # Small background to keep it fast
        background = x[: min(8, x.shape[0])]
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(x)

        plt.figure(figsize=(10, 6))
        try:
            shap.summary_plot(
                shap_values, x.cpu().numpy(),
                class_names=CLASS_NAMES, show=False, plot_type="bar",
            )
        except Exception:
            # Fallback: plot first-class values as image
            sv0 = np.array(shap_values)[0]
            shap.image_plot([sv0], x.cpu().numpy()[:1], show=False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print_warning(f"SHAP failed: {e}")


# ---------- Integrated Gradients (Captum) ----------
def _integrated_gradients_for_model(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    device: torch.device,
    save_path: str,
):
    """
    Integrated Gradients using captum on the samples.
    Saves a grid of attributions.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        print_warning("Captum not installed (pip install captum). Skipping.")
        return

    if len(samples) == 0:
        return

    ig = IntegratedGradients(model)
    model.eval()

    n = len(samples)
    ncols = min(4, n)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(-1)

    for i, s in enumerate(samples):
        ax = axes[i]
        img = cv2.imread(s["path"])
        if img is None:
            ax.axis("off"); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        sd = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_n = ((t - m) / sd).unsqueeze(0).to(device)

        try:
            attributions = ig.attribute(
                img_n,
                target=int(s["pred_label"]),
                n_steps=32,
                internal_batch_size=1,
            )
            attr = attributions[0].detach().cpu().numpy()
            attr = np.mean(np.abs(attr), axis=0)   # (H, W)
            if attr.max() > attr.min():
                attr = (attr - attr.min()) / (attr.max() - attr.min())
        except Exception as e:
            print_warning(f"IG failed: {e}")
            ax.axis("off"); continue

        ax.imshow(img)
        ax.imshow(attr, cmap="hot", alpha=0.5)
        ax.axis("off")
        ax.set_title(
            f"T:{CLASS_NAMES[s['true_label']]} "
            f"P:{CLASS_NAMES[s['pred_label']]}",
            fontsize=10,
        )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Integrated Gradients — KD Final (ResNet50)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


# ---------- Main entry ----------
def generate_kdfinal_interpretability(
    student_models: List[nn.Module],
    test_paths: List[str],
    test_labels: List[int],
    test_preds: np.ndarray,
    device: torch.device,
    n_per_class: int = 1,
) -> Dict[str, str]:
    """
    Generate all interpretability plots and return their paths.

    Returns:
        {
            "gradcam": path,
            "lime":    path,
            "shap":    path,
            "ig":      path,
        }
    """
    print_step(6, 10, "Generating interpretability (Grad-CAM, LIME, SHAP, IG)")

    interp_plots: Dict[str, str] = {}

    # Pick samples (correct predictions preferred)
    samples = _select_samples_for_interpretability(
        test_paths, test_labels, test_preds, n_per_class=n_per_class,
    )
    if len(samples) == 0:
        print_warning("No samples available for interpretability.")
        return interp_plots

    print_info(f"Selected {len(samples)} samples for interpretability")

    # Use the first student model for interpretability
    model = student_models[0]

    # 1) Grad-CAM
    try:
        p = os.path.join(KD_FINAL_INTERP_DIR, "gradcam.png")
        _save_gradcam_grid(model, samples, device, p)
        if os.path.exists(p) and os.path.getsize(p) > 100:
            interp_plots["gradcam"] = p
            print_success(f"Grad-CAM saved: {p}")
    except Exception as e:
        print_warning(f"Grad-CAM failed: {e}")

    # 2) LIME (one sample)
    try:
        if len(samples) > 0:
            p = os.path.join(KD_FINAL_INTERP_DIR, "lime.png")
            _lime_for_model(model, samples[0]["path"], device, p)
            if os.path.exists(p) and os.path.getsize(p) > 100:
                interp_plots["lime"] = p
                print_success(f"LIME saved: {p}")
    except Exception as e:
        print_warning(f"LIME failed: {e}")

    # 3) SHAP (on all selected samples)
    try:
        p = os.path.join(KD_FINAL_INTERP_DIR, "shap.png")
        _shap_for_model(model, samples, device, p)
        if os.path.exists(p) and os.path.getsize(p) > 100:
            interp_plots["shap"] = p
            print_success(f"SHAP saved: {p}")
    except Exception as e:
        print_warning(f"SHAP failed: {e}")

    # 4) Integrated Gradients
    try:
        p = os.path.join(KD_FINAL_INTERP_DIR, "integrated_gradients.png")
        _integrated_gradients_for_model(model, samples, device, p)
        if os.path.exists(p) and os.path.getsize(p) > 100:
            interp_plots["ig"] = p
            print_success(f"Integrated Gradients saved: {p}")
    except Exception as e:
        print_warning(f"Integrated Gradients failed: {e}")

    return interp_plots

# ================================================================
# ✅ PDF Generator — 24 pages (with ALL plots + interpretability)
# ================================================================
# This is the FIXED version: every plot/image uses add_img() with
# os.path.exists check, so nothing crashes and images actually appear.
# ================================================================

def generate_kdfinal_pdf(
    pdf_path: str,
    test_metrics: Dict[str, Any],
    cv_summary: Dict[str, Any],
    ci95: Dict[str, Dict[str, float]],
    seed_results: List[Dict[str, Any]],
    kd_history: Dict[str, List[float]],
    polish_history: Optional[Dict[str, List[float]]],
    plots: Dict[str, str],
    interp_plots: Dict[str, str],
    config: Dict[str, Any],
    data_split: Dict[str, int],
    teacher_info: Dict[str, Any],
    n_test: int,
    device: torch.device,
) -> Optional[str]:
    """
    Build a 24-page PDF report for KDFinal.

    Every image uses add_img() with existence check → no crashes,
    actual images embedded (not placeholders) when files exist.
    """
    try:
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm, inch
        from reportlab.platypus import (
            Image as RLImage, PageBreak, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle, Preformatted,
        )
    except ImportError:
        print_error("reportlab not installed: pip install reportlab")
        return None

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=1.5 * cm, leftMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title="KD Final Report — Teacher (MobileNetV2) → Student (ResNet50)",
        author="Babak Yousefian",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "T", parent=styles["Heading1"],
        fontSize=24, spaceAfter=20, alignment=TA_CENTER,
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
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14, alignment=TA_JUSTIFY,
    )
    center = ParagraphStyle("C", parent=body, alignment=TA_CENTER)
    mono = ParagraphStyle("Mono", parent=body, fontName="Courier",
                         fontSize=9, leading=12)

    def add_img(path, w=6.5 * inch, h=4.5 * inch):
        """Safe image insertion: returns Image if exists, else a short note."""
        if path and os.path.exists(path) and os.path.getsize(path) > 100:
            try:
                return RLImage(path, width=w, height=h)
            except Exception:
                return Paragraph("⚠️ Image load failed", body)
        return Paragraph("⚠️ Image not available", body)

    story = []

    # ============================================================
    # PAGE 1 — Title
    # ============================================================
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("KD Final Report", title_style))
    story.append(Paragraph(
        "Teacher: MobileNetV2 (3-seed ensemble) → Student: ResNet50",
        ParagraphStyle("sub", parent=body, alignment=TA_CENTER,
                       fontSize=13, textColor=rl_colors.grey)))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", center))
    story.append(Paragraph(f"Test Samples: {n_test}", center))
    story.append(Paragraph(f"Device: {device}", center))
    story.append(Paragraph("Author: Babak Yousefian", center))
    story.append(Paragraph("Version: 1.0.0 (KDFinal)", center))
    story.append(PageBreak())

    # ============================================================
    # PAGE 2 — Executive Summary
    # ============================================================
    story.append(Paragraph("Executive Summary", h1))
    story.append(Paragraph(
        f"This report documents the FINAL Knowledge Distillation stage. "
        f"A MobileNetV2 3-seed ensemble (teacher, test acc = "
        f"<b>{teacher_info.get('teacher_acc', 0.0):.4f}</b>) teaches a ResNet50 student "
        f"(initial test acc = <b>{teacher_info.get('student_acc', 0.0):.4f}</b>). "
        f"The student is trained with 3 independent seeds "
        f"({', '.join(map(str, SEEDS))}), Focal Loss with soft γ=0.5, "
        f"label smoothing 0.10, gap monitor, and early stopping on accuracy. "
        f"Final evaluation combines 3-seed probability averaging, full TTA "
        f"(32 augmentations), 5-fold CV, and bootstrap 95% CI.",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>Headline Results (TTA on Test Set):</b>", h2))

    head_table = [["Metric", "Value"]]
    for k in ["accuracy", "balanced_accuracy", "precision", "recall",
              "f1_score", "f1_macro", "roc_auc_mean"]:
        if k in test_metrics:
            head_table.append([k.replace("_", " ").title(),
                               f"{test_metrics[k]:.4f}"])
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

    # ============================================================
    # PAGE 3 — Pipeline Description
    # ============================================================
    story.append(Paragraph("Pipeline", h1))
    story.append(Paragraph(
        "<b>Step 1 — Data:</b> Load labeled images from "
        "finalReports/labels.csv + cropped/. Stratified 80/10/10 split.<br/>"
        "<b>Step 2 — Teacher:</b> Load 3 MobileNetV2 seeds from "
        "best_ensemble_models.pth (bundle).<br/>"
        "<b>Step 3 — Student:</b> Load ResNet50 from final_model.pth "
        "(initial fine-tuned weights).<br/>"
        "<b>Step 4 — KD training:</b> 3 seeds (42, 123, 2024), KL+Focal, "
        "gap monitor, ES on accuracy.<br/>"
        "<b>Step 5 — Polish:</b> Short refinement without teacher.<br/>"
        "<b>Step 6 — Ensemble:</b> Average softmax of 3 students.<br/>"
        "<b>Step 7 — TTA:</b> 32 augmentations per test image.<br/>"
        "<b>Step 8 — CV5 + CI95:</b> Stratified 5-fold + bootstrap 2000.<br/>"
        "<b>Step 9 — Interpretability:</b> Grad-CAM, LIME, SHAP, IG.",
        body,
    ))
    story.append(PageBreak())

    # ============================================================
    # PAGE 4 — Dataset & Split
    # ============================================================
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
    story.append(PageBreak())

    # ============================================================
    # PAGE 5 — Training Config
    # ============================================================
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
    ]))
    story.append(t)
    story.append(PageBreak())

    # ============================================================
    # PAGE 6 — Overall Metrics
    # ============================================================
    story.append(Paragraph("Detailed Metrics (TTA on Test)", h1))
    met_data = [["Metric", "Value"]]
    for k in ["accuracy", "balanced_accuracy", "precision", "recall",
              "f1_score", "f1_macro", "roc_auc_mean", "num_samples"]:
        v = test_metrics.get(k, "N/A")
        met_data.append([
            k.replace("_", " ").title(),
            f"{v:.4f}" if isinstance(v, float) else str(v),
        ])
    t = Table(met_data, colWidths=[2.5 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#3498DB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ============================================================
    # PAGE 7 — Per-Class Metrics Table
    # ============================================================
    story.append(Paragraph("Per-Class Metrics", h1))
    pc_data = [["Class", "Precision", "Recall", "F1", "Support"]]
    # ── classification_report ممکنه توی kd_metrics.json نباشه ──
    cr = test_metrics.get("classification_report", None)
    if cr is None:
        # Build a fallback from confusion matrix + per-class counts
        cm_fb = test_metrics.get("confusion_matrix", None)
        if cm_fb is not None:
            import numpy as np
            cm_arr = np.array(cm_fb)
            cr = {}
            for i, cname in enumerate(CLASS_NAMES):
                tp = cm_arr[i, i]
                fp = cm_arr[:, i].sum() - tp
                fn = cm_arr[i, :].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                support = int(cm_arr[i, :].sum())
                cr[cname] = {
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1-score": float(f1),
                    "support": support,
                }
        else:
            # Last resort: zeros
            cr = {
                c: {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
                for c in CLASS_NAMES
            }

    for cname in CLASS_NAMES:
        if cname in cr:
            r = cr[cname]
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
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ============================================================
    # PAGES 8-11 — Plots (confusion, ROC, metrics, per-class)
    # ============================================================
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

    # ============================================================
    # PAGE 12 — Per-Seed Comparison
    # ============================================================
    if plots.get("per_seed"):
        story.append(Paragraph("Per-Seed Performance", h1))
        story.append(Paragraph(
            "Three ResNet50 students trained with seeds "
            f"{SEEDS}. Each seed's best validation accuracy (used for "
            "early stopping) and final accuracy before ensembling.",
            body,
        ))
        story.append(add_img(plots["per_seed"], w=6.5 * inch, h=4 * inch))
        story.append(PageBreak())

    # ============================================================
    # PAGE 13 — CV5 Table
    # ============================================================
    story.append(Paragraph("5-Fold Cross-Validation", h1))
    story.append(Paragraph(
        "The trained 3-seed ensemble is evaluated on 5 stratified folds. "
        "Measures robustness on different data distributions.",
        body,
    ))
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

    # ============================================================
    # PAGE 14 — CV5 Plot
    # ============================================================
    if plots.get("cv_folds"):
        story.append(Paragraph("CV Folds Visualization", h1))
        story.append(add_img(plots["cv_folds"], w=6.5 * inch, h=4 * inch))
        story.append(PageBreak())

    # ============================================================
    # PAGE 15 — CI95 Table
    # ============================================================
    story.append(Paragraph("95% Confidence Interval (Bootstrap)", h1))
    story.append(Paragraph(
        "Bootstrap resampling (2000 samples) estimates the distribution "
        "of each metric on the test set.",
        body,
    ))
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

    # ============================================================
    # PAGE 16 — CI95 Plot
    # ============================================================
    if plots.get("ci95"):
        story.append(Paragraph("CI95 Visualization", h1))
        story.append(add_img(plots["ci95"], w=6.5 * inch, h=4.5 * inch))
        story.append(PageBreak())

    # ============================================================
    # PAGE 17 — Training Curves
    # ============================================================
    story.append(Paragraph("Training Curves", h1))
    story.append(Paragraph(
        "KD training trajectory (train/val loss, val accuracy, "
        "train/val F1) with polish fine-tune overlay. Early stopping "
        "on accuracy; gap monitor prevents runaway overfit.",
        body,
    ))
    if plots.get("kd_curves"):
        story.append(add_img(plots["kd_curves"], w=6.5 * inch, h=4.5 * inch))
        story.append(PageBreak())

    # ============================================================
    # PAGE 18 — Ensemble Analysis (per-seed table)
    # ============================================================
    story.append(Paragraph("Ensemble Analysis", h1))
    story.append(Paragraph(
        f"Three ResNet50 students (seeds {SEEDS}). "
        f"Each contributes 1/3 weight to the ensemble.",
        body,
    ))
    seed_data = [["Seed", "Best Acc (ES)", "Final Val Acc", "Best Epoch"]]
    for r in seed_results:
        seed_data.append([
            str(r["seed"]),
            f"{r['best_accuracy']:.4f}",
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

    # ============================================================
    # PAGE 19 — Interpretability: Grad-CAM
    # ============================================================
    story.append(Paragraph("Interpretability — Grad-CAM", h1))
    story.append(Paragraph(
        "Grad-CAM highlights regions the ResNet50 student attends to "
        "when predicting each class. Warm colors = high importance.",
        body,
    ))
    if interp_plots.get("gradcam"):
        story.append(add_img(interp_plots["gradcam"], w=6.5 * inch, h=5 * inch))
    else:
        story.append(Paragraph("⚠️ Grad-CAM image not available.", body))
    story.append(PageBreak())

    # ============================================================
    # PAGE 20 — Interpretability: LIME
    # ============================================================
    story.append(Paragraph("Interpretability — LIME", h1))
    story.append(Paragraph(
        "LIME shows which superpixels positively contribute to the "
        "predicted class. Green boundaries indicate supportive regions.",
        body,
    ))
    if interp_plots.get("lime"):
        story.append(add_img(interp_plots["lime"], w=6 * inch, h=6 * inch))
    else:
        story.append(Paragraph("⚠️ LIME image not available.", body))
    story.append(PageBreak())

    # ============================================================
    # PAGE 21 — Interpretability: SHAP
    # ============================================================
    story.append(Paragraph("Interpretability — SHAP", h1))
    story.append(Paragraph(
        "SHAP summary plot showing feature importance per class.",
        body,
    ))
    if interp_plots.get("shap"):
        story.append(add_img(interp_plots["shap"], w=6.5 * inch, h=5 * inch))
    else:
        story.append(Paragraph("⚠️ SHAP image not available.", body))
    story.append(PageBreak())

    # ============================================================
    # PAGE 22 — Interpretability: Integrated Gradients
    # ============================================================
    story.append(Paragraph("Interpretability — Integrated Gradients", h1))
    story.append(Paragraph(
        "Integrated Gradients (Captum) highlights pixel-level attribution "
        "for each prediction.",
        body,
    ))
    if interp_plots.get("ig"):
        story.append(add_img(interp_plots["ig"], w=6.5 * inch, h=5 * inch))
    else:
        story.append(Paragraph("⚠️ Integrated Gradients image not available.", body))
    story.append(PageBreak())

    # ============================================================
    # PAGE 23 — Final Verdict
    # ============================================================
    story.append(Paragraph("Final Verdict", h1))
    verdict = f"""
    <b>Teacher:</b> MobileNetV2 (3-seed ensemble) — {teacher_info['teacher_acc']:.4f}<br/>
    <b>Student (init):</b> ResNet50 — {teacher_info.get('student_acc', 0.0):.4f}<br/>
    <b>Student (after KD, ensemble + TTA):</b><br/>
    - Accuracy: {test_metrics['accuracy']:.4f}<br/>
    - Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}<br/>
    - F1 (weighted): {test_metrics['f1_score']:.4f}<br/>
    - F1 (macro): {test_metrics['f1_macro']:.4f}<br/>
    - AUC (mean): {test_metrics['roc_auc_mean']:.4f}<br/><br/>

    <b>CV5 mean accuracy:</b> {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}<br/>
    <b>95% CI (accuracy):</b> [{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]<br/><br/>

    <b>Deployment recommendation:</b> use the 3-seed ensemble with TTA. 
    For a single-model deployment, use the seed with the highest 
    validation accuracy (saved as <i>final_best_model.pth</i>).
    """
    story.append(Paragraph(verdict, body))
    story.append(PageBreak())

    # ============================================================
    # PAGE 24 — Closing
    # ============================================================
    story.append(Spacer(1, 3 * inch))
    story.append(Paragraph("✅ Report Complete", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"Author: Babak Yousefian<br/>"
        f"Version: 1.0.0 (KDFinal)",
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
# ✅ NEW: Regenerate PDF from existing artifacts (no retraining)
# ================================================================
# This is used by main.py option 7: it reads the already-saved
# kd_metrics.json + all plots from KDFinal/ and rebuilds the
# 24-page PDF in a few seconds.
# ================================================================

def regenerate_kdfinal_pdf() -> Optional[str]:
    """
    Regenerate the 24-page PDF for KDFinal using ONLY the already-saved
    artifacts (kd_metrics.json + plots/*.png + interpretability/*.png).

    No model loading, no training, no TTA. Just PDF assembly.
    Time: ~5-10 seconds.

    Returns the PDF path on success, None on failure.
    """
    print_banner(
        "Regenerate PDF — KD Final (no retraining)",
        "Reads kd_metrics.json + plots + interpretability → 24-page PDF",
    )

    # ----- 1. Check kd_metrics.json exists -----
    if not os.path.exists(KD_FINAL_METRICS_JSON):
        print_error(f"kd_metrics.json not found: {KD_FINAL_METRICS_JSON}")
        print_info("👉 You need to run the full pipeline first (option 6).")
        return None

    # ----- 2. Load metrics -----
    try:
        with open(KD_FINAL_METRICS_JSON, "r") as f:
            data = json.load(f)
    except Exception as e:
        print_error(f"Could not read kd_metrics.json: {e}")
        return None

    test_metrics = data.get("test_metrics", {})
    cv_summary = data.get("cv_summary", {})
    ci95 = data.get("ci95", {})
    config = data.get("config", {})
    data_split = data.get("data_split", {})
    seed_results = data.get("seed_results", [])
    kd_history = data.get("kd_history", {})
    polish_history = data.get("polish_history", {})

    # ----- 3. Reconstruct teacher_info from config -----
    teacher_info = {
        "teacher_arch": config.get("Teacher", "MOBILENETV2 (ensemble)").split()[0].lower(),
        "teacher_acc": float(config.get("Teacher Init Acc", "0.8264")),
        "teacher_path": BEST_ENSEMBLE_PATH,
        "teacher_type": "ensemble",
    }
    student_acc = float(config.get("Student Init Acc", "0.7736"))

    # ----- 4. Reconstruct plots dict from disk -----
    plots = {}
    plot_files = {
        "cm":        ("confusion_matrix.png", KD_FINAL_PLOTS_DIR),
        "roc":       ("roc_curves.png",       KD_FINAL_PLOTS_DIR),
        "metrics":   ("metrics_bar.png",      KD_FINAL_PLOTS_DIR),
        "per_class": ("per_class.png",        KD_FINAL_PLOTS_DIR),
        "ci95":      ("ci95.png",             KD_FINAL_CI_DIR),
        "cv_folds":  ("cv_folds.png",         KD_FINAL_CV_DIR),
        "kd_curves": ("training_curves.png",  KD_FINAL_PLOTS_DIR),
        "per_seed":  ("per_seed.png",         KD_FINAL_PLOTS_DIR),
    }
    for key, (fname, base_dir) in plot_files.items():
        p = os.path.join(base_dir, fname)
        if os.path.exists(p):
            plots[key] = p
    print_info(f"Plots found: {len(plots)} / {len(plot_files)}")

    # ----- 5. Reconstruct interpretability plots -----
    interp_plots = {}
    interp_files = {
        "gradcam": "gradcam.png",
        "lime":    "lime.png",
        "shap":    "shap.png",
        "ig":      "integrated_gradients.png",
    }
    for key, fname in interp_files.items():
        p = os.path.join(KD_FINAL_INTERP_DIR, fname)
        if os.path.exists(p) and os.path.getsize(p) > 100:
            interp_plots[key] = p
    print_info(f"Interpretability plots found: {len(interp_plots)} / {len(interp_files)}")

    # ----- 6. Reconstruct seed_results with final_val -----
    if not seed_results:
        # Fallback if not present
        seed_results = [
            {"seed": s, "best_accuracy": 0.0, "best_epoch": 0,
             "final_val": {"accuracy": 0.0}}
            for s in SEEDS
        ]

    n_test = int(test_metrics.get("num_samples", 0))

    # ----- 7. Build PDF -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Building PDF → {KD_FINAL_PDF_PATH}")

    result = generate_kdfinal_pdf(
        pdf_path=KD_FINAL_PDF_PATH,
        test_metrics=test_metrics,
        cv_summary=cv_summary,
        ci95=ci95,
        seed_results=seed_results,
        kd_history=kd_history,
        polish_history=polish_history,
        plots=plots,
        interp_plots=interp_plots,
        config=config,
        data_split=data_split,
        teacher_info=teacher_info,
        n_test=n_test,
        device=device,
    )

    if result:
        print_success(f"✅ PDF regenerated: {KD_FINAL_PDF_PATH}")
        print_info(f"   Size: {os.path.getsize(KD_FINAL_PDF_PATH) / 1024:.1f} KB")
    else:
        print_error("PDF regeneration failed.")

    return result


# ================================================================
# ✅ Orchestrator — Full KDFinal Pipeline
# ================================================================

def run_kd_final() -> Dict[str, Any]:
    """
    Full KDFinal pipeline:
      1. Resolve teacher (MobileNetV2 ensemble) and student (ResNet50)
      2. Load teacher ensemble (3 MobileNetV2 seeds)
      3. Load student ResNet50 (init from final_model.pth)
      4. Train 3 seeds via KD (Profile v2.2 + Gap Monitor)
      5. Polish each seed
      6. Ensemble + TTA + CV5 + CI95
      7. Generate 8 standard plots
      8. Generate 4 interpretability plots
      9. Build 24-page PDF
     10. Save final_best_model.pth + kd_metrics.json
    """
    print_banner(
        "KD FINAL — Teacher (MobileNetV2 ensemble) → Student (ResNet50)",
        "3 Seeds · Profile v2.2 · TTA · CV5 · CI95 · Interpretability · 24-page PDF",
    )

    # ✅ Start music
    music_player = _start_music_kdfinal()
    try:
        # ----- 0. Setup -----
        ensure_kdfinal_dirs()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_info(f"Device: {device}")

        # ----- 1. Resolve teacher / student -----
        try:
            roles = resolve_teacher_student()
        except Exception as e:
            print_error(f"resolve_teacher_student failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}

        teacher_info = {
            "teacher_path": roles["teacher_path"],
            "teacher_arch": roles["teacher_arch"],
            "teacher_acc": roles["teacher_acc"],
            "teacher_type": roles["teacher_type"],
        }
        student_info = {
            "student_path": roles["student_path"],
            "student_arch": roles["student_arch"],
            "student_acc": roles["student_acc"],
        }

        # ----- 2. Load teacher ensemble -----
        print_step(1, 10, "Loading teacher (MobileNetV2 ensemble)")
        try:
            teacher_models = load_teacher_ensemble(teacher_info, device)
        except Exception as e:
            print_error(f"Teacher load failed: {e}")
            return {"status": "failed", "error": str(e)}

        # ----- 3. Data split (same random_state as newModels.py) -----
        print_step(2, 10, "Loading labeled data + splitting")
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

        # Class weights
        tr_counts = np.bincount(np.array(train_labels), minlength=NUM_CLASSES)
        tr_counts = np.maximum(tr_counts, 1)
        class_weights_np = len(train_labels) / (NUM_CLASSES * tr_counts)
        class_weights_t = torch.tensor(class_weights_np, dtype=torch.float32)
        print_info(f"Class weights: {class_weights_np.round(4).tolist()}")

        # ----- 4. Build DataLoaders -----
        print_step(3, 10, "Building DataLoaders")
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

        # ----- 5. KD training (3 seeds) -----
        print_step(4, 10, f"Training {len(SEEDS)} seeds via KD")
        seed_results: List[Dict[str, Any]] = []
        seed_model_paths: List[str] = []

        for si, seed in enumerate(SEEDS, start=1):
            print_banner(
                f"Seed {si}/{len(SEEDS)} — {seed}",
                f"Student: {student_info['student_arch'].upper()} | "
                f"Profile: {KDFINAL_PROFILE.upper()}",
            )
            set_seed(seed)

            # Load fresh student (init from final_model.pth)
            try:
                student = load_student_resnet50(student_info, device)
            except Exception as e:
                print_error(f"Student load failed for seed {seed}: {e}")
                continue

            seed_save = os.path.join(KD_FINAL_SEEDS_DIR, f"student_seed{seed}.pth")
            kd_only_save = os.path.join(KD_FINAL_SEEDS_DIR, f"student_seed{seed}_kd_only.pth")
            polish_save = os.path.join(KD_FINAL_SEEDS_DIR, f"student_seed{seed}_polish.pth")
            seed_model_paths.append(seed_save)

            # ---- KD training ----
            t0 = time.time()
            kd_res = kd_train_kdfinal_seed(
                student=student,
                teacher_models=teacher_models,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                profile=KDFINAL_PROFILE,
                alpha=0.3,
                temperature=2.0,
                class_weights=class_weights_t,
                save_path=kd_only_save,  # ← KD فقط اینجا ذخیره می‌شه
                verbose=True,
                seed=seed,
            )
            kd_time = (time.time() - t0) / 60.0
            print_success(
                f"KD done | best Acc={kd_res['best_accuracy']:.4f} | "
                f"epoch={kd_res['best_epoch']} | time={kd_time:.1f} min"
            )

            # ---- SHORT Polish (max 5 epochs, very low LR) ----
            # Save a snapshot of KD state before polish
            kd_snapshot = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            kd_best_acc = kd_res["best_accuracy"]
            kd_best_ep = kd_res["best_epoch"]
            kd_final_val = kd_res["final_val"]
            kd_history = kd_res["history"]

            # Run polish with very small patience (force <5 epochs)
            print_info(f"Seed {seed}: running short polish (max 5 epochs, lr=1e-7)")
            t0 = time.time()
            pol_res = polish_finetune_kdfinal_short(
                student=student,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                max_epochs=5,  # ← سخت‌گیرانه: حداکثر 5 epoch
                lr=1e-7,  # ← خیلی کم
                class_weights=class_weights_t,
                save_path=polish_save,  # ← جدا از KD
                verbose=True,
                seed=seed,
            )
            pol_time = (time.time() - t0) / 60.0
            print_success(
                f"Polish done | best Acc={pol_res['best_accuracy']:.4f} | "
                f"epoch={pol_res['best_epoch']} | time={pol_time:.1f} min"
            )

            # ---- Pick the BEST of KD vs Polish ----
            if pol_res["best_accuracy"] > kd_best_acc + 1e-4:
                # Polish better → use polish state
                final_state = pol_res["state"]
                final_acc = pol_res["best_accuracy"]
                final_ep = pol_res["best_epoch"]
                final_val = pol_res["final_val"]
                pol_history = pol_res["history"]
                source = "polish"
                print_success(
                    f"Seed {seed}: POLISH wins ({final_acc:.4f} > {kd_best_acc:.4f})"
                )
            else:
                # KD better (or polish didn't improve) → keep KD state
                final_state = kd_snapshot
                final_acc = kd_best_acc
                final_ep = kd_best_ep
                final_val = kd_final_val
                pol_history = pol_res["history"]
                source = "kd"
                print_success(
                    f"Seed {seed}: KD wins ({kd_best_acc:.4f} >= {pol_res['best_accuracy']:.4f})"
                )

            # ---- Save final chosen state to seed_save ----
            torch.save({
                "model_state_dict": final_state,
                "epoch": final_ep,
                "best_accuracy": final_acc,
                "seed": seed,
                "profile": KDFINAL_PROFILE,
                "source": source,  # "kd" or "polish"
                "history": pol_history if source == "polish" else kd_history,  # ← اصلاح شد
            }, seed_save)

            seed_results.append({
                "seed": seed,
                "best_accuracy": final_acc,
                "best_epoch": final_ep,
                "final_val": final_val,
                "kd_history": kd_history,
                "polish_history": pol_history,
                "source": source,
                "save_path": seed_save,
            })

            del student
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ----- 6. Load 3 best students into ensemble -----
        print_step(5, 10, "Loading 3 best students into ensemble")
        student_models: List[nn.Module] = []
        for sr in seed_results:
            m = load_student_resnet50(
                {"student_path": sr["save_path"], "student_arch": "resnet50"},
                device,
            )
            m.eval()
            student_models.append(m)
        print_success(f"Ensemble of {len(student_models)} models ready.")

        # ----- 7. TTA + CV5 + CI95 -----
        print_step(6, 10, "Evaluation (TTA + CV5 + CI95)")
        eval_results = evaluate_kdfinal_pipeline(
            student_models=student_models,
            test_paths=test_paths,
            test_labels=test_labels,
            all_paths=all_paths,
            all_labels=all_labels,
            device=device,
            n_aug_tta=32,
            n_folds=5,
            n_bootstrap=2000,
            verbose=True,
        )

        # ----- 8. Standard plots -----
        plots = generate_kdfinal_plots(
            eval_results=eval_results,
            seed_results=seed_results,
            student_label=student_info["student_arch"].upper(),
        )

        # ----- 9. Interpretability -----
        interp_plots = generate_kdfinal_interpretability(
            student_models=student_models,
            test_paths=test_paths,
            test_labels=test_labels,
            test_preds=eval_results["test_preds"],
            device=device,
            n_per_class=1,
        )

        # ----- 10. Config + save bundle -----
        config = {
            "Profile": KDFINAL_PROFILE.upper(),
            "Student": student_info["student_arch"].upper(),
            "Teacher": f"{teacher_info['teacher_arch'].upper()} (ensemble)",
            "Seeds": ", ".join(map(str, SEEDS)),
            "Dropout": DROPOUT_MAP.get(student_info["student_arch"], 0.20),
            "Focal gamma": FOCAL_GAMMA,
            "Label smoothing": TRAINING_PROFILES[KDFINAL_PROFILE]["label_smoothing"],
            "KD alpha": 0.7,
            "KD temperature": 3.0,
            "KD epochs": TRAINING_PROFILES[KDFINAL_PROFILE]["kd_epochs"],
            "Polish epochs": TRAINING_PROFILES[KDFINAL_PROFILE]["polish_epochs"],
            "KD Patience": TRAINING_PROFILES[KDFINAL_PROFILE]["kd_patience"],
            "Polish Patience": TRAINING_PROFILES[KDFINAL_PROFILE]["polish_patience"],
            "ES Metric": "accuracy",
            "Backbone LR": TRAINING_PROFILES[KDFINAL_PROFILE]["backbone_lr"],
            "Head LR": TRAINING_PROFILES[KDFINAL_PROFILE]["head_lr"],
            "Backbone WD": TRAINING_PROFILES[KDFINAL_PROFILE]["backbone_wd"],
            "Head WD": TRAINING_PROFILES[KDFINAL_PROFILE]["head_wd"],
            "Grad clip": TRAINING_PROFILES[KDFINAL_PROFILE]["grad_clip"],
            "Gap Monitor": (
                f"ON (>{TRAINING_PROFILES[KDFINAL_PROFILE]['gap_monitor_threshold']}×, "
                f"patience={TRAINING_PROFILES[KDFINAL_PROFILE]['gap_monitor_patience']})"
            ),
            "TTA augs": 32,
            "CV folds": 5,
            "Bootstrap": 2000,
            "Mixup/CutMix": "NONE",
            "Teacher Init Acc": f"{teacher_info['teacher_acc']:.4f}",
            "Student Init Acc": f"{student_info['student_acc']:.4f}",
        }

        data_split = {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
            "total": len(all_paths),
        }

        print_step(7, 10, "Saving final bundle")
        save_kdfinal_bundle(
            student_models=student_models,
            seed_results=seed_results,
            seed_model_paths=seed_model_paths,
            eval_results=eval_results,
            teacher_info=teacher_info,
            student_info=student_info,      # ← این خط اضافه شد
            config=config,
            data_split=data_split,
        )

        # ----- 11. PDF (24 pages) -----
        print_step(8, 10, "Generating 24-page PDF")
        generate_kdfinal_pdf(
            pdf_path=KD_FINAL_PDF_PATH,
            test_metrics=eval_results["test_metrics"],
            cv_summary=eval_results["cv_summary"],
            ci95=eval_results["ci95"],
            seed_results=seed_results,
            kd_history=seed_results[-1]["kd_history"],
            polish_history=seed_results[-1]["polish_history"],
            plots=plots,
            interp_plots=interp_plots,
            config=config,
            data_split=data_split,
            teacher_info=teacher_info,
            n_test=len(test_paths),
            device=device,
        )

        # ----- 12. Final summary -----
        tm = eval_results["test_metrics"]
        cv = eval_results["cv_summary"]
        ci = eval_results["ci95"]

        if HAS_RICH:
            console.print()
            table = Table(
                title="🎉 KDFinal Complete!",
                box=box.DOUBLE_EDGE, border_style="green",
            )
            table.add_column("Metric", style="cyan bold")
            table.add_column("Value", style="green bold")
            table.add_row("Teacher", f"{teacher_info['teacher_arch'].upper()} ({teacher_info['teacher_acc']:.4f})")
            table.add_row("Student Init", f"{student_info['student_arch'].upper()} ({student_info['student_acc']:.4f})")
            table.add_row("Test Accuracy", f"{tm['accuracy']:.4f}")
            table.add_row("Test Balanced Acc", f"{tm['balanced_accuracy']:.4f}")
            table.add_row("Test F1 (weighted)", f"{tm['f1_score']:.4f}")
            table.add_row("Test F1 (macro)", f"{tm['f1_macro']:.4f}")
            table.add_row("Test AUC", f"{tm['roc_auc_mean']:.4f}")
            table.add_row("CV Mean Acc", f"{cv['mean_accuracy']:.4f} ± {cv['std_accuracy']:.4f}")
            table.add_row("CI95 Accuracy", f"[{ci['accuracy']['ci95_low']:.4f}, {ci['accuracy']['ci95_high']:.4f}]")
            console.print(table)
            console.print(f"\n[bold green]📄 PDF:[/bold green] {KD_FINAL_PDF_PATH}")
            console.print(f"[bold green]🎯 Model:[/bold green] {KD_FINAL_MODEL_PATH}")

        return {
            "status": "success",
            "test_metrics": tm,
            "cv_summary": cv,
            "ci95": ci,
            "pdf_path": KD_FINAL_PDF_PATH,
            "model_path": KD_FINAL_MODEL_PATH,
        }
    except Exception as e:
        print(f"\n Error occur : {e}")
    finally:
        _stop_music_kdfinal(music_player)



