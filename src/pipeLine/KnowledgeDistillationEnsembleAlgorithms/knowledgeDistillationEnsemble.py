# src/pipeLine/knowledgeDistillationEnsemble.py
# -*- coding: utf-8 -*-

"""
Knowledge Distillation Ensemble Pipeline (ULTIMATE EDITION)

Combines ResNet50 (SimCLR) + EfficientNet-B1 (BYOL) via Knowledge Distillation
into a SINGLE model (.pth file).

Pipeline:
1. Load two teacher models (ResNet50 + B1)
2. Weighted ensemble (learned on val set)
3. Knowledge Distillation to a single student (ResNet50)
4. Polish fine-tune (20 epochs)
5. Save student: ensemble_kd_student_final.pth
6. Full evaluation: TTA + CV5 + CI 95%
7. Save final: final_model.pth
8. Generate 30+ page PDF report

Author: Babak Yousefian
Version: 1.0.0
"""

import copy
import gc
import json
import os
import random
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

# ================================================================
# ✅ Rich UI
# ================================================================

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.text import Text
    from rich.align import Align
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    print("⚠️ Rich not installed")

# ================================================================
# ✅ Paths
# ================================================================

PROJECT_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")

# ✅ Teacher models
RESNET50_TEACHER = os.path.join(
    SELF_SUPERVISED_ROOT, "final_models",
    "resnet50_finetuned_from_simclr_final.pth",
)
B1_TEACHER = os.path.join(
    SELF_SUPERVISED_ROOT, "SSL_Reporter", "b1_finetune",
    "b1_finetuned_final.pth",
)

# ✅ Output paths (per user request)
FINAL_MODELS_DIR = os.path.join(SELF_SUPERVISED_ROOT, "final_models")
STUDENT_PATH = os.path.join(FINAL_MODELS_DIR, "ensemble_kd_student_final.pth")
FINAL_MODEL_PATH = os.path.join(FINAL_MODELS_DIR, "final_model.pth")
KD_ROOT = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "kd_ensemble")
KD_PLOTS_DIR = os.path.join(KD_ROOT, "plots")
KD_CV_DIR = os.path.join(KD_ROOT, "../CV")
KD_INTERP_DIR = os.path.join(KD_ROOT, "interpretability")
KD_PDF_PATH = os.path.join(KD_ROOT, "KD_Ensemble_Report.pdf")
KD_METRICS_JSON = os.path.join(KD_ROOT, "kd_metrics.json")

# ================================================================
# ✅ Constants
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = 4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42

# ================================================================
# ✅ Console Helpers
# ================================================================

def print_header(title: str, subtitle: str = ""):
    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]",
            border_style="bright_cyan",
            padding=(1, 4),
        ))
    else:
        print(f"\n{'='*80}\n  {title}\n  {subtitle}\n{'='*80}")


def print_step(step: int, total: int, text: str):
    if HAS_RICH:
        console.print(f"\n[bold yellow]▶ Step {step}/{total}:[/bold yellow] [bold white]{text}[/bold white]")
    else:
        print(f"\n[{step}/{total}] {text}")


def print_success(text: str):
    if HAS_RICH:
        console.print(f"[bold green]✅ {text}[/bold green]")
    else:
        print(f"✅ {text}")


def print_error(text: str):
    if HAS_RICH:
        console.print(f"[bold red]❌ {text}[/bold red]")
    else:
        print(f"❌ {text}")


def print_info(text: str):
    if HAS_RICH:
        console.print(f"[cyan]ℹ️  {text}[/cyan]")
    else:
        print(f"ℹ️  {text}")


# ================================================================
# ✅ Dirs + Seed
# ================================================================

def ensure_all_dirs():
    for d in [FINAL_MODELS_DIR, KD_ROOT, KD_PLOTS_DIR, KD_CV_DIR, KD_INTERP_DIR]:
        os.makedirs(d, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_teachers_exist() -> Dict[str, bool]:
    return {
        "resnet50_teacher": os.path.exists(RESNET50_TEACHER),
        "b1_teacher": os.path.exists(B1_TEACHER),
    }


# ================================================================
# ✅ Dataset
# ================================================================

class LabeledDataset(Dataset):
    def __init__(self, image_paths, labels, input_size=224, augment=False,
                 augment_strength="strong"):
        self.image_paths = image_paths
        self.labels = labels
        self.input_size = input_size
        self.augment = augment
        self.augment_strength = augment_strength

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        return img

    def _augment(self, img):
        strength = self.augment_strength
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
        if random.random() < (0.3 if strength == "light" else 0.5):
            angle = random.uniform(-15, 15) if strength == "strong" else random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if random.random() < 0.6:
            alpha = random.uniform(0.85, 1.15) if strength == "strong" else random.uniform(0.9, 1.1)
            beta = random.uniform(-15, 15) if strength == "strong" else random.uniform(-10, 10)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        if strength == "strong":
            if random.random() < 0.3:
                zoom = random.uniform(0.9, 1.1)
                h, w = img.shape[:2]
                nh, nw = int(h * zoom), int(w * zoom)
                resized = cv2.resize(img, (nw, nh))
                if zoom < 1:
                    ph, pw = (h - nh) // 2, (w - nw) // 2
                    out = np.zeros((h, w, 3), dtype=np.uint8)
                    out[ph:ph+nh, pw:pw+nw] = resized
                    img = out
                else:
                    sh, sw = (nh - h) // 2, (nw - w) // 2
                    img = resized[sh:sh+h, sw:sw+w]
            if random.random() < 0.3:
                gamma = random.uniform(0.85, 1.15)
                img = np.clip((img.astype(np.float32) / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        if self.augment:
            img = self._augment(img)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_t = (img_t - mean) / std
        return img_t, self.labels[idx]


class TTADataset(Dataset):
    def __init__(self, image_paths, labels, input_size=224, n_aug=32):
        self.image_paths = image_paths
        self.labels = labels
        self.input_size = input_size
        self.n_aug = n_aug

    def __len__(self):
        return len(self.image_paths) * self.n_aug

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        return img

    def _tta_augment(self, img, aug_idx):
        if aug_idx == 0: return img
        elif aug_idx == 1: return np.fliplr(img).copy()
        elif aug_idx == 2: return np.flipud(img).copy()
        elif aug_idx == 3:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), 5, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 4:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -5, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 5:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 6:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -10, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 7: return np.clip(img.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        elif aug_idx == 8: return np.clip(img.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
        elif aug_idx == 9: return np.clip(img.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        elif aug_idx == 10: return np.clip(img.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
        elif aug_idx == 11:
            mean = img.mean()
            return np.clip((img.astype(np.float32) - mean) * 1.1 + mean, 0, 255).astype(np.uint8)
        elif aug_idx == 12:
            mean = img.mean()
            return np.clip((img.astype(np.float32) - mean) * 0.9 + mean, 0, 255).astype(np.uint8)
        elif aug_idx == 13:
            mean = img.mean()
            return np.clip((img.astype(np.float32) - mean) * 1.2 + mean, 0, 255).astype(np.uint8)
        elif aug_idx == 14:
            mean = img.mean()
            return np.clip((img.astype(np.float32) - mean) * 0.8 + mean, 0, 255).astype(np.uint8)
        elif aug_idx == 15: return np.clip((img.astype(np.float32) / 255.0) ** 0.9 * 255, 0, 255).astype(np.uint8)
        elif aug_idx == 16:
            h, w = img.shape[:2]
            ch, cw = int(h * 0.9), int(w * 0.9)
            y1, x1 = (h - ch) // 2, (w - cw) // 2
            return cv2.resize(img[y1:y1+ch, x1:x1+cw], (w, h))
        elif aug_idx == 17:
            h, w = img.shape[:2]
            ch, cw = int(h * 0.85), int(w * 0.85)
            y1, x1 = (h - ch) // 2, (w - cw) // 2
            return cv2.resize(img[y1:y1+ch, x1:x1+cw], (w, h))
        elif aug_idx == 18:
            h, w = img.shape[:2]
            nh, nw = int(h * 1.1), int(w * 1.1)
            resized = cv2.resize(img, (nw, nh))
            y1, x1 = (nh - h) // 2, (nw - w) // 2
            return resized[y1:y1+h, x1:x1+w]
        elif aug_idx == 19:
            h, w = img.shape[:2]
            nh, nw = int(h * 1.15), int(w * 1.15)
            resized = cv2.resize(img, (nw, nh))
            y1, x1 = (nh - h) // 2, (nw - w) // 2
            return resized[y1:y1+h, x1:x1+w]
        elif aug_idx == 20:
            h, w = img.shape[:2]
            M = np.float32([[1, 0, -5], [0, 1, 0]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 21:
            h, w = img.shape[:2]
            M = np.float32([[1, 0, 5], [0, 1, 0]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 22:
            h, w = img.shape[:2]
            M = np.float32([[1, 0, 0], [0, 1, -5]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 23:
            h, w = img.shape[:2]
            M = np.float32([[1, 0, 0], [0, 1, 5]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 24:
            flipped = np.fliplr(img).copy()
            h, w = flipped.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), 5, 1.0)
            return cv2.warpAffine(flipped, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 25:
            flipped = np.fliplr(img).copy()
            h, w = flipped.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -5, 1.0)
            return cv2.warpAffine(flipped, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 26:
            flipped = np.fliplr(img).copy()
            return np.clip(flipped.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        elif aug_idx == 27:
            flipped = np.fliplr(img).copy()
            return np.clip(flipped.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
        elif aug_idx == 28:
            flipped = np.fliplr(img).copy()
            h, w = flipped.shape[:2]
            ch, cw = int(h * 0.9), int(w * 0.9)
            y1, x1 = (h - ch) // 2, (w - cw) // 2
            return cv2.resize(flipped[y1:y1+ch, x1:x1+cw], (w, h))
        elif aug_idx == 29:
            flipped = np.fliplr(img).copy()
            mean = flipped.mean()
            return np.clip((flipped.astype(np.float32) - mean) * 1.1 + mean, 0, 255).astype(np.uint8)
        elif aug_idx == 30:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 31:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -15, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return img

    def __getitem__(self, idx):
        img_idx = idx // self.n_aug
        aug_idx = idx % self.n_aug
        img = self._load_image(self.image_paths[img_idx])
        img = self._tta_augment(img, aug_idx)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_t = (img_t - mean) / std
        return img_t, self.labels[img_idx], img_idx


# ================================================================
# ✅ Load Data
# ================================================================

def load_labeled_data() -> Tuple[List[str], List[int]]:
    labels_csv = os.path.join("finalReports", "labels.csv")
    image_dir = os.path.join("finalReports", "cropped")
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    df = pd.read_csv(labels_csv)
    df = df[df["liverLabel"].isin(CLASS_NAMES)]
    image_paths, labels = [], []
    for _, row in df.iterrows():
        img_name = row["liverName"]
        label_name = row["liverLabel"]
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(CLASS_NAMES.index(label_name))
        else:
            base_name = os.path.splitext(img_name)[0]
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                test_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(test_path):
                    image_paths.append(test_path)
                    labels.append(CLASS_NAMES.index(label_name))
                    break
    return image_paths, labels


# ================================================================
# ✅ Loss Functions
# ================================================================

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        reduction="mean",
        label_smoothing=0.05,   # ✅ NEW
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        weight = self.alpha
        if weight is not None:
            weight = weight.to(inputs.device)

        # ✅ Label smoothing in CE
        ce = F.cross_entropy(
            inputs, targets,
            weight=weight,
            reduction="none",
            label_smoothing=self.label_smoothing,   # ✅
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss (v3.0):
    L = α_kl * KL(student || teacher, T) * T² + (1-α_kl) * Focal(student, true)

    ✅ v3.0 CHANGES:
    - REMOVED MSE (was causing gradient explosion)
    - RESTORED T² scaling (standard KD)
    - Single hyperparameter: α_kl (default 0.5)
    """
    def __init__(
        self,
        alpha=0.5,            # KL weight (was 0.15)
        temperature=2.0,
        gamma=2.0,
        class_weights=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.focal = FocalLoss(alpha=class_weights, gamma=gamma)
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_probs, true_labels):
        # ✅ Ensure same device
        teacher_probs = teacher_probs.to(student_logits.device)

        # ✅ KL Divergence with T² scaling (STANDARD KD)
        student_log_softmax = F.log_softmax(student_logits / self.T, dim=1)
        kl_loss = self.kl(student_log_softmax, teacher_probs) * (self.T ** 2)

        # ✅ Focal Loss (true labels)
        focal_loss = self.focal(student_logits, true_labels)

        # ✅ Weighted combination
        return self.alpha * kl_loss + (1 - self.alpha) * focal_loss


# ================================================================
# ✅ Metrics
# ================================================================

def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_recall = []
    for c in range(NUM_CLASSES):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_recall.append(float((y_pred[mask] == c).mean()))
    balanced_acc = float(np.mean(per_class_recall)) if per_class_recall else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    roc_auc, fpr_dict, tpr_dict = {}, {}, {}
    try:
        y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
        for i in range(NUM_CLASSES):
            if y_bin[:, i].sum() > 0 and (1 - y_bin[:, i]).sum() > 0:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                fpr_dict[i] = fpr
                tpr_dict[i] = tpr
                roc_auc[i] = float(auc(fpr, tpr))
    except Exception:
        pass
    roc_auc_mean = float(np.mean(list(roc_auc.values()))) if roc_auc else 0.0
    class_report = classification_report(
        y_true, y_pred, labels=range(NUM_CLASSES),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    for cname in CLASS_NAMES:
        if cname not in class_report:
            class_report[cname] = {
                "precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0
            }
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1_score": f1, "f1_macro": f1_macro, "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc, "roc_auc_mean": roc_auc_mean,
        "fpr": fpr_dict, "tpr": tpr_dict,
        "confusion_matrix": cm, "classification_report": class_report,
        "num_samples": int(len(y_true)),
    }


# ================================================================
# ✅ Load Teacher Architects
# ================================================================

def load_resnet50_teacher(path, device):
    from src.models.resnet.resnet50 import create_resnet50
    ckpt = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "encoder_state_dict" in ckpt:
        state = ckpt["encoder_state_dict"]
    else:
        state = ckpt
    model = create_resnet50(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_b1_teacher(path, device):
    from src.models.efficientNet.B1 import create_efficientnet_b1
    ckpt = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "encoder_state_dict" in ckpt:
        state = ckpt["encoder_state_dict"]
    else:
        state = ckpt
    model = create_efficientnet_b1(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_student_resnet50(device):
    """
    Create a ResNet50 student initialized from the ResNet50 teacher.
    ✅ v3.0: NO dropout modification — keep teacher weights intact.
    """
    from src.models.resnet.resnet50 import create_resnet50

    model = create_resnet50(num_classes=NUM_CLASSES, pretrained=False, verbose=False)

    if os.path.exists(RESNET50_TEACHER):
        try:
            ckpt = torch.load(RESNET50_TEACHER, map_location="cpu")
            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif "encoder_state_dict" in ckpt:
                state = ckpt["encoder_state_dict"]
            else:
                state = ckpt
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"   ✅ Student initialized from ResNet50 teacher weights")
            print(f"      Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            print(f"   ⚠️ Could not init from teacher: {e}")

    # ✅ NO dropout modification — ResNet50 already has enough regularization
    # ✅ Reduce BN momentum (safe, doesn't change weights)
    try:
        bn_count = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.05
                bn_count += 1
        print(f"   ✅ Reduced BN momentum to 0.05 for {bn_count} layers")
    except Exception as e:
        print(f"   ⚠️ Could not reduce BN momentum: {e}")

    return model.to(device)


# ================================================================
# ✅ Teacher Ensemble Inference (returns probs)
# ================================================================

@torch.no_grad()
def teacher_ensemble_probs(resnet_model, b1_model, images_224, images_240, w):
    """Given pre-processed images, return weighted ensemble probs."""
    resnet_logits = resnet_model(images_224)
    b1_logits = b1_model(images_240)
    resnet_probs = F.softmax(resnet_logits.float(), dim=1)
    b1_probs = F.softmax(b1_logits.float(), dim=1)
    return w * resnet_probs + (1 - w) * b1_probs


# ================================================================
# ✅ Learn Ensemble Weights (on val set)
# ================================================================

@torch.no_grad()
def learn_ensemble_weights(resnet_model, b1_model, val_loader, device, step=0.05, verbose=True):
    """
    Find best weight w in [0,1] for weighted ensemble:
        probs = w * resnet_probs + (1-w) * b1_probs
    """
    resnet_model.eval()
    b1_model.eval()

    all_resnet_probs = []
    all_b1_probs = []
    all_labels = []

    for imgs, labels in tqdm(val_loader, desc="   Learning weights", leave=False, disable=not verbose):
        imgs = imgs.to(device)
        # ResNet: input 224 (already)
        resnet_logits = resnet_model(imgs)
        resnet_probs = F.softmax(resnet_logits.float(), dim=1)
        # B1: resize to 240
        imgs_240 = F.interpolate(imgs, size=(240, 240), mode="bilinear", align_corners=False)
        b1_logits = b1_model(imgs_240)
        b1_probs = F.softmax(b1_logits.float(), dim=1)

        all_resnet_probs.append(resnet_probs.cpu())
        all_b1_probs.append(b1_probs.cpu())
        all_labels.append(labels)

    resnet_probs_all = torch.cat(all_resnet_probs, dim=0).numpy()
    b1_probs_all = torch.cat(all_b1_probs, dim=0).numpy()
    y_val = torch.cat(all_labels, dim=0).numpy()

    best_w = 0.5
    best_acc = -1.0
    for w in np.arange(0.0, 1.0 + step, step):
        probs = w * resnet_probs_all + (1 - w) * b1_probs_all
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_w = w

    if verbose and HAS_RICH:
        console.print(f"   [cyan]Best Weight:[/cyan] ResNet50={best_w:.2f}, B1={1-best_w:.2f} | Val Acc={best_acc:.4f}")

    return float(best_w)


# ================================================================
# ✅ KD Training
# ================================================================

def kd_train_student(
    student, resnet_teacher, b1_teacher, train_loader, val_loader,
    device, ensemble_w, epochs=50, alpha=0.7, temperature=3.0,
    backbone_lr=1e-5, head_lr=1e-3, weight_decay=1e-4, patience=15,
    class_weights=None, save_path=None, verbose=True,
) -> Dict[str, Any]:
    """
    Train student via Knowledge Distillation from teacher ensemble.
    """
    best_f1 = -float("inf")
    best_val_loss = float("inf")   # ← جدید
    best_state = None
    patience_counter = 0
    best_epoch = 0
    student = student.to(device)
    resnet_teacher = resnet_teacher.to(device).eval()
    b1_teacher = b1_teacher.to(device).eval()

    # Loss
    criterion = KDLoss(alpha=alpha, temperature=temperature,
                       gamma=2.0, class_weights=class_weights)
    criterion = criterion.to(device)

    # ✅ FIX: Move class_weights to device too
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Layer-wise LR
    backbone_params, head_params = [], []
    HEAD_KEYWORDS = ["fc", "classifier", "head"]
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        is_head = any(k in name.lower() for k in HEAD_KEYWORDS)
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    # ✅ FIX: Stronger weight decay (L2 regularization)
    # Weight decay = 1e-2 for backbone, 1e-3 for head
    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": 1e-2,   # ✅ L2 قوی‌تر
            "name": "backbone",
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": 1e-3,   # ✅ head یکم کمتر
            "name": "head",
        })
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))

    # Scheduler
    warmup_epochs = min(3, max(1, epochs // 10))
    cosine_T = max(1, epochs - warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_T, eta_min=backbone_lr * 0.01)
    warmup_sch = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_sch, cosine], milestones=[warmup_epochs])

    # Mixed precision
    use_amp = device.type == "cuda"
    amp_dtype = None
    scaler = None
    amp_name = "disabled"
    if use_amp:
        try:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
                scaler = None
                amp_name = "BF16"
            else:
                amp_dtype = torch.float16
                scaler = torch.amp.GradScaler("cuda", enabled=True)
                amp_name = "FP16"
        except Exception:
            use_amp = False

    if verbose and HAS_RICH:
        console.print(f"   [cyan]Mixed Precision:[/cyan] [bold]{amp_name}[/bold]")
        console.print(f"   [cyan]Teacher Weight:[/cyan] ResNet50={ensemble_w:.2f}, B1={1-ensemble_w:.2f}")

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "kd_loss": [], "focal_loss": [],
        "learning_rates": [],
    }

    best_f1 = -float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        # ===== TRAIN =====
        student.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        pbar = tqdm(train_loader, desc=f"   KD Epoch {epoch+1}/{epochs}", leave=False,
                    disable=not verbose, ncols=100)

        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Get teacher probs (no_grad)
            with torch.no_grad():
                # ResNet input: 224 (already)
                resnet_logits = resnet_teacher(imgs)
                resnet_probs = F.softmax(resnet_logits.float(), dim=1)
                # B1 input: 240
                imgs_240 = F.interpolate(imgs, size=(240, 240), mode="bilinear", align_corners=False)
                b1_logits = b1_teacher(imgs_240)
                b1_probs = F.softmax(b1_logits.float(), dim=1)
                # Ensemble probs
                teacher_probs = ensemble_w * resnet_probs + (1 - ensemble_w) * b1_probs

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    student_logits = student(imgs)
                    loss = criterion(student_logits, teacher_probs, labels)
            else:
                student_logits = student(imgs)
                loss = criterion(student_logits, teacher_probs, labels)

            # ✅ FIX: Stronger gradient clipping (0.5 instead of 1.0)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
                optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                preds = student_logits.float().argmax(dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Compute train metrics (dummy prob for speed)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="weighted", zero_division=0)

        # ===== VALIDATION =====
        student.eval()
        val_loss = 0.0
        val_preds, val_labels, val_probs = [], [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.no_grad():
                    resnet_logits = resnet_teacher(imgs)
                    resnet_probs = F.softmax(resnet_logits.float(), dim=1)
                    imgs_240 = F.interpolate(imgs, size=(240, 240), mode="bilinear", align_corners=False)
                    b1_logits = b1_teacher(imgs_240)
                    b1_probs = F.softmax(b1_logits.float(), dim=1)
                    teacher_probs = ensemble_w * resnet_probs + (1 - ensemble_w) * b1_probs

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        student_logits = student(imgs)
                        loss = criterion(student_logits, teacher_probs, labels)
                else:
                    student_logits = student(imgs)
                    loss = criterion(student_logits, teacher_probs, labels)

                val_loss += loss.item()
                probs = F.softmax(student_logits.float(), dim=1)
                val_preds.extend(probs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.append(probs.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_metrics = compute_metrics(
            np.array(val_labels), np.array(val_preds),
            np.concatenate(val_probs, axis=0),
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_f1"].append(float(train_f1))
        history["val_f1"].append(val_metrics["f1_score"])
        history["learning_rates"].append(current_lr)

        scheduler.step()

        if verbose:
            print(f"   📊 Epoch {epoch+1}/{epochs} | "
                  f"Train L: {avg_train_loss:.4f} F1: {train_f1:.4f} | "
                  f"Val L: {avg_val_loss:.4f} F1: {val_metrics['f1_score']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f}")

        # ✅ FIX: Early stopping with F1 + Val Loss check
        improved = False
        if val_metrics["f1_score"] > best_f1 + 1e-3:  # ← بزرگ‌تر از 1e-4
            improved = True

        # ✅ FIX: Also stop if Val Loss explodes (>3x best)
        if avg_val_loss > 3.0 * (best_val_loss if best_val_loss < float('inf') else avg_val_loss):
            if verbose:
                print(f"   🛑 VAL LOSS EXPLOSION at epoch {epoch + 1} — stopping!")
            break

        if improved:
            best_f1 = val_metrics["f1_score"]
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "model_state_dict": best_state,
                    "epoch": epoch + 1,
                    "best_f1": best_f1,
                    "history": history,
                }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"   🛑 Early stopping at epoch {epoch + 1}")
                break

    # Load best
    if best_state is not None:
        student.load_state_dict(best_state)
        student = student.to(device)

    return {
        "history": history,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_val_metrics": val_metrics,
    }


# ================================================================
# ✅ Polish Fine-tune (short, on true labels)
# ================================================================

def polish_finetune(
    student, train_loader, val_loader, device,
    epochs=20, lr=1e-6, patience=10, class_weights=None,
    save_path=None, verbose=True,
) -> Dict[str, Any]:
    """Polish fine-tune with plain Focal Loss (no KD)."""
    student = student.to(device)

    # ✅ FIX: Move class_weights to device
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    use_amp = device.type == "cuda"
    amp_dtype = None
    scaler = None
    if use_amp:
        try:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
                scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            use_amp = False

    history = {"train_loss": [], "val_loss": [],
               "train_f1": [], "val_f1": [], "val_acc": []}

    best_f1 = -float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        # Train
        student.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        for imgs, labels in tqdm(train_loader, desc=f"   Polish {epoch+1}/{epochs}",
                                  leave=False, disable=not verbose):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = student(imgs)
                    loss = criterion(logits, labels)
            else:
                logits = student(imgs)
                loss = criterion(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item()
            with torch.no_grad():
                train_preds.extend(logits.float().argmax(dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_f1 = f1_score(train_labels, train_preds, average="weighted", zero_division=0)

        # Val
        student.eval()
        val_loss = 0.0
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits = student(imgs)
                        loss = criterion(logits, labels)
                else:
                    logits = student(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                probs = F.softmax(logits.float(), dim=1)
                val_preds.extend(probs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.append(probs.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_metrics = compute_metrics(
            np.array(val_labels), np.array(val_preds),
            np.concatenate(val_probs, axis=0),
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_f1"].append(float(train_f1))
        history["val_f1"].append(val_metrics["f1_score"])
        history["val_acc"].append(val_metrics["accuracy"])

        if verbose:
            print(f"   📊 Polish {epoch+1}/{epochs} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_metrics['f1_score']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics["f1_score"] > best_f1 + 1e-4:
            best_f1 = val_metrics["f1_score"]
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
            if save_path:
                torch.save({
                    "model_state_dict": best_state,
                    "epoch": epoch + 1,
                    "best_f1": best_f1,
                    "history": history,
                }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"   🛑 Polish early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        student.load_state_dict(best_state)
        student = student.to(device)

    return {
        "history": history,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_val_metrics": val_metrics,
    }


# ================================================================
# ⚠️ ادامه در پیام بعدی ⚠️
# ================================================================
# بخش ۲ شامل:
# - TTA evaluation
# - CV5
# - CI95% Bootstrap
# - Plot generators
# - PDF generator (30+ pages)
# - Main pipeline
# ================================================================

# ================================================================
# ✅ TTA Evaluation
# ================================================================

@torch.no_grad()
def evaluate_with_tta_student(
        model, image_paths, labels, device, input_size=224,
        n_aug=32, batch_size=8, verbose=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate single student model with TTA."""
    model.eval()
    model.to(device)
    num_samples = len(image_paths)

    tta_dataset = TTADataset(image_paths, labels, input_size, n_aug)
    tta_loader = DataLoader(tta_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    per_aug_probs = [[] for _ in range(num_samples)]
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    pbar = tqdm(tta_loader, desc="   TTA Inference", leave=False, disable=not verbose)
    for imgs, lbls, img_idx in pbar:
        imgs = imgs.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(imgs)
        else:
            logits = model(imgs)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        for i, idx in enumerate(img_idx.numpy()):
            per_aug_probs[idx].append(probs[i])

    averaged_probs = np.zeros((num_samples, NUM_CLASSES))
    for i in range(num_samples):
        if len(per_aug_probs[i]) > 0:
            averaged_probs[i] = np.mean(per_aug_probs[i], axis=0)

    preds = averaged_probs.argmax(axis=1)
    return preds, averaged_probs


# ================================================================
# ✅ CI 95% Bootstrap
# ================================================================

def compute_ci95_bootstrap(y_true, y_pred, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_accs, boot_f1s, boot_precs, boot_recs = [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t, y_p = y_true[idx], y_pred[idx]
        if len(np.unique(y_t)) < 2:
            continue
        boot_accs.append(accuracy_score(y_t, y_p))
        boot_f1s.append(f1_score(y_t, y_p, average="weighted", zero_division=0))
        boot_precs.append(precision_score(y_t, y_p, average="weighted", zero_division=0))
        boot_recs.append(recall_score(y_t, y_p, average="weighted", zero_division=0))

    def ci(arr):
        arr = np.array(arr)
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
        return {
            "mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }

    return {
        "accuracy": ci(boot_accs), "f1_score": ci(boot_f1s),
        "precision": ci(boot_precs), "recall": ci(boot_recs),
    }


# ================================================================
# ✅ CV5 (lightweight — student eval on each fold)
# ================================================================

def run_cv5_student(
        student, image_paths, labels, device, input_size=224,
        n_folds=5, n_aug_tta=32, verbose=True,
) -> Dict[str, Any]:
    """
    Lightweight CV5: Use the trained student, evaluate on 5 different splits.
    This is a sanity check — not a full CV training.
    """
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_results = []

    if verbose and HAS_RICH:
        console.print(f"\n   [bold cyan]Running {n_folds}-fold evaluation...[/bold cyan]")

    for fold_idx, (_, val_idx) in enumerate(skf.split(image_paths, labels_arr)):
        if verbose:
            print(f"\n   📌 Fold {fold_idx + 1}/{n_folds}")

        val_paths = [image_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # TTA eval on fold
        preds, probs = evaluate_with_tta_student(
            student, val_paths, val_labels, device,
            input_size=input_size, n_aug=n_aug_tta, batch_size=8, verbose=False,
        )
        metrics = compute_metrics(val_labels, preds, probs)

        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "roc_auc_mean": metrics["roc_auc_mean"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        })

        if verbose:
            print(f"      Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

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

    if verbose and HAS_RICH:
        console.print(f"\n   [bold cyan]CV Summary:[/bold cyan]")
        console.print(f"      Mean Accuracy: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        console.print(f"      Mean F1:       {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        console.print(f"      Mean AUC:      {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}")

    return cv_summary


# ================================================================
# ✅ Plot Generators
# ================================================================

def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 14, "weight": "bold"})
    plt.ylabel("True Label", fontweight="bold", fontsize=12)
    plt.xlabel("Predicted Label", fontweight="bold", fontsize=12)
    plt.title(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_roc_curves(metrics, save_path, title="ROC Curves"):
    plt.figure(figsize=(9, 8))
    colors_roc = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i in range(NUM_CLASSES):
        if i in metrics["fpr"] and i in metrics["tpr"]:
            fpr = metrics["fpr"][i]
            tpr = metrics["tpr"][i]
            auc_val = metrics["roc_auc"].get(i, 0)
            plt.plot(fpr, tpr, color=colors_roc[i], linewidth=2.5,
                     label=f"{CLASS_NAMES[i]} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Random")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontweight="bold", fontsize=12)
    plt.ylabel("True Positive Rate", fontweight="bold", fontsize=12)
    plt.title(title, fontweight="bold", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.text(0.55, 0.15, f"Mean AUC = {metrics['roc_auc_mean']:.4f}",
             fontsize=12, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F39C12", alpha=0.3))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_metrics_bar(metrics, save_path, title="Metrics"):
    names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1", "AUC"]
    values = [
        metrics["accuracy"], metrics["balanced_accuracy"],
        metrics["precision"], metrics["recall"],
        metrics["f1_score"], metrics["roc_auc_mean"],
    ]
    colors_bar = ["#27ae60", "#2ecc71", "#3498db", "#2980b9", "#e74c3c", "#c0392b"]
    plt.figure(figsize=(11, 6))
    bars = plt.bar(names, values, color=colors_bar, edgecolor="black", linewidth=1.2)
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


def plot_per_class_metrics(metrics, save_path, title="Per-Class Metrics"):
    per_class = metrics["classification_report"]
    precisions = [per_class[c]["precision"] for c in CLASS_NAMES]
    recalls = [per_class[c]["recall"] for c in CLASS_NAMES]
    f1s = [per_class[c]["f1-score"] for c in CLASS_NAMES]
    x = np.arange(len(CLASS_NAMES))
    width = 0.27
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width, precisions, width, label="Precision", color="#3498db")
    bars2 = plt.bar(x, recalls, width, label="Recall", color="#2ecc71")
    bars3 = plt.bar(x + width, f1s, width, label="F1-Score", color="#e74c3c")
    for bars in [bars1, bars2, bars3]:
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


def plot_ci_bars(ci95, save_path):
    metrics_to_plot = ["accuracy", "f1_score", "precision", "recall"]
    labels = ["Accuracy", "F1", "Precision", "Recall"]
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, metric in enumerate(metrics_to_plot):
        ci = ci95[metric]
        mean, low, high = ci["mean"], ci["ci95_low"], ci["ci95_high"]
        y_pos = len(metrics_to_plot) - i - 1
        ax.plot([low, high], [y_pos, y_pos], color="#3498db", linewidth=8, alpha=0.4)
        ax.scatter([mean], [y_pos], color="#e74c3c", s=200, marker="D", zorder=4,
                   edgecolor="white", linewidth=2)
        ax.text(high + 0.005, y_pos, f"  {mean:.4f} [{low:.4f}, {high:.4f}]",
                va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(metrics_to_plot)))
    ax.set_yticklabels(labels[::-1], fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.15)
    plt.title("95% Confidence Interval (Bootstrap, 1000 samples)", fontweight="bold", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_cv_folds(cv_summary, save_path):
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
               linewidth=2, label=f"Mean Acc: {cv_summary['mean_accuracy']:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("5-Fold CV Results (Lightweight)", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_training_curves(history, save_path, model_name="KD Student"):
    """Plot training curves from KD and Polish."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_f1 = history.get("train_f1", [])
    val_f1 = history.get("val_f1", [])
    val_acc = history.get("val_acc", [])
    lr = history.get("learning_rates", train_loss)

    epochs = range(1, len(train_loss) + 1)

    axes[0, 0].plot(epochs, train_loss, label="Train", linewidth=2, color="#3498db")
    axes[0, 0].plot(epochs, val_loss, label="Val", linewidth=2, color="#e74c3c")
    axes[0, 0].set_xlabel("Epoch");
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves", fontweight="bold");
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, val_acc, label="Val Accuracy", linewidth=2, color="#2ecc71")
    axes[0, 1].set_xlabel("Epoch");
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Validation Accuracy", fontweight="bold");
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, train_f1, label="Train", linewidth=2, color="#9b59b6")
    axes[1, 0].plot(epochs, val_f1, label="Val", linewidth=2, color="#1abc9c")
    axes[1, 0].set_xlabel("Epoch");
    axes[1, 0].set_ylabel("F1-Score")
    axes[1, 0].set_title("F1-Score Curves", fontweight="bold");
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if lr:
        axes[1, 1].plot(epochs[:len(lr)], lr, linewidth=2, color="#e67e22")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xlabel("Epoch");
        axes[1, 1].set_ylabel("Learning Rate (log)")
        axes[1, 1].set_title("LR Schedule", fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} — Training Curves", fontweight="bold", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ================================================================
# ✅ PDF Generator (30+ pages)
# ================================================================

def generate_kd_pdf(
        pdf_path, kd_results, polish_results, student_metrics, ci95, cv_summary,
        plots, ensemble_w, n_test, device,
):
    """Generate comprehensive PDF report."""
    try:
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm, inch
        from reportlab.platypus import (
            Image as RLImage, PageBreak, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        print_error("reportlab not installed")
        return None

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=1.5 * cm, leftMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title="Knowledge Distillation Ensemble Report",
        author="Babak Yousefian",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Heading1"],
                                 fontSize=26, spaceAfter=20, alignment=TA_CENTER,
                                 textColor=rl_colors.HexColor("#1a252f"))
    h1_style = ParagraphStyle("H1", parent=styles["Heading2"],
                              fontSize=18, spaceAfter=12, spaceBefore=12,
                              textColor=rl_colors.HexColor("#2c3e50"))
    h2_style = ParagraphStyle("H2", parent=styles["Heading3"],
                              fontSize=14, spaceAfter=8,
                              textColor=rl_colors.HexColor("#34495e"))
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
    center = ParagraphStyle("C", parent=body, alignment=TA_CENTER)

    def add_img(path, w=6.5 * inch, h=4.5 * inch):
        if path and os.path.exists(path) and os.path.getsize(path) > 100:
            try:
                return RLImage(path, width=w, height=h)
            except Exception:
                return Paragraph("⚠️ Image load failed", body)
        return Paragraph("⚠️ Image not available", body)

    story = []

    # ===== PAGE 1: Title =====
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Knowledge Distillation Ensemble Report", title_style))
    story.append(Paragraph("ResNet50 (SimCLR) + EfficientNet-B1 (BYOL) → Student ResNet50",
                           ParagraphStyle("sub", parent=body, alignment=TA_CENTER,
                                          fontSize=14, textColor=rl_colors.grey)))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", center))
    story.append(Paragraph(f"Test Samples: {n_test}", center))
    story.append(Paragraph(f"Device: {device}", center))
    story.append(Paragraph("Author: Babak Yousefian", center))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Student: ResNet50 (trained via KD)", center))
    story.append(Paragraph("Teacher Ensemble: ResNet50 + B1", center))
    story.append(Paragraph("Pipeline: KD → Polish → TTA → CV5 → CI95% → PDF", center))
    story.append(PageBreak())

    # ===== PAGE 2: Executive Summary =====
    story.append(Paragraph("Executive Summary", h1_style))

    story.append(Paragraph("<b>Pipeline Overview:</b>", h2_style))
    story.append(Paragraph(
        f"<b>Step 1:</b> Loaded two teacher models (ResNet50 + B1)<br/>"
        f"<b>Step 2:</b> Learned ensemble weight on val set: "
        f"ResNet50={ensemble_w:.2f}, B1={1 - ensemble_w:.2f}<br/>"
        f"<b>Step 3:</b> Trained ResNet50 student via Knowledge Distillation "
        f"({kd_results['best_epoch']} epochs, best F1={kd_results['best_f1']:.4f})<br/>"
        f"<b>Step 4:</b> Polish fine-tune "
        f"({polish_results['best_epoch']} epochs, best F1={polish_results['best_f1']:.4f})<br/>"
        f"<b>Step 5:</b> Full evaluation with TTA + CV5 + CI 95%<br/>",
        body,
    ))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("<b>Final Student Metrics (TTA on Test Set):</b>", h2_style))

    summary_data = [["Metric", "Value"]]
    for k in ["accuracy", "balanced_accuracy", "precision", "recall",
              "f1_score", "f1_macro", "roc_auc_mean"]:
        if k in student_metrics:
            summary_data.append([k.replace("_", " ").title(), f"{student_metrics[k]:.4f}"])

    t = Table(summary_data, colWidths=[2.5 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ===== PAGE 3: Detailed Metrics =====
    story.append(Paragraph("Detailed Metrics", h1_style))

    # Main
    story.append(Paragraph("<b>Overall Performance:</b>", h2_style))
    met_data = [["Metric", "Value"]]
    for k in ["accuracy", "balanced_accuracy", "precision", "recall",
              "f1_score", "f1_macro", "roc_auc_mean", "num_samples"]:
        v = student_metrics.get(k, "N/A")
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

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>Per-Class Metrics:</b>", h2_style))
    pc_data = [["Class", "Precision", "Recall", "F1", "Support"]]
    for cname in CLASS_NAMES:
        if cname in student_metrics["classification_report"]:
            r = student_metrics["classification_report"][cname]
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
    story.append(PageBreak())

    # ===== PAGE 4+: Plots =====
    plot_pages = [
        ("cm", "Confusion Matrix"),
        ("roc", "ROC Curves"),
        ("metrics", "Metrics Bar"),
        ("per_class", "Per-Class Metrics"),
    ]
    for key, title in plot_pages:
        if plots.get(key):
            story.append(Paragraph(title, h1_style))
            story.append(add_img(plots[key], w=6 * inch, h=5 * inch))
            story.append(PageBreak())

    # ===== CV Section =====
    story.append(Paragraph("5-Fold Cross-Validation (Lightweight)", h1_style))

    story.append(Paragraph(
        "The trained student model is evaluated on 5 different stratified folds. "
        "This measures robustness of the learned features on different data splits.",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))

    cv_data = [["Fold", "Accuracy", "F1", "AUC", "Balanced Acc"]]
    for r in cv_summary["fold_results"]:
        cv_data.append([
            f"Fold {r['fold']}",
            f"{r['accuracy']:.4f}",
            f"{r['f1_score']:.4f}",
            f"{r['roc_auc_mean']:.4f}",
            f"{r['balanced_accuracy']:.4f}",
        ])
    cv_data.append([
        "Mean ± Std",
        f"{cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}",
        f"{cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}",
        f"{cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}",
        "—",
    ])

    t = Table(cv_data, colWidths=[1.2 * inch, 1.5 * inch, 1.5 * inch, 1.3 * inch, 1.3 * inch])
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

    if plots.get("cv_folds"):
        story.append(Spacer(1, 0.2 * inch))
        story.append(add_img(plots["cv_folds"], w=6.5 * inch, h=4 * inch))
    story.append(PageBreak())

    # ===== CI 95% =====
    story.append(Paragraph("95% Confidence Interval (Bootstrap)", h1_style))

    ci_data = [["Metric", "Mean", "95% CI Low", "95% CI High", "Std"]]
    for metric_name, ci in ci95.items():
        ci_data.append([
            metric_name.replace("_", " ").title(),
            f"{ci['mean']:.4f}",
            f"{ci['ci95_low']:.4f}",
            f"{ci['ci95_high']:.4f}",
            f"{ci['std']:.4f}",
        ])

    t = Table(ci_data, colWidths=[1.8 * inch, 1.2 * inch, 1.4 * inch, 1.4 * inch, 1.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#E74C3C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ]))
    story.append(t)

    if plots.get("ci95"):
        story.append(Spacer(1, 0.2 * inch))
        story.append(add_img(plots["ci95"], w=6.5 * inch, h=4.5 * inch))
    story.append(PageBreak())

    # ===== Training Curves =====
    story.append(Paragraph("Training Curves", h1_style))
    story.append(Paragraph(
        "This section shows KD training curves (ResNet50 student learning from teacher ensemble). "
        "Polish fine-tune is a short additional phase for refinement.",
        body,
    ))
    if plots.get("kd_curves"):
        story.append(add_img(plots["kd_curves"], w=6.5 * inch, h=4.5 * inch))
        story.append(PageBreak())

    # ===== Final Verdict =====
    story.append(Paragraph("Final Verdict", h1_style))

    verdict = f"""
    <b>Knowledge Distillation Result:</b><br/>
    The ResNet50 student was successfully trained to mimic the ensemble 
    (ResNet50 + B1) with weight {ensemble_w:.2f} / {1 - ensemble_w:.2f}.<br/><br/>

    <b>Final Student Metrics (TTA on Test):</b><br/>
    - Accuracy: {student_metrics['accuracy']:.4f}<br/>
    - F1-Score: {student_metrics['f1_score']:.4f}<br/>
    - AUC: {student_metrics['roc_auc_mean']:.4f}<br/>
    - Balanced Accuracy: {student_metrics['balanced_accuracy']:.4f}<br/><br/>

    <b>CV Mean (Lightweight):</b> {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}<br/><br/>

    <b>95% CI (Bootstrap):</b> [{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]<br/><br/>

    <b>Recommendation:</b> The distilled student is a single .pth model that 
    captures the ensemble's knowledge. It's ready for deployment.
    """
    story.append(Paragraph(verdict, body))
    story.append(PageBreak())

    # ===== Last Page =====
    story.append(Spacer(1, 3 * inch))
    story.append(Paragraph("✅ Report Complete", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"Author: Babak Yousefian<br/>"
        f"Version: 1.0.0 (Knowledge Distillation)",
        center,
    ))

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
# ✅ MAIN PIPELINE
# ================================================================

def run_kd_ensemble_pipeline(
        device="cuda",
        kd_epochs=50,
        polish_epochs=20,
        batch_size=16,
        backbone_lr=1e-5,
        head_lr=1e-3,
        polish_lr=1e-6,
        patience=15,
        polish_patience=10,
        alpha=0.7,
        temperature=3.0,
        n_aug_tta=32,
        n_bootstrap=1000,
        cv_folds=5,
        verbose=True,
) -> Dict[str, Any]:
    """Main Knowledge Distillation pipeline."""

    print_header("KNOWLEDGE DISTILLATION ENSEMBLE",
                 "ResNet50 + B1 → Student ResNet50 (.pth)")

    set_seed(SEED)
    ensure_all_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if HAS_RICH:
        console.print(f"[cyan]Device:[/cyan] [bold]{device}[/bold]")

    # ===== Step 1: Check teachers =====
    print_step(1, 8, "Checking teacher models...")
    status = check_teachers_exist()
    if HAS_RICH:
        table = Table(box=box.ROUNDED, title="Teacher Check")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="bold")
        for key, exists in status.items():
            icon = "[green]✅ Found[/green]" if exists else "[red]❌ Not Found[/red]"
            table.add_row(key, icon)
        console.print(table)
    if not all(status.values()):
        print_error("Missing teachers")
        return {"status": "failed"}
    print_success("Both teachers found!")

    # ===== Step 2: Load data =====
    print_step(2, 8, "Loading data...")
    image_paths, labels = load_labeled_data()
    print_success(f"Loaded {len(image_paths)} labeled images")

    # Split: 80/10/10
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.20, stratify=labels, random_state=SEED,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED,
    )
    if HAS_RICH:
        console.print(f"[cyan]Train:[/cyan] {len(train_paths)} | "
                      f"[cyan]Val:[/cyan] {len(val_paths)} | "
                      f"[cyan]Test:[/cyan] {len(test_paths)}")

    # Class weights
    train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = len(train_labels) / (NUM_CLASSES * np.maximum(train_counts, 1))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    # ===== Step 3: Load teachers =====
    print_step(3, 8, "Loading teacher models...")
    resnet_teacher = load_resnet50_teacher(RESNET50_TEACHER, device)
    b1_teacher = load_b1_teacher(B1_TEACHER, device)
    print_success("Teachers loaded")

    # ===== Step 4: Learn ensemble weight =====
    print_step(4, 8, "Learning ensemble weights...")

    # Build val loader (ResNet size 224)
    val_ds = LabeledDataset(val_paths, val_labels, input_size=224, augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    best_w = learn_ensemble_weights(resnet_teacher, b1_teacher, val_loader, device)
    print_success(f"Ensemble weight: ResNet50={best_w:.2f}, B1={1 - best_w:.2f}")

    # ===== Step 5: KD training =====
    print_step(5, 8, "Knowledge Distillation training...")

    # ✅ FIX: Use lighter augmentation to reduce overfit
    # ✅ FIX: Lighter augmentation
    train_ds = LabeledDataset(train_paths, train_labels, input_size=224,
                              augment=True, augment_strength="light")

    # ✅ FIX: WeightedRandomSampler for class balance
    from torch.utils.data import WeightedRandomSampler
    train_labels_arr = np.array(train_labels)
    class_counts = np.bincount(train_labels_arr, minlength=NUM_CLASSES)
    class_counts = np.maximum(class_counts, 1)
    class_weights_sampler = 1.0 / class_counts
    sample_weights = class_weights_sampler[train_labels_arr]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,  # ✅ جای shuffle
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    # Student
    student = load_student_resnet50(device)

    kd_results = kd_train_student(
        student=student,
        resnet_teacher=resnet_teacher,
        b1_teacher=b1_teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        ensemble_w=best_w,
        epochs=kd_epochs,
        alpha=alpha,
        temperature=temperature,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=1e-4,
        patience=patience,
        class_weights=class_weights_t,
        save_path=STUDENT_PATH,
        verbose=verbose,
    )
    print_success(f"KD done — Best F1: {kd_results['best_f1']:.4f}")

    # ===== Step 6: Polish fine-tune =====
    print_step(6, 8, "Polish fine-tune...")

    # Load best KD state
    if os.path.exists(STUDENT_PATH):
        ckpt = torch.load(STUDENT_PATH, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        student = student.to(device)

    polish_results = polish_finetune(
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=polish_epochs,
        lr=polish_lr,
        patience=polish_patience,
        class_weights=class_weights_t,
        save_path=STUDENT_PATH,
        verbose=verbose,
    )
    print_success(f"Polish done — Best F1: {polish_results['best_f1']:.4f}")

    # ===== Step 7: Full evaluation =====
    print_step(7, 8, "Full evaluation (TTA + CV5 + CI95%)...")

    # Load final student
    if os.path.exists(STUDENT_PATH):
        ckpt = torch.load(STUDENT_PATH, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        student = student.to(device)

    # TTA on test
    print_info(f"TTA on test set ({len(test_paths)} imgs × {n_aug_tta} augs)...")
    test_preds, test_probs = evaluate_with_tta_student(
        student, test_paths, test_labels, device,
        input_size=224, n_aug=n_aug_tta, batch_size=8, verbose=True,
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)

    # CV5
    print_info(f"CV5 (lightweight)...")
    cv_summary = run_cv5_student(
        student, image_paths, labels, device,
        input_size=224, n_folds=cv_folds, n_aug_tta=n_aug_tta, verbose=True,
    )

    # CI 95%
    print_info(f"CI 95% bootstrap ({n_bootstrap} samples)...")
    ci95 = compute_ci95_bootstrap(
        np.array(test_labels), test_preds, n_bootstrap=n_bootstrap,
    )

    # ===== Step 8: Save + PDF =====
    print_step(8, 8, "Saving final_model.pth + Generating PDF...")

    # Save final_model.pth
    torch.save({
        "model_state_dict": student.state_dict(),
        "architecture": "resnet50",
        "pipeline": "knowledge_distillation_ensemble",
        "teacher_ensemble": {
            "resnet50_teacher_path": RESNET50_TEACHER,
            "b1_teacher_path": B1_TEACHER,
            "ensemble_weight_resnet": best_w,
            "ensemble_weight_b1": 1 - best_w,
        },
        "kd_config": {
            "alpha": alpha,
            "temperature": temperature,
            "kd_epochs": kd_epochs,
            "polish_epochs": polish_epochs,
        },
        "test_metrics_tta": {
            k: v for k, v in test_metrics.items()
            if k not in ["all_preds", "all_labels", "all_probs",
                         "fpr", "tpr", "confusion_matrix", "classification_report"]
        },
        "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "test_classification_report": test_metrics["classification_report"],
        "cv_summary": cv_summary,
        "ci95": ci95,
        "kd_history": kd_results["history"],
        "polish_history": polish_results["history"],
        "kd_best_epoch": kd_results["best_epoch"],
        "polish_best_epoch": polish_results["best_epoch"],
        "timestamp": datetime.now().isoformat(),
    }, FINAL_MODEL_PATH)
    print_success(f"Final model saved: {FINAL_MODEL_PATH}")

    # Generate plots
    print_info("Generating plots...")
    plots = {}
    plots["cm"] = plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        os.path.join(KD_PLOTS_DIR, "confusion_matrix.png"),
        "KD Student — Confusion Matrix",
    )
    plots["roc"] = plot_roc_curves(
        test_metrics,
        os.path.join(KD_PLOTS_DIR, "roc_curves.png"),
        "KD Student — ROC Curves",
    )
    plots["metrics"] = plot_metrics_bar(
        test_metrics,
        os.path.join(KD_PLOTS_DIR, "metrics_bar.png"),
        "KD Student — Metrics",
    )
    plots["per_class"] = plot_per_class_metrics(
        test_metrics,
        os.path.join(KD_PLOTS_DIR, "per_class.png"),
        "KD Student — Per-Class Metrics",
    )
    plots["ci95"] = plot_ci_bars(
        ci95,
        os.path.join(KD_PLOTS_DIR, "ci95.png"),
    )
    plots["cv_folds"] = plot_cv_folds(
        cv_summary,
        os.path.join(KD_PLOTS_DIR, "cv_folds.png"),
    )
    plots["kd_curves"] = plot_training_curves(
        kd_results["history"],
        os.path.join(KD_PLOTS_DIR, "kd_curves.png"),
        "KD Student (Teacher Distillation)",
    )

    # Generate PDF
    pdf_path = generate_kd_pdf(
        pdf_path=KD_PDF_PATH,
        kd_results=kd_results,
        polish_results=polish_results,
        student_metrics=test_metrics,
        ci95=ci95,
        cv_summary=cv_summary,
        plots=plots,
        ensemble_w=best_w,
        n_test=len(test_paths),
        device=device,
    )

    # Save metrics JSON
    metrics_json = {
        "timestamp": datetime.now().isoformat(),
        "ensemble_weight_resnet": best_w,
        "ensemble_weight_b1": 1 - best_w,
        "kd_results": {
            "best_f1": kd_results["best_f1"],
            "best_epoch": kd_results["best_epoch"],
        },
        "polish_results": {
            "best_f1": polish_results["best_f1"],
            "best_epoch": polish_results["best_epoch"],
        },
        "test_metrics": {
            k: v for k, v in test_metrics.items()
            if k not in ["all_preds", "all_labels", "all_probs",
                         "fpr", "tpr", "confusion_matrix", "classification_report"]
        },
        "cv_summary": cv_summary,
        "ci95": ci95,
    }
    with open(KD_METRICS_JSON, "w") as f:
        json.dump(metrics_json, f, indent=4, default=str)

    # ===== Summary =====
    if HAS_RICH:
        console.print()
        table = Table(box=box.DOUBLE_EDGE, title="🎉 Final Results", border_style="green")
        table.add_column("Metric", style="cyan bold")
        table.add_column("Value", style="green bold")
        table.add_row("Ensemble Weight (ResNet50)", f"{best_w:.4f}")
        table.add_row("Ensemble Weight (B1)", f"{1 - best_w:.4f}")
        table.add_row("KD Best F1", f"{kd_results['best_f1']:.4f}")
        table.add_row("Polish Best F1", f"{polish_results['best_f1']:.4f}")
        table.add_row("Test Accuracy (TTA)", f"{test_metrics['accuracy']:.4f}")
        table.add_row("Test F1 (TTA)", f"{test_metrics['f1_score']:.4f}")
        table.add_row("Test AUC (TTA)", f"{test_metrics['roc_auc_mean']:.4f}")
        table.add_row("CV Mean Accuracy", f"{cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        table.add_row("CI 95% Accuracy",
                      f"[{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]")
        console.print(table)

        console.print(f"\n[bold green]📄 PDF:[/bold green] {pdf_path}")
        console.print(f"[bold green]🎯 Final model:[/bold green] {FINAL_MODEL_PATH}")
        console.print(f"[bold green]📊 Metrics JSON:[/bold green] {KD_METRICS_JSON}")

    return {
        "status": "success",
        "final_model_path": FINAL_MODEL_PATH,
        "student_path": STUDENT_PATH,
        "pdf_path": pdf_path,
        "metrics_json": KD_METRICS_JSON,
        "ensemble_weight": best_w,
        "kd_results": kd_results,
        "polish_results": polish_results,
        "test_metrics": test_metrics,
        "cv_summary": cv_summary,
        "ci95": ci95,
    }


# ================================================================
# ✅ CLI Entry
# ================================================================

if __name__ == "__main__":
    results = run_kd_ensemble_pipeline()
    if results.get("status") == "success":
        print("\n🎉 KD Ensemble pipeline complete!")
        print(f"   Final model: {results['final_model_path']}")
        print(f"   PDF: {results['pdf_path']}")
