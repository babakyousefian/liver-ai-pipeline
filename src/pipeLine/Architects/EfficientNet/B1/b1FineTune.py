# src/pipeLine/b1FineTune.py
# -*- coding: utf-8 -*-

"""
EfficientNet-B1 Complete Fine-Tune Pipeline (ULTIMATE EDITION v8.0)

✅ v8.0 IMPROVEMENTS:
- WeightedRandomSampler for imbalanced classes
- Early stopping on F1 (not loss)
- Layer-wise Learning Rates (backbone vs head)
- Two-Stage Fine-Tuning (freeze → unfreeze)
- Test-Time Augmentation (TTA) for evaluation
- Multi-Seed Ensemble (3 seeds)

Author: Babak Yousefian
"""

import copy
import gc
import json
import os
import random
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
# ✅ Paths
# ================================================================

PROJECT_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")
EFFICIENTNET_WEIGHT_TRANSFORMER_DIR = os.path.join(
    SELF_SUPERVISED_ROOT, "EfficientNet", "weightTransformer"
)
REPORTER_ROOT = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "b1_finetune")

ARCH_DIR = os.path.join(REPORTER_ROOT, "architecture")
CV_DIR = os.path.join(REPORTER_ROOT, "cv")
INTERP_DIR = os.path.join(REPORTER_ROOT, "interpretability")
PLOTS_DIR = os.path.join(REPORTER_ROOT, "plots")
ENSEMBLE_DIR = os.path.join(REPORTER_ROOT, "ensemble")
FINAL_MODEL_PATH = os.path.join(REPORTER_ROOT, "b1_finetuned_final.pth")
METRICS_JSON_PATH = os.path.join(REPORTER_ROOT, "metrics.json")

# ================================================================
# ✅ Constants
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE_B1 = 240
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ✅ Ensemble seeds (v8.1: expanded)
ENSEMBLE_SEEDS = [42, 123, 999, 1337, 2024]  # 5 seeds


# ================================================================
# ✅ Seed
# ================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set to: {seed}")


# ================================================================
# ✅ Ensure Directories
# ================================================================

def ensure_all_dirs() -> None:
    dirs = [REPORTER_ROOT, ARCH_DIR, INTERP_DIR, PLOTS_DIR, ENSEMBLE_DIR]
    for fold in range(1, 6):
        dirs.append(os.path.join(CV_DIR, f"fold_{fold}"))
    for seed in ENSEMBLE_SEEDS:
        dirs.append(os.path.join(ENSEMBLE_DIR, f"seed_{seed}"))
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ All directories created under: {REPORTER_ROOT}")


# ================================================================
# ✅ Check Transferred Model
# ================================================================

def check_transferred_model(path: str = None) -> Tuple[bool, str]:
    if path is None:
        candidates = [
            os.path.join(EFFICIENTNET_WEIGHT_TRANSFORMER_DIR, "b1_transferred_from_byol.pth"),
            os.path.join(EFFICIENTNET_WEIGHT_TRANSFORMER_DIR, "b1_transferred_from_simclr.pth"),
            os.path.join(EFFICIENTNET_WEIGHT_TRANSFORMER_DIR, "b1_transferred_from_mae.pth"),
            os.path.join(EFFICIENTNET_WEIGHT_TRANSFORMER_DIR, "efficinetnet_b1_best_base.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return True, c
        return False, candidates[0]
    return os.path.exists(path), path


# ================================================================
# ✅ FOCAL LOSS (v8.1)
# ================================================================

class FocalLoss(nn.Module):
    """
    Enhanced Focal Loss with Label Smoothing + MSE (v9.0).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    ✅ v9.0 ADDITIONS:
    - Label Smoothing (reduces overconfidence)
    - MSE on probs vs one-hot (extra regularization)

    Args:
        alpha: Weighting factor for each class
        gamma: Focusing parameter (2.0 recommended)
        reduction: 'mean' | 'sum' | 'none'
        label_smoothing: Smoothing factor (0.05 recommended) ✅ NEW
        mse_lambda: MSE weight in total loss ✅ NEW

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean',
                 label_smoothing=0.05, mse_lambda=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.mse_lambda = mse_lambda

    def forward(self, inputs, targets):
        # ✅ Compute CE with label smoothing (per-sample)
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing,   # ✅ NEW
        )

        # Get probabilities
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal modulating factor
        focal_weight = (1.0 - p_t) ** self.gamma

        # Apply alpha (class weighting)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.to(inputs.device)[targets]
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        # ✅ MSE on probs vs one-hot (NEW)
        if self.mse_lambda > 0:
            one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()
            mse_loss = F.mse_loss(p, one_hot, reduction='mean')
            loss = loss + self.mse_lambda * mse_loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ================================================================
# ✅ TTA-Augmented Dataset (for Test-Time Augmentation)
# ================================================================

class TTADataset(Dataset):
    """
    Dataset for Test-Time Augmentation.
    Returns N augmented versions of each image.
    """
    def __init__(self, image_paths, labels, input_size=240, n_aug=8):
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
        """
        ✅ v8.1: 32 deterministic TTA variants.
        Comprehensive coverage of typical image variations.
        """
        # idx 0-7: base augmentations (same as v8.0)
        if aug_idx == 0:  # original
            return img
        elif aug_idx == 1:  # hflip
            return np.fliplr(img).copy()
        elif aug_idx == 2:  # vflip
            return np.flipud(img).copy()
        elif aug_idx == 3:  # rotate +5
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), 5, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 4:  # rotate -5
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -5, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 5:  # rotate +10
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 6:  # rotate -10
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -10, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 7:  # brightness up +10%
            return np.clip(img.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)

        # idx 8-15: intensity variations
        elif aug_idx == 8:  # brightness down -10%
            return np.clip(img.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
        elif aug_idx == 9:  # brightness up +20%
            return np.clip(img.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        elif aug_idx == 10:  # brightness down -20%
            return np.clip(img.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
        elif aug_idx == 11:  # contrast +10%
            mean = img.mean()
            return np.clip((img.astype(np.float32) - mean) * 1.1 + mean, 0, 255).astype(np.uint8)
        elif aug_idx == 12:  # contrast -10%
            mean = img.mean()
            return np.clip((img.astype(np.float32) - mean) * 0.9 + mean, 0, 255).astype(np.uint8)
        elif aug_idx == 13:  # contrast +20%
            mean = img.mean()
            return np.clip((img.astype(np.float32) - mean) * 1.2 + mean, 0, 255).astype(np.uint8)
        elif aug_idx == 14:  # contrast -20%
            mean = img.mean()
            return np.clip((img.astype(np.float32) - mean) * 0.8 + mean, 0, 255).astype(np.uint8)
        elif aug_idx == 15:  # gamma 0.9
            return np.clip((img.astype(np.float32) / 255.0) ** 0.9 * 255, 0, 255).astype(np.uint8)

        # idx 16-23: zoom/scale variations
        elif aug_idx == 16:  # zoom in 0.9
            h, w = img.shape[:2]
            ch, cw = int(h * 0.9), int(w * 0.9)
            y1, x1 = (h - ch) // 2, (w - cw) // 2
            cropped = img[y1:y1+ch, x1:x1+cw]
            return cv2.resize(cropped, (w, h))
        elif aug_idx == 17:  # zoom in 0.85
            h, w = img.shape[:2]
            ch, cw = int(h * 0.85), int(w * 0.85)
            y1, x1 = (h - ch) // 2, (w - cw) // 2
            cropped = img[y1:y1+ch, x1:x1+cw]
            return cv2.resize(cropped, (w, h))
        elif aug_idx == 18:  # zoom out 1.1
            h, w = img.shape[:2]
            nh, nw = int(h * 1.1), int(w * 1.1)
            resized = cv2.resize(img, (nw, nh))
            y1 = (nh - h) // 2
            x1 = (nw - w) // 2
            return resized[y1:y1+h, x1:x1+w]
        elif aug_idx == 19:  # zoom out 1.15
            h, w = img.shape[:2]
            nh, nw = int(h * 1.15), int(w * 1.15)
            resized = cv2.resize(img, (nw, nh))
            y1 = (nh - h) // 2
            x1 = (nw - w) // 2
            return resized[y1:y1+h, x1:x1+w]
        elif aug_idx == 20:  # translate left
            h, w = img.shape[:2]
            M = np.float32([[1, 0, -5], [0, 1, 0]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 21:  # translate right
            h, w = img.shape[:2]
            M = np.float32([[1, 0, 5], [0, 1, 0]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 22:  # translate up
            h, w = img.shape[:2]
            M = np.float32([[1, 0, 0], [0, 1, -5]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 23:  # translate down
            h, w = img.shape[:2]
            M = np.float32([[1, 0, 0], [0, 1, 5]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # idx 24-31: combined transformations (hflip + others)
        elif aug_idx == 24:  # hflip + rot +5
            flipped = np.fliplr(img).copy()
            h, w = flipped.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), 5, 1.0)
            return cv2.warpAffine(flipped, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 25:  # hflip + rot -5
            flipped = np.fliplr(img).copy()
            h, w = flipped.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -5, 1.0)
            return cv2.warpAffine(flipped, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 26:  # hflip + brightness +10
            flipped = np.fliplr(img).copy()
            return np.clip(flipped.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        elif aug_idx == 27:  # hflip + brightness -10
            flipped = np.fliplr(img).copy()
            return np.clip(flipped.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
        elif aug_idx == 28:  # hflip + zoom in 0.9
            flipped = np.fliplr(img).copy()
            h, w = flipped.shape[:2]
            ch, cw = int(h * 0.9), int(w * 0.9)
            y1, x1 = (h - ch) // 2, (w - cw) // 2
            cropped = flipped[y1:y1+ch, x1:x1+cw]
            return cv2.resize(cropped, (w, h))
        elif aug_idx == 29:  # hflip + contrast +10
            flipped = np.fliplr(img).copy()
            mean = flipped.mean()
            return np.clip((flipped.astype(np.float32) - mean) * 1.1 + mean, 0, 255).astype(np.uint8)
        elif aug_idx == 30:  # rotate +15
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif aug_idx == 31:  # rotate -15
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
# ✅ Labeled Dataset (with Two-Stage Augmentation)
# ================================================================

class LabeledDataset(Dataset):
    def __init__(self, image_paths, labels, input_size=240, augment=False,
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
        # ✅ Stronger augmentation for minority classes
        strength = self.augment_strength

        # Horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img).copy()

        # Rotation
        rot_p = 0.3 if strength == "light" else 0.5
        if random.random() < rot_p:
            angle = random.uniform(-15, 15) if strength == "strong" else random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Brightness/Contrast
        if random.random() < 0.6:
            alpha = random.uniform(0.85, 1.15) if strength == "strong" else random.uniform(0.9, 1.1)
            beta = random.uniform(-15, 15) if strength == "strong" else random.uniform(-10, 10)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # ✅ Extra for strong (minority)
        if strength == "strong":
            # Zoom
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
            # Gamma
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


# ================================================================
# ✅ Data Loader
# ================================================================

def load_labeled_data(labels_csv: str = None, image_dir: str = None) -> Tuple[List[str], List[int]]:
    if labels_csv is None:
        labels_csv = os.path.join("finalReports", "labels.csv")
    if image_dir is None:
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

    print(f"✅ Loaded {len(image_paths)} labeled images")
    print(f"   Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = sum(1 for l in labels if l == i)
        print(f"      {name}: {count}")

    return image_paths, labels


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

    # ✅ FIX: Ensure all classes present in report
    # This prevents KeyError in b1Reporter when a class is absent
    for cname in CLASS_NAMES:
        if cname not in class_report:
            class_report[cname] = {
                "precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0
            }

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1_score": f1, "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc, "roc_auc_mean": roc_auc_mean,
        "fpr": fpr_dict, "tpr": tpr_dict,
        "confusion_matrix": cm, "classification_report": class_report,
        "num_samples": int(len(y_true)),
    }


# ================================================================
# ✅ WeightedRandomSampler Builder
# ================================================================

def build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    ✅ Create WeightedRandomSampler for imbalanced classes.
    Minority classes get sampled more frequently.
    """
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr, minlength=NUM_CLASSES)
    class_counts = np.maximum(class_counts, 1)

    # Inverse frequency weights
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_arr]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    print(f"   ✅ WeightedRandomSampler:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"      {name}: count={class_counts[i]}, weight={class_weights[i]:.6f}")

    return sampler


# ================================================================
# ✅ Layer-wise LR Setup
# ================================================================

def get_layerwise_optimizer(model, backbone_lr=1e-5, head_lr=1e-3, weight_decay=5e-3):
    """
    ✅ Layer-wise Learning Rate + STRONG L2 Regularization (v9.0).

    v9.0 CHANGES:
    - Backbone: weight_decay = 2× (stronger L2)
    - Head:     weight_decay = 1× (standard L2)
    - Default weight_decay: 5e-3 (was 1e-4)

    ✅ FIX v8.1: Only the FINAL classifier layer gets head_lr.
    """
    backbone_params = []
    head_params = []

    HEAD_PREFIXES = (
        "fc.",
        "classifier.",
        "_fc.",
        "head.",
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = any(name.startswith(prefix) for prefix in HEAD_PREFIXES)
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    # ✅ v9.0: Different weight_decay for backbone vs head
    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay * 2,   # ✅ L2 قوی‌تر برای backbone
            "name": "backbone",
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,       # ✅ head عادی
            "name": "head",
        })

    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))

    print(f"   ✅ Layer-wise Optimizer (v9.0):")
    print(f"      Backbone params: {len(backbone_params)} (LR={backbone_lr}, WD={weight_decay * 2})")
    print(f"      Head params:     {len(head_params)} (LR={head_lr}, WD={weight_decay})")

    if head_params:
        head_ids = {id(p) for p in head_params}
        head_names = [n for n, p in model.named_parameters() if id(p) in head_ids]
        print(f"      Head layer names: {head_names}")

    return optimizer, backbone_params, head_params


# ================================================================
# ✅ B1 Fine-Tuner (v8.0)
# ================================================================

class B1FineTuner:
    """
    v8.0: Early stopping on F1, WeightedRandomSampler, Layer-wise LR, Two-Stage FT.
    """

    def __init__(self, device: str = "cuda", verbose: bool = True):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.verbose = verbose
        print(f"✅ B1FineTuner device: {self.device}")

    def _get_amp_config(self):
        if self.device.type != "cuda":
            return False, None, None
        try:
            if torch.cuda.is_bf16_supported():
                return True, torch.bfloat16, None
            else:
                return True, torch.float16, torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            return False, None, None

    def _freeze_backbone(self, model):
        """
        Stage 1: Freeze all except classifier head.
        ✅ FIX v8.1: Match by PREFIX to avoid freezing the SE block fc1/fc2 in backbone.
        """
        frozen, trainable = 0, 0

        HEAD_PREFIXES = (
            "fc.",
            "classifier.",
            "_fc.",
            "head.",
        )

        for name, param in model.named_parameters():
            is_head = any(name.startswith(prefix) for prefix in HEAD_PREFIXES)

            if is_head:
                param.requires_grad = True
                trainable += 1
            else:
                param.requires_grad = False
                frozen += 1

        if self.verbose:
            print(f"   🔒 Stage 1: Froze {frozen} params, trainable {trainable}")

    def _unfreeze_all(self, model):
        """Stage 2: Unfreeze all."""
        for param in model.parameters():
            param.requires_grad = True
        if self.verbose:
            print(f"   🔓 Stage 2: All params unfrozen")

    def finetune(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-5,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        save_path: str = None,
        class_weights: torch.Tensor = None,
        warmup_epochs: int = 3,
        two_stage: bool = True,
        stage1_epochs: int = 5,
        # ✅ NEW v8.1: Focal Loss support
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = None,
    ) -> Dict[str, Any]:
        """
        v8.1: Two-Stage FT + Layer-wise LR + Early stopping on F1 + Optional Focal Loss.

        Args:
            use_focal_loss: If True, use Focal Loss instead of CrossEntropy
            focal_gamma: Focusing parameter (2.0 recommended)
            focal_alpha: Class weighting. If None, no alpha. If float, uniform alpha.
                         If tensor, per-class alpha (recommended: inverse class freq).
        """
        model = model.to(self.device)

        # ✅ Loss selection (v8.1)
        if use_focal_loss:
            # Focal Loss: NO class_weights in CE, alpha handles it
            criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction='mean',
            )
            if self.verbose:
                print(f"🎯 Using FOCAL LOSS (gamma={focal_gamma}, alpha={focal_alpha})")
        else:
            # Standard CrossEntropy with class_weights
            if class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            else:
                criterion = nn.CrossEntropyLoss()
            if self.verbose:
                print(f"📊 Using CrossEntropyLoss (weighted={class_weights is not None})")

        use_amp, amp_dtype, scaler = self._get_amp_config()

        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [],
            "train_auc": [], "val_auc": [],
            "learning_rates": [],
        }

        best_metric = -float("inf")  # ← F1
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        best_epoch = 0
        global_epoch = 0

        # ══════════════════════════════════════════════════════════
        # STAGE 1: Frozen backbone, only train head
        # ══════════════════════════════════════════════════════════
        if two_stage:
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"🔒 STAGE 1: Frozen backbone ({stage1_epochs} epochs)")
                print(f"{'='*70}")

            self._freeze_backbone(model)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=head_lr, weight_decay=weight_decay,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=stage1_epochs, eta_min=head_lr * 0.01)

            for epoch in range(stage1_epochs):
                global_epoch += 1
                train_metrics = self._train_one_epoch(
                    model, train_loader, optimizer, criterion, use_amp, amp_dtype, scaler
                )
                val_metrics, avg_val_loss = self._validate(
                    model, val_loader, criterion, use_amp, amp_dtype
                )
                scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                self._record_history(history, train_metrics, val_metrics, avg_val_loss, current_lr)

                if self.verbose:
                    print(
                        f"📊 [S1] Epoch {global_epoch}/{epochs} | "
                        f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {avg_val_loss:.4f} | "
                        f"Val F1: {val_metrics['f1_score']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
                    )

                # Save best based on F1
                if val_metrics["f1_score"] > best_metric + 1e-4:
                    best_metric = val_metrics["f1_score"]
                    best_val_loss = avg_val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = global_epoch
                    if save_path:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save({
                            "model_state_dict": best_state,
                            "epoch": global_epoch,
                            "best_f1": best_metric,
                            "best_val_loss": best_val_loss,
                        }, save_path)

        # ══════════════════════════════════════════════════════════
        # STAGE 2: Unfrozen, layer-wise LR
        # ══════════════════════════════════════════════════════════
        if two_stage:
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"🔓 STAGE 2: Full fine-tune with layer-wise LR")
                print(f"{'='*70}")
            self._unfreeze_all(model)

        remaining_epochs = epochs - (stage1_epochs if two_stage else 0)
        if remaining_epochs <= 0:
            remaining_epochs = epochs

        # Layer-wise optimizer
        optimizer, backbone_params, head_params = get_layerwise_optimizer(
            model, backbone_lr=lr, head_lr=head_lr, weight_decay=weight_decay
        )

        # Scheduler: warmup + cosine
        warmup = min(warmup_epochs, max(1, remaining_epochs // 10))
        cosine_T = max(1, remaining_epochs - warmup)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_T, eta_min=lr * 0.01)
        if warmup > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup]
            )
        else:
            scheduler = cosine_scheduler

        if self.verbose:
            print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            amp_name = "BF16" if amp_dtype == torch.bfloat16 else ("FP16" if use_amp else "disabled")
            print(f"   Mixed precision: {amp_name}")
            print(f"   Early stopping on: F1-Score (patience={patience})")
            print("=" * 70)

        for epoch in range(remaining_epochs):
            global_epoch += 1
            train_metrics = self._train_one_epoch(
                model, train_loader, optimizer, criterion, use_amp, amp_dtype, scaler
            )
            val_metrics, avg_val_loss = self._validate(
                model, val_loader, criterion, use_amp, amp_dtype
            )
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            self._record_history(history, train_metrics, val_metrics, avg_val_loss, current_lr)

            if self.verbose:
                print(
                    f"📊 Epoch {global_epoch}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {avg_val_loss:.4f} | "
                    f"Train F1: {train_metrics['f1_score']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | LR: {current_lr:.2e}"
                )

            # ══════════════════════════════════════════════════════
            # Early stopping on F1 (NOT loss!)
            # ══════════════════════════════════════════════════════
            if val_metrics["f1_score"] > best_metric + 1e-4:
                best_metric = val_metrics["f1_score"]
                best_val_loss = avg_val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = global_epoch
                patience_counter = 0

                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        "model_state_dict": best_state,
                        "epoch": global_epoch,
                        "best_f1": best_metric,
                        "best_val_loss": best_val_loss,
                    }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.verbose:
                        print(f"🛑 Early stopping at epoch {global_epoch} (patience={patience})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(self.device)

        return {
            "history": history,
            "best_val_loss": best_val_loss,
            "best_f1": best_metric,
            "best_epoch": best_epoch,
            "final_val_metrics": val_metrics,
        }

    def _train_one_epoch(self, model, loader, optimizer, criterion, use_amp, amp_dtype, scaler):
        model.train()
        total_loss = 0.0
        preds_all, labels_all, probs_all = [], [], []

        for imgs, labels in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=1)
                preds_all.extend(probs.argmax(dim=1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
                probs_all.append(probs.cpu().numpy())

        avg_loss = total_loss / max(1, len(loader))
        metrics = compute_metrics(np.array(labels_all), np.array(preds_all),
                                   np.concatenate(probs_all, axis=0))
        metrics["loss"] = avg_loss
        return metrics

    def _validate(self, model, loader, criterion, use_amp, amp_dtype):
        model.eval()
        total_loss = 0.0
        preds_all, labels_all, probs_all = [], [], []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                total_loss += loss.item()
                probs = F.softmax(logits.float(), dim=1)
                preds_all.extend(probs.argmax(dim=1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
                probs_all.append(probs.cpu().numpy())

        avg_loss = total_loss / max(1, len(loader))
        metrics = compute_metrics(np.array(labels_all), np.array(preds_all),
                                   np.concatenate(probs_all, axis=0))
        return metrics, avg_loss

    def _record_history(self, history, train_m, val_m, val_loss, lr):
        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])
        history["train_f1"].append(train_m["f1_score"])
        history["val_f1"].append(val_m["f1_score"])
        history["train_auc"].append(train_m["roc_auc_mean"])
        history["val_auc"].append(val_m["roc_auc_mean"])
        history["learning_rates"].append(lr)

    # ═══════════════════════════════════════════════════════════════
    # ✅ TTA Evaluate
    # ═══════════════════════════════════════════════════════════════

    def evaluate_with_tta(
        self,
        model: nn.Module,
        image_paths: List[str],
        labels: List[int],
        n_aug: int = 8,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        ✅ Test-Time Augmentation: average probabilities over N augmented versions.
        """
        model = model.to(self.device)
        model.eval()

        tta_dataset = TTADataset(image_paths, labels, INPUT_SIZE_B1, n_aug=n_aug)
        tta_loader = DataLoader(
            tta_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        use_amp, amp_dtype, _ = self._get_amp_config()

        all_probs = None
        all_labels = None
        num_samples = len(image_paths)

        # Accumulate probs per (sample, aug)
        per_aug_probs = [[] for _ in range(num_samples)]

        with torch.no_grad():
            for imgs, lbls, img_idx in tqdm(tta_loader, desc="TTA", leave=False, disable=not self.verbose):
                imgs = imgs.to(self.device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits = model(imgs)
                else:
                    logits = model(imgs)
                probs = F.softmax(logits.float(), dim=1).cpu().numpy()

                for i, idx in enumerate(img_idx.numpy()):
                    per_aug_probs[idx].append(probs[i])

                if all_labels is None:
                    all_labels = lbls.numpy()

        # Average over augmentations
        averaged_probs = np.zeros((num_samples, NUM_CLASSES))
        for i in range(num_samples):
            if len(per_aug_probs[i]) > 0:
                averaged_probs[i] = np.mean(per_aug_probs[i], axis=0)

        preds = averaged_probs.argmax(axis=1)

        metrics = compute_metrics(np.array(labels), preds, averaged_probs)
        metrics["all_preds"] = preds.tolist()
        metrics["all_labels"] = labels
        metrics["all_probs"] = averaged_probs
        metrics["tta_n_aug"] = n_aug
        return metrics

    def evaluate(self, model, test_loader) -> Dict[str, Any]:
        """Standard evaluate (no TTA)."""
        model = model.to(self.device)
        model.eval()
        use_amp, amp_dtype, _ = self._get_amp_config()

        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Evaluating", leave=False, disable=not self.verbose):
                imgs = imgs.to(self.device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits = model(imgs)
                else:
                    logits = model(imgs)
                probs = F.softmax(logits.float(), dim=1)
                all_preds.extend(probs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.append(probs.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), all_probs)
        metrics["all_preds"] = all_preds
        metrics["all_labels"] = all_labels
        metrics["all_probs"] = all_probs
        return metrics


# ================================================================
# ✅ Bootstrap CI 95%
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
# ✅ 5-Fold CV (with WeightedRandomSampler + Two-Stage FT)
# ================================================================

def cross_validate_5fold(
    transferred_path, image_paths, labels,
    device="cuda", epochs=50, lr=1e-5, head_lr=1e-3,
    batch_size=32, patience=10, verbose=True,
) -> Dict[str, Any]:
    ensure_all_dirs()
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels_arr)):
        if verbose:
            print(f"\n{'='*70}")
            print(f"📌 FOLD {fold_idx+1}/5")
            print(f"{'='*70}")

        train_paths = [image_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # ✅ Two-stage augmentation: strong for training
        train_dataset = LabeledDataset(train_paths, train_labels, INPUT_SIZE_B1,
                                        augment=True, augment_strength="strong")
        val_dataset = LabeledDataset(val_paths, val_labels, INPUT_SIZE_B1, augment=False)

        # ✅ WeightedRandomSampler
        sampler = build_weighted_sampler(train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                   num_workers=0, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True)

        from src.models.efficientNet.B1 import create_efficientnet_b1
        model = create_efficientnet_b1(num_classes=NUM_CLASSES, pretrained=False, verbose=False)

        model = create_efficientnet_b1(num_classes=NUM_CLASSES, pretrained=False, verbose=False)

        ckpt = torch.load(transferred_path, map_location="cpu")
        if "encoder_state_dict" in ckpt:
            state_dict = ckpt["encoder_state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if verbose:
            print(f"   ✅ Loaded transferred: {len(missing)} missing, {len(unexpected)} unexpected")

        # ✅ v9.0: Dropout + BN momentum adjustment for regularization
        try:
            from src.models.efficientNet.B1 import EfficientNetB1  # fallback

            # ✅ Add Dropout before classifier if not already present
            if hasattr(model, "classifier"):
                if isinstance(model.classifier, nn.Sequential):
                    if not any(isinstance(m, nn.Dropout) for m in model.classifier):
                        model.classifier = nn.Sequential(
                            nn.Dropout(p=0.1),  # ✅ Dropout 0.1
                            *list(model.classifier.children()),
                        )
                        if verbose:
                            print("   ✅ Added Dropout(0.1) before classifier (Sequential)")
                elif isinstance(model.classifier, nn.Linear):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Sequential(
                        nn.Dropout(p=0.1),  # ✅ Dropout 0.1
                        nn.Linear(in_features, NUM_CLASSES),
                    )
                    if verbose:
                        print("   ✅ Wrapped classifier with Dropout(0.1) + Linear")
            elif hasattr(model, "_fc"):
                if isinstance(model._fc, nn.Linear):
                    in_features = model._fc.in_features
                    model._fc = nn.Sequential(
                        nn.Dropout(p=0.1),
                        nn.Linear(in_features, NUM_CLASSES),
                    )
                    if verbose:
                        print("   ✅ Added Dropout(0.1) to _fc")

            # ✅ Reduce BatchNorm2D momentum for stronger regularization
            bn_count = 0
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.momentum = 0.05  # از 0.1 به 0.05
                    bn_count += 1
            if verbose and bn_count > 0:
                print(f"   ✅ Reduced BN momentum to 0.05 for {bn_count} BN2d layers")

        except Exception as e:
            if verbose:
                print(f"   ✅ Loaded transferred: {len(missing)} missing, {len(unexpected)} unexpected")

        model = model.to(device)

        # Class weights
        class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        class_weights = len(train_labels) / (NUM_CLASSES * np.maximum(class_counts, 1))
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

        fold_save_path = os.path.join(CV_DIR, f"fold_{fold_idx+1}", "b1_finetuned_best.pth")
        trainer = B1FineTuner(device=device, verbose=verbose)

        finetune_results = trainer.finetune(
            model=model, train_loader=train_loader, val_loader=val_loader,
            epochs=epochs, lr=lr, head_lr=head_lr,
            weight_decay=5e-3,  # ✅ از 1e-4 به 5e-3
            patience=patience, save_path=fold_save_path,
            class_weights=class_weights_t,
            two_stage=True, stage1_epochs=3,
            use_focal_loss=True,
            focal_gamma=2.0,
            focal_alpha=class_weights_t,
        )

        # ✅ TTA evaluate on val
        val_metrics = trainer.evaluate_with_tta(
            model, val_paths, val_labels, n_aug=8, batch_size=batch_size,
        )

        fold_results.append({
            "fold": fold_idx + 1,
            "train_size": len(train_paths), "val_size": len(val_paths),
            "best_epoch": finetune_results["best_epoch"],
            "best_f1": finetune_results["best_f1"],
            "best_val_loss": finetune_results["best_val_loss"],
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1_score": val_metrics["f1_score"],
            "balanced_accuracy": val_metrics["balanced_accuracy"],
            "roc_auc_mean": val_metrics["roc_auc_mean"],
            "confusion_matrix": val_metrics["confusion_matrix"].tolist(),
            "history": finetune_results["history"],
            "saved_path": fold_save_path,
        })

        if verbose:
            print(f"\n   ✅ Fold {fold_idx+1} complete (with TTA):")
            print(f"      Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"      F1:       {val_metrics['f1_score']:.4f}")
            print(f"      AUC:      {val_metrics['roc_auc_mean']:.4f}")

        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_score"] for r in fold_results]
    aucs = [r["roc_auc_mean"] for r in fold_results]

    cv_summary = {
        "fold_results": fold_results,
        "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
        "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
    }

    with open(os.path.join(CV_DIR, "cv_summary.json"), "w") as f:
        json.dump(cv_summary, f, indent=4, default=str)

    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 CV SUMMARY (with TTA)")
        print(f"{'='*70}")
        print(f"   Mean Accuracy: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        print(f"   Mean F1:       {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        print(f"   Mean AUC:      {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}")

    return cv_summary


# ================================================================
# ✅ Single-Seed Final Fine-Tune (with Two-Stage + Layer-wise LR)
# ================================================================

def finetune_final_single_seed(
    transferred_path, image_paths, labels, seed=42,
    device="cuda", epochs=100, lr=1e-5, head_lr=1e-3,
    batch_size=32, patience=15, verbose=True,
    split_seed=42,   # ✅ NEW: fixed split for all seeds
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, List], str]:
    """
    ✅ Single-seed fine-tune with all improvements.
    Returns (model, results, split_info, save_path)

    ✅ FIX: split_seed controls the train/val/test split (SAME for all seeds).
            seed controls model initialization (DIFFERENT per seed).
    """
    set_seed(seed)

    from src.models.efficientNet.B1 import create_efficientnet_b1

    # ✅ FIX: Use split_seed (not seed) for consistent test set across ensemble
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.20, random_state=split_seed, stratify=labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=split_seed, stratify=temp_labels,
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"📌 FINAL FINE-TUNE (seed={seed})")
        print(f"{'='*70}")
        print(f"   Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
        for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
            counts = [sum(1 for l in split_labels if l == i) for i in range(NUM_CLASSES)]
            print(f"      {split_name}: " + ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(NUM_CLASSES)))

    train_dataset = LabeledDataset(train_paths, train_labels, INPUT_SIZE_B1,
                                    augment=True, augment_strength="strong")
    val_dataset = LabeledDataset(val_paths, val_labels, INPUT_SIZE_B1, augment=False)

    # ✅ WeightedRandomSampler
    sampler = build_weighted_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                               num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = create_efficientnet_b1(num_classes=NUM_CLASSES, pretrained=False, verbose=False)

    ckpt = torch.load(transferred_path, map_location="cpu")
    if "encoder_state_dict" in ckpt:
        state_dict = ckpt["encoder_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if verbose:
        print(f"   ✅ Loaded transferred: {len(missing)} missing, {len(unexpected)} unexpected")

    # ✅ v9.0: Adjust Dropout + BN momentum for regularization
    try:
        # ✅ Set Dropout to 0.1 (model already has "dropout" module)
        if hasattr(model, "dropout") and isinstance(model.dropout, nn.Dropout):
            old_p = model.dropout.p
            model.dropout.p = 0.1  # ✅ از 0.2 به 0.1
            if verbose:
                print(f"   ✅ Dropout: {old_p} → 0.1")
        else:
            if verbose:
                print(f"   ⚠️ No 'dropout' module found; skipping dropout adjustment")

        # ✅ Reduce BatchNorm2D momentum from 0.1 to 0.05
        bn_count = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.05
                bn_count += 1
        if verbose and bn_count > 0:
            print(f"   ✅ Reduced BN momentum 0.1 → 0.05 for {bn_count} BN2d layers")

    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not adjust regularization: {e}")

    model = model.to(device)

    # ✅ v9.0: Dropout + BN momentum adjustment for regularization
    try:
        from src.models.efficientNet.B1 import EfficientNetB1  # fallback

        # ✅ Add Dropout before classifier if not already present
        if hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                if not any(isinstance(m, nn.Dropout) for m in model.classifier):
                    model.classifier = nn.Sequential(
                        nn.Dropout(p=0.1),   # ✅ Dropout 0.1
                        *list(model.classifier.children()),
                    )
                    if verbose:
                        print("   ✅ Added Dropout(0.1) before classifier (Sequential)")
            elif isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.1),   # ✅ Dropout 0.1
                    nn.Linear(in_features, NUM_CLASSES),
                )
                if verbose:
                    print("   ✅ Wrapped classifier with Dropout(0.1) + Linear")
        elif hasattr(model, "_fc"):
            if isinstance(model._fc, nn.Linear):
                in_features = model._fc.in_features
                model._fc = nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Linear(in_features, NUM_CLASSES),
                )
                if verbose:
                    print("   ✅ Added Dropout(0.1) to _fc")

        # ✅ Reduce BatchNorm2D momentum for stronger regularization
        bn_count = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.05   # از 0.1 به 0.05
                bn_count += 1
        if verbose and bn_count > 0:
            print(f"   ✅ Reduced BN momentum to 0.05 for {bn_count} BN2d layers")

    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not adjust model regularization: {e}")

    model = model.to(device)

    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = len(train_labels) / (NUM_CLASSES * np.maximum(class_counts, 1))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    seed_save_path = os.path.join(ENSEMBLE_DIR, f"seed_{seed}", "b1_best.pth")
    os.makedirs(os.path.dirname(seed_save_path), exist_ok=True)

    trainer = B1FineTuner(device=device, verbose=verbose)
    finetune_results = trainer.finetune(
        model=model, train_loader=train_loader, val_loader=val_loader,
        epochs=epochs, lr=lr, head_lr=head_lr,
        weight_decay=5e-3,  # ✅ از 1e-4 به 5e-3
        patience=patience, save_path=seed_save_path,
        class_weights=class_weights_t,
        two_stage=True, stage1_epochs=5,
        use_focal_loss=True,
        focal_gamma=2.0,
        focal_alpha=class_weights_t,
    )

    split_info = {
        "train_paths": train_paths, "train_labels": train_labels,
        "val_paths": val_paths, "val_labels": val_labels,
        "test_paths": test_paths, "test_labels": test_labels,
    }

    return model, finetune_results, split_info, seed_save_path


# ================================================================
# ✅ Ensemble of 3 Seeds
# ================================================================

def train_ensemble_and_evaluate(
    transferred_path, image_paths, labels,
    device="cuda", epochs=100, lr=1e-5, head_lr=1e-3,
    batch_size=32, patience=15, verbose=True, n_aug_tta=32,
) -> Dict[str, Any]:
    """
    ✅ Train 3 models with different seeds, ensemble their TTA-averaged probabilities.
    """
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING ENSEMBLE OF {len(ENSEMBLE_SEEDS)} SEEDS")
    print(f"{'='*80}")

    ensemble_models = []
    ensemble_results = []
    test_paths_shared = None
    test_labels_shared = None

    for seed in ENSEMBLE_SEEDS:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")

        model, results, split_info, save_path = finetune_final_single_seed(
            transferred_path=transferred_path,
            image_paths=image_paths, labels=labels, seed=seed,
            device=device, epochs=epochs, lr=lr, head_lr=head_lr,
            batch_size=batch_size, patience=patience, verbose=verbose,
        )

        # Reload best state
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        ensemble_models.append(model)
        ensemble_results.append(results)

        if test_paths_shared is None:
            test_paths_shared = split_info["test_paths"]
            test_labels_shared = split_info["test_labels"]

    # ══════════════════════════════════════════════════════════
    # Ensemble Evaluation with TTA
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"🎯 ENSEMBLE EVALUATION (TTA n_aug={n_aug_tta})")
    print(f"{'='*80}")

    trainer = B1FineTuner(device=device, verbose=False)

    # Build TTA loader ONCE
    tta_dataset = TTADataset(test_paths_shared, test_labels_shared, INPUT_SIZE_B1, n_aug=n_aug_tta)
    tta_loader = DataLoader(tta_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    num_samples = len(test_paths_shared)
    per_aug_probs = [np.zeros(NUM_CLASSES) for _ in range(num_samples)]
    counts = [0] * num_samples

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for imgs, _, img_idx in tqdm(tta_loader, desc="Ensemble TTA", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            img_idx_np = img_idx.numpy()

            # Average across ensemble models
            batch_probs = None
            for model in ensemble_models:
                model.eval()
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits = model(imgs)
                else:
                    logits = model(imgs)
                probs = F.softmax(logits.float(), dim=1).cpu().numpy()
                if batch_probs is None:
                    batch_probs = probs
                else:
                    batch_probs = batch_probs + probs

            batch_probs = batch_probs / len(ensemble_models)

            for i, idx in enumerate(img_idx_np):
                per_aug_probs[idx] = per_aug_probs[idx] + batch_probs[i]
                counts[idx] += 1

    # Average over augmentations
    final_probs = np.zeros((num_samples, NUM_CLASSES))
    for i in range(num_samples):
        if counts[i] > 0:
            final_probs[i] = per_aug_probs[i] / counts[i]

    preds = final_probs.argmax(axis=1)

    ensemble_metrics = compute_metrics(np.array(test_labels_shared), preds, final_probs)
    ensemble_metrics["all_preds"] = preds.tolist()
    ensemble_metrics["all_labels"] = test_labels_shared
    ensemble_metrics["all_probs"] = final_probs

    print(f"\n🏆 ENSEMBLE RESULTS (with TTA):")
    print(f"   Accuracy:  {ensemble_metrics['accuracy']:.4f}")
    print(f"   F1-Score:  {ensemble_metrics['f1_score']:.4f}")
    print(f"   AUC:       {ensemble_metrics['roc_auc_mean']:.4f}")
    print(f"   Balanced:  {ensemble_metrics['balanced_accuracy']:.4f}")

    return {
        "ensemble_models": ensemble_models,
        "ensemble_results": ensemble_results,
        "ensemble_metrics": ensemble_metrics,
        "test_paths": test_paths_shared,
        "test_labels": test_labels_shared,
        "seeds": ENSEMBLE_SEEDS,
    }


# ================================================================
# ✅ Main Entry
# ================================================================

def run_b1_finetune_pipeline(
    transferred_path=None,
    epochs_final=100, epochs_cv=50,
    lr_final=5e-6, lr_cv=5e-6, head_lr=5e-4,   # ✅ کاهش LR
    batch_size=32, patience_final=15, patience_cv=10,
    n_bootstrap=1000, device="cuda", verbose=True,
) -> Dict[str, Any]:
    """
    v8.0 Pipeline:
    1. Ensemble of 3 seeds (with Two-Stage FT + WeightedSampler + Layer-wise LR + TTA)
    2. 5-Fold CV (with same improvements + TTA)
    3. Bootstrap CI on ensemble predictions
    """
    print("\n" + "=" * 80)
    print("🚀 B1 COMPLETE FINE-TUNE PIPELINE (v8.0)")
    print("=" * 80)

    ensure_all_dirs()

    print("\n🔍 Step 1: Checking transferred model...")
    exists, actual_path = check_transferred_model(transferred_path)
    if not exists:
        raise FileNotFoundError(f"❌ Transferred model not found! Please run option 12 first.")
    print(f"✅ Found: {actual_path}")

    print("\n🔍 Step 2: Loading labeled data...")
    image_paths, labels = load_labeled_data()

    # ══════════════════════════════════════════════════════════
    # Step 3: Ensemble (3 seeds) + TTA
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"🧠 Step 3: ENSEMBLE of {len(ENSEMBLE_SEEDS)} seeds + TTA")
    print("=" * 80)

    ensemble_out = train_ensemble_and_evaluate(
        transferred_path=actual_path,
        image_paths=image_paths, labels=labels,
        device=device, epochs=epochs_final, lr=lr_final, head_lr=head_lr,
        batch_size=batch_size, patience=patience_final, verbose=verbose,
        n_aug_tta=32,
    )

    ensemble_metrics = ensemble_out["ensemble_metrics"]
    test_paths = ensemble_out["test_paths"]
    test_labels = ensemble_out["test_labels"]

    # Use single-seed history for reporting (from best seed)
    best_seed_idx = int(np.argmax([r["best_f1"] for r in ensemble_out["ensemble_results"]]))
    best_seed_history = ensemble_out["ensemble_results"][best_seed_idx]["history"]

    # ══════════════════════════════════════════════════════════
    # Step 4: 5-Fold CV
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("🔬 Step 4: 5-Fold Cross-Validation")
    print("=" * 80)

    cv_summary = cross_validate_5fold(
        transferred_path=actual_path,
        image_paths=image_paths, labels=labels,
        device=device, epochs=epochs_cv, lr=lr_cv, head_lr=head_lr,
        batch_size=batch_size, patience=patience_cv, verbose=verbose,
    )

    # ══════════════════════════════════════════════════════════
    # Step 5: Bootstrap CI on ensemble TEST predictions
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📊 Step 5: Bootstrap CI 95% (on ensemble TEST predictions)")
    print("=" * 80)

    ci95 = compute_ci95_bootstrap(
        np.array(ensemble_metrics["all_labels"]),
        np.array(ensemble_metrics["all_preds"]),
        n_bootstrap=n_bootstrap,
    )

    # ══════════════════════════════════════════════════════════
    # Save metrics
    # ══════════════════════════════════════════════════════════
    metrics_to_save = {
        "version": "8.0.0",
        "improvements": [
            "WeightedRandomSampler", "EarlyStoppingOnF1",
            "LayerwiseLR", "TwoStageFT", "TTA", "3-SeedEnsemble",
        ],
        "test_metrics_ensemble_tta": {
            "accuracy": ensemble_metrics["accuracy"],
            "precision": ensemble_metrics["precision"],
            "recall": ensemble_metrics["recall"],
            "f1_score": ensemble_metrics["f1_score"],
            "balanced_accuracy": ensemble_metrics["balanced_accuracy"],
            "roc_auc": ensemble_metrics["roc_auc"],
            "roc_auc_mean": ensemble_metrics["roc_auc_mean"],
            "confusion_matrix": ensemble_metrics["confusion_matrix"].tolist(),
            "classification_report": ensemble_metrics["classification_report"],
            "num_samples": ensemble_metrics["num_samples"],
        },
        "per_seed_metrics": [
            {"seed": s, "best_f1": r["best_f1"], "best_val_loss": r["best_val_loss"]}
            for s, r in zip(ENSEMBLE_SEEDS, ensemble_out["ensemble_results"])
        ],
        "cv_summary": {
            "mean_accuracy": cv_summary["mean_accuracy"],
            "std_accuracy": cv_summary["std_accuracy"],
            "mean_f1": cv_summary["mean_f1"],
            "std_f1": cv_summary["std_f1"],
            "mean_auc": cv_summary["mean_auc"],
            "std_auc": cv_summary["std_auc"],
            "fold_results": [
                {k: v for k, v in r.items() if k not in ("history",)}
                for r in cv_summary["fold_results"]
            ],
        },
        "ci95_bootstrap": ci95,
        "history": best_seed_history,
        "transferred_path": actual_path,
        "final_model_path": FINAL_MODEL_PATH,
        "ensemble_dir": ENSEMBLE_DIR,
        "n_samples": len(image_paths),
        "timestamp": datetime.now().isoformat(),
    }

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=4, default=str)

    # ══════════════════════════════════════════════════════════
    # Save final "ensemble" model as the best single seed
    # ══════════════════════════════════════════════════════════
    best_model = ensemble_out["ensemble_models"][best_seed_idx]
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "best_seed": ENSEMBLE_SEEDS[best_seed_idx],
        "test_metrics_ensemble_tta": {
            k: v for k, v in ensemble_metrics.items()
            if k not in ("all_preds", "all_labels", "all_probs",
                          "confusion_matrix", "fpr", "tpr", "classification_report")
        },
    }, FINAL_MODEL_PATH)

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("🎉 B1 FINE-TUNE v8.0 COMPLETE!")
    print("=" * 80)

    print(f"\n📊 TEST SET Metrics (ENSEMBLE + TTA):")
    print(f"   Accuracy:          {ensemble_metrics['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {ensemble_metrics['balanced_accuracy']:.4f}")
    print(f"   Precision:         {ensemble_metrics['precision']:.4f}")
    print(f"   Recall:            {ensemble_metrics['recall']:.4f}")
    print(f"   F1-Score:          {ensemble_metrics['f1_score']:.4f}")
    print(f"   AUC:               {ensemble_metrics['roc_auc_mean']:.4f}")

    print(f"\n📊 5-Fold CV (with TTA):")
    print(f"   Mean Accuracy: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
    print(f"   Mean F1:       {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")

    print(f"\n📊 95% CI:")
    print(f"   Accuracy: [{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]")
    print(f"   F1:       [{ci95['f1_score']['ci95_low']:.4f}, {ci95['f1_score']['ci95_high']:.4f}]")

    print(f"\n📂 Outputs:")
    print(f"   Final model: {FINAL_MODEL_PATH}")
    print(f"   Metrics:     {METRICS_JSON_PATH}")
    print(f"   Ensemble:    {ENSEMBLE_DIR}")
    print(f"   CV:          {CV_DIR}")

    return {
        "status": "success",
        "transferred_path": actual_path,
        "final_model_path": FINAL_MODEL_PATH,
        "final_metrics": ensemble_metrics,       # ← ENSEMBLE+TTA metrics (TEST)
        "val_metrics": ensemble_metrics,          # alias for compatibility
        "train_metrics_sample": {},               # not used
        "cv_summary": cv_summary,
        "ci95": ci95,
        "history": best_seed_history,
        "split_info": {
            "test_paths": test_paths, "test_labels": test_labels,
            "train_paths": [], "train_labels": [],
            "val_paths": [], "val_labels": [],
        },
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "ensemble_results": [
            {"seed": s, "best_f1": r["best_f1"]}
            for s, r in zip(ENSEMBLE_SEEDS, ensemble_out["ensemble_results"])
        ],
        "ensemble_models": ensemble_out["ensemble_models"],
        "test_paths": test_paths,
        "test_labels": test_labels,
    }
