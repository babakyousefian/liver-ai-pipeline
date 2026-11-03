# src/pipeLine/dualModelReporter.py
# -*- coding: utf-8 -*-

"""
Dual Model Report Generator (v1.0)

Generates a comprehensive PDF report for TWO specific models:
1. ResNet50 Fine-Tuned (from resnet50_finetuned_from_simclr_final.pth)
2. EfficientNet-B1 Fine-Tuned (from b1_finetuned_final.pth)

Features:
- Load both models from predefined paths
- Evaluate with Test-Time Augmentation (TTA)
- Generate comprehensive plots (ROC, CM, metrics, per-class)
- Interpretability (SHAP, Captum, LIME, Grad-CAM)
- Side-by-side comparison
- ~25-page PDF report with all details

Author: Babak Yousefian
Version: 1.0.0
"""

import gc
import json
import os
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
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

# ================================================================
# ✅ Paths
# ================================================================

PROJECT_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")

# ✅ Model paths (exact paths from user)
RESNET50_MODEL_PATH = os.path.join(
    SELF_SUPERVISED_ROOT, "final_models",
    "resnet50_finetuned_from_simclr_final.pth",
)
B1_MODEL_PATH = os.path.join(
    SELF_SUPERVISED_ROOT, "SSL_Reporter", "b1_finetune",
    "b1_finetuned_final.pth",
)

# ✅ Output
OUTPUT_DIR = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "dual_model_report")
PDF_PATH = os.path.join(OUTPUT_DIR, "Dual_Model_Comparison_Report.pdf")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
INTERP_DIR = os.path.join(OUTPUT_DIR, "interpretability")

# ================================================================
# ✅ Constants
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = 4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ================================================================
# ✅ Directories
# ================================================================

def ensure_dirs():
    for d in [OUTPUT_DIR, PLOTS_DIR, INTERP_DIR]:
        os.makedirs(d, exist_ok=True)
    # Sub-dirs for each model
    for model_name in ["resnet50", "b1"]:
        os.makedirs(os.path.join(PLOTS_DIR, model_name), exist_ok=True)
        os.makedirs(os.path.join(INTERP_DIR, model_name), exist_ok=True)


# ================================================================
# ✅ Check Architects Exist
# ================================================================

def check_models_exist() -> Tuple[bool, bool]:
    """
    Check if both models exist.
    Returns: (resnet_exists, b1_exists)
    """
    resnet_exists = os.path.exists(RESNET50_MODEL_PATH)
    b1_exists = os.path.exists(B1_MODEL_PATH)
    return resnet_exists, b1_exists


# ================================================================
# ✅ Dataset (Labeled)
# ================================================================

class LabeledDataset(Dataset):
    """Simple labeled dataset for evaluation."""

    def __init__(self, image_paths, labels, input_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.input_size = input_size

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        return img

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_t = (img_t - mean) / std
        return img_t, self.labels[idx]


# ================================================================
# ✅ TTA Dataset (32 augmentations)
# ================================================================

class TTADataset(Dataset):
    """TTA dataset — 32 deterministic augmentations."""

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
        """32 deterministic TTA variants."""
        # Base
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
        elif aug_idx == 7:
            return np.clip(img.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        # Intensity
        elif aug_idx == 8:
            return np.clip(img.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
        elif aug_idx == 9:
            return np.clip(img.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        elif aug_idx == 10:
            return np.clip(img.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
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
        elif aug_idx == 15:
            return np.clip((img.astype(np.float32) / 255.0) ** 0.9 * 255, 0, 255).astype(np.uint8)
        # Zoom
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
        # Translate
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
        # Combined
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
# ✅ Load Labeled Data
# ================================================================

def load_labeled_data() -> Tuple[List[str], List[int]]:
    """Load labels.csv + cropped/ images."""
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

    print(f"✅ Loaded {len(image_paths)} labeled images")
    for i, name in enumerate(CLASS_NAMES):
        count = sum(1 for l in labels if l == i)
        print(f"   {name}: {count}")

    return image_paths, labels


# ================================================================
# ✅ Compute Metrics
# ================================================================

def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, Any]:
    """Compute all metrics (safe, robust)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Balanced accuracy
    per_class_recall = []
    for c in range(NUM_CLASSES):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_recall.append(float((y_pred[mask] == c).mean()))
    balanced_acc = float(np.mean(per_class_recall)) if per_class_recall else 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

    # ROC per class
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

    # Classification report
    class_report = classification_report(
        y_true, y_pred, labels=range(NUM_CLASSES),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    # Ensure all classes present
    for cname in CLASS_NAMES:
        if cname not in class_report:
            class_report[cname] = {
                "precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0
            }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f1_macro": f1_macro,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "roc_auc_mean": roc_auc_mean,
        "fpr": fpr_dict,
        "tpr": tpr_dict,
        "confusion_matrix": cm,
        "classification_report": class_report,
        "num_samples": int(len(y_true)),
    }


# ================================================================
# ✅ Evaluate with TTA
# ================================================================

def evaluate_with_tta(
    model: nn.Module,
    image_paths: List[str],
    labels: List[int],
    device: torch.device,
    input_size: int = 224,
    n_aug: int = 32,
    batch_size: int = 8,
    num_classes: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model with TTA (average probabilities over N augmentations).
    """
    # ✅ FIX: Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    num_samples = len(image_paths)
    print(f"   🔬 TTA Evaluation: {num_samples} images × {n_aug} augmentations")

    tta_dataset = TTADataset(image_paths, labels, input_size, n_aug)
    tta_loader = DataLoader(
        tta_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    per_aug_probs = [[] for _ in range(num_samples)]
    use_amp = device.type == "cuda"

    pbar = tqdm(tta_loader, desc="   TTA Inference", leave=False, disable=not verbose)
    with torch.no_grad():
        for imgs, lbls, img_idx in pbar:
            imgs = imgs.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
            else:
                logits = model(imgs)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()

            for i, idx in enumerate(img_idx.numpy()):
                per_aug_probs[idx].append(probs[i])

    # Average over augmentations
    averaged_probs = np.zeros((num_samples, num_classes))
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


# ================================================================
# ✅ Plot Generators
# ================================================================

def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    """Confusion matrix heatmap."""
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


def plot_roc_curves(metrics, save_path, title="ROC Curves"):
    """ROC curves per class + mean AUC."""
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

    # Annotate mean AUC
    plt.text(0.55, 0.15, f"Mean AUC = {metrics['roc_auc_mean']:.4f}",
             fontsize=12, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#F39C12", alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_metrics_bar(metrics, save_path, title="Metrics"):
    """Bar chart of main metrics."""
    names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1 (weighted)", "F1 (macro)"]
    values = [
        metrics["accuracy"], metrics["balanced_accuracy"],
        metrics["precision"], metrics["recall"],
        metrics["f1_score"], metrics["f1_macro"],
    ]
    colors_bar = ["#27ae60", "#2ecc71", "#3498db", "#2980b9", "#e74c3c", "#c0392b"]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(names, values, color=colors_bar, edgecolor="black", linewidth=1.2)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.015,
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
    """Per-class precision/recall/F1 bar chart."""
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
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.01,
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


def plot_class_distribution(metrics, save_path, title="Class Distribution"):
    """Pie chart of test set class distribution."""
    sizes = []
    for c in CLASS_NAMES:
        if c in metrics["classification_report"]:
            sizes.append(metrics["classification_report"][c]["support"])

    if not sizes or sum(sizes) == 0:
        plt.close()
        return None

    plt.figure(figsize=(8, 8))
    colors_pie = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    wedges, texts, autotexts = plt.pie(
        sizes, labels=CLASS_NAMES, autopct="%1.1f%%",
        colors=colors_pie, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    plt.title(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_comparison_bar(metrics_resnet, metrics_b1, save_path):
    """Side-by-side comparison bar chart."""
    metrics_names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1", "AUC"]
    resnet_vals = [
        metrics_resnet["accuracy"], metrics_resnet["balanced_accuracy"],
        metrics_resnet["precision"], metrics_resnet["recall"],
        metrics_resnet["f1_score"], metrics_resnet["roc_auc_mean"],
    ]
    b1_vals = [
        metrics_b1["accuracy"], metrics_b1["balanced_accuracy"],
        metrics_b1["precision"], metrics_b1["recall"],
        metrics_b1["f1_score"], metrics_b1["roc_auc_mean"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.38

    plt.figure(figsize=(13, 6))
    bars1 = plt.bar(x - width/2, resnet_vals, width, label="ResNet50 (SimCLR)",
                    color="#3498db", edgecolor="black")
    bars2 = plt.bar(x + width/2, b1_vals, width, label="EfficientNet-B1 (BYOL)",
                    color="#e74c3c", edgecolor="black")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.xticks(x, metrics_names, fontsize=11, fontweight="bold")
    plt.ylim(0, 1.10)
    plt.ylabel("Score", fontweight="bold", fontsize=12)
    plt.title("Side-by-Side Comparison: ResNet50 vs EfficientNet-B1",
              fontweight="bold", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_comparison_radar(metrics_resnet, metrics_b1, save_path):
    """Radar chart comparison."""
    metrics_names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1", "AUC"]
    resnet_vals = [
        metrics_resnet["accuracy"], metrics_resnet["balanced_accuracy"],
        metrics_resnet["precision"], metrics_resnet["recall"],
        metrics_resnet["f1_score"], metrics_resnet["roc_auc_mean"],
    ]
    b1_vals = [
        metrics_b1["accuracy"], metrics_b1["balanced_accuracy"],
        metrics_b1["precision"], metrics_b1["recall"],
        metrics_b1["f1_score"], metrics_b1["roc_auc_mean"],
    ]

    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    resnet_vals += resnet_vals[:1]
    b1_vals += b1_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.plot(angles, resnet_vals, "o-", linewidth=2, label="ResNet50 (SimCLR)",
            color="#3498db")
    ax.fill(angles, resnet_vals, alpha=0.25, color="#3498db")
    ax.plot(angles, b1_vals, "o-", linewidth=2, label="EfficientNet-B1 (BYOL)",
            color="#e74c3c")
    ax.fill(angles, b1_vals, alpha=0.25, color="#e74c3c")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Radar Comparison: ResNet50 vs EfficientNet-B1",
                 fontweight="bold", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.15), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ================================================================
# ✅ Generate All Plots for a Model
# ================================================================

def generate_model_plots(metrics, model_name, plots_dir):
    """Generate all plots for a single model."""
    plots = {}

    # Confusion matrix
    plots["confusion_matrix"] = plot_confusion_matrix(
        metrics["confusion_matrix"],
        os.path.join(plots_dir, "confusion_matrix.png"),
        title=f"{model_name} — Confusion Matrix",
    )

    # ROC curves
    plots["roc_curves"] = plot_roc_curves(
        metrics,
        os.path.join(plots_dir, "roc_curves.png"),
        title=f"{model_name} — ROC Curves",
    )

    # Metrics bar
    plots["metrics_bar"] = plot_metrics_bar(
        metrics,
        os.path.join(plots_dir, "metrics_bar.png"),
        title=f"{model_name} — Metrics",
    )

    # Per-class
    plots["per_class"] = plot_per_class_metrics(
        metrics,
        os.path.join(plots_dir, "per_class_metrics.png"),
        title=f"{model_name} — Per-Class Metrics",
    )

    # Class distribution
    plots["class_dist"] = plot_class_distribution(
        metrics,
        os.path.join(plots_dir, "class_distribution.png"),
        title=f"{model_name} — Test Set Distribution",
    )

    return plots


# ================================================================
# ✅ Load Architects
# ================================================================

def load_resnet50_model(device):
    """Load ResNet50 fine-tuned model."""
    from src.models.resnet.resnet50 import create_resnet50

    # ✅ FIX: Convert to torch.device
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else "cpu")

    if not os.path.exists(RESNET50_MODEL_PATH):
        raise FileNotFoundError(f"ResNet50 model not found: {RESNET50_MODEL_PATH}")

    print(f"\n📂 Loading ResNet50 from: {RESNET50_MODEL_PATH}")
    ckpt = torch.load(RESNET50_MODEL_PATH, map_location="cpu")

    # Try multiple keys
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "encoder_state_dict" in ckpt:
        state_dict = ckpt["encoder_state_dict"]
    else:
        state_dict = ckpt

    model = create_resnet50(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Extract metadata
    metadata = {
        "path": RESNET50_MODEL_PATH,
        "source_method": ckpt.get("best_method", "simclr"),
        "best_lp_accuracy": ckpt.get("best_lp_accuracy", 0.0),
        "test_metrics": ckpt.get("test_metrics", {}),
        "cv_summary": ckpt.get("cv_summary", {}),
        "size_mb": os.path.getsize(RESNET50_MODEL_PATH) / 1e6,
    }

    print(f"   ✅ Loaded ResNet50 ({metadata['size_mb']:.1f} MB)")
    print(f"   Source method: {metadata['source_method'].upper()}")

    return model, metadata


def load_b1_model(device):
    """Load EfficientNet-B1 fine-tuned model."""
    from src.models.efficientNet.B1 import create_efficientnet_b1

    # ✅ FIX: Convert to torch.device
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else "cpu")

    if not os.path.exists(B1_MODEL_PATH):
        raise FileNotFoundError(f"B1 model not found: {B1_MODEL_PATH}")

    print(f"\n📂 Loading EfficientNet-B1 from: {B1_MODEL_PATH}")
    ckpt = torch.load(B1_MODEL_PATH, map_location="cpu")

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "encoder_state_dict" in ckpt:
        state_dict = ckpt["encoder_state_dict"]
    else:
        state_dict = ckpt

    model = create_efficientnet_b1(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    metadata = {
        "path": B1_MODEL_PATH,
        "ensemble_seeds": ckpt.get("ensemble_seeds", []),
        "best_seed": ckpt.get("best_seed", None),
        "test_metrics": ckpt.get("test_metrics_ensemble_tta", {}),
        "size_mb": os.path.getsize(B1_MODEL_PATH) / 1e6,
    }

    print(f"   ✅ Loaded EfficientNet-B1 ({metadata['size_mb']:.1f} MB)")
    print(f"   Ensemble seeds: {metadata['ensemble_seeds']}")

    return model, metadata


# ================================================================
# ✅ Interpretability (SHAP, Captum, LIME)
# ================================================================

def generate_interpretability(model, image_paths, labels, output_dir, device, model_name="model"):
    """Generate SHAP, Captum, LIME."""
    results = {}

    # Build small eval loader
    dataset = LabeledDataset(image_paths[:50], labels[:50], input_size=224)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # ✅ Captum (Grad-CAM + IG)
    try:
        from src.reporter.captum_reporter import generate_interpretability_report
        captum_res = generate_interpretability_report(
            model, loader, output_dir, device=str(device),
            class_names=CLASS_NAMES, num_samples=8,
        )
        results["gradcam"] = captum_res.get("gradcam_path")
        results["ig"] = captum_res.get("ig_path")
        print(f"   ✅ Captum done")
    except Exception as e:
        print(f"   ⚠️ Captum failed: {e}")

    # ✅ SHAP
    try:
        from src.reporter.shap_reporter import generate_shap_explanation
        shap_res = generate_shap_explanation(
            model, loader, output_dir, device=str(device),
            num_background=30, num_explain=8, class_names=CLASS_NAMES,
        )
        results["shap"] = shap_res.get("shap_path")
        print(f"   ✅ SHAP done")
    except Exception as e:
        print(f"   ⚠️ SHAP failed: {e}")

    # ✅ LIME
    try:
        from src.reporter.lime_reporter import explain_predictions_lime
        lime_res = explain_predictions_lime(
            model, loader, output_dir, class_names=CLASS_NAMES,
            device=str(device), num_samples=3,
        )
        if isinstance(lime_res, dict):
            results["lime"] = lime_res.get("lime_path")
        print(f"   ✅ LIME done")
    except Exception as e:
        print(f"   ⚠️ LIME failed: {e}")

    return results


# ================================================================
# ✅ PDF Generator (25 pages)
# ================================================================

def generate_dual_model_pdf(
    output_path,
    resnet_metrics,
    b1_metrics,
    resnet_metadata,
    b1_metadata,
    resnet_plots,
    b1_plots,
    comparison_plots,
    resnet_interp,
    b1_interp,
    n_samples,
):
    """Generate ~25-page comprehensive PDF report."""
    try:
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm, inch
        from reportlab.platypus import (
            Image as RLImage, PageBreak, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        print("❌ reportlab not installed")
        return None

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        title="Dual Model Comparison Report",
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

    def add_img(path, w=6.5*inch, h=4.5*inch):
        if path and os.path.exists(path) and os.path.getsize(path) > 100:
            try:
                return RLImage(path, width=w, height=h)
            except Exception:
                return Paragraph("⚠️ Image load failed", body)
        return Paragraph("⚠️ Image not available", body)

    story = []

    # ============================================================
    # PAGE 1: Title
    # ============================================================
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("Dual Model Comparison Report", title_style))
    story.append(Paragraph("ResNet50 (SimCLR) vs EfficientNet-B1 (BYOL)",
                            ParagraphStyle("sub", parent=body, alignment=TA_CENTER,
                                            fontSize=14, textColor=rl_colors.grey)))
    story.append(Spacer(1, 0.5*inch))

    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            center))
    story.append(Paragraph(f"<b>Test Samples:</b> {n_samples}", center))
    story.append(Paragraph("<b>Author:</b> Babak Yousefian", center))

    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("📊 Both models evaluated on the same held-out test set",
                            center))
    story.append(Paragraph("with Test-Time Augmentation (TTA, n_aug=32)",
                            center))
    story.append(PageBreak())

    # ============================================================
    # PAGE 2: Executive Summary
    # ============================================================
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Spacer(1, 0.2*inch))

    # Comparison table
    summary_data = [
        ["Metric", "ResNet50 (SimCLR)", "EfficientNet-B1 (BYOL)", "Winner"],
        ["Accuracy", f"{resnet_metrics['accuracy']:.4f}",
         f"{b1_metrics['accuracy']:.4f}",
         "ResNet50" if resnet_metrics['accuracy'] > b1_metrics['accuracy'] else "B1"],
        ["Balanced Acc", f"{resnet_metrics['balanced_accuracy']:.4f}",
         f"{b1_metrics['balanced_accuracy']:.4f}",
         "ResNet50" if resnet_metrics['balanced_accuracy'] > b1_metrics['balanced_accuracy'] else "B1"],
        ["Precision", f"{resnet_metrics['precision']:.4f}",
         f"{b1_metrics['precision']:.4f}",
         "ResNet50" if resnet_metrics['precision'] > b1_metrics['precision'] else "B1"],
        ["Recall", f"{resnet_metrics['recall']:.4f}",
         f"{b1_metrics['recall']:.4f}",
         "ResNet50" if resnet_metrics['recall'] > b1_metrics['recall'] else "B1"],
        ["F1 (weighted)", f"{resnet_metrics['f1_score']:.4f}",
         f"{b1_metrics['f1_score']:.4f}",
         "ResNet50" if resnet_metrics['f1_score'] > b1_metrics['f1_score'] else "B1"],
        ["F1 (macro)", f"{resnet_metrics['f1_macro']:.4f}",
         f"{b1_metrics['f1_macro']:.4f}",
         "ResNet50" if resnet_metrics['f1_macro'] > b1_metrics['f1_macro'] else "B1"],
        ["AUC (mean)", f"{resnet_metrics['roc_auc_mean']:.4f}",
         f"{b1_metrics['roc_auc_mean']:.4f}",
         "ResNet50" if resnet_metrics['roc_auc_mean'] > b1_metrics['roc_auc_mean'] else "B1"],
    ]

    t = Table(summary_data, colWidths=[1.6*inch, 1.6*inch, 1.6*inch, 1.2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ============================================================
    # PAGE 3-4: Model 1 — ResNet50
    # ============================================================
    story.append(Paragraph("Model 1: ResNet50 (from SimCLR SSL)", h1_style))

    story.append(Paragraph("<b>Configuration:</b>", h2_style))
    story.append(Paragraph(
        f"<b>Path:</b> {resnet_metadata['path']}<br/>"
        f"<b>Size:</b> {resnet_metadata['size_mb']:.2f} MB<br/>"
        f"<b>Source Method:</b> {resnet_metadata['source_method'].upper()}<br/>"
        f"<b>Best LP Accuracy:</b> {resnet_metadata['best_lp_accuracy']:.4f}<br/>"
        f"<b>Test Samples:</b> {resnet_metrics['num_samples']}<br/>"
        f"<b>TTA Augmentations:</b> {resnet_metrics['tta_n_aug']}<br/>",
        body,
    ))

    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Test Metrics:</b>", h2_style))

    metrics_data = [
        ["Metric", "Value"],
        ["Accuracy", f"{resnet_metrics['accuracy']:.4f}"],
        ["Balanced Accuracy", f"{resnet_metrics['balanced_accuracy']:.4f}"],
        ["Precision", f"{resnet_metrics['precision']:.4f}"],
        ["Recall", f"{resnet_metrics['recall']:.4f}"],
        ["F1 (weighted)", f"{resnet_metrics['f1_score']:.4f}"],
        ["F1 (macro)", f"{resnet_metrics['f1_macro']:.4f}"],
        ["AUC (mean)", f"{resnet_metrics['roc_auc_mean']:.4f}"],
    ]
    t = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#3498DB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
    ]))
    story.append(t)

    # Per-class
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Per-Class Metrics:</b>", h2_style))
    pc_data = [["Class", "Precision", "Recall", "F1", "Support"]]
    for cname in CLASS_NAMES:
        r = resnet_metrics["classification_report"].get(cname, {})
        pc_data.append([
            cname,
            f"{r.get('precision', 0):.4f}",
            f"{r.get('recall', 0):.4f}",
            f"{r.get('f1-score', 0):.4f}",
            str(int(r.get('support', 0))),
        ])
    t = Table(pc_data, colWidths=[1.2*inch]*5)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#3498DB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ResNet50 Plots
    story.append(Paragraph("ResNet50 — Plots", h1_style))
    story.append(add_img(resnet_plots.get("confusion_matrix"), w=5.5*inch, h=4.5*inch))
    story.append(PageBreak())
    story.append(add_img(resnet_plots.get("roc_curves"), w=5.5*inch, h=5*inch))
    story.append(PageBreak())
    story.append(add_img(resnet_plots.get("metrics_bar"), w=6*inch, h=4*inch))
    story.append(PageBreak())
    story.append(add_img(resnet_plots.get("per_class"), w=6*inch, h=4*inch))
    story.append(PageBreak())

    # ============================================================
    # PAGE 8-9: Model 2 — EfficientNet-B1
    # ============================================================
    story.append(Paragraph("Model 2: EfficientNet-B1 (from BYOL SSL)", h1_style))

    story.append(Paragraph("<b>Configuration:</b>", h2_style))
    story.append(Paragraph(
        f"<b>Path:</b> {b1_metadata['path']}<br/>"
        f"<b>Size:</b> {b1_metadata['size_mb']:.2f} MB<br/>"
        f"<b>Ensemble Seeds:</b> {b1_metadata.get('ensemble_seeds', [])}<br/>"
        f"<b>Best Seed:</b> {b1_metadata.get('best_seed', 'N/A')}<br/>"
        f"<b>Test Samples:</b> {b1_metrics['num_samples']}<br/>"
        f"<b>TTA Augmentations:</b> {b1_metrics['tta_n_aug']}<br/>",
        body,
    ))

    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Test Metrics:</b>", h2_style))

    metrics_data = [
        ["Metric", "Value"],
        ["Accuracy", f"{b1_metrics['accuracy']:.4f}"],
        ["Balanced Accuracy", f"{b1_metrics['balanced_accuracy']:.4f}"],
        ["Precision", f"{b1_metrics['precision']:.4f}"],
        ["Recall", f"{b1_metrics['recall']:.4f}"],
        ["F1 (weighted)", f"{b1_metrics['f1_score']:.4f}"],
        ["F1 (macro)", f"{b1_metrics['f1_macro']:.4f}"],
        ["AUC (mean)", f"{b1_metrics['roc_auc_mean']:.4f}"],
    ]
    t = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#E74C3C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
    ]))
    story.append(t)

    # Per-class
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Per-Class Metrics:</b>", h2_style))
    pc_data = [["Class", "Precision", "Recall", "F1", "Support"]]
    for cname in CLASS_NAMES:
        r = b1_metrics["classification_report"].get(cname, {})
        pc_data.append([
            cname,
            f"{r.get('precision', 0):.4f}",
            f"{r.get('recall', 0):.4f}",
            f"{r.get('f1-score', 0):.4f}",
            str(int(r.get('support', 0))),
        ])
    t = Table(pc_data, colWidths=[1.2*inch]*5)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#E74C3C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ]))
    story.append(t)
    story.append(PageBreak())

    # B1 Plots
    story.append(Paragraph("EfficientNet-B1 — Plots", h1_style))
    story.append(add_img(b1_plots.get("confusion_matrix"), w=5.5*inch, h=4.5*inch))
    story.append(PageBreak())
    story.append(add_img(b1_plots.get("roc_curves"), w=5.5*inch, h=5*inch))
    story.append(PageBreak())
    story.append(add_img(b1_plots.get("metrics_bar"), w=6*inch, h=4*inch))
    story.append(PageBreak())
    story.append(add_img(b1_plots.get("per_class"), w=6*inch, h=4*inch))
    story.append(PageBreak())

    # ============================================================
    # COMPARISON SECTION
    # ============================================================
    story.append(Paragraph("Side-by-Side Comparison", h1_style))
    story.append(add_img(comparison_plots.get("comparison_bar"), w=6.5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(add_img(comparison_plots.get("comparison_radar"), w=6*inch, h=6*inch))
    story.append(PageBreak())

    # ============================================================
    # INTERPRETABILITY — ResNet50
    # ============================================================
    story.append(Paragraph("Interpretability — ResNet50", h1_style))

    if resnet_interp.get("gradcam"):
        story.append(Paragraph("Grad-CAM:", h2_style))
        story.append(add_img(resnet_interp["gradcam"], w=6*inch, h=4.5*inch))
        story.append(PageBreak())

    if resnet_interp.get("ig"):
        story.append(Paragraph("Integrated Gradients:", h2_style))
        story.append(add_img(resnet_interp["ig"], w=6*inch, h=4.5*inch))
        story.append(PageBreak())

    if resnet_interp.get("shap"):
        story.append(Paragraph("SHAP Summary:", h2_style))
        story.append(add_img(resnet_interp["shap"], w=6*inch, h=4.5*inch))
        story.append(PageBreak())

    if resnet_interp.get("lime"):
        story.append(Paragraph("LIME Explanation:", h2_style))
        story.append(add_img(resnet_interp["lime"], w=6*inch, h=4.5*inch))
        story.append(PageBreak())

    # ============================================================
    # INTERPRETABILITY — B1
    # ============================================================
    story.append(Paragraph("Interpretability — EfficientNet-B1", h1_style))

    if b1_interp.get("gradcam"):
        story.append(Paragraph("Grad-CAM:", h2_style))
        story.append(add_img(b1_interp["gradcam"], w=6*inch, h=4.5*inch))
        story.append(PageBreak())

    if b1_interp.get("ig"):
        story.append(Paragraph("Integrated Gradients:", h2_style))
        story.append(add_img(b1_interp["ig"], w=6*inch, h=4.5*inch))
        story.append(PageBreak())

    if b1_interp.get("shap"):
        story.append(Paragraph("SHAP Summary:", h2_style))
        story.append(add_img(b1_interp["shap"], w=6*inch, h=4.5*inch))
        story.append(PageBreak())

    if b1_interp.get("lime"):
        story.append(Paragraph("LIME Explanation:", h2_style))
        story.append(add_img(b1_interp["lime"], w=6*inch, h=4.5*inch))
        story.append(PageBreak())

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    story.append(Paragraph("Final Verdict", h1_style))

    # Compute differences
    acc_diff = resnet_metrics['accuracy'] - b1_metrics['accuracy']
    f1_diff = resnet_metrics['f1_score'] - b1_metrics['f1_score']
    auc_diff = resnet_metrics['roc_auc_mean'] - b1_metrics['roc_auc_mean']

    if acc_diff > 0:
        winner_acc = "ResNet50"
    else:
        winner_acc = "EfficientNet-B1"

    if f1_diff > 0:
        winner_f1 = "ResNet50"
    else:
        winner_f1 = "EfficientNet-B1"

    if auc_diff > 0:
        winner_auc = "ResNet50"
    else:
        winner_auc = "EfficientNet-B1"

    verdict_text = f"""
    <b>Best Accuracy:</b> {winner_acc} (diff = {abs(acc_diff):.4f})<br/>
    <b>Best F1:</b> {winner_f1} (diff = {abs(f1_diff):.4f})<br/>
    <b>Best AUC:</b> {winner_auc} (diff = {abs(auc_diff):.4f})<br/>
    <br/>
    <b>Recommendation:</b><br/>
    """
    if acc_diff > 0.02:
        verdict_text += "ResNet50 demonstrates clearly superior performance. " \
                        "Recommend using ResNet50 for deployment."
    elif acc_diff < -0.02:
        verdict_text += "EfficientNet-B1 demonstrates clearly superior performance. " \
                        "Recommend using EfficientNet-B1 for deployment."
    else:
        verdict_text += "Both models are approximately equal in performance. " \
                        "Consider ensemble for best results, or choose based on " \
                        "inference speed and memory constraints."

    story.append(Paragraph(verdict_text, body))
    story.append(Spacer(1, 0.3*inch))

    # Final table
    story.append(Paragraph("<b>Metric-by-Metric Comparison:</b>", h2_style))
    final_data = [["Metric", "ResNet50", "EfficientNet-B1", "Diff", "Winner"]]
    for name, r_val, b_val in [
        ("Accuracy", resnet_metrics["accuracy"], b1_metrics["accuracy"]),
        ("Balanced Acc", resnet_metrics["balanced_accuracy"], b1_metrics["balanced_accuracy"]),
        ("Precision", resnet_metrics["precision"], b1_metrics["precision"]),
        ("Recall", resnet_metrics["recall"], b1_metrics["recall"]),
        ("F1 (weighted)", resnet_metrics["f1_score"], b1_metrics["f1_score"]),
        ("F1 (macro)", resnet_metrics["f1_macro"], b1_metrics["f1_macro"]),
        ("AUC (mean)", resnet_metrics["roc_auc_mean"], b1_metrics["roc_auc_mean"]),
    ]:
        diff = r_val - b_val
        winner = "ResNet50" if diff > 0 else "B1"
        final_data.append([
            name,
            f"{r_val:.4f}",
            f"{b_val:.4f}",
            f"{diff:+.4f}",
            winner,
        ])

    t = Table(final_data, colWidths=[1.4*inch, 1.1*inch, 1.4*inch, 1.0*inch, 1.0*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ]))
    story.append(t)

    story.append(PageBreak())

    # Final page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("✅ Report Complete", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"Author: Babak Yousefian<br/>"
        f"Version: 1.0.0 (Dual Model Edition)",
        center,
    ))

    try:
        doc.build(story)
        print(f"✅ PDF generated: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ PDF build failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ================================================================
# ✅ Main Entry
# ================================================================

def run_dual_model_report(device="cuda"):
    """
    Main entry: generate comprehensive dual-model report.
    """
    # ✅ FIX: Convert device to torch.device ONCE at the beginning
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("📊 DUAL MODEL COMPARISON REPORT")
    print("=" * 80)
    print(f"   Device: {device}")

    ensure_dirs()

    # ✅ Step 1: Check both models exist
    print("\n🔍 Step 1: Checking for models...")
    resnet_exists, b1_exists = check_models_exist()

    print(f"\n   Model 1 (ResNet50):")
    print(f"      Path: {RESNET50_MODEL_PATH}")
    print(f"      Status: {'✅ Found' if resnet_exists else '❌ NOT FOUND'}")

    print(f"\n   Model 2 (EfficientNet-B1):")
    print(f"      Path: {B1_MODEL_PATH}")
    print(f"      Status: {'✅ Found' if b1_exists else '❌ NOT FOUND'}")

    if not resnet_exists or not b1_exists:
        print("\n" + "=" * 80)
        print("❌ ERROR: Required models missing!")
        print("=" * 80)
        if not resnet_exists:
            print(f"   ❌ ResNet50 not found: {RESNET50_MODEL_PATH}")
            print(f"      👉 Please run option 7 first (ResNet50 Fine-Tune)")
        if not b1_exists:
            print(f"   ❌ EfficientNet-B1 not found: {B1_MODEL_PATH}")
            print(f"      👉 Please run option 13 first (B1 Fine-Tune)")
        print("=" * 80)
        return None

    print("\n✅ Both models found!")

    # ✅ Step 2: Load labeled data
    print("\n🔍 Step 2: Loading labeled data...")
    image_paths, labels = load_labeled_data()

    # ✅ Step 3: Use same test split as training (80/10/10 with seed=42)
    print("\n🔍 Step 3: Creating test split...")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.20, stratify=labels, random_state=42,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42,
    )

    print(f"   Test set: {len(test_paths)} samples")
    for i, cname in enumerate(CLASS_NAMES):
        count = sum(1 for l in test_labels if l == i)
        print(f"      {cname}: {count}")

    # ✅ Step 4: Load and evaluate ResNet50
    print("\n" + "=" * 80)
    print("🔬 Step 4: ResNet50 Evaluation")
    print("=" * 80)

    resnet_model, resnet_metadata = load_resnet50_model(device)

    print("\n   Running TTA evaluation...")
    resnet_metrics = evaluate_with_tta(
        resnet_model, test_paths, test_labels,
        device=device, input_size=224, n_aug=32, batch_size=8,
        num_classes=NUM_CLASSES, verbose=True,
    )

    print(f"\n   📊 ResNet50 Test Metrics:")
    print(f"      Accuracy:  {resnet_metrics['accuracy']:.4f}")
    print(f"      F1:        {resnet_metrics['f1_score']:.4f}")
    print(f"      AUC:       {resnet_metrics['roc_auc_mean']:.4f}")

    # ✅ Step 5: Load and evaluate B1
    print("\n" + "=" * 80)
    print("🔬 Step 5: EfficientNet-B1 Evaluation")
    print("=" * 80)

    b1_model, b1_metadata = load_b1_model(device)

    print("\n   Running TTA evaluation...")
    b1_metrics = evaluate_with_tta(
        b1_model, test_paths, test_labels,
        device=device, input_size=240, n_aug=32, batch_size=8,
        num_classes=NUM_CLASSES, verbose=True,
    )

    print(f"\n   📊 EfficientNet-B1 Test Metrics:")
    print(f"      Accuracy:  {b1_metrics['accuracy']:.4f}")
    print(f"      F1:        {b1_metrics['f1_score']:.4f}")
    print(f"      AUC:       {b1_metrics['roc_auc_mean']:.4f}")

    # ✅ Step 6: Generate plots
    print("\n" + "=" * 80)
    print("📈 Step 6: Generating Plots")
    print("=" * 80)

    resnet_plots = generate_model_plots(
        resnet_metrics, "ResNet50",
        os.path.join(PLOTS_DIR, "resnet50"),
    )
    print(f"   ✅ ResNet50 plots: {len(resnet_plots)}")

    b1_plots = generate_model_plots(
        b1_metrics, "EfficientNet-B1",
        os.path.join(PLOTS_DIR, "../Architects/EfficientNet/B1"),
    )
    print(f"   ✅ B1 plots: {len(b1_plots)}")

    # Comparison plots
    comparison_plots = {
        "comparison_bar": plot_comparison_bar(
            resnet_metrics, b1_metrics,
            os.path.join(PLOTS_DIR, "comparison_bar.png"),
        ),
        "comparison_radar": plot_comparison_radar(
            resnet_metrics, b1_metrics,
            os.path.join(PLOTS_DIR, "comparison_radar.png"),
        ),
    }
    print(f"   ✅ Comparison plots: {len(comparison_plots)}")

    # ✅ Step 7: Interpretability
    print("\n" + "=" * 80)
    print("🔬 Step 7: Interpretability")
    print("=" * 80)

    print("\n   ResNet50...")
    resnet_interp = generate_interpretability(
        resnet_model, test_paths, test_labels,
        os.path.join(INTERP_DIR, "resnet50"), device, "ResNet50",
    )

    print("\n   EfficientNet-B1...")
    b1_interp = generate_interpretability(
        b1_model, test_paths, test_labels,
        os.path.join(INTERP_DIR, "../Architects/EfficientNet/B1"), device, "EfficientNet-B1",
    )

    # ✅ Step 8: Save metrics JSON
    print("\n" + "=" * 80)
    print("💾 Step 8: Saving Metrics JSON")
    print("=" * 80)

    metrics_json = {
        "timestamp": datetime.now().isoformat(),
        "test_samples": len(test_paths),
        "resnet50": {
            "metadata": resnet_metadata,
            "metrics": {
                k: v for k, v in resnet_metrics.items()
                if k not in ["all_preds", "all_labels", "all_probs",
                             "fpr", "tpr", "confusion_matrix",
                             "classification_report"]
            },
            "confusion_matrix": resnet_metrics["confusion_matrix"].tolist(),
            "classification_report": resnet_metrics["classification_report"],
        },
        "b1": {
            "metadata": b1_metadata,
            "metrics": {
                k: v for k, v in b1_metrics.items()
                if k not in ["all_preds", "all_labels", "all_probs",
                             "fpr", "tpr", "confusion_matrix",
                             "classification_report"]
            },
            "confusion_matrix": b1_metrics["confusion_matrix"].tolist(),
            "classification_report": b1_metrics["classification_report"],
        },
    }

    metrics_path = os.path.join(OUTPUT_DIR, "dual_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=4, default=str)
    print(f"   ✅ Metrics saved: {metrics_path}")

    # ✅ Step 9: Generate PDF
    print("\n" + "=" * 80)
    print("📄 Step 9: Generating PDF Report")
    print("=" * 80)

    pdf_path = generate_dual_model_pdf(
        output_path=PDF_PATH,
        resnet_metrics=resnet_metrics,
        b1_metrics=b1_metrics,
        resnet_metadata=resnet_metadata,
        b1_metadata=b1_metadata,
        resnet_plots=resnet_plots,
        b1_plots=b1_plots,
        comparison_plots=comparison_plots,
        resnet_interp=resnet_interp,
        b1_interp=b1_interp,
        n_samples=len(test_paths),
    )

    # Cleanup
    del resnet_model, b1_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ✅ Summary
    print("\n" + "=" * 80)
    print("🎉 DUAL MODEL REPORT COMPLETE!")
    print("=" * 80)
    print(f"\n📊 ResNet50 (SimCLR):")
    print(f"   Accuracy:  {resnet_metrics['accuracy']:.4f}")
    print(f"   F1:        {resnet_metrics['f1_score']:.4f}")
    print(f"   AUC:       {resnet_metrics['roc_auc_mean']:.4f}")

    print(f"\n📊 EfficientNet-B1 (BYOL):")
    print(f"   Accuracy:  {b1_metrics['accuracy']:.4f}")
    print(f"   F1:        {b1_metrics['f1_score']:.4f}")
    print(f"   AUC:       {b1_metrics['roc_auc_mean']:.4f}")

    acc_diff = resnet_metrics['accuracy'] - b1_metrics['accuracy']
    if acc_diff > 0:
        print(f"\n🏆 Winner: ResNet50 (+{acc_diff:.4f} accuracy)")
    else:
        print(f"\n🏆 Winner: EfficientNet-B1 (+{abs(acc_diff):.4f} accuracy)")

    print(f"\n📂 Outputs:")
    print(f"   PDF:          {pdf_path}")
    print(f"   Plots:        {PLOTS_DIR}")
    print(f"   Interp:       {INTERP_DIR}")
    print(f"   Metrics JSON: {metrics_path}")

    return {
        "status": "success",
        "pdf_path": pdf_path,
        "metrics_json": metrics_path,
        "resnet50_metrics": resnet_metrics,
        "b1_metrics": b1_metrics,
    }
