# src/pipeLine/sslDualComparison.py
# -*- coding: utf-8 -*-

"""
SSL Dual Comparison Report Generator (v1.0)

Generates a comprehensive PDF report for 6 SSL encoders compared in 3 pairs:
- Pair 1 (BYOL):   ResNet18-BYOL vs EfficientNet-B0-BYOL
- Pair 2 (MAE):    ResNet18-MAE  vs EfficientNet-B0-MAE
- Pair 3 (SimCLR): ResNet18-SimCLR vs EfficientNet-B0-SimCLR

Each pair uses Linear Probe + TTA for evaluation on the held-out test set.

Features:
- Load 6 SSL encoders from predefined paths
- Linear Probe on embeddings (LogisticRegression)
- TTA on embeddings (32 augmentations averaged)
- Comprehensive plots + SHAP + Captum + LIME
- Cross-method comparison (which SSL method is best?)
- ~40-page PDF report

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

# ================================================================
# ✅ Paths
# ================================================================

PROJECT_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")

# ✅ Six model paths (exact from user)
MODEL_PATHS = {
    # Pair 1: BYOL
    "resnet18_byol": os.path.join(
        SELF_SUPERVISED_ROOT, "../Resnet", "18", "SSL_checkpoints",
        "architecture", "byol", "resnet18_best_update_byol.pth",
    ),
    "b0_byol": os.path.join(
        SELF_SUPERVISED_ROOT, "../EfficientNet", "EfficientNet/B0", "SSL_checkpoints",
        "architecture", "byol", "b0_ssl_byol_best.pth",
    ),
    # Pair 2: MAE
    "resnet18_mae": os.path.join(
        SELF_SUPERVISED_ROOT, "../Resnet", "18", "SSL_checkpoints",
        "architecture", "mae", "resnet18_best_update_mae.pth",
    ),
    "b0_mae": os.path.join(
        SELF_SUPERVISED_ROOT, "../EfficientNet", "EfficientNet/B0", "SSL_checkpoints",
        "architecture", "mae", "b0_ssl_mae_best.pth",
    ),
    # Pair 3: SimCLR
    "resnet18_simclr": os.path.join(
        SELF_SUPERVISED_ROOT, "../Resnet", "18", "SSL_checkpoints",
        "architecture", "simclr", "resnet18_best_update_simclr.pth",
    ),
    "b0_simclr": os.path.join(
        SELF_SUPERVISED_ROOT, "../EfficientNet", "EfficientNet/B0", "SSL_checkpoints",
        "architecture", "simclr", "b0_ssl_simclr_best.pth",
    ),
}

# ✅ Output
OUTPUT_DIR = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "ssl_dual_comparison")
PDF_PATH = os.path.join(OUTPUT_DIR, "SSL_Dual_Comparison_Report.pdf")
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
    for key in MODEL_PATHS.keys():
        os.makedirs(os.path.join(PLOTS_DIR, key), exist_ok=True)
        os.makedirs(os.path.join(INTERP_DIR, key), exist_ok=True)


# ================================================================
# ✅ Check All Architects Exist
# ================================================================

def check_all_models_exist() -> Dict[str, bool]:
    """Check existence of all 6 models."""
    return {k: os.path.exists(p) for k, p in MODEL_PATHS.items()}


# ================================================================
# ✅ TTA Dataset for SSL Encoders
# ================================================================

class SSLTTAImageDataset(Dataset):
    """
    Dataset returning raw images with TTA variants.
    The encoder will be called to extract embeddings.
    """
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


class SimpleImageDataset(Dataset):
    """Simple image dataset for non-TTA extraction."""

    def __init__(self, image_paths, labels, input_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.input_size = input_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_size, self.input_size))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_t = (img_t - mean) / std
        return img_t, self.labels[idx]


# ================================================================
# ✅ Load Labeled Data
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

    print(f"✅ Loaded {len(image_paths)} labeled images")
    for i, name in enumerate(CLASS_NAMES):
        count = sum(1 for l in labels if l == i)
        print(f"   {name}: {count}")

    return image_paths, labels


# ================================================================
# ✅ Compute Metrics
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
# ✅ Extract Embeddings (Regular + TTA)
# ================================================================

def extract_embeddings_regular(
    encoder: nn.Module,
    image_paths: List[str],
    labels: List[int],
    input_size: int,
    device: torch.device,
    batch_size: int = 32,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings WITHOUT TTA."""
    dataset = SimpleImageDataset(image_paths, labels, input_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    encoder.eval()
    encoder.to(device)

    all_emb = []
    all_lbl = []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="   Extracting", leave=False, disable=not verbose):
            imgs = imgs.to(device, non_blocking=True)
            if hasattr(encoder, "get_embedding"):
                emb = encoder.get_embedding(imgs)
            else:
                emb = encoder(imgs)
                if emb.dim() > 2:
                    emb = emb.view(emb.size(0), -1)
            all_emb.append(emb.float().cpu().numpy())
            all_lbl.append(lbls.numpy())

    X = np.concatenate(all_emb, axis=0)
    y = np.concatenate(all_lbl, axis=0)
    return X, y


def extract_embeddings_tta(
    encoder: nn.Module,
    image_paths: List[str],
    labels: List[int],
    input_size: int,
    device: torch.device,
    n_aug: int = 32,
    batch_size: int = 8,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings WITH TTA.
    Returns averaged embeddings per sample.
    """
    num_samples = len(image_paths)
    print(f"   🔬 TTA Embedding Extraction: {num_samples} images × {n_aug} augmentations")

    tta_dataset = SSLTTAImageDataset(image_paths, labels, input_size, n_aug)
    tta_loader = DataLoader(tta_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    per_aug_emb = [[] for _ in range(num_samples)]

    encoder.eval()
    encoder.to(device)
    use_amp = device.type == "cuda"

    pbar = tqdm(tta_loader, desc="   TTA Extract", leave=False, disable=not verbose)
    with torch.no_grad():
        for imgs, lbls, img_idx in pbar:
            imgs = imgs.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    if hasattr(encoder, "get_embedding"):
                        emb = encoder.get_embedding(imgs)
                    else:
                        emb = encoder(imgs)
                        if emb.dim() > 2:
                            emb = emb.view(emb.size(0), -1)
            else:
                if hasattr(encoder, "get_embedding"):
                    emb = encoder.get_embedding(imgs)
                else:
                    emb = encoder(imgs)
                    if emb.dim() > 2:
                        emb = emb.view(emb.size(0), -1)
            emb_np = emb.float().cpu().numpy()

            for i, idx in enumerate(img_idx.numpy()):
                per_aug_emb[idx].append(emb_np[i])

    # Average over augmentations
    emb_dim = per_aug_emb[0][0].shape[0] if per_aug_emb[0] else 0
    averaged_emb = np.zeros((num_samples, emb_dim))
    for i in range(num_samples):
        if len(per_aug_emb[i]) > 0:
            averaged_emb[i] = np.mean(per_aug_emb[i], axis=0)

    return averaged_emb, np.array(labels)


# ================================================================
# ✅ Linear Probe + TTA evaluation
# ================================================================

def linear_probe_with_tta(
    encoder: nn.Module,
    train_paths: List[str],
    train_labels: List[int],
    test_paths: List[str],
    test_labels: List[int],
    input_size: int,
    device: torch.device,
    n_aug_tta: int = 32,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Linear Probe on embeddings + TTA for test evaluation.

    Steps:
    1. Extract train embeddings (NO TTA)
    2. Train LogisticRegression
    3. Extract test embeddings WITH TTA
    4. Predict and compute metrics
    """
    # 1. Extract train embeddings
    print("   📊 Extracting TRAIN embeddings (no TTA)...")
    X_train, y_train = extract_embeddings_regular(
        encoder, train_paths, train_labels, input_size, device,
        batch_size=32, verbose=verbose,
    )

    # 2. Train linear probe
    print("   🎯 Training LogisticRegression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    # 3. Extract test embeddings WITH TTA
    print("   🔬 Extracting TEST embeddings WITH TTA...")
    X_test, y_test = extract_embeddings_tta(
        encoder, test_paths, test_labels, input_size, device,
        n_aug=n_aug_tta, batch_size=8, verbose=verbose,
    )

    # 4. Predict
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)

    # 5. Metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["all_preds"] = y_pred.tolist()
    metrics["all_labels"] = y_test.tolist()
    metrics["all_probs"] = y_prob
    metrics["embedding_dim"] = int(X_train.shape[1])
    metrics["train_samples"] = len(train_paths)
    metrics["test_samples"] = len(test_paths)

    return metrics


# ================================================================
# ✅ Load Encoders
# ================================================================

def load_resnet18_encoder(model_path: str, device):
    """Load ResNet18 SSL encoder."""
    from src.models.resnet.resnet18 import create_resnet18

    ckpt = torch.load(model_path, map_location="cpu")
    if "encoder_state_dict" in ckpt:
        state = ckpt["encoder_state_dict"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model = create_resnet18(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_b0_encoder(model_path: str, device):
    """Load EfficientNet-B0 SSL encoder."""
    from src.models.efficientNet.B0 import create_efficientnet_b0

    ckpt = torch.load(model_path, map_location="cpu")
    if "encoder_state_dict" in ckpt:
        state = ckpt["encoder_state_dict"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model = create_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


# ================================================================
# ✅ Plot Generators
# ================================================================

def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
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


def plot_pair_comparison_bar(metrics_resnet, metrics_b0, save_path, method_name):
    """Side-by-side bar chart for a pair."""
    metrics_names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1", "AUC"]
    resnet_vals = [
        metrics_resnet["accuracy"], metrics_resnet["balanced_accuracy"],
        metrics_resnet["precision"], metrics_resnet["recall"],
        metrics_resnet["f1_score"], metrics_resnet["roc_auc_mean"],
    ]
    b0_vals = [
        metrics_b0["accuracy"], metrics_b0["balanced_accuracy"],
        metrics_b0["precision"], metrics_b0["recall"],
        metrics_b0["f1_score"], metrics_b0["roc_auc_mean"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.38

    plt.figure(figsize=(13, 6))
    bars1 = plt.bar(x - width/2, resnet_vals, width, label=f"ResNet18 ({method_name.upper()})",
                    color="#3498db", edgecolor="black")
    bars2 = plt.bar(x + width/2, b0_vals, width, label=f"EfficientNet-B0 ({method_name.upper()})",
                    color="#e74c3c", edgecolor="black")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.xticks(x, metrics_names, fontsize=11, fontweight="bold")
    plt.ylim(0, 1.10)
    plt.ylabel("Score", fontweight="bold", fontsize=12)
    plt.title(f"Pair Comparison: ResNet18 vs EfficientNet-B0 ({method_name.upper()})",
              fontweight="bold", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_pair_comparison_radar(metrics_resnet, metrics_b0, save_path, method_name):
    """Radar chart for a pair."""
    metrics_names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1", "AUC"]
    resnet_vals = [
        metrics_resnet["accuracy"], metrics_resnet["balanced_accuracy"],
        metrics_resnet["precision"], metrics_resnet["recall"],
        metrics_resnet["f1_score"], metrics_resnet["roc_auc_mean"],
    ]
    b0_vals = [
        metrics_b0["accuracy"], metrics_b0["balanced_accuracy"],
        metrics_b0["precision"], metrics_b0["recall"],
        metrics_b0["f1_score"], metrics_b0["roc_auc_mean"],
    ]

    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    resnet_vals += resnet_vals[:1]
    b0_vals += b0_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.plot(angles, resnet_vals, "o-", linewidth=2,
            label=f"ResNet18 ({method_name.upper()})", color="#3498db")
    ax.fill(angles, resnet_vals, alpha=0.25, color="#3498db")
    ax.plot(angles, b0_vals, "o-", linewidth=2,
            label=f"EfficientNet-B0 ({method_name.upper()})", color="#e74c3c")
    ax.fill(angles, b0_vals, alpha=0.25, color="#e74c3c")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Radar Comparison: ResNet18 vs EfficientNet-B0 ({method_name.upper()})",
                 fontweight="bold", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.15), fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_cross_method_summary(all_metrics, save_path):
    """
    Cross-method summary: compare all 6 models across 3 methods.
    """
    model_names = [
        "ResNet18-BYOL", "B0-BYOL",
        "ResNet18-MAE", "B0-MAE",
        "ResNet18-SimCLR", "B0-SimCLR",
    ]
    accs = [
        all_metrics["resnet18_byol"]["accuracy"],
        all_metrics["b0_byol"]["accuracy"],
        all_metrics["resnet18_mae"]["accuracy"],
        all_metrics["b0_mae"]["accuracy"],
        all_metrics["resnet18_simclr"]["accuracy"],
        all_metrics["b0_simclr"]["accuracy"],
    ]
    f1s = [
        all_metrics["resnet18_byol"]["f1_score"],
        all_metrics["b0_byol"]["f1_score"],
        all_metrics["resnet18_mae"]["f1_score"],
        all_metrics["b0_mae"]["f1_score"],
        all_metrics["resnet18_simclr"]["f1_score"],
        all_metrics["b0_simclr"]["f1_score"],
    ]
    aucs = [
        all_metrics["resnet18_byol"]["roc_auc_mean"],
        all_metrics["b0_byol"]["roc_auc_mean"],
        all_metrics["resnet18_mae"]["roc_auc_mean"],
        all_metrics["b0_mae"]["roc_auc_mean"],
        all_metrics["resnet18_simclr"]["roc_auc_mean"],
        all_metrics["b0_simclr"]["roc_auc_mean"],
    ]

    x = np.arange(len(model_names))
    width = 0.27

    plt.figure(figsize=(14, 7))
    colors_bar = ["#27ae60", "#3498db", "#e74c3c"]

    bars1 = plt.bar(x - width, accs, width, label="Accuracy", color=colors_bar[0], edgecolor="black")
    bars2 = plt.bar(x, f1s, width, label="F1-Score", color=colors_bar[1], edgecolor="black")
    bars3 = plt.bar(x + width, aucs, width, label="AUC", color=colors_bar[2], edgecolor="black")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.xticks(x, model_names, fontsize=11, fontweight="bold", rotation=15)
    plt.ylim(0, 1.10)
    plt.ylabel("Score", fontweight="bold", fontsize=12)
    plt.title("Cross-Method Summary: All 6 SSL Architects", fontweight="bold", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_cross_method_heatmap(all_metrics, save_path):
    """Heatmap of all metrics across all 6 models."""
    model_names = [
        "ResNet18-BYOL", "B0-BYOL",
        "ResNet18-MAE", "B0-MAE",
        "ResNet18-SimCLR", "B0-SimCLR",
    ]
    keys = ["resnet18_byol", "b0_byol", "resnet18_mae", "b0_mae",
            "resnet18_simclr", "b0_simclr"]
    metric_names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1", "AUC"]

    data = []
    for key in keys:
        m = all_metrics[key]
        data.append([
            m["accuracy"], m["balanced_accuracy"],
            m["precision"], m["recall"],
            m["f1_score"], m["roc_auc_mean"],
        ])
    data = np.array(data)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="YlGnBu",
        xticklabels=metric_names, yticklabels=model_names,
        cbar_kws={"label": "Score"}, vmin=0, vmax=1,
    )
    plt.title("Cross-Method Metrics Heatmap", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ================================================================
# ✅ Interpretability (with LogisticRegression wrapper)
# ================================================================

class LinearProbeClassifier(nn.Module):
    """
    Wraps encoder + scaler + LogisticRegression as a single nn.Module
    to allow interpretability methods (SHAP, Captum) to work.

    ✅ v1.1: Compatible with Captum, SHAP, LIME reporters.
    """
    def __init__(self, encoder, scaler, clf, device):
        super().__init__()
        self.encoder = encoder
        self.scaler = scaler
        self.clf = clf
        self.device = device

        # ✅ Store as plain attributes (for sklearn-like compatibility)
        self._coef = torch.tensor(clf.coef_, dtype=torch.float32).to(device)
        self._intercept = torch.tensor(clf.intercept_, dtype=torch.float32).to(device)
        self._mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(device)
        self._scale = torch.tensor(scaler.scale_, dtype=torch.float32).to(device)

    # ✅ Properties for sklearn-compatibility
    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @property
    def classes_(self):
        return np.arange(NUM_CLASSES)

    @property
    def n_features_in_(self):
        return int(self._coef.shape[1])

    def forward(self, x):
        """
        Return logits.
        ✅ v1.1: NO torch.no_grad() — required for gradient-based interpretability.
        """
        # Forward through encoder (with gradient support)
        if hasattr(self.encoder, "get_embedding"):
            emb = self.encoder.get_embedding(x)
        else:
            emb = self.encoder(x)
            if emb.dim() > 2:
                emb = emb.view(emb.size(0), -1)

        # Standardize
        emb_scaled = (emb - self._mean) / self._scale

        # Logistic regression logits
        logits = emb_scaled @ self._coef.T + self._intercept
        return logits


def generate_interpretability(
    encoder, scaler, clf, image_paths, labels, output_dir, device, model_name="model",
):
    """Generate SHAP, Captum, LIME for a linear probe."""
    results = {}
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Wrap as a classifier
    classifier = LinearProbeClassifier(encoder, scaler, clf, device)
    classifier.eval()

    # ✅ FIX: Register encoder as a child module so hooks work
    # The wrapper class already has self.encoder, so Captum can find it
    # But we need to make sure `model.encoder.get_embedding` is accessible
    # ✅ Just make sure classifier's forward calls encoder with gradient support

    # ✅ Build small eval loader
    dataset = SimpleImageDataset(image_paths[:50], labels[:50], 224)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # ✅ FIX: Test the wrapper works before passing to reporters
    try:
        test_batch = next(iter(loader))[0].to(device)
        test_out = classifier(test_batch)
        print(f"   ✅ Wrapper test OK: output shape = {test_out.shape}")
    except Exception as e:
        print(f"   ⚠️ Wrapper test failed: {e}")

    # ✅ Captum
    try:
        from src.reporter.captum_reporter import generate_interpretability_report
        captum_res = generate_interpretability_report(
            classifier, loader, output_dir, device=str(device),
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
            classifier, loader, output_dir, device=str(device),
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
            classifier, loader, output_dir, class_names=CLASS_NAMES,
            device=str(device), num_samples=3,
        )
        if isinstance(lime_res, dict):
            results["lime"] = lime_res.get("lime_path")
        print(f"   ✅ LIME done")
    except Exception as e:
        print(f"   ⚠️ LIME failed: {e}")

    return results


# ================================================================
# ✅ PDF Generator
# ================================================================

def generate_pdf(
    output_path,
    all_metrics, all_meta, pair_plots, cross_plots, interp_paths, n_test,
):
    """Generate comprehensive PDF report."""
    try:
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
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
        title="SSL Dual Comparison Report",
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
    story.append(Paragraph("SSL Dual Comparison Report", title_style))
    story.append(Paragraph("6 SSL Encoders → 3 Pairwise Comparisons",
                            ParagraphStyle("sub", parent=body, alignment=TA_CENTER,
                                            fontSize=14, textColor=rl_colors.grey)))
    story.append(Spacer(1, 0.5*inch))

    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            center))
    story.append(Paragraph(f"<b>Test Samples:</b> {n_test}", center))
    story.append(Paragraph("<b>Author:</b> Babak Yousefian", center))

    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("📊 Methods Evaluated via Linear Probe + TTA (32 augs)", center))
    story.append(Paragraph("Pair 1: BYOL | Pair 2: MAE | Pair 3: SimCLR", center))
    story.append(PageBreak())

    # ============================================================
    # PAGE 2: Executive Summary (Cross-Method)
    # ============================================================
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Spacer(1, 0.2*inch))

    # Table of all 6 models
    summary_data = [["Model", "Method", "Accuracy", "Balanced Acc", "F1", "AUC"]]
    for key, label in [
        ("resnet18_byol", "ResNet18-BYOL"),
        ("b0_byol", "B0-BYOL"),
        ("resnet18_mae", "ResNet18-MAE"),
        ("b0_mae", "B0-MAE"),
        ("resnet18_simclr", "ResNet18-SimCLR"),
        ("b0_simclr", "B0-SimCLR"),
    ]:
        m = all_metrics[key]
        summary_data.append([
            label.split("-")[0],
            label.split("-")[1],
            f"{m['accuracy']:.4f}",
            f"{m['balanced_accuracy']:.4f}",
            f"{m['f1_score']:.4f}",
            f"{m['roc_auc_mean']:.4f}",
        ])

    t = Table(summary_data, colWidths=[1.2*inch, 1.0*inch, 1.0*inch, 1.2*inch, 1.0*inch, 1.0*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
    ]))
    story.append(t)

    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Winner Per Metric:</b>", h2_style))

    # Determine winners
    best_acc = max(all_metrics.items(), key=lambda x: x[1]["accuracy"])
    best_f1 = max(all_metrics.items(), key=lambda x: x[1]["f1_score"])
    best_auc = max(all_metrics.items(), key=lambda x: x[1]["roc_auc_mean"])

    winner_data = [
        ["Metric", "Best Model", "Value"],
        ["Accuracy", best_acc[0].replace("_", " ").upper(), f"{best_acc[1]['accuracy']:.4f}"],
        ["F1-Score", best_f1[0].replace("_", " ").upper(), f"{best_f1[1]['f1_score']:.4f}"],
        ["AUC", best_auc[0].replace("_", " ").upper(), f"{best_auc[1]['roc_auc_mean']:.4f}"],
    ]
    t = Table(winner_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#27AE60")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ]))
    story.append(t)

    story.append(PageBreak())

    # ============================================================
    # CROSS-METHOD PLOTS
    # ============================================================
    story.append(Paragraph("Cross-Method Summary (All 6 Architects)", h1_style))
    story.append(add_img(cross_plots.get("summary_bar"), w=6.5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(add_img(cross_plots.get("heatmap"), w=6*inch, h=4*inch))
    story.append(PageBreak())

    # ============================================================
    # PAIR 1: BYOL
    # ============================================================
    story.append(Paragraph("Pair 1: BYOL — ResNet18 vs EfficientNet-B0", h1_style))

    # ResNet18-BYOL
    story.append(Paragraph("<b>ResNet18-BYOL:</b>", h2_style))
    m = all_metrics["resnet18_byol"]
    story.append(Paragraph(
        f"Path: {all_meta['resnet18_byol']['path']}<br/>"
        f"Size: {all_meta['resnet18_byol']['size_mb']:.2f} MB<br/>"
        f"Embedding Dim: {m.get('embedding_dim', '?')}<br/>"
        f"<br/>"
        f"<b>Test Metrics:</b><br/>"
        f"Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | AUC: {m['roc_auc_mean']:.4f}<br/>"
        f"Balanced Acc: {m['balanced_accuracy']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}",
        body,
    ))

    story.append(Paragraph("<b>EfficientNet-B0-BYOL:</b>", h2_style))
    m = all_metrics["b0_byol"]
    story.append(Paragraph(
        f"Path: {all_meta['b0_byol']['path']}<br/>"
        f"Size: {all_meta['b0_byol']['size_mb']:.2f} MB<br/>"
        f"Embedding Dim: {m.get('embedding_dim', '?')}<br/>"
        f"<br/>"
        f"<b>Test Metrics:</b><br/>"
        f"Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | AUC: {m['roc_auc_mean']:.4f}<br/>"
        f"Balanced Acc: {m['balanced_accuracy']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}",
        body,
    ))
    story.append(PageBreak())

    # BYOL plots
    story.append(Paragraph("BYOL — Plots", h1_style))
    story.append(Paragraph("ResNet18-BYOL Confusion Matrix:", h2_style))
    story.append(add_img(pair_plots["byol_resnet_cm"], w=5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(Paragraph("B0-BYOL Confusion Matrix:", h2_style))
    story.append(add_img(pair_plots["byol_b0_cm"], w=5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(Paragraph("BYOL Side-by-Side Comparison:", h2_style))
    story.append(add_img(pair_plots["byol_bar"], w=6.5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(add_img(pair_plots["byol_radar"], w=6*inch, h=6*inch))
    story.append(PageBreak())
    story.append(Paragraph("BYOL ROC Curves:", h2_style))
    story.append(add_img(pair_plots["byol_resnet_roc"], w=5*inch, h=4.5*inch))
    story.append(add_img(pair_plots["byol_b0_roc"], w=5*inch, h=4.5*inch))
    story.append(PageBreak())

    # ============================================================
    # PAIR 2: MAE
    # ============================================================
    story.append(Paragraph("Pair 2: MAE — ResNet18 vs EfficientNet-B0", h1_style))

    story.append(Paragraph("<b>ResNet18-MAE:</b>", h2_style))
    m = all_metrics["resnet18_mae"]
    story.append(Paragraph(
        f"Path: {all_meta['resnet18_mae']['path']}<br/>"
        f"<b>Test Metrics:</b><br/>"
        f"Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | AUC: {m['roc_auc_mean']:.4f}",
        body,
    ))

    story.append(Paragraph("<b>EfficientNet-B0-MAE:</b>", h2_style))
    m = all_metrics["b0_mae"]
    story.append(Paragraph(
        f"Path: {all_meta['b0_mae']['path']}<br/>"
        f"<b>Test Metrics:</b><br/>"
        f"Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | AUC: {m['roc_auc_mean']:.4f}",
        body,
    ))
    story.append(PageBreak())

    # MAE plots
    story.append(Paragraph("MAE — Plots", h1_style))
    story.append(Paragraph("ResNet18-MAE Confusion Matrix:", h2_style))
    story.append(add_img(pair_plots["mae_resnet_cm"], w=5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(Paragraph("B0-MAE Confusion Matrix:", h2_style))
    story.append(add_img(pair_plots["mae_b0_cm"], w=5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(Paragraph("MAE Side-by-Side Comparison:", h2_style))
    story.append(add_img(pair_plots["mae_bar"], w=6.5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(add_img(pair_plots["mae_radar"], w=6*inch, h=6*inch))
    story.append(PageBreak())
    story.append(Paragraph("MAE ROC Curves:", h2_style))
    story.append(add_img(pair_plots["mae_resnet_roc"], w=5*inch, h=4.5*inch))
    story.append(add_img(pair_plots["mae_b0_roc"], w=5*inch, h=4.5*inch))
    story.append(PageBreak())

    # ============================================================
    # PAIR 3: SimCLR
    # ============================================================
    story.append(Paragraph("Pair 3: SimCLR — ResNet18 vs EfficientNet-B0", h1_style))

    story.append(Paragraph("<b>ResNet18-SimCLR:</b>", h2_style))
    m = all_metrics["resnet18_simclr"]
    story.append(Paragraph(
        f"Path: {all_meta['resnet18_simclr']['path']}<br/>"
        f"<b>Test Metrics:</b><br/>"
        f"Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | AUC: {m['roc_auc_mean']:.4f}",
        body,
    ))

    story.append(Paragraph("<b>EfficientNet-B0-SimCLR:</b>", h2_style))
    m = all_metrics["b0_simclr"]
    story.append(Paragraph(
        f"Path: {all_meta['b0_simclr']['path']}<br/>"
        f"<b>Test Metrics:</b><br/>"
        f"Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | AUC: {m['roc_auc_mean']:.4f}",
        body,
    ))
    story.append(PageBreak())

    # SimCLR plots
    story.append(Paragraph("SimCLR — Plots", h1_style))
    story.append(Paragraph("ResNet18-SimCLR Confusion Matrix:", h2_style))
    story.append(add_img(pair_plots["simclr_resnet_cm"], w=5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(Paragraph("B0-SimCLR Confusion Matrix:", h2_style))
    story.append(add_img(pair_plots["simclr_b0_cm"], w=5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(Paragraph("SimCLR Side-by-Side Comparison:", h2_style))
    story.append(add_img(pair_plots["simclr_bar"], w=6.5*inch, h=4*inch))
    story.append(PageBreak())
    story.append(add_img(pair_plots["simclr_radar"], w=6*inch, h=6*inch))
    story.append(PageBreak())
    story.append(Paragraph("SimCLR ROC Curves:", h2_style))
    story.append(add_img(pair_plots["simclr_resnet_roc"], w=5*inch, h=4.5*inch))
    story.append(add_img(pair_plots["simclr_b0_roc"], w=5*inch, h=4.5*inch))
    story.append(PageBreak())

    # ============================================================
    # INTERPRETABILITY SECTION
    # ============================================================
    story.append(Paragraph("Interpretability — Best Model Per Method", h1_style))

    # Determine best model per method
    for method_key, method_name in [
        ("byol", "BYOL"),
        ("mae", "MAE"),
        ("simclr", "SimCLR"),
    ]:
        key_r = f"resnet18_{method_key}"
        key_b = f"b0_{method_key}"
        winner = key_r if all_metrics[key_r]["accuracy"] > all_metrics[key_b]["accuracy"] else key_b
        interp = interp_paths.get(winner, {})

        story.append(Paragraph(f"{method_name} — Best: {winner.replace('_', ' ').upper()}", h2_style))

        for label, key in [("Grad-CAM", "gradcam"), ("Integrated Gradients", "ig"),
                            ("SHAP", "shap"), ("LIME", "lime")]:
            if interp.get(key):
                story.append(Paragraph(f"{label}:", h2_style))
                story.append(add_img(interp[key], w=6*inch, h=4.5*inch))
                story.append(PageBreak())

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    story.append(Paragraph("Final Verdict", h1_style))

    # Best per method (within pair)
    verdict_text = "<b>Best Model Per SSL Method (within pair):</b><br/><br/>"
    for method_key, method_name in [
        ("byol", "BYOL"), ("mae", "MAE"), ("simclr", "SimCLR"),
    ]:
        key_r = f"resnet18_{method_key}"
        key_b = f"b0_{method_key}"
        r_acc = all_metrics[key_r]["accuracy"]
        b_acc = all_metrics[key_b]["accuracy"]
        winner = "ResNet18" if r_acc > b_acc else "EfficientNet-B0"
        diff = abs(r_acc - b_acc)
        verdict_text += f"<b>{method_name}:</b> {winner} wins (+{diff:.4f} accuracy)<br/>"

    verdict_text += "<br/><b>Overall Best Model (across all 6):</b><br/>"
    best_overall = max(all_metrics.items(), key=lambda x: x[1]["accuracy"])
    verdict_text += f"<b>{best_overall[0].replace('_', ' ').upper()}</b> " \
                     f"(Accuracy = {best_overall[1]['accuracy']:.4f})<br/>"

    story.append(Paragraph(verdict_text, body))
    story.append(PageBreak())

    # Final summary table
    story.append(Paragraph("Final Comparison Table", h1_style))
    final_data = [["Model", "Accuracy", "F1", "AUC", "Rank"]]

    # Rank by accuracy
    sorted_models = sorted(all_metrics.items(),
                            key=lambda x: x[1]["accuracy"], reverse=True)
    for rank, (key, m) in enumerate(sorted_models, 1):
        final_data.append([
            key.replace("_", " ").upper(),
            f"{m['accuracy']:.4f}",
            f"{m['f1_score']:.4f}",
            f"{m['roc_auc_mean']:.4f}",
            f"#{rank}",
        ])

    t = Table(final_data, colWidths=[2.2*inch, 1.1*inch, 1.1*inch, 1.1*inch, 0.8*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
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
        f"Version: 1.0.0",
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

def run_ssl_dual_comparison(device="cuda"):
    """Main entry: generate SSL dual comparison report."""
    # ✅ Convert device
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("📊 SSL DUAL COMPARISON REPORT")
    print("=" * 80)
    print(f"   Device: {device}")

    ensure_dirs()

    # ✅ Check all models
    print("\n🔍 Step 1: Checking all 6 models...")
    status = check_all_models_exist()

    all_exist = True
    for key, exists in status.items():
        icon = "✅" if exists else "❌"
        print(f"   {icon} {key:<20} : {MODEL_PATHS[key]}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n" + "=" * 80)
        print("❌ ERROR: Some models are missing!")
        print("=" * 80)
        for key, exists in status.items():
            if not exists:
                print(f"   ❌ {key}: {MODEL_PATHS[key]}")
        print("\n👉 Please run options 3-5 (ResNet18 SSL) and 9-11 (B0 SSL) first.")
        print("=" * 80)
        return None

    print("\n✅ All 6 models found!")

    # ✅ Load labeled data
    print("\n🔍 Step 2: Loading labeled data...")
    image_paths, labels = load_labeled_data()

    # ✅ Create split
    print("\n🔍 Step 3: Creating test split...")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.20, stratify=labels, random_state=42,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42,
    )

    print(f"   Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    for i, cname in enumerate(CLASS_NAMES):
        count = sum(1 for l in test_labels if l == i)
        print(f"      Test {cname}: {count}")

    all_metrics = {}
    all_meta = {}

    # ✅ Process each model
    model_configs = [
        ("resnet18_byol", "resnet18", 224, "BYOL"),
        ("b0_byol", "b0", 224, "BYOL"),
        ("resnet18_mae", "resnet18", 224, "MAE"),
        ("b0_mae", "b0", 224, "MAE"),
        ("resnet18_simclr", "resnet18", 224, "SimCLR"),
        ("b0_simclr", "b0", 224, "SimCLR"),
    ]

    for key, model_type, input_size, method in model_configs:
        print(f"\n{'='*80}")
        print(f"🔬 Processing: {key.upper()} ({method})")
        print(f"{'='*80}")

        try:
            if model_type == "resnet18":
                encoder = load_resnet18_encoder(MODEL_PATHS[key], device)
            else:
                encoder = load_b0_encoder(MODEL_PATHS[key], device)

            print(f"   ✅ Encoder loaded")

            # ✅ Linear Probe + TTA
            metrics = linear_probe_with_tta(
                encoder, train_paths, train_labels, test_paths, test_labels,
                input_size, device, n_aug_tta=32, verbose=True,
            )

            print(f"\n   📊 Test Metrics:")
            print(f"      Accuracy:  {metrics['accuracy']:.4f}")
            print(f"      F1:        {metrics['f1_score']:.4f}")
            print(f"      AUC:       {metrics['roc_auc_mean']:.4f}")

            all_metrics[key] = metrics
            all_meta[key] = {
                "path": MODEL_PATHS[key],
                "method": method,
                "model_type": model_type,
                "size_mb": os.path.getsize(MODEL_PATHS[key]) / 1e6,
                "device": str(device),
            }

            # Free memory
            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ✅ Generate plots
    print("\n" + "=" * 80)
    print("📈 Step 4: Generating Plots")
    print("=" * 80)

    pair_plots = {}
    # Per-model plots
    for key in all_metrics.keys():
        m = all_metrics[key]
        subdir = os.path.join(PLOTS_DIR, key)
        plot_confusion_matrix(m["confusion_matrix"],
                                os.path.join(subdir, "confusion_matrix.png"),
                                title=f"{key.replace('_', ' ').upper()} — Confusion Matrix")
        plot_roc_curves(m, os.path.join(subdir, "roc_curves.png"),
                         title=f"{key.replace('_', ' ').upper()} — ROC Curves")
        plot_metrics_bar(m, os.path.join(subdir, "metrics_bar.png"),
                          title=f"{key.replace('_', ' ').upper()} — Metrics")
        plot_per_class_metrics(m, os.path.join(subdir, "per_class.png"),
                                title=f"{key.replace('_', ' ').upper()} — Per-Class")

    # Pair plots
    for method_key, method_name in [("byol", "BYOL"), ("mae", "MAE"), ("simclr", "SimCLR")]:
        key_r = f"resnet18_{method_key}"
        key_b = f"b0_{method_key}"

        pair_plots[f"{method_key}_resnet_cm"] = os.path.join(
            PLOTS_DIR, key_r, "confusion_matrix.png")
        pair_plots[f"{method_key}_b0_cm"] = os.path.join(
            PLOTS_DIR, key_b, "confusion_matrix.png")
        pair_plots[f"{method_key}_resnet_roc"] = os.path.join(
            PLOTS_DIR, key_r, "roc_curves.png")
        pair_plots[f"{method_key}_b0_roc"] = os.path.join(
            PLOTS_DIR, key_b, "roc_curves.png")

        # Comparison
        pair_plots[f"{method_key}_bar"] = plot_pair_comparison_bar(
            all_metrics[key_r], all_metrics[key_b],
            os.path.join(PLOTS_DIR, f"{method_key}_comparison_bar.png"),
            method_name,
        )
        pair_plots[f"{method_key}_radar"] = plot_pair_comparison_radar(
            all_metrics[key_r], all_metrics[key_b],
            os.path.join(PLOTS_DIR, f"{method_key}_comparison_radar.png"),
            method_name,
        )

    print(f"   ✅ Pair plots generated")

    # Cross-method plots
    cross_plots = {
        "summary_bar": plot_cross_method_summary(
            all_metrics, os.path.join(PLOTS_DIR, "cross_method_summary.png")),
        "heatmap": plot_cross_method_heatmap(
            all_metrics, os.path.join(PLOTS_DIR, "cross_method_heatmap.png")),
    }
    print(f"   ✅ Cross-method plots generated")

    # ✅ Interpretability — for best model per method
    print("\n" + "=" * 80)
    print("🔬 Step 5: Interpretability (Best Model Per Method)")
    print("=" * 80)

    interp_paths = {}
    for method_key, method_name in [("byol", "BYOL"), ("mae", "MAE"), ("simclr", "SimCLR")]:
        key_r = f"resnet18_{method_key}"
        key_b = f"b0_{method_key}"
        winner_key = key_r if all_metrics[key_r]["accuracy"] > all_metrics[key_b]["accuracy"] else key_b

        print(f"\n   {method_name}: Best = {winner_key.upper()}")

        try:
            # Load winner
            if winner_key.startswith("resnet18"):
                encoder = load_resnet18_encoder(MODEL_PATHS[winner_key], device)
                input_size = 224
            else:
                encoder = load_b0_encoder(MODEL_PATHS[winner_key], device)
                input_size = 224

            # Rebuild linear probe for interpretability
            X_train, y_train = extract_embeddings_regular(
                encoder, train_paths, train_labels, input_size, device,
                batch_size=32, verbose=False,
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=42, n_jobs=-1)
            clf.fit(X_train_scaled, y_train)

            interp = generate_interpretability(
                encoder, scaler, clf, test_paths, test_labels,
                os.path.join(INTERP_DIR, winner_key), device, winner_key,
            )
            interp_paths[winner_key] = interp

            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            interp_paths[winner_key] = {}

    # ✅ Save metrics JSON
    print("\n" + "=" * 80)
    print("💾 Step 6: Saving Metrics JSON")
    print("=" * 80)

    metrics_json = {
        "timestamp": datetime.now().isoformat(),
        "num_test_samples": len(test_paths),
        "models": {},
    }
    for key, m in all_metrics.items():
        metrics_json["models"][key] = {
            "metadata": all_meta[key],
            "metrics": {
                k: v for k, v in m.items()
                if k not in ["all_preds", "all_labels", "all_probs",
                             "fpr", "tpr", "confusion_matrix",
                             "classification_report"]
            },
            "confusion_matrix": m["confusion_matrix"].tolist(),
            "classification_report": m["classification_report"],
        }

    metrics_path = os.path.join(OUTPUT_DIR, "ssl_dual_comparison_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=4, default=str)
    print(f"   ✅ Metrics saved: {metrics_path}")

    # ✅ Generate PDF
    print("\n" + "=" * 80)
    print("📄 Step 7: Generating PDF Report")
    print("=" * 80)

    pdf_path = generate_pdf(
        output_path=PDF_PATH,
        all_metrics=all_metrics,
        all_meta=all_meta,
        pair_plots=pair_plots,
        cross_plots=cross_plots,
        interp_paths=interp_paths,
        n_test=len(test_paths),
    )

    # ✅ Summary
    print("\n" + "=" * 80)
    print("🎉 SSL DUAL COMPARISON COMPLETE!")
    print("=" * 80)

    for key, m in sorted(all_metrics.items(),
                          key=lambda x: x[1]["accuracy"], reverse=True):
        print(f"   {key.replace('_', ' ').upper():<22} "
              f"Acc={m['accuracy']:.4f}  F1={m['f1_score']:.4f}  AUC={m['roc_auc_mean']:.4f}")

    print(f"\n📂 Outputs:")
    print(f"   PDF:          {pdf_path}")
    print(f"   Plots:        {PLOTS_DIR}")
    print(f"   Interp:       {INTERP_DIR}")
    print(f"   Metrics JSON: {metrics_path}")

    return {
        "status": "success",
        "pdf_path": pdf_path,
        "metrics_json": metrics_path,
        "all_metrics": all_metrics,
    }
