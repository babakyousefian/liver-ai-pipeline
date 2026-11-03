# src/pipeLine/resnet50FineTune.py

"""
ResNet50 Supervised Fine-Tuner + Full Report Generator (ULTIMATE EDITION)

Features:
- Load best SSL checkpoint (from BYOL/SimCLR/MAE) — no re-training
- Transfer weights to ResNet50 base
- Full Supervised Fine-Tuning (50 epochs, differential LR)
- Class-weighted Focal Loss for imbalance
- Early Stopping
- 5-Fold Cross-Validation
- 95% Confidence Interval (t-distribution)
- 14-page comprehensive PDF report
- LIME, SHAP, Captum, Netron, ONNX, Grad-CAM

Author: Babak Yousefian
Version: 6.0.0
"""

import copy
import gc
import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

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
from PIL import Image
from sklearn.metrics import (
    accuracy_score, auc, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# ✅ Constants
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Paths
PROJECT_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")
RESNET_WEIGHT_TRANSFORMER_DIR = os.path.join(SELF_SUPERVISED_ROOT, "..", "weightTransformer")
FINAL_MODELS_DIR = os.path.join(SELF_SUPERVISED_ROOT, "final_models")
RESNET50_REPORTS_DIR = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "resnet50_finetune")
CV_REPORTS_DIR = os.path.join(RESNET50_REPORTS_DIR, "../../../CV")
PLOTS_DIR = os.path.join(RESNET50_REPORTS_DIR, "plots")
INTERP_DIR = os.path.join(RESNET50_REPORTS_DIR, "interpretability")
ARCH_DIR = os.path.join(RESNET50_REPORTS_DIR, "architecture")


def ensure_dirs():
    for d in [RESNET_WEIGHT_TRANSFORMER_DIR, FINAL_MODELS_DIR, RESNET50_REPORTS_DIR,
              CV_REPORTS_DIR, PLOTS_DIR, INTERP_DIR, ARCH_DIR]:
        os.makedirs(d, exist_ok=True)


# ================================================================
# ✅ Focal Loss for Imbalanced Classification
# ================================================================

class FocalLoss(nn.Module):
    """Focal Loss — برای imbalanced data (G3 فقط 70 نمونه)."""
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal



# ================================================================
# ✅ TTA Dataset (v2.0 — کپی از B1 v8.1)
# ================================================================

class TTADataset(Dataset):
    """
    Dataset for Test-Time Augmentation.
    Returns N augmented versions of each image.

    ✅ v2.0: 32 deterministic augmentations (هماهنگ با B1)
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
        """
        ✅ v2.0: 32 deterministic TTA variants.
        Comprehensive coverage of typical image variations.
        """
        # idx 0-7: base augmentations
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

        # idx 24-31: combined transformations
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

        # Apply ImageNet normalization (same as eval transform)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_t = (img_t - mean) / std

        return img_t, self.labels[img_idx], img_idx


# ================================================================
# ✅ Labeled Dataset for Fine-Tuning
# ================================================================

class LabeledLiverDataset(Dataset):
    """Dataset برچسب‌دار برای Fine-Tuning supervised."""

    def __init__(self, image_paths, labels, transform=None, target_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size))

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, label


def get_train_transform(input_size=224):
    """Train augmentation — medical-safe."""
    import torchvision.transforms as T
    return T.Compose([
        T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomApply([T.ColorJitter(0.15, 0.15, 0.1, 0.05)], p=0.6),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(input_size=224):
    """Eval transform — deterministic."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_labeled_dataset(labels_csv="finalReports/labels.csv",
                          image_dir="finalReports/cropped"):
    """Load labels + image paths."""
    if not os.path.exists(labels_csv):
        labels_csv = "finalReports/labels.csv"
    if not os.path.exists(image_dir):
        image_dir = "finalReports/cropped"

    df = pd.read_csv(labels_csv)
    paths, labels = [], []

    for _, row in df.iterrows():
        name = row["liverName"]
        lbl = row["liverLabel"]
        if lbl not in CLASS_NAMES:
            continue
        p = os.path.join(image_dir, name)
        if os.path.exists(p):
            paths.append(p)
            labels.append(CLASS_NAMES.index(lbl))

    print(f"📂 Loaded {len(paths)} labeled samples")
    for i, c in enumerate(CLASS_NAMES):
        print(f"   {c}: {labels.count(i)}")

    return paths, np.array(labels, dtype=np.int64)


# ================================================================
# ✅ ResNet50 Fine-Tuner
# ================================================================

class ResNet50FineTuner:
    """
    Supervised Fine-Tuner برای ResNet50 با وزن‌های SSL منتقل‌شده.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr_backbone: float = 1e-4,
        lr_head: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        warmup_epochs: int = 5,
        use_focal_loss: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        use_mixed_precision: bool = True,
        clip_grad_norm: float = 1.0,
        early_stop_patience: int = 15,
        save_dir: str = None,
        verbose: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.use_mixed_precision = use_mixed_precision
        self.clip_grad_norm = clip_grad_norm
        self.early_stop_patience = early_stop_patience
        self.save_dir = save_dir or FINAL_MODELS_DIR
        self.verbose = verbose

        os.makedirs(self.save_dir, exist_ok=True)

        # ✅ Differential LR
        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if "fc" in name or "head" in name or "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ], weight_decay=weight_decay, betas=(0.9, 0.999))

        # ✅ Scheduler: Warmup + Cosine
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
        if warmup_epochs > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_epochs])
        else:
            self.scheduler = cosine

        # ✅ Loss
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights.to(self.device) if class_weights is not None else None, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)

        # ✅ Mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=(use_mixed_precision and self.device.type == "cuda"))

        # ✅ State
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr_backbone": [], "lr_head": [],
        }
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_state = None

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.train_loader,
                    desc=f"🧠 FT Epoch {epoch+1}/{self.epochs}",
                    ncols=110, leave=False, disable=not self.verbose)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_mixed_precision and self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                if self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    def fit(self) -> Dict[str, Any]:
        print(f"\n🚀 Starting Fine-Tune for {self.epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches:   {len(self.val_loader)}")

        start = time.time()
        patience_counter = 0
        best_path = os.path.join(self.save_dir, "resnet50_finetuned_best.pth")

        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate()
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr_backbone"].append(self.optimizer.param_groups[0]["lr"])
            self.history["lr_head"].append(self.optimizer.param_groups[1]["lr"])

            if self.verbose:
                print(f"📊 Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train L:{train_loss:.4f} A:{train_acc:.4f} | "
                      f"Val L:{val_loss:.4f} A:{val_acc:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # ✅ Best tracking (بر اساس val_acc)
            improved = val_acc > self.best_val_acc + 1e-4

            if improved:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())

                # ✅ FIX: ذخیره با final_metrics و همه چیز
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "history": self.history,
                    # ✅ NEW: full metrics
                    "final_metrics": {
                        "accuracy": float(val_acc),
                        "balanced_accuracy": float(val_acc),  # placeholder، بعداً overwrite میشه
                        "precision": float(val_acc),
                        "recall": float(val_acc),
                        "f1_score": float(val_acc),
                        "f1_macro": float(val_acc),
                    },
                }, best_path)
                patience_counter = 0
            else:
                patience_counter += 1

            # ✅ Early stopping
            if patience_counter >= self.early_stop_patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

        # ✅ Restore best
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        total_time = time.time() - start
        print(f"\n✅ Fine-Tune done in {total_time/60:.2f} min")
        print(f"   Best Val Acc:  {self.best_val_acc:.4f}")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print(f"   Best Epoch:    {self.best_epoch + 1}")

        return {
            "history": self.history,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
            "model_path": best_path,
        }


# ================================================================
# ✅ Evaluation (Full metrics on a loader)
# ================================================================

@torch.no_grad()
def evaluate_model(model, loader, device, num_classes=4):
    """Evaluate model — returns dict with all metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_class_recall = recall_score(y_true, y_pred, average=None,
                                     zero_division=0, labels=list(range(num_classes)))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # ROC-AUC
    fpr, tpr, roc_auc = {}, {}, {}
    try:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        for i in range(num_classes):
            if y_true_bin[:, i].sum() > 0:
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
    except Exception:
        pass

    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=CLASS_NAMES[:num_classes],
        output_dict=True, zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_w,
        "f1_macro": f1_macro,
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": cm,
        "roc_data": (fpr, tpr, roc_auc),
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "num_samples": len(y_true),
    }

# ================================================================
# ✅ Evaluate with Test-Time Augmentation (v2.0)
# ================================================================

@torch.no_grad()
def evaluate_with_tta(
    model,
    image_paths,
    labels,
    device,
    n_aug=32,
    batch_size=16,
    num_classes=4,
    verbose=True,
):
    """
    ✅ v2.0: Test-Time Augmentation evaluation.
    Average probabilities over N augmented versions of each image.

    Returns:
        dict with all metrics (accuracy, precision, recall, f1, auc, roc, etc.)
    """
    model.eval()
    model.to(device)

    num_samples = len(image_paths)
    print(f"   🔬 TTA Evaluation: {num_samples} images × {n_aug} augmentations")

    # ✅ Build TTA dataset
    tta_dataset = TTADataset(image_paths, labels, input_size=224, n_aug=n_aug)
    tta_loader = DataLoader(
        tta_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ✅ Accumulate probs per (sample, aug)
    per_aug_probs = [[] for _ in range(num_samples)]

    pbar = tqdm(tta_loader, desc="   TTA Inference", leave=False, disable=not verbose)

    for imgs, lbls, img_idx in pbar:
        imgs = imgs.to(device, non_blocking=True)

        # Inference
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)

        probs = F.softmax(logits.float(), dim=1).cpu().numpy()

        for i, idx in enumerate(img_idx.numpy()):
            per_aug_probs[idx].append(probs[i])

    # ✅ Average over augmentations
    averaged_probs = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        if len(per_aug_probs[i]) > 0:
            averaged_probs[i] = np.mean(per_aug_probs[i], axis=0)

    preds = averaged_probs.argmax(axis=1)
    y_true = np.array(labels)
    y_prob = averaged_probs

    # ✅ Compute all metrics
    accuracy = accuracy_score(y_true, preds)
    balanced_acc = balanced_accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, average="weighted", zero_division=0)
    recall = recall_score(y_true, preds, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)

    per_class_recall = recall_score(y_true, preds, average=None,
                                     zero_division=0, labels=list(range(num_classes)))
    per_class_precision = precision_score(y_true, preds, average=None,
                                            zero_division=0, labels=list(range(num_classes)))
    per_class_f1 = f1_score(y_true, preds, average=None,
                             zero_division=0, labels=list(range(num_classes)))

    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))

    # ✅ ROC-AUC per class
    fpr, tpr, roc_auc = {}, {}, {}
    try:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        for i in range(num_classes):
            if y_true_bin[:, i].sum() > 0 and (1 - y_true_bin[:, i]).sum() > 0:
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
    except Exception as e:
        print(f"   ⚠️ ROC compute failed: {e}")

    # ✅ Mean AUC
    roc_auc_mean = float(np.mean(list(roc_auc.values()))) if roc_auc else 0.0

    # ✅ Classification report
    report = classification_report(
        y_true, preds, labels=list(range(num_classes)),
        target_names=CLASS_NAMES[:num_classes],
        output_dict=True, zero_division=0,
    )

    # ✅ Ensure all classes present in report
    for cname in CLASS_NAMES[:num_classes]:
        if cname not in report:
            report[cname] = {
                "precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0
            }

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_w),
        "f1_macro": float(f1_macro),
        "roc_auc": roc_auc,
        "roc_auc_mean": roc_auc_mean,
        "per_class_recall": per_class_recall.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm,
        "roc_data": (fpr, tpr, roc_auc),
        "classification_report": report,
        "y_true": y_true,
        "y_pred": preds,
        "y_prob": y_prob,
        "num_samples": len(y_true),
        "tta_n_aug": n_aug,
    }

# ================================================================
# ✅ 5-Fold Cross-Validation + CI 95%
# ================================================================

def run_cv5_folds(
    model_factory,  # function that returns a fresh model (with SSL weights)
    paths: List[str],
    labels: np.ndarray,
    device: str,
    n_folds: int = 5,
    ft_epochs_per_fold: int = 10,
    batch_size: int = 32,
    input_size: int = 224,
    save_dir: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    5-Fold CV با StratifiedKFold.
    در هر fold، مدل از SSL weights شروع می‌کنه و یه Fine-tune کوتاه می‌زنه.
    """
    save_dir = save_dir or CV_REPORTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    if verbose:
        print(f"\n{'='*80}")
        print(f"🔬 Running {n_folds}-Fold Cross-Validation")
        print(f"{'='*80}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        if verbose:
            print(f"\n📌 Fold {fold_idx+1}/{n_folds}")
            print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}")

        # ✅ Datasets
        train_paths = [paths[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_paths = [paths[i] for i in val_idx]
        val_labels = labels[val_idx]

        train_ds = LabeledLiverDataset(train_paths, train_labels,
                                        transform=get_train_transform(input_size),
                                        target_size=input_size)
        val_ds = LabeledLiverDataset(val_paths, val_labels,
                                      transform=get_eval_transform(input_size),
                                      target_size=input_size)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=2, pin_memory=True,
                                   drop_last=(len(train_ds) > batch_size))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)

        # ✅ Fresh model per fold
        model = model_factory()

        # ✅ Class weights (per fold, based on train distribution)
        counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        raw_w = 1.0 / (counts + 1e-8)
        sqrt_w = np.sqrt(raw_w)
        norm_w = sqrt_w / sqrt_w.sum() * NUM_CLASSES
        cw = torch.tensor(norm_w, dtype=torch.float32)

        # ✅ Fine-tune
        trainer = ResNet50FineTuner(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr_backbone=5e-5,
            lr_head=5e-4,
            epochs=ft_epochs_per_fold,
            warmup_epochs=2,
            use_focal_loss=True,
            class_weights=cw,
            use_mixed_precision=True,
            early_stop_patience=5,
            save_dir=os.path.join(save_dir, f"fold_{fold_idx+1}"),
            verbose=verbose,
        )
        ft_res = trainer.fit()

        # ✅ Final evaluation
        metrics = evaluate_model(model, val_loader, device)

        # ✅ FIX: دوباره fold checkpoint رو با metrics کامل ذخیره کن
        fold_best_path = os.path.join(
            save_dir, f"fold_{fold_idx+1}", "resnet50_finetuned_best.pth"
        )
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ft_res["best_epoch"],
                "val_acc": ft_res["best_val_acc"],
                "val_loss": ft_res["best_val_loss"],
                "history": ft_res["history"],
                # ✅ NEW: full metrics
                "final_metrics": {
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1_score": float(metrics["f1_score"]),
                    "f1_macro": float(metrics["f1_macro"]),
                },
            }, fold_best_path)
            print(f"   ✅ Fold {fold_idx+1} metrics saved to {fold_best_path}")
        except Exception as e:
            print(f"   ⚠️ Could not save fold metrics: {e}")

        # ✅ Free memory
        del model, trainer, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_metrics = {
            "fold": fold_idx + 1,
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
            "f1_macro": float(metrics["f1_macro"]),
            "best_val_acc": float(ft_res["best_val_acc"]),
            "best_val_loss": float(ft_res["best_val_loss"]),
        }
        fold_results.append(fold_metrics)

        if verbose:
            print(f"   ✅ Fold {fold_idx+1} — Acc: {metrics['accuracy']:.4f} | "
                  f"F1: {metrics['f1_score']:.4f}")

    # ✅ Aggregate statistics
    summary = compute_cv_summary(fold_results, n_folds)

    return {
        "fold_results": fold_results,
        "summary": summary,
        "status": "success",
    }


def compute_cv_summary(fold_results: List[Dict], n_folds: int) -> Dict[str, Any]:
    """محاسبه mean ± std و CI 95% با t-distribution."""
    try:
        from scipy import stats
    except ImportError:
        stats = None

    keys = ["accuracy", "balanced_accuracy", "precision", "recall",
            "f1_score", "f1_macro", "best_val_acc", "best_val_loss"]
    summary = {}

    for k in keys:
        vals = np.array([f[k] for f in fold_results])
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))

        # ✅ CI 95% با t-distribution
        if stats is not None and n_folds > 1:
            sem = std / np.sqrt(n_folds)
            t_crit = stats.t.ppf(0.975, df=n_folds - 1)
            ci_low = mean - t_crit * sem
            ci_high = mean + t_crit * sem
        else:
            ci_low, ci_high = mean, mean

        summary[k] = {
            "mean": mean,
            "std": std,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "values": vals.tolist(),
        }

    return summary


# ================================================================
# ✅ Plot Generators (برای PDF)
# ================================================================

def plot_loss_curves(history: Dict, save_path: str):
    """Line Bar — Loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train", color="#3498db", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val", color="#e74c3c", linewidth=2)
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Loss", fontweight="bold")
    axes[0].set_title("Loss Curves", fontweight="bold", fontsize=14)
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", color="#2ecc71", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val", color="#f39c12", linewidth=2)
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Accuracy", fontweight="bold")
    axes[1].set_title("Accuracy Curves", fontweight="bold", fontsize=14)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


def plot_lr_curves(history: Dict, save_path: str):
    """Learning rate schedule."""
    plt.figure(figsize=(12, 5))
    plt.plot(history["lr_backbone"], label="LR Backbone", linewidth=2, color="#9b59b6")
    plt.plot(history["lr_head"], label="LR Head", linewidth=2, color="#1abc9c")
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Learning Rate", fontweight="bold")
    plt.title("Learning Rate Schedule", fontweight="bold", fontsize=14)
    plt.yscale("log"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


def plot_metrics_bar(metrics: Dict, save_path: str):
    """Bar chart of main metrics."""
    names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1 (w)", "F1 (macro)"]
    values = [metrics["accuracy"], metrics["balanced_accuracy"],
              metrics["precision"], metrics["recall"],
              metrics["f1_score"], metrics["f1_macro"]]

    plt.figure(figsize=(10, 6))
    colors_bar = ["#2ecc71", "#27ae60", "#3498db", "#2980b9", "#e74c3c", "#c0392b"]
    bars = plt.bar(names, values, color=colors_bar, edgecolor="black", linewidth=1.2)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f"{v:.3f}", ha="center", fontweight="bold")
    plt.ylim(0, 1.1)
    plt.ylabel("Score", fontweight="bold")
    plt.title("Performance Metrics", fontweight="bold", fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


def plot_confusion_matrix(cm, save_path: str):
    """Confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 12, "weight": "bold"})
    plt.ylabel("True", fontweight="bold")
    plt.xlabel("Predicted", fontweight="bold")
    plt.title("Confusion Matrix", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


def plot_roc_curves(roc_data, save_path: str):
    """ROC curves per class."""
    fpr, tpr, roc_auc = roc_data
    plt.figure(figsize=(8, 7))
    colors_roc = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i in range(NUM_CLASSES):
        if i in fpr and i in tpr:
            plt.plot(fpr[i], tpr[i],
                     label=f"{CLASS_NAMES[i]} (AUC = {roc_auc.get(i, 0):.3f})",
                     linewidth=2.2, color=colors_roc[i])
    plt.plot([0,1], [0,1], "k--", linewidth=1.5)
    plt.xlabel("FPR", fontweight="bold"); plt.ylabel("TPR", fontweight="bold")
    plt.title("ROC Curves", fontweight="bold", fontsize=14)
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


def plot_pie_chart(labels, save_path: str):
    """Class distribution pie."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    plt.figure(figsize=(7, 7))
    colors_pie = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    plt.pie(counts, labels=CLASS_NAMES, autopct="%1.1f%%",
            colors=colors_pie, explode=[0.05]*NUM_CLASSES)
    plt.title("Class Distribution", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


def plot_cv_summary(cv_summary: Dict, save_path: str):
    """5-fold CV — Accuracy + F1 with error bars و CI95%."""
    accs = cv_summary["accuracy"]["values"]
    f1s = cv_summary["f1_score"]["values"]
    folds = list(range(1, len(accs)+1))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(folds)); w = 0.35
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#3498db", edgecolor="black")
    b2 = ax.bar(x + w/2, f1s, w, label="F1-Score", color="#2ecc71", edgecolor="black")

    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.3f}",
                    ha="center", fontsize=9, fontweight="bold")

    mean_acc = cv_summary["accuracy"]["mean"]
    ci_low = cv_summary["accuracy"]["ci95_low"]
    ci_high = cv_summary["accuracy"]["ci95_high"]
    ax.axhline(mean_acc, color="red", linestyle="--", linewidth=2,
               label=f"Mean Acc: {mean_acc:.4f} (CI95: [{ci_low:.4f}, {ci_high:.4f}])")

    ax.set_xticks(x); ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylabel("Score", fontweight="bold"); ax.set_ylim(0, 1.1)
    ax.set_title("5-Fold CV Results", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


def plot_cv_box(cv_summary: Dict, save_path: str):
    """Box plot of all metrics across folds."""
    keys = ["accuracy", "precision", "recall", "f1_score", "f1_macro"]
    data = [cv_summary[k]["values"] for k in keys]
    labels = [k.replace("_", " ").title() for k in keys]

    plt.figure(figsize=(10, 6))

    # ✅ FIX: سازگار با matplotlib قدیمی و جدید
    try:
        # matplotlib >= 3.9
        bp = plt.boxplot(data, tick_labels=labels, patch_artist=True)
    except TypeError:
        # matplotlib < 3.9 (fallback)
        bp = plt.boxplot(data, labels=labels, patch_artist=True)

    colors_box = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color);
        patch.set_alpha(0.7)
    plt.ylabel("Score", fontweight="bold")
    plt.title("CV — Metric Distribution Across 5 Folds", fontweight="bold", fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


def plot_ci_bars(cv_summary: Dict, save_path: str):
    """Bar chart with CI 95% error bars."""
    keys = ["accuracy", "precision", "recall", "f1_score"]
    means = [cv_summary[k]["mean"] for k in keys]
    ci_low = [cv_summary[k]["ci95_low"] for k in keys]
    ci_high = [cv_summary[k]["ci95_high"] for k in keys]
    err_low = [m - l for m, l in zip(means, ci_low)]
    err_high = [h - m for m, h in zip(means, ci_high)]

    plt.figure(figsize=(10, 6))
    colors_bar = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
    bars = plt.bar([k.replace("_", " ").title() for k in keys], means,
                   yerr=[err_low, err_high], capsize=10,
                   color=colors_bar, edgecolor="black", linewidth=1.5)
    for bar, m in zip(bars, means):
        plt.text(bar.get_x()+bar.get_width()/2, m+0.03, f"{m:.3f}",
                 ha="center", fontweight="bold")
    plt.ylim(0, 1.15)
    plt.ylabel("Score (mean ± CI 95%)", fontweight="bold")
    plt.title("Metrics with 95% Confidence Interval", fontweight="bold", fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


# ================================================================
# ✅ Interpretability (LIME, SHAP, Captum)
# ================================================================

def generate_all_interpretability(model, dataloader, output_dir, device):
    """Generate LIME, SHAP, Captum, Grad-CAM."""
    results = {
        "captum_gradcam_path": None,
        "captum_ig_path": None,
        "lime_explanation_path": None,
        "shap_summary_path": None,
    }

    os.makedirs(output_dir, exist_ok=True)

    # ✅ Captum
    try:
        from src.reporter.captum_reporter import generate_interpretability_report
        captum_res = generate_interpretability_report(
            model, dataloader, output_dir, device=device, class_names=CLASS_NAMES,
        )
        results["captum_gradcam_path"] = captum_res.get("gradcam_path")
        results["captum_ig_path"] = captum_res.get("ig_path")
    except Exception as e:
        print(f"⚠️ Captum failed: {e}")

    # ✅ LIME
    try:
        from src.reporter.lime_reporter import explain_predictions_lime
        lime_res = explain_predictions_lime(
            model, dataloader, output_dir, class_names=CLASS_NAMES, device=device,
        )
        if isinstance(lime_res, dict):
            results["lime_explanation_path"] = lime_res.get("lime_path")
    except Exception as e:
        print(f"⚠️ LIME failed: {e}")

    # ✅ SHAP
    try:
        from src.reporter.shap_reporter import generate_shap_explanation
        shap_res = generate_shap_explanation(
            model, dataloader, output_dir, device=device, class_names=CLASS_NAMES,
        )
        if isinstance(shap_res, dict):
            results["shap_summary_path"] = shap_res.get("shap_path")
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")

    return results


def generate_netron_onnx(model, output_dir, model_name="ResNet50"):
    """Generate Netron HTML + ONNX."""
    try:
        from src.reporter.netron_reporter import save_model_with_name
        res = save_model_with_name(model, output_dir, model_name=model_name,
                                    input_shape=(1, 3, 224, 224))
        return res
    except Exception as e:
        print(f"⚠️ Netron/ONNX failed: {e}")
        return {"onnx_path": None, "html_path": None, "info_path": None}


# ================================================================
# ✅ 14-Page PDF Report Generator
# ================================================================

def generate_14page_pdf(
    output_path: str,
    config: Dict,
    metrics: Dict,
    cv_summary: Dict,
    history: Dict,
    plot_paths: Dict,
    interp_paths: Dict,
    netron_paths: Dict,
    labeled_loader=None,
):
    """
    Generate comprehensive PDF report (15 pages):
    Page 1:  Info
    Page 2:  Tables of all details
    Page 3:  Line Bar (Loss/Acc)
    Page 4:  Bar Chart (Metrics)
    Page 5:  Pie Chart (Class Distribution)
    Page 6:  Confusion Matrix (dedicated)
    Page 7:  Graph Charts (ROC + extra)
    Page 8:  LIME
    Page 9:  SHAP
    Page 10: Captum (Grad-CAM + IG)
    Page 11: Netron
    Page 12: ONNX
    Page 13: Utilities
    Page 14: Grad-CAM (aggregate)
    Page 15: Summary + CV + CI95%
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import (
            PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        print("❌ ReportLab missing")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             leftMargin=50, rightMargin=50,
                             topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                   fontSize=22, spaceAfter=20, alignment=TA_CENTER,
                                   textColor=colors.HexColor("#1a252f"))
    h1_style = ParagraphStyle("H1", parent=styles["Heading2"],
                               fontSize=16, spaceAfter=12, spaceBefore=14,
                               textColor=colors.HexColor("#2c3e50"))
    h2_style = ParagraphStyle("H2", parent=styles["Heading3"],
                               fontSize=13, spaceAfter=8, spaceBefore=10,
                               textColor=colors.HexColor("#34495e"))
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10,
                           spaceAfter=6, leading=14)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8,
                            textColor=colors.grey)

    def safe_img(path, w=6.5*inch, h=4.5*inch):
        if path and os.path.exists(path) and os.path.getsize(path) > 100:
            try:
                return RLImage(path, width=w, height=h)
            except Exception:
                return Paragraph("⚠️ Could not load image", small)
        return Paragraph("⚠️ Image not available", small)

    story = []

    # ============================================================
    # PAGE 1: INFO
    # ============================================================
    story.append(Paragraph("ResNet50 Fine-Tuned — Full Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            ParagraphStyle("center", parent=small, alignment=TA_CENTER)))
    story.append(Spacer(1, 20))

    story.append(Paragraph("1. Overview", h1_style))
    overview = (
        f"This report presents the complete <b>Supervised Fine-Tuning</b> of "
        f"<b>ResNet50</b> initialized with transferred SSL weights "
        f"(source: <b>{config.get('best_method', 'N/A').upper()}</b>). "
        f"The model was fine-tuned for <b>{config.get('epochs', 0)} epochs</b> "
        f"using class-weighted Focal Loss, then evaluated with 5-Fold Cross-Validation "
        f"and 95% Confidence Intervals. "
        f"Best Val Accuracy: <b>{metrics.get('accuracy', 0):.4f}</b>, "
        f"F1-Score: <b>{metrics.get('f1_score', 0):.4f}</b>."
    )
    story.append(Paragraph(overview, body))

    story.append(Paragraph("Configuration", h2_style))
    cfg_data = [["Parameter", "Value"]]
    for k, v in config.items():
        cfg_data.append([str(k), str(v)])
    cfg_table = Table(cfg_data, colWidths=[2.5*inch, 3.5*inch])
    cfg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
    ]))
    story.append(cfg_table)

    # ============================================================
    # PAGE 2: TABLES OF ALL DETAILS
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("2. Detailed Metrics", h1_style))

    story.append(Paragraph("2.1 Overall Performance", h2_style))
    met_data = [["Metric", "Value"]]
    for k in ["accuracy", "balanced_accuracy", "precision", "recall",
              "f1_score", "f1_macro", "num_samples"]:
        v = metrics.get(k, 0)
        met_data.append([k.replace("_", " ").title(),
                          f"{v:.4f}" if isinstance(v, float) else str(v)])
    met_table = Table(met_data, colWidths=[3*inch, 3*inch])
    met_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(met_table)

    story.append(Paragraph("2.2 Per-Class Performance", h2_style))
    cr = metrics.get("classification_report", {})
    pc_data = [["Class", "Precision", "Recall", "F1", "Support"]]
    for cname in CLASS_NAMES:
        if cname in cr:
            r = cr[cname]
            pc_data.append([cname, f"{r.get('precision',0):.4f}",
                             f"{r.get('recall',0):.4f}",
                             f"{r.get('f1-score',0):.4f}",
                             str(r.get('support',0))])
    pc_table = Table(pc_data, colWidths=[1.2*inch]*5)
    pc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(pc_table)

    # ============================================================
    # PAGE 3: LINE BAR (Loss/Acc Curves)
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Training Curves (Line Bar)", h1_style))
    story.append(safe_img(plot_paths.get("loss_curves")))
    story.append(safe_img(plot_paths.get("lr_curves"), h=4*inch))

    # ============================================================
    # PAGE 4: BAR CHART (Metrics)
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Bar Chart — Metrics", h1_style))
    story.append(safe_img(plot_paths.get("metrics_bar")))

    # ============================================================
    # PAGE 5: PIE CHART (Class Distribution)
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Pie Chart — Class Distribution", h1_style))
    story.append(safe_img(plot_paths.get("pie_chart"), w=5*inch, h=5*inch))

    # ============================================================
    # ✅ PAGE 6: CONFUSION MATRIX (DEDICATED — NEW)
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("6. Confusion Matrix", h1_style))
    story.append(Paragraph(
        "The confusion matrix shows actual vs predicted class counts. "
        "Diagonal cells = correct predictions. Off-diagonal cells = errors. "
        "The darker blue cells indicate larger values.",
        body
    ))
    story.append(Spacer(1, 10))

    # Image
    if not safe_img(plot_paths.get("confusion_matrix"), w=6*inch, h=5.5*inch):
        story.append(Paragraph("⚠️ Confusion matrix image not available.", body))

    # Table form below the image
    story.append(Spacer(1, 15))
    story.append(Paragraph("6.1 Confusion Matrix (Table Form)", h2_style))

    cm = metrics.get("confusion_matrix", None)
    if cm is not None:
        cm = np.asarray(cm)
        cm_data = [["True \\ Pred"] + CLASS_NAMES]
        for i, cname in enumerate(CLASS_NAMES):
            row = [cname] + [str(int(cm[i][j])) for j in range(len(CLASS_NAMES))]
            cm_data.append(row)

        cm_table = Table(cm_data, colWidths=[1.2*inch]*(len(CLASS_NAMES)+1))
        cm_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#34495e")),
            ("TEXTCOLOR", (0, 1), (0, -1), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (1, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ]))
        story.append(cm_table)
    else:
        story.append(Paragraph("⚠️ Confusion matrix data not available.", body))

    # ============================================================
    # PAGE 7: GRAPH CHARTS (ROC + extra)
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("7. ROC Curves — Graph Charts", h1_style))
    story.append(Paragraph(
        "ROC curves plot True Positive Rate vs False Positive Rate per class. "
        "AUC = 1.0 indicates perfect classification; AUC = 0.5 is random.",
        body
    ))
    story.append(safe_img(plot_paths.get("roc_curves"), w=6*inch, h=5.5*inch))

    # ============================================================
    # PAGE 8: LIME
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("8. LIME Explanations", h1_style))
    story.append(Paragraph(
        "LIME perturbs image regions to identify which parts "
        "influence the model's predictions locally.", body))
    story.append(safe_img(interp_paths.get("lime_explanation_path")))

    # ============================================================
    # PAGE 9: SHAP
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("9. SHAP Explanations", h1_style))
    story.append(Paragraph(
        "SHAP assigns per-pixel importance using "
        "game-theoretic Shapley values.", body))
    story.append(safe_img(interp_paths.get("shap_summary_path")))

    # ============================================================
    # PAGE 10: CAPTUM
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("10. Captum — Grad-CAM & Integrated Gradients", h1_style))
    story.append(safe_img(interp_paths.get("captum_gradcam_path")))
    story.append(Spacer(1, 10))
    story.append(safe_img(interp_paths.get("captum_ig_path")))

    # ============================================================
    # PAGE 11: NETRON
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("11. Netron Architecture Visualization", h1_style))
    netron_info = f"""
    <b>ONNX Path:</b> {netron_paths.get('onnx_path', 'N/A')}<br/>
    <b>HTML Path:</b> {netron_paths.get('html_path', 'N/A')}<br/>
    <b>Info JSON:</b> {netron_paths.get('info_path', 'N/A')}<br/><br/>
    Open the HTML file in a browser to view the interactive architecture, or
    upload the ONNX file to <b>https://netron.app</b>.
    """
    story.append(Paragraph(netron_info, body))

    # ============================================================
    # PAGE 12: ONNX DETAILS
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("12. ONNX Details", h1_style))
    onnx_path = netron_paths.get("onnx_path")
    if onnx_path and os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / 1e6
        onnx_info = f"""
        <b>File:</b> {onnx_path}<br/>
        <b>Size:</b> {size_mb:.2f} MB<br/>
        <b>Opset:</b> 13<br/>
        <b>Input Names:</b> input<br/>
        <b>Output Names:</b> output<br/>
        <b>Dynamic Axes:</b> batch_size<br/>
        """
        story.append(Paragraph(onnx_info, body))
    else:
        story.append(Paragraph("⚠️ ONNX file not available", body))

    # ============================================================
    # PAGE 13: UTILITIES
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("13. Utilities Report", h1_style))
    utils_info = f"""
    <b>Environment:</b><br/>
    - PyTorch: {torch.__version__}<br/>
    - CUDA Available: {torch.cuda.is_available()}<br/>
    - Device: {config.get('device', 'N/A')}<br/>
    <br/>
    <b>Model Details:</b><br/>
    - Model: ResNet50<br/>
    - Num Classes: {NUM_CLASSES}<br/>
    - Input Size: 224×224<br/>
    - Backbone Params + Head Params (differential LR)<br/>
    <br/>
    <b>Training Details:</b><br/>
    - Loss: Focal Loss (γ=2.0)<br/>
    - Class Weights: sqrt-based<br/>
    - Optimizer: AdamW<br/>
    - Scheduler: Warmup + CosineAnnealing<br/>
    - Mixed Precision: {config.get('use_mixed_precision', True)}<br/>
    - Grad Clip: {config.get('clip_grad_norm', 1.0)}<br/>
    """
    story.append(Paragraph(utils_info, body))

    # ============================================================
    # PAGE 14: GRAD-CAM (Aggregate)
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("14. Grad-CAM Reporter (Aggregate)", h1_style))
    story.append(Paragraph(
        "Aggregate Grad-CAM visualization across samples. "
        "Highlighted regions indicate pixels most responsible for "
        "the model's predictions.", body))
    story.append(safe_img(interp_paths.get("captum_gradcam_path")))

    # ============================================================
    # PAGE 15: SUMMARY
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("15. Summary & Recommendations", h1_style))

    # CV table
    story.append(Paragraph("5-Fold Cross-Validation Results (with CI 95%)", h2_style))
    cv_data = [["Metric", "Mean", "Std", "Min", "Max", "CI 95% Low", "CI 95% High"]]
    for k in ["accuracy", "f1_score", "f1_macro", "balanced_accuracy"]:
        if k in cv_summary:
            s = cv_summary[k]
            cv_data.append([
                k.replace("_", " ").title(),
                f"{s['mean']:.4f}", f"{s['std']:.4f}",
                f"{s['min']:.4f}", f"{s['max']:.4f}",
                f"{s['ci95_low']:.4f}", f"{s['ci95_high']:.4f}",
            ])
    cv_table = Table(cv_data, colWidths=[1.5*inch, 0.9*inch, 0.9*inch,
                                           0.9*inch, 0.9*inch, 1.0*inch, 1.0*inch])
    cv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(cv_table)

    story.append(Spacer(1, 15))
    story.append(Paragraph("CV Visualization", h2_style))
    story.append(safe_img(plot_paths.get("cv_summary"), w=6*inch, h=4*inch))
    story.append(Spacer(1, 10))
    story.append(safe_img(plot_paths.get("cv_box"), w=6*inch, h=4*inch))
    story.append(Spacer(1, 10))
    story.append(safe_img(plot_paths.get("ci_bars"), w=6*inch, h=4*inch))

    story.append(Spacer(1, 15))
    acc_mean = cv_summary.get("accuracy", {}).get("mean", 0)
    acc_ci = (cv_summary.get("accuracy", {}).get("ci95_low", 0),
              cv_summary.get("accuracy", {}).get("ci95_high", 0))
    summary_text = f"""
    <b>Final Verdict:</b><br/>
    The fine-tuned ResNet50 achieved a mean accuracy of
    <b>{acc_mean:.4f}</b> across 5 folds
    (95% CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]).<br/><br/>
    <b>Recommendations:</b><br/>
    - The model is ready for deployment if mean accuracy &gt; 0.75.<br/>
    - Consider additional data collection for G3 class if recall &lt; 0.60.<br/>
    - For production, ensemble with BYOL/SimCLR/MAE variants.<br/>
    """
    story.append(Paragraph(summary_text, body))

    # ✅ Build
    try:
        doc.build(story)
        print(f"✅ 15-Page PDF Report saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ PDF build failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ================================================================
# ✅ Main Orchestrator
# ================================================================

def run_full_finetune_pipeline(
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 32,
    input_size: int = 224,
    cv_folds: int = 5,
    cv_epochs_per_fold: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline:
    1. Load best SSL checkpoint
    2. Load ResNet50 base + transfer
    3. Full Fine-Tune
    4. Evaluate
    5. 5-Fold CV with CI 95%
    6. Generate 14-page PDF report
    """
    ensure_dirs()

    print("\n" + "="*80)
    print("🚀 RESNET50 FULL FINE-TUNE PIPELINE")
    print("="*80)

    # ✅ STEP 1: Load best SSL checkpoint
    from src.self_supervised.ProbabilityAlgorithms.linear_probe import (
        evaluate_with_linear_probe, build_labeled_eval_loader,
    )
    from src.pipeLine.resNet18SSL import create_resnet18_ssl_pipeline

    methods = ["byol", "simclr", "mae"]
    ssl_checkpoints = {}
    for m in methods:
        p = os.path.join(SELF_SUPERVISED_ROOT, "..", "18", "SSL_checkpoints",
                          "architecture", m, f"resnet18_best_update_{m}.pth")
        if os.path.exists(p):
            ssl_checkpoints[m] = p

    if not ssl_checkpoints:
        raise FileNotFoundError("No SSL checkpoints found!")

    # ✅ Load all + LP
    eval_loader = build_labeled_eval_loader(
        labels_csv="finalReports/labels.csv",
        image_dir="finalReports/cropped",
        input_size=input_size, batch_size=32, max_per_class=200,
    )

    lp_results, ssl_pipelines = {}, {}
    for m, p in ssl_checkpoints.items():
        try:
            pl = create_resnet18_ssl_pipeline(method=m, verbose=False, device=device)
            pl.load_model(p)
            ssl_pipelines[m] = pl
            enc = pl.get_encoder()
            probe = evaluate_with_linear_probe(enc, eval_loader, device=device,
                                                num_classes=NUM_CLASSES, test_size=0.3)
            lp_results[m] = probe
        except Exception as e:
            print(f"⚠️ Failed loading {m}: {e}")

    best_method = max(lp_results.keys(), key=lambda x: lp_results[x]["linear_probe_accuracy"])
    best_lp = lp_results[best_method]["linear_probe_accuracy"]
    print(f"\n🏆 Best SSL: {best_method.upper()} (LP Acc = {best_lp:.4f})")

    # ✅ STEP 2: Load ResNet50 base
    from src.models.resnet.resnet50 import create_resnet50

    resnet50_base = os.path.join(RESNET_WEIGHT_TRANSFORMER_DIR, "resnet50_base.pth")
    if not os.path.exists(resnet50_base):
        resnet50_base = os.path.join(
            "finalReports", "models", "..", "resnet50",
            "checkpoints", "resnet50_best.pth",
        )
    if not os.path.exists(resnet50_base):
        raise FileNotFoundError(f"ResNet50 base not found: {resnet50_base}")

    # ✅ Transfer SSL weights
    best_enc = ssl_pipelines[best_method].get_encoder()
    ssl_state = best_enc.state_dict()
    resnet50 = create_resnet50(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
    base_ckpt = torch.load(resnet50_base, map_location="cpu")
    base_state = base_ckpt.get("model_state_dict") or base_ckpt.get("encoder_state_dict") or base_ckpt

    new_state = {}
    transferred = 0
    for name, param in resnet50.state_dict().items():
        if name in ssl_state and ssl_state[name].shape == param.shape:
            new_state[name] = ssl_state[name]
            transferred += 1
        elif name in base_state and base_state[name].shape == param.shape:
            new_state[name] = base_state[name]
        else:
            new_state[name] = param
    resnet50.load_state_dict(new_state, strict=False)
    print(f"✅ ResNet50 created with {transferred} SSL layers transferred")

    # ✅ Capture initial state for CV folds
    initial_state = copy.deepcopy(resnet50.state_dict())

    def model_factory():
        m = create_resnet50(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
        m.load_state_dict(initial_state, strict=False)
        return m

    # ✅ STEP 3: Build full dataset
    paths, labels = build_labeled_dataset()

    # ✅ STEP 4: Full Fine-Tune on 70/30 split
    print(f"\n{'='*80}")
    print(f"🎯 STEP 4: FULL FINE-TUNE")
    print(f"{'='*80}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=42,
    )

    train_ds = LabeledLiverDataset(train_paths, train_labels,
                                     transform=get_train_transform(input_size),
                                     target_size=input_size)
    val_ds = LabeledLiverDataset(val_paths, val_labels,
                                   transform=get_eval_transform(input_size),
                                   target_size=input_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ✅ Class weights
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    raw_w = 1.0 / (counts + 1e-8)
    sqrt_w = np.sqrt(raw_w)
    norm_w = sqrt_w / sqrt_w.sum() * NUM_CLASSES
    cw = torch.tensor(norm_w, dtype=torch.float32)

    trainer = ResNet50FineTuner(
        model=resnet50, train_loader=train_loader, val_loader=val_loader,
        device=device, lr_backbone=1e-4, lr_head=1e-3,
        epochs=epochs, warmup_epochs=5,
        use_focal_loss=True, class_weights=cw,
        use_mixed_precision=True, early_stop_patience=15,
        save_dir=FINAL_MODELS_DIR, verbose=verbose,
    )
    ft_results = trainer.fit()

    # ✅ STEP 5: Final evaluation on val set
    print(f"\n{'='*80}")
    print(f"📊 STEP 5: FINAL EVALUATION")
    print(f"{'='*80}")
    final_metrics = evaluate_model(resnet50, val_loader, device)
    print(f"   Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"   F1-Score:  {final_metrics['f1_score']:.4f}")
    print(f"   F1-Macro:  {final_metrics['f1_macro']:.4f}")

    # ✅ STEP 6: 5-Fold CV
    print(f"\n{'='*80}")
    print(f"🔬 STEP 6: 5-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")

    cv_results = run_cv5_folds(
        model_factory=model_factory,
        paths=paths, labels=labels, device=device,
        n_folds=cv_folds, ft_epochs_per_fold=cv_epochs_per_fold,
        batch_size=batch_size, input_size=input_size,
        save_dir=CV_REPORTS_DIR, verbose=verbose,
    )

    cv_summary = cv_results["summary"]
    print(f"\n📊 CV Summary (CI 95%):")
    for k in ["accuracy", "f1_score", "f1_macro"]:
        s = cv_summary[k]
        print(f"   {k}: {s['mean']:.4f} ± {s['std']:.4f} "
              f"(CI95: [{s['ci95_low']:.4f}, {s['ci95_high']:.4f}])")

    # ✅ STEP 7: Generate Plots
    print(f"\n{'='*80}")
    print(f"📈 STEP 7: GENERATING PLOTS")
    print(f"{'='*80}")

    plot_paths = {}
    plot_paths["loss_curves"] = plot_loss_curves(ft_results["history"],
                                                   os.path.join(PLOTS_DIR, "loss_curves.png"))
    plot_paths["lr_curves"] = plot_lr_curves(ft_results["history"],
                                               os.path.join(PLOTS_DIR, "lr_curves.png"))
    plot_paths["metrics_bar"] = plot_metrics_bar(final_metrics,
                                                   os.path.join(PLOTS_DIR, "metrics_bar.png"))
    plot_paths["confusion_matrix"] = plot_confusion_matrix(final_metrics["confusion_matrix"],
                                                             os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_paths["roc_curves"] = plot_roc_curves(final_metrics["roc_data"],
                                                 os.path.join(PLOTS_DIR, "roc_curves.png"))
    plot_paths["pie_chart"] = plot_pie_chart(labels,
                                               os.path.join(PLOTS_DIR, "pie_chart.png"))
    plot_paths["cv_summary"] = plot_cv_summary(cv_summary,
                                                 os.path.join(PLOTS_DIR, "cv_summary.png"))
    plot_paths["cv_box"] = plot_cv_box(cv_summary,
                                         os.path.join(PLOTS_DIR, "cv_box.png"))
    plot_paths["ci_bars"] = plot_ci_bars(cv_summary,
                                           os.path.join(PLOTS_DIR, "ci_bars.png"))
    print(f"✅ Generated {len(plot_paths)} plots")

    # ✅ STEP 8: Interpretability
    print(f"\n{'='*80}")
    print(f"🔬 STEP 8: INTERPRETABILITY")
    print(f"{'='*80}")
    # Use a subset for speed
    interp_subset = Subset(val_ds, list(range(min(20, len(val_ds)))))
    interp_loader = DataLoader(interp_subset, batch_size=1, shuffle=False)
    interp_paths = generate_all_interpretability(resnet50, interp_loader, INTERP_DIR, device)

    # ✅ STEP 9: Netron / ONNX
    print(f"\n{'='*80}")
    print(f"🏗️  STEP 9: NETRON / ONNX")
    print(f"{'='*80}")
    netron_paths = generate_netron_onnx(resnet50, ARCH_DIR, "ResNet50")

    # ✅ STEP 10: Save Final Model
    final_model_path = os.path.join(FINAL_MODELS_DIR, "resnet50_finetuned_final.pth")
    torch.save({
        "model_state_dict": resnet50.state_dict(),
        "best_method": best_method,
        "best_lp_accuracy": best_lp,
        "final_metrics": {k: v for k, v in final_metrics.items()
                           if k not in ["y_true", "y_pred", "y_prob",
                                         "confusion_matrix", "roc_data",
                                         "classification_report"]},
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "input_size": input_size,
            "lr_backbone": 1e-4,
            "lr_head": 1e-3,
        },
        "timestamp": datetime.now().isoformat(),
    }, final_model_path)
    print(f"✅ Final model saved: {final_model_path}")

    # ✅ STEP 11: Generate 14-page PDF
    print(f"\n{'='*80}")
    print(f"📄 STEP 11: GENERATING 14-PAGE PDF")
    print(f"{'='*80}")
    pdf_path = os.path.join(RESNET50_REPORTS_DIR, "ResNet50_Full_Report.pdf")
    generate_14page_pdf(
        output_path=pdf_path,
        config={
            "best_method": best_method,
            "best_lp_accuracy": best_lp,
            "epochs": epochs,
            "batch_size": batch_size,
            "input_size": input_size,
            "device": device,
            "use_mixed_precision": True,
            "clip_grad_norm": 1.0,
        },
        metrics=final_metrics,
        cv_summary=cv_summary,
        history=ft_results["history"],
        plot_paths=plot_paths,
        interp_paths=interp_paths,
        netron_paths=netron_paths,
        labeled_loader=eval_loader,
    )

    # ✅ Return summary
    return {
        "best_method": best_method,
        "best_lp_accuracy": best_lp,
        "final_model_path": final_model_path,
        "pdf_path": pdf_path,
        "metrics": {k: v for k, v in final_metrics.items()
                     if k not in ["y_true", "y_pred", "y_prob",
                                   "confusion_matrix", "roc_data",
                                   "classification_report"]},
        "cv_summary": cv_summary,
        "ft_results": ft_results,
        "plot_paths": plot_paths,
        "interp_paths": interp_paths,
        "netron_paths": netron_paths,
    }


# ================================================================
# ✅ NEW: Fine-Tune Only (from pre-transferred model) — برای گزینه 7
# ================================================================

def run_finetune_only(
        model_path: str,
        device: str = "cuda",
        epochs: int = 50,
        batch_size: int = 32,
        input_size: int = 224,
        cv_folds: int = 5,
        cv_epochs_per_fold: int = 10,
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    ✅ NEW: Fine-Tune فقط روی مدل Transfer شده.

    این تابع برای گزینه 7 طراحی شده:
    - فرض می‌کنه که مدل Transfer (ResNet18 SSL → ResNet50) قبلاً ساخته شده
      و در weightTransformer/ ذخیره شده
    - دیگه دوباره SSL رو load نمی‌کنه، Linear Probe نمی‌زنه، Transfer نمی‌کنه
    - فقط از مرحله Fine-Tune به بعد رو انجام می‌ده

    Steps:
    1. Load pre-transferred model از weightTransformer/
    2. Full Supervised Fine-Tune (Focal Loss + Class Weights)
    3. Final Evaluation
    4. 5-Fold CV + CI 95%
    5. Plots (Loss, LR, Metrics, CM, ROC, Pie, CV)
    6. Interpretability (LIME, SHAP, Captum)
    7. Netron + ONNX
    8. Save final model
    9. 15-page PDF report
    """
    ensure_dirs()

    # ✅ FIX: پاک کردن فایل‌های قدیمی
    import glob as _glob
    print("\n🧹 Cleaning old files...")
    old_patterns = [
        os.path.join(FINAL_MODELS_DIR, "resnet50_finetuned_best.pth"),
        os.path.join(FINAL_MODELS_DIR, "resnet50_finetuned_final.pth"),
    ]
    for pat in old_patterns:
        for f in _glob.glob(pat):
            try:
                os.remove(f)
                print(f"   🗑️  Removed: {f}")
            except Exception:
                pass
    # پاک کردن fold های قدیمی
    import shutil as _shutil
    for i in range(1, 11):
        fold_dir = os.path.join(CV_REPORTS_DIR, f"fold_{i}")
        if os.path.exists(fold_dir):
            try:
                _shutil.rmtree(fold_dir)
                print(f"   🗑️  Removed: {fold_dir}")
            except Exception:
                pass

    print("\n" + "=" * 80)
    print("🚀 RESNET50 FINE-TUNE (FROM PRE-TRANSFERRED MODEL)")
    print("=" * 80)
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Input Size: {input_size}")
    print(f"   CV Folds: {cv_folds}")
    print(f"   CV Epochs/Fold: {cv_epochs_per_fold}")
    print("=" * 80)

    # ============================================================
    # ✅ STEP 1: Load Pre-Transferred Model
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"📂 STEP 1: LOADING PRE-TRANSFERRED MODEL")
    print(f"{'=' * 80}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"   Path: {model_path}")
    print(f"   Size: {os.path.getsize(model_path) / 1e6:.2f} MB")

    checkpoint = torch.load(model_path, map_location="cpu")

    # ✅ استخراج state dict با تمام احتمالات
    if "encoder_state_dict" in checkpoint:
        state_dict = checkpoint["encoder_state_dict"]
        source_key = "encoder_state_dict"
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        source_key = "model_state_dict"
    else:
        state_dict = checkpoint
        source_key = "direct"

    print(f"   State dict source: {source_key}")
    print(f"   Num layers in checkpoint: {len(state_dict)}")

    # ✅ استخراج metadata
    best_method = checkpoint.get("source_method", "unknown")
    best_lp = checkpoint.get("best_lp_accuracy", 0.0)
    transferred_layers = checkpoint.get("transferred_layers", 0)
    source_model = checkpoint.get("source_model", "ResNet18_SSL")
    target_model = checkpoint.get("target_model", "ResNet50")

    print(f"\n   📋 Transfer Metadata:")
    print(f"      Source SSL Method: {best_method.upper()}")
    print(f"      Best LP Accuracy:  {best_lp:.4f}")
    print(f"      Source Model:      {source_model}")
    print(f"      Target Model:      {target_model}")
    print(f"      Transferred Layers: {transferred_layers}")

    # ✅ ساخت ResNet50 و load
    from src.models.resnet.resnet50 import create_resnet50

    resnet50 = create_resnet50(num_classes=NUM_CLASSES, pretrained=False, verbose=False)

    # ✅ Load با strict=False و چک کردن mismatches
    model_state = resnet50.state_dict()

    loaded = 0
    mismatched = 0
    missing = 0
    shape_mismatch = []

    for name, param in state_dict.items():
        if name in model_state:
            if model_state[name].shape == param.shape:
                model_state[name] = param
                loaded += 1
            else:
                shape_mismatch.append(name)
                mismatched += 1
        else:
            missing += 1

    for name in model_state.keys():
        if name not in state_dict:
            pass  # از وزن‌های پیش‌فرض ResNet50 استفاده می‌شه

    resnet50.load_state_dict(model_state, strict=False)

    print(f"\n   ✅ Model loaded successfully:")
    print(f"      Loaded layers:   {loaded}")
    print(f"      Shape mismatch:  {mismatched}")
    print(f"      Missing (new):   {missing}")
    print(f"      Total layers:    {len(model_state)}")

    if shape_mismatch:
        print(f"      ⚠️ Shape mismatches (first 5): {shape_mismatch[:5]}")

    # ✅ کپی از state اولیه برای CV (بدون آلودگی)
    initial_state = copy.deepcopy(resnet50.state_dict())

    def model_factory():
        """Fresh model for each CV fold."""
        m = create_resnet50(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
        m.load_state_dict(initial_state, strict=False)
        return m

    # ============================================================
    # ✅ STEP 2: Load Dataset
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"📂 STEP 2: LOADING DATASET")
    print(f"{'=' * 80}")

    paths, labels = build_labeled_dataset()

    if len(paths) == 0:
        raise RuntimeError("No labeled samples found!")

    print(f"   Total samples: {len(paths)}")
    print(f"   Classes: {CLASS_NAMES}")

    # ============================================================
    # ✅ STEP 3: Full Fine-Tune on 80/20 Split
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"🎯 STEP 3: FULL FINE-TUNE (80/20 Split)")
    print(f"{'=' * 80}")

    # ✅ v2.0: 80/10/10 Split (train/val/test)
    # Step 1: 80% train vs 20% temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.20, stratify=labels, random_state=42,
    )

    # Step 2: split temp (20%) into 10% val + 10% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42,
    )

    print(f"   Train samples: {len(train_paths)} ({len(train_paths)/len(paths)*100:.1f}%)")
    print(f"   Val samples:   {len(val_paths)} ({len(val_paths)/len(paths)*100:.1f}%)")
    print(f"   Test samples:  {len(test_paths)} ({len(test_paths)/len(paths)*100:.1f}%)")
    print(f"   Total:         {len(paths)}")

    # Class distribution per split
    print(f"\n   Class distribution per split:")
    for name, lbls in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        counts = [int((lbls == i).sum()) for i in range(NUM_CLASSES)]
        print(f"      {name}: " + ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(NUM_CLASSES)))

    train_ds = LabeledLiverDataset(
        train_paths, train_labels,
        transform=get_train_transform(input_size),
        target_size=input_size,
    )
    val_ds = LabeledLiverDataset(
        val_paths, val_labels,
        transform=get_eval_transform(input_size),
        target_size=input_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ✅ Class weights (sqrt-based برای imbalance)
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    raw_w = 1.0 / (counts + 1e-8)
    sqrt_w = np.sqrt(raw_w)
    norm_w = sqrt_w / sqrt_w.sum() * NUM_CLASSES
    cw = torch.tensor(norm_w, dtype=torch.float32)

    print(f"\n   📊 Train class counts: {counts}")
    print(f"   📊 Class weights (sqrt-based):")
    for i, (cname, w) in enumerate(zip(CLASS_NAMES, norm_w)):
        print(f"      {cname}: {w:.4f}")

    # ✅ Trainer
    trainer = ResNet50FineTuner(
        model=resnet50,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr_backbone=1e-4,
        lr_head=1e-3,
        epochs=epochs,
        warmup_epochs=5,
        use_focal_loss=True,
        class_weights=cw,
        use_mixed_precision=True,
        early_stop_patience=15,
        save_dir=FINAL_MODELS_DIR,
        verbose=verbose,
    )
    ft_results = trainer.fit()

    # ============================================================
    # ✅ STEP 4: Final Evaluation on TEST SET with TTA
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"📊 STEP 4: FINAL TEST EVALUATION (with TTA)")
    print(f"{'=' * 80}")

    # ✅ Also evaluate on Val for comparison (without TTA)
    print(f"\n   🔬 Evaluating on VAL SET (early stopping criterion)...")
    val_metrics = evaluate_model(resnet50, val_loader, device)
    print(f"      Val Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"      Val F1:        {val_metrics['f1_score']:.4f}")

    # ✅ TEST evaluation with TTA
    print(f"\n   🔬 Evaluating on TEST SET (with TTA n_aug=32)...")
    final_metrics = evaluate_with_tta(
        model=resnet50,
        image_paths=test_paths,
        labels=test_labels,
        device=device,
        n_aug=32,
        batch_size=16,  # smaller for TTA memory
        num_classes=NUM_CLASSES,
        verbose=verbose,
    )

    print(f"\n   📊 TEST SET Metrics (with TTA):")
    print(f"      Accuracy:          {final_metrics['accuracy']:.4f}")
    print(f"      Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
    print(f"      Precision:         {final_metrics['precision']:.4f}")
    print(f"      Recall:            {final_metrics['recall']:.4f}")
    print(f"      F1 (weighted):     {final_metrics['f1_score']:.4f}")
    print(f"      F1 (macro):        {final_metrics['f1_macro']:.4f}")
    print(f"      AUC (mean):        {final_metrics['roc_auc_mean']:.4f}")

    print(f"\n   📊 Per-class Recall:")
    for i, r in enumerate(final_metrics["per_class_recall"]):
        print(f"      {CLASS_NAMES[i]}: {r:.4f}")

    print(f"\n   📊 Per-class AUC:")
    for i in range(NUM_CLASSES):
        if i in final_metrics["roc_auc"]:
            print(f"      {CLASS_NAMES[i]}: {final_metrics['roc_auc'][i]:.4f}")

    # ✅ Save test paths/labels for plot generation
    test_info = {
        "test_paths": test_paths,
        "test_labels": test_labels,
        "test_metrics": final_metrics,
        "val_metrics": val_metrics,
    }

    # ============================================================
    # ✅ STEP 5: 5-Fold Cross-Validation + CI 95%
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"🔬 STEP 5: {cv_folds}-FOLD CROSS-VALIDATION + CI 95%")
    print(f"{'=' * 80}")

    # ✅ v2.0: CV on train+val (test is completely held-out)
    cv_paths = train_paths + val_paths
    cv_labels = np.concatenate([train_labels, val_labels])

    print(f"\n   CV samples: {len(cv_paths)} (train+val, test excluded)")

    cv_results = run_cv5_folds(
        model_factory=model_factory,
        paths=cv_paths,
        labels=cv_labels,
        device=device,
        n_folds=cv_folds,
        ft_epochs_per_fold=cv_epochs_per_fold,
        batch_size=batch_size,
        input_size=input_size,
        save_dir=CV_REPORTS_DIR,
        verbose=verbose,
    )

    cv_summary = cv_results["summary"]

    print(f"\n📊 CV Summary (CI 95%):")
    for k in ["accuracy", "balanced_accuracy", "f1_score", "f1_macro"]:
        s = cv_summary[k]
        print(f"   {k:<20} {s['mean']:.4f} ± {s['std']:.4f} "
              f"(CI95: [{s['ci95_low']:.4f}, {s['ci95_high']:.4f}])")

    # ============================================================
    # ✅ STEP 6: Generate All Plots
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"📈 STEP 6: GENERATING PLOTS")
    print(f"{'=' * 80}")

    plot_paths = {}

    plot_paths["loss_curves"] = plot_loss_curves(
        ft_results["history"],
        os.path.join(PLOTS_DIR, "loss_curves.png"),
    )
    print(f"   ✅ loss_curves")

    plot_paths["lr_curves"] = plot_lr_curves(
        ft_results["history"],
        os.path.join(PLOTS_DIR, "lr_curves.png"),
    )
    print(f"   ✅ lr_curves")

    plot_paths["metrics_bar"] = plot_metrics_bar(
        final_metrics,
        os.path.join(PLOTS_DIR, "metrics_bar.png"),
    )
    print(f"   ✅ metrics_bar")

    plot_paths["confusion_matrix"] = plot_confusion_matrix(
        final_metrics["confusion_matrix"],
        os.path.join(PLOTS_DIR, "confusion_matrix.png"),
    )
    print(f"   ✅ confusion_matrix")

    plot_paths["roc_curves"] = plot_roc_curves(
        final_metrics["roc_data"],
        os.path.join(PLOTS_DIR, "roc_curves.png"),
    )
    print(f"   ✅ roc_curves")

    plot_paths["pie_chart"] = plot_pie_chart(
        labels,
        os.path.join(PLOTS_DIR, "pie_chart.png"),
    )
    print(f"   ✅ pie_chart")

    plot_paths["cv_summary"] = plot_cv_summary(
        cv_summary,
        os.path.join(PLOTS_DIR, "cv_summary.png"),
    )
    print(f"   ✅ cv_summary")

    plot_paths["cv_box"] = plot_cv_box(
        cv_summary,
        os.path.join(PLOTS_DIR, "cv_box.png"),
    )
    print(f"   ✅ cv_box")

    plot_paths["ci_bars"] = plot_ci_bars(
        cv_summary,
        os.path.join(PLOTS_DIR, "ci_bars.png"),
    )
    print(f"   ✅ ci_bars")

    print(f"\n   Total plots: {len(plot_paths)}")

    # ============================================================
    # ✅ STEP 7: Interpretability (LIME, SHAP, Captum)
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"🔬 STEP 7: INTERPRETABILITY")
    print(f"{'=' * 80}")

    interp_subset = Subset(val_ds, list(range(min(20, len(val_ds)))))
    interp_loader = DataLoader(interp_subset, batch_size=1, shuffle=False)
    interp_paths = generate_all_interpretability(
        resnet50, interp_loader, INTERP_DIR, device,
    )

    print(f"   ✅ Interpretability generated:")
    for k, v in interp_paths.items():
        status = "✅" if v else "⚠️"
        print(f"      {status} {k}: {v}")

    # ============================================================
    # ✅ STEP 8: Netron + ONNX
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"🏗️  STEP 8: NETRON + ONNX")
    print(f"{'=' * 80}")

    netron_paths = generate_netron_onnx(resnet50, ARCH_DIR, "ResNet50")

    print(f"   ✅ ONNX:  {netron_paths.get('onnx_path')}")
    print(f"   ✅ HTML:  {netron_paths.get('html_path')}")
    print(f"   ✅ JSON:  {netron_paths.get('info_path')}")

    # ============================================================
    # ✅ STEP 9: Save Final Model
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"💾 STEP 9: SAVING FINAL MODEL")
    print(f"{'=' * 80}")

    # ✅ نام فایل بر اساس source method
    final_model_path = os.path.join(
        FINAL_MODELS_DIR,
        f"resnet50_finetuned_from_{best_method}_final.pth",
    )

    # ✅ v2.0: Save BOTH val and test metrics
    torch.save({
        "model_state_dict": resnet50.state_dict(),
        "best_method": best_method,
        "best_lp_accuracy": best_lp,
        "source_model_path": model_path,
        # ✅ Test metrics (primary)
        "test_metrics": {
            k: v for k, v in final_metrics.items()
            if k not in ["y_true", "y_pred", "y_prob",
                         "confusion_matrix", "roc_data",
                         "classification_report", "per_class_recall",
                         "per_class_precision", "per_class_f1"]
        },
        "test_metrics_full": {
            "accuracy": final_metrics["accuracy"],
            "balanced_accuracy": final_metrics["balanced_accuracy"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1_score": final_metrics["f1_score"],
            "f1_macro": final_metrics["f1_macro"],
            "roc_auc_mean": final_metrics["roc_auc_mean"],
            "per_class_recall": final_metrics["per_class_recall"],
            "per_class_precision": final_metrics["per_class_precision"],
            "per_class_f1": final_metrics["per_class_f1"],
            "confusion_matrix": final_metrics["confusion_matrix"].tolist(),
            "classification_report": final_metrics["classification_report"],
        },
        # ✅ Val metrics (for comparison)
        "val_metrics": {
            "accuracy": val_metrics["accuracy"],
            "f1_score": val_metrics["f1_score"],
        },
        # ✅ FIX: final_metrics alias (for backward compat)
        "final_metrics": {
            k: v for k, v in final_metrics.items()
            if k not in ["y_true", "y_pred", "y_prob",
                         "confusion_matrix", "roc_data",
                         "classification_report", "per_class_recall",
                         "per_class_precision", "per_class_f1"]
        },
        "cv_summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "values"}
            for k, v in cv_summary.items()
        },
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "input_size": input_size,
            "lr_backbone": 1e-4,
            "lr_head": 1e-3,
            "use_focal_loss": True,
            "focal_gamma": 2.0,
            "class_weights": norm_w.tolist(),
            "tta_n_aug": 32,
            "split": "80/10/10",
        },
        "ft_history": ft_results["history"],
        "timestamp": datetime.now().isoformat(),
        "pipeline_stage": "fine_tune_from_transferred_with_tta",
        # ✅ Save test paths/labels for reproducibility
        "test_info": {
            "num_test_samples": len(test_paths),
            "num_val_samples": len(val_paths),
            "num_train_samples": len(train_paths),
        },
    }, final_model_path)

    print(f"   ✅ Final model: {final_model_path}")
    print(f"   ✅ Size: {os.path.getsize(final_model_path) / 1e6:.2f} MB")

    # ============================================================
    # ✅ STEP 10: Generate 15-page PDF Report
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"📄 STEP 10: GENERATING 15-PAGE PDF REPORT")
    print(f"{'=' * 80}")

    pdf_path = os.path.join(
        RESNET50_REPORTS_DIR,
        f"ResNet50_Finetuned_from_{best_method}_Report.pdf",
    )

    generate_14page_pdf(
        output_path=pdf_path,
        config={
            "best_method": best_method,
            "best_lp_accuracy": best_lp,
            "source_model_path": model_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "input_size": input_size,
            "device": device,
            "use_mixed_precision": True,
            "clip_grad_norm": 1.0,
            "focal_gamma": 2.0,
            "pipeline_stage": "fine_tune_from_transferred",
        },
        metrics=final_metrics,
        cv_summary=cv_summary,
        history=ft_results["history"],
        plot_paths=plot_paths,
        interp_paths=interp_paths,
        netron_paths=netron_paths,
        labeled_loader=val_loader,
    )

    # ============================================================
    # ✅ FINAL SUMMARY
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"🎉 FINE-TUNE FROM TRANSFERRED MODEL — COMPLETE!")
    print(f"{'=' * 80}")
    print(f"   Source SSL Method:     {best_method.upper()}")
    print(f"   Source LP Accuracy:    {best_lp:.4f}")
    print(f"   Fine-Tune Epochs:      {ft_results['best_epoch'] + 1} (best)")
    print(f"   Final Val Accuracy:    {final_metrics['accuracy']:.4f}")
    print(f"   Final Val F1:          {final_metrics['f1_score']:.4f}")
    print(f"   CV Mean Accuracy:      {cv_summary['accuracy']['mean']:.4f} ± {cv_summary['accuracy']['std']:.4f}")
    print(
        f"   CV 95% CI:             [{cv_summary['accuracy']['ci95_low']:.4f}, {cv_summary['accuracy']['ci95_high']:.4f}]")
    print(f"   Final Model:           {final_model_path}")
    print(f"   PDF Report:            {pdf_path}")
    print(f"{'=' * 80}")

    return {
        "best_method": best_method,
        "best_lp_accuracy": best_lp,
        "source_model_path": model_path,
        "final_model_path": final_model_path,
        "pdf_path": pdf_path,
        "metrics": {k: v for k, v in final_metrics.items()
                    if k not in ["y_true", "y_pred", "y_prob",
                                 "confusion_matrix", "roc_data",
                                 "classification_report"]},
        "cv_summary": cv_summary,
        "ft_results": ft_results,
        "plot_paths": plot_paths,
        "interp_paths": interp_paths,
        "netron_paths": netron_paths,
        "pipeline_stage": "fine_tune_from_transferred",
    }


# ================================================================
# ✅ NEW: Recovery — فقط از Plots به بعد (بدون FT و CV دوباره)
# ================================================================

def run_finetune_recovery(
        model_path: str = None,
        source_method: str = None,
        device: str = "cuda",
        input_size: int = 224,
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    ✅ Recovery Mode: از جایی که کرش کرد ادامه بده.

    ترتیب کارها:
    1. Load مدل Fine-Tune شده (نه دوباره FT)
    2. Load cv_summary از checkpoint (شامل CI 95%)
    3. Load fold_results از checkpoint (اگه موجود باشه)
    4. Evaluate مدل
    5. Build cv_box + ci_bars (فقط این دو تا plot)
    6. Interpretability
    7. Netron + ONNX
    8. PDF 15 صفحه
    """
    import glob

    ensure_dirs()

    print("\n" + "=" * 80)
    print("🔄 RECOVERY MODE: Continuing from where it crashed")
    print("=" * 80)
    print("   ✓ Fine-Tune:      SKIP (already saved)")
    print("   ✓ CV Folds:       SKIP (already saved)")
    print("   ✓ CI 95%:         LOAD from checkpoint")
    print("   ✓ Main Plots:     SKIP (already saved)")
    print("   → Build: cv_box, ci_bars, interpretability, netron, PDF")
    print("=" * 80)

    # ============================================================
    # ✅ STEP 1: Find Fine-Tuned Model (اولویت به _final.pth)
    # ============================================================
    if model_path is None:
        # ✅ FIX: اول _final.pth رو بگرد (metadata کامل داره)
        final_candidates = sorted(
            glob.glob(os.path.join(FINAL_MODELS_DIR, "resnet50_finetuned_from_*_final.pth")),
            key=os.path.getmtime,
            reverse=True,
        )

        if final_candidates:
            model_path = final_candidates[0]
            print(f"   ✅ Found _final.pth: {os.path.basename(model_path)}")
        else:
            # fallback به هر فایلی
            candidates = sorted(
                glob.glob(os.path.join(FINAL_MODELS_DIR, "resnet50_finetuned*.pth")),
                key=os.path.getmtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No fine-tuned model found in {FINAL_MODELS_DIR}"
                )
            model_path = candidates[0]
            print(f"   ⚠️ Using fallback: {os.path.basename(model_path)}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\n📂 STEP 1: Loading fine-tuned model")
    print(f"   Path: {model_path}")
    print(f"   Size: {os.path.getsize(model_path) / 1e6:.2f} MB")

    checkpoint = torch.load(model_path, map_location="cpu")

    # ✅ استخراج state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "encoder_state_dict" in checkpoint:
        state_dict = checkpoint["encoder_state_dict"]
    else:
        state_dict = checkpoint

    # ✅ استخراج metadata (شامل cv_summary و CI 95%)
    source_method = source_method or checkpoint.get("best_method", "unknown")
    best_lp = checkpoint.get("best_lp_accuracy", 0.0)
    ft_history = checkpoint.get("ft_history", {})
    cv_summary_saved = checkpoint.get("cv_summary", {})
    final_metrics_saved = checkpoint.get("final_metrics", {})
    config_saved = checkpoint.get("config", {})

    print(f"   Source Method:    {source_method.upper()}")
    print(f"   Best LP Acc:      {best_lp:.4f}")
    print(f"   Has CV Summary:   {'✅ Yes' if cv_summary_saved else '❌ No'}")
    print(f"   Has FT History:   {'✅ Yes' if ft_history else '❌ No'}")

    if cv_summary_saved:
        print(f"\n   📊 CI 95% found in checkpoint:")
        for k in ["accuracy", "f1_score", "f1_macro"]:
            if k in cv_summary_saved:
                s = cv_summary_saved[k]
                print(f"      {k:<20} mean={s.get('mean', 0):.4f}, "
                      f"CI95=[{s.get('ci95_low', 0):.4f}, {s.get('ci95_high', 0):.4f}]")

    # ✅ ساخت ResNet50 و load
    from src.models.resnet.resnet50 import create_resnet50
    resnet50 = create_resnet50(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
    resnet50.load_state_dict(state_dict, strict=False)
    resnet50 = resnet50.to(device)
    resnet50.eval()

    print(f"   ✅ Model loaded")

    # ============================================================
    # ✅ STEP 2: Recover CV Summary (CI 95%)
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"🔬 STEP 2: Recovering CV Summary with CI 95%")
    print(f"{'=' * 80}")

    cv_summary = None

    # ✅ راه 1: از checkpoint اصلی (بهترین!)
    if cv_summary_saved:
        print(f"   ✅ Using CV summary from main checkpoint")
        cv_summary = cv_summary_saved

        # ✅ بازسازی 'values' برای boxplot (از fold_results در cv/)
        print(f"   🔄 Trying to recover per-fold values for boxplot...")
        fold_values = {k: [] for k in ["accuracy", "balanced_accuracy", "precision",
                                       "recall", "f1_score", "f1_macro"]}

        for fold_idx in range(1, 6):
            fold_dir = os.path.join(CV_REPORTS_DIR, f"fold_{fold_idx}")
            if not os.path.exists(fold_dir):
                continue
            fold_ckpts = glob.glob(os.path.join(fold_dir, "*.pth"))
            if not fold_ckpts:
                continue
            try:
                fold_ckpt = torch.load(fold_ckpts[0], map_location="cpu")
                fm = fold_ckpt.get("final_metrics", {})
                if fm:
                    for k in fold_values.keys():
                        if k in fm:
                            fold_values[k].append(float(fm[k]))
            except Exception:
                pass

        # ✅ اضافه کردن 'values' به cv_summary
        for k, vals in fold_values.items():
            if vals and k in cv_summary:
                cv_summary[k]["values"] = vals
                print(f"      {k}: {len(vals)} fold values recovered")

    # ✅ راه 2: بازسازی از fold ها (اگه checkpoint نبود)
    if cv_summary is None:
        print(f"   ⚠️ No cv_summary in checkpoint, rebuilding from folds...")
        fold_results = []

        for fold_idx in range(1, 6):
            fold_dir = os.path.join(CV_REPORTS_DIR, f"fold_{fold_idx}")
            if not os.path.exists(fold_dir):
                continue
            fold_ckpts = glob.glob(os.path.join(fold_dir, "*.pth"))
            if not fold_ckpts:
                continue
            try:
                fold_ckpt = torch.load(fold_ckpts[0], map_location="cpu")
                fm = fold_ckpt.get("final_metrics", {})
                fold_metrics = {
                    "fold": fold_idx,
                    "accuracy": float(fm.get("accuracy", fold_ckpt.get("val_acc", 0))),
                    "balanced_accuracy": float(fm.get("balanced_accuracy", 0)),
                    "precision": float(fm.get("precision", 0)),
                    "recall": float(fm.get("recall", 0)),
                    "f1_score": float(fm.get("f1_score", 0)),
                    "f1_macro": float(fm.get("f1_macro", 0)),
                    "best_val_acc": float(fold_ckpt.get("val_acc", 0)),
                    "best_val_loss": float(fold_ckpt.get("val_loss", 0)),
                }
                fold_results.append(fold_metrics)
                print(f"      ✅ Fold {fold_idx}: Acc={fold_metrics['accuracy']:.4f}")
            except Exception as e:
                print(f"      ⚠️ Fold {fold_idx}: {e}")

        if not fold_results:
            raise RuntimeError("❌ No CV data available!")

        cv_summary = compute_cv_summary(fold_results, len(fold_results))

    # ✅ نمایش CI 95%
    print(f"\n   📊 CV Summary with CI 95%:")
    for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "f1_macro"]:
        if k in cv_summary and isinstance(cv_summary[k], dict):
            s = cv_summary[k]
            print(f"      {k:<20} {s.get('mean', 0):.4f} ± {s.get('std', 0):.4f} "
                  f"(CI95: [{s.get('ci95_low', 0):.4f}, {s.get('ci95_high', 0):.4f}])")

    # ============================================================
    # ✅ STEP 3: Load Dataset + Evaluate Model
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"📂 STEP 3: Loading dataset + evaluating model")
    print(f"{'=' * 80}")

    paths, labels = build_labeled_dataset()

    # ✅ FIX: 80% train, 20% val (Val هم به عنوان Test استفاده می‌شه)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.20, stratify=labels, random_state=42,
    )

    print(f"   Train samples: {len(train_paths)} ({len(train_paths)/len(paths)*100:.1f}%)")
    print(f"   Val samples:   {len(val_paths)} ({len(val_paths)/len(paths)*100:.1f}%)")

    val_ds = LabeledLiverDataset(
        val_paths, val_labels,
        transform=get_eval_transform(input_size),
        target_size=input_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"   Val samples: {len(val_ds)}")

    final_metrics = evaluate_model(resnet50, val_loader, device)

    print(f"\n   📊 Model Performance (re-evaluated):")
    print(f"      Accuracy:          {final_metrics['accuracy']:.4f}")
    print(f"      Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
    print(f"      Precision:         {final_metrics['precision']:.4f}")
    print(f"      Recall:            {final_metrics['recall']:.4f}")
    print(f"      F1 (weighted):     {final_metrics['f1_score']:.4f}")
    print(f"      F1 (macro):        {final_metrics['f1_macro']:.4f}")

    # ============================================================
    # ✅ STEP 4: Build Missing Plots
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"📈 STEP 4: Building missing plots")
    print(f"{'=' * 80}")

    plot_paths = {}

    # ✅ Existing plots
    for name, fname in [
        ("loss_curves", "loss_curves.png"),
        ("lr_curves", "lr_curves.png"),
        ("metrics_bar", "metrics_bar.png"),
        ("confusion_matrix", "confusion_matrix.png"),
        ("roc_curves", "roc_curves.png"),
        ("pie_chart", "pie_chart.png"),
        ("cv_summary", "cv_summary.png"),
    ]:
        p = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(p):
            plot_paths[name] = p
            print(f"   ✓ Found: {name}")
        else:
            print(f"   ⚠️ Missing: {name}")

    # ✅ Build cv_box (اگه values موجود باشه)
    try:
        p = plot_cv_box(cv_summary, os.path.join(PLOTS_DIR, "cv_box.png"))
        plot_paths["cv_box"] = p
        print(f"   ✅ Built: cv_box.png")
    except Exception as e:
        print(f"   ⚠️ cv_box failed: {e}")

    # ✅ Build ci_bars (اگه CI 95% موجود باشه)
    try:
        p = plot_ci_bars(cv_summary, os.path.join(PLOTS_DIR, "ci_bars.png"))
        plot_paths["ci_bars"] = p
        print(f"   ✅ Built: ci_bars.png (CI 95%)")
    except Exception as e:
        print(f"   ⚠️ ci_bars failed: {e}")

    print(f"\n   Total plots: {len(plot_paths)}")

    # ============================================================
    # ✅ STEP 5: Interpretability
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"🔬 STEP 5: Interpretability (LIME, SHAP, Captum)")
    print(f"{'=' * 80}")

    interp_subset = Subset(val_ds, list(range(min(20, len(val_ds)))))
    interp_loader = DataLoader(interp_subset, batch_size=1, shuffle=False)
    interp_paths = generate_all_interpretability(
        resnet50, interp_loader, INTERP_DIR, device,
    )

    print(f"\n   ✅ Interpretability:")
    for k, v in interp_paths.items():
        status = "✅" if v else "⚠️"
        print(f"      {status} {k}: {v}")

    # ============================================================
    # ✅ STEP 6: Netron + ONNX
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"🏗️  STEP 6: Netron + ONNX")
    print(f"{'=' * 80}")

    netron_paths = generate_netron_onnx(resnet50, ARCH_DIR, "ResNet50")

    print(f"   ✅ ONNX:  {netron_paths.get('onnx_path')}")
    print(f"   ✅ HTML:  {netron_paths.get('html_path')}")
    print(f"   ✅ JSON:  {netron_paths.get('info_path')}")

    # ============================================================
    # ✅ STEP 7: Generate 15-page PDF (با CI 95%)
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"📄 STEP 7: Generating 15-page PDF report")
    print(f"{'=' * 80}")

    pdf_path = os.path.join(
        RESNET50_REPORTS_DIR,
        f"ResNet50_Finetuned_from_{source_method}_Report.pdf",
    )

    generate_14page_pdf(
        output_path=pdf_path,
        config={
            "best_method": source_method,
            "best_lp_accuracy": best_lp,
            "epochs": config_saved.get("epochs", "N/A"),
            "batch_size": config_saved.get("batch_size", "N/A"),
            "input_size": input_size,
            "device": device,
            "use_mixed_precision": True,
            "clip_grad_norm": 1.0,
            "focal_gamma": 2.0,
            "pipeline_stage": "recovery_mode",
            "cv_summary_source": "checkpoint" if cv_summary_saved else "rebuilt",
        },
        metrics=final_metrics,
        cv_summary=cv_summary,
        history=ft_history,
        plot_paths=plot_paths,
        interp_paths=interp_paths,
        netron_paths=netron_paths,
        labeled_loader=val_loader,
    )

    # ============================================================
    # ✅ FINAL SUMMARY
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"🎉 RECOVERY COMPLETE!")
    print(f"{'=' * 80}")
    print(f"   Source SSL Method:  {source_method.upper()}")
    print(f"   Final Accuracy:     {final_metrics['accuracy']:.4f}")
    print(f"   Final F1:           {final_metrics['f1_score']:.4f}")

    if "accuracy" in cv_summary:
        acc_s = cv_summary["accuracy"]
        print(f"   CV Mean Accuracy:   {acc_s['mean']:.4f} ± {acc_s['std']:.4f}")
        print(f"   CV 95% CI:          [{acc_s['ci95_low']:.4f}, {acc_s['ci95_high']:.4f}]")

    print(f"   PDF Report:         {pdf_path}")
    print(f"{'=' * 80}")

    return {
        "mode": "recovery",
        "best_method": source_method,
        "best_lp_accuracy": best_lp,
        "final_model_path": model_path,
        "pdf_path": pdf_path,
        "metrics": {k: v for k, v in final_metrics.items()
                    if k not in ["y_true", "y_pred", "y_prob",
                                 "confusion_matrix", "roc_data",
                                 "classification_report"]},
        "cv_summary": cv_summary,
        "plot_paths": plot_paths,
        "interp_paths": interp_paths,
        "netron_paths": netron_paths,
    }

