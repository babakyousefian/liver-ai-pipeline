# src/pipeLine/vitWithoutFineTuning.py
# -*- coding: utf-8 -*-

"""
Vision Transformer (ViT) Without Fine-Tuning - Task 5
=====================================================
Author: Babak Yousefian

Full training pipeline for ViT with:
- Class weighting (sqrt method)
- WeightedRandomSampler (power=0.5)
- Label smoothing (0.12)
- Mixup + CutMix
- Ultrasound-specific augmentations
- Cosine scheduler + long warmup (10 epochs, needed for ViT)
- TTA at evaluation (flip + multi-crop)
- Comprehensive metrics (Acc, Precision, Recall, F1, AUC, ROC, Loss)
- Gap analysis (train vs val loss)
- Cross-validation
- Full PDF report
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

from src.models.ViT import ViT, create_vit
from src.models.ViT.utils import (
    count_parameters,
    get_device,
    get_model_size,
    set_seed,
)


# ================================================================
# ViT Wrapper
# ================================================================

class ViTWrapper:
    """ViT Wrapper for Task 5 (from scratch)."""

    def __init__(
        self,
        num_classes: int = 4,
        checkpoints_dir: str = None,
        configs_dir: str = None,
        plots_dir: str = None,
        # Architecture
        variant: str = "tiny",       # 'tiny' | 'small' | 'base' | 'custom'
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        head_dropout: float = 0.2,
        use_cls_token: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        target_size: int = 224,
        # Regularization
        label_smoothing: float = 0.12,
        use_focal_loss: bool = False,
        focal_gamma: float = 1.0,
        use_weighted_sampler: bool = True,
        sampler_power: float = 0.5,
        use_class_weights: bool = True,
        class_weight_method: str = "sqrt",
        # Augmentation
        augmentation_level: str = "basic",
        use_ultrasound_transforms: bool = True,
        # Scheduler (ViT needs longer warmup)
        scheduler_type: str = "cosine",
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
        total_epochs: int = 80,
        # TTA & Mixup
        use_tta: bool = True,
        use_mixup: bool = True,
        mixup_alpha: float = 0.2,
        mixup_start_stage: int = 4,
        use_cutmix: bool = False,
        cutmix_alpha: float = 1.0,
        # Optimizer (ViT prefers AdamW with lower LR)
        learning_rate: float = 5e-4,
        weight_decay: float = 0.05,
        # Data
        labels_csv_path: str = None,
        verbose: bool = True,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.checkpoints_dir = checkpoints_dir or "checkpoints"
        self.configs_dir = configs_dir or "configs"
        self.plots_dir = plots_dir or "plots"
        self.device = torch.device(device)
        self.scaler = (
            torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        )

        # Architecture
        self.variant = variant
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.drop_path_rate = drop_path_rate
        self.head_dropout = head_dropout
        self.use_cls_token = use_cls_token
        self.target_size = target_size

        # Regularization
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.use_weighted_sampler = use_weighted_sampler
        self.sampler_power = sampler_power
        self.use_class_weights = use_class_weights
        self.class_weight_method = class_weight_method

        # Augmentation
        self.augmentation_level = augmentation_level
        self.use_ultrasound_transforms = use_ultrasound_transforms

        # Scheduler
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.total_epochs = total_epochs

        # TTA & Mixup
        self.use_tta = use_tta
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_start_stage = mixup_start_stage
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha

        # Optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.labels_csv_path = labels_csv_path
        self.verbose = verbose

        # State
        self.model = None
        self.is_trained = False
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.train_history = []
        self.current_epoch = 0
        self.class_weights_tensor = None

        # Create directories
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        if self.verbose:
            print("\n" + "=" * 70)
            print("🚀 ViT Wrapper Initialized")
            print("=" * 70)
            print(f"   Variant:            {self.variant}")
            print(f"   Num Classes:        {self.num_classes}")
            print(f"   Device:             {self.device}")
            print(f"   Target Size:        {self.target_size}")
            print(f"   Patch Size:         {self.patch_size}")
            print(f"   Embed Dim:          {self.embed_dim}")
            print(f"   Depth / Heads:      {self.depth} / {self.num_heads}")
            print(f"   Dropout:            {self.dropout}")
            print(f"   Drop Path Rate:     {self.drop_path_rate}")
            print(f"   Head Dropout:       {self.head_dropout}")
            print(f"   Label Smoothing:    {self.label_smoothing}")
            print(f"   Sampler Power:      {self.sampler_power}")
            print(f"   Focal Loss:         {self.use_focal_loss}")
            print(f"   Mixup:              {self.use_mixup}")
            print(f"   CutMix:             {self.use_cutmix}")
            print(f"   TTA:                {self.use_tta}")
            print(f"   Warmup Epochs:      {self.warmup_epochs}")
            print(f"   Total Epochs:       {self.total_epochs}")
            print(f"   Learning Rate:      {self.learning_rate}")
            print(f"   Weight Decay:       {self.weight_decay}")
            print("=" * 70 + "\n")

    # ============================================================
    # Model Building
    # ============================================================

    def build_model(self) -> nn.Module:
        if self.variant in ("tiny", "small", "base"):
            # Use factory with variant
            self.model = create_vit(
                num_classes=self.num_classes,
                variant=self.variant,
                pretrained=False,
                verbose=False,
                img_size=self.img_size,
                patch_size=self.patch_size,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                drop_path_rate=self.drop_path_rate,
                head_dropout=self.head_dropout,
                use_cls_token=self.use_cls_token,
            )
        else:
            # Custom config
            self.model = ViT(
                num_classes=self.num_classes,
                img_size=self.img_size,
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                drop_path_rate=self.drop_path_rate,
                head_dropout=self.head_dropout,
                use_cls_token=self.use_cls_token,
                verbose=False,
            )

        self.model = self.model.to(self.device)
        if self.verbose:
            print(f"✅ ViT built on {self.device}")
        return self.model

    def print_summary(self) -> None:
        if self.model is None:
            self.build_model()
        params = count_parameters(self.model)
        size = get_model_size(self.model)
        print("\n" + "=" * 70)
        print("📊 ViT MODEL SUMMARY")
        print("=" * 70)
        print(f"   Total Parameters:     {params['total']:,}")
        print(f"   Trainable Parameters: {params['trainable']:,}")
        print(f"   Model Size:           {size['mb']:.2f} MB")
        print(f"   Device:               {self.device}")
        print("=" * 70 + "\n")

    # ============================================================
    # Loss / Optimizer / Scheduler
    # ============================================================

    def _compute_class_weights(self, train_labels: np.ndarray) -> torch.Tensor:
        """Compute class weights for loss function."""
        class_counts = np.bincount(train_labels, minlength=self.num_classes).astype(float)
        class_counts = np.maximum(class_counts, 1.0)

        if self.class_weight_method == "sqrt":
            weights = 1.0 / np.sqrt(class_counts)
        elif self.class_weight_method == "inverse":
            weights = 1.0 / class_counts
        elif self.class_weight_method == "balanced":
            weights = len(train_labels) / (self.num_classes * class_counts)
        else:
            weights = np.ones(self.num_classes, dtype=float)

        # Normalize so mean = 1
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _get_criterion(self) -> nn.Module:
        if self.use_focal_loss:
            class FocalLoss(nn.Module):
                def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
                    super().__init__()
                    self.gamma = gamma
                    self.weight = weight
                    self.label_smoothing = label_smoothing

                def forward(self, inputs, targets):
                    ce = nn.functional.cross_entropy(
                        inputs,
                        targets,
                        weight=self.weight,
                        reduction="none",
                        label_smoothing=self.label_smoothing,
                    )
                    pt = torch.exp(-ce)
                    loss = ((1 - pt) ** self.gamma) * ce
                    return loss.mean()

            if self.verbose:
                print(
                    f"   ✅ Using Focal Loss (gamma={self.focal_gamma}, "
                    f"weights={'yes' if self.use_class_weights else 'no'})"
                )
            return FocalLoss(
                gamma=self.focal_gamma,
                weight=self.class_weights_tensor,
                label_smoothing=self.label_smoothing,
            )
        else:
            if self.verbose:
                print(
                    f"   ✅ Using CrossEntropy "
                    f"(label_smoothing={self.label_smoothing}, "
                    f"weights={'yes' if self.use_class_weights else 'no'})"
                )
            return nn.CrossEntropyLoss(
                weight=self.class_weights_tensor,
                label_smoothing=self.label_smoothing,
            )

    def _get_optimizer_scheduler(self):
        # ✅ ViT: AdamW with weight_decay=0.05, lower LR (5e-4)
        # Separate weight decay: no decay for bias/norm/pos_embed/cls_token
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or "bias" in name or "norm" in name or "pos_embed" in name or "cls_token" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
        )

        # ✅ ViT: warmup + cosine (ViT needs longer warmup)
        if self.scheduler_type == "warmup_cosine" or self.warmup_epochs > 0:
            def warmup_cosine(epoch):
                if epoch < self.warmup_epochs:
                    return (epoch + 1) / max(1, self.warmup_epochs)
                progress = (epoch - self.warmup_epochs) / max(
                    1, self.total_epochs - self.warmup_epochs
                )
                return self.min_lr / self.learning_rate + (
                    1 - self.min_lr / self.learning_rate
                ) * 0.5 * (1 + np.cos(np.pi * progress))

            scheduler = LambdaLR(optimizer, warmup_cosine)
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.total_epochs, eta_min=self.min_lr
            )

        return optimizer, scheduler

    # ============================================================
    # Mixup / CutMix
    # ============================================================

    def _mixup_data(self, x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _cutmix_data(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size, _, H, W = x.size()
        index = torch.randperm(batch_size, device=x.device)

        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cy = np.random.randint(H)
        cx = np.random.randint(W)

        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)

        mixed_x = x.clone()
        mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

        lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _mix_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # ============================================================
    # Training
    # ============================================================

    def train_full(
        self,
        train_loader,
        val_loader,
        epochs: int = 80,
        save_best: bool = True,
    ) -> Dict[str, Any]:
        if self.model is None:
            self.build_model()

        # ✅ Compute class weights from train labels
        train_labels_array = np.array(
            [train_loader.dataset[i][1] for i in range(len(train_loader.dataset))]
        )
        self.class_weights_tensor = self._compute_class_weights(train_labels_array)

        if self.verbose:
            print(f"\n📊 Class Weights (method={self.class_weight_method}):")
            for i, w in enumerate(self.class_weights_tensor.cpu().numpy()):
                print(f"   Class {i}: {w:.4f}")
            print()

        criterion = self._get_criterion()
        optimizer, scheduler = self._get_optimizer_scheduler()

        best_acc = 0.0
        best_epoch = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "gap_loss": [],
            "gap_acc": [],
        }

        patience_counter = 0
        patience_limit = 25  # ViT needs patience

        print("\n" + "=" * 70)
        print(f"🚀 TRAINING ViT FOR {epochs} EPOCHS")
        print("=" * 70)

        for epoch in range(epochs):
            epoch_start = time.time()
            self.current_epoch = epoch + 1

            # =====================================================
            # TRAIN
            # =====================================================
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_preds_all = []
            train_labels_all = []

            # ✅ Decide augmentation strategy
            use_mixup_this_epoch = (
                self.use_mixup and self.current_epoch >= self.mixup_start_stage
            )
            use_cutmix_this_epoch = (
                self.use_cutmix and self.current_epoch >= self.mixup_start_stage
            )

            train_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", ncols=100
            )
            for batch_idx, (inputs, targets) in enumerate(train_bar):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # Mix strategy (mixup OR cutmix, not both)
                if use_cutmix_this_epoch:
                    mixed_inputs, y_a, y_b, lam = self._cutmix_data(
                        inputs, targets, alpha=self.cutmix_alpha
                    )
                    with torch.amp.autocast(
                        "cuda", enabled=self.scaler is not None
                    ):
                        outputs, _ = self.model(mixed_inputs)
                        loss = self._mix_criterion(
                            criterion, outputs, y_a, y_b, lam
                        )
                elif use_mixup_this_epoch:
                    mixed_inputs, y_a, y_b, lam = self._mixup_data(
                        inputs, targets, alpha=self.mixup_alpha
                    )
                    with torch.amp.autocast(
                        "cuda", enabled=self.scaler is not None
                    ):
                        outputs, _ = self.model(mixed_inputs)
                        loss = self._mix_criterion(
                            criterion, outputs, y_a, y_b, lam
                        )
                else:
                    with torch.amp.autocast(
                        "cuda", enabled=self.scaler is not None
                    ):
                        outputs, _ = self.model(inputs)
                        loss = criterion(outputs, targets)

                # Backward
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss.item()
                with torch.no_grad():
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()
                    train_preds_all.extend(predicted.cpu().numpy().tolist())
                    train_labels_all.extend(targets.cpu().numpy().tolist())

                train_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{train_correct / max(1, train_total):.4f}",
                    }
                )

            train_loss /= max(1, len(train_loader))
            train_acc = train_correct / max(1, train_total)

            # Train metrics
            train_preds_arr = np.array(train_preds_all)
            train_labels_arr = np.array(train_labels_all)
            try:
                train_precision = precision_score(
                    train_labels_arr, train_preds_arr, average="weighted", zero_division=0
                )
                train_recall = recall_score(
                    train_labels_arr, train_preds_arr, average="weighted", zero_division=0
                )
                train_f1 = f1_score(
                    train_labels_arr, train_preds_arr, average="weighted", zero_division=0
                )
            except Exception:
                train_precision = train_recall = train_f1 = 0.0

            # =====================================================
            # VALIDATION
            # =====================================================
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds_all = []
            val_labels_all = []

            with torch.no_grad():
                val_bar = tqdm(
                    val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]  ", ncols=100
                )
                for inputs, targets in val_bar:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    outputs, _ = self.model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    val_preds_all.extend(predicted.cpu().numpy().tolist())
                    val_labels_all.extend(targets.cpu().numpy().tolist())

                    val_bar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{val_correct / max(1, val_total):.4f}",
                        }
                    )

            val_loss /= max(1, len(val_loader))
            val_acc = val_correct / max(1, val_total)

            # Val metrics
            val_preds_arr = np.array(val_preds_all)
            val_labels_arr = np.array(val_labels_all)
            try:
                val_precision = precision_score(
                    val_labels_arr, val_preds_arr, average="weighted", zero_division=0
                )
                val_recall = recall_score(
                    val_labels_arr, val_preds_arr, average="weighted", zero_division=0
                )
                val_f1 = f1_score(
                    val_labels_arr, val_preds_arr, average="weighted", zero_division=0
                )
            except Exception:
                val_precision = val_recall = val_f1 = 0.0

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # ⭐ Gap tracking
            gap_loss = val_loss - train_loss
            gap_acc = train_acc - val_acc

            # Save history
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["train_acc"].append(float(train_acc))
            history["val_acc"].append(float(val_acc))
            history["learning_rates"].append(float(current_lr))
            history["train_precision"].append(float(train_precision))
            history["train_recall"].append(float(train_recall))
            history["train_f1"].append(float(train_f1))
            history["val_precision"].append(float(val_precision))
            history["val_recall"].append(float(val_recall))
            history["val_f1"].append(float(val_f1))
            history["gap_loss"].append(float(gap_loss))
            history["gap_acc"].append(float(gap_acc))

            epoch_time = time.time() - epoch_start

            print(
                f"\n📊 Epoch {epoch + 1}/{epochs} | "
                f"Train L: {train_loss:.4f} | Train A: {train_acc:.4f} | "
                f"Val L: {val_loss:.4f} | Val A: {val_acc:.4f} | "
                f"Gap(L): {gap_loss:+.4f} | Gap(A): {gap_acc:+.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                self.best_accuracy = best_acc
                self.best_epoch = best_epoch
                patience_counter = 0

                if save_best:
                    self._save_checkpoint(
                        "visio_transformer_best.pth",
                        epoch,
                        optimizer,
                        scheduler,
                        val_acc,
                        val_loss,
                        train_acc,
                        train_loss,
                    )
                    print(f"   ✅ Best model saved! (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"\n🛑 Early Stopping at epoch {epoch + 1}")
                    print(f"   Best Val Acc: {best_acc:.4f} at epoch {best_epoch}")
                    break

            # Save last every epoch
            self._save_checkpoint(
                "visio_transformer_last.pth",
                epoch,
                optimizer,
                scheduler,
                val_acc,
                val_loss,
                train_acc,
                train_loss,
            )

        # Save final
        self._save_checkpoint(
            "visio_transformer_final.pth",
            epochs - 1,
            optimizer,
            scheduler,
            val_acc,
            val_loss,
            train_acc,
            train_loss,
        )

        self.is_trained = True
        self.train_history = history
        self._save_config()

        print("\n" + "=" * 70)
        print(f"✅ TRAINING COMPLETE!")
        print(f"   Best Val Acc: {best_acc:.4f} (Epoch {best_epoch})")
        print("=" * 70)

        return {
            "best_accuracy": best_acc,
            "best_epoch": best_epoch,
            "history": history,
        }

    # ============================================================
    # Checkpoint / Config Save
    # ============================================================

    def _save_checkpoint(
        self,
        filename,
        epoch,
        optimizer,
        scheduler,
        val_acc,
        val_loss,
        train_acc,
        train_loss,
    ):
        path = os.path.join(self.checkpoints_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "gap_loss": float(val_loss - train_loss),
                "gap_acc": float(train_acc - val_acc),
                "best_accuracy": self.best_accuracy,
                "best_epoch": self.best_epoch,
                "num_classes": self.num_classes,
                "config": {
                    "variant": self.variant,
                    "embed_dim": self.embed_dim,
                    "depth": self.depth,
                    "num_heads": self.num_heads,
                    "patch_size": self.patch_size,
                    "dropout": self.dropout,
                    "drop_path_rate": self.drop_path_rate,
                },
            },
            path,
        )

    def _save_config(self):
        config = {
            "model": "VisionTransformer",
            "variant": self.variant,
            "implementation": "from_scratch_modernized",
            "class_names": ["NL", "G1", "G2", "G3"],
            "num_classes": self.num_classes,
            "target_size": self.target_size,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "drop_path_rate": self.drop_path_rate,
            "head_dropout": self.head_dropout,
            "use_cls_token": self.use_cls_token,
            "label_smoothing": self.label_smoothing,
            "use_focal_loss": self.use_focal_loss,
            "focal_gamma": self.focal_gamma,
            "use_weighted_sampler": self.use_weighted_sampler,
            "sampler_power": self.sampler_power,
            "use_class_weights": self.use_class_weights,
            "class_weight_method": self.class_weight_method,
            "augmentation_level": self.augmentation_level,
            "use_ultrasound_transforms": self.use_ultrasound_transforms,
            "scheduler_type": self.scheduler_type,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "use_tta": self.use_tta,
            "use_mixup": self.use_mixup,
            "mixup_alpha": self.mixup_alpha,
            "mixup_start_stage": self.mixup_start_stage,
            "use_cutmix": self.use_cutmix,
            "cutmix_alpha": self.cutmix_alpha,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "device": str(self.device),
            "best_accuracy": float(self.best_accuracy),
            "best_epoch": int(self.best_epoch),
            "created_at": datetime.now().isoformat(),
        }
        path = os.path.join(self.configs_dir, "config_visio_transformer.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        if self.verbose:
            print(f"✅ Config saved: {path}")

    def load_checkpoint(self, path: str):
        if self.model is None:
            self.build_model()
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.is_trained = True
        print(f"✅ Loaded: {path}")
        print(f"   Best Accuracy: {self.best_accuracy:.4f} (Epoch {self.best_epoch})")

    def save_model(self, filename: str = "visio_transformer_final.pth") -> str:
        if self.model is None:
            raise ValueError("Model not built.")
        path = os.path.join(self.checkpoints_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_classes": self.num_classes,
                "best_accuracy": self.best_accuracy,
                "best_epoch": self.best_epoch,
            },
            path,
        )
        print(f"✅ Model saved: {path}")
        return path

    # ============================================================
    # Comprehensive Evaluation (with TTA)
    # ============================================================

    def evaluate_comprehensive(
        self,
        test_loader,
        generate_report: bool = True,
        cv_folds: int = 0,
        report_name: str = "ViT_Task5_Report.pdf",
    ) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not built.")

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        print("\n" + "=" * 70)
        print("📊 EVALUATING ON TEST SET")
        print("=" * 70)

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Test", ncols=100):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs, _ = self.model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)

                all_labels.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # ✅ TTA: average of original + horizontal flip
        if self.use_tta:
            print("   ✅ Applying TTA (horizontal flip)")
            all_probs_tta = []
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(self.device)
                    inputs_flipped = torch.flip(inputs, dims=[3])
                    outputs, _ = self.model(inputs_flipped)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs_tta.extend(probs.cpu().numpy())
            all_probs = (all_probs + np.array(all_probs_tta)) / 2
            all_preds = all_probs.argmax(axis=1)

        test_acc = accuracy_score(all_labels, all_preds)
        test_precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        test_recall = recall_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        test_f1 = f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        try:
            test_auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="weighted"
            )
        except Exception as e:
            print(f"   ⚠️ AUC failed: {e}")
            test_auc = 0.0

        test_loss = total_loss / max(1, len(test_loader))

        print(f"\n✅ Test Accuracy:  {test_acc:.4f}")
        print(f"✅ Test Precision: {test_precision:.4f}")
        print(f"✅ Test Recall:    {test_recall:.4f}")
        print(f"✅ Test F1-Score:  {test_f1:.4f}")
        print(f"✅ Test AUC:       {test_auc:.4f}")
        print(f"✅ Test Loss:      {test_loss:.4f}")
        print("=" * 70)

        cm = confusion_matrix(all_labels, all_preds)
        class_names = ["NL", "G1", "G2", "G3"]

        # Generate plots
        self._generate_plots(all_labels, all_preds, all_probs, cm, class_names)

        # Metrics
        metrics = {
            "test_accuracy": float(test_acc),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1_score": float(test_f1),
            "test_auc": float(test_auc),
            "test_loss": float(test_loss),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                all_labels, all_preds, target_names=class_names, digits=4, zero_division=0
            ),
            "num_samples": int(len(all_labels)),
            "best_accuracy": float(self.best_accuracy),
            "best_epoch": int(self.best_epoch),
        }

        # Save metrics.json
        reports_dir = os.path.join(os.path.dirname(self.plots_dir), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        metrics_path = os.path.join(reports_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"✅ Metrics saved: {metrics_path}")

        # Save task5_report.json
        task5_report = {
            "model_info": {
                "model": "VisionTransformer",
                "variant": self.variant,
                "version": "5.0.0",
                "implementation": "from_scratch_modernized",
                "class_names": class_names,
                "num_classes": self.num_classes,
                "device": str(self.device),
                "checkpoints_dir": self.checkpoints_dir,
                "reports_dir": reports_dir,
                "configs_dir": self.configs_dir,
                "plots_dir": self.plots_dir,
                "created_at": datetime.now().isoformat(),
            },
            "training_summary": {
                "best_validation_accuracy": float(self.best_accuracy),
                "best_epoch": int(self.best_epoch),
                "total_epochs": len(self.train_history.get("train_loss", []))
                if isinstance(self.train_history, dict)
                else 0,
                "final_train_loss": float(self.train_history["train_loss"][-1])
                if isinstance(self.train_history, dict) and self.train_history.get("train_loss")
                else 0.0,
                "final_val_loss": float(self.train_history["val_loss"][-1])
                if isinstance(self.train_history, dict) and self.train_history.get("val_loss")
                else 0.0,
                "final_gap_loss": float(self.train_history["gap_loss"][-1])
                if isinstance(self.train_history, dict) and self.train_history.get("gap_loss")
                else 0.0,
                "final_gap_acc": float(self.train_history["gap_acc"][-1])
                if isinstance(self.train_history, dict) and self.train_history.get("gap_acc")
                else 0.0,
            },
            "performance_metrics": {
                "test_accuracy": float(test_acc),
                "test_precision": float(test_precision),
                "test_recall": float(test_recall),
                "test_f1_score": float(test_f1),
                "test_auc": float(test_auc),
                "test_loss": float(test_loss),
                "confusion_matrix": cm.tolist(),
            },
            "config": {
                "variant": self.variant,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "patch_size": self.patch_size,
                "dropout": self.dropout,
                "drop_path_rate": self.drop_path_rate,
                "head_dropout": self.head_dropout,
                "label_smoothing": self.label_smoothing,
                "sampler_power": self.sampler_power,
                "use_focal_loss": self.use_focal_loss,
                "augmentation_level": self.augmentation_level,
                "use_ultrasound_transforms": self.use_ultrasound_transforms,
                "scheduler_type": self.scheduler_type,
                "warmup_epochs": self.warmup_epochs,
                "total_epochs": self.total_epochs,
                "use_tta": self.use_tta,
                "use_mixup": self.use_mixup,
                "use_cutmix": self.use_cutmix,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "target_size": self.target_size,
            },
            "generated_at": datetime.now().isoformat(),
        }
        task5_path = os.path.join(reports_dir, "visio_transformer_task5_report.json")
        with open(task5_path, "w") as f:
            json.dump(task5_report, f, indent=4)
        print(f"✅ Task5 report saved: {task5_path}")

        if generate_report:
            self._generate_pdf_report(metrics, report_name)

        return {
            "metrics": metrics,
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist(),
            "probabilities": all_probs.tolist(),
            "task5_report": task5_report,
        }

    # ============================================================
    # Plots
    # ============================================================

    def _generate_plots(self, y_true, y_pred, y_prob, cm, class_names):
        os.makedirs(self.plots_dir, exist_ok=True)
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]

        # 1) Confusion Matrix (Normalized)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            square=True,
            annot_kws={"size": 13, "fontweight": "bold"},
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_ylabel("True", fontsize=12, fontweight="bold")
        ax.set_title(
            "ViT — Confusion Matrix (Normalized)", fontsize=14, fontweight="bold"
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "confusion_matrix.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        # 2) Raw Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            square=True,
            annot_kws={"size": 13, "fontweight": "bold"},
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_ylabel("True", fontsize=12, fontweight="bold")
        ax.set_title(
            "ViT — Confusion Matrix (Raw Counts)", fontsize=14, fontweight="bold"
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "confusion_matrix_raw.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        # 3) ROC Curves
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        for i, name in enumerate(class_names):
            y_bin = (y_true == i).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, linewidth=2.5, color=colors[i], label=f"{name} (AUC = {roc_auc:.4f})"
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.6, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax.set_title("ViT — ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "roc_curves.png"), dpi=200, bbox_inches="tight"
        )
        plt.close()

        # 4) Metrics Bar
        metrics_vals = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        bars = ax.bar(
            metrics_vals.keys(),
            metrics_vals.values(),
            color=colors[:4],
            edgecolor="black",
        )
        for bar, v in zip(bars, metrics_vals.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("ViT — Performance Metrics (Test Set)", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "metrics_bar.png"), dpi=200, bbox_inches="tight"
        )
        plt.close()

        # 5) Per-Class Metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))

        x = np.arange(len(class_names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        ax.bar(x - width, per_class_precision, width, label="Precision", color="#3498db")
        ax.bar(x, per_class_recall, width, label="Recall", color="#2ecc71")
        ax.bar(x + width, per_class_f1, width, label="F1-Score", color="#9b59b6")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("ViT — Per-Class Metrics", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "per_class_metrics.png"), dpi=200, bbox_inches="tight"
        )
        plt.close()

        # 6) Class Distribution Pie
        unique, counts = np.unique(y_true, return_counts=True)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        ax.pie(
            counts,
            labels=[class_names[i] for i in unique],
            autopct="%1.1f%%",
            colors=colors[: len(unique)],
            startangle=90,
            wedgeprops={"edgecolor": "black", "linewidth": 1.5},
        )
        ax.set_title("ViT — Test Set Class Distribution", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "class_distribution_pie.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        # 7) Predictions Distribution
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        full_pred = np.zeros(len(class_names))
        full_true = np.zeros(len(class_names))
        for u, c in zip(unique_pred, counts_pred):
            full_pred[u] = c
        for u, c in zip(unique_true, counts_true):
            full_true[u] = c
        x = np.arange(len(class_names))
        width = 0.35
        ax.bar(x - width / 2, full_true, width, label="True", color="#3498db")
        ax.bar(x + width / 2, full_pred, width, label="Predicted", color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax.set_title("ViT — True vs Predicted Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "predictions_distribution.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        # ============================================================
        # Training History Plots
        # ============================================================
        if isinstance(self.train_history, dict) and self.train_history.get("train_loss"):
            ep = range(1, len(self.train_history["train_loss"]) + 1)

            # 8) Loss History
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            ax.plot(ep, self.train_history["train_loss"], "o-", color="#e74c3c", label="Train", linewidth=2)
            ax.plot(ep, self.train_history["val_loss"], "s-", color="#3498db", label="Val", linewidth=2)
            ax.set_xlabel("Epoch", fontweight="bold")
            ax.set_ylabel("Loss", fontweight="bold")
            ax.set_title("ViT — Loss History (Train vs Val)", fontweight="bold", fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "01_loss_history.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # 9) Accuracy History
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            ax.plot(ep, self.train_history["train_acc"], "o-", color="#e74c3c", label="Train", linewidth=2)
            ax.plot(ep, self.train_history["val_acc"], "s-", color="#2ecc71", label="Val", linewidth=2)
            ax.set_xlabel("Epoch", fontweight="bold")
            ax.set_ylabel("Accuracy", fontweight="bold")
            ax.set_title("ViT — Accuracy History (Train vs Val)", fontweight="bold", fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "02_accuracy_history.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # 10) ⭐ Gap Analysis (Loss)
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            gaps = self.train_history["gap_loss"]
            gap_colors = ["#e74c3c" if g > 0.05 else "#2ecc71" if g < 0.02 else "#f39c12" for g in gaps]
            ax.bar(ep, gaps, color=gap_colors, alpha=0.75, edgecolor="black", linewidth=0.5)
            ax.axhline(y=0, color="black", linewidth=1.5)
            ax.axhline(
                y=np.mean(gaps),
                color="#f39c12",
                linewidth=2,
                linestyle="--",
                label=f"Avg Gap: {np.mean(gaps):+.4f}",
            )
            ax.set_xlabel("Epoch", fontweight="bold")
            ax.set_ylabel("Gap (Val Loss - Train Loss)", fontweight="bold")
            ax.set_title(
                "ViT — Gap Analysis (Loss): Overfit/Underfit Indicator",
                fontweight="bold",
                fontsize=14,
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "03_gap_loss.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # 11) ⭐ Gap Analysis (Accuracy)
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            gaps_acc = self.train_history["gap_acc"]
            gap_colors_acc = ["#e74c3c" if g > 0.05 else "#2ecc71" if g < 0.02 else "#f39c12" for g in gaps_acc]
            ax.bar(ep, gaps_acc, color=gap_colors_acc, alpha=0.75, edgecolor="black", linewidth=0.5)
            ax.axhline(y=0, color="black", linewidth=1.5)
            ax.axhline(
                y=np.mean(gaps_acc),
                color="#f39c12",
                linewidth=2,
                linestyle="--",
                label=f"Avg Gap: {np.mean(gaps_acc):+.4f}",
            )
            ax.set_xlabel("Epoch", fontweight="bold")
            ax.set_ylabel("Gap (Train Acc - Val Acc)", fontweight="bold")
            ax.set_title(
                "ViT — Gap Analysis (Accuracy): Overfit/Underfit Indicator",
                fontweight="bold",
                fontsize=14,
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "04_gap_accuracy.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # 12) Combined Loss + Gap
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=200)
            axes[0].plot(ep, self.train_history["train_loss"], "o-", color="#3498db", linewidth=2.5, label="Train")
            axes[0].plot(ep, self.train_history["val_loss"], "s-", color="#e74c3c", linewidth=2.5, label="Val")
            axes[0].fill_between(
                ep,
                self.train_history["train_loss"],
                self.train_history["val_loss"],
                alpha=0.2,
                color="#f39c12",
                label="Gap",
            )
            axes[0].set_xlabel("Epoch", fontweight="bold")
            axes[0].set_ylabel("Loss", fontweight="bold")
            axes[0].set_title("Train vs Validation Loss — Gap Analysis", fontweight="bold")
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            axes[1].bar(ep, gaps, color=gap_colors, alpha=0.75, edgecolor="black", linewidth=0.5)
            axes[1].axhline(y=0, color="black", linewidth=1.5)
            axes[1].axhline(
                y=np.mean(gaps),
                color="#f39c12",
                linewidth=2,
                linestyle="--",
                label=f"Avg Gap: {np.mean(gaps):+.4f}",
            )
            axes[1].set_xlabel("Epoch", fontweight="bold")
            axes[1].set_ylabel("Gap (Val - Train)", fontweight="bold")
            axes[1].set_title("Overfitting / Underfitting Indicator", fontweight="bold")
            axes[1].legend()
            axes[1].grid(alpha=0.3, axis="y")
            plt.suptitle("ViT — Loss & Gap Analysis", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "05_loss_and_gap.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # 13) LR Schedule
            fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
            ax.plot(ep, self.train_history["learning_rates"], "o-", color="#9b59b6", linewidth=2.5, markersize=4)
            ax.set_xlabel("Epoch", fontweight="bold")
            ax.set_ylabel("Learning Rate (log scale)", fontweight="bold")
            ax.set_yscale("log")
            ax.set_title("ViT — Learning Rate Schedule (Warmup + Cosine)", fontweight="bold", fontsize=14)
            ax.grid(alpha=0.3, which="both", linestyle="--")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "06_lr_schedule.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # 14) Metrics History
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)
            axes[0].plot(ep, self.train_history["train_precision"], "o-", color="#3498db", label="Train", linewidth=2)
            axes[0].plot(ep, self.train_history["val_precision"], "s-", color="#e74c3c", label="Val", linewidth=2)
            axes[0].set_xlabel("Epoch", fontweight="bold")
            axes[0].set_ylabel("Precision", fontweight="bold")
            axes[0].set_title("Precision History", fontweight="bold")
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            axes[1].plot(ep, self.train_history["train_recall"], "o-", color="#3498db", label="Train", linewidth=2)
            axes[1].plot(ep, self.train_history["val_recall"], "s-", color="#e74c3c", label="Val", linewidth=2)
            axes[1].set_xlabel("Epoch", fontweight="bold")
            axes[1].set_ylabel("Recall", fontweight="bold")
            axes[1].set_title("Recall History", fontweight="bold")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            axes[2].plot(ep, self.train_history["train_f1"], "o-", color="#3498db", label="Train", linewidth=2)
            axes[2].plot(ep, self.train_history["val_f1"], "s-", color="#e74c3c", label="Val", linewidth=2)
            axes[2].set_xlabel("Epoch", fontweight="bold")
            axes[2].set_ylabel("F1-Score", fontweight="bold")
            axes[2].set_title("F1-Score History", fontweight="bold")
            axes[2].legend()
            axes[2].grid(alpha=0.3)
            plt.suptitle("ViT — Training Metrics History", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "07_metrics_history.png"), dpi=200, bbox_inches="tight")
            plt.close()

        print(f"✅ Plots saved to: {self.plots_dir}")

    # ============================================================
    # PDF Report
    # ============================================================

    def _generate_pdf_report(self, metrics, report_name):
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("⚠️ PyMuPDF not installed. Skipping PDF generation.")
            return

        reports_dir = os.path.join(os.path.dirname(self.plots_dir), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        pdf_path = os.path.join(reports_dir, report_name)

        doc = fitz.open()

        # Page 1: Cover
        page = doc.new_page(width=595, height=842)
        page.draw_rect(fitz.Rect(0, 0, 595, 200), color=None, fill=(0.1, 0.1, 0.18), width=0)
        page.insert_text(
            (50, 80), "Vision Transformer", fontsize=30, fontname="helvetica-bold", color=(0, 0.83, 1)
        )
        page.insert_text(
            (50, 130),
            "Task 5 — From Scratch (Modernized)",
            fontsize=18,
            fontname="helvetica",
            color=(1, 1, 1),
        )
        page.insert_text(
            (50, 165),
            "Liver Ultrasound Classification (NL / G1 / G2 / G3)",
            fontsize=12,
            fontname="helvetica",
            color=(0.65, 0.75, 0.85),
        )

        acc = metrics["test_accuracy"]
        page.draw_rect(
            fitz.Rect(50, 230, 545, 320),
            color=(0, 0.83, 1),
            fill=(0.9, 0.97, 1),
            width=2,
        )
        page.insert_text(
            (70, 265),
            "🎯 TEST ACCURACY",
            fontsize=14,
            fontname="helvetica-bold",
            color=(0.1, 0.1, 0.2),
        )
        page.insert_text(
            (70, 305),
            f"{acc * 100:.2f}%",
            fontsize=32,
            fontname="helvetica-bold",
            color=(0, 0.55, 0.85),
        )

        info = [
            ("Model", f"ViT-{self.variant.capitalize()} (from scratch)"),
            ("Total Params", f"{count_parameters(self.model)['total']:,}"),
            ("Test Samples", str(metrics["num_samples"])),
            ("Best Epoch", str(metrics["best_epoch"])),
            ("Best Val Acc", f"{metrics['best_accuracy']:.4f}"),
            ("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Author", "Babak Yousefian"),
        ]
        y = 370
        for key, val in info:
            page.draw_rect(
                fitz.Rect(50, y, 545, y + 28), color=None, fill=(0.97, 0.98, 1)
            )
            page.insert_text(
                (60, y + 19), key, fontsize=11, fontname="helvetica-bold", color=(0.2, 0.2, 0.3)
            )
            page.insert_text(
                (200, y + 19), val, fontsize=11, fontname="helvetica", color=(0.1, 0.1, 0.2)
            )
            y += 32

        # Page 2: Test Metrics
        page = doc.new_page(width=595, height=842)
        page.insert_text(
            (50, 50),
            "TEST METRICS",
            fontsize=18,
            fontname="helvetica-bold",
            color=(0, 0.55, 0.85),
        )
        page.draw_line(
            fitz.Point(50, 62), fitz.Point(545, 62), color=(0, 0.83, 1), width=2
        )

        metrics_list = [
            ("Accuracy", metrics["test_accuracy"]),
            ("Precision", metrics["test_precision"]),
            ("Recall", metrics["test_recall"]),
            ("F1-Score", metrics["test_f1_score"]),
            ("AUC", metrics["test_auc"]),
            ("Loss", metrics["test_loss"]),
        ]
        y = 100
        for label, val in metrics_list:
            page.draw_rect(
                fitz.Rect(50, y, 545, y + 26),
                color=(0.8, 0.85, 0.9),
                fill=(0.97, 0.98, 1),
                width=0.5,
            )
            page.insert_text(
                (60, y + 18), label, fontsize=11, fontname="helvetica", color=(0.15, 0.15, 0.2)
            )
            page.insert_text(
                (400, y + 18),
                f"{val:.4f}",
                fontsize=12,
                fontname="helvetica-bold",
                color=(0, 0.55, 0.85),
            )
            y += 30

        # Page 3: Gap Analysis
        if isinstance(self.train_history, dict) and self.train_history.get("gap_loss"):
            page = doc.new_page(width=595, height=842)
            page.insert_text(
                (50, 50),
                "GAP ANALYSIS (TRAIN vs VAL)",
                fontsize=18,
                fontname="helvetica-bold",
                color=(0, 0.55, 0.85),
            )
            page.draw_line(
                fitz.Point(50, 62), fitz.Point(545, 62), color=(0, 0.83, 1), width=2
            )

            final_train_loss = self.train_history["train_loss"][-1]
            final_val_loss = self.train_history["val_loss"][-1]
            final_gap_loss = self.train_history["gap_loss"][-1]
            final_train_acc = self.train_history["train_acc"][-1]
            final_val_acc = self.train_history["val_acc"][-1]
            final_gap_acc = self.train_history["gap_acc"][-1]
            avg_gap_loss = float(np.mean(self.train_history["gap_loss"]))
            avg_gap_acc = float(np.mean(self.train_history["gap_acc"]))

            gap_items = [
                ("Final Train Loss", f"{final_train_loss:.4f}"),
                ("Final Val Loss", f"{final_val_loss:.4f}"),
                ("Final Gap (Loss)", f"{final_gap_loss:+.4f}"),
                ("Avg Gap (Loss)", f"{avg_gap_loss:+.4f}"),
                ("Final Train Acc", f"{final_train_acc:.4f}"),
                ("Final Val Acc", f"{final_val_acc:.4f}"),
                ("Final Gap (Acc)", f"{final_gap_acc:+.4f}"),
                ("Avg Gap (Acc)", f"{avg_gap_acc:+.4f}"),
            ]
            y = 100
            for label, val in gap_items:
                page.draw_rect(
                    fitz.Rect(50, y, 545, y + 26),
                    color=(0.8, 0.85, 0.9),
                    fill=(0.97, 0.98, 1),
                    width=0.5,
                )
                page.insert_text(
                    (60, y + 18), label, fontsize=11, fontname="helvetica", color=(0.15, 0.15, 0.2)
                )
                page.insert_text(
                    (400, y + 18),
                    val,
                    fontsize=12,
                    fontname="helvetica-bold",
                    color=(0, 0.55, 0.85),
                )
                y += 30

        # Pages: Plots
        plot_files = [
            ("confusion_matrix.png", "CONFUSION MATRIX (NORMALIZED)"),
            ("confusion_matrix_raw.png", "CONFUSION MATRIX (RAW COUNTS)"),
            ("roc_curves.png", "ROC CURVES (ONE-VS-REST)"),
            ("metrics_bar.png", "PERFORMANCE METRICS"),
            ("per_class_metrics.png", "PER-CLASS METRICS"),
            ("class_distribution_pie.png", "CLASS DISTRIBUTION"),
            ("predictions_distribution.png", "TRUE vs PREDICTED DISTRIBUTION"),
            ("01_loss_history.png", "LOSS HISTORY (TRAIN vs VAL)"),
            ("02_accuracy_history.png", "ACCURACY HISTORY (TRAIN vs VAL)"),
            ("03_gap_loss.png", "GAP ANALYSIS (LOSS)"),
            ("04_gap_accuracy.png", "GAP ANALYSIS (ACCURACY)"),
            ("05_loss_and_gap.png", "LOSS & GAP ANALYSIS"),
            ("06_lr_schedule.png", "LEARNING RATE SCHEDULE"),
            ("07_metrics_history.png", "TRAINING METRICS HISTORY"),
        ]

        for plot_file, title in plot_files:
            plot_path = os.path.join(self.plots_dir, plot_file)
            if os.path.exists(plot_path):
                page = doc.new_page(width=595, height=842)
                page.insert_text(
                    (50, 50),
                    title,
                    fontsize=16,
                    fontname="helvetica-bold",
                    color=(0.1, 0.1, 0.2),
                )
                page.draw_line(
                    fitz.Point(50, 62), fitz.Point(545, 62), color=(0, 0.83, 1), width=2
                )
                try:
                    page.insert_image(fitz.Rect(50, 90, 545, 780), filename=plot_path)
                except Exception as e:
                    page.insert_text(
                        (50, 120),
                        f"⚠️ Could not embed image: {e}",
                        fontsize=10,
                        color=(1, 0, 0),
                    )

        # Classification Report (text)
        page = doc.new_page(width=595, height=842)
        page.insert_text(
            (50, 50),
            "CLASSIFICATION REPORT",
            fontsize=16,
            fontname="helvetica-bold",
            color=(0.1, 0.1, 0.2),
        )
        page.draw_line(
            fitz.Point(50, 62), fitz.Point(545, 62), color=(0, 0.83, 1), width=2
        )

        report_text = metrics.get("classification_report", "")
        y = 90
        for line in report_text.split("\n"):
            if y > 800:
                page = doc.new_page(width=595, height=842)
                y = 50
            page.insert_text(
                (50, y), line, fontsize=9, fontname="courier", color=(0.1, 0.1, 0.2)
            )
            y += 12

        doc.save(pdf_path)
        doc.close()
        print(f"✅ PDF Report: {pdf_path}")

    # ============================================================
    # Cross-Validation
    # ============================================================

    def perform_cross_validation_from_saved(
        self, dataset, n_folds: int = 5, batch_size: int = 32
    ) -> Dict[str, Any]:
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import DataLoader, Subset

        if self.model is None:
            raise ValueError("Model not built.")

        labels = [dataset[i][1] for i in range(len(dataset))]
        labels = np.array(labels)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "auc": [],
        }

        print(f"\n🔬 {n_folds}-Fold Cross-Validation (no retraining)")
        for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f"   Fold {fold + 1}/{n_folds}...")
            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            all_preds = []
            all_labels = []
            all_probs = []

            self.model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    outputs, _ = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = outputs.max(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(targets.numpy())
                    all_probs.extend(probs.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)

            results["accuracy"].append(float(accuracy_score(all_labels, all_preds)))
            results["precision"].append(
                float(precision_score(all_labels, all_preds, average="weighted", zero_division=0))
            )
            results["recall"].append(
                float(recall_score(all_labels, all_preds, average="weighted", zero_division=0))
            )
            results["f1_score"].append(
                float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
            )
            try:
                results["auc"].append(
                    float(
                        roc_auc_score(
                            all_labels, all_probs, multi_class="ovr", average="weighted"
                        )
                    )
                )
            except Exception:
                results["auc"].append(0.0)

            print(
                f"      Acc: {results['accuracy'][-1]:.4f} | "
                f"F1: {results['f1_score'][-1]:.4f}"
            )

        return {
            "mean_accuracy": float(np.mean(results["accuracy"])),
            "std_accuracy": float(np.std(results["accuracy"])),
            "mean_precision": float(np.mean(results["precision"])),
            "std_precision": float(np.std(results["precision"])),
            "mean_recall": float(np.mean(results["recall"])),
            "std_recall": float(np.std(results["recall"])),
            "mean_f1": float(np.mean(results["f1_score"])),
            "std_f1": float(np.std(results["f1_score"])),
            "mean_auc": float(np.mean(results["auc"])),
            "std_auc": float(np.std(results["auc"])),
            **results,
        }


# ================================================================
# Factory
# ================================================================

def create_vit_wrapper(**kwargs) -> ViTWrapper:
    return ViTWrapper(**kwargs)


def ensure_directories():
    """Create all required directories for ViT."""
    dirs = [
        "finalReports/models/visio_transformer/checkpoints",
        "finalReports/models/visio_transformer/configs",
        "finalReports/models/visio_transformer/plots",
        "finalReports/models/visio_transformer/reports",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
