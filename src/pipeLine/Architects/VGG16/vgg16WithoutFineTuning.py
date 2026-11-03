# src/pipeLine/vgg16WithoutFineTuning.py
# -*- coding: utf-8 -*-

"""
VGG16 Without Fine-Tuning - Task 5
===================================
Wrapper class for VGG16 that provides training/evaluation methods
for `run_advanced_model_generic()`.

Author: Babak Yousefian
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

from src.models.VGG16 import VGG16, create_vgg16
from src.models.VGG16.utils import (
    calculate_flops,
    count_parameters,
    get_device,
    get_model_size,
    profile_model,
    set_seed,
    validate_forward_pass,
)


# ================================================================
# VGG16 WRAPPER CLASS
# ================================================================
class VGG16Wrapper:
    """
    VGG16 Wrapper for Task 5 (Without Fine-Tuning).

    Provides:
    - build_model()
    - print_summary()
    - train_full()
    - evaluate_comprehensive()
    - save_model()
    - load_checkpoint()
    - perform_cross_validation_from_saved()
    """

    def __init__(
        self,
        num_classes: int = 5,
        checkpoints_dir: str = None,
        configs_dir: str = None,
        plots_dir: str = None,
        # Architecture
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # Data
        target_size: int = 224,
        # Regularization
        use_class_weights: bool = True,
        class_weight_method: str = "sqrt",
        label_smoothing: float = 0.1,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        use_weighted_sampler: bool = True,
        sampler_power: float = 0.5,
        # Augmentation (handled by loader)
        augmentation_level: str = "basic",
        use_ultrasound_transforms: bool = True,
        # Scheduler
        scheduler_type: str = "cosine",
        warmup_epochs: int = 0,
        min_lr: float = 1e-6,
        total_epochs: int = 100,
        # TTA & Mixup
        use_tta: bool = True,
        use_mixup: bool = True,
        mixup_alpha: float = 0.2,
        mixup_start_stage: int = 4,
        # Labels
        labels_csv_path: str = None,
        # Other
        verbose: bool = True,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.checkpoints_dir = checkpoints_dir or "checkpoints"
        self.configs_dir = configs_dir or "configs"
        self.plots_dir = plots_dir or "plots"
        self.device = torch.device(device)
        # ✅ AMP Scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        # Architecture
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.target_size = target_size

        # Regularization
        self.use_class_weights = use_class_weights
        self.class_weight_method = class_weight_method
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

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

        self.labels_csv_path = labels_csv_path
        self.verbose = verbose

        # State
        self.model = None
        self.is_trained = False
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.train_history = []
        self.current_epoch = 0

        # Create dirs
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        if self.verbose:
            print("\n" + "=" * 70)
            print("🚀 VGG16 Wrapper Initialized")
            print("=" * 70)
            print(f"   Num Classes: {self.num_classes}")
            print(f"   Device: {self.device}")
            print(f"   Target Size: {self.target_size}")
            print(f"   Dropout: {self.dropout_rate}")
            print(f"   BatchNorm: {self.use_batch_norm}")
            print(f"   Weight Decay: 5e-3 (AdamW)")
            print(f"   Label Smoothing: {self.label_smoothing}")
            print(f"   Focal Loss: {self.use_focal_loss}")
            print(f"   Mixup: {self.use_mixup}")
            print(f"   TTA: {self.use_tta}")
            print(f"   Scheduler: {self.scheduler_type}")
            print(f"   Total Epochs: {self.total_epochs}")
            print("=" * 70 + "\n")

    # ================================================================
    # BUILD MODEL
    # ================================================================
    def build_model(self) -> nn.Module:
        """Build VGG16 model."""
        self.model = create_vgg16(
            num_classes=self.num_classes,
            pretrained=False,
            verbose=False,
            use_batch_norm=self.use_batch_norm,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
        )
        self.model = self.model.to(self.device)

        if self.verbose:
            print(f"✅ VGG16 built on {self.device}")

        return self.model

    def print_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            self.build_model()

        params = count_parameters(self.model)
        size = get_model_size(self.model)

        print("\n" + "=" * 70)
        print("📊 VGG16 MODEL SUMMARY")
        print("=" * 70)
        print(f"   Total Parameters:     {params['total']:,}")
        print(f"   Trainable Parameters: {params['trainable']:,}")
        print(f"   Model Size:           {size['mb']:.2f} MB")
        print(f"   Device:               {self.device}")
        print("=" * 70 + "\n")

    # ================================================================
    # LOSS FUNCTION (Focal Loss or CrossEntropy)
    # ================================================================
    def _get_criterion(self) -> nn.Module:
        """Get loss function."""
        if self.use_focal_loss:
            # Focal Loss
            gamma = self.focal_gamma

            class FocalLoss(nn.Module):
                def __init__(self, gamma=2.0, reduction="mean"):
                    super().__init__()
                    self.gamma = gamma
                    self.reduction = reduction

                def forward(self, inputs, targets):
                    ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
                    pt = torch.exp(-ce)
                    focal = (1 - pt) ** self.gamma * ce

                    if self.reduction == "mean":
                        return focal.mean()
                    elif self.reduction == "sum":
                        return focal.sum()
                    return focal

            if self.verbose:
                print(f"   ✅ Using Focal Loss (gamma={gamma})")
            return FocalLoss(gamma=gamma)
        else:
            # CrossEntropy with label smoothing
            if self.verbose:
                print(f"   ✅ Using CrossEntropy (label_smoothing={self.label_smoothing})")
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    # ================================================================
    # OPTIMIZER + SCHEDULER
    # ================================================================
    def _get_optimizer_scheduler(self):
        """Get optimizer and scheduler."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=5e-3,  # ← 1e-3 → 5e-3
            betas=(0.9, 0.999),
        )

        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.total_epochs, eta_min=self.min_lr
            )
        elif self.scheduler_type == "warmup_cosine":
            def warmup_cosine(epoch):
                if epoch < self.warmup_epochs:
                    return (epoch + 1) / self.warmup_epochs
                progress = (epoch - self.warmup_epochs) / max(
                    1, self.total_epochs - self.warmup_epochs
                )
                return 0.5 * (1 + np.cos(np.pi * progress))

            scheduler = LambdaLR(optimizer, warmup_cosine)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.total_epochs)

        return optimizer, scheduler

    # ================================================================
    # TRAIN FULL
    # ================================================================
    def train_full(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        save_best: bool = True,
    ) -> Dict[str, Any]:
        """Train model fully."""
        if self.model is None:
            self.build_model()

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
        }

        print("\n" + "=" * 70)
        print(f"🚀 TRAINING VGG16 FOR {epochs} EPOCHS")
        print("=" * 70)

        # ✅ Early Stopping Setup (patience=20 برای مدل‌های کوچیک)
        patience_counter = 0
        patience_limit = 20  # ← 5 → 20 (حداقل 40 epoch)

        for epoch in range(epochs):
            epoch_start = time.time()
            self.current_epoch = epoch + 1

            # ==================== TRAIN ====================
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_idx, (inputs, targets) in enumerate(train_bar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Mixup
                # ✅ AMP Autocast
                with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                    if self.use_mixup and epoch >= self.mixup_start_stage:
                        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                        rand_idx = torch.randperm(inputs.size(0)).to(self.device)
                        mixed_inputs = lam * inputs + (1 - lam) * inputs[rand_idx]
                        mixed_targets_a, mixed_targets_b = targets, targets[rand_idx]

                        outputs = self.model(mixed_inputs)
                        loss = lam * criterion(outputs, mixed_targets_a) + (1 - lam) * criterion(
                            outputs, mixed_targets_b
                        )
                    else:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)

                optimizer.zero_grad()

                # ✅ AMP Backward
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

                train_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total:.4f}"}
                )

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # ==================== VALIDATION ====================
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for inputs, targets in val_bar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

                    val_bar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "acc": f"{val_correct/val_total:.4f}"}
                    )

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            # Step scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # Save history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["learning_rates"].append(current_lr)

            epoch_time = time.time() - epoch_start

            print(
                f"\n📊 Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

            # Save best
            # ✅ Early Stopping
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                self.best_accuracy = best_acc
                self.best_epoch = best_epoch
                patience_counter = 0

                if save_best:
                    self._save_checkpoint(
                        filename="vgg16_best.pth",
                        epoch=epoch,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        val_acc=val_acc,
                        val_loss=val_loss,
                    )
                    print(f"   ✅ Best model saved! (Val Acc: {val_acc:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:  # ← patience_limit (20)
                        print(f"\n🛑 Early Stopping at epoch {epoch + 1}")
                        print(f"   Best Val Acc: {best_acc:.4f} at epoch {best_epoch}")
                        break

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(
                    filename="vgg16_checkpoint.pth",
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_acc=val_acc,
                    val_loss=val_loss,
                )

            # Save last
            self._save_checkpoint(
                filename="vgg16_last.pth",
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                val_acc=val_acc,
                val_loss=val_loss,
            )

        # Save final
        self._save_checkpoint(
            filename="vgg16_final.pth",
            epoch=epochs - 1,
            optimizer=optimizer,
            scheduler=scheduler,
            val_acc=val_acc,
            val_loss=val_loss,
        )

        self.is_trained = True
        self.train_history = history

        # Save config
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

    # ================================================================
    # CHECKPOINT SAVE
    # ================================================================
    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        optimizer,
        scheduler,
        val_acc: float,
        val_loss: float,
    ) -> None:
        """Save checkpoint."""
        path = os.path.join(self.checkpoints_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "best_accuracy": self.best_accuracy,
                "num_classes": self.num_classes,
                "config": {
                    "use_batch_norm": self.use_batch_norm,
                    "activation": self.activation,
                    "dropout_rate": self.dropout_rate,
                },
            },
            path,
        )

    def _save_config(self) -> None:
        """Save config to JSON."""
        config = {
            "model": "VGG16",
            "variant": "vgg16",
            "class_names": ["NL", "G1", "G2", "G3"],
            "num_classes": self.num_classes,
            "target_size": self.target_size,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "label_smoothing": self.label_smoothing,
            "use_focal_loss": self.use_focal_loss,
            "focal_gamma": self.focal_gamma,
            "scheduler_type": self.scheduler_type,
            "warmup_epochs": self.warmup_epochs,
            "min_lr": self.min_lr,
            "total_epochs": self.total_epochs,
            "use_mixup": self.use_mixup,
            "mixup_alpha": self.mixup_alpha,
            "use_tta": self.use_tta,
            "device": str(self.device),
            "created_at": datetime.now().isoformat(),
        }

        path = os.path.join(self.configs_dir, "config_vgg16.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

        if self.verbose:
            print(f"✅ Config saved: {path}")

    # ================================================================
    # LOAD CHECKPOINT
    # ================================================================
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        if self.model is None:
            self.build_model()

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
        self.is_trained = True

        print(f"✅ Loaded: {path}")
        print(f"   Best Accuracy: {self.best_accuracy:.4f}")

    # ================================================================
    # EVALUATE COMPREHENSIVE
    # ================================================================
    def evaluate_comprehensive(
        self,
        test_loader,
        generate_report: bool = True,
        cv_folds: int = 0,
        report_name: str = "VGG16_Task5_Report.pdf",
    ) -> Dict[str, Any]:
        """Comprehensive evaluation on test set."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

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
            for inputs, targets in tqdm(test_loader, desc="Test"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
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

        # TTA (Test Time Augmentation) - optional
        if self.use_tta:
            print("   ✅ TTA enabled (predicting twice)")
            all_probs_tta = []
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs_tta.extend(probs.cpu().numpy())

            all_probs = (all_probs + np.array(all_probs_tta)) / 2
            all_preds = all_probs.argmax(axis=1)

        # ==========================================
        # Compute Metrics
        # ==========================================
        test_acc = accuracy_score(all_labels, all_preds)
        test_precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        test_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        test_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        # AUC
        try:
            test_auc = roc_auc_score(
                all_labels,
                all_probs,
                multi_class="ovr",
                average="weighted",
            )
        except Exception as e:
            print(f"   ⚠️ AUC failed: {e}")
            test_auc = 0.0

        test_loss = total_loss / len(test_loader)

        print(f"\n✅ Test Accuracy:  {test_acc:.4f}")
        print(f"✅ Test Precision: {test_precision:.4f}")
        print(f"✅ Test Recall:    {test_recall:.4f}")
        print(f"✅ Test F1-Score:  {test_f1:.4f}")
        print(f"✅ Test AUC:       {test_auc:.4f}")
        print(f"✅ Test Loss:      {test_loss:.4f}")
        print("=" * 70)

        # ==========================================
        # Confusion Matrix
        # ==========================================
        cm = confusion_matrix(all_labels, all_preds)
        class_names = ["NL", "G1", "G2", "G3"]

        # ==========================================
        # Generate Plots
        # ==========================================
        self._generate_plots(all_labels, all_preds, all_probs, cm, class_names)

        # ==========================================
        # Save Metrics JSON
        # ==========================================
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
            "num_samples": len(all_labels),
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
        }

        # ==========================================
        # Generate PDF Report
        # ==========================================
        if generate_report:
            self._generate_pdf_report(metrics, report_name)

        return {
            "metrics": metrics,
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist(),
            "probabilities": all_probs.tolist(),
        }

    # ================================================================
    # GENERATE PLOTS
    # ================================================================
    def _generate_plots(self, y_true, y_pred, y_prob, cm, class_names):
        """Generate all plots."""
        os.makedirs(self.plots_dir, exist_ok=True)

        # ==========================================
        # 1. Confusion Matrix
        # ==========================================
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
        ax.set_title("VGG16 — Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # ==========================================
        # 2. ROC Curves
        # ==========================================
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]

        for i, name in enumerate(class_names):
            y_bin = (y_true == i).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2.5, color=colors[i], label=f"{name} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.6, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax.set_title("VGG16 — ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "roc_curves.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # ==========================================
        # 3. Metrics Bar
        # ==========================================
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics_vals = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        colors_bar = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12"]
        bars = ax.bar(metrics_vals.keys(), metrics_vals.values(), color=colors_bar, edgecolor="black")

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
        ax.set_title("VGG16 — Performance Metrics (Test Set)", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "metrics_bar.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # ==========================================
        # 4. Training History (if available)
        # ==========================================
        if self.train_history:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
            ep = range(1, len(self.train_history["train_loss"]) + 1)

            axes[0].plot(ep, self.train_history["train_loss"], "o-", color="#e74c3c", label="Train", linewidth=2)
            axes[0].plot(ep, self.train_history["val_loss"], "s-", color="#3498db", label="Val", linewidth=2)
            axes[0].set_xlabel("Epoch", fontweight="bold")
            axes[0].set_ylabel("Loss", fontweight="bold")
            axes[0].set_title("Loss History", fontweight="bold")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].plot(ep, self.train_history["train_acc"], "o-", color="#e74c3c", label="Train", linewidth=2)
            axes[1].plot(ep, self.train_history["val_acc"], "s-", color="#2ecc71", label="Val", linewidth=2)
            axes[1].set_xlabel("Epoch", fontweight="bold")
            axes[1].set_ylabel("Accuracy", fontweight="bold")
            axes[1].set_title("Accuracy History", fontweight="bold")
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.suptitle("VGG16 — Training History", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "training_history.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # ==========================================
            # 5. Train vs Val Gap
            # ==========================================
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=200)

            train_loss = self.train_history["train_loss"]
            val_loss = self.train_history["val_loss"]
            gap = [v - t for t, v in zip(train_loss, val_loss)]

            axes[0].plot(ep, train_loss, "o-", color="#3498db", linewidth=2.5, label="Train")
            axes[0].plot(ep, val_loss, "s-", color="#e74c3c", linewidth=2.5, label="Val")
            axes[0].fill_between(ep, train_loss, val_loss, alpha=0.2, color="#f39c12", label="Gap")
            axes[0].set_xlabel("Epoch", fontweight="bold")
            axes[0].set_ylabel("Loss", fontweight="bold")
            axes[0].set_title("Train vs Validation Loss — Gap Analysis", fontweight="bold")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            colors_gap = ["#e74c3c" if g > 0 else "#2ecc71" for g in gap]
            axes[1].bar(ep, gap, color=colors_gap, alpha=0.75, edgecolor="black", linewidth=0.5)
            axes[1].axhline(y=0, color="black", linewidth=1.5)
            axes[1].axhline(y=np.mean(gap), color="#f39c12", linewidth=2, linestyle="--", label=f"Avg Gap: {np.mean(gap):+.4f}")
            axes[1].set_xlabel("Epoch", fontweight="bold")
            axes[1].set_ylabel("Gap (Val - Train)", fontweight="bold")
            axes[1].set_title("Overfitting / Underfitting Indicator", fontweight="bold")
            axes[1].legend()
            axes[1].grid(alpha=0.3, axis="y")

            plt.suptitle("VGG16 — Loss & Gap Analysis", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "train_val_gap.png"), dpi=200, bbox_inches="tight")
            plt.close()

        print(f"✅ Plots saved to: {self.plots_dir}")

    # ================================================================
    # GENERATE PDF REPORT
    # ================================================================
    def _generate_pdf_report(self, metrics, report_name):
        """Generate PDF report."""
        try:
            import fitz
        except ImportError:
            print("⚠️ PyMuPDF not installed. Skipping PDF.")
            return

        pdf_path = os.path.join(
            os.path.dirname(self.plots_dir), "reports", report_name
        )
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        doc = fitz.open()

        # Page 1: Cover
        page = doc.new_page(width=595, height=842)
        page.draw_rect(fitz.Rect(0, 0, 595, 200), color=None, fill=(0.1, 0.1, 0.18), width=0)
        page.insert_text((50, 80), "VGG16", fontsize=36, fontname="helvetica-bold", color=(0, 0.83, 1))
        page.insert_text((50, 130), "Fine-Tuning Report", fontsize=22, fontname="helvetica", color=(1, 1, 1))
        page.insert_text((50, 165), "Liver Ultrasound Classification (NL / G1 / G2 / G3)", fontsize=12, fontname="helvetica", color=(0.65, 0.75, 0.85))

        acc = metrics["test_accuracy"]
        page.draw_rect(fitz.Rect(50, 230, 545, 320), color=(0, 0.83, 1), fill=(0.9, 0.97, 1), width=2)
        page.insert_text((70, 265), "🎯 TEST ACCURACY", fontsize=14, fontname="helvetica-bold", color=(0.1, 0.1, 0.2))
        page.insert_text((70, 305), f"{acc * 100:.2f}%", fontsize=32, fontname="helvetica-bold", color=(0, 0.55, 0.85))

        info = [
            ("Model", "VGG16 (from scratch)"),
            ("Test Samples", str(metrics["num_samples"])),
            ("Best Epoch", str(metrics["best_epoch"])),
            ("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Author", "Babak Yousefian"),
        ]
        y = 370
        for key, val in info:
            page.draw_rect(fitz.Rect(50, y, 545, y + 28), color=None, fill=(0.97, 0.98, 1))
            page.insert_text((60, y + 19), key, fontsize=11, fontname="helvetica-bold", color=(0.2, 0.2, 0.3))
            page.insert_text((200, y + 19), val, fontsize=11, fontname="helvetica", color=(0.1, 0.1, 0.2))
            y += 32

        # Page 2: Metrics
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), "TEST METRICS", fontsize=18, fontname="helvetica-bold", color=(0, 0.55, 0.85))
        page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62), color=(0, 0.83, 1), width=2)

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
            page.draw_rect(fitz.Rect(50, y, 545, y + 26), color=(0.8, 0.85, 0.9), fill=(0.97, 0.98, 1), width=0.5)
            page.insert_text((60, y + 18), label, fontsize=11, fontname="helvetica", color=(0.15, 0.15, 0.2))
            page.insert_text((400, y + 18), f"{val:.4f}", fontsize=12, fontname="helvetica-bold", color=(0, 0.55, 0.85))
            y += 30

        # Pages 3+: Plots
        plot_files = [
            ("confusion_matrix.png", "CONFUSION MATRIX"),
            ("roc_curves.png", "ROC CURVES"),
            ("metrics_bar.png", "PERFORMANCE METRICS"),
            ("training_history.png", "TRAINING HISTORY"),
            ("train_val_gap.png", "TRAIN vs VAL LOSS GAP"),
        ]

        for plot_file, title in plot_files:
            plot_path = os.path.join(self.plots_dir, plot_file)
            if os.path.exists(plot_path):
                page = doc.new_page(width=595, height=842)
                page.insert_text((50, 50), title, fontsize=16, fontname="helvetica-bold", color=(0.1, 0.1, 0.2))
                page.insert_image(fitz.Rect(50, 90, 545, 750), filename=plot_path)

        doc.save(pdf_path)
        doc.close()
        print(f"✅ PDF Report: {pdf_path}")

    # ================================================================
    # SAVE MODEL
    # ================================================================
    def save_model(self, filename: str = "vgg16_final.pth") -> str:
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not built.")
        path = os.path.join(self.checkpoints_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_classes": self.num_classes,
                "best_accuracy": self.best_accuracy,
                "config": {
                    "use_batch_norm": self.use_batch_norm,
                    "activation": self.activation,
                    "dropout_rate": self.dropout_rate,
                },
            },
            path,
        )
        print(f"✅ Model saved: {path}")
        return path

    # ================================================================
    # CROSS-VALIDATION (from saved)
    # ================================================================
    def perform_cross_validation_from_saved(
        self, dataset, n_folds: int = 5, batch_size: int = 32
    ):
        """Perform CV using the saved model (no retraining)."""
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import DataLoader, Subset

        # Get labels for stratification
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)

        labels = np.array(labels)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        results = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
        }

        print(f"\n🔬 {n_folds}-Fold Cross-Validation (from saved model)")

        for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f"   Fold {fold+1}/{n_folds}...")

            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            all_preds = []
            all_labels = []

            self.model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = outputs.max(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(targets.numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

            results["accuracy"].append(acc)
            results["precision"].append(prec)
            results["recall"].append(rec)
            results["f1_score"].append(f1)

            print(f"      Accuracy: {acc:.4f}")

        return {
            "mean_accuracy": float(np.mean(results["accuracy"])),
            "std_accuracy": float(np.std(results["accuracy"])),
            "mean_precision": float(np.mean(results["precision"])),
            "std_precision": float(np.std(results["precision"])),
            "mean_recall": float(np.mean(results["recall"])),
            "std_recall": float(np.std(results["recall"])),
            "mean_f1": float(np.mean(results["f1_score"])),
            "std_f1": float(np.std(results["f1_score"])),
            **results,
        }


# ================================================================
# FACTORY FUNCTION
# ================================================================
def create_vgg16_wrapper(**kwargs) -> VGG16Wrapper:
    """Factory function to create VGG16Wrapper."""
    return VGG16Wrapper(**kwargs)


def ensure_directories():
    """Ensure directories exist."""
    dirs = [
        "finalReports/models/VGG16/checkpoints",
        "finalReports/models/VGG16/configs",
        "finalReports/models/VGG16/plots",
        "finalReports/models/VGG16/reports",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ================================================================
# TEST
# ================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing VGG16 Wrapper")
    print("=" * 80)

    ensure_directories()

    wrapper = VGG16Wrapper(
        num_classes=4,
        checkpoints_dir="finalReports/models/VGG16/checkpoints",
        configs_dir="finalReports/models/VGG16/configs",
        plots_dir="finalReports/models/VGG16/plots",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    wrapper.build_model()
    wrapper.print_summary()

    print("\n✅ VGG16 Wrapper is ready!")
    print("=" * 80)
