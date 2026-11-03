# src/AI/ResNet/resnet18.py

"""
ResNet18 Trainer for Liver Disease Classification
This module is designed to train ResNet18 model on liver image data
ResNet18 is a lightweight and fast model with 18 layers, suitable for medium-sized data
"""

import json
import math
import os
import random
import time
import warnings
from datetime import datetime
from typing import *

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =====================================================
# ✅ IMPORTS FROM UTILS
# =====================================================
from .utils import RESNET_REPORTS_DIR  # ✅ اضافه شد
from .utils import (
    CLASS_NAMES,
    CROPPED,
    HYPERPARAMS,
    INV_LABEL_MAP,
    LABEL_MAP,
    OUTPUT_DIR,
    CSVAddress,
    LabelSmoothingCrossEntropy,
    WarmupCosineScheduler,
    apply_advanced_augmentation,
    apply_augmentation,
    check_paths,
    compute_metrics,
    create_weighted_sampler,
    export_predictions,
    get_class_weights,
    get_device,
    load_model,
    normalize_resnet,
    print_metrics_summary,
    resize_with_padding,
    save_model,
    set_seed,
    validate_csv,
    validate_images,
)

# ✅ اضافه کردن save_model_checkpoint و load_model_checkpoint
# src/AI/ResNet/resnet18.py


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filename: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    save_dir: Optional[str] = None,
) -> str:
    """Save a complete model checkpoint with all training state."""
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, "resnet18")

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "model_architecture": "ResNet18",
        "version": "3.0.0",
    }

    # ✅ فقط اگه scheduler وجود داشته باشه و state_dict داشته باشه
    if scheduler is not None:
        try:
            if hasattr(scheduler, "state_dict"):
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            else:
                # ✅ برای schedulerهای سفارشی مثل WarmupCosineScheduler
                checkpoint["scheduler_state_dict"] = {
                    "current_epoch": scheduler.current_epoch,
                    "current_lr": scheduler.current_lr,
                }
        except Exception as e:
            print(f"⚠️ Could not save scheduler state: {e}")

    if extra_info is not None:
        checkpoint["extra_info"] = extra_info

    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint saved: {filepath}")
    return filepath


# src/AI/ResNet/resnet18.py


def load_model_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load a complete model checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ✅ بارگذاری scheduler اگه وجود داشته باشه
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                # ✅ برای schedulerهای سفارشی
                if hasattr(scheduler, "current_epoch"):
                    scheduler.current_epoch = checkpoint["scheduler_state_dict"].get(
                        "current_epoch", 0
                    )
                if hasattr(scheduler, "current_lr"):
                    scheduler.current_lr = checkpoint["scheduler_state_dict"].get(
                        "current_lr", scheduler.max_lr
                    )
        except Exception as e:
            print(f"⚠️ Could not load scheduler state: {e}")

    print(f"✅ Checkpoint loaded: {filepath}")
    return checkpoint


# =====================================================
# GLOBAL SETTINGS
# =====================================================
DEVICE = get_device()
torch.set_num_threads(4)
set_seed(42)


# =====================================================
# RESNET18 DATASET
# =====================================================
class ResNet18Dataset:
    def __init__(
        self,
        df,
        img_dir=None,
        augment=False,
        target_size=256,
        apply_normalize=True,
        use_advanced_augment=True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir if img_dir is not None else CROPPED
        self.augment = augment
        self.target_size = target_size
        self.apply_normalize = apply_normalize
        self.use_advanced_augment = use_advanced_augment

        self.df = self.df[self.df["liverLabel"] != "UNKNOWN"]
        class_counts = self.df["liverLabel"].value_counts()
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        self.df = self.df[self.df["liverLabel"].isin(valid_classes)]

        if len(self.df) == 0:
            raise ValueError("No valid labels found in dataset!")

        print(f"✅ ResNet18 Dataset initialized with {len(self.df)} samples")
        print(f"   Classes: {sorted(self.df['liverLabel'].unique())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["liverName"])
            label = LABEL_MAP[row["liverLabel"]]

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            else:
                img = resize_with_padding(img, self.target_size)

            if self.augment:
                if self.use_advanced_augment:
                    img = apply_advanced_augmentation(img, use_erasing=True)
                else:
                    img = apply_augmentation(img)

            img = img.astype(np.float32) / 255.0
            img = np.stack([img, img, img], axis=0)
            img_tensor = torch.tensor(img, dtype=torch.float32)

            if self.apply_normalize:
                img_tensor = normalize_resnet(img_tensor)

            return (img_tensor, torch.tensor(label, dtype=torch.long), row["liverName"])

        except Exception as e:
            print(f"❌ Error in __getitem__ (idx={idx}): {e}")
            img = np.zeros((3, self.target_size, self.target_size), dtype=np.float32)
            img_tensor = torch.tensor(img, dtype=torch.float32)
            if self.apply_normalize:
                img_tensor = normalize_resnet(img_tensor)
            return (img_tensor, torch.tensor(0, dtype=torch.long), "error_image.png")


# =====================================================
# RESNET18 TRAINER - FULLY OPTIMIZED 🔥
# =====================================================
class ResNet18Trainer:
    def __init__(
        self,
        csv_path=None,
        img_dir=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        num_workers=0,
        target_size=224,
        dropout_rate=None,
        use_mixed_precision=True,
        use_sampler=True,
        label_smoothing=None,
        accumulation_steps=None,
        warmup_epochs=None,
    ):
        hp = HYPERPARAMS["resnet18"]

        self.csv_path = csv_path if csv_path is not None else CSVAddress
        self.img_dir = img_dir if img_dir is not None else CROPPED
        self.batch_size = batch_size if batch_size is not None else hp["batch_size"]
        self.lr = lr if lr is not None else hp["lr"]
        self.weight_decay = weight_decay if weight_decay is not None else hp["weight_decay"]
        self.dropout_rate = dropout_rate if dropout_rate is not None else hp["dropout"]
        self.label_smoothing = (
            label_smoothing if label_smoothing is not None else hp["label_smoothing"]
        )
        self.accumulation_steps = (
            accumulation_steps if accumulation_steps is not None else hp["accumulation_steps"]
        )
        self.warmup_epochs = warmup_epochs if warmup_epochs is not None else hp["warmup_epochs"]
        self.epochs_per_phase = hp["epochs_per_phase"]
        self.target_size = target_size
        self.use_mixed_precision = use_mixed_precision
        self.use_sampler = use_sampler
        self.num_workers = num_workers
        # ✅ نرخ یادگیری برای هر فاز (پیشفرض: همه برابر base_lr)
        self.phase_lrs = None  # ← اضافه کن

        print("\n" + "=" * 70)
        print("🚀 Initializing ResNet18 Trainer (OPTIMIZED)")
        print("=" * 70)
        print(f"📁 CSV Path: {self.csv_path}")
        print(f"📁 Image Dir: {self.img_dir}")
        print(f"📦 Batch Size: {self.batch_size}")
        print(f"📊 Learning Rate: {self.lr}")
        print(f"⚙️  Weight Decay: {self.weight_decay}")
        print(f"🎯 Dropout Rate: {self.dropout_rate}")
        print(f"🎯 Label Smoothing: {self.label_smoothing}")
        print(f"📊 Accumulation Steps: {self.accumulation_steps}")
        print(f"🔥 Warmup Epochs: {self.warmup_epochs}")
        print(f"📈 Epochs per Phase: {self.epochs_per_phase}")
        print(f"🖥️  Device: {DEVICE}")
        print("=" * 70)

        if not check_paths():
            raise ValueError("Path validation failed!")

        is_valid, df, message = validate_csv(self.csv_path)
        if not is_valid:
            raise ValueError(f"CSV validation failed: {message}")
        self.df = df

        total, missing, missing_list = validate_images(self.csv_path, self.img_dir)
        if missing > 0:
            print(f"⚠️ Warning: {missing} images are missing!")

        print("\n📊 Loading and splitting data...")
        self.train_ds, self.val_ds = self._load_data()
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()

        print("\n🏗️ Building ResNet18 model...")
        self.model = self._build_model()

        print("\n⚖️ Setting up loss function...")
        class_weights = get_class_weights(self.csv_path)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=self.label_smoothing)
        print(f"   ✅ Label Smoothing: {self.label_smoothing}")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_epochs = sum(self.epochs_per_phase)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=total_epochs,
            min_lr=1e-7,
            max_lr=self.lr,
        )
        print(f"   ✅ Warmup Cosine Scheduler: {total_epochs} total epochs")

        if self.use_mixed_precision and DEVICE.type == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
            if self.use_mixed_precision:
                print("⚠️ Mixed precision requested but CUDA not available. Disabling.")

        self.best_acc = 0.0
        self.best_epoch = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
            "epoch_times": [],
        }
        self.metrics = None
        self.cm = None
        self.auc_score = None
        self.y_true = None
        self.y_pred = None
        self.y_prob = None

        # ✅ Storage paths - مطابق با ساختار
        self.base_dir = OUTPUT_DIR
        self.model_dir = os.path.join(self.base_dir, "resnet18")
        os.makedirs(self.model_dir, exist_ok=True)

        self.save_path = os.path.join(self.model_dir, "resnet18_best.pth")
        self.checkpoint_path = os.path.join(self.model_dir, "resnet18_checkpoint.pth")
        self.last_path = os.path.join(self.model_dir, "resnet18_last.pth")
        self.config_path = os.path.join(self.model_dir, "config_resnet18.json")

        self._save_config()

        print(f"\n✅ ResNet18 Trainer initialized successfully!")
        print(f"📁 Model directory: {self.model_dir}")
        print(f"📁 Best model: {self.save_path}")
        print("=" * 70 + "\n")

    # =====================================================
    # PRIVATE METHODS
    # =====================================================

    def _load_data(self):
        """Loading and splitting data into Train/Val/Test (80/10/10)"""
        from sklearn.model_selection import train_test_split

        df_valid = self.df[self.df["liverLabel"] != "UNKNOWN"].copy()

        if len(df_valid) == 0:
            raise ValueError("No valid labels found after removing UNKNOWN!")

        class_counts = df_valid["liverLabel"].value_counts()
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        df_valid = df_valid[df_valid["liverLabel"].isin(valid_classes)]

        if len(df_valid) == 0:
            raise ValueError("No classes with enough samples!")

        print("\n📊 Label Distribution (after filtering):")
        for label in sorted(df_valid["liverLabel"].unique()):
            count = len(df_valid[df_valid["liverLabel"] == label])
            print(f"   {label}: {count}")

        # ✅ Step 1: Test (10%)
        try:
            train_val_df, test_df = train_test_split(
                df_valid,
                test_size=0.10,
                stratify=df_valid["liverLabel"],
                random_state=42,
            )
        except ValueError as e:
            print(f"⚠️ Stratify failed (test split): {e}")
            train_val_df, test_df = train_test_split(
                df_valid, test_size=0.10, random_state=42
            )

        # ✅ Step 2: Validation (10% از کل = 1/9 از train_val)
        try:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=1 / 9,  # ← همون فرمول withoutFineTuner
                stratify=train_val_df["liverLabel"],
                random_state=42,
            )
        except ValueError as e:
            print(f"⚠️ Stratify failed (val split): {e}")
            train_df, val_df = train_test_split(
                train_val_df, test_size=1 / 9, random_state=42
            )

        train_ds = ResNet18Dataset(
            train_df, self.img_dir, augment=True,
            target_size=self.target_size, apply_normalize=True,
            use_advanced_augment=True,
        )
        val_ds = ResNet18Dataset(
            val_df, self.img_dir, augment=False,
            target_size=self.target_size, apply_normalize=True,
            use_advanced_augment=False,
        )
        test_ds = ResNet18Dataset(
            test_df, self.img_dir, augment=False,
            target_size=self.target_size, apply_normalize=True,
            use_advanced_augment=False,
        )

        print(f"\n✅ Train: {len(train_ds)} samples ({len(train_ds) / len(df_valid) * 100:.1f}%)")
        print(f"✅ Val:   {len(val_ds)} samples ({len(val_ds) / len(df_valid) * 100:.1f}%)")
        print(f"✅ Test:  {len(test_ds)} samples ({len(test_ds) / len(df_valid) * 100:.1f}%)")

        self.test_ds = test_ds
        return train_ds, val_ds

    def _create_dataloaders(self):
        train_sampler = None
        if self.use_sampler:
            print("\n🎯 Creating WeightedRandomSampler for training...")
            train_sampler = create_weighted_sampler(self.train_ds, self.csv_path)

        train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=(train_sampler is None), sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True if DEVICE.type == "cuda" else False,
            drop_last=False,
        )
        val_loader = DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if DEVICE.type == "cuda" else False,
            drop_last=False,
        )
        test_loader = DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if DEVICE.type == "cuda" else False,
            drop_last=False,
        )
        return train_loader, val_loader, test_loader

    def _build_model(self):
        model = models.resnet18(weights="IMAGENET1K_V1")

        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.3),
            nn.Linear(256, len(CLASS_NAMES)),
        )

        model = model.to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✅ Model built successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Non-trainable parameters: {total_params - trainable_params:,}")

        return model

    def _save_config(self):
        config = {
            "model": "ResNet18",
            "class_names": CLASS_NAMES,
            "csv_path": self.csv_path,
            "img_dir": self.img_dir,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "target_size": self.target_size,
            "dropout_rate": self.dropout_rate,
            "label_smoothing": self.label_smoothing,
            "accumulation_steps": self.accumulation_steps,
            "warmup_epochs": self.warmup_epochs,
            "epochs_per_phase": self.epochs_per_phase,
            "num_workers": self.num_workers,
            "use_mixed_precision": self.use_mixed_precision,
            "use_sampler": self.use_sampler,
            "device": str(DEVICE),
            "created_at": datetime.now().isoformat(),
        }

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"✅ Config saved: {self.config_path}")

    def _set_phase_lr(self, new_lr: float):
        """✅ تنظیم نرخ یادگیری برای فاز جدید"""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        if hasattr(self.scheduler, 'max_lr'):
            self.scheduler.max_lr = new_lr
        if hasattr(self.scheduler, 'current_lr'):
            self.scheduler.current_lr = new_lr
        print(f"   🔧 Learning Rate set to: {new_lr:.6f}")

    # =====================================================
    # FREEZING METHODS
    # =====================================================

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Backbone frozen (only FC layer trainable)")
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    def unfreeze_layer4(self):
        for name, param in self.model.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Layer4 and FC layer unfrozen")
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    def unfreeze_layer3_4(self):
        for name, param in self.model.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Layer3, Layer4 and FC layer unfrozen")
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ All layers unfrozen (full fine-tuning)")
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # =====================================================
    # TRAINING METHODS
    # =====================================================

    def train_epoch(self, epochs=1, desc=""):
        for epoch in range(epochs):
            epoch_start = time.time()

            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            self.optimizer.zero_grad()

            loop = tqdm(self.train_loader, desc=f"[ResNet18] {desc} Epoch {epoch+1}/{epochs}")

            for batch_idx, (x, y, _) in enumerate(loop):
                x, y = x.to(DEVICE), y.to(DEVICE)

                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        out = self.model(x)
                        loss = self.criterion(out, y)
                        loss = loss / self.accumulation_steps

                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    out = self.model(x)
                    loss = self.criterion(out, y)
                    loss = loss / self.accumulation_steps

                    loss.backward()

                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                total_loss += loss.item() * self.accumulation_steps
                pred = out.argmax(dim=1)
                total_correct += (pred == y).sum().item()
                total_samples += y.size(0)

                loop.set_postfix(
                    {
                        "loss": f"{loss.item() * self.accumulation_steps:.4f}",
                        "acc": f"{total_correct/total_samples:.4f}",
                    }
                )

            if batch_idx % self.accumulation_steps != 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            avg_loss = total_loss / len(self.train_loader)
            train_acc = total_correct / total_samples

            val_loss, val_acc = self.evaluate(return_loss=True)
            current_lr = self.scheduler.step()

            epoch_time = time.time() - epoch_start
            self.history["train_loss"].append(avg_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)
            self.history["epoch_times"].append(epoch_time)

            # ✅ استفاده از save_model_checkpoint
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                save_model_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics={"val_acc": val_acc, "val_loss": val_loss},
                    filename="resnet18_best.pth",
                    scheduler=self.scheduler,
                    extra_info={"best_accuracy": self.best_acc},
                )
                print(f"✅ Best model saved with accuracy: {val_acc:.4f}")

            if (epoch + 1) % 5 == 0:
                save_model_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics={"val_acc": val_acc, "val_loss": val_loss},
                    filename="resnet18_checkpoint.pth",
                    scheduler=self.scheduler,
                )

            save_model_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics={"val_acc": val_acc, "val_loss": val_loss},
                filename="resnet18_last.pth",
                scheduler=self.scheduler,
            )

            print(
                f"📊 Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
            )

            if len(self.history["val_acc"]) > 15:
                last_10 = self.history["val_acc"][-15:]
                if len(last_10) >= 15 and all(v <= last_10[0] for v in last_10[1:]):
                    print(f"⚠️ Early stopping triggered (no improvement for 10 epochs)")
                    break

        print(f"\n🔥 Best Accuracy: {self.best_acc:.4f} (Epoch {self.best_epoch+1})")
        return self.best_acc

    def train_full(self):
        print("\n" + "=" * 70)
        print("🚀 STARTING FULL TRAINING PIPELINE (ResNet18 - OPTIMIZED)")
        print("=" * 70)

        # ✅ اگر phase_lrs تعریف شده، استفاده کن
        phase_lrs = self.phase_lrs if self.phase_lrs else {
            0: self.lr, 1: self.lr, 2: self.lr, 3: self.lr
        }

        # Phase 1
        print(
            f"\n📌 Phase 1: Training FC layer only ({self.epochs_per_phase[0]} epochs) - LR: {phase_lrs.get(0, self.lr)}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(0, self.lr))
        self.freeze_backbone()
        self.train_epoch(epochs=self.epochs_per_phase[0], desc="Phase 1")

        # Phase 2
        print(
            f"\n📌 Phase 2: Training Layer4 + FC ({self.epochs_per_phase[1]} epochs) - LR: {phase_lrs.get(1, self.lr)}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(1, self.lr))
        self.unfreeze_layer4()
        self.train_epoch(epochs=self.epochs_per_phase[1], desc="Phase 2")

        # Phase 3
        print(
            f"\n📌 Phase 3: Training Layer3 + Layer4 + FC ({self.epochs_per_phase[2]} epochs) - LR: {phase_lrs.get(2, self.lr)}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(2, self.lr))
        self.unfreeze_layer3_4()
        self.train_epoch(epochs=self.epochs_per_phase[2], desc="Phase 3")

        # Phase 4
        print(
            f"\n📌 Phase 4: Fine-tuning all layers ({self.epochs_per_phase[3]} epochs) - LR: {phase_lrs.get(3, self.lr)}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(3, self.lr))
        self.unfreeze_all()
        self.train_epoch(epochs=self.epochs_per_phase[3], desc="Phase 4")

        print("\n📥 Loading best model...")
        self.load(self.save_path)

        # ✅ ارزیابی روی TEST
        print("\n📊 Evaluating on TEST set (unseen data)...")
        self.evaluate_on_test()

        print("\n" + "=" * 70)
        print(f"✅ Training Complete!")
        print(f"   Best Validation Accuracy: {self.best_acc:.4f}")
        print(f"   Test Accuracy:            {getattr(self, 'test_accuracy', 0):.4f}")
        print("=" * 70)

        return self.best_acc

    # =====================================================
    # EVALUATION METHODS
    # =====================================================

    def evaluate(self, return_loss=False):
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():
            for x, y, _ in self.val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                out = self.model(x)
                loss = self.criterion(out, y)
                prob = torch.softmax(out, dim=1)
                pred = prob.argmax(dim=1)

                total_loss += loss.item()
                total_correct += (pred == y).sum().item()
                total_samples += y.size(0)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_prob.extend(prob.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        report, cm, auc_score = compute_metrics(y_true, y_pred, y_prob)

        self.metrics = report
        self.cm = cm
        self.auc_score = auc_score
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(self.val_loader)

        if return_loss:
            return avg_loss, accuracy

        return accuracy

    def evaluate_on_test(self):
        """Evaluate on the separate TEST set"""
        print("\n" + "=" * 70)
        print("📊 EVALUATING ON TEST SET (UNSEEN DATA)")
        print("=" * 70)

        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for x, y, _ in self.test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = self.model(x)
                loss = self.criterion(out, y)
                prob = torch.softmax(out, dim=1)
                pred = prob.argmax(dim=1)
                total_loss += loss.item()
                total_correct += (pred == y).sum().item()
                total_samples += y.size(0)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_prob.extend(prob.cpu().numpy())

        y_true = np.array(y_true);
        y_pred = np.array(y_pred);
        y_prob = np.array(y_prob)
        report, cm, auc_score = compute_metrics(y_true, y_pred, y_prob)

        test_accuracy = total_correct / total_samples
        test_loss = total_loss / len(self.test_loader)

        print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")
        print(f"✅ Test Loss:     {test_loss:.4f}")
        print(f"✅ Test AUC:      {auc_score:.4f}")
        print("=" * 70)

        self.test_metrics = report
        self.test_cm = cm
        self.test_auc = auc_score
        self.test_y_true = y_true
        self.test_y_pred = y_pred
        self.test_y_prob = y_prob
        self.test_accuracy = test_accuracy
        self.test_loss = test_loss

        return test_accuracy, report, cm, auc_score

    # =====================================================
    # LOAD METHODS
    # =====================================================

    def load(self, path=None):
        if path is None:
            path = self.save_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

        print(f"✅ Model loaded from: {path}")
        if "epoch" in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if "metrics" in checkpoint:
            print(f"   Metrics: {checkpoint['metrics']}")

        return self.model

    # =====================================================
    # REPORT GENERATION - FIXED 🔥
    # =====================================================

    def generate_report(self, output_path=None):
        """Generate a full report of the results"""
        if self.metrics is None:
            print("⚠️ No metrics available. Running evaluation...")
            self.evaluate()

        if output_path is None:
            reports_dir = RESNET_REPORTS_DIR
            os.makedirs(reports_dir, exist_ok=True)
            output_path = os.path.join(reports_dir, "ResNet18_Report.pdf")

        print(f"\n📄 Generating report: {output_path}")

        # ✅ FIX: best_acc should be from history if 0
        if self.best_acc == 0.0 and self.history['val_acc']:
            self.best_acc = max(self.history['val_acc'])
            self.best_epoch = self.history['val_acc'].index(self.best_acc)

        # ✅ ALWAYS GENERATE JSON FIRST
        json_path = os.path.join(RESNET_REPORTS_DIR, "resnet18_report.json")

        # ✅ انتخاب metrics برای JSON
        _metrics_for_json = getattr(self, 'test_metrics', None) or self.metrics
        _cm_for_json = getattr(self, 'test_cm', None)
        if _cm_for_json is None:
            _cm_for_json = self.cm
        _auc_for_json = getattr(self, 'test_auc', self.auc_score)

        report_json = {
            "model_info": {
                "model": "ResNet18",
                "version": "3.0.0",
                "class_names": CLASS_NAMES,
                "num_classes": len(CLASS_NAMES),
                "device": str(DEVICE),
                "created_at": datetime.now().isoformat(),
            },
            "training_summary": {
                "best_validation_accuracy": self.best_acc,
                "best_epoch": self.best_epoch + 1,
                "total_epochs": len(self.history["val_acc"]),
                "best_val_loss": min(self.history["val_loss"]) if self.history["val_loss"] else None,
                "best_train_loss": min(self.history["train_loss"]) if self.history["train_loss"] else None,
                "total_training_time": sum(self.history["epoch_times"]) if self.history["epoch_times"] else 0,
            },
            "performance_metrics": {
                "test_accuracy": getattr(self, 'test_accuracy', self.metrics.get("accuracy", 0)),
                "weighted_avg": {
                    "precision": (getattr(self, 'test_metrics', self.metrics)).get("weighted avg", {}).get("precision",
                                                                                                           0),
                    "recall": (getattr(self, 'test_metrics', self.metrics)).get("weighted avg", {}).get("recall", 0),
                    "f1_score": (getattr(self, 'test_metrics', self.metrics)).get("weighted avg", {}).get("f1-score",
                                                                                                          0),
                },
                "auc_score": getattr(self, 'test_auc', self.auc_score),
                "confusion_matrix": (
                    getattr(self, 'test_cm', self.cm).tolist()
                    if getattr(self, 'test_cm', self.cm) is not None else None
                ),
            },
            "per_class_metrics": {},
            "training_history": {
                "train_loss": self.history.get("train_loss", []),
                "val_loss": self.history.get("val_loss", []),
                "val_accuracy": self.history.get("val_acc", []),
                "learning_rates": self.history.get("learning_rates", []),
                "epoch_times": self.history.get("epoch_times", []),
            },
            "config": {
                "csv_path": self.csv_path,
                "img_dir": self.img_dir,
                "batch_size": self.batch_size,
                "base_lr": self.lr,
                "weight_decay": self.weight_decay,
                "target_size": self.target_size,
                "dropout_rate": self.dropout_rate,
                "label_smoothing": self.label_smoothing,
                "accumulation_steps": self.accumulation_steps,
                "warmup_epochs": self.warmup_epochs,
                "epochs_per_phase": self.epochs_per_phase,
                "use_sampler": self.use_sampler,
                "use_mixed_precision": self.use_mixed_precision,
            },
            "generated_at": datetime.now().isoformat(),
        }

        if _metrics_for_json:
            for class_name in CLASS_NAMES:
                if class_name in _metrics_for_json:
                    report_json["per_class_metrics"][class_name] = {
                        "precision": _metrics_for_json[class_name].get("precision", 0),
                        "recall": _metrics_for_json[class_name].get("recall", 0),
                        "f1_score": _metrics_for_json[class_name].get("f1-score", 0),
                        "support": _metrics_for_json[class_name].get("support", 0),
                    }

        with open(json_path, "w") as f:
            json.dump(report_json, f, indent=4, default=str)
        print(f"✅ JSON report saved: {json_path}")

        try:
            from .resnetReports import generate_full_pdf

            # ✅ انتخاب metrics: TEST اگر موجود باشه، وگرنه validation
            if hasattr(self, 'test_metrics') and self.test_metrics is not None:
                report_metrics = self.test_metrics
                report_cm = self.test_cm
                report_y_true = self.test_y_true
                report_y_prob = self.test_y_prob
                test_acc = getattr(self, 'test_accuracy', 0)
                print(f"📊 Using TEST metrics for PDF (Test Acc: {test_acc:.4f})")
            else:
                report_metrics = self.metrics
                report_cm = self.cm
                report_y_true = self.y_true
                report_y_prob = self.y_prob
                test_acc = self.metrics.get("accuracy", 0)
                print(f"⚠️ No test metrics available, using validation metrics")

            sample_img = None
            for batch in self.val_loader:
                if len(batch) == 3:
                    sample_img = batch[0][0]
                else:
                    sample_img = batch[0]
                break

            if sample_img is not None:
                generate_full_pdf(
                    metrics=report_metrics,          # ← ✅ TEST metrics
                    cm=report_cm,                    # ← ✅ TEST cm
                    y_true=report_y_true,            # ← ✅ TEST y_true
                    y_prob=report_y_prob,            # ← ✅ TEST y_prob
                    model=self.model,
                    sample_image_tensor=sample_img,
                    output_path=output_path,
                    model_name="ResNet18",
                    history=self.history,
                    class_names=CLASS_NAMES,
                    best_accuracy=self.best_acc,     # ← Validation (بهترین)
                    best_accuracy_label="Validation",
                    test_accuracy=test_acc,          # ← Test واقعی
                    config={
                        "batch_size": self.batch_size,
                        "learning_rate": self.lr,
                        "weight_decay": self.weight_decay,
                        "dropout": self.dropout_rate,
                        "label_smoothing": self.label_smoothing,
                        "epochs_per_phase": self.epochs_per_phase,
                        "use_sampler": self.use_sampler,
                    },
                )
                print(f"✅ PDF Report saved: {output_path}")
            else:
                print("⚠️ No sample image available for report")
        except Exception as e:
            print(f"⚠️ PDF generation failed: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 70)
        print("📊 RESNET18 FINAL RESULTS")
        print("=" * 70)
        test_acc_final = getattr(self, 'test_accuracy', self.metrics.get("accuracy", 0))
        print(f"   ✅ Best Validation Accuracy: {self.best_acc:.4f} ({self.best_acc * 100:.2f}%)")
        print(f"   ✅ Test Accuracy:             {test_acc_final:.4f} ({test_acc_final * 100:.2f}%)")
        print("=" * 70)

        # ✅ استفاده از test metrics برای summary
        summary_metrics = self.test_metrics if hasattr(self, 'test_metrics') and self.test_metrics else self.metrics
        summary_cm = self.test_cm if hasattr(self, 'test_cm') and self.test_cm is not None else self.cm
        print_metrics_summary(summary_metrics, summary_cm)
        return output_path

    # =====================================================
    # OTHER METHODS
    # =====================================================

    def get_metrics(self):
        if self.metrics is None:
            self.evaluate()
        return self.metrics, self.cm, self.y_true, self.y_prob

    def get_history(self):
        return self.history

    def get_model(self):
        return self.model


# =====================================================
# MAIN - FOR TESTING
# =====================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Testing ResNet18 Trainer (FULLY OPTIMIZED)")
    print("=" * 70)

    try:
        print("\n📌 Creating trainer with optimized settings...")

        trainer = ResNet18Trainer(
            num_workers=0, target_size=224, use_sampler=True, use_mixed_precision=True
        )

        print("\n📌 Testing quick training (1 epoch per phase)...")

        trainer.epochs_per_phase = [1, 1, 1, 1]
        trainer.freeze_backbone()
        trainer.train_epoch(epochs=1, desc="Test Phase 1")
        trainer.unfreeze_layer4()
        trainer.train_epoch(epochs=1, desc="Test Phase 2")
        trainer.unfreeze_layer3_4()
        trainer.train_epoch(epochs=1, desc="Test Phase 3")
        trainer.unfreeze_all()
        trainer.train_epoch(epochs=1, desc="Test Phase 4")

        print("\n📌 Testing evaluation...")
        acc = trainer.evaluate()
        print(f"✅ Evaluation accuracy: {acc:.4f}")

        print("\n📌 Testing report generation...")
        trainer.generate_report()

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
