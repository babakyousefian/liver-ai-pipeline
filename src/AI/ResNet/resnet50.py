# src/AI/ResNet/resnet50.py

"""
ResNet50 Trainer for Liver Disease Classification
This module is designed to train the ResNet50 model on liver image data
ResNet50 has 50 layers and is suitable for more complex and larger data
"""

import json
import math
import os
import random
import time
import warnings
from datetime import datetime

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
# IMPORTS FROM UTILS
# =====================================================
from .utils import HYPERPARAMS  # ✅ NEW
from .utils import LabelSmoothingCrossEntropy  # ✅ NEW
from .utils import WarmupCosineScheduler  # ✅ NEW
from .utils import apply_advanced_augmentation  # ✅ NEW
from .utils import (
    CLASS_NAMES,
    CROPPED,
    INV_LABEL_MAP,
    LABEL_MAP,
    OUTPUT_DIR,
    CSVAddress,
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

# =====================================================
# GLOBAL SETTINGS
# =====================================================
DEVICE = get_device()
torch.set_num_threads(4)

# ✅ Seed setting
set_seed(42)


# =====================================================
# RESNET50 DATASET - WITH ADVANCED AUGMENTATION 🔥
# =====================================================
class ResNet50Dataset:
    """
    ResNet50 dedicated dataset with normalization and advanced augmentation

    Args:
        df (pd.DataFrame): DataFrame containing liverName and liverLabel
        img_dir (str): Image directory path
        augment (bool): Apply augmentation
        target_size (int): Target image size
        apply_normalize (bool): Whether to apply normalization
        use_advanced_augment (bool): Whether to use advanced augmentation
    """

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

        # Remove UNKNOWN rows
        self.df = self.df[self.df["liverLabel"] != "UNKNOWN"]

        # ✅ Remove classes with very few samples
        class_counts = self.df["liverLabel"].value_counts()
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        self.df = self.df[self.df["liverLabel"].isin(valid_classes)]

        if len(self.df) == 0:
            raise ValueError("No classes with enough samples after filtering!")

        time.sleep(0.5)
        print(f"✅ ResNet50 Dataset initialized with {len(self.df)} samples")
        time.sleep(0.5)
        print(f"   Classes: {sorted(self.df['liverLabel'].unique())}")
        time.sleep(0.5)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]

            img_path = os.path.join(self.img_dir, row["liverName"])
            label = LABEL_MAP[row["liverLabel"]]

            # Reading image as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                time.sleep(0.5)
                print(f"⚠️ Image not found: {img_path}")
                time.sleep(0.5)
                img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            else:
                img = resize_with_padding(img, self.target_size)

            # ✅ Advanced Augmentation 🔥
            if self.augment:
                if self.use_advanced_augment:
                    img = apply_advanced_augmentation(img, use_erasing=True)
                else:
                    img = apply_augmentation(img)

            # ✅ Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0

            # ✅ Convert to 3 channels (ResNet50 needs 3 channels)
            img = np.stack([img, img, img], axis=0)  # (3, H, W)

            # ✅ Convert to tensor
            img_tensor = torch.tensor(img, dtype=torch.float32)

            # ✅ Apply normalization
            if self.apply_normalize:
                img_tensor = normalize_resnet(img_tensor)

            return (img_tensor, torch.tensor(label, dtype=torch.long), row["liverName"])

        except Exception as e:
            time.sleep(0.5)
            print(f"❌ Error in __getitem__ (idx={idx}): {e}")
            time.sleep(0.5)

            # Return an empty instance on error
            img = np.zeros((3, self.target_size, self.target_size), dtype=np.float32)
            img_tensor = torch.tensor(img, dtype=torch.float32)
            if self.apply_normalize:
                img_tensor = normalize_resnet(img_tensor)

            return (img_tensor, torch.tensor(0, dtype=torch.long), "error_image.png")


# =====================================================
# RESNET50 TRAINER - FULLY OPTIMIZED 🔥
# =====================================================
class ResNet50Trainer:
    """
    ResNet50 model trainer with all optimization techniques

    Features:
    - Label Smoothing
    - Warmup + Cosine Annealing
    - Gradient Accumulation
    - Advanced Augmentation
    - WeightedRandomSampler
    - Mixed Precision Training
    - Multi-stage Training (5 phases)
    """

    def __init__(
        self,
        csv_path=None,
        img_dir=None,
        batch_size=None,  # ✅ از HYPERPARAMS
        lr=None,  # ✅ از HYPERPARAMS
        weight_decay=None,  # ✅ از HYPERPARAMS
        num_workers=0,
        target_size=256,
        dropout_rate=None,  # ✅ از HYPERPARAMS
        use_mixed_precision=True,
        use_sampler=True,
        label_smoothing=None,  # ✅ از HYPERPARAMS
        accumulation_steps=None,  # ✅ از HYPERPARAMS
        warmup_epochs=None,  # ✅ از HYPERPARAMS
    ):
        # ✅ Load hyperparameters from config
        hp = HYPERPARAMS["resnet50"]

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
        self.epochs_per_phase = hp["epochs_per_phase"]  # [10, 15, 15, 15, 15]

        self.num_workers = num_workers
        self.target_size = target_size
        self.use_mixed_precision = use_mixed_precision
        self.use_sampler = use_sampler

        # ✅ نمایش تنظیمات
        time.sleep(0.5)
        print("\n" + "=" * 70)
        time.sleep(0.5)
        print("🚀 Initializing ResNet50 Trainer (OPTIMIZED)")
        time.sleep(0.5)
        print("=" * 70)
        time.sleep(0.5)
        print(f"📁 CSV Path: {self.csv_path}")
        time.sleep(0.5)
        print(f"📁 Image Dir: {self.img_dir}")
        time.sleep(0.5)
        print(f"📦 Batch Size: {self.batch_size}")
        time.sleep(0.5)
        print(f"📊 Learning Rate: {self.lr}")
        time.sleep(0.5)
        print(f"⚙️  Weight Decay: {self.weight_decay}")
        time.sleep(0.5)
        print(f"🎯 Dropout Rate: {self.dropout_rate}")
        time.sleep(0.5)
        print(f"🎯 Label Smoothing: {self.label_smoothing}")
        time.sleep(0.5)
        print(f"📊 Accumulation Steps: {self.accumulation_steps}")
        time.sleep(0.5)
        print(f"🔥 Warmup Epochs: {self.warmup_epochs}")
        time.sleep(0.5)
        print(f"📈 Epochs per Phase: {self.epochs_per_phase}")
        time.sleep(0.5)
        print(f"🖥️  Device: {DEVICE}")
        time.sleep(0.5)
        print(f"🔧 Mixed Precision: {self.use_mixed_precision}")
        time.sleep(0.5)
        print(f"🎯 Use Sampler: {self.use_sampler}")
        time.sleep(0.5)
        print("=" * 70)
        time.sleep(0.5)

        # Check paths
        if not check_paths():
            raise ValueError("Path validation failed!")

        # CSV validation
        is_valid, df, message = validate_csv(self.csv_path)
        if not is_valid:
            raise ValueError(f"CSV validation failed: {message}")
        self.df = df

        # Image validation
        total, missing, missing_list = validate_images(self.csv_path, self.img_dir)
        if missing > 0:
            time.sleep(0.5)
            print(f"⚠️ Warning: {missing} images are missing!")
            time.sleep(0.5)
            if len(missing_list) <= 10:
                time.sleep(0.5)
                print(f"   Missing: {missing_list}")
                time.sleep(0.5)

        # Loading and splitting data
        time.sleep(0.5)
        print("\n📊 Loading and splitting data...")
        time.sleep(0.5)
        self.train_ds, self.val_ds = self._load_data()

        # ✅ Create DataLoaders with Sampler
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()

        # Build model
        time.sleep(0.5)
        print("\n🏗️ Building ResNet50 model...")
        time.sleep(0.5)
        self.model = self._build_model()

        # ✅ Loss function with Label Smoothing 🔥
        time.sleep(0.5)
        print("\n⚖️ Setting up loss function...")
        time.sleep(0.5)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=self.label_smoothing)
        print(f"   ✅ Label Smoothing: {self.label_smoothing}")
        time.sleep(0.5)

        # ✅ Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # ✅ Warmup + Cosine Annealing Scheduler 🔥
        total_epochs = sum(self.epochs_per_phase)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=total_epochs,
            min_lr=1e-7,
            max_lr=self.lr,
        )
        print(f"   ✅ Warmup Cosine Scheduler: {total_epochs} total epochs")
        time.sleep(0.5)

        # ✅ Mixed precision scaler
        if self.use_mixed_precision and DEVICE.type == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
            if self.use_mixed_precision:
                time.sleep(0.5)
                print("⚠️ Mixed precision requested but CUDA not available. Disabling.")
                time.sleep(0.5)

        # Variables to store results
        self.best_acc = 0.0
        self.best_epoch = 0
        self.phase_lrs = None  # ✅ NEW: for per-phase learning rates
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

        # ✅ Storage paths
        self.base_dir = OUTPUT_DIR

        self.model_dir = os.path.join(self.base_dir, "resnet50")
        os.makedirs(self.model_dir, exist_ok=True)

        self.save_path = os.path.join(self.model_dir, "resnet50_best.pth")
        self.checkpoint_path = os.path.join(self.model_dir, "resnet50_checkpoint.pth")
        self.last_path = os.path.join(self.model_dir, "resnet50_last.pth")
        self.config_path = os.path.join(self.model_dir, "config_resnet50.json")

        self.reports_dir = os.path.join(self.base_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

        self._save_config()

        time.sleep(0.5)
        print(f"\n✅ ResNet50 Trainer initialized successfully!")
        time.sleep(0.5)
        print(f"📁 Model directory: {self.model_dir}")
        time.sleep(0.5)
        print(f"📁 Best model: {self.save_path}")
        time.sleep(0.5)
        print(f"📁 Checkpoint: {self.checkpoint_path}")
        time.sleep(0.5)
        print(f"📁 Last model: {self.last_path}")
        time.sleep(0.5)
        print(f"📁 Reports directory: {self.reports_dir}")
        time.sleep(0.5)
        print("=" * 70 + "\n")
        time.sleep(0.5)

    # =====================================================
    # PRIVATE METHODS
    # =====================================================

    def _load_data(self):
        """Loading and splitting data into Train/Val/Test (80/10/10)"""
        from sklearn.model_selection import train_test_split

        df_valid = self.df[self.df["liverLabel"] != "UNKNOWN"].copy()

        if len(df_valid) == 0:
            raise ValueError("No valid labels found after removing UNKNOWN!")

        # Remove classes with few samples
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

        # ✅ Step 1: Test split (10%)
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

        # ✅ Step 2: Validation split (10% of total = 1/9 of train_val)
        try:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=1 / 9,
                stratify=train_val_df["liverLabel"],
                random_state=42,
            )
        except ValueError as e:
            print(f"⚠️ Stratify failed (val split): {e}")
            train_df, val_df = train_test_split(
                train_val_df, test_size=1 / 9, random_state=42
            )

        # ✅ Create datasets
        train_ds = ResNet50Dataset(
            train_df,
            self.img_dir,
            augment=True,
            target_size=self.target_size,
            apply_normalize=True,
            use_advanced_augment=True,
        )
        val_ds = ResNet50Dataset(
            val_df,
            self.img_dir,
            augment=False,
            target_size=self.target_size,
            apply_normalize=True,
            use_advanced_augment=False,
        )
        test_ds = ResNet50Dataset(
            test_df,
            self.img_dir,
            augment=False,
            target_size=self.target_size,
            apply_normalize=True,
            use_advanced_augment=False,
        )

        total = len(df_valid)
        print(f"\n✅ Train: {len(train_ds)} samples ({len(train_ds) / total * 100:.1f}%)")
        print(f"✅ Val:   {len(val_ds)} samples ({len(val_ds) / total * 100:.1f}%)")
        print(f"✅ Test:  {len(test_ds)} samples ({len(test_ds) / total * 100:.1f}%)")

        # ✅ ذخیره test_ds
        self.test_ds = test_ds

        return train_ds, val_ds

    def _create_dataloaders(self):
        """✅ Create DataLoaders with WeightedRandomSampler + Test Loader"""

        train_sampler = None
        if self.use_sampler:
            print("\n🎯 Creating WeightedRandomSampler for training...")
            train_sampler = create_weighted_sampler(self.train_ds, self.csv_path)

        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True if DEVICE.type == "cuda" else False,
            drop_last=False,
        )

        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if DEVICE.type == "cuda" else False,
            drop_last=False,
        )

        # ✅ Test DataLoader
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if DEVICE.type == "cuda" else False,
            drop_last=False,
        )

        return train_loader, val_loader, test_loader

    def _build_model(self):
        """Building the ResNet50 model with optimized settings"""

        # Load model with ImageNet weights
        model = models.resnet50(weights="IMAGENET1K_V1")

        # ✅ First layer - 3 channels
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # ✅ Last layer with optimized dropout
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

        # Display model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        time.sleep(0.5)
        print(f"✅ Model built successfully")
        time.sleep(0.5)
        print(f"   Total parameters: {total_params:,}")
        time.sleep(0.5)
        print(f"   Trainable parameters: {trainable_params:,}")
        time.sleep(0.5)
        print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
        time.sleep(0.5)

        return model

    def _save_config(self):
        """Save initial settings in the correct path"""
        config = {
            "model": "ResNet50",
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

        time.sleep(0.5)
        print(f"✅ Config saved: {self.config_path}")
        time.sleep(0.5)

    def _set_phase_lr(self, new_lr: float):
        """✅ Set learning rate for the new phase"""
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
        """Freeze all backbone layers except the FC layer"""
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
            else:
                param.requires_grad = True

        time.sleep(0.5)
        print("✅ Backbone frozen (only FC layer trainable)")
        time.sleep(0.5)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        time.sleep(0.5)
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        time.sleep(0.5)

    def unfreeze_layer4(self):
        """Unfreeze layer4 and FC layer"""
        for name, param in self.model.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True

        time.sleep(0.5)
        print("✅ Layer4 and FC layer unfrozen")
        time.sleep(0.5)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        time.sleep(0.5)
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        time.sleep(0.5)

    def unfreeze_layer3_4(self):
        """Unfreeze layer3, layer4 and FC layer"""
        for name, param in self.model.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True

        time.sleep(0.5)
        print("✅ Layer3, Layer4 and FC layer unfrozen")
        time.sleep(0.5)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        time.sleep(0.5)
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        time.sleep(0.5)

    def unfreeze_layer2_3_4(self):
        """Unfreeze layer2, layer3, layer4 and FC layer"""
        for name, param in self.model.named_parameters():
            if (
                name.startswith("layer2")
                or name.startswith("layer3")
                or name.startswith("layer4")
                or name.startswith("fc")
            ):
                param.requires_grad = True

        time.sleep(0.5)
        print("✅ Layer2, Layer3, Layer4 and FC layer unfrozen")
        time.sleep(0.5)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        time.sleep(0.5)
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        time.sleep(0.5)

    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = True

        time.sleep(0.5)
        print("✅ All layers unfrozen (full fine-tuning)")
        time.sleep(0.5)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        time.sleep(0.5)
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        time.sleep(0.5)

    # =====================================================
    # TRAINING METHODS WITH GRADIENT ACCUMULATION 🔥
    # =====================================================

    def train_epoch(self, epochs=1, desc=""):
        """
        Train the model for a given number of epochs with gradient accumulation

        Args:
            epochs (int): number of epochs
            desc (str): description to display

        Returns:
            float: best accuracy
        """
        for epoch in range(epochs):
            epoch_start = time.time()

            # ===== TRAINING PHASE =====
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            # ✅ Reset gradients
            self.optimizer.zero_grad()

            loop = tqdm(self.train_loader, desc=f"[ResNet50] {desc} Epoch {epoch+1}/{epochs}")

            for batch_idx, (x, y, _) in enumerate(loop):
                x, y = x.to(DEVICE), y.to(DEVICE)

                if self.scaler is not None:
                    # Mixed precision training
                    with torch.amp.autocast('cuda'):
                        out = self.model(x)
                        loss = self.criterion(out, y)
                        loss = loss / self.accumulation_steps  # ✅ Scale loss

                    self.scaler.scale(loss).backward()

                    # ✅ Gradient accumulation step
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    # Standard training
                    out = self.model(x)
                    loss = self.criterion(out, y)
                    loss = loss / self.accumulation_steps  # ✅ Scale loss

                    loss.backward()

                    # ✅ Gradient accumulation step
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                # Calculate statistics
                total_loss += loss.item() * self.accumulation_steps  # ✅ Undo scaling
                pred = out.argmax(dim=1)
                total_correct += (pred == y).sum().item()
                total_samples += y.size(0)

                # Update progress bar
                loop.set_postfix(
                    {
                        "loss": f"{loss.item() * self.accumulation_steps:.4f}",
                        "acc": f"{total_correct/total_samples:.4f}",
                    }
                )

            # ✅ Handle remaining gradients
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

            # ===== VALIDATION PHASE =====
            val_loss, val_acc = self.evaluate(return_loss=True)

            # ===== UPDATE SCHEDULER =====
            current_lr = self.scheduler.step()

            # ===== SAVE HISTORY =====
            epoch_time = time.time() - epoch_start
            self.history["train_loss"].append(avg_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)
            self.history["epoch_times"].append(epoch_time)

            # ===== SAVE BEST MODEL =====
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                save_model(
                    self.model,
                    self.save_path,
                    self.optimizer,
                    epoch,
                    {"val_acc": val_acc, "val_loss": val_loss},
                )
                time.sleep(0.5)
                print(f"✅ Best model saved with accuracy: {val_acc:.4f}")
                time.sleep(0.5)

            # ===== SAVE CHECKPOINT =====
            if (epoch + 1) % 5 == 0:
                save_model(
                    self.model,
                    self.checkpoint_path,
                    self.optimizer,
                    epoch,
                    {"val_acc": val_acc, "val_loss": val_loss},
                )

            # ===== SAVE LAST MODEL =====
            save_model(
                self.model,
                self.last_path,
                self.optimizer,
                epoch,
                {"val_acc": val_acc, "val_loss": val_loss},
            )

            # ===== PRINT RESULTS =====
            time.sleep(0.5)
            print(
                f"📊 Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
            )
            time.sleep(0.5)

            # ===== EARLY STOPPING =====
            if len(self.history["val_acc"]) > 15:
                last_10 = self.history["val_acc"][-15:]
                if len(last_10) >= 15 and all(v <= last_10[0] for v in last_10[1:]):
                    time.sleep(0.5)
                    print(f"⚠️ Early stopping triggered (no improvement for 15 epochs)")
                    time.sleep(0.5)
                    break

        time.sleep(0.5)
        print(f"\n🔥 Best Accuracy: {self.best_acc:.4f} (Epoch {self.best_epoch+1})")
        time.sleep(0.5)

        return self.best_acc

    def train_full(self):
        """
        Full training with multi-stage strategy (OPTIMIZED):
            1. FC layer only training (Phase 1)
            2. FC + Layer4 (Phase 2)
            3. FC + Layer3 + Layer4 (Phase 3)
            4. FC + Layer2 + Layer3 + Layer4 (Phase 4)
            5. Full fine-tuning (Phase 5)
        """
        print("\n" + "=" * 70)
        print("🚀 STARTING FULL TRAINING PIPELINE (ResNet50 - OPTIMIZED)")
        print("=" * 70)

        # ✅ Phase learning rates
        phase_lrs = self.phase_lrs if self.phase_lrs else {
            0: self.lr, 1: self.lr, 2: self.lr, 3: self.lr, 4: self.lr
        }

        # Phase 1: FC layer only
        print(
            f"\n📌 Phase 1: Training FC layer only ({self.epochs_per_phase[0]} epochs) - LR: {phase_lrs.get(0, self.lr):.6f}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(0, self.lr))
        self.freeze_backbone()
        self.train_epoch(epochs=self.epochs_per_phase[0], desc="Phase 1")

        # Phase 2: FC + Layer4
        print(
            f"\n📌 Phase 2: Training Layer4 + FC ({self.epochs_per_phase[1]} epochs) - LR: {phase_lrs.get(1, self.lr):.6f}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(1, self.lr))
        self.unfreeze_layer4()
        self.train_epoch(epochs=self.epochs_per_phase[1], desc="Phase 2")

        # Phase 3: FC + Layer3 + Layer4
        print(
            f"\n📌 Phase 3: Training Layer3 + Layer4 + FC ({self.epochs_per_phase[2]} epochs) - LR: {phase_lrs.get(2, self.lr):.6f}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(2, self.lr))
        self.unfreeze_layer3_4()
        self.train_epoch(epochs=self.epochs_per_phase[2], desc="Phase 3")

        # Phase 4: FC + Layer2 + Layer3 + Layer4
        print(
            f"\n📌 Phase 4: Training Layer2 + Layer3 + Layer4 + FC ({self.epochs_per_phase[3]} epochs) - LR: {phase_lrs.get(3, self.lr):.6f}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(3, self.lr))
        self.unfreeze_layer2_3_4()
        self.train_epoch(epochs=self.epochs_per_phase[3], desc="Phase 4")

        # Phase 5: All layers
        print(
            f"\n📌 Phase 5: Fine-tuning all layers ({self.epochs_per_phase[4]} epochs) - LR: {phase_lrs.get(4, self.lr):.6f}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(4, self.lr))
        self.unfreeze_all()
        self.train_epoch(epochs=self.epochs_per_phase[4], desc="Phase 5")

        # Loading the best model
        print("\n📥 Loading best model...")
        self.load(self.save_path)

        # ✅ Evaluate on TEST set
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
        """
        Evaluate the model on the validation set

        Args:
            return_loss (bool): whether to return loss

        Returns:
            tuple: (loss, accuracy) if return_loss=True
            float: accuracy if return_loss=False
        """
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

        # Calculating metrics
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

    # =====================================================
    # REST OF METHODS (evaluate_on_test, predict, load, etc.)
    # =====================================================

    def evaluate_on_test(self):
        """Evaluate on the separate TEST set (only once at the end)"""
        print("\n" + "=" * 70)
        print("📊 EVALUATING ON TEST SET (UNSEEN DATA)")
        print("=" * 70)

        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        y_true = []
        y_pred = []
        y_prob = []

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

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        report, cm, auc_score = compute_metrics(y_true, y_pred, y_prob)

        test_accuracy = total_correct / total_samples
        test_loss = total_loss / len(self.test_loader)

        print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")
        print(f"✅ Test Loss:     {test_loss:.4f}")
        print(f"✅ Test AUC:      {auc_score:.4f}")
        print("=" * 70)

        # Store test results
        self.test_metrics = report
        self.test_cm = cm
        self.test_auc = auc_score
        self.test_y_true = y_true
        self.test_y_pred = y_pred
        self.test_y_prob = y_prob
        self.test_accuracy = test_accuracy
        self.test_loss = test_loss

        return test_accuracy, report, cm, auc_score

    def predict(self, image_tensor):
        """Prediction on an image"""
        self.model.eval()

        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
        elif image_tensor.dim() == 3 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        elif image_tensor.dim() == 3 and image_tensor.shape[0] != 3:
            image_tensor = image_tensor.unsqueeze(0)

        if image_tensor.max() <= 1.0:
            image_tensor = normalize_resnet(image_tensor)

        image_tensor = image_tensor.to(DEVICE)

        with torch.no_grad():
            out = self.model(image_tensor.unsqueeze(0))
            prob = torch.softmax(out, dim=1)
            pred = prob.argmax(dim=1).item()
            confidence = prob[0][pred].item()

        predicted_label = INV_LABEL_MAP[pred]

        probabilities = {label: prob[0][i].item() for i, label in enumerate(CLASS_NAMES)}

        return predicted_label, probabilities, confidence

    def predict_batch(self, image_tensors):
        """Prediction on a set of images"""
        self.model.eval()

        if image_tensors.dim() == 3:
            image_tensors = image_tensors.unsqueeze(0)

        if image_tensors.max() <= 1.0:
            image_tensors = normalize_resnet(image_tensors)

        image_tensors = image_tensors.to(DEVICE)

        results = []
        with torch.no_grad():
            out = self.model(image_tensors)
            prob = torch.softmax(out, dim=1)
            preds = prob.argmax(dim=1)
            confidences = prob.max(dim=1)[0]

            for i in range(len(preds)):
                label = INV_LABEL_MAP[preds[i].item()]
                confidence = confidences[i].item()
                results.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "probabilities": {
                            label: prob[i][j].item() for j, label in enumerate(CLASS_NAMES)
                        },
                    }
                )

        return results

    def predict_on_folder(self, folder_path, output_csv=None):
        """Prediction on all images in a folder"""
        time.sleep(0.5)
        print(f"\n🔍 Predicting on folder: {folder_path}")
        time.sleep(0.5)

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        image_files = [
            f
            for f in os.listdir(folder_path)
            if any(f.lower().endswith(ext) for ext in image_extensions)
        ]

        if not image_files:
            time.sleep(0.5)
            print("❌ No images found in folder!")
            time.sleep(0.5)
            return pd.DataFrame()

        results = []
        self.model.eval()

        with torch.no_grad():
            for img_file in tqdm(image_files, desc="Predicting"):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    time.sleep(0.5)
                    print(f"⚠️ Could not read: {img_file}")
                    time.sleep(0.5)
                    continue

                img = resize_with_padding(img, self.target_size)
                img = img.astype(np.float32) / 255.0
                img = np.stack([img, img, img], axis=0)
                img_tensor = torch.tensor(img, dtype=torch.float32)

                img_tensor = normalize_resnet(img_tensor)
                img_tensor = img_tensor.to(DEVICE)

                out = self.model(img_tensor.unsqueeze(0))
                prob = torch.softmax(out, dim=1)
                pred = prob.argmax(dim=1).item()
                confidence = prob[0][pred].item()

                results.append(
                    {
                        "image_name": img_file,
                        "predicted_label": INV_LABEL_MAP[pred],
                        "confidence": confidence,
                        **{label: prob[0][i].item() for i, label in enumerate(CLASS_NAMES)},
                    }
                )

        df = pd.DataFrame(results)

        if output_csv:
            os.makedirs(
                os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".",
                exist_ok=True,
            )
            df.to_csv(output_csv, index=False)
            time.sleep(0.5)
            print(f"✅ Results saved: {output_csv}")
            time.sleep(0.5)

        time.sleep(0.5)
        print(f"\n✅ Predictions completed on {len(df)} images")
        time.sleep(0.5)

        return df

    def load(self, path=None):
        """Load model from file"""
        if path is None:
            path = self.save_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

        time.sleep(0.5)
        print(f"✅ Model loaded from: {path}")
        time.sleep(0.5)

        if "epoch" in checkpoint:
            time.sleep(0.5)
            print(f"   Epoch: {checkpoint['epoch']}")
            time.sleep(0.5)

        if "metrics" in checkpoint:
            time.sleep(0.5)
            print(f"   Metrics: {checkpoint['metrics']}")
            time.sleep(0.5)

        return self.model

    def load_best(self):
        return self.load(self.save_path)

    def load_checkpoint(self):
        return self.load(self.checkpoint_path)

    def load_last(self):
        return self.load(self.last_path)

    def generate_report(self, output_path=None):
        """Generate a full report of the results"""
        if self.metrics is None:
            time.sleep(0.5)
            print("⚠️ No metrics available. Running evaluation...")
            time.sleep(0.5)
            self.evaluate()

        if output_path is None:
            output_path = os.path.join(self.reports_dir, "ResNet50_Report.pdf")

        # ✅ ============================================
        # ✅ FIX: اطمینان از مقداردهی best_acc
        # ✅ ============================================
        if self.best_acc == 0.0 and hasattr(self, 'history') and self.history['val_acc']:
            self.best_acc = max(self.history['val_acc'])
            self.best_epoch = self.history['val_acc'].index(self.best_acc)
    
        # ✅ ============================================
        # ✅ ALWAYS GENERATE JSON (BEFORE PDF)
        # ✅ ============================================
        json_path = os.path.join(self.reports_dir, "resnet50_report.json")
    
        # ✅ Get class weights
        class_weights = self._get_class_weights() if hasattr(self, '_get_class_weights') else None

        # ✅ انتخاب metrics برای JSON
        _metrics_for_json = getattr(self, 'test_metrics', None) or self.metrics
        _cm_for_json = getattr(self, 'test_cm', None)
        if _cm_for_json is None:
            _cm_for_json = self.cm
        _auc_for_json = getattr(self, 'test_auc', self.auc_score)

        # ✅ Build JSON report
        report_json = {
            # ============================================
            # 1. MODEL INFORMATION
            # ============================================
            "model_info": {
                "model": "ResNet50",
                "version": "1.0.0",
                "class_names": CLASS_NAMES,
                "num_classes": len(CLASS_NAMES),
                "device": str(DEVICE),
                "created_at": datetime.now().isoformat(),
            },
        
            # ============================================
            # 2. TRAINING SUMMARY
            # ============================================
            "training_summary": {
                "best_validation_accuracy": self.best_acc,  # ✅ Validation Accuracy
                "best_epoch": self.best_epoch + 1,
                "total_epochs": len(self.history["val_acc"]),
                "best_val_loss": min(self.history["val_loss"]) if self.history["val_loss"] else None,
                "best_train_loss": min(self.history["train_loss"]) if self.history["train_loss"] else None,
                "total_training_time": sum(self.history["epoch_times"]) if self.history["epoch_times"] else 0,
                "early_stopped": len(self.history["val_acc"]) < sum(self.epochs_per_phase),
            },
        
            # ============================================
            # 3. PERFORMANCE METRICS (TEST)
            # ============================================

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
        
            # ============================================
            # 4. PER-CLASS METRICS
            # ============================================
            "per_class_metrics": {},
            "class_distribution": class_weights if class_weights else {},
        
            # ============================================
            # 5. TRAINING HISTORY
            # ============================================
            "training_history": {
                "train_loss": self.history.get("train_loss", []),
                "val_loss": self.history.get("val_loss", []),
                "val_accuracy": self.history.get("val_acc", []),
                "learning_rates": self.history.get("learning_rates", []),
                "epoch_times": self.history.get("epoch_times", []),
            },
        
            # ============================================
            # 6. CONFIGURATION
            # ============================================
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
                "num_workers": self.num_workers,
            },
        
            # ============================================
            # 7. STATISTICS
            # ============================================
            "statistics": {
                "train_samples": len(self.train_ds) if hasattr(self, 'train_ds') else 0,
                "val_samples": len(self.val_ds) if hasattr(self, 'val_ds') else 0,
                "total_samples": len(self.train_ds) + len(self.val_ds) if hasattr(self, 'train_ds') and hasattr(self, 'val_ds') else 0,
                "class_weights": class_weights,
            },
        
            # ============================================
            # 8. IMPROVEMENT SUGGESTIONS (Automated)
            # ============================================
            "suggestions": self._generate_suggestions() if hasattr(self, '_generate_suggestions') else [],
        
            # ============================================
            # 9. TIMESTAMP
            # ============================================
            "generated_at": datetime.now().isoformat(),
        }
    
        # ✅ Fill per-class metrics
        if _metrics_for_json:
            for class_name in CLASS_NAMES:
                if class_name in _metrics_for_json:
                    report_json["per_class_metrics"][class_name] = {
                        "precision": _metrics_for_json[class_name].get("precision", 0),
                        "recall": _metrics_for_json[class_name].get("recall", 0),
                        "f1_score": _metrics_for_json[class_name].get("f1-score", 0),
                        "support": _metrics_for_json[class_name].get("support", 0),
                    }
    
        # ✅ Save JSON
        with open(json_path, "w") as f:
            json.dump(report_json, f, indent=4, default=str)
    
        time.sleep(0.5)
        print(f"✅ JSON report saved: {json_path}")
        time.sleep(0.5)

        # ============================================
        # ✅ GENERATE PDF (ALWAYS) — با TEST metrics
        # ============================================
        from .resnetReports import generate_full_pdf

        # ✅ انتخاب metrics: TEST اگر موجود باشه
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
            print(f"⚠️ No test metrics, using validation metrics")

        sample_img = None
        for batch in self.val_loader:
            sample_img = batch[0][0]
            break

        if sample_img is not None:
            generate_full_pdf(
                metrics=report_metrics,
                cm=report_cm,
                y_true=report_y_true,
                y_prob=report_y_prob,
                model=self.model,
                sample_image_tensor=sample_img,
                output_path=output_path,
                model_name="ResNet50",
                history=self.history,
                class_names=CLASS_NAMES,
                best_accuracy=self.best_acc,
                best_accuracy_label="Validation",
                test_accuracy=test_acc,
            )
        else:
            time.sleep(0.5)
            print(f"⚠️ No sample image for PDF. JSON saved at: {json_path}")
            time.sleep(0.5)

        # ✅ Print summary با TEST metrics
        summary_metrics = self.test_metrics if hasattr(self, 'test_metrics') and self.test_metrics else self.metrics
        summary_cm = self.test_cm if hasattr(self, 'test_cm') and self.test_cm is not None else self.cm

        print("\n" + "=" * 70)
        print("📊 RESNET50 FINAL RESULTS")
        print("=" * 70)
        print(f"   ✅ Best Validation Accuracy: {self.best_acc:.4f} ({self.best_acc*100:.2f}%)")
        print(f"   ✅ Test Accuracy:             {test_acc:.4f} ({test_acc * 100:.2f}%)")
        print("=" * 70)
        time.sleep(0.5)

        print_metrics_summary(summary_metrics, summary_cm)
        time.sleep(0.5)

        return output_path

    def _get_class_weights(self):
        """Get class weights for imbalance analysis."""
        if not hasattr(self, 'train_ds'):
            return None
    
        class_counts = {}
        for idx in range(len(self.train_ds)):
            _, label, _ = self.train_ds[idx]
            class_name = CLASS_NAMES[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
        total = sum(class_counts.values())
        weights = {cls: count/total for cls, count in class_counts.items()}
        return weights

    def _generate_suggestions(self):
        """Generate automated improvement suggestions."""
        suggestions = []
    
        if self.metrics:
            acc = self.metrics.get('accuracy', 0)
        
            # Accuracy suggestions
            if acc < 0.7:
                suggestions.append("⚠️ Accuracy is below 70%. Consider:")
                suggestions.append("   - Increase epochs")
                suggestions.append("   - Reduce learning rate")
                suggestions.append("   - Use stronger augmentation")
            elif acc < 0.8:
                suggestions.append("📈 Accuracy is between 70-80%. Try:")
                suggestions.append("   - Add more data augmentation")
                suggestions.append("   - Increase model capacity")
                suggestions.append("   - Fine-tune for more epochs")
            else:
                suggestions.append("✅ Accuracy is above 80%. Good performance!")
                suggestions.append("   - Consider model ensemble for further improvement")
                suggestions.append("   - Try cross-validation for stability")
        
            # Class imbalance suggestions
            if hasattr(self, 'metrics') and 'per_class' in self.metrics:
                for cls, metrics in self.metrics.get('per_class', {}).items():
                    if metrics.get('recall', 0) < 0.5:
                        suggestions.append(f"⚠️ Class '{cls}' has low recall ({metrics.get('recall', 0):.2f})")
                        suggestions.append(f"   - Add more samples for class '{cls}'")
                        suggestions.append(f"   - Increase augmentation for class '{cls}'")
    
        # Overfitting suggestions
        if self.history and len(self.history.get('val_acc', [])) > 10:
            train_acc = self.history['val_acc'][-1] if self.history['val_acc'] else 0
            val_acc = self.history['val_acc'][-1] if self.history['val_acc'] else 0
            if train_acc - val_acc > 0.15:
                suggestions.append("⚠️ Possible overfitting detected")
                suggestions.append("   - Increase dropout rate")
                suggestions.append("   - Add more regularization")
                suggestions.append("   - Reduce model complexity")
    
        return suggestions

    def export_predictions(self, output_path=None):
        """Save prediction results"""
        if self.y_true is None or self.y_pred is None:
            time.sleep(0.5)
            print("⚠️ No predictions available. Running evaluation...")
            time.sleep(0.5)
            self.evaluate()

        if output_path is None:
            output_path = os.path.join(self.reports_dir, "resnet50_predictions.csv")

        export_predictions(self.y_true, self.y_pred, self.y_prob, CLASS_NAMES, output_path)

    # =====================================================
    # COMPARISON METHODS
    # =====================================================

    def compare_with_resnet18(self, resnet18_trainer, output_path=None):
        """Compare ResNet50 with ResNet18"""
        metrics50, cm50, y_true50, y_prob50 = self.get_metrics()
        metrics18, cm18, y_true18, y_prob18 = resnet18_trainer.get_metrics()

        comparison = {
            "model_comparison": {
                "ResNet50": {
                    "accuracy": metrics50.get("accuracy", 0),
                    "precision": metrics50.get("weighted avg", {}).get("precision", 0),
                    "recall": metrics50.get("weighted avg", {}).get("recall", 0),
                    "f1_score": metrics50.get("weighted avg", {}).get("f1-score", 0),
                    "auc_score": self.auc_score or 0,
                },
                "ResNet18": {
                    "accuracy": metrics18.get("accuracy", 0),
                    "precision": metrics18.get("weighted avg", {}).get("precision", 0),
                    "recall": metrics18.get("weighted avg", {}).get("recall", 0),
                    "f1_score": metrics18.get("weighted avg", {}).get("f1-score", 0),
                    "auc_score": resnet18_trainer.auc_score or 0,
                },
            },
            "class_names": CLASS_NAMES,
            "created_at": datetime.now().isoformat(),
        }

        if output_path is None:
            output_path = os.path.join(self.reports_dir, "resnet_comparison.json")

        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=4)

        time.sleep(0.5)
        print("\n" + "=" * 70)
        time.sleep(0.5)
        print("📊 COMPARISON: ResNet50 vs ResNet18")
        time.sleep(0.5)
        print("=" * 70)
        time.sleep(0.5)

        time.sleep(0.5)
        print(
            f"\n{'Metric':<15} {'ResNet50':<12} {'ResNet18':<12} {'Difference':<12} {'Status':<10}"
        )
        time.sleep(0.5)
        print("-" * 70)
        time.sleep(0.5)

        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            val50 = comparison["model_comparison"]["ResNet50"][metric]
            val18 = comparison["model_comparison"]["ResNet18"][metric]
            diff = val50 - val18

            if diff > 0.01:
                status = "✅ Better"
            elif diff < -0.01:
                status = "⚠️ Worse"
            else:
                status = "➖ Equal"

            time.sleep(0.5)
            print(
                f"{metric.capitalize():<15} {val50:<12.4f} {val18:<12.4f} {diff:+.4f}     {status}"
            )
            time.sleep(0.5)

        time.sleep(0.5)
        print(
            f"\nAUC Score:   {comparison['model_comparison']['ResNet50']['auc_score']:.4f} vs {comparison['model_comparison']['ResNet18']['auc_score']:.4f}"
        )
        time.sleep(0.5)
        print("=" * 70)
        time.sleep(0.5)

        conclusion = {
            "better_model": (
                "ResNet50"
                if comparison["model_comparison"]["ResNet50"]["accuracy"]
                > comparison["model_comparison"]["ResNet18"]["accuracy"]
                else "ResNet18"
            ),
            "accuracy_improvement": comparison["model_comparison"]["ResNet50"]["accuracy"]
            - comparison["model_comparison"]["ResNet18"]["accuracy"],
            "recommendation": (
                "ResNet50"
                if comparison["model_comparison"]["ResNet50"]["accuracy"]
                > comparison["model_comparison"]["ResNet18"]["accuracy"] + 0.01
                else "ResNet18"
            ),
        }

        comparison["conclusion"] = conclusion

        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=4)

        time.sleep(0.5)
        print(f"\n✅ Comparison saved to: {output_path}")
        time.sleep(0.5)

        return comparison

    # =====================================================
    # GETTER METHODS
    # =====================================================

    def get_metrics(self):
        if self.metrics is None:
            self.evaluate()
        return self.metrics, self.cm, self.y_true, self.y_prob

    def get_history(self):
        return self.history

    def get_model(self):
        return self.model

    def get_device(self):
        return DEVICE

    def get_class_names(self):
        return CLASS_NAMES

    def get_num_classes(self):
        return len(CLASS_NAMES)


# =====================================================
# MAIN - FOR TESTING
# =====================================================
if __name__ == "__main__":
    time.sleep(0.5)
    print("=" * 70)
    time.sleep(0.5)
    print("🧪 Testing ResNet50 Trainer (FULLY OPTIMIZED)")
    time.sleep(0.5)
    print("=" * 70)
    time.sleep(0.5)

    try:
        time.sleep(0.5)
        print("\n📌 Creating trainer with optimized settings...")
        time.sleep(0.5)

        trainer = ResNet50Trainer(
            num_workers=0, target_size=256, use_sampler=True, use_mixed_precision=True
        )

        # Quick training test (1 epoch per phase)
        time.sleep(0.5)
        print("\n📌 Testing quick training (1 epoch per phase)...")
        time.sleep(0.5)

        trainer.epochs_per_phase = [1, 1, 1, 1, 1]
        trainer.freeze_backbone()
        trainer.train_epoch(epochs=1, desc="Test Phase 1")
        trainer.unfreeze_layer4()
        trainer.train_epoch(epochs=1, desc="Test Phase 2")
        trainer.unfreeze_layer3_4()
        trainer.train_epoch(epochs=1, desc="Test Phase 3")
        trainer.unfreeze_layer2_3_4()
        trainer.train_epoch(epochs=1, desc="Test Phase 4")
        trainer.unfreeze_all()
        trainer.train_epoch(epochs=1, desc="Test Phase 5")

        # Assessment test
        time.sleep(0.5)
        print("\n📌 Testing evaluation...")
        time.sleep(0.5)
        acc = trainer.evaluate()
        time.sleep(0.5)
        print(f"✅ Evaluation accuracy: {acc:.4f}")
        time.sleep(0.5)

        # Prediction test
        time.sleep(0.5)
        print("\n📌 Testing prediction...")
        time.sleep(0.5)
        sample_img = np.ones((256, 256), dtype=np.uint8) * 128
        sample_img = np.stack([sample_img, sample_img, sample_img], axis=0)
        sample_img = torch.tensor(sample_img, dtype=torch.float32) / 255.0
        label, probs, conf = trainer.predict(sample_img)
        time.sleep(0.5)
        print(f"✅ Prediction: {label}, Confidence: {conf:.4f}")
        time.sleep(0.5)

        # Test saving the report
        time.sleep(0.5)
        print("\n📌 Testing report generation...")
        time.sleep(0.5)
        trainer.generate_report()

        time.sleep(0.5)
        print("\n✅ All tests passed!")
        time.sleep(0.5)

    except Exception as e:
        time.sleep(0.5)
        print(f"\n❌ Test failed: {e}")
        time.sleep(0.5)
        import traceback

        traceback.print_exc()
