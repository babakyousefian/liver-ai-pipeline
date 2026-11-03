# src/AI/CNNLSTM/CNNLSTM.py
# -*- coding: utf-8 -*-

"""
CNN+LSTM (Spatial-Temporal) Trainer for Liver Disease Classification
=====================================================================
Two-Stage Fine-Tuning: Freeze backbone -> Unfreeze all

Architecture:
- CNN Backbone: Extracts spatial features from each patch of the image
- LSTM Head: Architects the temporal/sequential relationship between patches

Author: Babak Yousefian
"""

import json
import os
import time
import warnings
from datetime import datetime
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from .CNNLSTMReports import CNNLSTMReporter

warnings.filterwarnings("ignore")

from src.AI.ResNet.utils import (
    CLASS_NAMES,
    CROPPED,
    CSVAddress,
    OUTPUT_DIR,
    LabelSmoothingCrossEntropy,
    WarmupCosineScheduler,
    check_paths,
    compute_metrics,
    create_weighted_sampler,
    get_class_weights,
    get_device,
    set_seed,
    validate_csv,
    validate_images,
)
from src.AI.ResNet.resnet18 import save_model_checkpoint

# =====================================================
# GLOBAL SETTINGS
# =====================================================
DEVICE = get_device()
torch.set_num_threads(4)
set_seed(42)

# ✅ CNN+LSTM output dir
CNNLSTM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "..", "CNNLSTM")
CNNLSTM_OUTPUT_DIR = os.path.abspath(CNNLSTM_OUTPUT_DIR)


# =====================================================
# CUSTOM MODEL DEFINITION (CNN+LSTM)
# =====================================================
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.2):
        super(CNNLSTM, self).__init__()

        # 1. CNN Backbone (Feature Extractor)
        # استفاده از ResNet18 (بدون لایه‌های آخر)
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # خروجی: (batch, 512, 7, 7)

        # 2. LSTM
        self.lstm_input_size = 512
        self.lstm_hidden_size = 256
        self.lstm_num_layers = 2

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if self.lstm_num_layers > 1 else 0
        )

        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.lstm_hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        features = self.cnn(x)  # (batch, 512, 7, 7)

        # تبدیل به دنباله: (batch, seq_len, features)
        # seq_len = 7*7 = 49 (پچ‌ها)
        batch_size, channels, h, w = features.size()
        features = features.view(batch_size, channels, h * w)
        features = features.permute(0, 2, 1)  # (batch, 49, 512)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)

        # استفاده از آخرین خروجی LSTM
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Classifier
        out = self.classifier(last_out)
        return out


# =====================================================
# EARLY STOPPING
# =====================================================
class EarlyStopping:
    def __init__(self, patience=12, min_delta=1e-4, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, score, epoch, model):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            if self.verbose:
                print(f"  ✅ EarlyStopping: improved to {score:.4f} (epoch {epoch + 1})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ⏳ EarlyStopping: no improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n⚠️ EarlyStopping triggered at epoch {epoch + 1}")
                    print(f"   Best score: {self.best_score:.4f} at epoch {self.best_epoch + 1}")
        return self.early_stop

    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"  ✅ Restored best weights from epoch {self.best_epoch + 1}")


# =====================================================
# CNN+LSTM TRAINER
# =====================================================
class CNNLSTMTrainer:
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
        # ✅ CNN+LSTM hyperparams (Optimized for Gap and Accuracy)
        DEFAULT_HP = {
            "batch_size": 16,  # ← کمتر برای LSTM (VRAM)
            "lr": 1e-4,
            "weight_decay": 1e-3,  # ← بهینه
            "dropout": 0.2,  # ← بهینه
            "label_smoothing": 0.1,  # ← استاندارد
            "accumulation_steps": 2,  # ← بیشتر برای batch کوچیک
            "warmup_epochs": 8,
            "epochs_per_phase": [5, 30],
        }
        hp = DEFAULT_HP

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
        self.phase_lrs = None

        print("\n" + "=" * 70)
        print("🚀 Initializing CNN+LSTM Trainer (TWO-STAGE)")
        print("=" * 70)
        print(f"📁 CSV: {self.csv_path}")
        print(f"📁 IMG: {self.img_dir}")
        print(f"📦 Batch: {self.batch_size} | Acc: {self.accumulation_steps}")
        print(f"📊 LR: {self.lr} | WD: {self.weight_decay}")
        print(f"🎯 Dropout: {self.dropout_rate} | LS: {self.label_smoothing}")
        print(f"📈 Epochs: {self.epochs_per_phase}")
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
            print(f"⚠️ {missing} images missing!")

        print("\n📊 Loading and splitting data...")
        self.train_ds, self.val_ds = self._load_data()
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()

        print("\n🏗️ Building CNN+LSTM model...")
        self.model = self._build_model()

        print("\n⚖️ Setting up loss function...")
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
            min_lr=1e-6,
            max_lr=self.lr,
        )
        print(f"   ✅ Warmup Cosine Scheduler: {total_epochs} epochs")

        if self.use_mixed_precision and DEVICE.type == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

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

        self.test_metrics = None
        self.test_cm = None
        self.test_auc = None
        self.test_y_true = None
        self.test_y_pred = None
        self.test_y_prob = None
        self.test_accuracy = 0.0
        self.test_loss = 0.0

        # ✅ Storage paths
        self.base_dir = CNNLSTM_OUTPUT_DIR
        self.model_dir = os.path.join(self.base_dir, "CNNLSTM")
        os.makedirs(self.model_dir, exist_ok=True)

        self.save_path = os.path.join(self.model_dir, "CNNLSTM_best.pth")
        self.checkpoint_path = os.path.join(self.model_dir, "CNNLSTM_checkpoint.pth")
        self.last_path = os.path.join(self.model_dir, "CNNLSTM_last.pth")
        self.config_path = os.path.join(self.model_dir, "config_CNNLSTM.json")

        self._save_config()

        print(f"\n✅ CNN+LSTM Trainer initialized!")
        print(f"📁 Model dir: {self.model_dir}")
        print("=" * 70 + "\n")

    # =====================================================
    # PRIVATE
    # =====================================================

    def _load_data(self):
        from sklearn.model_selection import train_test_split
        from src.AI.ResNet.utils import LiverDataset

        df_valid = self.df[self.df["liverLabel"] != "UNKNOWN"].copy()
        if len(df_valid) == 0:
            raise ValueError("No valid labels!")

        class_counts = df_valid["liverLabel"].value_counts()
        valid_classes = class_counts[class_counts >= 2].index.tolist()
        df_valid = df_valid[df_valid["liverLabel"].isin(valid_classes)]

        print("\n📊 Label Distribution:")
        for label in sorted(df_valid["liverLabel"].unique()):
            count = len(df_valid[df_valid["liverLabel"] == label])
            print(f"   {label}: {count}")

        try:
            train_val_df, test_df = train_test_split(
                df_valid, test_size=0.10,
                stratify=df_valid["liverLabel"], random_state=42,
            )
        except ValueError:
            train_val_df, test_df = train_test_split(
                df_valid, test_size=0.10, random_state=42
            )

        try:
            train_df, val_df = train_test_split(
                train_val_df, test_size=1 / 9,
                stratify=train_val_df["liverLabel"], random_state=42,
            )
        except ValueError:
            train_df, val_df = train_test_split(
                train_val_df, test_size=1 / 9, random_state=42
            )

        train_ds = LiverDataset(
            train_df, self.img_dir, augment=True,
            target_size=self.target_size, apply_normalize=True,
            use_advanced_augment=True, return_path=True,
        )
        val_ds = LiverDataset(
            val_df, self.img_dir, augment=False,
            target_size=self.target_size, apply_normalize=True,
            use_advanced_augment=False, return_path=True,
        )
        test_ds = LiverDataset(
            test_df, self.img_dir, augment=False,
            target_size=self.target_size, apply_normalize=True,
            use_advanced_augment=False, return_path=True,
        )

        print(f"\n✅ Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

        self.test_ds = test_ds
        return train_ds, val_ds

    def _create_dataloaders(self):
        train_sampler = None
        if self.use_sampler:
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
        model = CNNLSTM(num_classes=len(CLASS_NAMES), dropout_rate=self.dropout_rate)
        model = model.to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✅ CNN+LSTM built")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")

        return model

    def _save_config(self):
        config = {
            "model": "CNNLSTM",
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
            "effective_batch_size": self.batch_size * self.accumulation_steps,
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

    def _save_history(self):
        history_path = os.path.join(self.model_dir, "history_CNNLSTM.json")
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, default=str)
            print(f"✅ History saved: {history_path}")

            meta_path = os.path.join(self.model_dir, "training_meta_CNNLSTM.json")
            meta = {
                "best_acc": float(getattr(self, 'test_accuracy', 0.0)),
                "best_val_acc": float(self.best_acc),
                "best_epoch": int(self.best_epoch),
                "test_accuracy": float(getattr(self, 'test_accuracy', 0.0)),
                "test_auc": float(getattr(self, 'test_auc', 0.0)),
                "test_loss": float(getattr(self, 'test_loss', 0.0)),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"✅ Meta saved: {meta_path}")
        except Exception as e:
            print(f"⚠️ Could not save history: {e}")

    def _load_history(self):
        history_path = os.path.join(self.model_dir, "history_CNNLSTM.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                print(f"✅ History loaded: {history_path}")
                return True
            except Exception as e:
                print(f"⚠️ Could not load history: {e}")
        return False

    def _set_phase_lr(self, new_lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        if hasattr(self.scheduler, 'max_lr'):
            self.scheduler.max_lr = new_lr
        if hasattr(self.scheduler, 'current_lr'):
            self.scheduler.current_lr = new_lr
        print(f"   🔧 LR set to: {new_lr:.6f}")

    # =====================================================
    # FREEZING
    # =====================================================

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if "cnn" not in name:  # فریز کردن CNN (Feature Extractor)
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ CNN Backbone FROZEN (LSTM only)")
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ All layers UNFROZEN")
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    # =====================================================
    # TRAINING
    # =====================================================

    def train_epoch(self, epochs=1, desc=""):
        for epoch in range(epochs):
            epoch_start = time.time()

            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            self.optimizer.zero_grad()

            loop = tqdm(self.train_loader, desc=f"[CNNLSTM] {desc} Epoch {epoch + 1}/{epochs}")

            batch_idx = 0
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

                loop.set_postfix({
                    "loss": f"{loss.item() * self.accumulation_steps:.4f}",
                    "acc": f"{total_correct / total_samples:.4f}",
                })

            if (batch_idx + 1) % self.accumulation_steps != 0:
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

            # ✅ Early Stopping Check
            if hasattr(self, 'early_stopping'):
                if self.early_stopping(val_loss, epoch, self.model):
                    print(f"\n🛑 Early Stopping at epoch {epoch + 1}")
                    print(f"   Best Val Loss: {self.early_stopping.best_score:.4f}")
                    print(f"   Best Epoch: {self.early_stopping.best_epoch + 1}")
                    break

            epoch_time = time.time() - epoch_start
            self.history["train_loss"].append(avg_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)
            self.history["epoch_times"].append(epoch_time)

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                save_model_checkpoint(
                    model=self.model, optimizer=self.optimizer, epoch=epoch,
                    metrics={"val_acc": val_acc, "val_loss": val_loss},
                    filename="CNNLSTM_best.pth",
                    scheduler=self.scheduler,
                    extra_info={"best_accuracy": self.best_acc},
                    save_dir=self.model_dir,
                )
                print(f"✅ Best model saved: {val_acc:.4f}")

            if (epoch + 1) % 5 == 0:
                save_model_checkpoint(
                    model=self.model, optimizer=self.optimizer, epoch=epoch,
                    metrics={"val_acc": val_acc, "val_loss": val_loss},
                    filename="CNNLSTM_checkpoint.pth",
                    scheduler=self.scheduler,
                    save_dir=self.model_dir,
                )

            save_model_checkpoint(
                model=self.model, optimizer=self.optimizer, epoch=epoch,
                metrics={"val_acc": val_acc, "val_loss": val_loss},
                filename="CNNLSTM_last.pth",
                scheduler=self.scheduler,
                save_dir=self.model_dir,
            )

            print(
                f"📊 Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s"
            )

        print(f"\n🔥 Best Accuracy: {self.best_acc:.4f} (Epoch {self.best_epoch + 1})")
        return self.best_acc

    def train_full(self):
        print("\n" + "=" * 70)
        print("🚀 CNN+LSTM FULL TRAINING PIPELINE (TWO-STAGE)")
        print("=" * 70)

        phase_lrs = self.phase_lrs if self.phase_lrs else {
            0: self.lr, 1: self.lr * 0.2
        }

        # ---- Stage 1: Frozen backbone ----
        print(f"\n📌 STAGE 1: Frozen CNN Backbone ({self.epochs_per_phase[0]} epochs) - LR: {phase_lrs.get(0, self.lr)}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(0, self.lr))
        self.freeze_backbone()
        self.train_epoch(epochs=self.epochs_per_phase[0], desc="Stage 1")

        # ---- Stage 2: Full fine-tune ----
        print(
            f"\n📌 STAGE 2: Full fine-tune ({self.epochs_per_phase[1]} epochs) - LR: {phase_lrs.get(1, self.lr * 0.2)}")
        print("-" * 40)
        self._set_phase_lr(phase_lrs.get(1, self.lr * 0.2))
        self.unfreeze_all()

        # ✅ Early Stopping Setup
        self.early_stopping = EarlyStopping(
            patience=12,
            min_delta=1e-4,
            mode='min',
            verbose=True,
        )

        self.train_epoch(epochs=self.epochs_per_phase[1], desc="Stage 2")

        # ✅ Restore بهترین وزن‌ها
        if hasattr(self, 'early_stopping') and self.early_stopping.best_weights is not None:
            self.early_stopping.restore_best_weights(self.model)

        print("\n📥 Loading best model...")
        self.load(self.save_path)

        print("\n📊 Evaluating on TEST set...")
        self.evaluate_on_test()

        # ✅✅✅ ذخیره history
        self._save_history()

        print("\n" + "=" * 70)
        print(f"✅ Training Complete!")
        print(f"   Best Val Acc: {self.best_acc:.4f}")
        print(f"   Test Acc:     {getattr(self, 'test_accuracy', 0):.4f}")
        print("=" * 70)

        return self.best_acc

    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(self, return_loss=False):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        y_true, y_pred, y_prob = [], [], []

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
        print("\n" + "=" * 70)
        print("📊 EVALUATING ON TEST SET")
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

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        report, cm, auc_score = compute_metrics(y_true, y_pred, y_prob)

        test_accuracy = total_correct / total_samples
        test_loss = total_loss / len(self.test_loader)

        print(f"\n✅ Test Acc:  {test_accuracy:.4f}")
        print(f"✅ Test Loss: {test_loss:.4f}")
        print(f"✅ Test AUC:  {auc_score:.4f}")
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

    def load(self, path=None):
        if path is None:
            path = self.save_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()
        print(f"✅ Model loaded: {path}")
        if "epoch" in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        return self.model

    # =====================================================
    # REPORT
    # =====================================================

    def generate_report(self, output_path=None):
        if self.metrics is None:
            self.evaluate()

        if not self.history.get("train_loss"):
            print("📊 History empty. Loading from saved file...")
            self._load_history()

        if self.best_acc == 0.0 and self.history['val_acc']:
            self.best_acc = max(self.history['val_acc'])
            self.best_epoch = self.history['val_acc'].index(self.best_acc)

        if output_path is None:
            reports_dir = os.path.join(self.base_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            output_path = os.path.join(reports_dir, "CNNLSTMReport.pdf")

        print(f"\n📄 Generating CNN+LSTM report: {output_path}")

        if self.test_metrics is None:
            print("📊 No test metrics found. Evaluating on test set...")
            try:
                self.evaluate_on_test()
            except Exception as e:
                print(f"⚠️ Could not evaluate on test set: {e}")

        try:
            if self.test_metrics is not None:
                used_metrics = self.test_metrics
                used_y_true = self.test_y_true
                used_y_pred = self.test_y_pred
                used_y_prob = self.test_y_prob
                used_acc = self.test_accuracy
                print(f"   ✅ Using TEST metrics (Acc: {used_acc:.4f})")
            else:
                used_metrics = self.metrics
                used_y_true = self.y_true
                used_y_pred = self.y_pred
                used_y_prob = self.y_prob
                used_acc = self.metrics.get("accuracy", 0)
                print(f"   ⚠️ Using VALIDATION metrics (Acc: {used_acc:.4f})")

            reporter = CNNLSTMReporter({
                "config": {
                    "input_size": self.target_size,
                    "batch_size": self.batch_size,
                    "accumulation_steps": self.accumulation_steps,
                    "effective_batch_size": self.batch_size * self.accumulation_steps,
                    "epochs_stage1": self.epochs_per_phase[0],
                    "epochs_stage2": self.epochs_per_phase[1],
                    "lr_stage1": self.lr,
                    "lr_backbone": self.lr * 0.2,
                    "lr_head": self.lr,
                    "weight_decay": self.weight_decay,
                    "label_smoothing": self.label_smoothing,
                    "grad_clip": 1.0,
                    "dropout": self.dropout_rate,
                    "hidden_dim": 256,
                    "scheduler": "WarmupCosine",
                    "use_amp": self.use_mixed_precision,
                    "use_weighted_sampler": self.use_sampler,
                    "seed": 42,
                    "total_params": sum(p.numel() for p in self.model.parameters()),
                    "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    "test_accuracy": used_acc,
                },
                "history": self.history,
                "best_metrics": self.metrics,
                "test_metrics": used_metrics,
                "y_true": used_y_true,
                "y_pred": used_y_pred,
                "y_prob": used_y_prob,
                "class_names": CLASS_NAMES,
                "best_epoch": self.best_epoch,
                "training_time_sec": sum(self.history["epoch_times"]) if self.history["epoch_times"] else 0,
            })

            reporter.generate_all(report_dir=os.path.dirname(output_path))
            print(f"✅ PDF Report saved: {output_path}")

        except Exception as e:
            print(f"⚠️ Report generation failed: {e}")
            import traceback
            traceback.print_exc()

        return output_path

    # =====================================================
    # GETTERS
    # =====================================================

    def get_metrics(self):
        if self.metrics is None:
            self.evaluate()
        return self.metrics, self.cm, self.y_true, self.y_prob

    def get_history(self):
        return self.history

    def get_model(self):
        return self.model
    