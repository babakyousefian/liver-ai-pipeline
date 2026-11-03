# src/pipeLine/resNetEfficientNetModels.py

"""
Liver AI Pipeline - COMPLETE ULTIMATE EDITION with ALL Optimizations

This module provides the complete pipeline with ALL optimizations:
- Advanced Hyperparameters (batch=32, lr=5e-4)
- Multi-stage training with optimized epochs (10,15,15,15)
- Label Smoothing (0.1)
- Warmup Cosine Scheduler
- Gradient Accumulation (4 steps)
- Cross-Validation (5-fold)
- Early Stopping (patience=10)
- WeightedRandomSampler
- Mixed Precision Training
- Model Checkpointing
- Comprehensive Logging & Resource Monitoring

Author: Babak Yousefian
Version: 4.0.0 (COMPLETE ULTIMATE EDITION)
"""

import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gc
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch

warnings.filterwarnings("ignore")

# =====================================================
# ✅ SMART LEARNING RATE SCHEDULER 🔥
# =====================================================


def get_learning_rate_schedule(model_type: str, phase: int) -> float:
    """
    Get the optimal learning rate for each phase based on global standards.

    Args:
        model_type: Type of model ('resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b1')
        phase: Phase number (0-indexed)

    Returns:
        float: Optimal learning rate for the given phase
    """
    # ✅ استانداردهای جهانی برای هر مدل
    LR_SCHEDULES = {
        "resnet18": {
            0: 1e-3,  # Phase 1: FC layer only (higher LR for new layer)
            1: 1e-4,  # Phase 2: Layer4 + FC
            2: 5e-5,  # Phase 3: Full fine-tuning (lower LR)
            3: 1e-5,  # Phase 4: Final fine-tuning (very low LR)
        },
        "resnet50": {
            0: 1e-3,  # Phase 1: FC layer only
            1: 5e-4,  # Phase 2: Layer4 + FC
            2: 1e-4,  # Phase 3: Layer3+4 + FC
            3: 5e-5,  # Phase 4: Layer2+3+4 + FC
            4: 1e-5,  # Phase 5: Full fine-tuning
        },
        "efficientnet_b0": {
            0: 1e-3,  # Phase 1: Classifier only
            1: 1e-4,  # Phase 2: Classifier + 2 blocks
            2: 5e-5,  # Phase 3: Full fine-tuning
            3: 1e-5,  # Phase 4: Final fine-tuning
        },
        "efficientnet_b1": {
            0: 1e-3,  # Phase 1: Classifier only
            1: 1e-4,  # Phase 2: Classifier + 3 blocks
            2: 5e-5,  # Phase 3: Full fine-tuning
            3: 1e-5,  # Phase 4: Final fine-tuning
        },
        "vgg16": {
            0: 1e-3,  # Stage 1: Classifier only
            1: 1e-4,  # Stage 2: Full fine-tune
        },
        "mobilenetv2": {
            0: 1e-3,  # Stage 1: Classifier only
            1: 2e-4,  # Stage 2: Full fine-tune
        },
        "googlenet": {
            0: 1e-3,  # Stage 1: Classifier only
            1: 2e-4,  # Stage 2: Full fine-tune
        },
        "swin_transformer": {
            0: 1e-3,  # Stage 1: Head only
            1: 5e-5,  # Stage 2: Full fine-tune (lower for transformer)
        },
        "alexnet": {
            0: 1e-3,  # Stage 1: Classifier only (heavy model, needs higher LR for new layers)
            1: 5e-5,  # Stage 2: Full fine-tune (lower for 60M params)
        },
        "visio_transformer": {
            0: 1e-3,  # Stage 1: Head only
            1: 5e-5,  # Stage 2: Full fine-tune (lower for ViT)
        },
        "convnext": {
            0: 1e-3,  # Stage 1: Classifier only
            1: 5e-5,  # Stage 2: Full fine-tune
        },
        "densnet": {
            0: 1e-3,  # Stage 1: Classifier only
            1: 5e-5,  # Stage 2: Full fine-tune
        },
        "cnn": {
            0: 1e-3,  # Stage 1: Classifier only
            1: 5e-5,  # Stage 2: Full fine-tune
        },
        "lstm": {
            0: 1e-3,  # Stage 1: LSTM + Classifier only
            1: 5e-5,  # Stage 2: Full fine-tune
        },
        "cnn_lstm": {
            0: 1e-3,  # Stage 1: LSTM + Classifier only
            1: 5e-5,  # Stage 2: Full fine-tune
        },
    }

    # ✅ بازگشت نرخ یادگیری برای فاز مورد نظر
    schedule = LR_SCHEDULES.get(model_type, {})
    lr = schedule.get(phase, 1e-4)

    return lr


# =====================================================
# ✅ GPU OPTIMIZED HYPERPARAMETERS 🔥
# =====================================================

HYPERPARAMS = {
    "resnet18": {
        "batch_size": 64,
        "base_lr": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 1,
        "warmup_epochs": 5,
        "epochs_per_phase": [10, 15, 15, 15],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 256,
        "use_smart_lr": True,
        "early_stopping_patience": 25
    },
    "resnet50": {
        "batch_size": 64,
        "base_lr": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 1,
        "warmup_epochs": 5,
        "epochs_per_phase": [15, 20, 20, 20, 25],
        "phase_lrs": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 256,
        "use_smart_lr": True,
        "early_stopping_patience": 25,
    },
    "efficientnet_b0": {
        "batch_size": 64,
        "base_lr": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 1,
        "warmup_epochs": 5,
        "epochs_per_phase": [15, 20, 20, 20],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 25,
    },
    "efficientnet_b1": {
        "batch_size": 64,
        "base_lr": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 1,
        "warmup_epochs": 5,
        "epochs_per_phase": [10, 15, 15, 20],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 240,
        "use_smart_lr": True,
        "early_stopping_patience": 25,
    },
    "vgg16": {
        "batch_size": 16,
        "base_lr": 1e-4,
        "weight_decay": 5e-3,           # ← 10x بالاتر
        "dropout": 0.4,                  # ← 0.15 → 0.4
        "label_smoothing": 0.15,         # ← 0.1 → 0.15
        "accumulation_steps": 2,
        "warmup_epochs": 6,              # ← 5 → 6
        "epochs_per_phase": [5, 35],     # ← 40 → 35
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 30,
    },
    "mobilenetv2": {
        "batch_size": 64,
        "base_lr": 5e-4,
        "weight_decay": 1e-3,         # ← 2x بالاتر
        "dropout": 0.15,               # ✅ نگه دار (استادت گفت)
        "label_smoothing": 0.12,       # ← 0.1 → 0.12
        "accumulation_steps": 1,
        "warmup_epochs": 6,            # ← 5 → 6
        "epochs_per_phase": [5, 45],   # ← 50 → 45
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 40,
    },
    "googlenet": {
        "batch_size": 64,
        "base_lr": 5e-4,
        "weight_decay": 1e-3,         # ← 2x بالاتر
        "dropout": 0.15,               # ✅ نگه دار
        "label_smoothing": 0.12,       # ← 0.1 → 0.12
        "accumulation_steps": 1,
        "warmup_epochs": 6,            # ← 5 → 6
        "epochs_per_phase": [5, 40],   # ← 45 → 40
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 40,
    },
    "swin_transformer": {
        "batch_size": 16,
        "base_lr": 1e-4,
        "weight_decay": 0.01,          # ← 20x بالاتر (Transformer نیاز داره)
        "dropout": 0.3,                # ← 0.15 → 0.3 (Transformer نیاز داره)
        "label_smoothing": 0.12,       # ← 0.1 → 0.12
        "accumulation_steps": 2,
        "warmup_epochs": 8,            # ← 5 → 8
        "epochs_per_phase": [5, 35],   # ✅ نگه دار
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 25,
    },
    "alexnet": {
         "batch_size": 32,
        "base_lr": 5e-4,
        "weight_decay": 2e-3,           # ← 4x بالاتر
        "dropout": 0.4,                  # ← 0.15 → 0.4
        "label_smoothing": 0.15,         # ← 0.1 → 0.15
        "accumulation_steps": 1,
        "warmup_epochs": 6,
        "epochs_per_phase": [5, 35],     # ← 45 → 35
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 40,
    },
    "visio_transformer": {
        "batch_size": 16,
        "base_lr": 1e-4,
        "weight_decay": 0.05,          # ✅ نگه دار
        "dropout": 0.3,                # ← 0.1 → 0.3 (ViT نیاز داره)
        "label_smoothing": 0.12,       # ← 0.1 → 0.12
        "accumulation_steps": 2,
        "warmup_epochs": 10,           # ✅ نگه دار
        "epochs_per_phase": [5, 35],   # ✅ نگه دار
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 30,
    },
     "convnext": {
        "batch_size": 16,
        "base_lr": 1e-4,
        "weight_decay": 0.05,            # ← 0.01 → 0.05
        "dropout": 0.3,                  # ← 0.25 → 0.3
        "label_smoothing": 0.15,         # ← 0.12 → 0.15
        "accumulation_steps": 2,
        "warmup_epochs": 8,              # ✅
        "epochs_per_phase": [5, 30],     # ← 35 → 30
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 30,
    },
    "densnet": {
        "batch_size": 32,
        "base_lr": 1e-4,
        "weight_decay": 0.01,
        "dropout": 0.3,
        "label_smoothing": 0.15,
        "accumulation_steps": 1,
        "warmup_epochs": 8,
        "epochs_per_phase": [5, 35],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 20,
    },
    "cnn": {
        "batch_size": 32,
        "base_lr": 1e-4,
        "weight_decay": 1e-3,  # ← بهینه
        "dropout": 0.15,  # ← بهینه
        "label_smoothing": 0.1,  # ← استاندارد
        "accumulation_steps": 1,
        "warmup_epochs": 8,
        "epochs_per_phase": [5, 35],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 15,  # ← بهینه
    },
    "lstm": {
        "batch_size": 16,  # ← کمتر برای LSTM (VRAM)
        "base_lr": 1e-4,
        "weight_decay": 1e-3,  # ← بهینه
        "dropout": 0.2,  # ← بهینه
        "label_smoothing": 0.1,  # ← استاندارد
        "accumulation_steps": 2,
        "warmup_epochs": 8,
        "epochs_per_phase": [5, 30],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 15,
    },
    "cnn_lstm": {
        "batch_size": 16,
        "base_lr": 1e-4,
        "weight_decay": 1e-3,  # ← بهینه
        "dropout": 0.2,  # ← بهینه
        "label_smoothing": 0.1,  # ← استاندارد
        "accumulation_steps": 2,
        "warmup_epochs": 8,
        "epochs_per_phase": [5, 30],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "use_smart_lr": True,
        "early_stopping_patience": 15,
    },
}


# =====================================================
# ✅ LOGGING SYSTEM
# =====================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    import io

    # ✅ تنظیم UTF-8 برای ویندوز (بالای فایل، بعد از import ها)
    if sys.platform == "win32":
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("LiverAI")
    logger.setLevel(logging.DEBUG)

    log_file = os.path.join(
        log_dir, f"liver_ai_{datetime.now().strftime('%Y%m%d')}.log"
    )
    # ✅ اضافه کردن encoding='utf-8'
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    # ✅ console handler با UTF-8
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # ✅ اگر باز هم مشکل بود
    if sys.platform == "win32" and hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


LOGGER = setup_logging()


# =====================================================
# ✅ ADVANCED TRAINER WRAPPER 🔥
# =====================================================


class AdvancedModelTrainer:
    """
    Advanced trainer with ALL optimizations.

    Features:
    - Optimized hyperparameters (batch=32, lr=5e-4)
    - Multi-stage training with optimized epochs
    - Label Smoothing (0.1)
    - Warmup Cosine Scheduler
    - Gradient Accumulation (4 steps)
    - Cross-Validation (5-fold)
    - Early Stopping (patience=10)
    - WeightedRandomSampler
    - Mixed Precision Training
    - Model Checkpointing
    """

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.hp = HYPERPARAMS[model_type]
        self.logger = LOGGER
        self.trainer = None
        self.results = {}
        self.phase_lrs = []
        self._cv_done = True
        self._cv_scores = {}  # ✅ ذخیره CV scores برای گزارش


    def create_trainer(self) -> Any:
        """Create trainer with optimized hyperparameters."""
        self.logger.info(
            f"🚀 Creating {self.model_type} trainer with optimized settings..."
        )
        self.logger.info(f"   Batch Size: {self.hp['batch_size']} (GPU optimized)")
        self.logger.info(f"   Learning Rate: Smart Schedule (per phase)")
        self.logger.info(f"   Dropout: {self.hp['dropout']}")
        self.logger.info(f"   Label Smoothing: {self.hp['label_smoothing']}")
        self.logger.info(f"   Use Sampler: {self.hp['use_sampler']}")

        # ✅ محاسبه نرخ‌های یادگیری هوشمند برای هر فاز
        num_phases = len(self.hp["epochs_per_phase"])
        self.phase_lrs = [
            get_learning_rate_schedule(self.model_type, phase)
            for phase in range(num_phases)
        ]

        # ✅ نمایش نرخ‌های یادگیری
        self.logger.info("📊 Smart Learning Rates per Phase:")
        for phase, lr in enumerate(self.phase_lrs):
            self.logger.info(f"   Phase {phase+1}: LR = {lr:.6f}")

        # ✅ استفاده از نرخ یادگیری فاز اول به عنوان base_lr
        base_lr = self.phase_lrs[0] if self.phase_lrs else self.hp["base_lr"]

        if "resnet" in self.model_type:
            if self.model_type == "resnet18":
                from src.AI.ResNet.resnet18 import ResNet18Trainer

                self.trainer = ResNet18Trainer(
                    batch_size=self.hp["batch_size"],
                    lr=base_lr,
                    weight_decay=self.hp["weight_decay"],
                    dropout_rate=self.hp["dropout"],
                    use_sampler=self.hp["use_sampler"],
                    use_mixed_precision=self.hp["use_mixed_precision"],
                    label_smoothing=self.hp["label_smoothing"],
                    accumulation_steps=self.hp["accumulation_steps"],
                    warmup_epochs=self.hp["warmup_epochs"],
                    num_workers=0,
                )
            else:
                from src.AI.ResNet.resnet50 import ResNet50Trainer

                self.trainer = ResNet50Trainer(
                    batch_size=self.hp["batch_size"],
                    lr=base_lr,
                    weight_decay=self.hp["weight_decay"],
                    dropout_rate=self.hp["dropout"],
                    use_sampler=self.hp["use_sampler"],
                    use_mixed_precision=self.hp["use_mixed_precision"],
                    label_smoothing=self.hp["label_smoothing"],
                    accumulation_steps=self.hp["accumulation_steps"],
                    warmup_epochs=self.hp["warmup_epochs"],
                    num_workers=0,
                )

        elif self.model_type == "vgg16":
            from src.AI.VGG16.VGG16 import VGG16Trainer

            self.trainer = VGG16Trainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        elif self.model_type == "mobilenetv2":
            from src.AI.MobileNet_V2.mobilenetv2 import MobileNetV2Trainer

            self.trainer = MobileNetV2Trainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        elif self.model_type == "googlenet":
            from src.AI.GoogleNet.GoogleNet import GoogleNetTrainer

            self.trainer = GoogleNetTrainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        elif self.model_type == "swin_transformer":
            from src.AI.Swin_Transformer.Swin_Transformer import SwinTransformerTrainer

            self.trainer = SwinTransformerTrainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        elif self.model_type == "alexnet":
            from src.AI.AlexNet.AlexNet import AlexNetTrainer

            self.trainer = AlexNetTrainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        elif self.model_type == "visio_transformer":
            from src.AI.visioTransformer.visioTransformer import VisioTransformerTrainer

            self.trainer = VisioTransformerTrainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        elif self.model_type == "convnext":
            from src.AI.ConvNext.convNext import ConvNextTrainer

            self.trainer = ConvNextTrainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        elif self.model_type == "densnet":
            from src.AI.DensNet.DensNet import DensNetTrainer

            self.trainer = DensNetTrainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        elif self.model_type == "cnn":
            from src.AI.CNN.CNN import CNNTrainer

            self.trainer = CNNTrainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        elif self.model_type == "lstm":
            from src.AI.LSTM.LSTM import LSTMTrainer

            self.trainer = LSTMTrainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        elif self.model_type == "cnn_lstm":
            from src.AI.CNNLSTM.CNNLSTM import CNNLSTMTrainer

            self.trainer = CNNLSTMTrainer(
                batch_size=self.hp["batch_size"],
                lr=base_lr,
                weight_decay=self.hp["weight_decay"],
                dropout_rate=self.hp["dropout"],
                use_sampler=self.hp["use_sampler"],
                use_mixed_precision=self.hp["use_mixed_precision"],
                label_smoothing=self.hp["label_smoothing"],
                accumulation_steps=self.hp["accumulation_steps"],
                warmup_epochs=self.hp["warmup_epochs"],
                num_workers=0,
                target_size=self.hp["target_size"],
            )

        else:
            if self.model_type == "efficientnet_b0":
                from src.AI.EfficientNet.efficientNetB0 import EfficientNetB0Trainer

                self.trainer = EfficientNetB0Trainer(
                    batch_size=self.hp["batch_size"],
                    lr=base_lr,
                    weight_decay=self.hp["weight_decay"],
                    dropout_rate=self.hp["dropout"],
                    use_sampler=self.hp["use_sampler"],
                    use_mixed_precision=self.hp["use_mixed_precision"],
                    label_smoothing=self.hp["label_smoothing"],
                    accumulation_steps=self.hp["accumulation_steps"],
                    warmup_epochs=self.hp["warmup_epochs"],
                    num_workers=0,
                )
            else:
                from src.AI.EfficientNet.efficientNetB1 import EfficientNetB1Trainer

                self.trainer = EfficientNetB1Trainer(
                    batch_size=self.hp["batch_size"],
                    lr=base_lr,
                    weight_decay=self.hp["weight_decay"],
                    dropout_rate=self.hp["dropout"],
                    use_sampler=self.hp["use_sampler"],
                    use_mixed_precision=self.hp["use_mixed_precision"],
                    label_smoothing=self.hp["label_smoothing"],
                    accumulation_steps=self.hp["accumulation_steps"],
                    warmup_epochs=self.hp["warmup_epochs"],
                    num_workers=0,
                )

        # ✅ ذخیره نرخ‌های یادگیری در trainer (اگه attribute داشته باشه)
        if hasattr(self.trainer, "phase_lrs"):
            self.trainer.phase_lrs = {
                phase: lr for phase, lr in enumerate(self.phase_lrs)
            }

        self.logger.info(f"✅ {self.model_type} trainer created")
        return self.trainer

    def train(self, epochs: int = None) -> Dict[str, Any]:
        """Train with full optimization."""
        if self.trainer is None:
            self.create_trainer()

        total_epochs = sum(self.hp["epochs_per_phase"])
        self.logger.info(f"📊 Training {self.model_type} for {total_epochs} epochs...")
        self.logger.info(f"   Phases: {self.hp['epochs_per_phase']}")

        # ✅ نمایش نرخ یادگیری هر فاز
        if self.phase_lrs:
            self.logger.info("   Learning Rates per Phase:")
            for phase, lr in enumerate(self.phase_lrs):
                self.logger.info(f"      Phase {phase+1}: {lr:.6f}")

        start_time = time.time()

        # ✅ Multi-stage training with optimized epochs
        self.trainer.epochs_per_phase = self.hp["epochs_per_phase"]

        # ✅ اگر trainer از phase_lrs پشتیبانی میکنه، تنظیم کن
        if hasattr(self.trainer, "phase_lrs"):
            self.trainer.phase_lrs = {
                phase: lr for phase, lr in enumerate(self.phase_lrs)
            }
        else:
            # ✅ اگر پشتیبانی نمیکنه، نرخ یادگیری رو به صورت دستی تنظیم کن
            self.logger.info(
                "   ⚠️ Trainer doesn't support phase_lrs. Using base LR with scheduler."
            )

        best_acc = self.trainer.train_full()

        elapsed = time.time() - start_time

        self.results = {
            "model_type": self.model_type,
            "best_accuracy": best_acc,
            "elapsed_time": elapsed,
            "hyperparameters": self.hp,
            "history": self.trainer.get_history(),
        }

        self.logger.info(
            f"✅ {self.model_type} completed: {best_acc:.4f} in {elapsed:.2f}s"
        )

        return self.results

    def cross_validate(self, n_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation using the SAVED best model.
        This does NOT retrain the model from scratch.
        """
        # ✅ جلوگیری از اجرای دوباره
        if self._cv_done:
            self.logger.info("⏭️ Cross-Validation already performed. Skipping...")
            return {}

        self.logger.info(f"🔬 Running {n_folds}-fold CV for {self.model_type}...")
        self.logger.info("📌 Using saved best model (no retraining)...")

        if self.trainer is None:
            self.create_trainer()

        from src.AI.ResNet.utils import CROPPED, CSVAddress, LiverDataset, validate_csv

        is_valid, df, _ = validate_csv(CSVAddress)

        if not is_valid:
            self.logger.error("❌ Invalid CSV for cross-validation")
            return {}

        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import DataLoader

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        labels = df["liverLabel"].values

        cv_results = {
            "folds": [],
            "accuracies": [],
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
            "cv_scores": {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
            },
        }

        # ✅ Load the saved best model once
        best_model_path = self.trainer.save_path
        self.logger.info(f"📥 Loading best model from: {best_model_path}")

        try:
            self.trainer.load(best_model_path)
        except Exception as e:
            self.logger.error(f"❌ Failed to load best model: {e}")
            self.logger.error("📌 Please train the model first (Option 1-5 in menu)")
            return {}

        for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
            self.logger.info(f"   Fold {fold+1}/{n_folds}...")

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            # ✅ Create datasets WITHOUT retraining the model
            train_ds = LiverDataset(train_df, CROPPED, augment=False, return_path=True)
            val_ds = LiverDataset(val_df, CROPPED, augment=False, return_path=True)

            train_loader = DataLoader(
                train_ds, batch_size=self.hp["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.hp["batch_size"], shuffle=False
            )

            # ✅ Evaluate the saved model on this fold WITHOUT retraining
            original_train_loader = self.trainer.train_loader
            original_val_loader = self.trainer.val_loader

            self.trainer.train_loader = train_loader
            self.trainer.val_loader = val_loader

            # ✅ Run evaluation and get metrics
            from src.AI.ResNet.utils import compute_metrics

            self.trainer.model.eval()

            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        x, y, _ = batch
                    else:
                        x, y = batch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    x = x.to(device)
                    y = y.to(device)

                    out = self.trainer.model(x)
                    prob = torch.softmax(out, dim=1)
                    pred = prob.argmax(dim=1)

                    all_labels.extend(y.cpu().numpy())
                    all_preds.extend(pred.cpu().numpy())
                    all_probs.extend(prob.cpu().numpy())

            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)

            metrics, _, _ = compute_metrics(all_labels, all_preds, all_probs)

            # ✅ ذخیره CV scores
            accuracy = metrics.get("accuracy", 0)
            precision = metrics.get("weighted avg", {}).get("precision", 0)
            recall = metrics.get("weighted avg", {}).get("recall", 0)
            f1 = metrics.get("weighted avg", {}).get("f1-score", 0)

            cv_results["cv_scores"]["accuracy"].append(accuracy)
            cv_results["cv_scores"]["precision"].append(precision)
            cv_results["cv_scores"]["recall"].append(recall)
            cv_results["cv_scores"]["f1_score"].append(f1)

            # ✅ Restore original dataloaders
            self.trainer.train_loader = original_train_loader
            self.trainer.val_loader = original_val_loader

            cv_results["folds"].append(fold + 1)
            cv_results["accuracies"].append(accuracy)

            self.logger.info(f"      Accuracy: {accuracy:.4f}")

        cv_results["mean_accuracy"] = np.mean(cv_results["accuracies"])
        cv_results["std_accuracy"] = np.std(cv_results["accuracies"])

        self.logger.info(
            f"✅ CV Results: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}"
        )

        self._cv_done = True
        self._cv_scores = cv_results["cv_scores"]

        return cv_results

    def generate_report(self) -> str:
        """Generate comprehensive report."""
        if self.trainer is None:
            self.logger.error("❌ No trainer available")
            return ""

        self.logger.info(f"📄 Generating report for {self.model_type}...")

        metrics, cm, y_true, y_prob = self.trainer.get_metrics()

        # ✅ دریافت بهترین دقت Validation
        best_acc = self.results.get('best_accuracy', 0.0)  # از نتایج آموزش

        if "resnet" in self.model_type:
            from src.AI.ResNet.resnetReports import generate_full_pdf
            from src.AI.ResNet.utils import OUTPUT_DIR as RESNET_OUTPUT_DIR

            output_path = os.path.join(
                RESNET_OUTPUT_DIR, "reports", f"{self.model_type.upper()}_Report.pdf"
            )

            sample_img = None
            for batch in self.trainer.val_loader:
                if len(batch) == 3:
                    sample_img = batch[0][0]
                else:
                    sample_img = batch[0]
                break

            if sample_img is not None:
                generate_full_pdf(
                    metrics=metrics,
                    cm=cm,
                    y_true=y_true,
                    y_prob=y_prob,
                    model=self.trainer.model,
                    sample_image_tensor=sample_img,
                    output_path=output_path,
                    model_name=self.model_type.upper(),
                    history=self.trainer.get_history(),
                    config=self.hp,
                    cv_scores=self._cv_scores if self._cv_done else None,
                    best_accuracy=best_acc,  # ✅ اضافه شد
                )
            else:
                self.logger.warning("⚠️ No sample image available for report")
        else:
            from src.AI.EfficientNet.efficientNetReports import generate_full_pdf
            from src.AI.EfficientNet.utils import OUTPUT_DIR as EFFICIENTNET_OUTPUT_DIR

            output_path = os.path.join(
                EFFICIENTNET_OUTPUT_DIR,
                "reports",
                f"{self.model_type.upper()}_Report.pdf",
            )

            sample_img = None
            for batch in self.trainer.val_loader:
                if len(batch) == 3:
                    sample_img = batch[0][0]
                else:
                    sample_img = batch[0]
                break

            if sample_img is not None:
                generate_full_pdf(
                    metrics=metrics,
                    cm=cm,
                    y_true=y_true,
                    y_prob=y_prob,
                    model=self.trainer.model,
                    sample_image_tensor=sample_img,
                    output_path=output_path,
                    model_name=self.model_type.upper(),
                    history=self.trainer.get_history(),
                    config=self.hp,
                )
            else:
                self.logger.warning("⚠️ No sample image available for report")

        self.logger.info(f"✅ Report generated: {output_path}")
        return output_path


# =====================================================
# ✅ TRAINING FUNCTIONS (بدون Cross-Validation) 🔥
# =====================================================


def train_resnet18_advanced() -> Dict[str, Any]:
    """Train ResNet18 with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("resnet18")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 ResNet18 Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results


def train_resnet50_advanced() -> Dict[str, Any]:
    """Train ResNet50 with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("resnet50")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 ResNet50 Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results


def train_efficientnet_b0_advanced() -> Dict[str, Any]:
    """Train EfficientNet-B0 with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("efficientnet_b0")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 EfficientNet-B0 Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results


def train_efficientnet_b1_advanced() -> Dict[str, Any]:
    """Train EfficientNet-B1 with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("efficientnet_b1")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 EfficientNet-B1 Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

def train_vgg16_advanced() -> Dict[str, Any]:
    """Train VGG16 with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("vgg16")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 VGG16 Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

def train_mobilenetv2_advanced() -> Dict[str, Any]:
    """Train MobileNetV2 with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("mobilenetv2")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 MobileNetV2 Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

def train_googlenet_advanced() -> Dict[str, Any]:
    """Train GoogleNet with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("googlenet")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 GoogleNet Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

def train_swin_transformer_advanced() -> Dict[str, Any]:
    """Train Swin Transformer with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("swin_transformer")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 Swin Transformer Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results


def train_alexnet_advanced() -> Dict[str, Any]:
    """Train AlexNet with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("alexnet")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 AlexNet Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

def train_visio_transformer_advanced() -> Dict[str, Any]:
    """Train Vision Transformer with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("visio_transformer")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 Vision Transformer Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

def train_convnext_advanced() -> Dict[str, Any]:
    """Train ConvNeXt with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("convnext")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 ConvNeXt Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

def train_densnet_advanced() -> Dict[str, Any]:
    """Train DenseNet with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("densnet")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 DenseNet Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

def train_cnn_advanced() -> Dict[str, Any]:
    """Train Custom CNN with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("cnn")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 CNN Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

# =====================================================
# LSTM TRAINING FUNCTION
# =====================================================
def train_lstm_advanced() -> Dict[str, Any]:
    """Train LSTM (CNN-LSTM) with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("lstm")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 LSTM Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

# =====================================================
# CNN+LSTM TRAINING FUNCTION
# =====================================================
def train_cnn_lstm_advanced() -> Dict[str, Any]:
    """Train CNN+LSTM with ALL optimizations (NO CV)."""
    trainer = AdvancedModelTrainer("cnn_lstm")
    trainer.create_trainer()
    results = trainer.train()
    trainer.generate_report()

    LOGGER.info(f"📊 CNN+LSTM Results:")
    LOGGER.info(f"   Accuracy: {results['best_accuracy']:.4f}")

    return results

# =====================================================
# ✅ CROSS-VALIDATION FUNCTIONS (جداگانه) 🔥
# =====================================================


def cross_validate_resnet18(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate ResNet18 using SAVED model."""
    trainer = AdvancedModelTrainer("resnet18")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)


def cross_validate_resnet50(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate ResNet50 using SAVED model."""
    trainer = AdvancedModelTrainer("resnet50")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)


def cross_validate_efficientnet_b0(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate EfficientNet-B0 using SAVED model."""
    trainer = AdvancedModelTrainer("efficientnet_b0")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)


def cross_validate_efficientnet_b1(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate EfficientNet-B1 using SAVED model."""
    trainer = AdvancedModelTrainer("efficientnet_b1")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)

def cross_validate_vgg16(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate VGG16 using SAVED model."""
    trainer = AdvancedModelTrainer("vgg16")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)

def cross_validate_mobilenetv2(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate MobileNetV2 using SAVED model."""
    trainer = AdvancedModelTrainer("mobilenetv2")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)

def cross_validate_googlenet(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate GoogleNet using SAVED model."""
    trainer = AdvancedModelTrainer("googlenet")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)

def cross_validate_swin_transformer(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate Swin Transformer using SAVED model."""
    trainer = AdvancedModelTrainer("swin_transformer")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)


def cross_validate_alexnet(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate AlexNet using SAVED model."""
    trainer = AdvancedModelTrainer("alexnet")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)

def cross_validate_visio_transformer(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate Vision Transformer using SAVED model."""
    trainer = AdvancedModelTrainer("visio_transformer")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)

def cross_validate_convnext(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate ConvNeXt using SAVED model."""
    trainer = AdvancedModelTrainer("convnext")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)

def cross_validate_densnet(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate DenseNet using SAVED model."""
    trainer = AdvancedModelTrainer("densnet")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)

def cross_validate_cnn(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate Custom CNN using SAVED model."""
    trainer = AdvancedModelTrainer("cnn")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)

# =====================================================
# LSTM CROSS-VALIDATION FUNCTION
# =====================================================
def cross_validate_lstm(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate LSTM using SAVED model."""
    trainer = AdvancedModelTrainer("lstm")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)


# =====================================================
# CNN+LSTM CROSS-VALIDATION FUNCTION
# =====================================================
def cross_validate_cnn_lstm(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate CNN+LSTM using SAVED model."""
    trainer = AdvancedModelTrainer("cnn_lstm")
    trainer.create_trainer()
    return trainer.cross_validate(n_folds=n_folds)

def cross_validate_all_models(n_folds: int = 5) -> Dict[str, Any]:
    """Cross-validate ALL models using SAVED models."""
    LOGGER.info("🔬 Cross-validating ALL models...")
    LOGGER.info("=" * 70)

    results = {}

    LOGGER.info("\n📌 1/8: ResNet18 CV")
    results["resnet18"] = cross_validate_resnet18(n_folds)

    LOGGER.info("\n📌 2/8: ResNet50 CV")
    results["resnet50"] = cross_validate_resnet50(n_folds)

    LOGGER.info("\n📌 3/8: EfficientNet-B0 CV")
    results["efficientnet_b0"] = cross_validate_efficientnet_b0(n_folds)

    LOGGER.info("\n📌 4/8: EfficientNet-B1 CV")
    results["efficientnet_b1"] = cross_validate_efficientnet_b1(n_folds)

    LOGGER.info("\n📌 5/8: VGG16 CV")
    results["vgg16"] = cross_validate_vgg16(n_folds)

    LOGGER.info("\n📌 6/8: MobileNetV2 CV")
    results["mobilenetv2"] = cross_validate_mobilenetv2(n_folds)

    LOGGER.info("\n📌 7/8: GoogleNet CV")
    results["googlenet"] = cross_validate_googlenet(n_folds)

    LOGGER.info("\n📌 8/9: Swin Transformer CV")
    results["swin_transformer"] = cross_validate_swin_transformer(n_folds)

    LOGGER.info("\n📌 9/9: AlexNet CV")
    results["alexnet"] = cross_validate_alexnet(n_folds)

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("📊 CROSS-VALIDATION SUMMARY")
    LOGGER.info("=" * 70)
    for name, data in results.items():
        if data:
            mean = data.get("mean_accuracy", 0)
            std = data.get("std_accuracy", 0)
            LOGGER.info(f"   {name}: {mean:.4f} ± {std:.4f}")
        else:
            LOGGER.info(f"   {name}: ❌ Failed")
    LOGGER.info("=" * 70)

    return results


# =====================================================
# ✅ TRAIN ALL MODELS
# =====================================================


def train_all_models_advanced() -> Dict[str, Any]:
    """Train ALL models with ALL optimizations (NO CV)."""
    LOGGER.info("🚀 Training ALL models with ALL optimizations...")
    LOGGER.info("=" * 70)

    results = {}

    # ResNet18
    LOGGER.info("\n📌 1/7: ResNet18")
    results["resnet18"] = train_resnet18_advanced()

    LOGGER.info("\n📌 2/7: ResNet50")
    results["resnet50"] = train_resnet50_advanced()

    LOGGER.info("\n📌 3/7: EfficientNet-B0")
    results["efficientnet_b0"] = train_efficientnet_b0_advanced()

    LOGGER.info("\n📌 4/7: EfficientNet-B1")
    results["efficientnet_b1"] = train_efficientnet_b1_advanced()

    # VGG16
    LOGGER.info("\n📌 5/7: VGG16")
    results["vgg16"] = train_vgg16_advanced()

    # MobileNetV2
    LOGGER.info("\n📌 6/7: MobileNetV2")
    results["mobilenetv2"] = train_mobilenetv2_advanced()

    # GoogleNet
    LOGGER.info("\n📌 7/8: GoogleNet")
    results["googlenet"] = train_googlenet_advanced()

    # Swin Transformer
    LOGGER.info("\n📌 8/9: Swin Transformer")
    results["swin_transformer"] = train_swin_transformer_advanced()

    # AlexNet
    LOGGER.info("\n📌 9/9: AlexNet")
    results["alexnet"] = train_alexnet_advanced()

    # ✅ Run comparisons after training
    run_comprehensive_comparison_advanced()
    run_literature_comparison_advanced()

    # Summary
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("📊 FINAL SUMMARY - ALL MODELS")
    LOGGER.info("=" * 70)
    for name, data in results.items():
        acc = data.get("best_accuracy", 0)
        LOGGER.info(f"   {name}: {acc:.4f}")
    LOGGER.info("=" * 70)

    return results


def run_comprehensive_comparison_advanced() -> bool:
    """Run comprehensive comparison."""
    LOGGER.info("🔬 Running comprehensive comparison...")

    try:
        from src.AI.Comparisons.resnet_efficientNet_comparisons import run_comprehensive_comparison

        result = run_comprehensive_comparison()
        LOGGER.info(f"✅ Comprehensive comparison completed")
        return True
    except Exception as e:
        LOGGER.error(f"❌ Comprehensive comparison failed: {e}")
        return False


def run_literature_comparison_advanced() -> bool:
    """Run literature comparison."""
    LOGGER.info("📊 Running literature comparison...")

    try:
        from src.AI.Comparisons.generate_comparison_with_literature import (
            generate_comparison_report,
        )

        result = generate_comparison_report()
        LOGGER.info(f"✅ Literature comparison completed")
        return True
    except Exception as e:
        LOGGER.error(f"❌ Literature comparison failed: {e}")
        return False

def generate_report_resnet18() -> str:
    """Generate report for ResNet18 from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating ResNet18 report from saved model...")
    try:
        from src.AI.ResNet.resnet18 import ResNet18Trainer

        trainer = ResNet18Trainer(
            num_workers=0, target_size=224,
            use_sampler=True, use_mixed_precision=True,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved model at {trainer.save_path}")
            return ""

        # ✅ مهم: قبل از گزارش، روی test set ارزیابی کن
        LOGGER.info("📊 Evaluating on TEST set...")
        trainer.evaluate_on_test()

        # ✅✅✅ بهترین accuracy رو از test بگیر
        if hasattr(trainer, 'test_accuracy') and trainer.test_accuracy > 0:
            trainer.best_acc = trainer.test_accuracy
            LOGGER.info(f"✅ Using TEST accuracy: {trainer.best_acc:.4f}")

        # Generate report
        output_path = trainer.generate_report()
        LOGGER.info(f"✅ ResNet18 Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ ResNet18 report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""


def generate_report_resnet50() -> str:
    """Generate report for ResNet50 from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating ResNet50 report from saved model...")
    try:
        from src.AI.ResNet.resnet50 import ResNet50Trainer

        trainer = ResNet50Trainer(
            num_workers=0, target_size=224,
            use_sampler=True, use_mixed_precision=True,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved model at {trainer.save_path}")
            return ""

        # ✅ مهم: قبل از گزارش، روی test set ارزیابی کن
        LOGGER.info("📊 Evaluating on TEST set...")
        trainer.evaluate_on_test()

        # Generate report
        output_path = trainer.generate_report()
        LOGGER.info(f"✅ ResNet50 Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ ResNet50 report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""


def generate_report_efficientnet_b0() -> str:
    """Generate report for EfficientNet-B0 from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating EfficientNet-B0 report from saved model...")
    try:
        from src.AI.EfficientNet.efficientNetB0 import EfficientNetB0Trainer

        trainer = EfficientNetB0Trainer(
            num_workers=0, target_size=224,
            use_sampler=True, use_mixed_precision=True,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved model at {trainer.save_path}")
            return ""

        LOGGER.info("📊 Evaluating on TEST set...")
        trainer.evaluate_on_test()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ EfficientNet-B0 Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ EfficientNet-B0 report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""


def generate_report_efficientnet_b1() -> str:
    """Generate report for EfficientNet-B1 from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating EfficientNet-B1 report from saved model...")
    try:
        from src.AI.EfficientNet.efficientNetB1 import EfficientNetB1Trainer

        trainer = EfficientNetB1Trainer(
            num_workers=0, target_size=240,
            use_sampler=True, use_mixed_precision=True,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved model at {trainer.save_path}")
            return ""

        LOGGER.info("📊 Evaluating on TEST set...")
        trainer.evaluate_on_test()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ EfficientNet-B1 Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ EfficientNet-B1 report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_all_reports() -> Dict[str, str]:
    """Generate reports for ALL models from saved models."""
    LOGGER.info("📄 Generating reports for ALL models...")
    LOGGER.info("=" * 70)

    results = {}

    LOGGER.info("\n📌 1/8: ResNet18 Report")
    results["resnet18"] = generate_report_resnet18()

    LOGGER.info("\n📌 2/8: ResNet50 Report")
    results["resnet50"] = generate_report_resnet50()

    LOGGER.info("\n📌 3/8: EfficientNet-B0 Report")
    results["efficientnet_b0"] = generate_report_efficientnet_b0()

    LOGGER.info("\n📌 4/8: EfficientNet-B1 Report")
    results["efficientnet_b1"] = generate_report_efficientnet_b1()

    LOGGER.info("\n📌 5/8: VGG16 Report")
    results["vgg16"] = generate_report_vgg16()

    LOGGER.info("\n📌 6/8: MobileNetV2 Report")
    results["mobilenetv2"] = generate_report_mobilenetv2()

    LOGGER.info("\n📌 7/8: GoogleNet Report")
    results["googlenet"] = generate_report_googlenet()

    LOGGER.info("\n📌 8/9: Swin Transformer Report")
    results["swin_transformer"] = generate_report_swin_transformer()

    LOGGER.info("\n📌 9/9: AlexNet Report")
    results["alexnet"] = generate_report_alexnet()

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("📊 REPORTS SUMMARY")
    LOGGER.info("=" * 70)
    for name, path in results.items():
        status = "✅" if path else "❌"
        LOGGER.info(f"   {status} {name}: {path if path else 'FAILED'}")
    LOGGER.info("=" * 70)

    return results

def generate_report_vgg16() -> str:
    """Generate report for VGG16 from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating VGG16 report from saved model...")
    try:
        from src.AI.VGG16.VGG16 import VGG16Trainer

        trainer = VGG16Trainer(
            batch_size=16,
            lr=1e-4,
            weight_decay=1e-3,              # ← مطابق با DEFAULT_HP جدید
            dropout_rate=0.15,               # ← مطابق با DEFAULT_HP جدید
            label_smoothing=0.12,            # ← مطابق با DEFAULT_HP جدید
            accumulation_steps=2,
            warmup_epochs=6,                 # ← مطابق با DEFAULT_HP جدید
            num_workers=0,
            target_size=224,
        )

        # Load best model
        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved VGG16 model found at {trainer.save_path}")
            return ""

        # ✅✅✅ مهم: قبل از گزارش، روی test set ارزیابی کن ✅✅✅
        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        # ✅✅✅ خیلی مهم: test_metrics رو ست کن تا generate_report ازش استفاده کنه
        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        # ✅✅✅ مهم: best_acc رو با test_accuracy جایگزین کن
        trainer.best_acc = test_acc
        LOGGER.info(f"✅ best_acc set to test_accuracy: {trainer.best_acc:.4f}")

        # ✅✅✅ History رو از فایل لود کن (برای Gap plot)
        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        # Generate report
        output_path = trainer.generate_report()
        LOGGER.info(f"✅ VGG16 Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ VGG16 report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_report_mobilenetv2() -> str:
    """Generate report for MobileNetV2 from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating MobileNetV2 report from saved model...")
    try:
        from src.AI.MobileNet_V2.mobilenetv2 import MobileNetV2Trainer

        trainer = MobileNetV2Trainer(
            batch_size=64,
            lr=5e-4,
            weight_decay=1e-3,               # ← مطابق با DEFAULT_HP جدید
            dropout_rate=0.15,                 # ← مطابق
            label_smoothing=0.12,             # ← مطابق
            accumulation_steps=1,
            warmup_epochs=6,                  # ← مطابق
            num_workers=0,
            target_size=224,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved MobileNetV2 model found at {trainer.save_path}")
            return ""

        # ✅✅✅ مهم: قبل از گزارش، روی test set ارزیابی کن
        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        # ✅✅✅ خیلی مهم: test_metrics رو ست کن
        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc
        trainer.best_acc = test_acc             # ← best_acc = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        # ✅ History رو از فایل لود کن
        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ MobileNetV2 Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ MobileNetV2 report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_report_googlenet() -> str:
    """Generate report for GoogleNet from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating GoogleNet report from saved model...")
    try:
        from src.AI.GoogleNet.GoogleNet import GoogleNetTrainer

        trainer = GoogleNetTrainer(
            batch_size=64,
            lr=5e-4,
            weight_decay=1e-3,               # ← مطابق با DEFAULT_HP جدید
            dropout_rate=0.15,                 # ← مطابق
            label_smoothing=0.12,             # ← مطابق
            accumulation_steps=1,
            warmup_epochs=6,                  # ← مطابق
            num_workers=0,
            target_size=224,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved GoogleNet model found at {trainer.save_path}")
            return ""

        # ✅✅✅ مهم: قبل از گزارش، روی test set ارزیابی کن
        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        # ✅✅✅ خیلی مهم: test_metrics رو ست کن
        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc
        trainer.best_acc = test_acc             # ← best_acc = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        # ✅ History رو از فایل لود کن
        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ GoogleNet Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ GoogleNet report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_report_swin_transformer() -> str:
    """Generate report for Swin Transformer from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating Swin Transformer report from saved model...")
    try:
        from src.AI.Swin_Transformer.Swin_Transformer import SwinTransformerTrainer

        trainer = SwinTransformerTrainer(
            batch_size=16,
            lr=1e-4,
            weight_decay=0.01,               # ← مطابق با DEFAULT_HP جدید
            dropout_rate=0.15,                 # ← مطابق
            label_smoothing=0.12,             # ← مطابق
            accumulation_steps=2,
            warmup_epochs=8,                  # ← مطابق
            num_workers=0,
            target_size=224,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved Swin model at {trainer.save_path}")
            return ""

        # ✅✅✅ مهم: قبل از گزارش، روی test set ارزیابی کن
        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        # ✅✅✅ خیلی مهم: test_metrics رو ست کن
        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc
        trainer.best_acc = test_acc             # ← best_acc = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        # ✅ History رو از فایل لود کن
        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ Swin Transformer Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ Swin Transformer report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_report_alexnet() -> str:
    """Generate report for AlexNet from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating AlexNet report from saved model...")
    try:
        from src.AI.AlexNet.AlexNet import AlexNetTrainer

        trainer = AlexNetTrainer(
            batch_size=32,
            lr=5e-4,
            weight_decay=5e-4,               # ← مطابق با DEFAULT_HP جدید
            dropout_rate=0.15,                # ← مطابق
            label_smoothing=0.12,             # ← مطابق
            accumulation_steps=1,
            warmup_epochs=6,                  # ← مطابق
            num_workers=0,
            target_size=224,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved AlexNet model found at {trainer.save_path}")
            return ""

        # ✅ مهم: قبل از گزارش، روی test set ارزیابی کن
        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        # ✅✅✅ خیلی مهم: test_metrics رو ست کن
        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc
        trainer.best_acc = test_acc             # ← best_acc = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        # ✅ History رو از فایل لود کن
        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ AlexNet Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ AlexNet report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_report_visio_transformer() -> str:
    """Generate report for Vision Transformer from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating Vision Transformer report from saved model...")
    try:
        from src.AI.visioTransformer.visioTransformer import VisioTransformerTrainer

        trainer = VisioTransformerTrainer(
            batch_size=16,
            lr=1e-4,
            weight_decay=0.05,               # ← مطابق با DEFAULT_HP جدید
            dropout_rate=0.15,                # ← مطابق
            label_smoothing=0.12,             # ← مطابق
            accumulation_steps=2,
            warmup_epochs=10,                 # ← مطابق
            num_workers=0,
            target_size=224,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved ViT model at {trainer.save_path}")
            return ""

        # ✅✅✅ مهم: قبل از گزارش، روی test set ارزیابی کن
        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        # ✅✅✅ خیلی مهم: test_metrics رو ست کن
        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc
        trainer.best_acc = test_acc             # ← best_acc = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        # ✅ History رو از فایل لود کن
        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ Vision Transformer Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ Vision Transformer report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_report_convnext() -> str:
    """Generate report for ConvNeXt from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating ConvNeXt report from saved model...")
    try:
        from src.AI.ConvNext.convNext import ConvNextTrainer

        trainer = ConvNextTrainer(
            batch_size=16,
            lr=1e-4,
            weight_decay=0.05,               # ← 0.01 → 0.05
            dropout_rate=0.3,                 # ← 0.25 → 0.3
            label_smoothing=0.15,             # ← 0.12 → 0.15
            accumulation_steps=2,
            warmup_epochs=8,                  # ✅
            num_workers=0,
            target_size=224,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved ConvNeXt model at {trainer.save_path}")
            return ""

        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc
        trainer.best_acc = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ ConvNeXt Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ ConvNeXt report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_report_densnet() -> str:
    """Generate report for DenseNet from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating DenseNet report from saved model...")
    try:
        from src.AI.DensNet.DensNet import DensNetTrainer

        trainer = DensNetTrainer(
            batch_size=32,
            lr=1e-4,
            weight_decay=0.01,
            dropout_rate=0.3,
            label_smoothing=0.15,
            accumulation_steps=1,
            warmup_epochs=8,
            num_workers=0,
            target_size=224,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved DenseNet model at {trainer.save_path}")
            return ""

        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc
        trainer.best_acc = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ DenseNet Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ DenseNet report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_report_cnn() -> str:
    """Generate report for Custom CNN from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating CNN report from saved model...")
    try:
        from src.AI.CNN.CNN import CNNTrainer

        trainer = CNNTrainer(
            batch_size=32,
            lr=1e-4,
            weight_decay=1e-3,
            dropout_rate=0.15,
            label_smoothing=0.1,
            accumulation_steps=1,
            warmup_epochs=8,
            num_workers=0,
            target_size=224,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved CNN model at {trainer.save_path}")
            return ""

        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc
        trainer.best_acc = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ CNN Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ CNN report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

# =====================================================
# LSTM REPORT FUNCTION
# =====================================================
def generate_report_lstm() -> str:
    """Generate report for LSTM from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating LSTM report from saved model...")
    try:
        from src.AI.LSTM.LSTM import LSTMTrainer

        trainer = LSTMTrainer(
            batch_size=16,
            lr=1e-4,
            weight_decay=1e-3,
            dropout_rate=0.2,
            label_smoothing=0.1,
            accumulation_steps=2,
            warmup_epochs=8,
            num_workers=0,
            target_size=224,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved LSTM model at {trainer.save_path}")
            return ""

        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc
        trainer.best_acc = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ LSTM Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ LSTM report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

# =====================================================
# CNN+LSTM REPORT FUNCTION
# =====================================================
def generate_report_cnn_lstm() -> str:
    """Generate report for CNN+LSTM from saved model (TEST metrics)."""
    LOGGER.info("📄 Generating CNN+LSTM report from saved model...")
    try:
        from src.AI.CNNLSTM.CNNLSTM import CNNLSTMTrainer

        trainer = CNNLSTMTrainer(
            batch_size=16,
            lr=1e-4,
            weight_decay=1e-3,
            dropout_rate=0.2,
            label_smoothing=0.1,
            accumulation_steps=2,
            warmup_epochs=8,
            num_workers=0,
            target_size=224,
        )

        if os.path.exists(trainer.save_path):
            trainer.load(trainer.save_path)
            LOGGER.info(f"✅ Loaded: {trainer.save_path}")
        else:
            LOGGER.error(f"❌ No saved CNN+LSTM model at {trainer.save_path}")
            return ""

        LOGGER.info("📊 Evaluating on TEST set...")
        test_acc, test_report, test_cm, test_auc = trainer.evaluate_on_test()

        trainer.test_metrics = test_report
        trainer.test_cm = test_cm
        trainer.test_auc = test_auc
        trainer.test_accuracy = test_acc
        trainer.best_acc = test_acc

        LOGGER.info(f"✅ Test Accuracy: {test_acc:.4f}")

        if not trainer.history.get("train_loss"):
            LOGGER.info("📊 Loading history from saved file...")
            trainer._load_history()

        output_path = trainer.generate_report()
        LOGGER.info(f"✅ CNN+LSTM Report: {output_path}")
        return output_path

    except Exception as e:
        LOGGER.error(f"❌ CNN+LSTM report failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def show_hyperparameters() -> None:
    """Show all hyperparameters."""

def show_hyperparameters() -> None:
    """Show all hyperparameters."""
    print("\n" + "=" * 70)
    print("📊 HYPERPARAMETERS")
    print("=" * 70)
    for model, hp in HYPERPARAMS.items():
        print(f"\n🔹 {model.upper()}:")
        for key, value in hp.items():
            print(f"   {key}: {value}")
    print("=" * 70)


# =====================================================
# ✅ MAIN MENU
# =====================================================


def runDeeply():
    """Main entry point."""
    print("=" * 80)
    print("🧠 LIVER AI PIPELINE - COMPLETE ULTIMATE EDITION v4.0.0")
    print("=" * 80)
    print(f"👨‍💻 Author: Babak Yousefian")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\n✨ SMART OPTIMIZATIONS ENABLED:")
    print("   ✅ Smart Learning Rate per Phase")
    print("   ✅ WeightedRandomSampler for Imbalanced Data")
    print("   ✅ GPU/CUDA Optimized Batch Sizes")
    print("   ✅ Label Smoothing for Better Generalization")
    print("   ✅ Multi-Stage Training")
    print("   ✅ 5-Fold Cross-Validation (separate menu)")
    print("   ✅ Early Stopping")
    print("   ✅ Model Checkpointing")
    print("=" * 80)

    while True:
        print("0. 🔩 PreProcessor Images")
        print("1. 🧠 Train ResNet18")
        print("2. 🧠 Train ResNet50")
        print("3. 🧠 Train EfficientNet-B0")
        print("4. 🧠 Train EfficientNet-B1")
        print("5. 🧠 Train VGG16")
        print("6. 🧠 Train MobileNetV2")
        print("7. 🧠 Train GoogleNet")
        print("8. 🧠 Train Swin Transformer")
        print("9. 🧠 Train AlexNet")
        print("10. 🧠 Train Vision Transformer")
        print("11. 🧠 Train ConvNeXt")
        print("12. 🧠 Train DenseNet")
        print("13. 🧠 Train Custom CNN")
        print("14. 🧠 Train LSTM")
        print("15. 🧠 Train CNN+LSTM")               # ← جدید
        print("16. 🚀 Train ALL Architects")
        print("")
        print("17. 🔬 Cross-Validate ResNet18 (using saved model)")
        print("18. 🔬 Cross-Validate ResNet50 (using saved model)")
        print("19. 🔬 Cross-Validate EfficientNet-B0 (using saved model)")
        print("20. 🔬 Cross-Validate EfficientNet-B1 (using saved model)")
        print("21. 🔬 Cross-Validate VGG16 (using saved model)")
        print("22. 🔬 Cross-Validate MobileNetV2 (using saved model)")
        print("23. 🔬 Cross-Validate GoogleNet (using saved model)")
        print("24. 🔬 Cross-Validate Swin Transformer (using saved model)")
        print("25. 🔬 Cross-Validate AlexNet (using saved model)")
        print("26. 🔬 Cross-Validate Vision Transformer (using saved model)")
        print("27. 🔬 Cross-Validate ConvNeXt (using saved model)")
        print("28. 🔬 Cross-Validate DenseNet (using saved model)")
        print("29. 🔬 Cross-Validate Custom CNN (using saved model)")
        print("30. 🔬 Cross-Validate LSTM (using saved model)")
        print("31. 🔬 Cross-Validate CNN+LSTM (using saved model)")  # ← جدید
        print("32. 🔬 Cross-Validate ALL Architects (using saved model)")
        print("")
        print("33. 🔬 Comprehensive Comparison")
        print("34. 📊 Literature Comparison")
        print("35. 📈 Show Hyperparameters")
        print("36. 📄 Generate-Report for VGG16")
        print("37. 📄 Generate-Report for ResNet18")
        print("38. 📄 Generate-Report for ResNet50")
        print("39. 📄 Generate-Report for EfficientNet-B0")
        print("40. 📄 Generate-Report for EfficientNet-B1")
        print("41. 📄 Generate-Report for MobileNetV2")
        print("42. 📄 Generate-Report for GoogleNet")
        print("43. 📄 Generate-Report for Swin Transformer")
        print("44. 📄 Generate-Report for AlexNet")
        print("45. 📄 Generate-Report for Vision Transformer")
        print("46. 📄 Generate-Report for ConvNeXt")
        print("47. 📄 Generate-Report for DenseNet")
        print("48. 📄 Generate-Report for Custom CNN")
        print("49. 📄 Generate-Report for LSTM")
        print("50. 📄 Generate-Report for CNN+LSTM")                 # ← جدید
        print("51. 📄 Generate-Report for ALL Architects")
        print("52. ❌ Exit")
        print("=" * 70)

        choice = input("\n👉 Enter your choice (0-52): ").strip()

        if choice == "0":
            from .preProcessoring import preProcessor
            preProcessor()
            input("\nPress Enter to continue...")
        elif choice == "1":
            train_resnet18_advanced()
            input("\nPress Enter to continue...")
        elif choice == "2":
            train_resnet50_advanced()
            input("\nPress Enter to continue...")
        elif choice == "3":
            train_efficientnet_b0_advanced()
            input("\nPress Enter to continue...")
        elif choice == "4":
            train_efficientnet_b1_advanced()
            input("\nPress Enter to continue...")
        elif choice == "5":
            train_vgg16_advanced()
            input("\nPress Enter to continue...")
        elif choice == "6":
            train_mobilenetv2_advanced()
            input("\nPress Enter to continue...")
        elif choice == "7":
            train_googlenet_advanced()
            input("\nPress Enter to continue...")
        elif choice == "8":
            train_swin_transformer_advanced()
            input("\nPress Enter to continue...")
        elif choice == "9":
            train_alexnet_advanced()
            input("\nPress Enter to continue...")
        elif choice == "10":
            train_visio_transformer_advanced()
            input("\nPress Enter to continue...")
        elif choice == "11":
            train_convnext_advanced()
            input("\nPress Enter to continue...")
        elif choice == "12":
            train_densnet_advanced()
            input("\nPress Enter to continue...")
        elif choice == "13":
            train_cnn_advanced()
            input("\nPress Enter to continue...")
        elif choice == "14":
            train_lstm_advanced()
            input("\nPress Enter to continue...")
        elif choice == "15":
            train_cnn_lstm_advanced()                   # ← جدید
            input("\nPress Enter to continue...")
        elif choice == "16":
            train_all_models_advanced()
            input("\nPress Enter to continue...")
        elif choice == "17":
            cross_validate_resnet18()
            input("\nPress Enter to continue...")
        elif choice == "18":
            cross_validate_resnet50()
            input("\nPress Enter to continue...")
        elif choice == "19":
            cross_validate_efficientnet_b0()
            input("\nPress Enter to continue...")
        elif choice == "20":
            cross_validate_efficientnet_b1()
            input("\nPress Enter to continue...")
        elif choice == "21":
            cross_validate_vgg16()
            input("\nPress Enter to continue...")
        elif choice == "22":
            cross_validate_mobilenetv2()
            input("\nPress Enter to continue...")
        elif choice == "23":
            cross_validate_googlenet()
            input("\nPress Enter to continue...")
        elif choice == "24":
            cross_validate_swin_transformer()
            input("\nPress Enter to continue...")
        elif choice == "25":
            cross_validate_alexnet()
            input("\nPress Enter to continue...")
        elif choice == "26":
            cross_validate_visio_transformer()
            input("\nPress Enter to continue...")
        elif choice == "27":
            cross_validate_convnext()
            input("\nPress Enter to continue...")
        elif choice == "28":
            cross_validate_densnet()
            input("\nPress Enter to continue...")
        elif choice == "29":
            cross_validate_cnn()
            input("\nPress Enter to continue...")
        elif choice == "30":
            cross_validate_lstm()
            input("\nPress Enter to continue...")
        elif choice == "31":
            cross_validate_cnn_lstm()                   # ← جدید
            input("\nPress Enter to continue...")
        elif choice == "32":
            cross_validate_all_models()
            input("\nPress Enter to continue...")
        elif choice == "33":
            run_comprehensive_comparison_advanced()
            input("\nPress Enter to continue...")
        elif choice == "34":
            run_literature_comparison_advanced()
            input("\nPress Enter to continue...")
        elif choice == "35":
            show_hyperparameters()
            input("\nPress Enter to continue...")
        elif choice == "36":
            generate_report_vgg16()
            input("\nPress Enter to continue...")
        elif choice == "37":
            generate_report_resnet18()
            input("\nPress Enter to continue...")
        elif choice == "38":
            generate_report_resnet50()
            input("\nPress Enter to continue...")
        elif choice == "39":
            generate_report_efficientnet_b0()
            input("\nPress Enter to continue...")
        elif choice == "40":
            generate_report_efficientnet_b1()
            input("\nPress Enter to continue...")
        elif choice == "41":
            generate_report_mobilenetv2()
            input("\nPress Enter to continue...")
        elif choice == "42":
            generate_report_googlenet()
            input("\nPress Enter to continue...")
        elif choice == "43":
            generate_report_swin_transformer()
            input("\nPress Enter to continue...")
        elif choice == "44":
            generate_report_alexnet()
            input("\nPress Enter to continue...")
        elif choice == "45":
            generate_report_visio_transformer()
            input("\nPress Enter to continue...")
        elif choice == "46":
            generate_report_convnext()
            input("\nPress Enter to continue...")
        elif choice == "47":
            generate_report_densnet()
            input("\nPress Enter to continue...")
        elif choice == "48":
            generate_report_cnn()
            input("\nPress Enter to continue...")
        elif choice == "49":
            generate_report_lstm()
            input("\nPress Enter to continue...")
        elif choice == "50":
            generate_report_cnn_lstm()                  # ← جدید
            input("\nPress Enter to continue...")
        elif choice == "51":
            generate_all_reports()
            input("\nPress Enter to continue...")
        elif choice == "52":
            print("\n👋 Goodbye!")
            print("=" * 70)
            break
        else:
            print("❌ Invalid choice! Please enter 0-52.")
            time.sleep(1)


def main():
    runDeeply()


if __name__ == "__main__":
    main()
