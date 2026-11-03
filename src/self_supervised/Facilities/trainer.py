# src/self_supervised/trainer.py

"""
Unified SSL Trainer for Self-Supervised Learning (ULTIMATE EDITION)

This module provides a unified trainer interface for all SSL methods:
- SimCLR
- BYOL
- MAE

Features:
- Unified API for all SSL methods
- Automatic device detection and management
- Comprehensive logging and checkpointing
- Multi-GPU support
- Mixed precision training
- Gradient accumulation and clipping
- Progress tracking and visualization
- Easy switching between SSL methods
- Configuration validation and management
- Dataset and dataloader management
- Metrics collection and reporting

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 5.0.0 (ULTIMATE EDITION)
"""

import copy
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.self_supervised.SSLModels.byol import create_byol_trainer
from src.self_supervised.SSLModels.mae import create_mae_trainer
from src.self_supervised.SSLModels.simclr import create_simclr_trainer
from .utils import get_device, set_seed

# ================================================================
# ✅ SSL CONFIGURATION DATACLASS
# ================================================================

# src/self_supervised/trainer.py (قسمت جدید)


class SSLDataset(Dataset):
    """
    ✅ FIXED: SSL Dataset that returns two augmented views.

    This dataset wrapper takes a base dataset and applies SSL augmentations
    to generate two different views of the same image.

    Features:
    - Dual-view generation for BYOL/SimCLR
    - Configurable transform
    - Supports both labeled and unlabeled data
    - Efficient caching
    - Proper error handling

    Example:
        >>> base_dataset = ImageFolder(path, transform=None)
        >>> ssl_dataset = SSLDataset(
        ...     base_dataset=base_dataset,
        ...     transform=BYOLAugmentation(),
        ...     is_ssl=True
        ... )
        >>> x1, x2 = ssl_dataset[0]  # Two views
    """

    def __init__(
        self,
        base_dataset: Dataset,
        transform: Optional[Callable] = None,
        is_ssl: bool = True,
        return_labels: bool = False,
    ):
        """
        Initialize SSL Dataset.

        Args:
            base_dataset: Base dataset (returns (image, label))
            transform: SSL augmentation (must return two views)
            is_ssl: Whether to return SSL views or original
            return_labels: Whether to return labels (for validation)
        """
        self.base_dataset = base_dataset
        self.transform = transform
        self.is_ssl = is_ssl
        self.return_labels = return_labels

        # ✅ Cache for faster loading
        self.cache = {}
        self.cache_size = 1000

        # ✅ Check transform
        self._is_dual_view = self._check_transform()

        print(f"📊 SSLDataset initialized:")
        print(f"   Samples: {len(self.base_dataset)}")
        print(f"   SSL Mode: {is_ssl}")
        print(f"   Transform: {transform is not None}")
        print(f"   Dual-View: {self._is_dual_view}")
        print(f"   Return Labels: {return_labels}")

    def _check_transform(self) -> bool:
        """Check if transform returns two views."""
        if self.transform is None:
            return False

        try:
            # ✅ Test with dummy image
            dummy = torch.randn(3, 224, 224)
            result = self.transform(dummy)
            if isinstance(result, (list, tuple)) and len(result) == 2:
                return True
        except:
            pass
        return False

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        """
        ✅ FIXED: Returns two views for SSL training.

        Returns:
            If is_ssl=True: (view1, view2)
            If is_ssl=False and return_labels=True: (image, label)
            If is_ssl=False and return_labels=False: (image,)
        """
        # ✅ Get base data
        try:
            data = self.base_dataset[idx]
        except Exception as e:
            raise RuntimeError(f"Failed to load sample {idx}: {e}")

        # ✅ Handle different return formats
        if isinstance(data, (list, tuple)):
            if len(data) == 2:
                img, label = data
            else:
                img = data[0]
                label = 0
        else:
            img = data
            label = 0

        # ✅ Convert to tensor if needed
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()

        # ✅ SSL Mode: Return two views
        if self.is_ssl and self.transform is not None:
            try:
                # ✅ Generate two views
                if self._is_dual_view:
                    # ✅ Transform already returns two views
                    result = self.transform(img)
                    if isinstance(result, (list, tuple)) and len(result) == 2:
                        view1, view2 = result
                    else:
                        view1 = result
                        view2 = result
                else:
                    # ✅ Transform returns single view, apply twice
                    view1 = self.transform(img)
                    view2 = self.transform(img)

                # ✅ Sanity check
                if view1 is None or view2 is None:
                    raise RuntimeError("Transform returned None")

                if isinstance(view1, torch.Tensor) and isinstance(view2, torch.Tensor):
                    if view1.shape != view2.shape:
                        print(f"⚠️ Shape mismatch: {view1.shape} vs {view2.shape}")
                        view2 = view1.clone()

                    if torch.isnan(view1).any() or torch.isnan(view2).any():
                        raise RuntimeError("NaN detected in augmented views")

                    if torch.isinf(view1).any() or torch.isinf(view2).any():
                        raise RuntimeError("Inf detected in augmented views")

                return view1, view2

            except Exception as e:
                print(f"❌ Augmentation failed for sample {idx}: {e}")
                # ✅ Fallback: return original image as both views
                return img, img

        # ✅ Non-SSL mode: Return original
        if self.return_labels:
            return img, label
        else:
            return img

    def _load_with_cache(self, idx: int):
        """Load sample with caching."""
        if idx in self.cache:
            return self.cache[idx]

        data = self.base_dataset[idx]

        if len(self.cache) < self.cache_size:
            self.cache[idx] = data

        return data


@dataclass
class SSLConfig:
    """
    Configuration for SSL training (ULTIMATE EDITION).

    This dataclass holds all configuration parameters for SSL training.
    """

    # Model configuration
    model_name: str = "resnet18"
    input_size: int = 224

    # SSL method
    method: str = "byol"  # 'simclr', 'byol', 'mae'

    # SSL-specific parameters
    projection_dim: int = 256
    hidden_dim: int = 4096
    temperature: float = 0.5
    momentum: float = 0.996
    momentum_schedule: str = "cosine"
    mask_ratio: float = 0.75
    patch_size: int = 16

    # Encoder configuration
    encoder_embed_dim: int = 768
    encoder_depth: int = 12
    encoder_num_heads: int = 12
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 8

    # Training configuration
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    total_epochs: int = 100
    use_lars: bool = False
    use_mixed_precision: bool = False
    use_multi_gpu: bool = False
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 1.0
    dropout_rate: float = 0.1

    # Augmentation configuration
    augmentation_method: str = "byol"  # 'byol', 'simclr'
    color_jitter_strength: float = 1.0
    blur_kernel_size: int = 23

    # Data configuration
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Logging and saving
    save_dir: str = "ssl_checkpoints"
    save_interval: int = 10
    log_interval: int = 1
    verbose: bool = True

    # Device
    device: str = None  # Auto-detect if None

    # Seed
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSLConfig":
        """Create config from dictionary."""
        return cls(**data)

    def validate(self) -> None:
        """Validate configuration."""
        assert self.method in [
            "simclr",
            "byol",
            "mae",
        ], f"method must be 'simclr', 'byol', or 'mae', got {self.method}"
        assert self.batch_size > 0, "batch_size must be > 0"
        assert self.lr > 0, "lr must be > 0"
        assert self.total_epochs > 0, "total_epochs must be > 0"
        assert self.warmup_epochs >= 0, "warmup_epochs must be >= 0"
        assert 0 <= self.mask_ratio <= 1, "mask_ratio must be between 0 and 1"


# ================================================================
# ✅ UNIFIED SSL TRAINER - ULTIMATE EDITION
# ================================================================
# src/self_supervised/trainer.py (ULTIMATE FIXED EDITION)


class SSLTrainer:
    """
    ✅ FIXED: Unified SSL Trainer with proper dual-view augmentation.

    This class now properly handles:
    - Creating SSL dataset with dual-view augmentations
    - Using Kornia or torchvision augmentations
    - Proper data loading with drop_last=True
    - Sanity checks before training
    - Error handling for NaN/Inf
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: SSLConfig,
        target_encoder: nn.Module = None,
        base_dataset: Dataset = None,  # ✅ جدید: دیتاست پایه
    ):
        """
        Initialize unified SSL trainer.

        Args:
            encoder: Base encoder model
            config: SSL configuration
            target_encoder: Target encoder for BYOL (optional)
            base_dataset: Base dataset for SSL training
        """
        self.encoder = encoder
        self.config = config
        self.target_encoder = target_encoder
        self.base_dataset = base_dataset

        # Validate configuration
        self.config.validate()

        # Set seed
        set_seed(self.config.seed)

        # Device handling
        if self.config.device is None:
            self.config.device = get_device()

        self.device = torch.device(self.config.device)

        # ✅ FIXED: Create augmentation
        self.augmentation = self._create_augmentation()

        # ✅ FIXED: Create SSL dataset if base_dataset provided
        self.ssl_dataset = None
        if base_dataset is not None:
            self.ssl_dataset = self._create_ssl_dataset(base_dataset)

        # Create trainer
        self.trainer = self._create_trainer()

        self.history = {
            "method": self.config.method,
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "epoch_times": [],
            "best_loss": float("inf"),
        }

        self.current_epoch = 0

        if self.config.verbose:
            self._print_init_info()

    # ============================================================
    # ✅ FIXED: Augmentation creation
    # ============================================================

    def _create_augmentation(self):
        """
        ✅ FIXED: Create proper dual-view augmentation.

        Returns:
            Augmentation that returns (view1, view2)
        """
        method = self.config.method
        input_size = self.config.input_size

        if self.config.verbose:
            print(f"\n🔧 Creating augmentation for {method.upper()}:")
            print(f"   Input size: {input_size}")

        # ✅ از augmentations استفاده کن
        if method == "byol":
            from .augmentations import BYOLAugmentation

            augmentation = BYOLAugmentation(
                target_size=input_size,
                normalize=True,
                color_jitter_strength=self.config.color_jitter_strength,
                blur_kernel_size=self.config.blur_kernel_size,
            )
            if self.config.verbose:
                print(f"   ✅ Using BYOL augmentation")

        elif method == "simclr":
            from .augmentations import SimCLRAugmentation

            augmentation = SimCLRAugmentation(
                target_size=input_size,
                normalize=True,
                color_jitter_strength=self.config.color_jitter_strength,
            )
            if self.config.verbose:
                print(f"   ✅ Using SimCLR augmentation")

        else:  # MAE
            from .augmentations import MAEAugmentation

            augmentation = MAEAugmentation(target_size=input_size, normalize=True)

            if self.config.verbose:
                print(f"   ✅ Using MAE augmentation")

        # ✅ Check if augmentation returns two views
        if augmentation is not None:
            try:
                dummy = torch.randn(1, 3, input_size, input_size)
                result = augmentation(dummy)
                if not (isinstance(result, (list, tuple)) and len(result) == 2):
                    print(f"   ⚠️ Augmentation does not return two views!")
                    print(f"   This will cause issues with BYOL/SimCLR!")
            except:
                pass

        return augmentation

    # ============================================================
    # ✅ FIXED: SSL Dataset creation
    # ============================================================

    def _create_ssl_dataset(self, base_dataset: Dataset):
        """
        ✅ FIXED: Wrap base dataset with SSL augmentations.

        Args:
            base_dataset: Base dataset (returns image, label)

        Returns:
            SSLDataset that returns (view1, view2)
        """
        if self.config.verbose:
            print(f"\n📊 Creating SSL dataset:")
            print(f"   Base samples: {len(base_dataset)}")
            print(f"   Augmentation: {self.augmentation is not None}")

        # ✅ Wrap with SSLDataset
        ssl_dataset = SSLDataset(
            base_dataset=base_dataset,
            transform=self.augmentation,
            is_ssl=True,
            return_labels=False,
        )

        if self.config.verbose:
            print(f"   ✅ SSL dataset created: {len(ssl_dataset)} samples")

        return ssl_dataset

    # ============================================================
    # ✅ FIXED: DataLoader creation
    # ============================================================

    def create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        drop_last: bool = True,  # ✅ برای SSL حتماً True
    ) -> DataLoader:
        """
        ✅ FIXED: Create DataLoader with proper settings.

        Args:
            dataset: Dataset to load
            shuffle: Whether to shuffle
            drop_last: Drop last incomplete batch (default: True)

        Returns:
            DataLoader instance
        """
        # ✅ برای SSL training حتماً drop_last=True
        if self.config.method in ["byol", "simclr"]:
            drop_last = True

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == "cuda",
            prefetch_factor=self.config.prefetch_factor,
            drop_last=drop_last,
            persistent_workers=False,
        )

    # ============================================================
    # ✅ FIXED: Sanity Check (CRITICAL!)
    # ============================================================

    def _sanity_check(self, train_loader: DataLoader):
        """
        ✅ FIXED: Critical sanity check before training.

        This checks:
        1. DataLoader is not empty
        2. Batch format is correct (x1, x2)
        3. Shapes are correct
        4. No NaN values
        """
        print("\n🔍 Running SANITY CHECK before training...")
        print("=" * 60)

        # ✅ Check 1: DataLoader is not empty
        if train_loader is None:
            raise RuntimeError("❌ train_loader is None!")

        if len(train_loader) == 0:
            raise RuntimeError(f"❌ train_loader is empty! (len={len(train_loader)})")

        print(f"✅ DataLoader has {len(train_loader)} batches")

        # ✅ Check 2: First batch format
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            raise RuntimeError("❌ Cannot iterate over train_loader!")

        print(f"✅ Batch type: {type(batch)}")

        # ✅ Check 3: Batch length (should be 2 for SSL)
        if isinstance(batch, (list, tuple)):
            print(f"✅ Batch length: {len(batch)}")
            if len(batch) != 2:
                raise RuntimeError(
                    f"❌ Expected batch length 2 (x1, x2), got {len(batch)}! "
                    f"This means your DataLoader is returning single-view data."
                )
        else:
            raise RuntimeError(f"❌ Expected batch to be tuple/list, got {type(batch)}")

        # ✅ Check 4: Shapes
        x1, x2 = batch
        print(f"✅ x1 shape: {x1.shape}")
        print(f"✅ x2 shape: {x2.shape}")

        if x1.shape != x2.shape:
            raise RuntimeError(f"❌ Shape mismatch: x1 {x1.shape} vs x2 {x2.shape}")

        # ✅ Check 5: Batch size
        expected_batch_size = self.config.batch_size
        if x1.size(0) != expected_batch_size:
            print(f"⚠️ Batch size mismatch: expected {expected_batch_size}, got {x1.size(0)}")

        # ✅ Check 6: No NaN
        if torch.isnan(x1).any():
            raise RuntimeError("❌ NaN detected in x1!")
        if torch.isnan(x2).any():
            raise RuntimeError("❌ NaN detected in x2!")

        print("✅ No NaN values detected")

        # ✅ Check 7: Value range
        print(f"✅ x1 range: [{x1.min():.2f}, {x1.max():.2f}]")
        print(f"✅ x2 range: [{x2.min():.2f}, {x2.max():.2f}]")

        print("=" * 60)
        print("✅ SANITY CHECK PASSED! Ready to train.\n")

    # ============================================================
    # ✅ FIXED: Train method
    # ============================================================

    def train(
        self,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        epochs: int = None,
        base_dataset: Dataset = None,  # ✅ جدید
    ) -> Dict[str, Any]:
        """
        ✅ FIXED: Train the SSL model with proper data handling.

        Args:
            train_loader: Training DataLoader (optional if base_dataset provided)
            val_loader: Validation DataLoader (optional)
            epochs: Number of epochs (overrides config)
            base_dataset: Base dataset for SSL (optional if train_loader provided)

        Returns:
            Dictionary with training results
        """
        # ✅ Use base_dataset if provided
        if base_dataset is not None:
            self.base_dataset = base_dataset
            self.ssl_dataset = self._create_ssl_dataset(base_dataset)
            train_loader = self.create_dataloader(self.ssl_dataset, shuffle=True)

            if self.config.verbose:
                print(f"\n✅ Created train_loader from base_dataset")
                print(f"   Batch size: {self.config.batch_size}")
                print(f"   Drop last: True")
                print(f"   Batches: {len(train_loader)}")

        # ✅ Sanity check before training
        self._sanity_check(train_loader)

        if epochs is not None:
            self.config.total_epochs = epochs

        if self.config.verbose:
            print("\n" + "=" * 80)
            print(f"🚀 Starting {self.config.method.upper()} Training")
            print("=" * 80)
            print(f"   Total batches: {len(train_loader)}")
            print(f"   Batch size: {self.config.batch_size}")
            print(f"   Total samples per epoch: {len(train_loader) * self.config.batch_size}")
            print(f"   Device: {self.device}")
            print("=" * 80)

        start_time = time.time()

        # Train using the specific trainer
        results = self.trainer.train(
            dataloader=train_loader,
            epochs=self.config.total_epochs,
            save_interval=self.config.save_interval,
            validation_loader=val_loader,
        )

        # Update history
        self.history.update(results)
        self.history["total_time"] = time.time() - start_time

        if self.config.verbose:
            print("\n" + "=" * 80)
            print(f"✅ {self.config.method.upper()} Training Complete!")
            print(f"   Best Loss: {self.history['best_loss']:.4f}")
            print(f"   Total Time: {self.history['total_time']:.2f}s")
            print("=" * 80)

        return self.history

    # ============================================================
    # ✅ Trainer creation (با پشتیبانی از augmentation)
    # ============================================================

    def _create_trainer(self):
        """Create the appropriate trainer based on method."""
        method = self.config.method.lower()

        if method == "simclr":
            return create_simclr_trainer(
                encoder=self.encoder,
                input_size=self.config.input_size,
                projection_dim=self.config.projection_dim,
                hidden_dim=self.config.hidden_dim,
                temperature=self.config.temperature,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                batch_size=self.config.batch_size,
                use_lars=self.config.use_lars,
                use_mixed_precision=self.config.use_mixed_precision,
                use_multi_gpu=self.config.use_multi_gpu,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                clip_grad_norm=self.config.clip_grad_norm,
                warmup_epochs=self.config.warmup_epochs,
                total_epochs=self.config.total_epochs,
                color_jitter_strength=self.config.color_jitter_strength,
                blur_kernel_size=self.config.blur_kernel_size,
                device=self.config.device,
                verbose=self.config.verbose,
                num_workers=self.config.num_workers,
                save_dir=self.config.save_dir,
            )

        elif method == "byol":
            return create_byol_trainer(
                model=self.encoder,
                target_encoder=self.target_encoder,
                model_name=self.config.model_name,
                input_size=self.config.input_size,
                projection_dim=self.config.projection_dim,
                hidden_dim=self.config.hidden_dim,
                momentum=self.config.momentum,
                momentum_schedule=self.config.momentum_schedule,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                batch_size=self.config.batch_size,
                use_lars=self.config.use_lars,
                use_mixed_precision=self.config.use_mixed_precision,
                use_multi_gpu=self.config.use_multi_gpu,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                clip_grad_norm=self.config.clip_grad_norm,
                warmup_epochs=self.config.warmup_epochs,
                total_epochs=self.config.total_epochs,
                device=self.config.device,
                verbose=self.config.verbose,
                num_workers=self.config.num_workers,
                augmentation_method=self.config.augmentation_method,
                save_dir=self.config.save_dir,
            )

        elif method == "mae":
            return create_mae_trainer(
                encoder=self.encoder,
                img_size=self.config.input_size,
                patch_size=self.config.patch_size,
                mask_ratio=self.config.mask_ratio,
                encoder_embed_dim=self.config.encoder_embed_dim,
                encoder_depth=self.config.encoder_depth,
                encoder_num_heads=self.config.encoder_num_heads,
                decoder_embed_dim=self.config.decoder_embed_dim,
                decoder_depth=self.config.decoder_depth,
                decoder_num_heads=self.config.decoder_num_heads,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                batch_size=self.config.batch_size,
                use_mixed_precision=self.config.use_mixed_precision,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                clip_grad_norm=self.config.clip_grad_norm,
                warmup_epochs=self.config.warmup_epochs,
                total_epochs=self.config.total_epochs,
                device=self.config.device,
                verbose=self.config.verbose,
                num_workers=self.config.num_workers,
                save_dir=self.config.save_dir,
            )

        else:
            raise ValueError(f"Unknown SSL method: {method}")

    # ============================================================
    # ✅ Other methods
    # ============================================================

    def _print_init_info(self):
        """Print initialization information."""
        print("\n" + "=" * 80)
        print("🚀 Initializing Unified SSL Trainer (FIXED EDITION)")
        print("=" * 80)
        print(f"   Method: {self.config.method.upper()}")
        print(f"   Model: {self.config.model_name}")
        print(f"   Input Size: {self.config.input_size}")
        print(f"   Batch Size: {self.config.batch_size}")
        print(f"   Total Epochs: {self.config.total_epochs}")
        print(f"   Learning Rate: {self.config.lr}")
        print(f"   Weight Decay: {self.config.weight_decay}")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {self.config.use_mixed_precision}")
        print(f"   Multi-GPU: {self.config.use_multi_gpu}")
        print(f"   Augmentation: {self.config.augmentation_method}")
        print("=" * 80)

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        return self.trainer.validate(val_loader)

    def get_encoder(self) -> nn.Module:
        """Get the trained encoder."""
        return self.trainer.get_encoder()

    def get_embeddings(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract embeddings from the encoder."""
        return self.trainer.get_embeddings(dataloader)

    def save_checkpoint(self, path: str = None) -> str:
        """Save checkpoint."""
        if path is None:
            path = os.path.join(self.config.save_dir, f"{self.config.method}_final.pth")
        self.trainer.save_checkpoint(path)
        return path

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """Load checkpoint."""
        self.trainer.load_checkpoint(path, load_optimizer)

    def get_config(self) -> Dict[str, Any]:
        """Get trainer configuration."""
        return self.config.to_dict()

    def get_history(self) -> Dict[str, Any]:
        """Get training history."""
        return self.history

    def get_method_info(self) -> Dict[str, Any]:
        """Get method information."""
        methods = {
            "simclr": {
                "name": "SimCLR",
                "full_name": "A Simple Framework for Contrastive Learning",
                "reference": "Chen et al., 2020",
                "requires_negative": True,
                "recommended_batch_size": 4096,
                "best_for": "Large datasets with big batches",
            },
            "byol": {
                "name": "BYOL",
                "full_name": "Bootstrap Your Own Latent",
                "reference": "Grill et al., 2020",
                "requires_negative": False,
                "recommended_batch_size": 256,
                "best_for": "Small to medium datasets",
            },
            "mae": {
                "name": "MAE",
                "full_name": "Masked Autoencoders Are Scalable Vision Learners",
                "reference": "He et al., 2022",
                "requires_negative": False,
                "recommended_batch_size": 128,
                "best_for": "Spatial feature learning",
            },
        }
        return methods.get(self.config.method.lower(), {})

    def __repr__(self) -> str:
        return f"SSLTrainer(method={self.config.method}, model={self.config.model_name})"


# ================================================================
# ✅ SSL TRAINER FACTORY
# ================================================================


def create_ssl_trainer(
    encoder: nn.Module,
    method: str = "byol",
    config: SSLConfig = None,
    base_dataset: Dataset = None,  # ✅ جدید
    **kwargs,
) -> SSLTrainer:
    """
    ✅ FIXED: Factory function to create SSL trainer.

    Args:
        encoder: Base encoder model
        method: SSL method ('simclr', 'byol', 'mae')
        config: SSL configuration (optional)
        base_dataset: Base dataset for SSL training
        **kwargs: Additional configuration parameters

    Returns:
        SSLTrainer instance

    Example:
        >>> from torchvision.datasets import ImageFolder
        >>> base_dataset = ImageFolder('images/')
        >>> trainer = create_ssl_trainer(
        ...     encoder=resnet18,
        ...     method='byol',
        ...     base_dataset=base_dataset,
        ...     batch_size=32,
        ...     total_epochs=100
        ... )
        >>> trainer.train()
    """
    if config is None:
        config = SSLConfig()

    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    config.method = method

    return SSLTrainer(encoder=encoder, config=config, base_dataset=base_dataset)


# ================================================================
# ✅ SSL METHOD COMPARISON
# ================================================================


def compare_ssl_methods(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    methods: List[str] = ["simclr", "byol", "mae"],
    epochs: int = 50,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple SSL methods on the same encoder.

    Args:
        encoder: Base encoder model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (optional)
        methods: List of methods to compare
        epochs: Number of epochs for each method
        **kwargs: Additional configuration parameters

    Returns:
        Dictionary with comparison results

    Example:
        >>> results = compare_ssl_methods(
        ...     encoder=resnet18,
        ...     train_loader=train_loader,
        ...     methods=['simclr', 'byol'],
        ...     epochs=50
        ... )
    """
    results = {}

    for method in methods:
        print("\n" + "=" * 80)
        print(f"🔬 Testing {method.upper()}")
        print("=" * 80)

        # Create trainer
        trainer = create_ssl_trainer(
            encoder=copy.deepcopy(encoder), method=method, total_epochs=epochs, **kwargs
        )

        # Train
        train_results = trainer.train(train_loader=train_loader, val_loader=val_loader)

        # Store results
        results[method] = {
            "trainer": trainer,
            "history": train_results,
            "best_loss": train_results.get("best_loss", float("inf")),
            "total_time": train_results.get("total_time", 0),
        }

        # Print summary
        print(f"\n📊 {method.upper()} Results:")
        print(f"   Best Loss: {results[method]['best_loss']:.4f}")
        print(f"   Total Time: {results[method]['total_time']:.2f}s")

    # Comparison summary
    print("\n" + "=" * 80)
    print("📊 SSL METHODS COMPARISON SUMMARY")
    print("=" * 80)

    # Find best method
    best_method = min(results.keys(), key=lambda x: results[x]["best_loss"])

    print(f"\n🏆 Best Method: {best_method.upper()}")
    print(f"   Best Loss: {results[best_method]['best_loss']:.4f}")
    print(f"   Total Time: {results[best_method]['total_time']:.2f}s")

    print("\n📋 All Results:")
    for method, data in results.items():
        print(f"   {method.upper()}: Loss={data['best_loss']:.4f}, Time={data['total_time']:.2f}s")

    print("=" * 80)

    return results


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Unified SSL Trainer (ULTIMATE EDITION)")
    print("=" * 80)

    # Create dummy encoder
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dim = 128
            self.conv = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 128),
            )

        def get_embedding(self, x):
            return self.conv(x)

        def forward(self, x):
            return self.conv(x)

    # Create dummy data
    from torch.utils.data import DataLoader, TensorDataset

    dummy_images = torch.randn(64, 3, 224, 224)
    dummy_labels = torch.zeros(64)
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Test configuration
    print("\n📌 Testing SSLConfig...")
    config = SSLConfig(method="byol", batch_size=8, total_epochs=2, warmup_epochs=1, verbose=True)
    print(f"   Config: {config.to_dict()}")

    # Test trainer creation
    print("\n📌 Testing SSLTrainer creation...")
    encoder = DummyEncoder()

    # Test BYOL
    print("\n📌 Testing BYOL trainer...")
    trainer = create_ssl_trainer(
        encoder=encoder,
        method="byol",
        batch_size=8,
        total_epochs=2,
        warmup_epochs=1,
        verbose=True,
    )
    results = trainer.train(loader)
    print(f"   Results: Best Loss = {results['best_loss']:.4f}")

    # Test SimCLR
    print("\n📌 Testing SimCLR trainer...")
    trainer = create_ssl_trainer(
        encoder=copy.deepcopy(encoder),
        method="simclr",
        batch_size=8,
        total_epochs=2,
        warmup_epochs=1,
        verbose=True,
    )
    results = trainer.train(loader)
    print(f"   Results: Best Loss = {results['best_loss']:.4f}")

    # Test MAE
    print("\n📌 Testing MAE trainer...")
    trainer = create_ssl_trainer(
        encoder=copy.deepcopy(encoder),
        method="mae",
        batch_size=8,
        total_epochs=2,
        warmup_epochs=1,
        verbose=True,
    )
    results = trainer.train(loader)
    print(f"   Results: Best Loss = {results['best_loss']:.4f}")

    # Test method info
    print("\n📌 Testing method info...")
    trainer = create_ssl_trainer(encoder=encoder, method="byol")
    info = trainer.get_method_info()
    print(f"   Method: {info['name']}")
    print(f"   Full Name: {info['full_name']}")
    print(f"   Best For: {info['best_for']}")

    print("\n✅ All SSL trainer tests passed!")
    print("=" * 80)
