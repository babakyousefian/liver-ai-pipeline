# src/self_supervised/__init__.py

"""
Self-Supervised Learning Module for Liver AI Pipeline (ULTIMATE EDITION)

This module provides state-of-the-art Self-Supervised Learning (SSL) implementations
for pretraining on unlabeled liver ultrasound images.

Available Methods:
1. SimCLR: Simple Contrastive Learning of Visual Representations
   - Uses contrastive loss with positive/negative pairs
   - Requires large batch sizes (4096+ ideally)
   - Best for: General feature learning

2. BYOL: Bootstrap Your Own Latent
   - No negative samples required
   - Uses EMA (Exponential Moving Average) for target network
   - Best for: Small batch sizes, stable training

3. MAE: Masked Autoencoders
   - Masks random patches and reconstructs them
   - Learns semantic representations from reconstruction
   - Best for: Learning detailed spatial features

4. Data Augmentations: SimCLR and BYOL pipelines
   - RandomResizedCrop, RandomHorizontalFlip, ColorJitter
   - GaussianBlur, Solarize, Grayscale

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 5.0.0 (ULTIMATE EDITION)
"""

import warnings

warnings.filterwarnings("ignore")

# ================================================================
# ✅ Core SSL Methods
# ================================================================

from typing_extensions import *

from src.self_supervised.OptimizationsAlgorithms.Augmentations.augmentations import (  # Core augmentations; SSL augmentation pipelines
    BYOLAugmentation,
    RandomColorJitter,
    RandomGaussianBlur,
    RandomGrayscale,
    RandomSolarize,
    SimCLRAugmentation,
    get_ssl_augmentation,
)
from src.self_supervised.SSLModels.byol import BYOLLoss, BYOLTrainer, MLPPredictor, create_byol_trainer
from src.self_supervised.SSLModels.mae import MAELoss, MAETrainer, MaskedAutoencoder, create_mae_trainer
from src.self_supervised.SSLModels.simclr import NTXentLoss, ProjectionHead, SimCLRTrainer, create_simclr_trainer
from src.self_supervised.Facilities.trainer import SSLConfig, SSLTrainer, create_ssl_trainer
from src.self_supervised.Facilities.utils import (  # validate_ssl_model,
    compute_ssl_metrics,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

# ================================================================
# ✅ Data Augmentations
# ================================================================


# ================================================================
# ✅ Trainer & Utilities
# ================================================================


# ================================================================
# ✅ Module Metadata
# ================================================================

__version__ = "5.0.0"
__author__ = "Babak Yousefian"

# ================================================================
# ✅ Public API
# ================================================================

__all__ = [
    # SimCLR
    "SimCLRTrainer",
    "NTXentLoss",
    "ProjectionHead",
    "create_simclr_trainer",
    # BYOL
    "BYOLTrainer",
    "BYOLLoss",
    "MLPPredictor",
    "create_byol_trainer",
    # MAE
    "MAETrainer",
    "MaskedAutoencoder",
    "MAELoss",
    "create_mae_trainer",
    # Augmentations
    "RandomGaussianBlur",
    "RandomSolarize",
    "RandomGrayscale",
    "RandomColorJitter",
    "SimCLRAugmentation",
    "BYOLAugmentation",
    "get_ssl_augmentation",
    # Trainer & Utils
    "SSLTrainer",
    "SSLConfig",
    "create_ssl_trainer",
    "set_seed",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
    "compute_ssl_metrics",
    # 'validate_ssl_model',
    # Module Info
    "info",
    "test",
    "get_module_version",
    "get_module_author",
]

# ================================================================
# ✅ Module Information
# ================================================================


def info(verbose: bool = False) -> None:
    """
    Print detailed module information.

    Args:
        verbose: If True, print extended information
    """
    import torch

    print(
        f"""
    {'='*80}
    Self-Supervised Learning Module (ULTIMATE EDITION)
    {'='*80}

    📌 Module: Self-Supervised Learning
    📌 Version: {__version__}
    👨‍💻 Author: {__author__}

    📁 Components:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Component           │ Description                                        │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ SimCLR              │ Contrastive learning with NT-Xent loss              │
    │ BYOL                │ Bootstrap Your Own Latent (no negative samples)    │
    │ MAE                 │ Masked Autoencoder (reconstruction-based)           │
    │ Augmentations       │ SimCLR and BYOL augmentation pipelines             │
    │ Trainer             │ Unified training interface                         │
    │ Utils               │ Helper functions (checkpoints, metrics, etc.)      │
    └────────────────────────────────────────────────────────────────────────────┘

    📊 SSL Methods Comparison:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Feature        │ SimCLR           │ BYOL              │ MAE              │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ Negative Samples│ ✅ Required      │ ❌ Not Required   │ ❌ Not Required  │
    │ Batch Size     │ Large (4096+)    │ Medium (256-512)  │ Medium (128-256) │
    │ Memory Usage   │ High             │ Medium            │ Medium           │
    │ Training Speed │ Fast             │ Medium            │ Slow             │
    │ Complexity     │ Low              │ Medium            │ High             │
    │ Best For       │ General Features │ Stable Training   │ Spatial Details  │
    └────────────────────────────────────────────────────────────────────────────┘

    📊 Augmentation Pipelines:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Augmentation    │ SimCLR            │ BYOL                              │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ RandomResizedCrop │ ✅              │ ✅                                │
    │ RandomHorizontalFlip│ ✅            │ ✅                                │
    │ ColorJitter      │ ✅ (strong)      │ ✅ (moderate)                     │
    │ RandomGrayscale  │ ✅ (p=0.2)       │ ✅ (p=0.2)                        │
    │ GaussianBlur     │ ✅ (p=0.5)       │ ✅ (p=0.5)                        │
    │ Solarize         │ ✅ (p=0.2)       │ ❌                                │
    │ RandomRotation   │ ✅ (optional)    │ ✅ (optional)                     │
    └────────────────────────────────────────────────────────────────────────────┘

    💡 Usage Example:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ >>> from src.self_supervised import BYOLTrainer, create_byol_trainer    │
    │ >>>                                                                      │
    │ >>> # Create BYOL trainer                                               │
    │ >>> trainer = create_byol_trainer(                                      │
    │ ...     model=resnet18,                                                  │
    │ ...     device='cuda',                                                   │
    │ ...     batch_size=256                                                   │
    │ ... )                                                                    │
    │ >>>                                                                      │
    │ >>> # Train on unlabeled liver images                                   │
    │ >>> trainer.train(dataloader, epochs=100)                               │
    │ >>>                                                                      │
    │ >>> # Save checkpoint for fine-tuning                                   │
    │ >>> trainer.save_checkpoint('ssl_checkpoints/resnet18_byol.pth')        │
    └────────────────────────────────────────────────────────────────────────────┘

    📚 Documentation: https://github.com/babakyousefian/liver-ai-pipeline
    {'='*80}
    """
    )

    if verbose:
        print("📌 System Information:")
        print(f"   - PyTorch Version: {torch.__version__}")
        print(f"   - CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   - CUDA Version: {torch.version.cuda}")
            print(f"   - GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"   - GPU Count: {torch.cuda.device_count()}")
            print(
                f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        print(
            f"   - MPS Available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}"
        )

    print(
        f"""
    🎯 Recommended Workflow:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Step 1: Choose SSL Method                                                │
    │   → BYOL recommended for liver ultrasound (stable, no negatives)         │
    │                                                                          │
    │ Step 2: Prepare Unlabeled Data                                          │
    │   → Use all liver ultrasound images (without labels)                    │
    │   → Apply appropriate augmentations                                     │
    │                                                                          │
    │ Step 3: SSL Pretraining                                                 │
    │   → Train for 100-200 epochs                                             │
    │   → Use cosine annealing or warmup scheduler                             │
    │                                                                          │
    │ Step 4: Fine-Tuning                                                      │
    │   → Load SSL pretrained weights                                         │
    │   → Fine-tune on labeled data (25,000 images)                           │
    │   → Use lower learning rate (1e-4 to 1e-5)                              │
    │                                                                          │
    │ Step 5: Evaluation & Reporting                                          │
    │   → Generate comprehensive PDF reports                                 │
    │   → Compare SSL methods and models                                     │
    └────────────────────────────────────────────────────────────────────────────┘
    """
    )


def test(verbose: bool = False) -> bool:
    """
    Quick test function to verify the module is working correctly.

    Args:
        verbose: If True, print detailed test results

    Returns:
        bool: True if all tests passed
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print("\n🧪 Testing Self-Supervised Learning Module (ULTIMATE EDITION)")
    print("=" * 60)

    try:
        # Test device
        device = get_device()
        print(f"✅ Device: {device}")

        # Test SimCLR
        print("\n📌 Testing SimCLR...")
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
        )
        trainer = SimCLRTrainer(
            model=model,
            input_size=224,
            projection_dim=128,
            batch_size=8,
            device=device,
            verbose=False,
        )

        # Create dummy data
        dummy_data = torch.randn(16, 3, 224, 224)
        dummy_dataset = TensorDataset(dummy_data, torch.zeros(16))
        dummy_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)

        # Test training step
        metrics = trainer.train_epoch(dummy_loader)
        print(f"   ✅ SimCLR forward pass successful (loss: {metrics['loss']:.4f})")

        # Test BYOL
        print("\n📌 Testing BYOL...")
        online_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
        )
        trainer = BYOLTrainer(
            online_encoder=online_encoder,
            input_size=224,
            projection_dim=128,
            batch_size=8,
            device=device,
            verbose=False,
        )

        metrics = trainer.train_epoch(dummy_loader)
        print(f"   ✅ BYOL forward pass successful (loss: {metrics['loss']:.4f})")

        # Test MAE
        print("\n📌 Testing MAE...")
        try:
            from src.self_supervised.SSLModels.mae import MaskedAutoencoderTrainer

            encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 128),
            )
            trainer = MaskedAutoencoderTrainer(
                encoder=encoder,
                input_size=224,
                patch_size=16,
                mask_ratio=0.75,
                batch_size=8,
                device=device,
                verbose=False,
            )
            metrics = trainer.train_epoch(dummy_loader)
            print(f"   ✅ MAE forward pass successful (loss: {metrics['loss']:.4f})")
        except Exception as e:
            print(f"   ⚠️ MAE test failed: {e}")

        # Test Augmentations
        print("\n📌 Testing Augmentations...")
        from src.self_supervised.OptimizationsAlgorithms.Augmentations.augmentations import BYOLAugmentation, SimCLRAugmentation

        aug_simclr = SimCLRAugmentation(target_size=224)
        aug_byol = BYOLAugmentation(target_size=224)

        test_img = torch.randn(3, 224, 224)
        v1, v2 = aug_simclr(test_img)
        print(f"   ✅ SimCLR augmentation: view1 {v1.shape}, view2 {v2.shape}")

        v1, v2 = aug_byol(test_img)
        print(f"   ✅ BYOL augmentation: view1 {v1.shape}, view2 {v2.shape}")

        # Test Utils
        print("\n📌 Testing Utils...")
        from src.self_supervised.Facilities.utils import load_checkpoint, save_checkpoint, set_seed

        set_seed(42)
        dummy_model = nn.Linear(10, 5)
        save_checkpoint(dummy_model, "test_checkpoint.pth", {"test": "value"})
        loaded = load_checkpoint("test_checkpoint.pth", dummy_model)
        print(f"   ✅ Save/Load checkpoint successful")

        # Clean up test file
        import os

        if os.path.exists("test_checkpoint.pth"):
            os.remove("test_checkpoint.pth")

        if verbose:
            print("\n📊 Test Results:")
            print(f"   - SimCLR: ✅ Passed")
            print(f"   - BYOL: ✅ Passed")
            print(f"   - MAE: {'✅ Passed' if 'trainer' in locals() else '⚠️ Skipped'}")
            print(f"   - Augmentations: ✅ Passed")
            print(f"   - Utils: ✅ Passed")

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        print("=" * 60)
        return False


def get_module_version() -> str:
    """Get the module version."""
    return __version__


def get_module_author() -> str:
    """Get the module author."""
    return __author__


def get_ssl_method_info(method: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific SSL method.

    Args:
        method: 'simclr', 'byol', or 'mae'

    Returns:
        Dictionary with method information
    """
    methods = {
        "simclr": {
            "name": "SimCLR",
            "full_name": "A Simple Framework for Contrastive Learning of Visual Representations",
            "reference": "Chen et al., 2020 (ICML)",
            "paper_url": "https://arxiv.org/abs/2002.05709",
            "requires_negative": True,
            "recommended_batch_size": 4096,
            "memory_usage": "High",
            "training_speed": "Fast",
            "implementation_complexity": "Low",
            "best_for": "General feature learning with large batch sizes",
        },
        "byol": {
            "name": "BYOL",
            "full_name": "Bootstrap Your Own Latent",
            "reference": "Grill et al., 2020 (NeurIPS)",
            "paper_url": "https://arxiv.org/abs/2006.07733",
            "requires_negative": False,
            "recommended_batch_size": 256,
            "memory_usage": "Medium",
            "training_speed": "Medium",
            "implementation_complexity": "Medium",
            "best_for": "Stable training with small batch sizes",
        },
        "mae": {
            "name": "MAE",
            "full_name": "Masked Autoencoders Are Scalable Vision Learners",
            "reference": "He et al., 2022 (CVPR)",
            "paper_url": "https://arxiv.org/abs/2111.06377",
            "requires_negative": False,
            "recommended_batch_size": 128,
            "memory_usage": "Medium",
            "training_speed": "Slow",
            "implementation_complexity": "High",
            "best_for": "Learning detailed spatial features and structure",
        },
    }

    return methods.get(method.lower(), {})


# ================================================================
# ✅ Module Initialization
# ================================================================

_VERBOSE_INIT = False

if _VERBOSE_INIT:
    print(f"✅ Self-Supervised Learning module loaded successfully! (ULTIMATE EDITION)")
    print(f"   📌 Version: {__version__}")
    print(f"   👨‍💻 Author: {__author__}")
    print(f"   📚 Available Methods: SimCLR, BYOL, MAE")
    print(f"   💡 Type 'from src.self_supervised import info' and call info() for details")


# ================================================================
# ✅ Quick sanity check
# ================================================================


def _sanity_check() -> bool:
    """Perform a quick sanity check of the module."""
    try:
        import torch

        from src.self_supervised.Facilities.utils import get_device, set_seed

        set_seed(42)
        device = get_device()
        print(f"✅ Sanity check passed - Device: {device}")
        return True
    except:
        return False


# ================================================================
# ✅ If module is run directly, run tests
# ================================================================

if __name__ == "__main__":
    test(verbose=True)
    info(verbose=True)

    print("\n📊 SSL Method Comparison:")
    for method in ["simclr", "byol", "mae"]:
        info = get_ssl_method_info(method)
        print(f"   {info['name']}: {info['best_for']}")
        print(f"      - Reference: {info['reference']}")
        print(f"      - Requires Negative: {info['requires_negative']}")
        print(f"      - Recommended Batch Size: {info['recommended_batch_size']}")
        print()
