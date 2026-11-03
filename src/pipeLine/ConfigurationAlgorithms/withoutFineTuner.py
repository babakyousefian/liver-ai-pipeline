"""
Main Pipeline - Liver AI Pipeline
Task 5 - Complete Implementation with Dual-Mode Menu and Reporting

This file provides a comprehensive menu system that allows users to choose between:
1. Traditional Fine-Tuning Mode (Using resNetEfficientNetModels)
2. Advanced Mode - No Fine-Tuning (ResNet18, ResNet50, EfficientNet-B0, EfficientNet-B1 from scratch)
3. Data Status Check
4. Report Generation from Saved Architects
5. Show Directory Structure
6. Cross-Validation for Architects
7. Model Comparison
8. Hyperparameter Optimization

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 7.0.0 (COMPLETE ULTIMATE EDITION)
"""

import os

# ✅ CRITICAL: Remove any CUDA restrictions
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import json
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,  # ← ✅ NEW
)

# ================================================================
# ✅ AUGMENTATION IMPORTS
# ================================================================

from torchvision import transforms

# ✅ Optional: kornia برای ultrasound enhancement
try:
    import kornia
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("⚠️ kornia not available, using torchvision only")

from src.MassiveClasses.Audio.music import Music

try:
    from src.pipeLine.Architects.VGG16.vgg16WithoutFineTuning import (
        create_vgg16_wrapper,
        ensure_directories as ensure_directories_vgg16,
    )
    WFT_VGG16_AVAILABLE = True
except ImportError as e:
    WFT_VGG16_AVAILABLE = False
    print(f"⚠️ VGG16 Without Fine-Tuning module not available: {e}")


try:
    from src.pipeLine.Architects.CNN.cnnlstmWithoutFineTuning import (
        create_cnnlstm_wrapper,
        ensure_directories as ensure_directories_cnnlstm,
    )
    WFT_CNNLSTM_AVAILABLE = True
except ImportError as e:
    WFT_CNNLSTM_AVAILABLE = False
    print(f"⚠️ CNNLSTM Without Fine-Tuning module not available: {e}")

try:
    from src.pipeLine.Architects.ConvNext.convnextWithoutFineTuning import (
        create_convnext_wrapper,
        ensure_directories as ensure_directories_convnext,
    )
    WFT_CONVNEXT_AVAILABLE = True
except ImportError as e:
    WFT_CONVNEXT_AVAILABLE = False
    print(f"⚠️ ConvNext Without Fine-Tuning module not available: {e}")

try:
    from src.pipeLine.Architects.MobileNetV2.mobilenetv2WithoutFineTuning import (
        create_mobilenetv2_wrapper,
        ensure_directories as ensure_directories_mobilenetv2,
    )
    WFT_MOBILENETV2_AVAILABLE = True
except ImportError as e:
    WFT_MOBILENETV2_AVAILABLE = False
    print(f"⚠️ MobileNetV2 Without Fine-Tuning module not available: {e}")

# ================================================================
# ✅ AlexNet Import
# ================================================================
try:
    from src.pipeLine.Architects.AlexnetFT.alexnetWithoutFineTuning import (
        create_alexnet_wrapper,
        ensure_directories as ensure_directories_alexnet,
    )
    WFT_ALEXNET_AVAILABLE = True
except ImportError as e:
    WFT_ALEXNET_AVAILABLE = False
    print(f"⚠️ AlexNet Without Fine-Tuning module not available: {e}")

# ================================================================
# ✅ ViT Import
# ================================================================
try:
    from src.pipeLine.Architects.Vision_Transformer.vitWithoutFineTuning import (
        create_vit_wrapper,
        ensure_directories as ensure_directories_vit,
    )
    WFT_VIT_AVAILABLE = True
except ImportError as e:
    WFT_VIT_AVAILABLE = False
    print(f"⚠️ ViT Without Fine-Tuning module not available: {e}")

try:
    from src.pipeLine.Architects.DenseNet.densenetWithoutFineTuning import (
        create_densenet_wrapper,
        ensure_directories as ensure_directories_densenet,
    )
    WFT_DENSENET_AVAILABLE = True
except ImportError as e:
    WFT_DENSENET_AVAILABLE = False
    print(f"⚠️ DenseNet Without Fine-Tuning module not available: {e}")

warnings.filterwarnings("ignore")

# ================================================================
# ✅ CENTRAL CONFIGURATION (NO G4)
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

VERSION = "7.0.0"
AUTHOR = "Babak Yousefian"

# ================================================================
# PATHS
# ================================================================

FINAL_REPORTS_DIR = "../../../finalReports"
CROPPED_DIR = os.path.join(FINAL_REPORTS_DIR, "cropped")
LABELS_CSV = os.path.join(FINAL_REPORTS_DIR, "labels.csv")


#FINAL_REPORTS_DIR = "finalReports"
#CROPPED_DIR = os.path.join(FINAL_REPORTS_DIR, "newLiver")
#LABELS_CSV = os.path.join(FINAL_REPORTS_DIR, "newLabel.csv")

MODELS_BASE_DIR = os.path.join(FINAL_REPORTS_DIR, "../Architects")
RESNET_BASE_DIR = os.path.join(MODELS_BASE_DIR, "../Architects/Resnet")
RESNET18_BASE_DIR = os.path.join(RESNET_BASE_DIR, "resnet18")
RESNET50_BASE_DIR = os.path.join(RESNET_BASE_DIR, "resnet50")
EFFICIENTNET_BASE_DIR = os.path.join(MODELS_BASE_DIR, "../Architects/EfficientNet")
EFFICIENTNET_B0_BASE_DIR = os.path.join(EFFICIENTNET_BASE_DIR, "../Architects/EfficientNet/B0")
EFFICIENTNET_B1_BASE_DIR = os.path.join(EFFICIENTNET_BASE_DIR, "../Architects/EfficientNet/B1")

# Sub-directories
RESNET18_CHECKPOINTS_DIR = os.path.join(RESNET18_BASE_DIR, "checkpoints")
RESNET18_CONFIGS_DIR = os.path.join(RESNET18_BASE_DIR, "configs")
RESNET18_PLOTS_DIR = os.path.join(RESNET18_BASE_DIR, "plots")
RESNET18_REPORTS_DIR = os.path.join(RESNET18_BASE_DIR, "reports")

RESNET50_CHECKPOINTS_DIR = os.path.join(RESNET50_BASE_DIR, "checkpoints")
RESNET50_CONFIGS_DIR = os.path.join(RESNET50_BASE_DIR, "configs")
RESNET50_PLOTS_DIR = os.path.join(RESNET50_BASE_DIR, "plots")
RESNET50_REPORTS_DIR = os.path.join(RESNET50_BASE_DIR, "reports")

EFFICIENTNET_B0_CHECKPOINTS_DIR = os.path.join(EFFICIENTNET_B0_BASE_DIR, "checkpoints")
EFFICIENTNET_B0_CONFIGS_DIR = os.path.join(EFFICIENTNET_B0_BASE_DIR, "configs")
EFFICIENTNET_B0_PLOTS_DIR = os.path.join(EFFICIENTNET_B0_BASE_DIR, "plots")
EFFICIENTNET_B0_REPORTS_DIR = os.path.join(EFFICIENTNET_B0_BASE_DIR, "reports")

EFFICIENTNET_B1_CHECKPOINTS_DIR = os.path.join(EFFICIENTNET_B1_BASE_DIR, "checkpoints")
EFFICIENTNET_B1_CONFIGS_DIR = os.path.join(EFFICIENTNET_B1_BASE_DIR, "configs")
EFFICIENTNET_B1_PLOTS_DIR = os.path.join(EFFICIENTNET_B1_BASE_DIR, "plots")
EFFICIENTNET_B1_REPORTS_DIR = os.path.join(EFFICIENTNET_B1_BASE_DIR, "reports")

# AI Comparisons (Traditional Mode)
AI_COMPARISONS_DIR = os.path.join(FINAL_REPORTS_DIR, "AI_Comparisons")
RESNET_AI_DIR = os.path.join(AI_COMPARISONS_DIR, "../Architects/Resnet")
EFFICIENTNET_AI_DIR = os.path.join(AI_COMPARISONS_DIR, "../Architects/EfficientNet")
FINAL_COMPARISONS_DIR = os.path.join(AI_COMPARISONS_DIR, "Final_Comparisons")
LITERATURE_COMPARISON_DIR = os.path.join(AI_COMPARISONS_DIR, "Literature_Comparison")

# VGG16
VGG16_BASE_DIR = os.path.join(MODELS_BASE_DIR, "../Architects/VGG16")
VGG16_CHECKPOINTS_DIR = os.path.join(VGG16_BASE_DIR, "checkpoints")
VGG16_CONFIGS_DIR = os.path.join(VGG16_BASE_DIR, "configs")
VGG16_PLOTS_DIR = os.path.join(VGG16_BASE_DIR, "plots")
VGG16_REPORTS_DIR = os.path.join(VGG16_BASE_DIR, "reports")

# CNNLSTM
CNNLSTM_BASE_DIR = os.path.join(MODELS_BASE_DIR, "CNNLSTM")
CNNLSTM_CHECKPOINTS_DIR = os.path.join(CNNLSTM_BASE_DIR, "checkpoints")
CNNLSTM_CONFIGS_DIR = os.path.join(CNNLSTM_BASE_DIR, "configs")
CNNLSTM_PLOTS_DIR = os.path.join(CNNLSTM_BASE_DIR, "plots")
CNNLSTM_REPORTS_DIR = os.path.join(CNNLSTM_BASE_DIR, "reports")

CONVNEXT_BASE_DIR = os.path.join(MODELS_BASE_DIR, "../Architects/ConvNext")
CONVNEXT_CHECKPOINTS_DIR = os.path.join(CONVNEXT_BASE_DIR, "checkpoints")
CONVNEXT_CONFIGS_DIR = os.path.join(CONVNEXT_BASE_DIR, "configs")
CONVNEXT_PLOTS_DIR = os.path.join(CONVNEXT_BASE_DIR, "plots")
CONVNEXT_REPORTS_DIR = os.path.join(CONVNEXT_BASE_DIR, "reports")

MOBILENETV2_BASE_DIR = os.path.join(MODELS_BASE_DIR, "../Architects/MobileNetV2")
MOBILENETV2_CHECKPOINTS_DIR = os.path.join(MOBILENETV2_BASE_DIR, "checkpoints")
MOBILENETV2_CONFIGS_DIR = os.path.join(MOBILENETV2_BASE_DIR, "configs")
MOBILENETV2_PLOTS_DIR = os.path.join(MOBILENETV2_BASE_DIR, "plots")
MOBILENETV2_REPORTS_DIR = os.path.join(MOBILENETV2_BASE_DIR, "reports")

# AlexNet
ALEXNET_BASE_DIR = os.path.join(MODELS_BASE_DIR, "AlexNet")
ALEXNET_CHECKPOINTS_DIR = os.path.join(ALEXNET_BASE_DIR, "checkpoints")
ALEXNET_CONFIGS_DIR = os.path.join(ALEXNET_BASE_DIR, "configs")
ALEXNET_PLOTS_DIR = os.path.join(ALEXNET_BASE_DIR, "plots")
ALEXNET_REPORTS_DIR = os.path.join(ALEXNET_BASE_DIR, "reports")

# ViT
VIT_BASE_DIR = os.path.join(MODELS_BASE_DIR, "visio_transformer")
VIT_CHECKPOINTS_DIR = os.path.join(VIT_BASE_DIR, "checkpoints")
VIT_CONFIGS_DIR = os.path.join(VIT_BASE_DIR, "configs")
VIT_PLOTS_DIR = os.path.join(VIT_BASE_DIR, "plots")
VIT_REPORTS_DIR = os.path.join(VIT_BASE_DIR, "reports")

DENSENET_BASE_DIR = os.path.join(MODELS_BASE_DIR, "../Architects/DenseNet")
DENSENET_CHECKPOINTS_DIR = os.path.join(DENSENET_BASE_DIR, "checkpoints")
DENSENET_CONFIGS_DIR = os.path.join(DENSENET_BASE_DIR, "configs")
DENSENET_PLOTS_DIR = os.path.join(DENSENET_BASE_DIR, "plots")
DENSENET_REPORTS_DIR = os.path.join(DENSENET_BASE_DIR, "reports")

# ================================================================
# ✅ CUSTOM TRANSFORMS FOR ULTRASOUND
# ================================================================

class UltrasoundSpeckleNoise:
    """
    Add speckle noise to simulate ultrasound artifacts.

    Speckle noise is multiplicative noise commonly seen in ultrasound images.
    Adding it helps the model become robust to real-world ultrasound noise.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.1, p: float = 0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        import random
        if random.random() > self.p:
            return img

        import numpy as np
        img_np = np.array(img).astype(np.float32) / 255.0

        # ✅ Speckle noise (multiplicative)
        noise = np.random.normal(self.mean, self.std, img_np.shape)
        noisy = img_np + img_np * noise

        # Clip to [0, 1]
        noisy = np.clip(noisy, 0, 1)

        # Convert back to PIL
        from PIL import Image
        return Image.fromarray((noisy * 255).astype(np.uint8))


class UltrasoundBrightnessShift:
    """
    Simulate brightness variations common in ultrasound.
    Ultrasound images often have varying gain/brightness settings.
    """

    def __init__(self, shift_range: Tuple[float, float] = (-0.2, 0.2), p: float = 0.5):
        self.shift_range = shift_range
        self.p = p

    def __call__(self, img):
        import random
        if random.random() > self.p:
            return img

        import numpy as np
        img_np = np.array(img).astype(np.float32) / 255.0

        # ✅ Random brightness shift
        shift = random.uniform(*self.shift_range)
        img_np = img_np + shift
        img_np = np.clip(img_np, 0, 1)

        from PIL import Image
        return Image.fromarray((img_np * 255).astype(np.uint8))


class UltrasoundContrastStretch:
    """
    Apply contrast stretching (histogram-like enhancement).
    Helps with varying contrast in ultrasound images.
    """

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img):
        import random
        if random.random() > self.p:
            return img

        import numpy as np
        img_np = np.array(img).astype(np.float32) / 255.0

        # ✅ Contrast stretch: (x - min) / (max - min)
        img_min, img_max = img_np.min(), img_np.max()
        if img_max - img_min > 1e-6:
            img_np = (img_np - img_min) / (img_max - img_min)

        from PIL import Image
        return Image.fromarray((img_np * 255).astype(np.uint8))


class RandomGammaCorrection:
    """
    Apply random gamma correction.
    Simulates different display/processing settings.
    """

    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        import random
        if random.random() > self.p:
            return img

        import numpy as np
        img_np = np.array(img).astype(np.float32) / 255.0

        # ✅ Random gamma
        gamma = random.uniform(*self.gamma_range)
        img_np = np.power(img_np, gamma)
        img_np = np.clip(img_np, 0, 1)

        from PIL import Image
        return Image.fromarray((img_np * 255).astype(np.uint8))


# ================================================================
# ✅ AUGMENTATION BUILDER
# ================================================================

def build_train_transform(
        target_size: int = 224,
        augmentation_level: str = "basic",
        use_ultrasound_transforms: bool = True,
) -> transforms.Compose:
    """
    Build training transform with configurable augmentation strength.

    Args:
        target_size: Target image size
        augmentation_level:
            - "none": No augmentation
            - "basic": Basic augmentation (Flip, Rotate, ColorJitter)
            - "medium": Medium augmentation (+ Affine, Perspective, Erasing)
            - "strong": Strong augmentation (+ RandAugment)
        use_ultrasound_transforms: Add ultrasound-specific transforms

    Returns:
        torchvision.transforms.Compose
    """

    # ==========================================
    # Base transforms (applied to PIL Image)
    # ==========================================
    base_transforms = [
        transforms.Resize((target_size + 32, target_size + 32)),
        transforms.RandomCrop((target_size, target_size)),
    ]

    # ==========================================
    # Basic augmentation (PIL Image)
    # ==========================================
    basic_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
        ),
    ]

    # ==========================================
    # Medium augmentation (PIL Image) - بدون RandomErasing
    # ==========================================
    medium_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=25),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10,
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomPerspective(
            distortion_scale=0.15,
            p=0.3,
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ], p=0.2),
        # ❌ RandomErasing از اینجا حذف شد! (باید بعد از ToTensor باشه)
    ]

    # ==========================================
    # Strong augmentation (PIL Image)
    # ==========================================
    strong_transforms = [
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=15,
        ),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        # ❌ RandomErasing از اینجا حذف شد!
    ]

    # ==========================================
    # Ultrasound-specific transforms (PIL Image)
    # ==========================================
    ultrasound_transforms = []
    if use_ultrasound_transforms:
        ultrasound_transforms = [
            UltrasoundSpeckleNoise(mean=0.0, std=0.1, p=0.3),
            UltrasoundBrightnessShift(shift_range=(-0.15, 0.15), p=0.4),
            RandomGammaCorrection(gamma_range=(0.8, 1.2), p=0.3),
        ]

    # ==========================================
    # Tensor transforms (applied AFTER ToTensor)
    # ==========================================
    tensor_transforms = []

    # ✅ RandomErasing فقط برای medium و strong
    if augmentation_level == "medium":
        tensor_transforms.append(
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3),
                value='random',
            )
        )
    elif augmentation_level == "strong":
        tensor_transforms.append(
            transforms.RandomErasing(
                p=0.3,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value='random',
            )
        )

    # ==========================================
    # Combine all
    # ==========================================
    if augmentation_level == "none":
        all_pil_transforms = base_transforms
    elif augmentation_level == "basic":
        all_pil_transforms = base_transforms + basic_transforms + ultrasound_transforms
    elif augmentation_level == "medium":
        all_pil_transforms = base_transforms + medium_transforms + ultrasound_transforms
    elif augmentation_level == "strong":
        all_pil_transforms = base_transforms + strong_transforms + ultrasound_transforms
    else:
        raise ValueError(f"Unknown augmentation_level: {augmentation_level}")

    # ✅ Final transforms (always applied)
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    # ✅ ترکیب نهایی: PIL transforms → ToTensor → Normalize → RandomErasing
    return transforms.Compose(
        all_pil_transforms + final_transforms + tensor_transforms
    )


def build_val_transform(target_size: int = 224) -> transforms.Compose:
    """
    Build validation/test transform (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ================================================================
# ✅ MIXUP / CUTMIX (Optional, for Strong augmentation)
# ================================================================

def mixup_data(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.2,
        device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup augmentation (Zhang et al., 2018).

    Creates virtual training examples by linearly interpolating between pairs
    of images and their labels.

    Args:
        x: Batch of images (B, C, H, W)
        y: Batch of labels (B,)
        alpha: Beta distribution parameter (default: 0.2)
        device: Device

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0,
        device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix augmentation (Yun et al., 2019).

    Cuts a patch from one image and pastes it onto another.

    Args:
        x: Batch of images (B, C, H, W)
        y: Batch of labels (B,)
        alpha: Beta distribution parameter (default: 1.0)
        device: Device

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=device)

    # ✅ Compute bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    # ✅ Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # ✅ Adjust lambda based on actual area
    lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
        criterion,
        pred: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float,
) -> torch.Tensor:
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ================================================================
# ✅ MODEL RESULTS STORAGE
# ================================================================


class ModelResults:
    """Singleton class to store results from all models."""

    _instance = None
    _results = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._results = {}
        return cls._instance

    @classmethod
    def add_result(cls, model_name: str, accuracy: float, metrics: Dict[str, Any]):
        cls._results[model_name] = {
            "accuracy": accuracy,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

    @classmethod
    def get_results(cls) -> Dict[str, Any]:
        return cls._results

    @classmethod
    def print_summary(cls):
        if not cls._results:
            print("\n⚠️ No results available yet. Please train some models first.")
            return

        print("\n" + "=" * 80)
        print("📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)

        sorted_results = sorted(cls._results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

        for model_name, data in sorted_results:
            metrics = data.get("metrics", {})
            print(
                f"{model_name:<20} {data['accuracy']:.4f}      "
                f"{metrics.get('precision', 0):.4f}      "
                f"{metrics.get('recall', 0):.4f}      "
                f"{metrics.get('f1_score', 0):.4f}"
            )

        print("-" * 80)

        if len(sorted_results) > 1:
            best_model = sorted_results[0]
            print(f"\n🏆 BEST MODEL: {best_model[0]} with Accuracy: {best_model[1]['accuracy']:.4f}")

        print("=" * 80)

    @classmethod
    def clear(cls):
        cls._results = {}


# ================================================================
# ✅ IMPORTS
# ================================================================

# Traditional Fine-Tuning Mode
try:
    import src.pipeLine.ConfigurationAlgorithms.resNetEfficientNetModels as resNetEfficientNetModels

    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"⚠️ Pipeline module not available: {e}")

# Advanced Mode - Without Fine-Tuning
try:
    from src.pipeLine.resnet18WithoutFineTuning import create_resnet18_wrapper
    from src.pipeLine.resnet18WithoutFineTuning import ensure_directories as ensure_directories18

    WFT_RESNET18_AVAILABLE = True
except ImportError as e:
    WFT_RESNET18_AVAILABLE = False
    print(f"⚠️ ResNet18 Without Fine-Tuning module not available: {e}")

try:
    from src.pipeLine.resnet50WithoutFineTuning import create_resnet50_wrapper
    from src.pipeLine.resnet50WithoutFineTuning import ensure_directories as ensure_directories50

    WFT_RESNET50_AVAILABLE = True
except ImportError as e:
    WFT_RESNET50_AVAILABLE = False
    print(f"⚠️ ResNet50 Without Fine-Tuning module not available: {e}")

try:
    from src.pipeLine.Architects.EfficientNet.B0.efficientNetB0WithoutFineTuning import create_efficientnet_b0_wrapper
    from src.pipeLine.Architects.EfficientNet.B0.efficientNetB0WithoutFineTuning import (
        ensure_directories as ensure_directories_b0,
    )

    WFT_EFFICIENTNET_B0_AVAILABLE = True
except ImportError as e:
    WFT_EFFICIENTNET_B0_AVAILABLE = False
    print(f"⚠️ EfficientNet-B0 Without Fine-Tuning module not available: {e}")

try:
    from src.pipeLine.Architects.EfficientNet.B1.efficientNetB1WithoutFineTuning import create_efficientnet_b1_wrapper
    from src.pipeLine.Architects.EfficientNet.B1.efficientNetB1WithoutFineTuning import (
        ensure_directories as ensure_directories_b1,
    )

    WFT_EFFICIENTNET_B1_AVAILABLE = True
except ImportError as e:
    WFT_EFFICIENTNET_B1_AVAILABLE = False
    print(f"⚠️ EfficientNet-B1 Without Fine-Tuning module not available: {e}")

# Comparisons Module
try:
    from src.AI.Comparisons.generate_comparison_with_literature import generate_comparison_report
    from src.AI.Comparisons.resnet_efficientNet_comparisons import run_comprehensive_comparison

    COMPARISONS_AVAILABLE = True
except ImportError as e:
    COMPARISONS_AVAILABLE = False
    print(f"⚠️ Comparisons module not available: {e}")


# ================================================================
# ✅ ADVANCED HYPERPARAMETERS
# ================================================================

OPTIMAL_HYPERPARAMS = {
    "resnet18": {
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 2,
        "warmup_epochs": 5,
        "epochs": 60,
        "epochs_per_phase": [10, 15, 15, 20],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 256,
        "augmentation_strength": 0.5,
    },
    "resnet50": {
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 2,
        "warmup_epochs": 5,
        "epochs": 70,
        "epochs_per_phase": [10, 12, 14, 16, 18],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 256,
        "augmentation_strength": 0.5,
    },
    "efficientnet_b0": {
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 2,
        "warmup_epochs": 5,
        "epochs": 60,
        "epochs_per_phase": [10, 15, 15, 20],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 224,
        "augmentation_strength": 0.5,
    },
    "efficientnet_b1": {
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.15,
        "label_smoothing": 0.1,
        "accumulation_steps": 2,
        "warmup_epochs": 5,
        "epochs": 60,
        "epochs_per_phase": [10, 15, 15, 20],
        "use_sampler": True,
        "use_mixed_precision": True,
        "target_size": 240,
        "augmentation_strength": 0.5,
    },
}


# ================================================================
# ✅ DIRECTORY MANAGEMENT
# ================================================================


def ensure_directories_main() -> None:
    """Create all necessary directories for the project."""
    print("\n📁 Checking and creating directories...")
    print("=" * 70)

    directories = [
        FINAL_REPORTS_DIR,
        CROPPED_DIR,
        MODELS_BASE_DIR,
        RESNET_BASE_DIR,
        RESNET18_BASE_DIR,
        RESNET50_BASE_DIR,
        EFFICIENTNET_BASE_DIR,
        EFFICIENTNET_B0_BASE_DIR,
        EFFICIENTNET_B1_BASE_DIR,
        RESNET18_CHECKPOINTS_DIR,
        RESNET18_CONFIGS_DIR,
        RESNET18_PLOTS_DIR,
        RESNET18_REPORTS_DIR,
        RESNET50_CHECKPOINTS_DIR,
        RESNET50_CONFIGS_DIR,
        RESNET50_PLOTS_DIR,
        RESNET50_REPORTS_DIR,
        EFFICIENTNET_B0_CHECKPOINTS_DIR,
        EFFICIENTNET_B0_CONFIGS_DIR,
        EFFICIENTNET_B0_PLOTS_DIR,
        EFFICIENTNET_B0_REPORTS_DIR,
        EFFICIENTNET_B1_CHECKPOINTS_DIR,
        EFFICIENTNET_B1_CONFIGS_DIR,
        EFFICIENTNET_B1_PLOTS_DIR,
        EFFICIENTNET_B1_REPORTS_DIR,
        AI_COMPARISONS_DIR,
        RESNET_AI_DIR,
        EFFICIENTNET_AI_DIR,
        FINAL_COMPARISONS_DIR,
        LITERATURE_COMPARISON_DIR,
        VGG16_BASE_DIR,
        VGG16_CHECKPOINTS_DIR,
        VGG16_CONFIGS_DIR,
        VGG16_PLOTS_DIR,
        VGG16_REPORTS_DIR,
        CNNLSTM_BASE_DIR,
        CNNLSTM_CHECKPOINTS_DIR,
        CNNLSTM_CONFIGS_DIR,
        CNNLSTM_PLOTS_DIR,
        CNNLSTM_REPORTS_DIR,
        CONVNEXT_BASE_DIR,
        CONVNEXT_CHECKPOINTS_DIR,
        CONVNEXT_CONFIGS_DIR,
        CONVNEXT_PLOTS_DIR,
        CONVNEXT_REPORTS_DIR,
        MOBILENETV2_BASE_DIR,
        MOBILENETV2_CHECKPOINTS_DIR,
        MOBILENETV2_CONFIGS_DIR,
        MOBILENETV2_PLOTS_DIR,
        MOBILENETV2_REPORTS_DIR,
        # ✅ AlexNet
        ALEXNET_BASE_DIR,
        ALEXNET_CHECKPOINTS_DIR,
        ALEXNET_CONFIGS_DIR,
        ALEXNET_PLOTS_DIR,
        ALEXNET_REPORTS_DIR,
        # ✅ ViT
        VIT_BASE_DIR,
        VIT_CHECKPOINTS_DIR,
        VIT_CONFIGS_DIR,
        VIT_PLOTS_DIR,
        VIT_REPORTS_DIR,
        # ✅ DenseNet
        DENSENET_BASE_DIR,
        DENSENET_CHECKPOINTS_DIR,
        DENSENET_CONFIGS_DIR,
        DENSENET_PLOTS_DIR,
        DENSENET_REPORTS_DIR,
    ]

    created_count = 0
    for dir_path in directories:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"   ✅ Created: {dir_path}")
                created_count += 1
            except Exception as e:
                print(f"   ❌ Error creating {dir_path}: {e}")

    print("=" * 70)
    print(f"📊 Created {created_count} new directories")
    print("✅ All directories are ready!\n")


def print_directory_structure() -> None:
    """Print the complete directory structure."""
    print("\n" + "=" * 80)
    print("📂 DIRECTORY STRUCTURE")
    print("=" * 80)

    def status(path):
        return "✅" if os.path.exists(path) else "❌"

    print(
        f"""
{FINAL_REPORTS_DIR}/
├── cropped/                                    {status(CROPPED_DIR)}
├── labels.csv                                  {'✅' if os.path.exists(LABELS_CSV) else '❌'}
├── unknown.json                                {'✅' if os.path.exists(os.path.join(FINAL_REPORTS_DIR, 'unknown.json')) else '❌'}
├── AI_Comparisons/                             {status(AI_COMPARISONS_DIR)}
│   ├── ResNet/                                 {status(RESNET_AI_DIR)}
│   │   ├── resnet18/                           {status(os.path.join(RESNET_AI_DIR, 'resnet18'))}
│   │   ├── resnet50/                           {status(os.path.join(RESNET_AI_DIR, 'resnet50'))}
│   │   └── reports/                            {status(os.path.join(RESNET_AI_DIR, 'reports'))}
│   ├── EfficientNet/                           {status(EFFICIENTNET_AI_DIR)}
│   │   ├── b0/                                 {status(os.path.join(EFFICIENTNET_AI_DIR, '../Architects/EfficientNet/B0'))}
│   │   ├── b1/                                 {status(os.path.join(EFFICIENTNET_AI_DIR, '../Architects/EfficientNet/B1'))}
│   │   └── reports/                            {status(os.path.join(EFFICIENTNET_AI_DIR, 'reports'))}
│   ├── Final_Comparisons/                      {status(FINAL_COMPARISONS_DIR)}
│   └── Literature_Comparison/                  {status(LITERATURE_COMPARISON_DIR)}
└── models/
    ├── resnet/
    │   ├── resnet18/
    │   │   ├── checkpoints/                    {status(RESNET18_CHECKPOINTS_DIR)}
    │   │   ├── configs/                        {status(RESNET18_CONFIGS_DIR)}
    │   │   ├── plots/                          {status(RESNET18_PLOTS_DIR)}
    │   │   └── reports/                        {status(RESNET18_REPORTS_DIR)}
    │   └── resnet50/
    │       ├── checkpoints/                    {status(RESNET50_CHECKPOINTS_DIR)}
    │       ├── configs/                        {status(RESNET50_CONFIGS_DIR)}
    │       ├── plots/                          {status(RESNET50_PLOTS_DIR)}
    │       └── reports/                        {status(RESNET50_REPORTS_DIR)}
    ├── efficientnet/
    │   ├── B0/
    │   │   ├── checkpoints/                    {status(EFFICIENTNET_B0_CHECKPOINTS_DIR)}
    │   │   ├── configs/                        {status(EFFICIENTNET_B0_CONFIGS_DIR)}
    │   │   ├── plots/                          {status(EFFICIENTNET_B0_PLOTS_DIR)}
    │   │   └── reports/                        {status(EFFICIENTNET_B0_REPORTS_DIR)}
    │   └── B1/
    │       ├── checkpoints/                    {status(EFFICIENTNET_B1_CHECKPOINTS_DIR)}
    │       ├── configs/                        {status(EFFICIENTNET_B1_CONFIGS_DIR)}
    │       ├── plots/                          {status(EFFICIENTNET_B1_PLOTS_DIR)}
    │       └── reports/                        {status(EFFICIENTNET_B1_REPORTS_DIR)}
    ├── VGG16/
    │   ├── checkpoints/                        {status(VGG16_CHECKPOINTS_DIR)}
    │   ├── configs/                            {status(VGG16_CONFIGS_DIR)}
    │   ├── plots/                              {status(VGG16_PLOTS_DIR)}
    │   └── reports/                            {status(VGG16_REPORTS_DIR)}
    ├── CNNLSTM/
    │   ├── checkpoints/                        {status(CNNLSTM_CHECKPOINTS_DIR)}
    │   ├── configs/                            {status(CNNLSTM_CONFIGS_DIR)}
    │   ├── plots/                              {status(CNNLSTM_PLOTS_DIR)}
    │   └── reports/                            {status(CNNLSTM_REPORTS_DIR)}
    ├── ConvNext/
    │   ├── checkpoints/                        {status(CONVNEXT_CHECKPOINTS_DIR)}
    │   ├── configs/                            {status(CONVNEXT_CONFIGS_DIR)}
    │   ├── plots/                              {status(CONVNEXT_PLOTS_DIR)}
    │   └── reports/                            {status(CONVNEXT_REPORTS_DIR)}
    ├── MobileNetV2/
    │   ├── checkpoints/                        {status(MOBILENETV2_CHECKPOINTS_DIR)}
    │   ├── configs/                            {status(MOBILENETV2_CONFIGS_DIR)}
    │   ├── plots/                              {status(MOBILENETV2_PLOTS_DIR)}
    │   └── reports/                            {status(MOBILENETV2_REPORTS_DIR)}
    └── AlexNet/
        ├── checkpoints/                        {status(ALEXNET_CHECKPOINTS_DIR)}
        ├── configs/                            {status(ALEXNET_CONFIGS_DIR)}
        ├── plots/                              {status(ALEXNET_PLOTS_DIR)}
        └── reports/                            {status(ALEXNET_REPORTS_DIR)}
    ├── visio_transformer/
    │   ├── checkpoints/                        {status(VIT_CHECKPOINTS_DIR)}
    │   ├── configs/                            {status(VIT_CONFIGS_DIR)}
    │   ├── plots/                              {status(VIT_PLOTS_DIR)}
    │   └── reports/                            {status(VIT_REPORTS_DIR)}
    ├── DenseNet/
    │   ├── checkpoints/                        {status(DENSENET_CHECKPOINTS_DIR)}
    │   ├── configs/                            {status(DENSENET_CONFIGS_DIR)}
    │   ├── plots/                              {status(DENSENET_PLOTS_DIR)}
    │   └── reports/                            {status(DENSENET_REPORTS_DIR)}
    """
    )

    print("=" * 80)

# ================================================================
# ✅ DATA LOADING
# ================================================================


class LiverDataset(Dataset):
    """Custom Dataset for loading liver images."""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def load_data_from_cropped(
        target_size: int = 224,
        use_weighted_sampler: bool = True,
        sampler_power: float = 0.5,
        augmentation_level: str = "medium",
        use_ultrasound_transforms: bool = True,
        batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load images from cropped directory and labels from CSV.

    Args:
        target_size: Target image size (default: 224)
        use_weighted_sampler: Use WeightedRandomSampler for training
        sampler_power: Power for sampler weights:
            - 1.0: Full balancing (1/count)
            - 0.5: Softer balancing (1/sqrt(count))
            - 0.0: No balancing (all weights = 1)

    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n📥 Loading data from cropped directory...")
    print("=" * 60)

    # ==========================================
    # Read CSV
    # ==========================================
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)
    print(f"   📄 Read CSV: {len(df)} rows")

    # Filter valid labels (no G4)
    df = df[df["liverLabel"].isin(CLASS_NAMES)]
    print(f"   🔍 Filtered valid labels: {len(df)} rows remaining")

    # Map labels to integers
    df["label"] = df["liverLabel"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"   📊 Valid labels: {len(df)}")

    # ==========================================
    # Get image paths
    # ==========================================
    image_paths = []
    labels = []
    missing_count = 0

    for idx, row in df.iterrows():
        img_name = row["liverName"]
        img_path = os.path.join(CROPPED_DIR, img_name)

        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(int(row["label"]))
        else:
            # Try with extensions
            base_name = os.path.splitext(img_name)[0]
            found = False
            for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
                test_path = os.path.join(CROPPED_DIR, base_name + ext)
                if os.path.exists(test_path):
                    image_paths.append(test_path)
                    labels.append(int(row["label"]))
                    found = True
                    break

            if not found:
                missing_count += 1
                if missing_count <= 5:
                    print(f"   ⚠️ Image not found: {img_name}")

    if missing_count > 5:
        print(f"   ⚠️ ... and {missing_count - 5} more images not found")

    print(f"   ✅ Found {len(image_paths)} images with valid labels")

    if len(image_paths) == 0:
        raise FileNotFoundError("No images found in cropped directory!")

    # ==========================================
    # Class distribution
    # ==========================================
    print(f"\n   📊 Class Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        class_name = INV_LABEL_MAP[label]
        print(f"      - {class_name}: {count} ({count / len(labels) * 100:.1f}%)")

    # ==========================================
    # Transforms
    # ==========================================
    print(f"\n   🎨 Data Augmentation:")
    print(f"      - Level: {augmentation_level.upper()}")
    print(f"      - Ultrasound transforms: {use_ultrasound_transforms}")

    train_transform = build_train_transform(
        target_size=target_size,
        augmentation_level=augmentation_level,
        use_ultrasound_transforms=use_ultrasound_transforms,
    )
    val_transform = build_val_transform(target_size=target_size)

    print(f"      - Train transform: {len(train_transform.transforms)} operations")
    print(f"      - Val transform: {len(val_transform.transforms)} operations")

    # ==========================================
    # Split data
    # ==========================================
    # ✅ 80% train / 10% val / 10% test
    # Step 1: split test (10%)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.10, random_state=42, stratify=labels
    )

    # Step 2: split val (10% of total = 1/9 of train_val)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=1 / 9,  # ← 1/9 × 90% = 10% from total
        random_state=42,
        stratify=train_val_labels,
    )

    # ==========================================
    # Create datasets
    # ==========================================
    train_dataset = LiverDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = LiverDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = LiverDataset(test_paths, test_labels, transform=val_transform)

    # ==========================================
    # ✅ NEW: WeightedRandomSampler for training
    # ==========================================
    num_workers = 0  # For Windows
    pin_memory = True if torch.cuda.is_available() else False

    # ✅ محاسبه sample weights
    if use_weighted_sampler:
        # محاسبه weight برای هر نمونه
        train_labels_array = np.array(train_labels)
        class_counts = np.bincount(train_labels_array, minlength=NUM_CLASSES)

        # جلوگیری از تقسیم بر صفر
        class_counts = np.maximum(class_counts, 1)

        # ✅ Inverse frequency weighting با power
        # power=1.0: 1/count (balancing کامل)
        # power=0.5: 1/sqrt(count) (balancing ملایم‌تر)
        # power=0.0: 1 (بدون balancing)
        class_weights = 1.0 / (class_counts ** sampler_power)

        # ✅ محاسبه weight هر نمونه بر اساس کلاسش
        sample_weights = class_weights[train_labels_array]

        # ✅ تبدیل به tensor
        sample_weights_tensor = torch.from_numpy(sample_weights).double()

        # ✅ ساخت WeightedRandomSampler
        train_sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True,  # ← با replacement (چون نامتوازنه)
        )

        print(f"\n   ⚖️ WeightedRandomSampler ENABLED")
        print(f"      - Power: {sampler_power}")
        print(f"      - Class counts: {class_counts.tolist()}")
        print(f"      - Class weights: {[f'{w:.4f}' for w in class_weights.tolist()]}")
        print(f"      - Sample weights (first 10): {[f'{w:.4f}' for w in sample_weights[:10].tolist()]}")
        print(f"      - Min weight: {sample_weights.min():.4f}")
        print(f"      - Max weight: {sample_weights.max():.4f}")
        print(f"      - Max/Min ratio: {sample_weights.max() / sample_weights.min():.2f}x")

        # ✅ train_loader با sampler (بدون shuffle!)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        # ✅ حالا بدون sampler (روش قبلی)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        print(f"\n   ℹ️ WeightedRandomSampler DISABLED (using shuffle=True)")

    # ==========================================
    # Val and Test loaders (بدون sampler)
    # ==========================================
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ==========================================
    # Data Split Summary
    # ==========================================
    print(f"\n   📊 Data Split:")
    print(f"      - Train: {len(train_dataset)} samples")
    print(f"      - Validation: {len(val_dataset)} samples")
    print(f"      - Test: {len(test_dataset)} samples")
    print("=" * 60)

    return train_loader, val_loader, test_loader

# ================================================================
# ✅ DATA VALIDATION
# ================================================================


def print_header() -> None:
    """Print the program header."""
    print("\n" + "=" * 80)
    print("🧠 LIVER AI PIPELINE - TASK 5")
    print("=" * 80)
    print(f"📌 Version: {VERSION}")
    print(f"👨‍💻 Author: {AUTHOR}")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\n📊 Classes: {CLASS_NAMES}")
    print(f"📊 Number of Classes: {NUM_CLASSES}")
    print("=" * 80)


def validate_labels_csv(csv_path: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate the labels CSV file."""
    validation_info = {
        "exists": False,
        "total_rows": 0,
        "valid_rows": 0,
        "unknown_rows": 0,
        "class_distribution": {},
        "has_required_columns": False,
        "errors": [],
        "warnings": [],
    }

    if not os.path.exists(csv_path):
        validation_info["errors"].append(f"CSV file not found: {csv_path}")
        return False, validation_info

    validation_info["exists"] = True

    try:
        df = pd.read_csv(csv_path)

        required_columns = ["liverName", "liverLabel"]
        if not all(col in df.columns for col in required_columns):
            validation_info["errors"].append(
                f"Missing required columns. Expected: {required_columns}"
            )
            validation_info["has_required_columns"] = False
            return False, validation_info

        validation_info["has_required_columns"] = True
        validation_info["total_rows"] = len(df)

        valid_labels = CLASS_NAMES
        unknown_count = 0
        valid_count = 0
        class_dist = {label: 0 for label in valid_labels}

        for label in df["liverLabel"].values:
            if label in valid_labels:
                valid_count += 1
                class_dist[label] += 1
            elif label == "UNKNOWN" or pd.isna(label):
                unknown_count += 1
            else:
                validation_info["warnings"].append(f"Unexpected label found: {label}")

        validation_info["valid_rows"] = valid_count
        validation_info["unknown_rows"] = unknown_count
        validation_info["class_distribution"] = class_dist

        if valid_count == 0:
            validation_info["errors"].append("No valid labels found in CSV")

        if validation_info["errors"]:
            return False, validation_info

        return True, validation_info

    except Exception as e:
        validation_info["errors"].append(f"Error reading CSV: {str(e)}")
        return False, validation_info


def check_directory_exists(directory_path: str) -> bool:
    """Check if a directory exists and is not empty."""
    return os.path.exists(directory_path) and os.path.isdir(directory_path)


def count_images_in_directory(directory_path: str) -> int:
    """Count the number of image files in a directory."""
    if not check_directory_exists(directory_path):
        return 0

    image_extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG"}
    count = 0

    try:
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                count += 1
    except Exception as e:
        print(f"⚠️ Error reading directory: {e}")
        return 0

    return count


def check_data_ready() -> Tuple[bool, Dict[str, Any]]:
    """Check if all data is ready for training."""
    status_info = {
        "cropped_exists": False,
        "cropped_images": 0,
        "csv_valid": False,
        "csv_info": {},
        "is_ready": False,
        "messages": [],
    }

    print("\n" + "=" * 80)
    print("🔍 DATA VALIDATION REPORT")
    print("=" * 80)

    # Check Cropped Directory
    print("\n📁 Checking Cropped Directory...")

    status_info["cropped_exists"] = check_directory_exists(CROPPED_DIR)

    if status_info["cropped_exists"]:
        image_count = count_images_in_directory(CROPPED_DIR)
        status_info["cropped_images"] = image_count

        if image_count > 0:
            print(f"   ✅ Cropped directory exists with {image_count} images")
            status_info["messages"].append(f"✅ Cropped: {image_count} images")
        else:
            print(f"   ⚠️ Cropped directory exists but contains no images")
            status_info["messages"].append("⚠️ Cropped directory is empty")
    else:
        print(f"   ❌ Cropped directory not found: {CROPPED_DIR}")
        status_info["messages"].append("❌ Cropped directory missing")

    # Check Labels CSV
    print("\n📄 Checking Labels CSV...")

    csv_valid, csv_info = validate_labels_csv(LABELS_CSV)
    status_info["csv_valid"] = csv_valid
    status_info["csv_info"] = csv_info

    if csv_valid:
        print(f"   ✅ CSV is valid!")
        print(f"      - Total Rows: {csv_info['total_rows']}")
        print(f"      - Valid Labels: {csv_info['valid_rows']}")
        print(f"      - UNKNOWN Labels: {csv_info['unknown_rows']}")

        status_info["messages"].append(f"✅ CSV: {csv_info['valid_rows']} valid labels")

        print(f"\n   📊 Class Distribution:")
        for label, count in csv_info["class_distribution"].items():
            if count > 0:
                print(f"      - {label}: {count}")
    else:
        print(f"   ❌ CSV validation failed:")
        for error in csv_info["errors"]:
            print(f"      - {error}")
        for warning in csv_info["warnings"]:
            print(f"      ⚠️ {warning}")

        status_info["messages"].append("❌ CSV validation failed")

    # Final Decision
    print("\n" + "-" * 80)

    if status_info["cropped_exists"] and status_info["cropped_images"] > 0 and csv_valid:
        status_info["is_ready"] = True
        print("✅ DATA IS READY FOR TRAINING!")
        status_info["messages"].append("✅ Data ready for training")
    else:
        status_info["is_ready"] = False
        print("❌ DATA IS NOT READY!")

        if not status_info["cropped_exists"] or status_info["cropped_images"] == 0:
            print("   - Missing cropped images. Please run preprocessing first.")
        if not csv_valid:
            print("   - CSV validation failed. Please check the labels file.")

        status_info["messages"].append("❌ Data not ready. Run preprocessing first.")

    print("=" * 80)

    return status_info["is_ready"], status_info


# ================================================================
# ✅ GENERIC MODEL RUNNER (WITHOUT FINE-TUNING)
# ================================================================

def run_advanced_model_generic(
    model_name: str,
    create_func: Callable,
    checkpoints_dir: str,
    reports_dir: str,
    configs_dir: str,
    plots_dir: str,
    report_name: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generic function to run any advanced model.
    """
    print("\n" + "=" * 80)
    print(f"🚀 {model_name} - WITHOUT FINE-TUNING (TASK 5)")
    print(f"📌 {model_name} From Scratch")
    print("=" * 80)

    ensure_directories_main()

    # ✅ ایجاد دایرکتوری‌ها
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    data_ready, status = check_data_ready()
    if not data_ready:
        print("\n" + "!" * 80)
        print("⚠️ DATA NOT FOUND OR INVALID!")
        print("!" * 80)
        print("\n📌 Please run the Traditional Mode first.")
        return {}

    print("\n✅ Data validation passed! Proceeding with advanced training...")

    print(f"\n📊 Data Summary:")
    print(f"   - Cropped Images: {status['cropped_images']}")
    print(f"   - Valid Labels: {status['csv_info']['valid_rows']}")
    print(f"   - Class Distribution: {status['csv_info']['class_distribution']}")

    try:
        print(f"\n🔨 Creating {model_name} model from scratch...")

        # ✅ ارسال مسیرها به مدل
        model = create_func(
            num_classes=NUM_CLASSES,
            checkpoints_dir=checkpoints_dir,  # ✅ جدید
            configs_dir=configs_dir,          # ✅ جدید
            plots_dir=plots_dir,              # ✅ جدید
            **kwargs
        )
        model.build_model()

        print("\n📊 Model Summary:")
        model.print_summary()

        # بعد:
        target_size = kwargs.get("target_size", 224)
        use_weighted_sampler = kwargs.get("use_weighted_sampler", True)
        sampler_power = kwargs.get("sampler_power", 0.5)

        # ✅ NEW: Augmentation parameters
        augmentation_level = kwargs.get("augmentation_level", "medium")
        use_ultrasound_transforms = kwargs.get("use_ultrasound_transforms", True)

        train_loader, val_loader, test_loader = load_data_from_cropped(
            target_size=target_size,
            use_weighted_sampler=use_weighted_sampler,
            sampler_power=sampler_power,
            augmentation_level=augmentation_level,
            use_ultrasound_transforms=use_ultrasound_transforms,
            batch_size=kwargs.get("batch_size", 32),
        )

        print("\n" + "=" * 80)
        print("🚀 Starting Training...")
        print("=" * 80)

        train_results = model.train_full(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=kwargs.get("epochs", 100),
            save_best=True,
        )

        print("\n" + "=" * 80)
        print("📊 Running Comprehensive Evaluation...")
        print("=" * 80)

        # ✅ Read cv_folds from kwargs (default: 0)
        cv_folds_from_kwargs = kwargs.get("cv_folds", 0)
        print(f"📊 CV Folds for this model: {cv_folds_from_kwargs}")

        # ✅ Generate report with fallback - NEVER crash on report generation
        try:
            results = model.evaluate_comprehensive(
                test_loader=test_loader,
                generate_report=True,
                cv_folds=cv_folds_from_kwargs,  # ← ✅ از kwargs می‌خونه
                report_name=report_name,
            )
        except Exception as e:
            print(f"\n⚠️ Warning during report generation: {e}")
            print("📌 Attempting fallback: generate report without checkpoint reload...")
            import traceback
            traceback.print_exc()

            # ✅ Fallback: retry evaluation
            try:
                results = model.evaluate_comprehensive(
                    test_loader=test_loader,
                    generate_report=False,  # ← No PDF
                    cv_folds=0,
                    report_name=report_name,
                )
                print("✅ Fallback evaluation succeeded (without PDF)")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                results = {
                    "metrics": {
                        "accuracy": model.best_accuracy if hasattr(model, "best_accuracy") else 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                    },
                    "cv_scores": None,
                    "model_info": {},
                    "predictions": [],
                    "labels": [],
                    "probabilities": [],
                }

        # ✅ ============================================
        # ✅ GENERATE JSON REPORT
        # ✅ ============================================
        print("\n📄 Generating JSON report...")
        
        json_path = os.path.join(reports_dir, f"{model_name.lower()}_task5_report.json")
        
        best_validation_accuracy = model.best_accuracy
        test_accuracy = results["metrics"].get("accuracy", 0)
        
        # ✅ ذخیره در reports_dir
        json_report = {
            "model_info": {
                "model": model_name,
                "version": "5.0.0",
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "checkpoints_dir": checkpoints_dir,  # ✅ ذخیره مسیر
                "reports_dir": reports_dir,          # ✅ ذخیره مسیر
                "configs_dir": configs_dir,          # ✅ ذخیره مسیر
                "plots_dir": plots_dir,              # ✅ ذخیره مسیر
                "created_at": datetime.now().isoformat(),
            },
            "training_summary": {
                "best_validation_accuracy": best_validation_accuracy,
                "best_epoch": model.best_epoch if hasattr(model, 'best_epoch') else 0,
                "total_epochs": len(model.train_history) if hasattr(model, 'train_history') else 0,
            },
            "performance_metrics": {
                "test_accuracy": test_accuracy,
                "precision": results["metrics"].get("precision", 0),
                "recall": results["metrics"].get("recall", 0),
                "f1_score": results["metrics"].get("f1_score", 0),
                "auc_score": results["metrics"].get("roc_auc", {}),
                "confusion_matrix": results["metrics"].get("confusion_matrix", []).tolist() if hasattr(results["metrics"].get("confusion_matrix", []), 'tolist') else results["metrics"].get("confusion_matrix", []),
            },
            "config": {
                "batch_size": 32,
                "base_lr": 0.001,
                "weight_decay": 1e-4,
                "dropout_rate": 0.15,
                "activation": "relu",
                "target_size": target_size,
                "epochs": kwargs.get("epochs", 100),
            },
            "class_distribution": status['csv_info']['class_distribution'],
            "generated_at": datetime.now().isoformat(),
        }
        
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=4, default=str)
        
        print(f"✅ JSON report saved: {json_path}")

        # ✅ Save model (در checkpoints_dir)
        model_path = model.save_model()
        
        # ✅ همچنین کپی مدل به checkpoints_dir
        import shutil
        if os.path.exists(model_path) and os.path.dirname(model_path) != checkpoints_dir:
            dest_path = os.path.join(checkpoints_dir, os.path.basename(model_path))
            shutil.copy2(model_path, dest_path)
            print(f"✅ Model copied to: {dest_path}")

        # ✅ Store results
        ModelResults.add_result(
            model_name=model_name,
            accuracy=model.best_accuracy,
            metrics=results["metrics"],
        )

        print("\n" + "=" * 80)
        print(f"✅ {model_name} ADVANCED MODE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📌 Checkpoints: {checkpoints_dir}")
        print(f"📌 Reports: {reports_dir}")
        print(f"📌 Configs: {configs_dir}")
        print(f"📌 Plots: {plots_dir}")
        print(f"📌 JSON: {json_path}")
        print(f"📌 Best Validation Accuracy: {model.best_accuracy:.4f}")

        print("\n📊 Performance Metrics:")
        metrics = results["metrics"]
        print(f"   - Test Accuracy:  {metrics.get('test_accuracy', 0):.4f}")
        print(f"   - Test Precision: {metrics.get('test_precision', 0):.4f}")
        print(f"   - Test Recall:    {metrics.get('test_recall', 0):.4f}")
        print(f"   - Test F1-Score:  {metrics.get('test_f1_score', 0):.4f}")
        print(f"   - Test AUC:       {metrics.get('test_auc', 0):.4f}")
        print(f"   - Test Loss:      {metrics.get('test_loss', 0):.4f}")

        print("\n" + "=" * 80)

        return {
            "model_name": model_name,
            "best_validation_accuracy": model.best_accuracy,
            "test_accuracy": metrics.get('test_accuracy', 0.0),  # ← get به جای []
            "metrics": metrics,
            "model_path": model_path,
            "json_path": json_path,
            "results": results,
        }

    except Exception as e:
        print(f"\n❌ ERROR in advanced mode: {e}")
        import traceback
        traceback.print_exc()
        return {}

# ================================================================
# ✅ ADVANCED MODE FUNCTIONS (Without Fine-Tuning)
# ================================================================
def run_advanced_mode_resnet18() -> None:
    """
    Run ResNet18 without fine-tuning with class weighting + sampler + focal loss.
    Run ResNet18 with all regularization techniques + strong augmentation.
    """
    if not WFT_RESNET18_AVAILABLE:
        print("❌ ResNet18 Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="RESNET18",
        create_func=create_resnet18_wrapper,
        checkpoints_dir=RESNET18_CHECKPOINTS_DIR,
        reports_dir=RESNET18_REPORTS_DIR,
        configs_dir=RESNET18_CONFIGS_DIR,
        plots_dir=RESNET18_PLOTS_DIR,
        report_name="ResNet18_Task5_Report.pdf",
        zero_init_residual=True,
        use_batch_norm=True,
        activation="relu",
        dropout_rate=0.15,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=256,
        # ✅ Class Weighting
        use_class_weights=True,
        class_weight_method="sqrt",
        label_smoothing=0.1,
        use_adaptive_smoothing=False,  # ← ✅ NEW (اختیاری)
        labels_csv_path=LABELS_CSV,
        # ✅ WeightedRandomSampler
        use_weighted_sampler=True,
        sampler_power=0.5,
        # ✅ NEW: Focal Loss
        use_focal_loss=False,           # ← ✅ فعال
        focal_gamma=1.0,                # ← ✅ استاندارد
        # ✅ NEW: Data Augmentation
        augmentation_level="basic",  # ← ✅ "none", "basic", "medium", "strong"
        use_ultrasound_transforms=False,  # ← ✅ برای ultrasound
        scheduler_type="cosine",  # ← ✅ Warmup + Cosine
        warmup_epochs=0,  # ← ✅ 2 epoch warmup
        min_lr=1e-6,  # ← ✅ حداقل LR
        total_epochs=100,
    )

def run_advanced_mode_resnet50() -> None:
    """
    Run ResNet50 without fine-tuning with ALL regularization techniques.

    ✅ This includes:
    - Class Weighting (sqrt method for imbalance)
    - Focal Loss (gamma=2.0)
    - Label Smoothing (0.1)
    - WeightedRandomSampler (power=0.5)
    - Strong Augmentation + Ultrasound transforms
    - Warmup + Cosine Scheduler
    - TTA in evaluation
    - Mixup in Stage 5
    """
    if not WFT_RESNET50_AVAILABLE:
        print("❌ ResNet50 Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="RESNET50",
        create_func=create_resnet50_wrapper,
        checkpoints_dir=RESNET50_CHECKPOINTS_DIR,
        reports_dir=RESNET50_REPORTS_DIR,
        configs_dir=RESNET50_CONFIGS_DIR,
        plots_dir=RESNET50_PLOTS_DIR,
        report_name="ResNet50_Task5_Report.pdf",
        # ✅ Architecture
        zero_init_residual=True,
        use_batch_norm=True,
        activation="relu",
        dropout_rate=0.15,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=256,
        # ✅ Class Weighting
        use_class_weights=True,
        class_weight_method="sqrt",
        label_smoothing=0.1,
        use_adaptive_smoothing=False,
        labels_csv_path=LABELS_CSV,
        # ✅ WeightedRandomSampler
        use_weighted_sampler=True,
        sampler_power=0.5,
        # ✅ Focal Loss
        use_focal_loss=False,
        focal_gamma=1.0,
        # ✅ Data Augmentation
        augmentation_level="basic",
        use_ultrasound_transforms=True,
        # ✅ Scheduler
        scheduler_type="cosine",
        warmup_epochs=0,
        min_lr=1e-6,
        total_epochs=100,
        # ✅ TTA & Mixup
        use_tta=True,
        use_mixup=True,
        mixup_alpha=0.2,
        mixup_start_stage=4,
        # ✅ Epochs
        epochs=100,
    )


def run_advanced_mode_efficientnet_b0() -> None:
    """Run EfficientNet-B0 without fine-tuning with all regularization."""
    if not WFT_EFFICIENTNET_B0_AVAILABLE:
        print("❌ EfficientNet-B0 Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="EFFICIENTNET-B0",
        create_func=create_efficientnet_b0_wrapper,
        checkpoints_dir=EFFICIENTNET_B0_CHECKPOINTS_DIR,
        reports_dir=EFFICIENTNET_B0_REPORTS_DIR,
        configs_dir=EFFICIENTNET_B0_CONFIGS_DIR,
        plots_dir=EFFICIENTNET_B0_PLOTS_DIR,
        report_name="EfficientNet-B0_Task5_Report.pdf",
        zero_init_residual=True,
        use_batch_norm=True,
        activation="swish",
        # ✅ Architecture
        dropout_rate=0.15,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=224,
        # ✅ Class Weighting
        use_class_weights=True,
        class_weight_method="sqrt",
        label_smoothing=0.1,
        use_adaptive_smoothing=False,
        labels_csv_path=LABELS_CSV,
        # ✅ WeightedRandomSampler
        use_weighted_sampler=True,
        sampler_power=0.5,
        # ✅ Focal Loss
        use_focal_loss=False,
        focal_gamma=1.0,
        # ✅ Data Augmentation
        augmentation_level="basic",
        use_ultrasound_transforms=True,
        # ✅ Scheduler
        scheduler_type="cosine",
        warmup_epochs=0,
        min_lr=1e-6,
        total_epochs=100,
        # ✅ TTA & Mixup
        use_tta=True,
        use_mixup=True,
        mixup_alpha=0.2,
        mixup_start_stage=4,
        # ✅ Epochs
        epochs=100,
        # ✅ CV Folds (فقط برای این مدل)
        cv_folds=5,
    )


def run_advanced_mode_efficientnet_b1() -> None:
    """
    Run EfficientNet-B1 without fine-tuning with imbalance-aware config.

    ✅ Optimized for IMBALANCED dataset (2166 images, G3=47):
    - Class Weights ENABLED (sqrt method)
    - WeightedRandomSampler ENABLED (power=0.5)
    - Focal Loss ENABLED (gamma=1.5, softer than 2.0)
    - Stronger augmentation to prevent overfit on small classes
    """
    if not WFT_EFFICIENTNET_B1_AVAILABLE:
        print("❌ EfficientNet-B1 Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="EFFICIENTNET-B1",
        create_func=create_efficientnet_b1_wrapper,
        checkpoints_dir=EFFICIENTNET_B1_CHECKPOINTS_DIR,
        reports_dir=EFFICIENTNET_B1_REPORTS_DIR,
        configs_dir=EFFICIENTNET_B1_CONFIGS_DIR,
        plots_dir=EFFICIENTNET_B1_PLOTS_DIR,
        report_name="EfficientNet-B1_Task5_Report.pdf",
        # ✅ Architecture
        zero_init_residual=True,
        use_batch_norm=True,
        activation="swish",
        dropout_rate=0.15,  # ← از 0.1 به 0.2 (imbalance → بیشتر)
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=240,
        # ✅ Class Weighting (ENABLED for imbalance)
        use_class_weights=True,  # ← از False به True
        class_weight_method="sqrt",  # ← sqrt برای imbalance
        label_smoothing=0.1,
        use_adaptive_smoothing=False,
        labels_csv_path=LABELS_CSV,
        # ✅ WeightedRandomSampler (ENABLED)
        use_weighted_sampler=True,  # ← از False به True
        sampler_power=0.5,  # ← ملایم
        # ✅ Focal Loss (ENABLED - milder gamma)
        use_focal_loss=False,  # ← از False به True
        focal_gamma=1,  # ← 1.5 (نه 2.0، برای imbalance 29:1 مناسبه)
        # ✅ Data Augmentation (STRONGER)
        augmentation_level="basic",  # ← از basic به medium
        use_ultrasound_transforms=True,  # ← از False به True
        # ✅ Scheduler
        scheduler_type="cosine",
        warmup_epochs=0,
        min_lr=1e-6,
        total_epochs=100,
        # ✅ TTA & Mixup
        use_tta=True,
        use_mixup=True,
        mixup_alpha=0.2,
        mixup_start_stage=4,
        # ✅ CV
        cv_folds=5,
        # ✅ Epochs
        epochs=100,
    )

def run_advanced_mode_vgg16() -> None:
    """
    Run VGG16 without fine-tuning with ALL regularization techniques.

    ✅ This includes:
    - Class Weighting (sqrt method for imbalance)
    - Label Smoothing (0.1)
    - WeightedRandomSampler (power=0.5)
    - Strong Augmentation + Ultrasound transforms
    - Warmup + Cosine Scheduler
    - TTA in evaluation
    - Mixup in Stage 5
    """
    if not WFT_VGG16_AVAILABLE:
        print("❌ VGG16 Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="VGG16",
        create_func=create_vgg16_wrapper,
        checkpoints_dir=VGG16_CHECKPOINTS_DIR,
        reports_dir=VGG16_REPORTS_DIR,
        configs_dir=VGG16_CONFIGS_DIR,
        plots_dir=VGG16_PLOTS_DIR,
        report_name="VGG16_Task5_Report.pdf",
        # ✅ Architecture
        use_batch_norm=True,
        activation="relu",
        dropout_rate=0.5,                # ← VGG16 به dropout بالا نیاز داره
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=224,
        # ✅ Class Weighting
        use_class_weights=True,
        class_weight_method="sqrt",
        label_smoothing=0.1,
        labels_csv_path=LABELS_CSV,
        # ✅ WeightedRandomSampler
        use_weighted_sampler=True,
        sampler_power=0.3,  # ← 0.5 → 0.3 (کمتر aggressive)
        # ✅ Focal Loss (optional)
        use_focal_loss=False,
        focal_gamma=1.0,
        # ✅ Data Augmentation
        augmentation_level="basic",
        use_ultrasound_transforms=True,
        # ✅ Scheduler
        scheduler_type="cosine",
        warmup_epochs=0,
        min_lr=1e-6,
        total_epochs=50,
        # ✅ TTA & Mixup
        use_tta=True,
        use_mixup=False,  # ← True → False (VGG16 from-scratch با mixup مشکل داره)
        mixup_alpha=0.2,
        mixup_start_stage=4,
        # ✅ Epochs
        epochs=50,
    )

def run_advanced_mode_cnnlstm() -> None:
    """Run CNNLSTM without fine-tuning with ALL regularization techniques."""
    if not WFT_CNNLSTM_AVAILABLE:
        print("❌ CNNLSTM Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="CNNLSTM",
        create_func=create_cnnlstm_wrapper,
        checkpoints_dir=CNNLSTM_CHECKPOINTS_DIR,
        reports_dir=CNNLSTM_REPORTS_DIR,
        configs_dir=CNNLSTM_CONFIGS_DIR,
        plots_dir=CNNLSTM_PLOTS_DIR,
        report_name="CNNLSTM_Task5_Report.pdf",
        # ✅ Architecture
        use_batch_norm=True,
        activation="relu",
        dropout_rate=0.2,
        hidden_dim=256,
        num_layers=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=224,
        # ✅ Class Weighting
        use_class_weights=True,
        class_weight_method="sqrt",
        label_smoothing=0.1,
        labels_csv_path=LABELS_CSV,
        # ✅ WeightedRandomSampler
        use_weighted_sampler=True,
        sampler_power=0.3,
        # ✅ Focal Loss (optional)
        use_focal_loss=False,
        focal_gamma=1.0,
        # ✅ Data Augmentation
        augmentation_level="basic",
        use_ultrasound_transforms=True,
        # ✅ Scheduler
        scheduler_type="cosine",
        warmup_epochs=0,
        min_lr=1e-6,
        total_epochs=60,
        # ✅ TTA & Mixup
        use_tta=True,
        use_mixup=True,
        mixup_alpha=0.2,
        mixup_start_stage=4,
        # ✅ Batch Size
        batch_size=16,
        # ✅ Epochs
        epochs=60,
    )

def run_advanced_mode_convnext() -> None:
    """Run ConvNext without fine-tuning with ALL regularization techniques."""
    if not WFT_CONVNEXT_AVAILABLE:
        print("❌ ConvNext Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="ConvNext",
        create_func=create_convnext_wrapper,
        checkpoints_dir=CONVNEXT_CHECKPOINTS_DIR,
        reports_dir=CONVNEXT_REPORTS_DIR,
        configs_dir=CONVNEXT_CONFIGS_DIR,
        plots_dir=CONVNEXT_PLOTS_DIR,
        report_name="ConvNext_Task5_Report.pdf",
        # ✅ Architecture
        dims=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
        dropout_rate=0.2,                 # ← 0.3 → 0.2 (کمتر)
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=224,
        # ✅ Class Weighting
        use_class_weights=True,
        class_weight_method="sqrt",
        label_smoothing=0.1,              # ← 0.15 → 0.1 (استاندارد)
        labels_csv_path=LABELS_CSV,
        # ✅ WeightedRandomSampler
        use_weighted_sampler=True,
        sampler_power=0.3,
        # ✅ Focal Loss (optional)
        use_focal_loss=False,
        focal_gamma=1.0,
        # ✅ Data Augmentation
        augmentation_level="basic",
        use_ultrasound_transforms=True,
        # ✅ Scheduler
        scheduler_type="cosine",
        warmup_epochs=5,                  # ✅ خوبه
        min_lr=1e-6,                      # ← 1e-4 → 1e-6 (خیلی پایین، برای convergence نهایی)
        total_epochs=60,                 # ← 80 → 100 (فرصت بیشتر)
        # ✅ TTA & Mixup
        use_tta=True,
        use_mixup=False,
        mixup_alpha=0.2,
        mixup_start_stage=4,
        # ✅ Batch Size
        batch_size=16,
        # ✅ Epochs
        epochs=60,                       # ← 80 → 100
    )

def run_advanced_mode_mobilenetv2() -> None:
    """Run MobileNetV2 without fine-tuning with ALL regularization techniques."""
    if not WFT_MOBILENETV2_AVAILABLE:
        print("❌ MobileNetV2 Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="MobileNetV2",
        create_func=create_mobilenetv2_wrapper,
        checkpoints_dir=MOBILENETV2_CHECKPOINTS_DIR,
        reports_dir=MOBILENETV2_REPORTS_DIR,
        configs_dir=MOBILENETV2_CONFIGS_DIR,
        plots_dir=MOBILENETV2_PLOTS_DIR,
        report_name="MobileNetV2_Task5_Report.pdf",
        # ✅ Architecture
        width_mult=1.0,
        dropout_rate=0.15,               # ✅ از تجربه‌ی CNNLSTM
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=224,
        # ✅ Class Weighting
        use_class_weights=True,
        class_weight_method="sqrt",
        label_smoothing=0.12,
        labels_csv_path=LABELS_CSV,
        # ✅ WeightedRandomSampler
        use_weighted_sampler=True,
        sampler_power=0.5,
        # ✅ Focal Loss
        use_focal_loss=False,
        focal_gamma=1.0,
        # ✅ Data Augmentation
        augmentation_level="basic",
        use_ultrasound_transforms=True,
        # ✅ Scheduler
        scheduler_type="cosine",
        warmup_epochs=6,
        min_lr=1e-6,
        total_epochs=60,
        # ✅ TTA & Mixup
        use_tta=True,
        use_mixup=True,
        mixup_alpha=0.2,
        mixup_start_stage=4,
        # ✅ Batch Size
        batch_size=64,
        # ✅ Epochs
        epochs=60,
    )

def run_advanced_mode_alexnet() -> None:
    """
    Run AlexNet without fine-tuning with ALL regularization techniques.

    ✅ Optimized config (validated on MobileNetV2 → 83.40%):
    - width_mult=0.5 (reduces params from 60M to ~10M)
    - dropout_fc=0.4
    - sampler_power=0.5
    - label_smoothing=0.12
    - use_focal_loss=False (sampler+weights enough)
    - augmentation_level="basic" + ultrasound_transforms
    - warmup_epochs=6, total_epochs=60
    - use_mixup=True, use_tta=True
    - batch_size=64
    """
    if not WFT_ALEXNET_AVAILABLE:
        print("❌ AlexNet Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="AlexNet",
        create_func=create_alexnet_wrapper,
        checkpoints_dir=ALEXNET_CHECKPOINTS_DIR,
        reports_dir=ALEXNET_REPORTS_DIR,
        configs_dir=ALEXNET_CONFIGS_DIR,
        plots_dir=ALEXNET_PLOTS_DIR,
        report_name="AlexNet_Task5_Report.pdf",
        # ✅ Architecture (modernized)
        width_mult=0.5,                    # ~10M params (vs 60M original)
        fc6_features=1024,                 # 4096 → 1024
        fc7_features=512,                  # 4096 → 512
        dropout_fc=0.4,                    # 0.5 → 0.4
        dropout_conv_last=0.0,
        use_batch_norm=True,               # replaces LRN
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=224,
        # ✅ Class Weighting
        use_class_weights=True,
        class_weight_method="sqrt",
        label_smoothing=0.12,
        labels_csv_path=LABELS_CSV,
        # ✅ WeightedRandomSampler
        use_weighted_sampler=True,
        sampler_power=0.5,
        # ✅ Focal Loss (off — sampler+weights enough)
        use_focal_loss=False,
        focal_gamma=1.0,
        # ✅ Data Augmentation
        augmentation_level="basic",
        use_ultrasound_transforms=True,
        # ✅ Scheduler
        scheduler_type="cosine",
        warmup_epochs=6,
        min_lr=1e-6,
        total_epochs=60,
        # ✅ TTA & Mixup
        use_tta=True,
        use_mixup=True,
        mixup_alpha=0.2,
        mixup_start_stage=4,
        # ✅ Optimizer
        learning_rate=1e-3,
        weight_decay=2e-3,
        # ✅ Batch Size
        batch_size=32,
        # ✅ Epochs
        epochs=60,
    )

def run_advanced_mode_vit() -> None:
    """
    Run Vision Transformer (ViT) without fine-tuning.

    ✅ Optimized config (validated on MobileNetV2 → 83.40%, AlexNet pending):
    - ViT-Tiny (embed_dim=192, depth=6, heads=3, ~5.7M params)
    - patch_size=16
    - dropout=0.2, drop_path_rate=0.1
    - sampler_power=0.5
    - label_smoothing=0.12
    - use_focal_loss=False (sampler+weights enough)
    - augmentation_level="basic" + ultrasound_transforms
    - warmup_epochs=10 (ViT needs longer warmup)
    - total_epochs=80 (ViT needs more epochs)
    - use_mixup=True, use_tta=True
    - batch_size=64
    - learning_rate=5e-4, weight_decay=0.05 (ViT standard)
    """
    if not WFT_VIT_AVAILABLE:
        print("❌ ViT Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="visio_transformer",
        create_func=create_vit_wrapper,
        checkpoints_dir=VIT_CHECKPOINTS_DIR,
        reports_dir=VIT_REPORTS_DIR,
        configs_dir=VIT_CONFIGS_DIR,
        plots_dir=VIT_PLOTS_DIR,
        report_name="ViT_Task5_Report.pdf",
        # ✅ Architecture (ViT-Tiny)
        variant="tiny",
        img_size=224,
        patch_size=16,
        dropout=0.2,
        attn_dropout=0.0,
        drop_path_rate=0.1,
        head_dropout=0.2,
        use_cls_token=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=224,
        # ✅ Class Weighting
        use_class_weights=True,
        class_weight_method="sqrt",
        label_smoothing=0.12,
        labels_csv_path=LABELS_CSV,
        # ✅ WeightedRandomSampler
        use_weighted_sampler=True,
        sampler_power=0.5,
        # ✅ Focal Loss (off)
        use_focal_loss=False,
        focal_gamma=1.0,
        # ✅ Data Augmentation
        augmentation_level="basic",
        use_ultrasound_transforms=True,
        # ✅ Scheduler (ViT needs long warmup)
        scheduler_type="warmup_cosine",
        warmup_epochs=10,
        min_lr=1e-6,
        total_epochs=80,
        # ✅ TTA & Mixup
        use_tta=True,
        use_mixup=True,
        mixup_alpha=0.2,
        mixup_start_stage=5,
        use_cutmix=False,
        # ✅ Optimizer (ViT standard)
        learning_rate=5e-4,
        weight_decay=0.05,
        # ✅ Batch
        batch_size=64,
        # ✅ Epochs
        epochs=80,
    )

def run_advanced_mode_densenet() -> None:
    """Run DenseNet without fine-tuning — MEDIUM variant (if small underfits)."""
    if not WFT_DENSENET_AVAILABLE:
        print("❌ DenseNet Without Fine-Tuning module not available!")
        return

    run_advanced_model_generic(
        model_name="DenseNet",
        create_func=create_densenet_wrapper,
        checkpoints_dir=DENSENET_CHECKPOINTS_DIR,
        reports_dir=DENSENET_REPORTS_DIR,
        configs_dir=DENSENET_CONFIGS_DIR,
        plots_dir=DENSENET_PLOTS_DIR,
        report_name="DenseNet_Task5_Report.pdf",
        # ✅ Architecture — MEDIUM (larger, if small underfits)
        variant="medium",
        layers=[6, 12, 24, 16],          # ← medium
        growth_rate=24,                  # ← medium
        bn_size=4,
        compression=0.5,
        init_channels=32,
        dropout=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=224,
        use_class_weights=True,
        class_weight_method="sqrt",
        label_smoothing=0.12,
        labels_csv_path=LABELS_CSV,
        use_weighted_sampler=True,
        sampler_power=0.5,
        use_focal_loss=False,
        focal_gamma=1.0,
        augmentation_level="basic",
        use_ultrasound_transforms=True,
        scheduler_type="warmup_cosine",
        warmup_epochs=6,
        min_lr=1e-6,
        total_epochs=70,
        use_tta=True,
        use_mixup=True,
        mixup_alpha=0.2,
        mixup_start_stage=4,
        learning_rate=1e-3,
        weight_decay=1e-2,               # ← بالاتر برای medium
        batch_size=64,
        epochs=70,
    )

# ================================================================
# ✅ RUN MULTIPLE MODELS (Without Fine-Tuning)
# ================================================================


def run_both_models() -> None:
    """Run both ResNet18 and ResNet50."""
    print("\n" + "=" * 80)
    print("🚀 RUNNING BOTH MODELS (RESNET18 & RESNET50)")
    print("=" * 80)

    run_advanced_mode_resnet18()
    run_advanced_mode_resnet50()

    ModelResults.print_summary()

    print("\n" + "=" * 80)
    print("✅ BOTH MODELS COMPLETED!")
    print("=" * 80)


def run_all_models() -> None:
    """Run ResNet18, ResNet50, EfficientNet-B0."""
    print("\n" + "=" * 80)
    print("🚀 RUNNING ALL MODELS (ResNet18, ResNet50, EfficientNet-B0)")
    print("=" * 80)

    run_advanced_mode_resnet18()
    run_advanced_mode_resnet50()
    run_advanced_mode_efficientnet_b0()

    ModelResults.print_summary()

    print("\n" + "=" * 80)
    print("✅ ALL MODELS COMPLETED!")
    print("=" * 80)


def run_all_models_with_b1() -> None:
    """Run all models including EfficientNet-B1."""
    print("\n" + "=" * 80)
    print("🚀 RUNNING ALL MODELS (WITH EFFICIENTNET-B1)")
    print("=" * 80)

    run_advanced_mode_resnet18()
    run_advanced_mode_resnet50()
    run_advanced_mode_efficientnet_b0()
    run_advanced_mode_efficientnet_b1()

    ModelResults.print_summary()

    print("\n" + "=" * 80)
    print("✅ ALL MODELS (WITH B1) COMPLETED!")
    print("=" * 80)

def run_all_models_with_vgg16() -> None:
    """Run ALL models including VGG16."""
    print("\n" + "=" * 80)
    print("🚀 RUNNING ALL MODELS (WITH VGG16)")
    print("=" * 80)

    run_advanced_mode_resnet18()
    run_advanced_mode_resnet50()
    run_advanced_mode_efficientnet_b0()
    run_advanced_mode_efficientnet_b1()
    run_advanced_mode_vgg16()

    ModelResults.print_summary()

    print("\n" + "=" * 80)
    print("✅ ALL MODELS (WITH VGG16) COMPLETED!")
    print("=" * 80)

def run_all_models_with_cnnlstm() -> None:
    """Run ALL models including CNNLSTM."""
    print("\n" + "=" * 80)
    print("🚀 RUNNING ALL MODELS (WITH CNNLSTM)")
    print("=" * 80)

    run_advanced_mode_resnet18()
    run_advanced_mode_resnet50()
    run_advanced_mode_efficientnet_b0()
    run_advanced_mode_efficientnet_b1()
    run_advanced_mode_vgg16()
    run_advanced_mode_cnnlstm()

    ModelResults.print_summary()

    print("\n" + "=" * 80)
    print("✅ ALL MODELS (WITH CNNLSTM) COMPLETED!")
    print("=" * 80)

def run_report_only_alexnet() -> None:
    """Generate report for AlexNet (Without Fine-Tuning)."""
    if not WFT_ALEXNET_AVAILABLE:
        print("❌ AlexNet Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="AlexNet",
        create_func=create_alexnet_wrapper,
        checkpoints_dir=ALEXNET_CHECKPOINTS_DIR,
        reports_dir=ALEXNET_REPORTS_DIR,
        model_filename="alexnet_best.pth",
        report_filename="AlexNet_Report_From_Model.pdf",
        target_size=224,
        width_mult=0.5,
        fc6_features=1024,
        fc7_features=512,
        dropout_fc=0.4,
        use_batch_norm=True,
    )

def run_cross_validation_alexnet() -> None:
    if not WFT_ALEXNET_AVAILABLE:
        print("❌ AlexNet module not available!")
        return
    run_cross_validation_generic(
        model_name="AlexNet",
        create_func=create_alexnet_wrapper,
        checkpoints_dir=ALEXNET_CHECKPOINTS_DIR,
        model_filename="alexnet_best.pth",
        target_size=224,
        width_mult=0.5,
        fc6_features=1024,
        fc7_features=512,
        dropout_fc=0.4,
        use_batch_norm=True,
    )

def run_all_models_with_alexnet() -> None:
    print("\n" + "=" * 80)
    print("🚀 RUNNING ALL MODELS (WITH ALEXNET)")
    print("=" * 80)
    run_advanced_mode_resnet18()
    run_advanced_mode_resnet50()
    run_advanced_mode_efficientnet_b0()
    run_advanced_mode_efficientnet_b1()
    run_advanced_mode_vgg16()
    run_advanced_mode_cnnlstm()
    run_advanced_mode_convnext()
    run_advanced_mode_mobilenetv2()
    run_advanced_mode_alexnet()
    ModelResults.print_summary()
    print("\n" + "=" * 80)
    print("✅ ALL MODELS (WITH ALEXNET) COMPLETED!")
    print("=" * 80)


def run_report_all_models_with_alexnet() -> None:
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS (WITH ALEXNET)")
    print("=" * 80)
    run_report_only_resnet18()
    run_report_only_resnet50()
    run_report_only_efficientnet_b0()
    run_report_only_efficientnet_b1()
    run_report_only_vgg16()
    run_report_only_cnnlstm()
    run_report_only_convnext()
    run_report_only_mobilenetv2()
    run_report_only_alexnet()
    print("\n" + "=" * 80)
    print("✅ ALL REPORTS (WITH ALEXNET) GENERATED!")
    print("=" * 80)


def run_report_only_densenet() -> None:
    if not WFT_DENSENET_AVAILABLE:
        print("❌ DenseNet Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="DenseNet",
        create_func=create_densenet_wrapper,
        checkpoints_dir=DENSENET_CHECKPOINTS_DIR,
        reports_dir=DENSENET_REPORTS_DIR,
        model_filename="DenseNet_best.pth",
        report_filename="DenseNet_Report_From_Model.pdf",
        target_size=224,
        variant="small",
        dropout=0.2,
    )

def run_cross_validation_densenet() -> None:
    if not WFT_DENSENET_AVAILABLE:
        print("❌ DenseNet module not available!")
        return
    run_cross_validation_generic(
        model_name="DenseNet",
        create_func=create_densenet_wrapper,
        checkpoints_dir=DENSENET_CHECKPOINTS_DIR,
        model_filename="DenseNet_best.pth",
        target_size=224,
        variant="small",
        dropout=0.2,
    )

def run_all_models_with_densenet() -> None:
    print("\n" + "=" * 80)
    print("🚀 RUNNING ALL MODELS (WITH DENSENET)")
    print("=" * 80)
    run_advanced_mode_resnet18()
    run_advanced_mode_resnet50()
    run_advanced_mode_efficientnet_b0()
    run_advanced_mode_efficientnet_b1()
    run_advanced_mode_vgg16()
    run_advanced_mode_cnnlstm()
    run_advanced_mode_convnext()
    run_advanced_mode_mobilenetv2()
    run_advanced_mode_alexnet()
    run_advanced_mode_vit()
    run_advanced_mode_densenet()
    ModelResults.print_summary()
    print("\n" + "=" * 80)
    print("✅ ALL MODELS (WITH DENSENET) COMPLETED!")
    print("=" * 80)


def run_report_all_models_with_densenet() -> None:
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS (WITH DENSENET)")
    print("=" * 80)
    run_report_only_resnet18()
    run_report_only_resnet50()
    run_report_only_efficientnet_b0()
    run_report_only_efficientnet_b1()
    run_report_only_vgg16()
    run_report_only_cnnlstm()
    run_report_only_convnext()
    run_report_only_mobilenetv2()
    run_report_only_alexnet()
    run_report_only_vit()
    run_report_only_densenet()
    print("\n" + "=" * 80)
    print("✅ ALL REPORTS (WITH DENSENET) GENERATED!")
    print("=" * 80)

# ================================================================
# ✅ REPORT GENERATION FUNCTIONS
# ================================================================


def run_report_generic(
    model_name: str,
    create_func: Callable,
    checkpoints_dir: str,
    reports_dir: str,
    model_filename: str,
    report_filename: str,
    **kwargs,
) -> None:
    """Generic function to generate report from saved model."""
    print("\n" + "=" * 80)
    print(f"📄 GENERATE REPORT FOR {model_name}")
    print("=" * 80)

    ensure_directories_main()

    model_path = os.path.join(checkpoints_dir, model_filename)

    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print(f"📌 Please run {model_name} Advanced Mode first to train the model.")
        return

    try:
        print("📥 Loading saved model...")
        model = create_func(num_classes=NUM_CLASSES, **kwargs)
        model.load_checkpoint(model_path)

        print(f"✅ Model loaded successfully!")
        print(f"   - Best Accuracy: {model.best_accuracy:.4f}")
        print(f"   - Is Trained: {model.is_trained}")

        # Load test data
        target_size = kwargs.get("target_size", 224)
        _, _, test_loader = load_data_from_cropped(target_size=target_size)

        print("\n📄 Generating report from saved model...")
        results = model.evaluate_comprehensive(
            test_loader=test_loader,
            generate_report=True,
            cv_folds=0,
            report_name=report_filename,
        )

        print("\n" + "=" * 80)
        print("✅ REPORT GENERATED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📄 Report saved to: {reports_dir}/{report_filename}")

        print("\n📊 Performance Metrics:")
        metrics = results["metrics"]
        print(f"   - Accuracy: {metrics['accuracy']:.4f}")
        print(f"   - Precision: {metrics['precision']:.4f}")
        print(f"   - Recall: {metrics['recall']:.4f}")
        print(f"   - F1-Score: {metrics['f1_score']:.4f}")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback

        traceback.print_exc()


# ================================================================
# ✅ REPORT GENERATION (Without Fine-Tuning)
# ================================================================


def run_report_only_resnet18() -> None:
    """Generate report for ResNet18 (Without Fine-Tuning)."""
    if not WFT_RESNET18_AVAILABLE:
        print("❌ ResNet18 Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="RESNET18",
        create_func=create_resnet18_wrapper,
        checkpoints_dir=RESNET18_CHECKPOINTS_DIR,
        reports_dir=RESNET18_REPORTS_DIR,
        model_filename="resnet18_best.pth",
        report_filename="ResNet18_Report_From_Model.pdf",
        target_size=224,
    )


def run_report_only_resnet50() -> None:
    """Generate report for ResNet50 (Without Fine-Tuning)."""
    if not WFT_RESNET50_AVAILABLE:
        print("❌ ResNet50 Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="RESNET50",
        create_func=create_resnet50_wrapper,
        checkpoints_dir=RESNET50_CHECKPOINTS_DIR,
        reports_dir=RESNET50_REPORTS_DIR,
        model_filename="resnet50_best.pth",
        report_filename="ResNet50_Report_From_Model.pdf",
        target_size=224,
    )


def run_report_only_efficientnet_b0() -> None:
    """Generate report for EfficientNet-B0 (Without Fine-Tuning)."""
    if not WFT_EFFICIENTNET_B0_AVAILABLE:
        print("❌ EfficientNet-B0 Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="EFFICIENTNET-B0",
        create_func=create_efficientnet_b0_wrapper,
        checkpoints_dir=EFFICIENTNET_B0_CHECKPOINTS_DIR,
        reports_dir=EFFICIENTNET_B0_REPORTS_DIR,
        model_filename="efficientnet_b0_best.pth",
        report_filename="EfficientNet-B0_Report_From_Model.pdf",
        target_size=224,
    )


def run_report_only_efficientnet_b1() -> None:
    """Generate report for EfficientNet-B1 (Without Fine-Tuning)."""
    if not WFT_EFFICIENTNET_B1_AVAILABLE:
        print("❌ EfficientNet-B1 Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="EFFICIENTNET-B1",
        create_func=create_efficientnet_b1_wrapper,
        checkpoints_dir=EFFICIENTNET_B1_CHECKPOINTS_DIR,
        reports_dir=EFFICIENTNET_B1_REPORTS_DIR,
        model_filename="efficientnet_b1_best.pth",
        report_filename="EfficientNet-B1_Report_From_Model.pdf",
        target_size=240,
    )


def run_report_both_models() -> None:
    """Generate reports for ResNet18 and ResNet50 (Without Fine-Tuning)."""
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR BOTH MODELS")
    print("=" * 80)
    run_report_only_resnet18()
    run_report_only_resnet50()
    print("\n" + "=" * 80)
    print("✅ BOTH REPORTS GENERATED!")
    print("=" * 80)


def run_report_all_models() -> None:
    """Generate reports for all models (Without Fine-Tuning)."""
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS")
    print("=" * 80)
    run_report_only_resnet18()
    run_report_only_resnet50()
    run_report_only_efficientnet_b0()
    print("\n" + "=" * 80)
    print("✅ ALL REPORTS GENERATED!")
    print("=" * 80)


def run_report_all_models_with_b1() -> None:
    """Generate reports for all models including B1 (Without Fine-Tuning)."""
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS (WITH B1)")
    print("=" * 80)
    run_report_only_resnet18()
    run_report_only_resnet50()
    run_report_only_efficientnet_b0()
    run_report_only_efficientnet_b1()
    print("\n" + "=" * 80)
    print("✅ ALL REPORTS (WITH B1) GENERATED!")
    print("=" * 80)

def run_report_only_vgg16() -> None:
    """Generate report for VGG16 (Without Fine-Tuning)."""
    if not WFT_VGG16_AVAILABLE:
        print("❌ VGG16 Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="VGG16",
        create_func=create_vgg16_wrapper,
        checkpoints_dir=VGG16_CHECKPOINTS_DIR,
        reports_dir=VGG16_REPORTS_DIR,
        model_filename="vgg16_best.pth",
        report_filename="VGG16_Report_From_Model.pdf",
        target_size=224,
    )

def run_report_only_cnnlstm() -> None:
    """Generate report for CNNLSTM (Without Fine-Tuning)."""
    if not WFT_CNNLSTM_AVAILABLE:
        print("❌ CNNLSTM Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="CNNLSTM",
        create_func=create_cnnlstm_wrapper,
        checkpoints_dir=CNNLSTM_CHECKPOINTS_DIR,
        reports_dir=CNNLSTM_REPORTS_DIR,
        model_filename="cnnlstm_best.pth",
        report_filename="CNNLSTM_Report_From_Model.pdf",
        target_size=224,
    )

def run_report_only_convnext() -> None:
    if not WFT_CONVNEXT_AVAILABLE:
        print("❌ ConvNext Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="ConvNext",
        create_func=create_convnext_wrapper,
        checkpoints_dir=CONVNEXT_CHECKPOINTS_DIR,
        reports_dir=CONVNEXT_REPORTS_DIR,
        model_filename="ConvNext_best.pth",
        report_filename="ConvNext_Report_From_Model.pdf",
        target_size=224,
    )

def run_cross_validation_convnext() -> None:
    if not WFT_CONVNEXT_AVAILABLE:
        print("❌ ConvNext module not available!")
        return
    run_cross_validation_generic(
        model_name="ConvNext",
        create_func=create_convnext_wrapper,
        checkpoints_dir=CONVNEXT_CHECKPOINTS_DIR,
        model_filename="ConvNext_best.pth",
        target_size=224,
    )

def run_report_all_models_with_cnnlstm() -> None:
    """Generate reports for all models including CNNLSTM."""
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS (WITH CNNLSTM)")
    print("=" * 80)
    run_report_only_resnet18()
    run_report_only_resnet50()
    run_report_only_efficientnet_b0()
    run_report_only_efficientnet_b1()
    run_report_only_vgg16()
    run_report_only_cnnlstm()
    print("\n" + "=" * 80)
    print("✅ ALL REPORTS (WITH CNNLSTM) GENERATED!")
    print("=" * 80)


def run_cross_validation_cnnlstm() -> None:
    if not WFT_CNNLSTM_AVAILABLE:
        print("❌ CNNLSTM module not available!")
        return
    run_cross_validation_generic(
        model_name="CNNLSTM",
        create_func=create_cnnlstm_wrapper,
        checkpoints_dir=CNNLSTM_CHECKPOINTS_DIR,
        model_filename="cnnlstm_best.pth",
        target_size=224,
    )


def run_report_all_models_with_vgg16() -> None:
    """Generate reports for all models including VGG16."""
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS (WITH VGG16)")
    print("=" * 80)
    run_report_only_resnet18()
    run_report_only_resnet50()
    run_report_only_efficientnet_b0()
    run_report_only_efficientnet_b1()
    run_report_only_vgg16()
    print("\n" + "=" * 80)
    print("✅ ALL REPORTS (WITH VGG16) GENERATED!")
    print("=" * 80)

def run_report_only_vit() -> None:
    """Generate report for ViT (Without Fine-Tuning)."""
    if not WFT_VIT_AVAILABLE:
        print("❌ ViT Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="visio_transformer",
        create_func=create_vit_wrapper,
        checkpoints_dir=VIT_CHECKPOINTS_DIR,
        reports_dir=VIT_REPORTS_DIR,
        model_filename="visio_transformer_best.pth",
        report_filename="ViT_Report_From_Model.pdf",
        target_size=224,
        variant="tiny",
        patch_size=16,
        dropout=0.2,
        drop_path_rate=0.1,
        head_dropout=0.2,
    )


def run_traditional_report_generic(
    model_name: str,
    model_type: str,
    checkpoints_dir: str,
    report_filename: str,
    **kwargs,
) -> None:
    """
    Generic function to generate report from Traditional Mode saved model.
    """
    print("\n" + "=" * 80)
    print(f"📄 GENERATE REPORT FOR {model_name} (Traditional Mode - With Fine-Tuning)")
    print("=" * 80)

    ensure_directories_main()

    if "resnet" in model_type:
        if model_type == "resnet18":
            from src.AI.ResNet.resnet18 import ResNet18Trainer

            model_path = os.path.join(RESNET_AI_DIR, "resnet18", "resnet18_best.pth")
            reports_dir = os.path.join(RESNET_AI_DIR, "reports")
            target_size = 224
        else:
            from src.AI.ResNet.resnet50 import ResNet50Trainer

            model_path = os.path.join(RESNET_AI_DIR, "resnet50", "resnet50_best.pth")
            reports_dir = os.path.join(RESNET_AI_DIR, "reports")
            target_size = 256

        if not os.path.exists(model_path):
            print(f"❌ Model not found at: {model_path}")
            print(f"📌 Please run Traditional Mode (Option 1) first to train the model.")
            return

        try:
            print("📥 Loading saved model...")
            trainer = ResNet18Trainer() if model_type == "resnet18" else ResNet50Trainer()

            checkpoint = torch.load(model_path, map_location="cpu")
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            trainer.model.eval()

            # ✅ استخراج بهترین دقت Validation از checkpoint
            best_validation_accuracy = checkpoint.get("best_accuracy", 0.0)
            
            # ✅ اگر در metrics ذخیره شده
            if best_validation_accuracy == 0.0 and "metrics" in checkpoint:
                best_validation_accuracy = checkpoint["metrics"].get("best_accuracy", 0.0)
            
            # ✅ اگر در extra_info ذخیره شده
            if best_validation_accuracy == 0.0 and "extra_info" in checkpoint:
                best_validation_accuracy = checkpoint["extra_info"].get("best_accuracy", 0.0)

            print(f"✅ Model loaded successfully!")
            print(f"   Best Validation Accuracy: {best_validation_accuracy:.4f}")

            _, _, test_loader = load_data_from_cropped(target_size=target_size)

            print("\n📄 Generating report from saved model...")

            # ✅ ارزیابی دستی - بدون استفاده از trainer.evaluate()
            from src.AI.ResNet.utils import compute_metrics

            trainer.model.eval()

            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for batch in test_loader:
                    if len(batch) == 2:
                        x, y = batch
                    else:
                        x, y, _ = batch

                    x = x.to("cpu")
                    y = y.to("cpu")

                    out = trainer.model(x)
                    prob = torch.softmax(out, dim=1)
                    pred = prob.argmax(dim=1)

                    all_labels.extend(y.cpu().numpy())
                    all_preds.extend(pred.cpu().numpy())
                    all_probs.extend(prob.cpu().numpy())

            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)

            metrics, cm, auc_score = compute_metrics(all_labels, all_preds, all_probs)

            from src.AI.ResNet.resnetReports import generate_full_pdf

            sample_img = None
            for batch in test_loader:
                if len(batch) == 2:
                    sample_img = batch[0][0]
                else:
                    sample_img = batch[0][0]
                break

            if sample_img is not None:
                output_path = os.path.join(reports_dir, report_filename)
                os.makedirs(reports_dir, exist_ok=True)

                # ✅ ارسال best_validation_accuracy و test_accuracy به generate_full_pdf
                generate_full_pdf(
                    metrics=metrics,
                    cm=cm,
                    y_true=all_labels,
                    y_prob=all_probs,
                    model=trainer.model,
                    sample_image_tensor=sample_img,
                    output_path=output_path,
                    model_name=model_name,
                    history=None,
                    config=None,
                    best_accuracy=best_validation_accuracy,      # ✅ Validation Accuracy
                    best_accuracy_label="Validation",            # ✅ برچسب Validation
                    test_accuracy=metrics.get("accuracy", 0),    # ✅ NEW: Test Accuracy
                )

                print("\n" + "=" * 80)
                print("✅ REPORT GENERATED SUCCESSFULLY!")
                print("=" * 80)
                print(f"\n📄 Report saved to: {output_path}")

                print("\n📊 Performance Metrics:")
                print(f"   - Best Validation Accuracy: {best_validation_accuracy:.4f} ({best_validation_accuracy*100:.2f}%)")
                print(f"   - Test Accuracy:             {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
                if metrics:
                    print(f"   - Precision: {metrics.get('weighted avg', {}).get('precision', 0):.4f}")
                    print(f"   - Recall: {metrics.get('weighted avg', {}).get('recall', 0):.4f}")
                    print(f"   - F1-Score: {metrics.get('weighted avg', {}).get('f1-score', 0):.4f}")
                    print(f"   - AUC: {auc_score:.4f}")

                print("\n" + "=" * 80)

        except Exception as e:
            print(f"❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()

    elif "efficientnet" in model_type:
        # ✅ همین تغییرات برای EfficientNet
        if model_type == "efficientnet_b0":
            from src.AI.EfficientNet.efficientNetB0 import EfficientNetB0Trainer

            model_path = os.path.join(EFFICIENTNET_AI_DIR, "../Architects/EfficientNet/B0", "efficientnet_b0_best.pth")
            reports_dir = os.path.join(EFFICIENTNET_AI_DIR, "reports")
            target_size = 224
        else:
            from src.AI.EfficientNet.efficientNetB1 import EfficientNetB1Trainer

            model_path = os.path.join(EFFICIENTNET_AI_DIR, "../Architects/EfficientNet/B1", "efficientnet_b1_best.pth")
            reports_dir = os.path.join(EFFICIENTNET_AI_DIR, "reports")
            target_size = 240

        if not os.path.exists(model_path):
            print(f"❌ Model not found at: {model_path}")
            print(f"📌 Please run Traditional Mode (Option 1) first to train the model.")
            return

        try:
            print("📥 Loading saved model...")
            trainer = EfficientNetB0Trainer() if model_type == "efficientnet_b0" else EfficientNetB1Trainer()

            checkpoint = torch.load(model_path, map_location="cpu")
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            trainer.model.eval()

            # ✅ استخراج بهترین دقت Validation
            best_validation_accuracy = checkpoint.get("best_accuracy", 0.0)
            if best_validation_accuracy == 0.0 and "metrics" in checkpoint:
                best_validation_accuracy = checkpoint["metrics"].get("best_accuracy", 0.0)
            if best_validation_accuracy == 0.0 and "extra_info" in checkpoint:
                best_validation_accuracy = checkpoint["extra_info"].get("best_accuracy", 0.0)

            print(f"✅ Model loaded successfully!")
            print(f"   Best Validation Accuracy: {best_validation_accuracy:.4f}")

            _, _, test_loader = load_data_from_cropped(target_size=target_size)

            print("\n📄 Generating report from saved model...")

            from src.AI.EfficientNet.utils import compute_metrics

            trainer.model.eval()

            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for batch in test_loader:
                    if len(batch) == 2:
                        x, y = batch
                    else:
                        x, y, _ = batch

                    x = x.to("cpu")
                    y = y.to("cpu")

                    out = trainer.model(x)
                    prob = torch.softmax(out, dim=1)
                    pred = prob.argmax(dim=1)

                    all_labels.extend(y.cpu().numpy())
                    all_preds.extend(pred.cpu().numpy())
                    all_probs.extend(prob.cpu().numpy())

            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)

            metrics, cm, auc_score = compute_metrics(all_labels, all_preds, all_probs)

            from src.AI.EfficientNet.efficientNetReports import generate_full_pdf

            sample_img = None
            for batch in test_loader:
                if len(batch) == 2:
                    sample_img = batch[0][0]
                else:
                    sample_img = batch[0][0]
                break

            if sample_img is not None:
                output_path = os.path.join(reports_dir, report_filename)
                os.makedirs(reports_dir, exist_ok=True)

                # ✅ ارسال best_validation_accuracy و test_accuracy به generate_full_pdf
                generate_full_pdf(
                    metrics=metrics,
                    cm=cm,
                    y_true=all_labels,
                    y_prob=all_probs,
                    model=trainer.model,
                    sample_image_tensor=sample_img,
                    output_path=output_path,
                    model_name=model_name,
                    history=None,
                    config=None,
                    best_accuracy=best_validation_accuracy,      # ✅ Validation Accuracy
                    best_accuracy_label="Validation",            # ✅ برچسب Validation
                    test_accuracy=metrics.get("accuracy", 0),    # ✅ NEW: Test Accuracy
                )

                print("\n" + "=" * 80)
                print("✅ REPORT GENERATED SUCCESSFULLY!")
                print("=" * 80)
                print(f"\n📄 Report saved to: {output_path}")

                print("\n📊 Performance Metrics:")
                print(f"   - Best Validation Accuracy: {best_validation_accuracy:.4f} ({best_validation_accuracy*100:.2f}%)")
                print(f"   - Test Accuracy:             {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
                if metrics:
                    print(f"   - Precision: {metrics.get('weighted avg', {}).get('precision', 0):.4f}")
                    print(f"   - Recall: {metrics.get('weighted avg', {}).get('recall', 0):.4f}")
                    print(f"   - F1-Score: {metrics.get('weighted avg', {}).get('f1-score', 0):.4f}")
                    print(f"   - AUC: {auc_score:.4f}")

                print("\n" + "=" * 80)

        except Exception as e:
            print(f"❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()


# ================================================================
# ✅ REPORT GENERATION (Traditional Mode - Individual)
# ================================================================


def run_traditional_report_resnet18() -> None:
    """Generate report for ResNet18 (Traditional Mode)."""
    run_traditional_report_generic(
        model_name="RESNET18",
        model_type="resnet18",
        checkpoints_dir=RESNET_AI_DIR,
        report_filename="ResNet18_Report_Traditional.pdf",
    )


def run_traditional_report_resnet50() -> None:
    """Generate report for ResNet50 (Traditional Mode)."""
    run_traditional_report_generic(
        model_name="RESNET50",
        model_type="resnet50",
        checkpoints_dir=RESNET_AI_DIR,
        report_filename="ResNet50_Report_Traditional.pdf",
    )


def run_traditional_report_efficientnet_b0() -> None:
    """Generate report for EfficientNet-B0 (Traditional Mode)."""
    run_traditional_report_generic(
        model_name="EFFICIENTNET-B0",
        model_type="efficientnet_b0",
        checkpoints_dir=EFFICIENTNET_AI_DIR,
        report_filename="EfficientNet-B0_Report_Traditional.pdf",
    )


def run_traditional_report_efficientnet_b1() -> None:
    """Generate report for EfficientNet-B1 (Traditional Mode)."""
    run_traditional_report_generic(
        model_name="EFFICIENTNET-B1",
        model_type="efficientnet_b1",
        checkpoints_dir=EFFICIENTNET_AI_DIR,
        report_filename="EfficientNet-B1_Report_Traditional.pdf",
    )

def run_report_only_mobilenetv2() -> None:
    if not WFT_MOBILENETV2_AVAILABLE:
        print("❌ MobileNetV2 Without Fine-Tuning module not available!")
        return
    run_report_generic(
        model_name="MobileNetV2",
        create_func=create_mobilenetv2_wrapper,
        checkpoints_dir=MOBILENETV2_CHECKPOINTS_DIR,
        reports_dir=MOBILENETV2_REPORTS_DIR,
        model_filename="mobilenetV2_best.pth",
        report_filename="MobileNetV2_Report_From_Model.pdf",
        target_size=224,
    )

def run_cross_validation_mobilenetv2() -> None:
    if not WFT_MOBILENETV2_AVAILABLE:
        print("❌ MobileNetV2 module not available!")
        return
    run_cross_validation_generic(
        model_name="MobileNetV2",
        create_func=create_mobilenetv2_wrapper,
        checkpoints_dir=MOBILENETV2_CHECKPOINTS_DIR,
        model_filename="mobilenetV2_best.pth",
        target_size=224,
    )

def run_all_models_with_mobilenetv2() -> None:
    print("\n" + "=" * 80)
    print("🚀 RUNNING ALL MODELS (WITH MOBILENETV2)")
    print("=" * 80)
    run_advanced_mode_resnet18()
    run_advanced_mode_resnet50()
    run_advanced_mode_efficientnet_b0()
    run_advanced_mode_efficientnet_b1()
    run_advanced_mode_vgg16()
    run_advanced_mode_cnnlstm()
    run_advanced_mode_convnext()
    run_advanced_mode_mobilenetv2()
    ModelResults.print_summary()
    print("\n" + "=" * 80)
    print("✅ ALL MODELS (WITH MOBILENETV2) COMPLETED!")
    print("=" * 80)

def run_report_all_models_with_mobilenetv2() -> None:
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS (WITH MOBILENETV2)")
    print("=" * 80)
    run_report_only_resnet18()
    run_report_only_resnet50()
    run_report_only_efficientnet_b0()
    run_report_only_efficientnet_b1()
    run_report_only_vgg16()
    run_report_only_cnnlstm()
    run_report_only_convnext()
    run_report_only_mobilenetv2()
    print("\n" + "=" * 80)
    print("✅ ALL REPORTS (WITH MOBILENETV2) GENERATED!")
    print("=" * 80)

def run_cross_validation_vit() -> None:
    if not WFT_VIT_AVAILABLE:
        print("❌ ViT module not available!")
        return
    run_cross_validation_generic(
        model_name="visio_transformer",
        create_func=create_vit_wrapper,
        checkpoints_dir=VIT_CHECKPOINTS_DIR,
        model_filename="visio_transformer_best.pth",
        target_size=224,
        variant="tiny",
        patch_size=16,
        dropout=0.2,
        drop_path_rate=0.1,
        head_dropout=0.2,
    )

# ================================================================
# ✅ REPORT GENERATION (Traditional Mode - Multiple Architects)
# ================================================================


def run_traditional_report_both() -> None:
    """Generate reports for both ResNet18 and ResNet50 (Traditional Mode)."""
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR BOTH MODELS (Traditional Mode)")
    print("📌 ResNet18 & ResNet50")
    print("=" * 80)

    run_traditional_report_resnet18()
    run_traditional_report_resnet50()

    print("\n" + "=" * 80)
    print("✅ BOTH REPORTS GENERATED SUCCESSFULLY!")
    print("=" * 80)


def run_traditional_report_all() -> None:
    """Generate reports for all models (Traditional Mode)."""
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS (Traditional Mode)")
    print("📌 ResNet18 & ResNet50 & EfficientNet-B0")
    print("=" * 80)

    run_traditional_report_resnet18()
    run_traditional_report_resnet50()
    run_traditional_report_efficientnet_b0()

    print("\n" + "=" * 80)
    print("✅ ALL REPORTS GENERATED SUCCESSFULLY!")
    print("=" * 80)


def run_traditional_report_all_with_b1() -> None:
    """Generate reports for all models including B1 (Traditional Mode)."""
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS (Traditional Mode)")
    print("📌 ResNet18 & ResNet50 & EfficientNet-B0 & EfficientNet-B1")
    print("=" * 80)

    run_traditional_report_resnet18()
    run_traditional_report_resnet50()
    run_traditional_report_efficientnet_b0()
    run_traditional_report_efficientnet_b1()

    print("\n" + "=" * 80)
    print("✅ ALL REPORTS (WITH B1) GENERATED SUCCESSFULLY!")
    print("=" * 80)

def run_all_models_with_vit() -> None:
    print("\n" + "=" * 80)
    print("🚀 RUNNING ALL MODELS (WITH ViT)")
    print("=" * 80)
    run_advanced_mode_resnet18()
    run_advanced_mode_resnet50()
    run_advanced_mode_efficientnet_b0()
    run_advanced_mode_efficientnet_b1()
    run_advanced_mode_vgg16()
    run_advanced_mode_cnnlstm()
    run_advanced_mode_convnext()
    run_advanced_mode_mobilenetv2()
    run_advanced_mode_alexnet()
    run_advanced_mode_vit()
    ModelResults.print_summary()
    print("\n" + "=" * 80)
    print("✅ ALL MODELS (WITH ViT) COMPLETED!")
    print("=" * 80)


def run_report_all_models_with_vit() -> None:
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS (WITH ViT)")
    print("=" * 80)
    run_report_only_resnet18()
    run_report_only_resnet50()
    run_report_only_efficientnet_b0()
    run_report_only_efficientnet_b1()
    run_report_only_vgg16()
    run_report_only_cnnlstm()
    run_report_only_convnext()
    run_report_only_mobilenetv2()
    run_report_only_alexnet()
    run_report_only_vit()
    print("\n" + "=" * 80)
    print("✅ ALL REPORTS (WITH ViT) GENERATED!")
    print("=" * 80)


# ================================================================
# ✅ CROSS-VALIDATION
# ================================================================


def run_cross_validation_generic(
    model_name: str,
    create_func: Callable,
    checkpoints_dir: str,
    model_filename: str,
    **kwargs,
) -> None:
    """Generic function for cross-validation using SAVED model."""
    print("\n" + "=" * 80)
    print(f"🔬 CROSS-VALIDATION FOR {model_name} (Using Saved Model)")
    print("=" * 80)

    ensure_directories_main()

    model_path = os.path.join(checkpoints_dir, model_filename)

    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print(f"📌 Please run {model_name} Advanced Mode first to train the model.")
        return

    try:
        data_ready, status = check_data_ready()
        if not data_ready:
            print("\n❌ Data is not ready for cross-validation!")
            return

        print("\n✅ Data validation passed! Proceeding with Cross-Validation...")

        # Load model
        print("\n📥 Loading saved model...")
        model = create_func(num_classes=NUM_CLASSES, **kwargs)
        model.load_checkpoint(model_path)

        print(f"✅ Model loaded successfully!")
        print(f"   - Best Accuracy: {model.best_accuracy:.4f}")
        print(f"   - Is Trained: {model.is_trained}")

        # Load data
        target_size = kwargs.get("target_size", 224)
        train_loader, _, _ = load_data_from_cropped(target_size=target_size)

        print(f"\n📊 Training Data for CV: {len(train_loader.dataset)} samples")

        # ============================================================
        # ✅ RUN CV WITHOUT RETRAINING - استفاده از تابع جدید
        # ============================================================
        print("\n" + "=" * 80)
        print(f"🔬 Running {5}-Fold Cross-Validation on Training Data...")
        print("📌 Using SAVED model (NO RETRAINING!)")
        print("=" * 80)

        cv_results = model.perform_cross_validation_from_saved(
            dataset=train_loader.dataset, n_folds=5, batch_size=32
        )

        print("\n" + "=" * 80)
        print(f"📊 CROSS-VALIDATION RESULTS (5-Fold) - {model_name}")
        print("=" * 80)
        print(
            f"\n   Accuracy:  {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}"
        )
        print(
            f"   Precision: {cv_results['mean_precision']:.4f} ± {cv_results['std_precision']:.4f}"
        )
        print(f"   Recall:    {cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}")
        print(f"   F1-Score:  {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")

        print("\n📋 Detailed Results:")
        print("-" * 70)
        print(f"{'Fold':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        for i in range(5):
            print(
                f"{i+1:<8} {cv_results['accuracy'][i]:.4f}      {cv_results['precision'][i]:.4f}      {cv_results['recall'][i]:.4f}      {cv_results['f1_score'][i]:.4f}"
            )
        print("-" * 70)

        print("\n" + "=" * 80)
        print("✅ CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
        print("📌 No retraining was performed - used saved model only!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR in cross-validation: {e}")
        import traceback

        traceback.print_exc()


def run_cross_validation_resnet18() -> None:
    if not WFT_RESNET18_AVAILABLE:
        print("❌ ResNet18 module not available!")
        return
    run_cross_validation_generic(
        model_name="RESNET18",
        create_func=create_resnet18_wrapper,
        checkpoints_dir=RESNET18_CHECKPOINTS_DIR,
        model_filename="resnet18_best.pth",
        target_size=224,
    )


def run_cross_validation_resnet50() -> None:
    if not WFT_RESNET50_AVAILABLE:
        print("❌ ResNet50 module not available!")
        return
    run_cross_validation_generic(
        model_name="RESNET50",
        create_func=create_resnet50_wrapper,
        checkpoints_dir=RESNET50_CHECKPOINTS_DIR,
        model_filename="resnet50_best.pth",
        target_size=224,
    )


def run_cross_validation_efficientnet_b0() -> None:
    if not WFT_EFFICIENTNET_B0_AVAILABLE:
        print("❌ EfficientNet-B0 module not available!")
        return
    run_cross_validation_generic(
        model_name="EFFICIENTNET-B0",
        create_func=create_efficientnet_b0_wrapper,
        checkpoints_dir=EFFICIENTNET_B0_CHECKPOINTS_DIR,
        model_filename="efficientnet_b0_best.pth",
        target_size=224,
    )


def run_cross_validation_efficientnet_b1() -> None:
    if not WFT_EFFICIENTNET_B1_AVAILABLE:
        print("❌ EfficientNet-B1 module not available!")
        return
    run_cross_validation_generic(
        model_name="EFFICIENTNET-B1",
        create_func=create_efficientnet_b1_wrapper,
        checkpoints_dir=EFFICIENTNET_B1_CHECKPOINTS_DIR,
        model_filename="efficientnet_b1_best.pth",
        target_size=240,
    )


def run_cross_validation_both() -> None:
    print("\n" + "=" * 80)
    print("🔬 CROSS-VALIDATION FOR BOTH MODELS")
    print("=" * 80)
    run_cross_validation_resnet18()
    run_cross_validation_resnet50()
    print("\n" + "=" * 80)
    print("✅ BOTH CROSS-VALIDATIONS COMPLETED!")
    print("=" * 80)


def run_cross_validation_all() -> None:
    print("\n" + "=" * 80)
    print("🔬 CROSS-VALIDATION FOR ALL MODELS")
    print("=" * 80)
    run_cross_validation_resnet18()
    run_cross_validation_resnet50()
    run_cross_validation_efficientnet_b0()
    print("\n" + "=" * 80)
    print("✅ ALL CROSS-VALIDATIONS COMPLETED!")
    print("=" * 80)


def run_cross_validation_all_with_b1() -> None:
    print("\n" + "=" * 80)
    print("🔬 CROSS-VALIDATION FOR ALL MODELS (WITH B1)")
    print("=" * 80)
    run_cross_validation_resnet18()
    run_cross_validation_resnet50()
    run_cross_validation_efficientnet_b0()
    run_cross_validation_efficientnet_b1()
    print("\n" + "=" * 80)
    print("✅ ALL CROSS-VALIDATIONS (WITH B1) COMPLETED!")
    print("=" * 80)


def run_cross_validation_vgg16() -> None:
    if not WFT_VGG16_AVAILABLE:
        print("❌ VGG16 module not available!")
        return
    run_cross_validation_generic(
        model_name="VGG16",
        create_func=create_vgg16_wrapper,
        checkpoints_dir=VGG16_CHECKPOINTS_DIR,
        model_filename="vgg16_best.pth",
        target_size=224,
    )

def run_all_models_with_convnext() -> None:
    """Run ALL models including ConvNext."""
    print("\n" + "=" * 80)
    print("🚀 RUNNING ALL MODELS (WITH CONVNEXT)")
    print("=" * 80)

    run_advanced_mode_resnet18()
    run_advanced_mode_resnet50()
    run_advanced_mode_efficientnet_b0()
    run_advanced_mode_efficientnet_b1()
    run_advanced_mode_vgg16()
    run_advanced_mode_cnnlstm()
    run_advanced_mode_convnext()

    ModelResults.print_summary()

    print("\n" + "=" * 80)
    print("✅ ALL MODELS (WITH CONVNEXT) COMPLETED!")
    print("=" * 80)

def run_report_all_models_with_convnext() -> None:
    """Generate reports for all models including ConvNext."""
    print("\n" + "=" * 80)
    print("📄 GENERATE REPORTS FOR ALL MODELS (WITH CONVNEXT)")
    print("=" * 80)

    run_report_only_resnet18()
    run_report_only_resnet50()
    run_report_only_efficientnet_b0()
    run_report_only_efficientnet_b1()
    run_report_only_vgg16()
    run_report_only_cnnlstm()
    run_report_only_convnext()

    print("\n" + "=" * 80)
    print("✅ ALL REPORTS (WITH CONVNEXT) GENERATED!")
    print("=" * 80)

# ================================================================
# ✅ COMPARISONS
# ================================================================


def run_model_comparison() -> None:
    """Run comprehensive model comparison."""
    print("\n" + "=" * 80)
    print("🔬 RUNNING COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)

    if not COMPARISONS_AVAILABLE:
        print("❌ Comparisons module not available!")
        return

    try:
        run_comprehensive_comparison()
        print("\n✅ Comprehensive comparison completed!")
        print(f"📄 Results saved to: {FINAL_COMPARISONS_DIR}")
    except Exception as e:
        print(f"❌ Error in comparison: {e}")


def run_literature_comparison() -> None:
    """Run literature comparison."""
    print("\n" + "=" * 80)
    print("📊 RUNNING LITERATURE COMPARISON")
    print("=" * 80)

    if not COMPARISONS_AVAILABLE:
        print("❌ Comparisons module not available!")
        return

    try:
        generate_comparison_report()
        print("\n✅ Literature comparison completed!")
        print(f"📄 Results saved to: {LITERATURE_COMPARISON_DIR}")
    except Exception as e:
        print(f"❌ Error in literature comparison: {e}")


# ================================================================
# ✅ SHOW HYPERPARAMETERS
# ================================================================


def show_hyperparameters() -> None:
    """Display all hyperparameters."""
    print("\n" + "=" * 80)
    print("📊 OPTIMAL HYPERPARAMETERS")
    print("=" * 80)

    for model_name, params in OPTIMAL_HYPERPARAMS.items():
        print(f"\n🔹 {model_name.upper()}:")
        for key, value in params.items():
            print(f"   {key}: {value}")

    print("\n" + "=" * 80)


# ================================================================
# ✅ SHOW SYSTEM INFO
# ================================================================


def show_system_info() -> None:
    """Display system information."""
    print("\n" + "=" * 80)
    print("💻 SYSTEM INFORMATION")
    print("=" * 80)

    print(f"\n📌 Python Version: {sys.version}")
    print(f"📌 PyTorch Version: {torch.__version__}")
    print(f"📌 CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"📌 CUDA Version: {torch.version.cuda}")
        print(f"📌 GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"📌 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"\n📌 Available Modules:")
    print(f"   - WFT ResNet18: {'✅' if WFT_RESNET18_AVAILABLE else '❌'}")
    print(f"   - WFT ResNet50: {'✅' if WFT_RESNET50_AVAILABLE else '❌'}")
    print(f"   - WFT EfficientNet-B0: {'✅' if WFT_EFFICIENTNET_B0_AVAILABLE else '❌'}")
    print(f"   - WFT EfficientNet-B1: {'✅' if WFT_EFFICIENTNET_B1_AVAILABLE else '❌'}")
    print(f"   - Pipeline: {'✅' if PIPELINE_AVAILABLE else '❌'}")
    print(f"   - Comparisons: {'✅' if COMPARISONS_AVAILABLE else '❌'}")

    print("\n" + "=" * 80)


# ================================================================
# ✅ TRADITIONAL MODE
# ================================================================


def run_traditional_mode() -> None:
    """Run traditional fine-tuning mode."""
    print("\n" + "=" * 80)
    print("🔧 TRADITIONAL MODE - WITH FINE-TUNING")
    print("=" * 80)
    print("📌 Running the original pipeline with pre-trained models...")
    print("=" * 80)

    if not PIPELINE_AVAILABLE:
        print("❌ Pipeline module not available!")
        return

    try:
        # ✅ Call runDeeply() from resNetEfficientNetModels
        resNetEfficientNetModels.runDeeply()
    except Exception as e:
        print(f"\n❌ Error in Traditional Mode: {e}")
        import traceback

        traceback.print_exc()


def run_advanced_mode_resnet18_strong_aug() -> None:
    """Run ResNet18 with STRONG augmentation."""
    if not WFT_RESNET18_AVAILABLE:
        print("❌ ResNet18 module not available!")
        return

    run_advanced_model_generic(
        model_name="RESNET18",
        create_func=create_resnet18_wrapper,
        checkpoints_dir=RESNET18_CHECKPOINTS_DIR,
        reports_dir=RESNET18_REPORTS_DIR,
        configs_dir=RESNET18_CONFIGS_DIR,
        plots_dir=RESNET18_PLOTS_DIR,
        report_name="ResNet18_Task5_StrongAug_Report.pdf",
        zero_init_residual=True,
        use_batch_norm=True,
        activation="relu",
        dropout_rate=0.15,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=224,
        use_class_weights=True,
        class_weight_method="sqrt",
        labels_csv_path=LABELS_CSV,
        label_smoothing=0.1,
        use_adaptive_smoothing=False,
        use_focal_loss=True,
        focal_gamma=2.0,
        use_weighted_sampler=True,
        sampler_power=0.3,  # ← 0.5 → 0.3 (کمتر aggressive)
        augmentation_level="strong",       # ← ✅ قوی!
        use_ultrasound_transforms=True,
        total_epochs=80,
    )


def run_advanced_mode_resnet18_no_aug() -> None:
    """Run ResNet18 without augmentation (baseline)."""
    if not WFT_RESNET18_AVAILABLE:
        print("❌ ResNet18 module not available!")
        return

    run_advanced_model_generic(
        model_name="RESNET18",
        create_func=create_resnet18_wrapper,
        checkpoints_dir=RESNET18_CHECKPOINTS_DIR,
        reports_dir=RESNET18_REPORTS_DIR,
        configs_dir=RESNET18_CONFIGS_DIR,
        plots_dir=RESNET18_PLOTS_DIR,
        report_name="ResNet18_Task5_NoAug_Report.pdf",
        zero_init_residual=True,
        use_batch_norm=True,
        activation="relu",
        dropout_rate=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_size=224,
        use_class_weights=True,
        class_weight_method="sqrt",
        labels_csv_path=LABELS_CSV,
        label_smoothing=0.1,
        use_adaptive_smoothing=False,
        use_focal_loss=True,
        focal_gamma=1.0,
        use_weighted_sampler=True,
        sampler_power=0.5,
        augmentation_level="none",          # ← ✅ بدون augmentation
        use_ultrasound_transforms=False,
        total_epochs=80,
    )


# ================================================================
# ✅ MAIN MENU
# ================================================================


def show_main_menu() -> None:
    print("\n" + "=" * 80)
    print("🎯 MAIN MENU")
    print("=" * 80)
    print("📌 Please select the mode you want to run:")
    print("")
    print("   [1] 🔧 Traditional Mode - With Fine-Tuning")
    print("")
    print("   " + "-" * 40)
    print("   📌 Advanced Architects (Without Fine-Tuning):")
    print("")
    print("   [2] 🚀 ResNet18")
    print("   [3] 🚀 ResNet50")
    print("   [4] 🚀 EfficientNet-B0")
    print("   [5] 🚀 EfficientNet-B1")
    print("   [6] 🚀 VGG16")
    print("   [7] 🚀 CNNLSTM")
    print("   [8] 🚀 ConvNext")
    print("   [9] 🚀 MobileNetV2")
    print("   [10] 🚀 AlexNet")
    print("   [11] 🚀 ViT")
    print("   [12] 🚀 DenseNet (NEW)")
    print("")
    print("   [13] 🚀 Run Both ResNet Architects (18 & 50)")
    print("   [14] 🚀 Run All Architects (18 & 50 & B0)")
    print("   [15] 🚀 Run All Architects with B1")
    print("   [16] 🚀 Run All Architects with VGG16")
    print("   [17] 🚀 Run All Architects with CNNLSTM")
    print("   [18] 🚀 Run All Architects with ConvNext")
    print("   [19] 🚀 Run All Architects with MobileNetV2")
    print("   [20] 🚀 Run All Architects with AlexNet")
    print("   [21] 🚀 Run All Architects with ViT")
    print("   [22] 🚀 Run All Architects with DenseNet (NEW)")
    print("")
    print("   " + "-" * 40)
    print("   [23] 📊 Check Data Status Only")
    print("")
    print("   " + "-" * 40)
    print("   📌 Generate Reports (Without Fine-Tuning):")
    print("")
    print("   [24] 📄 ResNet18 Report")
    print("   [25] 📄 ResNet50 Report")
    print("   [26] 📄 EfficientNet-B0 Report")
    print("   [27] 📄 EfficientNet-B1 Report")
    print("   [28] 📄 VGG16 Report")
    print("   [29] 📄 CNNLSTM Report")
    print("   [30] 📄 ConvNext Report")
    print("   [31] 📄 MobileNetV2 Report")
    print("   [32] 📄 AlexNet Report")
    print("   [33] 📄 ViT Report")
    print("   [34] 📄 DenseNet Report (NEW)")
    print("")
    print("   [35] 📄 Reports for ResNet Architects (18 & 50)")
    print("   [36] 📄 Reports for All Architects (18 & 50 & B0)")
    print("   [37] 📄 Reports for All Architects with B1")
    print("   [38] 📄 Reports for All Architects with VGG16")
    print("   [39] 📄 Reports for All Architects with CNNLSTM")
    print("   [40] 📄 Reports for All Architects with ConvNext")
    print("   [41] 📄 Reports for All Architects with MobileNetV2")
    print("   [42] 📄 Reports for All Architects with AlexNet")
    print("   [43] 📄 Reports for All Architects with ViT")
    print("   [44] 📄 Reports for All Architects with DenseNet (NEW)")
    print("")
    print("   " + "-" * 40)
    print("   📌 Generate Reports (Traditional Mode):")
    print("")
    print("   [45] 📄 Traditional ResNet18 Report")
    print("   [46] 📄 Traditional ResNet50 Report")
    print("   [47] 📄 Traditional EfficientNet-B0 Report")
    print("   [48] 📄 Traditional EfficientNet-B1 Report")
    print("")
    print("   [49] 📄 Traditional Reports for Both ResNet Architects")
    print("   [50] 📄 Traditional Reports for All Architects (18 & 50 & B0)")
    print("   [51] 📄 Traditional Reports for All Architects with B1")
    print("")
    print("   " + "-" * 40)
    print("   📌 Cross-Validation:")
    print("")
    print("   [52] 🔬 ResNet18 CV")
    print("   [53] 🔬 ResNet50 CV")
    print("   [54] 🔬 EfficientNet-B0 CV")
    print("   [55] 🔬 EfficientNet-B1 CV")
    print("   [56] 🔬 VGG16 CV")
    print("   [57] 🔬 CNNLSTM CV")
    print("   [58] 🔬 ConvNext CV")
    print("   [59] 🔬 MobileNetV2 CV")
    print("   [60] 🔬 AlexNet CV")
    print("   [61] 🔬 ViT CV")
    print("   [62] 🔬 DenseNet CV (NEW)")
    print("")
    print("   [63] 🔬 ResNet Architects CV (18 & 50)")
    print("   [64] 🔬 All Architects CV (18 & 50 & B0)")
    print("   [65] 🔬 All Architects CV with B1")
    print("")
    print("   " + "-" * 40)
    print("   [66] 📂 Show Directory Structure")
    print("   [67] 🔬 Run Model Comparison")
    print("   [68] 📊 Literature Comparison")
    print("   [69] 📈 Show Hyperparameters")
    print("   [70] 💻 System Info")
    print("   [71] 🎨 ResNet18 with STRONG Augmentation")
    print("   [72] 🎨 ResNet18 with NO Augmentation (baseline)")
    print("")
    print("   " + "-" * 40)
    print("   [73] ❌ Exit")
    print("=" * 80)


def wFTAlgorithms() -> None:
    """Main entry point."""
    musicPlayer = None
    try:
        # ============================================================
        # ✅ Music Player (safe init)
        # ============================================================
        MUSIC_PATH = "../../../packages/facilities/assets/musics/music2.mp3"
        try:
            musicPlayer = Music(MUSIC_PATH=MUSIC_PATH)
            musicPlayer.play_music()
        except Exception as e:
            print(f"⚠️ Could not start music: {e}")
            musicPlayer = None

        print_header()
        ensure_directories_main()

        while True:
            show_main_menu()
            choice = input("\n👉 Enter your choice (1-73): ").strip()

            # ========================================================
            # [1] Traditional Mode
            # ========================================================
            if choice == "1":
                run_traditional_mode()
                input("\nPress Enter to continue...")

            # ========================================================
            # [2-12] Advanced Architects (Without Fine-Tuning)
            # ========================================================
            elif choice == "2":
                run_advanced_mode_resnet18()
                input("\nPress Enter to continue...")

            elif choice == "3":
                run_advanced_mode_resnet50()
                input("\nPress Enter to continue...")

            elif choice == "4":
                run_advanced_mode_efficientnet_b0()
                input("\nPress Enter to continue...")

            elif choice == "5":
                run_advanced_mode_efficientnet_b1()
                input("\nPress Enter to continue...")

            elif choice == "6":
                run_advanced_mode_vgg16()
                input("\nPress Enter to continue...")

            elif choice == "7":
                run_advanced_mode_cnnlstm()
                input("\nPress Enter to continue...")

            elif choice == "8":
                run_advanced_mode_convnext()
                input("\nPress Enter to continue...")

            elif choice == "9":
                run_advanced_mode_mobilenetv2()
                input("\nPress Enter to continue...")

            elif choice == "10":
                run_advanced_mode_alexnet()
                input("\nPress Enter to continue...")

            elif choice == "11":
                run_advanced_mode_vit()
                input("\nPress Enter to continue...")

            elif choice == "12":
                run_advanced_mode_densenet()
                input("\nPress Enter to continue...")

            # ========================================================
            # [13-22] Run Multiple Architects
            # ========================================================
            elif choice == "13":
                run_both_models()
                input("\nPress Enter to continue...")

            elif choice == "14":
                run_all_models()
                input("\nPress Enter to continue...")

            elif choice == "15":
                run_all_models_with_b1()
                input("\nPress Enter to continue...")

            elif choice == "16":
                run_all_models_with_vgg16()
                input("\nPress Enter to continue...")

            elif choice == "17":
                run_all_models_with_cnnlstm()
                input("\nPress Enter to continue...")

            elif choice == "18":
                run_all_models_with_convnext()
                input("\nPress Enter to continue...")

            elif choice == "19":
                run_all_models_with_mobilenetv2()
                input("\nPress Enter to continue...")

            elif choice == "20":
                run_all_models_with_alexnet()
                input("\nPress Enter to continue...")

            elif choice == "21":
                run_all_models_with_vit()
                input("\nPress Enter to continue...")

            elif choice == "22":
                run_all_models_with_densenet()
                input("\nPress Enter to continue...")

            # ========================================================
            # [23] Check Data Status
            # ========================================================
            elif choice == "23":
                check_data_ready()
                input("\nPress Enter to continue...")

            # ========================================================
            # [24-34] Reports (Without Fine-Tuning) — Individual
            # ========================================================
            elif choice == "24":
                run_report_only_resnet18()
                input("\nPress Enter to continue...")

            elif choice == "25":
                run_report_only_resnet50()
                input("\nPress Enter to continue...")

            elif choice == "26":
                run_report_only_efficientnet_b0()
                input("\nPress Enter to continue...")

            elif choice == "27":
                run_report_only_efficientnet_b1()
                input("\nPress Enter to continue...")

            elif choice == "28":
                run_report_only_vgg16()
                input("\nPress Enter to continue...")

            elif choice == "29":
                run_report_only_cnnlstm()
                input("\nPress Enter to continue...")

            elif choice == "30":
                run_report_only_convnext()
                input("\nPress Enter to continue...")

            elif choice == "31":
                run_report_only_mobilenetv2()
                input("\nPress Enter to continue...")

            elif choice == "32":
                run_report_only_alexnet()
                input("\nPress Enter to continue...")

            elif choice == "33":
                run_report_only_vit()
                input("\nPress Enter to continue...")

            elif choice == "34":
                run_report_only_densenet()
                input("\nPress Enter to continue...")

            # ========================================================
            # [35-44] Reports (Without Fine-Tuning) — Multiple
            # ========================================================
            elif choice == "35":
                run_report_both_models()
                input("\nPress Enter to continue...")

            elif choice == "36":
                run_report_all_models()
                input("\nPress Enter to continue...")

            elif choice == "37":
                run_report_all_models_with_b1()
                input("\nPress Enter to continue...")

            elif choice == "38":
                run_report_all_models_with_vgg16()
                input("\nPress Enter to continue...")

            elif choice == "39":
                run_report_all_models_with_cnnlstm()
                input("\nPress Enter to continue...")

            elif choice == "40":
                run_report_all_models_with_convnext()
                input("\nPress Enter to continue...")

            elif choice == "41":
                run_report_all_models_with_mobilenetv2()
                input("\nPress Enter to continue...")

            elif choice == "42":
                run_report_all_models_with_alexnet()
                input("\nPress Enter to continue...")

            elif choice == "43":
                run_report_all_models_with_vit()
                input("\nPress Enter to continue...")

            elif choice == "44":
                run_report_all_models_with_densenet()
                input("\nPress Enter to continue...")

            # ========================================================
            # [45-51] Reports (Traditional Mode)
            # ========================================================
            elif choice == "45":
                run_traditional_report_resnet18()
                input("\nPress Enter to continue...")

            elif choice == "46":
                run_traditional_report_resnet50()
                input("\nPress Enter to continue...")

            elif choice == "47":
                run_traditional_report_efficientnet_b0()
                input("\nPress Enter to continue...")

            elif choice == "48":
                run_traditional_report_efficientnet_b1()
                input("\nPress Enter to continue...")

            elif choice == "49":
                run_traditional_report_both()
                input("\nPress Enter to continue...")

            elif choice == "50":
                run_traditional_report_all()
                input("\nPress Enter to continue...")

            elif choice == "51":
                run_traditional_report_all_with_b1()
                input("\nPress Enter to continue...")

            # ========================================================
            # [52-62] Cross-Validation — Individual
            # ========================================================
            elif choice == "52":
                run_cross_validation_resnet18()
                input("\nPress Enter to continue...")

            elif choice == "53":
                run_cross_validation_resnet50()
                input("\nPress Enter to continue...")

            elif choice == "54":
                run_cross_validation_efficientnet_b0()
                input("\nPress Enter to continue...")

            elif choice == "55":
                run_cross_validation_efficientnet_b1()
                input("\nPress Enter to continue...")

            elif choice == "56":
                run_cross_validation_vgg16()
                input("\nPress Enter to continue...")

            elif choice == "57":
                run_cross_validation_cnnlstm()
                input("\nPress Enter to continue...")

            elif choice == "58":
                run_cross_validation_convnext()
                input("\nPress Enter to continue...")

            elif choice == "59":
                run_cross_validation_mobilenetv2()
                input("\nPress Enter to continue...")

            elif choice == "60":
                run_cross_validation_alexnet()
                input("\nPress Enter to continue...")

            elif choice == "61":
                run_cross_validation_vit()
                input("\nPress Enter to continue...")

            elif choice == "62":
                run_cross_validation_densenet()
                input("\nPress Enter to continue...")

            # ========================================================
            # [63-65] Cross-Validation — Multiple
            # ========================================================
            elif choice == "63":
                run_cross_validation_both()
                input("\nPress Enter to continue...")

            elif choice == "64":
                run_cross_validation_all()
                input("\nPress Enter to continue...")

            elif choice == "65":
                run_cross_validation_all_with_b1()
                input("\nPress Enter to continue...")

            # ========================================================
            # [66-70] Utilities
            # ========================================================
            elif choice == "66":
                print_directory_structure()
                input("\nPress Enter to continue...")

            elif choice == "67":
                run_model_comparison()
                input("\nPress Enter to continue...")

            elif choice == "68":
                run_literature_comparison()
                input("\nPress Enter to continue...")

            elif choice == "69":
                show_hyperparameters()
                input("\nPress Enter to continue...")

            elif choice == "70":
                show_system_info()
                input("\nPress Enter to continue...")

            # ========================================================
            # [71-72] ResNet18 Augmentation Experiments
            # ========================================================
            elif choice == "71":
                run_advanced_mode_resnet18_strong_aug()
                input("\nPress Enter to continue...")

            elif choice == "72":
                run_advanced_mode_resnet18_no_aug()
                input("\nPress Enter to continue...")

            # ========================================================
            # [73] Exit
            # ========================================================
            elif choice == "73":
                if musicPlayer is not None:
                    try:
                        musicPlayer.stop_music()
                    except Exception:
                        pass
                print("\n" + "=" * 80)
                print("👋 Goodbye! Thanks for using Liver AI Pipeline.")
                print("=" * 80)
                print(f"\n📌 Author: {AUTHOR}")
                print(f"📌 Version: {VERSION}")
                print("\n💡 If you found this helpful, please cite our work!")
                print("=" * 80)
                break

            # ========================================================
            # Invalid Choice
            # ========================================================
            else:
                print("\n❌ Invalid choice! Please enter a number between 1 and 73.")
                time.sleep(1)

    # ============================================================
    # KeyboardInterrupt / Exception Handling
    # ============================================================
    except KeyboardInterrupt:
        if musicPlayer is not None:
            try:
                musicPlayer.stop_music()
            except Exception:
                pass
        print("\n\n⚠️ Process interrupted by user.")
        print("👋 Goodbye!")

    except Exception as e:
        if musicPlayer is not None:
            try:
                musicPlayer.stop_music()
            except Exception:
                pass
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    wFTAlgorithms()
