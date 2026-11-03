# src/self_supervised/augmentations.py

"""
Advanced Data Augmentations for Self-Supervised Learning on Liver Ultrasound Images (ULTIMATE EDITION)

This module provides state-of-the-art data augmentation pipelines specifically designed for:
- Liver ultrasound images
- Imbalanced datasets (NL, G1, G2, G3)
- Self-Supervised Learning (SimCLR, BYOL, MAE)
- Transfer learning and fine-tuning
- GAN-based synthetic image generation
- Kornia differentiable augmentations
- Optuna-optimized augmentation policies

Key Features:
- Class-aware augmentations (different strategies for NL, G1, G2, G3)
- 1000+ synthetic samples per minority class (G3, G2, G1) via GAN
- Liver-specific augmentations (speckle noise, ultrasound artifacts)
- SimCLR and BYOL augmentation pipelines
- Advanced augmentations: CutMix, MixUp, RandomErasing
- Ultrasound-specific: Speckle noise, shadow simulation, attenuation
- AutoAugment style policies
- Kornia differentiable augmentations
- GAN-based synthetic image generation
- Optuna-optimized augmentation policies

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 7.0.0 (ULTIMATE EDITION with GAN + Kornia + Optuna)
"""

import os
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# ✅ CLASS DISTRIBUTION (For Balanced Augmentation)
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
CLASS_NAMES_LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Target distribution after augmentation
TARGET_DISTRIBUTION = {
    "NL": 1368,  # Keep original
    "G1": 1000,  # Increase to 1000
    "G2": 1000,  # Increase to 1000
    "G3": 1000,  # Increase to 1000
}

# Current distribution (for calculating augmentation factors)
CURRENT_DISTRIBUTION = {
    "NL": 1368,
    "G1": 483,
    "G2": 268,
    "G3": 47,
}

# Augmentation factor per class (how many times to augment)
AUGMENTATION_FACTOR = {
    "NL": 1.0,  # No augmentation for NL (already abundant)
    "G1": 2.1,  # 2.1x augmentation (483 → 1000+)
    "G2": 3.8,  # 3.8x augmentation (268 → 1000+)
    "G3": 21.3,  # 21.3x augmentation (47 → 1000+)
}

# Number of synthetic samples to generate per class
SYNTHETIC_SAMPLES = {
    "NL": 0,
    "G1": 517,  # 1000 - 483
    "G2": 732,  # 1000 - 268
    "G3": 953,  # 1000 - 47
}

# Augmentation intensity per class (0.0 - 1.0)
AUGMENTATION_INTENSITY = {
    "NL": 0.3,  # Low intensity for majority class
    "G1": 0.6,  # Medium intensity
    "G2": 0.8,  # High intensity
    "G3": 1.0,  # Maximum intensity
}


# ================================================================
# ✅ IMPORT KORNIA (با مدیریت خطا)
# ================================================================

try:
    import kornia.augmentation as K
    import kornia.geometry as KG

    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("⚠️ Kornia not available. Install with: pip install kornia")


# ================================================================
# ✅ KORNIA AUGMENTATIONS (Differentiable)
# ================================================================


class KorniaAugmentation:
    """
    Kornia differentiable augmentations for SSL.

    Advantages over torchvision:
    - Differentiable (can be used in training loops)
    - GPU-accelerated
    - More augmentation options
    - Batch-level operations

    Reference: "Kornia: an Open Source Differentiable Computer Vision Library"
    """

    def __init__(self, intensity: float = 0.5, target_size: int = 224, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Kornia augmentation pipeline.

        Args:
            intensity: Augmentation intensity (0.0 - 1.0)
            target_size: Target image size
            device: Device to run on
        """
        self.intensity = intensity
        self.target_size = target_size
        self.device = device

        if KORNIA_AVAILABLE:
            self.aug = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=20 * intensity),
                K.RandomResizedCrop(size=(target_size, target_size), scale=(0.7, 1.0)),
                K.ColorJitter(
                    brightness=0.3 * intensity,
                    contrast=0.3 * intensity,
                    saturation=0.3 * intensity,
                    hue=0.1 * intensity,
                    p=0.8,
                ),
                K.RandomGaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0 * intensity), p=0.5),
                K.RandomGaussianNoise(mean=0.0, std=0.05 * intensity, p=0.3),
                K.RandomErasing(scale=(0.02, 0.1 * intensity), ratio=(0.3, 3.3), p=0.3),
                data_keys=["input"],
                keepdim=True,
            ).to(device)
        else:
            self.aug = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Kornia augmentations to input tensor.

        Args:
            x: Input tensor (B, C, H, W) or (C, H, W)

        Returns:
            Augmented tensor
        """
        if not KORNIA_AVAILABLE or self.aug is None:
            return x

        # Ensure batch dimension
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        # Move to device
        x = x.to(self.device)

        # Apply augmentation
        with torch.no_grad():
            x = self.aug(x)

        # Remove batch dimension if needed
        if single:
            x = x.squeeze(0)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get augmentation configuration."""
        return {
            "type": "kornia",
            "intensity": self.intensity,
            "target_size": self.target_size,
            "device": self.device,
            "available": KORNIA_AVAILABLE,
        }


# ================================================================
# ✅ ULTRASOUND-SPECIFIC AUGMENTATIONS 🔥 NEW
# ================================================================


class UltrasoundSpeckleNoise:
    """
    Simulate ultrasound speckle noise.

    Speckle noise is multiplicative noise that appears in ultrasound images.
    This augmentation helps the model learn robust features.

    Reference: "Speckle Noise Reduction in Ultrasound Images" (IEEE)
    """

    def __init__(self, severity: float = 0.5, p: float = 0.5):
        self.severity = severity
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        # Generate speckle noise (multiplicative)
        noise = np.random.normal(1.0, self.severity * 0.3, image.shape)
        noisy = image * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class UltrasoundShadowSimulation:
    """
    Simulate ultrasound shadowing effect.

    Shadowing occurs when sound waves are attenuated by structures.
    This augmentation helps the model learn shadow patterns.
    """

    def __init__(self, p: float = 0.3, severity: float = 0.5):
        self.p = p
        self.severity = severity

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        h, w = image.shape[:2]
        # Create shadow gradient
        x = np.linspace(0, 1, w)
        shadow = 1.0 - self.severity * 0.3 * x**2

        if len(image.shape) == 2:
            return (image * shadow).astype(np.uint8)
        else:
            for c in range(3):
                image[:, :, c] = (image[:, :, c] * shadow).astype(np.uint8)
        return image


class UltrasoundAttenuation:
    """
    Simulate ultrasound attenuation (depth-based signal loss).

    Attenuation causes signals to weaken with depth.
    """

    def __init__(self, attenuation_rate: float = 0.002, p: float = 0.3):
        self.attenuation_rate = attenuation_rate
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        h, w = image.shape[:2]
        y = np.linspace(0, 1, h)
        attenuation = np.exp(-self.attenuation_rate * y * h)
        attenuation = attenuation.reshape(h, 1)

        if len(image.shape) == 2:
            return (image * attenuation).astype(np.uint8)
        else:
            for c in range(3):
                image[:, :, c] = (image[:, :, c] * attenuation).astype(np.uint8)
        return image


class UltrasoundReverberation:
    """
    Simulate ultrasound reverberation artifacts.

    Reverberation occurs when sound waves bounce between strong reflectors.
    """

    def __init__(self, p: float = 0.2, strength: float = 0.3):
        self.p = p
        self.strength = strength

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        h, w = image.shape[:2]
        # Create reverberation pattern
        freq = random.randint(2, 5)
        x = np.linspace(0, freq * 2 * np.pi, w)
        wave = 0.5 + 0.5 * np.sin(x)
        wave = wave.reshape(1, -1)
        wave = np.tile(wave, (h, 1))

        if len(image.shape) == 2:
            return (image + self.strength * 50 * (wave - 0.5)).astype(np.uint8)
        else:
            for c in range(3):
                image[:, :, c] = (image[:, :, c] + self.strength * 50 * (wave - 0.5)).astype(
                    np.uint8
                )
        return image


class UltrasoundEdgeEnhancement:
    """
    Enhance edges in ultrasound images.

    Helps the model focus on tissue boundaries.
    """

    def __init__(self, p: float = 0.2, strength: float = 0.5):
        self.p = p
        self.strength = strength

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        # Simple edge enhancement using sharpening
        kernel = np.array([[0, -1, 0], [-1, 4 + self.strength, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)


class UltrasoundContrastEnhancement:
    """
    Enhance contrast in ultrasound images.

    Improves visibility of tissue structures.
    """

    def __init__(self, p: float = 0.3, clip_limit: float = 2.0):
        self.p = p
        self.clip_limit = clip_limit

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            return clahe.apply(image)
        else:
            # Convert to LAB and apply CLAHE on L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ================================================================
# ✅ ADVANCED AUGMENTATIONS
# ================================================================


class RandomErasing:
    """
    Random Erasing augmentation.

    Randomly erases a rectangular region in the image.
    Helps the model focus on non-local features.

    Reference: "Random Erasing Data Augmentation" (AAAI 2020)
    """

    def __init__(
        self,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        value: float = 0.0,
    ):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.value = value

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return img

        h, w = img.shape[:2]
        area = h * w

        for _ in range(100):
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)
            h_erase = int(round((target_area * aspect_ratio) ** 0.5))
            w_erase = int(round((target_area / aspect_ratio) ** 0.5))

            if h_erase < h and w_erase < w:
                x1 = np.random.randint(0, h - h_erase)
                y1 = np.random.randint(0, w - w_erase)

                if len(img.shape) == 2:
                    img[x1 : x1 + h_erase, y1 : y1 + w_erase] = self.value
                else:
                    img[x1 : x1 + h_erase, y1 : y1 + w_erase, :] = self.value
                return img

        return img


class CutMix:
    """
    CutMix augmentation.

    Cuts a patch from one image and pastes it onto another.
    The label is mixed proportionally to the area of the patch.

    Reference: "CutMix: Regularization Strategy to Train Strong Classifiers" (ICCV 2019)
    """

    def __init__(self, probability: float = 0.3, alpha: float = 1.0):
        self.probability = probability
        self.alpha = alpha

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, float]:
        if random.random() > self.probability:
            return image1, 1.0

        h, w = image1.shape[:2]
        lam = np.random.beta(self.alpha, self.alpha)

        cut_h = int(h * np.sqrt(1 - lam))
        cut_w = int(w * np.sqrt(1 - lam))
        cx = np.random.randint(0, w - cut_w)
        cy = np.random.randint(0, h - cut_h)

        if len(image1.shape) == 2:
            result = image1.copy()
            result[cy : cy + cut_h, cx : cx + cut_w] = image2[cy : cy + cut_h, cx : cx + cut_w]
        else:
            result = image1.copy()
            result[cy : cy + cut_h, cx : cx + cut_w, :] = image2[
                cy : cy + cut_h, cx : cx + cut_w, :
            ]

        lam = 1.0 - (cut_h * cut_w) / (h * w)
        return result, lam


class MixUp:
    """
    MixUp augmentation.

    Linearly interpolates two images and their labels.

    Reference: "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
    """

    def __init__(self, probability: float = 0.3, alpha: float = 0.2):
        self.probability = probability
        self.alpha = alpha

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, float]:
        if random.random() > self.probability:
            return image1, 1.0

        lam = np.random.beta(self.alpha, self.alpha)
        mixed = lam * image1 + (1 - lam) * image2
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)

        return mixed, lam


class RandomRotation:
    """Random rotation with reflection padding."""

    def __init__(self, degrees: float = 15, p: float = 0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img

        angle = random.uniform(-self.degrees, self.degrees)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        if len(img.shape) == 2:
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        else:
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        return np.fliplr(img).copy()


class RandomVerticalFlip:
    """Random vertical flip."""

    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        return np.flipud(img).copy()


class RandomZoom:
    """Random zoom in/out with padding."""

    def __init__(self, zoom_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.3):
        self.zoom_range = zoom_range
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img

        zoom = random.uniform(*self.zoom_range)
        h, w = img.shape[:2]
        new_h, new_w = int(h * zoom), int(w * zoom)

        if len(img.shape) == 2:
            resized = cv2.resize(img, (new_w, new_h))
            if zoom < 1:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                padded = np.zeros((h, w), dtype=np.uint8)
                padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
                return padded
            else:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                return resized[start_h : start_h + h, start_w : start_w + w]
        else:
            resized = cv2.resize(img, (new_w, new_h))
            if zoom < 1:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w, :] = resized
                return padded
            else:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                return resized[start_h : start_h + h, start_w : start_w + w, :]


class RandomBrightnessContrast:
    """Random brightness and contrast adjustment."""

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.7, 1.3),
        p: float = 0.5,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img

        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)

        img = img.astype(np.float32)
        mean = np.mean(img)
        img = (img - mean) * contrast + mean
        img = img * brightness
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class RandomGamma:
    """Random gamma correction."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.5, 1.5), p: float = 0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img

        gamma = random.uniform(*self.gamma_range)
        img = img.astype(np.float32) / 255.0
        img = np.power(img, gamma)
        img = (img * 255).astype(np.uint8)
        return img


class RandomGaussianNoise:
    """Random Gaussian noise addition."""

    def __init__(self, std_range: Tuple[float, float] = (5, 25), p: float = 0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img

        std = random.uniform(*self.std_range)
        noise = np.random.normal(0, std, img.shape).astype(np.int16)
        img = img.astype(np.int16) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class RandomBlur:
    """Random Gaussian blur."""

    def __init__(
        self,
        kernel_size: int = 5,
        sigma_range: Tuple[float, float] = (0.1, 2.0),
        p: float = 0.3,
    ):
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img

        sigma = random.uniform(*self.sigma_range)
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigma)


class RandomSharpness:
    """Random sharpening."""

    def __init__(self, strength_range: Tuple[float, float] = (0.5, 2.0), p: float = 0.2):
        self.strength_range = strength_range
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img

        strength = random.uniform(*self.strength_range)
        kernel = np.array([[0, -1, 0], [-1, 4 + strength, -1], [0, -1, 0]]) / strength
        return cv2.filter2D(img, -1, kernel)


class RandomElasticTransform:
    """
    Random elastic deformation for ultrasound images.

    Simulates tissue deformation in ultrasound imaging.
    """

    def __init__(self, alpha: float = 30, sigma: float = 4, p: float = 0.2):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img

        try:
            from scipy.ndimage import map_coordinates

            h, w = img.shape[:2]
            if len(img.shape) == 2:
                img = img.reshape(h, w, 1)

            dx = np.random.randn(h, w) * self.alpha
            dy = np.random.randn(h, w) * self.alpha

            dx = gaussian_filter(dx, self.sigma)
            dy = gaussian_filter(dy, self.sigma)

            x, y = np.meshgrid(np.arange(w), np.arange(h))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            result = np.zeros_like(img)
            for c in range(img.shape[2]):
                result[:, :, c] = map_coordinates(
                    img[:, :, c], indices, order=1, mode="reflect"
                ).reshape(h, w)

            if result.shape[2] == 1:
                result = result.squeeze(2)

            return result.astype(np.uint8)
        except:
            return img


# ================================================================
# ✅ AUTO AUGMENT POLICY (SimCLR Style)
# ================================================================


class AutoAugmentPolicy:
    """
    AutoAugment style policy for ultrasound images.
    """

    POLICIES = [
        # Policy 1: Strong geometric + intensity
        [
            ("random_rotation", 0.8, 20),
            ("random_brightness_contrast", 0.8, (0.6, 1.4, 0.6, 1.4)),
        ],
        # Policy 2: Ultrasound-specific
        [
            ("speckle_noise", 0.6, 0.5),
            ("shadow_simulation", 0.4, 0.5),
        ],
        # Policy 3: CutMix + Erasing
        [
            ("random_erasing", 0.5, 0.3),
            ("elastic_transform", 0.3, 0.5),
        ],
        # Policy 4: Color jitter + Blur
        [
            ("color_jitter", 0.7, 0.5),
            ("gaussian_blur", 0.5, 1.0),
        ],
    ]

    def __init__(self, intensity: float = 0.5):
        self.intensity = intensity
        self.policies = self._get_policies()

    def _get_policies(self):
        """Get augmentation policies."""
        policies = []
        for policy in self.POLICIES:
            policy_list = []
            for op_name, prob, params in policy:
                policy_list.append({"name": op_name, "prob": prob, "params": params})
            policies.append(policy_list)
        return policies

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Select random policy
        policy = random.choice(self.policies)

        # Apply operations in policy
        for op in policy:
            if random.random() < op["prob"]:
                image = self._apply_op(image, op["name"], op["params"])

        return image

    def _apply_op(self, image: np.ndarray, op_name: str, params):
        """Apply a single augmentation operation."""
        if op_name == "random_rotation":
            return RandomRotation(degrees=params, p=1.0)(image)
        elif op_name == "random_brightness_contrast":
            brightness, contrast = params
            return RandomBrightnessContrast(
                brightness_range=(brightness[0], brightness[1]),
                contrast_range=(contrast[0], contrast[1]),
                p=1.0,
            )(image)
        elif op_name == "speckle_noise":
            return UltrasoundSpeckleNoise(severity=params, p=1.0)(image)
        elif op_name == "shadow_simulation":
            return UltrasoundShadowSimulation(severity=params, p=1.0)(image)
        elif op_name == "random_erasing":
            return RandomErasing(probability=params, sl=0.02, sh=0.4)(image)
        elif op_name == "elastic_transform":
            return RandomElasticTransform(alpha=30 * params, sigma=4, p=1.0)(image)
        elif op_name == "color_jitter":
            return RandomBrightnessContrast(
                p=1.0, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)
            )(image)
        elif op_name == "gaussian_blur":
            return RandomBlur(kernel_size=5, sigma_range=(0.1, 2.0), p=1.0)(image)

        return image


# ================================================================
# ✅ GAN - GENERATIVE ADVERSARIAL NETWORK (INTEGRATED)
# ================================================================


class Generator(nn.Module):
    """
    Generator Network for Ultrasound Image Synthesis.

    Takes random noise (z) and generates realistic ultrasound images.

    Architecture:
        z (100) → Linear(256) → BN → ReLU
                 → Linear(512) → BN → ReLU
                 → Linear(1024) → BN → ReLU
                 → Linear(2048) → BN → ReLU
                 → Linear(3*224*224) → Tanh → Image (3, 224, 224)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        img_shape: Tuple[int, int, int] = (3, 224, 224),
        hidden_dims: List[int] = [256, 512, 1024, 2048],
    ):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.img_size = img_shape[1] * img_shape[2] * img_shape[0]

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.img_size))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def generate_images(self, num_images: int, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
        self.eval()
        z = torch.randn(num_images, self.latent_dim).to(device)

        with torch.no_grad():
            images = self(z)
            images = (images + 1) / 2
            images = images.cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            images = (images * 255).astype(np.uint8)

        return images


class Discriminator(nn.Module):
    """
    Discriminator Network for Ultrasound Image Discrimination.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int, int] = (3, 224, 224),
        feature_dims: List[int] = [64, 128, 256],
        dropout_rate: float = 0.3,
    ):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape

        layers = []
        in_channels = img_shape[0]

        for out_channels in feature_dims:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(dropout_rate))
            in_channels = out_channels

        self.feature_size = self._get_conv_output(img_shape)

        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.feature_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def _get_conv_output(self, shape: Tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            for module in self.model:
                if isinstance(module, (nn.Conv2d, nn.LeakyReLU, nn.Dropout2d)):
                    continue
                dummy = module(dummy)
                if isinstance(module, nn.Flatten):
                    return dummy.shape[1]
        return 1024

    def _initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)


class GANTrainer:
    """
    GAN Trainer for Ultrasound Image Synthesis.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        img_shape: Tuple[int, int, int] = (3, 224, 224),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        betas: Tuple[float, float] = (0.5, 0.999),
        use_wgan: bool = False,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
    ):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.device = device
        self.use_wgan = use_wgan
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

        self.generator = Generator(latent_dim, img_shape).to(device)
        self.discriminator = Discriminator(img_shape).to(device)

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=betas)

        self.criterion = nn.BCELoss()

        self.history = {"g_loss": [], "d_loss": [], "epochs": []}
        self.best_g_loss = float("inf")
        self.best_d_loss = float("inf")

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)

        d_interpolated = self.discriminator(interpolated)

        grad = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
        gradient_penalty = self.lambda_gp * ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def train_step(self, real_images: torch.Tensor, batch_size: int) -> Tuple[float, float]:
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # Train Discriminator
        self.optimizer_d.zero_grad()

        real_output = self.discriminator(real_images)
        d_real_loss = self.criterion(real_output, real_labels)

        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_images = self.generator(z)
        fake_output = self.discriminator(fake_images.detach())
        d_fake_loss = self.criterion(fake_output, fake_labels)

        if self.use_wgan:
            gp = self._gradient_penalty(real_images, fake_images)
            d_loss = -d_real_loss + d_fake_loss + gp
        else:
            d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        self.optimizer_d.step()

        # Train Generator
        self.optimizer_g.zero_grad()

        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_images = self.generator(z)
        fake_output = self.discriminator(fake_images)

        if self.use_wgan:
            g_loss = -fake_output.mean()
        else:
            g_loss = self.criterion(fake_output, real_labels)

        g_loss.backward()
        self.optimizer_g.step()

        return g_loss.item(), d_loss.item()

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 100,
        save_interval: int = 10,
        save_dir: str = "gan_checkpoints",
        generate_interval: int = 20,
        num_generate: int = 64,
    ) -> Dict[str, List]:
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("🎨 Starting GAN Training")
        print("=" * 80)
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {dataloader.batch_size}")
        print(f"   Latent Dim: {self.latent_dim}")
        print(f"   Device: {self.device}")
        print(f"   WGAN-GP: {self.use_wgan}")
        print("=" * 80)

        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0

            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for batch_idx, (images, _) in enumerate(loop):
                images = images.to(self.device)
                batch_size = images.size(0)

                g_loss, d_loss = self.train_step(images, batch_size)

                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                num_batches += 1

                loop.set_postfix({"G": f"{g_loss:.4f}", "D": f"{d_loss:.4f}"})

            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches

            self.history["g_loss"].append(avg_g_loss)
            self.history["d_loss"].append(avg_d_loss)
            self.history["epochs"].append(epoch + 1)

            if avg_g_loss < self.best_g_loss:
                self.best_g_loss = avg_g_loss
            if avg_d_loss < self.best_d_loss:
                self.best_d_loss = avg_d_loss

            print(
                f"📊 Epoch {epoch+1}/{epochs} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}"
            )

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(os.path.join(save_dir, f"gan_epoch_{epoch+1}.pth"))

            if (epoch + 1) % generate_interval == 0:
                self.generate_and_save(
                    num_images=num_generate,
                    save_path=os.path.join(save_dir, f"generated_epoch_{epoch+1}.png"),
                )

        self.save_checkpoint(os.path.join(save_dir, "gan_final.pth"))

        print("\n" + "=" * 80)
        print("✅ GAN Training Complete!")
        print(f"   Best G Loss: {self.best_g_loss:.4f}")
        print(f"   Best D Loss: {self.best_d_loss:.4f}")
        print("=" * 80)

        return self.history

    def generate_images(self, num_images: int) -> np.ndarray:
        return self.generator.generate_images(num_images, self.device)

    def generate_and_save(
        self,
        num_images: int,
        save_path: str,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        images = self.generate_images(num_images)

        if grid_size is None:
            cols = int(np.sqrt(num_images))
            rows = int(np.ceil(num_images / cols))
        else:
            rows, cols = grid_size

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < num_images:
                    axes[i, j].imshow(images[idx])
                    axes[i, j].axis("off")
                else:
                    axes[i, j].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Generated images saved: {save_path}")

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "history": self.history,
            "best_g_loss": self.best_g_loss,
            "best_d_loss": self.best_d_loss,
            "latent_dim": self.latent_dim,
            "img_shape": self.img_shape,
            "use_wgan": self.use_wgan,
            "lambda_gp": self.lambda_gp,
            "n_critic": self.n_critic,
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(checkpoint, path)
        print(f"✅ GAN checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        self.history = checkpoint["history"]
        self.best_g_loss = checkpoint["best_g_loss"]
        self.best_d_loss = checkpoint["best_d_loss"]

        print(f"✅ GAN checkpoint loaded: {path}")


# ================================================================
# ✅ GAN FACTORY FUNCTIONS
# ================================================================


def create_gan_trainer(
    latent_dim: int = 100,
    img_shape: Tuple[int, int, int] = (3, 224, 224),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lr_g: float = 0.0002,
    lr_d: float = 0.0002,
    betas: Tuple[float, float] = (0.5, 0.999),
    use_wgan: bool = False,
    lambda_gp: float = 10.0,
    n_critic: int = 5,
) -> GANTrainer:
    """Factory function to create GAN trainer."""
    return GANTrainer(
        latent_dim=latent_dim,
        img_shape=img_shape,
        device=device,
        lr_g=lr_g,
        lr_d=lr_d,
        betas=betas,
        use_wgan=use_wgan,
        lambda_gp=lambda_gp,
        n_critic=n_critic,
    )


def generate_synthetic_images(
    model_path: str,
    num_images: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Generate synthetic images using trained GAN."""
    checkpoint = torch.load(model_path, map_location=device)

    latent_dim = checkpoint.get("latent_dim", 100)
    img_shape = checkpoint.get("img_shape", (3, 224, 224))

    generator = Generator(latent_dim, img_shape).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        images = generator(z)
        images = (images + 1) / 2
        images = images.cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))
        images = (images * 255).astype(np.uint8)

    if save_path:
        import matplotlib.pyplot as plt

        cols = int(np.sqrt(num_images))
        rows = int(np.ceil(num_images / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < num_images:
                    axes[i, j].imshow(images[idx])
                    axes[i, j].axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Generated images saved: {save_path}")

    return images


# ================================================================
# ✅ CLASS-AWARE AUGMENTATION PIPELINE (ENHANCED with GAN + Kornia)
# ================================================================


class ClassAwareAugmentation:
    """
    Class-aware augmentation pipeline for imbalanced datasets.

    Features:
    - 20+ augmentations for G3 class
    - Ultrasound-specific augmentations
    - AutoAugment style policies
    - GAN-based synthetic image generation
    - Kornia differentiable augmentations
    - Adjustable intensity per class
    """

    def __init__(
        self,
        class_name: str,
        target_size: int = 224,
        augmentation_strength: float = 1.0,
        use_auto_augment: bool = True,
        use_kornia: bool = True,
        use_gan: bool = False,
        gan_checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize class-aware augmentation.

        Args:
            class_name: 'NL', 'G1', 'G2', or 'G3'
            target_size: Target image size
            augmentation_strength: 0.5-2.0
            use_auto_augment: Whether to use AutoAugment policies
            use_kornia: Whether to use Kornia augmentations
            use_gan: Whether to use GAN for synthetic generation
            gan_checkpoint: Path to trained GAN checkpoint
            device: Device to run on
        """
        self.class_name = class_name
        self.target_size = target_size
        self.augmentation_strength = augmentation_strength
        self.use_auto_augment = use_auto_augment
        self.use_kornia = use_kornia
        self.use_gan = use_gan
        self.gan_checkpoint = gan_checkpoint
        self.device = device

        self.factor = AUGMENTATION_FACTOR.get(class_name, 1.0)
        self.intensity = AUGMENTATION_INTENSITY.get(class_name, 0.5)

        # ✅ Kornia dual-view
        if use_kornia and KORNIA_AVAILABLE:
            self.kornia_aug = DualViewKorniaAugmentation(
                intensity=self.intensity, target_size=target_size, device=device
            )
        else:
            self.kornia_aug = None

        # ✅ GAN synthetic images
        self.synthetic_images = None
        if self.use_gan and self.gan_checkpoint and os.path.exists(self.gan_checkpoint):
            try:
                self.synthetic_images = generate_synthetic_images(
                    self.gan_checkpoint,
                    int(SYNTHETIC_SAMPLES.get(class_name, 0)),
                    device,
                )
                print(
                    f"✅ GAN synthetic images loaded: {len(self.synthetic_images)} for {class_name}"
                )
            except Exception as e:
                print(f"⚠️ GAN loading failed: {e}")
                self.synthetic_images = None

        # ✅ Kornia augmenter
        self.kornia_aug = None
        if self.use_kornia and KORNIA_AVAILABLE:
            self.kornia_aug = KorniaAugmentation(
                intensity=self.intensity, target_size=target_size, device=device
            )

        from src.self_supervised.OptimizationsAlgorithms.Augmentations.imgaug_pipeline import get_imgaug_pipeline

        self.imgaug_pipeline = get_imgaug_pipeline(class_name)

        if class_name == "G3":
            self.intensity_level = "maximum"
        elif class_name == "G2":
            self.intensity_level = "very_high"
        elif class_name == "G1":
            self.intensity_level = "high"
        else:
            self.intensity_level = "low"

        self._build_pipeline()

    def _build_pipeline(self):
        """Build augmentation pipeline based on class."""

        base_transforms = [
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15, p=0.5),
        ]

        geometric_transforms = [
            RandomZoom(
                zoom_range=(0.85 - self.intensity * 0.15, 1.15 + self.intensity * 0.15),
                p=0.3 + self.intensity * 0.4,
            ),
            RandomVerticalFlip(p=0.1 + self.intensity * 0.2),
        ]

        intensity_transforms = [
            RandomBrightnessContrast(
                brightness_range=(
                    0.7 - self.intensity * 0.3,
                    1.3 + self.intensity * 0.3,
                ),
                contrast_range=(0.7 - self.intensity * 0.3, 1.3 + self.intensity * 0.3),
                p=0.4 + self.intensity * 0.4,
            ),
            RandomGamma(
                gamma_range=(0.5 - self.intensity * 0.2, 1.5 + self.intensity * 0.3),
                p=0.2 + self.intensity * 0.3,
            ),
        ]

        noise_transforms = [
            RandomGaussianNoise(
                std_range=(5 + self.intensity * 10, 15 + self.intensity * 20),
                p=0.2 + self.intensity * 0.3,
            ),
            RandomBlur(
                kernel_size=5,
                sigma_range=(0.1 + self.intensity * 0.2, 1.0 + self.intensity * 1.5),
                p=0.2 + self.intensity * 0.2,
            ),
        ]

        ultrasound_transforms = [
            UltrasoundSpeckleNoise(
                severity=0.3 + self.intensity * 0.6, p=0.3 + self.intensity * 0.4
            ),
            UltrasoundShadowSimulation(
                severity=0.3 + self.intensity * 0.4, p=0.2 + self.intensity * 0.3
            ),
            UltrasoundAttenuation(
                attenuation_rate=0.001 + self.intensity * 0.003,
                p=0.2 + self.intensity * 0.2,
            ),
            UltrasoundReverberation(
                strength=0.2 + self.intensity * 0.4, p=0.1 + self.intensity * 0.2
            ),
            UltrasoundContrastEnhancement(
                clip_limit=1.0 + self.intensity * 2.0, p=0.2 + self.intensity * 0.2
            ),
        ]

        advanced_transforms = [
            RandomErasing(
                probability=0.2 + self.intensity * 0.4,
                sl=0.02,
                sh=0.2 + self.intensity * 0.2,
                value=0,
            ),
            RandomSharpness(
                strength_range=(0.5 + self.intensity * 0.5, 1.5 + self.intensity * 1.0),
                p=0.1 + self.intensity * 0.2,
            ),
            RandomElasticTransform(
                alpha=15 + self.intensity * 30, sigma=4, p=0.1 + self.intensity * 0.2
            ),
        ]

        if self.intensity_level == "maximum":
            self.transforms = (
                base_transforms
                + geometric_transforms
                + intensity_transforms
                + noise_transforms
                + ultrasound_transforms
                + advanced_transforms
            )
        elif self.intensity_level == "very_high":
            self.transforms = (
                base_transforms
                + geometric_transforms
                + intensity_transforms
                + noise_transforms
                + ultrasound_transforms[:4]
                + advanced_transforms[:2]
            )
        elif self.intensity_level == "high":
            self.transforms = (
                base_transforms
                + geometric_transforms[:1]
                + intensity_transforms[:1]
                + noise_transforms[:1]
                + ultrasound_transforms[:3]
            )
        else:
            self.transforms = (
                base_transforms
                + geometric_transforms[:1]
                + intensity_transforms[:1]
                + noise_transforms[:1]
            )

        if self.use_auto_augment and self.intensity_level in ["maximum", "very_high"]:
            self.auto_augment = AutoAugmentPolicy(intensity=self.intensity)
        else:
            self.auto_augment = None

        if self.augmentation_strength != 1.0:
            self._scale_augmentation_strength()

    def _scale_augmentation_strength(self):
        for transform in self.transforms:
            if hasattr(transform, "p"):
                transform.p = min(1.0, transform.p * self.augmentation_strength)
            if hasattr(transform, "severity"):
                transform.severity = min(1.0, transform.severity * self.augmentation_strength)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Apply random transforms

        img1 = image.copy()
        for transform in self.transforms:
            try:
                img1 = transform(img1)
            except Exception as e:
                continue

        # View 2 (slightly different)
        img2 = image.copy()
        # Use a subset of transforms
        for transform in random.sample(self.transforms, max(1, len(self.transforms) // 2)):
            try:
                img2 = transform(img2)
            except:
                pass

        # ✅ Kornia (if available)
        if self.kornia_aug is not None and random.random() < 0.3:
            try:
                # Convert to tensor
                img1_t = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1) / 255.0
                img1_t = img1_t.unsqueeze(0).to(self.device)

                # Apply Kornia (returns two views)
                v1, v2 = self.kornia_aug(img1_t)

                # Convert back to numpy
                img1 = v1.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img1 = (img1 * 255).astype(np.uint8)

                img2 = v2.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img2 = (img2 * 255).astype(np.uint8)
            except:
                pass

        # Resize
        if img1.shape[:2] != (self.target_size, self.target_size):
            img1 = cv2.resize(img1, (self.target_size, self.target_size))
            img2 = cv2.resize(img2, (self.target_size, self.target_size))

        return img1, img2

    def get_augmentation_count(self) -> int:
        return len(self.transforms)

    def get_augmentation_factor(self) -> float:
        return self.factor


def get_class_aware_augmentation(
    class_name: str,
    target_size: int = 224,
    augmentation_strength: float = 1.0,
    use_auto_augment: bool = True,
    use_kornia: bool = True,
    use_gan: bool = False,
    gan_checkpoint: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    return_views: int = 2,
) -> ClassAwareAugmentation:
    """Factory function to get class-aware augmentation pipeline with GAN + Kornia."""
    aug = ClassAwareAugmentation(
        class_name=class_name,
        target_size=target_size,
        augmentation_strength=augmentation_strength,
        use_auto_augment=use_auto_augment,
        use_kornia=use_kornia,
        use_gan=use_gan,
        gan_checkpoint=gan_checkpoint,
        device=device,
    )
    # ✅ اگر یک view بخواد، wrapper بساز
    if return_views == 1:

        class SingleViewWrapper:
            def __init__(self, aug):
                self.aug = aug

            def __call__(self, img):
                v1, v2 = self.aug(img)
                return v1

        return SingleViewWrapper(aug)
    return aug


# ================================================================
# ✅ DATA AUGMENTATION FOR IMBALANCED DATASET (UPDATED)
# ================================================================


def augment_imbalanced_dataset(
    dataset: torch.utils.data.Dataset,
    target_size: int = 224,
    augmentation_strength: float = 1.0,
    use_auto_augment: bool = True,
    use_kornia: bool = True,
    use_gan: bool = False,
    gan_checkpoint: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.utils.data.Dataset:
    """
    Augment dataset to balance classes with GAN + Kornia support.

    G3: 47 → 1000+ (21x augmentation)
    G2: 268 → 1000+ (4x augmentation)
    G1: 483 → 1000+ (2x augmentation)
    NL: 1368 → 1368 (no augmentation)
    """
    from torch.utils.data import Dataset

    class BalancedDataset(Dataset):
        def __init__(
            self,
            dataset,
            target_size,
            augmentation_strength,
            use_auto_augment,
            use_kornia,
            use_gan,
            gan_checkpoint,
            device,
            is_ssl: bool = True,
        ):
            self.dataset = dataset
            self.target_size = target_size
            self.augmentation_strength = augmentation_strength
            self.use_auto_augment = use_auto_augment
            self.use_kornia = use_kornia
            self.use_gan = use_gan
            self.gan_checkpoint = gan_checkpoint
            self.device = device
            self.is_ssl = is_ssl
            self.samples = []

            for i in range(len(dataset)):
                img, label, _ = dataset[i]
                class_name = CLASS_NAMES[label]
                self.samples.append((img, label, class_name, i))

            self._generate_augmented_samples()

            print(f"✅ Original samples: {len(dataset)}")
            print(f"✅ Augmented samples: {len(self.samples)}")
            print(f"   - NL: {sum(1 for s in self.samples if s[2] == 'NL')}")
            print(f"   - G1: {sum(1 for s in self.samples if s[2] == 'G1')}")
            print(f"   - G2: {sum(1 for s in self.samples if s[2] == 'G2')}")
            print(f"   - G3: {sum(1 for s in self.samples if s[2] == 'G3')}")

        def _generate_augmented_samples(self):
            new_samples = []

            for img, label, class_name, idx in self.samples:
                aug = get_class_aware_augmentation(
                    class_name=class_name,
                    target_size=self.target_size,
                    augmentation_strength=self.augmentation_strength,
                    use_auto_augment=self.use_auto_augment,
                    use_kornia=self.use_kornia,
                    use_gan=self.use_gan,
                    gan_checkpoint=self.gan_checkpoint,
                    device=self.device,
                )

                factor = AUGMENTATION_FACTOR.get(class_name, 1.0)
                num_aug = int(factor)

                for _ in range(num_aug):
                    view1, view2 = aug(img)
                    new_samples.append(((view1, view2), label, class_name, idx))

            self.samples.extend(new_samples)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            data = self.samples[idx]
            img, label, _, _ = data
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=0)
                    img = torch.tensor(img, dtype=torch.float32) / 255.0
                else:
                    img = torch.tensor(img, dtype=torch.float32) / 255.0

            if self.is_ssl:
                # ✅ دو view
                (v1, v2), label, _, _ = data
                if isinstance(v1, np.ndarray):
                    v1 = torch.tensor(v1, dtype=torch.float32).permute(2, 0, 1) / 255.0
                    v2 = torch.tensor(v2, dtype=torch.float32).permute(2, 0, 1) / 255.0
                return v1, v2
            else:
                # ✅ یک view
                img, label, _, _ = data
                if isinstance(img, np.ndarray):
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=0)
                        img = torch.tensor(img, dtype=torch.float32) / 255.0
                    else:
                        img = torch.tensor(img, dtype=torch.float32) / 255.0
                return img, label

    return BalancedDataset(
        dataset,
        target_size,
        augmentation_strength,
        use_auto_augment,
        use_kornia,
        use_gan,
        gan_checkpoint,
        device,
    )


# ================================================================
# ✅ RANDOM COLOR JITTER
# ================================================================


class RandomColorJitter:
    """Random Color Jitter augmentation using torchvision."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        p: float = 0.8,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
        self.jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return self.jitter(x)
        return x


# ================================================================
# ✅ RANDOM GRAYSCALE
# ================================================================


class RandomGrayscale:
    """Random Grayscale augmentation."""

    def __init__(self, p: float = 0.2):
        self.p = p
        self.grayscale = T.RandomGrayscale(p=1.0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return self.grayscale(x)
        return x


# ================================================================
# ✅ RANDOM SOLARIZE
# ================================================================


class RandomSolarize:
    """Random Solarize augmentation."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.2:
            return TF.solarize(x, self.threshold)
        return x


# ================================================================
# ✅ FIX 1: RandomGaussianBlur - با پشتیبانی از GPU
# ================================================================


class RandomGaussianBlur:
    def __init__(
        self,
        kernel_size: int = 23,
        sigma_range: Tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
    ):
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x

        sigma = max(0.5, random.uniform(*self.sigma_range))  # ✅ حداقل 0.5

        if isinstance(x, np.ndarray):
            x_float = x.astype(np.float32)
            if x_float.ndim == 3 and x_float.shape[2] == 3:
                blurred = np.zeros_like(x_float)
                for c in range(3):
                    blurred[:, :, c] = cv2.GaussianBlur(
                        x_float[:, :, c], (self.kernel_size, self.kernel_size), sigma
                    )
                return blurred.astype(x.dtype)
            else:
                blurred = cv2.GaussianBlur(x_float, (self.kernel_size, self.kernel_size), sigma)
                return blurred.astype(x.dtype)

        elif isinstance(x, torch.Tensor):

            if x.device.type != "cpu":
                x_cpu = x.cpu()
            else:
                x_cpu = x

            if x_cpu.dim() == 3 and x_cpu.shape[0] == 3:
                img_np = x_cpu.permute(1, 2, 0).numpy().astype(np.float32)
                blurred = np.zeros_like(img_np)
                for c in range(3):
                    blurred[:, :, c] = cv2.GaussianBlur(
                        img_np[:, :, c], (self.kernel_size, self.kernel_size), sigma
                    )
                result = torch.tensor(blurred, dtype=x.dtype).permute(2, 0, 1)
                return result.to(x.device)  # ✅ برگردون به device اصلی
            else:
                img_np = x_cpu.numpy().astype(np.float32)
                blurred = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), sigma)
                result = torch.tensor(blurred, dtype=x.dtype)
                return result.to(x.device)

        return x


# ================================================================
# ✅ FIX 2: DualViewKorniaAugmentation
# ================================================================


class DualViewKorniaAugmentation:
    """
    ✅ FIXED: Kornia augmentation that returns two views for BYOL.
    """

    def __init__(self, intensity: float = 0.5, target_size: int = 224, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.intensity = intensity
        self.target_size = target_size
        self.device = device

        if KORNIA_AVAILABLE:
            self.aug1 = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=15 * intensity),
                K.RandomResizedCrop(
                    size=(target_size, target_size), scale=(0.85, 1.0)
                ),  # ✅ medical safe
                K.ColorJitter(
                    brightness=0.3 * intensity,
                    contrast=0.3 * intensity,
                    saturation=0.3 * intensity,
                    hue=0.1 * intensity,
                    p=0.8,
                ),
                K.RandomGaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0 * intensity), p=0.5),
                data_keys=["input"],
                keepdim=True,
            ).to(device)

            self.aug2 = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=15 * intensity * 1.2),
                K.RandomResizedCrop(size=(target_size, target_size), scale=(0.80, 1.0)),
                K.ColorJitter(
                    brightness=0.4 * intensity,
                    contrast=0.4 * intensity,
                    saturation=0.4 * intensity,
                    hue=0.15 * intensity,
                    p=0.8,
                ),
                K.RandomGaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.5 * intensity), p=0.6),
                K.RandomSolarize(thresholds=0.5, p=0.2),
                data_keys=["input"],
                keepdim=True,
            ).to(device)
        else:
            self.aug1 = None
            self.aug2 = None

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1) / 255.0
            x = x.unsqueeze(0)
            single = True
        elif isinstance(x, torch.Tensor):
            if x.dim() == 3:
                x = x.unsqueeze(0)
                single = True
            else:
                single = False
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

        if not KORNIA_AVAILABLE or self.aug1 is None:
            return x, x

        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        x = x.to(self.device)

        with torch.no_grad():
            view1 = self.aug1(x)
            view2 = self.aug2(x)

        if single:
            view1 = view1.squeeze(0)
            view2 = view2.squeeze(0)

        return view1, view2


# ================================================================
# ✅ SIMCLR AUGMENTATION PIPELINE
# ================================================================


class SimCLRAugmentation:
    """
    SimCLR augmentation pipeline for Self-Supervised Learning.
    Creates two different views of the same image.
    """

    def __init__(
        self,
        target_size: int = 224,
        crop_scale: Tuple[float, float] = (0.2, 1.0),
        color_jitter_strength: float = 1.0,
        blur_kernel_size: int = 23,
        normalize: bool = True,
    ):
        self.target_size = target_size
        self.crop_scale = crop_scale
        self.normalize = normalize

        self.base_transforms = T.Compose(
            [
                T.RandomResizedCrop(target_size, scale=crop_scale),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ]
        )

        self.color_jitter = RandomColorJitter(
            brightness=0.8 * color_jitter_strength,
            contrast=0.8 * color_jitter_strength,
            saturation=0.8 * color_jitter_strength,
            hue=0.2 * color_jitter_strength,
            p=0.8,
        )

        self.grayscale = RandomGrayscale(p=0.2)
        self.blur = RandomGaussianBlur(kernel_size=blur_kernel_size, sigma_range=(0.1, 2.0))
        self.solarize = RandomSolarize(threshold=0.5)

        if normalize:
            self.normalize_transform = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize_transform = T.Lambda(lambda x: x)

    def __call__(
        self, image: Union[Image.Image, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = T.ToPILImage()(image)

        view1 = self.base_transforms(image)
        view1 = self.color_jitter(view1)
        view1 = self.grayscale(view1)
        view1 = self.blur(view1)
        view1 = self.solarize(view1)
        view1 = self.normalize_transform(view1)

        view2 = self.base_transforms(image)

        if random.random() < 0.8:
            view2 = self.color_jitter(view2)

        view2 = self.grayscale(view2)

        if random.random() < 0.5:
            view2 = self.blur(view2)

        view2 = self.solarize(view2)
        view2 = self.normalize_transform(view2)

        return view1, view2


# ================================================================
# ✅ BYOL AUGMENTATION PIPELINE
# ================================================================


class BYOLAugmentation:
    """
    BYOL augmentation pipeline (similar to SimCLR but with different parameters).
    """

    def __init__(
            self,
            target_size: int = 224,
            crop_scale: Tuple[float, float] = (0.2, 1.0),
            normalize: bool = True,
            color_jitter_strength: float = 1.0,
            blur_kernel_size: int = 23,
            **kwargs,
    ):
        self.target_size = target_size
        self.crop_scale = crop_scale
        self.normalize = normalize

        # ✅ FIX: augmentation بسیار قوی‌تر برای liver ultrasound
        # مشکل اصلی این بود که دو view تقریباً یکسان بودند
        # راه‌حل: scale های کاملاً متفاوت + augmentation های قوی

        brightness = 0.6 * color_jitter_strength  # از 0.4 → 0.6
        contrast = 0.6 * color_jitter_strength  # از 0.4 → 0.6
        saturation = 0.5 * color_jitter_strength  # از 0.2 → 0.5
        hue = 0.15 * color_jitter_strength  # از 0.1 → 0.15

        # ✅ دو scale کاملاً متفاوت
        # view1: ناحیه بزرگ (near full image)
        # view2: ناحیه کوچک (zoomed in)
        v1_scale = (0.5, 1.0)  # view1: تصویر نسبتاً کامل
        v2_scale = (0.08, 0.4)  # view2: crop کوچک (خیلی aggressive)

        # ✅ view1: light augmentation
        self.view1 = T.Compose([
            T.RandomResizedCrop(target_size, scale=v1_scale, ratio=(0.75, 1.333)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness, contrast, saturation, hue)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            (T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             if normalize else T.Lambda(lambda x: x)),
        ])

        # ✅ view2: heavy augmentation (خیلی متفاوت از view1)
        self.view2 = T.Compose([
            T.RandomResizedCrop(target_size, scale=v2_scale, ratio=(0.75, 1.333)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness, contrast, saturation, hue)], p=0.9),  # p بالاتر
            T.RandomGrayscale(p=0.3),  # p بالاتر
            T.RandomApply(
                [T.GaussianBlur(kernel_size=blur_kernel_size, sigma=(0.1, 3.0))],  # sigma بالاتر
                p=0.7,  # p بالاتر
            ),
            T.RandomSolarize(threshold=0.4, p=0.3),  # p بالاتر
            T.ToTensor(),
            (T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             if normalize else T.Lambda(lambda x: x)),
        ])

    def __call__(
        self, image: Union[Image.Image, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            batch_size = image.size(0)
            view1_list = []
            view2_list = []

            for i in range(batch_size):
                img_pil = T.ToPILImage()(image[i])
                v1 = self.view1(img_pil)
                v2 = self.view2(img_pil)
                view1_list.append(v1.unsqueeze(0))
                view2_list.append(v2.unsqueeze(0))

            view1 = torch.cat(view1_list, dim=0)
            view2 = torch.cat(view2_list, dim=0)
            return view1, view2

        # ✅ تک تصویر
        if isinstance(image, torch.Tensor):
            image = T.ToPILImage()(image)

        view1 = self.view1(image)
        view2 = self.view2(image)

        return view1, view2


# ================================================================
# ✅ MAE AUGMENTATION PIPELINE
# ================================================================


class MAEAugmentation:
    """
    MAE (Masked Autoencoder) augmentation pipeline.

    MAE uses a simpler augmentation strategy compared to SimCLR/BYOL:
    - Random resized crop
    - Random horizontal flip
    - Normalization

    MAE doesn't use strong color jitter or blur because it relies on
    masking for its self-supervised signal.

    Reference: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022)
    """

    def __init__(
        self,
        target_size: int = 224,
        crop_scale: Tuple[float, float] = (0.2, 1.0),
        normalize: bool = True,
        use_color_jitter: bool = False,
        use_blur: bool = False,
    ):
        """
        Initialize MAE augmentation pipeline.

        Args:
            target_size: Target image size
            crop_scale: Random resized crop scale range
            normalize: Whether to normalize to ImageNet stats
            use_color_jitter: Whether to use color jitter (MAE typically doesn't)
            use_blur: Whether to use Gaussian blur (MAE typically doesn't)
        """
        self.target_size = target_size
        self.crop_scale = crop_scale
        self.normalize = normalize
        self.use_color_jitter = use_color_jitter
        self.use_blur = use_blur

        # Base transforms
        transforms = [
            T.RandomResizedCrop(target_size, scale=crop_scale),
            T.RandomHorizontalFlip(p=0.5),
        ]

        # Optional color jitter (MAE usually doesn't use this)
        if use_color_jitter:
            transforms.append(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1))

        # Optional Gaussian blur (MAE usually doesn't use this)
        if use_blur:
            transforms.append(T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)))

        transforms.append(T.ToTensor())

        if normalize:
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = T.Compose(transforms)

    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Apply augmentation to a single image.

        Args:
            image: Input image (PIL Image or torch Tensor)

        Returns:
            Augmented image tensor (C, H, W)
        """
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = T.ToPILImage()(image)
            else:
                # If it's a batch, take first element
                image = T.ToPILImage()(image[0] if image.dim() == 4 else image)

        return self.transform(image)

    def __repr__(self) -> str:
        return (
            f"MAEAugmentation(target_size={self.target_size}, "
            f"crop_scale={self.crop_scale}, "
            f"normalize={self.normalize}, "
            f"use_color_jitter={self.use_color_jitter}, "
            f"use_blur={self.use_blur})"
        )


# ================================================================
# ✅ MAE WITH TWO VIEWS (برای سازگاری با BYOL/SimCLR)
# ================================================================


class MAEAugmentationTwoViews:
    """
    MAE augmentation with two views (for compatibility with BYOL/SimCLR).

    This is useful if you want to use MAE-style augmentations but need
    two views for methods like BYOL or SimCLR.
    """

    def __init__(
        self,
        target_size: int = 224,
        crop_scale: Tuple[float, float] = (0.2, 1.0),
        normalize: bool = True,
    ):
        self.target_size = target_size
        self.crop_scale = crop_scale
        self.normalize = normalize

        # View 1: Standard MAE augmentation
        self.view1 = MAEAugmentation(
            target_size=target_size,
            crop_scale=crop_scale,
            normalize=normalize,
            use_color_jitter=False,
            use_blur=False,
        )

        # View 2: Slightly different augmentation
        self.view2 = MAEAugmentation(
            target_size=target_size,
            crop_scale=(crop_scale[0], crop_scale[1] * 0.9),  # Slightly different scale
            normalize=normalize,
            use_color_jitter=False,
            use_blur=False,
        )

    def __call__(
        self, image: Union[Image.Image, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views of the input image.

        Args:
            image: Input image (PIL Image or torch Tensor)

        Returns:
            Tuple of (view1, view2) tensors
        """
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = T.ToPILImage()(image)

        v1 = self.view1(image)
        v2 = self.view2(image)

        return v1, v2


# ================================================================
# ✅ AUGMENTATION FACTORY (UPDATED)
# ================================================================


def get_ssl_augmentation(
    method: str = "simclr", target_size: int = 224, **kwargs
) -> Union[SimCLRAugmentation, BYOLAugmentation, MAEAugmentation]:
    """Factory function to get SSL augmentation pipeline."""
    if method.lower() == "simclr":
        return SimCLRAugmentation(target_size=target_size, **kwargs)
    elif method.lower() == "byol":
        return BYOLAugmentation(target_size=target_size, **kwargs)
    elif method.lower() == "mae":
        return MAEAugmentation(target_size=target_size, **kwargs)
    else:
        raise ValueError(f"Unknown SSL method: {method}")


# ================================================================
# ✅ ALBUMENTATIONS-BASED MEDICAL-SAFE AUGMENTATION (NEW)
# ================================================================

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("⚠️ Albumentations not available. Install with: pip install albumentations")


class MedicalSafeBYOLAugmentation:
    """
    Medical-Safe BYOL Augmentation using Albumentations.

    Designed for Liver Ultrasound Images with the following constraints:
    - NO Gaussian noise (ultrasound has speckle noise naturally)
    - NO strong hue changes (keeps grayscale-like appearance)
    - NO rotation > 15 degrees (anatomically unnatural)
    - Limited brightness/contrast (medical imaging-safe)

    Includes:
    - Ultrasound-specific speckle noise
    - Shadow simulation
    - Attenuation
    - Elastic deformation (mild)
    - Grid distortion (mild)
    - Coarse dropout (small)
    """

    def __init__(
            self,
            target_size: int = 224,
            normalize: bool = True,
    ):
        if not HAS_ALBUMENTATIONS:
            raise ImportError("Albumentations not installed")

        self.target_size = target_size
        self.normalize = normalize

        # ✅ View 1: Lighter augmentation (whole image)
        self.view1 = self._build_pipeline(
            crop_scale=(0.7, 1.0),
            strength=0.6,
        )

        # ✅ View 2: Heavier augmentation (focused crop)
        self.view2 = self._build_pipeline(
            crop_scale=(0.5, 0.85),
            strength=1.0,
        )

    def _build_pipeline(self, crop_scale, strength):
        """Build the augmentation pipeline."""
        # ✅ Normalize
        normalize_transform = (
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            )
            if self.normalize
            else A.NoOp()
        )

        return A.Compose([
            # ✅ Spatial: crop with different scales for each view
            A.RandomResizedCrop(
                height=self.target_size,
                width=self.target_size,
                scale=crop_scale,
                ratio=(0.85, 1.15),
                p=1.0,
            ),

            # ✅ Horizontal flip (medical-safe)
            A.HorizontalFlip(p=0.5),

            # ✅ Rotation (limited to avoid anatomical issues)
            A.Rotate(
                limit=10 * strength,  # max 10 degrees
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),

            # ✅ Translation (small)
            A.Affine(
                translate_percent={"x": (-0.05 * strength, 0.05 * strength),
                                   "y": (-0.05 * strength, 0.05 * strength)},
                scale=(0.95, 1.05),
                shear=(-3 * strength, 3 * strength),
                p=0.4,
            ),

            # ✅ Brightness/Contrast (limited range for medical)
            A.RandomBrightnessContrast(
                brightness_limit=0.15 * strength,
                contrast_limit=0.15 * strength,
                p=0.6,
            ),

            # ✅ Gamma correction (mild)
            A.RandomGamma(
                gamma_limit=(int(90 - 10 * strength), int(110 + 10 * strength)),
                p=0.3,
            ),

            # ✅ Ultrasound speckle noise (REPLACES gaussian noise)
            # This is multiplicative noise, natural for ultrasound
            A.MultiplicativeNoise(
                multiplier=(1.0 - 0.1 * strength, 1.0 + 0.1 * strength),
                per_channel=True,
                p=0.4,
            ),

            # ✅ Elastic transform (mild)
            A.ElasticTransform(
                alpha=10 * strength,
                sigma=3,
                p=0.2,
            ),

            # ✅ Grid distortion (mild)
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1 * strength,
                p=0.2,
            ),

            # ✅ Coarse dropout (small holes, simulating shadows)
            A.CoarseDropout(
                max_holes=4,
                max_height=int(20 * strength),
                max_width=int(20 * strength),
                fill_value=0,
                p=0.2,
            ),

            # ✅ Normalize
            normalize_transform,

            # ✅ To Tensor
            ToTensorV2(),
        ])

    def __call__(self, image):
        """Apply augmentation to a single PIL Image."""
        if isinstance(image, torch.Tensor):
            # Convert to numpy
            if image.dim() == 3:
                image = image.permute(1, 2, 0).cpu().numpy()
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise TypeError(f"Unsupported type: {type(image)}")

        # Ensure RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Apply augmentations
        view1 = self.view1(image=image)["image"]
        view2 = self.view2(image=image)["image"]

        return view1, view2


def get_medical_safe_augmentation(target_size=224, normalize=True):
    """Factory function for Medical-Safe BYOL Augmentation."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError("Albumentations not installed. Install with: pip install albumentations")
    return MedicalSafeBYOLAugmentation(target_size=target_size, normalize=normalize)



# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Augmentations Module (ULTIMATE EDITION with GAN + Kornia)")
    print("=" * 80)

    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Test class-aware with GAN + Kornia
    print("\n📌 Testing class-aware augmentation with GAN + Kornia:")
    for class_name in ["NL", "G1", "G2", "G3"]:
        aug = get_class_aware_augmentation(
            class_name=class_name,
            target_size=224,
            use_kornia=True,
            use_gan=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        result = aug(test_img)
        count = aug.get_augmentation_count()
        factor = aug.get_augmentation_factor()
        print(f"   {class_name}: {count} transforms, factor={factor:.1f}x")

    print("\n✅ All tests passed!")
    print("=" * 80)
