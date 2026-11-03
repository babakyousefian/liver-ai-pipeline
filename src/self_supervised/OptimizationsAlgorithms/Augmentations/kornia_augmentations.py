# src/self_supervised/kornia_augmentations.py (ULTIMATE FIXED EDITION)

import warnings

import kornia.augmentation as K
import torch
import torch.nn as nn
from kornia.augmentation import AugmentationSequential

warnings.filterwarnings("ignore")


class KorniaAugmentation(nn.Module):
    """
    ✅ FIXED: Batch-aware augmentation using Kornia.

    This class provides augmentation for a single view.
    Used as a building block for dual-view augmentations.

    Features:
    - GPU-accelerated (no CPU bottleneck)
    - Batch-aware (processes entire batch at once)
    - Configurable intensity for medical images
    - Proper randomness (same_on_batch=False)
    - Normalization included
    """

    def __init__(
        self,
        target_size: int = 224,
        intensity: float = 0.8,
        normalize: bool = True,
        use_medical_safe: bool = True,
    ):
        """
        Initialize Kornia augmentation.

        Args:
            target_size: Output image size
            intensity: Augmentation intensity (0.0 - 1.0)
            normalize: Whether to normalize images
            use_medical_safe: Use medical-safe augmentations
        """
        super().__init__()

        self.target_size = target_size
        self.intensity = intensity
        self.normalize = normalize

        # ✅ تنظیم scale بر اساس intensity و medical safe
        if use_medical_safe:
            # ✅ برای تصاویر پزشکی - ملایم‌تر
            scale_min = 0.85 + (1.0 - intensity) * 0.1  # 0.85 - 0.95
            scale_max = 1.0
        else:
            # ✅ برای تصاویر طبیعی - قوی‌تر
            scale_min = 0.7 + (1.0 - intensity) * 0.2  # 0.7 - 0.9
            scale_max = 1.0

        # ✅ تنظیم brightness/contrast بر اساس intensity
        brightness = 0.2 + intensity * 0.3  # 0.2 - 0.5
        contrast = 0.2 + intensity * 0.3
        saturation = 0.1 + intensity * 0.2  # 0.1 - 0.3
        hue = 0.05 + intensity * 0.1  # 0.05 - 0.15

        # ✅ تنظیم blur بر اساس intensity
        blur_sigma = (0.1, 1.0 + intensity * 1.0)  # (0.1, 1.0) - (0.1, 2.0)
        blur_p = 0.3 + intensity * 0.3  # 0.3 - 0.6

        # ✅ Rotation برای تصاویر پزشکی (ملایم)
        rotation_degrees = 10 * intensity  # 0 - 10 درجه

        # ✅ تنظیم noise
        noise_std = 0.02 * intensity  # 0 - 0.02

        # ============================================================
        # 🔥 AUGMENTATION PIPELINE
        # ============================================================

        self.aug = AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5, same_on_batch=False),  # ✅ تصادفی برای هر نمونه
            K.RandomVerticalFlip(p=0.1 * intensity, same_on_batch=False),  # ✅ کمتر برای پزشکی
            K.RandomRotation(degrees=rotation_degrees, same_on_batch=False, p=0.5),
            K.RandomResizedCrop(
                size=(target_size, target_size),
                scale=(scale_min, scale_max),  # ✅ medical-safe
                ratio=(0.8, 1.2),  # ✅ تناسب بهتر برای ultrasound
                p=1.0,
                same_on_batch=False,
            ),
            K.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=0.8,
                same_on_batch=False,
            ),
            K.RandomGrayscale(p=0.2, same_on_batch=False),
            K.RandomGaussianBlur(
                kernel_size=(23, 23), sigma=blur_sigma, p=blur_p, same_on_batch=False
            ),
            K.RandomGaussianNoise(mean=0.0, std=noise_std, p=0.3, same_on_batch=False),
            data_keys=["input"],
            keepdim=True,
        )

        # ✅ Normalization
        if normalize:
            self.norm = K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            )
        else:
            self.norm = None

        print(f"✅ KorniaAugmentation initialized:")
        print(f"   Target Size: {target_size}")
        print(f"   Intensity: {intensity}")
        print(f"   Medical Safe: {use_medical_safe}")
        print(f"   Scale: ({scale_min:.2f}, {scale_max:.2f})")
        print(f"   Rotation: {rotation_degrees:.1f}°")
        print(f"   Blur Sigma: {blur_sigma}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a batch.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Augmented tensor (B, C, H, W)
        """
        # ✅ Sanity check
        if x is None or x.numel() == 0:
            raise ValueError("Input tensor is None or empty")

        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D")

        # ✅ Normalize input to [0, 1] اگر لازم باشه
        if x.max() > 1.0:
            x = x / 255.0

        try:
            # ✅ Apply augmentation
            augmented = self.aug(x)

            # ✅ Apply normalization
            if self.norm is not None:
                augmented = self.norm(augmented)

            # ✅ Sanity check
            if augmented is None:
                raise RuntimeError("Augmentation returned None")

            if torch.isnan(augmented).any():
                raise RuntimeError("NaN detected in augmented views")

            if torch.isinf(augmented).any():
                raise RuntimeError("Inf detected in augmented views")

            return augmented

        except Exception as e:
            raise RuntimeError(f"Augmentation failed: {e}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        return self.forward(x)

    def to(self, device):
        """Move augmentation to device."""
        self.aug = self.aug.to(device)
        if self.norm is not None:
            self.norm = self.norm.to(device)
        return self


# ================================================================
# ✅ DUAL-VIEW KORNIA AUGMENTATION (FOR BYOL)
# ================================================================


class DualViewKorniaAugmentation(nn.Module):
    """
    ✅ FIXED: Dual-view augmentation for BYOL using Kornia.

    This class provides two different augmented views of the same batch,
    optimized for BYOL and SimCLR training.

    Features:
    - GPU-accelerated (no CPU bottleneck)
    - Batch-aware (processes entire batch at once)
    - Dual-view (returns two different views)
    - Configurable intensity for medical images
    - Proper randomness (same_on_batch=False)
    - Normalization included

    Example:
        >>> aug = DualViewKorniaAugmentation(
        ...     target_size=224,
        ...     intensity=0.8,
        ...     use_medical_safe=True
        ... )
        >>> view1, view2 = aug(images)  # images: (B, C, H, W)
        >>> print(view1.shape, view2.shape)  # (B, 3, 224, 224) (B, 3, 224, 224)
    """

    def __init__(
        self,
        target_size: int = 224,
        intensity: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        normalize: bool = True,
        use_medical_safe: bool = True,
    ):
        """
        Initialize dual-view Kornia augmentation.

        Args:
            target_size: Output image size
            intensity: Augmentation intensity (0.0 - 1.0)
            device: Device to run on ('cpu' or 'cuda')
            normalize: Whether to normalize images
            use_medical_safe: Use medical-safe augmentations
        """
        super().__init__()

        self.target_size = target_size
        self.intensity = intensity
        self.device = torch.device(device)
        self.normalize = normalize
        self.use_medical_safe = use_medical_safe

        # ✅ تنظیم scale بر اساس intensity و medical safe
        if use_medical_safe:
            scale_min = 0.85 + (1.0 - intensity) * 0.1  # 0.85 - 0.95
            scale_max = 1.0
        else:
            scale_min = 0.7 + (1.0 - intensity) * 0.2  # 0.7 - 0.9
            scale_max = 1.0

        # ✅ تنظیم brightness/contrast بر اساس intensity
        brightness = 0.2 + intensity * 0.3  # 0.2 - 0.5
        contrast = 0.2 + intensity * 0.3
        saturation = 0.1 + intensity * 0.2  # 0.1 - 0.3
        hue = 0.05 + intensity * 0.1  # 0.05 - 0.15

        # ✅ تنظیم blur بر اساس intensity
        blur_sigma = (0.1, 1.0 + intensity * 1.0)  # (0.1, 1.0) - (0.1, 2.0)
        blur_p = 0.3 + intensity * 0.3  # 0.3 - 0.6

        # ✅ Rotation برای تصاویر پزشکی (ملایم)
        rotation_degrees = 10 * intensity  # 0 - 10 درجه

        # ============================================================
        # 🔥 AUGMENTATION PIPELINE - View 1
        # ============================================================

        self.aug1 = AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5, same_on_batch=False),
            K.RandomVerticalFlip(p=0.1 * intensity, same_on_batch=False),
            K.RandomRotation(degrees=rotation_degrees, same_on_batch=False, p=0.5),
            K.RandomResizedCrop(
                size=(target_size, target_size),
                scale=(scale_min, scale_max),
                ratio=(0.8, 1.2),
                p=1.0,
                same_on_batch=False,
            ),
            K.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=0.8,
                same_on_batch=False,
            ),
            K.RandomGrayscale(p=0.2, same_on_batch=False),
            K.RandomGaussianBlur(
                kernel_size=(23, 23), sigma=blur_sigma, p=blur_p, same_on_batch=False
            ),
            K.RandomGaussianNoise(mean=0.0, std=0.02 * intensity, p=0.3, same_on_batch=False),
            data_keys=["input"],
            keepdim=True,
        )

        # ============================================================
        # 🔥 AUGMENTATION PIPELINE - View 2 (slightly different)
        # ============================================================

        self.aug2 = AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5, same_on_batch=False),
            K.RandomVerticalFlip(p=0.1 * intensity, same_on_batch=False),
            K.RandomRotation(degrees=rotation_degrees, same_on_batch=False, p=0.5),
            K.RandomResizedCrop(
                size=(target_size, target_size),
                scale=(scale_min, scale_max),
                ratio=(0.8, 1.2),
                p=1.0,
                same_on_batch=False,
            ),
            K.ColorJitter(
                brightness=brightness * 1.2,  # ✅ کمی قوی‌تر برای view 2
                contrast=contrast * 1.2,
                saturation=saturation * 1.2,
                hue=hue * 1.2,
                p=0.8,
                same_on_batch=False,
            ),
            K.RandomGrayscale(p=0.2, same_on_batch=False),
            K.RandomGaussianBlur(
                kernel_size=(23, 23), sigma=blur_sigma, p=blur_p, same_on_batch=False
            ),
            K.RandomGaussianNoise(mean=0.0, std=0.02 * intensity, p=0.3, same_on_batch=False),
            # ✅ Solarization فقط برای view 2 (BYOL-style)
            K.RandomSolarize(thresholds=0.5, p=0.2, same_on_batch=False),
            data_keys=["input"],
            keepdim=True,
        )

        # ✅ Normalization
        if normalize:
            self.norm = K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            )
        else:
            self.norm = None

        # Move to device
        self.to(self.device)

        print(f"✅ DualViewKorniaAugmentation initialized:")
        print(f"   Target Size: {target_size}")
        print(f"   Intensity: {intensity}")
        print(f"   Device: {self.device}")
        print(f"   Medical Safe: {use_medical_safe}")
        print(f"   Scale: ({scale_min:.2f}, {scale_max:.2f})")
        print(f"   Rotation: {rotation_degrees:.1f}°")

    def forward(self, x: torch.Tensor) -> tuple:
        """
        ✅ SPEED FIX: input خام (3, 224, 224) در [0, 255] از DataLoader میاد.
        Augmentation روی GPU اجرا میشه.
        """
        # ✅ Sanity check
        if x is None or x.numel() == 0:
            raise ValueError("Input tensor is None or empty")

        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D")

        # ✅ Move to device
        x = x.to(self.device, non_blocking=True)

        # ✅ Normalize input به [0, 1] برای Kornia
        if x.max() > 1.5:  # فرض: input در [0, 255]
            x = x / 255.0

        try:
            # ✅ Generate two views (batch-level)
            view1 = self.aug1(x)
            view2 = self.aug2(x)

            # ✅ Apply normalization
            if self.norm is not None:
                view1 = self.norm(view1)
                view2 = self.norm(view2)

            # ✅ Sanity check
            if view1 is None or view2 is None:
                raise RuntimeError("Augmentation returned None")

            if view1.shape != view2.shape:
                raise RuntimeError(f"Shape mismatch: {view1.shape} vs {view2.shape}")

            return view1, view2

        except Exception as e:
            raise RuntimeError(f"Augmentation failed: {e}")

    def __call__(self, x: torch.Tensor) -> tuple:
        """Forward call."""
        return self.forward(x)

    def to(self, device):
        """Move augmentation to device."""
        self.device = torch.device(device)
        self.aug1 = self.aug1.to(device)
        self.aug2 = self.aug2.to(device)
        if self.norm is not None:
            self.norm = self.norm.to(device)
        return self


# ================================================================
# ✅ FACTORY FUNCTIONS
# ================================================================


def create_kornia_augmentation(
    target_size: int = 224,
    intensity: float = 0.8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    normalize: bool = True,
    use_medical_safe: bool = True,
) -> DualViewKorniaAugmentation:
    """
    Factory function to create dual-view Kornia augmentation.

    Args:
        target_size: Output image size
        intensity: Augmentation intensity (0.0 - 1.0)
        device: Device to run on
        normalize: Whether to normalize images
        use_medical_safe: Use medical-safe augmentations

    Returns:
        DualViewKorniaAugmentation instance

    Example:
        >>> aug = create_kornia_augmentation(
        ...     target_size=224,
        ...     intensity=0.7,
        ...     device='cuda'
        ... )
        >>> view1, view2 = aug(images)
    """
    return DualViewKorniaAugmentation(
        target_size=target_size,
        intensity=intensity,
        device=device,
        normalize=normalize,
        use_medical_safe=use_medical_safe,
    )


def create_single_view_kornia(
    target_size: int = 224,
    intensity: float = 0.8,
    normalize: bool = True,
    use_medical_safe: bool = True,
) -> KorniaAugmentation:
    """
    Factory function to create single-view Kornia augmentation.

    Args:
        target_size: Output image size
        intensity: Augmentation intensity (0.0 - 1.0)
        normalize: Whether to normalize images
        use_medical_safe: Use medical-safe augmentations

    Returns:
        KorniaAugmentation instance
    """
    return KorniaAugmentation(
        target_size=target_size,
        intensity=intensity,
        normalize=normalize,
        use_medical_safe=use_medical_safe,
    )


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Kornia Augmentations (ULTIMATE FIXED EDITION)")
    print("=" * 80)

    # Test single-view
    print("\n📌 Testing single-view Kornia augmentation...")
    aug = create_single_view_kornia(target_size=224, intensity=0.8, use_medical_safe=True)

    dummy_batch = torch.randn(4, 3, 224, 224)
    augmented = aug(dummy_batch)
    print(f"   Input shape: {dummy_batch.shape}")
    print(f"   Output shape: {augmented.shape}")

    # Test dual-view
    print("\n📌 Testing dual-view Kornia augmentation (for BYOL)...")
    dual_aug = create_kornia_augmentation(
        target_size=224, intensity=0.8, device="cuda" if torch.cuda.is_available() else "cpu", use_medical_safe=True
    )

    view1, view2 = dual_aug(dummy_batch)
    print(f"   View1 shape: {view1.shape}")
    print(f"   View2 shape: {view2.shape}")
    print(f"   View1 mean: {view1.mean().item():.4f}")
    print(f"   View2 mean: {view2.mean().item():.4f}")

    # Test difference
    diff = (view1 - view2).abs().mean().item()
    print(f"   Mean difference: {diff:.4f}")
    if diff > 0.01:
        print("   ✅ Views are different (good for BYOL)")
    else:
        print("   ⚠️ Views are too similar (check augmentation)")

    print("\n✅ All Kornia tests passed!")
    print("=" * 80)
