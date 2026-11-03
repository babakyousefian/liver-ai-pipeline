# src/self_supervised/fid_score.py

"""
FID (Fréchet Inception Distance) Score Calculator (ULTIMATE EDITION)

Computes FID between real and synthetic images to evaluate GAN quality.

Features:
- Uses InceptionV3 pretrained on ImageNet
- Handles both RGB and grayscale ultrasound images
- Supports batches to avoid memory issues
- Robust to different image sizes
- Fast and memory-efficient

Interpretation:
- FID < 10: excellent
- FID 10-30: good
- FID 30-50: acceptable
- FID 50-100: poor
- FID > 100: bad

Author: Babak Yousefian
Version: 1.0.0
"""

import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ================================================================
# ✅ INCEPTION V3 FEATURE EXTRACTOR
# ================================================================

class InceptionFeatureExtractor(nn.Module):
    """
    Extract 2048-dim features from images using InceptionV3.

    Removes the final classification layer and uses the
    global average pooling output.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

        # ✅ Load InceptionV3 with ImageNet weights
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, aux_logits=True)
        model.eval()
        model.to(device)

        # ✅ Remove final FC layer (keep up to avgpool)
        # InceptionV3 has: Conv → ... → avgpool → dropout → fc
        # We want the output of avgpool (2048-dim)
        self.features = nn.Sequential(
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            model.maxpool1,
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            model.maxpool2,
            model.Mixed_5b,
            model.Mixed_5c,
            model.Mixed_5d,
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.0)

        # ✅ Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 2048-dim features from batch of images.

        Args:
            x: (B, 3, H, W), values in [0, 1] or normalized

        Returns:
            (B, 2048) feature tensor
        """
        # ✅ InceptionV3 expects 299×299
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # ✅ Convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # ✅ ImageNet normalization (InceptionV3 was trained with this)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # ✅ Forward pass
        with torch.no_grad():
            features = self.features(x)
            features = self.avgpool(features)
            features = features.view(features.size(0), -1)

        return features


# ================================================================
# ✅ SIMPLE DATASET
# ================================================================

class ImagePathDataset(Dataset):
    """Dataset that loads images from paths."""

    def __init__(self, image_paths: List[str], target_size: int = 299):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),  # → [0, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


class ImageTensorDataset(Dataset):
    """Dataset that wraps pre-loaded numpy arrays."""

    def __init__(self, images: Union[np.ndarray, List[np.ndarray]], target_size: int = 299):
        self.images = images
        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            # Ensure uint8 or float
            if img.dtype != np.uint8 and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img.astype(np.uint8))
        elif isinstance(img, Image.Image):
            pass
        else:
            raise TypeError(f"Unsupported type: {type(img)}")

        transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
        ])
        return transform(img)


# ================================================================
# ✅ FID CALCULATION
# ================================================================

def _compute_statistics(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of features.

    Args:
        features: (N, D) array

    Returns:
        mu: (D,) mean
        sigma: (D, D) covariance
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _compute_fid(
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
    mu_fake: np.ndarray,
    sigma_fake: np.ndarray,
) -> float:
    """
    Compute FID between two Gaussian distributions.

    FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2√(Σ_real · Σ_fake))
    """
    diff = mu_real - mu_fake

    # ✅ Compute sqrt of product of covariances
    # Use scipy.linalg.sqrtm for numerical stability
    # ✅ FIX: scipy 1.18+ removed `disp` argument from sqrtm()
    try:
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    except TypeError:
        # Fallback for newer scipy versions
        covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))

    # ✅ Handle complex numbers (should be real but numerical errors happen)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # If imaginary part is too large, FID is unreliable
            if m > 1e-2:
                print(f"⚠️ Warning: large imaginary component in FID: {m}")
        covmean = covmean.real

    # ✅ Compute FID
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)

    return float(fid)


def _extract_features(
    dataloader: DataLoader,
    extractor: InceptionFeatureExtractor,
    device: str,
    desc: str = "Extracting features",
) -> np.ndarray:
    """Extract features from a DataLoader."""
    all_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            features = extractor(batch)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


# ================================================================
# ✅ MAIN FID FUNCTION (Paths)
# ================================================================

def calculate_fid_from_paths(
    real_paths: List[str],
    fake_paths: List[str],
    device: str = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
    verbose: bool = True,
) -> float:
    """
    Compute FID between two sets of image paths.

    Args:
        real_paths: List of real image paths
        fake_paths: List of synthetic image paths
        device: 'cpu' or 'cuda'
        batch_size: Batch size for feature extraction
        num_workers: Number of workers
        verbose: Print progress

    Returns:
        FID score (lower is better)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("📊 Computing FID Score")
        print("=" * 70)
        print(f"   Real images: {len(real_paths)}")
        print(f"   Fake images: {len(fake_paths)}")
        print(f"   Device: {device}")
        print("=" * 70)

    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise ValueError("Empty image list")

    # ✅ Build feature extractor
    extractor = InceptionFeatureExtractor(device=device)

    # ✅ Real features
    real_dataset = ImagePathDataset(real_paths)
    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    real_features = _extract_features(real_loader, extractor, device, "Real features")

    # ✅ Fake features
    fake_dataset = ImagePathDataset(fake_paths)
    fake_loader = DataLoader(
        fake_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    fake_features = _extract_features(fake_loader, extractor, device, "Fake features")

    # ✅ Compute statistics
    mu_real, sigma_real = _compute_statistics(real_features)
    mu_fake, sigma_fake = _compute_statistics(fake_features)

    # ✅ FID
    fid = _compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)

    if verbose:
        print("\n" + "=" * 70)
        print(f"📊 FID Score: {fid:.4f}")
        print("=" * 70)
        if fid < 10:
            print("   🟢 Excellent — synthetic images are very realistic")
        elif fid < 30:
            print("   🟢 Good — high quality")
        elif fid < 50:
            print("   🟡 Acceptable")
        elif fid < 100:
            print("   🟠 Poor — consider improving GAN")
        else:
            print("   🔴 Bad — GAN didn't learn well")
        print("=" * 70)

    return fid


# ================================================================
# ✅ MAIN FID FUNCTION (Arrays)
# ================================================================

def calculate_fid_from_arrays(
    real_images: Union[np.ndarray, List[np.ndarray]],
    fake_images: Union[np.ndarray, List[np.ndarray]],
    device: str = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
    verbose: bool = True,
) -> float:
    """
    Compute FID between two numpy arrays (or lists) of images.

    Args:
        real_images: (N, H, W, C) numpy array or list
        fake_images: (M, H, W, C) numpy array or list
        device: 'cpu' or 'cuda'
        batch_size: Batch size
        num_workers: Number of workers
        verbose: Print progress

    Returns:
        FID score
    """
    if verbose:
        print("\n" + "=" * 70)
        print("📊 Computing FID Score (from arrays)")
        print("=" * 70)
        print(f"   Real images: {len(real_images)}")
        print(f"   Fake images: {len(fake_images)}")
        print(f"   Device: {device}")
        print("=" * 70)

    # ✅ Build extractor
    extractor = InceptionFeatureExtractor(device=device)

    # ✅ Real
    real_dataset = ImageTensorDataset(real_images)
    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    real_features = _extract_features(real_loader, extractor, device, "Real features")

    # ✅ Fake
    fake_dataset = ImageTensorDataset(fake_images)
    fake_loader = DataLoader(
        fake_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    fake_features = _extract_features(fake_loader, extractor, device, "Fake features")

    # ✅ Statistics
    mu_real, sigma_real = _compute_statistics(real_features)
    mu_fake, sigma_fake = _compute_statistics(fake_features)

    # ✅ FID
    fid = _compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)

    if verbose:
        print("\n" + "=" * 70)
        print(f"📊 FID Score: {fid:.4f}")
        print("=" * 70)
        if fid < 10:
            print("   🟢 Excellent")
        elif fid < 30:
            print("   🟢 Good")
        elif fid < 50:
            print("   🟡 Acceptable")
        elif fid < 100:
            print("   🟠 Poor")
        else:
            print("   🔴 Bad")
        print("=" * 70)

    return fid


# ================================================================
# ✅ UTILITY: Split train/test real images
# ================================================================

def compute_fid_for_gan(
    real_dir: str,
    synthetic_dir: str,
    device: str = "cpu",
    max_real: int = 500,
    max_fake: int = 500,
    batch_size: int = 16,
    verbose: bool = True,
) -> float:
    """
    Compute FID between a folder of real images and a folder of synthetic images.

    Args:
        real_dir: Path to directory containing real images
        synthetic_dir: Path to directory containing synthetic images
        device: 'cpu' or 'cuda'
        max_real: Max number of real images to use (to save memory)
        max_fake: Max number of synthetic images to use
        batch_size: Batch size
        verbose: Print progress

    Returns:
        FID score
    """
    # ✅ Collect real paths
    real_exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".bmp"}
    real_paths = [
        os.path.join(real_dir, f) for f in os.listdir(real_dir)
        if os.path.splitext(f)[1] in real_exts
    ]
    real_paths = real_paths[:max_real]

    # ✅ Collect synthetic paths
    fake_paths = [
        os.path.join(synthetic_dir, f) for f in os.listdir(synthetic_dir)
        if os.path.splitext(f)[1] in real_exts
    ]
    fake_paths = fake_paths[:max_fake]

    if len(real_paths) == 0:
        raise ValueError(f"No real images found in {real_dir}")
    if len(fake_paths) == 0:
        raise ValueError(f"No synthetic images found in {synthetic_dir}")

    return calculate_fid_from_paths(
        real_paths=real_paths,
        fake_paths=fake_paths,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
    )


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Testing FID Score Calculator")
    print("=" * 70)

    # Create dummy images
    real_imgs = np.random.randint(0, 255, (20, 224, 224, 3), dtype=np.uint8)
    fake_imgs = np.random.randint(0, 255, (20, 224, 224, 3), dtype=np.uint8)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fid = calculate_fid_from_arrays(
        real_images=real_imgs,
        fake_images=fake_imgs,
        device=device,
        batch_size=8,
        verbose=True,
    )

    print(f"\n✅ FID test passed: {fid:.4f}")