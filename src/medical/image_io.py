# src/medical/image_io.py

import os
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# ================================================================
# ✅ MEDPY - Ultrasound Image Processing
# ================================================================

try:
    import medpy
    from medpy import io
    from medpy.filter import smoothing

    MEDPY_AVAILABLE = True
except ImportError:
    MEDPY_AVAILABLE = False
    print("⚠️ MedPy not installed. Install with: pip install medpy")


def enhance_ultrasound(image: np.ndarray) -> np.ndarray:
    """Enhance ultrasound image using MedPy."""
    if image is None or image.size == 0:
        raise ValueError("Image is None or empty!")
    if not MEDPY_AVAILABLE:
        print("⚠️ MedPy not available, returning original")
        return image

    try:
        # Apply anisotropic diffusion for noise reduction
        filtered = medpy.filter.smoothing.anisotropic_diffusion(
            image.astype(np.float32), niter=5, kappa=50, gamma=0.1
        )
        return np.clip(filtered, 0, 255).astype(np.uint8)
    except:
        return image


# ================================================================
# ✅ ITK - Medical Image I/O
# ================================================================

try:
    import itk

    ITK_AVAILABLE = True
except ImportError:
    ITK_AVAILABLE = False
    print("⚠️ ITK not installed. Install with: pip install itk")


def read_image_itk(filepath: str):
    """Read medical image using ITK."""
    if not ITK_AVAILABLE:
        return None

    try:
        img = itk.imread(filepath)
        return img
    except:
        return None


# ================================================================
# ✅ PYDICOM - DICOM Files
# ================================================================

try:
    import pydicom
    from pydicom import dcmread

    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("⚠️ PyDICOM not installed. Install with: pip install pydicom")


def read_dicom(filepath: str):
    """Read DICOM file."""
    if not PYDICOM_AVAILABLE:
        return None

    try:
        ds = dcmread(filepath)
        return ds
    except:
        return None


def dicom_to_numpy(ds, normalize=True) -> np.ndarray:
    """Convert DICOM to numpy array."""
    if not PYDICOM_AVAILABLE:
        return None

    try:
        if ds is None:
            raise ValueError("DICOM dataset is None!")
        img = ds.pixel_array.astype(np.float32)
        if normalize:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
        return img
    except:
        return None
