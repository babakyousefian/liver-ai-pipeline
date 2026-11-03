# src/self_supervised/imgaug_pipeline.py

"""
imgaug Pipeline for Advanced Augmentations (OPTIONAL)
If imgaug is not available, silently use fallback without errors.
"""

import warnings

warnings.filterwarnings("ignore")

# ================================================================
# ✅ TRY TO IMPORT IMGAUG (SILENT FAIL)
# ================================================================

IMGAUG_AVAILABLE = False

try:
    import imgaug as ia
    import imgaug.augmenters as iaa
    import numpy as np

    # ✅ تست واقعی
    try:
        test_img = np.zeros((10, 10, 3), dtype=np.uint8)
        seq = iaa.Fliplr(0.5)
        _ = seq(image=test_img)
        IMGAUG_AVAILABLE = True
        # ✅ خاموش کردن پیام (برای تمیزی)
        # print("✅ imgaug available")
    except:
        IMGAUG_AVAILABLE = False
except:
    IMGAUG_AVAILABLE = False


def get_imgaug_pipeline(class_name: str = "G3") -> object:
    """Get imgaug pipeline or None if not available."""
    if not IMGAUG_AVAILABLE:
        return None

    try:
        # Intensity based on class
        if class_name == "G3":
            intensity = 1.0
        elif class_name == "G2":
            intensity = 0.8
        elif class_name == "G1":
            intensity = 0.6
        else:
            intensity = 0.3

        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.1 * intensity),
                iaa.Affine(
                    rotate=(-20 * intensity, 20 * intensity),
                    scale=(0.8, 1.2),
                    translate_percent=(-0.1, 0.1),
                    mode="reflect",
                ),
                iaa.Multiply((0.7 * intensity, 1.3 * intensity)),
                iaa.LinearContrast((0.7 * intensity, 1.3 * intensity)),
                iaa.GaussianBlur(sigma=(0.0, 2.0 * intensity)),
                iaa.AdditiveGaussianNoise(scale=(0, 20 * intensity)),
                iaa.Dropout(p=(0.0, 0.05 * intensity)),
            ],
            random_order=True,
        )

        return seq

    except Exception as e:
        return None


class DummyAugmenter:
    """Dummy augmenter for when imgaug is not available."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image, *args, **kwargs):
        return image
