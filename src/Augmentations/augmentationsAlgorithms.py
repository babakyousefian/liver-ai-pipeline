"""
Liver Dataset Augmentation - PURE MEDICAL EDITION v3.2 (FIXED)

Key fixes:
- Removed UltrasoundShadow & UltrasoundAttenuation (they over-darken)
- Brightness/Contrast only LIGHTEN (not darken)
- Gamma range only LIGHTEN
- Added minimum brightness guard
- NO rotation, NO flip, NO affine
- NO CoarseDropout

Author: Adapted from v3
Version: 3.2
"""

import os
import cv2
import random
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
import torch


# =========================
# CONFIG
# =========================
INPUT_DIR = "../../finalReports/cropped/"
CSV_PATH = "../../finalReports/labels.csv"
OUTPUT_DIR = "../../finalReports/newLiver_v3/"


def resolve_path(path):
    if os.path.exists(path):
        return path
    if path.startswith("./FinalReports"):
        alt = path.replace("./FinalReports", "./finalReports", 1)
        if os.path.exists(alt):
            return alt
    if path.startswith("./finalReports"):
        alt = path.replace("./finalReports", "./FinalReports", 1)
        if os.path.exists(alt):
            return alt
    return path


INPUT_DIR = resolve_path(INPUT_DIR)
CSV_PATH = resolve_path(CSV_PATH)

print(f"\n📂 INPUT_DIR: {INPUT_DIR}")
print(f"📄 CSV_PATH:  {CSV_PATH}")

if not os.path.isdir(INPUT_DIR):
    raise FileNotFoundError(f"❌ Not found: {INPUT_DIR}")
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"❌ Not found: {CSV_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ TARGETS: balanced 1:1:1:1
TARGETS = {
    "NL": 1584,
    "G1": 1585,
    "G2": 1584,
    "G3": 1589,
}

random.seed(42)
np.random.seed(42)


# =========================
# ULTRASOUND-SPECIFIC TRANSFORMS (SAFE)
# =========================

class UltrasoundSpeckleNoise:
    """Mild multiplicative speckle noise (grayscale only)."""

    def __init__(self, severity=0.05, p=0.4):
        self.severity = severity
        self.p = p

    def __call__(self, image, **kwargs):
        if random.random() > self.p:
            return image

        # ✅ FIX: Single noise for all channels (keeps grayscale)
        h, w = image.shape[:2]
        noise_2d = np.random.normal(1.0, self.severity, (h, w))

        if image.ndim == 3:
            noise_2d = noise_2d[:, :, np.newaxis]  # (H, W, 1) → broadcast to all channels

        noisy = image.astype(np.float32) * noise_2d
        return np.clip(noisy, 0, 255).astype(np.uint8)


class BrightnessGuard:
    """
    ✅ UPDATED: Ensure image is not too dark AND preserves grayscale.
    """

    def __init__(self, min_brightness=35, p=1.0):
        self.min_brightness = min_brightness
        self.p = p

    def __call__(self, image, **kwargs):
        if random.random() > self.p:
            return image

        # Check mean brightness on grayscale version
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        mean_brightness = gray.mean()

        if mean_brightness < self.min_brightness:
            scale = self.min_brightness / (mean_brightness + 1e-6)
            scale = min(scale, 2.5)
            result = image.astype(np.float32) * scale
            result = np.clip(result, 0, 255).astype(np.uint8)

            # ✅ Ensure still grayscale after brightness boost
            if result.ndim == 3:
                gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                result = cv2.cvtColor(gray_result, cv2.COLOR_GRAY2RGB)

            return result

        return image


# =========================
# ✅ PURE MEDICAL-SAFE TRANSFORM v3.2 (FIXED)
# =========================
def get_transform(label):
    """
    Pure medical-safe augmentation v3.2 (FIXED).

    Changes vs v3.0:
    - REMOVED UltrasoundShadow (was over-darkening)
    - REMOVED UltrasoundAttenuation (was over-darkening)
    - Brightness/Contrast ONLY lightens (not darkens)
    - Gamma ONLY lightens
    - Added BrightnessGuard at the end (prevents black images)
    """
    # ✅ Class-specific strength (all positive-only for brightness)
    if label == "G3":
        brightness_max = 0.25
        contrast_max = 0.25
        gamma_range = (100, 130)  # Only lighten
        speckle_sev = 0.08
        elastic_p = 0.35
    elif label == "G2":
        brightness_max = 0.22
        contrast_max = 0.22
        gamma_range = (100, 125)
        speckle_sev = 0.06
        elastic_p = 0.30
    elif label == "G1":
        brightness_max = 0.20
        contrast_max = 0.20
        gamma_range = (100, 120)
        speckle_sev = 0.05
        elastic_p = 0.25
    else:  # NL
        brightness_max = 0.15
        contrast_max = 0.15
        gamma_range = (100, 115)
        speckle_sev = 0.04
        elastic_p = 0.15

    return A.Compose([
        # ==========================================
        # ✅ 1. INTENSITY — ONLY LIGHTEN (no darkening)
        # ==========================================
        A.RandomBrightnessContrast(
            brightness_limit=(0.0, brightness_max),   # ← ONLY POSITIVE (lighten)
            contrast_limit=(-0.05, contrast_max),     # ← mostly positive
            p=0.85,
        ),
        A.RandomGamma(
            gamma_limit=gamma_range,                   # ← ONLY > 100 (lighten)
            p=0.6,
        ),

        # ==========================================
        # ✅ 2. ULTRASOUND SPECKLE (safe)
        # ==========================================
        A.Lambda(
            name="speckle",
            image=UltrasoundSpeckleNoise(severity=speckle_sev, p=1.0),
            p=0.5,
        ),

        # ==========================================
        # ✅ 3. MILD MORPHOLOGY (no rotation/flip)
        # ==========================================
        A.ElasticTransform(
            alpha=15,
            sigma=5,
            p=elastic_p,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.08,
            p=0.25,
        ),

        # ==========================================
        # ✅ 4. SUBTLE BLUR (motion artifact)
        # ==========================================
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.3, 0.8)),
            A.MedianBlur(blur_limit=3),
        ], p=0.15),

        # ==========================================
        # ✅ 5. BRIGHTNESS GUARD (prevents black images)
        # ==========================================
        A.Lambda(
            name="brightness_guard",
            image=BrightnessGuard(min_brightness=25, p=1.0),
            p=1.0,  # Always apply
        ),
    ])


# =========================
# SAVE
# =========================
def save(img, label):
    # ✅ FIX: Force grayscale (medical ultrasound is grayscale)
    if img.ndim == 3:
        # Convert RGB → grayscale → back to 3-channel (keeps compatibility)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    name = f"{label}_{uuid.uuid4().hex[:8]}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv(CSV_PATH)
    data = {"NL": [], "G1": [], "G2": [], "G3": []}
    for _, row in df.iterrows():
        label = row["liverLabel"]
        fname = row["liverName"]
        if label in data:
            data[label].append(fname)
    return data


# =========================
# MAIN
# =========================
def generate():
    data = load_data()

    print(f"\n{'=' * 60}")
    print(f"📦 Building image cache...")
    print(f"{'=' * 60}")

    cache = {}
    for lbl, files in data.items():
        cache[lbl] = []
        for f in tqdm(files, desc=f"Caching {lbl}", ncols=100):
            img = cv2.imread(os.path.join(INPUT_DIR, f))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cache[lbl].append(img)

    print(f"\n📊 Cache loaded:")
    for lbl, imgs in cache.items():
        print(f"   - {lbl}: {len(imgs)} images")

    for label, files in data.items():
        print(f"\n{'=' * 60}")
        print(f"🚀 Processing: {label}")
        print(f"{'=' * 60}")

        transform = get_transform(label)

        current = len(files)
        target = TARGETS[label]
        needed = target - current

        print(f"   Current: {current}")
        print(f"   Target:  {target}")
        print(f"   Needed:  {needed}")

        # ✅ Save originals
        print(f"\n   💾 Saving {current} originals...")
        for f in tqdm(files, desc=f"Originals {label}", ncols=100):
            img = cv2.imread(os.path.join(INPUT_DIR, f))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                save(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), label)

        if needed <= 0:
            print(f"   ✅ No augmentation needed")
            continue

        if len(cache[label]) == 0:
            print(f"   ⚠️ Empty cache, skipping")
            continue

        # ✅ Augment
        print(f"\n   🔄 Augmenting {needed} images...")

        for _ in tqdm(range(needed), desc=f"Augmenting {label}", ncols=100):
            img1 = random.choice(cache[label])

            aug = transform(image=img1)["image"]
            aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            save(aug_bgr, label)

        print(f"   ✅ Done: {label}")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print(f"🧬 Liver Augmentation - PURE MEDICAL EDITION v3.2 (FIXED)")
    print(f"{'=' * 60}")
    print(f"📂 Input:  {INPUT_DIR}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print(f"✅ NO rotation, flip, or affine!")
    print(f"✅ NO black squares (CoarseDropout removed)")
    print(f"✅ Brightness ONLY lightens (no darkening)")
    print(f"✅ BrightnessGuard prevents black images")
    print(f"{'=' * 60}")

    generate()

    # Report
    print(f"\n{'=' * 60}")
    print(f"✅ AUGMENTATION COMPLETE!")
    print(f"{'=' * 60}")

    output_files = os.listdir(OUTPUT_DIR)
    output_images = [f for f in output_files if f.lower().endswith(".png")]

    # ✅ NEW: Verify grayscale
    print(f"\n   🔍 Verifying grayscale...")
    non_gray_count = 0
    for f in tqdm(output_images[:200], desc="Checking grayscale", ncols=100):
        img = cv2.imread(os.path.join(OUTPUT_DIR, f))
        if img is not None:
            # Check if R=G=B (grayscale)
            b, g, r = cv2.split(img)
            if not (np.allclose(b, g, atol=2) and np.allclose(g, r, atol=2)):
                non_gray_count += 1

    print(f"   Found {non_gray_count} non-grayscale images out of 200 checked")
    if non_gray_count == 0:
        print(f"   ✅ All images are grayscale!")
    else:
        print(f"   ⚠️ {non_gray_count} images have color artifacts")

    from collections import Counter
    class_counts = Counter()
    for f in output_images:
        if f.startswith("NL_"):
            class_counts["NL"] += 1
        elif f.startswith("G1_"):
            class_counts["G1"] += 1
        elif f.startswith("G2_"):
            class_counts["G2"] += 1
        elif f.startswith("G3_"):
            class_counts["G3"] += 1

    print(f"\n   Class Distribution:")
    total = len(output_images)
    for cls in ["NL", "G1", "G2", "G3"]:
        count = class_counts.get(cls, 0)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"      - {cls}: {count:,} ({pct:.2f}%)")

    # ✅ NEW: Check for black images
    print(f"\n   🔍 Checking for black images...")
    black_count = 0
    for f in tqdm(output_images[:200], desc="Checking brightness", ncols=100):
        img = cv2.imread(os.path.join(OUTPUT_DIR, f))
        if img is not None:
            mean_brightness = img.mean()
            if mean_brightness < 15:  # too dark
                black_count += 1

    print(f"   Found {black_count} black images out of 200 checked")

    print(f"\n📂 Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"🎉 DONE!\n")