# tools/build_liver_model.py
# -*- coding: utf-8 -*-
"""
Build pre-trained liver model with generalization check.

Usage:
    python tools/build_liver_model.py

Output:
    core/security/models/liver_model.pkl
"""

import os
import sys
import random

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

os.environ["LIVER_GUARD_ALLOW_TRAIN"] = "1"

from core.security import image_guard
from core.security.image_guard import (
    _get_liver_model,
    _compute_features,
    _BUNDLE_MODEL_FILE,
    LIVER_REFERENCE_DIR,
)


def augment_image(img):
    """Augmentation ساده برای چک generalization."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    augs = []

    # 1. چرخش کوچک
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 5, 1.0)
    augs.append(cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0)))

    # 2. تغییر روشنایی
    bright = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    augs.append(bright)

    # 3. تغییر کنتراست
    dark = cv2.convertScaleAbs(img, alpha=0.9, beta=-5)
    augs.append(dark)

    # 4. blur ملایم
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    augs.append(blur)

    # 5. flip افقی
    augs.append(cv2.flip(img, 1))

    return augs


def main():
    import cv2
    import numpy as np

    print("=" * 70)
    print("  Build Pre-trained Liver Model (v3.1 — Generalization)")
    print("=" * 70)
    print(f"  Reference dir:  {LIVER_REFERENCE_DIR}")
    print(f"  Output pkl:     {_BUNDLE_MODEL_FILE}")
    print("=" * 70)
    print()

    if not os.path.isdir(LIVER_REFERENCE_DIR):
        print(f"❌ Reference dir not found: {LIVER_REFERENCE_DIR}")
        sys.exit(1)

    files = [
        f for f in os.listdir(LIVER_REFERENCE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]
    print(f"📚 Found {len(files)} reference images")
    if len(files) < 10:
        print("❌ Need at least 10 images")
        sys.exit(1)
    print()

    os.makedirs(os.path.dirname(_BUNDLE_MODEL_FILE), exist_ok=True)

    # ============================================================
    # STEP 1: Build model
    # ============================================================
    import time
    t0 = time.time()
    print("🔄 Training...")
    model = _get_liver_model()
    elapsed = time.time() - t0

    if model is None:
        print("❌ Training failed")
        sys.exit(1)

    # ============================================================
    # STEP 2: Generalization check on AUGMENTED images
    # ============================================================
    print()
    print("=" * 70)
    print("  Generalization check (augmented images)")
    print("=" * 70)

    random.seed(42)
    sample_files = random.sample(files, min(30, len(files)))
    sample_paths = [os.path.join(LIVER_REFERENCE_DIR, f) for f in sample_files]

    total_aug = 0
    passed_aug = 0
    scores = []

    for path in sample_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        for aug_img in augment_image(img):
            fvec = _compute_features(aug_img)
            if fvec is None:
                continue
            score = model.score(fvec)
            scores.append(score)
            total_aug += 1
            if score >= image_guard.REJECT_SCORE:
                passed_aug += 1

    if total_aug > 0:
        pass_rate = passed_aug / total_aug * 100
        print(f"   Tested on:      {total_aug} augmented images")
        print(f"   Pass rate:      {passed_aug}/{total_aug} ({pass_rate:.1f}%)")
        print(f"   Score range:    {min(scores):.2f} .. {max(scores):.2f}")
        print(f"   Mean score:     {np.mean(scores):.3f}")
        print()
        if pass_rate < 85:
            print(f"   ⚠️ WARNING: Augmentation pass rate < 85%")
            print(f"   Model may reject valid new images!")
        else:
            print(f"   ✅ Good generalization (≥85%)")

    # ============================================================
    # STEP 3: Original reference check (should be 100%)
    # ============================================================
    print()
    print("=" * 70)
    print("  Original reference check")
    print("=" * 70)

    orig_scores = []
    orig_passed = 0
    check_n = min(50, len(files))
    for fname in random.sample(files, check_n):
        path = os.path.join(LIVER_REFERENCE_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        fvec = _compute_features(img)
        if fvec is None:
            continue
        score = model.score(fvec)
        orig_scores.append(score)
        if score >= image_guard.REJECT_SCORE:
            orig_passed += 1

    print(f"   Tested on:      {check_n} originals")
    print(f"   Pass rate:      {orig_passed}/{check_n} "
          f"({orig_passed/max(1,check_n)*100:.1f}%)")
    print(f"   Score range:    {min(orig_scores):.2f} .. {max(orig_scores):.2f}")
    print(f"   Mean score:     {np.mean(orig_scores):.3f}")

    # ============================================================
    # STEP 4: Final report
    # ============================================================
    print()
    print("=" * 70)
    print("  ✅ Build complete (v3.1)")
    print("=" * 70)
    print(f"  Samples:            {model.n_samples}")
    print(f"  Threshold:          {model.threshold:.3f}")
    print(f"  Reject score:       {image_guard.REJECT_SCORE}")
    print(f"  PCA components:     {model.pca.n_components_}")
    print(f"  Time:               {elapsed:.1f}s")
    if os.path.exists(_BUNDLE_MODEL_FILE):
        size_kb = os.path.getsize(_BUNDLE_MODEL_FILE) / 1024
        print(f"  File:               {_BUNDLE_MODEL_FILE}")
        print(f"  Size:               {size_kb:.0f} KB")
    print()

    # ✅ هشدار نهایی
    if total_aug > 0 and passed_aug / total_aug < 0.85:
        print("⚠️  Augmentation pass rate is low.")
        print("    Consider: lower PCA_COMPONENTS, raise REJECT_SCORE, or add more diverse references.")
    else:
        print("✅ Model is ready to ship.")
    print("=" * 70)


if __name__ == "__main__":
    main()
