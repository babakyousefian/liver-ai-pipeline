# tools/test_no_label_case.py
# -*- coding: utf-8 -*-
"""
Test the preprocessor on a NO-LABEL image.

Verifies:
    1. Labels found = 0 (or only header text, not tissue)
    2. Tissue area is IDENTICAL before/after (no damage)
    3. Only crop happened

Usage:
    python tools/test_no_label_case.py path/to/no_label_image.png
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
from core.preprocessing import preprocess_medical_image, PreprocessorConfig
from core.preprocessing.medical_preprocessor import (
    _detect_fan_region,
    _build_colored_label_mask,
    _build_white_text_mask,
    _detect_and_inpaint_labels,
)


def compare_tissue(before: np.ndarray, after: np.ndarray, name: str):
    """مقایسه‌ی بافت تصویر قبل و بعد."""
    if before.shape != after.shape:
        # resize to compare
        after_r = cv2.resize(after, (before.shape[1], before.shape[0]))
    else:
        after_r = after

    diff = cv2.absdiff(before, after_r)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    print(f"\n   {name}:")
    print(f"      Shape before: {before.shape}")
    print(f"      Shape after:  {after.shape}")
    print(f"      Mean diff:    {diff_gray.mean():.2f}")
    print(f"      Max diff:     {diff_gray.max()}")
    print(f"      >10 pixels:   {(diff_gray > 10).sum()}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/test_no_label_case.py image.png")
        sys.exit(1)

    path = sys.argv[1]

    print("=" * 70)
    print("  No-Label Image Test")
    print("=" * 70)
    print(f"  Input: {path}")

    if not os.path.isfile(path):
        print(f"   ❌ Not found")
        sys.exit(1)

    # ---------- 1. تشخیص خودکار ----------
    img = cv2.imread(path)
    print(f"\n   Original: {img.shape}")

    # چک label
    colored_mask = _build_colored_label_mask(img, PreprocessorConfig())
    white_mask = _build_white_text_mask(img, PreprocessorConfig())
    combined = cv2.bitwise_or(colored_mask, white_mask)

    colored_px = int((colored_mask > 0).sum())
    white_px = int((white_mask > 0).sum())
    total_px = int((combined > 0).sum())

    print(f"\n   Label detection on FULL image:")
    print(f"      Colored label pixels: {colored_px}")
    print(f"      White text pixels:    {white_px}")
    print(f"      Total:                {total_px}")

    if colored_px > 0:
        print(f"      ⚠️  Colored labels detected — is this truly no-label?")

    # ---------- 2. کراپ ----------
    fan = _detect_fan_region(img, PreprocessorConfig())
    if fan:
        x, y, w, h = fan
        print(f"\n   Fan detected: ({x}, {y}, {w}, {h})")
        cropped = img[y:y + h, x:x + w]
    else:
        print(f"\n   No fan detected — image may already be cropped")
        cropped = img

    # ---------- 3. اجرای کامل ----------
    print(f"\n   Running preprocess...")
    result = preprocess_medical_image(path, PreprocessorConfig(debug=True))

    if not result.success:
        print(f"   ❌ Failed: {result.reason}")
        sys.exit(1)

    print(f"   ✅ Success")
    print(f"      Original:     {result.original_shape}")
    print(f"      Final:        {result.final_shape}")
    print(f"      Cropped:      {result.cropped}")
    print(f"      Inpainted:    {result.inpainted}")
    print(f"      Labels:       {result.labels_removed}")

    # ---------- 4. مقایسه‌ی بافت ----------
    # باید فقط کراپ اتفاق افتاده باشه، بافت تغییر نکنه
    print(f"\n   Tissue comparison (should be ~identical):")
    compare_tissue(cropped, result.image, "Cropped vs Final")

    # ---------- 5. ذخیره ----------
    out = path.replace(".png", "_cleaned.png").replace(".jpg", "_cleaned.png")
    result.save(out)
    print(f"\n   💾 Saved: {out}")

    # ---------- 6. نتیجه ----------
    print("\n" + "=" * 70)
    if total_px == 0:
        print("   ✅ NO labels detected — image is already clean")
    elif total_px < 100:
        print(f"   ℹ️  Only {total_px} colored pixels — likely just noise")
    else:
        print(f"   ⚠️  {total_px} colored pixels — check if genuinely no label")
    print("=" * 70)


if __name__ == "__main__":
    main()
