# tools/test_preprocessor.py
# -*- coding: utf-8 -*-
"""
Test the medical preprocessor on sample images.

Usage:
    python tools/test_preprocessor.py path/to/medical.jpg
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import cv2
from core.preprocessing import preprocess_medical_image, PreprocessorConfig


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/test_preprocessor.py image1.jpg [image2.jpg ...]")
        sys.exit(1)

    print("=" * 70)
    print("  Medical Preprocessor Test")
    print("=" * 70)

    for path in sys.argv[1:]:
        print(f"\n📷 {os.path.basename(path)}")
        print("-" * 70)

        if not os.path.isfile(path):
            print(f"   ❌ Not found")
            continue

        # ✅ اجرا
        result = preprocess_medical_image(path, PreprocessorConfig(debug=True))

        if not result.success:
            print(f"   ❌ Failed: {result.reason}")
            continue

        print(f"   ✅ Success")
        print(f"   Original:     {result.original_shape}")
        print(f"   Final:        {result.final_shape}")
        print(f"   Cropped:      {result.cropped}")
        print(f"   Inpainted:    {result.inpainted}")
        print(f"   Labels found: {result.labels_removed}")

        # ✅ ذخیره‌ی خروجی
        out_path = os.path.splitext(path)[0] + "_clean.png"
        if result.save(out_path):
            print(f"   💾 Saved:     {out_path}")
        else:
            print(f"   ⚠️ Could not save")


if __name__ == "__main__":
    main()
