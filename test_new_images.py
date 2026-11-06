# tools/test_new_images.py
# -*- coding: utf-8 -*-
"""
Test the liver model on YOUR new images (that were previously rejected).

Usage:
    python tools/test_new_images.py path/to/image1.png path/to/image2.png
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from core.security.image_guard import analyze_image


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/test_new_images.py image1.png image2.png ...")
        sys.exit(1)

    print("=" * 70)
    print("  Testing new images")
    print("=" * 70)

    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"❌ Not found: {path}")
            continue

        print(f"\n📷 {os.path.basename(path)}")
        print("-" * 70)

        r = analyze_image(path)
        print(f"   is_liver:        {r['is_liver_like']}")
        print(f"   final_score:     {r['score']:.3f}")
        m = r.get("metrics", {})
        print(f"   heuristic_score: {m.get('heuristic_score', 0):.3f}")
        print(f"   liver_score:     {m.get('liver_score', 0):.3f}")
        if "reason" in r and r["reason"]:
            print(f"   reason:          {r['reason']}")
        status = "✅ ACCEPTED" if r["is_liver_like"] else "❌ REJECTED"
        print(f"   → {status}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
