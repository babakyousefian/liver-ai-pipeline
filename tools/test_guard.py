# tools/test_guard.py
import os
import sys
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from core.security.image_guard import (
    analyze_image, _heuristic_gate, _remove_color_labels_for_metrics,
)
import cv2

if len(sys.argv) < 2:
    print("Usage: python tools/test_guard.py image.png")
    sys.exit(1)

path = sys.argv[1]
print(f"Testing: {path}")
print("=" * 70)

img = cv2.imread(path)
if img is None:
    print("❌ Could not read")
    sys.exit(1)

# ✅ Resize
h, w = img.shape[:2]
if max(h, w) > 800:
    s = 800.0 / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)))

# ✅ Clean
img_clean = _remove_color_labels_for_metrics(img)

# ✅ Gate
is_medical, score, metrics = _heuristic_gate(img_clean)

print(f"\n📊 Metrics:")
for k, v in metrics.items():
    flag = "❌" if v < 0.30 else "✅"
    print(f"   {flag} {k:20s} = {v:.3f}")

print(f"\n📈 Final score: {score:.3f}")
print(f"🎯 is_medical: {is_medical}")

# ✅ Full analysis
print("\n" + "=" * 70)
result = analyze_image(path)
print(f"\n🔍 Full analyze_image:")
for k, v in result.items():
    print(f"   {k}: {v}")

# ✅ Hard rejection check
print("\n" + "=" * 70)
print("🚨 Hard rejection checks:")
checks = [
    ("saturation", 0.30),
    ("grayscale", 0.30),
    ("edges", 0.20),
    ("texture", 0.15),
    ("histogram", 0.20),
]
for key, threshold in checks:
    val = metrics.get(key, 0)
    status = "❌ FAIL" if val < threshold else "✅ PASS"
    print(f"   {status} {key}={val:.3f} (threshold={threshold})")