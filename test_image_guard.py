# test_image_guard.py
# -*- coding: utf-8 -*-
"""
Complete test for image_guard.py
Run from project root:  python test_image_guard.py

Tests:
    1. Model is loaded (from pre-trained pkl, NOT re-trained)
    2. All liver reference images PASS (with stats)
    3. Non-liver images (if available) FAIL
    4. Synthetic non-medical images FAIL
    5. Performance: average analysis time
"""

import os
import sys
import time
import random
from collections import Counter

# ✅ اجازه‌ی import از root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.security.image_guard import (
    analyze_image,
    is_confident_result,
    _get_liver_model,
    _heuristic_gate,
    _compute_features,
    LiverModel,
    MODEL_CACHE_FILE,
    LIVER_REFERENCE_DIR,
    NON_LIVER_REFERENCE_DIR,
    MIN_LIVER_SCORE,
    MIN_MODEL_CONFIDENCE,
)

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("❌ OpenCV/numpy not available")
    sys.exit(1)


# ================================================================
# Helpers
# ================================================================

def print_header(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_result(ok: bool, label: str, detail: str = ""):
    icon = "✅" if ok else "❌"
    line = f"{icon} {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)


def create_synthetic_non_medical(path: str, kind: str = "noise") -> bool:
    """ساخت تصویر مصنوعی غیرپزشکی برای تست rejection."""
    try:
        h, w = 400, 400
        if kind == "noise":
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        elif kind == "colorful":
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :w // 3] = (255, 0, 0)
            img[:, w // 3: 2 * w // 3] = (0, 255, 0)
            img[:, 2 * w // 3:] = (0, 0, 255)
        elif kind == "blank":
            img = np.full((h, w, 3), 128, dtype=np.uint8)
        elif kind == "gradient":
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                img[y, :] = (y % 256, (y * 2) % 256, (y * 3) % 256)
        elif kind == "text":
            img = np.full((h, w, 3), 255, dtype=np.uint8)
            cv2.putText(img, "HELLO WORLD", (40, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
        else:
            img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(path, img)
        return True
    except Exception as e:
        print(f"⚠️ Failed to create synthetic image: {e}")
        return False


# ================================================================
# Test 1 — Model is available
# ================================================================

def test_model_available() -> bool:
    print_header("TEST 1 — Model availability")

    t0 = time.time()
    model = _get_liver_model()
    elapsed = time.time() - t0

    if model is None:
        print_result(False, "Model not available",
                     "No pre-trained pkl and not enough references")
        return False

    print_result(True, "Model loaded",
                 f"{model.n_samples} samples, threshold={model.threshold:.3f}")
    print_result(elapsed < 2.0, "Load time",
                 f"{elapsed * 1000:.0f}ms (target: <2000ms)")
    print_result(model.n_samples >= 10, "Sample count",
                 f"{model.n_samples} (need ≥10)")
    print_result(model.pca is not None, "PCA available", "yes")
    print_result(model.cov is not None, "Covariance available", "yes")
    return True


# ================================================================
# Test 2 — All liver references PASS
# ================================================================

def test_liver_references(sample_limit: int = 50) -> dict:
    print_header(f"TEST 2 — Liver references (sample of {sample_limit})")

    if not os.path.isdir(LIVER_REFERENCE_DIR):
        print_result(False, "Directory not found", LIVER_REFERENCE_DIR)
        return {"total": 0, "passed": 0, "failed": 0}

    files = sorted([
        f for f in os.listdir(LIVER_REFERENCE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])
    if not files:
        print_result(False, "No images found", LIVER_REFERENCE_DIR)
        return {"total": 0, "passed": 0, "failed": 0}

    # نمونه‌گیری یکنواخت
    if len(files) > sample_limit:
        random.seed(42)
        files = random.sample(files, sample_limit)

    passed, failed = 0, 0
    scores = []
    heuristic_scores = []
    times = []

    print(f"\n{'File':<40} {'H':<6} {'L':<6} {'F':<6} {'Status'}")
    print("-" * 72)

    for fname in files:
        path = os.path.join(LIVER_REFERENCE_DIR, fname)
        t0 = time.time()
        r = analyze_image(path)
        times.append(time.time() - t0)

        m = r.get("metrics", {})
        h_score = m.get("heuristic_score", 0.0)
        l_score = m.get("liver_score", 0.0)
        f_score = r.get("score", 0.0)

        scores.append(f_score)
        heuristic_scores.append(h_score)

        ok = r["is_liver_like"]
        if ok:
            passed += 1
        else:
            failed += 1

        print(
            f"{fname[:38]:<40} "
            f"{h_score:.2f}  {l_score:.2f}  {f_score:.2f}  "
            f"{'✅' if ok else '❌'}"
            + (f"  {r.get('reason', '')[:30]}" if not ok else "")
        )

    pass_rate = passed / max(1, passed + failed) * 100
    print("\n" + "-" * 72)
    print_result(pass_rate >= 85, f"Pass rate: {passed}/{passed+failed} "
                                  f"({pass_rate:.1f}%) — target ≥85%")
    if scores:
        print(f"   Mean score:      {np.mean(scores):.3f}")
        print(f"   Std score:       {np.std(scores):.3f}")
        print(f"   Mean heuristic:  {np.mean(heuristic_scores):.3f}")
        print(f"   Mean time:       {np.mean(times)*1000:.1f}ms")

    return {
        "total": passed + failed,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "mean_time_ms": np.mean(times) * 1000 if times else 0,
    }


# ================================================================
# Test 3 — Non-liver references FAIL
# ================================================================

def test_non_liver_references(sample_limit: int = 50) -> dict:
    print_header(f"TEST 3 — Non-liver references (sample of {sample_limit})")

    if not os.path.isdir(NON_LIVER_REFERENCE_DIR):
        print_result(False, "Directory not found (skip)",
                     NON_LIVER_REFERENCE_DIR)
        return {"total": 0, "rejected": 0, "passed": 0, "reject_rate": 0}

    files = sorted([
        f for f in os.listdir(NON_LIVER_REFERENCE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])
    if not files:
        print_result(False, "No images found (skip)", NON_LIVER_REFERENCE_DIR)
        return {"total": 0, "rejected": 0, "passed": 0, "reject_rate": 0}

    if len(files) > sample_limit:
        random.seed(43)
        files = random.sample(files, sample_limit)

    rejected, passed = 0, 0
    print(f"\n{'File':<40} {'Score':<8} {'Status'}")
    print("-" * 72)

    for fname in files:
        path = os.path.join(NON_LIVER_REFERENCE_DIR, fname)
        r = analyze_image(path)
        score = r.get("score", 0.0)
        is_rejected = not r["is_liver_like"]
        if is_rejected:
            rejected += 1
            status = "✅ (rejected)"
        else:
            passed += 1
            status = "❌ (passed — BAD)"
        print(f"{fname[:38]:<40} {score:.2f}     {status}")

    reject_rate = rejected / max(1, rejected + passed) * 100
    print("\n" + "-" * 72)
    print_result(reject_rate >= 80,
                 f"Reject rate: {rejected}/{rejected+passed} "
                 f"({reject_rate:.1f}%) — target ≥80%")
    return {
        "total": rejected + passed,
        "rejected": rejected,
        "passed": passed,
        "reject_rate": reject_rate,
    }


# ================================================================
# Test 4 — Synthetic non-medical images FAIL
# ================================================================

def test_synthetic_images(tmp_dir: str = "/tmp") -> dict:
    print_header("TEST 4 — Synthetic non-medical images (should FAIL)")

    kinds = ["noise", "colorful", "blank", "gradient", "text"]
    rejected, passed = 0, 0

    for kind in kinds:
        path = os.path.join(tmp_dir, f"_test_{kind}.jpg")
        if not create_synthetic_non_medical(path, kind):
            continue

        r = analyze_image(path)
        is_rejected = not r["is_liver_like"]
        if is_rejected:
            rejected += 1
            print_result(True, f"{kind:<10}",
                         f"score={r.get('score', 0):.2f} → {r.get('reason', '')[:40]}")
        else:
            passed += 1
            print_result(False, f"{kind:<10}",
                         f"score={r.get('score', 0):.2f} — BAD: passed!")

        try:
            os.remove(path)
        except Exception:
            pass

    total = rejected + passed
    print()
    print_result(rejected == total, f"All rejected: {rejected}/{total}")
    return {"total": total, "rejected": rejected, "passed": passed}


# ================================================================
# Test 5 — is_confident_result (final gate)
# ================================================================

def test_confident_result() -> bool:
    print_header("TEST 5 — is_confident_result()")

    if not os.path.isdir(LIVER_REFERENCE_DIR):
        print_result(False, "No liver references to test")
        return False

    files = sorted([
        f for f in os.listdir(LIVER_REFERENCE_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ])[:5]
    if not files:
        print_result(False, "Empty reference dir")
        return False

    # Case A: good liver + high confidence → should pass
    path = os.path.join(LIVER_REFERENCE_DIR, files[0])
    ok, reason, m = is_confident_result(
        image_path=path,
        model_confidence=0.90,
        all_probs={0: 0.02, 1: 0.05, 2: 0.90, 3: 0.03},
    )
    print_result(ok, "Good liver + high conf",
                 f"ok={ok}, reason={reason or '—'}")

    # Case B: good liver + low confidence → should fail
    ok2, reason2, _ = is_confident_result(
        image_path=path,
        model_confidence=0.20,
        all_probs={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
    )
    print_result(not ok2, "Good liver + low conf",
                 f"reason={reason2[:50]}")

    return ok and not ok2


# ================================================================
# Test 6 — Performance
# ================================================================

def test_performance(num_images: int = 10) -> dict:
    print_header(f"TEST 6 — Performance ({num_images} images)")

    if not os.path.isdir(LIVER_REFERENCE_DIR):
        return {"avg_ms": 0, "max_ms": 0}

    files = sorted([
        f for f in os.listdir(LIVER_REFERENCE_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ])
    if not files:
        return {"avg_ms": 0, "max_ms": 0}

    random.seed(44)
    files = random.sample(files, min(num_images, len(files)))

    times = []
    for fname in files:
        path = os.path.join(LIVER_REFERENCE_DIR, fname)
        t0 = time.time()
        analyze_image(path)
        times.append((time.time() - t0) * 1000)

    avg = float(np.mean(times))
    mx = float(np.max(times))
    print_result(avg < 500, f"Average time: {avg:.1f}ms (target <500ms)")
    print_result(mx < 1500, f"Max time:     {mx:.1f}ms (target <1500ms)")
    return {"avg_ms": avg, "max_ms": mx}


# ================================================================
# Main
# ================================================================

def main():
    print("\n" + "🔬" * 36)
    print("       IMAGE GUARD v3.0 — COMPLETE TEST SUITE")
    print("🔬" * 36)

    if not os.path.exists(MODEL_CACHE_FILE):
        print(f"\n⚠️  Pre-trained model NOT found: {MODEL_CACHE_FILE}")
        print("   → On a fresh install, guard will fail-open (no 5-minute training).")
        print("   → To generate the pkl once, run with env: FORCE_TRAIN=1\n")

    results = {}

    # 1
    if not test_model_available():
        print("\n🛑 Cannot continue — model unavailable")
        return 1

    # 2
    results["liver"] = test_liver_references(sample_limit=50)

    # 3
    results["non_liver"] = test_non_liver_references(sample_limit=50)

    # 4
    results["synthetic"] = test_synthetic_images()

    # 5
    results["confident"] = test_confident_result()

    # 6
    results["perf"] = test_performance()

    # ============================================================
    # Final report
    # ============================================================
    print_header("FINAL REPORT")

    liver = results["liver"]
    print(f"  Liver pass rate:      {liver['passed']}/{liver['total']} "
          f"({liver['pass_rate']:.1f}%)  {'✅' if liver['pass_rate'] >= 85 else '❌'}")

    nl = results["non_liver"]
    if nl["total"] > 0:
        print(f"  Non-liver reject:     {nl['rejected']}/{nl['total']} "
              f"({nl['reject_rate']:.1f}%)  "
              f"{'✅' if nl['reject_rate'] >= 80 else '❌'}")
    else:
        print(f"  Non-liver reject:     (no references)  ⏭️")

    syn = results["synthetic"]
    print(f"  Synthetic reject:     {syn['rejected']}/{syn['total']}  "
          f"{'✅' if syn['rejected'] == syn['total'] else '❌'}")

    print(f"  Confident gate:       {'✅' if results['confident'] else '❌'}")

    perf = results["perf"]
    print(f"  Avg analysis time:    {perf['avg_ms']:.1f}ms  "
          f"{'✅' if perf['avg_ms'] < 500 else '❌'}")

    # 🎯 Overall
    all_ok = (
        liver["pass_rate"] >= 85
        and (nl["total"] == 0 or nl["reject_rate"] >= 80)
        and syn["rejected"] == syn["total"]
        and results["confident"]
        and perf["avg_ms"] < 500
    )
    print("\n" + "=" * 72)
    print(f"  {'🎉 ALL TESTS PASSED' if all_ok else '⚠️  SOME TESTS FAILED'}")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
