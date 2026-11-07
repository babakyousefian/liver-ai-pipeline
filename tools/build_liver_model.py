# tools/build_liver_model.py
# -*- coding: utf-8 -*-
"""
Build pre-trained liver model (v3.2) with REAL generalization check.

Usage:
    # 1. Train + save pkl
    python tools/build_liver_model.py

    # 2. Evaluate only (assumes pkl exists)
    python tools/build_liver_model.py --eval-only

    # 3. Custom reference dir
    python tools/build_liver_model.py --ref-dir /path/to/images

    # 4. Force rebuild (ignore existing pkl)
    python tools/build_liver_model.py --force

Output:
    core/security/models/liver_model.pkl

What "generalization check" means here:
    We test the model on images from the SAME reference set, but with
    realistic perturbations (rotation, brightness, blur, flip, JPEG noise,
    slight crop). If the model rejects these, it will reject real-world
    user images too — that's a RED FLAG.

    The check is stricter than v3.1:
    - 6 augmentations per image (was 5)
    - Includes JPEG compression noise
    - Includes slight crop (like real user images)
    - Reports per-augmentation pass rates
    - Fails hard if pass rate < 90% (was 85%)
"""

import os
import sys
import time
import random
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# ✅ Enable training mode BEFORE importing image_guard
os.environ["LIVER_GUARD_ALLOW_TRAIN"] = "1"


# ================================================================
# Argument parsing
# ================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build pre-trained liver model (v3.2)",
    )
    parser.add_argument(
        "--ref-dir",
        default=None,
        help="Reference images directory (default: core/security/liver_reference)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=3000,
        help="Max reference images to train on (default: 3000)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training — only evaluate existing pkl",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild — ignore existing pkl",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=40,
        help="Number of images for generalization check (default: 40)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode — fail if augmentation pass rate < 95%%",
    )
    return parser.parse_args()


# ================================================================
# Augmentations (realistic, matches real-world user images)
# ================================================================

def augment_image(img, aug_name: str = None):
    """
    Realistic augmentations that mimic real user-image variations.

    Args:
        img: BGR numpy array
        aug_name: if given, return only that augmentation.
                  if None, return list of all augmentations.

    Returns:
        - if aug_name: single augmented image
        - else: list of (aug_name, augmented_image)
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    augs = []

    # 1. Slight rotation (±5°) — like user tilt
    def _rotate(angle):
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    augs.append(("rotate_+3", _rotate(3)))
    augs.append(("rotate_-3", _rotate(-3)))

    # 2. Brightness/contrast variations
    def _brightness(alpha, beta):
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    augs.append(("bright_+15", _brightness(1.0, 15)))
    augs.append(("bright_-15", _brightness(1.0, -15)))

    # 3. Slight blur (like lower-quality scans)
    augs.append(("blur_3x3", cv2.GaussianBlur(img, (3, 3), 0)))

    # 4. Horizontal flip
    augs.append(("flip_h", cv2.flip(img, 1)))

    # 5. JPEG compression noise (very realistic!)
    def _jpeg_noise(quality):
        try:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            return img

    augs.append(("jpeg_q60", _jpeg_noise(60)))
    augs.append(("jpeg_q40", _jpeg_noise(40)))

    # 6. Slight crop (like real user crops)
    def _slight_crop(ratio):
        dh = int(h * ratio)
        dw = int(w * ratio)
        return img[dh:h - dh, dw:w - dw]

    augs.append(("crop_5pct", _slight_crop(0.05)))

    if aug_name is not None:
        for name, aug in augs:
            if name == aug_name:
                return aug
        return img  # fallback

    return augs


def augment_image_basic(img):
    """Backward-compat wrapper — returns list of images (no names)."""
    return [aug for _name, aug in augment_image(img)]


# ================================================================
# Evaluation helpers
# ================================================================

def _evaluate_batch(model, image_paths, augment=False, verbose=True):
    """
    Evaluate model on a batch of images.

    Returns:
        dict with keys:
            - total: number of tested images
            - passed: number passed (score >= REJECT_SCORE)
            - scores: list of scores
            - per_aug: dict {aug_name: (total, passed, [scores])}
            - failures: list of (path, aug_name, score) for failures
    """
    import cv2
    from core.security.image_guard import _compute_features, REJECT_SCORE

    result = {
        "total": 0,
        "passed": 0,
        "scores": [],
        "per_aug": {},
        "failures": [],
    }

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            if verbose:
                print(f"   ⚠️ Could not read: {os.path.basename(path)}")
            continue

        if augment:
            aug_list = augment_image(img)
        else:
            aug_list = [("original", img)]

        for aug_name, aug_img in aug_list:
            fvec = _compute_features(aug_img)
            if fvec is None:
                continue

            score = model.score(fvec)
            result["total"] += 1
            result["scores"].append(score)

            # ✅ Track per-augmentation
            if aug_name not in result["per_aug"]:
                result["per_aug"][aug_name] = {"total": 0, "passed": 0, "scores": []}
            result["per_aug"][aug_name]["total"] += 1
            result["per_aug"][aug_name]["scores"].append(score)

            if score >= REJECT_SCORE:
                result["passed"] += 1
                result["per_aug"][aug_name]["passed"] += 1
            else:
                result["failures"].append(
                    (os.path.basename(path), aug_name, float(score))
                )

        if verbose and (i + 1) % 10 == 0:
            print(f"   ... {i + 1}/{len(image_paths)} images processed")

    return result


def _print_eval_report(name: str, result: dict, strict_threshold: float = 0.90):
    """Print a beautiful evaluation report."""
    print()
    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)

    if result["total"] == 0:
        print("   ⚠️ No images tested")
        return False

    pass_rate = result["passed"] / result["total"] * 100
    scores = result["scores"]

    print(f"   Total tested:   {result['total']}")
    print(f"   Passed:         {result['passed']} ({pass_rate:.1f}%)")
    print(f"   Failed:         {result['total'] - result['passed']}")
    print(f"   Score range:    {min(scores):.3f} .. {max(scores):.3f}")

    import numpy as np
    print(f"   Mean score:     {np.mean(scores):.3f}")
    print(f"   Median score:   {np.median(scores):.3f}")
    print(f"   Std score:      {np.std(scores):.3f}")

    # ✅ Per-augmentation breakdown
    if len(result["per_aug"]) > 1:
        print()
        print("   Per-augmentation breakdown:")
        print(f"   {'Augmentation':<20} {'Total':>6} {'Passed':>8} {'Rate':>8} {'Mean':>8}")
        print(f"   {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
        for aug_name in sorted(result["per_aug"].keys()):
            stats = result["per_aug"][aug_name]
            if stats["total"] == 0:
                continue
            aug_rate = stats["passed"] / stats["total"] * 100
            aug_mean = np.mean(stats["scores"]) if stats["scores"] else 0.0
            flag = "✅" if aug_rate >= strict_threshold * 100 else "⚠️"
            print(f"   {flag} {aug_name:<18} {stats['total']:>6} "
                  f"{stats['passed']:>8} {aug_rate:>7.1f}% {aug_mean:>8.3f}")

    # ✅ Show failures
    if result["failures"] and len(result["failures"]) <= 15:
        print()
        print("   Failures:")
        for fname, aug_name, score in result["failures"][:15]:
            print(f"      ❌ {fname:<30} | {aug_name:<12} | score={score:.3f}")
    elif result["failures"]:
        print()
        print(f"   Failures: {len(result['failures'])} (showing first 15)")
        for fname, aug_name, score in result["failures"][:15]:
            print(f"      ❌ {fname:<30} | {aug_name:<12} | score={score:.3f}")

    # ✅ Verdict
    print()
    if pass_rate >= strict_threshold * 100:
        print(f"   ✅ PASS (≥{strict_threshold*100:.0f}%)")
        return True
    else:
        print(f"   ❌ FAIL (<{strict_threshold*100:.0f}%)")
        print(f"   ⚠️ Model may reject valid real-world images.")
        print(f"   ⚠️ Consider: lower PCA_COMPONENTS, raise REJECT_SCORE,")
        print(f"      or add more diverse reference images.")
        return False


# ================================================================
# Main
# ================================================================

def main():
    args = _parse_args()

    from core.security import image_guard
    from core.security.image_guard import (
        _get_liver_model,
        _compute_features,
        _BUNDLE_MODEL_FILE,
        LIVER_REFERENCE_DIR,
        MODEL_CACHE_VERSION,
        REJECT_SCORE,
        PCA_COMPONENTS,
    )

    # ✅ Resolve reference dir
    ref_dir = args.ref_dir if args.ref_dir else LIVER_REFERENCE_DIR
    ref_dir = os.path.abspath(ref_dir)

    print("=" * 70)
    print("  Build Pre-trained Liver Model (v3.2 — Realistic Generalization)")
    print("=" * 70)
    print(f"  Reference dir:    {ref_dir}")
    print(f"  Output pkl:       {_BUNDLE_MODEL_FILE}")
    print(f"  Model version:    {MODEL_CACHE_VERSION}")
    print(f"  PCA components:   {PCA_COMPONENTS}")
    print(f"  REJECT_SCORE:     {REJECT_SCORE}")
    print(f"  Eval-only:        {args.eval_only}")
    print(f"  Force rebuild:    {args.force}")
    print(f"  Sample size:      {args.sample_size}")
    print(f"  Strict mode:      {args.strict}")
    print("=" * 70)
    print()

    # ============================================================
    # Validate reference dir
    # ============================================================
    if not os.path.isdir(ref_dir):
        print(f"❌ Reference dir not found: {ref_dir}")
        sys.exit(1)

    files = [
        f for f in os.listdir(ref_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]
    print(f"📚 Found {len(files)} reference images")

    if len(files) < 10:
        print("❌ Need at least 10 reference images")
        sys.exit(1)
    print()

    # ============================================================
    # STEP 1: Train or load model
    # ============================================================
    os.makedirs(os.path.dirname(_BUNDLE_MODEL_FILE), exist_ok=True)

    if args.eval_only:
        if not os.path.exists(_BUNDLE_MODEL_FILE):
            print(f"❌ Eval-only mode but pkl not found: {_BUNDLE_MODEL_FILE}")
            sys.exit(1)

        print(f"📦 Loading existing model: {_BUNDLE_MODEL_FILE}")
        model = image_guard.LiverModel.load(_BUNDLE_MODEL_FILE)
        if model is None:
            print("❌ Failed to load model")
            sys.exit(1)

        elapsed = 0.0
        print(f"✅ Model loaded ({model.n_samples} samples)")
    else:
        # ✅ Remove old pkl if --force
        if args.force:
            for p in (_BUNDLE_MODEL_FILE,
                      os.path.join(os.path.dirname(_BUNDLE_MODEL_FILE),
                                   "..", ".cache", "liver_model.pkl")):
                p = os.path.abspath(p)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        print(f"🗑️ Removed old pkl: {p}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {p}: {e}")

        # ✅ If pkl exists and not --force, load it
        if os.path.exists(_BUNDLE_MODEL_FILE) and not args.force:
            print(f"📦 Existing pkl found — loading (use --force to rebuild)")
            model = image_guard.LiverModel.load(_BUNDLE_MODEL_FILE)
            if model is None:
                print("⚠️ Failed to load — retraining")
                model = None
            else:
                elapsed = 0.0
                print(f"✅ Loaded ({model.n_samples} samples)")
        else:
            model = None

        if model is None:
            # ✅ Train
            print("🔄 Training model...")
            t0 = time.time()

            # ✅ Set MAX_REFERENCE_IMAGES for this run
            old_max = image_guard.MAX_REFERENCE_IMAGES
            image_guard.MAX_REFERENCE_IMAGES = args.max_images
            try:
                model = _get_liver_model()
            finally:
                image_guard.MAX_REFERENCE_IMAGES = old_max

            elapsed = time.time() - t0

            if model is None:
                print("❌ Training failed")
                sys.exit(1)

            print(f"✅ Training complete in {elapsed:.1f}s")

    # ============================================================
    # STEP 2: Generalization check (augmented)
    # ============================================================
    print()
    print("=" * 70)
    print("  Generalization check (augmented images)")
    print("=" * 70)
    print(f"  Testing {args.sample_size} random images × 10 augmentations each")
    print()

    random.seed(42)
    sample_size = min(args.sample_size, len(files))
    sample_files = random.sample(files, sample_size)
    sample_paths = [os.path.join(ref_dir, f) for f in sample_files]

    aug_result = _evaluate_batch(model, sample_paths, augment=True, verbose=True)

    strict_threshold = 0.95 if args.strict else 0.90
    aug_ok = _print_eval_report(
        "Augmented Generalization",
        aug_result,
        strict_threshold=strict_threshold,
    )

    # ============================================================
    # STEP 3: Original check (should be ~100%)
    # ============================================================
    print()
    print("=" * 70)
    print("  Original reference check")
    print("=" * 70)
    print()

    orig_result = _evaluate_batch(model, sample_paths, augment=False, verbose=True)
    orig_ok = _print_eval_report(
        "Original Images",
        orig_result,
        strict_threshold=0.98,
    )

    # ============================================================
    # STEP 4: Final report
    # ============================================================
    print()
    print("=" * 70)
    print("  ✅ Build Summary")
    print("=" * 70)
    print(f"  Model samples:      {model.n_samples}")
    print(f"  PCA components:     {model.pca.n_components_}")
    print(f"  Threshold:          {model.threshold:.3f}")
    print(f"  REJECT_SCORE:       {REJECT_SCORE}")
    print(f"  Training time:      {elapsed:.1f}s")

    if aug_result["total"] > 0:
        aug_pass = aug_result["passed"] / aug_result["total"] * 100
        print(f"  Augmented pass:     {aug_pass:.1f}%")
    if orig_result["total"] > 0:
        orig_pass = orig_result["passed"] / orig_result["total"] * 100
        print(f"  Original pass:      {orig_pass:.1f}%")

    if os.path.exists(_BUNDLE_MODEL_FILE):
        size_kb = os.path.getsize(_BUNDLE_MODEL_FILE) / 1024
        print(f"  File:               {_BUNDLE_MODEL_FILE}")
        print(f"  Size:               {size_kb:.0f} KB")
    print()

    # ============================================================
    # STEP 5: Verdict
    # ============================================================
    print("=" * 70)
    if aug_ok and orig_ok:
        print("  ✅ Model is ready to ship.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("  ❌ Model NOT ready.")
        print("=" * 70)
        if not orig_ok:
            print("  ⚠️ Original pass rate is too low.")
            print("     This means the model is rejecting its OWN training data.")
            print("     → Check pkl file, PCA_COMPONENTS, threshold calculation.")
        if not aug_ok:
            print("  ⚠️ Augmented pass rate is too low.")
            print("     This means the model will reject valid real-world images.")
            print("     → Lower PCA_COMPONENTS, raise REJECT_SCORE,")
            print("       or add more diverse reference images.")
        print()
        print(f"  Strict mode: {args.strict}")
        print(f"  Thresholds:  original ≥ 98%, augmented ≥ {strict_threshold*100:.0f}%")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
