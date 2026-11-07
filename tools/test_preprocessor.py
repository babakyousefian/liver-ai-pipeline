# tools/test_preprocessor.py
# -*- coding: utf-8 -*-
"""
Test the medical preprocessor (v4.0) on sample images.

Usage:
    # Test single image with visual comparison
    python tools/test_preprocessor.py --compare image.jpg

    # Test a folder with summary
    python tools/test_preprocessor.py --dir /path/to/folder --summary --compare

    # Force modes (override auto-detect)
    python tools/test_preprocessor.py --force-fan image.jpg
    python tools/test_preprocessor.py --no-crop image.jpg
"""

import os
import sys
import argparse
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np


# ================================================================
# Arguments
# ================================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="Test Medical Preprocessor v4.0")
    parser.add_argument("images", nargs="*", help="Image path(s)")
    parser.add_argument("--dir", default=None, help="Directory of images")
    parser.add_argument("--compare", action="store_true",
                        help="Save side-by-side comparison")
    parser.add_argument("--force-fan", action="store_true",
                        help="Force fan-crop")
    parser.add_argument("--no-crop", action="store_true",
                        help="Force no-crop")
    parser.add_argument("--summary", action="store_true",
                        help="Print aggregated summary")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--max", type=int, default=0, help="Max images")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    return parser.parse_args()


# ================================================================
# Comparison builder
# ================================================================

def _make_comparison(original: np.ndarray, cleaned: np.ndarray,
                     decision: str = "", has_header: bool = False,
                     has_panels: bool = False) -> np.ndarray:
    """Build side-by-side comparison with labels."""
    try:
        target_h = 500

        def _resize_h(img):
            h, w = img.shape[:2]
            if h == 0:
                return img
            scale = target_h / h
            return cv2.resize(img, (max(1, int(w * scale)), target_h),
                              interpolation=cv2.INTER_AREA)

        left = _resize_h(original)
        right = _resize_h(cleaned)

        def _add_label(img, text, color=(20, 20, 20)):
            img = img.copy()
            band_h = 56
            cv2.rectangle(img, (0, 0), (img.shape[1], band_h), color, -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # First line: main label
            line1 = text.split("\n")[0]
            (tw, th), _ = cv2.getTextSize(line1, font, font_scale, thickness)
            x = (img.shape[1] - tw) // 2
            y = 22
            cv2.putText(img, line1, (x, y), font, font_scale, (255, 255, 255), thickness)

            # Second line (if present)
            if "\n" in text:
                line2 = text.split("\n", 1)[1]
                font_scale2 = 0.45
                (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale2, 1)
                x2 = (img.shape[1] - tw2) // 2
                y2 = 48
                cv2.putText(img, line2, (x2, y2), font, font_scale2, (200, 200, 200), 1)

            return img

        left = _add_label(left, "Original")
        header_info = f"header={has_header}, panels={has_panels}"
        right = _add_label(right, f"Preprocessed\n{decision}\n{header_info}",
                           color=(20, 50, 20))

        gap = np.full((target_h, 8, 3), 50, dtype=np.uint8)
        return np.hstack([left, gap, right])
    except Exception as e:
        print(f"   ⚠️ Comparison failed: {e}")
        return original


# ================================================================
# Single image
# ================================================================

def _process_one(path: str, args, stats: dict = None) -> dict:
    from core.preprocessing import preprocess_medical_image, PreprocessorConfig
    from core.security.image_guard import analyze_image

    info = {
        "path": path,
        "success": False,
        "cropped": False,
        "labels_removed": 0,
        "inpainted": False,
        "original_shape": (0, 0),
        "final_shape": (0, 0),
        "has_header": False,
        "has_side_panels": False,
        "decision": "",
        "guard_liver": None,
        "guard_score": 0.0,
        "error": None,
    }

    print(f"\n📷 {os.path.basename(path)}")
    print("-" * 70)

    if not os.path.isfile(path):
        print(f"   ❌ Not found")
        info["error"] = "not found"
        return info

    original = cv2.imread(path)
    if original is None:
        print(f"   ❌ Could not read")
        info["error"] = "read failed"
        return info

    # ✅ Determine override mode
    already_cropped = None
    if args.force_fan:
        already_cropped = False
    elif args.no_crop:
        already_cropped = True

    # ✅ Run preprocessor
    t0 = time.time()
    try:
        result = preprocess_medical_image(
            path,
            PreprocessorConfig(debug=args.debug),
            already_cropped=already_cropped,
        )
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        info["error"] = str(e)
        return info

    elapsed = time.time() - t0
    info.update({
        "success": result.success,
        "original_shape": result.original_shape,
        "final_shape": result.final_shape,
        "cropped": result.cropped,
        "labels_removed": result.labels_removed,
        "inpainted": result.inpainted,
        "has_header": result.has_header,
        "has_side_panels": result.has_side_panels,
        "decision": result.crop_decision,
    })

    if not result.success:
        print(f"   ❌ Failed: {result.reason}")
        info["error"] = result.reason
        return info

    # ✅ Report
    print(f"   ✅ Success         ({elapsed*1000:.0f}ms)")
    print(f"   Original:          {result.original_shape[1]}×{result.original_shape[0]}")
    print(f"   Final:             {result.final_shape[1]}×{result.final_shape[0]}")
    print(f"   Has header:        {result.has_header}")
    print(f"   Has side panels:   {result.has_side_panels}")
    print(f"   Decision:          {result.crop_decision}")
    print(f"   Cropped:           {result.cropped}")
    print(f"   Inpainted:         {result.inpainted}")
    print(f"   Labels removed:    {result.labels_removed}")

    # ✅ Save output
    out_dir = args.out_dir if args.out_dir else os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{base}_clean.png")
    if result.save(out_path):
        print(f"   💾 Saved:          {out_path}")

    # ✅ Comparison
    if args.compare:
        cmp_path = os.path.join(out_dir, f"{base}_compare.png")
        cmp = _make_comparison(
            original, result.image,
            decision=result.crop_decision,
            has_header=result.has_header,
            has_panels=result.has_side_panels,
        )
        if cv2.imwrite(cmp_path, cmp):
            print(f"   🖼️ Comparison:     {cmp_path}")

    # ✅ Guard check
    try:
        guard = analyze_image(out_path)
        is_liver = guard.get("is_liver_like", True)
        score = guard.get("score", 0.0)
        info["guard_liver"] = is_liver
        info["guard_score"] = score
        icon = "✅" if is_liver else "❌"
        print(f"   🛡️ Guard:          {icon} is_liver={is_liver}, score={score:.3f}")
    except Exception as e:
        print(f"   ⚠️ Guard failed: {e}")

    # ✅ Stats
    if stats is not None:
        stats["total"] += 1
        if result.success:
            stats["success"] += 1
            if result.cropped:
                stats["cropped"] += 1
            if result.labels_removed > 0:
                stats["labels_removed"] += 1
            if result.has_header:
                stats["with_header"] += 1
            if result.has_side_panels:
                stats["with_panels"] += 1
        if info["guard_liver"] is not None:
            stats["guard_checked"] += 1
            if info["guard_liver"]:
                stats["guard_passed"] += 1

    return info


# ================================================================
# Summary
# ================================================================

def _print_summary(stats: dict, results: list):
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)

    total = stats["total"]
    if total == 0:
        print("  No images processed.")
        return

    print(f"  Total images:             {total}")
    print(f"  Success:                  {stats['success']} "
          f"({stats['success']/total*100:.1f}%)")

    if stats["success"] > 0:
        s = stats["success"]
        print(f"  Cropped:                  {stats['cropped']} "
              f"({stats['cropped']/s*100:.1f}%)")
        print(f"  Labels removed:           {stats['labels_removed']} "
              f"({stats['labels_removed']/s*100:.1f}%)")
        print(f"  With header (top/bottom): {stats['with_header']} "
              f"({stats['with_header']/s*100:.1f}%)")
        print(f"  With side panels:         {stats['with_panels']} "
              f"({stats['with_panels']/s*100:.1f}%)")

    if stats["guard_checked"] > 0:
        g = stats["guard_checked"]
        print(f"  Guard checked:            {g}")
        print(f"  Guard passed:             {stats['guard_passed']} "
              f"({stats['guard_passed']/g*100:.1f}%)")

    errors = [r for r in results if not r.get("success")]
    if errors:
        print()
        print(f"  ❌ Failed ({len(errors)}):")
        for r in errors:
            print(f"     - {os.path.basename(r['path'])}: {r.get('error')}")

    rejected = [r for r in results
                if r.get("success") and r.get("guard_liver") is False]
    if rejected:
        print()
        print(f"  🛡️ Rejected by guard ({len(rejected)}):")
        for r in rejected:
            print(f"     - {os.path.basename(r['path'])}: score={r['guard_score']:.3f}")


# ================================================================
# Main
# ================================================================

def main():
    args = _parse_args()

    paths = list(args.images) if args.images else []

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"❌ Directory not found: {args.dir}")
            sys.exit(1)
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        paths.extend(sorted([
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.lower().endswith(exts)
        ]))

    if not paths:
        print("❌ No images provided.")
        print("Usage:")
        print("  python tools/test_preprocessor.py image.jpg")
        print("  python tools/test_preprocessor.py --dir /path/to/folder")
        sys.exit(1)

    if args.max > 0:
        paths = paths[:args.max]

    print("=" * 70)
    print("  Medical Preprocessor Test (v4.0 — Header-aware)")
    print("=" * 70)
    print(f"  Images:      {len(paths)}")
    print(f"  Mode:        {'force-fan' if args.force_fan else 'no-crop' if args.no_crop else 'auto'}")
    print(f"  Compare:     {args.compare}")
    if args.out_dir:
        print(f"  Output dir:  {args.out_dir}")
    print("=" * 70)

    stats = {
        "total": 0, "success": 0, "cropped": 0,
        "labels_removed": 0, "with_header": 0, "with_panels": 0,
        "guard_checked": 0, "guard_passed": 0,
    }
    results = []
    t_start = time.time()

    for path in paths:
        try:
            results.append(_process_one(path, args, stats=stats))
        except Exception as e:
            print(f"\n📷 {os.path.basename(path)}")
            print(f"   ❌ Unexpected: {e}")
            import traceback
            traceback.print_exc()
            results.append({"path": path, "success": False, "error": str(e)})

    elapsed = time.time() - t_start

    if args.summary or len(paths) > 1:
        _print_summary(stats, results)
        print()
        print(f"  Total time: {elapsed:.1f}s "
              f"({elapsed/max(1,len(paths))*1000:.0f}ms/image)")

    print()
    print("=" * 70)
    print("  ✅ Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
