# merge_datasets.py
"""
Merge Real + Synthetic Datasets for Extended Training

This script:
1. Creates a new directory `cropped_extended/` with all real + synthetic images
2. Creates `labels_extended.csv` with all labels
3. Reports statistics for verification

Usage:
    python merge_datasets.py
    python merge_datasets.py --copy-mode hardlink
    python merge_datasets.py --dry-run

Author: Babak Yousefian
Version: 1.0.0
"""

import argparse
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


# ================================================================
# ✅ CONSTANTS
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]

# Default paths (relative to project root)
DEFAULT_REAL_DIR = "../../finalReports/cropped"
DEFAULT_REAL_LABELS = "finalReports/labels.csv"
DEFAULT_GAN_ROOT = "finalReports/selfSupervisedLearning/GAN"
DEFAULT_EXTENDED_DIR = "finalReports/cropped_extended"
DEFAULT_EXTENDED_LABELS = "finalReports/labels_extended.csv"

# Image extensions to look for
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".bmp", ".BMP"}


# ================================================================
# ✅ HELPERS
# ================================================================

def print_header(title: str):
    """Print a nice header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def find_images_in_dir(directory: str) -> list:
    """Find all image files in a directory (non-recursive)."""
    if not os.path.exists(directory):
        return []

    images = []
    for f in os.listdir(directory):
        ext = os.path.splitext(f)[1]
        if ext in IMAGE_EXTS:
            images.append(f)
    return sorted(images)


def copy_file(src: str, dst: str, mode: str = "copy"):
    """
    Copy or link a file from src to dst.

    Args:
        src: Source file path
        dst: Destination file path
        mode: 'copy', 'hardlink', or 'symlink'
    """
    if os.path.exists(dst):
        return  # skip if already exists

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except (OSError, NotImplementedError):
            # Fallback to copy if hardlink fails (e.g., different filesystem)
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            os.symlink(os.path.abspath(src), dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


# ================================================================
# ✅ MAIN MERGE FUNCTION
# ================================================================

def merge_datasets(
    real_dir: str = DEFAULT_REAL_DIR,
    real_labels: str = DEFAULT_REAL_LABELS,
    gan_root: str = DEFAULT_GAN_ROOT,
    extended_dir: str = DEFAULT_EXTENDED_DIR,
    extended_labels: str = DEFAULT_EXTENDED_LABELS,
    copy_mode: str = "copy",
    dry_run: bool = False,
    verbose: bool = True,
):
    """
    Merge real and synthetic datasets.
    """
    print_header("🔄 MERGING REAL + SYNTHETIC DATASETS")

    print(f"   📂 Real images dir:    {real_dir}")
    print(f"   📄 Real labels:        {real_labels}")
    print(f"   📂 GAN root:           {gan_root}")
    print(f"   📂 Extended dir:       {extended_dir}")
    print(f"   📄 Extended labels:    {extended_labels}")
    print(f"   🔧 Copy mode:          {copy_mode}")
    print(f"   🧪 Dry run:            {dry_run}")

    # ============================================================
    # ✅ Step 1: Validate inputs
    # ============================================================
    if not os.path.exists(real_dir):
        print(f"\n❌ ERROR: Real images directory not found: {real_dir}")
        return False

    if not os.path.exists(real_labels):
        print(f"\n❌ ERROR: Real labels file not found: {real_labels}")
        return False

    if not os.path.exists(gan_root):
        print(f"\n❌ ERROR: GAN root directory not found: {gan_root}")
        print(f"   Please run GAN training first (see run_gan.py)")
        return False

    # ============================================================
    # ✅ Step 2: Load real labels
    # ============================================================
    print_header("📄 Loading real labels")
    real_df = pd.read_csv(real_labels)
    print(f"   Real labels loaded: {len(real_df)} rows")
    print(f"   Columns: {list(real_df.columns)}")

    if "liverName" not in real_df.columns or "liverLabel" not in real_df.columns:
        print(f"❌ ERROR: Required columns not found. Need 'liverName' and 'liverLabel'")
        return False

    # Normalize liverName column (strip whitespace)
    real_df["liverName"] = real_df["liverName"].astype(str).str.strip()
    real_df["liverLabel"] = real_df["liverLabel"].astype(str).str.strip()

    # ============================================================
    # ✅ Step 3: Find real images that actually exist
    # ============================================================
    print_header("🔍 Verifying real images")

    real_images_in_dir = find_images_in_dir(real_dir)
    real_images_in_dir_set = set(real_images_in_dir)

    print(f"   Images found in {real_dir}: {len(real_images_in_dir)}")
    print(f"   Labels in {real_labels}: {len(real_df)}")

    # Match labels with actual images
    matched_real = []
    missing_files = []
    for _, row in real_df.iterrows():
        name = row["liverName"]
        label = row["liverLabel"]

        # Try exact match first
        if name in real_images_in_dir_set:
            matched_real.append({"liverName": name, "liverLabel": label})
            continue

        # Try with .png
        name_png = f"{name}.png"
        if name_png in real_images_in_dir_set:
            matched_real.append({"liverName": name_png, "liverLabel": label})
            continue

        # Try without extension
        name_no_ext = os.path.splitext(name)[0]
        for img in real_images_in_dir:
            if os.path.splitext(img)[0] == name_no_ext:
                matched_real.append({"liverName": img, "liverLabel": label})
                break
        else:
            missing_files.append(name)

    print(f"   ✅ Matched labels: {len(matched_real)}")
    print(f"   ❌ Missing files:  {len(missing_files)}")

    if missing_files[:5]:
        print(f"   Example missing: {missing_files[:5]}")

    # ============================================================
    # ✅ Step 4: Find synthetic images
    # ============================================================
    print_header("🔍 Finding synthetic images")

    synthetic_rows = []  # list of (class_name, synthetic_image_name, gan_class_dir)

    for class_name in CLASS_NAMES:
        # Skip NL (we don't generate synthetic for NL)
        if class_name == "NL":
            continue

        synthetic_dir = os.path.join(gan_root, class_name, "synthetic")

        if not os.path.exists(synthetic_dir):
            print(f"   ⚠️  {class_name}: synthetic dir not found: {synthetic_dir}")
            continue

        synthetic_images = find_images_in_dir(synthetic_dir)
        print(f"   ✅ {class_name}: {len(synthetic_images)} synthetic images")

        for img_name in synthetic_images:
            synthetic_rows.append({
                "liverName": img_name,
                "liverLabel": class_name,
                "source_dir": synthetic_dir,
            })

    print(f"\n   📊 Total synthetic images: {len(synthetic_rows)}")

    # ============================================================
    # ✅ Step 5: Create extended directory
    # ============================================================
    if not dry_run:
        print_header(f"📁 Creating extended directory: {extended_dir}")
        os.makedirs(extended_dir, exist_ok=True)
        print(f"   ✅ Created: {extended_dir}")
    else:
        print_header("🧪 DRY RUN — not creating directories")

    # ============================================================
    # ✅ Step 6: Copy real images
    # ============================================================
    print_header("📋 Copying real images")

    copied_real = 0
    skipped_real = 0

    for i, row in enumerate(matched_real):
        src = os.path.join(real_dir, row["liverName"])
        dst = os.path.join(extended_dir, row["liverName"])

        if dry_run:
            copied_real += 1
            continue

        if os.path.exists(dst):
            skipped_real += 1
            continue

        try:
            copy_file(src, dst, mode=copy_mode)
            copied_real += 1
        except Exception as e:
            print(f"   ⚠️  Failed to copy {row['liverName']}: {e}")

        if verbose and (i + 1) % 500 == 0:
            print(f"   Progress: {i + 1}/{len(matched_real)}")

    print(f"   ✅ Copied: {copied_real}")
    print(f"   ⏭️  Skipped (already exist): {skipped_real}")

    # ============================================================
    # ✅ Step 7: Copy synthetic images
    # ============================================================
    print_header("📋 Copying synthetic images")

    copied_synthetic = 0
    skipped_synthetic = 0

    for i, row in enumerate(synthetic_rows):
        src = os.path.join(row["source_dir"], row["liverName"])
        dst = os.path.join(extended_dir, row["liverName"])

        if dry_run:
            copied_synthetic += 1
            continue

        if os.path.exists(dst):
            skipped_synthetic += 1
            continue

        try:
            copy_file(src, dst, mode=copy_mode)
            copied_synthetic += 1
        except Exception as e:
            print(f"   ⚠️  Failed to copy {row['liverName']}: {e}")

        if verbose and (i + 1) % 500 == 0:
            print(f"   Progress: {i + 1}/{len(synthetic_rows)}")

    print(f"   ✅ Copied: {copied_synthetic}")
    print(f"   ⏭️  Skipped (already exist): {skipped_synthetic}")

    # ============================================================
    # ✅ Step 8: Create extended labels CSV
    # ============================================================
    print_header("📄 Creating extended labels CSV")

    # Combine real + synthetic
    extended_rows = []

    # Real
    for row in matched_real:
        extended_rows.append({
            "liverName": row["liverName"],
            "liverLabel": row["liverLabel"],
            "source": "real",
        })

    # Synthetic
    for row in synthetic_rows:
        extended_rows.append({
            "liverName": row["liverName"],
            "liverLabel": row["liverLabel"],
            "source": "synthetic",
        })

    extended_df = pd.DataFrame(extended_rows)

    # Remove duplicates (keep first)
    extended_df = extended_df.drop_duplicates(subset=["liverName"], keep="first")

    # ✅ Keep original column format (liverName, liverLabel only)
    extended_df_final = extended_df[["liverName", "liverLabel"]].copy()

    if not dry_run:
        os.makedirs(os.path.dirname(extended_labels) if os.path.dirname(extended_labels) else ".", exist_ok=True)
        extended_df_final.to_csv(extended_labels, index=False)
        print(f"   ✅ Saved: {extended_labels}")

        # Also save with source info for reference
        extended_df.to_csv(
            extended_labels.replace(".csv", "_with_source.csv"),
            index=False,
        )
        print(f"   ✅ Saved (with source): {extended_labels.replace('.csv', '_with_source.csv')}")
    else:
        print(f"   🧪 DRY RUN — would save {len(extended_df_final)} rows to {extended_labels}")

    # ============================================================
    # ✅ Step 9: Report statistics
    # ============================================================
    print_header("📊 FINAL STATISTICS")

    # Label distribution
    label_counts = extended_df_final["liverLabel"].value_counts().sort_index()

    print("\n   📈 Label distribution (extended):")
    print(f"   {'Class':<8} {'Real':>8} {'Synthetic':>12} {'Total':>8} {'Percent':>10}")
    print(f"   {'-'*50}")

    total = len(extended_df_final)
    for class_name in CLASS_NAMES:
        real_count = sum(1 for _, r in extended_df.iterrows()
                        if r["liverLabel"] == class_name and r["source"] == "real")
        synthetic_count = sum(1 for _, r in extended_df.iterrows()
                             if r["liverLabel"] == class_name and r["source"] == "synthetic")
        total_count = label_counts.get(class_name, 0)
        percent = 100 * total_count / total if total > 0 else 0

        print(f"   {class_name:<8} {real_count:>8} {synthetic_count:>12} {total_count:>8} {percent:>9.1f}%")

    print(f"   {'-'*50}")
    print(f"   {'TOTAL':<8} {len(matched_real):>8} {len(synthetic_rows):>12} {total:>8} {'100.0%':>10}")

    # Verify files in extended dir
    if not dry_run:
        extended_images = find_images_in_dir(extended_dir)
        print(f"\n   📁 Files in {extended_dir}: {len(extended_images)}")

        if len(extended_images) == total:
            print(f"   ✅ File count matches label count")
        else:
            print(f"   ⚠️  File count ({len(extended_images)}) != label count ({total})")
            print(f"   This may be due to duplicate filenames — check!")

    # ============================================================
    # ✅ Step 10: Save summary JSON
    # ============================================================
    if not dry_run:
        import json

        summary = {
            "timestamp": datetime.now().isoformat(),
            "real_dir": real_dir,
            "extended_dir": extended_dir,
            "gan_root": gan_root,
            "extended_labels": extended_labels,
            "copy_mode": copy_mode,
            "real_images_found": len(real_images_in_dir),
            "real_labels_matched": len(matched_real),
            "real_labels_missing": len(missing_files),
            "synthetic_images_found": len(synthetic_rows),
            "total_extended": total,
            "label_distribution": {k: int(v) for k, v in label_counts.items()},
        }

        summary_path = os.path.join(
            os.path.dirname(extended_labels) if os.path.dirname(extended_labels) else ".",
            "merge_summary.json",
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\n   ✅ Summary saved: {summary_path}")

    print_header("✅ MERGE COMPLETE")
    return True


# ================================================================
# ✅ ENTRY POINT
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge real + synthetic datasets for extended training"
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default=DEFAULT_REAL_DIR,
        help="Directory with real images",
    )
    parser.add_argument(
        "--real-labels",
        type=str,
        default=DEFAULT_REAL_LABELS,
        help="CSV with real labels",
    )
    parser.add_argument(
        "--gan-root",
        type=str,
        default=DEFAULT_GAN_ROOT,
        help="Root of GAN output (contains G1/, G2/, G3/)",
    )
    parser.add_argument(
        "--extended-dir",
        type=str,
        default=DEFAULT_EXTENDED_DIR,
        help="Output directory for merged images",
    )
    parser.add_argument(
        "--extended-labels",
        type=str,
        default=DEFAULT_EXTENDED_LABELS,
        help="Output CSV for merged labels",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        default="copy",
        choices=["copy", "hardlink", "symlink"],
        help="How to transfer files (copy is safest)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually copy files, just report",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    success = merge_datasets(
        real_dir=args.real_dir,
        real_labels=args.real_labels,
        gan_root=args.gan_root,
        extended_dir=args.extended_dir,
        extended_labels=args.extended_labels,
        copy_mode=args.copy_mode,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()