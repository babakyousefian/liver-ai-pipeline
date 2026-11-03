# generate_new_labels.py

"""
Generate newLabel.csv from augmented dataset in newLiver/

This script scans the newLiver/ directory, extracts the class label
from each filename (format: {label}_{uuid}.png), and creates a CSV
file compatible with the pipeline.

Features:
- Robust filename parsing (regex-based)
- Verification of class distribution
- Detailed statistics
- CSV format matches original labels.csv

Author: Babak Yousefian
Version: 1.0.0
"""

import os
import re
import sys
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


# ================================================================
# CONFIG
# ================================================================

NEW_LIVER_DIR = "./FinalReports/newLiver"                    # ← پوشه‌ی تصاویر augmented
OUTPUT_CSV = "./FinalReports/newLabel.csv"                   # ← فایل خروجی
CLASS_NAMES = ["NL", "G1", "G2", "G3"]        # ← کلاس‌های مجاز

# ✅ Regex: نام فایل باید شروع بشه با یکی از این labelها + underscore
LABEL_PATTERN = re.compile(r"^(NL|G1|G2|G3)_")


# ================================================================
# MAIN LOGIC
# ================================================================

def scan_new_liver_dir(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Scan the newLiver directory and return list of files + class counts.

    Args:
        directory: Path to newLiver directory

    Returns:
        (list_of_filenames, {class_name: count})
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"❌ Directory not found: {directory}\n"
            f"📌 Please make sure the augmentation has been completed."
        )

    # ✅ Collect all image files (files only, not directories)
    all_files = os.listdir(directory)
    image_files = [
        f for f in all_files
        if os.path.isfile(os.path.join(directory, f))   # ← ✅ فقط فایل‌ها
        and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"\n📂 Scanning: {directory}")
    print(f"   Found {len(image_files)} image files")
    print(f"   Found {len(all_files) - len(image_files)} non-image items")

    # ✅ Parse labels from filenames
    valid_files: List[str] = []
    invalid_files: List[str] = []
    class_counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES}

    print(f"\n🔍 Parsing labels from filenames...")
    for filename in tqdm(image_files, desc="Parsing", ncols=100):
        match = LABEL_PATTERN.match(filename)
        if match:
            label = match.group(1)
            valid_files.append(filename)
            class_counts[label] += 1
        else:
            invalid_files.append(filename)

    # ✅ Report results
    print(f"\n📊 Parsing Results:")
    print(f"   Valid files:   {len(valid_files)}")
    print(f"   Invalid files: {len(invalid_files)}")

    if invalid_files:
        print(f"\n⚠️ First 10 invalid filenames (no NL_/G1_/G2_/G3_ prefix):")
        for f in invalid_files[:10]:
            print(f"      - {f}")

    return valid_files, class_counts


def build_dataframe(valid_files: List[str]) -> pd.DataFrame:
    """
    Build DataFrame with liverName and liverLabel columns.

    Args:
        valid_files: List of valid filenames

    Returns:
        DataFrame with liverName, liverLabel
    """
    print(f"\n🔨 Building DataFrame...")

    rows = []
    for filename in tqdm(valid_files, desc="Building CSV", ncols=100):
        label = LABEL_PATTERN.match(filename).group(1)
        rows.append({
            "liverName": filename,
            "liverLabel": label,
        })

    df = pd.DataFrame(rows)

    # ✅ Shuffle rows (so classes are interleaved)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"\n✅ CSV saved to: {output_path}")


def print_statistics(class_counts: Dict[str, int], df: pd.DataFrame) -> None:
    """Print detailed statistics."""
    total = sum(class_counts.values())

    print(f"\n" + "=" * 70)
    print(f"📊 CLASS DISTRIBUTION STATISTICS")
    print(f"=" * 70)
    print(f"\n{'Class':<10} {'Count':<12} {'Percentage':<12} {'Ratio to Min':<15}")
    print("-" * 70)

    min_count = min(class_counts.values())

    for class_name in CLASS_NAMES:
        count = class_counts[class_name]
        pct = (count / total) * 100 if total > 0 else 0
        ratio = count / min_count if min_count > 0 else 0
        print(f"{class_name:<10} {count:<12,} {pct:>10.2f}%  {ratio:>13.2f}x")

    print("-" * 70)
    print(f"{'TOTAL':<10} {total:<12,} {100.0:>10.2f}%")

    # ✅ Balance check
    if min_count > 0:
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count
        print(f"\n⚖️ Imbalance ratio (Max/Min): {imbalance_ratio:.2f}x")
        if imbalance_ratio < 1.5:
            print(f"   ✅ Excellent balance!")
        elif imbalance_ratio < 2.0:
            print(f"   ✅ Good balance")
        elif imbalance_ratio < 3.0:
            print(f"   ⚠️ Moderate imbalance")
        else:
            print(f"   ❌ High imbalance — consider re-augmentation")

    # ✅ Summary
    print(f"\n✅ Total images in CSV: {len(df):,}")
    print(f"✅ Unique labels: {df['liverLabel'].nunique()}")
    print(f"✅ Labels: {sorted(df['liverLabel'].unique())}")
    print("=" * 70)


def verify_output(df: pd.DataFrame) -> None:
    """Quick verification of the output CSV."""
    print(f"\n🔍 Verifying output CSV...")

    # Check required columns
    required_cols = ["liverName", "liverLabel"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"   ❌ Missing columns: {missing}")
        return
    print(f"   ✅ Required columns present: {required_cols}")

    # Check for duplicates
    n_dupes = df["liverName"].duplicated().sum()
    if n_dupes > 0:
        print(f"   ⚠️ {n_dupes} duplicate filenames found")
    else:
        print(f"   ✅ No duplicates")

    # Check for null values
    n_null = df.isnull().sum().sum()
    if n_null > 0:
        print(f"   ⚠️ {n_null} null values found")
    else:
        print(f"   ✅ No null values")

    # Check that all labels are in CLASS_NAMES
    unknown_labels = set(df["liverLabel"].unique()) - set(CLASS_NAMES)
    if unknown_labels:
        print(f"   ❌ Unknown labels: {unknown_labels}")
    else:
        print(f"   ✅ All labels are valid")


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 GENERATE newLabel.csv from Augmented Dataset")
    print("=" * 70)

    # ✅ Step 1: Scan directory
    valid_files, class_counts = scan_new_liver_dir(NEW_LIVER_DIR)

    if len(valid_files) == 0:
        print("\n❌ No valid files found! Aborting.")
        sys.exit(1)

    # ✅ Step 2: Build DataFrame
    df = build_dataframe(valid_files)

    # ✅ Step 3: Save CSV
    save_csv(df, OUTPUT_CSV)

    # ✅ Step 4: Print statistics
    print_statistics(class_counts, df)

    # ✅ Step 5: Verify output
    verify_output(df)

    print(f"\n🎉 DONE! newLabel.csv is ready at: {os.path.abspath(OUTPUT_CSV)}")
    print(f"\n📌 Next step: copy it to finalReports/ and rename as needed")
    print(f"   Or update LABELS_CSV in your pipeline to point to newLabel.csv\n")


if __name__ == "__main__":
    main()
