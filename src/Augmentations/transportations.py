import os
import csv
import shutil
from collections import deque
from src.pipeLine.ConfigurationAlgorithms.withoutFineTuner import CROPPED_DIR, LABELS_CSV

# =====================================================
# =====================================================
FINAL_REPORTS_DIR = os.path.dirname(CROPPED_DIR)  # → finalReports/
BACKUP_DIR = os.path.join(FINAL_REPORTS_DIR, "croppedBackups")

CLASS_FOLDERS = {
    "NL": os.path.join(BACKUP_DIR, "nl"),
    "G1": os.path.join(BACKUP_DIR, "g1"),
    "G2": os.path.join(BACKUP_DIR, "g2"),
    "G3": os.path.join(BACKUP_DIR, "g3"),
}

for folder in CLASS_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

print(f"✅ Backup directory: {BACKUP_DIR}")
print(f"✅ Class folders created")

# =====================================================
# =====================================================
nll = deque()
g1 = deque()
g2 = deque()
g3 = deque()

with open(LABELS_CSV, mode="r", encoding="utf-8") as f:
    labels = csv.reader(f, delimiter=",")
    header = next(labels, None)

    for row in labels:
        if len(row) < 2:
            continue

        img_name, img_label = row[0].strip(), row[1].strip()

        if img_label == "NL":
            nll.append(img_name)
        elif img_label == "G1":
            g1.append(img_name)
        elif img_label == "G2":
            g2.append(img_name)
        elif img_label == "G3":
            g3.append(img_name)
        # ✅ بقیه (UNKNOWN یا ...) نادیده گرفته میشن

print(f"\n📊 Label counts:")
print(f"   NL: {len(nll)}")
print(f"   G1: {len(g1)}")
print(f"   G2: {len(g2)}")
print(f"   G3: {len(g3)}")
print(f"   Total: {len(nll) + len(g1) + len(g2) + len(g3)}")

# =====================================================
# 3. کپی تصاویر به پوشه‌های مربوطه
# =====================================================
copied = {"NL": 0, "G1": 0, "G2": 0, "G3": 0}
missing = {"NL": 0, "G1": 0, "G2": 0, "G3": 0}
skipped = 0

# ✅ نگاشت label → deque و پوشه
label_map = {
    "NL": (nll, CLASS_FOLDERS["NL"]),
    "G1": (g1, CLASS_FOLDERS["G1"]),
    "G2": (g2, CLASS_FOLDERS["G2"]),
    "G3": (g3, CLASS_FOLDERS["G3"]),
}

for label_name, (deq, dest_folder) in label_map.items():
    print(f"\n📂 Processing {label_name} ({len(deq)} images)...")

    for img_name in deq:
        src = os.path.join(CROPPED_DIR, img_name)
        dst = os.path.join(dest_folder, img_name)

        # ✅ چک کن فایل مبدأ وجود داره
        if not os.path.exists(src):
            # ✅ شاید اسم فایل بدون پسوند ذخیره شده باشه
            # جستجو با پسوندهای رایج
            found = False
            base_name = os.path.splitext(img_name)[0]
            for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".bmp", ".tiff"]:
                test_src = os.path.join(CROPPED_DIR, base_name + ext)
                if os.path.exists(test_src):
                    src = test_src
                    dst = os.path.join(dest_folder, base_name + ext)
                    found = True
                    break

            if not found:
                print(f"   ⚠️ Not found: {img_name}")
                missing[label_name] += 1
                continue

        # ✅ اگه فایل قبلاً کپی شده، skip کن
        if os.path.exists(dst):
            skipped += 1
            continue

        # ✅ کپی فایل
        try:
            shutil.copy2(src, dst)  # copy2 → حفظ metadata
            copied[label_name] += 1
        except Exception as e:
            print(f"   ❌ Error copying {img_name}: {e}")
            missing[label_name] += 1

# =====================================================
# 4. گزارش نهایی
# =====================================================
print("\n" + "=" * 60)
print("📊 COPY SUMMARY")
print("=" * 60)

total_copied = 0
total_missing = 0

for label_name in ["NL", "G1", "G2", "G3"]:
    print(f"   {label_name}:")
    print(f"      ✅ Copied:  {copied[label_name]}")
    print(f"      ⚠️ Missing: {missing[label_name]}")
    print(f"      📁 Folder:  {CLASS_FOLDERS[label_name]}")
    total_copied += copied[label_name]
    total_missing += missing[label_name]

print(f"\n   ✅ Total copied:  {total_copied}")
print(f"   ⚠️ Total missing: {total_missing}")
print(f"   ⏭️ Skipped (already exist): {skipped}")
print("=" * 60)

print(f"\n🎉 Done! Backup folder: {BACKUP_DIR}")