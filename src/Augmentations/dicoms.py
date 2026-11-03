import os
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # چون PS هستی، بدون GUI
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============ تنظیمات ============
SRC_ROOT = r"/packages/data/DICOM/main/nifti_and_segms/nifti_and_segms"
DST_ROOT = r"/finalReports/pngs"
AXIS = 2                       # محور اسلایس (معمولاً 2 = axial)
KEEP_FRAC = (0.2, 0.8)         # فقط اسلایس‌های وسط
WINDOW_PCT = (1, 99)
SAVE_PREVIEW = True
# =================================

def nii_to_png(nii_path, out_dir, axis=AXIS,
               window_percentile=WINDOW_PCT,
               keep_frac=KEEP_FRAC,
               save_preview=SAVE_PREVIEW):
    try:
        img = nib.load(nii_path)
        data = img.get_fdata().astype(np.float32)
    except Exception as e:
        print(f"   ❌ خطا در خواندن {nii_path}: {e}")
        return 0

    if data.size == 0 or data.max() == data.min():
        print(f"   ⚠️ داده ثابت یا خالی: {nii_path}")
        return 0

    # نرمال‌سازی percentile
    lo, hi = np.percentile(data, window_percentile)
    data = np.clip(data, lo, hi)
    data = (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
    data = data.astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    n_slices = data.shape[axis]

    if n_slices < 3:
        print(f"   ⚠️ اسلایس کافی نداره ({n_slices}): {nii_path}")
        return 0

    start = int(n_slices * keep_frac[0])
    end   = int(n_slices * keep_frac[1])
    count = 0

    for i in range(start, end):
        sl = np.take(data, i, axis=axis)
        # اسلایس کاملاً سیاه رو ذخیره نکن
        if sl.max() < 5:
            continue
        Image.fromarray(sl).save(
            os.path.join(out_dir, f"slice_{i:03d}.png"))
        count += 1

    # preview
    if save_preview:
        idxs = np.linspace(start, end - 1, min(6, end - start)).astype(int)
        fig, axes = plt.subplots(1, len(idxs), figsize=(3*len(idxs), 3))
        if len(idxs) == 1:
            axes = [axes]
        for ax, i in zip(axes, idxs):
            ax.imshow(np.take(data, i, axis=axis), cmap='gray')
            ax.set_title(f"slice {i}")
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "_preview.png"), dpi=100)
        plt.close()

    return count


def process_all(src_root, dst_root):
    # پیدا کردن همه nii.gz
    nii_files = []
    for dirpath, _, filenames in os.walk(src_root):
        for f in filenames:
            if f.endswith(".nii.gz") and not f.startswith("._"):
                nii_files.append(os.path.join(dirpath, f))

    print(f"🔍 {len(nii_files)} فایل nii.gz پیدا شد.\n")

    summary = []
    for nii_path in tqdm(nii_files, desc="Converting", ncols=80):
        rel = os.path.relpath(nii_path, src_root)
        out_dir = os.path.join(dst_root, os.path.splitext(os.path.splitext(rel)[0])[0])

        n = nii_to_png(nii_path, out_dir)
        summary.append((rel, n))

    print("\n=== خلاصه ===")
    total = 0
    for rel, n in summary:
        print(f"  {rel:60s} → {n} اسلایس")
        total += n
    print(f"\n✅ مجموع: {total} تصویر png")


if __name__ == "__main__":
    process_all(SRC_ROOT, DST_ROOT)