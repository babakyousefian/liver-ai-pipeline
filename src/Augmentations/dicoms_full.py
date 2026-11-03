import os
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

DCM_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\packages\data\DICOM\main\dicoms"
DST_ROOT = r"/finalReports/pngs_from_dcm"

def dcm_to_png_array(ds, win_pct=(1, 99)):
    arr = ds.pixel_array.astype(np.float32)

    # multi-frame
    if arr.ndim == 3:
        frames = []
        for i in range(arr.shape[0]):
            frames.append(dcm_to_png_array(arr[i], win_pct))
        return np.stack(frames, axis=0)

    # rescale slope/intercept
    slope = float(getattr(ds, "RescaleSlope", 1))
    inter = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + inter

    lo, hi = np.percentile(arr, win_pct)
    arr = np.clip(arr, lo, hi)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    return arr.astype(np.uint8)


def process_all_dcm(src_root, dst_root):
    dcm_files = []
    for dirpath, _, filenames in os.walk(src_root):
        if "__MACOSX" in dirpath:
            continue
        for f in filenames:
            if f.endswith(".dcm") and not f.startswith("._"):
                dcm_files.append(os.path.join(dirpath, f))

    print(f"🔍 {len(dcm_files)} فایل dcm پیدا شد.\n")

    total = 0
    for dcm_path in tqdm(dcm_files, desc="DCM→PNG", ncols=80):
        try:
            ds = pydicom.dcmread(dcm_path, force=True)
            if not hasattr(ds, "PixelData"):
                continue

            arr = dcm_to_png_array(ds)

            rel = os.path.relpath(dcm_path, src_root)
            out_dir = os.path.join(dst_root, os.path.dirname(rel))
            os.makedirs(out_dir, exist_ok=True)

            base = os.path.splitext(os.path.basename(dcm_path))[0]

            if arr.ndim == 2:
                Image.fromarray(arr).save(os.path.join(out_dir, f"{base}.png"))
                total += 1
            else:  # multi-frame
                for i in range(arr.shape[0]):
                    Image.fromarray(arr[i]).save(
                        os.path.join(out_dir, f"{base}_f{i:03d}.png"))
                    total += 1
        except Exception as e:
            print(f"   ❌ {dcm_path}: {e}")

    print(f"\n✅ مجموع: {total} تصویر png")


if __name__ == "__main__":
    process_all_dcm(DCM_ROOT, DST_ROOT)