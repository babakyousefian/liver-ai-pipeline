# core/security/image_guard.py
# -*- coding: utf-8 -*-
"""
Image Guard v4.0 — STRICT medical image validation

KEY CHANGES:
- Heuristic gate now REJECTS clear non-medical images
- Adds texture-energy check (rejects trees, textures, nature)
- Adds histogram-shape check (rejects uniform-color images)
- Adds edge-structure check (rejects random-edge images like trees)
- Runs BEFORE preprocessing (on raw image)
"""

import os
import math
import pickle
from typing import Optional, Dict, Tuple, List

try:
    import cv2
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.covariance import EmpiricalCovariance
    HAS_CV2 = True
    HAS_SKLEARN = True
except ImportError:
    HAS_CV2 = False
    HAS_SKLEARN = False


# ================================================================
# Configuration
# ================================================================

MIN_LIVER_SCORE = 0.40
MIN_MODEL_CONFIDENCE = 0.2

PCA_COMPONENTS = 23
KNN_K = 7
REJECT_SCORE = 0.02

_HERE = os.path.dirname(os.path.abspath(__file__))
LIVER_REFERENCE_DIR = os.path.join(_HERE, "liver_reference")
CACHE_DIR = os.path.join(_HERE, ".cache")

_BUNDLE_MODEL_FILE = os.path.join(_HERE, "models", "liver_model.pkl")
_DEV_CACHE_FILE = os.path.join(CACHE_DIR, "liver_model.pkl")

MODEL_CACHE_FILE = _BUNDLE_MODEL_FILE
MODEL_CACHE_VERSION = 7

FEATURE_SIZE = (256, 256)
MAX_REFERENCE_IMAGES = 3000

_ALLOW_TRAINING = os.environ.get("LIVER_GUARD_ALLOW_TRAIN", "0") == "1"


# ================================================================
# ✅ NEW: Strong heuristic gate (rejects non-medical images)
# ================================================================

def _remove_color_labels_for_metrics(bgr: np.ndarray) -> np.ndarray:
    """
    حذف لیبل‌های رنگی کوچک برای محاسبه‌ی متریک.
    فقط لیبل‌های 20-800 پیکسل با اشباع بالا را حذف می‌کند.
    """
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 80, 100]),
            np.array([180, 255, 255]),
        )
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean_mask = np.zeros_like(mask)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if 20 <= area <= 800:
                clean_mask[labels == i] = 255

        if clean_mask.sum() == 0:
            return bgr

        return cv2.inpaint(bgr, clean_mask, 3, cv2.INPAINT_TELEA)
    except Exception:
        return bgr

def _saturation_score(bgr: np.ndarray) -> float:
    """Low score = too colorful (not ultrasound)."""
    try:
        # ✅ حذف لیبل‌های رنگی قبل از محاسبه
        bgr_clean = _remove_color_labels_for_metrics(bgr)

        gray = cv2.cvtColor(bgr_clean, cv2.COLOR_BGR2GRAY)
        tissue = gray > 15
        if tissue.sum() < 100:
            return 0.5
        hsv = cv2.cvtColor(bgr_clean, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        mean_sat = float(sat[tissue].mean())
        if mean_sat < 0.15:
            return 1.0
        elif mean_sat < 0.25:
            return 0.85
        elif mean_sat < 0.40:
            return 0.50
        elif mean_sat < 0.55:
            return 0.20
        else:
            return 0.0
    except Exception:
        return 0.5


def _brightness_score(bgr: np.ndarray) -> float:
    """Ultrasound has moderate brightness."""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tissue = gray > 15
        if tissue.sum() < 100:
            return 0.5
        vals = gray[tissue].astype(np.float32)
        mean_b = float(vals.mean()) / 255.0
        if 0.05 <= mean_b <= 0.85:
            return 1.0
        elif 0.02 <= mean_b < 0.05 or 0.85 < mean_b <= 0.95:
            return 0.5
        else:
            return 0.1
    except Exception:
        return 0.5


def _grayscale_score(bgr: np.ndarray) -> float:
    """Ultrasound is grayscale — no strong color tint."""
    try:
        # ✅ حذف لیبل‌های رنگی قبل از محاسبه
        bgr_clean = _remove_color_labels_for_metrics(bgr)

        gray = cv2.cvtColor(bgr_clean, cv2.COLOR_BGR2GRAY)
        tissue = gray > 15
        if tissue.sum() < 100:
            return 0.5
        b, g, r = cv2.split(bgr_clean.astype(np.float32))
        mr = float(r[tissue].mean())
        mg = float(g[tissue].mean())
        mb = float(b[tissue].mean())
        mean_all = (mr + mg + mb) / 3.0 + 1e-6
        diff = (abs(mr - mg) + abs(mg - mb) + abs(mb - mr)) / 3.0
        color_ratio = diff / mean_all
        if color_ratio < 0.05:
            return 1.0
        elif color_ratio < 0.10:
            return 0.85
        elif color_ratio < 0.20:
            return 0.55
        elif color_ratio < 0.35:
            return 0.20
        else:
            return 0.0
    except Exception:
        return 0.5


def _edge_density_score(bgr: np.ndarray) -> float:
    """نسخه نرم‌تر."""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tissue = gray > 15
        if tissue.sum() < 100:
            return 0.5
        edges = cv2.Canny(gray, 50, 150)
        density = float((edges > 0)[tissue].mean())
        # ✅ بازه‌ی وسیع‌تر
        if 0.02 <= density <= 0.30:      # ← از 0.05-0.ResNet18 به 0.02-0.30
            return 1.0
        elif 0.005 <= density < 0.02 or 0.30 < density <= 0.45:
            return 0.6
        elif 0.45 < density <= 0.60:
            return 0.25
        else:
            return 0.10
    except Exception:
        return 0.5


def _extract_fan_roi_for_features(bgr: np.ndarray) -> np.ndarray:
    """
    استخراج ROI بادبزن برای feature extraction.
    این کار تطابق بین training (روی reference های کراپ‌شده) و
    inference (روی تصاویر با هدر) را تضمین می‌کند.
    """
    try:
        h, w = bgr.shape[:2]
        if h < 50 or w < 50:
            return bgr

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tissue_mask = (gray > 25).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask, 8)
        if num <= 1:
            return bgr

        total_area = w * h
        best_idx, best_area = -1, 0
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < total_area * 0.10:
                continue
            if area > best_area:
                best_area = area
                best_idx = i

        if best_idx == -1:
            return bgr

        x = stats[best_idx, cv2.CC_STAT_LEFT]
        y = stats[best_idx, cv2.CC_STAT_TOP]
        bw = stats[best_idx, cv2.CC_STAT_WIDTH]
        bh = stats[best_idx, cv2.CC_STAT_HEIGHT]

        roi = bgr[y:y + bh, x:x + bw].copy()
        roi_mask = (labels == best_idx).astype(np.uint8)
        roi_mask = roi_mask[y:y + bh, x:x + bw]

        # پس‌زمینه‌ی سیاه را با میانگین بافت پر کن
        bg = roi_mask == 0
        if bg.sum() > 0 and (~bg).sum() > 0:
            mean_color = roi[~bg].mean(axis=0)
            roi[bg] = mean_color.astype(np.uint8)

        return roi
    except Exception:
        return bgr

def _histogram_shape_score(bgr: np.ndarray) -> float:
    """نسخه نرم‌تر برای تصاویر با پس‌زمینه‌ی سیاه و لیبل."""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tissue = gray > 15
        if tissue.sum() < 100:
            return 0.5
        hist = np.histogram(gray[tissue], bins=256, range=(0, 256))[0].astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        mid_ratio = float(hist[40:230].sum())   # ← بازه‌ی وسیع‌تر
        extreme_ratio = float(hist[0:25].sum() + hist[240:256].sum())
        # ✅ آستانه‌های نرم‌تر
        if mid_ratio > 0.50 and extreme_ratio < 0.30:  # ← از 0.70/0.15
            return 1.0
        elif mid_ratio > 0.35 and extreme_ratio < 0.45:  # ← از 0.55/0.25
            return 0.75
        elif mid_ratio > 0.20 and extreme_ratio < 0.60:  # ← از 0.40/0.40
            return 0.45
        else:
            return 0.20
    except Exception:
        return 0.5


def _texture_energy_score(bgr: np.ndarray) -> float:
    """نسخه نرم‌تر برای بافت کبد چرب (که ناهموارتر از کبد سالم است)."""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tissue = gray > 15
        if tissue.sum() < 100:
            return 0.5
        gray_f = gray.astype(np.float32)
        mean = cv2.blur(gray_f, (15, 15))
        mean_sq = cv2.blur(gray_f ** 2, (15, 15))
        variance = np.maximum(mean_sq - mean ** 2, 0)
        local_var = float(variance[tissue].mean())
        # ✅ آستانه‌های نرم‌تر (بافت چرب = واریانس بالاتر)
        if local_var < 800:       # ← از 500 به 800
            return 1.0
        elif local_var < 2500:    # ← از 1500 به 2500
            return 0.85
        elif local_var < 6000:    # ← از 4000 به 6000
            return 0.60
        elif local_var < 15000:   # ← از 10000 به 15000
            return 0.35
        else:
            return 0.15
    except Exception:
        return 0.5

def _small_color_label_ratio(bgr: np.ndarray) -> float:
    """
    نسبت پیکسل‌های رنگی کوچک (لیبل) به کل تصویر.
    اگر کمتر از ۲٪ باشد → قابل قبول (لیبل قابل حذف است).
    اگر بیشتر از ۵٪ باشد → تصویر رنگی است (طبیعت، عکس).
    """
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 80, 100]),
            np.array([180, 255, 255]),
        )
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        small_label_pixels = 0
        total = mask.size
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            # ✅ فقط لیبل‌های کوچک (20-800 px)
            if 20 <= area <= 800:
                small_label_pixels += area
        return small_label_pixels / float(total)
    except Exception:
        return 0.0


def _heuristic_gate(bgr: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
    """
    v4.4 — نرم‌تر برای تصاویر واقعی با لیبل.

    تغییرات:
    - Hard rejections فقط برای موارد بسیار واضح
    - وزن histogram و texture کاهش یافت
    - آستانه‌ها نرم‌تر شدند
    """
    metrics = {
        "saturation": _saturation_score(bgr),
        "brightness": _brightness_score(bgr),
        "grayscale": _grayscale_score(bgr),
        "edges": _edge_density_score(bgr),
        "histogram": _histogram_shape_score(bgr),
        "texture": _texture_energy_score(bgr),
    }
    weights = {
        "saturation": 0.25,  # ← بیشتر (چون مهم‌ترین نشانه)
        "brightness": 0.10,
        "grayscale": 0.30,  # ← بیشتر
        "edges": 0.15,
        "histogram": 0.10,  # ← کمتر (نرم‌تر)
        "texture": 0.10,  # ← کمتر (نرم‌تر)
    }
    score = sum(metrics[k] * weights[k] for k in weights)

    # ✅ HARD REJECTIONS — فقط واضح‌ترین‌ها
    # (آستانه‌ها نرم‌تر شدند)
    if metrics["saturation"] < 0.15:  # ← از 0.30 به 0.15
        return False, score, metrics
    if metrics["grayscale"] < 0.15:  # ← از 0.30 به 0.15
        return False, score, metrics
    if metrics["edges"] < 0.10:  # ← از 0.20 به 0.10
        return False, score, metrics
    if metrics["texture"] < 0.08:  # ← از 0.15 به 0.08
        return False, score, metrics
    # ❌ histogram hard rejection حذف شد (خیلی سخت‌گیرانه بود)

    # ✅ Final threshold
    is_medical = score >= 0.45  # ← از 0.55 به 0.45
    return is_medical, score, metrics


# ================================================================
# Feature extraction (same as before, no L2 norm)
# ================================================================

def _extract_roi(bgr: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return bgr

        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        tissue_mask = (labels == largest).astype(np.uint8) * 255
        tissue_ratio = float((tissue_mask > 0).sum()) / float(tissue_mask.size)
        if tissue_ratio < 0.10 or tissue_ratio > 0.95:
            return bgr

        x = stats[largest, cv2.CC_STAT_LEFT]
        y = stats[largest, cv2.CC_STAT_TOP]
        w = stats[largest, cv2.CC_STAT_WIDTH]
        h = stats[largest, cv2.CC_STAT_HEIGHT]

        roi = bgr[y:y + h, x:x + w]
        m = tissue_mask[y:y + h, x:x + w]

        result = roi.copy()
        bg_mask = (m == 0)
        if bg_mask.sum() > 0 and (~bg_mask).sum() > 0:
            mean_color = roi[~bg_mask].mean(axis=0)
            result[bg_mask] = mean_color.astype(np.uint8)

        return result
    except Exception:
        return bgr


def _tissue_gray(bgr: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = gray > 15
        if mask.sum() < 100:
            return gray.flatten()
        return gray[mask]
    except Exception:
        return bgr.flatten()


def _compute_features(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    v5.0 — با ویژگی‌های منحصربه‌فرد سونوگرافی.

    ویژگی‌های جدید:
    - Speckle noise pattern (مخصوص ultrasound)
    - Depth-based brightness profile
    - Tissue-to-background ratio
    - Homogeneity map
    """
    if not HAS_CV2:
        return None
    try:
        roi = _extract_roi(bgr)
        if roi.size == 0:
            roi = bgr
        resized = cv2.resize(roi, FEATURE_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        tissue_mask = gray > 15
        if tissue_mask.sum() < 500:
            return None

        feats = []

        # ---------- 1. Intensity histogram (64 bins) ----------
        hist = cv2.calcHist([gray], [0], tissue_mask.astype(np.uint8),
                            [64], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.append(hist)

        # ---------- 2. LBP histogram (10 bins) ----------
        try:
            lbp = _local_binary_pattern(gray, radius=1, n_points=8)
            lbp_hist = np.histogram(lbp[tissue_mask], bins=10, range=(0, 256))[0].astype(np.float32)
            lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
            feats.append(lbp_hist)
        except Exception:
            feats.append(np.zeros(10, dtype=np.float32))

        # ---------- 3. Gradient histogram (32 bins) ----------
        try:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag_hist = np.histogram(mag[tissue_mask], bins=32, range=(0, 255))[0].astype(np.float32)
            mag_hist = mag_hist / (mag_hist.sum() + 1e-6)
            feats.append(mag_hist)
        except Exception:
            feats.append(np.zeros(32, dtype=np.float32))

        # ---------- 4. FFT radial profile (16 bins) ----------
        try:
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            hh, ww = magnitude.shape
            cy, cx = hh // 2, ww // 2
            Y, X = np.ogrid[:hh, :ww]
            R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.int32)
            max_r = min(cx, cy)
            radial = np.zeros(16, dtype=np.float32)
            for i in range(16):
                r0 = int(i * max_r / 16)
                r1 = int((i + 1) * max_r / 16)
                if r1 <= r0:
                    continue
                ring = (R >= r0) & (R < r1)
                if ring.sum() > 0:
                    radial[i] = float(magnitude[ring].mean())
            radial = radial / (radial.sum() + 1e-6)
            feats.append(radial)
        except Exception:
            feats.append(np.zeros(16, dtype=np.float32))

        # ---------- 5. Statistical moments (5 values) ----------
        try:
            tissue_vals = gray[tissue_mask].astype(np.float32)
            moments = np.array([
                float(tissue_vals.mean()) / 255.0,
                float(tissue_vals.std()) / 128.0,
                float(np.percentile(tissue_vals, 10)) / 255.0,
                float(np.percentile(tissue_vals, 90)) / 255.0,
                float(np.median(tissue_vals)) / 255.0,
            ], dtype=np.float32)
            feats.append(moments)
        except Exception:
            feats.append(np.zeros(5, dtype=np.float32))

        # ---------- 6. Color histogram (32 bins) ----------
        try:
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            color_hist = cv2.calcHist([hsv], [0, 1], tissue_mask.astype(np.uint8),
                                      [8, 4], [0, 180, 0, 256]).flatten()
            color_hist = color_hist / (color_hist.sum() + 1e-6)
            feats.append(color_hist.astype(np.float32))
        except Exception:
            feats.append(np.zeros(32, dtype=np.float32))

        # ---------- 7. Texture features (4 values) ----------
        try:
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            energy = float(np.mean(grad_x ** 2 + grad_y ** 2)) / 10000.0
            contrast = float(np.std(gray)) / 128.0
            homogeneity = 1.0 / (1.0 + float(np.var(gray)) / 1000.0)
            if gray.size > 1:
                correlation = float(np.corrcoef(
                    gray.flatten(),
                    np.roll(gray, 1, axis=1).flatten()
                )[0, 1])
            else:
                correlation = 0.0
            glcm = np.array([energy, contrast, homogeneity, correlation], dtype=np.float32)
            feats.append(glcm)
        except Exception:
            feats.append(np.zeros(4, dtype=np.float32))

        # ============================================================
        # ✅ NEW FEATURES (unique to ultrasound liver)
        # ============================================================

        # ---------- 8. Speckle noise pattern (8 values) ----------
        # Ultrasound has multiplicative speckle noise — unique signature
        try:
            gray_f = gray.astype(np.float32)
            # Speckle = signal / local_mean
            local_mean = cv2.blur(gray_f, (7, 7)) + 1e-6
            speckle = gray_f / local_mean

            # Statistics of speckle (should be near 1.0 mean, ~0.3 std for ultrasound)
            tissue_speckle = speckle[tissue_mask]
            speckle_feats = np.array([
                float(tissue_speckle.mean()) / 2.0,  # ~0.5
                float(tissue_speckle.std()),  # ~0.3
                float(np.percentile(tissue_speckle, 10)),  # ~0.7
                float(np.percentile(tissue_speckle, 50)),  # ~1.0
                float(np.percentile(tissue_speckle, 90)),  # ~1.3
                # Skewness (ultrasound: positive skew)
                float(((tissue_speckle - tissue_speckle.mean()) ** 3).mean() / (tissue_speckle.std() ** 3 + 1e-6)),
                # Kurtosis (ultrasound: higher than normal)
                float(
                    ((tissue_speckle - tissue_speckle.mean()) ** 4).mean() / (tissue_speckle.std() ** 4 + 1e-6) - 3.0),
                # Speckle autocorrelation (ultrasound: high = smooth graininess)
                float(np.corrcoef(tissue_speckle[:-1], tissue_speckle[1:])[0, 1]) if len(tissue_speckle) > 1 else 0.0,
            ], dtype=np.float32)
            feats.append(speckle_feats)
        except Exception:
            feats.append(np.zeros(8, dtype=np.float32))

        # ---------- 9. Depth-based brightness profile (8 values) ----------
        # Ultrasound: brightness decreases with depth (attenuation)
        try:
            hh, ww = gray.shape
            depth_bins = 8
            depth_profile = np.zeros(depth_bins, dtype=np.float32)
            bin_height = hh // depth_bins

            for i in range(depth_bins):
                y0 = i * bin_height
                y1 = (i + 1) * bin_height if i < depth_bins - 1 else hh
                strip = gray[y0:y1, :]
                strip_mask = strip > 15
                if strip_mask.sum() > 10:
                    depth_profile[i] = float(strip[strip_mask].mean()) / 255.0

            # Normalize (relative to first bin)
            if depth_profile[0] > 0.01:
                depth_profile = depth_profile / depth_profile[0]

            feats.append(depth_profile)
        except Exception:
            feats.append(np.zeros(8, dtype=np.float32))

        # ---------- 10. Tissue-to-background ratio (2 values) ----------
        try:
            total_pixels = gray.size
            tissue_pixels = int(tissue_mask.sum())
            tissue_ratio = tissue_pixels / float(total_pixels)

            # Border blackness (ultrasound has black border)
            border_size = min(20, gray.shape[0] // 10, gray.shape[1] // 10)
            border = np.concatenate([
                gray[:border_size, :].flatten(),
                gray[-border_size:, :].flatten(),
                gray[:, :border_size].flatten(),
                gray[:, -border_size:].flatten(),
            ])
            border_blackness = float((border < 30).mean())  # fraction of black border

            feats.append(np.array([tissue_ratio, border_blackness], dtype=np.float32))
        except Exception:
            feats.append(np.zeros(2, dtype=np.float32))

        # ---------- 11. Homogeneity map (4 values) ----------
        # Ultrasound liver = homogeneous; MRI/CT = structured
        try:
            # Local variance
            gray_f = gray.astype(np.float32)
            mean_local = cv2.blur(gray_f, (11, 11))
            mean_sq_local = cv2.blur(gray_f ** 2, (11, 11))
            local_var = np.maximum(mean_sq_local - mean_local ** 2, 0)

            tissue_var = local_var[tissue_mask]
            if len(tissue_var) > 0:
                homogeneity_feats = np.array([
                    float(tissue_var.mean()) / 1000.0,
                    float(tissue_var.std()) / 1000.0,
                    float(np.percentile(tissue_var, 50)) / 1000.0,
                    float(np.percentile(tissue_var, 90)) / 1000.0,
                ], dtype=np.float32)
            else:
                homogeneity_feats = np.zeros(4, dtype=np.float32)
            feats.append(homogeneity_feats)
        except Exception:
            feats.append(np.zeros(4, dtype=np.float32))

        # ---------- Concatenate ----------
        vec = np.concatenate(feats).astype(np.float32)
        return vec
    except Exception as e:
        print(f"⚠️ Feature extraction failed: {e}")
        return None


def _local_binary_pattern(gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    try:
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        for i, ang in enumerate(angles):
            dx = int(round(radius * np.cos(ang)))
            dy = int(round(radius * np.sin(ang)))
            shifted = np.zeros_like(gray)
            ys = slice(max(0, -dy), h - max(0, dy))
            xs = slice(max(0, -dx), w - max(0, dx))
            yd = slice(max(0, dy), h - max(0, -dy))
            xd = slice(max(0, dx), w - max(0, -dx))
            shifted[ys, xs] = gray[yd, xd]
            bit = (shifted >= gray).astype(np.uint8)
            lbp |= (bit << i)
        return lbp
    except Exception:
        return np.zeros_like(gray, dtype=np.uint8)


# ================================================================
# LiverModel (same as before)
# ================================================================

class LiverModel:
    def __init__(self):
        self.pca = None
        self.cov = None
        self.mean = None
        self.threshold = 0.0
        self.n_samples = 0
        self.reduced_features = None
        self.dist_stats = None

    def fit(self, features: List[np.ndarray]):
        if not features or len(features) < 10:
            raise ValueError("Need at least 10 reference features")

        X = np.array(features, dtype=np.float32)
        n_components = min(PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])

        self.pca = PCA(n_components=n_components, whiten=True)
        X_reduced = self.pca.fit_transform(X)
        self.reduced_features = X_reduced.astype(np.float32)

        self.mean = np.mean(X_reduced, axis=0).astype(np.float32)
        self.cov = EmpiricalCovariance()
        self.cov.fit(X_reduced)

        distances = self.cov.mahalanobis(X_reduced)

        p50 = float(np.percentile(distances, 50))
        p95 = float(np.percentile(distances, 95))  # ← جدید
        p99 = float(np.percentile(distances, 99))  # ← فقط برای گزارش

        spread = max(p99 - p50, 1e-3)
        self.threshold = p95 + 0.75 * spread

        self.dist_stats = {
            "min": float(distances.min()),
            "p25": float(np.percentile(distances, 25)),
            "p50": p50,
            "p75": float(np.percentile(distances, 75)),
            "p95": float(np.percentile(distances, 95)),
            "p99": p99,
            "max": float(distances.max()),
        }
        self.n_samples = len(features)

        print(f"✅ Liver model v4.0 trained on {self.n_samples} images")
        print(f"   PCA components:  {n_components}")
        print(f"   Threshold:       {self.threshold:.3f}  (p50={p50:.2f}, p99={p99:.2f})")

    def score(self, feature: np.ndarray) -> float:
        if self.pca is None or self.cov is None:
            return 0.5
        try:
            x_reduced = self.pca.transform(feature.reshape(1, -1))
            x_reduced = x_reduced.astype(np.float32)

            dist = float(self.cov.mahalanobis(x_reduced)[0])

            if self.reduced_features is not None and len(self.reduced_features) >= KNN_K:
                diffs = self.reduced_features - x_reduced
                dists_sq = np.sum(diffs * diffs, axis=1)
                nearest = np.partition(dists_sq, KNN_K)[:KNN_K]
                knn_dist = float(np.sqrt(np.mean(nearest)))
            else:
                knn_dist = dist

            combined = 0.6 * dist + 0.4 * knn_dist

            scale = max(self.threshold * 1.0, 1e-3)
            z = (combined - self.threshold) / scale
            z = max(-50.0, min(50.0, z))
            score = 1.0 / (1.0 + math.exp(z))
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({
                    "version": MODEL_CACHE_VERSION,
                    "pca": self.pca,
                    "cov": self.cov,
                    "mean": self.mean,
                    "threshold": self.threshold,
                    "n_samples": self.n_samples,
                    "reduced_features": self.reduced_features,
                    "dist_stats": self.dist_stats,
                }, f)
            print(f"💾 Model saved: {path}")
        except Exception as e:
            print(f"⚠️ Failed to save model: {e}")

    @classmethod
    def load(cls, path: str) -> Optional["LiverModel"]:
        try:
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data.get("version") != MODEL_CACHE_VERSION:
                return None
            model = cls()
            model.pca = data["pca"]
            model.cov = data["cov"]
            model.mean = data["mean"]
            model.threshold = data["threshold"]
            model.n_samples = data["n_samples"]
            model.reduced_features = data.get("reduced_features")
            model.dist_stats = data.get("dist_stats", {})
            print(f"📦 Model loaded from cache ({model.n_samples} samples, v4.0)")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            return None


_LIVER_MODEL: Optional[LiverModel] = None


def _get_liver_model() -> Optional[LiverModel]:
    global _LIVER_MODEL
    if _LIVER_MODEL is not None:
        return _LIVER_MODEL

    for path in (_BUNDLE_MODEL_FILE, _DEV_CACHE_FILE):
        if os.path.exists(path):
            m = LiverModel.load(path)
            if m is not None:
                _LIVER_MODEL = m
                return m

    if not _ALLOW_TRAINING:
        print("⚠️ Liver model pkl not found — guard in FAIL-OPEN mode.")
        return None

    print("🔄 [DEV] Training liver model from references...")
    features = _load_reference_features(LIVER_REFERENCE_DIR, max_images=MAX_REFERENCE_IMAGES)
    if not features or len(features) < 10:
        return None

    try:
        model = LiverModel()
        model.fit(features)
        model.save(_BUNDLE_MODEL_FILE)
        model.save(_DEV_CACHE_FILE)
        _LIVER_MODEL = model
        return model
    except Exception as e:
        print(f"⚠️ Training failed: {e}")
        return None


def _load_reference_features(directory: str, max_images: int = MAX_REFERENCE_IMAGES) -> List[np.ndarray]:
    features = []
    if not os.path.isdir(directory):
        return features
    files = sorted([f for f in os.listdir(directory)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))])

    if len(files) > max_images:
        import random
        random.seed(42)
        files = random.sample(files, max_images)

    for fname in files:
        path = os.path.join(directory, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        fvec = _compute_features(img)
        if fvec is not None:
            features.append(fvec)

    print(f"📚 Loaded {len(features)} feature vectors from {len(files)} files")
    return features


# ================================================================
# Public API
# ================================================================

def analyze_image(image_path: str) -> Dict:
    """
    v5.0 — Pure one-class anomaly detection (بدون ultrasound verifier).
    """
    result = {"is_liver_like": True, "score": 1.0,
              "metrics": {}, "reason": ""}

    if not HAS_CV2 or not HAS_SKLEARN:
        result["reason"] = "Required libraries not available"
        return result

    if not image_path or not os.path.isfile(image_path):
        result["reason"] = "File not found — fail-open"
        return result

    try:
        img = cv2.imread(image_path)
        if img is None:
            result["reason"] = "Could not read image — fail-open"
            return result

        h, w = img.shape[:2]
        if max(h, w) > 800:
            s = 800.0 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
            h, w = img.shape[:2]

        # ✅ Clean colored labels
        img_clean = _remove_color_labels_for_metrics(img)

        # ✅ Heuristic gate
        is_medical_like, heuristic_score, heuristic_metrics = _heuristic_gate(img_clean)
        result["metrics"].update(heuristic_metrics)
        result["metrics"]["heuristic_score"] = heuristic_score

        if not is_medical_like:
            result.update(is_liver_like=False, score=heuristic_score)
            worst = min(heuristic_metrics.items(), key=lambda kv: kv[1])
            reason_map = {
                "saturation": "Image is too colorful (not grayscale ultrasound)",
                "grayscale": "Image has strong color tint (not ultrasound)",
                "edges": "Image has wrong edge density (not ultrasound)",
                "texture": "Image has too much local texture (not ultrasound)",
                "histogram": "Histogram shape is not consistent with ultrasound",
                "brightness": "Brightness out of range",
            }
            result["reason"] = reason_map.get(worst[0],
                                              "Not a medical ultrasound image")
            return result

        # ✅ Liver model
        model = _get_liver_model()
        if model is None:
            result["is_liver_like"] = True
            result["score"] = heuristic_score
            result["reason"] = "No liver model — heuristic-only (fail-open)"
            return result

        test_vec = _compute_features(img_clean)
        if test_vec is None:
            result["is_liver_like"] = True
            result["reason"] = "Could not extract features — fail-open"
            return result

        liver_score = model.score(test_vec)
        result["metrics"]["liver_score"] = liver_score

        # ✅ v5.0: REJECT_SCORE
        if liver_score < REJECT_SCORE:
            result.update(
                is_liver_like=False,
                score=liver_score,
                reason=(
                    f"Does not match liver pattern "
                    f"(score={liver_score:.3f} < {REJECT_SCORE})"
                ),
            )
            return result

        final = 0.2 * heuristic_score + 0.8 * liver_score
        result["score"] = final
        result["metrics"]["final_score"] = final
        result["is_liver_like"] = final >= MIN_LIVER_SCORE
        if not result["is_liver_like"]:
            result["reason"] = (
                f"Score {final:.2f} below {MIN_LIVER_SCORE} "
                f"(heuristic={heuristic_score:.2f}, liver={liver_score:.2f})"
            )
        return result

    except Exception as e:
        result["is_liver_like"] = True
        result["reason"] = f"Error (fail-open): {e}"
        return result


def is_confident_result(image_path: str, model_confidence: float,
                        all_probs: Optional[Dict] = None) -> Tuple[bool, str, Dict]:
    metrics = {}
    img_result = analyze_image(image_path)
    metrics.update(img_result.get("metrics", {}))
    metrics["image_score"] = img_result.get("score", 1.0)

    if not img_result["is_liver_like"]:
        return (False, img_result.get("reason", "Not a liver ultrasound"), metrics)

    if model_confidence < MIN_MODEL_CONFIDENCE:
        return (False, f"Model confidence too low ({model_confidence*100:.1f}%)", metrics)

    return (True, "", metrics)


def reload_model():
    global _LIVER_MODEL
    _LIVER_MODEL = None
    try:
        if os.path.exists(MODEL_CACHE_FILE):
            os.remove(MODEL_CACHE_FILE)
    except Exception:
        pass
    _get_liver_model()
