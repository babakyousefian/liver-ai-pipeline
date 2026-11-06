# core/security/image_guard.py
# -*- coding: utf-8 -*-
"""
AI Detector for Fatty Liver Disease — v3.0 (Reference-Based Anomaly Detection)

Goal: Accept ONLY liver ultrasound images that match the training set.
Reject EVERYTHING else: fetus, kidney, brain, MRI, CT, nature, space, memes...

Strategy:
    Layer 1 — Heuristic gate (fast reject of non-medical images)
    Layer 2 — Statistical model (PCA + Mahalanobis) trained on YOUR liver images
    Layer 3 — Non-liver reference check (optional, for extra safety)

This uses the 2642 liver images you provided to learn the "liver pattern"
(color, texture, histogram, edges, FFT) without needing deep learning.
"""

import os
import math
import time
import pickle
import hashlib
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
# ✅ Configuration
# ================================================================

MIN_LIVER_SCORE = 0.45         # ← از 0.55 به 0.45 (کمی نرم‌تر)
MIN_MODEL_CONFIDENCE = 0.45    # ← از 0.50 به 0.45

# ✅ NEW: Generalization settings
PCA_COMPONENTS = 15              # ← از 10 به 15 (تعادل بهتر)
KNN_K = 7                        # ← از 5 به 7
REJECT_SCORE = 0.05             # ← از 0.15 به 0.05 (خیلی نرم‌تر — فقط نویز خالص رد بشه)

_HERE = os.path.dirname(os.path.abspath(__file__))
LIVER_REFERENCE_DIR = os.path.join(_HERE, "liver_reference")
NON_LIVER_REFERENCE_DIR = os.path.join(_HERE, "non_liver_reference")
CACHE_DIR = os.path.join(_HERE, ".cache")

# ✅ فایل مدل از پیش آموزش‌دیده — همراه پروژه، نه در .cache
_BUNDLE_MODEL_FILE = os.path.join(_HERE, "models", "liver_model.pkl")
# ✅ فایل قدیمی (dev cache) — فقط به‌عنوان fallback
_DEV_CACHE_FILE = os.path.join(_HERE, ".cache", "liver_model.pkl")

MODEL_CACHE_FILE = _BUNDLE_MODEL_FILE    # ← مسیر ترجیحی
MODEL_CACHE_VERSION = 4

# ابعاد ویژگی‌ها
FEATURE_SIZE = (256, 256)
MAX_REFERENCE_IMAGES = 3000    # حداکثر تصویر برای آموزش

# ================================================================
# ✅ NEW: Env flag — فقط برای تولید pkl (روی سیستم خودتان)
# ================================================================

_ALLOW_TRAINING = os.environ.get("LIVER_GUARD_ALLOW_TRAIN", "0") == "1"


# ================================================================
# ✅ Utility — ROI extraction (fan-shaped)
# ================================================================

def _extract_roi(bgr: np.ndarray) -> np.ndarray:
    """جدا کردن بافت (حذف پس‌زمینه‌ی سیاه سونوگرافی)."""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return bgr

        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        tissue_mask = (labels == largest).astype(np.uint8) * 255

        tissue_ratio = float((tissue_mask > 0).sum()) / float(tissue_mask.size)
        if tissue_ratio < 0.15 or tissue_ratio > 0.98:
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
        mask = gray > 20
        if mask.sum() < 100:
            return gray.flatten()
        return gray[mask]
    except Exception:
        return bgr.flatten()


# ================================================================
# ✅ Feature extraction (rich, robust, discriminative)
# ================================================================

def _compute_features(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    استخراج یک بردار ویژگی غنی از تصویر.
    این بردار شامل:
        - هیستوگرام شدت (64 bin)
        - هیستوگرام LBP (10 bin)
        - هیستوگرام گرادیان (32 bin)
        - پروفایل شعاعی FFT (16 bin)
        - گشتاورهای آماری (5 عدد)
        - هیستوگرام رنگ (32 bin)
        - ویژگی‌های بافت GLCM (4 عدد)
    """
    if not HAS_CV2:
        return None
    try:
        roi = _extract_roi(bgr)
        if roi.size == 0:
            roi = bgr
        resized = cv2.resize(roi, FEATURE_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        tissue_mask = gray > 20
        if tissue_mask.sum() < 500:
            return None

        feats = []

        # 1. Intensity histogram
        hist = cv2.calcHist([gray], [0], tissue_mask.astype(np.uint8),
                            [64], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.append(hist)

        # 2. LBP histogram
        try:
            lbp = _local_binary_pattern(gray, radius=1, n_points=8)
            lbp_hist = np.histogram(lbp[tissue_mask], bins=10, range=(0, 256))[0].astype(np.float32)
            lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
            feats.append(lbp_hist)
        except Exception:
            feats.append(np.zeros(10, dtype=np.float32))

        # 3. Gradient histogram
        try:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag_hist = np.histogram(mag[tissue_mask], bins=32, range=(0, 255))[0].astype(np.float32)
            mag_hist = mag_hist / (mag_hist.sum() + 1e-6)
            feats.append(mag_hist)
        except Exception:
            feats.append(np.zeros(32, dtype=np.float32))

        # 4. FFT radial profile
        try:
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
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

        # 5. Statistical moments
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

        # 6. Color histogram (ROI)
        try:
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            color_hist = cv2.calcHist([hsv], [0, 1], tissue_mask.astype(np.uint8),
                                      [8, 4], [0, 180, 0, 256]).flatten()
            color_hist = color_hist / (color_hist.sum() + 1e-6)
            feats.append(color_hist.astype(np.float32))
        except Exception:
            feats.append(np.zeros(32, dtype=np.float32))

        # 7. GLCM-like texture features (از گرادیان)
        try:
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            energy = float(np.mean(grad_x ** 2 + grad_y ** 2)) / 10000.0
            contrast = float(np.std(gray)) / 128.0
            homogeneity = 1.0 / (1.0 + float(np.var(gray)) / 1000.0)
            correlation = float(np.corrcoef(gray.flatten(), np.roll(gray, 1, axis=1).flatten())[0, 1]) if gray.size > 1 else 0.0
            glcm = np.array([energy, contrast, homogeneity, correlation], dtype=np.float32)
            feats.append(glcm)
        except Exception:
            feats.append(np.zeros(4, dtype=np.float32))

        vec = np.concatenate(feats).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
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
# ✅ Layer 1 — Heuristic gate (fast reject of non-medical images)
# ================================================================

def _saturation_score(bgr: np.ndarray) -> float:
    try:
        tissue_mask = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 20
        if tissue_mask.sum() < 100:
            return 0.5
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        mean_sat = float(sat[tissue_mask].mean())
        if mean_sat < 0.15: return 1.0
        elif mean_sat < 0.25: return 0.8
        elif mean_sat < 0.35: return 0.5
        else: return 0.1
    except Exception:
        return 0.5


def _brightness_score(bgr: np.ndarray) -> float:
    try:
        tissue = _tissue_gray(bgr)
        if tissue.size < 100:
            return 0.5
        mean_b = float(tissue.mean()) / 255.0
        if 0.08 <= mean_b <= 0.85: return 1.0
        elif 0.05 <= mean_b < 0.08 or 0.85 < mean_b <= 0.92: return 0.7
        else: return 0.1
    except Exception:
        return 0.5


def _grayscale_score(bgr: np.ndarray) -> float:
    try:
        tissue_mask = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 20
        if tissue_mask.sum() < 100: return 0.5
        b, g, r = cv2.split(bgr.astype(np.float32))
        mean_r, mean_g, mean_b = float(r[tissue_mask].mean()), float(g[tissue_mask].mean()), float(b[tissue_mask].mean())
        mean_all = (mean_r + mean_g + mean_b) / 3.0 + 1e-6
        diff = (abs(mean_r - mean_g) + abs(mean_g - mean_b) + abs(mean_b - mean_r)) / 3.0
        return max(0.0, 1.0 - (diff / mean_all) * 4.0)
    except Exception:
        return 0.5


def _edge_density_score(bgr: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tissue_mask = gray > 20
        if tissue_mask.sum() < 100: return 0.5
        edges = cv2.Canny(gray, 50, 150)
        density = float((edges > 0)[tissue_mask].mean())
        if density < 0.08: return 1.0
        elif density < 0.15: return 0.8
        elif density < 0.25: return 0.5
        else: return 0.1
    except Exception:
        return 0.5


def _histogram_score(bgr: np.ndarray) -> float:
    try:
        tissue = _tissue_gray(bgr)
        if tissue.size < 100: return 0.5
        hist = np.histogram(tissue, bins=256, range=(0, 256))[0].astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        mid = float(hist[50:210].sum())
        extreme = float(hist[0:20].sum() + hist[235:256].sum())
        if mid > 0.65 and extreme < 0.25: return 1.0
        elif mid > 0.45 and extreme < 0.40: return 0.7
        else: return 0.15
    except Exception:
        return 0.5


def _heuristic_gate(bgr: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
    metrics = {
        "saturation": _saturation_score(bgr),
        "brightness": _brightness_score(bgr),
        "grayscale": _grayscale_score(bgr),
        "edges": _edge_density_score(bgr),
        "histogram": _histogram_score(bgr),
    }
    weights = {"saturation": 0.35, "brightness": 0.15, "grayscale": 0.25, "edges": 0.10, "histogram": 0.15}
    score = sum(metrics[k] * weights[k] for k in metrics)

    # ✅ NEW: Hard rules — reject non-medical images definitively
    # اگه تصویر رنگی واضحه (saturation کم امتیاز)، یا grayscale نیست، رد کن
    if metrics["saturation"] < 0.30:  # ← خیلی رنگی = منظره/عکس معمولی
        return False, score, metrics
    if metrics["grayscale"] < 0.30:  # ← رنگی = سونوگرافی نیست
        return False, score, metrics
    if metrics["edges"] < 0.20:  # ← لبه‌های خیلی زیاد = عکس طبیعت
        return False, score, metrics

    # ✅ آستانه‌ی نهایی (نرم‌تر از قبل)
    is_medical_like = score >= 0.45  # ← از 0.50 به 0.45
    return is_medical_like, score, metrics


# ================================================================
# ✅ Layer 2 — Statistical model (PCA + Mahalanobis)
# ================================================================

class LiverModel:
    """
    ✅ v3.1 — Generalization-focused liver model.

    Key changes vs v3.0:
        - PCA 10 (not 50) → less memorization
        - KNN-Lite on reduced features → tolerance to new images
        - Threshold from percentile (not mean+2.5σ)
        - Score via sigmoid CDF (soft, not hard cutoff)
    """

    def __init__(self):
        self.pca = None
        self.cov = None
        self.mean = None
        self.threshold = 0.0
        self.n_samples = 0
        # ✅ NEW: reduced features for KNN
        self.reduced_features = None   # (N, n_components) numpy array
        self.dist_stats = None         # {"min", "p25", "p50", "p75", "p95", "max"}

    def fit(self, features: List[np.ndarray]):
        """آموزش از لیست بردارهای ویژگی."""
        if not features or len(features) < 10:
            raise ValueError("Need at least 10 reference features")

        X = np.array(features, dtype=np.float32)
        n_components = min(PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])

        # 1. PCA (کاهش ابعاد)
        self.pca = PCA(n_components=n_components, whiten=True)
        X_reduced = self.pca.fit_transform(X)
        self.reduced_features = X_reduced.astype(np.float32)

        # 2. میانگین + کوواریانس
        self.mean = np.mean(X_reduced, axis=0).astype(np.float32)
        self.cov = EmpiricalCovariance()
        self.cov.fit(X_reduced)

        # 3. محاسبه‌ی فاصله‌ی همه‌ی نمونه‌ها
        distances = self.cov.mahalanobis(X_reduced)

        # ✅ NEW: آستانه از percentile (نه mean+2.5σ)
        p50 = float(np.percentile(distances, 50))
        p75 = float(np.percentile(distances, 75))
        p95 = float(np.percentile(distances, 95))
        p99 = float(np.percentile(distances, 99))

        # آستانه = میانه + 1.5 * (p95 - p50)
        # این یعنی هر تصویر تا 2.5 برابر پراکندگی طبیعی → قبول
        spread = max(p95 - p50, 1e-3)
        self.threshold = p50 + 2.5 * spread

        # ✅ آمار برای امتیازدهی نرم
        self.dist_stats = {
            "min": float(distances.min()),
            "p25": float(np.percentile(distances, 25)),
            "p50": p50,
            "p75": p75,
            "p95": p95,
            "p99": p99,
            "max": float(distances.max()),
        }
        self.n_samples = len(features)

        # ✅ گزارش کامل برای دیباگ
        print(f"✅ Liver model v3.1 trained on {self.n_samples} images")
        print(f"   PCA components:  {n_components}")
        print(f"   Threshold:       {self.threshold:.3f}  (p50={p50:.2f}, p95={p95:.2f})")
        print(f"   Distance range:  {distances.min():.2f} .. {distances.max():.2f}")

    def score(self, feature: np.ndarray) -> float:
        """
        امتیاز شباهت به مدل (0 تا 1) — نرم و تحمل‌پذیر.

        روش:
            1. Mahalanobis distance در PCA space
            2. KNN-Lite: میانگین فاصله تا ۵ نزدیک‌ترین neighbor
            3. ترکیب دو → score

        برخلاف v3.0، اینجا از sigmoid (CDF) استفاده می‌شه،
        نه exp() ساده. یعنی حتی اگه فاصله بیشتر از threshold باشه،
        score صفر نمی‌شه (تا 0.05 شنا می‌کنه).
        """
        if self.pca is None or self.cov is None or self.mean is None:
            return 0.5
        try:
            # 1. PCA transform
            x_reduced = self.pca.transform(feature.reshape(1, -1))
            x_reduced = x_reduced.astype(np.float32)

            # 2. Mahalanobis distance
            dist = float(self.cov.mahalanobis(x_reduced)[0])

            # 3. KNN-Lite: میانگین فاصله تا K نزدیک‌ترین neighbor
            if self.reduced_features is not None and len(self.reduced_features) >= KNN_K:
                # Euclidean distance در PCA space
                diffs = self.reduced_features - x_reduced
                dists_sq = np.sum(diffs * diffs, axis=1)
                nearest = np.partition(dists_sq, KNN_K)[:KNN_K]
                knn_dist = float(np.sqrt(np.mean(nearest)))
            else:
                knn_dist = dist

            # 4. ترکیب: نیم‌نیم
            combined = 0.6 * dist + 0.4 * knn_dist

            # 5. تبدیل به score با sigmoid (CDF)
            #    score = 1 / (1 + exp((combined - threshold) / scale))
            #    - اگه combined << threshold → score ≈ 1.0
            #    - اگه combined == threshold → score = 0.5
            #    - اگه combined >> threshold → score ≈ 0.0
            scale = max(self.threshold * 0.4, 1e-3)
            z = (combined - self.threshold) / scale
            # clamp برای جلوگیری از overflow
            z = max(-50.0, min(50.0, z))
            score = 1.0 / (1.0 + math.exp(z))

            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️ LiverModel.score failed: {e}")
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
                    "reduced_features": self.reduced_features,  # ✅ NEW
                    "dist_stats": self.dist_stats,               # ✅ NEW
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
            model.reduced_features = data.get("reduced_features")  # ممکنه نباشه در pkl قدیمی
            model.dist_stats = data.get("dist_stats", {})
            print(f"📦 Model loaded from cache ({model.n_samples} samples, v3.1)")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            return None


# ================================================================
# ✅ Model singleton + training
# ================================================================

_LIVER_MODEL: Optional[LiverModel] = None


def _get_liver_model() -> Optional[LiverModel]:
    """
    ترتیب بارگذاری:
      1. کش در حافظه
      2. pkl از پیش آموزش‌دیده (shipped)  ← اصلی
      3. .cache/ قدیمی (dev fallback)
      4. آموزش جدید ONLY IF LIVER_GUARD_ALLOW_TRAIN=1
         وگرنه fail-open (None برمی‌گرداند — بدون کندی)
    """
    global _LIVER_MODEL
    if _LIVER_MODEL is not None:
        return _LIVER_MODEL

    # 1. pkl از پیش آموزش‌دیده
    for path in (_BUNDLE_MODEL_FILE, _DEV_CACHE_FILE):
        if os.path.exists(path):
            m = LiverModel.load(path)
            if m is not None:
                _LIVER_MODEL = m
                return m

    # 2. آموزش فقط اگر صریحاً اجازه داده شده باشد
    if not _ALLOW_TRAINING:
        print("⚠️ Liver model pkl not found — guard is in FAIL-OPEN mode.")
        print("   (To generate the pkl once, run: LIVER_GUARD_ALLOW_TRAIN=1 python tools/build_liver_model.py)")
        return None

    # 3. آموزش (فقط dev)
    print("🔄 [DEV] Training liver model from references...")
    features = _load_reference_features(LIVER_REFERENCE_DIR, max_images=MAX_REFERENCE_IMAGES)
    if not features or len(features) < 10:
        print("⚠️ Not enough liver reference images for training")
        return None

    try:
        model = LiverModel()
        model.fit(features)
        # ✅ ذخیره در مسیر bundle (تا در نصب‌های بعدی استفاده شود)
        model.save(_BUNDLE_MODEL_FILE)
        model.save(_DEV_CACHE_FILE)
        _LIVER_MODEL = model
        return model
    except Exception as e:
        print(f"⚠️ Training failed: {e}")
        return None


def _load_reference_features(directory: str, max_images: int = MAX_REFERENCE_IMAGES) -> List[np.ndarray]:
    """بارگذاری ویژگی‌های همه‌ی تصاویر یک پوشه."""
    features = []
    if not os.path.isdir(directory):
        return features
    files = sorted([f for f in os.listdir(directory)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))])

    # نمونه‌گیری تصادفی اگه بیشتر از max_images بود
    if len(files) > max_images:
        import random
        random.seed(42)
        files = random.sample(files, max_images)

    for i, fname in enumerate(files):
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
# ✅ Layer 3 — Non-liver reference check (optional)
# ================================================================

_NON_LIVER_FEATURES: Optional[List[np.ndarray]] = None


def _get_non_liver_features() -> List[np.ndarray]:
    global _NON_LIVER_FEATURES
    if _NON_LIVER_FEATURES is not None:
        return _NON_LIVER_FEATURES
    _NON_LIVER_FEATURES = _load_reference_features(NON_LIVER_REFERENCE_DIR, max_images=500)
    return _NON_LIVER_FEATURES


# ================================================================
# ✅ Public API
# ================================================================

def analyze_image(image_path: str) -> Dict:
    result = {"is_liver_like": True, "score": 1.0,
              "metrics": {}, "reason": ""}

    if not HAS_CV2 or not HAS_SKLEARN:
        result["reason"] = "Required libraries not available"
        return result

    if not image_path or not os.path.isfile(image_path):
        result.update(is_liver_like=False, score=0.0,
                      reason="File not found")
        return result

    try:
        img = cv2.imread(image_path)
        if img is None:
            result.update(is_liver_like=False, score=0.0,
                          reason="Could not read image")
            return result

        h, w = img.shape[:2]
        if max(h, w) > 600:
            s = 600.0 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))

        # ---------- Layer 1: Heuristic gate (ALWAYS runs — fast) ----------
        is_medical_like, heuristic_score, heuristic_metrics = _heuristic_gate(img)
        result["metrics"].update(heuristic_metrics)
        result["metrics"]["heuristic_score"] = heuristic_score

        if not is_medical_like:
            result.update(is_liver_like=False, score=heuristic_score)
            worst = min(heuristic_metrics.items(), key=lambda kv: kv[1])
            reason_map = {
                "saturation": "Image is too colorful (not grayscale ultrasound)",
                "grayscale": "Image is not grayscale (not ultrasound)",
                "brightness": "Image brightness is out of range",
                "edges": "Image has too many sharp edges",
                "histogram": "Histogram is not consistent with ultrasound",
            }
            result["reason"] = reason_map.get(worst[0],
                                              "Not a medical ultrasound image")
            return result

        # ---------- Layer 2: Statistical model ----------
        model = _get_liver_model()
        if model is None:
            # ✅ FAIL-OPEN: heuristic passed, we allow (no 5-min delay)
            result["is_liver_like"] = True
            result["score"] = heuristic_score
            result["reason"] = "No liver model — heuristic-only (fail-open)"
            return result

        test_vec = _compute_features(img)
        if test_vec is None:
            result["is_liver_like"] = True
            result["reason"] = "Could not extract features — fail-open"
            return result

        liver_score = model.score(test_vec)
        result["metrics"]["liver_score"] = liver_score

        if liver_score < REJECT_SCORE:  # ← از 0.35 به 0.15
            result.update(is_liver_like=False, score=liver_score,
                          reason=(
                              f"Image does not match liver ultrasound pattern "
                              f"(liver_score={liver_score:.2f} < {REJECT_SCORE})"
                          ))
            return result

        # ---------- Layer 3: optional non-liver ----------
        non_liver = _get_non_liver_features()
        if non_liver:
            try:
                sims = [float(np.dot(test_vec, r)) for r in non_liver]
                best = max(sims) if sims else 0.0
                result["metrics"]["non_liver_similarity"] = best
                if best > liver_score + 0.1:
                    result.update(is_liver_like=False, score=best,
                                  reason="More similar to non-liver references")
                    return result
            except Exception:
                pass

        final = 0.3 * heuristic_score + 0.7 * liver_score
        result["score"] = final
        result["metrics"]["final_score"] = final
        result["is_liver_like"] = final >= MIN_LIVER_SCORE
        if not result["is_liver_like"]:
            result["reason"] = (
                f"Score {final:.2f} below threshold {MIN_LIVER_SCORE} "
                f"(heuristic={heuristic_score:.2f}, liver={liver_score:.2f})"
            )
        return result

    except Exception as e:
        result["is_liver_like"] = True
        result["reason"] = f"Analysis error (fail-open): {e}"
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

    if all_probs and len(all_probs) >= 2:
        try:
            probs = list(all_probs.values())
            s = sum(probs) + 1e-9
            probs = [p/s for p in probs]
            entropy = -sum(p * math.log(p + 1e-9) for p in probs if p > 0)
            max_entropy = math.log(len(probs))
            norm_entropy = entropy / (max_entropy + 1e-9)
            metrics["entropy"] = norm_entropy
            if norm_entropy > 0.98:
                return (False, f"Model distribution too uniform (entropy={norm_entropy:.2f})", metrics)
        except Exception:
            pass

    return (True, "", metrics)


def reload_model():
    """برای زمانی که تصاویر reference عوض شدن."""
    global _LIVER_MODEL, _NON_LIVER_FEATURES
    _LIVER_MODEL = None
    _NON_LIVER_FEATURES = None
    try:
        if os.path.exists(MODEL_CACHE_FILE):
            os.remove(MODEL_CACHE_FILE)
    except Exception:
        pass
    _get_liver_model()