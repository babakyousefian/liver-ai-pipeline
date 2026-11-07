# core/security/image_guard.py
# -*- coding: utf-8 -*-
"""
Image Guard v3.2 — Robust Reference-Based Anomaly Detection

Key fixes vs v3.1:
- Feature vector is NOT L2-normalized (keeps cosine meaningful)
- Mahalanobis threshold uses 99th percentile + wider margin
- Heuristic gate runs on ROI only (not full resized image)
- Fail-open on ANY error (never blocks a valid image)
- Non-liver check removed (was causing false rejections)
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
# Configuration — VERY PERMISSIVE
# ================================================================

# ✅ Only reject if image is CLEARLY not liver (very low thresholds)
MIN_LIVER_SCORE = 0.35
MIN_MODEL_CONFIDENCE = 0.25       # ← lowered from 0.45

# ✅ Model settings
PCA_COMPONENTS = 15
KNN_K = 7

# ✅ Threshold: use p99 + wide margin (not p50 + 2.5*spread)
REJECT_SCORE = 0.02               # ← lowered from 0.05

_HERE = os.path.dirname(os.path.abspath(__file__))
LIVER_REFERENCE_DIR = os.path.join(_HERE, "liver_reference")
CACHE_DIR = os.path.join(_HERE, ".cache")

_BUNDLE_MODEL_FILE = os.path.join(_HERE, "models", "liver_model.pkl")
_DEV_CACHE_FILE = os.path.join(CACHE_DIR, "liver_model.pkl")

MODEL_CACHE_FILE = _BUNDLE_MODEL_FILE
MODEL_CACHE_VERSION = 5           # ← bumped: pkl باید دوباره ساخته شود

FEATURE_SIZE = (256, 256)
MAX_REFERENCE_IMAGES = 3000

_ALLOW_TRAINING = os.environ.get("LIVER_GUARD_ALLOW_TRAIN", "0") == "1"


# ================================================================
# Utility — ROI extraction
# ================================================================

def _extract_roi(bgr: np.ndarray) -> np.ndarray:
    """Extract tissue region (remove black background)."""
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
        # ✅ If tissue covers >95% or <10%, keep original (already cropped or too small)
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


# ================================================================
# Feature extraction — NO L2 normalization
# ================================================================

def _compute_features(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a rich feature vector.
    ✅ IMPORTANT: NO L2 normalization — keeps magnitudes meaningful.
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

        # 1. Intensity histogram (64 bins) — already sums to 1
        hist = cv2.calcHist([gray], [0], tissue_mask.astype(np.uint8),
                            [64], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.append(hist)

        # 2. LBP histogram (10 bins) — already sums to 1
        try:
            lbp = _local_binary_pattern(gray, radius=1, n_points=8)
            lbp_hist = np.histogram(lbp[tissue_mask], bins=10, range=(0, 256))[0].astype(np.float32)
            lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
            feats.append(lbp_hist)
        except Exception:
            feats.append(np.zeros(10, dtype=np.float32))

        # 3. Gradient histogram (32 bins) — already sums to 1
        try:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag_hist = np.histogram(mag[tissue_mask], bins=32, range=(0, 255))[0].astype(np.float32)
            mag_hist = mag_hist / (mag_hist.sum() + 1e-6)
            feats.append(mag_hist)
        except Exception:
            feats.append(np.zeros(32, dtype=np.float32))

        # 4. FFT radial profile (16 bins) — already sums to 1
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

        # 5. Statistical moments (5 values, NOT normalized to sum=1)
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

        # 6. Color histogram (32 bins) — already sums to 1
        try:
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            color_hist = cv2.calcHist([hsv], [0, 1], tissue_mask.astype(np.uint8),
                                      [8, 4], [0, 180, 0, 256]).flatten()
            color_hist = color_hist / (color_hist.sum() + 1e-6)
            feats.append(color_hist.astype(np.float32))
        except Exception:
            feats.append(np.zeros(32, dtype=np.float32))

        # 7. Texture features (4 values)
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

        # ✅ CONCATENATE — NO L2 NORMALIZATION
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
# Layer 1 — Heuristic gate (fast, on ROI)
# ================================================================

def _saturation_score(bgr: np.ndarray) -> float:
    """✅ Runs on ROI — very permissive."""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tissue_mask = gray > 15
        if tissue_mask.sum() < 100:
            return 0.7  # ✅ neutral, don't penalize
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        mean_sat = float(sat[tissue_mask].mean())
        # ✅ Very permissive: ultrasound can have slight color tint
        if mean_sat < 0.20: return 1.0
        elif mean_sat < 0.30: return 0.85
        elif mean_sat < 0.45: return 0.6
        else: return 0.3
    except Exception:
        return 0.7


def _brightness_score(bgr: np.ndarray) -> float:
    try:
        tissue = _tissue_gray(bgr)
        if tissue.size < 100:
            return 0.7
        mean_b = float(tissue.mean()) / 255.0
        # ✅ Very permissive: ultrasound brightness varies a lot
        if 0.05 <= mean_b <= 0.90: return 1.0
        elif 0.02 <= mean_b < 0.05 or 0.90 < mean_b <= 0.96: return 0.7
        else: return 0.3
    except Exception:
        return 0.7


def _grayscale_score(bgr: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tissue_mask = gray > 15
        if tissue_mask.sum() < 100:
            return 0.7
        b, g, r = cv2.split(bgr.astype(np.float32))
        mean_r = float(r[tissue_mask].mean())
        mean_g = float(g[tissue_mask].mean())
        mean_b = float(b[tissue_mask].mean())
        mean_all = (mean_r + mean_g + mean_b) / 3.0 + 1e-6
        diff = (abs(mean_r - mean_g) + abs(mean_g - mean_b) + abs(mean_b - mean_r)) / 3.0
        # ✅ Very permissive: allow slight color tint
        score = 1.0 - (diff / mean_all) * 2.5  # ← was 4.0
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.7


def _edge_density_score(bgr: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tissue_mask = gray > 15
        if tissue_mask.sum() < 100:
            return 0.7
        edges = cv2.Canny(gray, 50, 150)
        density = float((edges > 0)[tissue_mask].mean())
        # ✅ Very permissive
        if density < 0.12: return 1.0
        elif density < 0.20: return 0.85
        elif density < 0.30: return 0.6
        else: return 0.3
    except Exception:
        return 0.7


def _histogram_score(bgr: np.ndarray) -> float:
    try:
        tissue = _tissue_gray(bgr)
        if tissue.size < 100:
            return 0.7
        hist = np.histogram(tissue, bins=256, range=(0, 256))[0].astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        mid = float(hist[30:225].sum())
        # ✅ Very permissive
        if mid > 0.50: return 1.0
        elif mid > 0.35: return 0.7
        else: return 0.3
    except Exception:
        return 0.7


def _heuristic_gate(bgr: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
    """
    ✅ Runs on ROI (not full image).
    ✅ Very permissive — only rejects CLEARLY non-medical images.
    """
    # ✅ Extract ROI first
    roi = _extract_roi(bgr)
    if roi.size == 0:
        roi = bgr

    metrics = {
        "saturation": _saturation_score(roi),
        "brightness": _brightness_score(roi),
        "grayscale": _grayscale_score(roi),
        "edges": _edge_density_score(roi),
        "histogram": _histogram_score(roi),
    }
    weights = {
        "saturation": 0.35,
        "brightness": 0.15,
        "grayscale": 0.25,
        "edges": 0.10,
        "histogram": 0.15,
    }
    score = sum(metrics[k] * weights[k] for k in metrics)

    # ✅ Only reject if CATASTROPHICALLY bad (colorful photo, etc.)
    # ✅ REMOVED the "edges < 0.20" hard rule (was rejecting valid images)
    if metrics["saturation"] < 0.20:  # ← was 0.30
        return False, score, metrics
    if metrics["grayscale"] < 0.20:   # ← was 0.30
        return False, score, metrics

    # ✅ Very permissive final threshold
    is_medical_like = score >= 0.35  # ← was 0.45
    return is_medical_like, score, metrics


# ================================================================
# Layer 2 — Statistical model (PCA + Mahalanobis)
# ================================================================

class LiverModel:
    """
    v3.2 — Generalization-focused.

    Key changes:
    - Threshold = p99 + wide margin (not p50 + 2.5*spread)
    - Score via sigmoid with LARGER scale (smoother)
    - KNN-Lite with Euclidean distance
    """

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
        p95 = float(np.percentile(distances, 95))
        p99 = float(np.percentile(distances, 99))
        max_d = float(distances.max())

        # ✅ KEY FIX: threshold based on p99 + wide margin
        # This allows new images up to ~3x the training spread
        spread = max(p99 - p50, 1e-3)
        self.threshold = p99 + 2.0 * spread   # ← generous

        self.dist_stats = {
            "min": float(distances.min()),
            "p25": float(np.percentile(distances, 25)),
            "p50": p50,
            "p75": float(np.percentile(distances, 75)),
            "p95": p95,
            "p99": p99,
            "max": max_d,
        }
        self.n_samples = len(features)

        print(f"✅ Liver model v3.2 trained on {self.n_samples} images")
        print(f"   PCA components:  {n_components}")
        print(f"   Threshold:       {self.threshold:.3f}  (p50={p50:.2f}, p99={p99:.2f})")
        print(f"   Distance range:  {distances.min():.2f} .. {max_d:.2f}")

    def score(self, feature: np.ndarray) -> float:
        """
        Similarity score (0-1) — SMOOTH and FORGIVING.
        """
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

            # ✅ KEY FIX: MUCH larger scale → smoother sigmoid
            # Combined up to 3x threshold still gives reasonable score
            scale = max(self.threshold * 1.5, 1e-3)   # ← was 0.4
            z = (combined - self.threshold) / scale
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
                print(f"⚠️ Model version mismatch: {data.get('version')} != {MODEL_CACHE_VERSION}")
                return None
            model = cls()
            model.pca = data["pca"]
            model.cov = data["cov"]
            model.mean = data["mean"]
            model.threshold = data["threshold"]
            model.n_samples = data["n_samples"]
            model.reduced_features = data.get("reduced_features")
            model.dist_stats = data.get("dist_stats", {})
            print(f"📦 Model loaded from cache ({model.n_samples} samples, v3.2)")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            return None


# ================================================================
# Model singleton + training
# ================================================================

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
        print("⚠️ Liver model pkl not found — guard is in FAIL-OPEN mode.")
        print("   (To generate the pkl once, run: LIVER_GUARD_ALLOW_TRAIN=1 python tools/build_liver_model.py)")
        return None

    print("🔄 [DEV] Training liver model from references...")
    features = _load_reference_features(LIVER_REFERENCE_DIR, max_images=MAX_REFERENCE_IMAGES)
    if not features or len(features) < 10:
        print("⚠️ Not enough liver reference images for training")
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
# Public API — FAIL-OPEN
# ================================================================

def analyze_image(image_path: str) -> Dict:
    """
    ✅ FAIL-OPEN: On ANY error, returns is_liver_like=True.
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
        if max(h, w) > 600:
            s = 600.0 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))

        # ---------- Layer 1: Heuristic gate ----------
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

        # ✅ ONLY reject if score is EXTREMELY low
        if liver_score < REJECT_SCORE:
            result.update(is_liver_like=False, score=liver_score,
                          reason=(
                              f"Image does not match liver ultrasound pattern "
                              f"(liver_score={liver_score:.3f} < {REJECT_SCORE})"
                          ))
            return result

        # ✅ Layer 3 removed (was causing false rejections)

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
    """
    ✅ VERY PERMISSIVE: only rejects if model is COMPLETELY unsure.
    """
    metrics = {}
    img_result = analyze_image(image_path)
    metrics.update(img_result.get("metrics", {}))
    metrics["image_score"] = img_result.get("score", 1.0)

    if not img_result["is_liver_like"]:
        return (False, img_result.get("reason", "Not a liver ultrasound"), metrics)

    # ✅ Only reject if confidence is EXTREMELY low
    if model_confidence < MIN_MODEL_CONFIDENCE:
        return (False, f"Model confidence too low ({model_confidence*100:.1f}%)", metrics)

    # ✅ Entropy check removed (was rejecting Grid 0 cases)
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
