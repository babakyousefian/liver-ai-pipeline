# core/preprocessing/medical_preprocessor.py
# -*- coding: utf-8 -*-
"""
Medical Preprocessor v6.0 — Aggressive header removal + fan bounding box

Strategy:
    1. Detect uniform borders (top/bottom/left/right) — remove them
    2. In the cropped image, find the largest bright region (fan/tissue)
    3. Crop to that region's bounding box (with small padding)
    4. Inpaint small colored labels INSIDE the tissue only
    5. Return cleaned image

Why this works:
    - Headers (top/side) are always uniform-color or uniform-dark
    - Fan is always the largest bright connected region
    - Labels are always small colored patches INSIDE the fan
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class PreprocessorConfig:
    # Border detection
    border_uniformity_std: float = 18.0
    border_max_ratio: float = 0.30
    border_min_pixels: int = 6

    # Fan detection (after border removal)
    fan_min_area_ratio: float = 0.25   # fan must be ≥25% of remaining image
    fan_padding: float = 0.02          # 2% padding around fan

    # Label removal (only inside tissue)
    label_saturation_min: int = 80     # only strongly colored
    label_value_min: int = 100         # only bright
    label_min_area: int = 20
    label_max_area: int = 500          # small labels only
    label_dilate_size: int = 5
    label_inpaint_radius: int = 3

    # Output
    output_min_size: int = 224
    max_input_dim: int = 2000

    debug: bool = False


@dataclass
class PreprocessResult:
    success: bool
    image: Optional[np.ndarray] = None
    original_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    labels_removed: int = 0
    cropped: bool = False
    inpainted: bool = False
    has_header: bool = False
    has_side_panels: bool = False
    crop_decision: str = ""
    reason: str = ""
    borders: Optional[Dict[str, int]] = None
    fan_box: Optional[Tuple[int, int, int, int]] = None

    def save(self, path: str) -> bool:
        if self.image is None:
            return False
        try:
            return cv2.imwrite(path, self.image)
        except Exception:
            return False


# ================================================================
# Strip analysis
# ================================================================

def _analyze_strip(strip: np.ndarray) -> Dict[str, float]:
    if strip.size == 0:
        return {"std": 0.0, "mean": 0.0, "bright_ratio": 0.0, "dark_ratio": 0.0}
    gray = strip if strip.ndim == 2 else cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    return {
        "std": float(gray.std()),
        "mean": float(gray.mean()),
        "bright_ratio": float((gray > 180).sum()) / gray.size,
        "dark_ratio": float((gray < 90).sum()) / gray.size,
    }


def _is_header_strip(stats: Dict, cfg: PreprocessorConfig) -> bool:
    """
    v6.4 — تشخیص سخت‌گیرانه‌تر.
    فقط اگر واقعاً uniform یا dark-with-text باشد.
    """
    # ✅ uniformity خیلی سخت‌گیرانه (std < 10)
    if stats["std"] < 10.0:
        return True

    # ✅ dark with text (مثل هدر تصویر دوم)
    if (stats["mean"] < 100 and
            0.005 < stats["bright_ratio"] < 0.35 and
            stats["std"] < 25):
        return True

    # ✅ dark uniform (پس‌زمینه سیاه)
    if stats["std"] < 12 and stats["mean"] < 40:
        return True

    return False


# ================================================================
# Border detection (with sub-pixel precision)
# ================================================================

def _detect_side_border(img: np.ndarray, cfg: PreprocessorConfig,
                        side: str) -> int:
    """
    v6.4 — تشخیص دقیق‌تر border با محدودیت 25% (نه 40%).

    تغییرات:
    - حداکثر 25% ارتفاع (نه 40%)
    - چک uniformity دقیق‌تر
    - جلوگیری از کراپ بیش از حد
    """
    h, w = img.shape[:2]

    if side in ("top", "bottom"):
        max_scan = min(int(h * 0.25), 200)  # ← از 0.40 به 0.25
        min_scan = max(cfg.border_min_pixels, int(h * 0.01))
    else:
        max_scan = min(int(w * 0.25), 200)
        min_scan = max(cfg.border_min_pixels, int(w * 0.01))

    if max_scan < min_scan:
        return 0

    best = 0
    for scan in range(min_scan, max_scan, 3):
        if side == "top":
            strip = img[:scan, :]
        elif side == "bottom":
            strip = img[-scan:, :]
        elif side == "left":
            strip = img[:, :scan]
        else:
            strip = img[:, -scan:]

        stats = _analyze_strip(strip)
        if _is_header_strip(stats, cfg):
            best = scan
        else:
            if best > 0:
                break

    # ✅ حداقل 1.5% و حداکثر 25%
    if side in ("top", "bottom"):
        if best < int(h * 0.015):
            return 0
        if best > int(h * 0.25):
            return int(h * 0.25)  # ← کپ
    else:
        if best < int(w * 0.015):
            return 0
        if best > int(w * 0.25):
            return int(w * 0.25)

    return best


def _detect_borders(img: np.ndarray, cfg: PreprocessorConfig) -> Dict[str, int]:
    return {
        "top": _detect_side_border(img, cfg, "top"),
        "bottom": _detect_side_border(img, cfg, "bottom"),
        "left": _detect_side_border(img, cfg, "left"),
        "right": _detect_side_border(img, cfg, "right"),
    }


# ================================================================
# Fan / tissue detection
# ================================================================

def _detect_fan_box(img: np.ndarray, cfg: PreprocessorConfig) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the largest bright region (fan/tissue) in the image.
    Returns bounding box (x, y, w, h) or None.
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ✅ Tissue: bright pixels (> 25)
        tissue_mask = (gray > 25).astype(np.uint8) * 255

        # ✅ Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # ✅ Largest connected component
        num, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask, 8)
        if num <= 1:
            return None

        total_area = w * h
        best_idx, best_area = -1, 0
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < total_area * cfg.fan_min_area_ratio:
                continue
            if area > best_area:
                best_area = area
                best_idx = i

        if best_idx == -1:
            return None

        x = stats[best_idx, cv2.CC_STAT_LEFT]
        y = stats[best_idx, cv2.CC_STAT_TOP]
        bw = stats[best_idx, cv2.CC_STAT_WIDTH]
        bh = stats[best_idx, cv2.CC_STAT_HEIGHT]

        return (int(x), int(y), int(bw), int(bh))
    except Exception as e:
        if cfg.debug:
            print(f"   ⚠️ Fan detection failed: {e}")
        return None


def _crop_to_box(img: np.ndarray, box: Tuple[int, int, int, int],
                 padding: float = 0.02) -> np.ndarray:
    """Crop image to bounding box with padding."""
    h, w = img.shape[:2]
    x, y, bw, bh = box

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)

    if x1 <= x0 + 10 or y1 <= y0 + 10:
        return img

    return img[y0:y1, x0:x1].copy()


def _crop_to_box_no_bottom(img: np.ndarray, box: Tuple[int, int, int, int],
                           padding: float = 0.02) -> np.ndarray:
    """
    Crop image to bounding box — BUT never crop from bottom.
    ✅ این همان نسخه‌ای است که کار می‌کرد (کمی چپ بیشتر، پایین صفر).
    """
    h, w = img.shape[:2]
    x, y, bw, bh = box

    # ✅ padding متفاوت برای هر جهت
    pad_left = int(bw * 0.05)   # ۵٪ چپ (کمی بیشتر برای اطمینان)
    pad_right = int(bw * 0.02)  # ۲٪ راست
    pad_top = int(bh * 0.02)    # ۲٪ بالا

    x0 = max(0, x - pad_left)
    y0 = max(0, y - pad_top)
    x1 = min(w, x + bw + pad_right)
    y1 = h  # ← پایین: هرگز کراپ نکن

    if x1 <= x0 + 10 or y1 <= y0 + 10:
        return img

    return img[y0:y1, x0:x1].copy()

# ================================================================
# Label removal (only INSIDE tissue, only small colored patches)
# ================================================================

def _remove_labels(img: np.ndarray, cfg: PreprocessorConfig) -> Tuple[np.ndarray, int]:
    """
    Remove SMALL colored labels inside the tissue.
    Does NOT inpaint branches/trees because:
      - Saturation threshold is high (80) → only strong colors
      - Area limit is tight (20-500 px) → only text-sized blobs
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]

        # ✅ Strong colored pixels only
        mask = cv2.inRange(
            hsv,
            np.array([0, cfg.label_saturation_min, cfg.label_value_min]),
            np.array([180, 255, 255]),
        )

        # ✅ Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # ✅ Filter by size: only text-sized blobs
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean_mask = np.zeros_like(mask)
        count = 0

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            # ✅ Text-size: area 20-500, aspect ratio < 3
            if (cfg.label_min_area <= area <= cfg.label_max_area and
                    max(bw, bh) / max(1, min(bw, bh)) < 3.0):
                clean_mask[labels == i] = 255
                count += 1

        if count == 0:
            return img, 0

        # ✅ Dilate (small)
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.label_dilate_size, cfg.label_dilate_size),
        )
        dilated = cv2.dilate(clean_mask, dilate_kernel, iterations=1)

        # ✅ Inpaint
        result = cv2.inpaint(img, dilated, cfg.label_inpaint_radius, cv2.INPAINT_TELEA)
        return result, count
    except Exception:
        return img, 0


# ================================================================
# Main entry
# ================================================================
def preprocess_medical_image(
        image_path: str,
        config: Optional[PreprocessorConfig] = None,
        already_cropped: Optional[bool] = None,
) -> PreprocessResult:
    cfg = config or PreprocessorConfig()

    if not image_path or not os.path.isfile(image_path):
        return PreprocessResult(success=False, reason="File not found")

    img = cv2.imread(image_path)
    if img is None:
        return PreprocessResult(success=False, reason="Could not read image")

    original_shape = img.shape[:2]
    h, w = img.shape[:2]
    if max(h, w) > cfg.max_input_dim:
        scale = cfg.max_input_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # ============================================================
    # STEP 1: Detect TOP border only (bottom = 0)
    # ============================================================
    top_border = _detect_side_border(img, cfg, "top")
    bottom_border = 0  # ✅ همیشه صفر

    has_header = top_border > 0

    if cfg.debug:
        print(f"   🔍 Top={top_border}px, Bottom=0px (disabled)")

    # ============================================================
    # STEP 2: Crop TOP border only
    # ============================================================
    if already_cropped is True:
        after_vertical = img
        vertical_cropped = False
    else:
        if has_header:
            hh, ww = img.shape[:2]
            t = min(top_border, int(hh * 0.40))
            after_vertical = img[t:, :].copy()
            vertical_cropped = True
        else:
            after_vertical = img
            vertical_cropped = False

    if cfg.debug:
        print(f"   🔍 After top crop: {after_vertical.shape}")

    # ============================================================
    # STEP 3: Detect fan box
    # ============================================================
    fan_box = _detect_fan_box(after_vertical, cfg)

    if fan_box is not None:
        fx, fy, fw, fh = fan_box
        has_side_panels = fx > 5 or (after_vertical.shape[1] - (fx + fw)) > 5
    else:
        has_side_panels = False

    if cfg.debug:
        print(f"   🔍 Fan box: {fan_box}, has_side_panels={has_side_panels}")

    # ============================================================
    # STEP 4: Crop using fan box (no bottom crop)
    # ============================================================
    fan_cropped = False

    if already_cropped is True:
        after_fan = after_vertical
        decision = "forced no-crop"
    else:
        if fan_box is not None:
            hh, ww = after_vertical.shape[:2]
            fx, fy, fw, fh = fan_box
            fan_area_ratio = (fw * fh) / float(ww * hh)

            if fan_area_ratio < 0.92:
                after_fan = _crop_to_box_no_bottom(
                    after_vertical, fan_box, cfg.fan_padding
                )
                fan_cropped = True
                decision = "auto: fan-crop (no bottom)"
            else:
                after_fan = after_vertical
                decision = "auto: no fan-crop (fan fills image)"
        else:
            after_fan = after_vertical
            decision = "auto: no fan (fallback)"

    if cfg.debug:
        print(f"   🔍 After fan crop: {after_fan.shape}, decision={decision}")

    was_cropped = vertical_cropped or fan_cropped

    # ============================================================
    # STEP 5: Remove labels
    # ============================================================
    cleaned, labels_removed = _remove_labels(after_fan, cfg)

    # ============================================================
    # STEP 6: Resize if too small
    # ============================================================
    h2, w2 = cleaned.shape[:2]
    if min(h2, w2) < cfg.output_min_size:
        scale = cfg.output_min_size / float(min(h2, w2))
        cleaned = cv2.resize(
            cleaned,
            (int(w2 * scale), int(h2 * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    borders_dict = {
        "top": top_border,
        "bottom": 0,
        "left": 0,
        "right": 0,
    }

    return PreprocessResult(
        success=True,
        image=cleaned,
        original_shape=original_shape,
        final_shape=cleaned.shape[:2],
        labels_removed=labels_removed,
        cropped=was_cropped,
        inpainted=labels_removed > 0,
        has_header=has_header,
        has_side_panels=has_side_panels,
        crop_decision=decision,
        reason="OK",
        borders=borders_dict,
        fan_box=fan_box,
    )
