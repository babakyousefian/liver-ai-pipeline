# core/preprocessing/medical_preprocessor.py
# -*- coding: utf-8 -*-
"""
Medical Preprocessor v4.0 — Header-aware cropping.

KEY LOGIC:
    1. Detect if image has "header" (top bar) or "side panels"
       → If YES: crop the fan region + remove labels
       → If NO:  keep image as-is, only remove labels

    2. Label removal handles BOTH:
       - Colored labels on dark background (traditional)
       - Colored labels on light/white background

    3. Fan extraction ONLY runs when headers/panels detected.
       This prevents re-cropping already-cropped images.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


# ================================================================
# Configuration
# ================================================================

@dataclass
class PreprocessorConfig:
    # -------- Header detection --------
    # Top strip: if it's uniform-color OR contains bright text on dark, it's a header
    header_max_height_ratio: float = 0.20    # scan up to 20% of image height
    header_uniformity_std: float = 12.0      # std < this → uniform strip
    header_text_brightness_thresh: int = 180 # bright text threshold (white text)
    header_dark_bg_thresh: int = 90          # dark background threshold

    # -------- Side panel detection --------
    side_max_width_ratio: float = 0.20       # scan up to 20% of image width
    side_uniformity_std: float = 12.0        # std < this → uniform strip

    # -------- Fan mask --------
    fan_min_area_ratio: float = 0.15         # fan must cover ≥15% of image
    fan_crop_padding: float = 0.03           # 3% padding around fan

    # -------- Label removal --------
    label_saturation_min: int = 55           # colored label threshold (lower = more sensitive)
    label_value_min: int = 60
    label_min_area: int = 15                 # min area (px) to count as label
    label_max_area_ratio: float = 0.05       # max area as fraction of image
    label_dilate_size: int = 5
    label_inpaint_radius: int = 3
    label_white_bg_value_min: int = 200      # for labels on white bg
    label_white_bg_saturation_max: int = 35

    # -------- Output --------
    output_min_size: int = 224
    max_input_dim: int = 1600

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

    def save(self, path: str) -> bool:
        if self.image is None:
            return False
        try:
            return cv2.imwrite(path, self.image)
        except Exception:
            return False


# ================================================================
# ✅ STEP 1: Header / Side-panel detection
# ================================================================

def _analyze_strip(strip: np.ndarray) -> Dict[str, float]:
    """
    Analyze a strip (horizontal or vertical) and return statistics:
        - std: uniformity (low = uniform color)
        - mean: average brightness
        - bright_ratio: fraction of bright pixels (potential text)
        - dark_ratio: fraction of dark pixels
    """
    if strip.size == 0:
        return {"std": 0.0, "mean": 0.0, "bright_ratio": 0.0, "dark_ratio": 0.0}

    gray = strip if strip.ndim == 2 else cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    mean_v = float(gray.mean())
    std_v = float(gray.std())
    bright = float((gray > 180).sum()) / gray.size
    dark = float((gray < 90).sum()) / gray.size

    return {
        "std": std_v,
        "mean": mean_v,
        "bright_ratio": bright,
        "dark_ratio": dark,
    }


def _detect_top_header(img: np.ndarray, cfg: PreprocessorConfig) -> Tuple[bool, int]:
    """
    Detect if there's a "header" at the top of the image.

    A header is a horizontal strip that is:
      (a) uniform color (low std), OR
      (b) contains bright text on dark background (like "ECUBE15", "DR.ROSHANAE")

    Returns:
        (has_header, header_height_px)
    """
    h, w = img.shape[:2]
    max_h = min(int(h * cfg.header_max_height_ratio), 200)
    if max_h < 10:
        return False, 0

    # ✅ Scan from top, growing the strip until it stops looking like a header
    best_height = 0

    for scan_h in range(10, max_h, 4):
        strip = img[:scan_h, :]
        stats = _analyze_strip(strip)

        # Case A: Uniform colored strip (like a solid blue bar)
        is_uniform = stats["std"] < cfg.header_uniformity_std

        # Case B: Dark strip with bright text (like the second image's header)
        # The header has dark bg + some very bright pixels (text)
        is_dark_with_text = (
            stats["mean"] < 130 and                  # mostly dark
            stats["bright_ratio"] > 0.005 and        # some bright pixels (text)
            stats["bright_ratio"] < 0.40             # not too many (just text)
        )

        # Case C: Uniform dark strip (like black border)
        is_dark_uniform = stats["std"] < 15 and stats["mean"] < 60

        if is_uniform or is_dark_with_text or is_dark_uniform:
            best_height = scan_h
        else:
            # ✅ Once strip stops looking like a header, stop scanning
            if best_height > 0:
                break

    # ✅ Require minimum header height (at least 2% of image)
    if best_height < int(h * 0.02):
        return False, 0

    if cfg.debug:
        strip = img[:best_height, :]
        print(f"   🔍 Top header: h={best_height}px, stats={_analyze_strip(strip)}")

    return True, best_height


def _detect_side_panel(img: np.ndarray, cfg: PreprocessorConfig,
                       side: str = "left") -> Tuple[bool, int]:
    """
    Detect if there's a side panel (left or right).

    A side panel is a vertical strip that is:
      (a) uniform color, OR
      (b) contains text (like "ABD1", "P 100%", scale marks)

    Args:
        side: "left" or "right"

    Returns:
        (has_panel, panel_width_px)
    """
    h, w = img.shape[:2]
    max_w = min(int(w * cfg.side_max_width_ratio), 200)
    if max_w < 10:
        return False, 0

    best_width = 0

    for scan_w in range(10, max_w, 4):
        if side == "left":
            strip = img[:, :scan_w]
        else:
            strip = img[:, -scan_w:]

        stats = _analyze_strip(strip)

        # Same logic as top header
        is_uniform = stats["std"] < cfg.side_uniformity_std
        is_dark_with_text = (
            stats["mean"] < 130 and
            stats["bright_ratio"] > 0.005 and
            stats["bright_ratio"] < 0.40
        )
        is_dark_uniform = stats["std"] < 15 and stats["mean"] < 60

        if is_uniform or is_dark_with_text or is_dark_uniform:
            best_width = scan_w
        else:
            if best_width > 0:
                break

    if best_width < int(w * 0.02):
        return False, 0

    if cfg.debug:
        if side == "left":
            strip = img[:, :best_width]
        else:
            strip = img[:, -best_width:]
        print(f"   🔍 {side} panel: w={best_width}px, stats={_analyze_strip(strip)}")

    return True, best_width


def _detect_bottom_bar(img: np.ndarray, cfg: PreprocessorConfig) -> Tuple[bool, int]:
    """
    Detect if there's a bottom bar (sometimes has patient info).
    Same logic as top header but reversed.
    """
    h, w = img.shape[:2]
    max_h = min(int(h * cfg.header_max_height_ratio), 200)
    if max_h < 10:
        return False, 0

    best_height = 0

    for scan_h in range(10, max_h, 4):
        strip = img[-scan_h:, :]
        stats = _analyze_strip(strip)

        is_uniform = stats["std"] < cfg.header_uniformity_std
        is_dark_with_text = (
            stats["mean"] < 130 and
            stats["bright_ratio"] > 0.005 and
            stats["bright_ratio"] < 0.40
        )
        is_dark_uniform = stats["std"] < 15 and stats["mean"] < 60

        if is_uniform or is_dark_with_text or is_dark_uniform:
            best_height = scan_h
        else:
            if best_height > 0:
                break

    if best_height < int(h * 0.02):
        return False, 0

    return True, best_height


# ================================================================
# ✅ STEP 2: Crop based on detected boundaries
# ================================================================

def _crop_boundaries(img: np.ndarray, boundaries: Dict[str, int]) -> np.ndarray:
    """
    Crop image based on detected top/bottom/left/right boundaries.
    """
    h, w = img.shape[:2]
    top = boundaries.get("top", 0)
    bottom = boundaries.get("bottom", 0)
    left = boundaries.get("left", 0)
    right = boundaries.get("right", 0)

    # ✅ Safety: don't crop more than 30% from any side
    top = min(top, int(h * 0.30))
    bottom = min(bottom, int(h * 0.30))
    left = min(left, int(w * 0.30))
    right = min(right, int(w * 0.30))

    y0 = top
    y1 = h - bottom if bottom > 0 else h
    x0 = left
    x1 = w - right if right > 0 else w

    if y1 <= y0 or x1 <= x0:
        return img

    return img[y0:y1, x0:x1].copy()


# ================================================================
# ✅ STEP 3: Fan mask (only when needed)
# ================================================================

def _extract_fan_mask(img: np.ndarray, cfg: PreprocessorConfig) -> np.ndarray:
    """Extract fan-shaped tissue mask (bright region inside dark background)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ✅ Tissue = bright pixels (exclude black background)
    tissue = cv2.inRange(gray, 15, 235)

    # Remove small text
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, k_small, iterations=1)

    # Merge tissue regions
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, k_big, iterations=3)

    # Largest component
    num, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, 8)
    if num <= 1:
        return np.ones_like(gray) * 255

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
        return np.ones_like(gray) * 255

    mask = (labels == best_idx).astype(np.uint8) * 255

    # Convex hull (to include fan shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)

    clean_mask = np.zeros_like(gray)
    cv2.drawContours(clean_mask, [hull], -1, 255, -1)
    return clean_mask


def _crop_and_mask_fan(img: np.ndarray, cfg: PreprocessorConfig) -> Tuple[np.ndarray, bool]:
    """
    Crop fan region and black out everything outside the fan.
    ONLY called when headers/panels were detected.
    """
    h, w = img.shape[:2]
    fan_mask = _extract_fan_mask(img, cfg)

    x, y, ww, hh = cv2.boundingRect(fan_mask)
    if ww == 0 or hh == 0:
        return img, False

    # ✅ Small padding to avoid cutting tissue edges
    pad_x = int(ww * cfg.fan_crop_padding)
    pad_y = int(hh * cfg.fan_crop_padding)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + ww + pad_x)
    y1 = min(h, y + hh + pad_y)

    cropped = img[y0:y1, x0:x1].copy()
    cropped_mask = fan_mask[y0:y1, x0:x1]

    # Black out non-fan regions
    cropped[cropped_mask == 0] = [0, 0, 0]
    return cropped, True


# ================================================================
# ✅ STEP 4: Label removal (handles BOTH dark and light backgrounds)
# ================================================================

def _remove_labels(img: np.ndarray, cfg: PreprocessorConfig) -> Tuple[np.ndarray, int]:
    """
    Remove colored labels (NL, G1, G2, G3, etc.).

    Handles:
      - Colored labels on dark background
      - Colored labels on light/white background
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # ============================================================
    # Mask 1: Saturated colored labels (any hue)
    # ============================================================
    mask_colored = cv2.inRange(
        hsv,
        np.array([0, cfg.label_saturation_min, cfg.label_value_min]),
        np.array([180, 255, 255]),
    )

    # ============================================================
    # Mask 2: Labels on WHITE background
    # Small white blobs = labels (large white blobs = background)
    # ============================================================
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, cfg.label_white_bg_value_min]),
        np.array([180, cfg.label_white_bg_saturation_max, 255]),
    )
    white_ratio = white_mask.sum() / (255.0 * h * w)
    # Only consider as labels if white region is small (≤15%)
    mask_white_labels = white_mask if white_ratio < 0.15 else np.zeros_like(white_mask)

    # Combine
    combined = cv2.bitwise_or(mask_colored, mask_white_labels)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ============================================================
    # Filter connected components by area
    # ============================================================
    num, labels, stats, _ = cv2.connectedComponentsWithStats(combined, 8)
    clean_mask = np.zeros_like(combined)
    count = 0

    total_area = w * h
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if cfg.label_min_area <= area <= total_area * cfg.label_max_area_ratio:
            clean_mask[labels == i] = 255
            count += 1

    if count == 0:
        return img, 0

    # ============================================================
    # Inpaint
    # ============================================================
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.label_dilate_size, cfg.label_dilate_size),
    )
    dilated = cv2.dilate(clean_mask, dilate_kernel, iterations=2)
    result = cv2.inpaint(img, dilated, cfg.label_inpaint_radius, cv2.INPAINT_TELEA)

    if cfg.debug:
        print(f"   🔍 Labels removed: {count} component(s)")

    return result, count


# ================================================================
# ✅ MAIN: Smart preprocessing
# ================================================================

def preprocess_medical_image(
        image_path: str,
        config: Optional[PreprocessorConfig] = None,
        already_cropped: Optional[bool] = None,   # ← None = auto-detect
) -> PreprocessResult:
    """
    Smart preprocessing:
      - If image has header/side-panels → crop fan + remove labels
      - If image is already clean → only remove labels

    Args:
        image_path: path to input image
        config: PreprocessorConfig
        already_cropped: if True, skip fan-crop. If None, auto-detect.
    """
    cfg = config or PreprocessorConfig()

    if not image_path or not os.path.isfile(image_path):
        return PreprocessResult(success=False, reason="File not found")

    img = cv2.imread(image_path)
    if img is None:
        return PreprocessResult(success=False, reason="Could not read image")

    original_shape = img.shape[:2]

    # ✅ Downscale if too large
    h, w = img.shape[:2]
    if max(h, w) > cfg.max_input_dim:
        scale = cfg.max_input_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        if cfg.debug:
            print(f"   🔍 Resized: {w}×{h} → {int(w*scale)}×{int(h*scale)}")

    # ============================================================
    # STEP 1: Detect headers and side panels
    # ============================================================
    has_top_header, top_h = _detect_top_header(img, cfg)
    has_bottom_bar, bottom_h = _detect_bottom_bar(img, cfg)
    has_left_panel, left_w = _detect_side_panel(img, cfg, side="left")
    has_right_panel, right_w = _detect_side_panel(img, cfg, side="right")

    has_header = has_top_header or has_bottom_bar
    has_side_panels = has_left_panel or has_right_panel

    if cfg.debug:
        print(f"   🔍 Detected: top={has_top_header}({top_h}px) "
              f"bottom={has_bottom_bar}({bottom_h}px) "
              f"left={has_left_panel}({left_w}px) "
              f"right={has_right_panel}({right_w}px)")

    # ============================================================
    # STEP 2: Decide crop strategy
    # ============================================================
    if already_cropped is True:
        # ✅ Force no-crop (user or caller explicitly requested)
        cropped_img = img
        was_cropped = False
        decision = "forced no-crop"

    elif already_cropped is False:
        # ✅ Force crop (user explicitly requested)
        cropped_img, was_cropped = _crop_and_mask_fan(img, cfg)
        decision = "forced fan-crop"

    else:
        # ✅ Auto-detect
        if has_header or has_side_panels:
            # ✅ Header/side-panel present → crop fan
            cropped_img, was_cropped = _crop_and_mask_fan(img, cfg)
            decision = "auto: crop (header detected)"
        else:
            # ✅ Clean image → no crop
            cropped_img = img
            was_cropped = False
            decision = "auto: no crop (clean image)"

    if cfg.debug:
        print(f"   🔍 Decision: {decision}, cropped={was_cropped}")

    # ============================================================
    # STEP 3: Label removal (ALWAYS runs)
    # ============================================================
    cleaned_img, labels_removed = _remove_labels(cropped_img, cfg)

    # ============================================================
    # STEP 4: Ensure minimum size
    # ============================================================
    h2, w2 = cleaned_img.shape[:2]
    if min(h2, w2) < cfg.output_min_size:
        scale = cfg.output_min_size / float(min(h2, w2))
        cleaned_img = cv2.resize(
            cleaned_img,
            (int(w2 * scale), int(h2 * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    return PreprocessResult(
        success=True,
        image=cleaned_img,
        original_shape=original_shape,
        final_shape=cleaned_img.shape[:2],
        labels_removed=labels_removed,
        cropped=was_cropped,
        inpainted=labels_removed > 0,
        has_header=has_header,
        has_side_panels=has_side_panels,
        crop_decision=decision,
        reason="OK",
    )
