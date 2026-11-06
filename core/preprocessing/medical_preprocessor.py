# core/preprocessing/medical_preprocessor.py
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PreprocessorConfig:
    # Crop settings
    crop_threshold: int = 25
    crop_morph_size: int = 15
    min_crop_area_ratio: float = 0.10
    # Label settings
    label_saturation_min: int = 100
    label_value_min: int = 80
    label_min_area: int = 20
    label_dilate_size: int = 5
    label_inpaint_radius: int = 3
    output_min_size: int = 224
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
    reason: str = ""

    def save(self, path: str) -> bool:
        if self.image is None: return False
        try:
            return cv2.imwrite(path, self.image)
        except Exception:
            return False


def _extract_fan_mask(img: np.ndarray, cfg: PreprocessorConfig) -> np.ndarray:
    """ماسک دقیق بادبزن (فن) رو با استفاده از Convex Hull پیدا می‌کنه."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. ماسک پیکسل‌های روشن (بافت) - حذف متن‌های سفید
    tissue = cv2.inRange(gray, cfg.crop_threshold, 230)

    # 2. حذف متن‌های ریز
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, k_small, iterations=1)

    # 3. ادغام بافت‌ها
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.crop_morph_size, cfg.crop_morph_size))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, k_big, iterations=3)

    # 4. بزرگ‌ترین کامپوننت (خود فن)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, 8)
    if num <= 1: return np.ones_like(gray) * 255

    best_idx, best_area, total_area = -1, 0, w * h
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < total_area * cfg.min_crop_area_ratio: continue
        if area > best_area:
            best_area = area
            best_idx = i

    if best_idx == -1: return np.ones_like(gray) * 255

    mask = (labels == best_idx).astype(np.uint8) * 255

    # 5. ساخت Convex Hull برای شکل بادبزن
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return mask

    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)

    clean_mask = np.zeros_like(gray)
    cv2.drawContours(clean_mask, [hull], -1, 255, -1)
    return clean_mask


def _crop_and_mask_fan(img: np.ndarray, cfg: PreprocessorConfig) -> Tuple[np.ndarray, bool]:
    """برش دقیق بادبزن و سیاه کردن پس‌زمینه با حاشیه امن برای جلوگیری از بریدن بافت کبد."""
    h, w = img.shape[:2]
    fan_mask = _extract_fan_mask(img, cfg)

    x, y, ww, hh = cv2.boundingRect(fan_mask)
    if ww == 0 or hh == 0: return img, False

    # ✅ افزایش حاشیه امن (Padding) از ۲ درصد به ۵ درصد
    # این کار باعث میشه یه ذره از فضای اطراف فن هم بمونه تا لبه‌های کبد بریده نشن
    pad_x = int(ww * 0.05)
    pad_y = int(hh * 0.05)

    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(w, x + ww + pad_x), min(h, y + hh + pad_y)

    cropped_img = img[y0:y1, x0:x1].copy()
    cropped_mask = fan_mask[y0:y1, x0:x1]

    # پس‌زمینه خارج از بادبزن کاملاً سیاه میشه
    cropped_img[cropped_mask == 0] = [0, 0, 0]
    return cropped_img, True


def _remove_labels(img: np.ndarray, cfg: PreprocessorConfig) -> Tuple[np.ndarray, int]:
    """حذف لیبل‌های رنگی (مثل G1)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    mask = cv2.inRange(hsv, np.array([0, cfg.label_saturation_min, cfg.label_value_min]), np.array([180, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean_mask = np.zeros_like(mask)
    count = 0

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if cfg.label_min_area <= area <= (w * h * 0.05):
            clean_mask[labels == i] = 255
            count += 1

    if count == 0: return img, 0

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.label_dilate_size, cfg.label_dilate_size))
    dilated_mask = cv2.dilate(clean_mask, dilate_kernel, iterations=2)
    result = cv2.inpaint(img, dilated_mask, cfg.label_inpaint_radius, cv2.INPAINT_TELEA)
    return result, count


def preprocess_medical_image(image_path: str, config: Optional[PreprocessorConfig] = None,
                             already_cropped: bool = False) -> PreprocessResult:
    cfg = config or PreprocessorConfig()
    if not image_path or not os.path.isfile(image_path):
        return PreprocessResult(success=False, reason="File not found")

    img = cv2.imread(image_path)
    if img is None: return PreprocessResult(success=False, reason="Could not read")

    original_shape = img.shape[:2]
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        scale = 1600.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    if already_cropped:
        cropped_img, was_cropped = img, False
    else:
        cropped_img, was_cropped = _crop_and_mask_fan(img, cfg)

    cleaned_img, labels_removed = _remove_labels(cropped_img, cfg)

    h2, w2 = cleaned_img.shape[:2]
    if min(h2, w2) < cfg.output_min_size:
        scale = cfg.output_min_size / float(min(h2, w2))
        cleaned_img = cv2.resize(cleaned_img, (int(w2 * scale), int(h2 * scale)), interpolation=cv2.INTER_CUBIC)

    return PreprocessResult(
        success=True, image=cleaned_img, original_shape=original_shape,
        final_shape=cleaned_img.shape[:2], labels_removed=labels_removed,
        cropped=was_cropped, inpainted=labels_removed > 0, reason="OK"
    )