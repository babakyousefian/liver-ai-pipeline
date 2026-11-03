# src/ultra.py

import re

import cv2
import numpy as np


class UltraDetector:
    def __init__(self, reader):
        self.reader = reader
        self.LABEL_REGEX = re.compile(r"\b(NL|G[1-4])\b")

    def detect(self, img):
        H, W = img.shape[:2]

        # ---------- helpers ---------- #
        def gamma(img, g=1.5):
            inv = 1.0 / g
            table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
            return cv2.LUT(img, table)

        def enhance(roi):
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
            return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        # ---------- preprocess ---------- #
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---------- MSER ---------- #
        mser = cv2.MSER_create()
        mser.setMinArea(60)
        mser.setMaxArea(2000)

        regions, _ = mser.detectRegions(gray)

        rois = []

        for p in regions:
            x, y, w, h = cv2.boundingRect(p)

            if w < 10 or h < 10:
                continue

            if w > 150 or h > 80:
                continue

            ratio = w / float(h)
            if ratio < 0.5 or ratio > 6:
                continue

            rois.append((x, y, w, h))

        # 🔥 LIMIT REGIONS
        rois = sorted(rois, key=lambda r: r[2] * r[3], reverse=True)[:12]

        # ---------- OCR ---------- #
        for x, y, w, h in rois:
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(W, x + w + pad)
            y2 = min(H, y + h + pad)

            roi = img[y1:y2, x1:x2]

            roi = enhance(roi)
            roi = gamma(roi, 1.5)

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.resize(gray_roi, None, fx=2, fy=2)

            variants = [
                gray_roi,
                cv2.threshold(gray_roi, 0, 255, cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(
                    gray_roi,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                ),
                cv2.bitwise_not(gray_roi),
            ]

            for v in variants:
                try:
                    results = self.reader.readtext(v, detail=0)
                except:
                    continue

                for text in results:
                    text = text.strip().upper()

                    if len(text) <= 3:
                        m = self.LABEL_REGEX.search(text)
                        if m:
                            return m.group(1)

        # ---------- FINAL fallback ---------- #
        try:
            results = self.reader.readtext(img, detail=0)
            for text in results:
                text = text.strip().upper()
                m = self.LABEL_REGEX.search(text)
                if m:
                    return m.group(1)
        except:
            pass

        return "UNKNOWN"
