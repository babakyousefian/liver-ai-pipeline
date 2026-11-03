# src/ultra_ai.py

import re

import cv2
import numpy as np


class UltraAIDetector:
    def __init__(self, reader):
        self.reader = reader

    def detect(self, img):
        def gamma(img, g):
            inv = 1.0 / g
            table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
            return cv2.LUT(img, table)

        def preprocess_variants(gray):
            variants = []

            # OTSU
            _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(th1)

            # Adaptive
            th2 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            variants.append(th2)

            # Inverted
            variants.append(cv2.bitwise_not(gray))

            # Gamma variations
            variants.append(gamma(gray, 1.5))
            variants.append(gamma(gray, 2.0))

            return variants

        scales = [1.0, 1.5, 2.0]

        for scale in scales:
            resized = cv2.resize(img, None, fx=scale, fy=scale)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            step = int(0.15 * min(gray.shape))

            for y in range(0, gray.shape[0] - 40, step):
                for x in range(0, gray.shape[1] - 80, step):
                    roi = gray[y : y + 80, x : x + 160]

                    if roi.shape[0] < 40 or roi.shape[1] < 80:
                        continue

                    variants = preprocess_variants(roi)

                    votes = []

                    for v in variants:
                        results = self.reader.readtext(v, detail=0, paragraph=False)

                        for text in results:
                            text = text.strip().upper()

                            if len(text) <= 3:
                                match = re.search(r"\b(NL|G[1-4])\b", text)
                                if match:
                                    votes.append(match.group(1))

                    # 🎯 voting system
                    if votes:
                        return max(set(votes), key=votes.count)

        return "UNKNOWN"
