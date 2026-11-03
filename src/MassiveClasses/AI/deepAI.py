import re
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class DeepAI:
    def __init__(self, CSV_PATH, image_cache, reader):
        self.CSV_PATH = CSV_PATH
        self.image_cache = image_cache
        self.reader = reader
        self.img = ""

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def deepAI(self):
        time.sleep(1)
        print("\n🤖 ULTRA AI resolving UNKNOWN...\n")
        time.sleep(1)

        df = pd.read_csv(self.CSV_PATH)
        unknown_rows = df[df["liverLabel"] == "UNKNOWN"]

        for idx in tqdm(unknown_rows.index, desc="ULTRA AI Stage"):
            self.img = self.image_cache.get(df.at[idx, "liverName"], None)

            if self.img is None:
                continue

            label = self.__detect_label_ultra_ai()

            if label != "UNKNOWN":
                df.at[idx, "liverLabel"] = label

        df.to_csv(self.CSV_PATH, index=False)

    def __detect_label_ultra_ai(self):
        H, W = self.img.shape[:2]

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

        # 🔥 multi-scale sliding windows
        scales = [1.0, 1.5, 2.0]

        for scale in scales:
            resized = cv2.resize(self.img, None, fx=scale, fy=scale)
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

                    # 🎯 voting system (like ensemble model)
                    if votes:
                        return max(set(votes), key=votes.count)

        return "UNKNOWN"
