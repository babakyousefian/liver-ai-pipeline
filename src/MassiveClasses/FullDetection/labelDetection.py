import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


class LabelDetection:
    def __init__(self, INPUT_DIR, files, reader, LABEL_REGEX):
        self.INPUT_DIR = INPUT_DIR
        self.files = files
        self.reader = reader
        self.LABEL_REGEX = LABEL_REGEX
        self.img = ""

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def labelDetector(self):
        time.sleep(1)
        print("\n🧠 PHASE 1: PARALLEL LABEL DETECTION (FULL IMAGE)\n")
        time.sleep(1)

        def process_image(args):
            idx, file = args

            path = os.path.join(self.INPUT_DIR, file)
            self.img = cv2.imread(path)

            if self.img is None:
                return None

            name = f"Liver{idx:04d}.png"

            label = self.__detect_label()

            if label == "UNKNOWN":
                label = self.__detect_label_advanced()

            if label == "UNKNOWN":
                label = self.__detect_label_full_image()

            return {"liverName": name, "liverLabel": label, "img": self.img}

        with ThreadPoolExecutor(max_workers=8) as executor:
            processed = list(
                tqdm(
                    executor.map(process_image, enumerate(self.files, 1)),
                    total=len(self.files),
                    desc="Detection Phase",
                )
            )

        results = []
        image_cache = {}

        for item in processed:
            if item is None:
                continue

            results.append({"liverName": item["liverName"], "liverLabel": item["liverLabel"]})

            image_cache[item["liverName"]] = item["img"]

        return results, image_cache

    # ------------------------ detect fully ------------------------ #

    def __detect_label_full_image(self):
        mask = self.get_label_mask()

        # find contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        H, W = self.img.shape[:2]

        candidates = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            area = w * h

            # filter out too small or too large areas
            if area < 50 or area > 5000:
                continue

            # 🔥 expand ROI
            pad = 10
            roi = self.img[
                max(0, y - pad) : min(H, y + h + pad),
                max(0, x - pad) : min(W, x + w + pad),
            ]

            candidates.append(roi)

        # -------- OCR -------- #
        for roi in candidates:
            try:
                result = self.reader.readtext(roi)

                for _, text, conf in result:
                    text = text.strip().upper()

                    if len(text) <= 3:
                        match = self.LABEL_REGEX.search(text)
                        if match:
                            return match.group(1)

            except:
                continue

        return "UNKNOWN"

    # ------------------ BASIC DETECTION ------------------ #
    def __detect_label(self):
        mask = self.__get_green_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w < 15 or h < 15:
                continue

            roi = self.img[y : y + h, x : x + w]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2)

            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

            votes = []

            results = self.reader.readtext(thresh, detail=0)

            for text in results:
                text = text.strip().upper()

                if len(text) <= 3:
                    match = self.LABEL_REGEX.search(text)

                    if match:
                        votes.append(match.group(1))

            if votes:
                label = max(set(votes), key=votes.count)

                if label == "NL" and votes.count("NL") < 2:
                    return "UNKNOWN"

                return label

        return "UNKNOWN"

    # ------------------ ADVANCED DETECTION ------------------ #
    def __detect_label_advanced(self):
        def enhance(roi):
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(3.5, (8, 8)).apply(l)
            return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        mask = self.__get_green_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w < 10 or h < 10:
                continue

            roi = enhance(self.img[y : y + h, x : x + w])

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.5, fy=2.5)

            candidates = [
                cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                ),
                cv2.bitwise_not(gray),
                gray,
            ]

            votes = []

            for c in candidates:
                results = self.reader.readtext(c, detail=0)

                for t in results:
                    t = t.strip().upper()
                    if len(t) <= 3:
                        m = self.LABEL_REGEX.search(t)
                        if m:
                            votes.append(m.group(1))

            # 🎯 تصمیم نهایی
            if votes:
                label = max(set(votes), key=votes.count)

                # جلوگیری از NL fake
                if label == "NL" and votes.count("NL") < 2:
                    return "UNKNOWN"

                return label

        return "UNKNOWN"

    def get_label_mask(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # 🎯 WIDE RANGE (همه رنگ‌های تو)

        masks = []

        # 🟢 green (full spectrum)
        masks.append(cv2.inRange(hsv, (25, 30, 30), (95, 255, 255)))

        # 🟡 yellow / gold
        masks.append(cv2.inRange(hsv, (15, 80, 120), (40, 255, 255)))

        # 🟠 orange
        masks.append(cv2.inRange(hsv, (5, 120, 120), (20, 255, 255)))

        # ⚪ cream / pale (low saturation but bright)
        masks.append(cv2.inRange(hsv, (0, 0, 180), (180, 60, 255)))

        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    # ------------------ MULTI MASK ------------------ #
    def __get_green_mask(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        lower = np.array([25, 30, 30])
        upper = np.array([100, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

        return mask
