import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


# src/MajorProcessing/detection.py


import contextlib

# silence stdout temporarily
import sys


class DummyFile:
    def write(self, x):
        pass

    def flush(self):
        pass


@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = DummyFile()
    try:
        yield
    finally:
        sys.stdout = old_stdout


import re

import cv2
import easyocr
import numpy as np

# ------------------ OCR ------------------ #
import torch

use_gpu = torch.cuda.is_available()

if use_gpu:
    with suppress_stdout():
        reader = easyocr.Reader(
            ["en"], gpu=False, model_storage_directory="./packages/facilities/models"
        )

else:
    with suppress_stdout():
        reader = easyocr.Reader(
            ["en"], gpu=False, model_storage_directory="./packages/facilities/models"
        )

MODEL_PATH = r"/home/bobby/Documents/Germany/Tasks/packages/facilities/models"


class Detection:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False, model_storage_directory=MODEL_PATH)

        self.LABEL_REGEX = re.compile(r"\b(NL|G[1-4])\b")

    # ---------------- CROP ---------------- #
    def crop_image(self, img):
        h, w, _ = img.shape
        return img[int(0.15 * h) : h - int(0.10 * h), int(0.08 * w) : w - int(0.18 * w)]

    # ---------------- MASK ---------------- #
    def get_green_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([25, 30, 30])
        upper = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    def get_label_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([15, 100, 120])
        upper_yellow = np.array([35, 255, 255])

        lower_orange = np.array([5, 120, 120])
        upper_orange = np.array([15, 255, 255])

        lower_green = np.array([25, 30, 30])
        upper_green = np.array([90, 255, 255])

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_yellow, upper_yellow),
            cv2.inRange(hsv, lower_orange, upper_orange),
        )

        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_green, upper_green))

        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # ---------------- BASIC ---------------- #
    def detect_basic(self, img):
        mask = self.get_green_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w < 15 or h < 15:
                continue

            roi = img[y : y + h, x : x + w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2)

            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

            results = self.reader.readtext(thresh, detail=0)

            for text in results:
                text = text.strip().upper()
                if len(text) <= 3:
                    m = self.LABEL_REGEX.search(text)
                    if m:
                        return m.group(1)

        return "UNKNOWN"

    # ---------------- ADVANCED ---------------- #
    def detect_advanced(self, img):
        def enhance(roi):
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(3.5, (8, 8)).apply(l)
            return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        mask = self.get_green_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w < 10 or h < 10:
                continue

            roi = enhance(img[y : y + h, x : x + w])
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

            for c in candidates:
                results = self.reader.readtext(c, detail=0)
                for t in results:
                    t = t.strip().upper()
                    if len(t) <= 3:
                        m = self.LABEL_REGEX.search(t)
                        if m:
                            return m.group(1)

        return "UNKNOWN"

    # ---------------- FULL ---------------- #
    def detect_full(self, img):
        mask = self.get_label_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        H, W = img.shape[:2]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w * h < 50 or w * h > 5000:
                continue

            roi = img[max(0, y - 10) : min(H, y + h + 10), max(0, x - 10) : min(W, x + w + 10)]

            try:
                result = self.reader.readtext(roi)
                for _, text, conf in result:
                    text = text.upper().strip()
                    if conf > 0.4 and text in ["NL", "G1", "G2", "G3"]:
                        return text
            except:
                continue

        return "UNKNOWN"

    # ---------------- PIPELINE ---------------- #
    def detect_full_pipeline(self, img):
        label = self.detect_full(img)

        if label == "UNKNOWN":
            label = self.detect_basic(img)

        if label == "UNKNOWN":
            label = self.detect_advanced(img)

        return label

    # ---------------- INPAINT ---------------- #
    def remove_label(self, img):
        mask = self.get_label_mask(img)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=3)
        return cv2.inpaint(img, mask, 5, cv2.INPAINT_NS)

    # ---------------- ULTRA ---------------- #
    def detect_ultra(self, img):
        from src.MajorProcessing.ultra import UltraDetector

        return UltraDetector(self.reader).detect(img)

    def detect_ultra_ai(self, img):
        from src.MajorProcessing.ultra_ai import UltraAIDetector

        return UltraAIDetector(self.reader).detect(img)
