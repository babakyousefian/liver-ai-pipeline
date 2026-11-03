import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from src.MassiveClasses.FullDetection.labelDetection import LabelDetection


class Crop:
    def __init__(self, image_cache, RAW_DIR, OUTPUT_DIR, INPUT_DIR, files, reader, LABEL_REGEX):
        self.image_cache = image_cache
        self.RAW_DIR = RAW_DIR
        self.OUTPUT_DIR = OUTPUT_DIR
        self.INPUT_DIR = INPUT_DIR
        self.files = files
        self.reader = reader
        self.LABEL_REGEX = LABEL_REGEX
        self.img = ""
        self.cropped = ""
        self.clean = ""
        self.g = 1.5

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def cropper(self):
        time.sleep(1)
        print("\n✂️ FINAL CROPPING + CLEANING\n")
        time.sleep(1)

        for name, self.img in tqdm(self.image_cache.items(), desc="Cropping Stage"):
            # ✂️ CROP FIRST
            self.cropped = self.__crop_image()

            self.img = self.cropped

            # save RAW (cropped with label)
            cv2.imwrite(os.path.join(self.RAW_DIR, name), self.img)

            # 🧹 REMOVE LABEL
            self.clean = self.__remove_label()

            # save FINAL (cropped + cleaned)
            cv2.imwrite(os.path.join(self.OUTPUT_DIR, name), self.clean)

    def __crop_image(self):
        h, w, _ = self.img.shape
        return self.img[int(0.15 * h) : h - int(0.10 * h), int(0.08 * w) : w - int(0.18 * w)]

    def gamma_correction(self):
        inv = 1.0 / self.g
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(self.img, table)

    def __remove_label(self):
        ld_ = LabelDetection(
            INPUT_DIR=self.INPUT_DIR,
            files=self.files,
            reader=self.reader,
            LABEL_REGEX=self.LABEL_REGEX,
        )

        # mask = ld_.get_label_mask()

        def get_label_mask():
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

            # 🟡 yellow / gold
            lower_yellow = np.array([15, 100, 120])
            upper_yellow = np.array([35, 255, 255])

            # 🟠 orange
            lower_orange = np.array([5, 120, 120])
            upper_orange = np.array([15, 255, 255])

            # 🟢 green
            lower_green = np.array([25, 30, 30])
            upper_green = np.array([90, 255, 255])

            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            mask = cv2.bitwise_or(mask_yellow, mask_orange)
            mask = cv2.bitwise_or(mask, mask_green)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            return mask

        mask = get_label_mask()
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=3)
        return cv2.inpaint(self.img, mask, 5, cv2.INPAINT_NS)
