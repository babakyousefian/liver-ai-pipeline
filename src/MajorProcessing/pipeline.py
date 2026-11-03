# src/pipeline.py

import os
import time

import cv2
import pandas as pd
from tqdm import tqdm

from src.MajorProcessing.detection import Detection
from src.MassiveClasses.Ending.ender import End

# from src.MassiveClasses.Audio.music import Music
from src.MassiveClasses.LoadFiles.loader import Loader
from src.MassiveClasses.Removing.removerUF import RemoverUF
from src.MassiveClasses.Reporter.finalReport import FinalReporter
from src.MassiveClasses.Reporter.reportBeforeCrop import ReportBC
from src.MassiveClasses.SavingData.saver import Save
from src.MassiveClasses.SavingData.saveUnknown import SaveUnknown
from src.MassiveClasses.Timer.timer import Timer

INPUT_DIR = "./packages/data/my_Data/liver/liver/"
OUTPUT_DIR = "./finalReports/cropped/"
RAW_DIR = "./finalReports/cropped_raw/"
CSV_PATH = "./finalReports/labels.csv"
# MUSIC_PATH = "./packages/facilities/assets/musics/music.mp3"
JSON_PATH = "./finalReports/unknown.json"
FINAL_REPORTS = "./finalReports/Liver_AI_Final_Report.pdf"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


class Pipeline:
    def __init__(self):
        self.detector = Detection()

    def run(self):
        timer__ = Timer()
        total_start = timer__.timer()

        # myMusic = Music(MUSIC_PATH=MUSIC_PATH)
        # myMusic.play_music()

        load__ = Loader(INPUT_DIR=INPUT_DIR)
        files = load__.loadSomeFiles()

        results = []

        print("\n🧠 PHASE 1: LABEL DETECTION (FULL IMAGE)\n")

        for idx, file in enumerate(tqdm(files, desc="Detection Phase"), 1):
            path = os.path.join(INPUT_DIR, file)
            img = cv2.imread(path)

            if img is None:
                continue

            name = f"Liver{idx:03d}.png"

            cv2.imwrite(os.path.join(RAW_DIR, name), img)

            label = self.detector.detect_full_pipeline(img)

            results.append({"liverName": name, "liverLabel": label})

        saver = Save(results=results, CSV_PATH=CSV_PATH)
        saver.saver()

        saveUImage = SaveUnknown(RAW_DIR=RAW_DIR, CSV_PATH=CSV_PATH, reader=self.detector.reader)
        saveUImage.saveUnknownImages()

        print("\n🚀 ULTRA resolving UNKNOWN...\n")

        df = pd.read_csv(CSV_PATH)
        unknown_rows = df[df["liverLabel"] == "UNKNOWN"]

        for idx in tqdm(unknown_rows.index, desc="ULTRA Stage"):
            img_path = os.path.join(RAW_DIR, df.at[idx, "liverName"])
            img = cv2.imread(img_path)

            if img is None:
                continue

            label = self.detector.detect_ultra(img)

            if label != "UNKNOWN":
                df.at[idx, "liverLabel"] = label

        df.to_csv(CSV_PATH, index=False)

        print("\n🤖 ULTRA AI resolving UNKNOWN...\n")

        df = pd.read_csv(CSV_PATH)
        unknown_rows = df[df["liverLabel"] == "UNKNOWN"]

        for idx in tqdm(unknown_rows.index, desc="ULTRA AI Stage"):
            img_path = os.path.join(RAW_DIR, df.at[idx, "liverName"])
            img = cv2.imread(img_path)

            if img is None:
                continue

            label = self.detector.detect_ultra_ai(img)

            if label != "UNKNOWN":
                df.at[idx, "liverLabel"] = label

        df.to_csv(CSV_PATH, index=False)

        reporter_ = ReportBC(CSV_PATH=CSV_PATH, JSON_PATH=JSON_PATH)
        reporter_.reporterForBC()

        print("\n✂️ FINAL CROPPING + CLEANING\n")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cropping Stage"):
            img_path = os.path.join(RAW_DIR, row["liverName"])
            img = cv2.imread(img_path)

            if img is None:
                continue

            cropped = self.detector.crop_image(img)
            clean = self.detector.remove_label(cropped)

            cv2.imwrite(os.path.join(OUTPUT_DIR, row["liverName"]), clean)

        removeUF_ = RemoverUF(OUTPUT_DIR=OUTPUT_DIR, CSV_PATH=CSV_PATH, JSON_PATH=JSON_PATH)
        removeUF_.remove()

        end_ = End(total_start=total_start)
        total_end = end_.ending()

        fr_ = FinalReporter(
            CSV_PATH=CSV_PATH,
            FINAL_REPORTS=FINAL_REPORTS,
            total_start=total_start,
            total_end=total_end,
            files=files,
            # myMusic=myMusic
        )
        fr_.finalReporter()
