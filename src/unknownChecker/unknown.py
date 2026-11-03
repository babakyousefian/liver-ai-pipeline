import csv
import json
import os
import re
import time

import cv2
import easyocr
import numpy as np
import pandas as pd

input_csv = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\labels.csv"

# BASE_DIR = os.path.dirname(__file__)

jsonAddress = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\unknown.json"


LABEL_REGEX = re.compile(r"\b(NL|G[0-4])\b")


def resolve_unknown_labels(image_dir, csv_path, reader):
    df = pd.read_csv(csv_path)

    unknown_rows = df[df["liverLabel"] == "UNKNOWN"]

    print(f"\n🧠 Resolving {len(unknown_rows)} UNKNOWN labels...\n")

    for idx, row in unknown_rows.iterrows():
        img_path = os.path.join(image_dir, row["liverName"])
        img = cv2.imread(img_path)

        if img is None:
            continue

        # resize
        img_small = cv2.resize(img, None, fx=0.5, fy=0.5)

        try:
            results = reader.readtext(img_small, detail=0)
        except Exception as e:
            print(f"⚠️ OCR failed on {row['liverName']}")
            continue

        for text in results:
            text = text.strip().upper()

            match = LABEL_REGEX.search(text)
            if match:
                df.at[idx, "liverLabel"] = match.group(1)
                print(f"✔ fixed: {row['liverName']} → {match.group(1)}")
                break

    df.to_csv(csv_path, index=False)

    print("\n🔥 UNKNOWN labels resolved safely")


# ---------- REPORT ---------- #


def unknownNumber():
    my_reader = ""
    unknownData = list()
    with open(input_csv, mode="r", newline="", encoding="utf-8") as f:
        my_reader = csv.reader(f, delimiter=",")
        for row in my_reader:
            if row[0] == "liverName":
                continue
            else:
                unknownData.append(row[-1])
        counter = 0
        checker_ = ["UNKNOWN"]
        for datum in unknownData:
            if datum in checker_:
                counter += 1
            else:
                continue
        print("the number of UNKNOWN data is : {}".format(counter))


def unknownName():
    my_reader = ""
    unknownDataName = list()
    checker_ = ["UNKNOWN"]
    with open(input_csv, mode="r", newline="", encoding="utf-8") as f:
        my_reader = csv.reader(f, delimiter=",")
        for row in my_reader:
            if row[-1] in checker_:
                unknownDataName.append(row[0])
            else:
                continue
    return unknownDataName


def generateJson(unknownDataName: list | None = None):
    if unknownDataName is None:
        unknownDataName = []
    else:
        if not isinstance(unknownDataName, list):
            raise TypeError("unknownDataName must be a list...!!!")
        else:
            with open(jsonAddress, mode="w", newline="\n", encoding="utf-8") as f:
                json.dump(unknownDataName, f, ensure_ascii=False, indent=4)
            print("\ngenerate json successfully...!!!")


def readJson(file_path: str | None = None):
    if file_path is None:
        file_path = ""
    else:
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a str...!!!")
        else:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                print("\n⚠️ JSON is empty or missing")
                return
            else:
                with open(file_path, mode="r", newline="\n", encoding="utf-8") as f:
                    try:
                        out_ = json.load(f)
                        print("\n🧠 UNKNOWN images:", out_)
                    except json.JSONDecodeError:
                        print("\n⚠️ JSON corrupted or empty")


def unknownLabels():
    print("\n🔥 NOW HERE IS DETAILS OF UNKNOWN labels")
    print("ready 🚀\n\n")
    unknownNumber()
    time.sleep(1)
    unknownDataName = unknownName()
    time.sleep(1)
    generateJson(unknownDataName)
    time.sleep(1)
    readJson(jsonAddress)
    time.sleep(1)
    print("\nall operations are complete successfully...!!!")
    time.sleep(1)


if __name__ == "__main__":
    unknownLabels()
