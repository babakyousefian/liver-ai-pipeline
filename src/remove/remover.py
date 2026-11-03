import csv
import os
import time

path_ = r"/home/bobby/Documents/Germany/Tasks/finalReports/labels.csv"

json_path_ = r"/home/bobby/Documents/Germany/Tasks/finalReports/unknown.json"

new_path_ = r"/home/bobby/Documents/Germany/Tasks/finalReports/cropped"


def readCSV(path: str) -> dict:
    if not isinstance(path, str):
        raise TypeError("\n must be string...!!!\n\n")
    else:
        time.sleep(1)
        print("\n prepare data\n")
        time.sleep(1)

        content_ = ""
        UNKNOWN = ["unknown", "Unknown", "UNKNOWN"]
        dictionary__ = dict()
        unknownLIST = list()

        time.sleep(1)
        print("\n read unknown labels... \n")
        time.sleep(1)

        with open(path, mode="r", newline="", encoding="utf-8") as file:
            content_ = csv.reader(file, delimiter=",")

            for row in content_:
                if row[-1] in UNKNOWN:
                    unknownLIST.append(row[:])
                else:
                    continue

            time.sleep(1)
            print("\n here is the final unknown images... \n")
            time.sleep(1)

            for i in unknownLIST:
                print(i, sep="\n", end="\n", flush=False)
                time.sleep(1)
            time.sleep(1)

            counter = 0
            for j in unknownLIST:
                dictionary__.setdefault(j[0], j[-1])
                counter += 1

            time.sleep(1)

        return dictionary__, counter


import json

import pandas as pd


def removeChecker(
    dictionary: dict, counter: int, absolutelyPatch: str, csv_path: str, json_path: str
):
    if not isinstance(dictionary, dict):
        raise TypeError("dictionary must be dict")

    elif not isinstance(counter, int):
        raise TypeError("counter must be int")

    elif not isinstance(absolutelyPatch, str):
        raise TypeError("absolutelyPatch must be str")

    elif not isinstance(csv_path, str):
        raise TypeError("csv_path must be str")

    elif not isinstance(json_path, str):
        raise TypeError("json_path must be str")

    else:
        time.sleep(1)
        print("\n🔍 scanning directory...\n")
        time.sleep(1)

        removed = []

        # -------- REMOVE FILES -------- #
        for filename in dictionary.keys():
            full_path = os.path.join(absolutelyPatch, filename)

            if os.path.exists(full_path):
                try:
                    os.remove(full_path)

                    time.sleep(0.1)
                    print(f"✅ removed: {filename}")
                    time.sleep(0.1)

                    removed.append(filename)

                except Exception as e:
                    time.sleep(1)
                    print(f"❌ error removing {filename}: {e}")
                    time.sleep(1)
            else:
                time.sleep(1)
                print(f"⚠️ not found: {filename}")
                time.sleep(1)

        print("\n-----------------------------\n")

        time.sleep(1)
        print(f"🧠 removed {len(removed)} / {counter} images")
        time.sleep(1)

        # -------- UPDATE CSV -------- #
        time.sleep(1)
        print("\n📄 updating CSV...\n")
        time.sleep(1)

        df = pd.read_csv(csv_path)

        before = len(df)

        df = df[~df["liverName"].isin(removed)]

        after = len(df)

        df.to_csv(csv_path, index=False)

        time.sleep(1)
        print(f"✅ CSV updated: {before} → {after}")
        time.sleep(1)

        # -------- UPDATE JSON -------- #
        time.sleep(1)
        print("\n📁 updating JSON...\n")
        time.sleep(1)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                unknown_list = json.load(f)
        except:
            unknown_list = []

        # remove deleted ones
        unknown_list = [x for x in unknown_list if x not in removed]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(unknown_list, f, indent=4)

        time.sleep(1)
        print(f"✅ JSON updated: {len(unknown_list)} remaining")
        time.sleep(1)

        time.sleep(1)
        print("\n🎯 DATASET CLEANED SUCCESSFULLY\n")
        time.sleep(1)


def removeUnknownImageLabels(
    cropped_path_: str, cammaSeperatedValue_: str, javaScriptObjectNotation_: str
):
    dictionary, counter = readCSV(path=cammaSeperatedValue_)

    removeChecker(
        dictionary=dictionary,
        counter=counter,
        absolutelyPatch=cropped_path_,
        csv_path=cammaSeperatedValue_,
        json_path=javaScriptObjectNotation_,
    )


def main():
    removeUnknownImageLabels(new_path_, path_, json_path_)


if __name__ == "__main__":
    main()
