import json
import time

import pandas as pd


class ReportBC:
    def __init__(self, CSV_PATH, JSON_PATH):
        self.CSV_PATH = CSV_PATH
        self.JSON_PATH = JSON_PATH

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def reporterForBC(self):
        time.sleep(1)
        print("\n📊 FINAL LABEL REPORT\n")
        time.sleep(1)

        df = pd.read_csv(self.CSV_PATH)

        unknown_count = (df["liverLabel"] == "UNKNOWN").sum()
        unknown_names = df[df["liverLabel"] == "UNKNOWN"]["liverName"].tolist()

        time.sleep(1)
        print(f"\n🧠 Remaining UNKNOWN: {unknown_count}")
        time.sleep(1)

        with open(self.JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(unknown_names, f, indent=4)
