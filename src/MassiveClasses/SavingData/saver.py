import time

import pandas as pd


class Save:
    def __init__(self, results, CSV_PATH):
        self.results = results
        self.CSV_PATH = CSV_PATH

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def saver(self):
        time.sleep(1)
        print("\n📄 Initial CSV saved\n")
        time.sleep(1)

        df = pd.DataFrame(self.results)
        df.to_csv(self.CSV_PATH, index=False)

        time.sleep(1)
        print("\n save to csv successfully...\n\n")
        time.sleep(1)
