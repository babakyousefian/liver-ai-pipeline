import os
import time


class Loader:
    def __init__(self, INPUT_DIR):
        self.INPUT_DIR = INPUT_DIR

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def loadSomeFiles(self):
        time.sleep(1)
        print("\n📂 Load Files\n")
        time.sleep(1)

        files = sorted(
            [f for f in os.listdir(self.INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

        return files
