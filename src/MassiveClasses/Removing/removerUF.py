import time

from src.remove import remover


class RemoverUF:
    def __init__(self, OUTPUT_DIR, CSV_PATH, JSON_PATH):
        self.OUTPUT_DIR = OUTPUT_DIR
        self.CSV_PATH = CSV_PATH
        self.JSON_PATH = JSON_PATH

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def remove(self):
        time.sleep(1)
        print("\n🚀 REMOVE UNKNOWN FILES\n")
        time.sleep(1)

        remover.removeUnknownImageLabels(self.OUTPUT_DIR, self.CSV_PATH, self.JSON_PATH)
