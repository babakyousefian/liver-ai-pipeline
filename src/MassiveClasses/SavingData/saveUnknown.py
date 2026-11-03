import time

from src.unknownChecker import unknown


class SaveUnknown:
    def __init__(self, RAW_DIR, CSV_PATH, reader):
        self.RAW_DIR = RAW_DIR
        self.CSV_PATH = CSV_PATH
        self.reader = reader

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def saveUnknownImages(self):
        time.sleep(1)
        print("\n⚡ SAFE UNKNOWN RESOLUTION\n")
        time.sleep(1)

        unknown.resolve_unknown_labels(self.RAW_DIR, self.CSV_PATH, self.reader)

        time.sleep(1)
        print("\n📊 AFTER SAFE RESOLVE\n")
        time.sleep(1)

        unknown.unknownLabels()
