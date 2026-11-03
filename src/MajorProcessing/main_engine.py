# src/main_engine.py

# ================= SILENCE EVERYTHING =================

import os

# pygame
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Torch / CUDA noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

# ------------------ SILENCE ------------------ #
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*Using CPU.*")
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="easyocr")
warnings.filterwarnings("ignore", message=".*CPU.*")


from src.MajorProcessing.pipeline import Pipeline


class PreProcessingImage:
    def __init__(self):
        self.pipeline = Pipeline()

    def __repr__(self):
        return "PreProcessingImage()"

    def __str__(self):
        return "Liver AI PreProcessing Pipeline"

    def runPreProcessor(self):
        self.pipeline.run()
