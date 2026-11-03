import warnings

warnings.filterwarnings("ignore")

from src.MajorProcessing.main_engine import PreProcessingImage


def preProcessor():
    pre_ = PreProcessingImage()
    pre_.runPreProcessor()


def main():
    preProcessor()


if __name__ == "__main__":
    main()
