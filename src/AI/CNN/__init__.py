# src/AI/CNN/__init__.py
# -*- coding: utf-8 -*-

"""CNN Fine-Tuning Pipeline for Liver Ultrasound Classification."""

from .CNN import CNNTrainer
from .CNNReports import CNNReporter

__version__ = "1.0.0"
__author__ = "Babak Yousefian"

__all__ = ["CNNTrainer", "CNNReporter"]