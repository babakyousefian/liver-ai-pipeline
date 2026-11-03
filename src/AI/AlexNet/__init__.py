# src/AI/AlexNet/__init__.py
# -*- coding: utf-8 -*-

"""AlexNet Fine-Tuning Pipeline for Liver Ultrasound Classification."""

from .AlexNet import AlexNetTrainer
from .AlexNetReports import AlexNetReporter

__version__ = "1.0.0"
__author__ = "Babak Yousefian"

__all__ = ["AlexNetTrainer", "AlexNetReporter"]