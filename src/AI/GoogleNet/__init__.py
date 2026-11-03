# src/AI/GoogleNet/__init__.py
# -*- coding: utf-8 -*-

"""GoogleNet (Inception-V1) Fine-Tuning Pipeline for Liver Ultrasound Classification."""

from .GoogleNet import GoogleNetTrainer
from .GoogleNetReports import GoogleNetReporter

__version__ = "1.0.0"
__author__ = "Babak Yousefian"

__all__ = ["GoogleNetTrainer", "GoogleNetReporter"]