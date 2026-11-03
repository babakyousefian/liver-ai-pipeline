# src/models/CNNLSTM/__init__.py
# -*- coding: utf-8 -*-

"""
CNNLSTM Module - Complete Implementation from Scratch
=====================================================
This module provides a complete implementation of CNN-LSTM architecture
without any pretrained weights.

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 1.0.0 (ULTIMATE EDITION)
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from .blocks import (
    CNNFeatureExtractor,
    LSTMHead,
    init_weights,
    get_block_info,
)
from .cnnlstm import CNNLSTM, CNNLSTMConfig, create_cnnlstm

from .cnnlstm import (
    analyze_memory_usage,
    calculate_flops,
    count_parameters,
    get_device,
    get_layer_output_shapes,
    get_model_size,
    print_model_summary,
    profile_model,
    set_seed,
    test_inference_speed,
    validate_forward_pass,
)

__version__ = "1.0.0"
__author__ = "Babak Yousefian"

__all__ = [
    # Model
    "CNNLSTM",
    "CNNLSTMConfig",
    "create_cnnlstm",
    # Blocks
    "CNNFeatureExtractor",
    "LSTMHead",
    "init_weights",
    "get_block_info",
    # Utils
    "count_parameters",
    "get_model_size",
    "print_model_summary",
    "validate_forward_pass",
    "get_layer_output_shapes",
    "profile_model",
    "calculate_flops",
    "analyze_memory_usage",
    "test_inference_speed",
    "set_seed",
    "get_device",
]


def info(verbose: bool = False) -> None:
    """Print module information."""
    print(
        f"""
{'='*80}
CNNLSTM Module Information (ULTIMATE EDITION)
{'='*80}

📌 Module: CNNLSTM
📌 Version: {__version__}
👨‍💻 Author: {__author__}

📁 Components:
┌───────────────────────────────────────────────────────────────┐
│ Component       │ Description                                │
├───────────────────────────────────────────────────────────────┤
│ CNNLSTM         │ Complete CNN-LSTM architecture             │
│ create_cnnlstm  │ Factory function                           │
│ CNNFeatureExtractor │ CNN feature extraction                │
│ LSTMHead        │ LSTM sequence modeling                     │
└───────────────────────────────────────────────────────────────┘

📊 Architecture:
- CNN: Convolutional feature extractor
- LSTM: Sequence modeling on CNN features
- Classifier: Final classification

💡 Usage:
>>> from src.models.CNNLSTM import create_cnnlstm
>>> model = create_cnnlstm(num_classes=5)
>>> output = model(torch.randn(2, 3, 224, 224))
{'='*80}
"""
    )


def get_module_version() -> str:
    return __version__


def get_module_author() -> str:
    return __author__


if __name__ == "__main__":
    info(verbose=True)
