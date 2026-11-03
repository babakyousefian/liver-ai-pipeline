# src/models/AlexNet/__init__.py
# -*- coding: utf-8 -*-

"""
AlexNet Module - Complete Modernized Implementation from Scratch
================================================================
Author: Babak Yousefian

A faithful re-implementation of AlexNet (Krizhevsky et al., 2012)
with modern adaptations for small medical datasets:

- BatchNorm instead of LRN (faster, more stable)
- AdaptiveAvgPool for flexible input sizes
- Reduced width multiplier (0.5) to prevent overfitting
- Reduced classifier size (4096 -> 1024 -> 512)
- Modern dropout (0.4 instead of 0.5)
"""

import warnings
warnings.filterwarnings("ignore")

from .blocks import (
    AlexNetConvBlock,
    AlexNetClassifierBlock,
    init_weights_alexnet,
)
from .alexnet import (
    AlexNet,
    AlexNetConfig,
    create_alexnet,
)
from .utils import (
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
    "AlexNet",
    "AlexNetConfig",
    "create_alexnet",
    # Blocks
    "AlexNetConvBlock",
    "AlexNetClassifierBlock",
    "init_weights_alexnet",
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
