# src/models/ConvNext/__init__.py
# -*- coding: utf-8 -*-

"""
ConvNext Module - Complete Implementation from Scratch
======================================================
Author: Babak Yousefian
"""

import warnings
warnings.filterwarnings("ignore")

from .blocks import ConvNextBlock, LayerNorm2d, init_weights
from .ConvNext import ConvNext, ConvNextConfig, create_convnext
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
    "ConvNext", "ConvNextConfig", "create_convnext",
    "ConvNextBlock", "LayerNorm2d", "init_weights",
    "count_parameters", "get_model_size", "print_model_summary",
    "validate_forward_pass", "get_layer_output_shapes",
    "profile_model", "calculate_flops", "analyze_memory_usage",
    "test_inference_speed", "set_seed", "get_device",
]