# src/models/DenseNet/__init__.py
# -*- coding: utf-8 -*-

"""
DenseNet Module - Complete Implementation from Scratch
=====================================================
Author: Babak Yousefian

Faithful re-implementation of DenseNet-BC (Huang et al., 2017)
with modern adaptations for small medical datasets:

- Reduced layers and growth rate (~1.2M params)
- Bottleneck layers (1x1 conv before 3x3)
- Compression in transition layers
- Dropout for regularization (critical for small data)
- Adaptive to any input size
"""

import warnings
warnings.filterwarnings("ignore")

from .blocks import (
    DenseLayer,
    DenseBlock,
    TransitionLayer,
    Bottleneck,
    init_weights_densenet,
)
from .DenseNet import (
    DenseNet,
    DenseNetConfig,
    create_densenet,
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
    "DenseNet",
    "DenseNetConfig",
    "create_densenet",
    # Blocks
    "DenseLayer",
    "DenseBlock",
    "TransitionLayer",
    "Bottleneck",
    "init_weights_densenet",
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
