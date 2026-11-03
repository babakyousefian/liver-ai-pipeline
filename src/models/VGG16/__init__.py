# src/models/VGG16/__init__.py
# -*- coding: utf-8 -*-

"""
VGG16 Module - Complete Implementation from Scratch

This module provides a complete implementation of VGG16 architecture
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
    ConvBlock,
    count_blocks_in_stage,
    get_block_info,
    get_stage_info,
    init_weights,
    make_stage,
)
from .vgg16 import VGG16, VGG16Config, create_vgg16

# ================================================================
# Exports
# ================================================================
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
    "VGG16",
    "VGG16Config",
    "create_vgg16",
    # Blocks
    "ConvBlock",
    "make_stage",
    "init_weights",
    "get_block_info",
    "get_stage_info",
    "count_blocks_in_stage",
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


# ================================================================
# Info Function
# ================================================================
def info(verbose: bool = False) -> None:
    """Print module information."""
    print(
        f"""
{'='*80}
VGG16 Module Information (ULTIMATE EDITION)
{'='*80}

📌 Module: VGG16
📌 Version: {__version__}
👨‍💻 Author: {__author__}

📁 Components:
┌───────────────────────────────────────────────────────────────┐
│ Component       │ Description                                │
├───────────────────────────────────────────────────────────────┤
│ VGG16           │ Complete VGG16 architecture (138M params)  │
│ create_vgg16    │ Factory function                           │
│ ConvBlock       │ Conv2d + BN + ReLU                         │
│ make_stage      │ Create sequence of ConvBlocks              │
└───────────────────────────────────────────────────────────────┘

📊 Architecture:
┌───────────────────────────────────────────────────────────────┐
│ Stage  │ Layers                │ Output Shape              │
├───────────────────────────────────────────────────────────────┤
│ 1      │ 2× Conv(64) + MaxPool │ 64×112×112                │
│ 2      │ 2× Conv(128) + MaxPool│ 128×56×56                 │
│ 3      │ 3× Conv(256) + MaxPool│ 256×28×28                 │
│ 4      │ 3× Conv(512) + MaxPool│ 512×14×14                 │
│ 5      │ 3× Conv(512) + MaxPool│ 512×7×7                   │
│ FC     │ 3× Linear              │ num_classes               │
└───────────────────────────────────────────────────────────────┘

💡 Usage:
>>> from src.models.VGG16 import create_vgg16
>>> model = create_vgg16(num_classes=5)
>>> output = model(torch.randn(2, 3, 224, 224))
{'='*80}
"""
    )


def get_module_version() -> str:
    return __version__


def get_module_author() -> str:
    return __author__


# ================================================================
# Sanity Check
# ================================================================
def _sanity_check() -> bool:
    """Quick sanity check."""
    try:
        model = create_vgg16(num_classes=5, verbose=False)
        dummy = torch.randn(1, 3, 224, 224)
        output = model(dummy)
        return output.shape == (1, 5)
    except:
        return False


if __name__ == "__main__":
    info(verbose=True)
    print(f"\n✅ Sanity check: {_sanity_check()}")
