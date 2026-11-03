# src/models/AlexNet/blocks.py
# -*- coding: utf-8 -*-

"""
AlexNet Building Blocks
=======================
Modern implementations of AlexNet components:

- AlexNetConvBlock: Conv2d + BatchNorm + ReLU (replaces LRN)
- AlexNetClassifierBlock: Linear + ReLU + Dropout
- init_weights_alexnet: Proper weight initialization

Author: Babak Yousefian
"""

import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ================================================================
# Weight Initialization
# ================================================================

def init_weights_alexnet(
    module: nn.Module,
    init_type: str = "kaiming",
    lstm_init: str = "orthogonal",
) -> None:
    """
    Initialize weights for AlexNet-style modules.

    Modern initialization strategy:
    - Conv2d: Kaiming normal (He initialization) for ReLU
    - BatchNorm2d: weight=1, bias=0
    - Linear: Kaiming normal (or normal with std=0.01)
    - LSTM (if present): Orthogonal

    Args:
        module: PyTorch module
        init_type: 'kaiming' | 'normal'
        lstm_init: 'orthogonal' | 'uniform'
    """
    if isinstance(module, nn.Conv2d):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


# ================================================================
# Conv Block (Conv2d + BN + ReLU)
# ================================================================

class AlexNetConvBlock(nn.Module):
    """
    Modern AlexNet Conv Block.

    Original AlexNet used Local Response Normalization (LRN).
    We replace it with BatchNorm2d, which is:
    - Faster on GPU
    - More stable training
    - Better gradient flow
    - Standard in modern architectures

    Structure:
        Conv2d -> [BatchNorm2d] -> ReLU -> [MaxPool2d]

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        use_bn: Whether to use BatchNorm (default: True)
        use_pool: Whether to use MaxPool after (default: False)
        pool_kernel: MaxPool kernel size (default: 3)
        pool_stride: MaxPool stride (default: 2)
        init_type: Weight init type
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        use_bn: bool = True,
        use_pool: bool = False,
        pool_kernel: int = 3,
        pool_stride: int = 2,
        init_type: str = "kaiming",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn,  # bias only if no BN
        )
        init_weights_alexnet(self.conv, init_type=init_type)

        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=True)

        self.pool = None
        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

        # Store config for introspection
        self._config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "use_bn": use_bn,
            "use_pool": use_pool,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        return dict(self._config)


# ================================================================
# Classifier Block (Linear + ReLU + Dropout)
# ================================================================

class AlexNetClassifierBlock(nn.Module):
    """
    Modern AlexNet Classifier Block.

    Structure:
        Linear -> ReLU -> [Dropout]

    Args:
        in_features: Input features
        out_features: Output features
        dropout: Dropout rate (0 = no dropout)
        use_relu: Whether to use ReLU (last layer usually False)
        init_type: Weight init type
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        use_relu: bool = True,
        init_type: str = "kaiming",
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        init_weights_alexnet(self.linear, init_type=init_type)

        self.relu = nn.ReLU(inplace=True) if use_relu else None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    