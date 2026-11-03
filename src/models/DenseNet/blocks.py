# src/models/DenseNet/blocks.py
# -*- coding: utf-8 -*-

"""
DenseNet Building Blocks
========================
Modern implementations of DenseNet components:

- Bottleneck: BN-ReLU-Conv1x1-BN-ReLU-Conv3x3 (with dropout)
- DenseLayer: Single layer in a dense block
- DenseBlock: Stack of DenseLayers with dense connectivity
- TransitionLayer: BN-ReLU-Conv1x1 + AvgPool (compression)
- init_weights_densenet: Proper initialization

Author: Babak Yousefian
"""

import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# ================================================================
# Weight Initialization
# ================================================================

def init_weights_densenet(module: nn.Module, init_type: str = "kaiming") -> None:
    """
    Initialize weights for DenseNet modules.

    DenseNet uses Kaiming normal (He) initialization for ReLU.
    """
    if isinstance(module, nn.Conv2d):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
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
# Bottleneck Layer (DenseNet-B)
# ================================================================

class Bottleneck(nn.Module):
    """
    Bottleneck layer for DenseNet-BC.

    Structure:
        BN -> ReLU -> Conv1x1 (4*growth) -> BN -> ReLU -> Conv3x3 (growth)
        [optional Dropout]

    The 1x1 conv reduces channels before the expensive 3x3 conv,
    significantly reducing parameters and computation.

    Args:
        in_channels: Input channels
        growth_rate: Growth rate (k)
        bn_size: Bottleneck multiplier (default: 4)
        dropout: Dropout rate (default: 0.0)
        init_type: Weight init type
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout: float = 0.0,
        init_type: str = "kaiming",
    ):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, bn_size * growth_rate, kernel_size=1, bias=False
        )
        init_weights_densenet(self.conv1, init_type)

        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )
        init_weights_densenet(self.conv2, init_type)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ================================================================
# Dense Layer
# ================================================================

class DenseLayer(nn.Module):
    """
    Single dense layer: Bottleneck + concatenation with input.

    This is the core building block of a DenseBlock. Each layer
    receives the concatenation of ALL previous feature maps in the block.

    Args:
        in_channels: Input channels (current feature map channels)
        growth_rate: Growth rate (k)
        bn_size: Bottleneck multiplier
        dropout: Dropout rate
        init_type: Weight init type
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout: float = 0.0,
        init_type: str = "kaiming",
    ):
        super().__init__()

        self.bottleneck = Bottleneck(
            in_channels=in_channels,
            growth_rate=growth_rate,
            bn_size=bn_size,
            dropout=dropout,
            init_type=init_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bottleneck(x)
        return torch.cat([x, out], dim=1)


# ================================================================
# Dense Block
# ================================================================

class DenseBlock(nn.Module):
    """
    Dense Block: stack of DenseLayers with dense connectivity.

    Each layer takes all previous outputs as input (concatenated).
    This is the key innovation of DenseNet.

    Args:
        num_layers: Number of DenseLayers
        in_channels: Input channels
        growth_rate: Growth rate (k)
        bn_size: Bottleneck multiplier
        dropout: Dropout rate
        init_type: Weight init type
    """

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout: float = 0.0,
        init_type: str = "kaiming",
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(
                DenseLayer(
                    in_channels=current_channels,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    dropout=dropout,
                    init_type=init_type,
                )
            )
            current_channels += growth_rate

        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ================================================================
# Transition Layer
# ================================================================

class TransitionLayer(nn.Module):
    """
    Transition layer between dense blocks.

    Structure:
        BN -> ReLU -> Conv1x1 (compress) -> AvgPool2d(2)

    Compression factor reduces channels, AvgPool reduces spatial size.

    Args:
        in_channels: Input channels
        compression: Compression factor (default: 0.5)
        init_type: Weight init type
    """

    def __init__(
        self,
        in_channels: int,
        compression: float = 0.5,
        init_type: str = "kaiming",
    ):
        super().__init__()

        out_channels = int(in_channels * compression)

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        init_weights_densenet(self.conv, init_type)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.bn(x), inplace=True))
        x = self.pool(x)
        return x
