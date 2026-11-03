# src/models/MobileNetV2/blocks.py
# -*- coding: utf-8 -*-

"""
MobileNetV2 Building Blocks
============================
- ConvBNReLU: Conv2d + BatchNorm + ReLU6
- InvertedResidual: Depthwise separable conv with expansion
"""

import warnings
from typing import Any, Dict

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


def init_weights(module: nn.Module, init_type: str = "kaiming") -> None:
    """Initialize weights."""
    if isinstance(module, nn.Conv2d):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class ConvBNReLU(nn.Sequential):
    """
    Conv2d + BatchNorm + ReLU6.

    Standard building block for MobileNetV2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        init_type: str = "kaiming",
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        init_weights(self[0], init_type)


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block (MobileNetV2).

    Architecture:
        1x1 Conv (expand) → 3x3 DepthwiseConv → 1x1 Conv (project) → Residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        init_type: str = "kaiming",
    ):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = (self.stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            # 1x1 expansion
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, init_type=init_type))

        # 3x3 depthwise
        layers.append(
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
                init_type=init_type,
            )
        )

        # 1x1 projection (no activation)
        layers.append(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)
        init_weights(self.conv[0], init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "InvertedResidual",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "stride": self.stride,
            "parameters": sum(p.numel() for p in self.parameters()),
        }
