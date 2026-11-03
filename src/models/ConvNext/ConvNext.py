# src/models/ConvNext/ConvNext.py
# -*- coding: utf-8 -*-

"""
ConvNeXt Architecture from Scratch
==================================
Author: Babak Yousefian
"""

import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNextBlock, LayerNorm2d, init_weights

warnings.filterwarnings("ignore")


@dataclass
class ConvNextConfig:
    """Configuration for ConvNext."""
    num_classes: int = 5
    input_channels: int = 3
    dims: List[int] = None
    depths: List[int] = None
    dropout_rate: float = 0.2
    init_type: str = "kaiming"
    verbose: bool = False

    def __post_init__(self):
        if self.dims is None:
            self.dims = [96, 192, 384, 768]
        if self.depths is None:
            self.depths = [3, 3, 9, 3]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "dims": self.dims,
            "depths": self.depths,
            "dropout_rate": self.dropout_rate,
            "init_type": self.init_type,
            "verbose": self.verbose,
        }


class ConvNext(nn.Module):
    """Complete ConvNeXt Architecture from Scratch."""

    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 3,
        dims: List[int] = None,
        depths: List[int] = None,
        dropout_rate: float = 0.2,
        init_type: str = "kaiming",
        verbose: bool = False,
    ):
        super().__init__()

        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [3, 3, 9, 3]

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dims = dims
        self.depths = depths
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        self.verbose = verbose
        self._config = ConvNextConfig(
            num_classes=num_classes,
            input_channels=input_channels,
            dims=dims,
            depths=depths,
            dropout_rate=dropout_rate,
            init_type=init_type,
            verbose=verbose,
        )

        # ==========================================
        # Stem: 4x4 conv, stride 4
        # ==========================================
        self.downsample_layers = nn.ModuleList()

        # Stem
        stem = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        init_weights(stem[0], init_type)
        self.downsample_layers.append(stem)

        # Downsample layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            init_weights(downsample_layer[1], init_type)
            self.downsample_layers.append(downsample_layer)

        # ==========================================
        # Stages: 4 stages of ConvNextBlock
        # ==========================================
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNextBlock(dim=dims[i], init_type=init_type) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        # ==========================================
        # Head: LayerNorm + AvgPool + Linear
        # ==========================================
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # Init head
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, 0)

        if verbose:
            print(self.summary())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # Global average pooling
        x = x.mean([-2, -1])  # (B, C)
        x = self.norm(x)
        x = self.head(x)
        return x

    def get_trainable_parameters(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "trainable_percentage": (trainable / total) * 100 if total > 0 else 0.0,
        }

    def summary(self) -> str:
        info = self.get_trainable_parameters()
        return f"""
{'='*80}
ConvNeXt Architecture Summary
{'='*80}
Input Shape: (batch, {self.input_channels}, 224, 224)
Output Shape: (batch, {self.num_classes})
Parameters:
- Total: {info['total']:,}
- Trainable: {info['trainable']:,}
- Dims: {self.dims}
- Depths: {self.depths}
- Dropout: {self.dropout_rate}
{'='*80}
"""

    def __repr__(self) -> str:
        return f"ConvNext(num_classes={self.num_classes}, dims={self.dims})"


def create_convnext(
    num_classes: int = 5, pretrained: bool = False, verbose: bool = True, **kwargs
) -> ConvNext:
    """Factory function."""
    if pretrained:
        print("⚠️ Warning: pretrained=True ignored for Task 5")
    model = ConvNext(num_classes=num_classes, verbose=verbose, **kwargs)
    if verbose:
        print(model.summary())
    return model
