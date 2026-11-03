# src/models/MobileNetV2/mobilenetV2.py
# -*- coding: utf-8 -*-

"""
MobileNetV2 Architecture from Scratch
=====================================
Author: Babak Yousefian
"""

import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .blocks import ConvBNReLU, InvertedResidual, init_weights

warnings.filterwarnings("ignore")


# ================================================================
# MobileNetV2 Configuration
# ================================================================
# (t, c, n, s) - expand_ratio, out_channels, num_blocks, stride
MOBILENETV2_CONFIG = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


@dataclass
class MobileNetV2Config:
    """Configuration for MobileNetV2."""
    num_classes: int = 5
    input_channels: int = 3
    width_mult: float = 1.0
    dropout_rate: float = 0.2
    init_type: str = "kaiming"
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "width_mult": self.width_mult,
            "dropout_rate": self.dropout_rate,
            "init_type": self.init_type,
            "verbose": self.verbose,
        }


class MobileNetV2(nn.Module):
    """Complete MobileNetV2 Architecture from Scratch."""

    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 3,
        width_mult: float = 1.0,
        dropout_rate: float = 0.2,
        init_type: str = "kaiming",
        verbose: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.width_mult = width_mult
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        self.verbose = verbose
        self._config = MobileNetV2Config(
            num_classes=num_classes,
            input_channels=input_channels,
            width_mult=width_mult,
            dropout_rate=dropout_rate,
            init_type=init_type,
            verbose=verbose,
        )

        # ✅ Stem: Conv 3x3, stride 2
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        features = [ConvBNReLU(input_channels, input_channel, stride=2, init_type=init_type)]

        # ✅ Inverted Residual Blocks
        for t, c, n, s in MOBILENETV2_CONFIG:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    features.append(
                        InvertedResidual(
                            input_channel,
                            output_channel,
                            stride=s,
                            expand_ratio=t,
                            init_type=init_type,
                        )
                    )
                else:
                    features.append(
                        InvertedResidual(
                            input_channel,
                            output_channel,
                            stride=1,
                            expand_ratio=t,
                            init_type=init_type,
                        )
                    )
                input_channel = output_channel

        # ✅ Final 1x1 conv
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, init_type=init_type))

        self.features = nn.Sequential(*features)

        # ✅ Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.last_channel, num_classes),
        )

        # ✅ Initialize Classifier
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if verbose:
            print(self.summary())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
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
MobileNetV2 Architecture Summary
{'='*80}
Input Shape: (batch, {self.input_channels}, 224, 224)
Output Shape: (batch, {self.num_classes})
Parameters:
- Total: {info['total']:,}
- Trainable: {info['trainable']:,}
- Width Mult: {self.width_mult}
- Dropout: {self.dropout_rate}
{'='*80}
"""

    def __repr__(self) -> str:
        return f"MobileNetV2(num_classes={self.num_classes}, dropout={self.dropout_rate})"


def create_mobilenetv2(
    num_classes: int = 5, pretrained: bool = False, verbose: bool = True, **kwargs
) -> MobileNetV2:
    """Factory function."""
    if pretrained:
        print("⚠️ Warning: pretrained=True ignored for Task 5")
    model = MobileNetV2(num_classes=num_classes, verbose=verbose, **kwargs)
    if verbose:
        print(model.summary())
    return model
