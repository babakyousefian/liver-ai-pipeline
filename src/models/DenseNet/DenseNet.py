# src/models/DenseNet/DenseNet.py
# -*- coding: utf-8 -*-

"""
DenseNet Architecture from Scratch (Modernized)
================================================

Faithful implementation of DenseNet-BC (Huang et al., 2017)
with modern adaptations for small medical datasets.

Original DenseNet-121:
    - 8M parameters
    - Layers: [6, 12, 24, 16]
    - Growth rate: 32
    - No dropout

Modernized DenseNet (this file):
    - ~1.2M parameters (for 2642 images)
    - Layers: [4, 8, 12, 8]
    - Growth rate: 16
    - Dropout: 0.2 (critical for small data)
    - Bottleneck + Compression

Author: Babak Yousefian
"""

import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    DenseBlock,
    TransitionLayer,
    init_weights_densenet,
)

warnings.filterwarnings("ignore")


# ================================================================
# DenseNet Configuration
# ================================================================

@dataclass
class DenseNetConfig:
    """Configuration for DenseNet."""
    num_classes: int = 4
    input_channels: int = 3
    input_size: int = 224

    # Architecture
    layers: List[int] = None  # [4, 8, 12, 8] for small data
    growth_rate: int = 16
    bn_size: int = 4
    compression: float = 0.5
    init_channels: int = 32

    # Regularization
    dropout: float = 0.2  # critical for small data
    init_type: str = "kaiming"

    # Misc
    verbose: bool = False

    def __post_init__(self):
        if self.layers is None:
            self.layers = [4, 8, 12, 8]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "input_size": self.input_size,
            "layers": self.layers,
            "growth_rate": self.growth_rate,
            "bn_size": self.bn_size,
            "compression": self.compression,
            "init_channels": self.init_channels,
            "dropout": self.dropout,
            "init_type": self.init_type,
        }


# ================================================================
# DenseNet Model
# ================================================================

class DenseNet(nn.Module):
    """
    Modernized DenseNet-BC from Scratch.

    Architecture:
        1. Stem: Conv7x7 stride 2 + MaxPool3x3 stride 2
        2. DenseBlock1 -> Transition1
        3. DenseBlock2 -> Transition2
        4. DenseBlock3 -> Transition3
        5. DenseBlock4 -> BN -> AdaptiveAvgPool -> FC

    Args:
        num_classes: Output classes
        input_channels: Input channels
        input_size: Input image size
        layers: Number of layers in each dense block
        growth_rate: Growth rate (k)
        bn_size: Bottleneck multiplier
        compression: Transition compression factor
        init_channels: Stem output channels
        dropout: Dropout rate in dense layers
        init_type: Weight init type
        verbose: Print summary on init
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 3,
        input_size: int = 224,
        layers: Optional[List[int]] = None,
        growth_rate: int = 16,
        bn_size: int = 4,
        compression: float = 0.5,
        init_channels: int = 32,
        dropout: float = 0.2,
        init_type: str = "kaiming",
        verbose: bool = False,
    ):
        super().__init__()

        if layers is None:
            layers = [4, 8, 12, 8]

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.layers = layers
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.compression = compression
        self.init_channels = init_channels
        self.dropout = dropout
        self.init_type = init_type
        self.verbose = verbose

        self._config = DenseNetConfig(
            num_classes=num_classes,
            input_channels=input_channels,
            input_size=input_size,
            layers=layers,
            growth_rate=growth_rate,
            bn_size=bn_size,
            compression=compression,
            init_channels=init_channels,
            dropout=dropout,
            init_type=init_type,
        )

        # ========================================================
        # Stem: Conv7x7 stride 2 + MaxPool3x3 stride 2
        # ========================================================
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                init_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # After stem: (B, init_channels, H/4, W/4)
        # For 224: 224 -> 112 -> 56

        # ========================================================
        # Dense Blocks + Transitions
        # ========================================================
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        current_channels = init_channels
        num_blocks = len(layers)

        for i, num_layers in enumerate(layers):
            # Dense block
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=current_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout=dropout,
                init_type=init_type,
            )
            self.dense_blocks.append(block)
            current_channels = block.out_channels

            # Transition (except after last block)
            if i < num_blocks - 1:
                trans = TransitionLayer(
                    in_channels=current_channels,
                    compression=compression,
                    init_type=init_type,
                )
                self.transitions.append(trans)
                current_channels = trans.out_channels

        # ========================================================
        # Final BN + Classifier
        # ========================================================
        self.final_bn = nn.BatchNorm2d(current_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, num_classes)

        # ========================================================
        # Initialize weights
        # ========================================================
        self._init_weights()

        if verbose:
            print(self.summary())

    def _init_weights(self):
        """Initialize weights for stem and classifier."""
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                init_weights_densenet(m, self.init_type)
        nn.init.kaiming_normal_(
            self.classifier.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, H, W)

        Returns:
            logits: (B, num_classes)
        """
        x = self.stem(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = F.relu(self.final_bn(x), inplace=True)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # ============================================================
    # Feature Extraction (for interpretability)
    # ============================================================

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return features before classifier."""
        x = self.stem(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = F.relu(self.final_bn(x), inplace=True)
        x = self.adaptive_pool(x)
        return torch.flatten(x, 1)

    # ============================================================
    # Introspection
    # ============================================================

    def get_trainable_parameters(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_percentage": (trainable / total * 100.0) if total > 0 else 0.0,
        }

    def get_config(self) -> Dict[str, Any]:
        return self._config.to_dict()

    def summary(self) -> str:
        info = self.get_trainable_parameters()
        lines = [
            "=" * 80,
            "DenseNet Architecture Summary (Modernized)",
            "=" * 80,
            f"Input Shape:        (B, {self.input_channels}, {self.input_size}, {self.input_size})",
            f"Output Shape:       (B, {self.num_classes})",
            f"Layers:             {self.layers}",
            f"Growth Rate:        {self.growth_rate}",
            f"BN Size:            {self.bn_size}",
            f"Compression:        {self.compression}",
            f"Init Channels:      {self.init_channels}",
            f"Dropout:            {self.dropout}",
            "-" * 80,
            f"Total Parameters:     {info['total']:,}",
            f"Trainable Parameters: {info['trainable']:,}",
            f"Frozen Parameters:    {info['frozen']:,}",
            f"Trainable %:          {info['trainable_percentage']:.2f}%",
            "=" * 80,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DenseNet(num_classes={self.num_classes}, "
            f"layers={self.layers}, growth={self.growth_rate}, "
            f"dropout={self.dropout})"
        )


# ================================================================
# Factory Function
# ================================================================

def create_densenet(
    num_classes: int = 4,
    variant: str = "small",
    pretrained: bool = False,
    verbose: bool = True,
    **kwargs,
) -> DenseNet:
    """
    Factory function to create a DenseNet model.

    Args:
        num_classes: Number of output classes
        variant: 'small' | 'medium' | 'custom'
            - small: layers=[4,8,12,8], growth=16 (~1.2M params) ← recommended
            - medium: layers=[6,12,24,16], growth=24 (~4M params)
            - custom: uses kwargs
        pretrained: If True, warns (from-scratch implementation)
        verbose: Print model summary
        **kwargs: Passed to DenseNet constructor

    Returns:
        DenseNet model
    """
    if pretrained:
        print(
            "⚠️  Warning: pretrained=True is IGNORED. "
            "This is a from-scratch implementation (Task 5)."
        )

    variants = {
        "small": {"layers": [4, 8, 12, 8], "growth_rate": 16},
        "medium": {"layers": [6, 12, 24, 16], "growth_rate": 24},
        "custom": {},
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")

    config = {**variants[variant], **kwargs}

    model = DenseNet(
        num_classes=num_classes,
        verbose=verbose,
        **config,
    )

    if verbose:
        print(model.summary())

    return model
