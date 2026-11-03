# src/models/AlexNet/alexnet.py
# -*- coding: utf-8 -*-

"""
AlexNet Architecture from Scratch (Modernized)
==============================================

A faithful implementation of AlexNet (Krizhevsky et al., 2012)
with modern adaptations for small medical datasets.

Original AlexNet:
    - 60M parameters
    - 227x227 input
    - LRN normalization
    - 4096-4096-1000 classifier
    - 0.5 dropout

Modernized AlexNet (this file):
    - ~9-10M parameters (width_mult=0.5)
    - 224x224 input (adaptive)
    - BatchNorm instead of LRN
    - 1024-512-num_classes classifier
    - 0.4 dropout

Author: Babak Yousefian
"""

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .blocks import (
    AlexNetClassifierBlock,
    AlexNetConvBlock,
    init_weights_alexnet,
)

warnings.filterwarnings("ignore")


# ================================================================
# AlexNet Configuration
# ================================================================

@dataclass
class AlexNetConfig:
    """
    Configuration for AlexNet.

    Defaults are tuned for small medical datasets (e.g., 2000-5000 images).
    """
    num_classes: int = 4
    input_channels: int = 3
    input_size: int = 224

    # Width multiplier: scales channel counts in conv layers
    # 1.0 = original AlexNet (60M params)
    # 0.5 = half width (~15M params)
    # 0.375 = ~9M params (recommended for 2642 images)
    width_mult: float = 0.5

    # Classifier hidden sizes
    fc6_features: int = 1024
    fc7_features: int = 512

    # Regularization
    dropout_conv: float = 0.0      # Dropout after convs (usually 0)
    dropout_fc: float = 0.4        # Dropout in classifier
    dropout_conv_last: float = 0.0  # Extra dropout after conv5

    # Normalization
    use_batch_norm: bool = True    # BatchNorm instead of LRN

    # Init
    init_type: str = "kaiming"

    # Misc
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "input_size": self.input_size,
            "width_mult": self.width_mult,
            "fc6_features": self.fc6_features,
            "fc7_features": self.fc7_features,
            "dropout_conv": self.dropout_conv,
            "dropout_fc": self.dropout_fc,
            "dropout_conv_last": self.dropout_conv_last,
            "use_batch_norm": self.use_batch_norm,
            "init_type": self.init_type,
        }


# ================================================================
# Helper: Compute channel counts
# ================================================================

def _channels(base: int, width_mult: float) -> int:
    """Compute channel count with width multiplier (min 8)."""
    return max(8, int(round(base * width_mult)))


# ================================================================
# AlexNet Model
# ================================================================

class AlexNet(nn.Module):
    """
    Modernized AlexNet from Scratch.

    Architecture (with width_mult=0.5, num_classes=4):

    FEATURES (Conv):
        Conv1: 3 -> 48, 11x11, stride=4, pad=2, BN, ReLU, MaxPool(3,2)
        Conv2: 48 -> 128, 5x5, pad=2, BN, ReLU, MaxPool(3,2)
        Conv3: 128 -> 192, 3x3, pad=1, BN, ReLU
        Conv4: 192 -> 192, 3x3, pad=1, BN, ReLU
        Conv5: 192 -> 128, 3x3, pad=1, BN, ReLU, MaxPool(3,2)
        AdaptiveAvgPool2d((6, 6))

    CLASSIFIER (FC):
        Flatten: 128 * 6 * 6 = 4608
        FC6: 4608 -> 1024, ReLU, Dropout(0.4)
        FC7: 1024 -> 512, ReLU, Dropout(0.4)
        FC8: 512 -> num_classes

    Total parameters: ~9-10M (depends on width_mult)

    Args:
        num_classes: Number of output classes
        input_channels: Input channels (3 for RGB)
        input_size: Expected input size (for summary only)
        width_mult: Channel width multiplier
        fc6_features: FC6 hidden size
        fc7_features: FC7 hidden size
        dropout_conv: Dropout after convs (rarely used)
        dropout_fc: Dropout in classifier
        dropout_conv_last: Extra dropout after conv5
        use_batch_norm: Use BatchNorm instead of LRN
        init_type: Weight init type
        verbose: Print summary on init
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 3,
        input_size: int = 224,
        width_mult: float = 0.5,
        fc6_features: int = 1024,
        fc7_features: int = 512,
        dropout_conv: float = 0.0,
        dropout_fc: float = 0.4,
        dropout_conv_last: float = 0.0,
        use_batch_norm: bool = True,
        init_type: str = "kaiming",
        verbose: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.width_mult = width_mult
        self.fc6_features = fc6_features
        self.fc7_features = fc7_features
        self.dropout_conv = dropout_conv
        self.dropout_fc = dropout_fc
        self.dropout_conv_last = dropout_conv_last
        self.use_batch_norm = use_batch_norm
        self.init_type = init_type
        self.verbose = verbose

        # Store config
        self._config = AlexNetConfig(
            num_classes=num_classes,
            input_channels=input_channels,
            input_size=input_size,
            width_mult=width_mult,
            fc6_features=fc6_features,
            fc7_features=fc7_features,
            dropout_conv=dropout_conv,
            dropout_fc=dropout_fc,
            dropout_conv_last=dropout_conv_last,
            use_batch_norm=use_batch_norm,
            init_type=init_type,
            verbose=verbose,
        )

        # ========================================================
        # Compute channel counts
        # ========================================================
        # Original AlexNet: [96, 256, 384, 384, 256]
        c1 = _channels(96, width_mult)
        c2 = _channels(256, width_mult)
        c3 = _channels(384, width_mult)
        c4 = _channels(384, width_mult)
        c5 = _channels(256, width_mult)

        self.channels = [c1, c2, c3, c4, c5]

        # ========================================================
        # FEATURES (Convolutional)
        # ========================================================
        self.features = nn.Sequential(
            # Conv1: 3 -> 48 (or 96), 11x11, stride=4, pad=2, BN, ReLU, Pool
            AlexNetConvBlock(
                in_channels=input_channels,
                out_channels=c1,
                kernel_size=11,
                stride=4,
                padding=2,
                use_bn=use_batch_norm,
                use_pool=True,
                pool_kernel=3,
                pool_stride=2,
                init_type=init_type,
            ),
            # Conv2: 48 -> 128, 5x5, pad=2, BN, ReLU, Pool
            AlexNetConvBlock(
                in_channels=c1,
                out_channels=c2,
                kernel_size=5,
                stride=1,
                padding=2,
                use_bn=use_batch_norm,
                use_pool=True,
                pool_kernel=3,
                pool_stride=2,
                init_type=init_type,
            ),
            # Conv3: 128 -> 192, 3x3, pad=1, BN, ReLU
            AlexNetConvBlock(
                in_channels=c2,
                out_channels=c3,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bn=use_batch_norm,
                use_pool=False,
                init_type=init_type,
            ),
            # Conv4: 192 -> 192, 3x3, pad=1, BN, ReLU
            AlexNetConvBlock(
                in_channels=c3,
                out_channels=c4,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bn=use_batch_norm,
                use_pool=False,
                init_type=init_type,
            ),
            # Conv5: 192 -> 128, 3x3, pad=1, BN, ReLU, Pool
            AlexNetConvBlock(
                in_channels=c4,
                out_channels=c5,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bn=use_batch_norm,
                use_pool=True,
                pool_kernel=3,
                pool_stride=2,
                init_type=init_type,
            ),
        )

        # Optional dropout after last conv
        self.conv_dropout = (
            nn.Dropout(p=dropout_conv_last)
            if dropout_conv_last > 0
            else nn.Identity()
        )

        # ========================================================
        # ADAPTIVE POOL (for flexible input size)
        # ========================================================
        # We use AdaptiveAvgPool2d((6, 6)) so the model accepts
        # any input size (>=~64x64) and always produces (B, C5, 6, 6)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        # ========================================================
        # CLASSIFIER (Fully Connected)
        # ========================================================
        flattened_size = c5 * 6 * 6

        self.classifier = nn.Sequential(
            # FC6: 4608 -> 1024, ReLU, Dropout
            AlexNetClassifierBlock(
                in_features=flattened_size,
                out_features=fc6_features,
                dropout=dropout_fc,
                use_relu=True,
                init_type=init_type,
            ),
            # FC7: 1024 -> 512, ReLU, Dropout
            AlexNetClassifierBlock(
                in_features=fc6_features,
                out_features=fc7_features,
                dropout=dropout_fc,
                use_relu=True,
                init_type=init_type,
            ),
            # FC8 (output): 512 -> num_classes, no ReLU, no Dropout
            nn.Linear(fc7_features, num_classes),
        )

        # Special init for last layer: small normal (like original AlexNet)
        nn.init.normal_(self.classifier[-1].weight, mean=0.0, std=0.01)
        if self.classifier[-1].bias is not None:
            nn.init.constant_(self.classifier[-1].bias, 0)

        if verbose:
            print(self.summary())

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Logits (B, num_classes)
        """
        # Features
        x = self.features(x)
        x = self.conv_dropout(x)

        # Adaptive pool to fixed size
        x = self.adaptive_pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classifier
        x = self.classifier(x)

        return x

    # ============================================================
    # Feature extraction (useful for interpretability)
    # ============================================================

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return conv features before adaptive pool (B, C5, H, W)."""
        return self.features(x)

    def forward_features_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened features after adaptive pool (B, 4608)."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        return torch.flatten(x, 1)

    # ============================================================
    # Introspection
    # ============================================================

    def get_trainable_parameters(self) -> Dict[str, Any]:
        """Return total / trainable / frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_percentage": (trainable / total * 100.0) if total > 0 else 0.0,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration as dict."""
        return self._config.to_dict()

    def summary(self) -> str:
        """Return a human-readable summary string."""
        info = self.get_trainable_parameters()
        lines = [
            "=" * 80,
            "AlexNet Architecture Summary (Modernized)",
            "=" * 80,
            f"Input Shape:        (B, {self.input_channels}, {self.input_size}, {self.input_size})",
            f"Output Shape:       (B, {self.num_classes})",
            f"Width Multiplier:   {self.width_mult}",
            f"Conv Channels:      {self.channels}",
            f"FC6 / FC7:          {self.fc6_features} / {self.fc7_features}",
            f"Dropout (FC):       {self.dropout_fc}",
            f"Dropout (Conv Last):{self.dropout_conv_last}",
            f"BatchNorm:          {self.use_batch_norm}",
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
            f"AlexNet(num_classes={self.num_classes}, "
            f"width_mult={self.width_mult}, "
            f"dropout_fc={self.dropout_fc}, "
            f"channels={self.channels})"
        )


# ================================================================
# Factory Function
# ================================================================

def create_alexnet(
    num_classes: int = 4,
    pretrained: bool = False,
    verbose: bool = True,
    **kwargs,
) -> AlexNet:
    """
    Factory function to create an AlexNet model.

    Args:
        num_classes: Number of output classes
        pretrained: If True, tries to load ImageNet weights (NOT SUPPORTED
                    in this from-scratch implementation - will warn)
        verbose: Print model summary
        **kwargs: Passed to AlexNet constructor

    Returns:
        AlexNet model
    """
    if pretrained:
        print(
            "⚠️  Warning: pretrained=True is IGNORED. "
            "This is a from-scratch implementation (Task 5)."
        )

    model = AlexNet(
        num_classes=num_classes,
        verbose=verbose,
        **kwargs,
    )

    if verbose:
        print(model.summary())

    return model
