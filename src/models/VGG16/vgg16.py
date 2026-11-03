# src/models/VGG16/vgg16.py
# -*- coding: utf-8 -*-

"""
VGG16 Architecture Implementation from Scratch (ULTIMATE EDITION)
==================================================================
This module implements the complete VGG16 architecture without pretrained weights.

Architecture (from "Very Deep Convolutional Networks for Large-Scale Image Recognition"):
- Input: 224×224×3
- Stage 1: 2× ConvBlock(64) + MaxPool → 112×112
- Stage 2: 2× ConvBlock(128) + MaxPool → 56×56
- Stage 3: 3× ConvBlock(256) + MaxPool → 28×28
- Stage 4: 3× ConvBlock(512) + MaxPool → 14×14
- Stage 5: 3× ConvBlock(512) + MaxPool → 7×7
- Classifier: Linear(512*7*7 → 4096) → ReLU → Dropout → Linear(4096 → 4096) → ReLU → Dropout → Linear(4096 → num_classes)

Total Parameters: ~138M

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 1.0.0 (ULTIMATE EDITION)
"""

import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, init_weights, make_stage

warnings.filterwarnings("ignore")


# ================================================================
# CONFIG DATACLASS
# ================================================================
@dataclass
class VGG16Config:
    """Configuration class for VGG16."""

    num_classes: int = 5
    use_batch_norm: bool = True
    activation: str = "relu"
    dropout_rate: float = 0.5
    input_channels: int = 3
    init_type: str = "kaiming"
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "input_channels": self.input_channels,
            "init_type": self.init_type,
            "verbose": self.verbose,
        }


# ================================================================
# VGG16 MODEL
# ================================================================
class VGG16(nn.Module):
    """
    Complete VGG16 Architecture from Scratch.

    Features:
        - 13 convolutional layers (5 stages)
        - 3 fully connected layers
        - BatchNorm option
        - Dropout for regularization
        - Configurable number of classes
    """

    def __init__(
        self,
        num_classes: int = 5,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.5,
        input_channels: int = 3,
        init_type: str = "kaiming",
        verbose: bool = False,
    ) -> None:
        super(VGG16, self).__init__()

        # ==========================================
        # Store configuration
        # ==========================================
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels
        self.init_type = init_type
        self.verbose = verbose

        # ==========================================
        # Feature Extractor (Convolutional Layers)
        # ==========================================
        # Stage 1: 2x Conv(64) → MaxPool → 112×112
        self.stage1 = make_stage(
            in_channels=input_channels,
            out_channels=64,
            num_blocks=2,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_type=init_type,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: 2x Conv(128) → MaxPool → 56×56
        self.stage2 = make_stage(
            in_channels=64,
            out_channels=128,
            num_blocks=2,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_type=init_type,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: 3x Conv(256) → MaxPool → 28×28
        self.stage3 = make_stage(
            in_channels=128,
            out_channels=256,
            num_blocks=3,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_type=init_type,
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 4: 3x Conv(512) → MaxPool → 14×14
        self.stage4 = make_stage(
            in_channels=256,
            out_channels=512,
            num_blocks=3,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_type=init_type,
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 5: 3x Conv(512) → MaxPool → 7×7
        self.stage5 = make_stage(
            in_channels=512,
            out_channels=512,
            num_blocks=3,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_type=init_type,
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ==========================================
        # Adaptive Pooling (to 7×7)
        # ==========================================
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ==========================================
        # Classifier (Fully Connected Layers)
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096, momentum=0.01),  # ← momentum کوچیک
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes),
        )

        # ==========================================
        # Initialize Classifier Weights
        # ==========================================
        # ✅ Initialize classifier with SMALL weights (critical for VGG16)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  # ← خیلی کوچیک
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # ==========================================
        # Stage Info
        # ==========================================
        self._stage_info = self._get_stage_info()

        if verbose:
            print(self.summary())

    def _get_stage_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about each stage."""
        return {
            "stage1": {
                "name": "Stage 1 (2× ConvBlock(64))",
                "blocks": 2,
                "out_channels": 64,
                "output_shape": (64, 112, 112),
                "parameters": sum(p.numel() for p in self.stage1.parameters()),
            },
            "stage2": {
                "name": "Stage 2 (2× ConvBlock(128))",
                "blocks": 2,
                "out_channels": 128,
                "output_shape": (128, 56, 56),
                "parameters": sum(p.numel() for p in self.stage2.parameters()),
            },
            "stage3": {
                "name": "Stage 3 (3× ConvBlock(256))",
                "blocks": 3,
                "out_channels": 256,
                "output_shape": (256, 28, 28),
                "parameters": sum(p.numel() for p in self.stage3.parameters()),
            },
            "stage4": {
                "name": "Stage 4 (3× ConvBlock(512))",
                "blocks": 3,
                "out_channels": 512,
                "output_shape": (512, 14, 14),
                "parameters": sum(p.numel() for p in self.stage4.parameters()),
            },
            "stage5": {
                "name": "Stage 5 (3× ConvBlock(512))",
                "blocks": 3,
                "out_channels": 512,
                "output_shape": (512, 7, 7),
                "parameters": sum(p.numel() for p in self.stage5.parameters()),
            },
            "classifier": {
                "name": "Classifier",
                "out_channels": self.num_classes,
                "output_shape": (self.num_classes,),
                "parameters": sum(p.numel() for p in self.classifier.parameters()),
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Stage 1-5
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.stage4(x)
        x = self.pool4(x)
        x = self.stage5(x)
        x = self.pool5(x)

        # Adaptive pool to 7×7
        x = self.avgpool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classifier
        x = self.classifier(x)

        return x

    # ==========================================
    # ADVANCED FEATURES
    # ==========================================

    def get_trainable_parameters(self) -> Dict[str, Union[int, float]]:
        """Get count of trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_percentage": (trainable / total) * 100 if total > 0 else 0.0,
        }

    def get_layer_parameters(self) -> Dict[str, int]:
        """Get parameter count for each major layer."""
        return {
            "stage1": sum(p.numel() for p in self.stage1.parameters()),
            "stage2": sum(p.numel() for p in self.stage2.parameters()),
            "stage3": sum(p.numel() for p in self.stage3.parameters()),
            "stage4": sum(p.numel() for p in self.stage4.parameters()),
            "stage5": sum(p.numel() for p in self.stage5.parameters()),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
        }

    def get_feature_extractor(self) -> nn.Sequential:
        """Get feature extractor (convolutional layers)."""
        return nn.Sequential(
            self.stage1,
            self.pool1,
            self.stage2,
            self.pool2,
            self.stage3,
            self.pool3,
            self.stage4,
            self.pool4,
            self.stage5,
            self.pool5,
            self.avgpool,
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding (features before classifier)."""
        with torch.no_grad():
            features = self.get_feature_extractor()(x)
            return torch.flatten(features, 1)

    def freeze_backbone(self) -> None:
        """Freeze all layers except classifier."""
        for name, param in self.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        if self.verbose:
            print("✅ Backbone frozen. Only classifier is trainable.")

    def unfreeze_all(self) -> None:
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True
        if self.verbose:
            print("✅ All layers unfrozen.")

    def summary(self) -> str:
        """Get a string summary of the model."""
        info = self.get_trainable_parameters()
        layer_params = self.get_layer_parameters()

        return f"""
{'='*80}
VGG16 Architecture Summary
{'='*80}

Input Shape: (batch, {self.input_channels}, 224, 224)
Output Shape: (batch, {self.num_classes})

Layer Configuration:
┌────────────────────────────────────────────────────────────────────┐
│ Layer        │ Output Shape    │ Parameters    │ Blocks             │
├────────────────────────────────────────────────────────────────────┤
│ Stage1       │ 64×112×112      │ {layer_params.get('stage1', 0):>12,} │ 2× ConvBlock(64)   │
│ Stage2       │ 128×56×56       │ {layer_params.get('stage2', 0):>12,} │ 2× ConvBlock(128)  │
│ Stage3       │ 256×28×28       │ {layer_params.get('stage3', 0):>12,} │ 3× ConvBlock(256)  │
│ Stage4       │ 512×14×14       │ {layer_params.get('stage4', 0):>12,} │ 3× ConvBlock(512)  │
│ Stage5       │ 512×7×7         │ {layer_params.get('stage5', 0):>12,} │ 3× ConvBlock(512)  │
│ Classifier   │ {self.num_classes}               │ {layer_params.get('classifier', 0):>12,} │ 3× Linear          │
└────────────────────────────────────────────────────────────────────┘

Parameter Count:
- Total: {info['total']:,}
- Trainable: {info['trainable']:,}
- Frozen: {info['frozen']:,}
- Trainable %: {info['trainable_percentage']:.1f}%

Configuration:
- Activation: {self.activation}
- BatchNorm: {'Enabled' if self.use_batch_norm else 'Disabled'}
- Dropout: {self.dropout_rate}
- Input Channels: {self.input_channels}
- Init Type: {self.init_type}

Status: ✅ Starting from Scratch (No Pre-trained Weights)
{'='*80}
"""

    def __repr__(self) -> str:
        return f"VGG16(num_classes={self.num_classes}, dropout={self.dropout_rate})"


def create_vgg16(
    num_classes: int = 5, pretrained: bool = False, verbose: bool = True, **kwargs
) -> VGG16:
    """
    Factory function to create VGG16 model.

    Args:
        num_classes: Number of output classes
        pretrained: Ignored (no pretrained weights)
        verbose: Print model summary
        **kwargs: Additional arguments

    Returns:
        VGG16 model
    """
    if pretrained:
        print("⚠️ Warning: pretrained=True ignored for Task 5")
        print("📌 Starting from scratch as per Task 5 requirements")

    model = VGG16(num_classes=num_classes, verbose=verbose, **kwargs)

    if verbose:
        print(model.summary())

    return model


if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing VGG16 Implementation")
    print("=" * 80)

    model = create_vgg16(num_classes=5, verbose=False)
    print("✅ Model created")

    dummy = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    print(f"   Input: {dummy.shape}")
    print(f"   Output: {output.shape}")

    print(model.summary())
    print("\n✅ VGG16 is ready!")
    print("=" * 80)
