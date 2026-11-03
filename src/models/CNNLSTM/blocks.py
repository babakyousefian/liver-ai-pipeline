# src/models/CNNLSTM/blocks.py
# -*- coding: utf-8 -*-

"""
CNNLSTM Core Building Blocks
=============================
This module implements the fundamental building blocks of CNN-LSTM:

- CNNFeatureExtractor: Convolutional layers to extract spatial features
- LSTMHead: LSTM layers to model temporal/sequential dependencies

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 1.0.0 (ULTIMATE EDITION)
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ================================================================
# WEIGHT INITIALIZATION
# ================================================================
def init_weights(module: nn.Module, init_type: str = "kaiming") -> None:
    """Initialize weights of a module."""
    if isinstance(module, nn.Conv2d):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            module.weight.data *= 0.1
        elif init_type == "xavier":
            nn.init.xavier_normal_(module.weight)
            module.weight.data *= 0.1
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.01)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        elif init_type == "xavier":
            nn.init.xavier_normal_(module.weight)
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.01)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


# ================================================================
# CNN FEATURE EXTRACTOR
# ================================================================
class CNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor for CNN-LSTM.

    Architecture:
        Conv2d → BatchNorm → ReLU → MaxPool (repeated)
        → AdaptiveAvgPool → Flatten

    Outputs a sequence of feature vectors (one per spatial location).
    """

    def __init__(
        self,
        input_channels: int = 3,
        use_batch_norm: bool = True,
        activation: str = "relu",
        init_type: str = "kaiming",
    ) -> None:
        super(CNNFeatureExtractor, self).__init__()

        self.input_channels = input_channels
        self.use_batch_norm = use_batch_norm
        self.activation = activation

        # ==========================================
        # Activation
        # ==========================================
        if activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation.lower() == "swish":
            self.act = nn.SiLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # ==========================================
        # Block 1: 3 → 64
        # ==========================================
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=not use_batch_norm)
        self.bn1 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        init_weights(self.conv1, init_type)

        # ==========================================
        # Block 2: 64 → 128
        # ==========================================
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        init_weights(self.conv2, init_type)

        # ==========================================
        # Block 3: 128 → 256
        # ==========================================
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=not use_batch_norm)
        self.bn3 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()
        init_weights(self.conv3, init_type)

        # ==========================================
        # Block 4: 256 → 512
        # ==========================================
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=not use_batch_norm)
        self.bn4 = nn.BatchNorm2d(512) if use_batch_norm else nn.Identity()
        init_weights(self.conv4, init_type)

        # ==========================================
        # MaxPool
        # ==========================================
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ==========================================
        # Adaptive Pool (to fixed size)
        # ==========================================
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ==========================================
        # Output
        # ==========================================
        self.output_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 3, 224, 224)

        Returns:
            (batch, seq_len, feature_dim)
        """
        # Block 1: 224 → 112
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)

        # Block 2: 112 → 56
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)

        # Block 3: 56 → 28
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.pool(x)

        # Block 4: 28 → 14
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.pool(x)

        # Adaptive pool: 14 → 7
        x = self.avgpool(x)

        # (batch, 512, 7, 7) → (batch, 49, 512)
        batch_size, channels, h, w = x.size()
        x = x.view(batch_size, channels, h * w)  # (batch, 512, 49)
        x = x.permute(0, 2, 1)  # (batch, 49, 512)

        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "CNNFeatureExtractor",
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


# ================================================================
# LSTM HEAD
# ================================================================
class LSTMHead(nn.Module):
    """
    LSTM Head for CNN-LSTM.

    Architecture:
        LSTM (input_dim=512, hidden_dim=256, num_layers=2)
        → Take last output
        → Classifier (Linear + BatchNorm + ReLU + Dropout + Linear)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        init_type: str = "kaiming",
    ) -> None:
        super(LSTMHead, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # ==========================================
        # LSTM
        # ==========================================
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False,
        )

        # ==========================================
        # Classifier
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256, momentum=0.01) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, momentum=0.01) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

        # ==========================================
        # Initialize Classifier
        # ==========================================
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, num_classes)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take last output
        last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # Classifier
        out = self.classifier(last_out)

        return out

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "LSTMHead",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================
def get_block_info(block: nn.Module) -> Dict[str, Any]:
    """Get information about a block."""
    info = {
        "type": block.__class__.__name__,
        "parameters": sum(p.numel() for p in block.parameters()),
    }

    if hasattr(block, "get_config"):
        info["config"] = block.get_config()

    return info


# ================================================================
# TEST
# ================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing CNNLSTM Blocks")
    print("=" * 80)

    # Test CNN Feature Extractor
    print("\n📌 Testing CNNFeatureExtractor:")
    cnn = CNNFeatureExtractor(input_channels=3)
    x = torch.randn(2, 3, 224, 224)
    y = cnn(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    print(f"   Params: {sum(p.numel() for p in cnn.parameters()):,}")

    # Test LSTM Head
    print("\n📌 Testing LSTMHead:")
    lstm = LSTMHead(input_dim=512, hidden_dim=256, num_classes=5)
    x = torch.randn(2, 49, 512)
    y = lstm(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    print(f"   Params: {sum(p.numel() for p in lstm.parameters()):,}")

    print("\n✅ All CNNLSTM blocks are working correctly!")
    print("=" * 80)
