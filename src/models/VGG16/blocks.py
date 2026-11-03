# src/models/VGG16/blocks.py
# -*- coding: utf-8 -*-

"""
VGG16 Core Building Blocks
===========================
This module implements the fundamental building blocks of VGG16 architecture:
- ConvBlock: Standard convolutional block (Conv2d + BN + ReLU)
- make_stage: Helper function to create repeated conv blocks

VGG16 Architecture:
- 13 convolutional layers (5 stages)
- 3 fully connected layers
- Uses small 3x3 filters throughout
- Max pooling for downsampling

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
    """
    Initialize weights of a module.

    Args:
        module: PyTorch module
        init_type: Type of initialization ('kaiming', 'xavier', 'normal')
    """
    if isinstance(module, nn.Conv2d):
        if init_type == "kaiming":
            # ✅ برای VGG16، fan_in بهتره
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            # ✅ scale down برای جلوگیری از انفجار
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
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif init_type == "xavier":
            nn.init.xavier_normal_(module.weight)
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.01)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


# ================================================================
# CONV BLOCK (VGG16 Building Block)
# ================================================================
class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for VGG16.

    Architecture:
        Conv2d → BatchNorm2d → ReLU

    This is the fundamental building block of VGG16.
    Each block consists of a single 3x3 convolution followed by
    batch normalization (optional) and ReLU activation.

    VGG16 uses:
    - kernel_size = 3
    - padding = 1
    - stride = 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        init_type: str = "kaiming",
    ) -> None:
        """
        Initialize ConvBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size (default: 3)
            stride: Convolution stride (default: 1)
            padding: Padding size (default: 1)
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'swish', 'leaky_relu')
            init_type: Weight initialization type
        """
        super(ConvBlock, self).__init__()

        # ==========================================
        # Validate arguments
        # ==========================================
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}")

        # ==========================================
        # Store configuration
        # ==========================================
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_batch_norm = use_batch_norm
        self.activation = activation

        # ==========================================
        # Convolution
        # ==========================================
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm,
        )
        init_weights(self.conv, init_type)

        # ==========================================
        # BatchNorm
        # ==========================================
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Get block configuration."""
        return {
            "type": "ConvBlock",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "parameters": sum(p.numel() for p in self.parameters()),
        }

    def __repr__(self) -> str:
        return (
            f"ConvBlock({self.in_channels} → {self.out_channels}, "
            f"k={self.kernel_size}, s={self.stride})"
        )


# ================================================================
# MAKE STAGE (Create Sequential Conv Blocks)
# ================================================================
def make_stage(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    kernel_size: int = 3,
    padding: int = 1,
    use_batch_norm: bool = True,
    activation: str = "relu",
    init_type: str = "kaiming",
) -> nn.Sequential:
    """
    Create a stage (sequential of ConvBlocks).

    Args:
        in_channels: Input channels for first block
        out_channels: Output channels for all blocks
        num_blocks: Number of blocks in this stage
        kernel_size: Kernel size
        padding: Padding
        use_batch_norm: Use batch norm
        activation: Activation function
        init_type: Weight initialization type

    Returns:
        Sequential container with all blocks
    """
    blocks = []

    # First block (may have different in_channels)
    blocks.append(
        ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_type=init_type,
        )
    )

    # Subsequent blocks (same channels)
    for _ in range(1, num_blocks):
        blocks.append(
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                use_batch_norm=use_batch_norm,
                activation=activation,
                init_type=init_type,
            )
        )

    return nn.Sequential(*blocks)


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================
def get_block_info(block: nn.Module) -> Dict[str, Any]:
    """Get information about a block."""
    info = {
        "type": block.__class__.__name__,
        "parameters": sum(p.numel() for p in block.parameters()),
    }

    for attr in ["in_channels", "out_channels", "kernel_size", "stride"]:
        if hasattr(block, attr):
            info[attr] = getattr(block, attr)

    if hasattr(block, "get_config"):
        info["config"] = block.get_config()

    return info


def get_stage_info(stage: nn.Sequential) -> Dict[str, Any]:
    """Get information about a stage."""
    info = {
        "num_blocks": len(stage),
        "total_parameters": sum(p.numel() for p in stage.parameters()),
        "block_infos": [],
    }

    for block in stage:
        info["block_infos"].append(get_block_info(block))

    return info


def count_blocks_in_stage(stage: nn.Sequential) -> Dict[str, int]:
    """Count block types in a stage."""
    counts = {}
    for block in stage:
        block_type = block.__class__.__name__
        counts[block_type] = counts.get(block_type, 0) + 1
    return counts


# ================================================================
# TEST
# ================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing VGG16 Blocks")
    print("=" * 80)

    # Test ConvBlock
    print("\n📌 Testing ConvBlock:")
    block = ConvBlock(3, 64, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 224, 224)
    y = block(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    print(f"   Params: {sum(p.numel() for p in block.parameters()):,}")
    print(f"   Config: {block.get_config()}")

    # Test make_stage
    print("\n📌 Testing make_stage:")
    stage = make_stage(3, 64, num_blocks=2)
    x = torch.randn(2, 3, 224, 224)
    y = stage(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    print(f"   Params: {sum(p.numel() for p in stage.parameters()):,}")
    print(f"   Stage info: {get_stage_info(stage)}")

    print("\n✅ All VGG16 blocks are working correctly!")
    print("=" * 80)
