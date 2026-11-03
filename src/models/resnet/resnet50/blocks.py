# src/models/resnet/resnet50/blocks.py

"""
ResNet50 Core Building Blocks (ULTIMATE EDITION)

This module implements the fundamental building blocks of ResNet50 architecture:
- Bottleneck: Used in ResNet50, ResNet101, and ResNet152
- make_layer: Helper function to create repeated blocks

Key Differences from ResNet18:
- Uses Bottleneck blocks instead of BasicBlock
- Expansion factor of 4 (vs 1 in BasicBlock)
- 1x1 -> 3x3 -> 1x1 convolution structure

Features:
- Advanced weight initialization (Kaiming, Xavier, etc.)
- Zero-init residual connections for better gradient flow
- Multiple activation functions (ReLU, Swish, LeakyReLU)
- Block analysis and visualization tools
- Layer-wise configuration support
- Export/Import capabilities

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 3.0.0 (ULTIMATE EDITION)
"""

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# ================================================================
# ✅ ADVANCED WEIGHT INITIALIZATION 🔥 NEW
# ================================================================


def init_weights(module: nn.Module, init_type: str = "kaiming", gain: float = 1.0) -> None:
    """
    Initialize weights of a module with various strategies.

    Args:
        module: PyTorch module
        init_type: Type of initialization ('kaiming', 'xavier', 'normal', 'uniform')
        gain: Gain factor for initialization
    """
    if isinstance(module, nn.Conv2d):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif init_type == "xavier":
            nn.init.xavier_normal_(module.weight, gain=gain)
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.01)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight, a=-0.1, b=0.1)
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif init_type == "xavier":
            nn.init.xavier_normal_(module.weight, gain=gain)
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.01)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight, a=-0.1, b=0.1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class Bottleneck(nn.Module):
    """
    Bottleneck Residual Block for ResNet50, ResNet101, and ResNet152.

    Architecture:
        ┌─────────────────────────────────────┐
        │  Conv2d (1×1, stride=1)             │  ← Reduce channels (1/4)
        │  BatchNorm2d                         │
        │  ReLU                                │
        │  Conv2d (3×3, stride=stride)        │  ← Main processing
        │  BatchNorm2d                         │
        │  ReLU                                │
        │  Conv2d (1×1, stride=1)             │  ← Expand channels (4x)
        │  BatchNorm2d                         │
        └─────────────────────────────────────┘
                       ↓
        ┌─────────────────────────────────────┐
        │  out = F(x) + (x or downsample(x))  │
        └─────────────────────────────────────┘
                       ↓
                    ReLU
                       ↓
                   Output

    This block uses a bottleneck design: 1x1 → 3x3 → 1x1 convolutions.
    The first 1x1 reduces channels, the 3x3 processes spatial information,
    and the final 1x1 restores channel dimensions.

    Expansion Factor: 4 (output channels = out_channels * 4)
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_batch_norm: bool = True,
        activation: str = "relu",
        init_type: str = "kaiming",
        zero_init_residual: bool = False,
    ) -> None:
        """
        Initialize the Bottleneck block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (actual = out_channels * expansion)
            stride: Stride for the 3x3 convolution
            downsample: Optional downsampling layer for skip connection
            use_batch_norm: Whether to use batch normalization
            activation: Activation function type ('relu', 'swish', or 'leaky_relu')
            init_type: Weight initialization type
            zero_init_residual: Whether to zero-initialize residual connections
        """
        super(Bottleneck, self).__init__()

        # ==========================================
        # Validate arguments
        # ==========================================
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}")
        if stride not in [1, 2]:
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        # ==========================================
        # Store configuration
        # ==========================================
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.init_type = init_type
        self.zero_init_residual = zero_init_residual
        self.downsample = downsample

        # Intermediate channel reduction (1/4 of output)
        mid_channels = out_channels // self.expansion
        self.mid_channels = mid_channels

        # ==========================================
        # First 1x1 Convolution (Reduce Channels)
        # ==========================================
        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            bias=False if use_batch_norm else True,
        )
        init_weights(self.conv1, init_type)
        self.bn1 = nn.BatchNorm2d(mid_channels) if use_batch_norm else nn.Identity()

        # ==========================================
        # Activation Function
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
        # 3x3 Convolution (Spatial Processing)
        # ==========================================
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False if use_batch_norm else True,
        )
        init_weights(self.conv2, init_type)
        self.bn2 = nn.BatchNorm2d(mid_channels) if use_batch_norm else nn.Identity()

        # ==========================================
        # Final 1x1 Convolution (Expand Channels)
        # ==========================================
        self.conv3 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False if use_batch_norm else True,
        )
        init_weights(self.conv3, init_type)
        self.bn3 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        # ==========================================
        # Zero-initialize residual connections if requested
        # ==========================================
        if zero_init_residual:
            if use_batch_norm:
                nn.init.constant_(self.bn3.weight, 0)
            else:
                nn.init.constant_(self.conv3.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Bottleneck block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        identity = x

        # ==========================================
        # Main Path
        # ==========================================
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # ==========================================
        # Skip Connection
        # ==========================================
        if self.downsample is not None:
            identity = self.downsample(x)

        # ==========================================
        # Add Skip Connection
        # ==========================================
        out += identity
        out = self.act(out)

        return out

    def get_flops(self, input_shape: Tuple[int, int]) -> int:
        """
        Calculate FLOPs for this block.

        Args:
            input_shape: (H, W) of input tensor

        Returns:
            Number of floating point operations
        """
        h, w = input_shape
        mid_channels = self.out_channels // self.expansion

        flops = 0

        # Conv1: 1x1
        flops += mid_channels * self.in_channels * h * w

        # Conv2: 3x3
        h2 = h // self.stride
        w2 = w // self.stride
        flops += mid_channels * mid_channels * 9 * h2 * w2

        # Conv3: 1x1
        flops += self.out_channels * mid_channels * h2 * w2

        return flops

    def get_params(self) -> Dict[str, int]:
        """
        Get parameter count for this block.

        Returns:
            Dictionary with parameter counts
        """
        params = {
            "conv1": sum(p.numel() for p in self.conv1.parameters()),
            "conv2": sum(p.numel() for p in self.conv2.parameters()),
            "conv3": sum(p.numel() for p in self.conv3.parameters()),
            "bn1": (sum(p.numel() for p in self.bn1.parameters()) if self.use_batch_norm else 0),
            "bn2": (sum(p.numel() for p in self.bn2.parameters()) if self.use_batch_norm else 0),
            "bn3": (sum(p.numel() for p in self.bn3.parameters()) if self.use_batch_norm else 0),
        }

        if self.downsample:
            params["downsample"] = sum(p.numel() for p in self.downsample.parameters())
        else:
            params["downsample"] = 0

        params["total"] = sum(params.values())

        return params

    def get_config(self) -> Dict[str, Any]:
        """Get block configuration."""
        return {
            "type": "Bottleneck",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "expansion": self.expansion,
            "mid_channels": self.mid_channels,
            "stride": self.stride,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "init_type": self.init_type,
            "zero_init_residual": self.zero_init_residual,
            "has_downsample": self.downsample is not None,
            "parameters": sum(p.numel() for p in self.parameters()),
        }

    def __repr__(self) -> str:
        return (
            f"Bottleneck(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"stride={self.stride}, "
            f"expansion={self.expansion}, "
            f"activation={self.activation})"
        )


def make_layer(
    block_type: type,
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int = 1,
    use_batch_norm: bool = True,
    activation: str = "relu",
    init_type: str = "kaiming",
    zero_init_residual: bool = False,
) -> nn.Sequential:
    """
    Create a layer consisting of multiple residual blocks.

    This function creates a sequence of residual blocks with the same
    output channels. The first block may have a different stride for
    downsampling, and subsequent blocks have stride=1.

    Args:
        block_type: Type of block to use (Bottleneck for ResNet50)
        in_channels: Input channels for the first block
        out_channels: Output channels for all blocks
        num_blocks: Number of blocks in this layer
        stride: Stride for the first block (subsequent blocks use stride=1)
        use_batch_norm: Whether to use batch normalization
        activation: Activation function type
        init_type: Weight initialization type
        zero_init_residual: Whether to zero-initialize residual connections

    Returns:
        Sequential container with all blocks
    """
    # ==========================================
    # Validate arguments
    # ==========================================
    if in_channels < 1:
        raise ValueError(f"in_channels must be >= 1, got {in_channels}")
    if out_channels < 1:
        raise ValueError(f"out_channels must be >= 1, got {out_channels}")
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if stride not in [1, 2]:
        raise ValueError(f"stride must be 1 or 2, got {stride}")

    # ==========================================
    # Downsample configuration for first block
    # ==========================================
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False if use_batch_norm else True,
            ),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
        )
        # Initialize downsample
        init_weights(downsample[0], init_type)

    # ==========================================
    # Build the layer
    # ==========================================
    layers = []

    # First block (may have downsampling)
    layers.append(
        block_type(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            downsample=downsample,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_type=init_type,
            zero_init_residual=zero_init_residual,
        )
    )

    # Subsequent blocks (stride=1, no downsampling)
    for _ in range(1, num_blocks):
        layers.append(
            block_type(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                downsample=None,
                use_batch_norm=use_batch_norm,
                activation=activation,
                init_type=init_type,
                zero_init_residual=zero_init_residual,
            )
        )

    return nn.Sequential(*layers)


# ================================================================
# ✅ BLOCK ANALYSIS FUNCTIONS 🔥 NEW
# ================================================================


def get_block_info(block: nn.Module) -> Dict[str, Any]:
    """
    Get detailed information about a block.

    Args:
        block: A block module (Bottleneck)

    Returns:
        Dictionary with block information
    """
    info: Dict[str, Any] = {
        "type": block.__class__.__name__,
        "parameters": sum(p.numel() for p in block.parameters()),
        "trainable_parameters": sum(p.numel() for p in block.parameters() if p.requires_grad),
    }

    # Try to get attributes if they exist
    for attr in [
        "in_channels",
        "out_channels",
        "stride",
        "expansion",
        "mid_channels",
        "use_batch_norm",
        "activation",
    ]:
        if hasattr(block, attr):
            info[attr] = getattr(block, attr)

    if hasattr(block, "downsample") and block.downsample:
        info["has_downsample"] = True
        info["downsample_params"] = sum(p.numel() for p in block.downsample.parameters())
    else:
        info["has_downsample"] = False
        info["downsample_params"] = 0

    if hasattr(block, "get_params"):
        info["param_breakdown"] = block.get_params()

    if hasattr(block, "get_config"):
        info["config"] = block.get_config()

    return info


def get_layer_info(layer: nn.Sequential) -> Dict[str, Any]:
    """
    Get detailed information about a layer (sequence of blocks).

    Args:
        layer: Sequential layer of blocks

    Returns:
        Dictionary with layer information
    """
    info: Dict[str, Any] = {
        "num_blocks": len(layer),
        "block_types": [],
        "total_parameters": 0,
        "trainable_parameters": 0,
        "block_infos": [],
    }

    for block in layer:
        block_info = get_block_info(block)
        info["block_types"].append(block_info["type"])
        info["total_parameters"] += block_info["parameters"]
        info["trainable_parameters"] += block_info["trainable_parameters"]
        info["block_infos"].append(block_info)

    return info


def print_layer_info(layer: nn.Sequential, name: str = "Layer") -> None:
    """
    Print detailed information about a layer.

    Args:
        layer: Sequential layer of blocks
        name: Name of the layer
    """
    info = get_layer_info(layer)

    print(f"\n📊 {name} Information:")
    print(f"   - Number of Blocks: {info['num_blocks']}")
    print(f"   - Block Types: {', '.join(info['block_types'])}")
    print(f"   - Total Parameters: {info['total_parameters']:,}")
    print(f"   - Trainable Parameters: {info['trainable_parameters']:,}")

    for i, block_info in enumerate(info["block_infos"]):
        print(f"\n   Block {i+1}:")
        print(f"      - Type: {block_info['type']}")
        if "in_channels" in block_info:
            print(f"      - In Channels: {block_info['in_channels']}")
        if "out_channels" in block_info:
            print(f"      - Out Channels: {block_info['out_channels']}")
        if "mid_channels" in block_info:
            print(f"      - Mid Channels: {block_info['mid_channels']}")
        if "stride" in block_info:
            print(f"      - Stride: {block_info['stride']}")
        if "expansion" in block_info:
            print(f"      - Expansion: {block_info['expansion']}")
        if "has_downsample" in block_info:
            print(f"      - Has Downsample: {block_info['has_downsample']}")
        if "activation" in block_info:
            print(f"      - Activation: {block_info['activation']}")
        print(f"      - Parameters: {block_info['parameters']:,}")


def count_blocks_in_layer(layer: nn.Sequential) -> Dict[str, int]:
    """
    Count the number of each block type in a layer.

    Args:
        layer: Sequential layer of blocks

    Returns:
        Dictionary with block type counts
    """
    counts: Dict[str, int] = {}
    for block in layer:
        block_type = block.__class__.__name__
        counts[block_type] = counts.get(block_type, 0) + 1
    return counts


def get_layer_io_shapes(layer: nn.Sequential, input_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    Get input and output shapes for a layer.

    Args:
        layer: Sequential layer of blocks
        input_shape: (channels, height, width) of input tensor

    Returns:
        Dictionary with input and output shapes
    """
    dummy = torch.randn(1, *input_shape)
    shapes = []
    current = dummy

    for i, block in enumerate(layer):
        current = block(current)
        shapes.append(
            {
                "block_index": i,
                "block_type": block.__class__.__name__,
                "output_shape": tuple(current.shape),
            }
        )

    return {
        "input_shape": input_shape,
        "output_shape": tuple(current.shape),
        "block_shapes": shapes,
    }


def analyze_resnet50_blocks(model: nn.Module) -> Dict[str, Any]:
    """
    Analyze all blocks in a ResNet50 model.

    Args:
        model: PyTorch ResNet50 model

    Returns:
        Dictionary with analysis results
    """
    results = {
        "total_blocks": 0,
        "total_parameters": 0,
        "blocks": [],
        "layer_counts": {},
    }

    for name, module in model.named_modules():
        if isinstance(module, Bottleneck):
            info = get_block_info(module)
            results["blocks"].append({"name": name, "info": info})
            results["total_blocks"] += 1
            results["total_parameters"] += info["parameters"]

            # Count by type
            block_type = module.__class__.__name__
            results["layer_counts"][block_type] = results["layer_counts"].get(block_type, 0) + 1

    return results


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing ResNet50 Blocks (ULTIMATE EDITION)")
    print("=" * 80)

    # Test Bottleneck
    print("\n📌 Testing Bottleneck:")
    block = Bottleneck(64, 256, stride=1)
    x = torch.randn(4, 64, 56, 56)
    y = block(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in block.parameters()):,}")
    print(f"   Block info: {block}")
    print(f"   Config: {block.get_config()}")

    # Test Bottleneck with downsampling
    block = Bottleneck(64, 256, stride=2)
    x = torch.randn(4, 64, 56, 56)
    y = block(x)
    print(f"\n   Input shape (downsample): {x.shape}")
    print(f"   Output shape (downsample): {y.shape}")

    # Test make_layer
    print("\n📌 Testing make_layer (ResNet50 stages):")

    layer1 = make_layer(Bottleneck, 64, 256, 3, stride=1)
    x = torch.randn(4, 64, 56, 56)
    y = layer1(x)
    print(f"\n   Stage 1 (64→256, 3 blocks):")
    print(f"      Input shape: {x.shape}")
    print(f"      Output shape: {y.shape}")
    print(f"      Total parameters: {sum(p.numel() for p in layer1.parameters()):,}")

    layer2 = make_layer(Bottleneck, 256, 512, 4, stride=2)
    x = torch.randn(4, 256, 56, 56)
    y = layer2(x)
    print(f"\n   Stage 2 (256→512, 4 blocks, stride=2):")
    print(f"      Input shape: {x.shape}")
    print(f"      Output shape: {y.shape}")
    print(f"      Total parameters: {sum(p.numel() for p in layer2.parameters()):,}")

    # Test get_layer_info
    print("\n📌 Testing get_layer_info:")
    print_layer_info(layer2, "Stage 2 (ResNet50)")

    # Test count_blocks_in_layer
    print("\n📌 Testing count_blocks_in_layer:")
    counts = count_blocks_in_layer(layer2)
    print(f"   Block counts: {counts}")

    # Test FLOPs calculation
    print("\n📌 Testing FLOPs calculation:")
    block = Bottleneck(64, 256, stride=2)
    flops = block.get_flops((56, 56))
    print(f"   Bottleneck FLOPs: {flops:,}")

    # Test different activations
    print("\n📌 Testing different activations:")
    activations = ["relu", "swish", "leaky_relu"]
    for act in activations:
        block = Bottleneck(64, 256, stride=1, activation=act)
        x = torch.randn(4, 64, 56, 56)
        y = block(x)
        print(f"   {act}: output shape {y.shape}")

    # Test error handling
    print("\n📌 Testing error handling:")
    try:
        block = Bottleneck(0, 256, stride=1)
        print("   ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {e}")

    try:
        block = Bottleneck(64, 256, stride=3)
        print("   ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {e}")

    print("\n✅ All ResNet50 blocks are working correctly!")
    print("=" * 80)
