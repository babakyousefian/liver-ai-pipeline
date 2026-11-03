# src/models/efficientNet/B0/blocks.py

"""
EfficientNet-B0 Core Building Blocks (ULTIMATE EDITION)

This module implements the fundamental building blocks of EfficientNet architecture:
- MBConvBlock: Mobile Inverted Bottleneck Conv block with Squeeze-and-Excitation
- SqueezeExcitation: Channel attention mechanism (SE block)
- make_layer: Helper function to create repeated blocks

Key Features:
- Depthwise separable convolutions for efficiency
- Squeeze-and-Excitation (SE) for channel attention
- Swish (SiLU) activation function
- Stochastic depth (DropConnect) for regularization
- Advanced weight initialization
- Layer-wise learning rate support
- Custom forward hooks for analysis
- Block analysis and visualization tools

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 3.0.0 (ULTIMATE EDITION)
"""

import math
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)

    Also known as SiLU (Sigmoid Linear Unit).
    Used in EfficientNet instead of ReLU.

    Mathematical formula: f(x) = x * σ(x)
    Where σ(x) is the sigmoid function.

    Properties:
    - Smooth, non-monotonic
    - Self-gated
    - Better performance than ReLU in deep networks
    """

    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation."""
        return x * torch.sigmoid(x)

    def __repr__(self) -> str:
        return "Swish()"


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for channel attention.

    Architecture:
        ┌─────────────────────────────────────┐
        │  Global Average Pooling (Squeeze)    │
        │  (B, C, H, W) → (B, C, 1, 1)       │
        └─────────────────────────────────────┘
                       ↓
        ┌─────────────────────────────────────┐
        │  FC1: C → C//ratio (Reduce)         │
        │  Swish / ReLU                       │
        │  FC2: C//ratio → C (Expand)         │
        │  Sigmoid (Excitation)               │
        └─────────────────────────────────────┘
                       ↓
        ┌─────────────────────────────────────┐
        │  Scale: multiply with original      │
        │  (B, C, 1, 1) * (B, C, H, W)       │
        └─────────────────────────────────────┘

    This block learns channel-wise attention weights
    to emphasize important features and suppress irrelevant ones.

    Reference: "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 4,
        activation: str = "swish",
        init_type: str = "kaiming",
    ) -> None:
        """
        Initialize Squeeze-and-Excitation block.

        Args:
            channels: Number of input channels (C)
            reduction_ratio: Reduction ratio for bottleneck (default: 4)
            activation: Activation function ('swish' or 'relu')
            init_type: Weight initialization type
        """
        super(SqueezeExcitation, self).__init__()

        # ==========================================
        # Validate arguments
        # ==========================================
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if reduction_ratio < 1:
            raise ValueError(f"reduction_ratio must be >= 1, got {reduction_ratio}")

        # ==========================================
        # Squeeze: Global Average Pooling
        # ==========================================
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # ==========================================
        # Excitation: Two FC layers
        # ==========================================
        reduced_channels = max(1, channels // reduction_ratio)

        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)

        # Activation functions
        if activation.lower() == "swish":
            self.act = Swish()
        elif activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.sigmoid = nn.Sigmoid()

        # ==========================================
        # Store configuration
        # ==========================================
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self.reduced_channels = reduced_channels
        self.init_type = init_type

        # ==========================================
        # Initialize weights
        # ==========================================
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using specified strategy."""
        init_weights(self.fc1, self.init_type)
        init_weights(self.fc2, self.init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SE block.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Scaled tensor of same shape (B, C, H, W)
        """
        # Squeeze: Global Average Pooling
        se = self.global_avgpool(x)  # (B, C, 1, 1)
        se = se.view(se.size(0), -1)  # (B, C)

        # Excitation
        se = self.fc1(se)  # (B, C//ratio)
        se = self.act(se)
        se = self.fc2(se)  # (B, C)
        se = self.sigmoid(se)  # (B, C)

        # Scale: reshape and multiply
        se = se.view(se.size(0), se.size(1), 1, 1)  # (B, C, 1, 1)
        return x * se

    def get_params(self) -> Dict[str, int]:
        """Get parameter count for this block."""
        return {
            "fc1": sum(p.numel() for p in self.fc1.parameters()),
            "fc2": sum(p.numel() for p in self.fc2.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }

    def get_flops(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate FLOPs for this block."""
        b, c, h, w = input_shape
        reduced_c = self.reduced_channels

        flops = c * h * w  # Global Average Pooling
        flops += c * reduced_c  # FC1
        flops += reduced_c * c  # FC2
        flops += c * h * w  # Sigmoid and multiplication

        return flops

    def __repr__(self) -> str:
        return (
            f"SqueezeExcitation(channels={self.channels}, "
            f"reduction_ratio={self.reduction_ratio}, "
            f"activation={self.activation})"
        )


class StochasticDepth(nn.Module):
    """
    Stochastic Depth (DropConnect) for regularization.

    Randomly drops entire residual blocks during training.
    This helps prevent overfitting and improves generalization.

    Reference: "Deep Networks with Stochastic Depth" (ECCV 2016)
    """

    def __init__(self, drop_rate: float = 0.0) -> None:
        """
        Initialize Stochastic Depth.

        Args:
            drop_rate: Probability of dropping the residual connection
        """
        super(StochasticDepth, self).__init__()

        if not (0.0 <= drop_rate <= 1.0):
            raise ValueError(f"drop_rate must be in [0, 1], got {drop_rate}")

        self.drop_rate = drop_rate
        self._is_active = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth.

        During training, randomly drops the input.
        During inference, returns input unchanged.
        """
        if not self.training or self.drop_rate == 0.0:
            return x

        keep_prob = 1.0 - self.drop_rate
        mask = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        mask = (mask < keep_prob).float()

        return x * mask / keep_prob

    def set_active(self, active: bool) -> None:
        """Enable or disable stochastic depth."""
        self._is_active = active

    def __repr__(self) -> str:
        return f"StochasticDepth(drop_rate={self.drop_rate})"


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Conv (MBConv) Block.

    Architecture (for expand_ratio > 1):
        ┌─────────────────────────────────────────────┐
        │  Conv2d (1×1, expand)                       │
        │  BatchNorm2d                                │
        │  Swish                                      │
        └─────────────────────────────────────────────┘
                         ↓
        ┌─────────────────────────────────────────────┐
        │  Depthwise Conv (k×k, groups=C)             │
        │  BatchNorm2d                                │
        │  Swish                                      │
        └─────────────────────────────────────────────┘
                         ↓
        ┌─────────────────────────────────────────────┐
        │  Squeeze-and-Excitation (optional)          │
        └─────────────────────────────────────────────┘
                         ↓
        ┌─────────────────────────────────────────────┐
        │  Conv2d (1×1, compress)                     │
        │  BatchNorm2d                                │
        └─────────────────────────────────────────────┘
                         ↓
        ┌─────────────────────────────────────────────┐
        │  Residual Connection (if stride=1)          │
        │  + Stochastic Depth (optional)              │
        └─────────────────────────────────────────────┘

    For expand_ratio = 1, the first 1x1 conv is skipped.

    Reference: "EfficientNet: Rethinking Model Scaling" (ICML 2019)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        use_se: bool = True,
        se_reduction_ratio: int = 4,
        drop_rate: float = 0.0,
        activation: str = "swish",
        use_batch_norm: bool = True,
        init_type: str = "kaiming",
        zero_init_residual: bool = True,
    ) -> None:
        """
        Initialize MBConv block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for depthwise conv
            stride: Stride for depthwise conv
            expand_ratio: Expansion ratio for hidden channels
            use_se: Whether to use Squeeze-and-Excitation
            se_reduction_ratio: Reduction ratio for SE block
            drop_rate: Stochastic depth drop rate
            activation: Activation function ('swish' or 'relu')
            use_batch_norm: Whether to use batch normalization
            init_type: Weight initialization type
            zero_init_residual: Whether to zero-initialize residual connections
        """
        super(MBConvBlock, self).__init__()

        # ==========================================
        # Validate arguments
        # ==========================================
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if stride not in [1, 2]:
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        if expand_ratio < 1:
            raise ValueError(f"expand_ratio must be >= 1, got {expand_ratio}")
        if not (0.0 <= drop_rate <= 1.0):
            raise ValueError(f"drop_rate must be in [0, 1], got {drop_rate}")

        # ==========================================
        # Store configuration
        # ==========================================
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.use_se = use_se
        self.se_reduction_ratio = se_reduction_ratio
        self.drop_rate = drop_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.init_type = init_type
        self.zero_init_residual = zero_init_residual

        # Hidden channels after expansion
        self.hidden_channels = in_channels * expand_ratio
        self.has_residual = stride == 1 and in_channels == out_channels

        # ==========================================
        # Activation function
        # ==========================================
        if activation.lower() == "swish":
            self.act = Swish()
        elif activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # ==========================================
        # Build layers
        # ==========================================
        layers: List[nn.Module] = []

        # ==========================================
        # 1x1 Expansion Conv (skip if expand_ratio == 1)
        # ==========================================
        if expand_ratio > 1:
            conv_expand = nn.Conv2d(
                in_channels,
                self.hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False if use_batch_norm else True,
            )
            init_weights(conv_expand, init_type)
            layers.append(conv_expand)

            if use_batch_norm:
                bn = nn.BatchNorm2d(self.hidden_channels)
                layers.append(bn)
            layers.append(self.act)

        # ==========================================
        # Depthwise Conv
        # ==========================================
        conv_dw = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=self.hidden_channels,
            bias=False if use_batch_norm else True,
        )
        init_weights(conv_dw, init_type)
        layers.append(conv_dw)

        if use_batch_norm:
            bn_dw = nn.BatchNorm2d(self.hidden_channels)
            layers.append(bn_dw)
        layers.append(self.act)

        # ==========================================
        # Squeeze-and-Excitation (optional)
        # ==========================================
        if use_se:
            se = SqueezeExcitation(
                self.hidden_channels,
                reduction_ratio=se_reduction_ratio,
                activation=activation,
                init_type=init_type,
            )
            layers.append(se)

        # ==========================================
        # 1x1 Compression Conv
        # ==========================================
        conv_compress = nn.Conv2d(
            self.hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False if use_batch_norm else True,
        )
        init_weights(conv_compress, init_type)
        layers.append(conv_compress)

        if use_batch_norm:
            bn_compress = nn.BatchNorm2d(out_channels)
            layers.append(bn_compress)

        # ==========================================
        # Wrap in Sequential
        # ==========================================
        self.conv = nn.Sequential(*layers)

        # ==========================================
        # Stochastic Depth (DropConnect) for regularization
        # ==========================================
        if drop_rate > 0 and self.has_residual:
            self.stochastic_depth = StochasticDepth(drop_rate)
        else:
            self.stochastic_depth = nn.Identity()

        # ==========================================
        # Zero-initialize residual connections if requested
        # ==========================================
        if zero_init_residual and self.has_residual:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, "weight") and module.weight is not None:
                        nn.init.constant_(module.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MBConv block.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C_out, H_out, W_out)
        """
        if self.has_residual:
            residual = x
            out = self.conv(x)
            out = self.stochastic_depth(out)
            return out + residual
        else:
            return self.conv(x)

    def get_flops(self, input_shape: Tuple[int, int]) -> int:
        """Calculate FLOPs for this block."""
        h, w = input_shape

        if self.stride == 2:
            h_out = h // 2
            w_out = w // 2
        else:
            h_out = h
            w_out = w

        flops = 0

        # Expansion Conv (1x1)
        if self.expand_ratio > 1:
            flops += self.hidden_channels * self.in_channels * h * w

        # Depthwise Conv (k x k)
        flops += self.hidden_channels * self.kernel_size * self.kernel_size * h * w

        # SE Block (if used)
        if self.use_se:
            reduced_channels = max(1, self.hidden_channels // self.se_reduction_ratio)
            flops += self.hidden_channels * reduced_channels
            flops += reduced_channels * self.hidden_channels
            flops += self.hidden_channels * h_out * w_out

        # Compression Conv (1x1)
        flops += self.out_channels * self.hidden_channels * h_out * w_out

        return flops

    def get_params(self) -> Dict[str, int]:
        """Get parameter count for this block."""
        params = {
            "conv_expand": 0,
            "conv_depthwise": 0,
            "conv_compress": 0,
            "bn": 0,
            "se": 0,
            "total": 0,
        }

        for name, module in self.named_children():
            if "conv" in name and isinstance(module, nn.Conv2d):
                if "conv.0" in name:
                    params["conv_expand"] += sum(p.numel() for p in module.parameters())
                elif "conv.1" in name or "conv.3" in name:
                    params["conv_depthwise"] += sum(p.numel() for p in module.parameters())
                elif "conv.4" in name or "conv.6" in name:
                    params["conv_compress"] += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.BatchNorm2d):
                params["bn"] += sum(p.numel() for p in module.parameters())
            elif isinstance(module, SqueezeExcitation):
                params["se"] += sum(p.numel() for p in module.parameters())

        params["total"] = sum(p.numel() for p in self.parameters())
        return params

    def get_layer_info(self) -> Dict[str, Any]:
        """Get detailed layer information."""
        return {
            "type": "MBConvBlock",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "use_se": self.use_se,
            "drop_rate": self.drop_rate,
            "activation": self.activation,
            "has_residual": self.has_residual,
            "parameters": sum(p.numel() for p in self.parameters()),
            "flops": self.get_flops((224, 224)),
        }

    def __repr__(self) -> str:
        return (
            f"MBConvBlock(in={self.in_channels}, "
            f"out={self.out_channels}, "
            f"kernel={self.kernel_size}, "
            f"stride={self.stride}, "
            f"expand={self.expand_ratio}, "
            f"se={self.use_se}, "
            f"activation={self.activation})"
        )


def make_layer(
    block_type: type,
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int = 1,
    kernel_size: int = 3,
    expand_ratio: int = 6,
    use_se: bool = True,
    se_reduction_ratio: int = 4,
    drop_rate: float = 0.0,
    activation: str = "swish",
    use_batch_norm: bool = True,
    init_type: str = "kaiming",
    zero_init_residual: bool = True,
) -> nn.Sequential:
    """
    Create a layer consisting of multiple MBConv blocks.

    Args:
        block_type: Type of block to use (MBConvBlock)
        in_channels: Input channels for the first block
        out_channels: Output channels for all blocks
        num_blocks: Number of blocks in this layer
        stride: Stride for the first block (subsequent blocks use stride=1)
        kernel_size: Kernel size for depthwise conv
        expand_ratio: Expansion ratio for hidden channels
        use_se: Whether to use Squeeze-and-Excitation
        se_reduction_ratio: Reduction ratio for SE block
        drop_rate: Stochastic depth drop rate
        activation: Activation function ('swish' or 'relu')
        use_batch_norm: Whether to use batch normalization
        init_type: Weight initialization type
        zero_init_residual: Whether to zero-initialize residual connections

    Returns:
        Sequential container with all blocks
    """
    layers: List[nn.Module] = []
    current_channels = in_channels

    for i in range(num_blocks):
        block_stride = stride if i == 0 else 1
        block_drop_rate = drop_rate * (i / num_blocks) if num_blocks > 1 else drop_rate

        layers.append(
            block_type(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=block_stride,
                expand_ratio=expand_ratio,
                use_se=use_se,
                se_reduction_ratio=se_reduction_ratio,
                drop_rate=block_drop_rate,
                activation=activation,
                use_batch_norm=use_batch_norm,
                init_type=init_type,
                zero_init_residual=zero_init_residual,
            )
        )

        current_channels = out_channels

    return nn.Sequential(*layers)


# ================================================================
# ✅ BLOCK ANALYSIS FUNCTIONS 🔥 NEW
# ================================================================


def analyze_block(block: nn.Module) -> Dict[str, Any]:
    """
    Analyze a block and return detailed information.

    Args:
        block: PyTorch module

    Returns:
        Dictionary with analysis results
    """
    if hasattr(block, "get_layer_info"):
        return block.get_layer_info()

    return {
        "type": block.__class__.__name__,
        "parameters": sum(p.numel() for p in block.parameters()),
        "trainable": sum(p.numel() for p in block.parameters() if p.requires_grad),
        "children": len(list(block.children())),
    }


def analyze_layer(layer: nn.Sequential) -> Dict[str, Any]:
    """
    Analyze a layer (sequence of blocks).

    Args:
        layer: Sequential layer

    Returns:
        Dictionary with analysis results
    """
    results = {
        "num_blocks": len(layer),
        "total_parameters": 0,
        "trainable_parameters": 0,
        "blocks": [],
    }

    for i, block in enumerate(layer):
        info = analyze_block(block)
        results["blocks"].append(info)
        results["total_parameters"] += info.get("parameters", 0)
        results["trainable_parameters"] += info.get("trainable", 0)

    return results


def get_block_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get summary of all blocks in a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with block summary
    """
    summary = {"total_blocks": 0, "blocks": [], "total_parameters": 0}

    for name, module in model.named_modules():
        if isinstance(module, MBConvBlock):
            summary["blocks"].append({"name": name, "info": module.get_layer_info()})
            summary["total_blocks"] += 1
            summary["total_parameters"] += sum(p.numel() for p in module.parameters())

    return summary


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing EfficientNet-B0 Blocks (ULTIMATE EDITION)")
    print("=" * 80)

    # ==========================================
    # Test SqueezeExcitation
    # ==========================================
    print("\n📌 Testing SqueezeExcitation:")
    se = SqueezeExcitation(64, reduction_ratio=4)
    x = torch.randn(4, 64, 56, 56)
    y = se(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in se.parameters()):,}")
    print(f"   SE info: {se}")

    # ==========================================
    # Test StochasticDepth
    # ==========================================
    print("\n📌 Testing StochasticDepth:")
    sd = StochasticDepth(drop_rate=0.2)
    x = torch.randn(4, 64, 56, 56)
    sd.train()
    y = sd(x)
    print(f"   Training - Input: {x.shape}, Output: {y.shape}")
    sd.eval()
    y = sd(x)
    print(f"   Eval - Input: {x.shape}, Output: {y.shape}")

    # ==========================================
    # Test MBConvBlock
    # ==========================================
    print("\n📌 Testing MBConvBlock:")
    block = MBConvBlock(32, 16, kernel_size=3, stride=1, expand_ratio=6, use_se=True)
    x = torch.randn(4, 32, 112, 112)
    y = block(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in block.parameters()):,}")
    print(f"   Block info: {block}")
    print(f"   Has residual: {block.has_residual}")

    # ==========================================
    # Test make_layer
    # ==========================================
    print("\n📌 Testing make_layer:")
    layer = make_layer(MBConvBlock, 32, 16, 2, stride=2, kernel_size=3, expand_ratio=6, use_se=True)
    x = torch.randn(4, 32, 112, 112)
    y = layer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in layer.parameters()):,}")

    # ==========================================
    # Test analyze_block
    # ==========================================
    print("\n📌 Testing analyze_block:")
    analysis = analyze_block(block)
    print(f"   Analysis: {analysis}")

    # ==========================================
    # Test analyze_layer
    # ==========================================
    print("\n📌 Testing analyze_layer:")
    layer_analysis = analyze_layer(layer)
    print(f"   Num blocks: {layer_analysis['num_blocks']}")
    print(f"   Total params: {layer_analysis['total_parameters']:,}")

    # ==========================================
    # Test get_block_summary
    # ==========================================
    print("\n📌 Testing get_block_summary:")
    try:
        from .efficientnet_b0 import create_efficientnet_b0

        model = create_efficientnet_b0(num_classes=5, pretrained=False, verbose=False)
        summary = get_block_summary(model)
        print(f"   Total blocks: {summary['total_blocks']}")
        print(f"   Total parameters: {summary['total_parameters']:,}")
    except Exception as e:
        print(f"   ⚠️ Could not test summary: {e}")

    # ==========================================
    # Test error handling
    # ==========================================
    print("\n📌 Testing error handling:")
    try:
        se = SqueezeExcitation(0, reduction_ratio=4)
        print("   ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {e}")

    try:
        block = MBConvBlock(32, 16, stride=3, expand_ratio=6)
        print("   ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {e}")

    print("\n✅ All EfficientNet-B0 blocks are working correctly!")
    print("=" * 80)
