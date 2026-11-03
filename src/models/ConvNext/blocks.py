# src/models/ConvNext/blocks.py
# -*- coding: utf-8 -*-

"""
ConvNext Building Blocks
========================
ConvNeXt block: 7x7 depthwise conv + LayerNorm + 1x1 conv + GELU + 1x1 conv + residual
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
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            module.weight.data *= 0.1
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, LayerNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D tensors (channels-last)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight + self.bias
        # (B, H, W, C) → (B, C, H, W)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNextBlock(nn.Module):
    """
    ConvNeXt Block.
    Architecture: 7x7 DepthwiseConv → LayerNorm → 1x1 Conv → GELU → 1x1 Conv → Residual
    """

    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        use_batch_norm: bool = False,
        init_type: str = "kaiming",
    ):
        super().__init__()
        self.dim = dim

        # 7x7 depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        init_weights(self.dwconv, init_type)

        # LayerNorm
        self.norm = LayerNorm2d(dim)

        # 1x1 conv (expand)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        init_weights(self.pwconv1, init_type)

        # GELU
        self.act = nn.GELU()

        # 1x1 conv (reduce)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        init_weights(self.pwconv2, init_type)

        # Layer scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # 7x7 depthwise conv
        x = self.dwconv(x)

        # LayerNorm
        x = self.norm(x)

        # 1x1 conv + GELU + 1x1 conv
        # (B, C, H, W) → (B, H, W, C) → Linear → (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2).contiguous()

        # Residual
        x = residual + x
        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "ConvNextBlock",
            "dim": self.dim,
            "parameters": sum(p.numel() for p in self.parameters()),
        }
