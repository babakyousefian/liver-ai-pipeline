# src/models/ViT/blocks.py
# -*- coding: utf-8 -*-

"""
Vision Transformer Building Blocks
===================================
Modern implementations of ViT components:

- PatchEmbedding: Image -> patches -> tokens
- MultiHeadSelfAttention: Self-attention mechanism
- MLPBlock: Feed-forward network
- TransformerBlock: Complete encoder layer
- StochasticDepth / DropPath: Regularization

Author: Babak Yousefian
"""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# ================================================================
# Stochastic Depth (DropPath)
# ================================================================

def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample.

    Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """DropPath module wrapper for nn.Module usage."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class StochasticDepth(nn.Module):
    """Alias for DropPath (common name in ViT papers)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_path = DropPath(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(x)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_path.drop_prob:.3f}"


# ================================================================
# Patch Embedding
# ================================================================

class PatchEmbedding(nn.Module):
    """
    Convert image into patch embeddings.

    Uses Conv2d with stride=kernel_size (patch_size) for efficiency.
    This is mathematically equivalent to:
        - Split image into non-overlapping patches
        - Flatten each patch
        - Apply linear projection

    Args:
        img_size: Input image size (assumed square)
        patch_size: Patch size (assumed square)
        in_channels: Input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()

        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by "
            f"patch_size ({patch_size})"
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # Conv2d projection (equivalent to flatten + linear)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, num_patches, embed_dim)
        """
        # (B, embed_dim, grid, grid)
        x = self.proj(x)
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

    def extra_repr(self) -> str:
        return (
            f"img_size={self.img_size}, patch_size={self.patch_size}, "
            f"num_patches={self.num_patches}, embed_dim={self.embed_dim}"
        )


# ================================================================
# Multi-Head Self-Attention
# ================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA).

    Standard scaled dot-product attention with:
        - QKV projection in a single Linear
        - Scaled dot-product: softmax(QK^T / sqrt(d_k)) V
        - Output projection

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        qkv_bias: Use bias in QKV projection
        attn_dropout: Dropout on attention weights
        proj_dropout: Dropout on output projection
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Single QKV projection (efficient)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, N, C)
            return_attention: If True, also return attention weights

        Returns:
            out: (B, N, C)
            attn_weights (optional): (B, num_heads, N, N)
        """
        B, N, C = x.shape

        # (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        # (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # (B, num_heads, N, head_dim)
        out = attn @ v
        # (B, N, num_heads, head_dim) -> (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_dropout(out)

        if return_attention:
            return out, attn
        return out, None

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )


# ================================================================
# MLP Block
# ================================================================

class MLPBlock(nn.Module):
    """
    Feed-forward MLP used inside Transformer block.

    Structure:
        Linear(C, hidden) -> GELU -> Dropout -> Linear(hidden, C) -> Dropout

    Args:
        embed_dim: Input/output dimension
        mlp_ratio: Hidden dimension multiplier (hidden = embed_dim * mlp_ratio)
        dropout: Dropout rate
        activation: 'gelu' | 'relu' | 'silu'
    """

    def __init__(
        self,
        embed_dim: int = 192,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = self._get_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        if name == "gelu":
            return nn.GELU()
        elif name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}"


# ================================================================
# Transformer Block (Pre-LN)
# ================================================================

class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block with Pre-LN architecture.

    Pre-LN (pre-normalization) is more stable during training than Post-LN
    and is the standard in modern ViTs (e.g., CLIP, DINO).

    Structure:
        x = x + DropPath( MHSA( LayerNorm(x) ) )
        x = x + DropPath( MLP( LayerNorm(x) ) )

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden ratio
        qkv_bias: Use bias in QKV projection
        dropout: General dropout
        attn_dropout: Dropout on attention weights
        drop_path: Stochastic depth rate
        activation: MLP activation
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation,
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Attention block (Pre-LN + residual)
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, return_attention=return_attention)
        x = x + self.drop_path1(attn_out)

        # MLP block (Pre-LN + residual)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x, attn_weights
