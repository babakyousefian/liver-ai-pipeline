# src/models/ViT/vit.py
# -*- coding: utf-8 -*-

"""
Vision Transformer (ViT) Architecture from Scratch (Modernized)
================================================================

Faithful implementation of ViT (Dosovitskiy et al., 2020)
with modern adaptations for small medical datasets.

Original ViT-Base:
    - 86M parameters
    - patch_size=16, embed_dim=768, depth=12, heads=12
    - 0.0 drop_path
    - 0.1 dropout
    - batch_size=4096, warmup ~10K steps

Modernized ViT (this file):
    - ViT-Tiny: ~5.7M params (embed_dim=192, depth=6, heads=3)
    - ViT-Small: ~22M params (embed_dim=384, depth=12, heads=6)
    - Pre-LN architecture
    - Stochastic Depth (drop_path) for regularization
    - Learnable positional embeddings
    - [CLS] token for classification

Author: Babak Yousefian
"""

import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import (
    MLPBlock,
    MultiHeadSelfAttention,
    PatchEmbedding,
    TransformerBlock,
)

warnings.filterwarnings("ignore")


# ================================================================
# ViT Configuration
# ================================================================

@dataclass
class ViTConfig:
    """
    Configuration for ViT.

    Defaults = ViT-Tiny variant, tuned for small medical datasets.
    """
    num_classes: int = 4
    input_channels: int = 3
    img_size: int = 224

    # Patch embedding
    patch_size: int = 16
    embed_dim: int = 192      # ViT-Tiny: 192, ViT-Small: 384, ViT-Base: 768
    depth: int = 6            # ViT-Tiny: 6, ViT-Small: 12, ViT-Base: 12
    num_heads: int = 3        # ViT-Tiny: 3, ViT-Small: 6, ViT-Base: 12
    mlp_ratio: float = 4.0

    # Regularization
    dropout: float = 0.2          # higher than CNN (ViT needs it)
    attn_dropout: float = 0.0
    drop_path_rate: float = 0.1   # stochastic depth
    qkv_bias: bool = True

    # Head
    head_dropout: float = 0.2
    use_cls_token: bool = True    # if False, use global avg pool instead

    # Activation
    activation: str = "gelu"

    # Init
    init_std: float = 0.02

    # Misc
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "drop_path_rate": self.drop_path_rate,
            "qkv_bias": self.qkv_bias,
            "head_dropout": self.head_dropout,
            "use_cls_token": self.use_cls_token,
            "activation": self.activation,
        }


# ================================================================
# ViT Model
# ================================================================

class ViT(nn.Module):
    """
    Vision Transformer (ViT-Tiny by default).

    Architecture:
        1. Patch Embedding: (B, 3, 224, 224) -> (B, 196, 192)
        2. Prepend [CLS] token + Positional Encoding
        3. N x TransformerBlock (Pre-LN, MHSA, MLP, DropPath)
        4. LayerNorm
        5. Classification Head: [CLS] -> Linear -> num_classes

    Args:
        num_classes: Output classes
        input_channels: Input channels (3 for RGB)
        img_size: Input image size (must be divisible by patch_size)
        patch_size: Patch size (16 recommended for small datasets)
        embed_dim: Embedding dimension
        depth: Number of Transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden ratio
        dropout: General dropout
        attn_dropout: Attention weights dropout
        drop_path_rate: Max stochastic depth rate (linearly scaled over depth)
        qkv_bias: Use bias in QKV projection
        head_dropout: Dropout in classification head
        use_cls_token: Use [CLS] token (True) or global avg pool (False)
        activation: MLP activation
        init_std: Init std for Linear/Conv weights
        verbose: Print summary on init
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        qkv_bias: bool = True,
        head_dropout: float = 0.2,
        use_cls_token: bool = True,
        activation: str = "gelu",
        init_std: float = 0.02,
        verbose: bool = False,
    ):
        super().__init__()

        # Store config
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.head_dropout = head_dropout
        self.use_cls_token = use_cls_token
        self.activation = activation
        self.init_std = init_std

        self._config = ViTConfig(
            num_classes=num_classes,
            input_channels=input_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            head_dropout=head_dropout,
            use_cls_token=use_cls_token,
            activation=activation,
        )

        # ========================================================
        # Patch Embedding
        # ========================================================
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=input_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # ========================================================
        # [CLS] Token + Positional Embedding
        # ========================================================
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim)
            )
        else:
            self.cls_token = None
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )

        # ========================================================
        # Dropout after positional embedding
        # ========================================================
        self.pos_drop = nn.Dropout(dropout)

        # ========================================================
        # Transformer Blocks (with linearly increasing drop_path)
        # ========================================================
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, depth)
        ]

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[i],
                    activation=activation,
                )
                for i in range(depth)
            ]
        )

        # ========================================================
        # Final Norm + Classification Head
        # ========================================================
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # ========================================================
        # Initialize weights
        # ========================================================
        self._init_weights()

        if verbose:
            print(self.summary())

    # ============================================================
    # Weight Initialization
    # ============================================================

    def _init_weights(self):
        """Initialize weights following ViT paper conventions."""
        # Linear and Conv: truncated normal (we use normal with small std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=self.init_std, a=-2 * self.init_std, b=2 * self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=self.init_std, a=-2 * self.init_std, b=2 * self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Positional embedding
        nn.init.trunc_normal_(
            self.pos_embed, std=0.02, a=-0.04, b=0.04
        )

        # CLS token
        if self.cls_token is not None:
            nn.init.trunc_normal_(
                self.cls_token, std=0.02, a=-0.04, b=0.04
            )

        # Classification head: small init
        if isinstance(self.head, nn.Sequential):
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    # ============================================================
    # Tokenization
    # ============================================================

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch tokens + [CLS]."""
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, C)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)
            return_attention: If True, return attention weights from all blocks

        Returns:
            logits: (B, num_classes)
            attentions (optional): List of (B, num_heads, N, N) per block
        """
        x = self._tokenize(x)

        all_attentions = [] if return_attention else None
        for block in self.blocks:
            x, attn = block(x, return_attention=return_attention)
            if return_attention:
                all_attentions.append(attn)

        x = self.norm(x)

        # Extract feature for classification
        if self.cls_token is not None:
            x = x[:, 0]  # [CLS] token
        else:
            x = x.mean(dim=1)  # global avg pool

        logits = self.head(x)

        if return_attention:
            return logits, all_attentions
        return logits, None

    # ============================================================
    # Feature Extraction (useful for interpretability)
    # ============================================================

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings after final norm."""
        x = self._tokenize(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        if self.cls_token is not None:
            return x[:, 0]
        return x.mean(dim=1)

    # ============================================================
    # Introspection
    # ============================================================

    def get_trainable_parameters(self) -> Dict[str, Any]:
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
        return self._config.to_dict()

    def summary(self) -> str:
        info = self.get_trainable_parameters()
        num_patches = self.patch_embed.num_patches
        lines = [
            "=" * 80,
            "Vision Transformer (ViT) Architecture Summary (Modernized)",
            "=" * 80,
            f"Input Shape:        (B, {self.input_channels}, {self.img_size}, {self.img_size})",
            f"Output Shape:       (B, {self.num_classes})",
            f"Patch Size:         {self.patch_size} x {self.patch_size}",
            f"Num Patches:        {num_patches}",
            f"Embed Dim:          {self.embed_dim}",
            f"Depth:              {self.depth}",
            f"Num Heads:          {self.num_heads} (head_dim={self.embed_dim // self.num_heads})",
            f"MLP Ratio:          {self.mlp_ratio}",
            f"Dropout:            {self.dropout}",
            f"Attn Dropout:       {self.attn_dropout}",
            f"Drop Path Rate:     {self.drop_path_rate}",
            f"Head Dropout:       {self.head_dropout}",
            f"Use CLS Token:      {self.use_cls_token}",
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
            f"ViT(num_classes={self.num_classes}, "
            f"embed_dim={self.embed_dim}, "
            f"depth={self.depth}, "
            f"heads={self.num_heads}, "
            f"patch_size={self.patch_size})"
        )


# ================================================================
# Factory Function
# ================================================================

def create_vit(
    num_classes: int = 4,
    variant: str = "tiny",
    pretrained: bool = False,
    verbose: bool = True,
    **kwargs,
) -> ViT:
    """
    Factory function to create a ViT model.

    Args:
        num_classes: Number of output classes
        variant: 'tiny' | 'small' | 'base' | 'custom'
            - tiny: embed_dim=192, depth=6, heads=3 (~5.7M params)
            - small: embed_dim=384, depth=12, heads=6 (~22M params)
            - base: embed_dim=768, depth=12, heads=12 (~86M params)
            - custom: uses kwargs
        pretrained: If True, warns (from-scratch implementation)
        verbose: Print model summary
        **kwargs: Passed to ViT constructor (overrides variant defaults)

    Returns:
        ViT model
    """
    if pretrained:
        print(
            "⚠️  Warning: pretrained=True is IGNORED. "
            "This is a from-scratch implementation (Task 5)."
        )

    # Variant presets
    variants = {
        "tiny": {"embed_dim": 192, "depth": 6, "num_heads": 3, "mlp_ratio": 4.0},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0},
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0},
        "custom": {},
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")

    # Merge: variant defaults + user kwargs (kwargs win)
    config = {**variants[variant], **kwargs}

    model = ViT(
        num_classes=num_classes,
        verbose=verbose,
        **config,
    )

    if verbose:
        print(model.summary())

    return model
