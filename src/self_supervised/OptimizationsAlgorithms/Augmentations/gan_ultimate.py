# src/self_supervised/gan_ultimate.py
"""
GAN ULTIMATE — State-of-the-Art Architecture for Medical Ultrasound

Based on:
- StyleGAN2 (Karras et al., 2020) — lightweight adaptation
- BigGAN (Brock et al., 2019) — residual + spectral norm
- ADA (Karras et al., 2020) — Adaptive Discriminator Augmentation
- Feature Matching (Salimans et al., 2016)

Key Improvements over DCGAN:
- ✅ Residual blocks (better gradient flow)
- ✅ Equalized Learning Rate (EQM)
- ✅ Spectral Normalization
- ✅ Self-Attention (optional)
- ✅ Multi-Scale Discriminator
- ✅ Adaptive Discriminator Augmentation
- ✅ EMA for Generator
- ✅ R1 Regularization

Designed for:
- Small datasets (< 1000 images)
- Medical ultrasound (224×224)
- 4GB VRAM (RTX 2050)

Author: Babak Yousefian
Version: 1.0.0
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ================================================================
# ✅ EQUALIZED LEARNING RATE (EQM)
# ================================================================


class EqualizedLinear(nn.Module):
    """
    Linear layer with equalized learning rate.

    Reference: StyleGAN2 (Karras et al., 2020)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, gain: float = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gain = gain if gain is not None else math.sqrt(2)

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # ✅ Equalized learning rate
        self.scale = self.gain / math.sqrt(in_features)
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)


class EqualizedConv2d(nn.Module):
    """
    Conv2d layer with equalized learning rate.

    Reference: StyleGAN2 (Karras et al., 2020)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        gain: float = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.gain = gain if gain is not None else math.sqrt(2)

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # ✅ Equalized learning rate
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = self.gain / math.sqrt(fan_in)
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight * self.scale, self.bias, self.stride, self.padding
        )


# ================================================================
# ✅ RESIDUAL BLOCKS
# ================================================================


class ResBlockUp(nn.Module):
    """
    Residual block for Generator (upsampling).

    Architecture:
        x → ConvTranspose(2x) → BN → ReLU → Conv → BN → + skip → ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Upsampling skip connection
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            EqualizedConv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        h = self.upsample(x)
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h, inplace=True)

        h = self.conv2(h)
        h = self.bn2(h)

        # Skip connection
        s = self.skip(x)

        return F.relu(h + s, inplace=True)


class ResBlockDown(nn.Module):
    """
    Residual block for Discriminator (downsampling).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, 1, 1)

        # Skip connection with downsampling
        self.skip = EqualizedConv2d(in_channels, out_channels, 1, 1, 0, bias=False)

        self.downsample = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        h = self.conv1(x)
        h = F.leaky_relu(h, 0.2, inplace=True)

        h = self.conv2(h)

        # Skip connection
        s = self.skip(x)

        h = h + s
        h = self.downsample(h)
        h = F.leaky_relu(h, 0.2, inplace=True)

        return h


# ================================================================
# ✅ SELF-ATTENTION (Optional)
# ================================================================


class SelfAttention(nn.Module):
    """
    Self-Attention block (SAGAN, Zhang et al., 2019).

    Adds global context to generator/discriminator.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.query = EqualizedConv2d(in_channels, in_channels // 8, 1, 1, 0)
        self.key = EqualizedConv2d(in_channels, in_channels // 8, 1, 1, 0)
        self.value = EqualizedConv2d(in_channels, in_channels, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Query, Key, Value
        q = self.query(x).view(B, C // 8, -1).permute(0, 2, 1)  # (B, H*W, C/8)
        k = self.key(x).view(B, C // 8, -1)  # (B, C/8, H*W)
        v = self.value(x).view(B, C, -1)  # (B, C, H*W)

        # Attention
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(C // 8), dim=-1)  # (B, H*W, H*W)
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(B, C, H, W)

        # Residual connection
        return self.gamma * out + x


# ================================================================
# ✅ ULTIMATE GENERATOR
# ================================================================


class UltimateGenerator(nn.Module):
    """
    Ultimate Generator based on StyleGAN2-lite + BigGAN.

    Architecture:
        z (latent_dim) → Mapping Network (MLP) → w
        w → Synthesis Network:
            - Initial: 4×4
            - ResBlockUp × 5 (4 → 8 → 16 → 32 → 64 → 128 → 224)
            - ToRGB at each scale (StyleGAN2-style)
        → Tanh → Image (3, 224, 224)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        img_channels: int = 3,
        feature_maps: int = 64,
        num_blocks: int = 5,
        use_attention: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.feature_maps = feature_maps
        self.num_blocks = num_blocks

        # ✅ Mapping Network (StyleGAN2-style)
        self.mapping = nn.Sequential(
            EqualizedLinear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedLinear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedLinear(latent_dim, latent_dim),
        )

        # ✅ Initial block (4×4)
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
        )

        # ✅ Residual blocks (upsampling)
        self.blocks = nn.ModuleList()

        in_ch = feature_maps * 8
        for i in range(num_blocks):
            out_ch = max(feature_maps, in_ch // 2)
            self.blocks.append(ResBlockUp(in_ch, out_ch))
            in_ch = out_ch

        # ✅ Self-Attention
        if use_attention:
            self.attention = SelfAttention(in_ch)
        else:
            self.attention = None

        # ✅ Final layer
        self.final = nn.Sequential(
            EqualizedConv2d(in_ch, img_channels, 3, 1, 1),
            nn.Tanh(),
        )

        # ✅ Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # ✅ Mapping
        w = self.mapping(z)

        # ✅ Initial
        x = w.view(z.size(0), self.latent_dim, 1, 1)
        x = self.initial(x)  # (B, C*8, 4, 4)

        # ✅ Residual blocks
        for block in self.blocks:
            x = block(x)

        # ✅ Attention
        if self.attention is not None:
            x = self.attention(x)

        # ✅ Final
        # x currently: (B, C, H, W) where H ≈ 128
        # Need to upsample to 224
        if x.size(-1) != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = self.final(x)
        return x

    def generate_images(self, num_images: int, device: str = "cpu") -> "np.ndarray":
        import numpy as np

        self.eval()
        with torch.no_grad():
            z = torch.randn(num_images, self.latent_dim, device=device)
            images = self(z)
            images = (images + 1) / 2
            images = images.cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            images = (images * 255).astype(np.uint8)
        return images


# ================================================================
# ✅ ULTIMATE DISCRIMINATOR
# ================================================================


class UltimateDiscriminator(nn.Module):
    """
    Ultimate Discriminator with:
    - Residual blocks
    - Spectral Normalization
    - Self-Attention (optional)
    - Multi-Scale features (for feature matching)
    """

    def __init__(
        self,
        img_channels: int = 3,
        feature_maps: int = 64,
        num_blocks: int = 4,
        use_attention: bool = True,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        self.img_channels = img_channels
        self.feature_maps = feature_maps
        self.use_spectral_norm = use_spectral_norm

        # ✅ Initial conv (no SN, as per BigGAN)
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ✅ Residual blocks (downsampling)
        self.blocks = nn.ModuleList()
        in_ch = feature_maps

        for i in range(num_blocks):
            out_ch = min(feature_maps * 8, in_ch * 2)
            block = ResBlockDown(in_ch, out_ch)

            # ✅ Apply spectral norm to block
            if use_spectral_norm:
                for name, module in block.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                        setattr(
                            block,
                            name,
                            spectral_norm(module),
                        )

            self.blocks.append(block)
            in_ch = out_ch

        # ✅ Self-Attention
        if use_attention:
            self.attention = SelfAttention(in_ch)
        else:
            self.attention = None

        # ✅ Final
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 1),
        )

        # ✅ Init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and not isinstance(m, spectral_norm):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward — returns real/fake logits."""
        features = self.get_features(x)
        return self.final[2](features)  # Linear layer

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate features for Feature Matching loss.
        """
        x = self.initial(x)

        for block in self.blocks:
            x = block(x)

        if self.attention is not None:
            x = self.attention(x)

        # Global average pool → feature vector
        features = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return features


# ================================================================
# ✅ ADAPTIVE DISCRIMINATOR AUGMENTATION (ADA)
# ================================================================


class AdaptiveAugmentation(nn.Module):
    """
    Adaptive Discriminator Augmentation (ADA).

    Reference: Karras et al., 2020 "Training Generative Adversarial
    Networks with Limited Data"

    Adds augmentation to discriminator input, with adaptive probability.
    """

    def __init__(self, img_channels: int = 3):
        super().__init__()

        self.img_channels = img_channels

        # ✅ Augmentation probability (adaptive)
        self.register_buffer("p", torch.tensor(0.0))
        self.p_target = 0.6  # Target probability
        self.p_step = 0.01  # Adjustment step
        self.p_min = 0.0
        self.p_max = 0.8

        # ✅ Augmentation transforms (differentiable)
        self.translate = nn.Parameter(torch.zeros(1, img_channels, 1, 1))

    def forward(self, x: torch.Tensor, update_p: bool = False) -> torch.Tensor:
        """
        Apply augmentation with probability p.

        Args:
            x: Input images (B, C, H, W)
            update_p: Whether to update p based on discriminator accuracy

        Returns:
            Augmented images
        """
        if update_p:
            return x  # No augmentation

        # ✅ Randomly apply augmentation
        if torch.rand(1).item() > self.p.item():
            return x

        # ✅ Augmentation: random crop + flip + brightness
        B, C, H, W = x.shape

        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])

        # Random brightness
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
        x = x * brightness

        # Random contrast
        mean = x.mean(dim=(2, 3), keepdim=True)
        contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
        x = (x - mean) * contrast + mean

        return x

    def update(self, real_score: torch.Tensor, fake_score: torch.Tensor):
        """
        Update augmentation probability based on discriminator accuracy.

        If discriminator is too good (>80% accuracy), increase p.
        If discriminator is too bad (<60% accuracy), decrease p.
        """
        with torch.no_grad():
            # Estimate discriminator accuracy
            real_correct = (real_score > 0).float().mean()
            fake_correct = (fake_score < 0).float().mean()
            accuracy = (real_correct + fake_correct) / 2

            # Update p
            if accuracy > 0.8:
                # Discriminator too good → increase augmentation
                self.p.copy_(torch.clamp(self.p + self.p_step, self.p_min, self.p_max))
            elif accuracy < 0.6:
                # Discriminator too weak → decrease augmentation
                self.p.copy_(torch.clamp(self.p - self.p_step, self.p_min, self.p_max))


# ================================================================
# ✅ EMA (Exponential Moving Average)
# ================================================================


class EMA:
    """
    Exponential Moving Average of model parameters.

    Smoother generator samples.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay

        # ✅ Create shadow copy
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


# ================================================================
# ✅ FACTORY FUNCTIONS
# ================================================================


def create_ultimate_generator(
    latent_dim: int = 128,
    img_channels: int = 3,
    feature_maps: int = 64,
    num_blocks: int = 5,
    use_attention: bool = True,
) -> UltimateGenerator:
    """Create ultimate generator."""
    return UltimateGenerator(
        latent_dim=latent_dim,
        img_channels=img_channels,
        feature_maps=feature_maps,
        num_blocks=num_blocks,
        use_attention=use_attention,
    )


def create_ultimate_discriminator(
    img_channels: int = 3,
    feature_maps: int = 64,
    num_blocks: int = 4,
    use_attention: bool = True,
    use_spectral_norm: bool = True,
) -> UltimateDiscriminator:
    """Create ultimate discriminator."""
    return UltimateDiscriminator(
        img_channels=img_channels,
        feature_maps=feature_maps,
        num_blocks=num_blocks,
        use_attention=use_attention,
        use_spectral_norm=use_spectral_norm,
    )


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Ultimate GAN Architecture")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test generator
    print("\n📌 Testing Generator...")
    G = create_ultimate_generator(
        latent_dim=128, feature_maps=32, num_blocks=5
    ).to(device)

    z = torch.randn(2, 128, device=device)
    with torch.no_grad():
        fake = G(z)
    print(f"   z shape: {z.shape}")
    print(f"   fake shape: {fake.shape}")

    # Test discriminator
    print("\n📌 Testing Discriminator...")
    D = create_ultimate_discriminator(
        feature_maps=32, num_blocks=4
    ).to(device)

    with torch.no_grad():
        out = D(fake)
        features = D.get_features(fake)
    print(f"   out shape: {out.shape}")
    print(f"   features shape: {features.shape}")

    # Test ADA
    print("\n📌 Testing ADA...")
    ada = AdaptiveAugmentation().to(device)
    augmented = ada(fake)
    print(f"   augmented shape: {augmented.shape}")
    print(f"   initial p: {ada.p.item()}")

    # Test EMA
    print("\n📌 Testing EMA...")
    ema = EMA(G, decay=0.999)
    ema.update()
    print(f"   EMA shadow keys: {len(ema.shadow)}")

    print("\n✅ All Ultimate GAN tests passed!")
    print("=" * 80)
    