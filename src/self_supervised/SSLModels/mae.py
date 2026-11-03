# src/self_supervised/mae.py

"""
MAE for ResNet — Complete Ultimate 2026 Edition

Hybrid MAE designed specifically for ResNet/EfficientNet backbones.
The encoder is the backbone (pretrained), and MAE masks patches of
the FEATURE MAP (not raw pixels) and reconstructs them.

Key Features (2026 Edition):
- ✅ CUDA-ONLY (raises error if no GPU)
- ✅ Uses pretrained ResNet/EfficientNet encoder
- ✅ Modern torch.amp (no deprecated APIs)
- ✅ BF16 auto-detect on Ampere+
- ✅ Correct MAE loss on masked feature patches
- ✅ Lightweight decoder (optional reconstruction)
- ✅ NaN/Inf guard
- ✅ Best model tracking
- ✅ Full state checkpointing

Reference: He et al., 2022 "Masked Autoencoders Are Scalable Vision Learners"
Adapted for CNN backbones (ResNet/EfficientNet).

Author: Babak Yousefian
Version: 6.0.0 (2026 ULTIMATE EDITION)
"""

import copy
import gc
import json
import math
import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ================================================================
# ✅ MAE MONITOR
# ================================================================

class MAEMonitor:
    """
    ✅ Comprehensive Monitoring for MAE Training.

    Tracks 7 metrics for both reconstruction and target.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history = {
            "epoch": [], "loss": [], "mask_ratio": [],
            "recon_std": [], "recon_variance": [], "recon_covariance": [],
            "recon_correlation": [], "recon_coefficient": [],
            "recon_cosine_similarity": [], "recon_tanimoto_similarity": [],
            "target_std": [], "target_variance": [], "target_covariance": [],
            "target_correlation": [], "target_coefficient": [],
            "target_cosine_similarity": [], "target_tanimoto_similarity": [],
            "recon_target_cosine": [], "status": [],
        }
        self.collapse_thresholds = {
            "std_min": 0.05, "std_max": 50.0,
            "variance_min": 0.0025, "variance_max": 2500.0,
            "covariance_max": 20.0, "correlation_max": 0.85,
            "cosine_max": 0.98, "tanimoto_max": 0.85,
        }

    def monitor(self, reconstruction, target, mask, epoch, loss):
        """
        ✅ FIXED: Monitor with explicit memory cleanup.
        """
        # ✅ FIX: Force CPU + float16 to halve memory
        with torch.no_grad():
            recon_flat = reconstruction.detach().cpu().float().reshape(reconstruction.size(0), -1).numpy()
            target_flat = target.detach().cpu().float().reshape(target.size(0), -1).numpy()

        # ✅ FIX: Compute metrics (now safe with B×B fix)
        recon_metrics = self._calculate_metrics(recon_flat)
        target_metrics = self._calculate_metrics(target_flat)

        # ✅ Recon vs Target cosine (per-sample, then mean) — already B×B
        recon_norm = recon_flat / (np.linalg.norm(recon_flat, axis=1, keepdims=True) + 1e-8)
        target_norm = target_flat / (np.linalg.norm(target_flat, axis=1, keepdims=True) + 1e-8)
        recon_target_cosine = float(np.mean(np.sum(recon_norm * target_norm, axis=1)))

        mask_ratio = float(mask.float().mean().item()) if mask is not None else 0.0

        health = self._check_health(recon_metrics, target_metrics, recon_target_cosine, loss)

        # ✅ FIX: Free large arrays immediately
        del recon_flat, target_flat, recon_norm, target_norm
        import gc
        gc.collect()

        self.history["epoch"].append(epoch)
        self.history["loss"].append(loss)
        self.history["mask_ratio"].append(mask_ratio)
        for k in ["std", "variance", "covariance", "correlation", "coefficient", "cosine_similarity",
                  "tanimoto_similarity"]:
            self.history[f"recon_{k}"].append(recon_metrics[k])
            self.history[f"target_{k}"].append(target_metrics[k])
        self.history["recon_target_cosine"].append(recon_target_cosine)
        self.history["status"].append(health)

        if self.verbose:
            self._print_report(epoch, loss, recon_metrics, target_metrics, recon_target_cosine, mask_ratio, health)

        return {
            "recon": recon_metrics, "target": target_metrics,
            "recon_target_cosine": recon_target_cosine,
            "mask_ratio": mask_ratio, "health_status": health,
        }

    def _calculate_metrics(self, data):
        """
        ✅ FIXED: Computes metrics without huge memory footprint.

        Key fix:
        - For covariance/correlation, use sample-wise (B×B) instead of feature-wise (D×D)
        - D can be 25088, B is only 32, so B×B = ~1K vs D×D = ~600M
        """
        std = float(np.std(data))
        variance = float(np.var(data))

        # ✅ FIX: Sample-wise covariance (B×B, NOT D×D)
        B = data.shape[0]
        if B > 1:
            # Center each feature independently
            centered = data - data.mean(axis=0, keepdims=True)
            # Sample-wise covariance: (B, B)
            cov_matrix = np.cov(centered, rowvar=True)  # ✅ B×B نه D×D

            if cov_matrix.ndim == 0:
                covariance = 0.0
                correlation = 0.0
            else:
                off_diag = cov_matrix[~np.eye(cov_matrix.shape[0], dtype=bool)]
                covariance = float(np.mean(np.abs(off_diag))) if len(off_diag) > 0 else 0.0

                # Correlation from sample-wise cov
                stds = np.sqrt(np.diag(cov_matrix))
                stds[stds < 1e-8] = 1e-8
                corr_matrix = cov_matrix / np.outer(stds, stds)
                off_diag_corr = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
                correlation = float(np.mean(np.abs(off_diag_corr))) if len(off_diag_corr) > 0 else 0.0
        else:
            covariance = 0.0
            correlation = 0.0

        mean_val = float(np.mean(data))
        coefficient = std / (abs(mean_val) + 1e-8)

        # ✅ FIX: Cosine similarity via normalized dot product (no sklearn)
        # Instead of sklearn.metrics.pairwise.cosine_similarity (which uses D×D)
        data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-8)
        # Sample-wise similarity: (B, B)
        cos_matrix = np.dot(data_norm, data_norm.T)  # ✅ B×B
        off_diag_cos = cos_matrix[~np.eye(cos_matrix.shape[0], dtype=bool)]
        cosine_sim = float(np.mean(off_diag_cos)) if len(off_diag_cos) > 0 else 0.0

        # ✅ Tanimoto (using same normalized matrix)
        dot = cos_matrix  # Already (B, B)
        tanimoto = dot / (2 - dot + 1e-8)
        off_diag_tan = tanimoto[~np.eye(tanimoto.shape[0], dtype=bool)]
        tanimoto_sim = float(np.mean(off_diag_tan)) if len(off_diag_tan) > 0 else 0.0

        return {
            "std": std, "variance": variance,
            "covariance": covariance, "correlation": correlation,
            "coefficient": coefficient, "cosine_similarity": cosine_sim,
            "tanimoto_similarity": tanimoto_sim,
        }

    def _check_health(self, recon_metrics, target_metrics, recon_target_cosine, loss):
        warnings_list = []
        is_healthy = True
        status = "✅ HEALTHY"

        warnings_list.extend(self._check_single(recon_metrics, "recon"))
        warnings_list.extend(self._check_single(target_metrics, "target"))

        # ✅ MAE health: loss باید پایین باشه
        if loss > 2.5:
            warnings_list.append(f"⚠️ Loss too high ({loss:.4f})")
        elif loss > 1.3:
            warnings_list.append(f"⚠️ Loss moderate ({loss:.4f})")

        # ✅ حذف cosine check (معیار بدی برای MAE)
        # if recon_target_cosine < 0.3: ...

        if any(("COLLAPSE" in w) or ("🚨" in w) for w in warnings_list):
            status = "💀 COLLAPSE"
            is_healthy = False
        elif any("⚠️" in w for w in warnings_list):
            status = "⚠️ WARNING"

        if not warnings_list:
            warnings_list.append("✅ All metrics within healthy ranges!")

        return {"status": status, "is_healthy": is_healthy, "warnings": warnings_list}

    def _check_single(self, metrics, name):
        w = []
        t = self.collapse_thresholds
        if metrics["std"] < t["std_min"]:
            w.append(f"⚠️ {name} STD too low ({metrics['std']:.4f})")
        elif metrics["std"] > t["std_max"]:
            w.append(f"⚠️ {name} STD too high ({metrics['std']:.2f})")
        if metrics["variance"] < t["variance_min"]:
            w.append(f"⚠️ {name} Variance too low ({metrics['variance']:.6f})")
        if metrics["covariance"] > t["covariance_max"]:
            w.append(f"⚠️ {name} Covariance too high ({metrics['covariance']:.4f})")
        if metrics["correlation"] > t["correlation_max"]:
            w.append(f"⚠️ {name} Correlation too high ({metrics['correlation']:.4f})")
        if metrics["cosine_similarity"] > t["cosine_max"]:
            w.append(f"🚨 {name} Cosine too high ({metrics['cosine_similarity']:.4f})")
        if metrics["tanimoto_similarity"] > t["tanimoto_max"]:
            w.append(f"⚠️ {name} Tanimoto too high ({metrics['tanimoto_similarity']:.4f})")
        return w

    def _print_report(self, epoch, loss, recon_metrics, target_metrics, recon_target_cosine, mask_ratio, health):
        status = health["status"]
        is_healthy = health["is_healthy"]
        status_color = "🟢" if is_healthy else "🔴"
        print("\n" + "=" * 90)
        print(f"📊 MAE MONITORING REPORT - Epoch {epoch}")
        print("=" * 90)
        print(f"   📉 Loss: {loss:.6f}")
        print(f"   🎭 Mask Ratio: {mask_ratio:.4f}")
        print("-" * 90)
        print("\n🎯 TARGET (Original) METRICS:")
        print("-" * 60)
        self._print_table(target_metrics)
        print("\n🔮 RECONSTRUCTION METRICS:")
        print("-" * 60)
        self._print_table(recon_metrics)
        print("\n📊 RECON vs TARGET:")
        print("-" * 60)
        print(f"   Recon-Target Cosine: {recon_target_cosine:.6f}  {'✅' if recon_target_cosine > 0.3 else '⚠️'}")
        print(f"\n🏥 HEALTH STATUS: {status}")
        print("-" * 60)
        for w in health["warnings"]:
            print(f"   {w}")
        print("=" * 90 + "\n")

    def _print_table(self, m):
        t = self.collapse_thresholds
        print(f"   🟢 1. STD:                {m['std']:.6f}  {'✅' if t['std_min'] < m['std'] < t['std_max'] else '⚠️'}")
        print(
            f"   🟢 2. Variance:            {m['variance']:.6f}  {'✅' if t['variance_min'] < m['variance'] < t['variance_max'] else '⚠️'}")
        print(f"   🟢 3. Covariance:          {m['covariance']:.6f}")
        print(f"   🟢 4. Correlation:         {m['correlation']:.6f}")
        print(f"   🟢 5. Coefficient (CV):    {m['coefficient']:.6f}")
        print(
            f"   🟢 6. Cosine Similarity:   {m['cosine_similarity']:.6f}  {'✅' if m['cosine_similarity'] < t['cosine_max'] else '🚨'}")
        print(f"   🟢 7. Tanimoto Similarity: {m['tanimoto_similarity']:.6f}")

# ================================================================
# ✅ CUDA ENFORCEMENT
# ================================================================

def _require_cuda(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n" + "=" * 70 +
            "\n❌ CUDA IS REQUIRED FOR MAE 2026 ULTIMATE EDITION"
            "\n" + "=" * 70 +
            "\n   Install PyTorch with CUDA:"
            "\n   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            "\n" + "=" * 70
        )
    if device != "cuda":
        print(f"⚠️  Forcing device='cuda' (was '{device}')")
        device = "cuda"
    dev = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"✅ CUDA Available: {gpu_name} ({vram_gb:.2f} GB)")
    return dev


# ================================================================
# ✅ LEARNABLE POSITIONAL EMBEDDING
# ================================================================

class PositionalEmbedding(nn.Module):
    """Learnable positional embedding for patch sequences."""

    def __init__(self, num_positions: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


# ================================================================
# ✅ LIGHTWEIGHT TRANSFORMER BLOCK
# ================================================================

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GELU."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm1(x)
        a, _ = self.attn(x, x, x, need_weights=False)
        x = r + self.dropout(a)

        r = x
        x = self.norm2(x)
        x = r + self.mlp(x)
        return x


# ================================================================
# ✅ HYBRID MAE (ResNet + Transformer)
# ================================================================

class MaskedAutoencoder(nn.Module):
    """
    Hybrid MAE for ResNet/EfficientNet backbones.

    Pipeline:
      1. Encoder (ResNet) → feature map (B, C, H', W')
      2. Patchify feature map → (B, N, C)
      3. Linear projection to embed_dim
      4. Random masking (75%)
      5. Lightweight transformer on visible tokens
      6. Reconstruct masked tokens (regression on feature)
      7. Loss only on masked patches
    """

    def __init__(
            self,
            encoder: nn.Module,
            img_size: int = 224,
            patch_size: int = 7,
            mask_ratio: float = 0.75,
            embed_dim: int = 256,
            depth: int = 4,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # Feature dim from encoder
        self.feature_dim = self._get_feature_dim(encoder)

        # ✅ FIX: Dynamically determine feature_dim and feature_size by running dummy forward
        self.feature_dim = self._get_feature_dim(encoder)

        # ✅ FIX: Run a dummy forward pass to detect actual feature map shape
        try:
            dummy_input = torch.randn(1, 3, img_size, img_size)
            # Move to same device as encoder
            try:
                dummy_input = dummy_input.to(next(encoder.parameters()).device)
            except StopIteration:
                pass
            with torch.no_grad():
                dummy_feat = self._encode_features(dummy_input)

            if dummy_feat.dim() == 4:
                B_d, C_d, H_d, W_d = dummy_feat.shape
                self.feature_dim = C_d
                self.feature_size = H_d  # assume square
                print(f"✅ MAE: Detected feature map = ({C_d}, {H_d}, {W_d})")
            else:
                # Fallback
                self.feature_size = img_size // 32
                print(f"⚠️ MAE: Non-4D feature, using fallback feature_size={self.feature_size}")
        except Exception as e:
            print(f"⚠️ MAE: Could not detect feature map size: {e}")
            self.feature_size = img_size // 32

        if self.feature_size < 1:
            self.feature_size = 1

        # ✅ FIX: If patch_size > feature_size, force patch_size=1
        if patch_size > self.feature_size:
            print(f"⚠️ MAE: patch_size ({patch_size}) > feature_size ({self.feature_size}), forcing patch_size=1")
            patch_size = 1
        self.patch_size = patch_size

        self.num_patches = (self.feature_size // patch_size) ** 2
        if self.num_patches < 1:
            self.patch_size = 1
            self.num_patches = self.feature_size ** 2

        # ✅ FIX: Build patch projection deterministically
        # Patchify: (B, C, F, F) → (B, C, F/p, F/p) → (B, N, C)
        # For patch_size=1, no projection needed (identity)
        # For patch_size>1, use Conv2d
        #if patch_size > 1:
            #self.patch_conv = nn.Conv2d(
                #self.feature_dim, self.feature_dim,
                #kernel_size=patch_size, stride=patch_size,
            #)
        #else:
            #self.patch_conv = nn.Identity()

        # ✅ FIX: Linear projection from patch features to embed_dim (built once)
        patch_dim = self.feature_dim * (patch_size ** 2)
        self.patch_linear = nn.Linear(patch_dim, embed_dim)

        # Positional embedding for tokens
        self.pos_embed = PositionalEmbedding(self.num_patches, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_dim)  # ← to patch_dim, not feature_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def _get_feature_dim(self, model: nn.Module) -> int:
        if hasattr(model, "feature_dim"):
            return model.feature_dim
        if hasattr(model, "_feature_dim"):
            return model._feature_dim
        try:
            dummy = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
            with torch.no_grad():
                if hasattr(model, "get_embedding"):
                    feat = model.get_embedding(dummy)
                else:
                    feat = model(dummy)
                    if feat.dim() > 2:
                        feat = feat.view(feat.size(0), -1)
            return feat.size(-1)
        except Exception:
            return 512

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 4D feature map from encoder.

        Priority:
        1. forward_features (if returns 4D)
        2. features (if returns 4D)
        3. backbone (if returns 4D)
        4. EfficientNet-style (conv_stem + blocks + conv_head)  ← ✅ NEW
        5. ResNet-style (conv1 + bn1 + layer1-3)
        6. fallback to embedding
        """
        feat = None

        # ================================================================
        # ✅ Priority 1: forward_features
        # ================================================================
        if hasattr(self.encoder, "forward_features"):
            try:
                feat = self.encoder.forward_features(x)
                if feat.dim() != 4:
                    print(f"⚠️ forward_features returned {feat.dim()}D, trying next")
                    feat = None
            except Exception as e:
                print(f"⚠️ forward_features failed: {e}")
                feat = None

        # ================================================================
        # ✅ Priority 2: features (EfficientNet-style, older versions)
        # ================================================================
        if feat is None and hasattr(self.encoder, "features"):
            try:
                feat = self.encoder.features(x)
                if feat.dim() != 4:
                    print(f"⚠️ features returned {feat.dim()}D, trying next")
                    feat = None
            except Exception as e:
                print(f"⚠️ features failed: {e}")
                feat = None

        # ================================================================
        # ✅ Priority 3: backbone
        # ================================================================
        if feat is None and hasattr(self.encoder, "backbone"):
            try:
                feat = self.encoder.backbone(x)
                if feat.dim() != 4:
                    print(f"⚠️ backbone returned {feat.dim()}D, trying next")
                    feat = None
            except Exception as e:
                print(f"⚠️ backbone failed: {e}")
                feat = None

        # ================================================================
        # ✅ Priority 4: EfficientNet-style (conv_stem + bn1 + act + blocks + conv_head + bn2 + act)
        # ================================================================
        if (
                feat is None
                and hasattr(self.encoder, "conv_stem")
                and hasattr(self.encoder, "blocks")
                and hasattr(self.encoder, "conv_head")
        ):
            try:
                h = x
                # Stem
                if hasattr(self.encoder, "conv_stem"):
                    h = self.encoder.conv_stem(h)
                if hasattr(self.encoder, "bn1"):
                    h = self.encoder.bn1(h)
                if hasattr(self.encoder, "act"):
                    h = self.encoder.act(h)

                # ✅ FIX: Iterate over blocks — ModuleList has NO forward method!
                # Each block in EfficientNet is a Sequential (MBConv)
                for block in self.encoder.blocks:
                    h = block(h)

                # Head
                if hasattr(self.encoder, "conv_head"):
                    h = self.encoder.conv_head(h)
                if hasattr(self.encoder, "bn2"):
                    h = self.encoder.bn2(h)
                if hasattr(self.encoder, "act"):
                    h = self.encoder.act(h)

                # Now h should be (B, 1280, 7, 7) for B0
                feat = h
            except Exception as e:
                print(f"⚠️ EfficientNet-style failed: {e}")
                feat = None

        # ================================================================
        # ✅ Priority 5: ResNet-style (conv1 + bn1 + layer1-3)
        # ================================================================
        if (
                feat is None
                and hasattr(self.encoder, "conv1")
                and hasattr(self.encoder, "layer1")
        ):
            try:
                import torch.nn.functional as F
                h = self.encoder.conv1(x)
                h = self.encoder.bn1(h)
                h = F.relu(h, inplace=False)
                if hasattr(self.encoder, "maxpool"):
                    h = self.encoder.maxpool(h)
                h = self.encoder.layer1(h)
                h = self.encoder.layer2(h)
                h = self.encoder.layer3(h)  # ✅ layer3 → feature_dim=256 (R18), 1024 (R50)
                feat = h
            except Exception as e:
                print(f"⚠️ ResNet-style failed: {e}")
                feat = None

        # ================================================================
        # ✅ Priority 6: fallback — use embedding (LAST resort)
        # ================================================================
        if feat is None:
            print(f"⚠️ No 4D feature path found — using fallback (embedding)")
            if hasattr(self.encoder, "get_embedding"):
                emb = self.encoder.get_embedding(x)
            else:
                emb = self.encoder(x)
            if emb.dim() == 2:
                B, D = emb.shape
                F_size = int(math.sqrt(D)) or 1
                feat = emb[:, : F_size * F_size].view(B, F_size, F_size, 1).permute(0, 3, 1, 2)
            else:
                feat = emb

        # ================================================================
        # ✅ Final sanity check
        # ================================================================
        if feat.dim() != 4:
            raise RuntimeError(
                f"❌ _encode_features: expected 4D feature map, got {feat.dim()}D "
                f"shape={feat.shape}"
            )

        return feat

    def patchify(self, feat: torch.Tensor) -> torch.Tensor:
        """
        (B, C, H, W) → (B, N, patch_dim) where patch_dim = C * p * p
        """
        B, C, H, W = feat.shape
        p = self.patch_size

        # ✅ FIX: If p > H or p > W, force p = 1
        if p > H or p > W:
            print(f"⚠️ patchify: patch_size ({p}) > feature size ({H}x{W}), forcing p=1")
            p = 1

        H_crop = (H // p) * p
        W_crop = (W // p) * p
        feat = feat[:, :, :H_crop, :W_crop]

        if p > 1:
            # (B, C, H, W) → (B, C, H//p, p, W//p, p) → (B, H//p, W//p, p, p, C)
            feat = feat.reshape(B, C, H_crop // p, p, W_crop // p, p)
            feat = feat.permute(0, 2, 4, 3, 5, 1)  # (B, H//p, W//p, p, p, C)
            N = (H_crop // p) * (W_crop // p)
            feat = feat.reshape(B, N, p * p * C)
        else:
            # p=1: just flatten spatial
            feat = feat.permute(0, 2, 3, 1)  # (B, H, W, C)
            N = H_crop * W_crop
            feat = feat.reshape(B, N, C)

        return feat

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking on token dimension.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x: torch.Tensor, mask_ratio: float = None):
        """
        Returns:
            reconstruction: (B, N, patch_dim)
            target:         (B, N, patch_dim)
            mask:           (B, N)
        """
        # 1) Extract features
        feat = self._encode_features(x)  # (B, C, F, F)

        # 2) Patchify
        patches = self.patchify(feat)  # (B, N, patch_dim)

        target = patches

        # 3) Project to embed_dim
        x_tokens = self.patch_linear(patches)  # (B, N, D)

        # 4) Add positional embedding
        x_tokens = self.pos_embed(x_tokens)

        # 5) Random masking
        x_visible, mask, ids_restore = self.random_masking(x_tokens, mask_ratio)

        # 6) Transformer on visible tokens
        for blk in self.blocks:
            x_visible = blk(x_visible)
        x_visible = self.norm(x_visible)

        # 7) Reconstruct full sequence
        B, N_visible, D = x_visible.shape
        N_total = x_tokens.size(1)

        mask_tokens = self.mask_token.expand(B, N_total - N_visible, -1)
        x_full = torch.cat([x_visible, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )

        # 8) Predict
        x_full = self.decoder_norm(x_full)
        reconstruction = self.decoder_pred(x_full)  # (B, N, patch_dim)

        return reconstruction, target, mask

    def get_encoder(self) -> nn.Module:
        """Return the backbone encoder."""
        return self.encoder

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings (mean pooled) for downstream tasks."""
        if hasattr(self.encoder, "get_embedding"):
            return self.encoder.get_embedding(x)
        feat = self._encode_features(x)
        # Global average pool
        return feat.mean(dim=(2, 3))


# ================================================================
# ✅ MAE LOSS (masked only)
# ================================================================

class MAELoss(nn.Module):
    """
    MAE loss — MSE computed only on masked patches.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            reconstruction: (B, N, D)
            target:         (B, N, D)
            mask:           (B, N) — 1 for masked
        """
        loss_per_token = self.mse(reconstruction, target).mean(dim=-1)  # (B, N)
        loss_per_sample = (loss_per_token * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return loss_per_sample.mean()


# ================================================================
# ✅ MAE TRAINER — 2026 ULTIMATE
# ================================================================

class MAETrainer:
    """
    MAE Trainer for ResNet/EfficientNet backbones.

    CUDA-only, mixed precision (BF16 on Ampere+), best-model tracking.
    """

    def __init__(
        self,
        encoder: nn.Module,
        img_size: int = 224,
        patch_size: int = 7,
        mask_ratio: float = 0.75,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # ✅ NEW: accept aliases for compatibility
        encoder_embed_dim: int = None,
        encoder_depth: int = None,
        encoder_num_heads: int = None,
        decoder_embed_dim: int = None,  # ignored (decoder is linear now)
        decoder_depth: int = None,  # ignored
        decoder_num_heads: int = None,  # ignored
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        use_mixed_precision: bool = True,
        use_bf16: bool = True,
        use_multi_gpu: bool = False,
        gradient_accumulation_steps: int = 1,
        clip_grad_norm: float = 1.0,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        device: str = "cuda",
        verbose: bool = True,
        num_workers: int = 4,
        save_dir: str = None,
        **kwargs,
    ):
        # ✅ CUDA-ONLY
        self.device = _require_cuda(device)
        self.verbose = verbose

        # ✅ BF16 auto-detect
        if use_bf16 and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.amp_dtype_name = "bfloat16"
        else:
            self.amp_dtype = torch.float16
            self.amp_dtype_name = "float16"

        self.use_mixed_precision = use_mixed_precision
        self.use_bf16 = use_bf16 and torch.cuda.is_bf16_supported()

        # ✅ Auto-tune batch size based on VRAM
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb < 4.0 and batch_size > 32:
            print(f"⚠️  Reducing batch_size from {batch_size} to 32 (low VRAM)")
            batch_size = 32

        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_workers = num_workers
        self.use_multi_gpu = use_multi_gpu
        self.save_dir = save_dir or "ssl_checkpoints/mae"
        self.lr = lr
        self.weight_decay = weight_decay

        # ✅ FIX: Resolve aliases
        if encoder_embed_dim is not None:
            embed_dim = encoder_embed_dim
        if encoder_depth is not None:
            depth = encoder_depth
        if encoder_num_heads is not None:
            num_heads = encoder_num_heads

        # ✅ Build MAE
        self.mae = MaskedAutoencoder(
            encoder=encoder,
            img_size=img_size,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        ).to(self.device)

        # ✅ FIX: Unfreeze encoder with small LR
        self.mae.encoder.train()

        if use_multi_gpu and torch.cuda.device_count() > 1:
            self.mae = nn.DataParallel(self.mae)

        self.criterion = MAELoss()

        # ✅ Optimizer: two param groups (encoder + decoder)
        encoder_params = []
        decoder_params = []
        for name, param in self.mae.named_parameters():
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        self.optimizer = optim.AdamW([
            {"params": encoder_params, "lr": lr * 0.1},
            {"params": decoder_params, "lr": lr},
        ], weight_decay=weight_decay, betas=(0.9, 0.95))

        # ✅ NEW: Add monitor
        self.monitor = MAEMonitor(verbose=True)
        self._last_reconstruction = None
        self._last_target = None
        self._last_mask = None
        self._last_health_status = "unknown"

        # ✅ Scheduler
        self.scheduler = self._get_scheduler()

        # ✅ AMP scaler only for FP16
        if self.use_mixed_precision and self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.scaler = None

        # ✅ State
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float("inf")
        self.best_epoch = 0

        self.history = {
            "loss": [],
            "loss_epoch": [],
            "learning_rates": [],
            "epoch_times": [],
        }

        if self.verbose:
            self._print_init_info()

    def _get_scheduler(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # ✅ FIX: جلوگیری از T_max = 0
        cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)
        effective_warmup = min(self.warmup_epochs, max(0, self.total_epochs - 1))

        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6,
        )
        if effective_warmup > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=effective_warmup,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[effective_warmup],
            )
        return cosine

    def _print_init_info(self):
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print("\n" + "=" * 80)
        print("🚀 MAE Trainer — 2026 ULTIMATE EDITION")
        print("=" * 80)
        print(f"   Device:            {self.device} ({torch.cuda.get_device_name(0)})")
        print(f"   VRAM:              {vram_gb:.2f} GB")
        print(f"   Image Size:        {self.mae.img_size}")
        print(f"   Patch Size:        {self.mae.patch_size}")
        print(f"   Num Patches:       {self.mae.num_patches}")
        print(f"   Mask Ratio:        {self.mae.mask_ratio}")
        print(f"   Embed Dim:         {self.mae.embed_dim}")
        print(f"   Depth:             {self.mae.depth}")
        print(f"   Num Heads:         {self.mae.num_heads}")
        print(f"   Batch Size:        {self.batch_size}")
        print(f"   LR:                {self.lr}")
        print(f"   Weight Decay:      {self.weight_decay}")
        print(f"   Mixed Precision:   {self.amp_dtype_name if self.use_mixed_precision else 'disabled'}")
        print(f"   Grad Accum:        {self.gradient_accumulation_steps}")
        print(f"   Warmup Epochs:     {self.warmup_epochs}")
        print(f"   Total Epochs:      {self.total_epochs}")
        print("=" * 80 + "\n")

    def train_step(self, batch: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        use_amp = self.use_mixed_precision

        if use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                reconstruction, target, mask = self.mae(batch)
                loss = self.criterion(reconstruction, target, mask)
        else:
            reconstruction, target, mask = self.mae(batch)
            loss = self.criterion(reconstruction, target, mask)

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(
                f"NaN/Inf loss! recon.std={reconstruction.std().item():.4f}, "
                f"target.std={target.std().item():.4f}, loss={loss.item()}"
            )

        raw_loss = loss.item()
        scaled_loss = loss / self.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # ✅ Store for monitoring
        self._last_reconstruction = reconstruction.detach()
        self._last_target = target.detach()
        self._last_mask = mask.detach() if mask is not None else None

        self.current_step += 1
        if self.current_step % self.gradient_accumulation_steps == 0:
            if self.clip_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.mae.parameters(), self.clip_grad_norm
                )
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return raw_loss, {"loss": raw_loss}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.mae.train()
        self.mae.encoder.train()

        total_loss = 0.0
        total_steps = 0
        losses = []
        epoch_start = time.time()

        loop = tqdm(
            dataloader,
            desc=f"[MAE] Epoch {self.current_epoch + 1}/{self.total_epochs}",
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
            colour="#87CEEB",
            ncols=110,
            leave=False,
            disable=not self.verbose,
        )

        for images, _ in loop:
            images = images.to(self.device, non_blocking=True)
            if images.size(0) <= 1:
                continue

            loss, _ = self.train_step(images)

            # ✅ FIX: Debug ONLY at first batch of each epoch (not every 23 batches)
            debug_interval = int(os.environ.get("MAE_DEBUG_INTERVAL", "999999"))

            if total_steps == 0 and self._last_reconstruction is not None:
                self._debug_metrics(
                    loss=loss,
                    reconstruction=self._last_reconstruction,
                    target=self._last_target,
                    mask=self._last_mask,
                    step=total_steps,
                    epoch=self.current_epoch,
                )

                # ✅ Monitor every 5 epochs (only once per epoch)
                if (self.current_epoch + 1) % 5 == 0:
                    monitor_results = self.monitor.monitor(
                        reconstruction=self._last_reconstruction,
                        target=self._last_target,
                        mask=self._last_mask,
                        epoch=self.current_epoch + 1,
                        loss=loss,
                    )
                    self._last_health_status = monitor_results["health_status"]["status"]

                # ✅ FIX: Free intermediates after debug
                self._last_reconstruction = None
                self._last_target = None
                self._last_mask = None
                import gc as _gc
                _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            total_loss += loss
            total_steps += 1
            losses.append(loss)

            avg = total_loss / total_steps
            loop.set_postfix({
                "loss": f"{avg:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        if total_steps == 0:
            raise RuntimeError("No training steps executed")

        avg_loss = total_loss / total_steps
        epoch_time = time.time() - epoch_start

        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        self.history["loss"].extend(losses)
        self.history["loss_epoch"].append(avg_loss)
        self.history["learning_rates"].append(current_lr)
        self.history["epoch_times"].append(epoch_time)

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_epoch = self.current_epoch

        return {
            "loss": avg_loss,
            "best_loss": self.best_loss,
            "learning_rate": current_lr,
            "epoch_time": epoch_time,
        }

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = None,
        save_interval: int = 10,
        validation_loader: DataLoader = None,
    ) -> Dict[str, Any]:
        if epochs is not None:
            self.total_epochs = epochs

        os.makedirs(self.save_dir, exist_ok=True)

        if self.verbose:
            print(f"\n🚀 Starting MAE training for {self.total_epochs} epochs\n")

        start = time.time()

        for epoch in range(self.total_epochs):
            self.current_epoch = epoch
            metrics = self.train_epoch(dataloader)

            if self.verbose:
                print(
                    f"📊 Epoch {epoch+1}/{self.total_epochs} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Best: {metrics['best_loss']:.4f} | "
                    f"LR: {metrics['learning_rate']:.2e} | "
                    f"Time: {metrics['epoch_time']:.1f}s"
                )

            if validation_loader is not None:
                val = self.validate(validation_loader)
                if self.verbose:
                    print(f"   📊 Val Loss: {val['loss']:.4f}")

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.save_dir, f"mae_epoch_{epoch+1}.pth")
                )

            if metrics["loss"] == self.best_loss:
                self.save_checkpoint(
                    os.path.join(self.save_dir, "mae_best.pth")
                )

        self.save_checkpoint(os.path.join(self.save_dir, "mae_final.pth"))

        total_time = time.time() - start

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"✅ MAE training complete!")
            print(f"   Best Loss: {self.best_loss:.4f} (epoch {self.best_epoch + 1})")
            print(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f}m)")
            print("=" * 80 + "\n")

        return {
            "history": self.history,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
            "epochs": self.current_epoch + 1,
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.mae.eval()

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device, non_blocking=True)
                if images.size(0) <= 1:
                    continue

                use_amp = self.use_mixed_precision
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        rec, tgt, mask = self.mae(images)
                        loss = self.criterion(rec, tgt, mask)
                else:
                    rec, tgt, mask = self.mae(images)
                    loss = self.criterion(rec, tgt, mask)

                total_loss += loss.item()
                total_steps += 1

        return {"loss": total_loss / max(1, total_steps)}

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        mae = self.mae.module if hasattr(self.mae, "module") else self.mae

        torch.save(
            {
                "mae_state_dict": mae.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "history": self.history,
                "best_loss": self.best_loss,
                "best_epoch": self.best_epoch,
                "current_epoch": self.current_epoch,
                "current_step": self.current_step,
                "config": self.get_config(),
                "timestamp": datetime.now().isoformat(),
                "version": "6.0.0",
            },
            path,
        )

        if self.verbose:
            size_mb = os.path.getsize(path) / 1e6
            print(f"✅ Checkpoint saved: {path} ({size_mb:.2f} MB)")

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ckpt = torch.load(path, map_location=self.device)
        mae = self.mae.module if hasattr(self.mae, "module") else self.mae
        mae.load_state_dict(ckpt["mae_state_dict"])

        if load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.history = ckpt.get("history", self.history)
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.best_epoch = ckpt.get("best_epoch", 0)
        self.current_epoch = ckpt.get("current_epoch", 0)
        self.current_step = ckpt.get("current_step", 0)

        if self.verbose:
            print(f"✅ Checkpoint loaded: {path}")

    def get_encoder(self) -> nn.Module:
        """Return the trained ResNet encoder."""
        mae = self.mae.module if hasattr(self.mae, "module") else self.mae
        mae.encoder.eval()
        return mae.encoder

    def get_history(self) -> Dict[str, List]:
        return self.history

    def get_config(self) -> Dict[str, Any]:
        mae = self.mae.module if hasattr(self.mae, "module") else self.mae
        return {
            "img_size": mae.img_size,
            "patch_size": mae.patch_size,
            "mask_ratio": mae.mask_ratio,
            "embed_dim": mae.embed_dim,
            "depth": mae.depth,
            "num_heads": mae.num_heads,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "use_mixed_precision": self.use_mixed_precision,
            "amp_dtype": self.amp_dtype_name,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "clip_grad_norm": self.clip_grad_norm,
            "device": str(self.device),
            "save_dir": self.save_dir,
        }

    def __repr__(self) -> str:
        return f"MAETrainer(device=cuda, batch={self.batch_size}, mask={self.mae.mask_ratio})"

    def _debug_metrics(self, loss, reconstruction, target, mask, step=0, epoch=0):
        """
        ✅ FIXED: Debug metrics for MAE with correct health criteria.

        Key fixes:
        - `cos_sim_same` properly defined (was causing NameError)
        - MAE health uses MSE (not cosine) — reconstruction is MSE-based
        - Safe handling of small batch sizes (B <= 1)
        - Memory-efficient (no cdist, no D×D allocation)
        """
        with torch.no_grad():
            # ============================================================
            # ✅ Basic stats
            # ============================================================
            raw_loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

            B = reconstruction.size(0)
            recon_flat = reconstruction.detach().float().reshape(B, -1)
            target_flat = target.detach().float().reshape(B, -1)

            variance_recon = recon_flat.var(dim=0).mean().item()
            variance_target = target_flat.var(dim=0).mean().item()

            mean_recon = recon_flat.mean().item()
            std_recon = recon_flat.std().item()
            mean_target = target_flat.mean().item()
            std_target = target_flat.std().item()

            min_recon = recon_flat.min().item()
            max_recon = recon_flat.max().item()

            # ============================================================
            # ✅ ✅ PRIMARY METRIC: Reconstruction MSE (not cosine!)
            # ============================================================
            mse_same = F.mse_loss(recon_flat, target_flat).item()

            # ============================================================
            # ✅ ✅ PATCH-WISE COSINE (correct for MAE)
            # ============================================================
            # reshape to (B, N, C) → cosine per patch → mean
            feature_dim = 128  # AlexNet feature_dim
            try:
                if reconstruction.dim() == 3 and reconstruction.size(-1) == feature_dim:
                    recon_patches = reconstruction.detach().float()  # (B, N, C)
                    target_patches = target.detach().float()  # (B, N, C)
                    cos_sim_same = F.cosine_similarity(
                        recon_patches, target_patches, dim=-1
                    ).mean().item()
                else:
                    # Fallback: flatten cosine (shouldn't happen)
                    cos_sim_same = F.cosine_similarity(
                        recon_flat, target_flat, dim=1
                    ).mean().item()
            except Exception:
                cos_sim_same = 0.0

            # ============================================================
            # ✅ Random baseline (target vs shuffled target)
            # ============================================================
            if B > 1:
                idx = torch.randperm(B, device=target_flat.device)
                target_shuffled = target_flat[idx]
                cos_sim_random = F.cosine_similarity(
                    target_flat, target_shuffled, dim=1
                ).mean().item()
            else:
                cos_sim_random = 0.0

            # ============================================================
            # ✅ Pairwise distance (memory-safe)
            # ============================================================
            if B > 1:
                t_norm = target_flat / (target_flat.norm(dim=1, keepdim=True) + 1e-8)
                cos_matrix = torch.mm(t_norm, t_norm.t())  # (B, B)
                mask_diag = ~torch.eye(B, dtype=torch.bool, device=cos_matrix.device)
                pairwise_cos = cos_matrix[mask_diag]
                pairwise_distance = torch.sqrt(
                    torch.clamp(2 * (1 - pairwise_cos), min=1e-8)
                ).mean().item()
            else:
                pairwise_distance = 0.0

            # ============================================================
            # ✅ Derived metrics
            # ============================================================
            cos_diff = abs(cos_sim_same - cos_sim_random)
            mask_ratio = mask.float().mean().item() if mask is not None else 0.0

            # ============================================================
            # ✅ ✅ HEALTH CHECK (correct for MAE)
            # ============================================================
            # MAE health is based on:
            # - Reconstruction MSE (primary) — should be < 0.3
            # - Feature diversity (std, pairwise distance)
            # - Patch-wise cosine (should be > 0.3 after warmup)

            if epoch < 5:
                # Warmup: lenient thresholds
                is_healthy = (
                        0.05 < std_recon < 100.0 and
                        0.05 < std_target < 100.0 and
                        mse_same < 1.0
                )
            else:
                # Steady state: stricter thresholds
                is_healthy = (
                        0.1 < std_recon < 100.0 and
                        0.1 < std_target < 100.0 and
                        mse_same < 0.5 and
                        pairwise_distance > 0.05 and
                        cos_sim_same > 0.3
                )

            # ============================================================
            # ✅ Warnings
            # ============================================================
            warnings_list = []

            if std_recon < 0.1:
                warnings_list.append(f"💀 recon Std too low ({std_recon:.4f})")
            elif std_recon > 50.0:
                warnings_list.append(f"⚠️ recon Std too high ({std_recon:.2f})")

            if std_target < 0.1:
                warnings_list.append(f"💀 target Std too low ({std_target:.4f})")
            elif std_target > 50.0:
                warnings_list.append(f"⚠️ target Std too high ({std_target:.2f})")

            if mse_same > 0.5:
                warnings_list.append(f"⚠️ Reconstruction MSE too high ({mse_same:.4f})")
            elif mse_same > 0.3:
                warnings_list.append(f"⚠️ Reconstruction MSE moderate ({mse_same:.4f})")

            if epoch >= 5 and cos_sim_same < 0.3:
                warnings_list.append(f"⚠️ Patch-wise cosine too low ({cos_sim_same:.4f})")

            if B > 1 and pairwise_distance < 0.05:
                warnings_list.append(f"⚠️ Pairwise distance too low ({pairwise_distance:.4f})")

            # ============================================================
            # ✅ Health status string
            # ============================================================
            if any("💀" in w for w in warnings_list):
                status_emoji = "💀 UNHEALTHY"
                status_color = "🔴"
                is_healthy = False
            elif any("⚠️" in w for w in warnings_list):
                status_emoji = "⚠️ WARNING"
                status_color = "🟡"
            else:
                status_emoji = "✅ HEALTHY"
                status_color = "🟢"

        # ============================================================
        # ✅ Print report (outside no_grad to avoid tensor lifetime)
        # ============================================================
        warnings_str = (
            "None ✅"
            if not warnings_list
            else "\n".join("   • " + w for w in warnings_list)
        )

        print(f"""
        📊 ==================== MAE DEBUG METRICS ====================
        Step: {step}  |  Epoch: {epoch}
        ────────────────────────────────────────────────────────────
           📉 Loss:                       {raw_loss_val:.6f}
           🎯 Recon MSE:                  {mse_same:.6f}   {'✅' if mse_same < 0.3 else '⚠️'}
           🎭 Mask Ratio:                 {mask_ratio:.4f}
        ────────────────────────────────────────────────────────────
        🧠 FEATURE VARIANCE:
        recon Variance:               {variance_recon:.6f}  {'✅' if variance_recon > 0.01 else '💀'}
        target Variance:              {variance_target:.6f}  {'✅' if variance_target > 0.01 else '💀'}
        ────────────────────────────────────────────────────────────
        📊 EMBEDDING DIVERSITY:
        recon Mean:                   {mean_recon:.6f}
        recon Std:                    {std_recon:.6f}   {'✅' if std_recon > 0.1 else '⚠️'}
        target Mean:                  {mean_target:.6f}
        target Std:                   {std_target:.6f}   {'✅' if std_target > 0.1 else '⚠️'}
        ────────────────────────────────────────────────────────────
           🔄 PATCH-WISE COSINE:
        Same (recon, target):         {cos_sim_same:.4f}   {'✅' if cos_sim_same > 0.3 else '⚠️'}
        Random (target, target[idx]): {cos_sim_random:.4f}
        Difference:                   {cos_diff:.4f}
        ────────────────────────────────────────────────────────────
        📏 PAIRWISE DISTANCE:
        target Pairwise Distance:     {pairwise_distance:.4f}   {'✅' if pairwise_distance > 0.05 else '⚠️'}
        ────────────────────────────────────────────────────────────
        📏 Min/Max (recon):
        recon Min/Max:                {min_recon:.4f} / {max_recon:.4f}
        ────────────────────────────────────────────────────────────
        💡 WARNINGS:
        {warnings_str}
        ────────────────────────────────────────────────────────────
        {status_color} STATUS: {status_emoji}
        ============================================================
        """)

        # ✅ Free intermediate tensors
        try:
            del recon_flat, target_flat
        except Exception:
            pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "loss": raw_loss_val,
            "mse_same": mse_same,
            "variance": variance_recon,
            "variance_recon": variance_recon,
            "variance_target": variance_target,
            "mean_recon": mean_recon,
            "std_recon": std_recon,
            "mean_target": mean_target,
            "std_target": std_target,
            "cosine_similarity": cos_sim_same,
            "cosine_same": cos_sim_same,
            "cosine_random": cos_sim_random,
            "cosine_diff": cos_diff,
            "pairwise_distance": pairwise_distance,
            "min": min_recon,
            "max": max_recon,
            "mask_ratio": mask_ratio,
            "is_healthy": is_healthy,
            "warnings": warnings_list,
        }


# ================================================================
# ✅ FACTORY
# ================================================================

def create_mae_trainer(
    encoder: nn.Module,
    img_size: int = 224,
    patch_size: int = 7,
    mask_ratio: float = 0.75,
    embed_dim: int = 256,
    depth: int = 4,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    use_mixed_precision: bool = True,
    use_bf16: bool = True,
    use_multi_gpu: bool = False,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: float = 1.0,
    warmup_epochs: int = 10,
    total_epochs: int = 100,
    device: str = "cuda",
    verbose: bool = True,
    num_workers: int = 4,
    save_dir: str = None,
    **kwargs,
) -> MAETrainer:
    """Create MAE trainer (CUDA-only)."""
    return MAETrainer(
        encoder=encoder,
        img_size=img_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        use_mixed_precision=use_mixed_precision,
        use_bf16=use_bf16,
        use_multi_gpu=use_multi_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad_norm=clip_grad_norm,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        device=device,
        verbose=verbose,
        num_workers=num_workers,
        save_dir=save_dir,
        **kwargs,
    )


# ================================================================
# ✅ TEST
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing MAE — 2026 Ultimate Edition")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available.")
        raise SystemExit(1)

    # Dummy ResNet-like encoder
    class DummyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dim = 512
            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        def forward(self, x):
            x = self.stem(x)
            return self.layer4(x)

        def get_embedding(self, x):
            return self(x).mean(dim=(2, 3))

    encoder = DummyResNet()
    trainer = create_mae_trainer(
        encoder=encoder,
        img_size=224,
        patch_size=1,
        mask_ratio=0.75,
        embed_dim=256,
        depth=2,
        num_heads=4,
        batch_size=4,
        total_epochs=2,
        warmup_epochs=1,
        verbose=True,
        save_dir="test_checkpoints_mae",
    )

    # Quick forward
    dummy = torch.randn(4, 3, 224, 224, device="cuda")
    rec, tgt, mask = trainer.mae(dummy)
    print(f"✅ Forward OK — rec: {rec.shape}, tgt: {tgt.shape}, mask: {mask.shape}")

    # Quick train step
    loss, _ = trainer.train_step(dummy)
    print(f"✅ Train step OK — loss: {loss:.4f}")

    print("\n✅ All tests passed!")
    print("=" * 80)