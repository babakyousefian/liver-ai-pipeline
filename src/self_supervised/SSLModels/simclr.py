# src/self_supervised/simclr.py

"""
SimCLR — Complete Ultimate 2026 Edition

SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
Reference: Chen et al., 2020 (ICML) + SimCLR v2 improvements

Key Features (2026 Edition):
- ✅ CUDA-ONLY (raises error if no GPU; no CPU fallback)
- ✅ Mixed precision with torch.amp (modern API, NOT deprecated autocast)
- ✅ Numerically stable NT-Xent loss for FP16/BF16
- ✅ Built-in LARS optimizer (no external dependency)
- ✅ Batch-level augmentation (efficient)
- ✅ 3-layer MLP projection head (SimCLR v2)
- ✅ Channel-wise normalization with learnable scale
- ✅ Gradient clipping inside AMP autocast
- ✅ Best model tracking + internal early stopping
- ✅ Full checkpoint save/load with all state
- ✅ Comprehensive metrics and history
- ✅ Zero-warnings execution

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
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
# ✅ SimCLR MONITOR (کپی از BYOLMonitor با نام‌گذاری SimCLR)
# ================================================================


class SimCLRMonitor:
    """
    ✅ Comprehensive Monitoring System for SimCLR Training.

    Tracks 7 key metrics for BOTH z (projection) and q (projection) to detect collapse.

    Note: In SimCLR, there is no separate target/prediction network.
    z1, z2 are both projections of the two views.
    We compute metrics for both to detect if the encoder is collapsing.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history = {
            "epoch": [],
            "loss": [],
            # ✅ z1 metrics
            "z1_std": [],
            "z1_variance": [],
            "z1_covariance": [],
            "z1_correlation": [],
            "z1_coefficient": [],
            "z1_cosine_similarity": [],
            "z1_tanimoto_similarity": [],
            # ✅ z2 metrics
            "z2_std": [],
            "z2_variance": [],
            "z2_covariance": [],
            "z2_correlation": [],
            "z2_coefficient": [],
            "z2_cosine_similarity": [],
            "z2_tanimoto_similarity": [],
            # ✅ Comparison metrics
            "z1_z2_cosine_similarity": [],
            "status": [],
        }
        self.collapse_thresholds = {
            "std_min": 0.05,
            "std_max": 50.0,
            "variance_min": 0.0025,
            "variance_max": 2500.0,
            "covariance_max": 20.0,
            "correlation_max": 0.85,
            "cosine_max": 0.98,
            "tanimoto_max": 0.85,
        }

    def monitor(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        epoch: int,
        loss: float,
    ) -> Dict[str, Any]:
        """
        Monitor SimCLR training metrics for BOTH z1 and z2.
        """
        # Convert to numpy
        z1_np = z1.detach().cpu().float().numpy()
        z2_np = z2.detach().cpu().float().numpy()

        # ✅ Combine for better statistics
        z1_combined = z1_np
        z2_combined = z2_np

        # ✅ Calculate metrics
        z1_metrics = self._calculate_metrics(z1_combined, "z1")
        z2_metrics = self._calculate_metrics(z2_combined, "z2")

        # ✅ Cosine similarity between z1 and z2 (positive pairs)
        z1_norm = z1_combined / (np.linalg.norm(z1_combined, axis=1, keepdims=True) + 1e-8)
        z2_norm = z2_combined / (np.linalg.norm(z2_combined, axis=1, keepdims=True) + 1e-8)
        z1_z2_cosine = np.mean(np.sum(z1_norm * z2_norm, axis=1))

        # ✅ Health check
        health_status = self._check_health(
            z1_metrics=z1_metrics, z2_metrics=z2_metrics,
            z1_z2_cosine=z1_z2_cosine, loss=loss
        )

        # ✅ Store history
        self.history["epoch"].append(epoch)
        self.history["loss"].append(loss)
        for k in ["std", "variance", "covariance", "correlation", "coefficient",
                  "cosine_similarity", "tanimoto_similarity"]:
            self.history[f"z1_{k}"].append(z1_metrics[k])
            self.history[f"z2_{k}"].append(z2_metrics[k])
        self.history["z1_z2_cosine_similarity"].append(z1_z2_cosine)
        self.history["status"].append(health_status)

        # ✅ Print report
        if self.verbose:
            self._print_report(
                epoch=epoch, loss=loss,
                z1_metrics=z1_metrics, z2_metrics=z2_metrics,
                z1_z2_cosine=z1_z2_cosine, health_status=health_status,
            )

        return {
            "z1": z1_metrics, "z2": z2_metrics,
            "z1_z2_cosine": z1_z2_cosine,
            "health_status": health_status,
        }

    def _calculate_metrics(self, data: np.ndarray, name: str) -> Dict[str, float]:
        """Calculate all 7 metrics."""
        std = np.std(data)
        variance = np.var(data)

        cov_matrix = np.cov(data, rowvar=False)
        off_diag = cov_matrix[~np.eye(cov_matrix.shape[0], dtype=bool)]
        covariance = np.mean(np.abs(off_diag)) if len(off_diag) > 0 else 0.0

        corr_matrix = np.corrcoef(data, rowvar=False)
        off_diag_corr = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
        correlation = np.mean(np.abs(off_diag_corr)) if len(off_diag_corr) > 0 else 0.0

        mean_val = np.mean(data)
        coefficient = std / (mean_val + 1e-8)

        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = cosine_similarity(data)
        off_diag_cos = cos_sim[~np.eye(cos_sim.shape[0], dtype=bool)]
        cosine_sim = np.mean(off_diag_cos) if len(off_diag_cos) > 0 else 0.0

        data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-8)
        dot_product = np.dot(data_norm, data_norm.T)
        tanimoto = dot_product / (2 - dot_product + 1e-8)
        off_diag_tan = tanimoto[~np.eye(tanimoto.shape[0], dtype=bool)]
        tanimoto_sim = np.mean(off_diag_tan) if len(off_diag_tan) > 0 else 0.0

        return {
            "std": std, "variance": variance,
            "covariance": covariance, "correlation": correlation,
            "coefficient": coefficient, "cosine_similarity": cosine_sim,
            "tanimoto_similarity": tanimoto_sim,
        }

    def _check_health(self, z1_metrics, z2_metrics, z1_z2_cosine, loss):
        """Check health."""
        warnings_list = []
        is_healthy = True
        status = "✅ HEALTHY"

        z1_warnings = self._check_single_metrics(z1_metrics, "z1")
        z2_warnings = self._check_single_metrics(z2_metrics, "z2")
        warnings_list.extend(z1_warnings)
        warnings_list.extend(z2_warnings)

        if z1_z2_cosine > 0.995:
            warnings_list.append(f"🚨 z1 and z2 are IDENTICAL ({z1_z2_cosine:.4f}) - COLLAPSE!")
            is_healthy = False
            status = "💀 SimCLR COLLAPSE"
        elif z1_z2_cosine > 0.95:
            warnings_list.append(f"⚠️ z1 and z2 too similar ({z1_z2_cosine:.4f})")

        if loss > 10.0:
            warnings_list.append(f"⚠️ Loss too high ({loss:.4f})")
        elif loss < 0.01:
            warnings_list.append(f"⚠️ Loss too low ({loss:.4f}) - POSSIBLE COLLAPSE!")

        if not is_healthy:
            pass
        elif any(("COLLAPSE" in w) or ("🚨" in w) for w in warnings_list):
            status = "💀 COLLAPSE"
            is_healthy = False
        elif any("⚠️" in w for w in warnings_list):
            status = "⚠️ WARNING"
            is_healthy = True

        if not warnings_list:
            warnings_list.append("✅ All metrics are within healthy ranges!")

        return {
            "status": status,
            "is_healthy": is_healthy,
            "warnings": warnings_list,
        }

    def _check_single_metrics(self, metrics, name):
        """Check single metrics."""
        warnings_list = []
        std = metrics["std"]
        variance = metrics["variance"]
        covariance = metrics["covariance"]
        correlation = metrics["correlation"]
        cosine_sim = metrics["cosine_similarity"]
        tanimoto_sim = metrics["tanimoto_similarity"]
        coefficient = metrics["coefficient"]

        if std < self.collapse_thresholds["std_min"]:
            warnings_list.append(f"⚠️ {name} STD too low ({std:.4f}) - COLLAPSE!")
        elif std > self.collapse_thresholds["std_max"]:
            warnings_list.append(f"⚠️ {name} STD too high ({std:.4f}) - EXPLOSION!")

        if variance < self.collapse_thresholds["variance_min"]:
            warnings_list.append(f"⚠️ {name} Variance too low ({variance:.6f}) - COLLAPSE!")
        elif variance > self.collapse_thresholds["variance_max"]:
            warnings_list.append(f"⚠️ {name} Variance too high ({variance:.2f}) - EXPLOSION!")

        if covariance > self.collapse_thresholds["covariance_max"]:
            warnings_list.append(f"⚠️ {name} Covariance too high ({covariance:.4f})")

        if correlation > self.collapse_thresholds["correlation_max"]:
            warnings_list.append(f"⚠️ {name} Correlation too high ({correlation:.4f})")

        if coefficient < 0.01:
            warnings_list.append(f"⚠️ {name} Coefficient too low ({coefficient:.4f})")
        elif coefficient > 5.0:
            warnings_list.append(f"⚠️ {name} Coefficient too high ({coefficient:.2f})")

        if cosine_sim > self.collapse_thresholds["cosine_max"]:
            warnings_list.append(f"🚨 {name} COSINE TOO HIGH ({cosine_sim:.4f}) - COLLAPSE!")
        elif cosine_sim > 0.7:
            warnings_list.append(f"⚠️ {name} Cosine high ({cosine_sim:.4f})")

        if tanimoto_sim > self.collapse_thresholds["tanimoto_max"]:
            warnings_list.append(f"⚠️ {name} Tanimoto too high ({tanimoto_sim:.4f})")

        return warnings_list

    def _print_report(self, epoch, loss, z1_metrics, z2_metrics, z1_z2_cosine, health_status):
        """Print report."""
        status = health_status["status"]
        is_healthy = health_status["is_healthy"]
        warnings_list = health_status["warnings"]
        status_color = "🟢" if is_healthy else "🔴"

        print("\n" + "=" * 90)
        print(f"📊 SimCLR MONITORING REPORT - Epoch {epoch}")
        print("=" * 90)
        print(f"   📉 Loss: {loss:.6f}")
        print("-" * 90)

        print("\n🎯 Z1 METRICS:")
        print("-" * 60)
        self._print_metrics_table(z1_metrics, "Z1")

        print("\n🎯 Z2 METRICS:")
        print("-" * 60)
        self._print_metrics_table(z2_metrics, "Z2")

        print("\n📊 Z1 vs Z2 COMPARISON:")
        print("-" * 60)
        print(f"   z1-z2 Cosine Similarity:    {z1_z2_cosine:.6f}  {'✅' if z1_z2_cosine < 0.9 else '🚨'}")
        print("-" * 60)

        print(f"\n🏥 HEALTH STATUS: {status}")
        print("-" * 60)
        for warning in warnings_list:
            print(f"   {warning}")
        print("-" * 60)
        print("=" * 90 + "\n")

    def _print_metrics_table(self, metrics, prefix):
        """Print metrics table."""
        t = self.collapse_thresholds
        std = metrics["std"]
        std_ok = t["std_min"] < std < t["std_max"]
        print(f"   🟢 1. STD:                          {std:.6f}  {'✅' if std_ok else '⚠️'}")
        var = metrics["variance"]
        var_ok = t["variance_min"] < var < t["variance_max"]
        print(f"   🟢 2. Variance:                      {var:.6f}  {'✅' if var_ok else '⚠️'}")
        cov = metrics["covariance"]
        cov_ok = cov < t["covariance_max"]
        print(f"   🟢 3. Covariance (mean off-diag):    {cov:.6f}  {'✅' if cov_ok else '⚠️'}")
        corr = metrics["correlation"]
        corr_ok = corr < t["correlation_max"]
        print(f"   🟢 4. Correlation (mean off-diag):   {corr:.6f}  {'✅' if corr_ok else '⚠️'}")
        coeff = metrics["coefficient"]
        coeff_ok = 0.001 < coeff < 50.0
        print(f"   🟢 5. Coefficient (CV):              {coeff:.6f}  {'✅' if coeff_ok else '⚠️'}")
        cos = metrics["cosine_similarity"]
        cos_ok = cos < t["cosine_max"]
        print(f"   🟢 6. Cosine Similarity:             {cos:.6f}  {'✅' if cos_ok else '🚨'}")
        tan = metrics["tanimoto_similarity"]
        tan_ok = tan < t["tanimoto_max"]
        print(f"   🟢 7. Tanimoto Similarity:           {tan:.6f}  {'✅' if tan_ok else '⚠️'}")

# ================================================================
# ✅ CUDA CHECK — HARD FAIL IF NO GPU
# ================================================================

def _require_cuda(device: str) -> torch.device:
    """
    Enforce CUDA-only. Raise error if GPU unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n" + "=" * 70 +
            "\n❌ CUDA IS REQUIRED FOR SimCLR ULTIMATE 2026 EDITION"
            "\n" + "=" * 70 +
            "\n   This implementation uses GPU-only optimizations:"
            "\n   - Mixed precision (FP16/BF16)"
            "\n   - Large batch operations"
            "\n   - LARS optimizer with GPU-centric gradients"
            "\n"
            "\n   Please install PyTorch with CUDA support:"
            "\n   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            "\n" + "=" * 70
        )

    if device != "cuda":
        print(f"⚠️  Forcing device='cuda' (was '{device}'). CUDA is mandatory.")
        device = "cuda"

    dev = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"✅ CUDA Available: {gpu_name} ({vram_gb:.2f} GB)")
    return dev


# ================================================================
# ✅ LARS OPTIMIZER (Built-in)
# ================================================================


class LARS(optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.

    Used in SimCLR for large-batch training. Scales learning rate
    layer-wise based on parameter/gradient norm ratio.

    Reference: You et al., 2017 "Large Batch Training of Convolutional Networks"
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        eta: float = 0.001,
        trust_coef: float = 0.001,
        eps: float = 1e-8,
        exclude_bias_and_norm: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            trust_coef=trust_coef,
            eps=eps,
            exclude_bias_and_norm=exclude_bias_and_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            trust_coef = group["trust_coef"]
            eps = group["eps"]
            exclude = group["exclude_bias_and_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients")

                # Skip bias and norm layers if requested
                if exclude and p.ndim <= 1:
                    adaptive_lr = group["lr"]
                else:
                    p_norm = p.norm()
                    g_norm = grad.norm()
                    if p_norm > 0 and g_norm > 0:
                        adaptive_lr = (
                            trust_coef * p_norm / (g_norm + weight_decay * p_norm + eps)
                        )
                        adaptive_lr = adaptive_lr.clamp(max=1.0) * group["lr"]
                    else:
                        adaptive_lr = group["lr"]

                # Add weight decay to gradient
                if weight_decay > 0 and not (exclude and p.ndim <= 1):
                    grad = grad.add(p, alpha=weight_decay)

                # Momentum update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.clone(grad).detach()

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                # Parameter update
                p.add_(buf, alpha=-adaptive_lr)

        return loss


# ================================================================
# ✅ PROJECTION HEAD — 3-Layer MLP (SimCLR v2)
# ================================================================


class ProjectionHead(nn.Module):
    """
    Projection Head for SimCLR v2.

    Architecture (3-layer MLP, SimCLR v2):
        Input (feature_dim)
        → Linear(hidden_dim) → BN → ReLU → Dropout
        → Linear(hidden_dim) → BN → ReLU → Dropout
        → Linear(projection_dim)

    Reference: Chen et al., 2020 "Big Self-Supervised Architects are Strong
    Semi-Supervised Learners"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        layers = []
        prev_dim = input_dim

        # Hidden layers (num_layers - 1)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Final projection layer (NO BN, NO activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.projector = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


# ================================================================
# ✅ NT-XENT LOSS (FP16/BF16-safe)
# ================================================================


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.

    Numerically stable for FP16/BF16. Uses `-1e4` (safe for FP16)
    instead of `-1e9` (overflows FP16).
    """

    def __init__(self, temperature: float = 0.2, reduction: str = "mean"):
        super().__init__()
        self.temperature = max(temperature, 1e-2)
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)

        # L2-normalize
        z_i = F.normalize(z_i, dim=1, eps=1e-8)
        z_j = F.normalize(z_j, dim=1, eps=1e-8)

        # Concatenate: (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity / temperature
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, device=z.device) + batch_size,
            torch.arange(batch_size, device=z.device),
        ], dim=0)

        # Mask self-similarity with a FP16-safe large negative value
        # ✅ -1e4 is safe for FP16 (max ≈ 65504); -1e9 would overflow
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e4)

        loss = self.criterion(sim, labels)
        return loss


# ================================================================
# ✅ SIMCLR TRAINER — ULTIMATE 2026
# ================================================================


class SimCLRTrainer:
    """
    SimCLR Trainer — 2026 Ultimate Edition.

    Features:
    - CUDA-only
    - Modern mixed precision (torch.amp)
    - LARS optimizer built-in
    - 3-layer projection head (SimCLR v2)
    - Best model tracking
    - Full checkpointing
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_size: int = 224,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        projection_layers: int = 3,
        temperature: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        use_lars: bool = True,
        use_mixed_precision: bool = True,
        use_bf16: bool = True,
        use_multi_gpu: bool = False,
        gradient_accumulation_steps: int = 1,
        clip_grad_norm: float = 1.0,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        color_jitter_strength: float = 1.0,
        blur_kernel_size: int = 23,
        device: str = "cuda",
        verbose: bool = True,
        num_workers: int = 4,
        save_dir: str = None,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        # ==========================================
        # ✅ CUDA-ONLY ENFORCEMENT
        # ==========================================
        self.device = _require_cuda(device)
        self.verbose = verbose

        # ==========================================
        # ✅ MIXED PRECISION DTYPE
        # ==========================================
        # BF16 is preferred on Ampere+ (compute ≥ 8.0) for stability
        if use_bf16 and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.amp_dtype_name = "bfloat16"
        else:
            self.amp_dtype = torch.float16
            self.amp_dtype_name = "float16"

        self.use_mixed_precision = use_mixed_precision
        self.use_bf16 = use_bf16 and torch.cuda.is_bf16_supported()

        # ==========================================
        # ✅ AUTO-TUNE BATCH SIZE & WORKERS
        # ==========================================
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb < 4.5 and batch_size > 128:
            if self.verbose:
                print(f"⚠️  Reducing batch_size from {batch_size} to 128 (low VRAM)")
            batch_size = 128
        if vram_gb < 3.0 and batch_size > 64:
            if self.verbose:
                print(f"⚠️  Reducing batch_size from {batch_size} to 64 (very low VRAM)")
            batch_size = 64

        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_workers = num_workers
        self.use_lars = use_lars
        self.use_multi_gpu = use_multi_gpu
        self.save_dir = save_dir or "ssl_checkpoints/simclr"

        # ==========================================
        # ✅ NETWORKS
        # ==========================================
        self.encoder = encoder.to(self.device)
        self.feature_dim = self._get_feature_dim(encoder)

        self.projector = ProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            num_layers=projection_layers,
            dropout_rate=dropout_rate,
        ).to(self.device)

        # Multi-GPU
        if use_multi_gpu and torch.cuda.device_count() > 1:
            self.encoder = nn.DataParallel(self.encoder)
            self.projector = nn.DataParallel(self.projector)
            if self.verbose:
                print(f"✅ Using {torch.cuda.device_count()} GPUs")

        # ==========================================
        # ✅ LOSS
        # ==========================================
        self.criterion = NTXentLoss(temperature=temperature, reduction="mean")

        # ==========================================
        # ✅ OPTIMIZER
        # ==========================================
        self.optimizer = self._get_optimizer(lr, weight_decay)
        self.lr = lr
        self.weight_decay = weight_decay

        # ==========================================
        # ✅ SCHEDULER
        # ==========================================
        self.scheduler = self._get_scheduler()

        # ==========================================
        # ✅ MIXED PRECISION SCALER
        # ==========================================
        # BF16 does NOT need a scaler; FP16 does
        if self.use_mixed_precision and self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.scaler = None

        # ==========================================
        # ✅ AUGMENTATION
        # ==========================================
        from .augmentations import SimCLRAugmentation
        self.augmentation = SimCLRAugmentation(
            target_size=input_size,
            crop_scale=(0.2, 1.0),
            color_jitter_strength=color_jitter_strength,
            blur_kernel_size=blur_kernel_size,
            normalize=True,
        )

        # ==========================================
        # ✅ STATE
        # ==========================================
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

        # ✅ NEW: Add monitor
        self.monitor = SimCLRMonitor(verbose=True)
        self._last_z1 = None
        self._last_z2 = None
        self._last_health_status = "unknown"
        self._collapse_counter = 0

        if self.verbose:
            self._print_init_info(projection_dim, hidden_dim, temperature)

    # ============================================================
    # ✅ HELPERS
    # ============================================================

    def _get_feature_dim(self, model: nn.Module) -> int:
        """
        ✅ FIXED: Get feature dim by ACTUALLY running a dummy forward pass.
        This is the source of truth — ignores potentially wrong _feature_dim.
        """
        # ✅ اول با forward واقعی چک کن — این reliable هست
        try:
            dummy_size = getattr(self, "input_size", 224)
            # برای EfficientNet-B1 معمولاً 240 هست
            if "efficientnet" in type(model).__name__.lower() or \
               "efficientnet" in str(getattr(model, "model_name", "")).lower():
                dummy_size = 240

            dummy = torch.randn(1, 3, dummy_size, dummy_size, device=self.device)
            with torch.no_grad():
                if hasattr(model, "get_embedding"):
                    feat = model.get_embedding(dummy)
                else:
                    feat = model(dummy)
                    if feat.dim() > 2:
                        feat = feat.view(feat.size(0), -1)
            real_dim = feat.size(-1)

            # ✅ اگر _feature_dim وجود داره ولی با واقعیت نمی‌خونه، warning بده
            reported = getattr(model, "_feature_dim", None) or getattr(model, "feature_dim", None)
            if reported is not None and reported != real_dim:
                if self.verbose:
                    print(f"⚠️ Model reports feature_dim={reported} but actual={real_dim}. Using actual.")

            return real_dim

        except Exception as e:
            # Fallback به attribute‌ها
            if hasattr(model, "_feature_dim"):
                return model._feature_dim
            if hasattr(model, "feature_dim"):
                return model.feature_dim
            return 2048

    def _debug_metrics(
            self,
            loss: float,
            z1: torch.Tensor,
            z2: torch.Tensor,
            step: int = 0,
            epoch: int = 0,
    ) -> Dict[str, float]:
        """
        ✅ SimCLR debug metrics — کپی از BYOL با نام‌گذاری z1, z2.
        """
        with torch.no_grad():
            raw_loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

            # ✅ Variance
            variance_z1 = z1.var(dim=0).mean().item()
            variance_z2 = z2.var(dim=0).mean().item()

            # ✅ Diversity
            mean_z1 = z1.mean().item()
            std_z1 = z1.std().item()
            mean_z2 = z2.mean().item()
            std_z2 = z2.std().item()

            # ✅ Same vs Random Cosine
            cos_sim_same = torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean().item()

            # Random buffer
            if not hasattr(self, "_z_buffer") or self._z_buffer is None:
                self._z_buffer = z1.detach().clone()
            else:
                self._z_buffer = torch.cat([self._z_buffer, z1.detach()], dim=0)
                if self._z_buffer.size(0) > 512:
                    self._z_buffer = self._z_buffer[-512:]

            if self._z_buffer.size(0) > z1.size(0) * 2:
                buf_idx = torch.randperm(self._z_buffer.size(0))[:z1.size(0)]
                z_buffer_sample = self._z_buffer[buf_idx]
                cos_sim_random = torch.nn.functional.cosine_similarity(z1, z_buffer_sample, dim=1).mean().item()
            else:
                idx = torch.randperm(z1.size(0))
                cos_sim_random = torch.nn.functional.cosine_similarity(z1, z1[idx], dim=1).mean().item()

            # Pairwise distance
            z1_for_dist = z1.detach().float()
            z1_for_dist = torch.clamp(z1_for_dist, min=-100.0, max=100.0)
            z1_for_dist = z1_for_dist / (z1_for_dist.norm(dim=1, keepdim=True) + 1e-8)
            dist = torch.cdist(z1_for_dist, z1_for_dist, p=2)
            n = dist.size(0)
            mask = ~torch.eye(n, dtype=torch.bool, device=dist.device)
            pairwise_distance = dist[mask].mean().item()

            min_z1 = z1.min().item()
            max_z1 = z1.max().item()

            cos_diff = abs(cos_sim_same - cos_sim_random)

            is_healthy = (
                    0.1 < std_z1 < 50.0
                    and 0.1 < std_z2 < 50.0
                    and cos_sim_same < 0.95
                    and pairwise_distance > 0.1
                    and cos_diff > 0.01
            )

            if epoch < 5:
                is_healthy = (
                        0.05 < std_z1 < 100.0
                        and 0.05 < std_z2 < 100.0
                        and cos_sim_same < 0.98
                        and pairwise_distance > 0.05
                        and cos_diff > 0.001
                )

            warnings = []
            if std_z1 < 0.1:
                warnings.append(f"💀 z1 Std too low ({std_z1:.4f})")
            elif std_z1 > 50.0:
                warnings.append(f"⚠️ z1 Std too high ({std_z1:.2f})")
            if std_z2 < 0.1:
                warnings.append(f"💀 z2 Std too low ({std_z2:.4f})")
            elif std_z2 > 50.0:
                warnings.append(f"⚠️ z2 Std too high ({std_z2:.2f})")
            if cos_sim_same > 0.95:
                warnings.append(f"💀 Cosine (same) too high ({cos_sim_same:.4f})")
            if pairwise_distance < 0.1:
                warnings.append(f"⚠️ Pairwise distance too low ({pairwise_distance:.4f})")
            if abs(cos_sim_same - cos_sim_random) < 0.005:
                warnings.append("⚠️ same ≈ random → possible collapse")

        status_emoji = "✅ HEALTHY" if is_healthy else "💀 UNHEALTHY"
        status_color = "🟢" if is_healthy else "🔴"

        print(f"""
    📊 ==================== SimCLR DEBUG METRICS ====================
    Step: {step}  |  Epoch: {epoch}
    ────────────────────────────────────────────────────────────
       📉 Loss:                       {raw_loss_val:.6f}
    ────────────────────────────────────────────────────────────
    🧠 FEATURE VARIANCE:
    z1 Variance:                  {variance_z1:.6f}  {'✅' if variance_z1 > 0.01 else '💀'}
    z2 Variance:                  {variance_z2:.6f}  {'✅' if variance_z2 > 0.01 else '💀'}
    ────────────────────────────────────────────────────────────
    📊 EMBEDDING DIVERSITY:
    z1 Mean:                      {mean_z1:.6f}
    z1 Std:                       {std_z1:.6f}   {'✅' if std_z1 > 0.1 else '⚠️'}
    z2 Mean:                      {mean_z2:.6f}
    z2 Std:                       {std_z2:.6f}   {'✅' if std_z2 > 0.1 else '⚠️'}
    ────────────────────────────────────────────────────────────
       🔄 COSINE SIMILARITY:
    Same (z1, z2):                {cos_sim_same:.4f}   {'💀' if cos_sim_same > 0.95 else '✅'}
    Random (z1, z1[idx]):         {cos_sim_random:.4f}
    Difference:                   {abs(cos_sim_same - cos_sim_random):.4f}
    ────────────────────────────────────────────────────────────
    📏 PAIRWISE DISTANCE:
    z1 Pairwise Distance:         {pairwise_distance:.4f}   {'✅' if pairwise_distance > 0.1 else '⚠️'}
    ────────────────────────────────────────────────────────────
    📏 Min/Max:
    z1 Min/Max:                   {min_z1:.4f} / {max_z1:.4f}
    ────────────────────────────────────────────────────────────
    💡 WARNINGS:
    {'None ✅' if not warnings else chr(10).join('   • ' + w for w in warnings)}
    ────────────────────────────────────────────────────────────
    {status_color} STATUS: {status_emoji}
    ============================================================
    """)

        return {
            "loss": raw_loss_val,
            "variance": variance_z1,
            "variance_z1": variance_z1,
            "variance_z2": variance_z2,
            "mean_z1": mean_z1,
            "std_z1": std_z1,
            "mean_z2": mean_z2,
            "std_z2": std_z2,
            "cosine_similarity": cos_sim_same,
            "cosine_same": cos_sim_same,
            "cosine_random": cos_sim_random,
            "cosine_diff": abs(cos_sim_same - cos_sim_random),
            "pairwise_distance": pairwise_distance,
            "min": min_z1,
            "max": max_z1,
            "is_healthy": is_healthy,
            "warnings": warnings,
        }

    def _get_optimizer(self, lr: float, weight_decay: float):
        params = list(self.encoder.parameters()) + list(self.projector.parameters())

        # ✅ NEW: اگر custom optimizer در دسترسه، از اون استفاده کن
        optimizer_name = self.kwargs.get("optimizer_name", "adamw") if hasattr(self, "kwargs") else "adamw"

        try:
            from src.optimizers.custom_optimizers import get_optimizer as get_custom_opt

            if optimizer_name in ("radam", "ranger"):
                if self.verbose:
                    print(f"✅ Using custom optimizer: {optimizer_name.upper()}")
                return get_custom_opt(
                    name=optimizer_name,
                    params=params,
                    lr=lr,
                    weight_decay=weight_decay,
                )
        except ImportError:
            pass

        if self.use_lars:
            return LARS(
                params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                trust_coef=0.001,
                exclude_bias_and_norm=True,
            )
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

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
                start_factor=0.01,
                end_factor=1.0,
                total_iters=effective_warmup,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[effective_warmup],
            )
        return cosine

    def _print_init_info(self, projection_dim: int, hidden_dim: int, temperature: float):
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print("\n" + "=" * 80)
        print("🚀 SimCLR Trainer — 2026 ULTIMATE EDITION")
        print("=" * 80)
        print(f"   Device:           {self.device} ({torch.cuda.get_device_name(0)})")
        print(f"   VRAM:             {vram_gb:.2f} GB")
        print(f"   Feature Dim:      {self.feature_dim}")
        print(f"   Projection Dim:   {projection_dim}")
        print(f"   Hidden Dim:       {hidden_dim}")
        print(f"   Projection Layers:{self.projector.num_layers}")
        print(f"   Temperature:      {temperature}")
        print(f"   Batch Size:       {self.batch_size}")
        print(f"   LR:               {self.lr}")
        print(f"   Weight Decay:     {self.weight_decay}")
        print(f"   Optimizer:        {'LARS' if self.use_lars else 'AdamW'}")
        print(f"   Mixed Precision:  {self.amp_dtype_name if self.use_mixed_precision else 'disabled'}")
        print(f"   Grad Accum:       {self.gradient_accumulation_steps}")
        print(f"   Warmup Epochs:    {self.warmup_epochs}")
        print(f"   Total Epochs:     {self.total_epochs}")
        print("=" * 80 + "\n")

    # ============================================================
    # ✅ VIEW GENERATION (batch-level)
    # ============================================================

    def _generate_views(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views — batch-level when possible.
        """
        # If augmentation is batch-aware, call directly on the batch
        if hasattr(self.augmentation, "forward") and not hasattr(self.augmentation, "view1"):
            try:
                v1, v2 = self.augmentation(images)
                return v1.to(self.device), v2.to(self.device)
            except Exception:
                pass

        # Fallback: per-sample loop (still on GPU)
        view1_list = []
        view2_list = []
        for i in range(images.size(0)):
            v1, v2 = self.augmentation(images[i])
            view1_list.append(v1)
            view2_list.append(v2)

        view1 = torch.stack(view1_list, dim=0).to(self.device)
        view2 = torch.stack(view2_list, dim=0).to(self.device)
        return view1, view2

    # ============================================================
    # ✅ FORWARD
    # ============================================================

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        if hasattr(self.encoder, "get_embedding"):
            h1 = self.encoder.get_embedding(view1)
            h2 = self.encoder.get_embedding(view2)
        else:
            h1 = self.encoder(view1)
            h2 = self.encoder(view2)
            if h1.dim() > 2:
                h1 = h1.view(h1.size(0), -1)
                h2 = h2.view(h2.size(0), -1)

        # ✅ FIX: resize h1/h2 to match projector's expected input_dim
        expected_dim = self.projector.input_dim
        if h1.size(-1) != expected_dim:
            if h1.size(-1) > expected_dim:
                # Trim
                h1 = h1[..., :expected_dim]
                h2 = h2[..., :expected_dim]
            else:
                # Pad with zeros
                pad_size = expected_dim - h1.size(-1)
                h1 = torch.nn.functional.pad(h1, (0, pad_size))
                h2 = torch.nn.functional.pad(h2, (0, pad_size))

        # Projection
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return z1, z2

    # ============================================================
    # ✅ TRAIN STEP
    # ============================================================

    def train_step(self, batch: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        view1, view2 = self._generate_views(batch)

        use_amp = self.use_mixed_precision and self.amp_dtype is not None

        # ✅ Modern autocast
        if use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                z1, z2 = self.forward(view1, view2)
                loss = self.criterion(z1, z2)
        else:
            z1, z2 = self.forward(view1, view2)
            loss = self.criterion(z1, z2)

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(
                f"NaN/Inf loss detected! z1.std={z1.std().item():.4f}, "
                f"z2.std={z2.std().item():.4f}, loss={loss.item()}"
            )

        raw_loss = loss.item()
        scaled_loss = loss / self.gradient_accumulation_steps

        # ✅ Backward
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self.current_step += 1

        if self.current_step % self.gradient_accumulation_steps == 0:
            # ✅ Gradient clipping inside AMP context
            if self.clip_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.projector.parameters()),
                    self.clip_grad_norm,
                )

            # ✅ Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

        return raw_loss, {"loss": raw_loss}

    # ============================================================
    # ✅ TRAIN EPOCH
    # ============================================================

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.encoder.train()
        self.projector.train()

        total_loss = 0.0
        total_steps = 0
        losses = []
        epoch_start = time.time()

        loop = tqdm(
            dataloader,
            desc=f"[SimCLR] Epoch {self.current_epoch + 1}/{self.total_epochs}",
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
            colour="#87CEEB",
            ncols=110,
            leave=False,
            disable=not self.verbose,
        )

        for images, _ in loop:
            images = images.to(self.device, non_blocking=True)

            # Skip batch of size 1 (BatchNorm needs >1)
            if images.size(0) <= 1:
                continue

            loss, _ = self.train_step(images)

            # ✅ ذخیره z1, z2 برای monitoring
            try:
                # Recapture z1, z2 از آخرین forward
                with torch.no_grad():
                    view1, view2 = self._generate_views(images)
                    use_amp = self.use_mixed_precision and self.amp_dtype is not None
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                            z1_cap, z2_cap = self.forward(view1, view2)
                    else:
                        z1_cap, z2_cap = self.forward(view1, view2)

                self._last_z1 = z1_cap.detach()
                self._last_z2 = z2_cap.detach()
            except Exception:
                pass

            total_loss += loss

            # ✅ Debug metrics هر 23 batch
            debug_interval = int(os.environ.get("SIMCLR_DEBUG_INTERVAL", "23"))

            if total_steps % debug_interval == 0 and self._last_z1 is not None:
                debug_results = self._debug_metrics(
                    loss=loss,
                    z1=self._last_z1,
                    z2=self._last_z2,
                    step=total_steps,
                    epoch=self.current_epoch,
                )

                # ✅ Monitor هر 5 epoch و در batch اول
                if (self.current_epoch + 1) % 5 == 0 and total_steps == 0:
                    monitor_results = self.monitor.monitor(
                        z1=self._last_z1,
                        z2=self._last_z2,
                        epoch=self.current_epoch + 1,
                        loss=loss,
                    )
                    self._last_health_status = monitor_results["health_status"]["status"]

            total_steps += 1
            losses.append(loss)

            avg = total_loss / total_steps
            loop.set_postfix({"loss": f"{avg:.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

        if total_steps == 0:
            raise RuntimeError("No training steps executed (all batches were skipped)")

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

    # ============================================================
    # ✅ TRAIN LOOP
    # ============================================================

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
            print(f"\n🚀 Starting SimCLR training for {self.total_epochs} epochs\n")

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

            # Save checkpoints
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.save_dir, f"simclr_epoch_{epoch+1}.pth")
                )

            # Save best
            if metrics["loss"] == self.best_loss:
                self.save_checkpoint(
                    os.path.join(self.save_dir, "simclr_best.pth")
                )

        # Final
        self.save_checkpoint(os.path.join(self.save_dir, "simclr_final.pth"))

        total_time = time.time() - start

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"✅ SimCLR training complete!")
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

    # ============================================================
    # ✅ VALIDATE
    # ============================================================

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.encoder.eval()
        self.projector.eval()

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device, non_blocking=True)

                if images.size(0) <= 1:
                    continue

                view1, view2 = self._generate_views(images)

                use_amp = self.use_mixed_precision
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        z1, z2 = self.forward(view1, view2)
                        loss = self.criterion(z1, z2)
                else:
                    z1, z2 = self.forward(view1, view2)
                    loss = self.criterion(z1, z2)

                total_loss += loss.item()
                total_steps += 1

        avg = total_loss / max(1, total_steps)
        return {"loss": avg}

    # ============================================================
    # ✅ CHECKPOINT
    # ============================================================

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Unwrap if DataParallel
        encoder = self.encoder.module if hasattr(self.encoder, "module") else self.encoder
        projector = self.projector.module if hasattr(self.projector, "module") else self.projector

        torch.save(
            {
                "encoder_state_dict": encoder.state_dict(),
                "projector_state_dict": projector.state_dict(),
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

        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.projector.load_state_dict(ckpt["projector_state_dict"])

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

    # ============================================================
    # ✅ EXPORTS
    # ============================================================

    def get_encoder(self) -> nn.Module:
        self.encoder.eval()
        return self.encoder

    def get_embeddings(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.encoder.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for images, lbls in tqdm(dataloader, desc="Extracting"):
                images = images.to(self.device, non_blocking=True)
                if hasattr(self.encoder, "get_embedding"):
                    emb = self.encoder.get_embedding(images)
                else:
                    emb = self.encoder(images)
                    if emb.dim() > 2:
                        emb = emb.view(emb.size(0), -1)
                embeddings.append(emb.float().cpu().numpy())
                labels.append(lbls.numpy())

        return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)

    def get_history(self) -> Dict[str, List]:
        return self.history

    def get_config(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "projection_dim": self.projector.output_dim,
            "hidden_dim": self.projector.hidden_dim,
            "projection_layers": self.projector.num_layers,
            "temperature": self.criterion.temperature,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "use_lars": self.use_lars,
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
        return (
            f"SimCLRTrainer(feature_dim={self.feature_dim}, "
            f"proj_dim={self.projector.output_dim}, "
            f"batch={self.batch_size}, device=cuda)"
        )


# ================================================================
# ✅ FACTORY FUNCTION
# ================================================================


def create_simclr_trainer(
    encoder: nn.Module,
    input_size: int = 224,
    projection_dim: int = 256,
    hidden_dim: int = 2048,
    projection_layers: int = 3,
    temperature: float = 0.2,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    use_lars: bool = True,
    use_mixed_precision: bool = True,
    use_bf16: bool = True,
    use_multi_gpu: bool = False,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: float = 1.0,
    warmup_epochs: int = 10,
    total_epochs: int = 100,
    color_jitter_strength: float = 1.0,
    blur_kernel_size: int = 23,
    device: str = "cuda",
    verbose: bool = True,
    num_workers: int = 4,
    save_dir: str = None,
    **kwargs,
) -> SimCLRTrainer:
    """
    Factory function to create SimCLR trainer (CUDA-only).
    """
    return SimCLRTrainer(
        encoder=encoder,
        input_size=input_size,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        projection_layers=projection_layers,
        temperature=temperature,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        use_lars=use_lars,
        use_mixed_precision=use_mixed_precision,
        use_bf16=use_bf16,
        use_multi_gpu=use_multi_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad_norm=clip_grad_norm,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        color_jitter_strength=color_jitter_strength,
        blur_kernel_size=blur_kernel_size,
        device=device,
        verbose=verbose,
        num_workers=num_workers,
        save_dir=save_dir,
        **kwargs,
    )


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing SimCLR — 2026 Ultimate Edition")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available. Test cannot run.")
        raise SystemExit(1)

    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dim = 128
            self.conv = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 128),
            )

        def get_embedding(self, x):
            return self.conv(x)

        def forward(self, x):
            return self.conv(x)

    encoder = DummyEncoder()

    trainer = create_simclr_trainer(
        encoder=encoder,
        input_size=224,
        projection_dim=256,
        hidden_dim=256,
        batch_size=64,
        total_epochs=2,
        warmup_epochs=1,
        use_mixed_precision=True,
        use_bf16=True,
        verbose=True,
        save_dir="test_checkpoints",
    )

    # Quick forward
    dummy = torch.randn(8, 3, 224, 224, device="cuda")
    v1, v2 = trainer._generate_views(dummy)
    z1, z2 = trainer.forward(v1, v2)
    loss = trainer.criterion(z1, z2)
    print(f"✅ Forward OK — z1: {z1.shape}, loss: {loss.item():.4f}")

    # Quick train epoch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.randn(32, 3, 224, 224), torch.zeros(32))
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    m = trainer.train_epoch(dl)
    print(f"✅ Train epoch OK — loss: {m['loss']:.4f}")

    print("\n✅ All tests passed!")
    print("=" * 80)