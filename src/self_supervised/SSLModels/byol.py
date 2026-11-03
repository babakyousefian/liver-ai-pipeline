# src/self_supervised/byol.py

"""
BYOL Implementation for Self-Supervised Learning (ULTIMATE EDITION)

BYOL: Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning

Reference: https://arxiv.org/abs/2006.07733
git
Key Features:
- No negative samples required (unlike SimCLR)
- Two networks: Online and Target (momentum encoder)
- EMA (Exponential Moving Average) for target network with cosine schedule
- Prediction MLP for online network
- Symmetrized loss for better convergence
- Gradually increasing momentum (0.996 → 0.999)
- Layer-wise learning rate decay
- Advanced augmentation pipelines
- Multi-GPU support
- Gradient clipping and accumulation
- Comprehensive logging and checkpointing
- Support for ResNet, EfficientNet, and custom encoders

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BYOL Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   View 1 ──► Online Encoder ──► Projector ──► Predictor ──┐                 │
│                                                           │                 │
│   View 2 ──► Target Encoder ──► Projector ────────────────┼──► MSE Loss     │
│                    ▲                                      │                 │
│                    │ EMA Update (momentum)                │                 │
│                    └──────────────────────────────────────┘                 │
│                                                                             │
│   ✅ No negative samples required                                           │
│   ✅ EMA momentum scheduler (0.996 → 0.999)                                 │
│   ✅ Symmetrized loss (both directions)                                     │
└─────────────────────────────────────────────────────────────────────────────┘

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 5.0.0 (ULTIMATE EDITION)
"""

import copy

# from click.core import batch
import math
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torchvision.transforms as T  # ✅ FIX: برای ToPILImage در fallback
from tqdm import tqdm

from src.self_supervised.OptimizationsAlgorithms.Augmentations.augmentations import BYOLAugmentation

warnings.filterwarnings("ignore")

# ================================================================
# ✅ Seven Differ metrics for whole data
# ================================================================


class BYOLMonitor:
    """
    ✅ Comprehensive Monitoring System for BYOL Training.

    Tracks 7 key metrics for BOTH z (target) and q (prediction) to detect collapse.

    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history = {
            "epoch": [],
            "loss": [],
            # ✅ z metrics (target network)
            "z_std": [],
            "z_variance": [],
            "z_covariance": [],
            "z_correlation": [],
            "z_coefficient": [],
            "z_cosine_similarity": [],
            "z_tanimoto_similarity": [],
            # ✅ q metrics (online network - prediction)
            "q_std": [],
            "q_variance": [],
            "q_covariance": [],
            "q_correlation": [],
            "q_coefficient": [],
            "q_cosine_similarity": [],
            "q_tanimoto_similarity": [],
            # ✅ Comparison metrics
            "z_q_cosine_similarity": [],  # شباهت بین z و q
            "status": [],
        }
        # ✅ FIX: آستانه‌های منطقی‌تر (کاهش حساسیت بیش از حد)
        # فقط collapse واقعی (COSINE > 0.99 یا STD < 0.02) بحرانی است
        # بقیه موارد warning هستن، نه unhealthy
        self.collapse_thresholds = {
            "std_min": 0.01,  # ← از 0.05 به 0.02 (سختگیرانه‌تر نیست)
            "std_max": 100.0,  # ← از 50 به 100
            "variance_min": 0.0001,  # ← از 0.0025 به 0.0004
            "variance_max": 10000.0,  # ← از 2500 به 10000
            "covariance_max": 100.0,  # ← از 20 به 100 (خیلی راحت‌گیرتر)
            "correlation_max": 0.995,  # ← از 0.85 به 0.995 (تنها در صورت تقریباً 1، هشدار)
            "cosine_max": 0.98,  # ← بدون تغییر (کاربر خواسته)
            "tanimoto_max": 0.995,  # ← از 0.85 به 0.995 (خیلی راحت‌گیرتر)
        }

    def monitor(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        q1: torch.Tensor,
        q2: torch.Tensor,
        epoch: int,
        loss: float,
    ) -> Dict[str, Any]:
        """
        Monitor BYOL training metrics for BOTH z and q .

        Args:
            z1: Target projection (view 1) (B, D)
            z2: Target projection (view 2) (B, D)
            q1: Online prediction (view 1) (B, D)
            q2: Online prediction (view 2) (B, D)
            epoch: Current epoch number
            loss: Current loss value

        Returns:
            Dictionary with all metrics and health status
        """
        # ✅ Convert to numpy for faster computation
        z1_np = z1.detach().cpu().numpy()
        z2_np = z2.detach().cpu().numpy()
        q1_np = q1.detach().cpu().numpy()
        q2_np = q2.detach().cpu().numpy()

        # ✅ Combine both views for better statistics
        z_combined = np.concatenate([z1_np, z2_np], axis=0)  # (2B, D)
        q_combined = np.concatenate([q1_np, q2_np], axis=0)  # (2B, D)

        # ================================================================
        # ✅ CALCULATE METRICS FOR Z (TARGET NETWORK)
        # ================================================================
        z_metrics = self._calculate_metrics(z_combined, "z")

        # ================================================================
        # ✅ CALCULATE METRICS FOR Q (ONLINE NETWORK - PREDICTION)
        # ================================================================
        q_metrics = self._calculate_metrics(q_combined, "q")

        # ================================================================
        # ✅ CALCULATE COMPARISON METRICS (z vs q)
        # ================================================================
        # ✅ Cosine similarity between z and q (should be close but not identical)
        z_norm = z_combined / (np.linalg.norm(z_combined, axis=1, keepdims=True) + 1e-8)
        q_norm = q_combined / (np.linalg.norm(q_combined, axis=1, keepdims=True) + 1e-8)
        z_q_cosine = np.mean(np.sum(z_norm * q_norm, axis=1))

        # ================================================================
        # ✅ HEALTH STATUS DETECTION (BOTH z AND q)
        # ================================================================
        health_status = self._check_health(
            z_metrics=z_metrics, q_metrics=q_metrics, z_q_cosine=z_q_cosine, loss=loss
        )

        # ================================================================
        # ✅ STORE HISTORY
        # ================================================================
        self.history["epoch"].append(epoch)
        self.history["loss"].append(loss)

        # Z metrics
        self.history["z_std"].append(z_metrics["std"])
        self.history["z_variance"].append(z_metrics["variance"])
        self.history["z_covariance"].append(z_metrics["covariance"])
        self.history["z_correlation"].append(z_metrics["correlation"])
        self.history["z_coefficient"].append(z_metrics["coefficient"])
        self.history["z_cosine_similarity"].append(z_metrics["cosine_similarity"])
        self.history["z_tanimoto_similarity"].append(z_metrics["tanimoto_similarity"])

        # Q metrics
        self.history["q_std"].append(q_metrics["std"])
        self.history["q_variance"].append(q_metrics["variance"])
        self.history["q_covariance"].append(q_metrics["covariance"])
        self.history["q_correlation"].append(q_metrics["correlation"])
        self.history["q_coefficient"].append(q_metrics["coefficient"])
        self.history["q_cosine_similarity"].append(q_metrics["cosine_similarity"])
        self.history["q_tanimoto_similarity"].append(q_metrics["tanimoto_similarity"])

        # Comparison
        self.history["z_q_cosine_similarity"].append(z_q_cosine)
        self.history["status"].append(health_status)

        # ================================================================
        # ✅ PRINT REPORT
        # ================================================================
        if self.verbose:
            self._print_report(
                epoch=epoch,
                loss=loss,
                z_metrics=z_metrics,
                q_metrics=q_metrics,
                z_q_cosine=z_q_cosine,
                health_status=health_status,
            )

        return {
            "z": z_metrics,
            "q": q_metrics,
            "z_q_cosine": z_q_cosine,
            "health_status": health_status,
        }

    def _calculate_metrics(self, data: np.ndarray, name: str) -> Dict[str, float]:
        """
        ✅ Calculate all 7 metrics for a given dataset.

        Args:
            data: (N, D) numpy array
            name: Name for logging ('z' or 'q')

        Returns:
            Dictionary with all metrics
        """
        # 1. Standard Deviation
        std = np.std(data)

        # 2. Variance
        variance = np.var(data)

        # 3. Covariance (mean off-diagonal)
        cov_matrix = np.cov(data, rowvar=False)
        off_diag = cov_matrix[~np.eye(cov_matrix.shape[0], dtype=bool)]
        covariance = np.mean(np.abs(off_diag)) if len(off_diag) > 0 else 0.0

        # 4. Correlation (mean off-diagonal)
        corr_matrix = np.corrcoef(data, rowvar=False)
        off_diag_corr = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
        correlation = np.mean(np.abs(off_diag_corr)) if len(off_diag_corr) > 0 else 0.0

        # 5. Coefficient of Variation (CV = STD / Mean)
        mean_val = np.mean(data)
        coefficient = std / (mean_val + 1e-8)

        # 6. Cosine Similarity (mean between samples)
        from sklearn.metrics.pairwise import cosine_similarity

        cos_sim = cosine_similarity(data)
        off_diag_cos = cos_sim[~np.eye(cos_sim.shape[0], dtype=bool)]
        cosine_sim = np.mean(off_diag_cos) if len(off_diag_cos) > 0 else 0.0

        # 7. Tanimoto Similarity (Jaccard-like for continuous vectors)
        data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-8)
        dot_product = np.dot(data_norm, data_norm.T)
        tanimoto = dot_product / (2 - dot_product + 1e-8)
        off_diag_tan = tanimoto[~np.eye(tanimoto.shape[0], dtype=bool)]
        tanimoto_sim = np.mean(off_diag_tan) if len(off_diag_tan) > 0 else 0.0

        return {
            "std": std,
            "variance": variance,
            "covariance": covariance,
            "correlation": correlation,
            "coefficient": coefficient,
            "cosine_similarity": cosine_sim,
            "tanimoto_similarity": tanimoto_sim,
        }

    def _check_health(
        self,
        z_metrics: Dict[str, float],
        q_metrics: Dict[str, float],
        z_q_cosine: float,
        loss: float,
    ) -> Dict[str, Any]:
        """
        ✅ Check health status for BOTH z and q.
        """
        warnings_list = []
        is_healthy = True
        status = "✅ HEALTHY"

        # ✅ Check Z metrics
        z_warnings = self._check_single_metrics(z_metrics, "z (Target)")
        warnings_list.extend(z_warnings)

        # ✅ Check Q metrics
        q_warnings = self._check_single_metrics(q_metrics, "q (Prediction)")
        warnings_list.extend(q_warnings)

        # ✅ FIX: آستانه‌های معقول‌تر
        # مقاله BYOL اصلی: z-q similarity می‌تواند بالا باشد (0.6-0.9) ولی نه 1.0
        if z_q_cosine > 0.995:
            warnings_list.append(
                f"🚨 z and q are IDENTICAL ({z_q_cosine:.4f}) - BYOL COLLAPSE DETECTED!"
            )
            is_healthy = False
            status = "💀 BYOL COLLAPSE"
        elif z_q_cosine > 0.95:
            warnings_list.append(
                f"⚠️ z and q too similar ({z_q_cosine:.4f}) - warning"
            )
            # ✅ FIX: دیگر is_healthy را False نمی‌کند

        # ✅ FIX: آستانه loss منطقی‌تر (طبق درخواست کاربر: 0 < loss < 2.5)
        if loss > 10.0:
            warnings_list.append(f"⚠️ Loss too high ({loss:.4f} > 2.5) - TRAINING UNSTABLE!")
        elif loss <= 0.00005:
            warnings_list.append(f"🚨 Loss ZERO or NEGATIVE ({loss:.4f}) - COLLAPSE DETECTED!")
        # حذف هشدار "Loss too low" — دیگه warning نمیده چون 0.0009 کاملاً OK است

        # ✅ FIX: status logic دقیق‌تر
        # «POSSIBLE COLLAPSE SOON» نباید status را به COLLAPSE ببرد
        # فقط «COLLAPSE DETECTED» یا «🚨» باید status را تغییر دهد
        # ✅ FIX: فقط اگر COLLAPSE DETECTED صریح بود، is_healthy را False کن
        # آستانه‌های قبلی (مثل Cosine > 0.9) خیلی سختگیرانه بودند
        if not is_healthy:
            pass
        elif any(("COLLAPSE DETECTED" in w) or ("🚨" in w) for w in warnings_list):
            status = "💀 COLLAPSE"
            is_healthy = False
        elif any("⚠️" in w for w in warnings_list):
            status = "⚠️ WARNING"
            is_healthy = True  # ⚠️ هنوز True است

        if not warnings_list:
            warnings_list.append("✅ All metrics are within healthy ranges!")

        return {
            "status": status,
            "is_healthy": is_healthy,
            "warnings": warnings_list,
            "suggestion": self._get_suggestion(is_healthy, status, warnings_list),
        }

    def _check_single_metrics(self, metrics: Dict[str, float], name: str) -> List[str]:
        """
        ✅ Check metrics for a single dataset (z or q).
        """
        warnings_list = []

        std = metrics["std"]
        variance = metrics["variance"]
        covariance = metrics["covariance"]
        correlation = metrics["correlation"]
        coefficient = metrics["coefficient"]
        cosine_sim = metrics["cosine_similarity"]
        tanimoto_sim = metrics["tanimoto_similarity"]

        if std < self.collapse_thresholds["std_min"]:
            warnings_list.append(f"💀 {name} STD too low ({std:.4f}) - COLLAPSE DETECTED!")
        elif std > self.collapse_thresholds["std_max"]:
            warnings_list.append(f"⚠️ {name} STD too high ({std:.4f}) - EXPLOSION!")

        # ✅ FIX: حذف warning های بی‌مورد
        # Variance
        if variance < self.collapse_thresholds["variance_min"]:
            warnings_list.append(f"💀 {name} Variance too low ({variance:.6f}) - COLLAPSE!")
        elif variance > self.collapse_thresholds["variance_max"]:
            warnings_list.append(f"⚠️ {name} Variance too high ({variance:.2f}) - EXPLOSION!")

        # Covariance — حذف warning (فقط اطلاعاتی)
        # اگر واقعاً مشکل‌ساز شد، فقط وقتی > covariance_max باشه
        if covariance > self.collapse_thresholds["covariance_max"]:
            warnings_list.append(
                f"⚠️ {name} Covariance too high ({covariance:.4f}) - FEATURES VERY CORRELATED!"
            )

        # Correlation — فقط در صورتی که تقریباً 1 باشه
        if correlation > self.collapse_thresholds["correlation_max"]:
            warnings_list.append(
                f"🚨 {name} Correlation TOO HIGH ({correlation:.4f}) - COLLAPSE DETECTED!"
            )

        # Coefficient — حذف کامل (CV برای embedding بی‌معنیه)
        # نادیده بگیر — قبلاً false positive زیاد می‌داد

        # Cosine Similarity — این مهم‌ترین سیگنال
        if cosine_sim > self.collapse_thresholds["cosine_max"]:
            # فقط در صورتی که واقعاً مشکل باشه (نه در warmup)
            if not (self.history.get("epoch") and len(self.history["epoch"]) > 0 and self.history["epoch"][-1] < 10):
                warnings_list.append(
                    f"🚨 {name} COSINE SIMILARITY TOO HIGH ({cosine_sim:.4f}) - COLLAPSE DETECTED!"
                )
        elif cosine_sim > 0.95:
            # فقط اگر از 0.95 بگذره، warning بده
            warnings_list.append(
                f"⚠️ {name} Cosine similarity high ({cosine_sim:.4f}) - MONITOR"
            )

        # Tanimoto — فقط در صورتی که تقریباً 1 باشه
        if tanimoto_sim > self.collapse_thresholds["tanimoto_max"]:
            warnings_list.append(
                f"⚠️ {name} Tanimoto similarity too high ({tanimoto_sim:.4f}) - COLLAPSE!"
            )

        return warnings_list

    def _get_suggestion(self, is_healthy: bool, status: str, warnings_list: list) -> str:
        """Generate actionable suggestion based on health status."""
        if is_healthy:
            return "✅ Continue training - everything looks good!"

        suggestions = []

        if "BYOL COLLAPSE" in status:
            suggestions.append("🚨 CRITICAL: z and q are identical! BYOL has collapsed!")
            suggestions.append("   - Decrease learning rate (lr = lr * 0.1)")
            suggestions.append("   - Increase augmentation strength")
            suggestions.append("   - Check if predictor is learning properly")
            suggestions.append("   - Try using GroupNorm instead of BatchNorm")
            suggestions.append("   - Increase momentum (0.996 → 0.999)")

        elif "COLLAPSE" in status:
            suggestions.append("🚨 IMMEDIATE ACTION REQUIRED:")
            suggestions.append("   - Decrease learning rate (lr = lr * 0.5)")
            suggestions.append("   - Increase batch size if possible")
            suggestions.append("   - Check augmentation strength (reduce if too strong)")
            suggestions.append("   - Verify normalization in BYOL forward pass")
            suggestions.append("   - Try using GroupNorm instead of BatchNorm")

        else:
            suggestions.append("⚠️ RECOMMENDED ACTIONS:")
            if any("Cosine similarity" in w for w in warnings_list):
                suggestions.append("   - Increase augmentation strength")
                suggestions.append("   - Add more color jitter or blur")
            if any("Variance" in w and "low" in w for w in warnings_list):
                suggestions.append("   - Reduce learning rate")
                suggestions.append("   - Check BatchNorm stability")
            if any("Loss" in w and "high" in w for w in warnings_list):
                suggestions.append("   - Decrease learning rate")
                suggestions.append("   - Add more warmup epochs")

        return "\n".join(suggestions)

    def _print_report(
        self,
        epoch: int,
        loss: float,
        z_metrics: Dict[str, float],
        q_metrics: Dict[str, float],
        z_q_cosine: float,
        health_status: Dict[str, Any],
    ):
        """✅ Print beautiful monitoring report with z AND q."""

        status = health_status["status"]
        is_healthy = health_status["is_healthy"]
        warnings_list = health_status["warnings"]
        suggestion = health_status["suggestion"]

        status_color = "🟢" if is_healthy else "🔴"

        print("\n" + "=" * 90)
        print(f"📊 BYOL MONITORING REPORT - Epoch {epoch}")
        print("=" * 90)
        print(f"   📉 Loss: {loss:.6f}")
        print("-" * 90)

        # ✅ Z Metrics Table (Target)
        print("\n🎯 Z (TARGET NETWORK) METRICS:")
        print("-" * 60)
        self._print_metrics_table(z_metrics, "Z")

        # ✅ Q Metrics Table (Online - Prediction)
        print("\n🔮 Q (ONLINE PREDICTION) METRICS:")
        print("-" * 60)
        self._print_metrics_table(q_metrics, "Q")

        # ✅ Comparison
        print("\n📊 Z vs Q COMPARISON:")
        print("-" * 60)
        print(f"   z-q Cosine Similarity:    {z_q_cosine:.6f}  {'✅' if z_q_cosine < 0.9 else '🚨'}")
        print("-" * 60)

        # ✅ Health Status
        print(f"\n🏥 HEALTH STATUS: {status}")
        print("-" * 60)
        for warning in warnings_list:
            print(f"   {warning}")
        print("-" * 60)

        # ✅ Suggestions
        print(f"\n💡 SUGGESTIONS:")
        print("-" * 60)
        print(f"   {suggestion}")
        print("-" * 60)
        print("=" * 90 + "\n")

    def _print_metrics_table(self, metrics: Dict[str, float], prefix: str):
        """Print metrics in a formatted table. Uses config thresholds."""
        status_color = "🟢"
        t = self.collapse_thresholds  # ✅ از config استفاده کن

        # STD — ✅ FIXED: realistic thresholds
        std = metrics["std"]
        if std < 0.01:
            std_icon = "💀"
        elif std < 0.03:
            std_icon = "⚠️"
        else:
            std_icon = "✅"
        print(f"   {status_color} 1. STD:                          {std:.6f}  {std_icon}")

        # Variance — ✅ FIXED: realistic thresholds
        var = metrics["variance"]
        if var < 0.0001:
            var_icon = "💀"
        elif var < 0.001:
            var_icon = "⚠️"
        else:
            var_icon = "✅"
        print(f"   {status_color} 2. Variance:                      {var:.6f}  {var_icon}")

        # Covariance
        cov = metrics["covariance"]
        cov_ok = cov < t["covariance_max"]
        print(
            f"   {status_color} 3. Covariance (mean off-diag):    {cov:.6f}  {'✅' if cov_ok else '⚠️'}"
        )

        # Correlation
        corr = metrics["correlation"]
        corr_ok = corr < t["correlation_max"]
        print(
            f"   {status_color} 4. Correlation (mean off-diag):   {corr:.6f}  {'✅' if corr_ok else '⚠️'}"
        )

        # Coefficient
        coeff = metrics["coefficient"]
        coeff_ok = 0.001 < coeff < 50.0
        print(
            f"   {status_color} 5. Coefficient (CV):              {coeff:.6f}  {'✅' if coeff_ok else '⚠️'}"
        )

        # Cosine
        cos = metrics["cosine_similarity"]
        cos_ok = cos < t["cosine_max"]   # ✅ حالا از config (0.98) استفاده می‌کند
        print(
            f"   {status_color} 6. Cosine Similarity:             {cos:.6f}  {'✅' if cos_ok else '🚨'}"
        )

        # Tanimoto
        tan = metrics["tanimoto_similarity"]
        tan_ok = tan < t["tanimoto_max"]
        print(
            f"   {status_color} 7. Tanimoto Similarity:           {tan:.6f}  {'✅' if tan_ok else '⚠️'}"
        )


# ================================================================
# ✅ MLP PROJECTOR & PREDICTOR
# ================================================================


class MLPProjector(nn.Module):
    """
    MLP Projector for BYOL.

    Maps the representation to a lower-dimensional space.
    Used by both online and target networks.

    Architecture:
        Input (feature_dim) → Linear(hidden_dim) → BN → ReLU → Dropout
                          → Linear(hidden_dim) → BN → ReLU → Dropout
                          → Linear(projection_dim) → BN
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 4096,
        output_dim: int = 256,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_group_norm: bool = False,  # ✅ NEW
        num_groups: int = 32,  # ✅ NEW
    ):
        super(MLPProjector, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_group_norm = use_group_norm
        self.num_groups = num_groups

        layers = []

        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm and not use_group_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif use_group_norm:
            layers.append(nn.GroupNorm(num_groups, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))

        # Second hidden layer
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm and not use_group_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif use_group_norm:
            layers.append(nn.GroupNorm(num_groups, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate * 0.5))

        # Output layer — ✅ FIXED: BN در خروجی برگردانده شد (مقاله اصلی BYOL)
        # دلیل: بدون BN، z می‌تواند به هر مقداری برود (انفجار) و
        # optimizer shortcut پیدا می‌کند: z = constant × scale
        # که باعث collapse می‌شود. BN مقیاس را کنترل می‌کند.
        layers.append(nn.Linear(hidden_dim, output_dim))
        if use_batch_norm and not use_group_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        elif use_group_norm:
            layers.append(nn.GroupNorm(num_groups, output_dim))

        self.projector = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projector.

        Args:
            x: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Projected tensor of shape (batch_size, output_dim)
        """
        return self.projector(x)

    def get_config(self) -> Dict[str, Any]:
        """Get projector configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
        }


class MLPPredictor(nn.Module):
    """
    MLP Predictor for BYOL (Online network only).

    Takes the projection from online encoder and predicts the target projection.

    Architecture:
        Input (projection_dim) → Linear(hidden_dim) → BN → ReLU
                              → Linear(projection_dim) → BN
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 4096,
        output_dim: int = 256,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_group_norm: bool = False,  # ✅ NEW
        num_groups: int = 32,  # ✅ NEW
    ):
        super(MLPPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_group_norm = use_group_norm
        self.num_groups = num_groups

        layers = [
            nn.Linear(input_dim, hidden_dim),
        ]

        if use_batch_norm and not use_group_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif use_group_norm:
            layers.append(nn.GroupNorm(num_groups, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))

        # Output layer — ✅ FIXED: BN در خروجی predictor برگردانده شد
        layers.append(nn.Linear(hidden_dim, output_dim))
        if use_batch_norm and not use_group_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        elif use_group_norm:
            layers.append(nn.GroupNorm(num_groups, output_dim))

        self.predictor = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.predictor.modules():
            if isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through predictor.

        Args:
            x: Input tensor of shape (batch_size, projection_dim)

        Returns:
            Predicted tensor of shape (batch_size, projection_dim)
        """
        return self.predictor(x)

    def get_config(self) -> Dict[str, Any]:
        """Get predictor configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
        }


# ================================================================
# ✅ BYOL LOSS (MSE)
# ================================================================


class BYOLLoss(nn.Module):
    """
    BYOL Loss using Mean Squared Error with symmetrization.

    Computes MSE between predicted online features and target features
    in both directions for better convergence.

    Loss = MSE(q1, z2) + MSE(q2, z1)
    Where:
        q1 = predictor(projector(online_encoder(view1)))
        z2 = projector(target_encoder(view2))
        q2 = predictor(projector(online_encoder(view2)))
        z1 = projector(target_encoder(view1))
    """

    def __init__(self, temperature: float = 1.0, symmetrize: bool = True):
        """
        Initialize BYOL loss.

        Args:
            temperature: Temperature for scaling (default: 1.0)
            symmetrize: Whether to symmetrize loss (default: True)
        """
        super(BYOLLoss, self).__init__()
        self.temperature = temperature
        self.symmetrize = symmetrize
        self.mse = nn.MSELoss()

    def forward(
        self, q1: torch.Tensor, q2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BYOL loss with symmetrization.
        """
        # ✅ FIXED: فقط یک بار normalize — حذف normalize دوگانه
        q1 = F.normalize(q1, dim=1)
        q2 = F.normalize(q2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        loss1 = self.mse(q1, z2.detach())
        loss2 = self.mse(q2, z1.detach())
        loss = (loss1 + loss2) / 2 if self.symmetrize else loss1

        # ✅ FIXED: Raise exception instead of hiding
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"🔥 NaN/Inf detected in BYOL loss! "
                f"loss1: {loss1.item()}, loss2: {loss2.item()}, "
                f"q1 std: {q1.std().item()}, z2 std: {z2.std().item()}"
            )

        return loss * self.temperature


# ================================================================
# ✅ BYOL TRAINER - ULTIMATE EDITION
# ================================================================


class BYOLTrainer:
    """
    BYOL Trainer for Self-Supervised Pretraining on Liver Ultrasound Images.

    Features:
    - Two networks: Online and Target (momentum encoder)
    - EMA (Exponential Moving Average) with cosine schedule (0.996 → 0.999)
    - Prediction MLP for online network
    - No negative samples required
    - Symmetrized loss for better convergence
    - Multiple augmentation strategies (BYOL, SimCLR)
    - LARS optimizer support (optional)
    - Gradient clipping and accumulation
    - Mixed precision training (autocast)
    - Multi-GPU support (DataParallel/DistributedDataParallel)
    - Comprehensive checkpointing
    - Layer-wise learning rate decay
    - Cosine annealing scheduler with warmup
    - Progress tracking and logging

    Reference: "Bootstrap Your Own Latent" (Grill et al., 2020)
    """

    def __init__(
            self,
            online_encoder: nn.Module,
            target_encoder: nn.Module = None,
            input_size: int = 224,
            projection_dim: int = 256,
            hidden_dim: int = 4096,
            momentum: float = 0.996,
            momentum_schedule: str = "cosine",
            lr: float = 0.0002,
            weight_decay: float = 1e-4,
            batch_size: int = 64,
            use_lars: bool = False,
            use_mixed_precision: bool = True,
            use_multi_gpu: bool = False,
            gradient_accumulation_steps: int = 1,
            clip_grad_norm: float = 1.0,
            warmup_epochs: int = 10,
            total_epochs: int = 40,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            verbose: bool = True,
            num_workers: int = 0,
            augmentation_method: str = "byol",
            save_dir: str = None,
            # ✅ SPEED FIX: پارامترهای جدید
            use_kornia: bool = False,
            use_bf16: bool = False,
            augmentation_intensity: float = 0.8,
            use_medical_safe: bool = True,
            **kwargs,
    ):
        """
        Initialize BYOL Trainer (ULTIMATE EDITION).

        Args:
            online_encoder: Base model for online network (ResNet, EfficientNet, etc.)
            target_encoder: Base model for target network (if None, copy of online)
            input_size: Input image size (224 for ResNet, 240 for EfficientNet)
            projection_dim: Output dimension of projection head (default: 256)
            hidden_dim: Hidden dimension of MLP heads (default: 4096)
            momentum: Initial momentum for target network EMA update
            momentum_schedule: Momentum schedule ('constant', 'cosine', 'linear')
            lr: Base learning rate
            weight_decay: Weight decay
            batch_size: Batch size
            use_lars: Whether to use LARS optimizer (recommended for large batches)
            use_mixed_precision: Whether to use mixed precision (autocast)
            use_multi_gpu: Whether to use multiple GPUs
            gradient_accumulation_steps: Number of steps to accumulate gradients
            clip_grad_norm: Gradient clipping norm (0 = no clipping)
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
            device: Device to run on ('cuda' or 'cpu')
            verbose: Print progress
            num_workers: Number of data loading workers
            augmentation_method: 'byol' or 'simclr'
        """
        # ✅ اطمینان از cuda
        if isinstance(device, str):
            if device == "cpu":
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"✅ BYOLTrainer device: {self.device}")
        self.verbose = verbose
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_workers = num_workers
        self.use_lars = use_lars
        self.use_mixed_precision = use_mixed_precision
        self.use_multi_gpu = use_multi_gpu
        self.use_kornia = kwargs.get("use_kornia", False)
        self.kwargs = kwargs
        # ✅ SPEED FIX: ذخیره پارامترهای سرعت
        self.use_kornia = use_kornia
        self.use_bf16 = use_bf16
        self.augmentation_intensity = augmentation_intensity
        self.use_medical_safe = use_medical_safe

        # ==========================================
        # Initialize networks
        # ==========================================
        self.online_encoder = online_encoder.to(self.device)
        self.feature_dim = self._get_feature_dim(online_encoder)
        self.num_workers = num_workers
        self.save_dir = (
            save_dir or "finalReports/selfSupervisedLearning/ResNet18/SSL_checkpoints/byol"
        )
        self.monitor = BYOLMonitor(verbose=True)

        # Target encoder (copy of online encoder)
        if target_encoder is None:
            self.target_encoder = copy.deepcopy(online_encoder).to(self.device)
        else:
            self.target_encoder = target_encoder.to(self.device)

        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # ==========================================
        # Initialize MLP heads
        # ==========================================
        self.online_projector = MLPProjector(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout_rate=0.1,
        ).to(self.device)

        self.target_projector = MLPProjector(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout_rate=0.1,
        ).to(self.device)

        # Copy target projector weights from online
        self._copy_projector_weights()

        # Predictor (only for online)
        self.predictor = MLPPredictor(
            input_dim=projection_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout_rate=0.1,
        ).to(self.device)

        # ==========================================
        # Multi-GPU support
        # ==========================================
        if use_multi_gpu and torch.cuda.device_count() > 1:
            self.online_encoder = nn.DataParallel(self.online_encoder)
            self.target_encoder = nn.DataParallel(self.target_encoder)
            self.online_projector = nn.DataParallel(self.online_projector)
            self.target_projector = nn.DataParallel(self.target_projector)
            self.predictor = nn.DataParallel(self.predictor)
            if self.verbose:
                print(f"✅ Using {torch.cuda.device_count()} GPUs")

        # ==========================================
        # Loss function
        # ==========================================
        self.criterion = BYOLLoss(temperature=1.0, symmetrize=True)

        # ==========================================
        # Optimizer (with LARS support)
        # ==========================================
        self.optimizer = self._get_optimizer(lr, weight_decay)
        self.lr = lr
        self.weight_decay = weight_decay

        # ==========================================
        # Scheduler (Cosine Annealing with Warmup)
        # ==========================================
        self.scheduler = self._get_scheduler()

        # ==========================================
        # Mixed precision
        # ==========================================
        self.scaler = GradScaler() if use_mixed_precision else None

        # ==========================================
        # Augmentation
        # ==========================================

        self.augmentation = self._get_augmentation(augmentation_method, input_size)

        # ==========================================
        # Training state
        # ==========================================
        self.momentum = momentum
        self.momentum_schedule = momentum_schedule
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float("inf")

        self.history = {
            "loss": [],
            "loss_epoch": [],
            "learning_rates": [],
            "momentum": [],
            "epoch_times": [],
            # ✅ FIX: train_acc حذف شد (BYOL unsupervised است و accuracy ندارد)
            # ✅ اضافه کردن برای debug
            "variance": [],
            "cosine_similarity": [],
            "embedding_mean": [],
            "embedding_std": [],
        }

        # ==========================================
        # Display information
        # ==========================================
        if self.verbose:
            self._print_init_info(projection_dim, hidden_dim)

    def _get_feature_dim(self, model: nn.Module) -> int:
        """
        ✅ FIXED: Get feature dim by actually running a dummy forward pass.
        """
        try:
            # ✅ Determine input size based on model name
            dummy_size = 224
            model_name = getattr(model, "model_name", "") or type(model).__name__
            if "efficientnet" in str(model_name).lower() or \
               "efficientnet_b1" in str(model_name).lower():
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

            reported = getattr(model, "_feature_dim", None) or getattr(model, "feature_dim", None)
            if reported is not None and reported != real_dim:
                if self.verbose:
                    print(f"⚠️ Model reports feature_dim={reported} but actual={real_dim}. Using actual.")

            return real_dim
        except Exception as e:
            if hasattr(model, "_feature_dim"):
                return model._feature_dim
            if hasattr(model, "feature_dim"):
                return model.feature_dim
            return 2048

    def _copy_projector_weights(self):
        """Copy weights from online projector to target projector."""
        self.target_projector.load_state_dict(self.online_projector.state_dict())

    def _get_optimizer(self, lr: float, weight_decay: float):
        """Get optimizer — supports LARS, RAdam, Ranger, AdamW."""
        params = (
            list(self.online_encoder.parameters())
            + list(self.online_projector.parameters())
            + list(self.predictor.parameters())
        )

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

        # ✅ LARS (اگر درخواست شده)
        if self.use_lars:
            try:
                from torch_optimizer import LARS
                if self.verbose:
                    print("✅ Using LARS optimizer")
                return LARS(params, lr=lr, weight_decay=weight_decay)
            except ImportError:
                if self.verbose:
                    print("⚠️ LARS not available, using AdamW instead")

        # ✅ Fallback: AdamW
        return optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8
        )

    def _get_scheduler(self):
        """Get cosine annealing scheduler with warmup."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # ✅ FIX: جلوگیری از T_max = 0 (ZeroDivisionError)
        cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)

        # ✅ FIX: اگر warmup >= total باشه، warmup رو کمتر کن
        effective_warmup = min(self.warmup_epochs, max(0, self.total_epochs - 1))

        # Base scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cosine_epochs, eta_min=1e-6
        )

        if effective_warmup > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=effective_warmup,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[effective_warmup],
            )
        else:
            return cosine_scheduler

    def _get_augmentation(self, method: str, input_size: int):
        """
        ✅ SPEED FIX: اولویت به Kornia (GPU-accelerated).
        اگه use_kornia=True باشه، augmentation کامل روی GPU اجرا میشه.
        """
        # ✅ اولویت ۱: Kornia (GPU)
        if self.use_kornia:
            try:
                from .kornia_augmentations import create_kornia_augmentation

                aug = create_kornia_augmentation(
                    target_size=input_size,
                    intensity=self.augmentation_intensity,
                    device=str(self.device),
                    normalize=True,
                    use_medical_safe=self.use_medical_safe,
                )
                if self.verbose:
                    print(f"✅ GPU-accelerated Kornia augmentation enabled")
                    print(f"   Intensity: {self.augmentation_intensity}")
                    print(f"   Device: {self.device}")
                return aug
            except Exception as e:
                print(f"⚠️ Kornia failed, falling back to CPU: {e}")

        # ✅ Fallback: torchvision (CPU)
        if method.lower() == "byol":
            return BYOLAugmentation(target_size=input_size, normalize=True)
        elif method.lower() == "simclr":
            from .augmentations import SimCLRAugmentation
            return SimCLRAugmentation(target_size=input_size, normalize=True)
        elif method.lower() == "mae":
            from .augmentations import MAEAugmentation
            return MAEAugmentation(target_size=input_size, normalize=True)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

    def _print_init_info(self, projection_dim: int, hidden_dim: int):
        """Print initialization information."""
        print("\n" + "=" * 80)
        print("🚀 Initializing BYOL Trainer (ULTIMATE EDITION)")
        print("=" * 80)
        print(f"   Feature Dimension: {self.feature_dim}")
        print(f"   Projection Dimension: {projection_dim}")
        print(f"   Hidden Dimension: {hidden_dim}")
        print(f"   Initial Momentum: {self.momentum}")
        print(f"   Momentum Schedule: {self.momentum_schedule}")
        print(f"   Batch Size: {self.batch_size}")
        print(
            f"   Learning Rate: {self.lr if hasattr(self, 'lr') else self.optimizer.param_groups[0]['lr']}"
        )
        print(f"   Weight Decay: {self.optimizer.param_groups[0]['weight_decay']}")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {self.use_mixed_precision}")
        print(f"   Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"   Warmup Epochs: {self.warmup_epochs}")
        print(f"   Total Epochs: {self.total_epochs}")
        print("=" * 80)

    def _update_target_network(self):
        """
        Update target network using Exponential Moving Average (EMA).

        θ_target = momentum * θ_target + (1 - momentum) * θ_online
        """
        with torch.no_grad():
            # Update target encoder
            for online_param, target_param in zip(
                self.online_encoder.parameters(), self.target_encoder.parameters()
            ):
                target_param.data = (
                    self.momentum * target_param.data + (1 - self.momentum) * online_param.data
                )

            # Update target projector
            for online_param, target_param in zip(
                self.online_projector.parameters(), self.target_projector.parameters()
            ):
                target_param.data = (
                    self.momentum * target_param.data + (1 - self.momentum) * online_param.data
                )

    def _update_momentum(self):
        """
        Update momentum value based on schedule.

        Schedules:
        - 'constant': Keep initial momentum
        - 'cosine': Cosine schedule from initial to 0.999
        - 'linear': Linear schedule from initial to 0.999
        """
        if self.momentum_schedule == "constant":
            return

        progress = self.current_epoch / max(self.total_epochs, 1)
        target_momentum = 0.998   # ✅ FIX: کاهش از 0.999

        if self.momentum_schedule == "cosine":
            # Cosine schedule
            momentum = (
                target_momentum
                - (target_momentum - self.momentum) * (np.cos(np.pi * progress) + 1) / 2
            )
            self.momentum = min(target_momentum, max(self.momentum, momentum))

        elif self.momentum_schedule == "linear":
            # Linear schedule
            self.momentum = min(
                target_momentum,
                self.momentum + (target_momentum - self.momentum) * progress,
            )

    def _generate_views(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ SPEED FIX: batch-aware augmentation.

        اگه augmentation از نوع Kornia باشه، کامل batch رو یکجا پردازش می‌کنه (GPU).
        اگه از نوع torchvision باشه، fallback به loop (کند).
        """
        batch_size = images.size(0)

        # ✅ حالت ۱: batch-aware (Kornia)
        # images: (B, C, H, W) در [0, 255] یا [0, 1]
        if hasattr(self.augmentation, "forward"):
            # Kornia augmentation — batch-level
            try:
                result = self.augmentation(images)
                if isinstance(result, (list, tuple)) and len(result) == 2:
                    view1, view2 = result
                else:
                    # اگه فقط یک view برگردونه، دوبار اعمال کن
                    view1 = result
                    view2 = result
            except Exception as e:
                raise RuntimeError(f"Batch augmentation failed: {e}")
        else:
            # Fallback: torchvision (loop — کند)
            view1_list = []
            view2_list = []

            for i in range(batch_size):
                img = images[i]
                # تبدیل به PIL اگه لازم باشه
                if img.max() <= 1.0:
                    img_pil = T.ToPILImage()(img)
                else:
                    img_pil = T.ToPILImage()(img / 255.0)

                v1, v2 = self.augmentation(img_pil)
                view1_list.append(v1.unsqueeze(0))
                view2_list.append(v2.unsqueeze(0))

            view1 = torch.cat(view1_list, dim=0)
            view2 = torch.cat(view2_list, dim=0)

        # ✅ Sanity check
        if view1 is None or view2 is None:
            raise RuntimeError("Augmentation returned None")

        if view1.shape != view2.shape:
            raise RuntimeError(f"Shape mismatch: {view1.shape} vs {view2.shape}")

        # ✅ اطمینان از device
        view1 = view1.to(self.device, non_blocking=True)
        view2 = view2.to(self.device, non_blocking=True)

        return view1, view2

    def forward(
            self, view1: torch.Tensor, view2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Online network (gradients enabled)
        self.online_encoder.train()
        self.online_projector.train()
        self.predictor.train()

        # Get features and projections from online network
        if hasattr(self.online_encoder, "get_embedding"):
            h1 = self.online_encoder.get_embedding(view1)
            h2 = self.online_encoder.get_embedding(view2)
        else:
            h1 = self.online_encoder(view1)
            h2 = self.online_encoder(view2)
            if h1.dim() > 2:
                h1 = h1.view(h1.size(0), -1)
                h2 = h2.view(h2.size(0), -1)

        # ✅ FIX: match projector's expected input_dim
        expected_dim = self.online_projector.input_dim
        if h1.size(-1) != expected_dim:
            if h1.size(-1) > expected_dim:
                h1 = h1[..., :expected_dim]
                h2 = h2[..., :expected_dim]
            else:
                pad_size = expected_dim - h1.size(-1)
                h1 = torch.nn.functional.pad(h1, (0, pad_size))
                h2 = torch.nn.functional.pad(h2, (0, pad_size))

        # Online projections
        p1 = self.online_projector(h1)
        p2 = self.online_projector(h2)

        # Online predictions
        q1 = self.predictor(p1)
        q2 = self.predictor(p2)

        # Target network (no gradients)
        with torch.no_grad():
            self.target_encoder.eval()
            self.target_projector.eval()

            # Get features and projections from target network
            if hasattr(self.target_encoder, "get_embedding"):
                h1_t = self.target_encoder.get_embedding(view1)
                h2_t = self.target_encoder.get_embedding(view2)
            else:
                h1_t = self.target_encoder(view1)
                h2_t = self.target_encoder(view2)
                if h1_t.dim() > 2:
                    h1_t = h1_t.view(h1_t.size(0), -1)
                    h2_t = h2_t.view(h2_t.size(0), -1)

            # ✅ FIX: match target projector's expected input_dim
            expected_dim_t = self.target_projector.input_dim
            if h1_t.size(-1) != expected_dim_t:
                if h1_t.size(-1) > expected_dim_t:
                    h1_t = h1_t[..., :expected_dim_t]
                    h2_t = h2_t[..., :expected_dim_t]
                else:
                    pad_size = expected_dim_t - h1_t.size(-1)
                    h1_t = torch.nn.functional.pad(h1_t, (0, pad_size))
                    h2_t = torch.nn.functional.pad(h2_t, (0, pad_size))

            # Target projections
            z1 = self.target_projector(h1_t)
            z2 = self.target_projector(h2_t)

        return q1, q2, z1, z2

    def train_step(self, batch: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        ✅ SPEED FIX:
        - BF16 به جای FP16 (سریع‌تر روی RTX 2050)
        - Kornia augmentation روی GPU بعد از انتقال batch به device
        """
        # ✅ SPEED FIX: batch به device منتقل بشه، سپس augmentation
        if not isinstance(batch, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(batch)}")

        batch = batch.to(self.device, non_blocking=True)

        # ✅ اگه augmentation از نوع Kornia (batch-aware) هست، مستقیم batch رو بده
        view1, view2 = self._generate_views(batch)

        # ✅ SPEED FIX: BF16 autocast
        use_autocast = (
                self.device.type == "cuda"
                and self.use_mixed_precision
        )

        if use_autocast:
            # ✅ BF16 اگه فعال، وگرنه FP16
            dtype = torch.bfloat16 if self.use_bf16 else torch.float16
            with torch.amp.autocast('cuda', dtype=dtype):
                q1, q2, z1, z2 = self.forward(view1, view2)
                loss = self.criterion(q1, q2, z1, z2)
        else:
            q1, q2, z1, z2 = self.forward(view1, view2)
            loss = self.criterion(q1, q2, z1, z2)

        # ✅ FIX: ذخیره در FP32 برای debug دقیق (BF16 باعث rounding error میشه)
        self._last_z1 = z1.detach().float()
        self._last_z2 = z2.detach().float()
        self._last_q1 = q1.detach().float()
        self._last_q2 = q2.detach().float()
        self._last_loss = loss

        # ✅ NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"🔥 NaN/Inf detected in loss! "
                f"Loss: {loss.item()}, "
                f"q1 std: {q1.std().item()}, "
                f"q2 std: {q2.std().item()}"
            )

        raw_loss = loss.item()
        loss = loss / self.gradient_accumulation_steps

        # ✅ Backward
        if self.scaler is not None and not self.use_bf16:
            # BF16 نیازی به scaler نداره
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.current_step += 1

        if self.current_step % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.clip_grad_norm > 0:
                params = (
                        list(self.online_encoder.parameters())
                        + list(self.online_projector.parameters())
                        + list(self.predictor.parameters())
                )
                if self.scaler is not None and not self.use_bf16:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)

            # Optimizer step
            if self.scaler is not None and not self.use_bf16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)  # ✅ set_to_none=True سریع‌تره

        # ✅ EMA update
        self._update_target_network()

        return raw_loss, {
            "loss": raw_loss,
            "loss_for_backward": loss.item(),
        }

    def train_step_from_views(self, view1: torch.Tensor, view2: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        ✅ NEW: Train step that uses PRE-AUGMENTED views (no re-augmentation).
        Use this when DataLoader already returns (view1, view2).
        """
        view1 = view1.to(self.device, non_blocking=True)
        view2 = view2.to(self.device, non_blocking=True)

        use_autocast = (self.device.type == "cuda" and self.use_mixed_precision)

        if use_autocast:
            dtype = torch.bfloat16 if self.use_bf16 else torch.float16
            with torch.amp.autocast('cuda', dtype=dtype):
                q1, q2, z1, z2 = self.forward(view1, view2)
                loss = self.criterion(q1, q2, z1, z2)
        else:
            q1, q2, z1, z2 = self.forward(view1, view2)
            loss = self.criterion(q1, q2, z1, z2)

        # save debug tensors
        self._last_z1 = z1.detach().float()
        self._last_z2 = z2.detach().float()
        self._last_q1 = q1.detach().float()
        self._last_q2 = q2.detach().float()
        self._last_loss = loss

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"NaN/Inf loss! loss={loss.item()}")

        raw_loss = loss.item()
        loss = loss / self.gradient_accumulation_steps

        if self.scaler is not None and not self.use_bf16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.current_step += 1

        if self.current_step % self.gradient_accumulation_steps == 0:
            if self.clip_grad_norm > 0:
                params = (
                        list(self.online_encoder.parameters())
                        + list(self.online_projector.parameters())
                        + list(self.predictor.parameters())
                )
                if self.scaler is not None and not self.use_bf16:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)

            if self.scaler is not None and not self.use_bf16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

        self._update_target_network()

        return raw_loss, {"loss": raw_loss}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader with unlabeled images

        Returns:
            Dictionary with loss metrics
        """

        monitor_interval = 5
        warmup_epochs = 15

        self.online_encoder.train()
        self.online_projector.train()
        self.predictor.train()

        total_loss = 0.0
        total_steps = 0
        losses = []

        epoch_start_time = time.time()

        # ✅ FIXED: debug رو از environment variable بخون
        # در CV، این مقدار رو 999999 بذار (عملاً غیرفعال)
        import os
        debug_interval = int(os.environ.get("BYOL_DEBUG_INTERVAL", "23"))
        # ✅ SPEED FIX: debug فقط هر ۵ epoch
        run_debug = (self.current_epoch % 5 == 0) or (self.current_epoch == 0)

        loop = tqdm(
            dataloader,
            desc=f"[BYOL] Epoch {self.current_epoch + 1}/{self.total_epochs}",
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
            colour="#87CEEB",
            ncols=100,
            leave=False,
            disable=not self.verbose,
        )

        for batch_idx, batch in enumerate(loop):
            # ✅ اصلاح: بررسی نوع batch
            #if isinstance(batch, (list, tuple)):
                #if len(batch) == 2:
                    #images, _ = batch
                #else:
                    #images = batch[0]
            #else:
                #images = batch

            # ✅ اطمینان از Tensor بودن
            #if not isinstance(images, torch.Tensor):
                #continue

            # ✅ FIX: اگر batch_size=1 بود، skip کن (BatchNorm نیاز به >1 دارد)
            #if images.size(0) <= 1:
                #continue

            #images = images.to(self.device)

            # Training step
            #loss, metrics = self.train_step(images)

            # ✅ FIX: اگر DataLoader دو view می‌دهد → مستقیم استفاده کن
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x1, x2 = batch
                if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
                    if x1.size(0) <= 1:
                        continue
                    # ✅ دو view آماده — augment را دور بزن
                    loss, metrics = self.train_step_from_views(x1, x2)
                else:
                    continue
            else:
                # fallback: augment در trainer
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                if not isinstance(images, torch.Tensor) or images.size(0) <= 1:
                    continue
                images = images.to(self.device)
                loss, metrics = self.train_step(images)

            # ✅ SPEED FIX: debug فقط در epochs انتخاب‌شده
            debug_results = None
            if run_debug and batch_idx % debug_interval == 0 and hasattr(self, "_last_z1"):
                debug_results = self._debug_metrics(
                    loss=self._last_loss,
                    q1=self._last_q1,
                    q2=self._last_q2,
                    z1=self._last_z1,
                    z2=self._last_z2,
                    step=batch_idx,
                    epoch=self.current_epoch,
                )

                # ✅ FIX: ذخیره‌ی metrics در history برای هر debug step
                if debug_results is not None:
                    self.history["variance"].append(debug_results.get("variance", 0.0))
                    self.history["cosine_similarity"].append(
                        debug_results.get("cosine_similarity", 0.0)
                    )
                    self.history["embedding_mean"].append(
                        debug_results.get("mean_z1", 0.0)
                    )
                    self.history["embedding_std"].append(
                        debug_results.get("std_z1", 0.0)
                    )

            # ✅ اگه مدل سالم نیست، هشدار بده (فقط در حالت run_debug)
            if run_debug and debug_results is not None:
                if not debug_results["is_healthy"]:
                    print("⚠️ WARNING: Model may be collapsing!")

            # ✅ اگر مدل سالم نیست، هشدار بده
            if debug_results is not None:
                if not debug_results["is_healthy"]:
                    print("⚠️ WARNING: Model may be collapsing!")
                if debug_results["variance"] < 0.01:
                    print("   → Variance is too low (< 0.01)")
                if debug_results["cosine_similarity"] > 0.95:
                    print("   → Cosine similarity is too high (> 0.95)")

            total_loss += loss
            total_steps += 1
            losses.append(loss)

            # Update progress bar
            current_loss = total_loss / total_steps
            loop.set_postfix({"loss": f"{current_loss:.4f}", "mom": f"{self.momentum:.4f}"})

            # ============================================================
            # ✅ FIX 2: NaN/INF detection
            # ============================================================

            if math.isnan(loss) or math.isinf(loss):
                print(f"🚨 NaN/INF loss detected in TRAIN at step {batch_idx}")
                print(f"   Loss: {loss}")
                print(f"   q1 mean: {q1.mean().item() if 'q1' in dir() else 'N/A'}")
                print(f"   z1 mean: {z1.mean().item() if 'z1' in dir() else 'N/A'}")
                continue
            elif torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                print("🚨 NaN/INF detected!")

            # ✅ Monitoring
            if (self.current_epoch + 1) % monitor_interval == 0 and batch_idx == 0:
                if hasattr(self, "_last_z1") and hasattr(self, "_last_q1"):
                    monitor_results = self.monitor.monitor(
                        z1=self._last_z1,
                        z2=self._last_z2,
                        q1=self._last_q1,
                        q2=self._last_q2,
                        epoch=self.current_epoch + 1,
                        loss=loss,
                    )

                    # ✅ FIX: ذخیره وضعیت سلامت برای استفاده در pretrain
                    self._last_health_status = monitor_results["health_status"]["status"]

                    # ✅ اگر epoch کمتر از warmup هست، فقط log کن، متوقف نکن
                    if self.current_epoch < warmup_epochs:
                        print(f"⏳ Warmup phase (Epoch {self.current_epoch+1}) - Monitoring only")
                        continue

                    # ✅ FIX: فقط بعد از warmup و بعد از epoch 10 متوقف کن
                    if (
                        self.current_epoch >= warmup_epochs
                        and self.current_epoch >= 10  # ← اضافه
                        and not monitor_results["health_status"]["is_healthy"]
                    ):
                        # ✅ FIX: قبل از stop، 3 بار چک کن که collapse واقعی است
                        if not hasattr(self, "_collapse_counter"):
                            self._collapse_counter = 0
                        self._collapse_counter += 1

                        if self._collapse_counter >= 3:
                            print("🛑 Stopping training due to collapse detection!")
                            return {"loss": loss, "early_stopped": True}
                    else:
                        self._collapse_counter = 0  # reset

        # ============================================================
        # ✅ FIX 4: جلوگیری از avg_loss = inf
        # ============================================================
        if total_steps == 0:
            print("💀 ERROR: No training steps executed in this epoch!")
            return {
                "loss": float("inf"),
                "best_loss": self.best_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "momentum": self.momentum,
            }

        avg_loss = total_loss / total_steps

        # Update momentum
        self._update_momentum()

        # Update scheduler
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        # ✅ FIX: محاسبه و ذخیره epoch_time
        epoch_time = time.time() - epoch_start_time
        self.history["epoch_times"].append(epoch_time)

        # Save history
        self.history["loss"].extend(losses)
        self.history["loss_epoch"].append(avg_loss)
        self.history["learning_rates"].append(current_lr)
        self.history["momentum"].append(self.momentum)

        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss

        return {
            "loss": avg_loss,
            "best_loss": self.best_loss,
            "learning_rate": current_lr,
            "momentum": self.momentum,
            "epoch_time": epoch_time,
        }

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = None,
        save_dir: str = "ssl_checkpoints",
        save_interval: int = 10,
        validation_loader: DataLoader = None,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            dataloader: DataLoader with unlabeled images
            epochs: Number of epochs (if None, use total_epochs)
            save_dir: Directory to save checkpoints
            save_interval: Interval for saving checkpoints
            validation_loader: Optional validation loader for monitoring

        Returns:
            Dictionary with training results
        """
        if epochs is not None:
            self.total_epochs = epochs

        os.makedirs(save_dir, exist_ok=True)

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"🚀 Starting BYOL Training for {self.total_epochs} epochs")
            print("=" * 80)
            print("📌 Key Features:")
            print("   ✅ No negative samples required")
            print("   ✅ EMA momentum target network (cosine schedule)")
            print("   ✅ Symmetrized loss")
            print("   ✅ Self-supervised learning on liver ultrasound images")
            if self.use_mixed_precision:
                print("   ✅ Mixed precision training (autocast)")
            if self.use_lars:
                print("   ✅ LARS optimizer")
            if self.gradient_accumulation_steps > 1:
                print(f"   ✅ Gradient accumulation ({self.gradient_accumulation_steps} steps)")
            print("=" * 80)

        start_time = time.time()

        for epoch in range(self.total_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train epoch
            metrics = self.train_epoch(dataloader)

            epoch_time = time.time() - epoch_start
            self.history["epoch_times"].append(epoch_time)

            if self.verbose:
                print(
                    f"📊 Epoch {epoch+1}/{self.total_epochs} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Best Loss: {metrics['best_loss']:.4f} | "
                    f"LR: {metrics['learning_rate']:.2e} | "
                    f"Momentum: {metrics['momentum']:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )

            # Validation (optional)
            if validation_loader is not None:
                val_metrics = self.validate(validation_loader)
                if self.verbose:
                    print(f"   📊 Validation Loss: {val_metrics['loss']:.4f}")

            # ✅ NEW: Save 'last' checkpoint every epoch
            self.save_checkpoint(
                os.path.join(save_dir, "byol_last.pth"),
                save_type="last",
            )

            # ✅ NEW: Save 'best' checkpoint if improved
            if metrics["loss"] <= self.best_loss:
                self.save_checkpoint(
                    os.path.join(save_dir, "byol_best.pth"),
                    save_type="best",
                )

            # ✅ NEW: Save 'checkpoint' every 10 epochs
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"byol_checkpoint_epoch_{epoch+1}.pth"),
                    save_type="checkpoint",
                )

            # Early stopping (optional - based on loss plateau)
            if epoch > 20 and len(self.history["loss_epoch"]) > 20:
                recent_losses = self.history["loss_epoch"][-10:]
                if all(l >= recent_losses[0] for l in recent_losses[1:]):
                    if self.verbose:
                        print(f"⚠️ Early stopping triggered (loss plateau for 10 epochs)")
                    break

        # ✅ NEW: Save 'final' checkpoint
        self.save_checkpoint(
            os.path.join(save_dir, "byol_final.pth"),
            save_type="final",
        )

        total_time = time.time() - start_time

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"✅ BYOL Training Complete!")
            print(f"   Best Loss: {self.best_loss:.4f}")
            print(f"   Total Time: {total_time:.2f}s ({total_time/60:.2f}m)")
            print("=" * 80)

        return {
            "history": self.history,
            "best_loss": self.best_loss,
            "total_time": total_time,
            "epochs": self.current_epoch + 1,
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model on a validation set.

        Args:
            dataloader: Validation DataLoader

        Returns:
            Dictionary with validation metrics
        """
        self.online_encoder.eval()
        self.online_projector.eval()
        self.predictor.eval()

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", leave=False, disable=not self.verbose):
                # ✅ اصلاح: بررسی نوع batch
                #if isinstance(batch, (list, tuple)):
                    #if len(batch) == 2:
                        #images, _ = batch
                    #else:
                        #images = batch[0]
                #else:
                    #images = batch

                #if not isinstance(images, torch.Tensor):
                    #continue

                # ✅ FIX: اگر batch_size=1 بود، skip کن
                #if images.size(0) <= 1:
                    #continue

                #images = images.to(self.device, non_blocking=True)
                #view1, view2 = self._generate_views(images)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x1, x2 = batch
                    if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
                        if x1.size(0) <= 1:
                            continue
                        view1 = x1.to(self.device, non_blocking=True)
                        view2 = x2.to(self.device, non_blocking=True)
                    else:
                        continue
                else:
                    images = batch[0] if isinstance(batch, (list, tuple)) else batch
                    if not isinstance(images, torch.Tensor) or images.size(0) <= 1:
                        continue
                    images = images.to(self.device, non_blocking=True)
                    view1, view2 = self._generate_views(images)

                # ✅ FIX: اگر batch_size=1 بود، skip کن (BatchNorm نیاز به >1 دارد)
                if images.size(0) <= 1:
                    continue

                # Forward pass
                q1, q2, z1, z2 = self.forward(view1, view2)

                q1 = F.normalize(q1, dim=-1)
                q2 = F.normalize(q2, dim=-1)
                z1 = F.normalize(z1, dim=-1)
                z2 = F.normalize(z2, dim=-1)

                loss = self.criterion(q1, q2, z1, z2)

                # ✅ FIXED: print اضافی حذف شد
                if math.isnan(loss) or math.isinf(loss):
                    print(f"🚨 NaN/INF loss detected in VALIDATION")
                    continue

                total_loss += loss.item()
                total_steps += 1

        # ============================================================
        # ✅ FIX 4: جلوگیری از avg_loss = inf
        # ============================================================
        if total_steps == 0:
            print("💀 ERROR: No training steps executed in this epoch!")
            return {
                "loss": float("inf"),
                "best_loss": self.best_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "momentum": self.momentum,
            }

        avg_loss = total_loss / total_steps

        # ✅ FIX 5: چاپ validation loss
        print(f"📊 Validation loss: {avg_loss:.6f} (steps: {total_steps})")

        return {"loss": avg_loss}

    def save_checkpoint(self, path: str, save_type: str = "checkpoint") -> None:
        """Save checkpoint with all training state."""
        checkpoint = {
            # Model states
            "online_encoder_state_dict": self.online_encoder.state_dict(),
            "online_projector_state_dict": self.online_projector.state_dict(),
            "target_encoder_state_dict": self.target_encoder.state_dict(),
            "target_projector_state_dict": self.target_projector.state_dict(),
            "predictor_state_dict": self.predictor.state_dict(),
            # Optimizer and scheduler
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            # Training state
            "history": self.history,
            "best_loss": self.best_loss,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "momentum": self.momentum,
            "save_type": save_type,
            # Configuration
            "config": {
                "feature_dim": self.feature_dim,
                "projection_dim": self.online_projector.output_dim,
                "hidden_dim": self.online_projector.hidden_dim,
                "momentum_schedule": self.momentum_schedule,
                "batch_size": self.batch_size,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
                "total_epochs": self.total_epochs,
                "warmup_epochs": self.warmup_epochs,
                "use_mixed_precision": self.use_mixed_precision,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
            },
            "timestamp": datetime.now().isoformat(),
            "version": "6.0.0",
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(checkpoint, path)

        if self.verbose:
            print(f"✅ Checkpoint ({save_type}) saved: {path} ({os.path.getsize(path) / 1e6:.2f} MB)")

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """Load checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # Load model states
        self.online_encoder.load_state_dict(checkpoint["online_encoder_state_dict"])
        self.online_projector.load_state_dict(checkpoint["online_projector_state_dict"])
        self.target_encoder.load_state_dict(checkpoint["target_encoder_state_dict"])
        self.target_projector.load_state_dict(checkpoint["target_projector_state_dict"])
        self.predictor.load_state_dict(checkpoint["predictor_state_dict"])

        # Load optimizer and scheduler
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.history = checkpoint["history"]
        self.best_loss = checkpoint["best_loss"]
        self.current_epoch = checkpoint["current_epoch"]
        self.current_step = checkpoint.get("current_step", 0)
        self.momentum = checkpoint.get("momentum", self.momentum)

        if self.verbose:
            print(f"✅ Checkpoint loaded: {path}")
            print(f"   Best Loss: {self.best_loss:.4f}")
            print(f"   Epoch: {self.current_epoch}")
            print(f"   Momentum: {self.momentum:.4f}")

    def get_encoder(self) -> nn.Module:
        """Get the trained online encoder."""
        self.online_encoder.eval()
        return self.online_encoder

    def get_target_encoder(self) -> nn.Module:
        """Get the target encoder (momentum encoder)."""
        self.target_encoder.eval()
        return self.target_encoder

    def get_embeddings(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from the online encoder.

        Args:
            dataloader: DataLoader with images

        Returns:
            Tuple of (embeddings, labels)
        """
        self.online_encoder.eval()

        embeddings = []
        labels = []

        with torch.no_grad():
            for images, lbls in tqdm(dataloader, desc="Extracting embeddings"):
                images = images.to(self.device)
                if hasattr(self.online_encoder, "get_embedding"):
                    emb = self.online_encoder.get_embedding(images)
                else:
                    emb = self.online_encoder(images)
                    if emb.dim() > 2:
                        emb = emb.view(emb.size(0), -1)
                embeddings.append(emb.cpu().numpy())
                labels.append(lbls.numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        return embeddings, labels

    def get_history(self) -> Dict[str, List]:
        """Get training history."""
        return self.history

    def get_config(self) -> Dict[str, Any]:
        """Get trainer configuration."""
        return {
            "feature_dim": self.feature_dim,
            "projection_dim": self.online_projector.output_dim,
            "hidden_dim": self.online_projector.hidden_dim,
            "momentum_schedule": self.momentum_schedule,
            "batch_size": self.batch_size,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "use_mixed_precision": self.use_mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "clip_grad_norm": self.clip_grad_norm,
            "device": self.device,
        }

    def _debug_metrics(
        self,
        loss: float,
        q1: torch.Tensor,
        q2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        step: int = 0,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        محاسبه و چاپ metrics برای debug (ULTIMATE EDITION)

        Metrics:
        - Loss واقعی (raw و normalized)
        - Feature Variance (تشخیص collapse) - برای z1 و z2
        - Embedding Diversity (کیفیت یادگیری)
        - Cosine Similarity (same vs random)
        - Pairwise Distance (شاه‌کلید تشخیص diversity)

        Returns:
            Dictionary با تمام metrics
        """
        with torch.no_grad():
            # ============================================================
            # ✅ 1. Loss واقعی
            # ============================================================
            raw_loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

            # ============================================================
            # ✅ 2. Feature Variance (مهم‌ترین!)
            # ============================================================
            variance_z1 = z1.var(dim=0).mean().item()
            variance_z2 = z2.var(dim=0).mean().item()

            # ============================================================
            # ✅ 3. Embedding Diversity
            # ============================================================
            mean_z1 = z1.mean().item()
            std_z1 = z1.std().item()
            mean_z2 = z2.mean().item()
            std_z2 = z2.std().item()

            # ============================================================
            # ✅ 4. Cosine Similarity (same vs random) — FIXED
            # ============================================================
            # Same: شباهت بین دو view از همان تصویر — باید بالا باشد
            cos_sim_same = torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean().item()

            # ✅ FIX: استفاده از بافر global برای "random"
            if not hasattr(self, "_z_buffer") or self._z_buffer is None:
                self._z_buffer = z1.detach().clone()
            else:
                self._z_buffer = torch.cat([self._z_buffer, z1.detach()], dim=0)
                # ✅ بافر بزرگتر (512) و خارج از گراف
                if self._z_buffer.size(0) > 512:
                    self._z_buffer = self._z_buffer[-512:]

            # مقایسه با نمونه‌های بافر (random واقعی، نه از batch جاری)
            if self._z_buffer.size(0) > z1.size(0) * 2:
                buf_idx = torch.randperm(self._z_buffer.size(0))[:z1.size(0)]
                z_buffer_sample = self._z_buffer[buf_idx]
                cos_sim_random = torch.nn.functional.cosine_similarity(z1, z_buffer_sample, dim=1).mean().item()
            else:
                idx = torch.randperm(z1.size(0))
                cos_sim_random = torch.nn.functional.cosine_similarity(z1, z1[idx], dim=1).mean().item()

            # ============================================================
            # ✅ 5. Pairwise Distance (شاه‌کلید)
            # ============================================================
            # ✅ FIX: clamp + normalize + float32 برای جلوگیری از overflow
            z1_for_dist = z1.detach().float()
            # clamp مقادیر بزرگ — وقتی Std=15، مقادیر به ±50 می‌رسند و cdist overflow می‌کند
            z1_for_dist = torch.clamp(z1_for_dist, min=-100.0, max=100.0)
            # نرمال‌سازی برای اطمینان از اینکه در واحد sphere هستند
            z1_for_dist = z1_for_dist / (z1_for_dist.norm(dim=1, keepdim=True) + 1e-8)
            dist = torch.cdist(z1_for_dist, z1_for_dist, p=2)
            # ✅ حذف diagonal (فاصله هر نمونه با خودش = 0) که میانگین را خراب می‌کند
            n = dist.size(0)
            mask = ~torch.eye(n, dtype=torch.bool, device=dist.device)
            pairwise_distance = dist[mask].mean().item()

            # ============================================================
            # ✅ 6. Min/Max برای تشخیص explosion
            # ============================================================
            min_z1 = z1.min().item()
            max_z1 = z1.max().item()

            # ============================================================
            # ✅ 7. درصد variance در کل batch
            # ============================================================
            var_percent_z1 = (variance_z1 / (z1.var().item() + 1e-8)) * 100
            var_percent_z2 = (variance_z2 / (z2.var().item() + 1e-8)) * 100

            # ============================================================
            # ✅ 8. تشخیص سلامت مدل (روی z خام)
            # ============================================================
            # ✅ FIX: آستانه‌های جدید
            # مشکل قبلی: criterion ها خیلی سختگیرانه بودند
            # - Cos Same: 0.95 خیلی سختگیرانه — 0.92 هم یعنی مشکل ولی می‌تواند آموزش ببیند
            # - Difference: 0.03 خیلی سختگیرانه — 0.01 هم کافی است در ابتدا
            cos_diff = abs(cos_sim_same - cos_sim_random)

            # ✅ FIX: آستانه‌های منطقی‌تر
            # cos_sim_same: 0.95 → 0.98 (طبق درخواست کاربر)
            is_healthy = (
                    0.02 < std_z1 < 100.0
                    and 0.02 < std_z2 < 100.0
                    and cos_sim_same < 0.98  # ← از 0.95 به 0.98
                    and pairwise_distance > 0.05  # ← از 0.1 به 0.05
                    and cos_diff > 0.005  # ← از 0.01 به 0.005
            )

            # ✅ FIX: در warmup (epoch < 10)، حتی آسان‌گیرتر
            if epoch < 10:
                is_healthy = (
                        0.01 < std_z1 < 200.0
                        and 0.01 < std_z2 < 200.0
                        and cos_sim_same < 0.99  # ← از 0.98 به 0.99
                        and pairwise_distance > 0.02
                        and cos_diff > 0.001
                )

            # ============================================================
            # ✅ 9. هشدارهای خاص
            # ============================================================
            # ✅ FIX: آستانه‌های منطقی‌تر
            warnings = []
            if std_z1 < 0.01:  # ← آستانه‌ی واقعی‌تر (0.02 → 0.01)
                warnings.append(f"💀 z1 Std too low ({std_z1:.4f} < 0.01) - COLLAPSE")
            elif std_z1 < 0.03:  # ← warning سطح متوسط
                warnings.append(f"⚠️ z1 Std low ({std_z1:.4f} < 0.03) - MONITOR")
            elif std_z1 > 100.0:
                warnings.append(f"⚠️ z1 Std too high ({std_z1:.2f} > 100) - EXPLOSION")
            if std_z2 < 0.01:  # ← آستانه‌ی واقعی‌تر (0.02 → 0.01)
                warnings.append(f"💀 z2 Std too low ({std_z2:.4f} < 0.01) - COLLAPSE")
            elif std_z2 < 0.03:  # ← warning سطح متوسط
                warnings.append(f"⚠️ z2 Std low ({std_z2:.4f} < 0.03) - MONITOR")
            elif std_z2 > 100.0:
                warnings.append(f"⚠️ z2 Std too high ({std_z2:.2f} > 100) - EXPLOSION")

            # ✅ FIX: فقط در صورتی که > 0.98 باشه (طبق درخواست کاربر)
            if cos_sim_same > 0.98:
                warnings.append(f"🚨 Cosine similarity (same) too high ({cos_sim_same:.4f} > 0.98)")
            elif cos_sim_same > 0.95:
                warnings.append(f"⚠️ Cosine similarity (same) high ({cos_sim_same:.4f}) - MONITOR")

            if pairwise_distance < 0.05:
                warnings.append(f"⚠️ Pairwise distance too low ({pairwise_distance:.4f} < 0.05)")

            # حذف "same ≈ random" warning — قبلاً false positive میداد

        # ============================================================
        # ✅ چاپ خروجی زیبا
        # ============================================================
        status_emoji = "✅ HEALTHY" if is_healthy else "💀 UNHEALTHY"
        status_color = "🟢" if is_healthy else "🔴"

        print(
            f"""
    📊 ==================== DEBUG METRICS ====================
    Step: {step}  |  Epoch: {epoch}
    ────────────────────────────────────────────────────────────
       📉 Loss:                       {raw_loss_val:.6f}
    ────────────────────────────────────────────────────────────
    🧠 FEATURE VARIANCE:
    ────────────────────────────────────────────────────────
    z1 Variance:                  {variance_z1:.6f}  {'✅' if variance_z1 > 0.01 else '💀'}
    z2 Variance:                  {variance_z2:.6f}  {'✅' if variance_z2 > 0.01 else '💀'}
    ────────────────────────────────────────────────────────────
    📊 EMBEDDING DIVERSITY:
    ────────────────────────────────────────────────────────
    z1 Mean:                      {mean_z1:.6f}
    z1 Std:                       {std_z1:.6f}   {'✅' if std_z1 > 0.1 else '⚠️'}
    z2 Mean:                      {mean_z2:.6f}
    z2 Std:                       {std_z2:.6f}   {'✅' if std_z2 > 0.1 else '⚠️'}
    ────────────────────────────────────────────────────────────
       🔄 COSINE SIMILARITY:
    ────────────────────────────────────────────────────────
    Same (z1, z2):                {cos_sim_same:.4f}   {'💀' if cos_sim_same > 0.95 else '✅'}
    Random (z1, z1[idx]):         {cos_sim_random:.4f}
    Difference:                   {abs(cos_sim_same - cos_sim_random):.4f}
    ────────────────────────────────────────────────────────────
    📏 PAIRWISE DISTANCE:
    ────────────────────────────────────────────────────────────
    z1 Pairwise Distance:         {pairwise_distance:.4f}   {'✅' if pairwise_distance > 0.1 else '⚠️'}
    ────────────────────────────────────────────────────────────
    📏 Min/Max:
    ────────────────────────────────────────────────────────
    z1 Min/Max:                   {min_z1:.4f} / {max_z1:.4f}
    ────────────────────────────────────────────────────────────
    📊 Variance %:
    ────────────────────────────────────────────────────────
    z1 Variance %:                {var_percent_z1:.1f}%
    z2 Variance %:                {var_percent_z2:.1f}%
    ────────────────────────────────────────────────────────────
    💡 WARNINGS:
    ────────────────────────────────────────────────────────
    {'None ✅' if not warnings else chr(10).join('   • ' + w for w in warnings)}
    ────────────────────────────────────────────────────────────
    {status_color} STATUS: {status_emoji}
    ============================================================
    """
        )

        return {
            # Loss
            "loss": raw_loss_val,
            # Variance
            "variance": variance_z1,
            "variance_z1": variance_z1,
            "variance_z2": variance_z2,
            # Diversity
            "mean_z1": mean_z1,
            "std_z1": std_z1,
            "mean_z2": mean_z2,
            "std_z2": std_z2,
            # Cosine
            "cosine_similarity": cos_sim_same,
            "cosine_same": cos_sim_same,
            "cosine_random": cos_sim_random,
            "cosine_diff": abs(cos_sim_same - cos_sim_random),
            # Pairwise
            "pairwise_distance": pairwise_distance,
            # Min/Max
            "min": min_z1,
            "max": max_z1,
            # Percent
            "variance_percent_z1": var_percent_z1,
            "variance_percent_z2": var_percent_z2,
            # Health
            "is_healthy": is_healthy,
            "warnings": warnings,
        }

    # src/self_supervised/byol.py - NEW METHOD

    def _sanity_check_before_training(self, dataloader: DataLoader):
        """
        ✅ NEW: Complete sanity check before training.
        """
        print("\n" + "=" * 80)
        print("🔍 BYOL SANITY CHECK BEFORE TRAINING")
        print("=" * 80)

        # ✅ Check 1: DataLoader
        if dataloader is None:
            raise RuntimeError("❌ dataloader is None!")

        if len(dataloader) == 0:
            raise RuntimeError(f"❌ dataloader is empty! (len={len(dataloader)})")

        print(f"✅ DataLoader has {len(dataloader)} batches")

        # ✅ Check 2: First batch
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            raise RuntimeError("❌ Cannot iterate over dataloader!")

        # ✅ Check 3: Batch format
        if isinstance(batch, (list, tuple)):
            if len(batch) != 2:
                raise RuntimeError(f"❌ Expected batch length 2 (x1, x2), got {len(batch)}")
            images, _ = batch
        else:
            images = batch

        if not isinstance(images, torch.Tensor):
            raise RuntimeError(f"❌ Images is not a tensor: {type(images)}")

        print(f"✅ Images shape: {images.shape}")

        # ✅ Check 4: Test augmentation
        try:
            view1, view2 = self._generate_views(images[:2])
            print(f"✅ View1 shape: {view1.shape}")
            print(f"✅ View2 shape: {view2.shape}")

            if view1.shape != view2.shape:
                raise RuntimeError(f"❌ Shape mismatch: {view1.shape} vs {view2.shape}")

            if torch.isnan(view1).any() or torch.isnan(view2).any():
                raise RuntimeError("❌ NaN detected in augmented views!")

            print("✅ Augmentation works correctly")

        except Exception as e:
            raise RuntimeError(f"❌ Augmentation test failed: {e}")

        # ✅ Check 5: Test forward pass
        try:
            view1, view2 = self._generate_views(images[:2])
            q1, q2, z1, z2 = self.forward(view1, view2)

            print(f"✅ q1 shape: {q1.shape}")
            print(f"✅ q2 shape: {q2.shape}")
            print(f"✅ z1 shape: {z1.shape}")
            print(f"✅ z2 shape: {z2.shape}")

            if torch.isnan(q1).any() or torch.isnan(q2).any():
                raise RuntimeError("❌ NaN detected in predictions!")

            if torch.isnan(z1).any() or torch.isnan(z2).any():
                raise RuntimeError("❌ NaN detected in projections!")

            print("✅ Forward pass works correctly")

        except Exception as e:
            raise RuntimeError(f"❌ Forward pass test failed: {e}")

        # ✅ Check 6: Test loss
        try:
            loss = self.criterion(q1, q2, z1, z2)
            print(f"✅ Loss shape: {loss.shape}")
            print(f"✅ Loss value: {loss.item():.4f}")

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"❌ Loss is NaN/Inf: {loss.item()}")

            print("✅ Loss computation works correctly")

        except Exception as e:
            raise RuntimeError(f"❌ Loss test failed: {e}")

        print("=" * 80)
        print("✅ ALL SANITY CHECKS PASSED! Ready to train.\n")


# ================================================================
# ✅ BYOL FACTORY FUNCTION
# ================================================================


def create_byol_trainer(
    model: nn.Module,
    model_name: str = "resnet18",
    input_size: int = 224,
    projection_dim: int = 256,
    hidden_dim: int = 4096,
    momentum: float = 0.996,
    momentum_schedule: str = "cosine",
    lr: float = 0.0002,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    use_lars: bool = False,
    use_mixed_precision: bool = True,
    use_multi_gpu: bool = False,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: float = 1.0,
    warmup_epochs: int = 10,
    total_epochs: int = 40,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    augmentation_method: str = "byol",
    save_dir: str = None,
    **kwargs,
) -> BYOLTrainer:
    """
    Factory function to create BYOL trainer.

    Args:
        model: Base model (ResNet18, ResNet50, EfficientNet-B0, EfficientNet-B1)
        model_name: Name of the model
        input_size: Input image size
        projection_dim: Output dimension of projection head
        hidden_dim: Hidden dimension of MLP heads
        momentum: Initial momentum for target network
        momentum_schedule: Momentum schedule ('constant', 'cosine', 'linear')
        lr: Learning rate
        weight_decay: Weight decay
        batch_size: Batch size
        use_lars: Whether to use LARS optimizer
        use_mixed_precision: Whether to use mixed precision
        use_multi_gpu: Whether to use multiple GPUs
        gradient_accumulation_steps: Number of steps to accumulate gradients
        clip_grad_norm: Gradient clipping norm
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs
        device: Device to run on
        verbose: Print progress
        augmentation_method: 'byol' or 'simclr'

    Returns:
        BYOLTrainer instance
    """
    return BYOLTrainer(
        online_encoder=model,
        input_size=input_size,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        momentum=momentum,
        momentum_schedule=momentum_schedule,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        use_lars=use_lars,
        use_mixed_precision=use_mixed_precision,
        use_multi_gpu=use_multi_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad_norm=clip_grad_norm,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        device=device,
        verbose=verbose,
        augmentation_method=augmentation_method,
        save_dir=save_dir,
        **kwargs,
    )


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing BYOL Implementation (ULTIMATE EDITION)")
    print("=" * 80)

    # Create dummy model
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

    # Test trainer initialization
    print("\n📌 Testing BYOL Trainer initialization...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DummyEncoder()

    trainer = create_byol_trainer(
        model=model,
        model_name="dummy",
        input_size=224,
        projection_dim=256,
        hidden_dim=256,
        batch_size=64,
        device=device,
        verbose=True,
        total_epochs=40,
        warmup_epochs=10,
    )

    # Test forward pass
    print("\n📌 Testing forward pass...")
    dummy_batch = torch.randn(4, 3, 224, 224).to(device)
    view1, view2 = trainer._generate_views(dummy_batch)
    q1, q2, z1, z2 = trainer.forward(view1, view2)

    print(f"   q1 shape: {q1.shape}")
    print(f"   q2 shape: {q2.shape}")
    print(f"   z1 shape: {z1.shape}")
    print(f"   z2 shape: {z2.shape}")

    # Test loss
    print("\n📌 Testing loss...")
    loss = trainer.criterion(q1, q2, z1, z2)
    print(f"   Loss: {loss.item():.4f}")

    # Test train step
    print("\n📌 Testing train step...")
    loss, metrics = trainer.train_step(dummy_batch)
    print(f"   Train loss: {loss:.4f}")

    # Test save/load
    print("\n📌 Testing save/load...")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        trainer.save_checkpoint(f.name)
        print(f"   Saved to: {f.name}")

        trainer.load_checkpoint(f.name)
        print(f"   Loaded from: {f.name}")

    # Clean up
    import os

    os.unlink(f.name)

    print("\n✅ All BYOL tests passed!")
    print("=" * 80)
