# src/optimization/fitness_evaluator.py
"""
Fast Fitness Evaluator for GA — FIXED VERSION.

Fixes:
1. Removed `num_workers` from create_simclr_trainer (unsupported).
2. Loads pretrained weights from ResNet18/50 checkpoint directly.
3. Handles model creation properly.
"""

import gc
import os
import random
import time
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

warnings.filterwarnings("ignore")


# ================================================================
# ✅ PRETRAINED WEIGHTS PATH
# ================================================================

PRETRAINED_CHECKPOINTS = {
    "resnet18": "finalReports/models/resnet/resnet18/checkpoints/resnet18_best.pth",
    "resnet50": "finalReports/models/resnet/resnet50/checkpoints/resnet50_best.pth",
    "efficientnet_b0": "finalReports/models/efficientNet/B0/checkpoints/efficientnet_b0_best.pth",
    "efficientnet_b1": "finalReports/models/efficientNet/B1/checkpoints/efficientnet_b1_best.pth",
}


class FitnessEvaluator:
    """
    Fast fitness evaluator for GA.
    Trains SSL model with given params and returns Linear Probe accuracy.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        data_dir: str = "finalReports/cropped",
        labels_csv: str = "finalReports/labels.csv",
        subset_size: int = 100,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.labels_csv = labels_csv
        self.subset_size = subset_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        self.image_paths = self._load_data_paths()

    def _load_data_paths(self) -> List[str]:
        """Load balanced subset (fixed seed)."""
        if not os.path.exists(self.labels_csv):
            raise FileNotFoundError(self.labels_csv)

        df = pd.read_csv(self.labels_csv)
        class_names = ["NL", "G1", "G2", "G3"]
        class_paths = {c: [] for c in class_names}

        for _, row in df.iterrows():
            label = str(row["liverLabel"]).strip()
            name = str(row["liverName"]).strip()
            if label in class_names:
                path = os.path.join(self.data_dir, name)
                if os.path.exists(path):
                    class_paths[label].append(path)

        random.seed(42)
        n_per_class = max(1, self.subset_size // len(class_names))

        subset = []
        for c in class_names:
            paths = class_paths[c]
            if paths:
                n = min(n_per_class, len(paths))
                subset.extend(random.sample(paths, n))

        if self.verbose:
            print(f"📊 Subset: {len(subset)} images ({n_per_class}/class)")

        return subset

    # ============================================================
    # ✅ EVALUATE
    # ============================================================

    def evaluate(self, params: Dict[str, Any]) -> float:
        """Fast evaluation — target < 60 seconds."""
        t0 = time.time()
        try:
            # ✅ Extract params (with FAST caps)
            input_size = params.get("input_size", 224)
            batch_size = min(params.get("batch_size", 16), 32)
            lr = params.get("lr", 1e-4)
            weight_decay = params.get("weight_decay", 1e-4)
            warmup_epochs = min(params.get("warmup_epochs", 2), 5)
            # ✅ FIX: با subset بزرگتر، epochs بیشتر معنی‌داره
            total_epochs = min(params.get("total_epochs", 10), 10)
            num_workers = params.get("num_workers", 0)
            aug_method = params.get("augmentation_method", "byol")
            projection_dim = params.get("projection_dim", 256)
            hidden_dim = min(params.get("hidden_dim", 1024), 2048)
            momentum = params.get("momentum", 0.996)
            clip_grad_norm = params.get("clip_grad_norm", 1.0)
            grad_accum = params.get("gradient_accumulation_steps", 1)
            use_amp = params.get("use_mixed_precision", True)

            # ✅ Build encoder
            encoder = self._build_encoder()

            # ✅ Create trainer
            trainer = self._create_trainer(
                encoder=encoder,
                params=params,
                input_size=input_size,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                num_workers=num_workers,
                aug_method=aug_method,
                projection_dim=projection_dim,
                hidden_dim=hidden_dim,
                momentum=momentum,
                clip_grad_norm=clip_grad_norm,
                grad_accum=grad_accum,
                use_amp=use_amp,
            )

            # ✅ Dataloader
            from src.pipeLine.resNet18SSL import create_ssl_dataloader
            loader = create_ssl_dataloader(
                image_paths=self.image_paths,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                target_size=input_size,
                augmentation_method=aug_method,
            )

            # ✅ Quick train (3 epochs)
            for epoch in range(total_epochs):
                metrics = trainer.train_epoch(loader)
                if metrics.get("early_stopped", False):
                    break

            # ✅ Cleanup temp checkpoints
            try:
                import shutil
                temp_dirs = [
                    "temp_ga_checkpoints",
                    "finalReports/selfSupervisedLearning/GA_Optimization/temp_checkpoints",
                ]
                for temp_dir in temp_dirs:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

            # ✅ Quick Linear Probe
            from src.self_supervised.ProbabilityAlgorithms.linear_probe import (
                build_labeled_eval_loader,
                evaluate_with_linear_probe,
            )

            eval_loader = build_labeled_eval_loader(
                labels_csv=self.labels_csv,
                image_dir=self.data_dir,
                input_size=input_size,
                batch_size=16,
                max_per_class=50,
            )

            probe = evaluate_with_linear_probe(
                encoder=trainer.get_encoder(),
                labeled_loader=eval_loader,
                device=self.device,
                num_classes=4,
                test_size=0.3,
            )

            acc = float(probe.get("linear_probe_accuracy", 0.0))
            elapsed = time.time() - t0

            if self.verbose:
                print(f"      ⏱️  {elapsed:.1f}s | Acc: {acc:.4f}")

            # ✅ Cleanup
            del trainer, encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return acc

        except Exception as e:
            if self.verbose:
                print(f"      ❌ Failed ({time.time()-t0:.1f}s): {e}")
                import traceback
                traceback.print_exc()
            return 0.0

    # ============================================================
    # ✅ BUILD ENCODER — FIXED: Load pretrained weights manually
    # ============================================================

    def _build_encoder(self):
        """Create model and load pretrained weights manually."""
        # Create model WITHOUT pretrained flag
        if self.model_name == "resnet18":
            from src.models.resnet.resnet18 import create_resnet18
            model = create_resnet18(num_classes=4, pretrained=False, verbose=False)
        elif self.model_name == "resnet50":
            from src.models.resnet.resnet50 import create_resnet50
            model = create_resnet50(num_classes=4, pretrained=False, verbose=False)
        elif self.model_name == "efficientnet_b0":
            from src.models.efficientNet.B0 import create_efficientnet_b0
            model = create_efficientnet_b0(num_classes=4, pretrained=False, verbose=False)
        elif self.model_name == "efficientnet_b1":
            from src.models.efficientNet.B1 import create_efficientnet_b1
            model = create_efficientnet_b1(num_classes=4, pretrained=False, verbose=False)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        # ✅ Load pretrained weights manually
        ckpt_path = PRETRAINED_CHECKPOINTS.get(self.model_name)
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu")

                # Extract state_dict
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "encoder_state_dict" in checkpoint:
                    state_dict = checkpoint["encoder_state_dict"]
                else:
                    state_dict = checkpoint

                # Load (strict=False for compatibility)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)

                if self.verbose:
                    print(f"      ✅ Pretrained loaded: {ckpt_path}")
                    print(f"         Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            except Exception as e:
                if self.verbose:
                    print(f"      ⚠️ Pretrained load failed: {e}")
        else:
            if self.verbose:
                print(f"      ⚠️ No pretrained checkpoint found: {ckpt_path}")

        # ✅ FIX: Ensure model reports correct feature_dim
        # Run a dummy forward to detect the real output dimension
        try:
            dummy_size = 240 if "efficientnet" in self.model_name else 224
            dummy = torch.randn(1, 3, dummy_size, dummy_size)
            model.eval()
            with torch.no_grad():
                if hasattr(model, "get_embedding"):
                    feat = model.get_embedding(dummy)
                else:
                    feat = model(dummy)
                    if feat.dim() > 2:
                        feat = feat.view(feat.size(0), -1)
            real_feature_dim = feat.size(-1)
            model._feature_dim = real_feature_dim
            model.feature_dim = real_feature_dim
            if self.verbose:
                print(f"      ✅ Detected feature_dim = {real_feature_dim}")
        except Exception as e:
            if self.verbose:
                print(f"      ⚠️ Could not detect feature_dim: {e}")

        return model

    # ============================================================
    # ✅ CREATE TRAINER — FIXED
    # ============================================================

    def _create_trainer(
        self, encoder, params, input_size, batch_size, lr, weight_decay,
        warmup_epochs, total_epochs, num_workers, aug_method,
        projection_dim, hidden_dim, momentum, clip_grad_norm,
        grad_accum, use_amp,
    ):
        """Create trainer (BYOL or SimCLR)."""
        if aug_method == "byol":
            from src.self_supervised import create_byol_trainer
            return create_byol_trainer(
                model=encoder,
                model_name=self.model_name,
                input_size=input_size,
                projection_dim=projection_dim,
                hidden_dim=hidden_dim,
                momentum=momentum,
                momentum_schedule=params.get("momentum_schedule", "cosine"),
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                use_lars=params.get("use_lars", False),
                use_mixed_precision=use_amp,
                gradient_accumulation_steps=grad_accum,
                clip_grad_norm=clip_grad_norm,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                device=self.device,
                verbose=False,
                num_workers=num_workers,
                augmentation_method="byol",
            )
        else:
            # ✅ FIX: SimCLR does NOT accept num_workers
            from src.self_supervised import create_simclr_trainer
            return create_simclr_trainer(
                encoder=encoder,
                input_size=input_size,
                projection_dim=projection_dim,
                hidden_dim=hidden_dim,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                use_lars=params.get("use_lars", False),
                use_mixed_precision=use_amp,
                gradient_accumulation_steps=grad_accum,
                clip_grad_norm=clip_grad_norm,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                device=self.device,
                verbose=False,
                # ❌ num_workers removed
            )