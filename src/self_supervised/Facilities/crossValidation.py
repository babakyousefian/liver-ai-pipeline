# src/self_supervised/crossValidation.py
"""
Cross-Validation for SSL Architects (ULTIMATE EDITION - NO COLLAPSE)

✅ CRITICAL FIX: CV no longer trains SSL from scratch!
- Loads pretrained encoder
- Extracts embeddings (one pass)
- Uses StratifiedKFold + Logistic Regression
- NO collapse possible

Author: Babak Yousefian
Version: 6.0.0 (NO-COLLAPSE EDITION)
"""

import copy
import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_supervised import get_device, set_seed


# ✅ مسیرهای architecture/{method}/
EFFICIENTNET_B0_SSL_DIR = os.path.join(
    "finalReports", "selfSupervisedLearning", "EfficientNet", "B0", "SSL_checkpoints"
)
RESNET18_SSL_DIR = os.path.join(
    "finalReports", "selfSupervisedLearning", "ResNet", "18", "SSL_checkpoints"
)


# ================================================================
# ✅ Constants
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)


# ================================================================
# ✅ Simple Dataset (No Augmentation - for Embedding Extraction)
# ================================================================

class SimpleDataset(Dataset):
    """Simple dataset for embedding extraction (NO augmentation)."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ================================================================
# ✅ CV Results Container
# ================================================================

class CVResults:
    """Container for Cross-Validation results."""

    def __init__(self):
        self.fold_results = []
        self.best_fold = None
        self.mean_metrics = {}
        self.std_metrics = {}
        self.all_fold_metrics = []  # For detailed analysis

    def add_fold_result(self, fold_idx: int, metrics: Dict[str, Any]):
        self.fold_results.append({"fold": fold_idx, "metrics": metrics})

    def compute_summary(self):
        if not self.fold_results:
            return

        # Collect scalar metrics
        metric_keys = [
            "accuracy", "precision", "recall", "f1_score",
            "balanced_accuracy", "roc_auc_mean",
        ]

        for key in metric_keys:
            values = []
            for r in self.fold_results:
                v = r["metrics"].get(key)
                if isinstance(v, (int, float)) and not np.isnan(v):
                    values.append(v)

            if values:
                self.mean_metrics[key] = float(np.mean(values))
                self.std_metrics[key] = float(np.std(values))

        # Best fold = highest accuracy
        if "accuracy" in self.mean_metrics:
            best_idx = max(
                range(len(self.fold_results)),
                key=lambda i: self.fold_results[i]["metrics"].get("accuracy", 0),
            )
            self.best_fold = self.fold_results[best_idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_results": self.fold_results,
            "mean_metrics": self.mean_metrics,
            "std_metrics": self.std_metrics,
            "best_fold": self.best_fold,
            "timestamp": datetime.now().isoformat(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.fold_results:
            row = {"fold": r["fold"]}
            for k, v in r["metrics"].items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    row[k] = float(v)
            rows.append(row)
        return pd.DataFrame(rows)


# ================================================================
# ✅ MAIN CLASS: Cross-Validation (NO COLLAPSE)
# ================================================================

class SSLCrossValidation:
    """
    ✅ FIXED: 5-Fold Cross-Validation for SSL Encoders.

    No SSL training from scratch!
    - Loads pretrained encoder
    - Extracts embeddings (one pass)
    - StratifiedKFold + Logistic Regression
    """

    def __init__(
            self,
            model_name: str = "resnet18",
            method: str = "byol",
            n_folds: int = 5,
            batch_size: int = 32,
            num_workers: int = 0,  # ✅ 0 برای CV (embedding extraction ساده‌ست)
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            verbose: bool = True,
            save_dir: str = None,
            pretrained_path: str = None,
            **kwargs,
    ):
        self.model_name = model_name
        self.method = method
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        # ✅ FIX: Convert to torch.device object
        if isinstance(device, str):
            self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        else:
            self.device = device
        self.pretrained_path = pretrained_path
        self.kwargs = kwargs

        if save_dir is None:
            save_dir = "cross_validation_results"
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        set_seed(42)

        # ✅ Create model (fresh, will load weights)
        self.model = self._create_model()
        self.feature_dim = self._get_feature_dim()

        self.results = CVResults()

        if self.verbose:
            self._print_info()

    def _create_model(self) -> nn.Module:
        """Create the appropriate model."""
        if self.model_name == "resnet18":
            from src.models.resnet.resnet18 import create_resnet18
            return create_resnet18(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
        elif self.model_name == "resnet50":
            from src.models.resnet.resnet50 import create_resnet50
            return create_resnet50(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
        elif self.model_name == "efficientnet_b0":
            from src.models.efficientNet.B0 import create_efficientnet_b0
            return create_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
        elif self.model_name == "efficientnet_b1":
            from src.models.efficientNet.B1 import create_efficientnet_b1
            return create_efficientnet_b1(num_classes=NUM_CLASSES, pretrained=False, verbose=False)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _get_feature_dim(self) -> int:
        if hasattr(self.model, "_feature_dim"):
            return self.model._feature_dim
        return 2048

    def _print_info(self):
        print("\n" + "=" * 80)
        print("🚀 SSL Cross-Validation (NO-COLLAPSE EDITION)")
        print("=" * 80)
        print(f"   Model: {self.model_name}")
        print(f"   Method: {self.method.upper()}")
        print(f"   Folds: {self.n_folds}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Device: {self.device}")
        print(f"   Feature Dim: {self.feature_dim}")
        print(f"   Pre-trained: {self.pretrained_path}")
        print("=" * 80)

    # ================================================================
    # ✅ LOAD PRETRAINED ENCODER
    # ================================================================

    def _load_pretrained(self) -> bool:
        """
        ✅ NEW: Load pretrained encoder weights.
        Supports multiple path formats including architecture/{method}/
        """
        # ✅ Try the given path first
        candidate_paths = []
        if self.pretrained_path:
            candidate_paths.append(self.pretrained_path)

        # ✅ Add architecture/{method}/ candidates
        if self.model_name in ["efficientnet_b0", "resnet18"]:
            if self.model_name == "efficientnet_b0":
                base_dir = EFFICIENTNET_B0_SSL_DIR
                prefix = "b0_ssl"
            else:  # resnet18
                base_dir = RESNET18_SSL_DIR
                prefix = "resnet18_ssl"

            method_dir = os.path.join(base_dir, "architecture", self.method.lower())

            # Try best, then final, then checkpoint
            for suffix in ["best", "final", "checkpoint", "last"]:
                candidate_paths.append(
                    os.path.join(method_dir, f"{prefix}_{self.method.lower()}_{suffix}.pth")
                )

            # Try old paths (backward compat)
            candidate_paths.append(
                os.path.join(base_dir, f"{prefix}_{self.method.lower()}_best.pth")
            )
            candidate_paths.append(
                os.path.join(base_dir, f"{prefix}_{self.method.lower()}_encoder.pth")
            )

        # ✅ Find first existing path
        actual_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                actual_path = p
                break

        if actual_path is None:
            if self.verbose:
                print(f"⚠️ No pretrained weights found. Tried:")
                for p in candidate_paths[:5]:
                    print(f"   ❌ {p}")
            return False

        try:
            ckpt = torch.load(actual_path, map_location=self.device)

            # Handle different checkpoint formats
            if "encoder_state_dict" in ckpt:
                state_dict = ckpt["encoder_state_dict"]
            elif "online_encoder_state_dict" in ckpt:
                state_dict = ckpt["online_encoder_state_dict"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

            if self.verbose:
                print(f"✅ Loaded: {actual_path}")
                print(f"   Missing: {len(missing)}, Unexpected: {len(unexpected)}")

            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"❌ Failed to load pretrained: {e}")
            return False

    # ================================================================
    # ✅ EXTRACT EMBEDDINGS
    # ================================================================

    def _extract_embeddings(
            self,
            images: torch.Tensor,
            labels: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from encoder in ONE pass.
        NO training, NO collapse possible.
        """
        dataset = SimpleDataset(images, labels)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.model.eval()

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc="Extracting embeddings", leave=False, disable=not self.verbose):
                imgs = imgs.to(self.device, non_blocking=True)

                # Extract features
                if hasattr(self.model, "get_embedding"):
                    emb = self.model.get_embedding(imgs)
                else:
                    emb = self.model(imgs)
                    if emb.dim() > 2:
                        emb = emb.view(emb.size(0), -1)

                all_embeddings.append(emb.float().cpu().numpy())
                all_labels.append(lbls.numpy())

        X = np.concatenate(all_embeddings, axis=0)
        y = np.concatenate(all_labels, axis=0)

        return X, y

    # ================================================================
    # ✅ RUN CROSS-VALIDATION (StratifiedKFold + LogisticRegression)
    # ================================================================

    def run(
            self,
            data: torch.Tensor,
            labels: torch.Tensor,
            stratified: bool = True,
    ) -> CVResults:
        """
        Run Cross-Validation.

        ✅ NO SSL training!
        ✅ NO collapse possible!
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"🔬 Running {self.n_folds}-Fold Cross-Validation")
            print("=" * 80)
            print(f"   Total Samples: {len(data)}")
            print(f"   Classes: {np.unique(labels.numpy())}")
            print("=" * 80)

        # ✅ Validate
        if data is None or data.numel() == 0:
            raise ValueError("Data is empty!")
        if data.size(0) != labels.size(0):
            raise ValueError(f"Data size mismatch: {data.size(0)} vs {labels.size(0)}")

        # ✅ Load pretrained weights
        if not self._load_pretrained():
            raise RuntimeError("Failed to load pretrained weights!")

        # ✅ Extract embeddings ONCE (not per fold!)
        if self.verbose:
            print("\n📊 Extracting embeddings (ONE pass)...")

        X, y = self._extract_embeddings(data, labels)

        if self.verbose:
            print(f"✅ Extracted: X={X.shape}, y={y.shape}")
            print(f"   Classes: {np.unique(y)}")
            print(f"   Class counts: {np.bincount(y)}")

        # ✅ Free encoder memory
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ✅ StratifiedKFold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        if self.verbose:
            print(f"\n🔬 Running {self.n_folds}-Fold CV on embeddings...")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # Logistic Regression (with class_weight for imbalance)
            clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=1.0,
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_train_s, y_train)

            # Predict
            y_pred = clf.predict(X_val_s)
            y_prob = clf.predict_proba(X_val_s)

            # Compute metrics
            metrics = self._compute_metrics(y_val, y_pred, y_prob)

            self.results.add_fold_result(fold_idx, metrics)

            if self.verbose:
                print(
                    f"   Fold {fold_idx + 1}/{self.n_folds}: "
                    f"Acc={metrics['accuracy']:.4f} | "
                    f"F1={metrics['f1_score']:.4f} | "
                    f"BalAcc={metrics['balanced_accuracy']:.4f}"
                )

        # ✅ Compute summary
        self.results.compute_summary()

        if self.verbose:
            self._print_summary()

        return self.results

    def _compute_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute comprehensive metrics."""
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Balanced accuracy = mean of per-class recall
        per_class_recall = []
        for c in np.unique(y_true):
            mask = y_true == c
            if mask.sum() > 0:
                per_class_recall.append(float((y_pred[mask] == c).mean()))
        balanced_accuracy = float(np.mean(per_class_recall)) if per_class_recall else 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

        # ROC AUC (per class)
        roc_auc = {}
        try:
            y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
            if y_bin.shape[1] == NUM_CLASSES:
                for i in range(NUM_CLASSES):
                    if y_bin[:, i].sum() > 0 and (1 - y_bin[:, i]).sum() > 0:
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                        roc_auc[i] = float(auc(fpr, tpr))
        except Exception:
            pass

        roc_auc_mean = float(np.mean(list(roc_auc.values()))) if roc_auc else 0.0

        # Classification report
        class_report = classification_report(
            y_true, y_pred,
            labels=range(NUM_CLASSES),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "balanced_accuracy": balanced_accuracy,
            "roc_auc_mean": roc_auc_mean,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),  # JSON serializable
            "classification_report": class_report,
            "num_samples": int(len(y_true)),
        }

    def _print_summary(self):
        """Print CV summary."""
        print("\n" + "=" * 80)
        print("📊 Cross-Validation Summary")
        print("=" * 80)

        if not self.results.mean_metrics:
            print("⚠️ No metrics available")
            return

        print(f"\n{'Metric':<25} {'Mean':<12} {'Std':<12}")
        print("-" * 60)

        for key in ["accuracy", "balanced_accuracy", "f1_score", "precision", "recall", "roc_auc_mean"]:
            if key in self.results.mean_metrics:
                mean = self.results.mean_metrics[key]
                std = self.results.std_metrics.get(key, 0)
                print(f"{key:<25} {mean:.4f}      {std:.4f}")

        print("-" * 60)

        if self.results.best_fold:
            best = self.results.best_fold
            print(f"\n🏆 Best Fold: {best['fold'] + 1}")
            print(f"   Accuracy: {best['metrics']['accuracy']:.4f}")

        print("=" * 80)

    # ================================================================
    # ✅ REPORT GENERATION
    # ================================================================

    def generate_report(self, results: CVResults, output_path: str = None) -> str:
        """Generate comprehensive report."""
        if output_path is None:
            output_path = os.path.join(
                self.save_dir, f"cv_report_{self.model_name}_{self.method}.pdf"
            )

        df = results.to_dataframe()

        # Save CSV
        csv_path = output_path.replace(".pdf", ".csv")
        df.to_csv(csv_path, index=False)

        # Save JSON
        json_path = output_path.replace(".pdf", ".json")
        with open(json_path, "w") as f:
            json.dump(results.to_dict(), f, indent=4, default=str)

        # Plots
        plots_dir = os.path.join(os.path.dirname(output_path), "plots")
        os.makedirs(plots_dir, exist_ok=True)

        try:
            # 1. Accuracy per fold
            plt.figure(figsize=(12, 6))
            folds = df["fold"] + 1
            accuracies = df["accuracy"]

            plt.bar(folds, accuracies, color="steelblue", edgecolor="black")
            plt.axhline(
                y=np.mean(accuracies), color="red", linestyle="--",
                label=f"Mean: {np.mean(accuracies):.4f}",
            )
            plt.fill_between(
                folds,
                np.mean(accuracies) - np.std(accuracies),
                np.mean(accuracies) + np.std(accuracies),
                alpha=0.2, color="red",
            )

            plt.xlabel("Fold", fontsize=12)
            plt.ylabel("Accuracy", fontsize=12)
            plt.title(f"{self.model_name.upper()} - {self.method.upper()} CV Accuracy", fontsize=14)
            plt.xticks(folds)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "accuracy_per_fold.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # 2. Metrics comparison
            if "precision" in df.columns:
                plt.figure(figsize=(12, 6))
                metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
                x = np.arange(len(folds))
                width = 0.2

                for i, metric in enumerate(metrics_to_plot):
                    if metric in df.columns:
                        plt.bar(x + i * width, df[metric], width, label=metric.capitalize())

                plt.xlabel("Fold", fontsize=12)
                plt.ylabel("Score", fontsize=12)
                plt.title(f"{self.model_name.upper()} - {self.method.upper()} Metrics", fontsize=14)
                plt.xticks(x + width * 1.5, folds)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.05)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "metrics_comparison.png"), dpi=200, bbox_inches="tight")
                plt.close()

            # 3. Confusion matrix (best fold)
            if results.best_fold and "confusion_matrix" in results.best_fold["metrics"]:
                cm = np.array(results.best_fold["metrics"]["confusion_matrix"])
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                )
                plt.xlabel("Predicted", fontsize=12)
                plt.ylabel("True", fontsize=12)
                plt.title(f"Best Fold Confusion Matrix (Fold {results.best_fold['fold'] + 1})", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "best_fold_confusion_matrix.png"), dpi=200, bbox_inches="tight")
                plt.close()
        except Exception as e:
            print(f"⚠️ Plot generation warning: {e}")

        if self.verbose:
            print(f"✅ Reports saved:")
            print(f"   CSV: {csv_path}")
            print(f"   JSON: {json_path}")

        return output_path


# ================================================================
# ✅ FACTORY FUNCTION
# ================================================================

def create_cross_validator(
        model_name: str = "resnet18",
        method: str = "byol",
        n_folds: int = 5,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
        save_dir: str = None,
        pretrained_path: str = None,
        **kwargs,
) -> SSLCrossValidation:
    """Factory function to create CV instance."""
    return SSLCrossValidation(
        model_name=model_name,
        method=method,
        n_folds=n_folds,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
        save_dir=save_dir,
        pretrained_path=pretrained_path,
        **kwargs,
    )


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing SSL Cross-Validation (NO-COLLAPSE EDITION)")
    print("=" * 80)

    dummy_data = torch.randn(100, 3, 224, 224)
    dummy_labels = torch.randint(0, 4, (100,))

    cv = create_cross_validator(
        model_name="resnet18",
        method="byol",
        n_folds=5,
        batch_size=32,
        verbose=True,
    )

    results = cv.run(dummy_data, dummy_labels)
    cv.generate_report(results)

    print("\n✅ Cross-Validation tests passed!")
    print("=" * 80)