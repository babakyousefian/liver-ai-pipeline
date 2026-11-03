# src/pipeLine/crossValidationRunner.py
"""
Cross-Validation Runner for SSL Architects (ULTIMATE FIXED EDITION)

✅ FIXES:
- Windows-compatible paths (no more /home/bobby/...)
- No more epochs parameter (CV no longer trains SSL)
- batch_size=32 (from 16)
- kwargs forwarding for batch_size
- Uses architecture/{method}/ paths

Author: Babak Yousefian
Version: 7.0.0 (WINDOWS FIXED)
"""

import json
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

warnings.filterwarnings("ignore")

# ================================================================
# ✅ Add parent directory to path
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================================================================
# ✅ Imports
# ================================================================

from src.self_supervised.Facilities.crossValidation import create_cross_validator

# ================================================================
# ✅ Paths (Windows-compatible — absolute)
# ================================================================

# ✅ FIX: Windows-compatible path (was /home/bobby/...)
SELF_SUPERVISED_ROOT = os.path.join(
    PROJECT_ROOT, "finalReports", "selfSupervisedLearning"
)

# ResNet paths
RESNET18_SSL_DIR = os.path.join(SELF_SUPERVISED_ROOT, "ResNet", "18", "SSL_checkpoints")
RESNET_WEIGHT_TRANSFORMER_DIR = os.path.join(SELF_SUPERVISED_ROOT, "ResNet", "weightTransformer")
FINAL_MODELS_DIR = os.path.join(SELF_SUPERVISED_ROOT, "final_models")

# EfficientNet paths
EFFICIENTNET_B0_SSL_DIR = os.path.join(
    SELF_SUPERVISED_ROOT, "EfficientNet", "B0", "SSL_checkpoints"
)
EFFICIENTNET_WEIGHT_TRANSFORMER_DIR = os.path.join(
    SELF_SUPERVISED_ROOT, "EfficientNet", "weightTransformer"
)

# CV results directory
CV_RESULTS_DIR = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "crossValidation")
os.makedirs(CV_RESULTS_DIR, exist_ok=True)


# ================================================================
# ✅ Data Loader for CV
# ================================================================


def load_data_for_cv(
        image_paths: List[str],
        labels: List[int],
        target_size: int = 224,
        batch_size: int = 16,
        num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load data for Cross-Validation."""
    import cv2
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_list = []
    labels_list = []

    for img_path, label in zip(image_paths, labels):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img)
            data_list.append(img)
            labels_list.append(label)
        except Exception:
            continue

    if len(data_list) == 0:
        raise ValueError("No valid images found for CV!")

    data_tensor = torch.stack(data_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    return data_tensor, labels_tensor


def get_cv_data(target_size: int = 224) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ✅ FIX: Get data for CV with configurable target_size.
    """
    import pandas as pd

    CROPPED_DIR = os.path.join("finalReports", "cropped")
    LABELS_CSV = os.path.join("finalReports", "labels.csv")

    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)
    CLASS_NAMES = ["NL", "G1", "G2", "G3"]
    LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    df = df[df["liverLabel"].isin(CLASS_NAMES)]
    df["label"] = df["liverLabel"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    image_paths = []
    labels = []

    for idx, row in df.iterrows():
        img_name = row["liverName"]
        img_path = os.path.join(CROPPED_DIR, img_name)

        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(int(row["label"]))
        else:
            base_name = os.path.splitext(img_name)[0]
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                test_path = os.path.join(CROPPED_DIR, base_name + ext)
                if os.path.exists(test_path):
                    image_paths.append(test_path)
                    labels.append(int(row["label"]))
                    break

    if len(image_paths) == 0:
        raise ValueError("No images found for CV!")

    return load_data_for_cv(image_paths, labels, target_size=target_size)


# ================================================================
# ✅ CV Runner Functions
# ================================================================


def run_cross_validation(
        model_name: str,
        method: str,
        model_path: str,
        n_folds: int = 5,
        batch_size: int = 32,  # ✅ FIX: از 16 به 32
        device: str = "cpu",
        verbose: bool = True,
        # ✅ REMOVED: epochs (CV no longer trains SSL)
) -> Dict[str, Any]:
    """
    ✅ FIXED: Run Cross-Validation on a trained model.

    NO MORE SSL training! Just:
    1. Load pretrained encoder
    2. Extract embeddings
    3. StratifiedKFold + Logistic Regression
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"🔬 Running Cross-Validation for {model_name.upper()} - {method.upper()}")
        print("=" * 80)
        print(f"   Model Path: {model_path}")
        print(f"   Folds: {n_folds}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Method: Linear Probe (NO SSL training)")
        print("=" * 80)

    # Check if model exists
    if not os.path.exists(model_path):
        if verbose:
            print(f"❌ Model not found: {model_path}")
            print("   Skipping Cross-Validation...")
        return {"status": "failed", "error": "Model not found"}

    try:
        # Determine input_size based on model
        input_size = 240 if model_name in ["efficientnet_b1"] else 224

        # Get data
        data, labels = get_cv_data(target_size=input_size)

        if verbose:
            print(f"✅ Data loaded: {len(data)} samples (input_size={input_size})")

        # Create validator
        cv = create_cross_validator(
            model_name=model_name,
            method=method,
            n_folds=n_folds,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
            save_dir=os.path.join(CV_RESULTS_DIR, f"{model_name}_{method}"),
            pretrained_path=model_path,
            # ✅ REMOVED: epochs=epochs (no longer needed)
        )

        # Run CV
        results = cv.run(data, labels)

        # Generate report
        report_path = cv.generate_report(results)

        result_dict = {
            "status": "success",
            "model_name": model_name,
            "method": method,
            "model_path": model_path,
            "n_folds": n_folds,
            "mean_accuracy": results.mean_metrics.get("accuracy", 0),
            "std_accuracy": results.std_metrics.get("accuracy", 0),
            "mean_balanced_accuracy": results.mean_metrics.get("balanced_accuracy", 0),
            "mean_f1": results.mean_metrics.get("f1_score", 0),
            "mean_roc_auc": results.mean_metrics.get("roc_auc_mean", 0),
            "best_fold_accuracy": (
                results.best_fold["metrics"]["accuracy"] if results.best_fold else 0
            ),
            "report_path": report_path,
            "timestamp": datetime.now().isoformat(),
        }

        if verbose:
            print(f"\n✅ Cross-Validation Complete!")
            print(
                f"   Mean Accuracy: {result_dict['mean_accuracy']:.4f} ± {result_dict['std_accuracy']:.4f}"
            )
            print(f"   Mean Balanced Acc: {result_dict['mean_balanced_accuracy']:.4f}")
            print(f"   Best Fold Accuracy: {result_dict['best_fold_accuracy']:.4f}")
            print(f"   Report: {report_path}")

        return result_dict

    except Exception as e:
        if verbose:
            print(f"❌ Cross-Validation failed: {e}")
            import traceback
            traceback.print_exc()

        return {"status": "failed", "error": str(e)}


# ================================================================
# ✅ Helper: Find Model Path (uses architecture/{method}/)
# ================================================================


def _find_model_path(
        base_dir: str,
        method: str,
        prefix: str,
) -> Optional[str]:
    """
    ✅ FIX: Find model path in architecture/{method}/ first, then fallback.
    """
    method_dir = os.path.join(base_dir, "architecture", method.lower())

    # ✅ Try new paths first
    candidates = [
        os.path.join(method_dir, f"{prefix}_{method.lower()}_best.pth"),
        os.path.join(method_dir, f"{prefix}_{method.lower()}_final.pth"),
        os.path.join(method_dir, f"{prefix}_{method.lower()}_checkpoint.pth"),
        os.path.join(method_dir, f"{prefix}_{method.lower()}_last.pth"),
    ]

    # ✅ Fallback to old paths
    candidates.extend([
        os.path.join(base_dir, f"{prefix}_{method.lower()}_best.pth"),
        os.path.join(base_dir, f"{prefix}_{method.lower()}_encoder.pth"),
        os.path.join(base_dir, f"{prefix}_{method.lower()}_final.pth"),
    ])

    for p in candidates:
        if os.path.exists(p):
            return p

    return None


# ================================================================
# ✅ Specific CV Runners for Each Model (FIXED PATHS)
# ================================================================


def run_cv_resnet18_byol(device: str = "cpu", verbose: bool = True) -> Dict[str, Any]:
    """✅ FIXED: Run CV for ResNet18 BYOL."""
    model_path = _find_model_path(RESNET18_SSL_DIR, "byol", "resnet18_ssl")
    if model_path is None:
        if verbose:
            print("❌ ResNet18 BYOL model not found!")
        return {"status": "failed", "error": "Model not found"}
    return run_cross_validation(
        model_name="resnet18",
        method="byol",
        model_path=model_path,
        device=device,
        verbose=verbose,
    )


def run_cv_resnet18_simclr(device: str = "cpu", verbose: bool = True) -> Dict[str, Any]:
    """✅ FIXED: Run CV for ResNet18 SimCLR."""
    model_path = _find_model_path(RESNET18_SSL_DIR, "simclr", "resnet18_ssl")
    if model_path is None:
        if verbose:
            print("❌ ResNet18 SimCLR model not found!")
        return {"status": "failed", "error": "Model not found"}
    return run_cross_validation(
        model_name="resnet18",
        method="simclr",
        model_path=model_path,
        device=device,
        verbose=verbose,
    )


def run_cv_resnet18_mae(device: str = "cpu", verbose: bool = True) -> Dict[str, Any]:
    """✅ FIXED: Run CV for ResNet18 MAE."""
    model_path = _find_model_path(RESNET18_SSL_DIR, "mae", "resnet18_ssl")
    if model_path is None:
        if verbose:
            print("❌ ResNet18 MAE model not found!")
        return {"status": "failed", "error": "Model not found"}
    return run_cross_validation(
        model_name="resnet18",
        method="mae",
        model_path=model_path,
        device=device,
        verbose=verbose,
    )


def run_cv_efficientnet_b0_byol(device: str = "cpu", verbose: bool = True) -> Dict[str, Any]:
    """✅ FIXED: Run CV for EfficientNet-B0 BYOL."""
    model_path = _find_model_path(EFFICIENTNET_B0_SSL_DIR, "byol", "b0_ssl")
    if model_path is None:
        if verbose:
            print("❌ EfficientNet-B0 BYOL model not found!")
        return {"status": "failed", "error": "Model not found"}
    return run_cross_validation(
        model_name="efficientnet_b0",
        method="byol",
        model_path=model_path,
        device=device,
        verbose=verbose,
    )


def run_cv_efficientnet_b0_simclr(device: str = "cpu", verbose: bool = True) -> Dict[str, Any]:
    """✅ FIXED: Run CV for EfficientNet-B0 SimCLR."""
    model_path = _find_model_path(EFFICIENTNET_B0_SSL_DIR, "simclr", "b0_ssl")
    if model_path is None:
        if verbose:
            print("❌ EfficientNet-B0 SimCLR model not found!")
        return {"status": "failed", "error": "Model not found"}
    return run_cross_validation(
        model_name="efficientnet_b0",
        method="simclr",
        model_path=model_path,
        device=device,
        verbose=verbose,
    )


def run_cv_efficientnet_b0_mae(device: str = "cpu", verbose: bool = True) -> Dict[str, Any]:
    """✅ FIXED: Run CV for EfficientNet-B0 MAE."""
    model_path = _find_model_path(EFFICIENTNET_B0_SSL_DIR, "mae", "b0_ssl")
    if model_path is None:
        if verbose:
            print("❌ EfficientNet-B0 MAE model not found!")
        return {"status": "failed", "error": "Model not found"}
    return run_cross_validation(
        model_name="efficientnet_b0",
        method="mae",
        model_path=model_path,
        device=device,
        verbose=verbose,
    )


# ================================================================
# ✅ All CV Runners
# ================================================================


def run_all_cross_validations(
        device: str = "cpu", verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Run all Cross-Validations for SSL models (B0 only)."""
    if verbose:
        print("\n" + "=" * 80)
        print("🚀 Running All Cross-Validations")
        print("=" * 80)

    results = {}

    # ✅ B0 SSL only (skip transfer/finetune for now)
    cv_functions = [
        ("efficientnet_b0_byol", run_cv_efficientnet_b0_byol),
        ("efficientnet_b0_simclr", run_cv_efficientnet_b0_simclr),
        ("efficientnet_b0_mae", run_cv_efficientnet_b0_mae),
        ("resnet18_byol", run_cv_resnet18_byol),
        ("resnet18_simclr", run_cv_resnet18_simclr),
        ("resnet18_mae", run_cv_resnet18_mae),
    ]

    for name, func in cv_functions:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"📌 Running: {name}")
            print(f"{'=' * 60}")

        try:
            result = func(device=device, verbose=verbose)
            results[name] = result
        except Exception as e:
            if verbose:
                print(f"❌ {name} failed: {e}")
            results[name] = {"status": "failed", "error": str(e)}

    # Save results summary
    summary_path = os.path.join(CV_RESULTS_DIR, "all_cv_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    if verbose:
        print("\n" + "=" * 80)
        print("✅ All Cross-Validations Complete!")
        print(f"📂 Results saved: {summary_path}")
        print("=" * 80)

    return results


# ================================================================
# ✅ Integration with Main Pipeline
# ================================================================


def run_cv_after_model(
        model_name: str,
        method: str,
        model_path: str,
        device: str = "cpu",
        verbose: bool = True,
        **kwargs,  # ✅ FIX: Accept extra params (batch_size)
) -> Dict[str, Any]:
    """✅ FIXED: Run CV after model training with kwargs forwarding."""
    if verbose:
        print(f"\n🔬 Running Cross-Validation for {model_name} - {method}")

    return run_cross_validation(
        model_name=model_name,
        method=method,
        model_path=model_path,
        device=device,
        verbose=verbose,
        **kwargs,  # ✅ FIX: Forward batch_size
    )


# ================================================================
# ✅ Testing
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Cross-Validation Runner (FIXED)")
    print("=" * 80)

    print("\n📌 Testing All CVs...")
    run_all_cross_validations(device="cuda" if torch.cuda.is_available() else "cpu", verbose=True)