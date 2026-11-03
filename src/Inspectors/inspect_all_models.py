# inspect_all_models.py
# -*- coding: utf-8 -*-

"""
Comprehensive Model Inspector (v1.0)

Inspects all models used in the Liver AI Pipeline:
- ResNet18, ResNet50
- EfficientNet-B0, EfficientNet-B1
- SSL Encoders (BYOL, SimCLR, MAE) for all 4 architectures
- Fine-Tuned Ensemble Architects
- Knowledge Distillation Student

For each model, prints:
- Top-level modules
- Dropout layers (with p value)
- BatchNorm2d layers (with momentum)
- Final layers (for identifying classifier head)
- Total parameters

Author: Babak Yousefian
Version: 1.0.0
"""

import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn


# ================================================================
# ✅ ANSI Colors (fallback if Rich not available)
# ================================================================

class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


# ================================================================
# ✅ Pretty Printing Helpers
# ================================================================

def print_header(title: str, char: str = "=", width: int = 80):
    """Print a section header."""
    print()
    print(f"{C.BOLD}{C.CYAN}{char * width}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  {title}{C.END}")
    print(f"{C.BOLD}{C.CYAN}{char * width}{C.END}")


def print_subheader(title: str, char: str = "-", width: int = 80):
    """Print a subsection header."""
    print()
    print(f"{C.YELLOW}{char * width}{C.END}")
    print(f"{C.YELLOW}  {title}{C.END}")
    print(f"{C.YELLOW}{char * width}{C.END}")


def print_key_value(key: str, value: Any, indent: int = 2):
    """Print a key-value pair."""
    indent_str = " " * indent
    print(f"{indent_str}{C.GREEN}{key:<30}{C.END} : {C.BOLD}{value}{C.END}")


def print_list(items: List[str], indent: int = 2):
    """Print a list of items."""
    indent_str = " " * indent
    for item in items:
        print(f"{indent_str}  • {item}")


# ================================================================
# ✅ Model Inspector
# ================================================================

def inspect_model(model: nn.Module, model_name: str):
    """Inspect a single model and print all details."""

    print_subheader(f"📦 {model_name}")

    # Top-level modules
    print(f"  {C.BOLD}Top-Level Modules:{C.END}")
    top_level = []
    for name, mod in model.named_children():
        mod_type = type(mod).__name__
        top_level.append(f"{name}: {mod_type}")
    print_list(top_level, indent=2)

    # Dropout layers
    print()
    print(f"  {C.BOLD}Dropout Layers:{C.END}")
    dropout_layers = []
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Dropout):
            dropout_layers.append(f"{n}: Dropout(p={mod.p}, inplace={mod.inplace})")
    if dropout_layers:
        print_list(dropout_layers, indent=2)
    else:
        print(f"    {C.DIM}(none){C.END}")

    # BatchNorm2d layers
    print()
    print(f"  {C.BOLD}BatchNorm2d Layers:{C.END}")
    bn2d_layers = []
    for n, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            bn2d_layers.append(f"{n}: momentum={mod.momentum}, eps={mod.eps}")
    if bn2d_layers:
        print_list(bn2d_layers[:5], indent=2)
        if len(bn2d_layers) > 5:
            print(f"    {C.DIM}... and {len(bn2d_layers) - 5} more{C.END}")
        print(f"    {C.BOLD}TOTAL: {len(bn2d_layers)} BatchNorm2d layers{C.END}")
    else:
        print(f"    {C.DIM}(none){C.END}")

    # BatchNorm1d layers
    bn1d_count = sum(1 for _ in model.modules() if isinstance(_, nn.BatchNorm1d))
    if bn1d_count > 0:
        print(f"    {C.DIM}(+ {bn1d_count} BatchNorm1d layers){C.END}")

    # Final layers (last 8 params)
    print()
    print(f"  {C.BOLD}Final Layers (last 8 params):{C.END}")
    params = list(model.named_parameters())
    for n, p in params[-8:]:
        print(f"    {n}: {tuple(p.shape)}")

    # Try to identify classifier head
    print()
    print(f"  {C.BOLD}Classifier Head Detection:{C.END}")
    classifier_names = ["fc", "classifier", "_fc", "head", "linear", "out"]
    found_classifier = False
    for cname in classifier_names:
        if hasattr(model, cname):
            mod = getattr(model, cname)
            print(f"    ✅ model.{cname}: {type(mod).__name__}")
            if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
                print(f"       in_features={mod.in_features}, out_features={mod.out_features}")
            found_classifier = True

    # Check _fc specifically (for EfficientNet)
    if hasattr(model, "_fc"):
        mod = model._fc
        print(f"    ✅ model._fc: {type(mod).__name__}")

    # Check classifier in Sequential
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        print(f"    ✅ model.classifier is Sequential:")
        for i, sub in enumerate(model.classifier):
            print(f"       [{i}] {type(sub).__name__}")

    if not found_classifier:
        print(f"    {C.DIM}⚠️ No standard classifier name found{C.END}")

    # Total parameters
    print()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {C.BOLD}Parameters:{C.END}")
    print_key_value("Total Parameters", f"{total_params:,}", indent=4)
    print_key_value("Trainable Parameters", f"{trainable_params:,}", indent=4)
    print_key_value("Frozen Parameters", f"{total_params - trainable_params:,}", indent=4)


# ================================================================
# ✅ Model Creators
# ================================================================

def create_resnet18_model():
    """Create ResNet18."""
    from src.models.resnet.resnet18 import create_resnet18
    return create_resnet18(num_classes=4, pretrained=False, verbose=False)


def create_resnet50_model():
    """Create ResNet50."""
    from src.models.resnet.resnet50 import create_resnet50
    return create_resnet50(num_classes=4, pretrained=False, verbose=False)


def create_b0_model():
    """Create EfficientNet-B0."""
    from src.models.efficientNet.B0 import create_efficientnet_b0
    return create_efficientnet_b0(num_classes=4, pretrained=False, verbose=False)


def create_b1_model():
    """Create EfficientNet-B1."""
    from src.models.efficientNet.B1 import create_efficientnet_b1
    return create_efficientnet_b1(num_classes=4, pretrained=False, verbose=False)


# ================================================================
# ✅ Load Architects From Checkpoints
# ================================================================

def load_model_from_checkpoint(path: str, model_type: str) -> Optional[nn.Module]:
    """
    Load a model from a checkpoint path.

    Args:
        path: Path to the .pth file
        model_type: 'resnet18', 'resnet50', 'b0', 'b1'

    Returns:
        Model instance or None if failed
    """
    if not os.path.exists(path):
        return None

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Extract state dict
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "encoder_state_dict" in ckpt:
                state_dict = ckpt["encoder_state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Create model based on type
        if model_type == "resnet18":
            model = create_resnet18_model()
        elif model_type == "resnet50":
            model = create_resnet50_model()
        elif model_type == "b0":
            model = create_b0_model()
        elif model_type == "b1":
            model = create_b1_model()
        else:
            return None

        # Load weights
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            return None

        return model

    except Exception as e:
        return None


# ================================================================
# ✅ Checkpoint Paths (Auto-Detect)
# ================================================================

def get_checkpoint_paths() -> Dict[str, List[Tuple[str, str]]]:
    """
    Get all checkpoint paths to inspect.

    Returns:
        Dict with keys:
        - 'architectures': [(name, path), ...]  ← fresh model creation
        - 'ssl_checkpoints': [(name, path, model_type), ...]
        - 'transferred': [(name, path, model_type), ...]
        - 'finetuned': [(name, path, model_type), ...]
    """
    project_root = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
    ssl_root = os.path.join(project_root, "../../finalReports", "selfSupervisedLearning")

    result = {
        "ssl_checkpoints": [],
        "transferred": [],
        "finetuned": [],
        "ensemble_kd": [],
    }

    # ============================================================
    # SSL Checkpoints
    # ============================================================
    ssl_root_paths = {
        # ResNet18
        "resnet18_byol": (
            os.path.join(ssl_root, "ResNet", "18", "SSL_checkpoints", "architecture", "byol",
                         "resnet18_best_update_byol.pth"),
            "resnet18",
        ),
        "resnet18_simclr": (
            os.path.join(ssl_root, "ResNet", "18", "SSL_checkpoints", "architecture", "simclr",
                         "resnet18_best_update_simclr.pth"),
            "resnet18",
        ),
        "resnet18_mae": (
            os.path.join(ssl_root, "ResNet", "18", "SSL_checkpoints", "architecture", "mae",
                         "resnet18_best_update_mae.pth"),
            "resnet18",
        ),
        # EfficientNet-B0
        "b0_byol": (
            os.path.join(ssl_root, "EfficientNet", "B0", "SSL_checkpoints", "architecture", "byol",
                         "b0_ssl_byol_best.pth"),
            "b0",
        ),
        "b0_simclr": (
            os.path.join(ssl_root, "EfficientNet", "B0", "SSL_checkpoints", "architecture", "simclr",
                         "b0_ssl_simclr_best.pth"),
            "b0",
        ),
        "b0_mae": (
            os.path.join(ssl_root, "EfficientNet", "B0", "SSL_checkpoints", "architecture", "mae",
                         "b0_ssl_mae_best.pth"),
            "b0",
        ),
    }

    for name, (path, mtype) in ssl_root_paths.items():
        if os.path.exists(path):
            result["ssl_checkpoints"].append((name, path, mtype))

    # ============================================================
    # Transferred Architects
    # ============================================================
    transferred_paths = {
        "resnet50_transferred_byol": (
            os.path.join(ssl_root, "ResNet", "weightTransformer",
                         "resnet50_transferred_from_byol.pth"),
            "resnet50",
        ),
        "resnet50_transferred_simclr": (
            os.path.join(ssl_root, "ResNet", "weightTransformer",
                         "resnet50_transferred_from_simclr.pth"),
            "resnet50",
        ),
        "resnet50_transferred_mae": (
            os.path.join(ssl_root, "ResNet", "weightTransformer",
                         "resnet50_transferred_from_mae.pth"),
            "resnet50",
        ),
        "b1_transferred_byol": (
            os.path.join(ssl_root, "EfficientNet", "weightTransformer",
                         "b1_transferred_from_byol.pth"),
            "b1",
        ),
        "b1_transferred_simclr": (
            os.path.join(ssl_root, "EfficientNet", "weightTransformer",
                         "b1_transferred_from_simclr.pth"),
            "b1",
        ),
        "b1_transferred_mae": (
            os.path.join(ssl_root, "EfficientNet", "weightTransformer",
                         "b1_transferred_from_mae.pth"),
            "b1",
        ),
    }

    for name, (path, mtype) in transferred_paths.items():
        if os.path.exists(path):
            result["transferred"].append((name, path, mtype))

    # ============================================================
    # Fine-Tuned Architects
    # ============================================================
    finetuned_paths = {
        # ResNet50 Fine-Tuned
        "resnet50_finetuned_from_simclr_final": (
            os.path.join(ssl_root, "final_models",
                         "resnet50_finetuned_from_simclr_final.pth"),
            "resnet50",
        ),
        "resnet50_finetuned_from_byol_final": (
            os.path.join(ssl_root, "final_models",
                         "resnet50_finetuned_from_byol_final.pth"),
            "resnet50",
        ),
        "resnet50_finetuned_from_mae_final": (
            os.path.join(ssl_root, "final_models",
                         "resnet50_finetuned_from_mae_final.pth"),
            "resnet50",
        ),
        # B1 Fine-Tuned
        "b1_finetuned_final": (
            os.path.join(ssl_root, "SSL_Reporter", "b1_finetune",
                         "b1_finetuned_final.pth"),
            "b1",
        ),
    }

    for name, (path, mtype) in finetuned_paths.items():
        if os.path.exists(path):
            result["finetuned"].append((name, path, mtype))

    # ============================================================
    # KD Ensemble Model
    # ============================================================
    kd_paths = {
        "final_model_kd": (
            os.path.join(ssl_root, "final_models", "final_model.pth"),
            "resnet50",  # KD student is ResNet50
        ),
        "ensemble_kd_student_final": (
            os.path.join(ssl_root, "final_models", "ensemble_kd_student_final.pth"),
            "resnet50",
        ),
    }

    for name, (path, mtype) in kd_paths.items():
        if os.path.exists(path):
            result["ensemble_kd"].append((name, path, mtype))

    return result


# ================================================================
# ✅ Main Function
# ================================================================

def main():
    """Inspect all models."""
    print_header("🔍 COMPREHENSIVE MODEL INSPECTOR", char="═")
    print(f"{C.BOLD}Project: Liver AI Pipeline{C.END}")
    print(f"{C.DIM}Inspecting all architectures, SSL encoders, transferred and fine-tuned models.{C.END}")

    # ============================================================
    # SECTION 1: Fresh Architectures
    # ============================================================
    print_header("📐 SECTION 1: Fresh Architectures (Structure Only)", char="═")

    architectures = [
        ("ResNet18", create_resnet18_model),
        ("ResNet50", create_resnet50_model),
        ("EfficientNet-B0", create_b0_model),
        ("EfficientNet-B1", create_b1_model),
    ]

    for name, creator in architectures:
        try:
            model = creator()
            inspect_model(model, name)
        except Exception as e:
            print(f"\n  {C.RED}❌ Failed to create {name}: {e}{C.END}")

    # ============================================================
    # SECTION 2: Existing Checkpoints
    # ============================================================
    print_header("📂 SECTION 2: Existing Checkpoints", char="═")

    paths = get_checkpoint_paths()

    for category, items in paths.items():
        if not items:
            continue

        category_title = {
            "ssl_checkpoints": "🔬 SSL Encoders",
            "transferred": "🔄 Transferred Architects",
            "finetuned": "🎯 Fine-Tuned Architects",
            "ensemble_kd": "🧠 KD Ensemble (Student)",
        }.get(category, category)

        print_header(category_title, char="─")
        print(f"  Found {len(items)} checkpoint(s)")

        for name, path, mtype in items:
            size_mb = os.path.getsize(path) / 1e6
            print(f"\n  {C.BOLD}{name}{C.END}")
            print(f"    Path: {path}")
            print(f"    Type: {mtype} | Size: {size_mb:.2f} MB")

            # Load and inspect
            model = load_model_from_checkpoint(path, mtype)
            if model is None:
                print(f"    {C.RED}⚠️ Could not load model from checkpoint{C.END}")
                continue

            # Print only essential info (dropout + BN) for loaded models
            print(f"    {C.BOLD}Dropout Layers:{C.END}")
            dropout_found = False
            for n, mod in model.named_modules():
                if isinstance(mod, nn.Dropout):
                    print(f"      • {n}: Dropout(p={mod.p})")
                    dropout_found = True
            if not dropout_found:
                print(f"      {C.DIM}(none){C.END}")

            bn2d_count = sum(1 for _ in model.modules() if isinstance(_, nn.BatchNorm2d))
            print(f"    {C.BOLD}BatchNorm2d Layers:{C.END} {bn2d_count}")

            # Classifier head
            print(f"    {C.BOLD}Classifier Head:{C.END}")
            for cname in ["fc", "classifier", "_fc", "head"]:
                if hasattr(model, cname):
                    print(f"      • model.{cname}: {type(getattr(model, cname)).__name__}")

    # ============================================================
    # SECTION 3: Summary
    # ============================================================
    print_header("📊 SECTION 3: Summary", char="═")

    total = sum(len(v) for v in paths.values())
    print(f"  {C.BOLD}Total Checkpoints Found:{C.END} {total}")
    print()
    for category, items in paths.items():
        if items:
            print(f"  {C.GREEN}{category:<25}{C.END}: {len(items)}")

    print()
    print(f"{C.BOLD}{C.GREEN}✅ Inspection Complete!{C.END}")
    print()


# ================================================================
# ✅ Entry Point
# ================================================================

if __name__ == "__main__":
    main()
