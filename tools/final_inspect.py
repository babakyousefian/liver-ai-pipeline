
# ================================================================
# CHECKPOINT INSPECTOR — ConvNeXt + KD Final Ensemble
# Version: 1.0
# Purpose: Full structural inspection before model fusion
# ================================================================

import os
import json
import torch
import hashlib
from collections import OrderedDict

# ================================================================
# 1. CONFIGURATION
# ================================================================

CHECKPOINTS = {
    "ConvNeXt_Tiny": (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\AI_Comparisons\ConvNext\convnext"
        r"\ConvNext_best.pth"
    ),

    "KD_Base_Model": (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\selfSupervisedLearning"
        r"\final_models\final_model.pth"
    ),

    # مسیر دقیق سه Seed را از پوشه KDFinal وارد کن
    "KD_Seed_42": r"PUT_SEED_42_PATH_HERE",

    "KD_Seed_123": r"PUT_SEED_123_PATH_HERE",

    "KD_Seed_2024": r"PUT_SEED_2024_PATH_HERE",

    # در صورت وجود
    "KD_Best_Ensemble": (
        r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
        r"\finalReports\AI_Comparisons"
        r"\best_ensemble_models.pth"
    ),
}

OUTPUT_FILE = "checkpoint_inspection_report.txt"

# ================================================================
# 2. HELPERS
# ================================================================

def print_line(char="=", length=90):
    print(char * length)


def safe_json(value):
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def tensor_info(tensor):
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "requires_grad": tensor.requires_grad,
    }


def calculate_file_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def count_parameters(state_dict):
    total = 0
    trainable = 0

    for value in state_dict.values():
        if torch.is_tensor(value):
            total += value.numel()

            if value.is_floating_point():
                trainable += value.numel()

    return total, trainable


def extract_state_dict(checkpoint):
    """
    Detect common checkpoint structures.
    Does not modify the original checkpoint.
    """

    if not isinstance(checkpoint, dict):
        return None, None

    possible_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "student_state_dict",
        "best_model_state_dict",
        "ema_state_dict",
        "weights",
    ]

    for key in possible_keys:
        candidate = checkpoint.get(key)

        if isinstance(candidate, (dict, OrderedDict)):
            if any(torch.is_tensor(v) for v in candidate.values()):
                return candidate, key

    # Raw state_dict
    if any(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint, "RAW_STATE_DICT"

    return None, None


def print_nested_structure(obj, prefix="", depth=0, max_depth=3):
    if depth > max_depth:
        print(prefix + "... (maximum depth reached)")
        return

    if isinstance(obj, dict):
        for key, value in obj.items():

            if torch.is_tensor(value):
                print(
                    f"{prefix}{key}: "
                    f"Tensor(shape={tuple(value.shape)}, "
                    f"dtype={value.dtype})"
                )

            elif isinstance(value, (dict, list, tuple)):
                print(f"{prefix}{key}: {type(value).__name__}")
                print_nested_structure(
                    value,
                    prefix + "    ",
                    depth + 1,
                    max_depth
                )

            else:
                print(
                    f"{prefix}{key}: "
                    f"{str(value)[:250]}"
                )

    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}Length: {len(obj)}")

        for i, value in enumerate(obj[:5]):
            print(f"{prefix}[{i}]")
            print_nested_structure(
                value,
                prefix + "    ",
                depth + 1,
                max_depth
            )

# ================================================================
# 3. CHECKPOINT INSPECTION
# ================================================================

def inspect_checkpoint(name, path):

    print_line()
    print(f"CHECKPOINT: {name}")
    print_line()

    if not path or "PUT_" in path:
        print("[SKIPPED] Please provide the actual checkpoint path.")
        return

    if not os.path.isfile(path):
        print("[NOT FOUND]")
        print("Path:", path)
        return

    print("Path:", path)
    print("File size:", round(os.path.getsize(path) / (1024**2), 3), "MB")

    try:
        print("SHA256:", calculate_file_hash(path))
    except Exception as e:
        print("Hash error:", repr(e))

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    except Exception as e:
        print("[LOAD ERROR]", repr(e))
        return

    print("\n[OBJECT INFORMATION]")
    print("Type:", type(checkpoint))

    if not isinstance(checkpoint, dict):
        print("Object value:", str(checkpoint)[:1000])
        return

    # ------------------------------------------------------------
    # TOP LEVEL
    # ------------------------------------------------------------

    print("\n[TOP LEVEL KEYS]")

    for key, value in checkpoint.items():

        if torch.is_tensor(value):
            print(
                f"{key}: Tensor "
                f"shape={tuple(value.shape)}, dtype={value.dtype}"
            )

        elif isinstance(value, dict):
            print(f"{key}: dict ({len(value)} keys)")

        elif isinstance(value, (list, tuple)):
            print(f"{key}: {type(value).__name__}, len={len(value)}")

        else:
            print(f"{key}: {str(value)[:200]}")

    # ------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------

    print("\n[METADATA]")

    metadata_keys = [
        "epoch",
        "best_epoch",
        "seed",
        "accuracy",
        "best_acc",
        "val_acc",
        "test_acc",
        "test_metrics",
        "val_metrics",
        "test_metrics_tta",
        "config",
        "metadata",
        "class_names",
        "classes",
        "num_classes",
        "model_name",
        "architecture",
        "backbone",
        "optimizer",
        "scheduler",
    ]

    for key in metadata_keys:
        if key in checkpoint:
            print(f"\n{key}:")
            print(safe_json(checkpoint[key])[:5000])

    # ------------------------------------------------------------
    # STATE DICT
    # ------------------------------------------------------------

    state_dict, state_key = extract_state_dict(checkpoint)

    print("\n[STATE DICT]")

    if state_dict is None:
        print("No recognizable state_dict found.")
        print("Nested structure:")
        print_nested_structure(checkpoint)
        return

    print("Detected key:", state_key)
    print("Number of entries:", len(state_dict))

    total, floating = count_parameters(state_dict)

    print("Total tensor elements:", f"{total:,}")
    print("Floating tensor elements:", f"{floating:,}")

    # ------------------------------------------------------------
    # ALL LAYER NAMES AND SHAPES
    # ------------------------------------------------------------

    print("\n[ALL STATE DICT PARAMETERS]")

    for key, value in state_dict.items():

        if torch.is_tensor(value):
            print(
                f"{key:<75} "
                f"{str(tuple(value.shape)):<25} "
                f"{str(value.dtype)}"
            )
        else:
            print(f"{key:<75} {type(value)}")

    # ------------------------------------------------------------
    # FINAL LAYERS
    # ------------------------------------------------------------

    print("\n[CLASSIFIER / FINAL LAYER CANDIDATES]")

    for key, value in state_dict.items():

        if not torch.is_tensor(value):
            continue

        lower_key = key.lower()

        if any(
            token in lower_key
            for token in [
                "classifier",
                "fc.",
                "head",
                "final",
                "linear",
                "logits",
            ]
        ):
            print(key, tuple(value.shape))

    # ------------------------------------------------------------
    # POSSIBLE CLASS COUNT
    # ------------------------------------------------------------

    print("\n[OUTPUT DIMENSION CANDIDATES]")

    for key, value in state_dict.items():

        if not torch.is_tensor(value):
            continue

        if value.ndim == 2 and any(
            token in key.lower()
            for token in ["classifier", "fc", "head", "final"]
        ):
            print(
                f"{key}: possible output dimension = {value.shape[0]}"
            )

    # ------------------------------------------------------------
    # NESTED STRUCTURE
    # ------------------------------------------------------------

    print("\n[OTHER NESTED CHECKPOINT INFORMATION]")

    for key, value in checkpoint.items():

        if key not in [
            state_key,
            "model_state_dict",
            "state_dict",
        ] and isinstance(value, (dict, list, tuple)):

            print(f"\nNested item: {key}")
            print_nested_structure(
                value,
                prefix="    ",
                max_depth=2
            )

# ================================================================
# 4. RUN INSPECTION
# ================================================================

import sys
from contextlib import redirect_stdout

with open(OUTPUT_FILE, "w", encoding="utf-8") as report:

    original_stdout = sys.stdout

    try:
        sys.stdout = report

        print_line()
        print("FULL CHECKPOINT INSPECTION REPORT")
        print("ConvNeXt + KD Final Ensemble")
        print_line()

        for name, path in CHECKPOINTS.items():
            inspect_checkpoint(name, path)

        print_line()
        print("INSPECTION FINISHED")
        print_line()

    finally:
        sys.stdout = original_stdout

# ================================================================
# 5. CONSOLE SUMMARY
# ================================================================

print("\nInspection completed.")
print("Report saved to:", os.path.abspath(OUTPUT_FILE))
print("File size:", round(os.path.getsize(OUTPUT_FILE) / 1024, 2), "KB")

print("\nPlease send me the complete contents of:")
print(OUTPUT_FILE)
