# tools/inspect_alexnet.py
# -*- coding: utf-8 -*-
"""
Deep inspection of AlexNet checkpoint.
Target: C:\\Users\\liana\\Documents\\pythonProjects\\FinalTasks\\finalReports\\bestModels\\AlexNet\\alexnetBEST.pth
"""

import os
from collections import OrderedDict
import torch
import torch.nn as nn

CKPT_PATH = (
    r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"
    r"\bestModels\AlexNet\alexnetBEST.pth"
)


def extract_state_dict(ckpt_obj):
    """Extract state_dict from various checkpoint formats."""
    if isinstance(ckpt_obj, nn.Module):
        return ckpt_obj.state_dict(), "nn.Module", {}
    if isinstance(ckpt_obj, dict):
        meta = {k: v for k, v in ckpt_obj.items()
                if k not in ("state_dict", "model_state_dict", "model", "weights", "net")}
        for key in ("state_dict", "model_state_dict", "model", "weights", "net"):
            if key in ckpt_obj:
                inner = ckpt_obj[key]
                if isinstance(inner, nn.Module):
                    return inner.state_dict(), f"dict[{key} -> nn.Module]", meta
                if isinstance(inner, dict):
                    return inner, f"dict[{key}]", meta
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj, "raw_state_dict", {}
    return {}, type(ckpt_obj).__name__, {}


def detect_alexnet_head(state_dict):
    """
    Analyze the classifier (head) structure from state_dict keys.
    AlexNet's `classifier` is a Sequential. Default torchvision:
        classifier.0 = Dropout
        classifier.1 = Linear(256*6*6=9216, 4096)
        classifier.2 = ReLU
        classifier.3 = Dropout
        classifier.4 = Linear(4096, 4096)
        classifier.5 = ReLU
        classifier.6 = Linear(4096, num_classes)
    But we need to detect the CUSTOM head from this checkpoint.
    """
    print("\n--- Classifier (Head) Analysis ---")
    classifier_keys = [k for k in state_dict.keys() if k.startswith("classifier.")]
    if not classifier_keys:
        print("  ❌ No classifier keys found!")
        return None

    # Group by layer index
    layers = {}
    for k in classifier_keys:
        parts = k.split(".")
        if len(parts) >= 2:
            idx = parts[1]
            if idx not in layers:
                layers[idx] = []
            layers[idx].append(k)

    # Sort by numeric index
    def sort_key(idx):
        try:
            return int(idx)
        except ValueError:
            return 999

    print(f"  Found {len(layers)} classifier sub-modules:")
    head_info = []
    for idx in sorted(layers.keys(), key=sort_key):
        keys = layers[idx]
        # Get shape from first key
        first_key = keys[0]
        shape = tuple(state_dict[first_key].shape) if isinstance(state_dict[first_key], torch.Tensor) else None
        # Detect type
        if "weight" in first_key and "bias" in " ".join(keys):
            if shape and len(shape) == 2:
                mtype = "Linear"
                in_f, out_f = shape[1], shape[0]
            elif shape and len(shape) == 1:
                mtype = "Norm/BN"
                in_f, out_f = shape[0], shape[0]
            else:
                mtype = "Unknown"
                in_f, out_f = None, None
        elif "running_mean" in " ".join(keys):
            mtype = "BatchNorm1d"
            in_f = out_f = shape[0] if shape else None
        elif "num_batches_tracked" in " ".join(keys):
            continue  # Skip counter
        else:
            mtype = "Unknown"
            in_f, out_f = None, None
        head_info.append((idx, mtype, in_f, out_f, first_key, shape))
        print(f"    classifier.{idx}: {mtype:12s} | in={in_f}, out={out_f} | {first_key} -> {shape}")

    return head_info


def try_build_alexnet(state_dict):
    """Try to build AlexNet with various head configurations."""
    try:
        import torchvision.models as tv_models
    except ImportError:
        print("  ❌ torchvision not installed")
        return None

    # Detect num_classes from last classifier layer
    num_classes = 4
    classifier_keys = [k for k in state_dict.keys() if k.startswith("classifier.")]
    for k in sorted(classifier_keys, key=lambda x: int(x.split(".")[1]) if x.split(".")[1].isdigit() else 999, reverse=True):
        if "weight" in k and len(state_dict[k].shape) == 2:
            num_classes = state_dict[k].shape[0]
            break

    print(f"\n  Detected num_classes = {num_classes}")

    # Build base model
    try:
        model = tv_models.alexnet(weights=None, num_classes=num_classes)
    except Exception as e:
        print(f"  ❌ Could not build alexnet: {e}")
        return None

    # Check if default head matches
    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())
    missing = model_keys - sd_keys
    unexpected = sd_keys - model_keys
    score = len(missing) + len(unexpected)

    if score == 0:
        print(f"\n  ✅ EXACT MATCH with torchvision.alexnet (default head)")
        return "torchvision.alexnet (default)", model, 0, 0

    # If not exact, try with custom head
    print(f"\n  ⚠️ Default torchvision.alexnet does NOT match (missing={len(missing)}, unexpected={len(unexpected)})")
    print(f"  → Need to build CUSTOM head based on detected structure")

    return "custom", model, len(missing), len(unexpected)


def build_custom_head_from_analysis(state_dict):
    """
    Build a Sequential head matching the EXACT structure from state_dict.
    """
    from collections import OrderedDict

    # Analyze classifier structure
    classifier_keys = [k for k in state_dict.keys() if k.startswith("classifier.")]
    layers_dict = {}
    for k in classifier_keys:
        parts = k.split(".")
        idx = parts[1]
        if idx not in layers_dict:
            layers_dict[idx] = {}
        key_type = parts[-1]  # weight, bias, running_mean, etc.
        layers_dict[idx][key_type] = state_dict[k]

    # Build Sequential
    head_layers = OrderedDict()
    for idx in sorted(layers_dict.keys(), key=lambda x: int(x)):
        info = layers_dict[idx]
        if "weight" in info and "bias" in info:
            shape = info["weight"].shape
            if len(shape) == 2:
                # Linear
                head_layers[idx] = nn.Linear(shape[1], shape[0])
            elif len(shape) == 1:
                # Could be LayerNorm or BatchNorm
                if "running_mean" in info:
                    head_layers[idx] = nn.BatchNorm1d(shape[0])
                else:
                    head_layers[idx] = nn.LayerNorm(shape[0])
        elif "running_mean" in info:
            shape = info["running_mean"].shape
            head_layers[idx] = nn.BatchNorm1d(shape[0])
        elif not info:
            # Empty - this is ReLU or Dropout (no params)
            # Can't infer which one from state_dict, so skip
            pass

    return nn.Sequential(head_layers)


def main():
    print("=" * 80)
    print("ALEXNET — DEEP INSPECTOR")
    print("=" * 80)
    print(f"Path   : {CKPT_PATH}")
    print(f"Exists : {os.path.exists(CKPT_PATH)}")
    if not os.path.exists(CKPT_PATH):
        return

    size_mb = os.path.getsize(CKPT_PATH) / (1024 * 1024)
    print(f"Size   : {size_mb:.2f} MB")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict, container_type, meta = extract_state_dict(ckpt)

    print(f"\nType   : {container_type}")
    if isinstance(ckpt, dict):
        print(f"Top keys: {list(ckpt.keys())}")
    if meta:
        print("\nMetadata / Extra keys:")
        for k, v in meta.items():
            preview = str(v)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"  - {k}: {preview}")

    total_tensors = len(state_dict)
    total_params = sum(v.numel() for v in state_dict.values()
                       if isinstance(v, torch.Tensor))
    print(f"\nState dict tensors : {total_tensors}")
    print(f"Total parameters   : {total_params:,}")

    # ── Detect num_classes ──
    num_classes = 4
    for k in sorted([k for k in state_dict if k.startswith("classifier.")],
                    key=lambda x: int(x.split(".")[1]) if x.split(".")[1].isdigit() else 999,
                    reverse=True):
        if "weight" in k and len(state_dict[k].shape) == 2:
            num_classes = state_dict[k].shape[0]
            break
    print(f"Detected num_classes: {num_classes}")

    # ── Detect architecture params ──
    print(f"\n--- Architecture Params ---")
    # Features: check first conv
    for k in state_dict:
        if k.startswith("features.0.weight"):
            print(f"  features.0 (Conv2d): {tuple(state_dict[k].shape)}")
            break
    # Check for BatchNorm in features
    has_bn = any("features" in k and "running_mean" in k for k in state_dict)
    print(f"  has BatchNorm in features: {has_bn}")

    # ── Head analysis ──
    detect_alexnet_head(state_dict)

    # ── First 15 keys ──
    print(f"\n--- First 15 Keys ---")
    for i, (k, v) in enumerate(state_dict.items()):
        if i >= 15:
            break
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
        print(f"  {k:70s} | {shape}")

    # ── Last 15 keys ──
    print(f"\n--- Last 15 Keys (Classifier) ---")
    items = list(state_dict.items())
    for k, v in items[-15:]:
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
        print(f"  {k:70s} | {shape}")

    # ── Try to build ──
    print(f"\n--- Attempting Standard AlexNet Build ---")
    result = try_build_alexnet(state_dict)
    if result:
        name, model, missing, unexpected = result
        print(f"  Result: {name}")
        if missing > 0 or unexpected > 0:
            print(f"  Missing: {missing}, Unexpected: {unexpected}")

            # Print detailed diff
            model_keys = set(model.state_dict().keys())
            sd_keys = set(state_dict.keys())
            only_in_sd = sd_keys - model_keys
            only_in_model = model_keys - sd_keys
            print(f"\n  Keys ONLY in checkpoint (unexpected in default):")
            for k in sorted(only_in_sd)[:20]:
                print(f"    + {k}")
            print(f"\n  Keys ONLY in default model (missing from checkpoint):")
            for k in sorted(only_in_model)[:20]:
                print(f"    - {k}")

    # ── Suggestion ──
    print(f"\n--- SUGGESTION FOR CONVNEXT→ALEXNET KD ---")
    print(f"  Use this code in `get_alexnet_student()`:")
    print(f"  from torchvision.models import alexnet")
    print(f"  from collections import OrderedDict")
    print(f"  model = alexnet(weights=None, num_classes={num_classes})")
    print(f"  # Then replace classifier if needed")
    print(f"  model.classifier = nn.Sequential(OrderedDict([")
    print(f"      # ... (based on analysis above)")
    print(f"  ]))")


if __name__ == "__main__":
    main()


