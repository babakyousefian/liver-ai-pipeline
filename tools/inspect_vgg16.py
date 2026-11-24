# -*- coding: utf-8 -*-
"""
Inspect VGG16 checkpoint & architecture details.
Target checkpoint: C:\\Users\\liana\\Documents\\pythonProjects\\FinalTasks\\finalReports\\bestModels\\VGG16\\vgg16BEST.pth
"""

import os
from collections import OrderedDict
import torch
import torch.nn as nn

CKPT_PATH = (
    r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"
    r"\bestModels\VGG16\vgg16BEST.pth"
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


def strip_prefix(state_dict, prefix):
    """Remove a prefix from all keys in state_dict."""
    if not prefix:
        return state_dict
    return OrderedDict(
        (k[len(prefix):] if k.startswith(prefix) else k, v)
        for k, v in state_dict.items()
    )


def compare_keys(state_dict, model):
    """
    Compare state_dict keys with model's keys WITHOUT loading weights.
    Returns (missing_keys, unexpected_keys).
    """
    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())
    missing = list(model_keys - sd_keys)
    unexpected = list(sd_keys - model_keys)
    return missing, unexpected


def try_build_vgg16(state_dict):
    """
    Try to find the best matching VGG16 architecture.
    Returns (best_label, best_model, missing, unexpected).
    """
    candidates = []

    try:
        import torchvision.models as tv_models
        for name, fn in (("tv.vgg16", tv_models.vgg16), ("tv.vgg16_bn", tv_models.vgg16_bn)):
            for num_classes in (4, 1000):
                try:
                    m = fn(weights=None, num_classes=num_classes)
                    candidates.append((f"{name}(num_classes={num_classes})", m))
                except Exception:
                    pass
    except ImportError:
        pass

    try:
        import timm
        for variant in ("vgg16", "vgg16_bn"):
            for num_classes in (4, 1000):
                try:
                    m = timm.create_model(variant, pretrained=False, num_classes=num_classes)
                    candidates.append((f"timm.{variant}(num_classes={num_classes})", m))
                except Exception:
                    pass
    except ImportError:
        pass

    prefixes = ["", "module.", "model.", "backbone.", "vgg."]

    best = None
    best_score = 10 ** 9

    for prefix in prefixes:
        sd = strip_prefix(state_dict, prefix)
        for label, model in candidates:
            missing, unexpected = compare_keys(sd, model)
            score = len(missing) + len(unexpected)

            if score < best_score:
                best_score = score
                best = (f"{label} [prefix='{prefix}'] -> "
                        f"missing={len(missing)}, unexpected={len(unexpected)}",
                        model, missing, unexpected)

            # Perfect match found
            if score == 0:
                return (f"{label} [prefix='{prefix}'] -> EXACT MATCH",
                        model, missing, unexpected)

    if best is None:
        return None, None, None, None

    return best


def main():
    print("=" * 80)
    print("VGG16 CHECKPOINT & ARCHITECTURE INSPECTOR")
    print("=" * 80)
    print(f"Path   : {CKPT_PATH}")
    print(f"Exists : {os.path.exists(CKPT_PATH)}")
    if not os.path.exists(CKPT_PATH):
        return

    size_mb = os.path.getsize(CKPT_PATH) / (1024 * 1024)
    print(f"Size   : {size_mb:.2f} MB")

    # Load checkpoint
    try:
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(CKPT_PATH, map_location="cpu")

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

    print("\n--- First 25 Parameter Shapes ---")
    for i, (k, v) in enumerate(state_dict.items()):
        if i >= 25:
            break
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
        dtype = str(v.dtype) if isinstance(v, torch.Tensor) else ""
        print(f"  {k:65s} | {str(shape):22s} | {dtype}")

    print("\n--- Last 20 Parameter Shapes (Classifier / Head) ---")
    items = list(state_dict.items())
    for k, v in items[-20:]:
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
        dtype = str(v.dtype) if isinstance(v, torch.Tensor) else ""
        print(f"  {k:65s} | {str(shape):22s} | {dtype}")

    print("\n--- Attempting Standard Architecture Reconstruction ---")
    result = try_build_vgg16(state_dict)
    if result[0] is None:
        print("❌ No matching architecture found.")
        return

    match_info, model, missing, unexpected = result
    print(f"Result : {match_info}")

    if missing:
        print(f"  Missing keys ({len(missing)}):")
        for k in missing[:10]:
            print(f"    - {k}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:10]:
            print(f"    - {k}")

    if model is not None:
        m_total = sum(p.numel() for p in model.parameters())
        m_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nReconstructed model params: total={m_total:,}, trainable={m_train:,}")

        print("\n--- Top-level Modules ---")
        for name, child in model.named_children():
            p_cnt = sum(p.numel() for p in child.parameters())
            print(f"  .{name:20s} -> {child.__class__.__name__:25s} ({p_cnt:,} params)")

        try:
            with torch.no_grad():
                out = model(torch.randn(1, 3, 224, 224))
            print(f"\nDummy forward (1, 3, 224, 224) -> output shape: {tuple(out.shape)}")
        except Exception as e:
            print(f"\nDummy forward failed: {e}")


if __name__ == "__main__":
    main()