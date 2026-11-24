# scripts/inspect_convnext.py
# -*- coding: utf-8 -*-

"""
Inspect ConvNext checkpoint — full architecture + head structure.

Usage:
    python scripts/inspect_convnext.py
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models

CKPT_PATH = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\ConvNext\convnextBEST.pth"


def print_separator(title=""):
    print("\n" + "=" * 90)
    if title:
        print(f"  {title}")
        print("=" * 90)


def main():
    print_separator("CONVNEXT CHECKPOINT INSPECTOR")
    print(f"Checkpoint: {CKPT_PATH}\n")

    if not os.path.exists(CKPT_PATH):
        print(f"❌ File not found: {CKPT_PATH}")
        sys.exit(1)

    # ---- Load checkpoint ----
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # ---- Top-level keys ----
    print_separator("TOP-LEVEL KEYS IN .pth")
    if isinstance(ckpt, dict):
        for k in ckpt.keys():
            v = ckpt[k]
            if isinstance(v, dict):
                print(f"  {k:<30s} → dict ({len(v)} entries)")
            elif isinstance(v, torch.Tensor):
                print(f"  {k:<30s} → tensor {tuple(v.shape)}")
            elif isinstance(v, (int, float, str, bool, type(None))):
                print(f"  {k:<30s} → {type(v).__name__} = {v}")
            else:
                print(f"  {k:<30s} → {type(v).__name__}")
    else:
        print(f"  ⚠️ Checkpoint is not a dict, type = {type(ckpt)}")

    # ---- Extract state_dict ----
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        print(f"\n✅ Using 'model_state_dict' ({len(state)} tensors)")
    else:
        state = ckpt
        print(f"\n✅ Using raw checkpoint as state_dict ({len(state)} tensors)")

    # ---- All keys with shapes ----
    print_separator("ALL KEYS + SHAPES (sorted)")
    keys_sorted = sorted(state.keys())
    for k in keys_sorted:
        t = state[k]
        if isinstance(t, torch.Tensor):
            print(f"  {k:<60s} {tuple(t.shape)}")

    # ---- Group by prefix ----
    print_separator("KEYS GROUPED BY TOP-LEVEL PREFIX")
    groups = {}
    for k in keys_sorted:
        prefix = k.split(".")[0]
        groups.setdefault(prefix, []).append(k)

    for prefix in sorted(groups.keys()):
        print(f"\n  🔹 {prefix}/ ({len(groups[prefix])} tensors)")
        for k in groups[prefix]:
            t = state[k]
            print(f"      {k:<55s} {tuple(t.shape)}")

    # ---- Focus on classifier (head) ----
    print_separator("HEAD / CLASSIFIER ANALYSIS")
    classifier_keys = [k for k in keys_sorted if k.startswith("classifier.")]
    if not classifier_keys:
        print("  ⚠️ No 'classifier.*' keys found")
    else:
        print(f"  Found {len(classifier_keys)} classifier keys:\n")
        for k in classifier_keys:
            t = state[k]
            print(f"    {k:<45s} {tuple(t.shape)}")

        # Detect structure
        print("\n  🔍 Structure detection:")
        has_0_weight = "classifier.0.weight" in state
        has_1_weight = "classifier.1.weight" in state
        has_2_1 = any(k.startswith("classifier.2.1.") for k in classifier_keys)
        has_2_5 = any(k.startswith("classifier.2.5.") for k in classifier_keys)
        has_5_weight = "classifier.5.weight" in state

        if has_2_1 and has_2_5:
            print("    ✅ Head = LayerNorm2d + Flatten + Sequential(...)")
            print("       classifier.0     → LayerNorm2d")
            print("       classifier.1     → Flatten")
            print("       classifier.2.1   → Linear")
            print("       classifier.2.2   → BatchNorm")
            print("       classifier.2.5   → Linear (final)")
        elif has_0_weight and has_1_weight and has_5_weight:
            print("    ✅ Head = Flatten + Linear + BatchNorm + ReLU + Dropout + Linear")
            print("       classifier.0     → Flatten")
            print("       classifier.1     → Linear")
            print("       classifier.2     → BatchNorm")
            print("       classifier.5     → Linear (final)")
        else:
            print("    ⚠️ Unrecognized head structure")

    # ---- Try building ConvNext and load ----
    print_separator("ATTEMPT: BUILD + LOAD ConvNext (default head)")
    try:
        model = models.convnext_tiny(weights=None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Total missing   : {len(missing)}")
        print(f"  Total unexpected: {len(unexpected)}")
        if missing:
            print("\n  Missing keys (first 20):")
            for k in missing[:20]:
                print(f"    - {k}")
        if unexpected:
            print("\n  Unexpected keys (first 20):")
            for k in unexpected[:20]:
                print(f"    + {k}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # ---- Print classifier summary ----
    print_separator("CLASSIFIER — DETAILED DUMP")
    print("  (printing repr of classifier-related keys in order)\n")
    for k in classifier_keys:
        t = state[k]
        if isinstance(t, torch.Tensor):
            stats = ""
            if t.numel() > 1:
                stats = f" | mean={t.float().mean():.4f} std={t.float().std():.4f}"
            print(f"    {k}")
            print(f"      shape = {tuple(t.shape)}{stats}")

    print_separator("DONE")


if __name__ == "__main__":
    main()
