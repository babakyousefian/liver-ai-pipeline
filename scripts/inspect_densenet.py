# scripts/inspect_densenet.py
# -*- coding: utf-8 -*-

"""
Inspect DenseNet checkpoint — full architecture + head structure.

Usage:
    python scripts/inspect_densenet.py
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models

CKPT_PATH = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\DenseNet\densenetBEST.pth"


def print_separator(title=""):
    print("\n" + "=" * 90)
    if title:
        print(f"  {title}")
        print("=" * 90)


def main():
    print_separator("DENSENET CHECKPOINT INSPECTOR")
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

    # ---- Group by top-level prefix ----
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

    # ---- Focus on classifier ----
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

        # Walk through classifier.* and infer each sub-layer
        classifier_indices = set()
        for k in classifier_keys:
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                classifier_indices.add(int(parts[1]))

        print(f"    Found {len(classifier_indices)} classifier sub-layers "
              f"(indices: {sorted(classifier_indices)})")

        # Detailed walk
        for idx in sorted(classifier_indices):
            prefix = f"classifier.{idx}."
            sub_keys = [k for k in classifier_keys if k.startswith(prefix)]
            if not sub_keys:
                continue
            sample = sub_keys[0]
            t = state[sample]

            # Infer layer type
            if "weight" in sample and len(t.shape) == 2:
                print(f"    classifier.{idx} → Linear  "
                      f"in={t.shape[1]} out={t.shape[0]}")
            elif "weight" in sample and len(t.shape) == 1:
                print(f"    classifier.{idx} → BatchNorm "
                      f"features={t.shape[0]}")
            elif "running_mean" in sample:
                print(f"    classifier.{idx} → BatchNorm "
                      f"features={state[f'classifier.{idx}.running_mean'].shape[0]}")
            else:
                print(f"    classifier.{idx} → ?  keys={len(sub_keys)}")

    # ---- Try building DenseNet and load ----
    print_separator("ATTEMPT: BUILD + LOAD DenseNet-121 (default head)")
    try:
        model = models.densenet121(weights=None)
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

    print_separator("DONE")


if __name__ == "__main__":
    main()

