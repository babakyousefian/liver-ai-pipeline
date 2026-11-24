# tools/inspect_cnnlstm.py
# -*- coding: utf-8 -*-
"""
Inspect CNNLSTM checkpoint to understand exact architecture.
Run: python tools/inspect_cnnlstm.py
"""

import os
import json
import torch
import torch.nn as nn
from collections import OrderedDict

CKPT_PATH = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\CNNLSTM\cnnlstmBEST.pth"


def inspect_checkpoint(path):
    print("=" * 80)
    print("  🔍 CNNLSTM Checkpoint Inspector")
    print("=" * 80)

    if not os.path.exists(path):
        print(f"❌ Checkpoint not found: {path}")
        return

    print(f"\n📁 File: {path}")
    print(f"📦 Size: {os.path.getsize(path) / (1024*1024):.2f} MB")

    # Load checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    print(f"\n🔎 Top-level type: {type(ckpt).__name__}")

    # Extract state_dict + metadata
    state_dict = None
    meta = {}

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
            print("✅ Found 'model_state_dict'")
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
            print("✅ Found 'state_dict'")
        else:
            # Maybe the dict itself is the state_dict
            all_tensors = all(isinstance(v, torch.Tensor) for v in ckpt.values())
            if all_tensors:
                state_dict = ckpt
                print("✅ Dict is itself a state_dict")
            else:
                print("⚠️ Unknown dict structure. Keys:")
                for k in ckpt.keys():
                    print(f"   - {k}: {type(ckpt[k]).__name__}")
                return
    elif isinstance(ckpt, nn.Module):
        state_dict = ckpt.state_dict()
        print("✅ Loaded as nn.Module")
    else:
        print(f"❌ Unknown checkpoint type: {type(ckpt)}")
        return

    # ── Metadata ──
    print("\n" + "=" * 80)
    print("  📋 METADATA (non-tensor keys)")
    print("=" * 80)
    if meta:
        for k, v in meta.items():
            v_str = str(v)
            if len(v_str) > 300:
                v_str = v_str[:300] + "..."
            print(f"  {k}: {v_str}")
    else:
        print("  (no metadata found)")

    # ── State dict structure ──
    print("\n" + "=" * 80)
    print(f"  🏗️  STATE DICT ({len(state_dict)} tensors)")
    print("=" * 80)

    total_params = 0
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            n = tensor.numel()
            total_params += n
            print(f"  {name:60s}  shape={tuple(tensor.shape)}  params={n:,}")
        else:
            print(f"  {name:60s}  (not a tensor: {type(tensor).__name__})")

    print(f"\n📊 TOTAL PARAMETERS: {total_params:,}")

    # ── Group by prefix ──
    print("\n" + "=" * 80)
    print("  🗂️  MODULE PREFIXES (top-level modules)")
    print("=" * 80)
    prefixes = OrderedDict()
    for name in state_dict.keys():
        parts = name.split(".")
        if len(parts) >= 2:
            prefix = ".".join(parts[:2]) if parts[0] in ("features", "classifier", "lstm", "fc", "cnn", "rnn") else parts[0]
        else:
            prefix = parts[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1

    for prefix, count in prefixes.items():
        print(f"  {prefix:40s}  → {count} tensors")

    # ── Detect architecture family ──
    print("\n" + "=" * 80)
    print("  🎯 ARCHITECTURE DETECTION")
    print("=" * 80)

    keys = list(state_dict.keys())
    has_cnn = any("features" in k or "conv" in k.lower() for k in keys)
    has_lstm = any("lstm" in k.lower() or "rnn" in k.lower() for k in keys)
    has_fc = any("fc" in k.lower() or "classifier" in k.lower() for k in keys)

    print(f"  Has CNN part:  {has_cnn}")
    print(f"  Has LSTM part: {has_lstm}")
    print(f"  Has FC/Head:   {has_fc}")

    # ── Try to identify CNN backbone ──
    if has_cnn:
        print("\n  🔍 CNN backbone keys (first 20):")
        cnn_keys = [k for k in keys if "features" in k][:20]
        for k in cnn_keys:
            print(f"     {k}: {tuple(state_dict[k].shape)}")

    # ── LSTM info ──
    if has_lstm:
        print("\n  🔍 LSTM keys:")
        lstm_keys = [k for k in keys if "lstm" in k.lower() or "rnn" in k.lower()]
        for k in lstm_keys:
            print(f"     {k}: {tuple(state_dict[k].shape)}")

    # ── FC / classifier info ──
    if has_fc:
        print("\n  🔍 FC/classifier keys:")
        fc_keys = [k for k in keys if "fc" in k.lower() or "classifier" in k.lower()]
        for k in fc_keys:
            print(f"     {k}: {tuple(state_dict[k].shape)}")

    # ── Config from metadata (if available) ──
    print("\n" + "=" * 80)
    print("  ⚙️  CONFIG HINTS (from metadata)")
    print("=" * 80)

    config_keys = ["config", "hyperparameters", "hp", "args", "params", "settings"]
    found_config = False
    for ck in config_keys:
        if ck in meta:
            found_config = True
            print(f"\n  Found '{ck}':")
            cfg = meta[ck]
            if isinstance(cfg, dict):
                for k, v in cfg.items():
                    print(f"    {k}: {v}")
            else:
                print(f"    {cfg}")

    if not found_config:
        print("  (no config dict found in metadata)")
        print("  Trying to infer from state_dict...")

        # Infer LSTM hidden size
        for k in keys:
            if "lstm" in k.lower() and "weight_ih" in k:
                w = state_dict[k]
                hidden_size = w.shape[0] // 4  # gate size
                input_size = w.shape[1]
                print(f"\n  🧠 LSTM detected:")
                print(f"     {k}: {tuple(w.shape)}")
                print(f"     → input_size={input_size}, hidden_size={hidden_size} (assumed 4 gates)")
                break

        for k in keys:
            if "lstm" in k.lower() and "weight_hh" in k:
                w = state_dict[k]
                print(f"     {k}: {tuple(w.shape)}")
                break

        for k in keys:
            if "fc" in k.lower() and "weight" in k:
                w = state_dict[k]
                print(f"\n  🎯 FC head:")
                print(f"     {k}: {tuple(w.shape)}")
                break

        for k in keys:
            if "classifier" in k.lower() and "weight" in k:
                w = state_dict[k]
                print(f"     {k}: {tuple(w.shape)}")
                break

    print("\n" + "=" * 80)
    print("  ✅ INSPECTION COMPLETE")
    print("=" * 80)
    print("\n📋 NEXT STEP:")
    print("   Copy-paste THIS ENTIRE OUTPUT back to the chat.")
    print("   I need it to build the correct CNNLSTM for KD.")


if __name__ == "__main__":
    inspect_checkpoint(CKPT_PATH)