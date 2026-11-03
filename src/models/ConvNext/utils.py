# src/models/ConvNext/utils.py
# -*- coding: utf-8 -*-

"""
Utility Functions for ConvNext Architects
=====================================
Author: Babak Yousefian
"""

import gc
import os
import random
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ================================================================
# PARAMETER COUNTING
# ================================================================
def count_parameters(model: nn.Module) -> Dict[str, Union[int, Dict[str, int], float]]:
    """Count parameters of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    layer_counts = defaultdict(int)
    for name, param in model.named_parameters():
        layer_type = name.split(".")[0]
        layer_counts[layer_type] += param.numel()

    param_type_counts = defaultdict(int)
    for name, param in model.named_parameters():
        if "weight" in name:
            param_type_counts["weights"] += param.numel()
        elif "bias" in name:
            param_type_counts["biases"] += param.numel()
        else:
            param_type_counts["other"] += param.numel()

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "trainable_percentage": (trainable / total) * 100 if total > 0 else 0.0,
        "layer_counts": dict(layer_counts),
        "param_type_counts": dict(param_type_counts),
    }


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Calculate model size in memory."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size

    return {
        "bytes": float(total_size),
        "kb": float(total_size / 1024),
        "mb": float(total_size / 1024**2),
        "gb": float(total_size / 1024**3),
        "params_mb": float(param_size / 1024**2),
    }


# ================================================================
# ADVANCED PROFILING
# ================================================================
def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    warmup: int = 5,
    iterations: int = 20,
) -> Dict[str, Any]:
    """Profile the model."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)

    return {
        "device": device,
        "input_shape": input_shape,
        "mean_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        "fps": iterations / np.sum(times),
    }


def calculate_flops(
    model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)
) -> Dict[str, Any]:
    """Calculate FLOPs."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        dummy = torch.randn(*input_shape).to(device)
        flops = 0

        def hook_fn(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Conv2d):
                b, c_out, h_out, w_out = output.shape
                k_h, k_w = module.kernel_size
                flops += b * c_out * h_out * w_out * k_h * k_w * module.in_channels
            elif isinstance(module, nn.Linear):
                b, c_out = output.shape
                flops += b * c_out * module.in_features

        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))

        start = time.time()
        with torch.no_grad():
            _ = model(dummy)
        elapsed = time.time() - start

        for hook in hooks:
            hook.remove()

        return {
            "total_flops": flops,
            "flops_per_second": flops / elapsed if elapsed > 0 else 0,
        }
    except Exception as e:
        return {"total_flops": 0, "error": str(e)}


def analyze_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
) -> Dict[str, Any]:
    """Analyze memory usage."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        _ = model(dummy)
    size = get_model_size(model)
    return {
        "model_size_mb": size["mb"],
        "input_size_mb": (np.prod(input_shape) * 4) / 1024**2,
        "device": device,
    }


def test_inference_speed(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Test inference speed."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)
    return {
        "mean_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        "fps": num_runs / np.sum(times),
    }


# ================================================================
# MODEL SUMMARY
# ================================================================
def print_model_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (3, 224, 224),
    verbose: bool = True,
    show_parameters: bool = True,
    show_flops: bool = True,
    device=None,
) -> None:
    """Print comprehensive model summary."""
    print("\n" + "=" * 80)
    print("📊 MODEL SUMMARY")
    print("=" * 80)
    print(f"\n🏷️ Model: {model.__class__.__name__}")

    if show_parameters:
        params = count_parameters(model)
        print(f"\n📈 Parameters:")
        print(f"   - Total: {params['total']:,}")
        print(f"   - Trainable: {params['trainable']:,}")
        print(f"   - Non-trainable: {params['non_trainable']:,}")
        print(f"   - Trainable: {params['trainable_percentage']:.1f}%")

    size = get_model_size(model)
    print(f"\n💾 Size: {size['mb']:.2f} MB")

    if show_flops:
        try:
            flops = calculate_flops(model, (1, *input_shape))
            if flops.get("total_flops", 0) > 0:
                print(f"\n⚡ FLOPs: {flops['total_flops']:,}")
        except Exception as e:
            print(f"\n⚠️ FLOPs failed: {e}")

    print(f"\n📥 Input: {input_shape}")

    try:
        if device is None:
            device = next(model.parameters()).device
        dummy = torch.randn(1, *input_shape, device=device)
        with torch.no_grad():
            output = model(dummy)
        print(f"📤 Output: {tuple(output.shape)}")
    except Exception as e:
        print(f"⚠️ Output shape failed: {e}")

    print("\n" + "=" * 80)


# ================================================================
# VALIDATION & STATS
# ================================================================
def validate_forward_pass(
    model: nn.Module,
    device: str = "cpu",
    batch_size: int = 4,
    input_shape: Tuple[int, ...] = (3, 224, 224),
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate forward pass."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(batch_size, *input_shape).to(device)
    results = {"success": False, "errors": []}
    try:
        with torch.no_grad():
            output = model(dummy)
        results["success"] = True
        results["output_shape"] = tuple(output.shape)
        if torch.isnan(output).any():
            results["errors"].append("Output has NaN")
        if torch.isinf(output).any():
            results["errors"].append("Output has Inf")
    except Exception as e:
        results["errors"].append(str(e))
    if verbose:
        print(f"\n🔍 Forward Pass: {'✅' if results['success'] else '❌'}")
        if results["success"]:
            print(f"   Output: {results['output_shape']}")
    return results


def get_layer_output_shapes(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (3, 224, 224),
    verbose: bool = False,
    device=None,
) -> Dict[str, Tuple[int, ...]]:
    """Get layer output shapes."""
    shapes = {}

    def hook_fn(name):
        def hook(module, input, output):
            shapes[name] = tuple(output.shape)
        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        if device is None:
            device = next(model.parameters()).device
        dummy = torch.randn(1, *input_shape, device=device)
        _ = model(dummy)

    for hook in hooks:
        hook.remove()

    if verbose:
        print("\n📐 Layer Output Shapes:")
        for name, shape in shapes.items():
            print(f"   {name}: {shape}")

    return shapes


# ================================================================
# REPRODUCIBILITY
# ================================================================
def set_seed(seed: int = 42) -> None:
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set: {seed}")


def get_device() -> str:
    """Get best device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
