# src/models/MobileNetV2/utils.py
# -*- coding: utf-8 -*-

"""
MobileNetV2 Utilities
=====================
Helper functions for model analysis, device management, and profiling.

Author: Babak Yousefian
"""

import os
import random
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ================================================================
# Device Management
# ================================================================

def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.

    Args:
        prefer_cuda: If True, prefer CUDA if available.

    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed.
        deterministic: If True, use deterministic algorithms (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ================================================================
# Parameter / Size Analysis
# ================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dict with 'total' and 'trainable'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Get model size in MB (float32).

    Args:
        model: PyTorch model.

    Returns:
        Dict with 'mb', 'bytes', and 'params'.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_size + buffer_size

    return {
        "mb": total_bytes / (1024 ** 2),
        "bytes": total_bytes,
        "params": sum(p.numel() for p in model.parameters()),
    }


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
    """
    Print a compact model summary.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.

    Returns:
        Summary string.
    """
    params = count_parameters(model)
    size = get_model_size(model)

    lines = [
        "=" * 70,
        f"Model: {model.__class__.__name__}",
        "=" * 70,
        f"Total Parameters:     {params['total']:,}",
        f"Trainable Parameters: {params['trainable']:,}",
        f"Frozen Parameters:    {params['frozen']:,}",
        f"Model Size:           {size['mb']:.2f} MB",
        f"Input Size:           {input_size}",
        "=" * 70,
    ]
    summary = "\n".join(lines)
    print(summary)
    return summary


# ================================================================
# Forward Pass Validation
# ================================================================

def validate_forward_pass(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Validate that the model can run a forward pass.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.
        device: Device to run on (defaults to model's device).

    Returns:
        Dict with 'success', 'output_shape', 'error', 'input_shape'.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    dummy = torch.randn(*input_size, device=device)

    try:
        with torch.no_grad():
            out = model(dummy)
        return {
            "success": True,
            "input_shape": tuple(input_size),
            "output_shape": tuple(out.shape),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "input_shape": tuple(input_size),
            "output_shape": None,
            "error": str(e),
        }


# ================================================================
# Layer Output Shapes
# ================================================================

def get_layer_output_shapes(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Get output shapes of all named modules via forward hooks.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.
        device: Device to run on.

    Returns:
        List of dicts with 'name', 'type', 'output_shape'.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    results: List[Dict[str, Any]] = []
    hooks: List[Any] = []

    def make_hook(name: str, module: nn.Module):
        def hook(_module, _inp, out):
            def shape_of(x):
                if isinstance(x, torch.Tensor):
                    return tuple(x.shape)
                if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
                    return tuple(x[0].shape)
                return None

            results.append({
                "name": name,
                "type": module.__class__.__name__,
                "output_shape": shape_of(out),
            })
        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        hooks.append(module.register_forward_hook(make_hook(name, module)))

    model.eval()
    dummy = torch.randn(*input_size, device=device)
    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for h in hooks:
            h.remove()

    return results


# ================================================================
# FLOPs (approximate) & Memory
# ================================================================

def calculate_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Approximate FLOPs for Conv2d and Linear layers.

    NOTE: This is an approximation — it counts MACs for Conv2d/Linear
    and multiplies by 2. Activation/normalization FLOPs are ignored.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.
        device: Device to run on.

    Returns:
        Dict with 'flops', 'gflops', 'params_counted'.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    total_flops = 0
    hooks: List[Any] = []

    def conv_hook(module: nn.Conv2d, inp, out):
        nonlocal total_flops
        # out shape: (B, Cout, Hout, Wout)
        batch = out.shape[0]
        cout = out.shape[1]
        hout = out.shape[2]
        wout = out.shape[3] if out.dim() > 3 else 1
        cin_per_group = module.in_channels // module.groups
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        # MACs
        macs = batch * cout * hout * wout * cin_per_group * kernel_ops
        total_flops += 2 * macs  # 2x for FLOPs

    def linear_hook(module: nn.Linear, inp, out):
        nonlocal total_flops
        batch = out.shape[0] if out.dim() > 1 else 1
        macs = batch * module.in_features * module.out_features
        total_flops += 2 * macs

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    dummy = torch.randn(*input_size, device=device)
    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for h in hooks:
            h.remove()

    return {
        "flops": float(total_flops),
        "gflops": float(total_flops) / 1e9,
        "mflops": float(total_flops) / 1e6,
    }


def analyze_memory_usage(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Analyze GPU memory usage (if CUDA) or fall back to param size.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.
        device: Device to run on.

    Returns:
        Dict with 'device', 'param_mb', 'allocated_mb', 'reserved_mb', 'peak_mb'.
    """
    if device is None:
        device = get_device()

    result: Dict[str, Any] = {
        "device": str(device),
        "param_mb": get_model_size(model)["mb"],
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "peak_mb": 0.0,
    }

    if device.type != "cuda":
        return result

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        was_training = model.training
        model.eval()
        dummy = torch.randn(*input_size, device=device)

        with torch.no_grad():
            _ = model(dummy)

        result["allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
        result["reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)
        result["peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        del dummy
        torch.cuda.empty_cache()

        if was_training:
            model.train()

    except Exception as e:
        result["error"] = str(e)

    return result


# ================================================================
# Inference Speed
# ================================================================

def test_inference_speed(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
    warmup: int = 10,
    runs: int = 50,
) -> Dict[str, float]:
    """
    Measure inference latency and throughput.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.
        device: Device to run on.
        warmup: Number of warmup runs.
        runs: Number of timed runs.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'fps'.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    was_training = model.training
    model.eval()

    dummy = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times: List[float] = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    if was_training:
        model.train()

    arr = np.array(times)
    mean_ms = float(arr.mean()) if len(arr) else 0.0
    return {
        "mean_ms": mean_ms,
        "std_ms": float(arr.std()) if len(arr) else 0.0,
        "min_ms": float(arr.min()) if len(arr) else 0.0,
        "max_ms": float(arr.max()) if len(arr) else 0.0,
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "device": str(device),
        "input_shape": tuple(input_size),
        "runs": runs,
    }


# ================================================================
# Profiling
# ================================================================

def profile_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run a comprehensive profile: forward pass, params, size, FLOPs,
    memory, and inference speed.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.
        device: Device to run on.

    Returns:
        Dict with all profiling results.
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    forward = validate_forward_pass(model, input_size=input_size, device=device)
    params = count_parameters(model)
    size = get_model_size(model)
    flops = calculate_flops(model, input_size=input_size, device=device)
    memory = analyze_memory_usage(model, input_size=input_size, device=device)
    speed = test_inference_speed(model, input_size=input_size, device=device)

    return {
        "model_name": model.__class__.__name__,
        "device": str(device),
        "forward_pass": forward,
        "parameters": params,
        "size": size,
        "flops": flops,
        "memory": memory,
        "inference": speed,
    }


# ================================================================
# __all__
# ================================================================

__all__ = [
    "get_device",
    "set_seed",
    "count_parameters",
    "get_model_size",
    "print_model_summary",
    "validate_forward_pass",
    "get_layer_output_shapes",
    "calculate_flops",
    "analyze_memory_usage",
    "test_inference_speed",
    "profile_model",
]
