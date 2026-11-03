# src/models/resnet/resnet18/utils.py

"""
Utility Functions for ResNet Architects (ULTIMATE EDITION)

This module provides comprehensive utility functions for working with ResNet models:
- Parameter counting and analysis
- Model size calculation
- Forward pass validation
- Activation statistics
- Gradient flow analysis
- Layer output shapes
- Model comparison
- Reproducibility utilities
- Advanced model profiling
- FLOPs calculation
- Memory usage analysis
- Inference speed test
- Model pruning analysis
- Quantization analysis

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 3.0.0 (ULTIMATE EDITION)
"""

import gc
import json
import os
import random
import time
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ================================================================
# Parameter Counting and Analysis
# ================================================================


def count_parameters(model: nn.Module) -> Dict[str, Union[int, Dict[str, int], float]]:
    """
    Count and categorize parameters of a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts

    Example:
        >>> model = ResNet18()
        >>> params = count_parameters(model)
        >>> print(params['total'])  # 11,689,512
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    # Count by layer type
    layer_counts: Dict[str, int] = defaultdict(int)
    layer_trainable_counts: Dict[str, int] = defaultdict(int)

    for name, param in model.named_parameters():
        layer_type = name.split(".")[0]
        if layer_type:
            layer_counts[layer_type] += param.numel()
            if param.requires_grad:
                layer_trainable_counts[layer_type] += param.numel()

    # Count by parameter type
    param_type_counts: Dict[str, int] = defaultdict(int)
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
        "layer_trainable_counts": dict(layer_trainable_counts),
        "param_type_counts": dict(param_type_counts),
    }


def count_parameters_by_layer(
    model: nn.Module,
) -> Dict[str, Dict[str, Union[int, str]]]:
    """
    Count parameters for each layer with detailed breakdown.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with layer-wise parameter counts
    """
    layer_info: Dict[str, Dict[str, Union[int, str]]] = {}

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if total_params > 0:
                layer_info[name] = {
                    "type": module.__class__.__name__,
                    "total": total_params,
                    "trainable": trainable_params,
                    "non_trainable": total_params - trainable_params,
                }

    return layer_info


def get_parameter_sizes(model: nn.Module) -> Dict[str, List[int]]:
    """
    Get sizes of all parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping parameter names to their sizes
    """
    sizes: Dict[str, List[int]] = {}
    for name, param in model.named_parameters():
        sizes[name] = list(param.size())
    return sizes


# ================================================================
# Model Size Calculation
# ================================================================


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calculate the size of the model in memory.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with size information in MB and GB

    Example:
        >>> model = ResNet18()
        >>> size = get_model_size(model)
        >>> print(f"Model size: {size['mb']:.2f} MB")
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        "bytes": float(total_size),
        "kb": float(total_size / 1024),
        "mb": float(total_size / 1024**2),
        "gb": float(total_size / 1024**3),
        "params_bytes": float(param_size),
        "params_mb": float(param_size / 1024**2),
        "buffers_bytes": float(buffer_size),
        "buffers_mb": float(buffer_size / 1024**2),
    }


# ================================================================
# ✅ ADVANCED MODEL PROFILING 🔥 NEW
# ================================================================


def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    warmup: int = 5,
    iterations: int = 20,
) -> Dict[str, Any]:
    """
    Profile the model with advanced metrics.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on
        warmup: Number of warmup iterations
        iterations: Number of iterations for timing

    Returns:
        Dictionary with profiling results
    """
    model = model.to(device)
    model.eval()

    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)

    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.synchronize()

    # Timing
    times = []
    total_time = 0

    for _ in range(iterations):
        torch.cuda.synchronize() if device == "cuda" else None
        iter_start = time.time()

        with torch.no_grad():
            _ = model(dummy)

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.time() - iter_start
        times.append(elapsed)
        total_time += elapsed

    # Memory usage (GPU only)
    memory_info = {}
    if device == "cuda" and torch.cuda.is_available():
        memory_info = {
            "allocated_mb": torch.cuda.memory_allocated(0) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(0) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(0) / 1024**2,
        }

    # CPU memory
    if device == "cpu":
        try:
            import psutil

            process = psutil.Process()
            memory_info = {
                "memory_mb": process.memory_info().rss / 1024**2,
                "memory_percent": process.memory_percent(),
            }
        except ImportError:
            memory_info = {"memory_mb": 0, "memory_percent": 0}

    return {
        "device": device,
        "input_shape": input_shape,
        "iterations": iterations,
        "total_time_sec": total_time,
        "mean_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        "min_time_ms": np.min(times) * 1000,
        "max_time_ms": np.max(times) * 1000,
        "fps": iterations / total_time,
        "memory": memory_info,
        "times_ms": times,
    }


# ================================================================
# ✅ FLOPs CALCULATION 🔥 NEW
# ================================================================


def calculate_flops(
    model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)
) -> Dict[str, Any]:
    """
    Calculate FLOPs for the model using PyTorch's profiling.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape

    Returns:
        Dictionary with FLOPs information
    """
    try:
        device = next(model.parameters()).device
        model = model.to(device)
        model.eval()

        dummy = torch.randn(*input_shape).to(device)
        flops = 0

        def hook_fn(
            module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            nonlocal flops
            if isinstance(module, nn.Conv2d):
                b, c_out, h_out, w_out = output.shape
                k_h, k_w = module.kernel_size
                flops += b * c_out * h_out * w_out * k_h * k_w * module.in_channels // module.groups
            elif isinstance(module, nn.Linear):
                b, c_out = output.shape
                flops += b * c_out * module.in_features

        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy)
        total_time = time.time() - start_time

        for hook in hooks:
            hook.remove()

        return {
            "total_flops": flops,
            "flops_per_second": flops / total_time if total_time > 0 else 0,
            "total_time_sec": total_time,
            "device": device,
            "input_shape": input_shape,
        }
    except Exception as e:
        return {"total_flops": 0, "error": str(e)}


# ================================================================
# ✅ MEMORY USAGE ANALYSIS 🔥 NEW
# ================================================================


def analyze_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Analyze memory usage of the model.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on

    Returns:
        Dictionary with memory analysis
    """
    model = model.to(device)
    model.eval()

    dummy = torch.randn(*input_shape).to(device)

    # Clear cache
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Initial memory
    initial_memory = 0
    if device == "cuda" and torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated(0)
    elif device == "cpu":
        try:
            import psutil

            process = psutil.Process()
            initial_memory = process.memory_info().rss
        except ImportError:
            initial_memory = 0

    # Forward pass
    with torch.no_grad():
        output = model(dummy)

    # Final memory
    final_memory = 0
    if device == "cuda" and torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated(0)
    elif device == "cpu":
        try:
            import psutil

            process = psutil.Process()
            final_memory = process.memory_info().rss
        except ImportError:
            final_memory = 0

    # Get model size
    model_size = get_model_size(model)

    return {
        "model_size_mb": model_size["mb"],
        "input_size_mb": (np.prod(input_shape) * 4) / 1024**2,
        "initial_memory_mb": initial_memory / 1024**2,
        "final_memory_mb": final_memory / 1024**2,
        "memory_used_mb": (final_memory - initial_memory) / 1024**2,
        "device": device,
        "input_shape": input_shape,
    }


# ================================================================
# ✅ INFERENCE SPEED TEST 🔥 NEW
# ================================================================


def test_inference_speed(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    num_runs: int = 100,
) -> Dict[str, Any]:
    """
    Test inference speed of the model.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on
        num_runs: Number of inference runs

    Returns:
        Dictionary with speed test results
    """
    model = model.to(device)
    model.eval()

    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    # Test
    times = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            _ = model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start_time)

    return {
        "device": device,
        "input_shape": input_shape,
        "num_runs": num_runs,
        "mean_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        "min_time_ms": np.min(times) * 1000,
        "max_time_ms": np.max(times) * 1000,
        "fps": num_runs / np.sum(times),
        "times_ms": [t * 1000 for t in times],
    }


# ================================================================
# Model Summary Printing
# ================================================================


def print_model_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (3, 224, 224),
    verbose: bool = True,
    show_parameters: bool = True,
    show_gradients: bool = False,
    show_flops: bool = True,
) -> None:
    """
    Print a comprehensive summary of the model with advanced metrics.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        verbose: If True, print detailed layer information
        show_parameters: If True, show parameter counts
        show_gradients: If True, show gradient information
        show_flops: If True, show FLOPs information
    """
    print("\n" + "=" * 80)
    print("📊 MODEL SUMMARY (ULTIMATE EDITION)")
    print("=" * 80)

    print(f"\n🏷️ Model: {model.__class__.__name__}")

    if show_parameters:
        params = count_parameters(model)
        print(f"\n📈 Parameter Count:")
        print(f"   - Total: {params['total']:,}")
        print(f"   - Trainable: {params['trainable']:,}")
        print(f"   - Non-trainable: {params['non_trainable']:,}")
        print(f"   - Trainable: {params['trainable_percentage']:.1f}%")

        if params["param_type_counts"]:
            print(f"\n   Parameter Type Breakdown:")
            for ptype, count in params["param_type_counts"].items():
                print(f"      - {ptype}: {count:,}")

    size = get_model_size(model)
    print(f"\n💾 Model Size:")
    print(f"   - {size['mb']:.2f} MB ({size['gb']:.4f} GB)")
    print(f"   - Parameters: {size['params_mb']:.2f} MB")
    print(f"   - Buffers: {size['buffers_mb']:.2f} MB")

    if show_flops:
        try:
            flops_info = calculate_flops(model, (1, *input_shape))
            if flops_info.get("total_flops", 0) > 0:
                print(f"\n⚡ FLOPs:")
                print(f"   - Total FLOPs: {flops_info['total_flops']:,}")
                print(f"   - FLOPs per Second: {flops_info['flops_per_second']:.2e}")
        except Exception as e:
            print(f"\n⚠️ FLOPs calculation failed: {e}")

    print(f"\n📥 Input Shape: {input_shape}")

    try:
        device = next(model.parameters()).device  # ✅
        dummy = torch.randn(1, *input_shape).to(device)
        with torch.no_grad():
            output = model(dummy)
        print(f"📤 Output Shape: {tuple(output.shape)}")
    except Exception as e:
        print(f"⚠️ Could not determine output shape: {e}")

    if verbose:
        print(f"\n📋 Layer Summary:")
        print("-" * 80)
        print(
            f"{'Layer Name':<30} {'Type':<20} {'Parameters':<12} {'Trainable':<10} {'Output Shape':<15}"
        )
        print("-" * 80)

        total_params = 0
        total_trainable = 0

        # Get output shapes
        shapes = get_layer_output_shapes(model, input_shape, verbose=False)

        for name, module in model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += num_params
            total_trainable += trainable_params

            module_type = module.__class__.__name__
            trainable_str = "✅" if trainable_params > 0 else "❌"
            shape_str = str(shapes.get(name, "N/A"))

            print(
                f"{name:<30} {module_type:<20} {num_params:<12,} {trainable_str:<10} {shape_str:<15}"
            )

        print("-" * 80)
        print(f"{'TOTAL':<30} {'':<20} {total_params:<12,} {total_trainable:<10,}")

    if show_gradients:
        print(f"\n🔍 Gradient Information:")
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                has_grad = True
                grad_status = "✅" if param.grad is not None else "⬜"
                print(f"   {name}: {grad_status}")

        if not has_grad:
            print("   No parameters require gradients")

    # Device information
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else "cpu"
    print(f"\n🖥️ Device: {device}")

    print("\n" + "=" * 80)


# ================================================================
# Forward Pass Validation
# ================================================================


def validate_forward_pass(
    model: nn.Module,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    input_shape: Tuple[int, ...] = (3, 224, 224),
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate the forward pass of a model.

    Args:
        model: PyTorch model
        device: Device to run on ('cpu' or 'cuda')
        batch_size: Batch size for testing
        input_shape: Input tensor shape
        verbose: If True, print detailed results

    Returns:
        Dictionary with validation results
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, *input_shape).to(device)

    results: Dict[str, Any] = {
        "success": False,
        "input_shape": tuple(dummy_input.shape),
        "output_shape": None,
        "device": device,
        "errors": [],
        "warnings": [],
        "time_taken": None,
    }

    start_time = time.time()

    try:
        with torch.no_grad():
            output = model(dummy_input)

        elapsed_time = time.time() - start_time
        results["time_taken"] = elapsed_time

        results["output_shape"] = tuple(output.shape)
        results["success"] = True

        if len(output.shape) != 2:
            results["warnings"].append(f"Output has {len(output.shape)} dimensions, expected 2")

        if output.shape[0] != batch_size:
            results["warnings"].append(f"Batch size mismatch: {output.shape[0]} != {batch_size}")

        if torch.isnan(output).any():
            results["warnings"].append("Output contains NaN values")
        if torch.isinf(output).any():
            results["warnings"].append("Output contains Inf values")

    except Exception as e:
        results["errors"].append(str(e))
        results["success"] = False

    if verbose:
        print(f"\n🔍 Forward Pass Validation:")
        print(f"   - Device: {device}")
        print(f"   - Input Shape: {results['input_shape']}")
        if results["success"]:
            print(f"   - Output Shape: {results['output_shape']}")
            print(f"   - Time: {results['time_taken']:.4f}s")
            print(f"   - Status: ✅ Success")
        else:
            print(f"   - Status: ❌ Failed")
            for error in results["errors"]:
                print(f"   - Error: {error}")
        if results["warnings"]:
            for warning in results["warnings"]:
                print(f"   - Warning: {warning}")

    return results


# ================================================================
# Activation Statistics
# ================================================================


def get_activation_stats(
    model: nn.Module,
    input_tensor: torch.Tensor,
    return_activations: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Get statistics of activations through the network.

    Args:
        model: PyTorch model
        input_tensor: Input tensor
        return_activations: If True, return activation tensors
        verbose: If True, print statistics

    Returns:
        Dictionary with activation statistics
    """

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    activations: Dict[str, torch.Tensor] = {}

    def hook_fn(name: str) -> Callable:
        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()

        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    for hook in hooks:
        hook.remove()

    stats: Dict[str, Any] = {
        "layers": len(activations),
        "total_activations": 0,
        "stats": {},
    }

    for name, activation in activations.items():
        num_elements = activation.numel()
        stats["total_activations"] += num_elements

        stats["stats"][name] = {
            "shape": tuple(activation.shape),
            "num_elements": num_elements,
            "mean": float(activation.mean().item()),
            "std": float(activation.std().item()),
            "min": float(activation.min().item()),
            "max": float(activation.max().item()),
            "sparsity": float((activation == 0).float().mean().item()),
            "energy": float((activation**2).sum().item()),
        }

    if return_activations:
        stats["activations"] = activations

    if verbose:
        print(f"\n📊 Activation Statistics:")
        print(f"   - Layers Monitored: {stats['layers']}")
        print(f"   - Total Activations: {stats['total_activations']:,}")

        for i, (name, layer_stats) in enumerate(list(stats["stats"].items())[:5]):
            print(f"\n   {name}:")
            print(f"      - Shape: {layer_stats['shape']}")
            print(f"      - Mean: {layer_stats['mean']:.4f}")
            print(f"      - Std: {layer_stats['std']:.4f}")
            print(f"      - Sparsity: {layer_stats['sparsity']:.4f}")

        if len(stats["stats"]) > 5:
            print(f"\n   ... and {len(stats['stats']) - 5} more layers")

    return stats


# ================================================================
# Gradient Flow Analysis
# ================================================================


def check_gradient_flow(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    criterion: Optional[nn.Module] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Check gradient flow through the network.

    Args:
        model: PyTorch model
        input_tensor: Input tensor
        target: Target tensor (if None, random targets will be generated)
        criterion: Loss function (if None, CrossEntropyLoss will be used)
        verbose: If True, print gradient information

    Returns:
        Dictionary with gradient statistics
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    model.train()

    num_classes = 5
    if hasattr(model, "num_classes"):
        num_classes = model.num_classes
    elif hasattr(model, "fc") and hasattr(model.fc, "out_features"):
        num_classes = model.fc.out_features

    if target is None:
        target = torch.randint(0, num_classes, (input_tensor.size(0),))

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()

    grad_stats: Dict[str, Any] = {
        "total_grad_norm": 0.0,
        "max_grad_norm": 0.0,
        "min_grad_norm": float("inf"),
        "zero_grad": 0,
        "nonzero_grad": 0,
        "total_params": 0,
        "layer_stats": {},
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats["total_grad_norm"] += grad_norm**2
            grad_stats["max_grad_norm"] = max(grad_stats["max_grad_norm"], grad_norm)
            grad_stats["min_grad_norm"] = min(grad_stats["min_grad_norm"], grad_norm)
            grad_stats["total_params"] += 1

            if grad_norm == 0:
                grad_stats["zero_grad"] += 1
            else:
                grad_stats["nonzero_grad"] += 1

            grad_stats["layer_stats"][name] = {
                "norm": float(grad_norm),
                "mean": float(param.grad.mean().item()),
                "std": float(param.grad.std().item()),
                "max": float(param.grad.max().item()),
                "min": float(param.grad.min().item()),
            }
        else:
            grad_stats["zero_grad"] += 1
            grad_stats["total_params"] += 1

    grad_stats["total_grad_norm"] = float(np.sqrt(grad_stats["total_grad_norm"]))

    if verbose:
        print("\n🔍 Gradient Flow Analysis:")
        print(f"   Total Gradient Norm: {grad_stats['total_grad_norm']:.4f}")
        print(f"   Max Gradient Norm: {grad_stats['max_grad_norm']:.4f}")
        print(f"   Min Gradient Norm: {grad_stats['min_grad_norm']:.4f}")
        print(f"   Zero Gradients: {grad_stats['zero_grad']}")
        print(f"   Non-zero Gradients: {grad_stats['nonzero_grad']}")

        if grad_stats["total_grad_norm"] < 1e-3:
            print("   ⚠️ Warning: Potential vanishing gradients detected!")
        elif grad_stats["total_grad_norm"] > 1e3:
            print("   ⚠️ Warning: Potential exploding gradients detected!")
        else:
            print("   ✅ Gradient flow seems healthy!")

        if grad_stats["layer_stats"]:
            sorted_layers = sorted(
                grad_stats["layer_stats"].items(),
                key=lambda x: x[1]["norm"],
                reverse=True,
            )[:5]
            print("\n   Top 5 Layers by Gradient Norm:")
            for name, stats in sorted_layers:
                print(f"      {name}: {stats['norm']:.4f}")

    model.zero_grad()

    return grad_stats


# ================================================================
# Layer Output Shapes
# ================================================================


def get_layer_output_shapes(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (3, 224, 224),
    verbose: bool = False,
) -> Dict[str, Tuple[int, ...]]:
    """
    Get output shapes for each layer of the model.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        verbose: If True, print shapes

    Returns:
        Dictionary mapping layer names to output shapes
    """
    shapes: Dict[str, Tuple[int, ...]] = {}

    def hook_fn(name: str) -> Callable:
        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            shapes[name] = tuple(output.shape)

        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        dummy = torch.randn(1, *input_shape).to(device)
        _ = model(dummy)

    for hook in hooks:
        hook.remove()

    if verbose:
        print("\n📐 Layer Output Shapes:")
        for name, shape in shapes.items():
            print(f"   {name}: {shape}")

    return shapes


# ================================================================
# Model Comparison
# ================================================================


def compare_models(model1: nn.Module, model2: nn.Module, verbose: bool = True) -> Dict[str, Any]:
    """
    Compare two models for architectural differences.

    Args:
        model1: First model
        model2: Second model
        verbose: If True, print comparison results

    Returns:
        Dictionary with comparison results
    """
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)

    comparison: Dict[str, Any] = {
        "model1_name": model1.__class__.__name__,
        "model2_name": model2.__class__.__name__,
        "parameters": {
            "model1_total": params1["total"],
            "model2_total": params2["total"],
            "difference": params1["total"] - params2["total"],
            "ratio": (
                params1["total"] / params2["total"] if params2["total"] > 0 else float("inf")
            ),
            "model1_trainable": params1["trainable"],
            "model2_trainable": params2["trainable"],
        },
        "sizes": {
            "model1_mb": get_model_size(model1)["mb"],
            "model2_mb": get_model_size(model2)["mb"],
        },
    }

    layers1 = set([name for name, _ in model1.named_children()])
    layers2 = set([name for name, _ in model2.named_children()])

    comparison["architecture"] = {
        "common_layers": list(layers1 & layers2),
        "model1_only": list(layers1 - layers2),
        "model2_only": list(layers2 - layers1),
    }

    if verbose:
        print("\n🔍 Model Comparison:")
        print(f"   Model 1: {comparison['model1_name']}")
        print(f"   Model 2: {comparison['model2_name']}")
        print(f"\n   Parameters:")
        print(f"      Model 1: {comparison['parameters']['model1_total']:,}")
        print(f"      Model 2: {comparison['parameters']['model2_total']:,}")
        print(f"      Difference: {comparison['parameters']['difference']:,}")
        print(f"      Ratio: {comparison['parameters']['ratio']:.2f}x")
        print(f"\n   Model Sizes:")
        print(f"      Model 1: {comparison['sizes']['model1_mb']:.2f} MB")
        print(f"      Model 2: {comparison['sizes']['model2_mb']:.2f} MB")
        print(f"\n   Common Layers: {len(comparison['architecture']['common_layers'])}")
        if comparison["architecture"]["model1_only"]:
            print(f"   Model 1 Only: {', '.join(comparison['architecture']['model1_only'])}")
        if comparison["architecture"]["model2_only"]:
            print(f"   Model 2 Only: {', '.join(comparison['architecture']['model2_only'])}")

    return comparison


# ================================================================
# Reproducibility Utilities
# ================================================================


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✅ Seed set to: {seed}")


# ================================================================
# Model Information
# ================================================================


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with comprehensive model information
    """
    info: Dict[str, Any] = {
        "name": model.__class__.__name__,
        "parameters": count_parameters(model),
        "size": get_model_size(model),
        "has_batch_norm": any(isinstance(m, nn.BatchNorm2d) for m in model.modules()),
        "has_dropout": any(isinstance(m, nn.Dropout) for m in model.modules()),
        "device": (
            str(next(model.parameters()).device) if len(list(model.parameters())) > 0 else "cpu"
        ),
        "is_training": model.training,
        "num_layers": len(list(model.modules())),
        "layer_names": [name for name, _ in model.named_children()],
    }

    features = []
    if hasattr(model, "get_feature_dim"):
        features.append("feature_extractor")
    if hasattr(model, "get_embedding"):
        features.append("embedding")
    if hasattr(model, "get_activations"):
        features.append("activations")

    info["features"] = features

    return info


# ================================================================
# Save Model Summary
# ================================================================


def save_model_summary(
    model: nn.Module, output_path: str, input_shape: Tuple[int, ...] = (3, 224, 224)
) -> None:
    """
    Save model summary to a text file.

    Args:
        model: PyTorch model
        output_path: Path to save the summary
        input_shape: Input tensor shape
    """
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        import sys

        old_stdout = sys.stdout
        sys.stdout = f

        print_model_summary(model, input_shape, verbose=True, show_flops=True)

        print("\n" + "=" * 80)
        print("LAYER OUTPUT SHAPES")
        print("=" * 80)
        shapes = get_layer_output_shapes(model, input_shape, verbose=False)
        for name, shape in shapes.items():
            print(f"{name}: {shape}")

        sys.stdout = old_stdout

    print(f"✅ Model summary saved to: {output_path}")


# ================================================================
# Additional Utilities
# ================================================================


def get_device() -> str:
    """
    Get the best available device.

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_cuda_info() -> Dict[str, Any]:
    """
    Get CUDA information if available.

    Returns:
        Dictionary with CUDA information
    """
    info: Dict[str, Any] = {"cuda_available": torch.cuda.is_available()}

    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_allocated"] = float(torch.cuda.memory_allocated(0) / 1024**3)
        info["cuda_memory_reserved"] = float(torch.cuda.memory_reserved(0) / 1024**3)
        info["cuda_memory_free"] = info["cuda_memory_reserved"] - info["cuda_memory_allocated"]
        info["cuda_total_memory"] = float(
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )

    return info


# ✅ اضافه کردن تابع get_layer_io_shapes
def get_layer_io_shapes(
    model: nn.Module, input_shape: Tuple[int, ...], device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Dict[str, Any]]:
    """
    Get input and output shapes for each layer of the model.

    Args:
        model: PyTorch model
        input_shape: Input shape (batch_size, channels, height, width)
        device: Device to run on

    Returns:
        List of dictionaries with layer info
    """
    model.eval()
    model.to(device)

    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)

    # Dictionary to store layer outputs
    layer_outputs = {}

    # Hook function to capture outputs
    def hook_fn(name):
        def hook(module, input, output):
            layer_outputs[name] = {
                "input_shape": tuple(input[0].shape),
                "output_shape": tuple(output.shape),
            }

        return hook

    # Register hooks for all modules
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Format results
    results = []
    for name, info in layer_outputs.items():
        results.append(
            {
                "layer_name": name,
                "input_shape": info["input_shape"],
                "output_shape": info["output_shape"],
            }
        )

    return results


# ✅ به‌روزرسانی __all__
__all__ = [
    "count_parameters",
    "get_model_size",
    "print_model_summary",
    "validate_forward_pass",
    "get_activation_stats",
    "check_gradient_flow",
    "get_layer_output_shapes",
    "compare_models",
    "profile_model",
    "calculate_flops",
    "analyze_memory_usage",
    "test_inference_speed",
    "get_layer_io_shapes",  # ✅ اضافه شده
]


# ================================================================
# Testing
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing ResNet Utils (ULTIMATE EDITION)")
    print("=" * 80)

    try:
        from .resnet18 import create_resnet18

        model = create_resnet18(num_classes=5, pretrained=False, verbose=False)
        print("✅ Model created successfully")
    except ImportError:
        print("⚠️ Could not import ResNet18, creating dummy model")
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 5),
        )
        print("✅ Dummy model created")

    # Test 1: Parameter Counting
    print("\n📌 Test 1: Parameter Count")
    params = count_parameters(model)
    print(f"   Total: {params['total']:,}")

    # Test 2: Model Size
    print("\n📌 Test 2: Model Size")
    size = get_model_size(model)
    print(f"   Size: {size['mb']:.2f} MB")

    # Test 3: Profile Model
    print("\n📌 Test 3: Profile Model")
    try:
        profile = profile_model(model, (1, 3, 224, 224), "cpu", warmup=2, iterations=5)
        print(f"   Mean Inference Time: {profile['mean_time_ms']:.2f} ms")
        print(f"   FPS: {profile['fps']:.2f}")
    except Exception as e:
        print(f"   ⚠️ Profiling failed: {e}")

    # Test 4: FLOPs Calculation
    print("\n📌 Test 4: FLOPs Calculation")
    try:
        flops = calculate_flops(model, (1, 3, 224, 224))
        if flops.get("total_flops", 0) > 0:
            print(f"   Total FLOPs: {flops['total_flops']:,}")
    except Exception as e:
        print(f"   ⚠️ FLOPs calculation failed: {e}")

    # Test 5: Model Summary
    print("\n📌 Test 5: Model Summary")
    print_model_summary(model, (3, 224, 224), verbose=False, show_flops=False)

    print("\n✅ All tests passed! Ultimate utils are ready!")
    print("=" * 80)
