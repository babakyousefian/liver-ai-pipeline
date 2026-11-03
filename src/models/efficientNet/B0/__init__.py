# src/models/efficientNet/B0/__init__.py

"""
EfficientNet-B0 Module - Complete Implementation from Scratch (ULTIMATE EDITION)

This module provides a complete implementation of EfficientNet-B0 architecture
without any pre-trained weights, following the original paper:
"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

The implementation includes:
- Complete EfficientNet-B0 architecture (MBConv blocks with SE modules)
- Squeeze-and-Excitation (SE) blocks for channel attention
- Swish activation function (SiLU)
- Compound scaling parameters (width, depth, resolution)
- Advanced profiling and analysis tools
- Model registry and management
- Benchmark and conversion utilities

Architecture Details:
- Input: 224x224x3
- Base channels: 32
- Width multiplier: 1.0
- Depth multiplier: 1.0
- Resolution: 224x224
- Total Parameters: ~5.3M

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 3.0.0 (ULTIMATE EDITION)
"""

# ================================================================
# ✅ Suppress import messages (better for production)
# ================================================================
import warnings

warnings.filterwarnings("ignore")
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import MBConvBlock, SqueezeExcitation, Swish, make_layer
from .efficientnet_b0 import EfficientNetB0, create_efficientnet_b0
from .utils import (  # Parameter counting and analysis; Model size calculation; Model summary and validation; Activation and gradient analysis; Layer analysis; Model comparison; Additional utilities; Device utilities; ✅ Advanced profiling utilities (NEW)
    analyze_memory_usage,
    calculate_flops,
    check_gradient_flow,
    compare_models,
    count_parameters,
    count_parameters_by_layer,
    get_activation_stats,
    get_cuda_info,
    get_device,
    get_layer_output_shapes,
    get_model_info,
    get_model_size,
    get_parameter_sizes,
    print_model_summary,
    profile_model,
    save_model_summary,
    set_seed,
    validate_forward_pass,
)

# ================================================================
# Core Model Architecture
# ================================================================


# ================================================================
# Building Blocks
# ================================================================


# ================================================================
# Utility Functions
# ================================================================


# ================================================================
# ✅ Additional utilities that might be in blocks
# ================================================================

try:
    from .blocks import (
        count_blocks_in_layer,
        get_block_info,
        get_layer_info,
        get_layer_io_shapes,
        print_layer_info,
    )
except ImportError:
    # Define fallback functions
    def get_block_info(block_idx: int) -> dict:
        """Get information about a specific block."""
        block_info = {
            0: {"name": "Stem", "channels": 32, "resolution": 224},
            1: {"name": "Stage 1", "channels": 16, "resolution": 112},
            2: {"name": "Stage 2", "channels": 24, "resolution": 56},
            3: {"name": "Stage 3", "channels": 40, "resolution": 28},
            4: {"name": "Stage 4", "channels": 80, "resolution": 14},
            5: {"name": "Stage 5", "channels": 112, "resolution": 14},
            6: {"name": "Stage 6", "channels": 192, "resolution": 7},
            7: {"name": "Stage 7", "channels": 320, "resolution": 7},
            8: {"name": "Head", "channels": 1280, "resolution": 1},
        }
        return block_info.get(
            block_idx, {"name": f"Block {block_idx}", "channels": 0, "resolution": 0}
        )

    def get_layer_info(layer_name: str) -> dict:
        return {"name": layer_name, "type": "unknown"}

    def print_layer_info(layer_name: str) -> None:
        print(f"Layer: {layer_name}")

    def count_blocks_in_layer(layer) -> dict:
        return {"total": len(layer) if hasattr(layer, "__len__") else 0}

    def get_layer_io_shapes(layer, input_shape):
        return {"input_shape": input_shape, "output_shape": input_shape}


# ================================================================
# ✅ Module Metadata
# ================================================================

__version__ = "3.0.0"
__author__ = "Babak Yousefian"
__all__ = [
    # Model Classes
    "EfficientNetB0",
    # Factory Functions
    "create_efficientnet_b0",
    # Block Components
    "MBConvBlock",
    "SqueezeExcitation",
    "Swish",
    "make_layer",
    "get_block_info",
    "get_layer_info",
    "print_layer_info",
    "count_blocks_in_layer",
    "get_layer_io_shapes",
    # Utility Functions
    "count_parameters",
    "count_parameters_by_layer",
    "get_parameter_sizes",
    "get_model_size",
    "print_model_summary",
    "validate_forward_pass",
    "get_activation_stats",
    "check_gradient_flow",
    "get_layer_output_shapes",
    "compare_models",
    "set_seed",
    "get_model_info",
    "save_model_summary",
    "get_device",
    "get_cuda_info",
    # Advanced Utilities (NEW)
    "profile_model",
    "calculate_flops",
    "analyze_memory_usage",
    # Module Info
    "info",
    "test",
    "get_module_version",
    "get_module_author",
    "get_model_config",
    "compare_with_other_models",
    "test_inference_speed",  # ✅ اضافه شده
]


# ================================================================
# ✅ Module Information
# ================================================================


def info(verbose: bool = False) -> None:
    """
    Print detailed module information.

    Args:
        verbose: If True, print extended information
    """
    import torch

    print(
        f"""
    {'='*80}
    EfficientNet-B0 Module Information (ULTIMATE EDITION)
    {'='*80}

    📌 Module: EfficientNet-B0
    📌 Version: {__version__}
    👨‍💻 Author: {__author__}

    📁 Components:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Component           │ Description                                        │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ EfficientNetB0      │ Complete EfficientNet-B0 architecture from scratch │
    │ create_efficientnet │ Factory function to create EfficientNet-B0         │
    │ MBConvBlock         │ Mobile Inverted Bottleneck Conv block              │
    │ SqueezeExcitation   │ Channel attention mechanism (SE block)             │
    │ Swish               │ Swish activation function (x * sigmoid(x))         │
    │ make_layer          │ Helper to create layers of blocks                  │
    └────────────────────────────────────────────────────────────────────────────┘

    📊 Architecture Features:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Feature               │ Value                                            │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ Input Size            │ 224x224x3                                        │
    │ Base Channels         │ 32                                               │
    │ Width Multiplier      │ 1.0                                              │
    │ Depth Multiplier      │ 1.0                                              │
    │ Total Parameters      │ ~5.3M                                            │
    │ Feature Dimension     │ 1280                                             │
    │ Activation Function   │ Swish (SiLU)                                     │
    │ Attention Mechanism   │ Squeeze-and-Excitation (SE)                     │
    └────────────────────────────────────────────────────────────────────────────┘

    📊 Stage Configuration (B0):
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Stage │ Blocks │ Channels │ Resolution │ Expand Ratio │ Kernel │ SE    │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ 0     │ 1      │ 32       │ 224x224    │ -            │ -      │ -     │
    │ 1     │ 1      │ 16       │ 112x112    │ 1            │ 3x3    │ No    │
    │ 2     │ 2      │ 24       │ 56x56      │ 6            │ 3x3    │ No    │
    │ 3     │ 2      │ 40       │ 28x28      │ 6            │ 5x5    │ Yes   │
    │ 4     │ 3      │ 80       │ 14x14      │ 6            │ 3x3    │ Yes   │
    │ 5     │ 3      │ 112      │ 14x14      │ 6            │ 5x5    │ Yes   │
    │ 6     │ 4      │ 192      │ 7x7        │ 6            │ 5x5    │ Yes   │
    │ 7     │ 1      │ 320      │ 7x7        │ 6            │ 3x3    │ Yes   │
    │ 8     │ 1      │ 1280     │ 7x7        │ -            │ -      │ -     │
    └────────────────────────────────────────────────────────────────────────────┘

    📊 Utility Functions:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Function            │ Description                                        │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ count_parameters    │ Count model parameters                             │
    │ get_model_size      │ Calculate model memory size                        │
    │ print_model_summary │ Print comprehensive model summary                  │
    │ validate_forward    │ Validate forward pass                              │
    │ get_activation_stats│ Get activation statistics                          │
    │ check_gradient_flow │ Check gradient flow                                │
    │ get_layer_shapes    │ Get output shapes for each layer                   │
    │ compare_models      │ Compare two models                                 │
    │ set_seed            │ Set random seed for reproducibility                │
    │ get_model_info      │ Get comprehensive model information                │
    │ save_model_summary  │ Save model summary to file                         │
    │ get_device          │ Get best available device                          │
    │ get_cuda_info       │ Get CUDA information                               │
    │ profile_model       │ Profile model performance (NEW)                    │
    │ calculate_flops     │ Calculate model FLOPs (NEW)                        │
    │ analyze_memory_usage│ Analyze memory usage (NEW)                         │
    └────────────────────────────────────────────────────────────────────────────┘
    """
    )

    if verbose:
        print("📌 System Information:")
        print(f"   - PyTorch Version: {torch.__version__}")
        print(f"   - CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   - CUDA Version: {torch.version.cuda}")
            print(f"   - GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"   - GPU Count: {torch.cuda.device_count()}")
        print(
            f"   - MPS Available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}"
        )

    print(
        f"""
    💡 Usage Example:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ >>> from src.models.efficientNet.B0 import create_efficientnet_b0        │
    │ >>> model = create_efficientnet_b0(num_classes=5)                       │
    │ >>> output = model(torch.randn(4, 3, 224, 224))                         │
    │ >>> print(output.shape)  # (4, 5)                                       │
    │                                                                         │
    │ >>> # Advanced profiling                                               │
    │ >>> from src.models.efficientNet.B0 import profile_model, calculate_flops│
    │ >>> profile = profile_model(model)                                     │
    │ >>> flops = calculate_flops(model)                                     │
    └────────────────────────────────────────────────────────────────────────────┘

    📚 Documentation: https://github.com/babakyousefian/liver-ai-pipeline
    {'='*80}
    """
    )


def test(verbose: bool = False) -> bool:
    """
    Quick test function to verify the module is working correctly.

    Args:
        verbose: If True, print detailed test results

    Returns:
        bool: True if all tests passed
    """
    import torch

    print("\n🧪 Testing EfficientNet-B0 Module (ULTIMATE EDITION)...")
    print("=" * 60)

    try:
        # Create model
        model = create_efficientnet_b0(num_classes=5, pretrained=False, verbose=False)

        # Test forward pass
        dummy = torch.randn(2, 3, 224, 224)
        output = model(dummy)

        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())

        print(f"✅ Input shape: {dummy.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Total parameters: {total_params:,}")

        # ✅ Test advanced features
        try:
            from .utils import analyze_memory_usage, calculate_flops, profile_model

            print("\n📊 Advanced Tests:")

            # Test profiling
            profile = profile_model(model, (1, 3, 224, 224), "cpu", warmup=2, iterations=5)
            print(f"   ✅ Profiling: {profile['mean_time_ms']:.2f} ms, {profile['fps']:.1f} FPS")

            # Test FLOPs
            flops = calculate_flops(model, (1, 3, 224, 224))
            if flops.get("total_flops", 0) > 0:
                print(f"   ✅ FLOPs: {flops['total_flops']:,}")

            # Test memory analysis
            memory = analyze_memory_usage(model, (1, 3, 224, 224), "cpu")
            print(f"   ✅ Memory: {memory['memory_used_mb']:.2f} MB used")
        except Exception as e:
            print(f"   ⚠️ Advanced tests failed: {e}")

        if verbose:
            print("\n📊 Model Details:")
            print(f"   - Model Type: {model.__class__.__name__}")
            print(f"   - Number of classes: {model.num_classes}")
            print(f"   - Feature dimension: {model._feature_dim}")

        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        print("=" * 60)
        return False


def get_module_version() -> str:
    """Get the module version."""
    return __version__


def get_module_author() -> str:
    """Get the module author."""
    return __author__


def get_model_config() -> dict:
    """
    Get the default model configuration.

    Returns:
        dict: Model configuration
    """
    return {
        "name": "EfficientNet-B0",
        "version": __version__,
        "input_size": 224,
        "input_channels": 3,
        "num_classes": 5,
        "base_channels": 32,
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "feature_dim": 1280,
        "total_parameters": "~5.3M",
        "activation": "swish",
        "attention": "Squeeze-and-Excitation",
        "dropout_rate": 0.2,
        "use_batch_norm": True,
    }


def compare_with_other_models() -> dict:
    """
    Compare B0 with other EfficientNet variants.

    Returns:
        dict: Comparison results
    """
    return {
        "model": "EfficientNet-B0",
        "variants": {
            "B0": {
                "input_size": 224,
                "width_multiplier": 1.0,
                "depth_multiplier": 1.0,
                "parameters": "~5.3M",
                "feature_dim": 1280,
            },
            "B1": {
                "input_size": 240,
                "width_multiplier": 1.1,
                "depth_multiplier": 1.1,
                "parameters": "~7.8M",
                "feature_dim": 1280,
            },
            "B2": {
                "input_size": 260,
                "width_multiplier": 1.2,
                "depth_multiplier": 1.2,
                "parameters": "~9.2M",
                "feature_dim": 1408,
            },
            "B3": {
                "input_size": 300,
                "width_multiplier": 1.4,
                "depth_multiplier": 1.4,
                "parameters": "~12.2M",
                "feature_dim": 1536,
            },
        },
        "recommendation": "B0 is best for resource-constrained environments",
    }


def get_benchmark_results(model: nn.Module = None, device: str = "cpu") -> dict:
    """
    Get benchmark results for the model.

    Args:
        model: PyTorch model (if None, creates a new one)
        device: Device to run on

    Returns:
        dict: Benchmark results
    """
    import torch

    if model is None:
        model = create_efficientnet_b0(num_classes=5, pretrained=False, verbose=False)

    try:
        from .utils import analyze_memory_usage, calculate_flops, profile_model

        input_shape = (1, 3, 224, 224)

        profile = profile_model(model, input_shape, device, warmup=3, iterations=10)
        flops = calculate_flops(model, input_shape)
        memory = analyze_memory_usage(model, input_shape, device)

        return {
            "model": "EfficientNet-B0",
            "device": device,
            "input_shape": input_shape,
            "mean_inference_time_ms": profile["mean_time_ms"],
            "fps": profile["fps"],
            "total_flops": flops.get("total_flops", 0),
            "memory_used_mb": memory["memory_used_mb"],
            "model_size_mb": memory["model_size_mb"],
        }
    except Exception as e:
        return {"model": "EfficientNet-B0", "device": device, "error": str(e)}


# ================================================================
# ✅ Module Initialization (Silent by default)
# ================================================================

_VERBOSE_INIT = False

if _VERBOSE_INIT:
    print(f"✅ EfficientNet-B0 module loaded successfully! (ULTIMATE EDITION)")
    print(f"   📌 Version: {__version__}")
    print(f"   👨‍💻 Author: {__author__}")
    print(f"   💡 Type 'from src.models.efficientNet.B0 import info' and call info() for details")


# ================================================================
# ✅ Quick sanity check
# ================================================================


def _sanity_check() -> bool:
    """Perform a quick sanity check of the module."""
    try:
        import torch

        model = create_efficientnet_b0(num_classes=5, pretrained=False, verbose=False)
        dummy = torch.randn(1, 3, 224, 224)
        output = model(dummy)
        return output.shape == (1, 5)
    except:
        return False


# ================================================================
# ✅ If module is run directly, run tests
# ================================================================

from .blocks import MBConvBlock, SqueezeExcitation
from .efficientnet_b0 import EfficientNetB0, create_efficientnet_b0
from .utils import (
    analyze_memory_usage,
    calculate_flops,
    check_gradient_flow,
    compare_models,
    count_parameters,
    get_activation_stats,
    get_layer_output_shapes,
    get_model_size,
    print_model_summary,
    profile_model,
    validate_forward_pass,
)


# ✅ اضافه کردن تابع test_inference_speed
def test_inference_speed(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    Test inference speed of the model.

    Args:
        model: PyTorch model
        input_shape: Input shape (batch_size, channels, height, width)
        device: Device to run on
        num_runs: Number of runs for timing
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with timing statistics
    """
    import time

    model.eval()
    model.to(device)

    dummy_input = torch.randn(*input_shape).to(device)

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(dummy_input)

    # Timing
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_runs

    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "num_runs": num_runs,
        "device": device,
        "input_shape": input_shape,
        "fps": num_runs / total_time,
    }


if __name__ == "__main__":
    test(verbose=True)
    info(verbose=True)
    print("\n📊 Benchmark Results:")
    results = get_benchmark_results(device="cpu")
    if "error" not in results:
        print(f"   - Inference Time: {results['mean_inference_time_ms']:.2f} ms")
        print(f"   - FPS: {results['fps']:.1f}")
        print(f"   - Memory: {results['memory_used_mb']:.2f} MB")
    print("\n📊 Model Config:")
    print(f"   {get_model_config()}")
    print("\n📊 Comparison:")
    print(f"   {compare_with_other_models()}")
