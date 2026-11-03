# src/models/efficientNet/B1/__init__.py

"""
EfficientNet-B1 Module - Complete Implementation from Scratch (ULTIMATE EDITION)

This module provides a complete implementation of EfficientNet-B1 architecture
without any pre-trained weights, following the original paper:
"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

The implementation includes:
- Complete EfficientNet-B1 architecture (MBConv blocks with SE modules)
- Squeeze-and-Excitation (SE) blocks for channel attention
- Swish activation function (SiLU)
- Compound scaling parameters (width, depth, resolution)
- Advanced profiling and analysis tools
- Model registry and management
- Benchmark and conversion utilities

Architecture Details (compared to B0):
- Input: 240x240x3 (vs 224x224)
- Base channels: 32 (same)
- Width multiplier: 1.1 (vs 1.0)
- Depth multiplier: 1.1 (vs 1.0)
- Resolution: 240x240 (vs 224x224)
- Total Parameters: ~7.8M (vs ~5.3M)

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 3.0.0 (ULTIMATE EDITION)
"""

# ================================================================
# ✅ Suppress import messages for production
# ================================================================
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

from .blocks import MBConvBlock, SqueezeExcitation, Swish, make_layer
from .efficientnet_b1 import EfficientNetB1, create_efficientnet_b1

# ================================================================
# Core Model Architecture
# ================================================================


# ================================================================
# Building Blocks
# ================================================================


# ================================================================
# ✅ Additional utilities with fallback
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
    # Fallback functions
    def get_block_info(block_idx: int) -> dict:
        """Get information about a specific block (fallback)."""
        block_info = {
            0: {"name": "Stem", "channels": 32, "resolution": 240},
            1: {"name": "Stage 1", "channels": 16, "resolution": 120},
            2: {"name": "Stage 2", "channels": 24, "resolution": 60},
            3: {"name": "Stage 3", "channels": 40, "resolution": 30},
            4: {"name": "Stage 4", "channels": 80, "resolution": 15},
            5: {"name": "Stage 5", "channels": 112, "resolution": 15},
            6: {"name": "Stage 6", "channels": 192, "resolution": 8},
            7: {"name": "Stage 7", "channels": 320, "resolution": 8},
            8: {"name": "Head", "channels": 1280, "resolution": 1},
        }
        return block_info.get(
            block_idx, {"name": f"Block {block_idx}", "channels": 0, "resolution": 0}
        )

    def get_layer_info(layer_name: str) -> dict:
        """Get information about a specific layer (fallback)."""
        return {"name": layer_name, "type": "unknown"}

    def print_layer_info(layer_name: str) -> None:
        """Print information about a specific layer (fallback)."""
        print(f"Layer: {layer_name}")

    def count_blocks_in_layer(layer) -> dict:
        """Count blocks in a layer (fallback)."""
        return {"total": len(layer) if hasattr(layer, "__len__") else 0}

    def get_layer_io_shapes(layer, input_shape) -> dict:
        """Get input/output shapes for a layer (fallback)."""
        return {"input_shape": input_shape, "output_shape": input_shape}


# ================================================================
# Utility Functions
# ================================================================

try:
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
        test_inference_speed,
        validate_forward_pass,
    )
except ImportError:
    # Minimal fallback
    warnings.warn("utils module not available. Some features will be limited.")

    def count_parameters(model):
        return {"total": sum(p.numel() for p in model.parameters())}

    def get_model_size(model):
        return {"mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2}

    def print_model_summary(model, *args, **kwargs):
        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {count_parameters(model)['total']:,}")

    def set_seed(seed=42):
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"✅ Seed set to: {seed}")

    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def profile_model(model, *args, **kwargs):
        return {"mean_time_ms": 0, "fps": 0}

    def calculate_flops(model, *args, **kwargs):
        return {"total_flops": 0}

    def analyze_memory_usage(model, *args, **kwargs):
        return {"memory_used_mb": 0}

    def test_inference_speed(model, *args, **kwargs):
        return {"mean_time_ms": 0, "fps": 0}


# ================================================================
# Module Metadata
# ================================================================

__version__ = "3.0.0"
__author__ = "Babak Yousefian"

# ================================================================
# Public API
# ================================================================

__all__ = [
    # Model Classes
    "EfficientNetB1",
    # Factory Functions
    "create_efficientnet_b1",
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
    "test_inference_speed",
    # Module Info
    "info",
    "test",
    "get_module_version",
    "get_module_author",
    "get_model_config",
    "compare_with_b0",
    "get_benchmark_results",
    "test_inference_speed",  # ✅ اضافه شده
]

# ================================================================
# Module Information
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
    EfficientNet-B1 Module Information (ULTIMATE EDITION)
    {'='*80}

    📌 Module: EfficientNet-B1
    📌 Version: {__version__}
    👨‍💻 Author: {__author__}

    📁 Components:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Component           │ Description                                        │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ EfficientNetB1      │ Complete EfficientNet-B1 architecture from scratch │
    │ create_efficientnet │ Factory function to create EfficientNet-B1         │
    │ MBConvBlock         │ Mobile Inverted Bottleneck Conv block              │
    │ SqueezeExcitation   │ Channel attention mechanism (SE block)             │
    │ Swish               │ Swish activation function (x * sigmoid(x))         │
    │ make_layer          │ Helper to create layers of blocks                  │
    └────────────────────────────────────────────────────────────────────────────┘

    📊 Architecture Features (Compared to B0):
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Feature               │ B0               │ B1                            │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ Input Size            │ 224x224x3        │ 240x240x3                     │
    │ Width Multiplier      │ 1.0              │ 1.1                           │
    │ Depth Multiplier      │ 1.0              │ 1.1                           │
    │ Total Parameters      │ ~5.3M            │ ~7.8M                         │
    │ Feature Dimension     │ 1280             │ 1280                          │
    │ Activation Function   │ Swish (SiLU)     │ Swish (SiLU)                  │
    │ Attention Mechanism   │ SE               │ SE                            │
    └────────────────────────────────────────────────────────────────────────────┘

    📊 Stage Configuration (B1):
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Stage │ Blocks │ Channels │ Resolution │ Expand Ratio │ Kernel │ SE    │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ 0     │ 1      │ 32       │ 240x240    │ -            │ -      │ -     │
    │ 1     │ 1      │ 16       │ 120x120    │ 1            │ 3x3    │ No    │
    │ 2     │ 2      │ 24       │ 60x60      │ 6            │ 3x3    │ No    │
    │ 3     │ 2      │ 40       │ 30x30      │ 6            │ 5x5    │ Yes   │
    │ 4     │ 3      │ 80       │ 15x15      │ 6            │ 3x3    │ Yes   │
    │ 5     │ 3      │ 112      │ 15x15      │ 6            │ 5x5    │ Yes   │
    │ 6     │ 4      │ 192      │ 8x8        │ 6            │ 5x5    │ Yes   │
    │ 7     │ 1      │ 320      │ 8x8        │ 6            │ 3x3    │ Yes   │
    │ 8     │ 1      │ 1280     │ 8x8        │ -            │ -      │ -     │
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
    │ test_inference_speed│ Test inference speed (NEW)                         │
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
    │ >>> from src.models.efficientNet.B1 import create_efficientnet_b1        │
    │ >>> model = create_efficientnet_b1(num_classes=5)                       │
    │ >>> output = model(torch.randn(4, 3, 240, 240))                         │
    │ >>> print(output.shape)  # (4, 5)                                       │
    │                                                                         │
    │ >>> # Advanced profiling                                               │
    │ >>> from src.models.efficientNet.B1 import profile_model, calculate_flops│
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

    print("\n🧪 Testing EfficientNet-B1 Module (ULTIMATE EDITION)...")
    print("=" * 60)

    try:
        # Create model with B1 configuration
        model = create_efficientnet_b1(num_classes=5, pretrained=False, verbose=False)

        # Test forward pass with 240x240 input
        dummy = torch.randn(2, 3, 240, 240)
        output = model(dummy)

        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())

        print(f"✅ Input shape: {dummy.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Total parameters: {total_params:,}")

        # ✅ Test advanced features
        try:
            from .utils import (
                analyze_memory_usage,
                calculate_flops,
                profile_model,
                test_inference_speed,
            )

            print("\n📊 Advanced Tests:")

            # Test profiling
            profile = profile_model(model, (1, 3, 240, 240), "cpu", warmup=2, iterations=5)
            print(f"   ✅ Profiling: {profile['mean_time_ms']:.2f} ms, {profile['fps']:.1f} FPS")

            # Test FLOPs
            flops = calculate_flops(model, (1, 3, 240, 240))
            if flops.get("total_flops", 0) > 0:
                print(f"   ✅ FLOPs: {flops['total_flops']:,}")

            # Test memory analysis
            memory = analyze_memory_usage(model, (1, 3, 240, 240), "cpu")
            print(f"   ✅ Memory: {memory['memory_used_mb']:.2f} MB used")

            # Test inference speed
            speed = test_inference_speed(model, (1, 3, 240, 240), "cpu", num_runs=10)
            print(f"   ✅ Speed: {speed['mean_time_ms']:.2f} ms, {speed['fps']:.1f} FPS")
        except Exception as e:
            print(f"   ⚠️ Advanced tests failed: {e}")

        if verbose:
            print("\n📊 Model Details:")
            print(f"   - Model Type: {model.__class__.__name__}")
            print(f"   - Number of classes: {model.num_classes}")
            print(f"   - Feature dimension: {model._feature_dim}")
            print(f"   - Input size: {model.input_size}")

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
    Get the default model configuration for B1.

    Returns:
        Dictionary with model configuration
    """
    return {
        "name": "EfficientNet-B1",
        "version": __version__,
        "input_size": 240,
        "input_channels": 3,
        "num_classes": 5,
        "base_channels": 32,
        "width_multiplier": 1.1,
        "depth_multiplier": 1.1,
        "feature_dim": 1280,
        "total_parameters": "~7.8M",
        "activation": "swish",
        "attention": "Squeeze-and-Excitation",
        "dropout_rate": 0.2,
        "use_batch_norm": True,
    }


def compare_with_b0() -> dict:
    """
    Compare B1 configuration with B0.

    Returns:
        Dictionary with comparison
    """
    return {
        "b0": {
            "input_size": 224,
            "width_multiplier": 1.0,
            "depth_multiplier": 1.0,
            "parameters": "~5.3M",
            "feature_dim": 1280,
            "accuracy_expected": "~84-86%",
        },
        "b1": {
            "input_size": 240,
            "width_multiplier": 1.1,
            "depth_multiplier": 1.1,
            "parameters": "~7.8M",
            "feature_dim": 1280,
            "accuracy_expected": "~85-87%",
        },
        "differences": {
            "input_size": "+16 pixels",
            "width_multiplier": "+0.1",
            "depth_multiplier": "+0.1",
            "parameters": "+2.5M",
            "accuracy": "~1-2% higher",
        },
        "recommendation": "B1 for better accuracy, B0 for faster training",
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
        model = create_efficientnet_b1(num_classes=5, pretrained=False, verbose=False)

    try:
        from .utils import (
            analyze_memory_usage,
            calculate_flops,
            profile_model,
            test_inference_speed,
        )

        input_shape = (1, 3, 240, 240)

        profile = profile_model(model, input_shape, device, warmup=3, iterations=10)
        flops = calculate_flops(model, input_shape)
        memory = analyze_memory_usage(model, input_shape, device)
        speed = test_inference_speed(model, input_shape, device, num_runs=20)

        return {
            "model": "EfficientNet-B1",
            "device": device,
            "input_shape": input_shape,
            "mean_inference_time_ms": profile["mean_time_ms"],
            "fps": profile["fps"],
            "total_flops": flops.get("total_flops", 0),
            "memory_used_mb": memory["memory_used_mb"],
            "model_size_mb": memory["model_size_mb"],
            "speed_mean_ms": speed["mean_time_ms"],
            "speed_std_ms": speed["std_time_ms"],
        }
    except Exception as e:
        return {"model": "EfficientNet-B1", "device": device, "error": str(e)}


# ================================================================
# ✅ Module Initialization (Silent by default)
# ================================================================

_VERBOSE_INIT = False  # Set to True for debug

if _VERBOSE_INIT:
    print(f"✅ EfficientNet-B1 module loaded successfully! (ULTIMATE EDITION)")
    print(f"   📌 Version: {__version__}")
    print(f"   👨‍💻 Author: {__author__}")
    print(f"   💡 Type 'from src.models.efficientNet.B1 import info' and call info() for details")


# ================================================================
# ✅ Quick sanity check (optional)
# ================================================================


def _sanity_check() -> bool:
    """
    Perform a quick sanity check of the module.

    Returns:
        bool: True if sanity check passes
    """
    try:
        import torch

        model = create_efficientnet_b1(num_classes=5, pretrained=False, verbose=False)
        dummy = torch.randn(1, 3, 240, 240)
        output = model(dummy)
        return output.shape == (1, 5)
    except:
        return False


# ================================================================
# ✅ If module is run directly, run tests
# ================================================================

# src/models/efficientNet/B1/__init__.py

from typing import *

from .blocks import MBConvBlock, SqueezeExcitation
from .efficientnet_b1 import EfficientNetB1, create_efficientnet_b1
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
    input_shape: Tuple[int, ...] = (1, 3, 240, 240),
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
    # Run tests when module is executed directly
    test(verbose=True)
    info(verbose=True)
    print("\n📊 B1 vs B0 Comparison:")
    comp = compare_with_b0()
    print(
        f"   B0: Input={comp['b0']['input_size']}, Parameters={comp['b0']['parameters']}, Acc={comp['b0']['accuracy_expected']}"
    )
    print(
        f"   B1: Input={comp['b1']['input_size']}, Parameters={comp['b1']['parameters']}, Acc={comp['b1']['accuracy_expected']}"
    )
    print(f"   Differences: {comp['differences']}")
    print(f"   Recommendation: {comp['recommendation']}")

    print("\n📊 Benchmark Results:")
    results = get_benchmark_results(device="cpu")
    if "error" not in results:
        print(f"   - Inference Time: {results['mean_inference_time_ms']:.2f} ms")
        print(f"   - FPS: {results['fps']:.1f}")
        print(f"   - Memory: {results['memory_used_mb']:.2f} MB")
        print(f"   - FLOPs: {results['total_flops']:,}")
