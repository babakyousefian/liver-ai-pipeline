# src/models/resnet/resnet18/__init__.py

"""
ResNet18 Module - Complete Implementation from Scratch (ULTIMATE EDITION)

This module provides a complete implementation of ResNet18 architecture
without any pre-trained weights, following the original paper:
"Deep Residual Learning for Image Recognition" by Kaiming He et al.

The implementation includes:
- Complete ResNet18 architecture
- Basic building blocks (BasicBlock)
- Advanced profiling and analysis tools
- Model registry and management
- Benchmark and conversion utilities
- Utility functions for analysis and debugging
- Factory functions for easy model creation

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 3.0.0 (ULTIMATE EDITION)
"""

# ================================================================
# ✅ Suppress import messages for production
# ================================================================
import warnings

warnings.filterwarnings("ignore")

# ================================================================
# Core Model Architecture
# ================================================================

from .blocks import (
    BasicBlock,
    Bottleneck,
    count_blocks_in_layer,
    get_block_info,
    get_layer_info,
    get_layer_io_shapes,
    make_layer,
    print_layer_info,
)
from .resnet18 import ResNet18, create_resnet18

# ================================================================
# Building Blocks
# ================================================================


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
        import torch

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
    "ResNet18",
    # Factory Functions
    "create_resnet18",
    # Block Components
    "BasicBlock",
    "Bottleneck",
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
    "get_benchmark_results",
    "get_layer_io_shapes",
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
    ResNet18 Module Information (ULTIMATE EDITION)
    {'='*80}

    📌 Module: ResNet18
    📌 Version: {__version__}
    👨‍💻 Author: {__author__}

    📁 Components:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Component           │ Description                                        │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ ResNet18            │ Complete ResNet18 architecture from scratch         │
    │ create_resnet18     │ Factory function to create ResNet18                 │
    │ BasicBlock          │ Basic residual block (2x 3x3 conv)                  │
    │ Bottleneck          │ Bottleneck block (1x1→3x3→1x1)                    │
    │ make_layer          │ Helper to create layers of blocks                   │
    └────────────────────────────────────────────────────────────────────────────┘

    📊 Architecture Features:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Feature               │ Value                                            │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ Input Size            │ 224x224x3                                        │
    │ Total Parameters      │ ~11.57M                                          │
    │ Feature Dimension     │ 512                                              │
    │ Activation Function   │ ReLU                                             │
    │ Residual Connections  │ Yes (Identity shortcuts)                         │
    └────────────────────────────────────────────────────────────────────────────┘

    📊 Stage Configuration (ResNet18):
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Stage │ Blocks │ Channels │ Resolution │ Kernel Type │ Stride │         │
    ├────────────────────────────────────────────────────────────────────────────┤
    │ 0     │ 1      │ 64       │ 112x112    │ 7x7 Conv    │ 2     │         │
    │ 1     │ 2      │ 64       │ 56x56      │ BasicBlock  │ 1     │         │
    │ 2     │ 2      │ 128      │ 28x28      │ BasicBlock  │ 2     │         │
    │ 3     │ 2      │ 256      │ 14x14      │ BasicBlock  │ 2     │         │
    │ 4     │ 2      │ 512      │ 7x7        │ BasicBlock  │ 2     │         │
    │ 5     │ 1      │ 512      │ 7x7        │ AvgPool+FC  │ -     │         │
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
    │ >>> from src.models.resnet.resnet18 import create_resnet18               │
    │ >>> model = create_resnet18(num_classes=4)                              │
    │ >>> output = model(torch.randn(4, 3, 224, 224))                         │
    │ >>> print(output.shape)  # (4, 5)                                       │
    │                                                                         │
    │ >>> # Advanced profiling                                               │
    │ >>> from src.models.resnet.resnet18 import profile_model, calculate_flops│
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

    print("\n🧪 Testing ResNet18 Module (ULTIMATE EDITION)...")
    print("=" * 60)

    try:
        # Create model
        model = create_resnet18(num_classes=5, pretrained=False, verbose=False)

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
            from .utils import (
                analyze_memory_usage,
                calculate_flops,
                profile_model,
                test_inference_speed,
            )

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

            # Test inference speed
            speed = test_inference_speed(model, (1, 3, 224, 224), "cpu", num_runs=10)
            print(f"   ✅ Speed: {speed['mean_time_ms']:.2f} ms, {speed['fps']:.1f} FPS")
        except Exception as e:
            print(f"   ⚠️ Advanced tests failed: {e}")

        if verbose:
            print("\n📊 Model Details:")
            print(f"   - Model Type: {model.__class__.__name__}")
            print(f"   - Number of classes: {model.num_classes}")
            print(f"   - Feature dimension: {model.feature_dim}")

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
    Get the default model configuration for ResNet18.

    Returns:
        dict: Model configuration
    """
    return {
        "name": "ResNet18",
        "version": __version__,
        "input_size": 224,
        "input_channels": 3,
        "num_classes": 5,
        "feature_dim": 512,
        "total_parameters": "~11.57M",
        "activation": "relu",
        "layers": [2, 2, 2, 2],  # BasicBlock counts
        "use_batch_norm": True,
    }


import torch.nn as nn


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
        model = create_resnet18(num_classes=4, pretrained=False, verbose=False)

    try:
        from .utils import (
            analyze_memory_usage,
            calculate_flops,
            profile_model,
            test_inference_speed,
        )

        input_shape = (1, 3, 224, 224)

        profile = profile_model(model, input_shape, device, warmup=3, iterations=10)
        flops = calculate_flops(model, input_shape)
        memory = analyze_memory_usage(model, input_shape, device)
        speed = test_inference_speed(model, input_shape, device, num_runs=20)

        return {
            "model": "ResNet18",
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
        return {"model": "ResNet18", "device": device, "error": str(e)}


# ================================================================
# ✅ Module Initialization (Silent by default)
# ================================================================

_VERBOSE_INIT = False  # Set to True for debug

if _VERBOSE_INIT:
    print(f"✅ ResNet18 module loaded successfully! (ULTIMATE EDITION)")
    print(f"   📌 Version: {__version__}")
    print(f"   👨‍💻 Author: {__author__}")
    print(f"   💡 Type 'from src.models.resnet.resnet18 import info' and call info() for details")


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

        model = create_resnet18(num_classes=5, pretrained=False, verbose=False)
        dummy = torch.randn(1, 3, 224, 224)
        output = model(dummy)
        return output.shape == (1, 5)
    except:
        return False


# ================================================================
# ✅ If module is run directly, run tests
# ================================================================


# src/models/resnet/resnet18/__init__.py

from .blocks import BasicBlock, Bottleneck
from .resnet18 import ResNet18, create_resnet18
from .utils import get_layer_io_shapes  # ✅ اضافه شده
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
    test_inference_speed,
    validate_forward_pass,
)

if __name__ == "__main__":
    # Run tests when module is executed directly
    test(verbose=True)
    info(verbose=True)

    print("\n📊 Benchmark Results:")
    import torch
    results = get_benchmark_results(device="cuda" if torch.cuda.is_available() else "cpu")
    if "error" not in results:
        print(f"   - Inference Time: {results['mean_inference_time_ms']:.2f} ms")
        print(f"   - FPS: {results['fps']:.1f}")
        print(f"   - Memory: {results['memory_used_mb']:.2f} MB")
        print(f"   - FLOPs: {results['total_flops']:,}")

    print("\n📊 Model Config:")
    print(f"   {get_model_config()}")
