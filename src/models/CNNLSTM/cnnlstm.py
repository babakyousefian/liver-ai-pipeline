# src/models/CNNLSTM/cnnlstm.py
# -*- coding: utf-8 -*-

"""
CNNLSTM Architecture Implementation from Scratch
=================================================
Author: Babak Yousefian
"""

import gc
import json
import os
import random
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .blocks import CNNFeatureExtractor, LSTMHead, init_weights

warnings.filterwarnings("ignore")


# ================================================================
# CONFIG DATACLASS
# ================================================================
@dataclass
class CNNLSTMConfig:
    """Configuration class for CNNLSTM."""
    num_classes: int = 5
    input_channels: int = 3
    hidden_dim: int = 256
    num_layers: int = 2
    dropout_rate: float = 0.5
    use_batch_norm: bool = True
    activation: str = "relu"
    init_type: str = "kaiming"
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "init_type": self.init_type,
            "verbose": self.verbose,
        }


# ================================================================
# CNNLSTM MODEL
# ================================================================
class CNNLSTM(nn.Module):
    """Complete CNN-LSTM Architecture from Scratch."""

    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        activation: str = "relu",
        init_type: str = "kaiming",
        verbose: bool = False,
    ) -> None:
        super(CNNLSTM, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.init_type = init_type
        self.verbose = verbose
        self._config = CNNLSTMConfig(
            num_classes=num_classes,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_type=init_type,
            verbose=verbose,
        )

        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(
            input_channels=input_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_type=init_type,
        )

        # LSTM Head
        self.lstm_head = LSTMHead(
            input_dim=self.cnn.output_channels,  # 512
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            init_type=init_type,
        )

        self._stage_info = self._get_stage_info()

        if verbose:
            print(self.summary())

    def _get_stage_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            "cnn": {
                "name": "CNN Feature Extractor",
                "output_channels": self.cnn.output_channels,
                "parameters": sum(p.numel() for p in self.cnn.parameters()),
            },
            "lstm": {
                "name": "LSTM Head",
                "input_dim": self.cnn.output_channels,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_classes": self.num_classes,
                "parameters": sum(p.numel() for p in self.lstm_head.parameters()),
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        out = self.lstm_head(features)
        return out

    def get_trainable_parameters(self) -> Dict[str, Union[int, float]]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_percentage": (trainable / total) * 100 if total > 0 else 0.0,
        }

    def get_layer_parameters(self) -> Dict[str, int]:
        return {
            "cnn": sum(p.numel() for p in self.cnn.parameters()),
            "lstm_head": sum(p.numel() for p in self.lstm_head.parameters()),
        }

    def summary(self) -> str:
        info = self.get_trainable_parameters()
        layer_params = self.get_layer_parameters()
        return f"""
{'='*80}
CNNLSTM Architecture Summary
{'='*80}
Input Shape: (batch, {self.input_channels}, 224, 224)
Output Shape: (batch, {self.num_classes})
Parameters:
- Total: {info['total']:,}
- Trainable: {info['trainable']:,}
- Frozen: {info['frozen']:,}
{'='*80}
"""

    def __repr__(self) -> str:
        return f"CNNLSTM(num_classes={self.num_classes}, hidden_dim={self.hidden_dim}, dropout={self.dropout_rate})"


def create_cnnlstm(
    num_classes: int = 5, pretrained: bool = False, verbose: bool = True, **kwargs
) -> CNNLSTM:
    if pretrained:
        print("⚠️ Warning: pretrained=True ignored for Task 5")
    model = CNNLSTM(num_classes=num_classes, verbose=verbose, **kwargs)
    if verbose:
        print(model.summary())
    return model


# ================================================================
# ✅ UTILS FUNCTIONS (همه توی همین فایل)
# ================================================================
def count_parameters(model: nn.Module) -> Dict[str, Union[int, Dict[str, int], float]]:
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


def profile_model(model, input_shape=(1, 3, 224, 224), device="cpu", warmup=5, iterations=20):
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


def calculate_flops(model, input_shape=(1, 3, 224, 224)):
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


def analyze_memory_usage(model, input_shape=(1, 3, 224, 224), device="cpu"):
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


def test_inference_speed(model, input_shape=(1, 3, 224, 224), device="cpu", num_runs=100):
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


def print_model_summary(model, input_shape=(3, 224, 224), verbose=True, show_parameters=True, show_flops=True, device=None):
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


def validate_forward_pass(model, device="cpu", batch_size=4, input_shape=(3, 224, 224), verbose=True):
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


def get_layer_output_shapes(model, input_shape=(3, 224, 224), verbose=False, device=None):
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set: {seed}")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing CNNLSTM Utils")
    print("=" * 80)

    try:
        from .cnnlstm import create_cnnlstm

        model = create_cnnlstm(num_classes=5, verbose=False)
        print("✅ Model created")

        params = count_parameters(model)
        print(f"   Params: {params['total']:,}")

        size = get_model_size(model)
        print(f"   Size: {size['mb']:.2f} MB")

        print_model_summary(model, (3, 224, 224), verbose=False)

        print("\n✅ All CNNLSTM utils are ready!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 80)
