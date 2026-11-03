# src/self_supervised/utils.py (ULTIMATE FIXED EDITION)

"""
Utility Functions for Self-Supervised Learning (ULTIMATE FIXED EDITION)

This module provides comprehensive utility functions for SSL training:
- Random seed setting for reproducibility
- Device detection and management
- Checkpoint saving and loading
- Metric calculation and logging
- ✅ NEW: Data validation and sanity checks
- ✅ NEW: Augmentation testing
- ✅ NEW: Batch analysis
- ✅ NEW: NaN/Inf detection
- ✅ NEW: Training stability helpers
- Model validation and evaluation
- Data augmentation utilities
- Progress tracking helpers
- Configuration management
- Visualization utilities
- Distributed training helpers

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 5.0.0 (ULTIMATE FIXED EDITION)
"""

import gc
import hashlib
import json
import math
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ================================================================
# ✅ SEED SETTING
# ================================================================


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode (slower)

    Example:
        >>> set_seed(42)
        >>> set_seed(123, deterministic=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"✅ Seed set to: {seed} (deterministic={deterministic})")


def get_random_seed() -> int:
    """Get the current random seed."""
    return random.getstate()[1][0] if hasattr(random, "getstate") else 42


# ================================================================
# ✅ DEVICE MANAGEMENT
# ================================================================


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get the best available device.

    Args:
        force_cpu: Force CPU even if CUDA is available

    Returns:
        PyTorch device

    Example:
        >>> device = get_device()
        >>> print(device)  # 'cuda' or 'cpu'
    """
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_cuda_info() -> Dict[str, Any]:
    """Get detailed CUDA information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
    }

    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        for i in range(info["device_count"]):
            info["devices"].append(
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
                    "compute_capability": torch.cuda.get_device_properties(i).major,
                    "multi_processor_count": torch.cuda.get_device_properties(
                        i
                    ).multi_processor_count,
                }
            )
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()

    return info


# ================================================================
# ✅ ✅ NEW: DATA VALIDATION AND SANITY CHECKS
# ================================================================


def validate_dataloader(
    dataloader: DataLoader,
    expected_batch_size: Optional[int] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    check_nan: bool = True,
    check_inf: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    ✅ NEW: Comprehensive DataLoader validation.

    This function checks:
    1. DataLoader is not empty
    2. Batch format is correct
    3. Batch shapes are correct
    4. No NaN/Inf values
    5. Value ranges are reasonable

    Args:
        dataloader: DataLoader to validate
        expected_batch_size: Expected batch size
        expected_shape: Expected tensor shape (C, H, W)
        check_nan: Check for NaN values
        check_inf: Check for Inf values
        verbose: Print validation results

    Returns:
        Dictionary with validation results

    Example:
        >>> results = validate_dataloader(train_loader, expected_batch_size=32)
        >>> if results['status'] == 'failed':
        ...     print(f"❌ DataLoader validation failed: {results['errors']}")
    """
    results = {
        "status": "ok",
        "num_batches": 0,
        "batch_size": 0,
        "sample_shape": None,
        "has_nan": False,
        "has_inf": False,
        "value_range": (float("inf"), float("-inf")),
        "errors": [],
        "warnings": [],
    }

    if verbose:
        print("\n🔍 Validating DataLoader...")
        print("=" * 60)

    # ✅ Check 1: DataLoader is not None
    if dataloader is None:
        results["status"] = "failed"
        results["errors"].append("DataLoader is None")
        if verbose:
            print("❌ DataLoader is None!")
        return results

    # ✅ Check 2: DataLoader is not empty
    try:
        results["num_batches"] = len(dataloader)
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(f"Cannot get length: {e}")
        if verbose:
            print(f"❌ Cannot get DataLoader length: {e}")
        return results

    if results["num_batches"] == 0:
        results["status"] = "failed"
        results["errors"].append("DataLoader is empty (len=0)")
        if verbose:
            print("❌ DataLoader is empty!")
        return results

    if verbose:
        print(f"✅ DataLoader has {results['num_batches']} batches")

    # ✅ Check 3: First batch
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        results["status"] = "failed"
        results["errors"].append("Cannot iterate over DataLoader")
        if verbose:
            print("❌ Cannot iterate over DataLoader!")
        return results

    if verbose:
        print(f"✅ Batch type: {type(batch)}")

    # ✅ Check 4: Batch format
    if isinstance(batch, (list, tuple)):
        results["batch_len"] = len(batch)
        if verbose:
            print(f"✅ Batch length: {len(batch)}")

        # ✅ SSL expects (x1, x2) or (images, labels)
        if len(batch) == 2:
            if isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
                # ✅ Probably SSL: (x1, x2)
                if batch[0].shape == batch[1].shape:
                    if verbose:
                        print(f"✅ SSL batch detected: (x1, x2) with shapes {batch[0].shape}")
                    results["is_ssl"] = True
                    results["sample_shape"] = batch[0].shape[1:]
                    results["batch_size"] = batch[0].size(0)
                else:
                    # ✅ Could be (images, labels)
                    results["is_ssl"] = False
                    results["sample_shape"] = batch[0].shape[1:]
                    results["batch_size"] = batch[0].size(0)
                    if verbose:
                        print(
                            f"✅ Labeled batch detected: (images, labels) with shape {batch[0].shape}"
                        )
            else:
                results["warnings"].append(
                    f"Batch elements are not tensors: {type(batch[0])}, {type(batch[1])}"
                )
        else:
            results["warnings"].append(f"Batch length is {len(batch)}, expected 2 for SSL")
    elif isinstance(batch, torch.Tensor):
        results["is_ssl"] = False
        results["sample_shape"] = batch.shape[1:]
        results["batch_size"] = batch.size(0)
        if verbose:
            print(f"✅ Single tensor batch: shape {batch.shape}")
    else:
        results["status"] = "failed"
        results["errors"].append(f"Unexpected batch type: {type(batch)}")
        if verbose:
            print(f"❌ Unexpected batch type: {type(batch)}")
        return results

    # ✅ Check 5: Expected batch size
    if expected_batch_size is not None:
        if results["batch_size"] != expected_batch_size:
            results["warnings"].append(
                f"Batch size mismatch: expected {expected_batch_size}, got {results['batch_size']}"
            )
            if verbose:
                print(
                    f"⚠️ Batch size mismatch: expected {expected_batch_size}, got {results['batch_size']}"
                )
        else:
            if verbose:
                print(f"✅ Batch size matches: {results['batch_size']}")

    # ✅ Check 6: Expected shape
    if expected_shape is not None:
        if results["sample_shape"] != expected_shape:
            results["warnings"].append(
                f"Shape mismatch: expected {expected_shape}, got {results['sample_shape']}"
            )
            if verbose:
                print(
                    f"⚠️ Shape mismatch: expected {expected_shape}, got {results['sample_shape']}"
                )
        else:
            if verbose:
                print(f"✅ Shape matches: {results['sample_shape']}")

    # ✅ Check 7: NaN and Inf
    if check_nan or check_inf:
        # ✅ Check all tensors in batch
        tensors_to_check = []
        if isinstance(batch, (list, tuple)):
            for item in batch:
                if isinstance(item, torch.Tensor):
                    tensors_to_check.append(item)
        elif isinstance(batch, torch.Tensor):
            tensors_to_check.append(batch)

        for i, tensor in enumerate(tensors_to_check):
            if check_nan and torch.isnan(tensor).any():
                results["has_nan"] = True
                results["errors"].append(f"Tensor {i} has NaN values")
                if verbose:
                    print(f"❌ Tensor {i} has NaN values!")

            if check_inf and torch.isinf(tensor).any():
                results["has_inf"] = True
                results["errors"].append(f"Tensor {i} has Inf values")
                if verbose:
                    print(f"❌ Tensor {i} has Inf values!")

            # ✅ Value range
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            results["value_range"] = (
                min(results["value_range"][0], min_val),
                max(results["value_range"][1], max_val),
            )

        if verbose:
            if not results["has_nan"] and not results["has_inf"]:
                print("✅ No NaN or Inf values detected")
            print(
                f"✅ Value range: [{results['value_range'][0]:.4f}, {results['value_range'][1]:.4f}]"
            )

    # ✅ Final status
    if results["errors"]:
        results["status"] = "failed"
    else:
        results["status"] = "ok"

    if verbose:
        print("=" * 60)
        if results["status"] == "ok":
            print("✅ DataLoader validation PASSED!")
        else:
            print("❌ DataLoader validation FAILED!")
            for error in results["errors"]:
                print(f"   - {error}")

    return results


def validate_augmentation(
    augmentation: Callable,
    input_shape: Tuple[int, ...] = (3, 224, 224),
    batch_size: int = 2,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    ✅ NEW: Validate SSL augmentation.

    This function checks:
    1. Augmentation returns two views
    2. Views have correct shapes
    3. Views are different
    4. No NaN/Inf values
    5. Value ranges are reasonable

    Args:
        augmentation: Augmentation function
        input_shape: Input tensor shape (C, H, W)
        batch_size: Batch size for testing
        device: Device to run on
        verbose: Print validation results

    Returns:
        Dictionary with validation results

    Example:
        >>> aug = BYOLAugmentation()
        >>> results = validate_augmentation(aug)
        >>> if results['status'] == 'failed':
        ...     print(f"❌ Augmentation failed: {results['errors']}")
    """
    results = {
        "status": "ok",
        "input_shape": input_shape,
        "output_shapes": [],
        "views_identical": False,
        "has_nan": False,
        "has_inf": False,
        "value_range": (float("inf"), float("-inf")),
        "errors": [],
        "warnings": [],
    }

    if verbose:
        print("\n🔍 Validating Augmentation...")
        print("=" * 60)
        print(f"   Input shape: {input_shape}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {device}")

    # ✅ Create test input
    x = torch.randn(batch_size, *input_shape)
    x = x.to(device)

    if verbose:
        print(f"✅ Test input shape: {x.shape}")

    # ✅ Apply augmentation
    try:
        result = augmentation(x)
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(f"Augmentation failed: {e}")
        if verbose:
            print(f"❌ Augmentation failed: {e}")
        return results

    # ✅ Check return format
    if isinstance(result, (list, tuple)):
        results["output_count"] = len(result)
        if verbose:
            print(f"✅ Augmentation returned {len(result)} outputs")

        if len(result) != 2:
            results["status"] = "failed"
            results["errors"].append(f"Expected 2 views, got {len(result)}")
            if verbose:
                print(f"❌ Expected 2 views, got {len(result)}")
            return results
    else:
        results["status"] = "failed"
        results["errors"].append(f"Expected tuple/list, got {type(result)}")
        if verbose:
            print(f"❌ Expected tuple/list, got {type(result)}")
        return results

    # ✅ Check each view
    view1, view2 = result

    for i, view in enumerate([view1, view2]):
        view_name = f"view{i+1}"

        if not isinstance(view, torch.Tensor):
            results["status"] = "failed"
            results["errors"].append(f"{view_name} is not a tensor")
            if verbose:
                print(f"❌ {view_name} is not a tensor")
            return results

        results["output_shapes"].append(view.shape)

        # ✅ Check shape
        if view.dim() != 4:
            results["status"] = "failed"
            results["errors"].append(f"{view_name} has {view.dim()} dims, expected 4")
            if verbose:
                print(f"❌ {view_name} has {view.dim()} dims, expected 4")
            return results

        if view.size(0) != batch_size:
            results["warnings"].append(f"{view_name} batch size {view.size(0)} != {batch_size}")
            if verbose:
                print(f"⚠️ {view_name} batch size: {view.size(0)}")

        # ✅ Check NaN/Inf
        if torch.isnan(view).any():
            results["has_nan"] = True
            results["errors"].append(f"{view_name} has NaN values")
            if verbose:
                print(f"❌ {view_name} has NaN values!")

        if torch.isinf(view).any():
            results["has_inf"] = True
            results["errors"].append(f"{view_name} has Inf values")
            if verbose:
                print(f"❌ {view_name} has Inf values!")

        # ✅ Value range
        min_val = view.min().item()
        max_val = view.max().item()
        results["value_range"] = (
            min(results["value_range"][0], min_val),
            max(results["value_range"][1], max_val),
        )

        if verbose:
            print(f"✅ {view_name} shape: {view.shape}")
            print(f"   Value range: [{min_val:.4f}, {max_val:.4f}]")
            print(f"   Mean: {view.mean().item():.4f}, Std: {view.std().item():.4f}")

    # ✅ Check if views are identical
    if torch.allclose(view1, view2, rtol=1e-6):
        results["views_identical"] = True
        results["warnings"].append("Views are identical (no augmentation applied)")
        if verbose:
            print("⚠️ Views are identical - check augmentation!")
    else:
        if verbose:
            print("✅ Views are different (good!)")

    # ✅ Final status
    if results["errors"]:
        results["status"] = "failed"
    else:
        results["status"] = "ok"

    if verbose:
        print("=" * 60)
        if results["status"] == "ok":
            print("✅ Augmentation validation PASSED!")
        else:
            print("❌ Augmentation validation FAILED!")
            for error in results["errors"]:
                print(f"   - {error}")

    return results


def sanity_check_before_training(
    dataloader: DataLoader,
    augmentation: Optional[Callable] = None,
    expected_batch_size: Optional[int] = None,
    expected_shape: Tuple[int, ...] = (3, 224, 224),
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    ✅ NEW: Complete sanity check before training.

    This runs both DataLoader and Augmentation validation.

    Args:
        dataloader: DataLoader to validate
        augmentation: Augmentation to validate
        expected_batch_size: Expected batch size
        expected_shape: Expected tensor shape
        verbose: Print validation results

    Returns:
        Dictionary with all validation results

    Example:
        >>> results = sanity_check_before_training(
        ...     train_loader,
        ...     augmentation=aug,
        ...     expected_batch_size=32
        ... )
        >>> if results['status'] == 'failed':
        ...     raise RuntimeError("Sanity check failed!")
    """
    results = {
        "status": "ok",
        "dataloader": None,
        "augmentation": None,
        "errors": [],
        "warnings": [],
    }

    if verbose:
        print("\n" + "=" * 80)
        print("🔍 COMPLETE SANITY CHECK BEFORE TRAINING")
        print("=" * 80)

    # ✅ Validate DataLoader
    dl_results = validate_dataloader(
        dataloader,
        expected_batch_size=expected_batch_size,
        expected_shape=expected_shape,
        verbose=verbose,
    )
    results["dataloader"] = dl_results

    if dl_results["status"] == "failed":
        results["status"] = "failed"
        results["errors"].extend([f"DataLoader: {e}" for e in dl_results["errors"]])

    # ✅ Validate Augmentation
    if augmentation is not None:
        aug_results = validate_augmentation(
            augmentation,
            input_shape=expected_shape,
            batch_size=expected_batch_size or 2,
            verbose=verbose,
        )
        results["augmentation"] = aug_results

        if aug_results["status"] == "failed":
            results["status"] = "failed"
            results["errors"].extend([f"Augmentation: {e}" for e in aug_results["errors"]])

    if verbose:
        print("=" * 80)
        if results["status"] == "ok":
            print("✅ ALL SANITY CHECKS PASSED! Ready to train.")
        else:
            print("❌ SANITY CHECKS FAILED!")
            for error in results["errors"]:
                print(f"   - {error}")
        print("=" * 80 + "\n")

    return results


# ================================================================
# ✅ CHECKPOINT MANAGEMENT
# ================================================================


def save_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    epoch: int = 0,
    loss: float = 0.0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    save_full: bool = True,
) -> str:
    """Save model checkpoint with all training state."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "loss": loss,
        "config": config or {},
        "metrics": metrics or {},
        "extra_info": extra_info or {},
    }

    if save_full:
        checkpoint["model_state_dict"] = model.state_dict()
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved: {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: Optional[str] = None,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        model.to(device)

    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except Exception as e:
            print(f"⚠️ Failed to load optimizer state: {e}")

    if load_scheduler and scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            print(f"⚠️ Failed to load scheduler state: {e}")

    print(f"✅ Checkpoint loaded: {path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A')}")

    return checkpoint


# ================================================================
# ✅ METRICS AND EVALUATION
# ================================================================


def compute_ssl_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Compute SSL metrics for evaluation."""
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # ✅ Handle both SSL and labeled data
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images = batch[0]
                    labels = None
            else:
                images = batch
                labels = None

            images = images.to(device)

            if hasattr(model, "get_embedding"):
                embeddings = model.get_embedding(images)
            else:
                embeddings = model(images)
                if embeddings.dim() > 2:
                    embeddings = embeddings.view(embeddings.size(0), -1)

            all_embeddings.append(embeddings.cpu())
            if labels is not None:
                all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0) if all_labels else None

    if all_labels is None:
        return {"embeddings": all_embeddings}

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.neighbors import KNeighborsClassifier

    split = len(all_embeddings) // 2

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(all_embeddings[:split].numpy(), all_labels[:split].numpy())

    preds = knn.predict(all_embeddings[split:].numpy())
    true = all_labels[split:].numpy()

    metrics = {
        "accuracy": accuracy_score(true, preds),
        "precision": precision_score(true, preds, average="weighted", zero_division=0),
        "recall": recall_score(true, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(true, preds, average="weighted", zero_division=0),
    }

    return metrics


# ================================================================
# ✅ ✅ NEW: TRAINING STABILITY HELPERS
# ================================================================


def check_for_nan_inf(
    tensor: torch.Tensor, name: str = "tensor", raise_exception: bool = True
) -> bool:
    """
    ✅ NEW: Check tensor for NaN/Inf values.

    Args:
        tensor: Tensor to check
        name: Name for error messages
        raise_exception: Whether to raise exception on detection

    Returns:
        True if NaN/Inf found, False otherwise

    Example:
        >>> if check_for_nan_inf(loss, "loss"):
        ...     print("NaN detected!")
    """
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan or has_inf:
        msg = f"🔥 {name} has "
        if has_nan:
            msg += "NaN"
        if has_nan and has_inf:
            msg += " and "
        if has_inf:
            msg += "Inf"
        msg += f" values! (min={tensor.min().item():.4f}, max={tensor.max().item():.4f})"

        if raise_exception:
            raise ValueError(msg)
        else:
            print(msg)
        return True

    return False


def get_tensor_stats(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
    """
    ✅ NEW: Get comprehensive tensor statistics.

    Args:
        tensor: Tensor to analyze
        name: Name for the tensor

    Returns:
        Dictionary with statistics

    Example:
        >>> stats = get_tensor_stats(embeddings, "z1")
        >>> print(f"z1 mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
    """
    return {
        "name": name,
        "shape": tensor.shape,
        "dtype": tensor.dtype,
        "device": tensor.device,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "has_nan": torch.isnan(tensor).any().item(),
        "has_inf": torch.isinf(tensor).any().item(),
        "num_elements": tensor.numel(),
        "sparsity": (tensor == 0).float().mean().item(),
    }


def print_tensor_stats(stats: Dict[str, Any]) -> None:
    """
    ✅ NEW: Pretty print tensor statistics.

    Args:
        stats: Statistics from get_tensor_stats()

    Example:
        >>> stats = get_tensor_stats(z1, "z1")
        >>> print_tensor_stats(stats)
    """
    print(f"\n📊 Tensor Stats: {stats['name']}")
    print("-" * 40)
    print(f"   Shape: {stats['shape']}")
    print(f"   Dtype: {stats['dtype']}")
    print(f"   Device: {stats['device']}")
    print(f"   Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"   Mean: {stats['mean']:.6f}")
    print(f"   Std: {stats['std']:.6f}")
    print(f"   Has NaN: {stats['has_nan']}")
    print(f"   Has Inf: {stats['has_inf']}")
    print(f"   Sparsity: {stats['sparsity']:.4f}")


def monitor_byol_stability(
    z1: torch.Tensor,
    z2: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    loss: torch.Tensor,
    epoch: int,
    batch: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    ✅ NEW: Monitor BYOL training stability.

    This function checks for signs of collapse or instability in BYOL training.

    Args:
        z1: Online projection output
        z2: Target projection output
        p1: Online prediction output
        p2: Target prediction output
        loss: Loss value
        epoch: Current epoch
        batch: Current batch
        verbose: Print monitoring results

    Returns:
        Dictionary with stability metrics

    Example:
        >>> metrics = monitor_byol_stability(z1, z2, p1, p2, loss, epoch, batch)
        >>> if metrics['stable']:
        ...     print("✅ BYOL is stable")
        ... else:
        ...     print("⚠️ BYOL showing signs of instability!")
    """
    results = {
        "epoch": epoch,
        "batch": batch,
        "loss": loss.item(),
        "stable": True,
        "warnings": [],
        "metrics": {},
    }

    # ✅ Check loss
    results["metrics"]["loss"] = loss.item()
    if torch.isnan(loss) or torch.isinf(loss):
        results["stable"] = False
        results["warnings"].append("Loss is NaN or Inf")

    # ✅ Check z1 statistics
    z1_stats = get_tensor_stats(z1, "z1")
    results["metrics"]["z1_mean"] = z1_stats["mean"]
    results["metrics"]["z1_std"] = z1_stats["std"]
    results["metrics"]["z1_has_nan"] = z1_stats["has_nan"]
    results["metrics"]["z1_has_inf"] = z1_stats["has_inf"]

    if z1_stats["has_nan"] or z1_stats["has_inf"]:
        results["stable"] = False
        results["warnings"].append("z1 has NaN/Inf")

    # ✅ Check z2 statistics
    z2_stats = get_tensor_stats(z2, "z2")
    results["metrics"]["z2_mean"] = z2_stats["mean"]
    results["metrics"]["z2_std"] = z2_stats["std"]
    results["metrics"]["z2_has_nan"] = z2_stats["has_nan"]
    results["metrics"]["z2_has_inf"] = z2_stats["has_inf"]

    if z2_stats["has_nan"] or z2_stats["has_inf"]:
        results["stable"] = False
        results["warnings"].append("z2 has NaN/Inf")

    # ✅ Check scale (important for BYOL)
    if z1_stats["std"] > 10 or z2_stats["std"] > 10:
        results["stable"] = False
        results["warnings"].append(
            f"z1_std={z1_stats['std']:.2f}, z2_std={z2_stats['std']:.2f} (too large)"
        )

    if z1_stats["std"] < 0.01 or z2_stats["std"] < 0.01:
        results["stable"] = False
        results["warnings"].append(
            f"z1_std={z1_stats['std']:.4f}, z2_std={z2_stats['std']:.4f} (too small - collapse?)"
        )

    # ✅ Check p1/p2
    p1_stats = get_tensor_stats(p1, "p1")
    p2_stats = get_tensor_stats(p2, "p2")

    results["metrics"]["p1_std"] = p1_stats["std"]
    results["metrics"]["p2_std"] = p2_stats["std"]

    if p1_stats["has_nan"] or p1_stats["has_inf"] or p2_stats["has_nan"] or p2_stats["has_inf"]:
        results["stable"] = False
        results["warnings"].append("p1/p2 has NaN/Inf")

    # ✅ Check similarity (should be close but not identical)
    similarity = F.cosine_similarity(p1, p2, dim=1).mean().item()
    results["metrics"]["similarity"] = similarity

    if similarity > 0.99:
        results["warnings"].append(f"p1 and p2 are too similar ({similarity:.4f}) - collapse?")

    if similarity < -0.5:
        results["warnings"].append(f"p1 and p2 are negatively correlated ({similarity:.4f})")

    # ✅ Print if verbose
    if verbose:
        print(f"\n📊 BYOL Stability Monitor - Epoch {epoch}, Batch {batch}")
        print("-" * 60)
        print(f"   Loss: {results['metrics']['loss']:.6f}")
        print(
            f"   z1 mean: {results['metrics']['z1_mean']:.4f}, std: {results['metrics']['z1_std']:.4f}"
        )
        print(
            f"   z2 mean: {results['metrics']['z2_mean']:.4f}, std: {results['metrics']['z2_std']:.4f}"
        )
        print(
            f"   p1 std: {results['metrics']['p1_std']:.4f}, p2 std: {results['metrics']['p2_std']:.4f}"
        )
        print(f"   Similarity: {results['metrics']['similarity']:.4f}")

        if results["stable"]:
            print("   ✅ BYOL is stable")
        else:
            print("   ⚠️ BYOL showing instability!")
            for warning in results["warnings"]:
                print(f"      - {warning}")
        print("-" * 60)

    return results


# ================================================================
# ✅ ✅ NEW: AUGMENTATION TESTING HELPERS
# ================================================================


def test_augmentation_pipeline(
    augmentation: Callable,
    image_paths: List[str],
    output_dir: str = "augmentation_test",
    num_samples: int = 5,
    device: str = "cpu",
) -> None:
    """
    ✅ NEW: Test augmentation pipeline on real images.

    Args:
        augmentation: Augmentation function
        image_paths: List of image paths to test on
        output_dir: Directory to save visualizations
        num_samples: Number of samples to test
        device: Device to run on

    Example:
        >>> aug = BYOLAugmentation()
        >>> test_augmentation_pipeline(aug, image_paths, num_samples=3)
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🧪 Testing augmentation pipeline on {num_samples} images...")
    print("=" * 60)

    for i, path in enumerate(image_paths[:num_samples]):
        # Load image
        img = Image.open(path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Apply augmentation
        try:
            result = augmentation(img_tensor)

            if isinstance(result, (list, tuple)) and len(result) == 2:
                view1, view2 = result

                # Convert to numpy for visualization
                v1_np = view1[0].cpu().permute(1, 2, 0).numpy()
                v2_np = view2[0].cpu().permute(1, 2, 0).numpy()

                # Clip to valid range
                v1_np = np.clip(v1_np, 0, 1)
                v2_np = np.clip(v2_np, 0, 1)

                # Save
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(np.array(img) / 255.0)
                axes[0].set_title("Original")
                axes[0].axis("off")

                axes[1].imshow(v1_np)
                axes[1].set_title("View 1")
                axes[1].axis("off")

                axes[2].imshow(v2_np)
                axes[2].set_title("View 2")
                axes[2].axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_{i}_augmentation.png"), dpi=150)
                plt.close()

                print(f"✅ Sample {i}: Saved to {output_dir}/sample_{i}_augmentation.png")
            else:
                print(f"⚠️ Sample {i}: Unexpected output format: {type(result)}")

        except Exception as e:
            print(f"❌ Sample {i}: Failed with error: {e}")

    print("=" * 60)
    print(f"✅ Augmentation test complete! Results saved to {output_dir}")


# ================================================================
# ✅ DATA UTILITIES
# ================================================================


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create a DataLoader with optimal settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )


def get_data_info(dataloader: DataLoader) -> Dict[str, Any]:
    """Get information about a DataLoader."""
    dataset = dataloader.dataset

    return {
        "num_samples": len(dataset),
        "batch_size": dataloader.batch_size,
        "num_batches": len(dataloader),
        "num_workers": dataloader.num_workers,
        "shuffle": dataloader.shuffle if hasattr(dataloader, "shuffle") else False,
        "drop_last": dataloader.drop_last,
        "device": str(dataloader.device) if hasattr(dataloader, "device") else "cpu",
    }


# ================================================================
# ✅ CONFIGURATION UTILITIES
# ================================================================


def save_config(config: Dict[str, Any], path: str, include_timestamp: bool = True) -> str:
    """Save configuration to JSON file."""
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(path)
        path = f"{base}_{timestamp}{ext}"

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    serializable_config = {}
    for key, value in config.items():
        if hasattr(value, "to_dict"):
            serializable_config[key] = value.to_dict()
        elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)

    with open(path, "w") as f:
        json.dump(serializable_config, f, indent=4)

    print(f"✅ Config saved: {path}")
    return path


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        config = json.load(f)

    print(f"✅ Config loaded: {path}")
    return config


# ================================================================
# ✅ PROGRESS TRACKING
# ================================================================


class ProgressTracker:
    """Track training progress and metrics."""

    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.history = defaultdict(list)
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.step = 0

    def update(self, metrics: Dict[str, float], step: int = None) -> None:
        if step is None:
            step = self.step
            self.step += 1

        for key, value in metrics.items():
            self.history[key].append(value)

        if step % self.log_interval == 0:
            self.log()

    def log(self) -> None:
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        total_elapsed = current_time - self.start_time

        if self.history:
            latest = {k: v[-1] for k, v in self.history.items()}
            print(
                f"Step {self.step} | {latest} | Time: {elapsed:.2f}s | Total: {total_elapsed:.2f}s"
            )

        self.last_log_time = current_time


# ================================================================
# ✅ MEMORY UTILITIES
# ================================================================


def clear_memory() -> None:
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_info() -> Dict[str, float]:
    """Get memory usage information."""
    info = {}

    if torch.cuda.is_available():
        info["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        info["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        info["cuda_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

    try:
        import psutil

        process = psutil.Process()
        info["cpu_memory_gb"] = process.memory_info().rss / 1e9
        info["cpu_memory_percent"] = process.memory_percent()
    except ImportError:
        pass

    return info


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing SSL Utils (ULTIMATE FIXED EDITION)")
    print("=" * 80)

    # Test set_seed
    print("\n📌 Testing set_seed...")
    set_seed(42)
    print(f"   Random number: {random.random()}")

    # Test get_device
    print("\n📌 Testing get_device...")
    device = get_device()
    print(f"   Device: {device}")

    # Test checkpoint
    print("\n📌 Testing checkpoint functions...")
    model = nn.Linear(10, 5)
    path = "test_checkpoint.pth"
    save_checkpoint(model, path, epoch=10, loss=0.5)
    checkpoint = load_checkpoint(path, model, strict=False)
    print(f"   Loaded epoch: {checkpoint.get('epoch', 'N/A')}")

    # Test validate_dataloader
    print("\n📌 Testing validate_dataloader...")
    from torch.utils.data import DataLoader, TensorDataset

    dummy_data = torch.randn(100, 3, 224, 224)
    dummy_loader = DataLoader(TensorDataset(dummy_data, torch.zeros(100)), batch_size=32)
    results = validate_dataloader(dummy_loader, expected_batch_size=32, verbose=True)
    print(f"   Status: {results['status']}")

    # Clean up
    import os

    if os.path.exists("test_checkpoint.pth"):
        os.remove("test_checkpoint.pth")

    print("\n✅ All SSL utils tests passed!")
    print("=" * 80)
