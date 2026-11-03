# src/AI/visioTransformer/utils.py
# -*- coding: utf-8 -*-

"""
Vision Transformer Utils — Minimal stub
========================================
This module is intentionally minimal because VisioTransformerTrainer
uses `src.AI.ResNet.utils` for all shared utilities.

Author: Babak Yousefian
"""

# ✅ Re-export everything from ResNet utils for compatibility
from src.AI.ResNet.utils import (
    CLASS_NAMES,
    CROPPED,
    CSVAddress,
    OUTPUT_DIR,
    LABEL_MAP,
    INV_LABEL_MAP,
    NUM_CLASSES,
    LabelSmoothingCrossEntropy,
    WarmupCosineScheduler,
    RandomErasing,
    CutMix,
    MixUp,
    check_paths,
    compute_metrics,
    create_weighted_sampler,
    get_class_weights,
    get_device,
    load_model,
    normalize_resnet,
    print_metrics_summary,
    resize_with_padding,
    save_model,
    set_seed,
    validate_csv,
    validate_images,
    apply_advanced_augmentation,
    apply_augmentation,
    cross_validate_model,
    export_predictions,
    export_to_csv,
    load_split_data,
    LiverDataset,
)

__all__ = [
    "CLASS_NAMES", "CROPPED", "CSVAddress", "OUTPUT_DIR",
    "LABEL_MAP", "INV_LABEL_MAP", "NUM_CLASSES",
    "LabelSmoothingCrossEntropy", "WarmupCosineScheduler",
    "RandomErasing", "CutMix", "MixUp",
    "check_paths", "compute_metrics", "create_weighted_sampler",
    "get_class_weights", "get_device", "load_model",
    "normalize_resnet", "print_metrics_summary", "resize_with_padding",
    "save_model", "set_seed", "validate_csv", "validate_images",
    "apply_advanced_augmentation", "apply_augmentation",
    "cross_validate_model", "export_predictions", "export_to_csv",
    "load_split_data", "LiverDataset",
]