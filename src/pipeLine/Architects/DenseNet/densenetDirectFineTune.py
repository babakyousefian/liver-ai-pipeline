# src/pipeLine/densenetDirectFineTune.py
# -*- coding: utf-8 -*-
"""
Direct Fine-Tune DenseNet121 (ImageNet pretrained) — bypass SSL.
Best for small datasets (< 5k images) where SSL collapses.
"""

import os
import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from src.pipeLine.Architects.DenseNet.densenetFineTune import (
    run_densenet_finetune_pipeline,
    DENSENET_WEIGHT_TRANSFORMER_DIR,
)


def run_direct_densenet_finetune():
    """Fine-tune DenseNet121 from ImageNet weights directly (no SSL)."""

    print("\n" + "=" * 80)
    print("🏆 DenseNet121 Direct Fine-Tune (ImageNet pretrained)")
    print("=" * 80)
    print("   Skipping SSL — using ImageNet weights directly.")
    print("   Reason: SSL collapses on small datasets (< 5k images).")
    print("=" * 80)

    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 4)

    os.makedirs(DENSENET_WEIGHT_TRANSFORMER_DIR, exist_ok=True)
    ckpt_path = os.path.join(
        DENSENET_WEIGHT_TRANSFORMER_DIR,
        "densenet_transferred_from_imagenet.pth",
    )

    torch.save({
        "encoder_state_dict": model.state_dict(),
        "source_method": "imagenet",
        "source_model": "DenseNet121",
        "target_model": "DenseNet121",
        "best_lp_accuracy": 0.0,
        "transferred_layers": len(model.state_dict()),
        "total_layers": len(model.state_dict()),
    }, ckpt_path)

    print(f"✅ ImageNet checkpoint saved: {ckpt_path}")

    return run_densenet_finetune_pipeline(
        ssl_checkpoint_path=ckpt_path,
        ssl_method="imagenet",
        epochs_final=60,
        epochs_cv=20,
        lr_final=1e-4,
        lr_cv=1e-4,
        batch_size=16,
        gradient_accumulation_steps=2,
        n_tta=16,
        cv_folds=5,
        n_bootstrap=1000,
        patience_final=15,
        patience_cv=8,
        input_size=224,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )
