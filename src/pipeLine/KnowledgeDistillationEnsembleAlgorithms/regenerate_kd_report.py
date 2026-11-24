# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/regenerate_kd_report.py
# -*- coding: utf-8 -*-

"""
Regenerate KD Report from existing artifacts (NO retraining).
================================================================
Reads:
  - kd_swin_best.pth (or kd_densenet_best.pth)
  - kd_metrics.json
  - history_kd_swin.json (or history_kd_densenet.json)
  - advanced_eval.json
  - predictions.csv
  - training_meta_*.json

Then rebuilds:
  - All plots (confusion, ROC, metrics bar, per-class, gap, LR, etc.)
  - LIME / SHAP / Captum / Netron images
  - 24-page PDF report

Author: Babak Yousefian
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import DenseNet121_Weights, Swin_T_Weights
from sklearn.model_selection import train_test_split

from src.AI.ResNet.utils import (
    CLASS_NAMES, CROPPED, CSVAddress, validate_csv, LiverDataset,
)
from .kd_reporter import generate_kd_report


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def _build_densenet_student(num_classes: int, dropout: float = 0.3) -> nn.Module:
    """Build DenseNet-121 with the SAME head used during KD training."""
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.5),
        nn.Linear(512, num_classes),
    )
    return model


def _build_swin_student(num_classes: int, dropout: float = 0.25) -> nn.Module:
    """Build Swin-Tiny with the SAME head used during KD training."""
    model = models.swin_t(weights=None)
    in_features = model.head.in_features  # 768
    model.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(256, num_classes),
    )
    return model


def _build_test_loader(batch_size: int = 16, target_size: int = 224):
    """Rebuild the TEST split with the SAME seed used during training."""
    is_valid, df, msg = validate_csv(CSVAddress)
    if not is_valid:
        raise ValueError(f"CSV validation failed: {msg}")

    df = df[df["liverLabel"] != "UNKNOWN"].copy()
    counts = df["liverLabel"].value_counts()
    valid = counts[counts >= 2].index.tolist()
    df = df[df["liverLabel"].isin(valid)]

    train_val_df, test_df = train_test_split(
        df, test_size=0.10, stratify=df["liverLabel"], random_state=42,
    )

    test_ds = LiverDataset(
        test_df, CROPPED, augment=False,
        target_size=target_size, apply_normalize=True,
        use_advanced_augment=False, return_path=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return test_loader, test_ds


# ═════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════

def regenerate_densenet_report(
    densenet_dir: str = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\bestKDEnsembles\DenseNet121",
    device: str = "cuda",
) -> str:
    """
    Regenerate 24-page PDF report for DenseNet KD from existing artifacts.
    Returns path to the generated PDF.
    """
    return _regenerate_report_common(
        artifact_dir=densenet_dir,
        student_arch="DenseNet121",
        device=device,
    )


def regenerate_swin_report(
    swin_dir: str = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\bestKDEnsembles\SwinTransformer",
    device: str = "cuda",
) -> str:
    """
    Regenerate 24-page PDF report for Swin KD from existing artifacts.
    """
    return _regenerate_report_common(
        artifact_dir=swin_dir,
        student_arch="SwinTransformer",
        device=device,
    )


# ═════════════════════════════════════════════════════════════════════
# COMMON REGENERATION LOGIC
# ═════════════════════════════════════════════════════════════════════

def _regenerate_report_common(
    artifact_dir: str,
    student_arch: str,
    device: str = "cuda",
) -> str:
    print("\n" + "=" * 78)
    print(f"  📄 REGENERATING KD REPORT — {student_arch}")
    print("=" * 78)
    print(f"  Artifact dir: {artifact_dir}\n")

    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"Directory not found: {artifact_dir}")

    # ── Find files (extension is REQUIRED — no ambiguity) ──
    def _find_file(prefix: str, extension: str) -> Optional[str]:
        """Find a file starting with `prefix` and ending with `extension`."""
        candidates = []
        for fname in os.listdir(artifact_dir):
            if fname.startswith(prefix) and fname.endswith(extension):
                candidates.append(os.path.join(artifact_dir, fname))
        if not candidates:
            return None
        # Prefer exact match (e.g., `advanced_eval.json` over `advanced_eval_2.json`)
        exact = f"{prefix}{extension}"
        for c in candidates:
            if os.path.basename(c) == exact:
                return c
        return sorted(candidates)[0]

    if student_arch == "DenseNet121":
        best_ckpt = _find_file("kd_densenet_best", ".pth")
        metrics_json = _find_file("kd_metrics", ".json")
        history_json = _find_file("history_kd_densenet", ".json")
        adv_json = _find_file("advanced_eval", ".json")
        meta_json = _find_file("training_meta_kd_densenet", ".json")
    elif student_arch == "SwinTransformer":
        best_ckpt = _find_file("kd_swin_best", ".pth")
        metrics_json = _find_file("kd_metrics", ".json")
        history_json = _find_file("history_kd_swin", ".json")
        adv_json = _find_file("advanced_eval", ".json")
        meta_json = _find_file("training_meta_kd_swin", ".json")
    else:
        raise ValueError(f"Unsupported arch: {student_arch}")

    if best_ckpt is None or not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Best checkpoint not found in: {artifact_dir}")

    print(f"  ✅ Found checkpoint: {os.path.basename(best_ckpt)}")
    if metrics_json:
        print(f"  ✅ Found metrics: {os.path.basename(metrics_json)}")
    if history_json:
        print(f"  ✅ Found history: {os.path.basename(history_json)}")
    if adv_json:
        print(f"  ✅ Found advanced eval: {os.path.basename(adv_json)}")

    # ── Load metrics ──
    metrics_data = {}
    if metrics_json and os.path.exists(metrics_json):
        with open(metrics_json, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    metrics = metrics_data.get("metrics", {})
    config = metrics_data.get("config", {})
    adv_results = metrics_data.get("advanced_eval", {})
    if adv_json and os.path.exists(adv_json):
        try:
            with open(adv_json, "r", encoding="utf-8") as f:
                adv_results = json.load(f)
            print(f"  ✅ Loaded advanced_eval: {os.path.basename(adv_json)}")
        except Exception as e:
            print(f"  ⚠️ Could not load advanced_eval.json: {e}")
            print(f"  ↪ Falling back to metrics_data['advanced_eval']")

    # ── Load history ──
    history = {}
    if history_json and os.path.exists(history_json):
        with open(history_json, "r", encoding="utf-8") as f:
            history = json.load(f)

    # ── Load model ──
    print("\n  🏗️  Building student model...")
    if student_arch == "DenseNet121":
        model = _build_densenet_student(num_classes=len(CLASS_NAMES))
    else:
        model = _build_swin_student(num_classes=len(CLASS_NAMES))

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print(f"  ✅ Model loaded: {os.path.basename(best_ckpt)}")

    # ── Re-evaluate on test set (to get fresh y_true/y_pred/y_prob) ──
    print("\n  📊 Re-evaluating on test set...")
    test_loader, test_ds = _build_test_loader()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1)
            pred = prob.argmax(dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    from sklearn.metrics import accuracy_score, roc_auc_score, \
        precision_recall_fscore_support, classification_report, confusion_matrix

    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                         average="macro"))
    except Exception:
        auc_macro = 0.0

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        average=None, zero_division=0,
    )
    wa = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    report_str = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )

    test_metrics = {
        "accuracy": acc,
        "auc_macro": auc_macro,
        "precision_weighted": float(wa[0]),
        "recall_weighted": float(wa[1]),
        "f1_weighted": float(wa[2]),
        "precision_per_class": p.tolist(),
        "recall_per_class": r.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": s.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    print(f"  ✅ Test Acc: {acc:.4f} | AUC: {auc_macro:.4f}")

    # ── Generate PDF ──
    print("\n  📄 Generating PDF report...")
    pdf_path = generate_kd_report(
        model=model,
        teacher=None,                        # not needed for regeneration
        test_ds=test_ds,
        test_metrics=test_metrics,
        history=history,
        adv_results=adv_results,
        output_dir=artifact_dir,
        student_name=student_arch,
        teacher_name="ConvNext",
        config=config,
    )

    # ── Save predictions CSV ──
    pred_csv = os.path.join(artifact_dir, "predictions.csv")
    pd.DataFrame({
        "true_idx": y_true,
        "pred_idx": y_pred,
        "true_label": [CLASS_NAMES[i] for i in y_true],
        "pred_label": [CLASS_NAMES[i] for i in y_pred],
        "correct": (y_true == y_pred).astype(int),
        **{f"prob_{c}": y_prob[:, i] for i, c in enumerate(CLASS_NAMES)},
    }).to_csv(pred_csv, index=False)
    print(f"  ✅ Predictions saved: {pred_csv}")

    print("\n" + "=" * 78)
    print(f"  ✅ REPORT REGENERATED")
    print(f"  PDF: {pdf_path}")
    print("=" * 78 + "\n")

    return pdf_path