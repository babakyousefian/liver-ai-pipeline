# src/AI/ConvNext/convnext_targeted_finetune.py
# -*- coding: utf-8 -*-
"""
ConvNeXt Targeted Fine-Tune — G1/G2 only
==========================================
Strategy:
  1. Load ConvNeXt baseline (FROZEN)
  2. Extract 768-d features
  3. Fine-tune ONLY the classifier head with:
     - Weighted CrossEntropy (G1=2x, G2=1.5x, NL=1x, G3=1x)
     - NO Mixup/CutMix
     - Early stopping on BALANCED accuracy
  4. Target: NL<4%, G1<12%, G2<7%, G3<15%

Key difference from v3:
  - Higher G1 weight (2x instead of 3x from CB)
  - NL protected explicitly (weight=1.0)
  - Early stop on balanced accuracy (not raw accuracy)
  - Save as a NEW model, not overwrite baseline

Why this should work:
  - Only head changes (backbone features preserved)
  - Weights are balanced (NL not forgotten)
  - No augmentation artifacts
"""

import os
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split

from src.AI.ResNet.utils import (
    CLASS_NAMES, CROPPED, CSVAddress, validate_csv, LiverDataset,
)

FINAL_REPORTS_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"
LABELS_CSV = os.path.join(FINAL_REPORTS_DIR, "labels.csv")
CROPPED_IMAGES_DIR = os.path.join(FINAL_REPORTS_DIR, "cropped")


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════
def sanity_check_data(csv_path, img_dir):
    print("\n" + "=" * 78)
    print("  🔍 DATA SANITY CHECK")
    print("=" * 78)
    report = {"overall_status": "FAILED"}
    if not os.path.exists(csv_path) or not os.path.isdir(img_dir):
        report["overall_status"] = "MISSING"
        return report
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        report["overall_status"] = "CSV_ERROR"
        return report
    label_col = next((c for c in ["liverLabel", "label"] if c in df.columns), None)
    if not label_col:
        report["overall_status"] = "MISSING_COLS"
        return report
    print(f"✅ CSV: {len(df)} rows | classes: {df[label_col].value_counts().to_dict()}")
    report["overall_status"] = "OK"
    report["csv_rows"] = len(df)
    print("✅ VERDICT: Data looks OK")
    print("=" * 78 + "\n")
    return report


def build_convnext(backbone_path, n_classes=4):
    """Load ConvNeXt-Tiny with the SAME architecture as baseline."""
    from torchvision import models
    model = models.convnext_tiny(weights=None)
    nf = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(nf, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(256, n_classes),
    )
    ckpt = torch.load(backbone_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return model


def freeze_backbone(model):
    """Freeze all params except classifier[2]."""
    frozen = 0
    trainable = 0
    for name, p in model.named_parameters():
        if "classifier.2" in name:
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
            frozen += p.numel()
    print(f"\n   🔒 Backbone FROZEN:")
    print(f"      Frozen:    {frozen:,}")
    print(f"      Trainable: {trainable:,} ({trainable/(frozen+trainable)*100:.2f}%)")


def compute_targeted_weights(y_train):
    """
    Explicit class weights for targeted improvement:
      - NL: 1.0 (PROTECT — never boost)
      - G1: 2.5 (BIGGEST boost — worst error)
      - G2: 1.5 (moderate boost)
      - G3: 1.0 (protect — only 70 samples)
    """
    counts = np.bincount(y_train, minlength=4)
    print(f"\n   Training class counts: {dict(zip(CLASS_NAMES, counts.tolist()))}")
    weights = np.array([1.0, 2.5, 1.5, 1.0], dtype=np.float32)
    print(f"   Targeted class weights:")
    for i, w in enumerate(weights):
        print(f"      {CLASS_NAMES[i]}: {w:.2f}")
    return torch.tensor(weights)


# ═════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1]
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    return {
        "acc": accuracy_score(all_labels, all_preds),
        "bal_acc": balanced_accuracy_score(all_labels, all_preds),
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }


# ═════════════════════════════════════════════════════════════════════
# Fine-tune head only
# ═════════════════════════════════════════════════════════════════════
def finetune_head(model, train_loader, val_loader, device,
                  class_weights, epochs=25, lr=5e-4, weight_decay=1e-4,
                  patience=12):
    print(f"\n🔥 Fine-tuning head only...")
    print(f"   Epochs: {epochs} | LR: {lr} | WD: {weight_decay} | Patience: {patience}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)

    best_state = None
    best_score = -1e9  # maximize: bal_acc - 0.5 * max_per_class_error
    best_epoch = -1
    no_improve = 0
    history = {"train_loss": [], "val_acc": [], "val_bal_acc": [],
               "per_class_err": []}

    for epoch in range(1, epochs + 1):
        model.train()
        # Keep backbone in eval mode (frozen)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                m.eval()

        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Evaluate on val
        val_metrics = evaluate(model, val_loader, device)
        cm = confusion_matrix(val_metrics["labels"], val_metrics["preds"], labels=[0, 1, 2, 3])
        per_class_err = []
        for c in range(4):
            total = max(int((val_metrics["labels"] == c).sum()), 1)
            per_class_err.append((total - cm[c, c]) / total)
        max_err = max(per_class_err)

        score = val_metrics["bal_acc"] - 0.5 * max_err

        avg_train = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)
        history["val_acc"].append(val_metrics["acc"])
        history["val_bal_acc"].append(val_metrics["bal_acc"])
        history["per_class_err"].append(per_class_err)

        print(f"   Epoch {epoch:2d}: train={avg_train:.4f} | "
              f"val_acc={val_metrics['acc']:.4f} | "
              f"bal_acc={val_metrics['bal_acc']:.4f} | "
              f"worst_err={max_err*100:.2f}%")
        print(f"             per-class: " +
              " | ".join([f"{CLASS_NAMES[i]}={e*100:.1f}%"
                          for i, e in enumerate(per_class_err)]))

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
            print(f"      ✅ New best (score={best_score:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break

    print(f"\n✅ Best epoch: {best_epoch} (score={best_score:.4f})")
    return best_state, history


# ═════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════
def run_convnext_boosted(baseline_ckpt, output_dir, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 78)
    print("  🚀 CONVNEXT TARGETED FINE-TUNE (G1/G2 only)")
    print("=" * 78)
    print(f"  🖥️  Device: {device.upper()}")

    os.makedirs(output_dir, exist_ok=True)
    sanity_check_data(LABELS_CSV, CROPPED_IMAGES_DIR)

    # Load baseline
    print("\n📥 Loading ConvNeXt baseline...")
    model = build_convnext(baseline_ckpt, len(CLASS_NAMES)).to(device)
    print("   ✅ Baseline loaded")
    freeze_backbone(model)

    # Datasets (same split as baseline)
    print("\n📊 Building datasets...")
    _, df, _ = validate_csv(LABELS_CSV)
    df = df[df["liverLabel"] != "UNKNOWN"].copy()
    counts = df["liverLabel"].value_counts()
    df = df[df["liverLabel"].isin(counts[counts >= 2].index.tolist())]

    train_val_df, test_df = train_test_split(
        df, test_size=0.10, stratify=df["liverLabel"], random_state=42,
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=1/9,
        stratify=train_val_df["liverLabel"], random_state=42,
    )

    def _make_ds(d, augment):
        return LiverDataset(
            d, CROPPED_IMAGES_DIR, augment=augment,
            target_size=224, apply_normalize=True,
            use_advanced_augment=False,   # simple augmentation only
            return_path=False,
        )

    train_ds = _make_ds(train_df, augment=True)
    val_ds = _make_ds(val_df, augment=False)
    test_ds = _make_ds(test_df, augment=False)
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Baseline eval
    print("\n📊 Baseline evaluation...")
    base_metrics = evaluate(model, test_loader, device)
    cm_base = confusion_matrix(base_metrics["labels"], base_metrics["preds"], labels=[0, 1, 2, 3])
    print(f"   Baseline test_acc: {base_metrics['acc']:.4f}")
    for c in range(4):
        total = max(int((base_metrics["labels"] == c).sum()), 1)
        err = (total - cm_base[c, c]) / total
        print(f"      {CLASS_NAMES[c]}: err={err*100:.2f}%")

    # Class weights
    y_train = np.array([train_ds[i][1] for i in range(len(train_ds))])
    class_weights = compute_targeted_weights(y_train)

    # Fine-tune head
    best_state, history = finetune_head(
        model, train_loader, val_loader, device,
        class_weights=class_weights,
        epochs=25, lr=5e-4, weight_decay=1e-4, patience=12,
    )

    # Load best
    model.load_state_dict(best_state)
    model.eval()

    # Final eval
    print("\n📊 Final evaluation on test...")
    test_metrics = evaluate(model, test_loader, device)
    cm_ft = confusion_matrix(test_metrics["labels"], test_metrics["preds"], labels=[0, 1, 2, 3])

    print(f"\n   Baseline:   acc={base_metrics['acc']:.4f} | "
          f"bal_acc={base_metrics['bal_acc']:.4f}")
    print(f"   Fine-tuned: acc={test_metrics['acc']:.4f} | "
          f"bal_acc={test_metrics['bal_acc']:.4f}")

    print("\n📊 Test Classification Report:")
    print(classification_report(test_metrics["labels"], test_metrics["preds"],
                                target_names=CLASS_NAMES, digits=4, zero_division=0))

    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm_ft, index=CLASS_NAMES, columns=CLASS_NAMES))

    print("\n📊 Per-Class Error Rates:")
    for c in range(4):
        total = int((test_metrics["labels"] == c).sum())
        wrong_base = total - cm_base[c, c]
        wrong_ft = total - cm_ft[c, c]
        err_base = wrong_base / max(total, 1)
        err_ft = wrong_ft / max(total, 1)
        delta = err_ft - err_base
        marker = "✅" if delta < -0.005 else ("❌" if delta > 0.005 else "➖")
        print(f"   {CLASS_NAMES[c]}: base={err_base*100:.2f}% → "
              f"ft={err_ft*100:.2f}% (Δ={delta*100:+.2f}%) {marker}")

    # Save model
    save_path = os.path.join(output_dir, "convnext_targeted.pth")
    torch.save({"model_state_dict": best_state}, save_path)
    print(f"\n💾 Saved: {save_path}")

    # Save JSON
    results = {
        "baseline_acc": float(base_metrics["acc"]),
        "baseline_bal_acc": float(base_metrics["bal_acc"]),
        "finetuned_acc": float(test_metrics["acc"]),
        "finetuned_bal_acc": float(test_metrics["bal_acc"]),
        "class_weights": class_weights.tolist(),
        "cm_baseline": cm_base.tolist(),
        "cm_finetuned": cm_ft.tolist(),
        "per_class_error_baseline": [
            float((int((test_metrics["labels"] == c).sum()) - cm_base[c, c]) /
                  max(int((test_metrics["labels"] == c).sum()), 1))
            for c in range(4)
        ],
        "per_class_error_finetuned": [
            float((int((test_metrics["labels"] == c).sum()) - cm_ft[c, c]) /
                  max(int((test_metrics["labels"] == c).sum()), 1))
            for c in range(4)
        ],
        "history": {k: [float(v) for v in vals] if k != "per_class_err"
                    else [[float(x) for x in row] for row in vals]
                    for k, vals in history.items()},
    }
    with open(os.path.join(output_dir, "targeted_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Saved: {os.path.join(output_dir, 'targeted_results.json')}")
    return results