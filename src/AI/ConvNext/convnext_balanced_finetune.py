# src/AI/ConvNext/convnext_balanced_finetune.py
# -*- coding: utf-8 -*-
"""
ConvNeXt Balanced Fine-Tune — v3 (FROZEN Backbone)
====================================================
Strategy:
  - Backbone FROZEN (never modified)
  - Only classifier head is trained
  - Class-Balanced Loss applied to HEAD only
  - No Mixup/CutMix (medical images)
  - Early stopping on val_acc (not bal_acc)
  - Target: NL<5%, G1<5%, G2<5%, G3<15%

This preserves baseline features (NL=2.52%) while improving head
for G1/G2 discrimination.
"""

import os
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from PIL import Image

from src.AI.ResNet.utils import (
    CLASS_NAMES, CROPPED, CSVAddress, validate_csv, LiverDataset,
)

FINAL_REPORTS_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"
LABELS_CSV = os.path.join(FINAL_REPORTS_DIR, "labels.csv")
CROPPED_IMAGES_DIR = os.path.join(FINAL_REPORTS_DIR, "cropped")


# ═════════════════════════════════════════════════════════════════════
# 1. Sanity check
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


# ═════════════════════════════════════════════════════════════════════
# 2. Class-Balanced Weights (soft)
# ═════════════════════════════════════════════════════════════════════
def compute_class_balanced_weights(y_train, beta=0.999, max_weight=3.0):
    """
    Class-Balanced Loss weights with max_weight=3.0 (gentler than before).
    Keeps NL weight high enough to preserve NL performance.
    """
    counts = np.bincount(y_train, minlength=4)
    print(f"\n   Training class counts: {dict(zip(CLASS_NAMES, counts.tolist()))}")

    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.mean()
    weights = np.minimum(weights, max_weight)

    print(f"   Class-balanced weights (β={beta}, max={max_weight}):")
    for i, w in enumerate(weights):
        print(f"      {CLASS_NAMES[i]}: {w:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════
# 3. Model builder (with classifier replacement)
# ═════════════════════════════════════════════════════════════════════
def build_convnext(n_classes=4):
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
    return model


def freeze_backbone(model):
    """
    FREEZE all parameters except classifier head.
    This preserves baseline features (NL=2.52%).
    """
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"\n   🔒 Backbone FROZEN:")
    print(f"      Frozen params:    {frozen_count:,}")
    print(f"      Trainable params: {trainable_count:,} "
          f"({trainable_count / (frozen_count + trainable_count) * 100:.2f}%)")


# ═════════════════════════════════════════════════════════════════════
# 4. Evaluation
# ═════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1]
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
        all_probs.append(probs.cpu().numpy())

        if criterion is not None:
            loss = criterion(logits, y.to(device))
            total_loss += loss.item()
            n_batches += 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = {
        "acc": accuracy_score(all_labels, all_preds),
        "balanced_acc": balanced_accuracy_score(all_labels, all_preds),
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }
    if n_batches > 0:
        metrics["loss"] = total_loss / n_batches
    return metrics


# ═════════════════════════════════════════════════════════════════════
# 5. Fine-tuning loop (head only)
# ═════════════════════════════════════════════════════════════════════
def finetune_head_only(model, train_loader, val_loader, device,
                       class_weights, epochs=30, lr=5e-4, weight_decay=1e-4,
                       patience=23):
    """
    Fine-tune ONLY the classifier head.
    Simple augmentation (no Mixup/CutMix).
    Early stopping on val_acc with patience.
    """
    print(f"\n🔥 Fine-tuning head only...")
    print(f"   Epochs: {epochs} | LR: {lr} | WD: {weight_decay} | Patience: {patience}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=0.05,
    )

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0
    history = {
        "train_loss": [], "val_loss": [], "val_acc": [],
        "val_bal_acc": [], "val_per_class_err": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        # Only train classifier head — but keep backbone in eval mode
        # (BatchNorm/LayerNorm in backbone should be frozen anyway)
        epoch_loss = 0.0
        n_batches = 0
        n_correct = 0
        n_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            x = batch[0].to(device)
            y = batch[1].to(device)

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
            preds = logits.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{n_correct / n_total:.4f}",
            })

        scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, criterion)
        avg_train_loss = epoch_loss / max(n_batches, 1)

        cm = confusion_matrix(
            val_metrics["labels"], val_metrics["preds"], labels=[0, 1, 2, 3],
        )
        per_class_err = []
        for c in range(4):
            total = max(int((val_metrics["labels"] == c).sum()), 1)
            per_class_err.append((total - cm[c, c]) / total)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_bal_acc"].append(val_metrics["balanced_acc"])
        history["val_per_class_err"].append(per_class_err)

        print(f"   Epoch {epoch:2d}: train_loss={avg_train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_acc={val_metrics['acc']:.4f} | "
              f"bal_acc={val_metrics['balanced_acc']:.4f}")
        print(f"             per-class err: " +
              " | ".join([f"{CLASS_NAMES[i]}={e*100:.1f}%"
                          for i, e in enumerate(per_class_err)]))

        # Early stopping on val_acc (not bal_acc)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
            print(f"      ✅ New best (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"\n✅ Best epoch: {best_epoch} (val_acc={best_val_acc:.4f})")
    return best_state, history


# ═════════════════════════════════════════════════════════════════════
# 6. Main pipeline
# ═════════════════════════════════════════════════════════════════════
def run_convnext_boosted(convnext_checkpoint, output_dir, device=None):
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 78)
    print("  🚀 CONVNEXT BALANCED FINE-TUNE (v3 — FROZEN BACKBONE)")
    print("=" * 78)
    print(f"  🖥️  Device: {device.upper()}")
    if device == "cuda":
        print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  ⚠️  WARNING: Running on CPU — this will be VERY SLOW!")

    os.makedirs(output_dir, exist_ok=True)
    sanity = sanity_check_data(LABELS_CSV, CROPPED_IMAGES_DIR)
    with open(os.path.join(output_dir, "data_sanity_check.json"), "w") as f:
        json.dump(sanity, f, indent=2, default=str)

    # Load checkpoint
    print("\n📥 Loading ConvNeXt checkpoint...")
    ckpt = torch.load(convnext_checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = build_convnext(len(CLASS_NAMES))
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    print("   ✅ ConvNeXt loaded")

    # FREEZE BACKBONE
    freeze_backbone(model)

    # Datasets
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
            use_advanced_augment=augment,
            return_path=False,
        )

    train_ds = _make_ds(train_df, augment=True)
    val_ds = _make_ds(val_df, augment=False)
    test_ds = _make_ds(test_df, augment=False)
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Baseline evaluation (with frozen backbone, head as-is)
    print("\n📊 Baseline evaluation (before head fine-tune)...")
    base_metrics = evaluate(model, test_loader, device)
    cm_base = confusion_matrix(base_metrics["labels"], base_metrics["preds"],
                               labels=[0, 1, 2, 3])
    print(f"   Baseline test_acc: {base_metrics['acc']:.4f}")
    print(f"   Baseline bal_acc:  {base_metrics['balanced_acc']:.4f}")
    for c in range(4):
        total = max(int((base_metrics["labels"] == c).sum()), 1)
        err = (total - cm_base[c, c]) / total
        print(f"      {CLASS_NAMES[c]}: err={err*100:.2f}%")

    # Compute class weights from train labels
    y_train = np.array([train_ds[i][1] for i in range(len(train_ds))])
    class_weights = compute_class_balanced_weights(y_train, beta=0.999, max_weight=3.0)

    # Fine-tune head only
    best_state, history = finetune_head_only(
        model, train_loader, val_loader, device,
        class_weights=class_weights,
        epochs=60, lr=5e-4, weight_decay=1e-4, patience=25,
    )

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation on test
    print("\n📊 Final evaluation on test...")
    test_metrics = evaluate(model, test_loader, device)
    test_preds = test_metrics["preds"]
    y_test = test_metrics["labels"]

    print(f"\n   Baseline:   acc={base_metrics['acc']:.4f} | bal_acc={base_metrics['balanced_acc']:.4f}")
    print(f"   Fine-tuned: acc={test_metrics['acc']:.4f} | bal_acc={test_metrics['balanced_acc']:.4f}")

    print("\n📊 Test Classification Report:")
    print(classification_report(y_test, test_preds,
                                target_names=CLASS_NAMES, digits=4,
                                zero_division=0))

    cm_ft = confusion_matrix(y_test, test_preds, labels=list(range(4)))
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm_ft, index=CLASS_NAMES, columns=CLASS_NAMES))

    print("\n📊 Per-Class Error Rates:")
    for c in range(4):
        total_c = int((y_test == c).sum())
        wrong_c = total_c - cm_ft[c, c]
        err = wrong_c / max(total_c, 1)
        status = "✅" if err < 0.05 else ("⚠️" if err < 0.10 else "🔴")
        print(f"   {CLASS_NAMES[c]}: error={err*100:.2f}% {status} "
              f"(wrong={wrong_c}/{total_c})")

    # Save model
    print("\n💾 Saving fine-tuned model...")
    save_path = os.path.join(output_dir, "convnext_balanced.pth")
    torch.save({"model_state_dict": best_state}, save_path)
    print(f"   ✅ Saved: {save_path}")

    # Generate PDF report
    print("\n📄 Generating PDF report...")
    try:
        from .balanced_report import BalancedReporter
        reporter = BalancedReporter({
            "output_dir": output_dir,
            "history": history,
            "confusion_matrix_baseline": cm_base.tolist(),
            "confusion_matrix_finetuned": cm_ft.tolist(),
            "y_test": y_test,
            "y_pred_baseline": base_metrics["preds"],
            "y_pred_finetuned": test_preds,
            "y_prob_baseline": base_metrics["probs"],
            "y_prob_finetuned": test_metrics["probs"],
            "baseline_acc": float(base_metrics["acc"]),
            "finetuned_acc": float(test_metrics["acc"]),
            "baseline_bal_acc": float(base_metrics["balanced_acc"]),
            "finetuned_bal_acc": float(test_metrics["balanced_acc"]),
            "class_weights": class_weights.tolist(),
        })
        pdf_path = reporter.generate_all()
        print(f"   ✅ PDF: {pdf_path}")
    except Exception as e:
        print(f"   ⚠️ Report generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    results = {
        "baseline_acc": float(base_metrics["acc"]),
        "baseline_bal_acc": float(base_metrics["balanced_acc"]),
        "finetuned_acc": float(test_metrics["acc"]),
        "finetuned_bal_acc": float(test_metrics["balanced_acc"]),
        "class_weights": class_weights.tolist(),
        "confusion_matrix_baseline": cm_base.tolist(),
        "confusion_matrix_finetuned": cm_ft.tolist(),
        "per_class_error_baseline": [
            float(int((y_test == c).sum()) - cm_base[c, c]) /
            max(int((y_test == c).sum()), 1)
            for c in range(4)
        ],
        "per_class_error_finetuned": [
            float(int((y_test == c).sum()) - cm_ft[c, c]) /
            max(int((y_test == c).sum()), 1)
            for c in range(4)
        ],
        "history": history,
    }
    with open(os.path.join(output_dir, "balanced_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Saved: {os.path.join(output_dir, 'balanced_results.json')}")
    return results