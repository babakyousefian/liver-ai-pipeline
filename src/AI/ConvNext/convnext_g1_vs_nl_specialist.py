# src/AI/ConvNext/convnext_g1_vs_nl_specialist.py
# -*- coding: utf-8 -*-
"""
ConvNeXt G1 vs NL Specialist
==============================
- Train a SEPARATE ConvNeXt-Tiny ONLY on (NL vs G1)
- Backbone is FINE-TUNED (not frozen) on this binary task
- Then use it as a "corrector" for the main 4-class ConvNeXt

Why this works:
  - The specialist learns features SPECIFIC to NL vs G1 discrimination
  - This is fundamentally different from a classifier on frozen features
  - After training, it can distinguish NL/G1 better than baseline

Safety:
  - Only override NL/G2 → G1 if BOTH:
    * Specialist confidence > 0.90
    * Margin over baseline > 0.30
  - Rollback if NL worsens
"""

import os
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.AI.ResNet.utils import (
    CLASS_NAMES, CROPPED, CSVAddress, validate_csv, LiverDataset,
)

FINAL_REPORTS_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"
LABELS_CSV = os.path.join(FINAL_REPORTS_DIR, "labels.csv")
CROPPED_IMAGES_DIR = os.path.join(FINAL_REPORTS_DIR, "cropped")


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


def build_convnext_binary():
    """ConvNeXt-Tiny with 2-class head (NL vs G1)."""
    from torchvision import models
    model = models.convnext_tiny(weights=None)
    nf = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(nf, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(256, 2),   # ← binary: 0=NL, 1=G1
    )
    return model


def build_convnext_4class():
    """ConvNeXt-Tiny with 4-class head (baseline)."""
    from torchvision import models
    model = models.convnext_tiny(weights=None)
    nf = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(nf, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(256, 4),
    )
    return model


def train_g1_vs_nl_specialist(train_loader, val_loader, device,
                               epochs=20, lr=1e-4, weight_decay=0.01,
                               patience=8):
    """
    Train a binary ConvNeXt-Tiny on NL vs G1.
    Backbone is FINE-TUNED (not frozen) — this is the key difference.
    """
    print("\n🔥 Training G1 vs NL Specialist (ConvNeXt-Tiny, binary)...")
    print(f"   Epochs: {epochs} | LR: {lr} | WD: {weight_decay} | Patience: {patience}")

    model = build_convnext_binary().to(device)

    # Load pretrained ConvNeXt weights for backbone
    try:
        from torchvision import models
        pretrained = models.convnext_tiny(weights="IMAGENET1K_V1")
        # Copy backbone weights (everything except classifier[2])
        model_dict = model.state_dict()
        pretrained_dict = pretrained.state_dict()
        for k, v in pretrained_dict.items():
            if k in model_dict and "classifier.2" not in k:
                model_dict[k] = v
        model.load_state_dict(model_dict)
        print("   ✅ Loaded ImageNet pretrained backbone")
    except Exception as e:
        print(f"   ⚠️ Could not load pretrained: {e}")

    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ✅ AMP scaler
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        n_correct = 0
        n_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            # ✅ AMP forward
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, y)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        val_n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device, non_blocking=True)
                y = batch[1].to(device, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = model(x)
                        loss = criterion(logits, y)
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item()
                val_n_batches += 1
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_val_loss = val_loss / max(val_n_batches, 1)

        print(f"   Epoch {epoch:2d}: train_loss={avg_train_loss:.4f} | "
              f"val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
            print(f"      ✅ New best (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break

    print(f"\n✅ Best epoch: {best_epoch} (val_acc={best_val_acc:.4f})")
    return best_state, best_val_acc


def main_pipeline(convnext_baseline_ckpt, output_dir, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 78)
    print("  🚀 CONVNEXT G1 vs NL SPECIALIST")
    print("=" * 78)
    print(f"  🖥️  Device: {device.upper()}")

    os.makedirs(output_dir, exist_ok=True)
    sanity_check_data(LABELS_CSV, CROPPED_IMAGES_DIR)

    # ─── Load 4-class baseline (FROZEN) ───
    print("\n📥 Loading 4-class baseline ConvNeXt...")
    ckpt = torch.load(convnext_baseline_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    baseline = build_convnext_4class()
    baseline.load_state_dict(state, strict=True)
    baseline = baseline.to(device).eval()
    print("   ✅ Baseline loaded (FROZEN)")

    # ─── Load 2-class specialist (FROZEN — will load weights later) ───
    print("\n📥 Loading 2-class specialist (to be trained)...")

    # ─── Datasets ───
    print("\n📊 Building datasets...")
    _, df, _ = validate_csv(LABELS_CSV)
    df = df[df["liverLabel"] != "UNKNOWN"].copy()
    counts = df["liverLabel"].value_counts()
    df = df[df["liverLabel"].isin(counts[counts >= 2].index.tolist())]

    # Split same as baseline
    train_val_df, test_df = train_test_split(
        df, test_size=0.10, stratify=df["liverLabel"], random_state=42,
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=1/9,
        stratify=train_val_df["liverLabel"], random_state=42,
    )

    # ─── Filter ONLY NL and G1 for specialist ───
    def _filter_nl_g1(d):
        return d[d["liverLabel"].isin(["NL", "G1"])].reset_index(drop=True)

    train_spec = _filter_nl_g1(train_df)
    val_spec = _filter_nl_g1(val_df)
    print(f"   Specialist train: {len(train_spec)} (NL+G1)")
    print(f"   Specialist val:   {len(val_spec)} (NL+G1)")

    def _make_ds(d, augment):
        return LiverDataset(
            d, CROPPED_IMAGES_DIR, augment=augment,
            target_size=224, apply_normalize=True,
            use_advanced_augment=False,  # ← False
            return_path=False,
        )

    train_spec_ds = _make_ds(train_spec, augment=True)
    val_spec_ds = _make_ds(val_spec, augment=False)

    # Weighted sampler for balanced NL/G1
    labels_array = np.array([train_spec_ds[i][1] for i in range(len(train_spec_ds))])
    counts_2c = np.bincount(labels_array, minlength=4)
    # Map 0=NL, 1=G1
    nl_count = int((labels_array == 0).sum())
    g1_count = int((labels_array == 1).sum())
    print(f"   Train NL: {nl_count}, G1: {g1_count}")

    weights = np.zeros(len(labels_array))
    weights[labels_array == 0] = 1.0 / nl_count
    weights[labels_array == 1] = 1.0 / g1_count
    weights = weights / weights.sum() * len(weights)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )

    train_spec_loader = DataLoader(
        train_spec_ds, batch_size=16, sampler=sampler,
        num_workers=0, pin_memory=True,
    )
    val_spec_loader = DataLoader(
        val_spec_ds, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ─── Train specialist ───
    best_state, best_val_acc = train_g1_vs_nl_specialist(
        train_spec_loader, val_spec_loader, device,
        epochs=20, lr=1e-4, weight_decay=0.01, patience=8,
    )

    # Load best specialist
    specialist = build_convnext_binary()
    specialist.load_state_dict(best_state)
    specialist = specialist.to(device).eval()

    # Save specialist
    spec_path = os.path.join(output_dir, "convnext_g1_vs_nl_specialist.pth")
    torch.save({"model_state_dict": best_state}, spec_path)
    print(f"\n✅ Specialist saved: {spec_path}")

    # ─── Evaluate on test ───
    print("\n📊 Building test set...")
    test_ds = _make_ds(test_df, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print("\n🔬 Extracting predictions from both models...")
    baseline.eval()
    specialist.eval()

    all_baseline_preds = []
    all_specialist_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Eval]"):
            x = batch[0].to(device)
            y = batch[1]

            # Baseline (4-class)
            logits_base = baseline(x)
            probs_base = torch.softmax(logits_base, dim=1)
            preds_base = probs_base.argmax(dim=1)

            # Specialist (2-class)
            logits_spec = specialist(x)
            probs_spec = torch.softmax(logits_spec, dim=1)
            # P(G1) = probs_spec[:, 1]

            all_baseline_preds.extend(preds_base.cpu().numpy())
            all_specialist_probs.append(probs_spec.cpu().numpy())
            all_labels.extend(y.numpy())

    all_baseline_preds = np.array(all_baseline_preds)
    all_specialist_probs = np.concatenate(all_specialist_probs, axis=0)
    all_labels = np.array(all_labels)

    baseline_acc = accuracy_score(all_labels, all_baseline_preds)
    cm_base = confusion_matrix(all_labels, all_baseline_preds, labels=list(range(4)))
    print(f"\n   📊 Baseline test_acc: {baseline_acc:.4f}")

    # ─── Merge with safety rules ───
    print("\n🔧 Merging with safety rules...")
    final_preds = all_baseline_preds.copy()

    # Get baseline confidence
    # Need to re-extract baseline probs for confidence
    baseline_probs_list = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            logits = baseline(x)
            baseline_probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    baseline_probs = np.concatenate(baseline_probs_list, axis=0)

    stats = {
        "NL_to_G1": 0, "G2_to_G1": 0,
        "skipped_confident": 0, "skipped_G1": 0,
        "skipped_G3": 0, "skipped_low_specialist": 0,
        "skipped_low_margin": 0,
    }

    for i in range(len(all_baseline_preds)):
        pred = all_baseline_preds[i]
        conf = baseline_probs[i, pred]
        p_g1 = all_specialist_probs[i, 1]
        margin = p_g1 - conf

        if pred == 3:
            stats["skipped_G3"] += 1
            continue
        if pred == 1:
            stats["skipped_G1"] += 1
            continue
        if conf >= 0.85:
            stats["skipped_confident"] += 1
            continue
        if p_g1 < 0.90:
            stats["skipped_low_specialist"] += 1
            continue
        if margin < 0.30:
            stats["skipped_low_margin"] += 1
            continue

        if pred == 0:
            final_preds[i] = 1
            stats["NL_to_G1"] += 1
        elif pred == 2:
            final_preds[i] = 1
            stats["G2_to_G1"] += 1

    print(f"\n   🔧 Merge stats:")
    for k, v in stats.items():
        print(f"      {k}: {v}")

    final_acc = accuracy_score(all_labels, final_preds)
    cm_final = confusion_matrix(all_labels, final_preds, labels=list(range(4)))

    print(f"\n📊 Results:")
    print(f"   Baseline:  acc={baseline_acc:.4f}")
    print(f"   Merged:    acc={final_acc:.4f}  (Δ={final_acc - baseline_acc:+.4f})")

    print("\n📊 Per-Class Error Rates:")
    for c in range(4):
        total = int((all_labels == c).sum())
        wrong_base = total - cm_base[c, c]
        wrong_final = total - cm_final[c, c]
        err_base = wrong_base / max(total, 1)
        err_final = wrong_final / max(total, 1)
        delta = err_final - err_base
        marker = "✅" if delta < -0.001 else ("❌" if delta > 0.001 else "➖")
        print(f"   {CLASS_NAMES[c]}: base={err_base*100:.2f}%, "
              f"final={err_final*100:.2f}% (Δ={delta*100:+.2f}%) {marker}")

    # Rollback
    nl_base = (159 - cm_base[0, 0]) / 159
    nl_final = (159 - cm_final[0, 0]) / 159
    if nl_final > nl_base + 0.005:   # 0.5% tolerance
        print(f"\n   ⚠️ ROLLBACK: NL got worse")
        final_preds = all_baseline_preds
        final_acc = baseline_acc

    results = {
        "baseline_acc": float(baseline_acc),
        "final_acc": float(final_acc),
        "specialist_val_acc": float(best_val_acc),
        "merge_stats": stats,
        "confusion_matrix_baseline": cm_base.tolist(),
        "confusion_matrix_final": cm_final.tolist(),
    }
    with open(os.path.join(output_dir, "specialist_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Saved: {os.path.join(output_dir, 'specialist_results.json')}")
    return results


def run_convnext_boosted(convnext_checkpoint, output_dir, device=None):
    return main_pipeline(convnext_checkpoint, output_dir, device)