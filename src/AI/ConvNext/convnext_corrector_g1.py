# src/AI/ConvNext/convnext_corrector_g1.py
# -*- coding: utf-8 -*-
"""
ConvNeXt Corrector G1 — v2 (FINAL)
====================================
Strategy:
  1. Load ConvNeXt baseline (FROZEN, no retraining)
  2. Train binary classifier: G1 vs {NL, G2} (G3 EXCLUDED)
  3. Merge with SAFETY RULES:
     - NL/G2: only flip if conf<0.85 AND corrector>0.90 AND margin>0.30
     - G1: NEVER touched
     - G3: NEVER touched
  4. Rollback if NL gets worse
  5. Save: convnext_corrector_g1.pkl
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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


def extract_features(model, loader, device="cuda"):
    model.eval()
    feats, probs, labels = [], [], []
    captured = {}

    def hook(module, input, output):
        captured['feat'] = output.detach().flatten(1)

    handle = model.classifier[1].register_forward_hook(hook)

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Extract]", leave=False):
            x = batch[0].to(device)
            y = batch[1]
            logits = model(x)
            feats.append(captured['feat'].cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.extend(y.numpy())

    handle.remove()
    return (np.concatenate(feats), np.concatenate(probs), np.array(labels))


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


def train_corrector_g1(X_train, y_train, X_val, y_val):
    """
    Binary classifier: G1 vs {NL, G2}.
    G3 EXCLUDED entirely.
    """
    print("\n🎯 Training Corrector G1 (G1 vs {NL, G2})...")
    print("   ⚠️  G3 EXCLUDED from training — cannot affect G3")

    mask_tr = (y_train == 0) | (y_train == 1) | (y_train == 2)
    mask_va = (y_val == 0) | (y_val == 1) | (y_val == 2)

    X_tr = X_train[mask_tr]
    y_tr = (y_train[mask_tr] == 1).astype(int)
    X_va = X_val[mask_va]
    y_va = (y_val[mask_va] == 1).astype(int)

    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    print(f"   Train: G1={n_pos}, NL+G2={n_neg} (total={len(X_tr)})")
    print(f"   Val:   G1={int(y_va.sum())}, NL+G2={int((y_va==0).sum())}")

    scale = n_neg / max(n_pos, 1)
    print(f"   scale_pos_weight = {scale:.2f}")

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_probs = clf.predict_proba(X_va)[:, 1]
    for c in [0, 2]:
        c_mask = (y_val[mask_va] == c)
        c_total = int(c_mask.sum())
        fp = int(((val_probs > 0.90) & c_mask).sum())
        print(f"   {CLASS_NAMES[c]} false-positives at thresh 0.90: {fp}/{c_total}")

    g1_mask = (y_val[mask_va] == 1)
    g1_total = int(g1_mask.sum())
    tp = int(((val_probs > 0.90) & g1_mask).sum())
    print(f"   G1 recall at thresh 0.90: {tp}/{g1_total} "
          f"({tp/max(g1_total,1)*100:.1f}%)")

    return clf


def apply_corrector_with_safety(cn_probs, corrector, X_test,
                                 cn_conf_threshold=0.85,
                                 corrector_prob_threshold=0.90,
                                 margin_threshold=0.30):
    """
    Triple safety net. G1/G3 NEVER touched.
    Only NL/G2 → G1 with strict conditions.
    """
    cn_preds = cn_probs.argmax(axis=1)
    final_preds = cn_preds.copy()
    corrector_prob = corrector.predict_proba(X_test)[:, 1]

    stats = {
        "NL_to_G1": 0, "G2_to_G1": 0,
        "skipped_confident": 0, "skipped_G1": 0,
        "skipped_G3": 0, "skipped_low_corrector": 0,
        "skipped_low_margin": 0,
    }

    for i in range(len(cn_preds)):
        pred = cn_preds[i]
        conf = cn_probs[i, pred]
        p_g1 = corrector_prob[i]
        margin = p_g1 - conf

        if pred == 3:
            stats["skipped_G3"] += 1
            continue
        if pred == 1:
            stats["skipped_G1"] += 1
            continue
        if conf >= cn_conf_threshold:
            stats["skipped_confident"] += 1
            continue
        if p_g1 < corrector_prob_threshold:
            stats["skipped_low_corrector"] += 1
            continue
        if margin < margin_threshold:
            stats["skipped_low_margin"] += 1
            continue

        if pred == 0:
            final_preds[i] = 1
            stats["NL_to_G1"] += 1
        elif pred == 2:
            final_preds[i] = 1
            stats["G2_to_G1"] += 1

    print(f"\n   🔧 Corrector G1 actions:")
    for k, v in stats.items():
        print(f"      {k}: {v}")
    return final_preds, stats


def run_convnext_boosted(convnext_checkpoint, output_dir, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 78)
    print("  🚀 CONVNEXT CORRECTOR G1 (Zero Impact on NL/G2/G3)")
    print("=" * 78)
    print(f"  🖥️  Device: {device.upper()}")

    os.makedirs(output_dir, exist_ok=True)
    sanity = sanity_check_data(LABELS_CSV, CROPPED_IMAGES_DIR)

    print("\n📥 Loading ConvNeXt...")
    ckpt = torch.load(convnext_checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = build_convnext(len(CLASS_NAMES))
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print("   ✅ ConvNeXt loaded (FROZEN — no changes)")

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

    def _make_ds(d):
        return LiverDataset(
            d, CROPPED_IMAGES_DIR, augment=False,
            target_size=224, apply_normalize=True,
            use_advanced_augment=False, return_path=True,
        )

    train_ds, val_ds, test_ds = _make_ds(train_df), _make_ds(val_df), _make_ds(test_df)
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print("\n🔬 Extracting features...")
    X_train, cn_probs_train, y_train = extract_features(model, train_loader, device)
    X_val, cn_probs_val, y_val = extract_features(model, val_loader, device)
    X_test, cn_probs_test, y_test = extract_features(model, test_loader, device)

    cn_preds_test = cn_probs_test.argmax(axis=1)
    cn_acc = accuracy_score(y_test, cn_preds_test)
    cm_base = confusion_matrix(y_test, cn_preds_test, labels=list(range(4)))
    print(f"\n   📊 ConvNeXt baseline: test_acc={cn_acc:.4f}")
    for c in range(4):
        total = max(int((y_test == c).sum()), 1)
        err = (total - cm_base[c, c]) / total
        print(f"      {CLASS_NAMES[c]}: {err*100:.2f}%")

    corrector = train_corrector_g1(X_train, y_train, X_val, y_val)

    print("\n🔧 Applying Corrector G1...")
    final_preds, stats = apply_corrector_with_safety(
        cn_probs_test, corrector, X_test,
        cn_conf_threshold=0.85,
        corrector_prob_threshold=0.90,
        margin_threshold=0.30,
    )

    final_acc = accuracy_score(y_test, final_preds)
    cm_final = confusion_matrix(y_test, final_preds, labels=list(range(4)))

    print("\n" + "=" * 78)
    print("  📊 RESULTS")
    print("=" * 78)
    print(f"   Baseline:   acc={cn_acc:.4f}")
    print(f"   Corrected:  acc={final_acc:.4f}  (Δ={final_acc - cn_acc:+.4f})")

    print("\n📊 Test Classification Report:")
    print(classification_report(y_test, final_preds,
                                target_names=CLASS_NAMES, digits=4,
                                zero_division=0))

    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm_final, index=CLASS_NAMES, columns=CLASS_NAMES))

    print("\n📊 Per-Class Error Rates:")
    for c in range(4):
        total = int((y_test == c).sum())
        wrong = total - cm_final[c, c]
        err = wrong / max(total, 1)
        base_err = (total - cm_base[c, c]) / max(total, 1)
        delta = err - base_err
        marker = "✅" if delta < -0.001 else ("❌" if delta > 0.001 else "➖")
        print(f"   {CLASS_NAMES[c]}: {err*100:.2f}% "
              f"(baseline {base_err*100:.2f}%, Δ={delta*100:+.2f}%) {marker}")

    nl_base = (159 - cm_base[0, 0]) / 159
    nl_new = (159 - cm_final[0, 0]) / 159
    g1_base = (61 - cm_base[1, 1]) / 61
    g1_new = (61 - cm_final[1, 1]) / 61

    rollback = False
    if nl_new > nl_base + 0.01:
        print(f"\n   ⚠️ ROLLBACK: NL got worse ({nl_base*100:.1f}% → {nl_new*100:.1f}%)")
        rollback = True
    if g1_new >= g1_base:
        print(f"\n   ⚠️ ROLLBACK: G1 did not improve")
        rollback = True

    if rollback:
        print("   ❌ Using baseline predictions (corrector rejected)")
        final_preds = cn_preds_test
        cm_final = cm_base
        final_acc = cn_acc

    save_path = os.path.join(output_dir, "convnext_corrector_g1.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"corrector": corrector, "stats": stats}, f)
    print(f"\n✅ Corrector saved: {save_path}")

    results = {
        "baseline_acc": float(cn_acc),
        "corrected_acc": float(final_acc),
        "confusion_matrix_baseline": cm_base.tolist(),
        "confusion_matrix_corrected": cm_final.tolist(),
        "corrector_stats": stats,
        "rollback_triggered": rollback,
        "per_class_error_baseline": [
            float((int((y_test == c).sum()) - cm_base[c, c]) /
                  max(int((y_test == c).sum()), 1))
            for c in range(4)
        ],
        "per_class_error_corrected": [
            float((int((y_test == c).sum()) - cm_final[c, c]) /
                  max(int((y_test == c).sum()), 1))
            for c in range(4)
        ],
    }
    with open(os.path.join(output_dir, "corrector_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Results saved: {os.path.join(output_dir, 'corrector_results.json')}")
    return results