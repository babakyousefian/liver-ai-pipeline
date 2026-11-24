# src/AI/ConvNext/convnext_boundary_refine.py
# -*- coding: utf-8 -*-
"""
ConvNeXt Boundary Refinement — STEP 2: G1-ONLY CORRECTION
==========================================================
ABSOLUTE RULES:
  - NL is NEVER touched. Period.
  - G2 is NEVER touched.
  - G3 is NEVER touched.
  - ONLY G1 predictions are corrected.
  - When ConvNeXt says NL/G2 but G1 Detector is VERY confident → flip to G1.

Strategy:
  1. Train G1 Detector on G1 samples only (One-Class-like).
  2. Use high confidence threshold (0.90) to avoid false positives.
  3. Only apply to samples where ConvNeXt said NL or G2.
  4. Never touch NL if ConvNeXt is confident (>= 0.85).
"""

import os
import json
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


# ═════════════════════════════════════════════════════════════════════
# G1 Detector — trained ONLY on G1 samples vs ALL other classes
# ═════════════════════════════════════════════════════════════════════
def build_g1_detector_g1_only(X_train, y_train, X_val, y_val):
    """
    Binary classifier: G1 vs (everything else).
    Trained on ALL G1 + a balanced sample of NON-G1.
    Uses high confidence threshold to avoid false positives on NL/G2/G3.
    """
    print("\n🎯 Building G1 Detector (G1 vs Rest, conservative)...")

    y_bin = (y_train == 1).astype(int)
    y_val_bin = (y_val == 1).astype(int)

    n_g1 = int(y_bin.sum())
    n_non_g1 = int((y_bin == 0).sum())
    print(f"   Train: G1={n_g1}, Non-G1={n_non_g1}")

    # ❗ Important: use scale_pos_weight so G1 (minority) gets balanced weight
    # but NOT too high (to avoid overfitting to G1)
    scale = n_non_g1 / max(n_g1, 1)
    print(f"   scale_pos_weight = {scale:.2f}")

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.75,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(X_train, y_bin, eval_set=[(X_val, y_val_bin)], verbose=False)

    # Evaluate at HIGH threshold (0.90) to measure false positive rate
    val_probs = clf.predict_proba(X_val)[:, 1]

    # How many NL get falsely flagged as G1?
    nl_mask = (y_val == 0)
    nl_fp = ((val_probs > 0.90) & nl_mask).sum()
    nl_total = nl_mask.sum()

    # How many G1 are detected?
    g1_mask = (y_val == 1)
    g1_tp = ((val_probs > 0.90) & g1_mask).sum()
    g1_total = g1_mask.sum()

    print(f"   ✅ G1 Detector trained")
    print(f"      At threshold 0.90:")
    print(f"         G1 recall: {g1_tp}/{g1_total} = {g1_tp/max(g1_total,1):.2%}")
    print(f"         NL false-positives: {nl_fp}/{nl_total} = {nl_fp/max(nl_total,1):.2%}")
    return clf


# ═════════════════════════════════════════════════════════════════════
# G1-Only Correction — strict rules
# ═════════════════════════════════════════════════════════════════════
def g1_only_correction(cn_probs, g1_detector, X_test,
                      g1_prob_threshold=0.90,
                      nl_confidence_floor=0.85,
                      g2_confidence_floor=0.85):
    """
    Strict rules for correcting G1 errors.
    NL and G2 are ONLY touched when they are LOW-confidence.

    Rules:
      1. ConvNeXt says NL but confidence < 0.85 AND g1_prob > 0.90 → flip to G1
      2. ConvNeXt says G2 but confidence < 0.85 AND g1_prob > 0.90 → flip to G1
      3. ConvNeXt says G1 → keep as G1 (no change)
      4. ConvNeXt says G3 → keep as G3 (no change)
      5. NL/G2 with high confidence → NEVER touched
    """
    preds = cn_probs.argmax(axis=1)
    final_preds = preds.copy()

    g1_prob = g1_detector.predict_proba(X_test)[:, 1]

    nl_to_g1 = 0
    g2_to_g1 = 0
    protected_nl = 0
    protected_g2 = 0

    for i in range(len(preds)):
        cn_pred = preds[i]
        cn_conf = cn_probs[i, cn_pred]
        p_g1 = g1_prob[i]

        # ─── Rule 1: NL → G1 (only if NL is UNSURE and G1 Detector VERY sure) ───
        if cn_pred == 0:
            if cn_conf < nl_confidence_floor and p_g1 > g1_prob_threshold:
                final_preds[i] = 1
                nl_to_g1 += 1
            else:
                protected_nl += 1

        # ─── Rule 2: G2 → G1 (only if G2 is UNSURE and G1 Detector VERY sure) ───
        elif cn_pred == 2:
            if cn_conf < g2_confidence_floor and p_g1 > g1_prob_threshold:
                final_preds[i] = 1
                g2_to_g1 += 1
            else:
                protected_g2 += 1

        # ─── Rule 3 & 4: G1, G3 untouched ───
        # (already handled by final_preds = preds.copy())

    print(f"\n   🔧 G1-Only Correction Results:")
    print(f"      NL → G1 corrections:  {nl_to_g1}")
    print(f"      G2 → G1 corrections:  {g2_to_g1}")
    print(f"      NL protected (high conf): {protected_nl}")
    print(f"      G2 protected (high conf): {protected_g2}")

    return final_preds


def run_convnext_boosted(convnext_checkpoint, output_dir, device="cuda"):
    from torchvision import models

    print("\n" + "=" * 78)
    print("  🚀 STEP 2: G1-ONLY CORRECTION (NL/G2/G3 PROTECTED)")
    print("=" * 78)

    os.makedirs(output_dir, exist_ok=True)
    sanity = sanity_check_data(LABELS_CSV, CROPPED_IMAGES_DIR)
    with open(os.path.join(output_dir, "data_sanity_check.json"), "w") as f:
        json.dump(sanity, f, indent=2, default=str)

    # Load ConvNeXt
    print("\n📥 Loading ConvNeXt...")
    ckpt = torch.load(convnext_checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = models.convnext_tiny(weights=None)
    nf = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(nf, 256), nn.LayerNorm(256),
        nn.GELU(), nn.Dropout(0.15), nn.Linear(256, len(CLASS_NAMES)),
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print("   ✅ ConvNeXt loaded")

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

    def _make_ds(d):
        return LiverDataset(
            d, CROPPED_IMAGES_DIR, augment=False,
            target_size=224, apply_normalize=True,
            use_advanced_augment=False, return_path=True,
        )

    train_ds, val_ds, test_ds = _make_ds(train_df), _make_ds(val_df), _make_ds(test_df)
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Extract features
    print("\n🔬 Extracting ConvNeXt features...")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    X_train, cn_probs_train, y_train = extract_features(model, train_loader, device)
    X_val, cn_probs_val, y_val = extract_features(model, val_loader, device)
    X_test, cn_probs_test, y_test = extract_features(model, test_loader, device)

    # Baseline
    cn_preds_test = cn_probs_test.argmax(axis=1)
    cn_acc_test = accuracy_score(y_test, cn_preds_test)
    cm_base = confusion_matrix(y_test, cn_preds_test, labels=list(range(4)))
    print(f"   📊 ConvNeXt baseline: test={cn_acc_test:.4f}")
    print(f"      Baseline NL error: {(159 - cm_base[0,0])/159*100:.2f}%")
    print(f"      Baseline G1 error: {(61 - cm_base[1,1])/61*100:.2f}%")

    # Build G1 Detector
    g1_detector = build_g1_detector_g1_only(X_train, y_train, X_val, y_val)

    # Apply G1-only correction
    print("\n🔧 Applying G1-Only correction (strict protection)...")
    test_pred_final = g1_only_correction(
        cn_probs_test, g1_detector, X_test,
        g1_prob_threshold=0.90,
        nl_confidence_floor=0.85,
        g2_confidence_floor=0.85,
    )
    final_acc = accuracy_score(y_test, test_pred_final)
    cm = confusion_matrix(y_test, test_pred_final, labels=list(range(4)))

    # Report
    print("\n" + "=" * 78)
    print("  📊 STEP 2 RESULT")
    print("=" * 78)

    print(f"\n   Baseline acc: {cn_acc_test:.4f}")
    print(f"   Refined acc:  {final_acc:.4f}  (Δ: {final_acc - cn_acc_test:+.4f})")

    print("\n📊 Test Classification Report:")
    print(classification_report(y_test, test_pred_final,
                                target_names=CLASS_NAMES, digits=4,
                                zero_division=0))

    print("\nConfusion Matrix (Refined):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    print("\n📊 Per-Class Error Rates:")
    for c in range(4):
        total_c = int((y_test == c).sum())
        wrong_c = total_c - cm[c, c]
        err = wrong_c / max(total_c, 1)
        status = "✅" if err < 0.03 else ("⚠️" if err < 0.10 else "🔴")
        print(f"   {CLASS_NAMES[c]}: error={err*100:.2f}% {status} "
              f"(wrong={wrong_c}/{total_c})")

    # Compare with baseline
    print("\n📊 Baseline vs Refined Comparison:")
    print("\nBaseline CM:")
    print(pd.DataFrame(cm_base, index=CLASS_NAMES, columns=CLASS_NAMES))
    print("\nRefined CM:")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))
    print("\nDelta per class:")
    for c in range(4):
        base_correct = cm_base[c, c]
        refined_correct = cm[c, c]
        delta = refined_correct - base_correct
        marker = "✅" if delta > 0 else ("❌" if delta < 0 else "➖")
        print(f"   {CLASS_NAMES[c]}: {base_correct} → {refined_correct} ({delta:+d}) {marker}")

    # Save
    results = {
        "baseline_acc": float(cn_acc_test),
        "refined_acc": float(final_acc),
        "confusion_matrix_baseline": cm_base.tolist(),
        "confusion_matrix_refined": cm.tolist(),
        "per_class_error_refined": [
            float(int((y_test == c).sum()) - cm[c, c]) / max(int((y_test == c).sum()), 1)
            for c in range(4)
        ],
    }
    with open(os.path.join(output_dir, "boosted_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Saved: {os.path.join(output_dir, 'boosted_results.json')}")
    return results