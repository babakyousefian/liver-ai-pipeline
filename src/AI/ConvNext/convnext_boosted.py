# src/AI/ConvNext/convnext_boosted.py
# -*- coding: utf-8 -*-
"""
ConvNeXt Boosted Ensemble — v6: Hierarchical XGBoost
=====================================================
Strategy:
  Stage 1: XGBoost binary (NL vs rest) on ConvNeXt features
  Stage 2: XGBoost 3-class (G1/G2/G3) on non-NL samples
  Stage 3: Threshold tuning per stage (on validation)
  Stage 4: Optional kNN assist for G3

Why this works:
  - NL/G1 confusion is the main error source
  - Hierarchical splits force the model to learn NL boundary separately
  - XGBoost handles class imbalance better than AdaBoost
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
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.AI.ResNet.utils import (
    CLASS_NAMES, CROPPED, CSVAddress, validate_csv, LiverDataset,
)

FINAL_REPORTS_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"
LABELS_CSV = os.path.join(FINAL_REPORTS_DIR, "labels.csv")
CROPPED_IMAGES_DIR = os.path.join(FINAL_REPORTS_DIR, "cropped")


# ═════════════════════════════════════════════════════════════════════
# Sanity check (unchanged)
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
# Feature extraction
# ═════════════════════════════════════════════════════════════════════
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
# Stage 1: Binary NL vs rest
# ═════════════════════════════════════════════════════════════════════
def train_stage1_nl(X_train, y_train, X_val, y_val):
    """Binary: is this NL or not?"""
    print("\n🚀 Stage 1: NL vs rest (XGBoost binary)...")
    y_bin = (y_train != 0).astype(int)  # 0=NL, 1=rest
    y_val_bin = (y_val != 0).astype(int)

    pos_weight = (y_bin == 0).sum() / max((y_bin == 1).sum(), 1)
    print(f"   Class balance: NL={int((y_bin==0).sum())}, rest={int((y_bin==1).sum())}")

    clf = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_weight,  # boost rest class
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(X_train, y_bin, eval_set=[(X_val, y_val_bin)], verbose=False)

    val_probs = clf.predict_proba(X_val)[:, 1]  # P(not NL)
    val_acc = accuracy_score(y_val_bin, (val_probs > 0.5).astype(int))
    print(f"   ✅ Stage 1 val_acc={val_acc:.4f}")
    return clf


# ═════════════════════════════════════════════════════════════════════
# Stage 2: G1 vs G2 vs G3 (only on non-NL samples)
# ═════════════════════════════════════════════════════════════════════
def train_stage2_grades(X_train, y_train, X_val, y_val):
    """3-class: G1 vs G2 vs G3 on non-NL samples."""
    print("\n🚀 Stage 2: G1 vs G2 vs G3 (XGBoost 3-class)...")

    mask_train = y_train != 0
    mask_val = y_val != 0
    X_tr = X_train[mask_train]
    y_tr = y_train[mask_train] - 1  # 0=G1, 1=G2, 2=G3
    X_va = X_val[mask_val]
    y_va = y_val[mask_val] - 1

    print(f"   Train samples (non-NL): {len(X_tr)} | Val: {len(X_va)}")
    print(f"   Class counts: G1={int((y_tr==0).sum())}, "
          f"G2={int((y_tr==1).sum())}, G3={int((y_tr==2).sum())}")

    # Class weights for imbalance
    counts = np.bincount(y_tr, minlength=3)
    weights = len(y_tr) / (3 * np.maximum(counts, 1))
    sample_weights = weights[y_tr]

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(X_tr, y_tr, sample_weight=sample_weights,
            eval_set=[(X_va, y_va)], verbose=False)

    val_probs = clf.predict_proba(X_va)
    val_acc = accuracy_score(y_va, val_probs.argmax(axis=1))
    print(f"   ✅ Stage 2 val_acc={val_acc:.4f}")
    return clf


# ═════════════════════════════════════════════════════════════════════
# Hierarchical prediction
# ═════════════════════════════════════════════════════════════════════
def hierarchical_predict(stage1, stage2, X, nl_threshold=0.5):
    """
    Stage 1: P(NL) vs P(rest) → if P(rest) > nl_threshold → not NL
    Stage 2: among non-NL, pick G1/G2/G3
    """
    p_not_nl = stage1.predict_proba(X)[:, 1]

    # Default: predict NL
    preds = np.zeros(len(X), dtype=int)

    # For samples predicted as non-NL, use stage 2
    non_nl_mask = p_not_nl > nl_threshold
    if non_nl_mask.sum() > 0:
        grade_probs = stage2.predict_proba(X[non_nl_mask])
        grade_preds = grade_probs.argmax(axis=1) + 1  # +1 to map 0→G1, etc.
        preds[non_nl_mask] = grade_preds

    return preds, p_not_nl


# ═════════════════════════════════════════════════════════════════════
# Tune NL threshold (the key decision!)
# ═════════════════════════════════════════════════════════════════════
def tune_nl_threshold(stage1, stage2, X_val, y_val):
    """
    Find threshold that minimizes max per-class error.
    Grid search on validation.
    """
    print("\n🎯 Tuning NL threshold (grid search on val)...")
    p_not_nl = stage1.predict_proba(X_val)[:, 1]

    # Pre-compute stage2 probs for all non-NL candidates
    # (we do this for the max threshold to avoid recomputing)
    best_thr = 0.5
    best_score = -1e9
    best_errs = None

    for thr in np.arange(0.20, 0.81, 0.02):
        preds = np.zeros(len(X_val), dtype=int)
        non_nl = p_not_nl > thr
        if non_nl.sum() > 0:
            grade_probs = stage2.predict_proba(X_val[non_nl])
            preds[non_nl] = grade_probs.argmax(axis=1) + 1

        cm = confusion_matrix(y_val, preds, labels=list(range(4)))
        errs = []
        for c in range(4):
            total = max(int((y_val == c).sum()), 1)
            errs.append((total - cm[c, c]) / total)
        errs = np.array(errs)

        # Score: minimize mean + 3*max (heavily penalize worst class)
        score = -(np.mean(errs) + 3.0 * np.max(errs))

        if score > best_score:
            best_score = score
            best_thr = thr
            best_errs = errs.copy()

    print(f"   ✅ Best NL threshold={best_thr:.2f}")
    print(f"      val errors: {dict(zip(CLASS_NAMES, (best_errs*100).round(2)))}")
    return best_thr


# ═════════════════════════════════════════════════════════════════════
# G3 assist: lower threshold for rare class
# ═════════════════════════════════════════════════════════════════════
def g3_knn_assist(stage1, stage2, X, X_train, y_train, nl_threshold,
                  g3_boost=0.0):
    """
    Get hierarchical predictions, then boost G3 if kNN strongly agrees.
    """
    preds, p_not_nl = hierarchical_predict(stage1, stage2, X, nl_threshold)

    # kNN for G3 detection
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X_train, y_train)
    knn_probs = knn.predict_proba(X)
    g3_col = list(knn.classes_).index(3) if 3 in knn.classes_ else None

    if g3_col is None:
        return preds, 0

    n_g3_overrides = 0
    for i in range(len(preds)):
        if preds[i] != 3 and knn_probs[i, g3_col] > (0.4 - g3_boost):
            preds[i] = 3
            n_g3_overrides += 1

    return preds, n_g3_overrides


# ═════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════
def run_convnext_boosted(convnext_checkpoint, output_dir, device="cuda"):
    from torchvision import models

    print("\n" + "=" * 78)
    print("  🚀 CONVNEXT + HIERARCHICAL XGBOOST (v6)")
    print("=" * 78)

    os.makedirs(output_dir, exist_ok=True)
    sanity = sanity_check_data(LABELS_CSV, CROPPED_IMAGES_DIR)
    with open(os.path.join(output_dir, "data_sanity_check.json"), "w") as f:
        json.dump(sanity, f, indent=2, default=str)

    # ─── Load ConvNeXt ───
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

    # ─── Datasets ───
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

    # ─── Extract ───
    print("\n🔬 Extracting ConvNeXt features...")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    X_train, cn_probs_train, y_train = extract_features(model, train_loader, device)
    X_val, cn_probs_val, y_val = extract_features(model, val_loader, device)
    X_test, cn_probs_test, y_test = extract_features(model, test_loader, device)
    print(f"   ✅ Features: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    cn_acc_test = accuracy_score(y_test, cn_probs_test.argmax(axis=1))
    print(f"   📊 ConvNeXt baseline: test={cn_acc_test:.4f}")

    # ─── Stage 1: NL vs rest ───
    stage1 = train_stage1_nl(X_train, y_train, X_val, y_val)

    # ─── Stage 2: grades ───
    stage2 = train_stage2_grades(X_train, y_train, X_val, y_val)

    # ─── Tune NL threshold on val ───
    nl_thr = tune_nl_threshold(stage1, stage2, X_val, y_val)

    # ─── Evaluate on test with hierarchical ───
    print("\n📊 Evaluating hierarchical on test...")
    test_pred_hier, _ = hierarchical_predict(stage1, stage2, X_test, nl_thr)
    hier_acc = accuracy_score(y_test, test_pred_hier)
    print(f"   📊 Hierarchical: test_acc={hier_acc:.4f}")

    # ─── G3 kNN assist (only if needed) ───
    print("\n🔍 G3 kNN assist...")
    test_pred_final, n_g3 = g3_knn_assist(
        stage1, stage2, X_test, X_train, y_train, nl_thr, g3_boost=0.0,
    )
    final_acc = accuracy_score(y_test, test_pred_final)
    print(f"   ✅ G3 overrides: {n_g3} → test_acc={final_acc:.4f}")

    # ─── Final report ───
    print("\n" + "=" * 78)
    print("  📊 FINAL EVALUATION")
    print("=" * 78)

    print(f"\n   ConvNeXt baseline:         {cn_acc_test:.4f}")
    print(f"   Hierarchical (v6):         {hier_acc:.4f}")
    print(f"   + G3 kNN assist (FINAL):   {final_acc:.4f}")

    print("\n📊 Final Test Classification Report:")
    print(classification_report(y_test, test_pred_final,
                                target_names=CLASS_NAMES, digits=4,
                                zero_division=0))

    cm = confusion_matrix(y_test, test_pred_final, labels=list(range(4)))
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    print("\n📊 Per-Class Error Rates (target < 3%):")
    all_under = True
    for c in range(4):
        total_c = int((y_test == c).sum())
        wrong_c = total_c - cm[c, c]
        err = wrong_c / max(total_c, 1)
        ok = err < 0.03
        all_under = all_under and ok
        status = "✅" if ok else ("⚠️" if err < 0.10 else "🔴")
        print(f"   {CLASS_NAMES[c]}: error={err*100:.2f}% {status} "
              f"(wrong={wrong_c}/{total_c})")

    # ─── Report ───
    print("\n📄 Generating report...")
    from .convnext_boosted_report import generate_boosted_report
    generate_boosted_report(
        y_test=y_test,
        test_pred_raw=cn_probs_test.argmax(axis=1),
        test_pred_thr=test_pred_final,
        test_probs=cn_probs_test,
        clf=None,  # hierarchical has no single clf
        thresholds=np.array([nl_thr, 0.5, 0.5, 0.5]),
        output_dir=output_dir,
    )

    results = {
        "convnext_acc": float(cn_acc_test),
        "hierarchical_acc": float(hier_acc),
        "final_acc": float(final_acc),
        "nl_threshold": float(nl_thr),
        "confusion_matrix": cm.tolist(),
        "per_class_error": [
            float(int((y_test == c).sum()) - cm[c, c]) / max(int((y_test == c).sum()), 1)
            for c in range(4)
        ],
        "all_under_3pct": bool(all_under),
        "sanity_check": sanity,
    }
    with open(os.path.join(output_dir, "boosted_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Saved: {os.path.join(output_dir, 'boosted_results.json')}")
    return results