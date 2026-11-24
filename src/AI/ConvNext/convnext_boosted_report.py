# src/AI/ConvNext/convnext_boosted_report.py
# -*- coding: utf-8 -*-

"""
Full Boosting Report — includes:
  - Confusion matrices
  - Per-class metrics
  - Baseline vs Boosted comparison
  - Feature importance (top 20)
  - Threshold visualization
  - ROC curves
  - PDF report
"""

import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    roc_curve, auc, classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
)

from src.AI.ResNet.utils import CLASS_NAMES

CLASS_COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]


def generate_boosted_report(
        y_test, test_pred_raw, test_pred_thr,
        test_probs, clf, thresholds,
        output_dir,
):
    """Generate all plots + PDF for boosted results."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n📄 Generating boosted report...")

    # ═══════════════════════════════════════════════════════════════
    # 1. Confusion Matrices
    # ═══════════════════════════════════════════════════════════════
    cm_raw = confusion_matrix(y_test, test_pred_raw, labels=list(range(4)))
    cm_thr = confusion_matrix(y_test, test_pred_thr, labels=list(range(4)))

    _plot_confusion(cm_raw, "Baseline (raw)",
                    os.path.join(output_dir, "cm_baseline.png"))
    _plot_confusion(cm_thr, "Boosted (with thresholds + overrides)",
                    os.path.join(output_dir, "cm_boosted.png"))

    # ═══════════════════════════════════════════════════════════════
    # 2. Per-class metrics comparison
    # ═══════════════════════════════════════════════════════════════
    _plot_per_class_comparison(y_test, test_pred_raw, test_pred_thr,
                               os.path.join(output_dir, "per_class_comparison.png"))

    # ═══════════════════════════════════════════════════════════════
    # 3. Error rates bar
    # ═══════════════════════════════════════════════════════════════
    _plot_error_rates(y_test, test_pred_raw, test_pred_thr,
                      os.path.join(output_dir, "error_rates.png"))

    # ═══════════════════════════════════════════════════════════════
    # 4. ROC curves (raw probs)
    # ═══════════════════════════════════════════════════════════════
    _plot_roc(y_test, test_probs, os.path.join(output_dir, "roc_curves.png"))

    # ═══════════════════════════════════════════════════════════════
    # 5. Feature importance (top 20)
    # ═══════════════════════════════════════════════════════════════
    try:
        _plot_feature_importance(clf, os.path.join(output_dir, "feature_importance.png"))
    except Exception as e:
        print(f"   ⚠️ Feature importance failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 6. Thresholds visualization
    # ═══════════════════════════════════════════════════════════════
    _plot_thresholds(thresholds, os.path.join(output_dir, "thresholds.png"))

    # ═══════════════════════════════════════════════════════════════
    # 7. Save JSON + TXT
    # ═══════════════════════════════════════════════════════════════
    acc_raw = accuracy_score(y_test, test_pred_raw)
    acc_thr = accuracy_score(y_test, test_pred_thr)

    results = {
        "baseline_acc": float(acc_raw),
        "boosted_acc": float(acc_thr),
        "thresholds": thresholds.tolist(),
        "confusion_matrix_baseline": cm_raw.tolist(),
        "confusion_matrix_boosted": cm_thr.tolist(),
        "per_class_error_baseline": [
            float((y_test == c).sum() - cm_raw[c, c]) / max(int((y_test == c).sum()), 1)
            for c in range(4)
        ],
        "per_class_error_boosted": [
            float((y_test == c).sum() - cm_thr[c, c]) / max(int((y_test == c).sum()), 1)
            for c in range(4)
        ],
    }

    with open(os.path.join(output_dir, "boosted_report.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # TXT summary
    with open(os.path.join(output_dir, "boosted_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  CONVNEXT BOOSTED ENSEMBLE — FULL REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Generated: {datetime.now().isoformat()}\n")
        f.write(f"  Baseline Acc: {acc_raw:.4f}\n")
        f.write(f"  Boosted Acc:  {acc_thr:.4f}\n\n")
        f.write("-" * 70 + "\n")
        f.write("  CONFUSION MATRIX (Boosted)\n")
        f.write("-" * 70 + "\n")
        df_cm = pd.DataFrame(cm_thr, index=CLASS_NAMES, columns=CLASS_NAMES)
        f.write(df_cm.to_string() + "\n\n")
        f.write("-" * 70 + "\n")
        f.write("  PER-CLASS ERROR RATES\n")
        f.write("-" * 70 + "\n")
        for c in range(4):
            f.write(f"  {CLASS_NAMES[c]}: {results['per_class_error_boosted'][c] * 100:.2f}%\n")
        f.write("\n" + "=" * 70 + "\n")

    # ═══════════════════════════════════════════════════════════════
    # 8. PDF report
    # ═══════════════════════════════════════════════════════════════
    _generate_pdf_report(
        y_test, test_pred_thr, test_probs,
        cm_thr, thresholds, results, output_dir,
    )

    print(f"   ✅ All reports saved to: {output_dir}")


def _plot_confusion(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar=True, square=True, linewidths=0.5,
                annot_kws={"size": 13, "fontweight": "bold"}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_per_class_comparison(y_true, pred_raw, pred_thr, save_path):
    p_raw, r_raw, f1_raw, _ = precision_recall_fscore_support(
        y_true, pred_raw, labels=list(range(4)), average=None, zero_division=0)
    p_thr, r_thr, f1_thr, _ = precision_recall_fscore_support(
        y_true, pred_thr, labels=list(range(4)), average=None, zero_division=0)

    x = np.arange(4)
    w = 0.2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

    for ax, (raw, thr, name) in zip(axes, [
        (r_raw, r_thr, "Recall"),
        (p_raw, p_thr, "Precision"),
        (f1_raw, f1_thr, "F1"),
    ]):
        ax.bar(x - w / 2, raw, w, label="Baseline", color="#3498db", edgecolor="black")
        ax.bar(x + w / 2, thr, w, label="Boosted", color="#2ecc71", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(name, fontweight="bold")
        ax.set_title(f"{name} Comparison", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_error_rates(y_true, pred_raw, pred_thr, save_path):
    errors_raw = []
    errors_thr = []
    for c in range(4):
        total_c = max(int((y_true == c).sum()), 1)
        errors_raw.append((total_c - (pred_raw[y_true == c] == c).sum()) / total_c * 100)
        errors_thr.append((total_c - (pred_thr[y_true == c] == c).sum()) / total_c * 100)

    x = np.arange(4)
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.bar(x - w / 2, errors_raw, w, label="Baseline", color="#e74c3c", edgecolor="black")
    ax.bar(x + w / 2, errors_thr, w, label="Boosted", color="#2ecc71", edgecolor="black")
    ax.axhline(y=3, color="red", linestyle="--", linewidth=2, label="Target (< 3%)")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Error Rate (%)", fontweight="bold")
    ax.set_title("Per-Class Error Rates", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_roc(y_true, y_prob, save_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    for i, name in enumerate(CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.5,
                color=CLASS_COLORS[i % len(CLASS_COLORS)],
                label=f"{name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.6)
    ax.set_xlabel("FPR", fontweight="bold")
    ax.set_ylabel("TPR", fontweight="bold")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_feature_importance(clf, save_path):
    """Top-20 feature importance from the first tree of AdaBoost."""
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-20:]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.barh(range(len(top_idx)), importances[top_idx], color="#9b59b6")
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([f"feat_{i}" for i in top_idx])
    ax.set_xlabel("Importance", fontweight="bold")
    ax.set_title("Top-20 Feature Importance", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_thresholds(thresholds, save_path):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.bar(CLASS_NAMES, thresholds, color=CLASS_COLORS, edgecolor="black")
    for i, v in enumerate(thresholds):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Threshold (prob / thr for argmax)", fontweight="bold")
    ax.set_title("Per-Class Decision Thresholds", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _generate_pdf_report(y_test, pred_thr, probs, cm_thr, thresholds, results, output_dir):
    """Generate PDF with all plots."""
    try:
        import fitz
    except ImportError:
        print("   ⚠️ PyMuPDF not installed, skipping PDF")
        return

    pdf_path = os.path.join(output_dir, "BoostedReport.pdf")
    doc = fitz.open()

    # Cover page
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 80), "ConvNeXt Boosted Ensemble",
                     fontsize=28, fontname="helvetica-bold", color=(0, 0.55, 0.85))
    page.insert_text((50, 130), f"Baseline: {results['baseline_acc']:.4f}",
                     fontsize=16, fontname="helvetica", color=(0.3, 0.3, 0.3))
    page.insert_text((50, 160), f"Boosted:  {results['boosted_acc']:.4f}",
                     fontsize=20, fontname="helvetica-bold", color=(0, 0.7, 0.3))

    # Add images
    image_pages = [
        ("cm_boosted.png", "Boosted Confusion Matrix"),
        ("cm_baseline.png", "Baseline Confusion Matrix"),
        ("per_class_comparison.png", "Per-Class Metrics"),
        ("error_rates.png", "Per-Class Error Rates"),
        ("roc_curves.png", "ROC Curves"),
        ("thresholds.png", "Learned Thresholds"),
        ("feature_importance.png", "Feature Importance"),
    ]

    for fname, title in image_pages:
        p = os.path.join(output_dir, fname)
        if os.path.exists(p):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), title,
                             fontsize=16, fontname="helvetica-bold",
                             color=(0, 0.55, 0.85))
            page.insert_image(fitz.Rect(50, 80, 545, 780), filename=p)

    doc.save(pdf_path)
    doc.close()
    print(f"   ✅ PDF saved: {pdf_path}")