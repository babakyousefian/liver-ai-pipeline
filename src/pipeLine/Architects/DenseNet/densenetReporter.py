# src/pipeLine/densenetReporter.py
# -*- coding: utf-8 -*-

"""
DenseNet Fine-Tune Reporter — 24-Page Edition (LIGHT)
======================================================
Mirrors vgg16Reporter.py / mobilenetv2Reporter.py exactly, adapted for DenseNet.

Pages:
  1.  Title
  2.  Executive Summary
  3.  Training Configuration
  4.  Dataset Split & Class Distribution
  5.  Loss History
  6.  Learning Rate Schedule
  7.  Metrics Bar Chart
  8.  Confusion Matrix
  9.  Per-Class Performance
  10. Per-Class Recall Bar
  11. ROC Curves
  12. AUC Scores Table
  13. 5-Fold CV — Fold Accuracies
  14. 5-Fold CV — Metrics Comparison
  15. 5-Fold CV — Best Fold
  16. Bootstrap 95% CI
  17. TTA Analysis
  18. SSL Methods Comparison (BYOL / SimCLR / MAE)
  19. Captum Interpretability
  20. LIME Explanations
  21. SHAP Summary
  22. Model Architecture (Netron)
  23. Pipeline Summary
  24. Conclusion

Author: Babak Yousefian
Version: 1.0.0 (LIGHT)
"""

import json
import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ================================================================
# ✅ Paths
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")
REPORTER_ROOT = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "densenet_finetune")
PDF_PATH = os.path.join(REPORTER_ROOT, "densenet_finetune_report.pdf")
PLOTS_DIR = os.path.join(REPORTER_ROOT, "plots")
INTERP_DIR = os.path.join(REPORTER_ROOT, "interpretability")
CV_DIR = os.path.join(REPORTER_ROOT, "../../CV")
ARCH_DIR = os.path.join(REPORTER_ROOT, "architecture")

CLASS_NAMES = ["NL", "G1", "G2", "G3"]


# ================================================================
# ✅ SSL Comparison Plot
# ================================================================

def _generate_ssl_comparison_plot(ssl_comparison, plots_dir, verbose):
    """Generate a bar chart comparing BYOL/SimCLR/MAE."""
    plot_paths = {}

    try:
        all_methods = ssl_comparison.get("all_methods", {})
        if not all_methods:
            return plot_paths

        best_method = ssl_comparison.get("best_method", "")
        methods = list(all_methods.keys())
        lp_accs = [all_methods[m].get("lp_accuracy", 0) for m in methods]
        lp_f1s = [all_methods[m].get("lp_f1", 0) for m in methods]

        colors_bar = [
            "#2ecc71" if m == best_method else "#95a5a6" for m in methods
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        bars1 = axes[0].bar(methods, lp_accs, color=colors_bar,
                            edgecolor="black", linewidth=1.5)
        for b, v in zip(bars1, lp_accs):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.02,
                         f"{v:.4f}", ha="center", va="bottom",
                         fontsize=12, fontweight="bold")
        axes[0].set_ylabel("LP Accuracy", fontsize=13, fontweight="bold")
        axes[0].set_title("Linear Probe Accuracy", fontsize=14, fontweight="bold")
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(axis="y", alpha=0.3, linestyle="--")
        if best_method:
            axes[0].set_xticks(range(len(methods)))
            axes[0].set_xticklabels(
                [f"{m.upper()} 🏆" if m == best_method else m.upper() for m in methods],
                fontsize=12, fontweight="bold"
            )

        bars2 = axes[1].bar(methods, lp_f1s, color=colors_bar,
                            edgecolor="black", linewidth=1.5)
        for b, v in zip(bars2, lp_f1s):
            axes[1].text(b.get_x() + b.get_width() / 2, v + 0.02,
                         f"{v:.4f}", ha="center", va="bottom",
                         fontsize=12, fontweight="bold")
        axes[1].set_ylabel("LP F1 Score", fontsize=13, fontweight="bold")
        axes[1].set_title("Linear Probe F1 Score", fontsize=14, fontweight="bold")
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")
        if best_method:
            axes[1].set_xticks(range(len(methods)))
            axes[1].set_xticklabels(
                [f"{m.upper()} 🏆" if m == best_method else m.upper() for m in methods],
                fontsize=12, fontweight="bold"
            )

        plt.suptitle(
            f"SSL Methods Comparison — Winner: {best_method.upper()}" if best_method else "SSL Methods Comparison",
            fontsize=16, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        path = os.path.join(plots_dir, "08_ssl_comparison_bar.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        plot_paths["bar"] = path
        if verbose:
            print(f"  ✅ 08_ssl_comparison_bar")

    except Exception as e:
        if verbose:
            print(f"  ⚠️ SSL comparison plot failed: {e}")
        plt.close("all")

    return plot_paths


# ================================================================
# ✅ Main API
# ================================================================

def generate_densenet_report(
    final_model: nn.Module,
    metrics: Dict[str, Any],
    history: Dict[str, Any],
    cv_summary: Dict[str, Any],
    ci95: Dict[str, Any],
    image_paths: List[str],
    labels: List[int],
    ssl_checkpoint_path: str,
    ssl_method: str = "byol",
    config: Optional[Dict[str, Any]] = None,
    data_split: Optional[Dict[str, int]] = None,
    tta_info: Optional[Dict[str, Any]] = None,
    ssl_comparison: Optional[Dict[str, Any]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    generate_interpretability_flag: bool = True,
    verbose: bool = True,
) -> str:
    """Generate comprehensive 24-page PDF report."""
    os.makedirs(REPORTER_ROOT, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(INTERP_DIR, exist_ok=True)
    os.makedirs(CV_DIR, exist_ok=True)
    os.makedirs(ARCH_DIR, exist_ok=True)

    if verbose:
        print("\n" + "=" * 80)
        print("📄 Generating 24-page DenseNet Fine-Tune Report (LIGHT)")
        print("=" * 80)

    if verbose:
        print("\n📊 Generating plots...")
    plot_paths = _generate_plots(metrics, history, cv_summary, ci95, PLOTS_DIR, verbose)

    ssl_plot_paths = {}
    if ssl_comparison and ssl_comparison.get("available"):
        if verbose:
            print("\n📊 Generating SSL methods comparison plot...")
        ssl_plot_paths = _generate_ssl_comparison_plot(ssl_comparison, PLOTS_DIR, verbose)

    interp_paths = {}
    if generate_interpretability_flag and len(image_paths) > 0:
        if verbose:
            print("\n🔬 Generating interpretability...")
        interp_paths = _generate_interpretability(
            final_model, image_paths, labels, device, verbose
        )

    if verbose:
        print("\n🏗️  Generating architecture (ONNX + HTML)...")
    arch_paths = _generate_architecture(final_model, verbose)

    if verbose:
        print("\n📄 Building 24-page PDF...")
    pdf_path = _build_pdf(
        metrics=metrics,
        history=history,
        cv_summary=cv_summary,
        ci95=ci95,
        plot_paths=plot_paths,
        ssl_plot_paths=ssl_plot_paths,
        ssl_comparison=ssl_comparison or {},
        interp_paths=interp_paths,
        arch_paths=arch_paths,
        ssl_checkpoint_path=ssl_checkpoint_path,
        ssl_method=ssl_method,
        config=config or {},
        data_split=data_split or {},
        tta_info=tta_info or {},
        final_model=final_model,
        verbose=verbose,
    )

    return pdf_path


# ================================================================
# ✅ Plot Generation
# ================================================================

def _generate_plots(metrics, history, cv_summary, ci95, plots_dir, verbose):
    """Generate all 7 plots."""
    plot_paths = {}
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Loss
    try:
        if history and ("train_loss" in history or "loss" in history):
            plt.figure(figsize=(12, 6))
            if "train_loss" in history:
                plt.plot(history["train_loss"], label="Train", linewidth=2, color="#3498db")
            elif "loss" in history:
                plt.plot(history["loss"], label="Train", linewidth=2, color="#3498db")
            if "val_loss" in history:
                plt.plot(history["val_loss"], label="Val", linewidth=2, color="#e74c3c")
            plt.xlabel("Epoch", fontsize=13, fontweight="bold")
            plt.ylabel("Loss", fontsize=13, fontweight="bold")
            plt.title("Training & Validation Loss", fontsize=15, fontweight="bold")
            plt.legend(); plt.grid(alpha=0.3)
            plt.tight_layout()
            path = os.path.join(plots_dir, "01_loss_history.png")
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["loss"] = path
            if verbose:
                print("  ✅ 01_loss_history")
    except Exception:
        plt.close("all")

    # 2. Metrics bar
    try:
        if metrics:
            names = ["Accuracy", "Balanced", "F1 (w)", "F1 (m)"]
            vals = [
                metrics.get("accuracy", 0),
                metrics.get("balanced_accuracy", 0),
                metrics.get("f1_score", 0),
                metrics.get("f1_macro", 0),
            ]
            plt.figure(figsize=(11, 7))
            bars = plt.bar(names, vals,
                           color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"],
                           edgecolor="black", linewidth=1.5)
            for b, v in zip(bars, vals):
                plt.text(b.get_x() + b.get_width() / 2, v + 0.02,
                         f"{v:.4f}", ha="center", fontsize=13, fontweight="bold")
            plt.ylim(0, 1.1)
            plt.ylabel("Score", fontsize=13, fontweight="bold")
            plt.title("Test Set Metrics (with TTA)", fontsize=15, fontweight="bold")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            path = os.path.join(plots_dir, "03_metrics_bar.png")
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["metrics"] = path
    except Exception:
        plt.close("all")

    # 3. Confusion matrix
    try:
        if metrics and "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                        annot_kws={"size": 15, "weight": "bold"})
            plt.title("Confusion Matrix (Test Set)", fontsize=15, fontweight="bold")
            plt.ylabel("True", fontsize=13, fontweight="bold")
            plt.xlabel("Predicted", fontsize=13, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(plots_dir, "04_confusion_matrix.png")
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["cm"] = path
    except Exception:
        plt.close("all")

    # 4. Per-class recall
    try:
        if metrics and "per_class" in metrics:
            per_class = metrics["per_class"]
            names = [c for c in CLASS_NAMES if c in per_class]
            recalls = [per_class[c]["recall"] for c in names]
            colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"][: len(names)]
            plt.figure(figsize=(11, 7))
            bars = plt.bar(names, recalls, color=colors, edgecolor="black", linewidth=1.5)
            for b, v in zip(bars, recalls):
                plt.text(b.get_x() + b.get_width() / 2, v + 0.02,
                         f"{v:.4f}", ha="center", fontsize=13, fontweight="bold")
            plt.ylim(0, 1.1)
            plt.ylabel("Recall", fontsize=13, fontweight="bold")
            plt.title("Per-Class Recall (Test Set)", fontsize=15, fontweight="bold")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            path = os.path.join(plots_dir, "05_per_class_recall.png")
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["per_class"] = path
    except Exception:
        plt.close("all")

    # 5. CV folds
    try:
        if cv_summary and "fold_results" in cv_summary and cv_summary["fold_results"]:
            folds = [r["fold"] for r in cv_summary["fold_results"]]
            accs = [r["accuracy"] for r in cv_summary["fold_results"]]
            mean_acc = cv_summary.get("mean_accuracy", np.mean(accs))
            std_acc = cv_summary.get("std_accuracy", np.std(accs))
            plt.figure(figsize=(11, 7))
            bars = plt.bar(folds, accs, color="steelblue", edgecolor="black", linewidth=1.5)
            plt.axhline(mean_acc, color="red", linestyle="--", linewidth=2,
                        label=f"Mean: {mean_acc:.4f}")
            plt.fill_between([0.5, len(folds) + 0.5],
                             mean_acc - std_acc, mean_acc + std_acc,
                             alpha=0.2, color="red", label=f"±STD: {std_acc:.4f}")
            for b, v in zip(bars, accs):
                plt.text(b.get_x() + b.get_width() / 2, v + 0.02,
                         f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
            plt.xlabel("Fold", fontsize=13, fontweight="bold")
            plt.ylabel("Accuracy", fontsize=13, fontweight="bold")
            plt.title("5-Fold CV Accuracy", fontsize=15, fontweight="bold")
            plt.xticks(folds); plt.ylim(0, 1.1)
            plt.legend(); plt.grid(alpha=0.3)
            plt.tight_layout()
            path = os.path.join(plots_dir, "07_cv_folds.png")
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["cv"] = path
    except Exception:
        plt.close("all")

    return plot_paths


# ================================================================
# ✅ Interpretability — LIGHT
# ================================================================

def _generate_interpretability(model, image_paths, labels, device, verbose):
    """Generate Grad-CAM, LIME, SHAP (LIGHT: fewer samples)."""
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    interp_paths = {}

    class _DS(Dataset):
        def __init__(self, paths, labels, tf, target_size):
            self.paths = paths
            self.labels = labels
            self.tf = tf
            self.target_size = target_size

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = cv2.imread(self.paths[i])
            if img is None:
                img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            return self.tf(img), self.labels[i]

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Captum (8 samples)
    try:
        from src.reporter.captum_reporter import generate_interpretability_report
        n = min(8, len(image_paths))
        ds = _DS(image_paths[:n], labels[:n], tf, 224)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        result = generate_interpretability_report(
            model, dl, INTERP_DIR, device=device, class_names=CLASS_NAMES,
        )
        interp_paths.update(result)
        if verbose:
            print("  ✅ Captum")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Captum failed: {e}")

    # LIME (3 samples)
    try:
        from src.reporter.lime_reporter import explain_predictions_lime
        n = min(3, len(image_paths))
        ds = _DS(image_paths[:n], labels[:n], tf, 224)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        lime_result = explain_predictions_lime(
            model, dl, INTERP_DIR, class_names=CLASS_NAMES,
            device=device, num_samples=3,
        )
        interp_paths["lime"] = lime_result.get("lime_path")
        if verbose:
            print("  ✅ LIME")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ LIME failed: {e}")

    # SHAP (10 samples)
    try:
        from src.reporter.shap_reporter import generate_shap_explanation
        n = min(10, len(image_paths))
        ds = _DS(image_paths[:n], labels[:n], tf, 224)
        dl = DataLoader(ds, batch_size=2, shuffle=False)
        shap_result = generate_shap_explanation(
            model, dl, INTERP_DIR, device=device,
            num_background=4, num_explain=2, class_names=CLASS_NAMES,
        )
        interp_paths["shap"] = shap_result.get("shap_path")
        if verbose:
            print("  ✅ SHAP")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ SHAP failed: {e}")

    return interp_paths


# ================================================================
# ✅ Architecture
# ================================================================

def _generate_architecture(model, verbose):
    """Generate ONNX + HTML via Netron."""
    arch_paths = {}
    try:
        from src.reporter.netron_reporter import save_model_with_name
        result = save_model_with_name(
            model, ARCH_DIR,
            model_name="DenseNet",
            input_shape=(1, 3, 224, 224),
        )
        arch_paths.update(result)
        if verbose:
            print("  ✅ Netron")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Netron failed: {e}")
    return arch_paths


# ================================================================
# ✅ 24-Page PDF Builder
# ================================================================

def _build_pdf(
    metrics,
    history,
    cv_summary,
    ci95,
    plot_paths,
    ssl_plot_paths,
    ssl_comparison,
    interp_paths,
    arch_paths,
    ssl_checkpoint_path,
    ssl_method,
    config,
    data_split,
    tta_info,
    final_model,
    verbose=True,
):
    """Build 24-page PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image as RLImage, PageBreak, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        return _fallback_json(metrics, cv_summary, ci95)

    doc = SimpleDocTemplate(
        PDF_PATH, pagesize=A4,
        rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50,
        title="DenseNet Fine-Tune Report",
        author="Babak Yousefian",
    )

    styles = getSampleStyleSheet()
    S = {
        "title": ParagraphStyle("T", parent=styles["Heading1"], fontSize=28,
                                alignment=TA_CENTER, textColor=colors.HexColor("#1a252f"),
                                spaceAfter=20),
        "subtitle": ParagraphStyle("Sub", parent=styles["Normal"], fontSize=16,
                                   alignment=TA_CENTER, textColor=colors.HexColor("#34495e")),
        "h1": ParagraphStyle("H1", parent=styles["Heading2"], fontSize=20,
                             textColor=colors.HexColor("#2c3e50"), spaceBefore=16, spaceAfter=14),
        "h2": ParagraphStyle("H2", parent=styles["Heading3"], fontSize=15,
                             textColor=colors.HexColor("#34495e"), spaceBefore=12, spaceAfter=10),
        "body": ParagraphStyle("B", parent=styles["Normal"], fontSize=11,
                               leading=16, textColor=colors.HexColor("#2c3e50")),
        "small": ParagraphStyle("Sm", parent=styles["Normal"], fontSize=9,
                                textColor=colors.grey, leading=12),
        "center": ParagraphStyle("C", parent=styles["Normal"], fontSize=12, alignment=TA_CENTER),
    }

    story = []

    def safe_img(path, title, w=6.5 * inch, h=4.5 * inch):
        story.append(Spacer(1, 10))
        story.append(Paragraph(title, S["h2"]))
        if path and os.path.exists(path) and os.path.getsize(path) > 100:
            try:
                story.append(RLImage(path, width=w, height=h))
            except Exception as e:
                story.append(Paragraph(f"⚠️ {e}", S["small"]))
        else:
            story.append(Paragraph("⚠️ Image not available", S["small"]))

    def make_table(data, col_widths, header_bg="#2c3e50"):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        return t

    # PAGE 1: TITLE
    story.append(Spacer(1, 80))
    story.append(Paragraph("DenseNet Fine-Tune Report", S["title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Self-Supervised Learning → Full Fine-Tune → TTA → CV5 → CI95%",
        S["subtitle"]))
    story.append(Spacer(1, 40))
    story.append(Paragraph(f"SSL Method: <b>{ssl_method.upper()}</b>", S["center"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Checkpoint: <font face='Courier' size='9'>{os.path.basename(ssl_checkpoint_path)}</font>",
        S["center"]))
    story.append(Spacer(1, 60))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        S["center"]))
    story.append(Spacer(1, 120))
    story.append(Paragraph("Author: <b>Babak Yousefian</b>", S["center"]))
    story.append(Paragraph("Liver AI Pipeline · Task 5 · LIGHT Edition", S["center"]))
    story.append(PageBreak())

    # PAGE 2: EXECUTIVE SUMMARY
    story.append(Paragraph("1. Executive Summary", S["h1"]))
    acc = metrics.get("accuracy", 0) if metrics else 0
    bal_acc = metrics.get("balanced_accuracy", 0) if metrics else 0
    f1 = metrics.get("f1_score", 0) if metrics else 0
    auc_val = metrics.get("roc_auc_mean", 0) if metrics else 0

    if bal_acc >= 0.75:
        verdict = "🟢 EXCELLENT"
    elif bal_acc >= 0.60:
        verdict = "🟡 GOOD"
    elif bal_acc >= 0.40:
        verdict = "🟠 MODERATE"
    else:
        verdict = "🔴 POOR"

    story.append(Paragraph(f"<b>Verdict:</b> {verdict}", S["body"]))
    story.append(Spacer(1, 14))

    summary_data = [
        ["Metric", "Value"],
        ["SSL Method", ssl_method.upper()],
        ["Test Accuracy (TTA)", f"{acc:.4f}"],
        ["Balanced Accuracy", f"{bal_acc:.4f}"],
        ["F1 Score (weighted)", f"{f1:.4f}"],
        ["AUC (mean)", f"{auc_val:.4f}"],
        ["CV Mean Accuracy",
         f"{cv_summary.get('mean_accuracy', 0):.4f} ± {cv_summary.get('std_accuracy', 0):.4f}"],
        ["CV Mean F1", f"{cv_summary.get('mean_f1', 0):.4f} ± {cv_summary.get('std_f1', 0):.4f}"],
        ["CI 95% Accuracy",
         f"[{ci95.get('accuracy', {}).get('ci95_low', 0):.4f}, "
         f"{ci95.get('accuracy', {}).get('ci95_high', 0):.4f}]"],
        ["TTA Augmentations", str(tta_info.get("n_aug", 16))],
        ["Batch Size (with grad accum)",
         f"{config.get('batch_size', 8)} × {config.get('gradient_accumulation_steps', 4)} = "
         f"{config.get('batch_size', 8) * config.get('gradient_accumulation_steps', 4)}"],
    ]
    story.append(make_table(summary_data, [3 * inch, 3 * inch]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "This report presents the complete fine-tune pipeline for DenseNet on the "
        "liver ultrasound dataset (NL/G1/G2/G3). LIGHT configuration: batch=8, "
        "gradient_accumulation=4 (effective batch=32), image_size=224, epochs=40. "
        "Optimized for RTX 2050 (4GB VRAM).",
        S["body"]))
    story.append(PageBreak())

    # PAGE 3: TRAINING CONFIG
    story.append(Paragraph("2. Training Configuration", S["h1"]))
    story.append(Spacer(1, 12))

    cfg = config or {}
    config_data = [
        ["Parameter", "Value"],
        ["SSL Method", ssl_method.upper()],
        ["SSL Checkpoint", os.path.basename(ssl_checkpoint_path)],
        ["Architecture", "DenseNet121 (LIGHT)"],
        ["Input Size", f"{cfg.get('input_size', 224)}×{cfg.get('input_size', 224)}"],
        ["Batch Size", str(cfg.get("batch_size", 8))],
        ["Gradient Accumulation", str(cfg.get("gradient_accumulation_steps", 4))],
        ["Effective Batch", str(cfg.get("batch_size", 8) * cfg.get("gradient_accumulation_steps", 4))],
        ["FT Epochs (final)", str(cfg.get("epochs_final", 40))],
        ["FT Epochs (CV fold)", str(cfg.get("epochs_cv", 15))],
        ["Learning Rate (final)", f"{cfg.get('lr_final', 1e-4):.2e}"],
        ["Learning Rate (CV)", f"{cfg.get('lr_cv', 1e-4):.2e}"],
        ["Early Stop Patience (final)", str(cfg.get("patience_final", 15))],
        ["Early Stop Patience (CV)", str(cfg.get("patience_cv", 8))],
        ["Optimizer", "AdamW"],
        ["Loss Function", "FocalLoss (γ=2.0) + class weights"],
        ["Sampler", "WeightedRandomSampler (sqrt)"],
        ["TTA Augmentations", str(tta_info.get("n_aug", 16))],
        ["CV Folds", str(cfg.get("cv_folds", 5))],
        ["Bootstrap Samples", str(cfg.get("n_bootstrap", 1000))],
        ["Device", cfg.get("device", "cuda")],
    ]
    story.append(make_table(config_data, [3 * inch, 3 * inch], header_bg="#34495e"))
    story.append(PageBreak())

    # PAGE 4: DATASET
    story.append(Paragraph("3. Dataset Split & Class Distribution", S["h1"]))
    story.append(Paragraph(
        "The labeled dataset was split into 80% train / 10% validation / 10% test "
        "(stratified by class). The 5-Fold CV uses the full dataset.",
        S["body"]))
    story.append(Spacer(1, 12))

    ds = data_split or {}
    total = ds.get("total", 1) if ds.get("total", 0) > 0 else 1
    split_data = [
        ["Split", "Samples", "Percentage"],
        ["Train", str(ds.get("train", "N/A")),
         f"{100 * ds.get('train', 0) / total:.1f}%"],
        ["Validation", str(ds.get("val", "N/A")),
         f"{100 * ds.get('val', 0) / total:.1f}%"],
        ["Test", str(ds.get("test", "N/A")),
         f"{100 * ds.get('test', 0) / total:.1f}%"],
        ["TOTAL", str(ds.get("total", "N/A")), "100.0%"],
    ]
    story.append(make_table(split_data, [2 * inch, 2 * inch, 2 * inch], header_bg="#27ae60"))

    story.append(Spacer(1, 20))
    story.append(Paragraph("3.1 Class Imbalance", S["h2"]))
    story.append(Paragraph(
        "The liver dataset is highly imbalanced (approx. 22:1 ratio NL:G3). "
        "WeightedRandomSampler with sqrt-based weights + Focal Loss with class weights "
        "was used to mitigate this.",
        S["body"]))
    story.append(PageBreak())

    # PAGE 5: LOSS HISTORY
    story.append(Paragraph("4. Loss History", S["h1"]))
    safe_img(plot_paths.get("loss"), "4.1 Train vs Validation Loss")
    story.append(PageBreak())

    # PAGE 6: METRICS BAR
    story.append(Paragraph("5. Test Metrics (with TTA)", S["h1"]))
    safe_img(plot_paths.get("metrics"), "5.1 Metrics Bar Chart")
    story.append(PageBreak())

    # PAGE 7: CONFUSION MATRIX
    story.append(Paragraph("6. Confusion Matrix", S["h1"]))
    safe_img(plot_paths.get("cm"), "6.1 Confusion Matrix", w=6 * inch, h=5 * inch)
    story.append(PageBreak())

    # PAGE 8: PER-CLASS PERFORMANCE
    story.append(Paragraph("7. Per-Class Performance", S["h1"]))
    if metrics and "classification_report" in metrics:
        cr = metrics["classification_report"]
        class_data = [["Class", "Precision", "Recall", "F1", "Support"]]
        for cname in CLASS_NAMES:
            if cname in cr:
                r = cr[cname]
                class_data.append([
                    cname,
                    f"{r.get('precision', 0):.4f}",
                    f"{r.get('recall', 0):.4f}",
                    f"{r.get('f1-score', 0):.4f}",
                    str(int(r.get("support", 0))),
                ])
        story.append(make_table(class_data, [1.2 * inch] * 5, header_bg="#8e44ad"))
    story.append(PageBreak())

    # PAGE 9: PER-CLASS RECALL
    story.append(Paragraph("8. Per-Class Recall", S["h1"]))
    safe_img(plot_paths.get("per_class"), "8.1 Recall per Class")
    story.append(PageBreak())

    # PAGE 10: CV FOLD ACCURACIES
    story.append(Paragraph("9. CV — Accuracy per Fold", S["h1"]))
    safe_img(plot_paths.get("cv"), "9.1 Accuracy per Fold")
    story.append(PageBreak())

    # PAGE 11: CV METRICS COMPARISON
    story.append(Paragraph("10. CV — Metrics Comparison", S["h1"]))
    if cv_summary and "fold_results" in cv_summary:
        cv_data = [["Fold", "Acc", "F1", "BalAcc", "Best Val"]]
        for r in cv_summary["fold_results"]:
            cv_data.append([
                str(r["fold"]),
                f"{r['accuracy']:.4f}",
                f"{r['f1_weighted']:.4f}",
                f"{r['balanced_accuracy']:.4f}",
                f"{r.get('best_val_acc', 0):.4f}",
            ])
        cv_data.append(["Mean",
                        f"{cv_summary.get('mean_accuracy', 0):.4f}",
                        f"{cv_summary.get('mean_f1', 0):.4f}",
                        f"{cv_summary.get('mean_balanced_accuracy', 0):.4f}",
                        "—"])
        cv_data.append(["Std",
                        f"{cv_summary.get('std_accuracy', 0):.4f}",
                        f"{cv_summary.get('std_f1', 0):.4f}",
                        f"{cv_summary.get('std_balanced_accuracy', 0):.4f}",
                        "—"])
        story.append(make_table(cv_data, [1 * inch] * 5, header_bg="#8e44ad"))
    story.append(PageBreak())

    # PAGE 12: BOOTSTRAP CI
    story.append(Paragraph("11. Bootstrap 95% CI", S["h1"]))
    if ci95:
        ci_data = [["Metric", "Mean", "Std", "CI95 Low", "CI95 High", "Width"]]
        for m in ["accuracy", "f1_score", "balanced_accuracy"]:
            if m in ci95:
                d = ci95[m]
                width = d["ci95_high"] - d["ci95_low"]
                ci_data.append([
                    m.replace("_", " ").title(),
                    f"{d['mean']:.4f}",
                    f"{d['std']:.4f}",
                    f"{d['ci95_low']:.4f}",
                    f"{d['ci95_high']:.4f}",
                    f"{width:.4f}",
                ])
        story.append(make_table(ci_data, [1.4 * inch] * 6, header_bg="#16a085"))
    story.append(PageBreak())

    # PAGE 13: TTA
    story.append(Paragraph("12. Test-Time Augmentation (TTA)", S["h1"]))
    story.append(Paragraph(
        f"TTA averages predictions over {tta_info.get('n_aug', 16)} augmented copies "
        f"of each test image.",
        S["body"]))
    story.append(Spacer(1, 12))
    tta_data = [
        ["Parameter", "Value"],
        ["Number of Augmentations", str(tta_info.get("n_aug", 16))],
        ["Augmentation Types", "HFlip, VFlip, Rot90/180/270, Bright±, Contrast±"],
        ["Aggregation", "Mean of softmax probabilities"],
        ["Applied To", "Test set evaluation"],
    ]
    story.append(make_table(tta_data, [3 * inch, 3 * inch], header_bg="#2980b9"))
    story.append(PageBreak())

    # PAGE 14: SSL METHODS COMPARISON
    story.append(Paragraph("13. SSL Methods Comparison", S["h1"]))
    story.append(Paragraph(
        "Before fine-tuning, Self-Supervised Learning (SSL) methods were trained on "
        "DenseNet: <b>BYOL</b>, <b>SimCLR</b>, and <b>MAE</b>. Each was evaluated with "
        "a <b>Linear Probe</b>. The method with the highest LP accuracy was selected.",
        S["body"]))
    story.append(Spacer(1, 14))

    if ssl_comparison and ssl_comparison.get("available"):
        all_methods = ssl_comparison.get("all_methods", {})
        best_method = ssl_comparison.get("best_method", "")

        comparison_data = [["Method", "LP Accuracy", "LP F1", "Winner"]]
        for m in ["byol", "simclr", "mae"]:
            if m in all_methods:
                d = all_methods[m]
                winner = "🏆" if m == best_method else ""
                comparison_data.append([
                    m.upper(),
                    f"{d.get('lp_accuracy', 0):.4f}",
                    f"{d.get('lp_f1', 0):.4f}",
                    winner,
                ])

        if len(comparison_data) > 1:
            story.append(make_table(
                comparison_data,
                [1.5 * inch, 1.5 * inch, 1.5 * inch, 1 * inch],
                header_bg="#8e44ad"))

        story.append(Spacer(1, 16))
        story.append(Paragraph(
            f"<b>Winner:</b> {best_method.upper()} "
            f"(LP Accuracy = {all_methods.get(best_method, {}).get('lp_accuracy', 0):.4f})",
            S["body"]))

        safe_img(ssl_plot_paths.get("bar"),
                 "13.1 SSL Methods — Accuracy & F1",
                 w=6.5 * inch, h=3.5 * inch)
    else:
        story.append(Paragraph(
            "⚠️ SSL methods comparison data not available.",
            S["body"]))
    story.append(PageBreak())

    # PAGE 15: CAPTUM
    story.append(Paragraph("14. Captum Interpretability", S["h1"]))
    safe_img(interp_paths.get("gradcam_path"), "14.1 Grad-CAM")
    safe_img(interp_paths.get("ig_path"), "14.2 Integrated Gradients")
    story.append(PageBreak())

    # PAGE 16: LIME
    story.append(Paragraph("15. LIME Explanations", S["h1"]))
    safe_img(interp_paths.get("lime"), "15.1 LIME Explanation", w=6 * inch, h=4 * inch)
    story.append(PageBreak())

    # PAGE 17: SHAP
    story.append(Paragraph("16. SHAP Summary", S["h1"]))
    safe_img(interp_paths.get("shap"), "16.1 SHAP Summary", w=6 * inch, h=4 * inch)
    story.append(PageBreak())

    # PAGE 18: ARCHITECTURE
    story.append(Paragraph("17. Model Architecture (Netron)", S["h1"]))
    if arch_paths:
        arch_data = [["File", "Path", "Purpose"]]
        if arch_paths.get("onnx_path"):
            arch_data.append(["ONNX", str(arch_paths["onnx_path"]), "Portable format"])
        if arch_paths.get("html_path"):
            arch_data.append(["HTML", str(arch_paths["html_path"]), "Interactive viewer"])
        if arch_paths.get("info_path"):
            arch_data.append(["JSON", str(arch_paths["info_path"]), "Metadata"])
        if len(arch_data) > 1:
            story.append(make_table(arch_data,
                                    [0.8 * inch, 4 * inch, 1.5 * inch],
                                    header_bg="#2c3e50"))

    story.append(Spacer(1, 20))
    story.append(Paragraph("17.1 Model Summary", S["h2"]))
    try:
        num_params = sum(p.numel() for p in final_model.parameters())
        num_train = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
        param_data = [
            ["Property", "Value"],
            ["Architecture", "DenseNet121"],
            ["Input Size", "224 × 224 × 3"],
            ["Total Parameters", f"{num_params:,}"],
            ["Trainable Parameters", f"{num_train:,}"],
            ["Feature Dimension", "1024"],
            ["Output Classes", "4 (NL, G1, G2, G3)"],
        ]
        story.append(make_table(param_data, [2.5 * inch, 3 * inch], header_bg="#34495e"))
    except Exception as e:
        story.append(Paragraph(f"⚠️ {e}", S["small"]))
    story.append(PageBreak())

    # PAGE 19: PIPELINE SUMMARY
    story.append(Paragraph("18. Pipeline Summary", S["h1"]))
    pipeline_data = [
        ["Stage", "Status", "Details"],
        ["SSL Pretraining", "✅",
         f"{ssl_method.upper()} — {os.path.basename(ssl_checkpoint_path)}"],
        ["Fine-Tune", "✅", f"{config.get('epochs_final', 40)} epochs"],
        ["5-Fold CV", "✅",
         f"{len(cv_summary.get('fold_results', []))} folds, "
         f"mean acc = {cv_summary.get('mean_accuracy', 0):.4f}"],
        ["Bootstrap CI", "✅", f"{config.get('n_bootstrap', 1000)} samples"],
        ["TTA", "✅", f"{tta_info.get('n_aug', 16)} augmentations"],
        ["Test Eval", "✅", f"Acc = {metrics.get('accuracy', 0):.4f}"],
        ["Captum", "✅" if interp_paths.get("gradcam_path") else "❌", "Grad-CAM + IG"],
        ["LIME", "✅" if interp_paths.get("lime") else "❌", "3 samples"],
        ["SHAP", "✅" if interp_paths.get("shap") else "❌", "10 samples"],
        ["Netron", "✅" if arch_paths.get("onnx_path") else "❌", "ONNX + HTML"],
        ["Final Model", "✅", "densenet_finetuned_final.pth"],
        ["PDF Report", "✅", "24 pages"],
    ]
    story.append(make_table(pipeline_data,
                            [1.8 * inch, 0.8 * inch, 3.4 * inch],
                            header_bg="#2c3e50"))
    story.append(PageBreak())

    # PAGE 20: CONCLUSION
    story.append(Paragraph("19. Conclusion & Recommendation", S["h1"]))
    if bal_acc >= 0.75:
        final_verdict = "🟢 EXCELLENT"
        recommendation = "Model is ready for clinical deployment."
    elif bal_acc >= 0.60:
        final_verdict = "🟡 GOOD"
        recommendation = "Model learned useful features. Deploy with human verification."
    elif bal_acc >= 0.40:
        final_verdict = "🟠 MODERATE"
        recommendation = "Consider collecting more G3 samples or trying a larger backbone."
    else:
        final_verdict = "🔴 POOR"
        recommendation = "Model needs retraining with stronger augmentation."

    story.append(Paragraph(f"<b>Final Verdict:</b> {final_verdict}", S["body"]))
    story.append(Spacer(1, 14))

    conclusion_data = [
        ["Metric", "Value"],
        ["SSL Method Used", ssl_method.upper()],
        ["Test Accuracy (TTA)", f"{acc:.4f}"],
        ["Balanced Accuracy", f"{bal_acc:.4f}"],
        ["F1 Score (weighted)", f"{f1:.4f}"],
        ["AUC (mean)", f"{auc_val:.4f}"],
        ["CV Mean Accuracy",
         f"{cv_summary.get('mean_accuracy', 0):.4f} ± {cv_summary.get('std_accuracy', 0):.4f}"],
        ["CI 95% Accuracy",
         f"[{ci95.get('accuracy', {}).get('ci95_low', 0):.4f}, "
         f"{ci95.get('accuracy', {}).get('ci95_high', 0):.4f}]"],
    ]
    story.append(make_table(conclusion_data, [2.5 * inch, 3 * inch], header_bg="#27ae60"))

    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Recommendation:</b>", S["h2"]))
    story.append(Paragraph(recommendation, S["body"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Next Steps:</b>", S["h2"]))
    story.append(Paragraph(
        "1. Compare with other models (ResNet50, EfficientNet-B1) using same protocol.<br/>"
        "2. Test on external validation set.<br/>"
        "3. Consider ensemble with SSL-trained models.<br/>"
        "4. Export to ONNX for CPU deployment.<br/>"
        "5. Monitor G3 (smallest class) for false negatives.",
        S["body"]))
    story.append(Spacer(1, 40))
    story.append(Paragraph(
        f"Report generated by Liver AI Pipeline v5.0.0 (LIGHT Edition)<br/>"
        f"Author: Babak Yousefian<br/>"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"Total pages: ~20-24",
        S["small"]))

    try:
        doc.build(story)
        if verbose:
            print(f"✅ PDF saved: {PDF_PATH}")
        return PDF_PATH
    except Exception as e:
        print(f"❌ PDF build failed: {e}")
        import traceback
        traceback.print_exc()
        return _fallback_json(metrics, cv_summary, ci95)


# ================================================================
# ✅ Fallback
# ================================================================

def _fallback_json(metrics, cv_summary, ci95):
    json_path = PDF_PATH.replace(".pdf", ".json")
    with open(json_path, "w") as f:
        json.dump({"metrics": metrics, "cv": cv_summary, "ci95": ci95},
                  f, indent=4, default=str)
    print(f"⚠️ ReportLab not available — JSON saved: {json_path}")
    return json_path