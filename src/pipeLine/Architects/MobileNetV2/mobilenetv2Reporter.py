# src/pipeLine/mobilenetv2Reporter.py
# -*- coding: utf-8 -*-

"""
MobileNetV2 Fine-Tune Reporter — 24-Page Edition
==================================================
Mirrors alexnetReporter.py (24 pages) with SSL methods comparison.

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
Version: 2.0.0 (24-page edition)
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
REPORTER_ROOT = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "mobilenetv2_finetune")
PDF_PATH = os.path.join(REPORTER_ROOT, "mobilenetv2_finetune_report.pdf")
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

        colors_bar = []
        for m in methods:
            if m == best_method:
                colors_bar.append("#2ecc71")
            else:
                colors_bar.append("#95a5a6")

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
        plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        plot_paths["bar"] = path
        if verbose:
            print(f"  ✅ 08_ssl_comparison_bar")

    except Exception as e:
        if verbose:
            print(f"  ⚠️ SSL comparison plot failed: {e}")
        plt.close("all")

    # Radar
    try:
        from math import pi
        all_methods = ssl_comparison.get("all_methods", {})
        if len(all_methods) >= 2:
            methods = list(all_methods.keys())
            lp_accs = [all_methods[m].get("lp_accuracy", 0) for m in methods]
            lp_f1s = [all_methods[m].get("lp_f1", 0) for m in methods]

            categories = ["LP Accuracy", "LP F1"]
            N = len(categories)

            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            colors_radar = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
            for i, m in enumerate(methods):
                values = [lp_accs[i], lp_f1s[i]]
                values += values[:1]
                ax.plot(angles, values, "o-", linewidth=2,
                        label=m.upper(),
                        color=colors_radar[i % len(colors_radar)])
                ax.fill(angles, values, alpha=0.15,
                        color=colors_radar[i % len(colors_radar)])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=13, fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.set_title("SSL Methods — Radar Comparison",
                         fontsize=14, fontweight="bold", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(plots_dir, "08b_ssl_comparison_radar.png")
            plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["radar"] = path
            if verbose:
                print(f"  ✅ 08b_ssl_comparison_radar")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Radar plot failed: {e}")
        plt.close("all")

    return plot_paths


# ================================================================
# ✅ Main API
# ================================================================

def generate_mobilenetv2_report(
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
        print("📄 Generating 24-page MobileNetV2 Fine-Tune Report")
        print("=" * 80)

    # ✅ Plots
    if verbose:
        print("\n📊 Generating plots (7 figures)...")
    plot_paths = _generate_plots(metrics, history, cv_summary, ci95, PLOTS_DIR, verbose)

    # ✅ SSL comparison
    ssl_plot_paths = {}
    if ssl_comparison and ssl_comparison.get("available"):
        if verbose:
            print("\n📊 Generating SSL methods comparison plot...")
        ssl_plot_paths = _generate_ssl_comparison_plot(ssl_comparison, PLOTS_DIR, verbose)

    # ✅ Interpretability
    interp_paths = {}
    if generate_interpretability_flag:
        if verbose:
            print("\n🔬 Generating interpretability (Captum, LIME, SHAP)...")
        interp_paths = _generate_interpretability(
            final_model, image_paths, labels, device, verbose
        )

    # ✅ Architecture
    if verbose:
        print("\n🏗️  Generating architecture (ONNX + HTML)...")
    arch_paths = _generate_architecture(final_model, verbose)

    # ✅ Build PDF
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
# ✅ Plot Generation (7 figures)
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
            plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["loss"] = path
            if verbose:
                print("  ✅ 01_loss_history")
    except Exception:
        plt.close("all")

    # 2. LR
    try:
        if history and ("learning_rates" in history or "lr" in history):
            lrs = history.get("learning_rates", history.get("lr", []))
            if lrs:
                plt.figure(figsize=(12, 6))
                plt.plot(lrs, linewidth=2.5, color="#9b59b6")
                plt.xlabel("Epoch", fontsize=13, fontweight="bold")
                plt.ylabel("LR", fontsize=13, fontweight="bold")
                plt.title("Learning Rate Schedule", fontsize=15, fontweight="bold")
                plt.yscale("log"); plt.grid(alpha=0.3)
                plt.tight_layout()
                path = os.path.join(plots_dir, "02_learning_rate.png")
                plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
                plt.close()
                plot_paths["lr"] = path
    except Exception:
        plt.close("all")

    # 3. Metrics bar
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
            plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["metrics"] = path
    except Exception:
        plt.close("all")

    # 4. Confusion matrix
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
            plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["cm"] = path
    except Exception:
        plt.close("all")

    # 5. Per-class recall
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
            plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["per_class"] = path
    except Exception:
        plt.close("all")

    # 6. ROC
    try:
        if metrics and "roc_curves" in metrics and metrics["roc_curves"]:
            roc_data = metrics["roc_curves"]
            plt.figure(figsize=(10, 8))
            colors_roc = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
            for i, cname in enumerate(CLASS_NAMES):
                key = str(i) if str(i) in roc_data else i
                if key in roc_data:
                    d = roc_data[key]
                    plt.plot(d["fpr"], d["tpr"], label=f"{cname} (AUC={d['auc']:.3f})",
                             linewidth=2.5, color=colors_roc[i % len(colors_roc)])
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("FPR", fontsize=13, fontweight="bold")
            plt.ylabel("TPR", fontsize=13, fontweight="bold")
            plt.title("ROC Curves", fontsize=15, fontweight="bold")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            path = os.path.join(plots_dir, "06_roc_curves.png")
            plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["roc"] = path
    except Exception:
        plt.close("all")

    # 7. CV folds
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
            plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()
            plot_paths["cv"] = path
    except Exception:
        plt.close("all")

    return plot_paths


# ================================================================
# ✅ Interpretability
# ================================================================

def _generate_interpretability(model, image_paths, labels, device, verbose):
    """Generate Grad-CAM, LIME, SHAP."""
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    interp_paths = {}

    class _DS(Dataset):
        def __init__(self, paths, labels, tf):
            self.paths = paths
            self.labels = labels
            self.tf = tf

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = cv2.imread(self.paths[i])
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
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

    # Captum
    try:
        from src.reporter.captum_reporter import generate_interpretability_report
        n = min(16, len(image_paths))
        ds = _DS(image_paths[:n], labels[:n], tf)
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

    # LIME
    try:
        from src.reporter.lime_reporter import explain_predictions_lime
        n = min(5, len(image_paths))
        ds = _DS(image_paths[:n], labels[:n], tf)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        lime_result = explain_predictions_lime(
            model, dl, INTERP_DIR, class_names=CLASS_NAMES,
            device=device, num_samples=5,
        )
        interp_paths["lime"] = lime_result.get("lime_path")
        if verbose:
            print("  ✅ LIME")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ LIME failed: {e}")

    # SHAP
    try:
        from src.reporter.shap_reporter import generate_shap_explanation
        n = min(20, len(image_paths))
        ds = _DS(image_paths[:n], labels[:n], tf)
        dl = DataLoader(ds, batch_size=4, shuffle=False)
        shap_result = generate_shap_explanation(
            model, dl, INTERP_DIR, device=device,
            num_background=8, num_explain=4, class_names=CLASS_NAMES,
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
            model_name="MobileNetV2",
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
        title="MobileNetV2 Fine-Tune Report",
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

    # ─────────────────────────────────────────────
    # PAGE 1: TITLE
    # ─────────────────────────────────────────────
    story.append(Spacer(1, 80))
    story.append(Paragraph("MobileNetV2 Fine-Tune Report", S["title"]))
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
    story.append(Paragraph("Liver AI Pipeline · Task 5 · Version 5.0.0", S["center"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 2: EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────
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
        ["Bootstrap Samples", str(config.get("n_bootstrap", 2000))],
        ["TTA Augmentations", str(tta_info.get("n_aug", 32))],
    ]
    story.append(make_table(summary_data, [3 * inch, 3 * inch]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "This report presents the complete fine-tune pipeline for MobileNetV2 on the "
        "liver ultrasound dataset (NL/G1/G2/G3). The encoder was initialized from an SSL "
        "checkpoint and refined via supervised fine-tuning with Focal Loss + class weights. "
        "Evaluation includes TTA, 5-Fold CV, and Bootstrap CI (95%). Interpretability is "
        "provided via Captum, LIME, and SHAP.",
        S["body"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 3: TRAINING CONFIG
    # ─────────────────────────────────────────────
    story.append(Paragraph("2. Training Configuration", S["h1"]))
    story.append(Spacer(1, 12))

    cfg = config or {}
    config_data = [
        ["Parameter", "Value"],
        ["SSL Method", ssl_method.upper()],
        ["SSL Checkpoint", os.path.basename(ssl_checkpoint_path)],
        ["Architecture", "MobileNetV2 (width_mult=1.0)"],
        ["Input Size", f"{cfg.get('input_size', 224)}×{cfg.get('input_size', 224)}"],
        ["FT Epochs (final)", str(cfg.get("epochs_final", 100))],
        ["FT Epochs (CV fold)", str(cfg.get("epochs_cv", 30))],
        ["Learning Rate (final)", f"{cfg.get('lr_final', 1e-4):.2e}"],
        ["Learning Rate (CV)", f"{cfg.get('lr_cv', 1e-4):.2e}"],
        ["Batch Size", str(cfg.get("batch_size", 32))],
        ["Early Stop Patience (final)", str(cfg.get("patience_final", 30))],
        ["Early Stop Patience (CV)", str(cfg.get("patience_cv", 15))],
        ["Optimizer", "AdamW"],
        ["Loss Function", "FocalLoss (γ=2.0) + class weights"],
        ["Weight Decay", "2e-3"],
        ["Sampler", "WeightedRandomSampler (sqrt)"],
        ["Augmentation", "RandomResizedCrop, Flip, Rotate, ColorJitter"],
        ["TTA Augmentations", str(tta_info.get("n_aug", 32))],
        ["CV Folds", str(cfg.get("cv_folds", 5))],
        ["Bootstrap Samples", str(cfg.get("n_bootstrap", 2000))],
        ["Device", cfg.get("device", "cuda")],
    ]
    story.append(make_table(config_data, [3 * inch, 3 * inch], header_bg="#34495e"))
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 4: DATASET
    # ─────────────────────────────────────────────
    story.append(Paragraph("3. Dataset Split & Class Distribution", S["h1"]))
    story.append(Paragraph(
        "The labeled dataset was split into 80% train / 10% validation / 10% test "
        "(stratified by class). The 5-Fold CV uses the full dataset.",
        S["body"]))
    story.append(Spacer(1, 12))

    ds = data_split or {}
    split_data = [
        ["Split", "Samples", "Percentage"],
        ["Train", str(ds.get("train", "N/A")),
         f"{100 * ds.get('train', 0) / max(ds.get('total', 1), 1):.1f}%"],
        ["Validation", str(ds.get("val", "N/A")),
         f"{100 * ds.get('val', 0) / max(ds.get('total', 1), 1):.1f}%"],
        ["Test", str(ds.get("test", "N/A")),
         f"{100 * ds.get('test', 0) / max(ds.get('total', 1), 1):.1f}%"],
        ["TOTAL", str(ds.get("total", "N/A")), "100.0%"],
    ]
    story.append(make_table(split_data, [2 * inch, 2 * inch, 2 * inch], header_bg="#27ae60"))

    story.append(Spacer(1, 20))
    story.append(Paragraph("3.1 Class Imbalance", S["h2"]))
    story.append(Paragraph(
        "The liver dataset is highly imbalanced (approx. 22:1 ratio NL:G3). "
        "WeightedRandomSampler with sqrt-based weights + Focal Loss with class weights "
        "was used to mitigate this. Balanced Accuracy, Macro F1, and per-class recall "
        "are the primary evaluation metrics.",
        S["body"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 5: LOSS HISTORY
    # ─────────────────────────────────────────────
    story.append(Paragraph("4. Loss History", S["h1"]))
    safe_img(plot_paths.get("loss"), "4.1 Train vs Validation Loss")
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 6: LR SCHEDULE
    # ─────────────────────────────────────────────
    story.append(Paragraph("5. Learning Rate Schedule", S["h1"]))
    safe_img(plot_paths.get("lr"), "5.1 LR over Epochs")
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 7: METRICS BAR
    # ─────────────────────────────────────────────
    story.append(Paragraph("6. Test Metrics (with TTA)", S["h1"]))
    safe_img(plot_paths.get("metrics"), "6.1 Metrics Bar Chart")
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 8: CONFUSION MATRIX
    # ─────────────────────────────────────────────
    story.append(Paragraph("7. Confusion Matrix", S["h1"]))
    safe_img(plot_paths.get("cm"), "7.1 Confusion Matrix", w=6 * inch, h=5 * inch)
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 9: PER-CLASS PERFORMANCE
    # ─────────────────────────────────────────────
    story.append(Paragraph("8. Per-Class Performance", S["h1"]))
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

    # ─────────────────────────────────────────────
    # PAGE 10: PER-CLASS RECALL
    # ─────────────────────────────────────────────
    story.append(Paragraph("9. Per-Class Recall", S["h1"]))
    safe_img(plot_paths.get("per_class"), "9.1 Recall per Class")
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 11: ROC CURVES
    # ─────────────────────────────────────────────
    story.append(Paragraph("10. ROC Curves", S["h1"]))
    safe_img(plot_paths.get("roc"), "10.1 ROC Curves per Class", w=6 * inch, h=5 * inch)
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 12: AUC SCORES
    # ─────────────────────────────────────────────
    story.append(Paragraph("11. AUC Scores", S["h1"]))
    if metrics and "roc_auc" in metrics and metrics["roc_auc"]:
        roc_auc = metrics["roc_auc"]
        auc_data = [["Class", "AUC", "Interpretation"]]
        for cname in CLASS_NAMES:
            idx = CLASS_NAMES.index(cname)
            key = str(idx) if str(idx) in roc_auc else idx
            if key in roc_auc:
                v = roc_auc[key]
                if v >= 0.9:
                    interp = "Excellent"
                elif v >= 0.8:
                    interp = "Good"
                elif v >= 0.7:
                    interp = "Fair"
                elif v >= 0.6:
                    interp = "Poor"
                else:
                    interp = "Fail"
                auc_data.append([cname, f"{v:.4f}", interp])
        if len(auc_data) > 1:
            mean_auc = np.mean([float(r[1]) for r in auc_data[1:]])
            auc_data.append(["Mean", f"{mean_auc:.4f}", ""])
        story.append(make_table(auc_data, [1.5 * inch, 1.5 * inch, 2 * inch], header_bg="#16a085"))
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 13: CV FOLD ACCURACIES
    # ─────────────────────────────────────────────
    story.append(Paragraph("12. CV — Accuracy per Fold", S["h1"]))
    safe_img(plot_paths.get("cv"), "12.1 Accuracy per Fold")
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 14: CV METRICS COMPARISON
    # ─────────────────────────────────────────────
    story.append(Paragraph("13. CV — Metrics Comparison", S["h1"]))
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

    # ─────────────────────────────────────────────
    # PAGE 15: BEST FOLD
    # ─────────────────────────────────────────────
    story.append(Paragraph("14. CV — Best Fold", S["h1"]))
    if cv_summary and "fold_results" in cv_summary:
        best_fold = max(cv_summary["fold_results"], key=lambda x: x["accuracy"])
        best_data = [
            ["Metric", "Value"],
            ["Fold #", str(best_fold["fold"])],
            ["Accuracy", f"{best_fold['accuracy']:.4f}"],
            ["F1 (weighted)", f"{best_fold['f1_weighted']:.4f}"],
            ["Balanced Accuracy", f"{best_fold['balanced_accuracy']:.4f}"],
            ["Best Val Acc", f"{best_fold.get('best_val_acc', 0):.4f}"],
        ]
        story.append(make_table(best_data, [2.5 * inch, 2.5 * inch], header_bg="#c0392b"))
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 16: BOOTSTRAP CI
    # ─────────────────────────────────────────────
    story.append(Paragraph("15. Bootstrap 95% CI", S["h1"]))
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

    # ─────────────────────────────────────────────
    # PAGE 17: TTA
    # ─────────────────────────────────────────────
    story.append(Paragraph("16. Test-Time Augmentation (TTA)", S["h1"]))
    story.append(Paragraph(
        f"TTA averages predictions over {tta_info.get('n_aug', 32)} augmented copies "
        f"of each test image. Augmentations include: HFlip, VFlip, Rot90/180/270, "
        f"Brightness±, Contrast±, and transpose.",
        S["body"]))
    story.append(Spacer(1, 12))
    tta_data = [
        ["Parameter", "Value"],
        ["Number of Augmentations", str(tta_info.get("n_aug", 32))],
        ["Augmentation Types", "HFlip, VFlip, Rot90/180/270, Bright±, Contrast±, Transpose"],
        ["Aggregation", "Mean of softmax probabilities"],
        ["Applied To", "Test set evaluation"],
    ]
    story.append(make_table(tta_data, [3 * inch, 3 * inch], header_bg="#2980b9"))
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 18: SSL METHODS COMPARISON
    # ─────────────────────────────────────────────
    story.append(Paragraph("17. SSL Methods Comparison", S["h1"]))
    story.append(Paragraph(
        "Before fine-tuning, three Self-Supervised Learning (SSL) methods were trained on "
        "MobileNetV2: <b>BYOL</b>, <b>SimCLR</b>, and <b>MAE</b>. Each was evaluated with "
        "a <b>Linear Probe</b> on the labeled dataset. The method with the highest LP "
        "accuracy was selected as the starting point for fine-tuning.",
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
                 "17.1 SSL Methods — Accuracy & F1",
                 w=6.5 * inch, h=3.5 * inch)

        if ssl_plot_paths.get("radar"):
            safe_img(ssl_plot_paths.get("radar"),
                     "17.2 SSL Methods — Radar",
                     w=5.5 * inch, h=5.5 * inch)

        story.append(Spacer(1, 14))
        story.append(Paragraph("17.3 Why This Winner?", S["h2"]))
        story.append(Paragraph(
            f"<b>{best_method.upper()}</b> achieved the highest Linear Probe accuracy, "
            f"indicating that its SSL pretraining produced the most discriminative "
            f"features for the liver ultrasound classification task. "
            f"BYOL learns stable representations without negative samples; SimCLR uses "
            f"contrastive learning; MAE leverages masked reconstruction. The winner was "
            f"used as initialization for fine-tuning (see Section 1).",
            S["body"]))
    else:
        story.append(Paragraph(
            "⚠️ SSL methods comparison data not available. This section requires "
            "<b>best_method_summary.json</b> from the SSL selection step (option 24).",
            S["body"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 19: CAPTUM
    # ─────────────────────────────────────────────
    story.append(Paragraph("18. Captum Interpretability", S["h1"]))
    safe_img(interp_paths.get("gradcam_path"), "18.1 Grad-CAM")
    safe_img(interp_paths.get("ig_path"), "18.2 Integrated Gradients")
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 20: LIME
    # ─────────────────────────────────────────────
    story.append(Paragraph("19. LIME Explanations", S["h1"]))
    safe_img(interp_paths.get("lime"), "19.1 LIME Explanation", w=6 * inch, h=4 * inch)
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 21: SHAP
    # ─────────────────────────────────────────────
    story.append(Paragraph("20. SHAP Summary", S["h1"]))
    safe_img(interp_paths.get("shap"), "20.1 SHAP Summary", w=6 * inch, h=4 * inch)
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 22: ARCHITECTURE
    # ─────────────────────────────────────────────
    story.append(Paragraph("21. Model Architecture (Netron)", S["h1"]))
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
    story.append(Paragraph("21.1 Model Summary", S["h2"]))
    try:
        num_params = sum(p.numel() for p in final_model.parameters())
        num_train = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
        param_data = [
            ["Property", "Value"],
            ["Architecture", "MobileNetV2 (width_mult=1.0)"],
            ["Input Size", "224 × 224 × 3"],
            ["Total Parameters", f"{num_params:,}"],
            ["Trainable Parameters", f"{num_train:,}"],
            ["Feature Dimension", "1280"],
            ["Normalization", "BatchNorm2d"],
            ["Activation", "ReLU6"],
            ["Output Classes", "4 (NL, G1, G2, G3)"],
        ]
        story.append(make_table(param_data, [2.5 * inch, 3 * inch], header_bg="#34495e"))
    except Exception as e:
        story.append(Paragraph(f"⚠️ {e}", S["small"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 23: PIPELINE SUMMARY
    # ─────────────────────────────────────────────
    story.append(Paragraph("22. Pipeline Summary", S["h1"]))
    pipeline_data = [
        ["Stage", "Status", "Details"],
        ["SSL Pretraining", "✅",
         f"{ssl_method.upper()} — {os.path.basename(ssl_checkpoint_path)}"],
        ["Fine-Tune", "✅", f"{config.get('epochs_final', 100)} epochs"],
        ["5-Fold CV", "✅",
         f"{len(cv_summary.get('fold_results', []))} folds, "
         f"mean acc = {cv_summary.get('mean_accuracy', 0):.4f}"],
        ["Bootstrap CI", "✅", f"{config.get('n_bootstrap', 2000)} samples"],
        ["TTA", "✅", f"{tta_info.get('n_aug', 32)} augmentations"],
        ["Test Eval", "✅", f"Acc = {metrics.get('accuracy', 0):.4f}"],
        ["Captum", "✅" if interp_paths.get("gradcam_path") else "❌", "Grad-CAM + IG"],
        ["LIME", "✅" if interp_paths.get("lime") else "❌", "5 samples"],
        ["SHAP", "✅" if interp_paths.get("shap") else "❌", "10 samples"],
        ["Netron", "✅" if arch_paths.get("onnx_path") else "❌", "ONNX + HTML"],
        ["Final Model", "✅", "mobilenetv2_finetuned_final.pth"],
        ["PDF Report", "✅", "24 pages"],
    ]
    story.append(make_table(pipeline_data,
                            [1.8 * inch, 0.8 * inch, 3.4 * inch],
                            header_bg="#2c3e50"))
    story.append(PageBreak())

    # ─────────────────────────────────────────────
    # PAGE 24: CONCLUSION
    # ─────────────────────────────────────────────
    story.append(Paragraph("23. Conclusion & Recommendation", S["h1"]))
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
        f"Report generated by Liver AI Pipeline v5.0.0<br/>"
        f"Author: Babak Yousefian<br/>"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"Total pages: 24",
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