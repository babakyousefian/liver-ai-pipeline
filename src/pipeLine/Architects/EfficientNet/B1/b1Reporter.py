# src/pipeLine/b1Reporter.py
# -*- coding: utf-8 -*-

"""
EfficientNet-B1 Comprehensive Reporter (ULTIMATE 23-PAGE EDITION)

This module generates the most comprehensive 23-page PDF report for
EfficientNet-B1 Fine-Tuned models:

Page 1:  Info (Title, Config)
Page 2:  Details (Transfer info, Data info)
Page 3:  Tables (Metrics + Per-class)
Page 4:  Accuracy - Precision - Recall - F1 (Bar chart)
Page 5:  AUC - ROC curves
Page 6:  Confusion Matrix
Page 7:  Diagrams (Training flow)
Page 8:  Bar chart (Per-class metrics)
Page 9:  Line Bar (Accuracy vs Epoch)
Page 10: Pie chart (Class distribution)
Page 11: Graphs (Loss + LR curves)
Page 12: ONNX (Architecture)
Page 13: SHAP Reporter
Page 14: Captum Reporter (Grad-CAM + IG)
Page 15: LIME Reporter
Page 16: Netron Reporter
Page 17: CV Results (5-Fold)
Page 18: CI Results (95% Bootstrap)
Page 19: Train/Test/Val AUC
Page 20: Grad-CAM Reporter
Page 21: Loss History
Page 22: Summary
Page 23: Finished Page

Author: Babak Yousefian
Version: 8.0.0 (COLORFUL 23-PAGE EDITION)
"""

import json
import os
import warnings
from datetime import datetime
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
from matplotlib.patches import FancyBboxPatch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

# ================================================================
# ✅ COLOR PALETTE (Beautiful, Professional)
# ================================================================

class Colors:
    """Beautiful color palette for all plots and reports."""
    # Main brand colors
    PRIMARY = "#2C3E50"      # Dark navy blue
    SECONDARY = "#3498DB"    # Bright blue
    ACCENT = "#E74C3C"       # Red
    SUCCESS = "#27AE60"      # Green
    WARNING = "#F39C12"      # Orange
    PURPLE = "#9B59B6"       # Purple
    TEAL = "#1ABC9C"         # Teal
    PINK = "#E91E63"         # Pink
    DARK = "#1A252F"         # Very dark

    # Class colors (for NL, G1, G2, G3)
    CLASS_COLORS = ["#3498DB", "#27AE60", "#F39C12", "#E74C3C"]
    #              NL=blue    G1=green   G2=orange  G3=red

    # Gradient
    GRADIENTS = [
        "#3498DB",  # blue
        "#9B59B6",  # purple
        "#E91E63",  # pink
        "#E74C3C",  # red
        "#F39C12",  # orange
        "#F1C40F",  # yellow
        "#2ECC71",  # green
        "#1ABC9C",  # teal
    ]

    # Background
    BG_LIGHT = "#F8F9FA"
    BG_WHITE = "#FFFFFF"
    BG_PAPER = "#FEFEFE"


# ================================================================
# ✅ Paths
# ================================================================

PROJECT_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
SELF_SUPERVISED_ROOT = os.path.join(PROJECT_ROOT, "finalReports", "selfSupervisedLearning")
REPORTER_ROOT = os.path.join(SELF_SUPERVISED_ROOT, "SSL_Reporter", "b1_finetune")

ARCH_DIR = os.path.join(REPORTER_ROOT, "architecture")
CV_DIR = os.path.join(REPORTER_ROOT, "cv")
INTERP_DIR = os.path.join(REPORTER_ROOT, "interpretability")
PLOTS_DIR = os.path.join(REPORTER_ROOT, "plots")
FINETUNE_DIR = os.path.join(REPORTER_ROOT, "finetune_plots")
PDF_PATH = os.path.join(REPORTER_ROOT, "b1_finetune_report.pdf")

METRICS_JSON = os.path.join(REPORTER_ROOT, "metrics.json")
FINAL_MODEL_PATH = os.path.join(REPORTER_ROOT, "b1_finetuned_final.pth")

# ================================================================
# ✅ Constants
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE_B1 = 240

# ================================================================
# ✅ Ensure Dirs
# ================================================================

def ensure_reporter_dirs():
    dirs = [REPORTER_ROOT, ARCH_DIR, CV_DIR, INTERP_DIR, PLOTS_DIR, FINETUNE_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ================================================================
# ✅ Setup matplotlib style (beautiful defaults)
# ================================================================

def setup_style():
    """Setup beautiful matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": Colors.BG_WHITE,
        "axes.facecolor": Colors.BG_LIGHT,
        "axes.edgecolor": Colors.PRIMARY,
        "axes.labelcolor": Colors.PRIMARY,
        "axes.titlecolor": Colors.PRIMARY,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "semibold",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "xtick.color": Colors.PRIMARY,
        "ytick.color": Colors.PRIMARY,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.facecolor": Colors.BG_WHITE,
        "legend.edgecolor": Colors.PRIMARY,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": Colors.BG_WHITE,
    })


# ================================================================
# ✅ Style Helper: Decorate Axes
# ================================================================

def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply beautiful style to an axis."""
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", color=Colors.PRIMARY, pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight="semibold", color=Colors.PRIMARY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight="semibold", color=Colors.PRIMARY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(Colors.PRIMARY)
    ax.spines["bottom"].set_color(Colors.PRIMARY)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


# ================================================================
# ✅ Generate All Plots
# ================================================================

def generate_loss_plots(history: Dict, save_dir: str) -> Dict[str, str]:
    """Generate loss + LR + F1 plots."""
    paths = {}

    # 1. Train + Val Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], color=Colors.SECONDARY, linewidth=2.5,
            marker="o", markersize=4, label="Train Loss", zorder=3)
    ax.plot(epochs, history["val_loss"], color=Colors.ACCENT, linewidth=2.5,
            marker="s", markersize=4, label="Val Loss", zorder=3)

    # Highlight min (safe)
    if history.get("val_loss"):
        min_val = min(history["val_loss"])
        min_epoch = history["val_loss"].index(min_val) + 1
        ax.scatter([min_epoch], [min_val], color=Colors.SUCCESS, s=200,
                   marker="*", zorder=5, label=f"Best Val: {min_val:.4f} (epoch {min_epoch})")

    style_axes(ax, "Training & Validation Loss", "Epoch", "Loss")
    ax.legend(loc="upper right", fancybox=True, shadow=True)
    plt.tight_layout()
    p = os.path.join(save_dir, "01_loss_train_val.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    paths["loss_train_val"] = p

    # 2. LR Schedule
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history["learning_rates"], color=Colors.PURPLE,
            linewidth=2.5, marker="o", markersize=4)
    ax.set_yscale("log")
    style_axes(ax, "Learning Rate Schedule", "Epoch", "LR (log scale)")
    plt.tight_layout()
    p = os.path.join(save_dir, "02_lr_schedule.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    paths["lr_schedule"] = p

    # ✅ 3. F1 vs Epoch (NEW - v8.0)
    if history.get("train_f1") and history.get("val_f1"):
        fig, ax = plt.subplots(figsize=(12, 6))
        epochs_f1 = range(1, len(history["train_f1"]) + 1)

        ax.plot(epochs_f1, history["train_f1"], color=Colors.SUCCESS,
                linewidth=2.5, marker="o", markersize=4, label="Train F1")
        ax.plot(epochs_f1, history["val_f1"], color=Colors.PINK,
                linewidth=2.5, marker="s", markersize=4, label="Val F1")

        # Highlight best val F1 (that's what early stopping uses now)
        if history["val_f1"]:
            best_f1 = max(history["val_f1"])
            best_f1_epoch = history["val_f1"].index(best_f1) + 1
            ax.scatter([best_f1_epoch], [best_f1], color=Colors.WARNING, s=200,
                       marker="*", zorder=5, label=f"Best F1: {best_f1:.4f} (epoch {best_f1_epoch})")

        ax.set_ylim(0, 1.05)
        style_axes(ax, "F1-Score vs Epoch (Early Stopping Metric)", "Epoch", "F1-Score")
        ax.legend(loc="lower right", fancybox=True, shadow=True)
        plt.tight_layout()
        p = os.path.join(save_dir, "13_f1_vs_epoch.png")
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close()
        paths["f1_vs_epoch"] = p

    return paths


def generate_metrics_plots(metrics: Dict, save_dir: str) -> Dict[str, str]:
    """Generate metrics plots."""
    paths = {}

    # 1. Main metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ["Accuracy", "Precision", "Recall", "F1-Score", "Balanced Acc"]
    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["balanced_accuracy"],
    ]
    colors = [Colors.SUCCESS, Colors.SECONDARY, Colors.WARNING, Colors.ACCENT, Colors.PURPLE]

    bars = ax.bar(names, values, color=colors, edgecolor="white",
                  linewidth=2, width=0.6, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=Colors.PRIMARY)

    ax.set_ylim(0, 1.10)
    style_axes(ax, "Performance Metrics", "Metric", "Score")
    plt.tight_layout()
    p = os.path.join(save_dir, "03_metrics_bar.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    paths["metrics_bar"] = p

    # 2. Per-class metrics
    fig, ax = plt.subplots(figsize=(12, 7))
    per_class_data = metrics["classification_report"]

    precisions = [per_class_data[c]["precision"] for c in CLASS_NAMES]
    recalls = [per_class_data[c]["recall"] for c in CLASS_NAMES]
    f1s = [per_class_data[c]["f1-score"] for c in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    width = 0.27

    bars1 = ax.bar(x - width, precisions, width, label="Precision",
                   color=Colors.SECONDARY, edgecolor="white", linewidth=1.5, zorder=3)
    bars2 = ax.bar(x, recalls, width, label="Recall",
                   color=Colors.SUCCESS, edgecolor="white", linewidth=1.5, zorder=3)
    bars3 = ax.bar(x + width, f1s, width, label="F1-Score",
                   color=Colors.ACCENT, edgecolor="white", linewidth=1.5, zorder=3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=Colors.PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.10)
    style_axes(ax, "Per-Class Performance", "Class", "Score")
    ax.legend(loc="upper right", fancybox=True, shadow=True)
    plt.tight_layout()
    p = os.path.join(save_dir, "04_per_class_metrics.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    paths["per_class_metrics"] = p

    return paths


def generate_confusion_matrix(metrics: Dict, save_dir: str) -> str:
    """Generate confusion matrix heatmap."""
    cm = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={"label": "Count"},
                linewidths=2, linecolor="white",
                annot_kws={"fontsize": 14, "fontweight": "bold"},
                ax=ax)

    style_axes(ax, "Confusion Matrix",
               "Predicted Label", "True Label")
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold", rotation=0)

    plt.tight_layout()
    p = os.path.join(save_dir, "05_confusion_matrix.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    return p


def generate_roc_curves(metrics: Dict, save_dir: str) -> str:
    """Generate ROC curves."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(NUM_CLASSES):
        if i in metrics["fpr"] and i in metrics["tpr"]:
            fpr = metrics["fpr"][i]
            tpr = metrics["tpr"][i]
            roc_auc = metrics["roc_auc"][i]
            ax.plot(fpr, tpr, color=Colors.CLASS_COLORS[i],
                    lw=3, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})",
                    zorder=3)

    ax.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.6, label="Random (AUC = 0.5)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    style_axes(ax, "ROC Curves (One-vs-Rest)", "False Positive Rate", "True Positive Rate")
    ax.legend(loc="lower right", fancybox=True, shadow=True)

    # Add AUC mean annotation
    ax.text(0.55, 0.15, f"Mean AUC = {metrics['roc_auc_mean']:.4f}",
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5",
                     facecolor=Colors.WARNING, alpha=0.3,
                     edgecolor=Colors.WARNING))

    plt.tight_layout()
    p = os.path.join(save_dir, "06_roc_curves.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    return p


def generate_class_distribution_pie(metrics: Dict, save_dir: str) -> str:
    """Generate class distribution pie chart."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sizes = []
    for c in CLASS_NAMES:
        if c in metrics["classification_report"]:
            sizes.append(metrics["classification_report"][c]["support"])

    if not sizes:
        plt.close()
        return None

    explode = [0.05] * len(CLASS_NAMES)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=CLASS_NAMES, autopct="%1.1f%%",
        colors=Colors.CLASS_COLORS, explode=explode,
        shadow=True, startangle=90,
        textprops={"fontsize": 13, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(14)
        autotext.set_fontweight("bold")

    ax.set_title("Class Distribution", fontsize=16, fontweight="bold",
                 color=Colors.PRIMARY, pad=20)

    plt.tight_layout()
    p = os.path.join(save_dir, "07_class_distribution_pie.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    return p


def generate_accuracy_vs_epoch(history: Dict, save_dir: str) -> str:
    """Generate accuracy vs epoch plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(history["train_acc"]) + 1)

    ax.plot(epochs, history["train_acc"], color=Colors.SECONDARY,
            linewidth=2.5, marker="o", markersize=4, label="Train Accuracy")
    ax.plot(epochs, history["val_acc"], color=Colors.ACCENT,
            linewidth=2.5, marker="s", markersize=4, label="Val Accuracy")

    ax.set_ylim(0, 1.05)
    style_axes(ax, "Accuracy vs Epoch", "Epoch", "Accuracy")
    ax.legend(loc="lower right", fancybox=True, shadow=True)

    plt.tight_layout()
    p = os.path.join(save_dir, "08_accuracy_vs_epoch.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    return p


def generate_auc_vs_epoch(history: Dict, save_dir: str) -> str:
    """Generate AUC vs epoch plot (train/val)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(history["train_auc"]) + 1)

    ax.plot(epochs, history["train_auc"], color=Colors.SUCCESS,
            linewidth=2.5, marker="o", markersize=4, label="Train AUC")
    ax.plot(epochs, history["val_auc"], color=Colors.PURPLE,
            linewidth=2.5, marker="s", markersize=4, label="Val AUC")

    ax.set_ylim(0, 1.05)
    style_axes(ax, "AUC vs Epoch", "Epoch", "AUC")
    ax.legend(loc="lower right", fancybox=True, shadow=True)

    plt.tight_layout()
    p = os.path.join(save_dir, "09_auc_vs_epoch.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    return p


def generate_cv_bars(cv_summary: Dict, save_dir: str) -> str:
    """Generate 5-fold CV bars."""
    fig, ax = plt.subplots(figsize=(12, 6))

    folds = [f"Fold {r['fold']}" for r in cv_summary["fold_results"]]
    accs = [r["accuracy"] for r in cv_summary["fold_results"]]
    f1s = [r["f1_score"] for r in cv_summary["fold_results"]]
    aucs = [r["roc_auc_mean"] for r in cv_summary["fold_results"]]

    x = np.arange(len(folds))
    width = 0.27

    bars1 = ax.bar(x - width, accs, width, label="Accuracy",
                   color=Colors.SECONDARY, edgecolor="white", linewidth=1.5, zorder=3)
    bars2 = ax.bar(x, f1s, width, label="F1-Score",
                   color=Colors.SUCCESS, edgecolor="white", linewidth=1.5, zorder=3)
    bars3 = ax.bar(x + width, aucs, width, label="AUC",
                   color=Colors.ACCENT, edgecolor="white", linewidth=1.5, zorder=3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=Colors.PRIMARY)

    # Mean line
    ax.axhline(y=cv_summary["mean_accuracy"], color=Colors.WARNING,
               linestyle="--", linewidth=2, alpha=0.7,
               label=f"Mean Acc = {cv_summary['mean_accuracy']:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.10)
    style_axes(ax, "5-Fold Cross-Validation", "Fold", "Score")
    ax.legend(loc="lower right", fancybox=True, shadow=True)

    plt.tight_layout()
    p = os.path.join(save_dir, "10_cv_folds.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    return p


def generate_ci_bars(ci95: Dict, save_dir: str) -> str:
    """Generate 95% CI bars."""
    fig, ax = plt.subplots(figsize=(12, 7))

    metrics_to_plot = ["accuracy", "f1_score", "precision", "recall"]
    labels = ["Accuracy", "F1-Score", "Precision", "Recall"]

    for i, metric in enumerate(metrics_to_plot):
        ci = ci95[metric]
        mean = ci["mean"]
        low = ci["ci95_low"]
        high = ci["ci95_high"]

        color = Colors.CLASS_COLORS[i % len(Colors.CLASS_COLORS)]
        y_pos = len(metrics_to_plot) - i - 1

        # Draw horizontal bar from low to high
        ax.plot([low, high], [y_pos, y_pos], color=color,
                linewidth=8, alpha=0.4, zorder=2)
        # Mean marker
        ax.scatter([mean], [y_pos], color=color, s=200,
                   marker="D", zorder=4, edgecolor="white", linewidth=2)

        # Text
        ax.text(high + 0.005, y_pos,
                f"  {mean:.4f} [{low:.4f}, {high:.4f}]",
                va="center", fontsize=11, fontweight="bold", color=Colors.PRIMARY)

    ax.set_yticks(range(len(metrics_to_plot)))
    ax.set_yticklabels(labels[::-1], fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.15)
    style_axes(ax, "95% Confidence Interval (Bootstrap, 1000 samples)",
               "Score", "")

    plt.tight_layout()
    p = os.path.join(save_dir, "11_ci95_bars.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    return p


def generate_flow_diagram(save_dir: str) -> str:
    """Generate training flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    # Title
    ax.text(7, 7.5, "EfficientNet-B1 Fine-Tune Pipeline Flow",
            ha="center", fontsize=18, fontweight="bold", color=Colors.PRIMARY)

    # Boxes
    boxes = [
        (1.5, 6, "Transferred Model\n(B0 SSL → B1)", Colors.SECONDARY),
        (5, 6, "Labeled Data\n(labels.csv + cropped/)", Colors.SUCCESS),
        (8.5, 6, "Data Split\n(80/10/10)", Colors.WARNING),
        (11.5, 6, "Fine-Tune\n(AdamW, Cosine)", Colors.ACCENT),

        (11.5, 4, "5-Fold CV\n(Stratified)", Colors.PURPLE),
        (8.5, 4, "Bootstrap CI\n(95%, 1000x)", Colors.TEAL),
        (5, 4, "Final Model\n(All data)", Colors.PINK),
        (1.5, 4, "Evaluation\n(Test set)", Colors.DARK),

        (2.5, 2, "Metrics\n(Acc, F1, AUC)", Colors.SECONDARY),
        (6, 2, "Interpretability\n(Grad-CAM, IG, SHAP, LIME)", Colors.SUCCESS),
        (10, 2, "Report\n(23-page PDF)", Colors.WARNING),
    ]

    for x, y, text, color in boxes:
        bbox = FancyBboxPatch((x - 1.2, y - 0.35), 2.4, 0.7,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="white",
                              linewidth=2, alpha=0.85, zorder=3)
        ax.add_patch(bbox)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4)

    # Arrows
    arrows = [
        (2.7, 6, 3.8, 6),
        (6.2, 6, 7.3, 6),
        (9.7, 6, 10.3, 6),
        (11.5, 5.65, 11.5, 4.35),
        (10.3, 4, 9.7, 4),
        (7.3, 4, 6.2, 4),
        (3.8, 4, 2.7, 4),
        (2.5, 3.65, 2.5, 2.35),
        (3.7, 2, 4.8, 2),
        (7.2, 2, 8.8, 2),
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=Colors.PRIMARY,
                                    linewidth=2.5, connectionstyle="arc3,rad=0.1"))

    plt.tight_layout()
    p = os.path.join(save_dir, "12_flow_diagram.png")
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor=Colors.BG_WHITE)
    plt.close()
    return p


def generate_all_plots(
    history: Dict,
    metrics: Dict,
    cv_summary: Dict,
    ci95: Dict,
) -> Dict[str, str]:
    """Generate all plots."""
    ensure_reporter_dirs()
    setup_style()

    all_paths = {}

    print("\n🎨 Generating all plots...")

    print("   📊 Loss plots...")
    all_paths.update(generate_loss_plots(history, PLOTS_DIR))

    print("   📊 Metrics plots...")
    all_paths.update(generate_metrics_plots(metrics, PLOTS_DIR))

    print("   📊 Confusion matrix...")
    all_paths["confusion_matrix"] = generate_confusion_matrix(metrics, PLOTS_DIR)

    print("   📊 ROC curves...")
    all_paths["roc_curves"] = generate_roc_curves(metrics, PLOTS_DIR)

    print("   📊 Class distribution pie...")
    all_paths["class_pie"] = generate_class_distribution_pie(metrics, PLOTS_DIR)

    print("   📊 Accuracy vs Epoch...")
    all_paths["acc_vs_epoch"] = generate_accuracy_vs_epoch(history, PLOTS_DIR)

    print("   📊 AUC vs Epoch...")
    all_paths["auc_vs_epoch"] = generate_auc_vs_epoch(history, PLOTS_DIR)

    print("   📊 CV bars...")
    all_paths["cv_bars"] = generate_cv_bars(cv_summary, PLOTS_DIR)

    print("   📊 CI 95% bars...")
    all_paths["ci_bars"] = generate_ci_bars(ci95, PLOTS_DIR)

    print("   📊 Flow diagram...")
    all_paths["flow_diagram"] = generate_flow_diagram(PLOTS_DIR)

    # ✅ v8.0: F1 plot is generated inside generate_loss_plots already
    # It's stored in all_paths via .update() above

    print(f"✅ Generated {len(all_paths)} plots")

    return all_paths


# ================================================================
# ✅ Generate Interpretability
# ================================================================

def generate_interpretability(
    model: nn.Module,
    image_paths: List[str],
    labels: List[int],
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, str]:
    """Generate all interpretability plots (SHAP, Captum, LIME, Netron)."""
    ensure_reporter_dirs()
    results = {}

    # Build a small eval loader
    from src.pipeLine.Architects.EfficientNet.B1.b1FineTune import LabeledDataset

    dataset = LabeledDataset(image_paths[:300], labels[:300], INPUT_SIZE_B1, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # 1. Captum (Grad-CAM + IG)
    print("\n🔍 Generating Captum (Grad-CAM + IG)...")
    try:
        from src.reporter.captum_reporter import generate_interpretability_report
        res = generate_interpretability_report(
            model, loader, INTERP_DIR, device=str(device),
            class_names=CLASS_NAMES, num_samples=8,
        )
        if res.get("gradcam_path"):
            results["gradcam"] = res["gradcam_path"]
        if res.get("ig_path"):
            results["integrated_gradients"] = res["ig_path"]
        print(f"   ✅ Captum done")
    except Exception as e:
        print(f"   ⚠️ Captum failed: {e}")

    # 2. SHAP
    print("\n🔍 Generating SHAP...")
    try:
        from src.reporter.shap_reporter import generate_shap_explanation
        res = generate_shap_explanation(
            model, loader, INTERP_DIR, device=str(device),
            num_background=50, num_explain=10, class_names=CLASS_NAMES,
        )
        if res.get("shap_path"):
            results["shap"] = res["shap_path"]
        print(f"   ✅ SHAP done")
    except Exception as e:
        print(f"   ⚠️ SHAP failed: {e}")

    # 3. LIME
    print("\n🔍 Generating LIME...")
    try:
        from src.reporter.lime_reporter import explain_predictions_lime
        res = explain_predictions_lime(
            model, loader, INTERP_DIR, class_names=CLASS_NAMES,
            device=str(device), num_samples=5,
        )
        if isinstance(res, dict) and res.get("lime_path"):
            results["lime"] = res["lime_path"]
        print(f"   ✅ LIME done")
    except Exception as e:
        print(f"   ⚠️ LIME failed: {e}")

    # 4. Netron (ONNX)
    print("\n🔍 Generating Netron (ONNX)...")
    try:
        from src.reporter.netron_reporter import save_model_with_name
        res = save_model_with_name(
            model, ARCH_DIR, model_name="EfficientNet-B1",
            input_shape=(1, 3, INPUT_SIZE_B1, INPUT_SIZE_B1),
        )
        if res.get("onnx_path"):
            results["onnx_path"] = res["onnx_path"]
        if res.get("html_path"):
            results["html_path"] = res["html_path"]
        if res.get("info_path"):
            results["architecture_info"] = res["info_path"]
        print(f"   ✅ Netron done")
    except Exception as e:
        print(f"   ⚠️ Netron failed: {e}")

    return results


# ================================================================
# ✅ Save Individual Interpretability Copy
# ================================================================

def copy_interpretability_to_finetune(interp_paths: Dict[str, str]):
    """Copy interpretability PNGs to finetune dir with clear names."""
    import shutil

    mapping = {
        "gradcam": "gradcam.png",
        "integrated_gradients": "integrated_gradients.png",
        "shap": "shap_summary.png",
        "lime": "lime_explanation.png",
    }

    for key, new_name in mapping.items():
        src = interp_paths.get(key)
        if src and os.path.exists(src):
            dst = os.path.join(INTERP_DIR, new_name)
            if os.path.abspath(src) != os.path.abspath(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass


# ================================================================
# ✅ PDF Generation (23 pages, colorful)
# ================================================================

def generate_23page_pdf(
    metrics: Dict,
    history: Dict,
    cv_summary: Dict,
    ci95: Dict,
    plot_paths: Dict[str, str],
    interp_paths: Dict[str, str],
    transferred_path: str,
    n_samples: int,
    output_path: str = None,
) -> str:
    """Generate the 23-page colorful PDF report."""
    if output_path is None:
        output_path = PDF_PATH

    try:
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm, inch
        from reportlab.platypus import (
            Image as RLImage, PageBreak, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        print("❌ reportlab not installed. Run: pip install reportlab")
        return None

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="EfficientNet-B1 Fine-Tune Report",
        author="Babak Yousefian",
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Heading1"],
        fontSize=28, spaceAfter=20, alignment=TA_CENTER,
        textColor=rl_colors.HexColor(Colors.PRIMARY),
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Heading2"],
        fontSize=16, spaceAfter=12, alignment=TA_CENTER,
        textColor=rl_colors.HexColor(Colors.SECONDARY),
        fontName="Helvetica-Oblique",
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading2"],
        fontSize=20, spaceAfter=12, spaceBefore=12,
        textColor=rl_colors.HexColor(Colors.PRIMARY),
        fontName="Helvetica-Bold",
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading3"],
        fontSize=14, spaceAfter=8, spaceBefore=8,
        textColor=rl_colors.HexColor(Colors.SECONDARY),
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=11, spaceAfter=8, leading=16,
        textColor=rl_colors.HexColor(Colors.DARK),
        fontName="Helvetica",
    )
    center_style = ParagraphStyle(
        "Center", parent=body_style, alignment=TA_CENTER,
    )
    small_style = ParagraphStyle(
        "Small", parent=body_style, fontSize=9,
    )

    story = []

    # ================================================================
    # PAGE 1: Info
    # ================================================================
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("EfficientNet-B1", title_style))
    story.append(Paragraph("Complete Fine-Tune Report", subtitle_style))
    story.append(Spacer(1, 0.5 * inch))

    story.append(Paragraph(
        "<b>23-Page Comprehensive Analysis</b>",
        center_style,
    ))
    story.append(Spacer(1, 0.3 * inch))

    info_data = [
        ["Parameter", "Value"],
        ["Model", "EfficientNet-B1 (240x240)"],
        ["Source", "Transferred from EfficientNet-B0 SSL"],
        ["Dataset", f"{n_samples} labeled images"],
        ["Task", "4-class Liver Disease Classification"],
        ["Classes", "NL, G1, G2, G3"],
        ["Author", "Babak Yousefian"],
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Pipeline Version", "8.0.0 (ULTIMATE 23-PAGE)"],
    ]

    t = Table(info_data, colWidths=[2 * inch, 4 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(Colors.PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 1, rl_colors.HexColor(Colors.SECONDARY)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
    ]))
    story.append(t)

    story.append(PageBreak())

    # ================================================================
    # PAGE 2: Details
    # ================================================================
    story.append(Paragraph("1. Details", h1_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Transfer Information:</b>", h2_style))
    story.append(Paragraph(
        f"<b>Source:</b> {os.path.basename(transferred_path)}<br/>"
        f"<b>Full path:</b> {transferred_path}<br/>"
        f"<b>Technique:</b> Transfer Learning from SSL (EfficientNet-B0)<br/>"
        f"<b>Fine-tune LR:</b> 1e-5 (AdamW)<br/>"
        f"<b>Input size:</b> {INPUT_SIZE_B1}x{INPUT_SIZE_B1}<br/>",
        body_style,
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Training Configuration:</b>", h2_style))
    epochs = len(history.get("train_loss", []))
    # ✅ FIX: find best epoch correctly from val_loss
    if history.get("val_loss") and len(history["val_loss"]) > 0:
        best_epoch = int(np.argmin(history["val_loss"])) + 1
    elif history.get("train_loss") and len(history["train_loss"]) > 0:
        best_epoch = int(np.argmin(history["train_loss"])) + 1
    else:
        best_epoch = 0

    config_data = [
        ["Parameter", "Value"],
        ["Epochs", f"{epochs}"],
        ["Best Epoch", f"{best_epoch}"],
        ["Backbone LR", "1e-5 (layer-wise)"],
        ["Head LR", "1e-3 (layer-wise)"],
        ["Optimizer", "AdamW"],
        ["Scheduler", "Cosine with warmup"],
        ["Batch Size", "32"],
        ["Weight Decay", "1e-4"],
        ["Mixed Precision", "BF16/FP16 (auto)"],
        ["Early Stopping", "Patience=15 (on Val F1)"],
        ["Class Weights", "Balanced (in loss)"],
        ["Sampling", "WeightedRandomSampler"],
        ["Fine-Tuning Strategy", "Two-Stage (freeze → unfreeze)"],
        ["Test-Time Augmentation", "Yes (8 augmentations)"],
        ["Ensemble", "3 seeds (42, 123, 2024)"],
        ["Augmentation", "Flip + Rotation + ColorJitter + Zoom"],
    ]

    t = Table(config_data, colWidths=[2.5 * inch, 3.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(Colors.PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
    ]))
    story.append(t)

    story.append(PageBreak())

    # ================================================================
    # PAGE 3: Tables
    # ================================================================
    story.append(Paragraph("2. Performance Tables", h1_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Main Metrics:</b>", h2_style))
    metrics_data = [
        ["Metric", "Value"],
        ["Accuracy", f"{metrics['accuracy']:.4f}"],
        ["Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}"],
        ["Precision (weighted)", f"{metrics['precision']:.4f}"],
        ["Recall (weighted)", f"{metrics['recall']:.4f}"],
        ["F1-Score (weighted)", f"{metrics['f1_score']:.4f}"],
        ["Mean AUC", f"{metrics['roc_auc_mean']:.4f}"],
        ["Samples", str(metrics.get("num_samples", 0))],
    ]
    t = Table(metrics_data, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(Colors.PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 1, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("<b>Per-Class Metrics:</b>", h2_style))
    class_data = [["Class", "Precision", "Recall", "F1", "Support"]]
    for cname in CLASS_NAMES:
        if cname in metrics["classification_report"]:
            r = metrics["classification_report"][cname]
            class_data.append([
                cname,
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1-score']:.4f}",
                str(int(r['support'])),
            ])
    t = Table(class_data, colWidths=[1.2 * inch] * 5)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(Colors.SECONDARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 1, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
    ]))
    story.append(t)

    story.append(PageBreak())

    # ================================================================
    # PAGES 4-11: Plots
    # ================================================================
    plot_pages = [
        ("03_metrics_bar.png", "4. Accuracy - Precision - Recall - F1"),
        ("06_roc_curves.png", "5. AUC - ROC Curves"),
        ("05_confusion_matrix.png", "6. Confusion Matrix"),
        ("12_flow_diagram.png", "7. Training Flow Diagram"),
        ("04_per_class_metrics.png", "8. Per-Class Metrics (Bar Chart)"),
        ("08_accuracy_vs_epoch.png", "9. Accuracy vs Epoch (Line Chart)"),
        ("13_f1_vs_epoch.png", "9.1 F1-Score vs Epoch (Early Stopping Metric)"),  # ✅ NEW
        ("07_class_distribution_pie.png", "10. Class Distribution (Pie Chart)"),
        ("01_loss_train_val.png", "11. Loss Curves"),
    ]

    for fname, title in plot_pages:
        story.append(Paragraph(title, h1_style))
        story.append(Spacer(1, 0.15 * inch))

        fpath = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(fpath):
            img = RLImage(fpath, width=6.5 * inch, height=4.5 * inch)
            story.append(img)
        else:
            story.append(Paragraph(f"⚠️ Plot not found: {fname}", body_style))

        story.append(PageBreak())

    # ================================================================
    # PAGE 12: ONNX
    # ================================================================
    story.append(Paragraph("12. ONNX Export", h1_style))
    story.append(Spacer(1, 0.2 * inch))

    onnx_path = interp_paths.get("onnx_path")
    info_path = interp_paths.get("architecture_info")

    if info_path and os.path.exists(info_path):
        try:
            with open(info_path) as f:
                arch_info = json.load(f)

            arch_data = [["Property", "Value"]]
            for k, v in arch_info.items():
                if not isinstance(v, (dict, list)):
                    arch_data.append([k, str(v)[:80]])

            t = Table(arch_data, colWidths=[2.5 * inch, 3.5 * inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(Colors.PURPLE)),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ]))
            story.append(t)
        except Exception as e:
            story.append(Paragraph(f"⚠️ Could not load info: {e}", body_style))
    else:
        story.append(Paragraph(
            f"ONNX exported to: {onnx_path if onnx_path else 'N/A'}",
            body_style,
        ))

    story.append(PageBreak())

    # ================================================================
    # PAGES 13-16: Interpretability
    # ================================================================
    interp_pages = [
        ("shap", "13. SHAP Feature Importance"),
        ("gradcam", "14. Captum - Grad-CAM"),
        ("integrated_gradients", "14.1 Captum - Integrated Gradients"),
        ("lime", "15. LIME Explanation"),
    ]

    for key, title in interp_pages:
        story.append(Paragraph(title, h1_style))
        story.append(Spacer(1, 0.15 * inch))

        src = interp_paths.get(key)
        if src and os.path.exists(src):
            img = RLImage(src, width=6 * inch, height=4.5 * inch)
            story.append(img)
        else:
            story.append(Paragraph(f"⚠️ Not found: {key}", body_style))

        story.append(PageBreak())

    # Page 16: Netron
    story.append(Paragraph("16. Netron Architecture Viewer", h1_style))
    story.append(Spacer(1, 0.2 * inch))

    html_path = interp_paths.get("html_path")
    story.append(Paragraph(
        f"<b>HTML Viewer:</b> {html_path if html_path else 'N/A'}<br/>"
        f"<b>ONNX:</b> {onnx_path if onnx_path else 'N/A'}<br/><br/>"
        f"Open the HTML file to interactively explore the architecture.",
        body_style,
    ))
    story.append(PageBreak())

    # ================================================================
    # PAGE 17: CV Results
    # ================================================================
    story.append(Paragraph("17. 5-Fold Cross-Validation Results", h1_style))
    story.append(Spacer(1, 0.2 * inch))

    cv_data = [["Fold", "Accuracy", "F1-Score", "AUC", "Best Epoch"]]
    for r in cv_summary["fold_results"]:
        cv_data.append([
            f"Fold {r['fold']}",
            f"{r['accuracy']:.4f}",
            f"{r['f1_score']:.4f}",
            f"{r['roc_auc_mean']:.4f}",
            str(r['best_epoch']),
        ])

    cv_data.append([
        "Mean ± Std",
        f"{cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}",
        f"{cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}",
        f"{cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}",
        "—",
    ])

    t = Table(cv_data, colWidths=[1.2 * inch, 1.6 * inch, 1.6 * inch, 1.4 * inch, 1 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(Colors.TEAL)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -2), "Helvetica"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), rl_colors.HexColor("#F0F8F0")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 1, rl_colors.grey),
    ]))
    story.append(t)

    story.append(Spacer(1, 0.3 * inch))
    cv_plot = os.path.join(PLOTS_DIR, "10_cv_folds.png")
    if os.path.exists(cv_plot):
        story.append(RLImage(cv_plot, width=6.5 * inch, height=3.5 * inch))

    story.append(PageBreak())

    # ================================================================
    # PAGE 18: CI 95%
    # ================================================================
    story.append(Paragraph("18. 95% Confidence Intervals (Bootstrap)", h1_style))
    story.append(Spacer(1, 0.2 * inch))

    ci_data = [["Metric", "Mean", "95% CI Lower", "95% CI Upper", "Std"]]
    for metric_name, ci in ci95.items():
        ci_data.append([
            metric_name.replace("_", " ").title(),
            f"{ci['mean']:.4f}",
            f"{ci['ci95_low']:.4f}",
            f"{ci['ci95_high']:.4f}",
            f"{ci['std']:.4f}",
        ])

    t = Table(ci_data, colWidths=[1.8 * inch, 1.2 * inch, 1.4 * inch, 1.4 * inch, 1.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(Colors.PINK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 1, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
    ]))
    story.append(t)

    # ================================================================
    # PAGE 18.5: Ensemble Information (v8.0) — فقط اگه ensemble داریم
    # ================================================================
    ensemble_seeds = metrics.get("ensemble_seeds", None) if isinstance(metrics, dict) else None
    per_seed_metrics = metrics.get("per_seed_metrics", None) if isinstance(metrics, dict) else None

    if ensemble_seeds and per_seed_metrics:
        story.append(Paragraph("18.5 Ensemble Configuration", h1_style))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph(
            f"<b>Method:</b> {len(ensemble_seeds)}-Seed Ensemble with Test-Time Augmentation (TTA)<br/>"
            f"<b>Seeds:</b> {', '.join(map(str, ensemble_seeds))}<br/>"
            f"<b>TTA augmentations per image:</b> 8 (original, hflip, ±5°, brightness, zoom, contrast)<br/>"
            f"<b>Early stopping metric:</b> F1-Score (not loss)<br/>"
            f"<b>Optimizer:</b> AdamW with layer-wise LR (backbone=1e-5, head=1e-3)<br/>"
            f"<b>Two-Stage Fine-Tuning:</b> Frozen backbone (5 epochs) → Full fine-tune<br/>"
            f"<b>Sampling:</b> WeightedRandomSampler (inverse class frequency)<br/>",
            body_style,
        ))

        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("<b>Per-Seed Performance:</b>", h2_style))

        seed_data = [["Seed", "Best F1", "Best Val Loss"]]
        for entry in per_seed_metrics:
            seed_data.append([
                str(entry.get("seed", "?")),
                f"{entry.get('best_f1', 0):.4f}",
                f"{entry.get('best_val_loss', 0):.4f}",
            ])

        t = Table(seed_data, colWidths=[1.5 * inch, 2 * inch, 2 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(Colors.PURPLE)),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 1, rl_colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.HexColor("#F8F9FA"), rl_colors.white]),
        ]))
        story.append(t)

        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph(
            "<i>Note: The reported metrics on Page 3 are from the ENSEMBLE "
            "(averaged across 3 seeds × 8 TTA augmentations), which typically "
            "outperforms any single seed.</i>",
            small_style,
        ))
        story.append(PageBreak())

    # ================================================================
    # PAGE 19: Train/Test/Val AUC
    # ================================================================
    story.append(Paragraph("19. Train / Val / Test AUC", h1_style))
    story.append(Spacer(1, 0.2 * inch))

    auc_plot = os.path.join(PLOTS_DIR, "09_auc_vs_epoch.png")
    if os.path.exists(auc_plot):
        story.append(RLImage(auc_plot, width=6.5 * inch, height=4 * inch))

    story.append(Spacer(1, 0.2 * inch))
    last_train_auc = history.get("train_auc", [0])[-1] if history.get("train_auc") else 0
    last_val_auc = history.get("val_auc", [0])[-1] if history.get("val_auc") else 0
    story.append(Paragraph(
        f"<b>Final Test AUC (mean over classes):</b> {metrics['roc_auc_mean']:.4f}<br/>"
        f"<b>Train AUC (last epoch):</b> {last_train_auc:.4f}<br/>"
        f"<b>Val AUC (last epoch):</b> {last_val_auc:.4f}<br/>",
        body_style,
    ))

    story.append(PageBreak())

    # ================================================================
    # PAGE 20: Grad-CAM
    # ================================================================
    story.append(Paragraph("20. Grad-CAM Visualizations", h1_style))
    story.append(Spacer(1, 0.2 * inch))

    gradcam_path = interp_paths.get("gradcam")
    if gradcam_path and os.path.exists(gradcam_path):
        story.append(RLImage(gradcam_path, width=6.5 * inch, height=4.5 * inch))
    else:
        story.append(Paragraph("⚠️ Grad-CAM not available", body_style))

    story.append(PageBreak())

    # ================================================================
    # PAGE 21: Loss History
    # ================================================================
    story.append(Paragraph("21. Loss History", h1_style))
    story.append(Spacer(1, 0.2 * inch))

    loss_plot = os.path.join(PLOTS_DIR, "01_loss_train_val.png")
    if os.path.exists(loss_plot):
        story.append(RLImage(loss_plot, width=6.5 * inch, height=4 * inch))

    story.append(Spacer(1, 0.2 * inch))
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    train_f1s = history.get("train_f1", [])
    val_f1s = history.get("val_f1", [])

    loss_info = (
        f"<b>Best Train Loss:</b> {min(train_losses):.4f}<br/>"
        f"<b>Best Val Loss:</b> {min(val_losses):.4f}<br/>"
        f"<b>Final Train Loss:</b> {train_losses[-1]:.4f}<br/>"
        f"<b>Final Val Loss:</b> {val_losses[-1]:.4f}<br/>"
    )
    if train_f1s and val_f1s:
        loss_info += (
            f"<br/><b>Best Train F1:</b> {max(train_f1s):.4f}<br/>"
            f"<b>Best Val F1:</b> {max(val_f1s):.4f} (used for early stopping)<br/>"
        )
    story.append(Paragraph(loss_info, body_style))

    story.append(PageBreak())

    # ================================================================
    # PAGE 22: Summary
    # ================================================================
    story.append(Paragraph("22. Summary", h1_style))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("<b>🎯 Final Results:</b>", h2_style))
    summary_text = f"""
    <b>Model:</b> EfficientNet-B1 (fine-tuned from B0 SSL transfer)<br/>
    <b>Training samples:</b> {n_samples}<br/>
    <b>Test Accuracy:</b> {metrics['accuracy']:.4f}<br/>
    <b>Test F1-Score:</b> {metrics['f1_score']:.4f}<br/>
    <b>Test AUC:</b> {metrics['roc_auc_mean']:.4f}<br/>
    <br/>
    <b>5-Fold CV:</b><br/>
    - Mean Accuracy: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}<br/>
    - Mean F1: {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}<br/>
    - Mean AUC: {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}<br/>
    <br/>
    <b>95% Confidence Interval (Bootstrap):</b><br/>
    - Accuracy: [{ci95['accuracy']['ci95_low']:.4f}, {ci95['accuracy']['ci95_high']:.4f}]<br/>
    - F1: [{ci95['f1_score']['ci95_low']:.4f}, {ci95['f1_score']['ci95_high']:.4f}]<br/>
    """

    t = Table([["Summary"]], colWidths=[6 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor("#E8F4F8")),
        ("BOX", (0, 0), (-1, -1), 2, rl_colors.HexColor(Colors.SECONDARY)),
    ]))
    story.append(t)
    story.append(Paragraph(summary_text, body_style))

    story.append(PageBreak())

    # ================================================================
    # PAGE 23: Finished
    # ================================================================
    story.append(Spacer(1, 3 * inch))
    story.append(Paragraph("✅ Report Complete", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "<b>Thank you for using the Liver AI Pipeline!</b>",
        center_style,
    ))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"Author: Babak Yousefian<br/>"
        f"Version: 8.0.0 (ULTIMATE 23-PAGE)",
        center_style,
    ))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "🎉 All Done! 🎉",
        ParagraphStyle("Big", parent=body_style, fontSize=24,
                       alignment=TA_CENTER, textColor=rl_colors.HexColor(Colors.SUCCESS),
                       fontName="Helvetica-Bold"),
    ))

    # Build
    try:
        doc.build(story)
        print(f"✅ 23-page PDF generated: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ PDF build failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ================================================================
# ✅ Main Entry: Generate Full Report
# ================================================================

def generate_b1_report(
    final_model: nn.Module,
    metrics: Dict,
    history: Dict,
    cv_summary: Dict,
    ci95: Dict,
    image_paths: List[str],
    labels: List[int],
    transferred_path: str,
    device: str = "cuda",
    generate_interpretability_flag: bool = True,
    verbose: bool = True,
) -> str:
    """
    Generate the complete 23-page B1 report.
    """
    ensure_reporter_dirs()
    setup_style()

    print("\n" + "=" * 80)
    print("📄 GENERATING 23-PAGE PDF REPORT")
    print("=" * 80)

    # Step 1: Generate all plots
    print("\n📊 Step 1: Generating plots...")
    plot_paths = generate_all_plots(history, metrics, cv_summary, ci95)

    # Step 2: Interpretability
    interp_paths = {}
    if generate_interpretability_flag:
        print("\n🔬 Step 2: Generating interpretability...")
        try:
            interp_paths = generate_interpretability(
                final_model, image_paths, labels, device, verbose,
            )
            copy_interpretability_to_finetune(interp_paths)
        except Exception as e:
            print(f"⚠️ Interpretability failed: {e}")
            import traceback
            traceback.print_exc()

    # Step 3: PDF
    print("\n📄 Step 3: Generating PDF...")
    pdf_path = generate_23page_pdf(
        metrics=metrics,
        history=history,
        cv_summary=cv_summary,
        ci95=ci95,
        plot_paths=plot_paths,
        interp_paths=interp_paths,
        transferred_path=transferred_path,
        n_samples=len(image_paths),
        output_path=PDF_PATH,
    )

    print("\n" + "=" * 80)
    print(f"✅ Report generation complete!")
    print(f"   PDF: {pdf_path}")
    print(f"   Plots: {PLOTS_DIR}")
    print(f"   Interpretability: {INTERP_DIR}")
    print(f"   Architecture: {ARCH_DIR}")
    print("=" * 80)

    return pdf_path


# ================================================================
# ✅ CLI
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 B1 Reporter - Standalone Test")
    print("=" * 80)
    print("✅ Module loaded. Use from SSLAlgorithms.py (option 13) for full report.")
    print(f"   REPORT_ROOT: {REPORTER_ROOT}")
    print(f"   PDF will be saved to: {PDF_PATH}")
