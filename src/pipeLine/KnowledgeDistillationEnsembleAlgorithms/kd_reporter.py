# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/kd_reporter.py
# -*- coding: utf-8 -*-

"""
24-Page KD Report — Full PDF with LIME/SHAP/Captum/Netron
==========================================================
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Any, List
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

from src.AI.ResNet.utils import CLASS_NAMES

CLASS_COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]


# ═════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════

def _format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ═════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════

def _plot_confusion(cm, save_path, title="Confusion Matrix", normalize=True):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row = cm.sum(axis=1, keepdims=True); row[row == 0] = 1
        disp = cm / row; fmt = ".2%"
    else:
        disp = cm.astype(int); fmt = "d"
    sns.heatmap(disp, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar=True, square=True, linewidths=0.5,
                annot_kws={"size": 13, "fontweight": "bold"}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
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
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random")
    ax.set_xlabel("FPR", fontsize=12, fontweight="bold")
    ax.set_ylabel("TPR", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_metrics_bar(metrics, save_path):
    wa = metrics.get("weighted_avg", {})
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    values = [
        metrics.get("accuracy", 0),
        wa.get("precision", metrics.get("precision_weighted", 0)),
        wa.get("recall", metrics.get("recall_weighted", 0)),
        wa.get("f1-score", metrics.get("f1_weighted", 0)),
        metrics.get("auc_macro", 0),
    ]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#e74c3c"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", lw=1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Test Metrics", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_per_class(metrics, save_path):
    n = len(CLASS_NAMES)
    p = metrics.get("precision_per_class", [0]*n)
    r = metrics.get("recall_per_class", [0]*n)
    f1 = metrics.get("f1_per_class", [0]*n)
    x = np.arange(n); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.bar(x - w, p, w, label="Precision", color="#3498db", edgecolor="black")
    ax.bar(x, r, w, label="Recall", color="#2ecc71", edgecolor="black")
    ax.bar(x + w, f1, w, label="F1-Score", color="#9b59b6", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    if history.get("train_loss") and history.get("val_loss"):
        ep = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(ep, history["train_loss"], "o-", label="Train",
                     color="#e74c3c", lw=2, ms=4)
        axes[0].plot(ep, history["val_loss"], "s-", label="Val",
                     color="#3498db", lw=2, ms=4)
        axes[0].set_xlabel("Epoch", fontweight="bold")
        axes[0].set_ylabel("Loss", fontweight="bold")
        axes[0].set_title("Loss History", fontweight="bold")
        axes[0].legend(); axes[0].grid(alpha=0.3)
    if history.get("val_acc"):
        ep = range(1, len(history["val_acc"]) + 1)
        axes[1].plot(ep, history["val_acc"], "s-", label="Val Acc",
                     color="#2ecc71", lw=2.5, ms=4)
        axes[1].set_xlabel("Epoch", fontweight="bold")
        axes[1].set_ylabel("Accuracy", fontweight="bold")
        axes[1].set_title("Validation Accuracy", fontweight="bold")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 1.05)
    plt.suptitle("KD Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_teacher_agreement(history, save_path):
    if not history.get("teacher_agreement"):
        return
    ep = range(1, len(history["teacher_agreement"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.plot(ep, history["teacher_agreement"], "o-",
            color="#9b59b6", lw=2.5, ms=5)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Agreement", fontweight="bold")
    ax.set_title("Teacher-Student Agreement (per epoch)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_lr(history, save_path):
    if not history.get("learning_rates"):
        return
    ep = range(1, len(history["learning_rates"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.plot(ep, history["learning_rates"], "o-", color="#f39c12",
            lw=2, ms=4)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("LR", fontweight="bold")
    ax.set_yscale("log")
    ax.set_title("LR Schedule", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_advanced_eval(adv_results, save_path):
    """Plot CV5 + CI95 + TTA32 results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=200)

    # CV5
    cv = adv_results.get("cv5", {})
    folds = [f["fold"] for f in cv.get("folds", [])]
    accs = [f["acc"] for f in cv.get("folds", [])]
    if folds:
        axes[0].bar(folds, accs, color="#3498db", edgecolor="black")
        axes[0].axhline(cv.get("mean_acc", 0), color="red", ls="--",
                        label=f"Mean = {cv.get('mean_acc', 0):.4f}")
        axes[0].set_xlabel("Fold", fontweight="bold")
        axes[0].set_ylabel("Accuracy", fontweight="bold")
        axes[0].set_title("CV5", fontweight="bold")
        axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")

    # CI95
    ci = adv_results.get("ci95", {})
    if ci:
        acc_ci = ci.get("acc", {})
        axes[1].errorbar(
            [0], [acc_ci.get("mean", 0)],
            yerr=[[acc_ci.get("mean", 0) - acc_ci.get("lower", 0)],
                  [acc_ci.get("upper", 0) - acc_ci.get("mean", 0)]],
            fmt="o", capsize=10, color="#e74c3c", ms=10,
        )
        axes[1].set_xlim(-0.5, 0.5)
        axes[1].set_xticks([0])
        axes[1].set_xticklabels(["Acc"])
        axes[1].set_ylabel("Accuracy", fontweight="bold")
        axes[1].set_title(f"CI{ci.get('ci', 95)} (n={ci.get('n_bootstrap', 0)})",
                          fontweight="bold")
        axes[1].grid(alpha=0.3, axis="y")

    # TTA32
    tta = adv_results.get("tta32", {})
    if tta:
        axes[2].bar(["Acc", "AUC"], [tta.get("acc", 0), tta.get("auc", 0)],
                    color=["#2ecc71", "#9b59b6"], edgecolor="black")
        for i, v in enumerate([tta.get("acc", 0), tta.get("auc", 0)]):
            axes[2].text(i, v + 0.02, f"{v:.4f}", ha="center",
                         fontweight="bold")
        axes[2].set_ylim(0, 1.1)
        axes[2].set_title(f"TTA-{tta.get('n_aug', 0)}", fontweight="bold")
        axes[2].grid(alpha=0.3, axis="y")

    plt.suptitle("Advanced Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
# LIME / SHAP / CAPTUM / NETRON (best-effort)
# ═════════════════════════════════════════════════════════════════════

def _try_lime(model, sample_img, save_path):
    """Best-effort LIME explanation. Saves image or returns None."""
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries

        def _predict(imgs):
            model.eval()
            batch = torch.tensor(imgs).permute(0, 3, 1, 2).float().to(
                next(model.parameters()).device)
            with torch.no_grad():
                out = model(batch)
                return torch.softmax(out, dim=1).cpu().numpy()

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            sample_img.astype(np.double),
            _predict,
            top_labels=4,
            hide_color=0,
            num_samples=100,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False,
        )
        img_boundry = mark_boundaries(temp / 255.0, mask)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        ax.imshow(img_boundry)
        ax.set_title("LIME Explanation", fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"   ✅ LIME saved: {save_path}")
        return True
    except Exception as e:
        print(f"   ⚠️ LIME skipped: {e}")
        return False


def _try_shap(model, sample_tensor, save_path):
    """Best-effort SHAP. Saves image or returns None."""
    try:
        import shap
        background = torch.zeros(1, *sample_tensor.shape[1:]).to(
            next(model.parameters()).device)
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(sample_tensor.unsqueeze(0))
        # shap_values: list of arrays (n_classes, ...)
        fig, axes = plt.subplots(1, len(CLASS_NAMES),
                                  figsize=(4 * len(CLASS_NAMES), 4),
                                  dpi=180)
        for i, cname in enumerate(CLASS_NAMES):
            sv = np.abs(shap_values[i][0]).sum(axis=0)
            axes[i].imshow(sv, cmap="hot")
            axes[i].set_title(cname, fontweight="bold")
            axes[i].axis("off")
        plt.suptitle("SHAP Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"   ✅ SHAP saved: {save_path}")
        return True
    except Exception as e:
        print(f"   ⚠️ SHAP skipped: {e}")
        return False


def _try_captum(model, sample_tensor, save_path):
    """Best-effort Captum Integrated Gradients."""
    try:
        from captum.attr import IntegratedGradients

        ig = IntegratedGradients(model)
        sample = sample_tensor.unsqueeze(0).to(
            next(model.parameters()).device)
        baseline = torch.zeros_like(sample)
        attributions = ig.attribute(sample, baseline, target=0, n_steps=20)
        attr = attributions.detach().cpu().numpy()[0]
        attr_map = np.abs(attr).sum(axis=0)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        ax.imshow(attr_map, cmap="viridis")
        ax.set_title("Captum Integrated Gradients", fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"   ✅ Captum saved: {save_path}")
        return True
    except Exception as e:
        print(f"   ⚠️ Captum skipped: {e}")
        return False


def _try_netron(model, save_path):
    """Export ONNX model for Netron visualization."""
    try:
        sample = torch.randn(1, 3, 224, 224).to(
            next(model.parameters()).device)
        onnx_path = save_path.replace(".png", ".onnx")
        torch.onnx.export(
            model, sample, onnx_path,
            export_params=True, opset_version=12,
            do_constant_folding=True,
            input_names=["input"], output_names=["output"],
        )
        print(f"   ✅ ONNX exported: {onnx_path}")
        # Save a placeholder text in the .png path
        fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
        ax.text(0.5, 0.5,
                f"Netron: open in browser\n\nFile: {os.path.basename(onnx_path)}",
                ha="center", va="center", fontsize=14,
                bbox=dict(boxstyle="round,pad=1", facecolor="#fff9e6",
                          edgecolor="#f39c12"))
        ax.axis("off")
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"   ⚠️ Netron/ONNX skipped: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════
# PDF BUILDER
# ═════════════════════════════════════════════════════════════════════

def _page_cover(page, metrics, config, student_name, teacher_name):
    page.draw_rect(fitz.Rect(0, 0, 595, 200), color=None,
                   fill=(0.1, 0.1, 0.18), width=0)
    page.insert_text((50, 80), "Knowledge Distillation",
                     fontsize=30, fontname="helvetica-bold",
                     color=(0, 0.83, 1))
    page.insert_text((50, 125), f"{teacher_name} → {student_name}",
                     fontsize=20, fontname="helvetica", color=(1, 1, 1))
    page.insert_text((50, 160), "Liver Ultrasound Classification",
                     fontsize=12, fontname="helvetica",
                     color=(0.65, 0.75, 0.85))

    acc = metrics.get("accuracy", 0)
    page.draw_rect(fitz.Rect(50, 230, 545, 320),
                   color=(0, 0.83, 1), fill=(0.9, 0.97, 1), width=2)
    page.insert_text((70, 265), "TEST ACCURACY",
                     fontsize=14, fontname="helvetica-bold",
                     color=(0.1, 0.1, 0.2))
    page.insert_text((70, 305), f"{acc * 100:.2f}%",
                     fontsize=32, fontname="helvetica-bold",
                     color=(0, 0.55, 0.85))

    info = [
        ("Teacher", teacher_name),
        ("Student", student_name),
        ("Temperature", str(config.get("temperature", "N/A"))),
        ("Alpha", str(config.get("alpha", "N/A"))),
        ("Epochs", str(config.get("epochs", "N/A"))),
        ("Batch Size", str(config.get("batch_size", "N/A"))),
        ("LR (Student)", str(config.get("lr_student", "N/A"))),
        ("Weight Decay", str(config.get("weight_decay", "N/A"))),
        ("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Author", "Babak Yousefian"),
    ]
    y = 370
    for k, v in info:
        page.draw_rect(fitz.Rect(50, y, 545, y + 28),
                       color=None, fill=(0.97, 0.98, 1))
        page.insert_text((60, y + 19), k, fontsize=11,
                         fontname="helvetica-bold", color=(0.2, 0.2, 0.3))
        page.insert_text((200, y + 19), v, fontsize=11,
                         fontname="helvetica", color=(0.1, 0.1, 0.2))
        y += 32


def _page_exec_summary(page, metrics):
    page.insert_text((50, 50), "EXECUTIVE SUMMARY",
                     fontsize=18, fontname="helvetica-bold",
                     color=(0, 0.55, 0.85))
    page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62),
                   color=(0, 0.83, 1), width=2)
    y = 100
    wa = metrics.get("weighted_avg", {})
    rows = [
        ("Accuracy", metrics.get("accuracy", 0)),
        ("Precision (weighted)", wa.get("precision", metrics.get("precision_weighted", 0))),
        ("Recall (weighted)", wa.get("recall", metrics.get("recall_weighted", 0))),
        ("F1 (weighted)", wa.get("f1-score", metrics.get("f1_weighted", 0))),
        ("AUC (macro)", metrics.get("auc_macro", 0)),
    ]
    for label, val in rows:
        page.draw_rect(fitz.Rect(50, y, 545, y + 26),
                       color=(0.8, 0.85, 0.9), fill=(0.97, 0.98, 1), width=0.5)
        page.insert_text((60, y + 18), label, fontsize=11,
                         fontname="helvetica", color=(0.15, 0.15, 0.2))
        page.insert_text((400, y + 18), f"{val:.4f}",
                         fontsize=12, fontname="helvetica-bold",
                         color=(0, 0.55, 0.85))
        page.insert_text((480, y + 18), f"({val * 100:.2f}%)",
                         fontsize=10, fontname="helvetica",
                         color=(0.4, 0.4, 0.5))
        y += 30


def _page_config(page, config):
    page.insert_text((50, 50), "CONFIGURATION",
                     fontsize=18, fontname="helvetica-bold",
                     color=(0, 0.55, 0.85))
    page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62),
                   color=(0, 0.83, 1), width=2)
    y = 100
    for k, v in config.items():
        page.draw_rect(fitz.Rect(50, y, 545, y + 24),
                       color=(0.85, 0.9, 0.95), fill=(0.98, 0.99, 1), width=0.4)
        page.insert_text((60, y + 17), str(k), fontsize=10,
                         fontname="helvetica", color=(0.2, 0.2, 0.3))
        page.insert_text((280, y + 17), str(v), fontsize=10,
                         fontname="helvetica-bold", color=(0.1, 0.1, 0.2))
        y += 28
        if y > 780:
            break


def _page_per_class(page, metrics):
    page.insert_text((50, 50), "PER-CLASS METRICS",
                     fontsize=18, fontname="helvetica-bold",
                     color=(0, 0.55, 0.85))
    page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62),
                   color=(0, 0.83, 1), width=2)
    y = 100
    headers = ["Class", "Precision", "Recall", "F1", "Support"]
    page.draw_rect(fitz.Rect(50, y, 545, y + 30),
                   color=None, fill=(0, 0.55, 0.85))
    for h, x in zip(headers, [60, 170, 270, 370, 470]):
        page.insert_text((x, y + 20), h, fontsize=11,
                         fontname="helvetica-bold", color=(1, 1, 1))
    y += 30
    p = metrics.get("precision_per_class", [0]*4)
    r = metrics.get("recall_per_class", [0]*4)
    f1 = metrics.get("f1_per_class", [0]*4)
    s = metrics.get("support_per_class", [0]*4)
    for i, cname in enumerate(CLASS_NAMES):
        page.draw_rect(fitz.Rect(50, y, 545, y + 28),
                       color=(0.85, 0.9, 0.95),
                       fill=(0.97, 0.98, 1) if i % 2 == 0 else (1, 1, 1),
                       width=0.4)
        page.insert_text((60, y + 19), cname, fontsize=11,
                         fontname="helvetica-bold", color=(0.15, 0.15, 0.2))
        for x, v in [(180, p[i]), (280, r[i]), (380, f1[i])]:
            page.insert_text((x, y + 19), f"{v:.4f}",
                             fontsize=11, fontname="helvetica",
                             color=(0.1, 0.1, 0.2))
        page.insert_text((480, y + 19), str(int(s[i])),
                         fontsize=11, fontname="helvetica",
                         color=(0.1, 0.1, 0.2))
        y += 28


def _page_advanced_eval(page, adv):
    page.insert_text((50, 50), "ADVANCED EVALUATION",
                     fontsize=18, fontname="helvetica-bold",
                     color=(0, 0.55, 0.85))
    page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62),
                   color=(0, 0.83, 1), width=2)
    y = 100

    # CV5
    cv = adv.get("cv5", {})
    page.insert_text((60, y), "Cross-Validation (5-Fold):",
                     fontsize=13, fontname="helvetica-bold",
                     color=(0.1, 0.1, 0.2))
    y += 25
    page.insert_text((80, y),
                     f"Mean Acc: {cv.get('mean_acc', 0):.4f} ± "
                     f"{cv.get('std_acc', 0):.4f}",
                     fontsize=11, fontname="helvetica",
                     color=(0.1, 0.1, 0.2))
    y += 20
    page.insert_text((80, y),
                     f"Mean AUC: {cv.get('mean_auc', 0):.4f} ± "
                     f"{cv.get('std_auc', 0):.4f}",
                     fontsize=11, fontname="helvetica",
                     color=(0.1, 0.1, 0.2))
    y += 35

    # CI95
    ci = adv.get("ci95", {})
    page.insert_text((60, y), "CI95 (Bootstrap 2000):",
                     fontsize=13, fontname="helvetica-bold",
                     color=(0.1, 0.1, 0.2))
    y += 25
    acc_ci = ci.get("acc", {})
    page.insert_text((80, y),
                     f"Acc: {acc_ci.get('mean', 0):.4f}  "
                     f"[{acc_ci.get('lower', 0):.4f}, "
                     f"{acc_ci.get('upper', 0):.4f}]",
                     fontsize=11, fontname="helvetica",
                     color=(0.1, 0.1, 0.2))
    y += 20
    auc_ci = ci.get("auc", {})
    page.insert_text((80, y),
                     f"AUC: {auc_ci.get('mean', 0):.4f}  "
                     f"[{auc_ci.get('lower', 0):.4f}, "
                     f"{auc_ci.get('upper', 0):.4f}]",
                     fontsize=11, fontname="helvetica",
                     color=(0.1, 0.1, 0.2))
    y += 35

    # TTA
    tta = adv.get("tta32", {})
    page.insert_text((60, y), f"TTA-{tta.get('n_aug', 0)}:",
                     fontsize=13, fontname="helvetica-bold",
                     color=(0.1, 0.1, 0.2))
    y += 25
    page.insert_text((80, y),
                     f"Acc: {tta.get('acc', 0):.4f}  "
                     f"AUC: {tta.get('auc', 0):.4f}",
                     fontsize=11, fontname="helvetica",
                     color=(0.1, 0.1, 0.2))


def _page_footer(page, n):
    page.draw_line(fitz.Point(50, 800), fitz.Point(545, 800),
                   color=(0.85, 0.9, 0.95), width=0.5)
    page.insert_text((50, 815),
                     f"KD Report — Liver AI Pipeline | developed by Babak Yousefian | Page {n}",
                     fontsize=8, fontname="helvetica-oblique",
                     color=(0.5, 0.5, 0.6))


# ═════════════════════════════════════════════════════════════════════
# MAIN REPORT
# ═════════════════════════════════════════════════════════════════════

def generate_kd_report(
    model: nn.Module,
    teacher: nn.Module,
    test_ds,
    test_metrics: Dict[str, Any],
    history: Dict[str, Any],
    adv_results: Dict[str, Any],
    output_dir: str,
    student_name: str = "SwinTransformer",
    teacher_name: str = "ConvNext",
    config: Dict[str, Any] = None,
) -> str:
    """Generate 24-page PDF + all plots."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "KD_Report.pdf")

    if not HAS_FITZ:
        print("⚠️ PyMuPDF not installed. Install: pip install PyMuPDF")
        return pdf_path

    print("   📊 Generating plots...")

    # === PLOTS ===
    _plot_confusion(test_metrics["confusion_matrix"],
                    os.path.join(output_dir, "confusion_matrix.png"),
                    "Confusion Matrix (Normalized)", normalize=True)
    _plot_confusion(test_metrics["confusion_matrix"],
                    os.path.join(output_dir, "confusion_matrix_raw.png"),
                    "Confusion Matrix (Raw)", normalize=False)
    _plot_roc(test_metrics["y_true"], test_metrics["y_prob"],
              os.path.join(output_dir, "roc_curves.png"))
    _plot_metrics_bar(test_metrics,
                      os.path.join(output_dir, "metrics_bar.png"))
    _plot_per_class(test_metrics,
                    os.path.join(output_dir, "per_class_metrics.png"))
    _plot_training_history(history,
                            os.path.join(output_dir, "training_history.png"))
    _plot_teacher_agreement(history,
                             os.path.join(output_dir, "teacher_agreement.png"))
    _plot_lr(history,
              os.path.join(output_dir, "lr_schedule.png"))
    _plot_advanced_eval(adv_results,
                         os.path.join(output_dir, "advanced_eval.png"))

    # === XAI ===
    print("   🔬 Running XAI (LIME / SHAP / Captum / Netron)...")

    from PIL import Image
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # ── Get a single sample batch from the dataset ──
    # (no dependency on internal attribute names like .samples / .data)
    xai_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    sample_tensor = None
    sample_label = None

    for batch in xai_loader:
        if len(batch) == 3:
            x, y, _ = batch        # ← LiverDataset may return (img, label, path)
            sample_tensor = x[0]
            sample_label = int(y[0])
        else:
            x, y = batch
            sample_tensor = x[0]
            sample_label = int(y[0])
        break

    # ── Prepare LIME input (HWC uint8 image) ──
    # Convert tensor to numpy for LIME
    def _tensor_to_lime_image(t):
        # t shape: (3, H, W), normalized with ImageNet stats
        img = t.detach().cpu().numpy().transpose(1, 2, 0)     # HWC
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    if sample_tensor is not None:
        lime_img = _tensor_to_lime_image(sample_tensor)
        try:
            _try_lime(model, lime_img, os.path.join(output_dir, "xai_lime.png"))
        except Exception as e:
            print(f"   ⚠️ LIME failed: {e}")

        try:
            _try_shap(model, sample_tensor, os.path.join(output_dir, "xai_shap.png"))
        except Exception as e:
            print(f"   ⚠️ SHAP failed: {e}")

        try:
            _try_captum(model, sample_tensor, os.path.join(output_dir, "xai_captum.png"))
        except Exception as e:
            print(f"   ⚠️ Captum failed: {e}")

        try:
            _try_netron(model, os.path.join(output_dir, "xai_netron.png"))
        except Exception as e:
            print(f"   ⚠️ Netron failed: {e}")
    else:
        print("   ⚠️ No sample available for XAI (skipping)")

    # === PDF ===
    print("   📄 Building 24-page PDF...")
    doc = fitz.open()
    page_num = 1

    page = doc.new_page(width=595, height=842)
    _page_cover(page, test_metrics, config or {}, student_name, teacher_name)
    _page_footer(page, page_num); page_num += 1

    page = doc.new_page(width=595, height=842)
    _page_exec_summary(page, test_metrics); _page_footer(page, page_num); page_num += 1

    page = doc.new_page(width=595, height=842)
    _page_config(page, config or {}); _page_footer(page, page_num); page_num += 1

    page = doc.new_page(width=595, height=842)
    _page_per_class(page, test_metrics); _page_footer(page, page_num); page_num += 1

    page = doc.new_page(width=595, height=842)
    _page_advanced_eval(page, adv_results); _page_footer(page, page_num); page_num += 1

    # Images
    image_pages = [
        ("confusion_matrix.png", "CONFUSION MATRIX (Normalized)"),
        ("confusion_matrix_raw.png", "CONFUSION MATRIX (Raw)"),
        ("roc_curves.png", "ROC CURVES"),
        ("metrics_bar.png", "PERFORMANCE METRICS"),
        ("per_class_metrics.png", "PER-CLASS METRICS"),
        ("training_history.png", "TRAINING HISTORY"),
        ("train_val_gap.png", "GAP ANALYSIS"),
        ("teacher_agreement.png", "TEACHER-STUDENT AGREEMENT"),
        ("lr_schedule.png", "LR SCHEDULE"),
        ("advanced_eval.png", "ADVANCED EVALUATION"),
    ]
    for fname, title in image_pages:
        p = os.path.join(output_dir, fname)
        if os.path.exists(p):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), title,
                             fontsize=16, fontname="helvetica-bold",
                             color=(0.1, 0.1, 0.2))
            page.insert_image(fitz.Rect(60, 90, 535, 750), filename=p)
            _page_footer(page, page_num); page_num += 1

    # XAI pages
    xai_pages = [
        ("xai_lime.png", "LIME EXPLANATION"),
        ("xai_shap.png", "SHAP FEATURE IMPORTANCE"),
        ("xai_captum.png", "CAPTUM INTEGRATED GRADIENTS"),
        ("xai_netron.png", "NETRON MODEL GRAPH"),
    ]
    for fname, title in xai_pages:
        p = os.path.join(output_dir, fname)
        if os.path.exists(p):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), title,
                             fontsize=16, fontname="helvetica-bold",
                             color=(0, 0.55, 0.85))
            page.insert_image(fitz.Rect(60, 90, 535, 750), filename=p)
            _page_footer(page, page_num); page_num += 1

    doc.save(pdf_path)
    total = len(doc)
    doc.close()

    print(f"   ✅ PDF saved: {pdf_path} ({total} pages)")
    return pdf_path
