# src/AI/VGG16/vgg16Reports.py
# -*- coding: utf-8 -*-

"""
VGG16 Report Generator — Independent Version
=============================================
Generates a complete, professional PDF report for VGG16 fine-tuning.

Outputs (in report_dir):
- VGG16_Report.pdf            : Multi-page PDF (Test metrics)
- metrics.json                : All numeric metrics (TEST)
- summary.txt                 : Human-readable summary
- classification_report.txt   : sklearn classification report
- predictions.csv             : Per-sample predictions
- confusion_matrix.png        : Normalized confusion matrix
- confusion_matrix_raw.png    : Raw confusion matrix
- roc_curves.png              : ROC curves (one-vs-rest)
- metrics_bar.png             : Bar chart of key metrics
- per_class_metrics.png       : Precision/Recall/F1 per class
- performance_pie.png         : Pie chart
- training_history.png        : Loss + Accuracy curves
- learning_rate.png           : LR schedule
- stage_markers.png           : Stage 1 vs Stage 2

Author: Babak Yousefian
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ---- Optional: PyMuPDF for PDF ----
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("⚠️ PyMuPDF not installed. Install with: pip install PyMuPDF")

# ---- Optional: GradCAM ----
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False


# ================================================================
#  CONSTANTS
# ================================================================
CLASS_NAMES = ["NL", "G1", "G2", "G3"]
CLASS_COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ================================================================
#  MAIN REPORTER CLASS
# ================================================================

class VGG16Reporter:
    """
    Independent report generator for VGG16.

    Usage:
        reporter = VGG16Reporter(results)
        reporter.generate_all(report_dir="...")
    """

    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.config = results.get("config", {})
        self.history = results.get("history", {})
        self.best_metrics = results.get("best_metrics", {})
        self.test_metrics = results.get("test_metrics", {}) or self.best_metrics
        self.y_true = results.get("y_true")
        self.y_pred = results.get("y_pred")
        self.y_prob = results.get("y_prob")
        self.class_names = results.get("class_names", CLASS_NAMES)
        self.training_time_sec = results.get("training_time_sec", 0)
        self.best_epoch = results.get("best_epoch", 0)
        self.best_acc = results.get("config", {}).get("test_accuracy", 0)

        # ✅ فقط TEST metrics برای گزارش
        self.primary_metrics = self.test_metrics

    def _load_history_from_trainer(self, model_dir=None):
        """اگه history خالیه، از فایل JSON لود کن."""
        if model_dir is None:
            model_dir = self.config.get("model_dir") or self.results.get("model_dir")

        if not model_dir or not os.path.exists(model_dir):
            print(f"  ⚠️ model_dir not found: {model_dir}")
            return False

        path = os.path.join(model_dir, "history_VGG16.json")
        if not os.path.exists(path):
            print(f"  ⚠️ History file not found: {path}")
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if loaded.get("train_loss"):
                self.history = loaded
                print(f"  ✅ History loaded: {path}")
                print(f"     train_loss: {len(self.history.get('train_loss', []))} items")
                print(f"     val_loss:   {len(self.history.get('val_loss', []))} items")
                return True
        except Exception as e:
            print(f"  ⚠️ Failed to load history: {e}")
        return False


    # ============================================================
    #  PUBLIC API
    # ============================================================
    def generate_all(self, report_dir: str) -> str:
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

        # ✅ اگه history خالیه، از فایل لود کن
        if not self.history.get("train_loss"):
            print("\n📊 History is empty. Loading from saved file...")
            parent = os.path.dirname(report_dir)
            model_dir = self.config.get("model_dir") or os.path.join(parent, "vgg16")
            self._load_history_from_trainer(model_dir)

        print("\n" + "=" * 70)
        print("  VGG16 REPORT GENERATION")
        print("=" * 70)

        print("\n[1/6] Generating plots...")
        self._generate_plots(report_dir)

        print("[2/6] Saving metrics JSON...")
        self._save_metrics_json(report_dir)

        print("[3/6] Saving summary TXT...")
        self._save_summary_txt(report_dir)

        print("[4/6] Saving classification report...")
        self._save_classification_report(report_dir)

        print("[5/6] Saving predictions CSV...")
        self._save_predictions_csv(report_dir)

        print("[6/6] Generating PDF report...")
        pdf_path = self._generate_pdf(report_dir)

        print("\n" + "=" * 70)
        print(f"  ✅ ALL REPORTS SAVED TO: {report_dir}")
        print("=" * 70)

        return pdf_path

    # ============================================================
    #  1. PLOTS
    # ============================================================

    def _generate_plots(self, report_dir):
        # ✅ FIX: از test metrics استفاده کن (نه primary)
        m = self.test_metrics if self.test_metrics else self.primary_metrics
        self.primary_metrics = m  # ← sync کن

        # ✅ FIX: confusion matrix از test
        cm_raw = m.get("confusion_matrix")
        if cm_raw is not None and len(cm_raw) == 4:
            cm = np.array(cm_raw, dtype=int)
        else:
            cm = np.zeros((4, 4), dtype=int)

        # ✅ FIX: اگه cm همه 0 بود، از y_true/y_pred دوباره حساب کن
        if cm.sum() == 0 and self.y_true is not None and self.y_pred is not None:
            from sklearn.metrics import confusion_matrix as sk_cm
            cm = sk_cm(self.y_true, self.y_pred, labels=[0, 1, 2, 3])
            print(f"   ⚠️ Re-computed confusion matrix from y_true/y_pred (sum={cm.sum()})")

        # ---- Confusion Matrix (normalized) ----
        self._plot_confusion_matrix(
            cm, self.class_names,
            os.path.join(report_dir, "confusion_matrix.png"),
            title="VGG16 — Confusion Matrix (Normalized)",
            normalize=True,
        )

        # ---- Confusion Matrix (raw) ----
        self._plot_confusion_matrix(
            cm, self.class_names,
            os.path.join(report_dir, "confusion_matrix_raw.png"),
            title="VGG16 — Confusion Matrix (Raw)",
            normalize=False,
        )

        # ---- ROC curves ----
        # ✅ FIX: فقط اگه y_true/y_prob برای test موجود باشن
        if self.y_true is not None and self.y_prob is not None:
            try:
                self._plot_roc_curves(
                    self.y_true, self.y_prob, self.class_names,
                    os.path.join(report_dir, "roc_curves.png"),
                )
            except Exception as e:
                print(f"  ⚠️ ROC curves failed: {e}")

        # ---- Metrics bar ----
        self._plot_metrics_bar(
            m, os.path.join(report_dir, "metrics_bar.png"),
        )

        # ---- Per-class metrics ----
        self._plot_per_class_metrics(
            m, self.class_names,
            os.path.join(report_dir, "per_class_metrics.png"),
        )

        # ---- Pie chart ----
        self._plot_pie(
            m, os.path.join(report_dir, "performance_pie.png"),
        )

        # ---- Training history ----
        if self.history.get("train_loss"):
            self._plot_training_history(
                self.history,
                os.path.join(report_dir, "training_history.png"),
            )

        # ✅ همیشه Gap رو صدا بزن
        print("  📊 Generating Gap analysis...")
        self._plot_train_val_gap(
            self.history,
            os.path.join(report_dir, "train_val_gap.png"),
        )

        # ---- Learning rate schedule ----
        if self.history.get("learning_rates"):
            self._plot_lr_schedule(
                self.history,
                os.path.join(report_dir, "learning_rate.png"),
            )

    def _plot_confusion_matrix(self, cm, class_names, save_path,
                               title="Confusion Matrix", normalize=True):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        cm = np.asarray(cm, dtype=float)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            disp = cm / row_sums
            fmt = ".2%"
        else:
            disp = cm.astype(int)
            fmt = "d"

        sns.heatmap(
            disp, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar=True, square=True, linewidths=0.5,
            annot_kws={"size": 13, "fontweight": "bold"},
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_ylabel("True", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_roc_curves(self, y_true, y_prob, class_names, save_path):
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob)
        n_classes = len(class_names)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

        for i, name in enumerate(class_names):
            y_bin = (y_true == i).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2.5,
                    color=CLASS_COLORS[i % len(CLASS_COLORS)],
                    label=f"{name} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.6,
                label="Random (AUC = 0.5)")
        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax.set_title("VGG16 — ROC Curves (One-vs-Rest)",
                     fontsize=14, fontweight="bold", pad=12)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_metrics_bar(self, m, save_path):
        # ✅ استخراج درست از ساختار sklearn
        wa = m.get("weighted avg", {})

        accuracy = m.get("accuracy", 0) or 0
        precision = m.get("precision_macro") or wa.get("precision", 0) or 0
        recall = m.get("recall_macro") or wa.get("recall", 0) or 0
        f1 = m.get("f1_macro") or wa.get("f1-score", 0) or 0

        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [accuracy, precision, recall, f1]

        auc_macro = m.get("auc_macro")
        if auc_macro is not None and auc_macro > 0:
            labels.append("AUC")
            values.append(auc_macro)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#e74c3c"][:len(values)]
        bars = ax.bar(labels, values, color=colors,
                      edgecolor="black", linewidth=1.2, alpha=0.9)

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("VGG16 — Performance Metrics (Test Set)",
                     fontsize=14, fontweight="bold", pad=12)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_per_class_metrics(self, m, class_names, save_path):
        n = len(class_names)

        # ✅ استخراج از ساختار sklearn (per-class keys)
        p = []
        r = []
        f1 = []
        for cname in class_names:
            if cname in m:
                p.append(m[cname].get("precision", 0) or 0)
                r.append(m[cname].get("recall", 0) or 0)
                f1.append(m[cname].get("f1-score", 0) or 0)
            else:
                p.append(0)
                r.append(0)
                f1.append(0)

        # ⚠️ اگه همه صفر بودن، skip کن
        if sum(p) == 0 and sum(r) == 0 and sum(f1) == 0:
            print("  ⚠️ Skipping per-class metrics (all zero)")
            return

        x = np.arange(n)
        w = 0.26

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        bars1 = ax.bar(x - w, p, w, label="Precision",
                       color="#3498db", edgecolor="black", linewidth=1.2)
        bars2 = ax.bar(x, r, w, label="Recall",
                       color="#2ecc71", edgecolor="black", linewidth=1.2)
        bars3 = ax.bar(x + w, f1, w, label="F1-Score",
                       color="#9b59b6", edgecolor="black", linewidth=1.2)

        for bars, values in [(bars1, p), (bars2, r), (bars3, f1)]:
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("VGG16 — Per-Class Metrics (Test Set)",
                     fontsize=14, fontweight="bold", pad=12)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_pie(self, m, save_path):
        # ✅ استخراج درست از ساختار sklearn
        wa = m.get("weighted avg", {})
        p = wa.get("precision", 0) or 0
        r = wa.get("recall", 0) or 0
        f1 = wa.get("f1-score", 0) or 0

        # ⚠️ اگه همه صفر بودن، از macro استفاده کن
        if p == 0 and r == 0 and f1 == 0:
            p = m.get("precision_macro", 0) or 0
            r = m.get("recall_macro", 0) or 0
            f1 = m.get("f1_macro", 0) or 0

        # ⚠️ اگه بازم همه صفر بودن، از accuracy استفاده کن
        if p == 0 and r == 0 and f1 == 0:
            acc = m.get("accuracy", 0) or 0
            p = r = f1 = acc

        # ⚠️ اگه بازم همه صفر بودن، اصلاً pie نساز
        if p == 0 and r == 0 and f1 == 0:
            print("  ⚠️ Skipping pie chart (all metrics are zero)")
            return

        values = [p, r, f1]
        labels = [f"Precision\n({p:.3f})",
                  f"Recall\n({r:.3f})",
                  f"F1-Score\n({f1:.3f})"]
        colors = ["#3498db", "#2ecc71", "#9b59b6"]

        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        wedges, texts = ax.pie(
            values, labels=labels, colors=colors,
            shadow=True, startangle=90,
            textprops={"fontsize": 11, "fontweight": "bold"},
        )
        ax.set_title("VGG16 — Metrics Distribution (Test Set)",
                     fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_training_history(self, history, save_path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

        if history.get("train_loss") and history.get("val_loss"):
            ep = range(1, len(history["train_loss"]) + 1)
            axes[0].plot(ep, history["train_loss"], "o-",
                         label="Train", linewidth=2, markersize=4,
                         color="#e74c3c")
            axes[0].plot(ep, history["val_loss"], "s-",
                         label="Val", linewidth=2, markersize=4,
                         color="#3498db")
            axes[0].set_xlabel("Epoch", fontweight="bold")
            axes[0].set_ylabel("Loss", fontweight="bold")
            axes[0].set_title("Loss History", fontweight="bold")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

        if history.get("val_acc"):
            ep = range(1, len(history["val_acc"]) + 1)
            axes[1].plot(ep, history["val_acc"], "s-",
                         label="Val Accuracy", linewidth=2.5,
                         markersize=4, color="#2ecc71")
            axes[1].set_xlabel("Epoch", fontweight="bold")
            axes[1].set_ylabel("Accuracy", fontweight="bold")
            axes[1].set_title("Validation Accuracy", fontweight="bold")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            axes[1].set_ylim(0, 1.05)

        plt.suptitle("VGG16 — Training History", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_train_val_gap(self, history, save_path):
        """نمودار Train Loss + Val Loss + Gap + تشخیص Overfitting/Underfitting"""
        print(f"  🔍 _plot_train_val_gap called")
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        print(f"     train_loss: {len(train_loss)} items")
        print(f"     val_loss:   {len(val_loss)} items")

        if not train_loss or not val_loss:
            print("  ⚠️ _plot_train_val_gap: SKIPPED (empty history)")
            return

        try:
            n = min(len(train_loss), len(val_loss))
            train_loss = train_loss[:n]
            val_loss = val_loss[:n]
            epochs = list(range(1, n + 1))
            gap = [v - t for t, v in zip(train_loss, val_loss)]

            avg_gap = sum(gap) / len(gap)
            final_gap = gap[-1]

            if avg_gap > 0.3:
                status, status_color = "STRONG OVERFITTING", "#dc2626"
            elif avg_gap > 0.15:
                status, status_color = "MODERATE OVERFITTING", "#f59e0b"
            elif avg_gap > 0.05:
                status, status_color = "MILD OVERFITTING", "#fbbf24"
            elif avg_gap > -0.05:
                status, status_color = "GOOD FIT", "#22c55e"
            elif avg_gap > -0.15:
                status, status_color = "MILD UNDERFITTING", "#fbbf24"
            else:
                status, status_color = "STRONG UNDERFITTING", "#dc2626"

            fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=200,
                                     gridspec_kw={"height_ratios": [2, 1.2]})

            axes[0].plot(epochs, train_loss, "o-", color="#3498db",
                         linewidth=2.5, markersize=5, label="Train Loss")
            axes[0].plot(epochs, val_loss, "s-", color="#e74c3c",
                         linewidth=2.5, markersize=5, label="Val Loss")
            axes[0].fill_between(epochs, train_loss, val_loss,
                                 alpha=0.2, color="#f39c12",
                                 label="Gap (Val - Train)")
            axes[0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
            axes[0].set_ylabel("Loss", fontsize=12, fontweight="bold")
            axes[0].set_title("Train vs Validation Loss — Gap Analysis",
                              fontsize=14, fontweight="bold", pad=12)
            axes[0].legend(loc="upper right", fontsize=11)
            axes[0].grid(alpha=0.3, linestyle="--")

            max_gap = max(gap)
            min_gap = min(gap)
            textstr = (f"Gap Statistics:\n  Max:   {max_gap:+.4f}\n"
                       f"  Min:   {min_gap:+.4f}\n  Avg:   {avg_gap:+.4f}\n"
                       f"  Final: {final_gap:+.4f}")
            axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes,
                         fontsize=10, verticalalignment="top",
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9e6",
                                   edgecolor="#f39c12", alpha=0.95, linewidth=1.5))

            if val_loss:
                best_epoch_idx = int(np.argmin(val_loss))
                axes[0].axvline(x=epochs[best_epoch_idx], color="#2ecc71",
                                linestyle="--", linewidth=2, alpha=0.7)
                axes[0].text(epochs[best_epoch_idx], axes[0].get_ylim()[1] * 0.95,
                             f"  Best Epoch: {epochs[best_epoch_idx]}",
                             color="#2ecc71", fontsize=9, fontweight="bold")

            colors = ["#e74c3c" if g > 0 else "#2ecc71" for g in gap]
            axes[1].bar(epochs, gap, color=colors, alpha=0.75,
                        edgecolor="black", linewidth=0.5)
            axes[1].axhline(y=0, color="black", linewidth=1.5)
            axes[1].axhline(y=avg_gap, color="#f39c12", linewidth=2,
                            linestyle="--", label=f"Avg Gap: {avg_gap:+.4f}")
            axes[1].axhline(y=0.05, color="#22c55e", linewidth=1, linestyle=":", alpha=0.5)
            axes[1].axhline(y=-0.05, color="#22c55e", linewidth=1, linestyle=":", alpha=0.5)
            axes[1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
            axes[1].set_ylabel("Gap (Val - Train)", fontsize=12, fontweight="bold")
            axes[1].set_title("Overfitting / Underfitting Indicator",
                              fontsize=13, fontweight="bold", pad=10)
            axes[1].legend(loc="upper right", fontsize=10)
            axes[1].grid(alpha=0.3, linestyle="--", axis="y")
            axes[1].text(0.98, 0.95, f"Status: {status}", transform=axes[1].transAxes,
                         fontsize=11, fontweight="bold", color="white",
                         verticalalignment="top", horizontalalignment="right",
                         bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color,
                                   edgecolor="black", linewidth=1.5, alpha=0.9))

            plt.suptitle("Training Diagnostics — Loss & Gap Analysis",
                         fontsize=15, fontweight="bold", y=1.00)
            plt.tight_layout()
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  ✓ train_val_gap.png saved ({n} epochs, status={status})")
        except Exception as e:
            print(f"  ❌ _plot_train_val_gap FAILED: {e}")
            import traceback
            traceback.print_exc()

    def _plot_lr_schedule(self, history, save_path):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        ep = range(1, len(history["learning_rates"]) + 1)
        ax.plot(ep, history["learning_rates"], "o-",
                linewidth=2, markersize=4, color="#f39c12")
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Learning Rate", fontweight="bold")
        ax.set_title("VGG16 — Learning Rate Schedule",
                     fontsize=14, fontweight="bold", pad=12)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ============================================================
    #  2. METRICS JSON
    # ============================================================

    def _save_metrics_json(self, report_dir):
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        m = self.primary_metrics

        data = {
            "model": "VGG16",
            "timestamp": datetime.now().isoformat(),
            "best_epoch": int(self.best_epoch),
            "training_time_sec": float(self.training_time_sec),
            "training_time_hms": self._format_time(self.training_time_sec),
            "class_names": list(self.class_names),
            "config": _clean(self.config),
            "metrics": _clean(m),
            "history_summary": {
                "total_epochs": len(self.history.get("train_loss", [])),
                "final_train_loss": float(self.history["train_loss"][-1]) if self.history.get("train_loss") else None,
                "final_val_loss": float(self.history["val_loss"][-1]) if self.history.get("val_loss") else None,
                "final_val_acc": float(self.history["val_acc"][-1]) if self.history.get("val_acc") else None,
                "best_val_acc": float(max(self.history["val_acc"])) if self.history.get("val_acc") else None,
            },
        }

        path = os.path.join(report_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=str)
        print(f"  ✓ {path}")

    # ============================================================
    #  3. SUMMARY TXT
    # ============================================================

    def _save_summary_txt(self, report_dir):
        m = self.primary_metrics
        wa = m.get("weighted avg", {})
        lines = []

        lines.append("=" * 70)
        lines.append("  VGG16 — TEST RESULTS SUMMARY")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Best Epoch: {self.best_epoch}")
        lines.append(f"  Training Time: {self._format_time(self.training_time_sec)}")
        lines.append(f"  Classes: {', '.join(self.class_names)}")
        lines.append("")

        # ---- Overall metrics ----
        lines.append("-" * 70)
        lines.append("  🎯 OVERALL METRICS")
        lines.append("-" * 70)
        lines.append(f"  {'Accuracy':30s}: {m.get('accuracy', 0):.4f}")
        lines.append(f"  {'Precision (weighted)':30s}: {wa.get('precision', 0):.4f}")
        lines.append(f"  {'Recall (weighted)':30s}: {wa.get('recall', 0):.4f}")
        lines.append(f"  {'F1-Score (weighted)':30s}: {wa.get('f1-score', 0):.4f}")

        if m.get("auc_macro") is not None:
            lines.append(f"  {'AUC (macro, OvR)':30s}: {m['auc_macro']:.4f}")
        lines.append("")

        # ---- Per-class ----
        lines.append("-" * 70)
        lines.append("  PER-CLASS METRICS")
        lines.append("-" * 70)
        lines.append(f"  {'Class':10s} {'Precision':>12s} {'Recall':>12s} {'F1':>12s} {'Support':>10s}")
        lines.append("  " + "-" * 60)
        for cname in self.class_names:
            if cname in m:
                cm = m[cname]
                lines.append(
                    f"  {cname:10s} {cm.get('precision', 0):12.4f} "
                    f"{cm.get('recall', 0):12.4f} {cm.get('f1-score', 0):12.4f} "
                    f"{int(cm.get('support', 0)):10d}"
                )
        lines.append("")

        # ---- Confusion matrix ----
        cm = m.get("confusion_matrix")
        if cm is not None:
            lines.append("-" * 70)
            lines.append("  CONFUSION MATRIX")
            lines.append("-" * 70)
            cm = np.array(cm)
            header = "  " + " " * 12 + "  ".join(f"{c:>8s}" for c in self.class_names)
            lines.append(header + "   (Predicted)")
            for i, cname in enumerate(self.class_names):
                row = "  " + f"{cname:12s}" + "  ".join(f"{v:>8d}" for v in cm[i])
                lines.append(row)
            lines.append("  " + " " * 12 + "  (True)")
            lines.append("")

        lines.append("=" * 70)
        lines.append("  Report generated by Liver AI Pipeline")
        lines.append("  developed by : babak yousefian")
        lines.append("=" * 70)

        path = os.path.join(report_dir, "summary.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  ✓ {path}")

    # ============================================================
    #  4. CLASSIFICATION REPORT
    # ============================================================

    def _save_classification_report(self, report_dir):
        report = self.primary_metrics.get("classification_report", "")
        path = os.path.join(report_dir, "classification_report.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("VGG16 — Classification Report (Test Set)\n")
            f.write("=" * 70 + "\n\n")
            f.write(report)
        print(f"  ✓ {path}")

    # ============================================================
    #  5. PREDICTIONS CSV
    # ============================================================

    def _save_predictions_csv(self, report_dir):
        if self.y_true is None or self.y_pred is None:
            return

        y_true = np.asarray(self.y_true)
        y_pred = np.asarray(self.y_pred)

        data = {
            "true_idx": y_true,
            "pred_idx": y_pred,
            "true_label": [self.class_names[int(i)] for i in y_true],
            "pred_label": [self.class_names[int(i)] for i in y_pred],
            "correct": (y_true == y_pred).astype(int),
        }

        if self.y_prob is not None:
            y_prob = np.asarray(self.y_prob)
            for i, cname in enumerate(self.class_names):
                data[f"prob_{cname}"] = y_prob[:, i]
            data["confidence"] = y_prob.max(axis=1)

        df = pd.DataFrame(data)
        path = os.path.join(report_dir, "predictions.csv")
        df.to_csv(path, index=False)
        print(f"  ✓ {path}")

    # ============================================================
    #  6. PDF REPORT
    # ============================================================

    def _generate_pdf(self, report_dir):
        pdf_path = os.path.join(report_dir, "VGG16_Report.pdf")
        m = self.primary_metrics

        if not HAS_FITZ:
            print("⚠️ PyMuPDF not available. Skipping PDF. Install: pip install PyMuPDF")
            return pdf_path

        # ---- Extract test metrics ----
        accuracy = m.get("accuracy", 0)
        precision = m.get("precision_macro", 0)
        recall = m.get("recall_macro", 0)
        f1 = m.get("f1_macro", 0)
        auc_macro = m.get("auc_macro", 0)
        kappa = m.get("kappa", 0)
        mcc = m.get("mcc", 0)

        doc = fitz.open()

        # ============================================================
        #  PAGE 1: COVER
        # ============================================================
        page = doc.new_page(width=595, height=842)
        self._page_cover(page, m)
        self._page_footer(page, 1)

        # ============================================================
        #  PAGE 2: EXECUTIVE SUMMARY (TEST METRICS)
        # ============================================================
        page = doc.new_page(width=595, height=842)
        self._page_executive_summary(page, m)
        self._page_footer(page, 2)

        # ============================================================
        #  PAGE 3: CONFIGURATION
        # ============================================================
        page = doc.new_page(width=595, height=842)
        self._page_configuration(page)
        self._page_footer(page, 3)

        # ============================================================
        #  PAGE 4: PER-CLASS METRICS
        # ============================================================
        page = doc.new_page(width=595, height=842)
        self._page_per_class(page, m)
        self._page_footer(page, 4)

        # ============================================================
        #  PAGE 5: CONFUSION MATRIX
        # ============================================================
        cm_path = os.path.join(report_dir, "confusion_matrix.png")
        if os.path.exists(cm_path):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), "CONFUSION MATRIX (Normalized)",
                             fontsize=16, fontname="helvetica-bold",
                             color=(0.1, 0.1, 0.2))
            page.insert_image(fitz.Rect(60, 90, 535, 660), filename=cm_path)
            self._page_footer(page, 5)

        # ============================================================
        #  PAGE 6: ROC CURVES
        # ============================================================
        roc_path = os.path.join(report_dir, "roc_curves.png")
        if os.path.exists(roc_path):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), "ROC CURVES (One-vs-Rest)",
                             fontsize=16, fontname="helvetica-bold",
                             color=(0.1, 0.1, 0.2))
            page.insert_image(fitz.Rect(60, 90, 535, 660), filename=roc_path)
            self._page_footer(page, 6)

        # ============================================================
        #  PAGE 7: METRICS BAR
        # ============================================================
        bar_path = os.path.join(report_dir, "metrics_bar.png")
        if os.path.exists(bar_path):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), "PERFORMANCE METRICS",
                             fontsize=16, fontname="helvetica-bold",
                             color=(0.1, 0.1, 0.2))
            page.insert_image(fitz.Rect(60, 90, 535, 660), filename=bar_path)
            self._page_footer(page, 7)

        # ============================================================
        #  PAGE 8: PER-CLASS METRICS PLOT
        # ============================================================
        pcm_path = os.path.join(report_dir, "per_class_metrics.png")
        if os.path.exists(pcm_path):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), "PER-CLASS METRICS",
                             fontsize=16, fontname="helvetica-bold",
                             color=(0.1, 0.1, 0.2))
            page.insert_image(fitz.Rect(60, 90, 535, 660), filename=pcm_path)
            self._page_footer(page, 8)

        # ============================================================
        #  PAGE 9: PIE CHART
        # ============================================================
        pie_path = os.path.join(report_dir, "performance_pie.png")
        if os.path.exists(pie_path):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), "PERFORMANCE DISTRIBUTION",
                             fontsize=16, fontname="helvetica-bold",
                             color=(0.1, 0.1, 0.2))
            page.insert_image(fitz.Rect(110, 150, 485, 525), filename=pie_path)
            self._page_footer(page, 9)

        # ============================================================
        #  PAGE: Train vs Val Loss + Gap (NEW)
        # ============================================================
        gap_path = os.path.join(report_dir, "train_val_gap.png")
        if os.path.exists(gap_path):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), "TRAIN vs VAL LOSS — GAP ANALYSIS",
                             fontsize=16, fontname="helvetica-bold",
                             color=(0, 0.55, 0.85))
            page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62),
                           color=(0, 0.83, 1), width=2)
            page.insert_text((50, 85),
                             "Gap = Val Loss - Train Loss   |   "
                             "Gap > 0 → Overfitting   |   Gap < 0 → Underfitting",
                             fontsize=9, fontname="helvetica-oblique",
                             color=(0.4, 0.4, 0.5))
            page.insert_image(fitz.Rect(50, 110, 545, 720), filename=gap_path)
            self._page_footer(page, 10)

        # ============================================================
        #  PAGE 11: TRAINING HISTORY + LR
        # ============================================================
        hist_path = os.path.join(report_dir, "training_history.png")
        lr_path = os.path.join(report_dir, "learning_rate.png")
        if os.path.exists(hist_path) and os.path.exists(lr_path):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), "TRAINING CURVES",
                             fontsize=16, fontname="helvetica-bold",
                             color=(0.1, 0.1, 0.2))
            page.insert_text((50, 90), "Training History",
                             fontsize=13, fontname="helvetica-bold",
                             color=(0, 0.55, 0.85))
            page.insert_image(fitz.Rect(50, 110, 545, 380), filename=hist_path)
            page.insert_text((50, 410), "Learning Rate Schedule",
                             fontsize=13, fontname="helvetica-bold",
                             color=(0, 0.55, 0.85))
            page.insert_image(fitz.Rect(50, 430, 545, 700), filename=lr_path)
            self._page_footer(page, 11)

        # ============================================================
        #  PAGE 12: CONFUSION MATRIX (RAW)
        # ============================================================
        cm_raw = os.path.join(report_dir, "confusion_matrix_raw.png")
        if os.path.exists(cm_raw):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 50), "CONFUSION MATRIX (Raw Counts)",
                             fontsize=16, fontname="helvetica-bold",
                             color=(0.1, 0.1, 0.2))
            page.insert_image(fitz.Rect(60, 90, 535, 660), filename=cm_raw)
            self._page_footer(page, 12)

        # ============================================================
        #  SAVE
        # ============================================================
        doc.save(pdf_path)
        total = len(doc)
        doc.close()
        print(f"  ✓ {pdf_path} ({total} pages)")
        return pdf_path

    # ============================================================
    #  PDF PAGE BUILDERS
    # ============================================================

    def _page_cover(self, page, m):
        page.draw_rect(
            fitz.Rect(0, 0, 595, 200),
            color=None, fill=(0.1, 0.1, 0.18), width=0,
        )
        page.insert_text((50, 80), "VGG16",
                         fontsize=36, fontname="helvetica-bold",
                         color=(0, 0.83, 1))
        page.insert_text((50, 130), "Fine-Tuning Report",
                         fontsize=22, fontname="helvetica",
                         color=(1, 1, 1))
        page.insert_text((50, 165),
                         "Liver Ultrasound Classification (NL / G1 / G2 / G3)",
                         fontsize=12, fontname="helvetica",
                         color=(0.65, 0.75, 0.85))

        acc = m.get("accuracy", 0)
        page.draw_rect(
            fitz.Rect(50, 230, 545, 320),
            color=(0, 0.83, 1), fill=(0.9, 0.97, 1), width=2,
        )
        page.insert_text((70, 265), "🎯 TEST ACCURACY",
                         fontsize=14, fontname="helvetica-bold",
                         color=(0.1, 0.1, 0.2))
        page.insert_text((70, 305), f"{acc * 100:.2f}%",
                         fontsize=32, fontname="helvetica-bold",
                         color=(0, 0.55, 0.85))
        page.insert_text((280, 300), "(on unseen test set)",
                         fontsize=11, fontname="helvetica-oblique",
                         color=(0.4, 0.4, 0.5))

        # ✅ Test sample count
        total_samples = 0
        for cname in self.class_names:
            if cname in m:
                total_samples += m[cname].get("support", 0)

        info = [
            ("Model", "VGG16 (ImageNet pretrained)"),
            ("Task", "4-Class Liver Ultrasound Classification"),
            ("Test Samples", str(int(total_samples))),
            ("Best Epoch", str(self.best_epoch)),
            ("Training Time", self._format_time(self.training_time_sec)),
            ("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Author", "Babak Yousefian"),
        ]
        y = 370
        for key, val in info:
            page.draw_rect(fitz.Rect(50, y, 545, y + 28),
                           color=None, fill=(0.97, 0.98, 1))
            page.insert_text((60, y + 19), key,
                             fontsize=11, fontname="helvetica-bold",
                             color=(0.2, 0.2, 0.3))
            page.insert_text((200, y + 19), val,
                             fontsize=11, fontname="helvetica",
                             color=(0.1, 0.1, 0.2))
            y += 32

        page.insert_text((50, 800), "developed by : babak yousefian",
                         fontsize=10, fontname="helvetica-oblique",
                         color=(0.5, 0.5, 0.6))

    def _page_executive_summary(self, page, m):
        page.insert_text((50, 50), "EXECUTIVE SUMMARY",
                         fontsize=18, fontname="helvetica-bold",
                         color=(0, 0.55, 0.85))
        page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62),
                       color=(0, 0.83, 1), width=2)

        wa = m.get("weighted avg", {})
        y = 100
        page.insert_text((50, y),
                         "🎯 TEST SET PERFORMANCE (Primary Metric)",
                         fontsize=13, fontname="helvetica-bold",
                         color=(0.1, 0.1, 0.2))
        y += 30

        metrics_data = [
            ("Accuracy", m.get("accuracy", 0)),
            ("Precision (weighted)", wa.get("precision", 0)),
            ("Recall (weighted)", wa.get("recall", 0)),
            ("F1-Score (weighted)", wa.get("f1-score", 0)),
        ]
        if m.get("auc_macro") is not None:
            metrics_data.append(("AUC (macro, OvR)", m["auc_macro"]))

        for label, val in metrics_data:
            page.draw_rect(fitz.Rect(50, y, 545, y + 26),
                           color=(0.8, 0.85, 0.9), fill=(0.97, 0.98, 1),
                           width=0.5)
            page.insert_text((60, y + 18), label,
                             fontsize=11, fontname="helvetica",
                             color=(0.15, 0.15, 0.2))
            page.insert_text((400, y + 18), f"{val:.4f}",
                             fontsize=12, fontname="helvetica-bold",
                             color=(0, 0.55, 0.85))
            page.insert_text((480, y + 18), f"({val * 100:.2f}%)",
                             fontsize=10, fontname="helvetica",
                             color=(0.4, 0.4, 0.5))
            y += 30

        y += 20
        page.insert_text((50, y), "📌 Notes:",
                         fontsize=11, fontname="helvetica-bold",
                         color=(0.1, 0.1, 0.2))
        y += 20
        notes = [
            "• All metrics are computed on the TEST set (unseen data).",
            "• AUC is computed using One-vs-Rest strategy with macro averaging.",
            "• All metrics are weighted for class imbalance.",
        ]
        for note in notes:
            page.insert_text((60, y), note,
                             fontsize=10, fontname="helvetica",
                             color=(0.3, 0.3, 0.4))
            y += 18

    def _page_configuration(self, page):
        page.insert_text((50, 50), "CONFIGURATION",
                         fontsize=18, fontname="helvetica-bold",
                         color=(0, 0.55, 0.85))
        page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62),
                       color=(0, 0.83, 1), width=2)

        y = 100
        config_items = [
            ("Input Size", self.config.get("input_size", "N/A")),
            ("Batch Size", self.config.get("batch_size", "N/A")),
            ("Accumulation Steps", self.config.get("accumulation_steps", "N/A")),
            ("Effective Batch", self.config.get("effective_batch_size", "N/A")),
            ("Epochs (Stage 1)", self.config.get("epochs_stage1", "N/A")),
            ("Epochs (Stage 2)", self.config.get("epochs_stage2", "N/A")),
            ("LR (Stage 1)", self.config.get("lr_stage1", "N/A")),
            ("LR (Backbone)", self.config.get("lr_backbone", "N/A")),
            ("LR (Head)", self.config.get("lr_head", "N/A")),
            ("Weight Decay", self.config.get("weight_decay", "N/A")),
            ("Label Smoothing", self.config.get("label_smoothing", "N/A")),
            ("Gradient Clip", self.config.get("grad_clip", "N/A")),
            ("Dropout", self.config.get("dropout", "N/A")),
            ("Hidden Dim", self.config.get("hidden_dim", "N/A")),
            ("Scheduler", self.config.get("scheduler", "N/A")),
            ("Scheduler T0", self.config.get("scheduler_T0", "N/A")),
            ("Scheduler Tmult", self.config.get("scheduler_Tmult", "N/A")),
            ("Mixed Precision", self.config.get("use_amp", "N/A")),
            ("Weighted Sampler", self.config.get("use_weighted_sampler", "N/A")),
            ("Seed", self.config.get("seed", "N/A")),
            ("Total Params", self.config.get("total_params", "N/A")),
            ("Trainable Params", self.config.get("trainable_params", "N/A")),
        ]

        for label, val in config_items:
            page.draw_rect(fitz.Rect(50, y, 545, y + 24),
                           color=(0.85, 0.9, 0.95), fill=(0.98, 0.99, 1),
                           width=0.4)
            page.insert_text((60, y + 17), str(label),
                             fontsize=10, fontname="helvetica",
                             color=(0.2, 0.2, 0.3))
            page.insert_text((280, y + 17), str(val),
                             fontsize=10, fontname="helvetica-bold",
                             color=(0.1, 0.1, 0.2))
            y += 28

    def _page_per_class(self, page, m):
        page.insert_text((50, 50), "PER-CLASS METRICS (Test Set)",
                         fontsize=18, fontname="helvetica-bold",
                         color=(0, 0.55, 0.85))
        page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62),
                       color=(0, 0.83, 1), width=2)

        y = 100
        page.draw_rect(fitz.Rect(50, y, 545, y + 30),
                       color=None, fill=(0, 0.55, 0.85))
        headers = [("Class", 60), ("Precision", 170),
                   ("Recall", 270), ("F1-Score", 370), ("Support", 470)]
        for htext, x in headers:
            page.insert_text((x, y + 20), htext,
                             fontsize=11, fontname="helvetica-bold",
                             color=(1, 1, 1))
        y += 30

        for i, cname in enumerate(self.class_names):
            if cname not in m:
                continue
            cm_data = m[cname]
            bg = (0.97, 0.98, 1) if i % 2 == 0 else (1, 1, 1)
            page.draw_rect(fitz.Rect(50, y, 545, y + 28),
                           color=(0.85, 0.9, 0.95), fill=bg, width=0.4)
            page.insert_text((60, y + 19), cname,
                             fontsize=11, fontname="helvetica-bold",
                             color=(0.15, 0.15, 0.2))
            page.insert_text((180, y + 19),
                             f"{cm_data.get('precision', 0):.4f}",
                             fontsize=11, fontname="helvetica",
                             color=(0.1, 0.1, 0.2))
            page.insert_text((280, y + 19),
                             f"{cm_data.get('recall', 0):.4f}",
                             fontsize=11, fontname="helvetica",
                             color=(0.1, 0.1, 0.2))
            page.insert_text((380, y + 19),
                             f"{cm_data.get('f1-score', 0):.4f}",
                             fontsize=11, fontname="helvetica",
                             color=(0.1, 0.1, 0.2))
            page.insert_text((480, y + 19),
                             str(cm_data.get('support', 0)),
                             fontsize=11, fontname="helvetica",
                             color=(0.1, 0.1, 0.2))
            y += 28

        # ---- Weighted avg ----
        y += 20
        page.insert_text((50, y), "Weighted Average:",
                         fontsize=12, fontname="helvetica-bold",
                         color=(0.1, 0.1, 0.2))
        y += 25
        wa = m.get("weighted avg", {})
        wa_items = [
            ("Precision", wa.get("precision", 0)),
            ("Recall", wa.get("recall", 0)),
            ("F1-Score", wa.get("f1-score", 0)),
        ]
        for k, v in wa_items:
            page.insert_text((70, y), f"{k}:",
                             fontsize=11, fontname="helvetica",
                             color=(0.2, 0.2, 0.3))
            page.insert_text((200, y), f"{v:.4f}",
                             fontsize=11, fontname="helvetica-bold",
                             color=(0, 0.55, 0.85))
            y += 22

    def _page_footer(self, page, page_num):
        page.draw_line(fitz.Point(50, 800), fitz.Point(545, 800),
                       color=(0.85, 0.9, 0.95), width=0.5)
        page.insert_text((50, 815),
                         f"VGG16 Report — Liver AI Pipeline  |  developed by : babak yousefian  |  Page {page_num}",
                         fontsize=8, fontname="helvetica-oblique",
                         color=(0.5, 0.5, 0.6))

    # ============================================================
    #  UTILS
    # ============================================================

    @staticmethod
    def _format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
