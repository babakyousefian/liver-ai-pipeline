# src/AI/ConvNext/balanced_report.py
# -*- coding: utf-8 -*-
"""
ConvNeXt Balanced Fine-Tune Report Generator
==============================================
Generates complete PDF with all plots and metrics.

Outputs:
- ConvNext_Balanced_Report.pdf
- confusion_matrix_baseline.png
- confusion_matrix_finetuned.png
- per_class_errors_comparison.png
- training_history.png
- gap_analysis.png
- roc_curves.png
- metrics.json
- summary.txt
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_curve, auc

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("⚠️ PyMuPDF not installed. Install: pip install PyMuPDF")

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
CLASS_COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]


class BalancedReporter:
    """Complete report generator for ConvNeXt Balanced Fine-Tune."""

    def __init__(self, results: dict):
        self.results = results
        self.output_dir = results["output_dir"]
        self.history = results["history"]
        self.cm_baseline = np.array(results["confusion_matrix_baseline"])
        self.cm_finetuned = np.array(results["confusion_matrix_finetuned"])
        self.y_test = results["y_test"]
        self.y_pred_base = results["y_pred_baseline"]
        self.y_pred_ft = results["y_pred_finetuned"]
        self.y_prob_base = results["y_prob_baseline"]
        self.y_prob_ft = results["y_prob_finetuned"]
        self.base_acc = results["baseline_acc"]
        self.ft_acc = results["finetuned_acc"]
        self.base_bal_acc = results["baseline_bal_acc"]
        self.ft_bal_acc = results["finetuned_bal_acc"]
        self.class_weights = results["class_weights"]

    def generate_all(self):
        print("\n" + "=" * 70)
        print("  📄 GENERATING BALANCED FINE-TUNE REPORT")
        print("=" * 70)

        print("\n[1/7] Plotting confusion matrices...")
        self._plot_confusion(self.cm_baseline, "Baseline",
                             "confusion_matrix_baseline.png")
        self._plot_confusion(self.cm_finetuned, "Fine-Tuned",
                             "confusion_matrix_finetuned.png")

        print("[2/7] Plotting per-class errors comparison...")
        self._plot_per_class_comparison()

        print("[3/7] Plotting training history...")
        self._plot_training_history()

        print("[4/7] Plotting gap analysis...")
        self._plot_gap_analysis()

        print("[5/7] Plotting ROC curves...")
        self._plot_roc_curves()

        print("[6/7] Saving metrics JSON + summary TXT...")
        self._save_metrics_json()
        self._save_summary_txt()

        print("[7/7] Generating PDF report...")
        pdf_path = self._generate_pdf()

        print("\n" + "=" * 70)
        print(f"  ✅ ALL REPORTS SAVED TO: {self.output_dir}")
        print("=" * 70)
        return pdf_path

    # ═══════════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════════
    def _plot_confusion(self, cm, title_prefix, filename):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar=True, square=True, linewidths=0.5,
            annot_kws={"size": 13, "fontweight": "bold"}, ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_ylabel("True", fontsize=12, fontweight="bold")
        ax.set_title(f"{title_prefix} — Confusion Matrix (Normalized)",
                     fontsize=14, fontweight="bold", pad=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"   ✓ {filename}")

    def _plot_per_class_comparison(self):
        """Bar chart comparing baseline vs fine-tuned per-class errors."""
        errors_base = []
        errors_ft = []
        for c in range(4):
            total = max(int((self.y_test == c).sum()), 1)
            errors_base.append((total - self.cm_baseline[c, c]) / total * 100)
            errors_ft.append((total - self.cm_finetuned[c, c]) / total * 100)

        x = np.arange(4)
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        bars1 = ax.bar(x - w/2, errors_base, w, label="Baseline",
                       color="#3498db", edgecolor="black", linewidth=1.2)
        bars2 = ax.bar(x + w/2, errors_ft, w, label="Fine-Tuned",
                       color="#2ecc71", edgecolor="black", linewidth=1.2)

        for bars, vals in [(bars1, errors_base), (bars2, errors_ft)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

        ax.axhline(y=3, color="red", linestyle="--", linewidth=2,
                   label="Target (< 3%)")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylabel("Error Rate (%)", fontsize=12, fontweight="bold")
        ax.set_title("Per-Class Error Rates: Baseline vs Fine-Tuned",
                     fontsize=14, fontweight="bold", pad=12)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "per_class_errors_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"   ✓ per_class_errors_comparison.png")

    def _plot_training_history(self):
        """Training loss, val loss, val acc, bal acc over epochs."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], "o-",
                     label="Train Loss", linewidth=2, markersize=4, color="#e74c3c")
        axes[0].plot(epochs, self.history["val_loss"], "s-",
                     label="Val Loss", linewidth=2, markersize=4, color="#3498db")
        axes[0].set_xlabel("Epoch", fontweight="bold")
        axes[0].set_ylabel("Loss", fontweight="bold")
        axes[0].set_title("Loss History", fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, self.history["val_acc"], "s-",
                     label="Val Accuracy", linewidth=2.5,
                     markersize=4, color="#2ecc71")
        axes[1].plot(epochs, self.history["val_bal_acc"], "^-",
                     label="Val Balanced Accuracy", linewidth=2.5,
                     markersize=4, color="#9b59b6")
        axes[1].set_xlabel("Epoch", fontweight="bold")
        axes[1].set_ylabel("Accuracy", fontweight="bold")
        axes[1].set_title("Validation Accuracy", fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 1.05)

        plt.suptitle("ConvNeXt Balanced Fine-Tune — Training History",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "training_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"   ✓ training_history.png")

    def _plot_gap_analysis(self):
        """Train vs Val Loss + Gap analysis."""
        train_loss = self.history["train_loss"]
        val_loss = self.history["val_loss"]

        if not train_loss or not val_loss:
            return

        gap = [v - t for t, v in zip(train_loss, val_loss)]
        epochs = list(range(1, len(gap) + 1))
        avg_gap = np.mean(gap)
        final_gap = gap[-1]

        fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=150,
                                gridspec_kw={"height_ratios": [2, 1.2]})

        # Top: losses + gap fill
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

        textstr = (f"Gap Statistics:\n"
                   f"  Max:   {max(gap):+.4f}\n"
                   f"  Min:   {min(gap):+.4f}\n"
                   f"  Avg:   {avg_gap:+.4f}\n"
                   f"  Final: {final_gap:+.4f}")
        axes[0].text(0.02, 0.98, textstr,
                     transform=axes[0].transAxes,
                     fontsize=10, verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor="#fff9e6", edgecolor="#f39c12",
                               alpha=0.95, linewidth=1.5))

        # Bottom: gap bars
        colors = ["#e74c3c" if g > 0 else "#2ecc71" for g in gap]
        axes[1].bar(epochs, gap, color=colors, alpha=0.75,
                    edgecolor="black", linewidth=0.5)
        axes[1].axhline(y=0, color="black", linewidth=1.5)
        axes[1].axhline(y=avg_gap, color="#f39c12",
                        linewidth=2, linestyle="--",
                        label=f"Avg Gap: {avg_gap:+.4f}")
        axes[1].axhline(y=0.15, color="#22c55e", linewidth=1,
                        linestyle=":", alpha=0.5, label="Good Fit Zone (±0.15)")
        axes[1].axhline(y=-0.15, color="#22c55e", linewidth=1,
                        linestyle=":", alpha=0.5)
        axes[1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Gap (Val - Train)", fontsize=12, fontweight="bold")
        axes[1].set_title("Overfitting / Underfitting Indicator",
                          fontsize=13, fontweight="bold", pad=10)
        axes[1].legend(loc="upper right", fontsize=10)
        axes[1].grid(alpha=0.3, linestyle="--", axis="y")

        # Status
        if avg_gap > 0.5:
            status, color = "STRONG OVERFITTING", "#dc2626"
        elif avg_gap > 0.3:
            status, color = "MODERATE OVERFITTING", "#f59e0b"
        elif avg_gap > 0.15:
            status, color = "MILD OVERFITTING", "#fbbf24"
        elif avg_gap > -0.15:
            status, color = "GOOD FIT", "#22c55e"
        else:
            status, color = "UNDERFITTING", "#dc2626"

        axes[1].text(0.98, 0.95, f"Status: {status}",
                     transform=axes[1].transAxes,
                     fontsize=11, fontweight="bold",
                     color="white", verticalalignment="top",
                     horizontalalignment="right",
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor=color, edgecolor="black",
                               linewidth=1.5, alpha=0.9))

        plt.suptitle("Training Diagnostics — Loss & Gap Analysis",
                     fontsize=15, fontweight="bold", y=1.00)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "gap_analysis.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"   ✓ gap_analysis.png")

    def _plot_roc_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        for ax, y_prob, title in [
            (axes[0], self.y_prob_base, "Baseline"),
            (axes[1], self.y_prob_ft, "Fine-Tuned"),
        ]:
            for i, name in enumerate(CLASS_NAMES):
                y_bin = (self.y_test == i).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, linewidth=2.5,
                        color=CLASS_COLORS[i],
                        label=f"{name} (AUC={roc_auc:.4f})")
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.6)
            ax.set_xlabel("False Positive Rate", fontweight="bold")
            ax.set_ylabel("True Positive Rate", fontweight="bold")
            ax.set_title(f"{title} — ROC Curves", fontweight="bold")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])

        plt.suptitle("ROC Curves: Baseline vs Fine-Tuned",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "roc_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"   ✓ roc_curves.png")

    # ═══════════════════════════════════════════════════════════
    # JSON / TXT
    # ═══════════════════════════════════════════════════════════
    def _save_metrics_json(self):
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

        data = {
            "model": "ConvNeXt-Balanced-FineTune",
            "timestamp": datetime.now().isoformat(),
            "baseline": {
                "accuracy": float(self.base_acc),
                "balanced_accuracy": float(self.base_bal_acc),
                "confusion_matrix": self.cm_baseline.tolist(),
            },
            "finetuned": {
                "accuracy": float(self.ft_acc),
                "balanced_accuracy": float(self.ft_bal_acc),
                "confusion_matrix": self.cm_finetuned.tolist(),
            },
            "class_weights": [float(w) for w in self.class_weights],
            "per_class_error_baseline": [
                float((int((self.y_test == c).sum()) - self.cm_baseline[c, c]) /
                      max(int((self.y_test == c).sum()), 1))
                for c in range(4)
            ],
            "per_class_error_finetuned": [
                float((int((self.y_test == c).sum()) - self.cm_finetuned[c, c]) /
                      max(int((self.y_test == c).sum()), 1))
                for c in range(4)
            ],
        }
        path = os.path.join(self.output_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_clean(data), f, indent=2, ensure_ascii=False)
        print(f"   ✓ metrics.json")

    def _save_summary_txt(self):
        lines = []
        lines.append("=" * 70)
        lines.append("  CONVNEXT BALANCED FINE-TUNE — SUMMARY")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("  🎯 BASELINE vs FINE-TUNED")
        lines.append("-" * 70)
        lines.append(f"  {'Metric':30s} {'Baseline':>12s} {'Fine-Tuned':>14s} {'Δ':>10s}")
        lines.append("  " + "-" * 66)
        d_acc = self.ft_acc - self.base_acc
        d_bal = self.ft_bal_acc - self.base_bal_acc
        lines.append(f"  {'Accuracy':30s} {self.base_acc:12.4f} {self.ft_acc:14.4f} "
                     f"{d_acc:+10.4f}")
        lines.append(f"  {'Balanced Accuracy':30s} {self.base_bal_acc:12.4f} "
                     f"{self.ft_bal_acc:14.4f} {d_bal:+10.4f}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("  PER-CLASS ERROR RATES")
        lines.append("-" * 70)
        lines.append(f"  {'Class':10s} {'Baseline':>12s} {'Fine-Tuned':>14s} {'Δ':>10s}")
        lines.append("  " + "-" * 60)
        for c in range(4):
            total = max(int((self.y_test == c).sum()), 1)
            eb = (total - self.cm_baseline[c, c]) / total * 100
            ef = (total - self.cm_finetuned[c, c]) / total * 100
            d = ef - eb
            lines.append(f"  {CLASS_NAMES[c]:10s} {eb:11.2f}% {ef:13.2f}% "
                         f"{d:+9.2f}%")
        lines.append("")
        lines.append("-" * 70)
        lines.append("  CLASS-BALANCED WEIGHTS")
        lines.append("-" * 70)
        for i, w in enumerate(self.class_weights):
            lines.append(f"  {CLASS_NAMES[i]}: {w:.3f}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("  Report generated by Liver AI Pipeline")
        lines.append("  developed by : babak yousefian")
        lines.append("=" * 70)

        path = os.path.join(self.output_dir, "summary.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"   ✓ summary.txt")

    # ═══════════════════════════════════════════════════════════
    # PDF
    # ═══════════════════════════════════════════════════════════
    def _generate_pdf(self):
        if not HAS_FITZ:
            print("   ⚠️ PyMuPDF not available. Skipping PDF.")
            return None

        pdf_path = os.path.join(self.output_dir, "ConvNext_Balanced_Report.pdf")
        doc = fitz.open()

        # Page 1: Cover
        page = doc.new_page(width=595, height=842)
        self._page_cover(page)
        self._page_footer(page, 1)

        # Page 2: Executive Summary
        page = doc.new_page(width=595, height=842)
        self._page_summary(page)
        self._page_footer(page, 2)

        # Page 3+: Charts
        chart_pages = [
            ("confusion_matrix_baseline.png", "Confusion Matrix — Baseline"),
            ("confusion_matrix_finetuned.png", "Confusion Matrix — Fine-Tuned"),
            ("per_class_errors_comparison.png", "Per-Class Error Comparison"),
            ("training_history.png", "Training History"),
            ("gap_analysis.png", "Train vs Val Loss — Gap Analysis"),
            ("roc_curves.png", "ROC Curves Comparison"),
        ]

        page_num = 3
        for fname, title in chart_pages:
            p = os.path.join(self.output_dir, fname)
            if os.path.exists(p):
                page = doc.new_page(width=595, height=842)
                page.insert_text((50, 50), title,
                                 fontsize=16, fontname="helvetica-bold",
                                 color=(0.1, 0.1, 0.2))
                page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62),
                               color=(0, 0.83, 1), width=2)
                page.insert_image(fitz.Rect(50, 80, 545, 760), filename=p)
                self._page_footer(page, page_num)
                page_num += 1

        doc.save(pdf_path)
        total = len(doc)
        doc.close()
        print(f"   ✓ ConvNext_Balanced_Report.pdf ({total} pages)")
        return pdf_path

    def _page_cover(self, page):
        # Top banner
        page.draw_rect(fitz.Rect(0, 0, 595, 220),
                       color=None, fill=(0.1, 0.1, 0.18), width=0)
        page.insert_text((50, 80), "ConvNeXt",
                         fontsize=42, fontname="helvetica-bold",
                         color=(0, 0.83, 1))
        page.insert_text((50, 130), "Balanced Fine-Tune Report",
                         fontsize=24, fontname="helvetica", color=(1, 1, 1))
        page.insert_text((50, 170),
                         "Class-Balanced Weighted Loss · 4-Class Liver Ultrasound",
                         fontsize=11, fontname="helvetica",
                         color=(0.65, 0.75, 0.85))

        # Big delta card
        d_acc = (self.ft_acc - self.base_acc) * 100
        delta_color = (0, 0.7, 0.3) if d_acc >= 0 else (0.9, 0.2, 0.2)
        page.draw_rect(fitz.Rect(50, 250, 545, 400),
                       color=(0, 0.83, 1), fill=(0.9, 0.97, 1), width=2)
        page.insert_text((70, 285), "🎯 TEST ACCURACY",
                         fontsize=13, fontname="helvetica-bold",
                         color=(0.1, 0.1, 0.2))
        page.insert_text((70, 320), f"Baseline:  {self.base_acc * 100:.2f}%",
                         fontsize=16, fontname="helvetica",
                         color=(0.3, 0.3, 0.4))
        page.insert_text((70, 360), f"Fine-Tuned: {self.ft_acc * 100:.2f}%",
                         fontsize=22, fontname="helvetica-bold",
                         color=delta_color)
        page.insert_text((70, 390), f"Δ = {d_acc:+.2f}%",
                         fontsize=14, fontname="helvetica-bold",
                         color=delta_color)

        # Info table
        info = [
            ("Model", "ConvNeXt-Tiny + Class-Balanced Loss"),
            ("Strategy", "Weighted CrossEntropy (β=0.999)"),
            ("Training", "12 epochs, LR=1e-5, Mixup+CutMix"),
            ("Test Samples", f"{len(self.y_test)}"),
            ("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Author", "Babak Yousefian"),
        ]
        y = 440
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

    def _page_summary(self, page):
        page.insert_text((50, 50), "EXECUTIVE SUMMARY",
                         fontsize=18, fontname="helvetica-bold",
                         color=(0, 0.55, 0.85))
        page.draw_line(fitz.Point(50, 62), fitz.Point(545, 62),
                       color=(0, 0.83, 1), width=2)

        y = 100
        page.insert_text((50, y), "📊 Baseline vs Fine-Tuned Comparison",
                         fontsize=13, fontname="helvetica-bold",
                         color=(0.1, 0.1, 0.2))
        y += 30

        # Header
        page.draw_rect(fitz.Rect(50, y, 545, y + 30),
                       color=None, fill=(0, 0.55, 0.85))
        for htext, x in [("Metric", 60), ("Baseline", 250),
                         ("Fine-Tuned", 370), ("Δ", 490)]:
            page.insert_text((x, y + 20), htext,
                             fontsize=11, fontname="helvetica-bold",
                             color=(1, 1, 1))
        y += 30

        # Rows
        d_acc = self.ft_acc - self.base_acc
        d_bal = self.ft_bal_acc - self.base_bal_acc
        rows = [
            ("Accuracy", f"{self.base_acc*100:.2f}%",
             f"{self.ft_acc*100:.2f}%", f"{d_acc*100:+.2f}%"),
            ("Balanced Acc", f"{self.base_bal_acc*100:.2f}%",
             f"{self.ft_bal_acc*100:.2f}%", f"{d_bal*100:+.2f}%"),
        ]
        for i, (m, b, f, d) in enumerate(rows):
            bg = (0.97, 0.98, 1) if i % 2 == 0 else (1, 1, 1)
            page.draw_rect(fitz.Rect(50, y, 545, y + 28),
                           color=(0.85, 0.9, 0.95), fill=bg, width=0.4)
            page.insert_text((60, y + 19), m,
                             fontsize=11, fontname="helvetica",
                             color=(0.15, 0.15, 0.2))
            page.insert_text((250, y + 19), b,
                             fontsize=11, fontname="helvetica",
                             color=(0.1, 0.1, 0.2))
            page.insert_text((370, y + 19), f,
                             fontsize=11, fontname="helvetica-bold",
                             color=(0, 0.55, 0.85))
            page.insert_text((490, y + 19), d,
                             fontsize=11, fontname="helvetica-bold",
                             color=(0, 0.7, 0.3) if d.startswith("+") else
                                   (0.9, 0.2, 0.2) if d.startswith("-") else
                                   (0.3, 0.3, 0.3))
            y += 28

        # Per-class errors
        y += 25
        page.insert_text((50, y), "📊 Per-Class Error Rates",
                         fontsize=13, fontname="helvetica-bold",
                         color=(0.1, 0.1, 0.2))
        y += 30

        page.draw_rect(fitz.Rect(50, y, 545, y + 30),
                       color=None, fill=(0, 0.55, 0.85))
        for htext, x in [("Class", 60), ("Baseline", 200),
                         ("Fine-Tuned", 330), ("Δ", 470)]:
            page.insert_text((x, y + 20), htext,
                             fontsize=11, fontname="helvetica-bold",
                             color=(1, 1, 1))
        y += 30

        for i, c in enumerate(CLASS_NAMES):
            total = max(int((self.y_test == c).sum()), 1)
            eb = (total - self.cm_baseline[i, i]) / total * 100
            ef = (total - self.cm_finetuned[i, i]) / total * 100
            d = ef - eb
            bg = (0.97, 0.98, 1) if i % 2 == 0 else (1, 1, 1)
            page.draw_rect(fitz.Rect(50, y, 545, y + 28),
                           color=(0.85, 0.9, 0.95), fill=bg, width=0.4)
            page.insert_text((60, y + 19), c,
                             fontsize=11, fontname="helvetica-bold",
                             color=(0.15, 0.15, 0.2))
            page.insert_text((200, y + 19), f"{eb:.2f}%",
                             fontsize=11, fontname="helvetica",
                             color=(0.1, 0.1, 0.2))
            page.insert_text((330, y + 19), f"{ef:.2f}%",
                             fontsize=11, fontname="helvetica-bold",
                             color=(0, 0.55, 0.85))
            page.insert_text((470, y + 19), f"{d:+.2f}%",
                             fontsize=11, fontname="helvetica-bold",
                             color=(0, 0.7, 0.3) if d < 0 else
                                   (0.9, 0.2, 0.2) if d > 0 else
                                   (0.3, 0.3, 0.3))
            y += 28

        # Weights
        y += 25
        page.insert_text((50, y), "⚖️ Class-Balanced Weights",
                         fontsize=13, fontname="helvetica-bold",
                         color=(0.1, 0.1, 0.2))
        y += 25
        for i, w in enumerate(self.class_weights):
            page.insert_text((60, y), f"{CLASS_NAMES[i]}: {w:.3f}",
                             fontsize=11, fontname="helvetica",
                             color=(0.2, 0.2, 0.3))
            y += 20

    def _page_footer(self, page, page_num):
        page.draw_line(fitz.Point(50, 800), fitz.Point(545, 800),
                       color=(0.85, 0.9, 0.95), width=0.5)
        page.insert_text((50, 815),
                         f"ConvNeXt Balanced Fine-Tune Report  |  "
                         f"developed by : babak yousefian  |  Page {page_num}",
                         fontsize=8, fontname="helvetica-oblique",
                         color=(0.5, 0.5, 0.6))