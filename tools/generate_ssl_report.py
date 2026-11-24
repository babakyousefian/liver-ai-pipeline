# tools/generate_ssl_report.py
# -*- coding: utf-8 -*-
"""
Generate 24-page PDF report for SSL final model.
Uses ALL metadata from checkpoint (test_metrics_tta, cv_summary, ci95, kd_history, polish_history).
Target: C:\\Users\\liana\\Documents\\pythonProjects\\FinalTasks\\finalReports\\selfSupervisedLearning\\final_models\\final_model.pth
"""

import os
import sys
import json
import time
import copy
import warnings
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report, roc_curve, auc,
)
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms
from PIL import Image

# ═════════════════════════════════════════════════════════════════════
# ⚠️ CRITICAL: Add kd_reporter path to sys.path
# ═════════════════════════════════════════════════════════════════════

PROJECT_ROOT = r"C:\Users\liana\Documents\pythonProjects\FinalTasks"
KD_REPORTER_DIR = os.path.join(
    PROJECT_ROOT, "src", "pipeLine", "KnowledgeDistillationEnsembleAlgorithms"
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if KD_REPORTER_DIR not in sys.path:
    sys.path.insert(0, KD_REPORTER_DIR)

# Now try to import
try:
    from kd_reporter import generate_kd_report
    HAS_KD_REPORTER = True
    print("✅ kd_reporter imported successfully")
except ImportError as e:
    print(f"⚠️ kd_reporter import failed: {e}")
    generate_kd_report = None
    HAS_KD_REPORTER = False

# ── Safe imports ────────────────────────────────────────────────────
try:
    from src.AI.ResNet.utils import CLASS_NAMES
except ImportError:
    CLASS_NAMES = ["NL", "G1", "G2", "G3"]

# ── Config ──────────────────────────────────────────────────────────
SSL_MODEL_PATH = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\selfSupervisedLearning\final_models\final_model.pth"
IMAGES_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\cropped"
LABELS_CSV = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\labels.csv"
OUTPUT_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\selfSupervisedLearning\final_models\report"

IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═════════════════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════════════════

class LiverDataset(torch.utils.data.Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["liverName"]
        label = CLASS_NAMES.index(row["liverLabel"])
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label, img_name


# ═════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═════════════════════════════════════════════════════════════════════

def load_ssl_model(path):
    """Load SSL checkpoint and build model."""
    print(f"\n🧠 Loading SSL model from {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"SSL model not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Extract state_dict and metadata
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    elif isinstance(ckpt, nn.Module):
        state_dict = ckpt.state_dict()
        meta = {}
    else:
        state_dict = ckpt
        meta = {}

    print(f"   State dict keys: {len(state_dict)}")
    print(f"   Metadata keys: {list(meta.keys())}")

    # Detect architecture
    arch = meta.get("architecture", None)
    if arch:
        print(f"   Architecture: {arch}")

    # Build model
    model = None
    if arch and "resnet" in str(arch).lower():
        from torchvision.models import resnet18, resnet34, resnet50
        builders = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
        }
        for name, builder in builders.items():
            if name in str(arch).lower():
                try:
                    m = builder(weights=None, num_classes=len(CLASS_NAMES))
                    m.load_state_dict(state_dict, strict=True)
                    model = m
                    print(f"   ✅ Loaded as torchvision.{name} (strict=True)")
                    break
                except RuntimeError:
                    m = builder(weights=None, num_classes=len(CLASS_NAMES))
                    m.load_state_dict(state_dict, strict=False)
                    model = m
                    print(f"   ⚠️ Loaded as torchvision.{name} (strict=False)")
                    break

    # Fallback: try all ResNet variants
    if model is None:
        from torchvision.models import resnet18, resnet34, resnet50
        for name, builder in [
            ("resnet18", resnet18),
            ("resnet34", resnet34),
            ("resnet50", resnet50),
        ]:
            try:
                m = builder(weights=None, num_classes=len(CLASS_NAMES))
                m.load_state_dict(state_dict, strict=True)
                model = m
                print(f"   ✅ Loaded as torchvision.{name}")
                break
            except RuntimeError:
                continue

    if model is None:
        if isinstance(ckpt, nn.Module):
            model = ckpt
            print(f"   ✅ Loaded as nn.Module")
        else:
            raise RuntimeError(
                f"Could not build model from {path}. "
                f"Architecture: {arch}, meta keys: {list(meta.keys())}"
            )

    model = model.to(DEVICE)
    model.eval()
    return model, meta


# ═════════════════════════════════════════════════════════════════════
# EVALUATE (fallback if metadata is missing)
# ═════════════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            with autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    avg_loss = total_loss / len(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, s = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CLASS_NAMES)))
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc_macro = 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "y_true": all_labels,
        "y_pred": all_preds,
        "y_prob": all_probs,
        "precision_per_class": p.tolist(),
        "recall_per_class": r.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": s.tolist(),
        "confusion_matrix": cm,
        "auc_macro": auc_macro,
        "weighted_avg": {
            "precision": float(np.average(p, weights=s)),
            "recall": float(np.average(r, weights=s)),
            "f1-score": float(np.average(f1, weights=s)),
        },
    }


# ═════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, save_path, title="Confusion Matrix (Normalized)"):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar=True, square=True, linewidths=0.5,
                annot_kws={"size": 13, "fontweight": "bold"}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Saved: {save_path}")


def plot_confusion_matrix_raw(cm, save_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar=True, square=True, linewidths=0.5,
                annot_kws={"size": 13, "fontweight": "bold"}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix (Raw)", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Saved: {save_path}")


def plot_metrics_bar(metrics, save_path):
    wa = metrics.get("weighted_avg", {})
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    values = [
        metrics.get("accuracy", 0),
        wa.get("precision", 0),
        wa.get("recall", 0),
        wa.get("f1-score", 0),
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
    print(f"   ✅ Saved: {save_path}")


def plot_per_class(metrics, save_path):
    n = len(CLASS_NAMES)
    p = metrics.get("precision_per_class", [0] * n)
    r = metrics.get("recall_per_class", [0] * n)
    f1 = metrics.get("f1_per_class", [0] * n)
    x = np.arange(n)
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.bar(x - w, p, w, label="Precision", color="#3498db", edgecolor="black")
    ax.bar(x, r, w, label="Recall", color="#2ecc71", edgecolor="black")
    ax.bar(x + w, f1, w, label="F1-Score", color="#9b59b6", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Saved: {save_path}")


def plot_roc_curves(y_true, y_prob, save_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    for i, name in enumerate(CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.5, color=colors[i % len(colors)],
                label=f"{name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random")
    ax.set_xlabel("FPR", fontsize=12, fontweight="bold")
    ax.set_ylabel("TPR", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Saved: {save_path}")


def plot_train_val_gap(history, save_path):
    """Plot Train vs Val Loss + Gap Analysis. Skip if history is empty."""
    if not history.get("train_loss") or not history.get("val_loss"):
        print(f"   ⚠️ No training history. Skipping Gap plot: {save_path}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=200)

    epochs = range(1, len(history["train_loss"]) + 1)
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    gap = np.array(val_loss) - np.array(train_loss)

    axes[0].plot(epochs, train_loss, "o-", label="Train Loss",
                 color="#3498db", lw=2, ms=4)
    axes[0].plot(epochs, val_loss, "s-", label="Val Loss",
                 color="#e74c3c", lw=2, ms=4)
    axes[0].fill_between(epochs, train_loss, val_loss, alpha=0.15, color="#f39c12")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Loss", fontweight="bold")
    axes[0].set_title("Train vs Validation Loss — Gap Analysis",
                      fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3, ls="--")

    stats_text = (f"Gap Statistics:\n"
                  f"Max:  {gap.max():+.4f}\n"
                  f"Min:  {gap.min():+.4f}\n"
                  f"Avg:  {gap.mean():+.4f}\n"
                  f"Final: {gap[-1]:+.4f}")
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                 fontsize=9, va="top", bbox=dict(boxstyle="round,pad=0.3",
                 facecolor="#fff9e6", edgecolor="#f39c12"))

    colors = ["#e74c3c" if g > 0 else "#2ecc71" for g in gap]
    axes[1].bar(epochs, gap, color=colors, edgecolor="black", alpha=0.7)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].axhline(gap.mean(), color="#f39c12", ls="--", lw=2,
                    label=f"Avg Gap = {gap.mean():+.4f}")
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Gap (Val - Train)", fontweight="bold")
    axes[1].set_title("Overfitting / Underfitting Indicator",
                      fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3, ls="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    global HAS_KD_REPORTER
    print("=" * 70)
    print("  📄 SSL FINAL MODEL — 24-PAGE REPORT GENERATOR")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load Data ──
    print("\n📂 Loading dataset...")
    df = pd.read_csv(LABELS_CSV)
    print(f"   Total samples: {len(df)}")
    print(f"   Class distribution:\n{df['liverLabel'].value_counts()}")

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["liverLabel"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["liverLabel"]
    )

    print(f"\n   📊 Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    test_ds = LiverDataset(test_df, IMAGES_DIR, val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 2. Load SSL Model ──
    model, meta = load_ssl_model(SSL_MODEL_PATH)

    # ── 3. Evaluate on TEST (always run, for y_true/y_prob) ──
    print("\n📊 Final evaluation on TEST set...")
    final_metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), DEVICE)
    print(f"   Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"   AUC macro: {final_metrics['auc_macro']:.4f}")

    # ── 4. Use metadata metrics (overwrite if available) ──
    if "test_metrics_tta" in meta:
        tta_meta = meta["test_metrics_tta"]
        if isinstance(tta_meta, dict):
            print(f"\n   Using TTA metrics from metadata:")
            print(f"     Accuracy (TTA): {tta_meta.get('accuracy', 'N/A')}")
            print(f"     AUC (TTA): {tta_meta.get('auc_macro', 'N/A')}")

    # ── 5. Save classification_report.txt ──
    report_txt = classification_report(
        final_metrics["y_true"], final_metrics["y_pred"],
        target_names=CLASS_NAMES, digits=4
    )
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("SSL Final Model — Classification Report (Test Set)\n")
        f.write("=" * 70 + "\n")
        f.write(report_txt)
    print(f"   ✅ classification_report.txt saved")

    # ── 6. Save metrics.json ──
    metrics_json = {
        "model": "SSL Final Model",
        "source": SSL_MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
        "class_names": CLASS_NAMES,
        "metadata": {k: str(v) for k, v in meta.items()},
        "metrics": {
            "accuracy": final_metrics["accuracy"],
            "auc_macro": final_metrics["auc_macro"],
            "weighted_avg": final_metrics["weighted_avg"],
            "per_class": {
                CLASS_NAMES[i]: {
                    "precision": final_metrics["precision_per_class"][i],
                    "recall": final_metrics["recall_per_class"][i],
                    "f1-score": final_metrics["f1_per_class"][i],
                    "support": final_metrics["support_per_class"][i],
                }
                for i in range(len(CLASS_NAMES))
            },
        },
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4, ensure_ascii=False)
    print(f"   ✅ metrics.json saved")

    # ── 7. Generate Plots ──
    print("\n📊 Generating plots...")
    plot_confusion_matrix(
        final_metrics["confusion_matrix"],
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    )
    plot_confusion_matrix_raw(
        final_metrics["confusion_matrix"],
        os.path.join(OUTPUT_DIR, "confusion_matrix_raw.png"),
    )
    plot_metrics_bar(
        final_metrics,
        os.path.join(OUTPUT_DIR, "metrics_bar.png"),
    )
    plot_per_class(
        final_metrics,
        os.path.join(OUTPUT_DIR, "per_class_metrics.png"),
    )
    plot_roc_curves(
        final_metrics["y_true"], final_metrics["y_prob"],
        os.path.join(OUTPUT_DIR, "roc_curves.png"),
    )

    # ── 8. Load training history from metadata ──
    history = {"train_loss": [], "val_loss": [], "val_acc": [],
               "learning_rates": [], "teacher_agreement": []}

    # Try kd_history first
    if "kd_history" in meta:
        kd_hist = meta["kd_history"]
        if isinstance(kd_hist, dict):
            print(f"\n📈 Loading KD training history from metadata...")
            history["train_loss"] = kd_hist.get("train_loss", [])
            history["val_loss"] = kd_hist.get("val_loss", [])
            history["val_acc"] = kd_hist.get("val_acc", [])
            history["learning_rates"] = kd_hist.get("learning_rates", [])
            print(f"   Train Loss: {len(history['train_loss'])} points")
            print(f"   Val Loss: {len(history['val_loss'])} points")
            print(f"   Val Acc: {len(history['val_acc'])} points")

    # Fallback to polish_history if kd_history empty
    if not history["train_loss"] and "polish_history" in meta:
        polish_hist = meta["polish_history"]
        if isinstance(polish_hist, dict):
            print(f"\n📈 Loading Polish history from metadata...")
            history["train_loss"] = polish_hist.get("train_loss", [])
            history["val_loss"] = polish_hist.get("val_loss", [])
            history["val_acc"] = polish_hist.get("val_acc", [])
            history["learning_rates"] = polish_hist.get("learning_rates", [])
            print(f"   Train Loss: {len(history['train_loss'])} points")

    # Plot Gap Analysis
    if history["train_loss"] and history["val_loss"]:
        print("\n📊 Generating Gap Analysis plot...")
        plot_train_val_gap(history, os.path.join(OUTPUT_DIR, "train_val_gap.png"))
    else:
        print("\n⚠️ No training history found. Skipping Gap plot.")

    # ── 9. Build adv_results from metadata ──
    adv_results = {}

    # CV5
    if "cv_summary" in meta:
        cv_meta = meta["cv_summary"]
        if isinstance(cv_meta, dict):
            adv_results["cv5"] = {
                "folds": cv_meta.get("folds", []),
                "mean_acc": cv_meta.get("mean_acc", 0.0),
                "std_acc": cv_meta.get("std_acc", 0.0),
                "mean_auc": cv_meta.get("mean_auc", 0.0),
                "std_auc": cv_meta.get("std_auc", 0.0),
            }
            print(f"\n   ✅ CV5 from metadata: mean_acc={adv_results['cv5']['mean_acc']:.4f}")
        else:
            adv_results["cv5"] = {
                "folds": [], "mean_acc": 0.0, "std_acc": 0.0,
                "mean_auc": 0.0, "std_auc": 0.0,
            }
    else:
        adv_results["cv5"] = {
            "folds": [],
            "mean_acc": float(final_metrics["accuracy"]),
            "std_acc": 0.0,
            "mean_auc": float(final_metrics["auc_macro"]),
            "std_auc": 0.0,
            "note": "CV5 not available in metadata.",
        }

    # CI95
    if "ci95" in meta:
        ci_meta = meta["ci95"]
        if isinstance(ci_meta, dict):
            adv_results["ci95"] = {
                "ci": ci_meta.get("ci", 95),
                "n_bootstrap": ci_meta.get("n_bootstrap", 0),
                "acc": ci_meta.get("acc", {"mean": 0.0, "lower": 0.0, "upper": 0.0}),
                "auc": ci_meta.get("auc", {"mean": 0.0, "lower": 0.0, "upper": 0.0}),
            }
            print(f"   ✅ CI95 from metadata")
        else:
            adv_results["ci95"] = {
                "ci": 95, "n_bootstrap": 0,
                "acc": {"mean": 0.0, "lower": 0.0, "upper": 0.0},
                "auc": {"mean": 0.0, "lower": 0.0, "upper": 0.0},
            }
    else:
        adv_results["ci95"] = {
            "ci": 95, "n_bootstrap": 0,
            "acc": {"mean": float(final_metrics["accuracy"]), "lower": 0.0, "upper": 0.0},
            "auc": {"mean": float(final_metrics["auc_macro"]), "lower": 0.0, "upper": 0.0},
        }

    # TTA32
    if "test_metrics_tta" in meta:
        tta_meta = meta["test_metrics_tta"]
        if isinstance(tta_meta, dict):
            adv_results["tta32"] = {
                "n_aug": tta_meta.get("n_aug", 32),
                "acc": tta_meta.get("accuracy", 0.0),
                "auc": tta_meta.get("auc_macro", 0.0),
            }
            print(f"   ✅ TTA32 from metadata: acc={adv_results['tta32']['acc']:.4f}")
        else:
            adv_results["tta32"] = {"n_aug": 32, "acc": 0.0, "auc": 0.0}
    else:
        adv_results["tta32"] = {"n_aug": 0, "acc": 0.0, "auc": 0.0}

    # ── 10. Generate 24-page PDF Report ──
    print("\n📄 Generating 24-page PDF report...")

    config = {
        "Model": "SSL Final Model",
        "Source": SSL_MODEL_PATH,
        "Architecture": str(meta.get("architecture", "unknown")),
        "Pipeline": str(meta.get("pipeline", "unknown")),
        "Teacher Ensemble": str(meta.get("teacher_ensemble", "N/A")),
        "KD Config": str(meta.get("kd_config", "N/A")),
        "KD Best Epoch": str(meta.get("kd_best_epoch", "N/A")),
        "Polish Best Epoch": str(meta.get("polish_best_epoch", "N/A")),
        "Timestamp": str(meta.get("timestamp", "N/A")),
    }

    if HAS_KD_REPORTER and generate_kd_report is not None:
        try:
            report_path = generate_kd_report(
                model=model,
                teacher=model,  # dummy teacher (same as student)
                test_ds=test_ds,
                test_metrics=final_metrics,
                history=history,
                adv_results=adv_results,
                output_dir=OUTPUT_DIR,
                student_name="SSL Final Model",
                teacher_name="N/A (SSL)",
                config=config,
            )
            print(f"\n✅ 24-page report generated: {report_path}")
        except Exception as e:
            print(f"\n❌ kd_reporter failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic PDF
            HAS_KD_REPORTER = False

    if not HAS_KD_REPORTER:
        print("\n⚠️ Generating basic PDF fallback...")
        try:
            import fitz
            doc = fitz.open()
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 80), "SSL Final Model — Report",
                             fontsize=24, fontname="helvetica-bold")
            page.insert_text((50, 120), f"Accuracy: {final_metrics['accuracy']:.4f}",
                             fontsize=14, fontname="helvetica")
            page.insert_text((50, 150), f"AUC macro: {final_metrics['auc_macro']:.4f}",
                             fontsize=14, fontname="helvetica")
            y = 200
            for i, name in enumerate(CLASS_NAMES):
                page.insert_text((50, y),
                                 f"{name}: P={final_metrics['precision_per_class'][i]:.4f} "
                                 f"R={final_metrics['recall_per_class'][i]:.4f} "
                                 f"F1={final_metrics['f1_per_class'][i]:.4f}",
                                 fontsize=12, fontname="helvetica")
                y += 25
            pdf_path = os.path.join(OUTPUT_DIR, "SSL_Report.pdf")
            doc.save(pdf_path)
            doc.close()
            print(f"✅ Basic PDF saved: {pdf_path}")
        except ImportError:
            print("❌ fitz (PyMuPDF) not installed. Install: pip install PyMuPDF")

    # ── 11. Summary ──
    print("\n" + "=" * 70)
    print(f"  ✅ All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)
    print("\n📁 Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"   - {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()