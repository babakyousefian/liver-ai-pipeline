# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/kd_convnext_swin.py
# -*- coding: utf-8 -*-

"""
Knowledge Distillation: ConvNext (Teacher) → Swin Transformer (Student)
========================================================================
Full pipeline:
  1. Load teacher from bestKDE.pth (ConvNext)
  2. Build Swin-T student with custom head
  3. KD training (α·T²·KLDiv + (1-α)·CE) with warmup-cosine
  4. After training, run:
       - Test evaluation (baseline)
       - CV5 (5-fold cross-validation on test set)
       - CI95 (2000 bootstrap samples)
       - TTA-32 (32 augmented passes)
  5. Save comprehensive report (24-page PDF) with LIME/SHAP/Captum/Netron

Author: Babak Yousefian
"""

import os
import json
import time
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.models import Swin_T_Weights
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
)
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── Project imports ───
from src.AI.ResNet.utils import (
    CLASS_NAMES, CROPPED, CSVAddress,
    LabelSmoothingCrossEntropy, WarmupCosineScheduler,
    create_weighted_sampler, get_device, set_seed,
    validate_csv, LiverDataset,
)
from src.AI.ResNet.resnet18 import save_model_checkpoint


# ═════════════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════════════

FINAL_REPORTS_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"
BEST_MODELS_DIR = os.path.join(FINAL_REPORTS_DIR, "bestModels")

KD_TEACHER_DIR = os.path.join(BEST_MODELS_DIR, "bestKDEnsemble")
TEACHER_CKPT = os.path.join(KD_TEACHER_DIR, "bestKDE.pth")

KD_STUDENTS_DIR = os.path.join(BEST_MODELS_DIR, "bestKDEnsembles")
STUDENT_NAME = "SwinTransformer"
KD_OUTPUT_DIR = os.path.join(KD_STUDENTS_DIR, STUDENT_NAME)

STUDENT_BEST = os.path.join(KD_OUTPUT_DIR, "kd_swin_best.pth")
STUDENT_LAST = os.path.join(KD_OUTPUT_DIR, "kd_swin_last.pth")
KD_HISTORY = os.path.join(KD_OUTPUT_DIR, "history_kd_swin.json")
KD_META = os.path.join(KD_OUTPUT_DIR, "training_meta_kd_swin.json")
KD_METRICS = os.path.join(KD_OUTPUT_DIR, "kd_metrics.json")
KD_GAP_PNG = os.path.join(KD_OUTPUT_DIR, "train_val_gap.png")
KD_PREDICTIONS = os.path.join(KD_OUTPUT_DIR, "predictions.csv")
ADV_EVAL_JSON = os.path.join(KD_OUTPUT_DIR, "advanced_eval.json")


# ═════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════════════

HP = {
    # data
    "target_size": 224,
    "batch_size": 16,
    "accumulation_steps": 2,
    "num_workers": 0,

    # KD
    "temperature": 4.0,
    "alpha": 0.65,                     # ← 0.65 برای توجه بیشتر به Teacher
    "label_smoothing": 0.10,

    # optimization — تخصصی برای Swin
    "epochs": 80,
    "lr_student": 1e-4,                # ← کمتر از DenseNet (Swin حساس‌تر)
    "weight_decay": 0.02,              # ← بیشتر (Swin نیاز به Regularization)
    "warmup_epochs": 8,                # ← بیشتر (Swin دیرتر converge می‌کند)
    "min_lr": 1e-6,
    "grad_clip": 1.0,

    # regularization
    "dropout": 0.25,

    # early stopping
    "patience": 25,                    # ← بیشتر (فرصت یادگیری)
    "min_delta": 5e-5,                 # ← حساس‌تر
}

# ─── Advanced evaluation params ───
ADV_EVAL = {
    "cv_folds": 5,
    "n_bootstrap": 2000,
    "ci": 95,
    "tta_n": 32,
}


# ═════════════════════════════════════════════════════════════════════
# DEVICE / SEED
# ═════════════════════════════════════════════════════════════════════

DEVICE = get_device()
set_seed(42)
torch.set_num_threads(4)


# ═════════════════════════════════════════════════════════════════════
# TEACHER LOADER (ConvNext from bestKDE.pth)
# ═════════════════════════════════════════════════════════════════════

def _detect_teacher_arch() -> str:
    meta_path = os.path.join(KD_TEACHER_DIR, "selection_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_name = meta.get("selected_model")
            if model_name:
                return model_name
        except Exception:
            pass
    return "ConvNext"


def _build_teacher_convnext(num_classes: int, state_dict: dict) -> nn.Module:
    """Build ConvNext teacher matching checkpoint structure."""
    teacher = models.convnext_tiny(weights=None)

    has_2_1 = any(k.startswith("classifier.2.1.") for k in state_dict.keys())
    has_bn_2_2 = any(k.startswith("classifier.2.2.running_mean") for k in state_dict.keys())

    if has_2_1:
        in_features = 768
        try:
            from torchvision.models.convnext import LayerNorm2d
            ln_layer = LayerNorm2d(in_features)
        except ImportError:
            ln_layer = nn.LayerNorm((in_features, 1, 1))

        norm_2_2 = nn.BatchNorm1d(256) if has_bn_2_2 else nn.LayerNorm(256)

        teacher.classifier = nn.Sequential(
            ln_layer,
            nn.Flatten(1),
            nn.Sequential(
                nn.Dropout(HP["dropout"]),
                nn.Linear(in_features, 256),
                norm_2_2,
                nn.ReLU(inplace=True),
                nn.Dropout(HP["dropout"] * 0.5),
                nn.Linear(256, num_classes),
            ),
        )
    else:
        in_features = teacher.classifier[2].in_features
        teacher.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(HP["dropout"]),
            nn.Linear(256, num_classes),
        )
    return teacher


def load_teacher(checkpoint_path: str, num_classes: int) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    arch = _detect_teacher_arch()
    print(f"   📌 Teacher arch: {arch}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    if arch != "ConvNext":
        raise ValueError(f"Only ConvNext teacher is supported here (got {arch}). "
                         f"Use a different KD script for {arch}.")

    teacher = _build_teacher_convnext(num_classes, state)
    teacher.load_state_dict(state, strict=True)
    teacher = teacher.to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"   ✅ Teacher loaded (strict=True)")
    return teacher


# ═════════════════════════════════════════════════════════════════════
# STUDENT (Swin-Tiny)
# ═════════════════════════════════════════════════════════════════════

def build_swin_student(num_classes: int) -> nn.Module:
    """
    Swin-Tiny with custom classifier matching pretrained Swin-T head format:
        model.head = Linear(768, 1000)  → replaced with Sequential
    """
    student = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

    in_features = student.head.in_features  # 768

    student.head = nn.Sequential(
        nn.Dropout(HP["dropout"]),
        nn.Linear(in_features, 256),
        nn.LayerNorm(256),                 # ← LayerNorm برای Swin
        nn.GELU(),
        nn.Dropout(HP["dropout"] * 0.5),
        nn.Linear(256, num_classes),
    )
    return student.to(DEVICE)


# ═════════════════════════════════════════════════════════════════════
# KD LOSS
# ═════════════════════════════════════════════════════════════════════

def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    T: float,
    alpha: float,
    hard_criterion: nn.Module,
) -> torch.Tensor:
    soft_s = F.log_softmax(student_logits / T, dim=1)
    soft_t = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_s, soft_t, reduction="batchmean") * (T ** 2)
    hard_loss = hard_criterion(student_logits, targets)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss


# ═════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═════════════════════════════════════════════════════════════════════

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    n_classes = len(CLASS_NAMES)
    acc = float(accuracy_score(y_true, y_pred))

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)),
        average=None, zero_division=0,
    )
    wa = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )

    try:
        auc_macro = float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro",
        ))
    except Exception:
        auc_macro = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    report = classification_report(
        y_true, y_pred, labels=list(range(n_classes)),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_weighted": float(wa[0]),
        "recall_weighted": float(wa[1]),
        "f1_weighted": float(wa[2]),
        "precision_per_class": p.tolist(),
        "recall_per_class": r.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": s.tolist(),
        "auc_macro": auc_macro,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    return_loss: bool = False,
) -> Dict[str, Any]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            loss = criterion(out, y)
            prob = torch.softmax(out, dim=1)
            pred = prob.argmax(dim=1)

            total_loss += loss.item()
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = _compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader)

    if return_loss:
        return metrics
    return metrics


# ═════════════════════════════════════════════════════════════════════
# MAIN KD TRAINING
# ═════════════════════════════════════════════════════════════════════

def run_kd_convnext_swin(
    epochs: Optional[int] = None,
    temperature: Optional[float] = None,
    alpha: Optional[float] = None,
    patience: Optional[int] = None,
) -> Dict[str, Any]:
    """Full KD pipeline: ConvNext (T) → Swin Transformer (S)."""
    if epochs is not None: HP["epochs"] = epochs
    if temperature is not None: HP["temperature"] = temperature
    if alpha is not None: HP["alpha"] = alpha
    if patience is not None: HP["patience"] = patience

    os.makedirs(KD_STUDENTS_DIR, exist_ok=True)
    os.makedirs(KD_OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("  🧠  KD: ConvNext (Teacher) → Swin Transformer (Student)")
    print("=" * 80)
    print(f"  Teacher: {TEACHER_CKPT}")
    print(f"  Student: {KD_OUTPUT_DIR}")
    print(f"  Epochs: {HP['epochs']} | T={HP['temperature']} | α={HP['alpha']}")
    print(f"  LR={HP['lr_student']} | WD={HP['weight_decay']} | Patience={HP['patience']}")
    print("=" * 80 + "\n")

    if not os.path.exists(TEACHER_CKPT):
        raise FileNotFoundError(f"Teacher checkpoint not found: {TEACHER_CKPT}")

    # ─── Data ───
    print("📊 Loading data...")
    is_valid, df, msg = validate_csv(CSVAddress)
    if not is_valid:
        raise ValueError(f"CSV validation failed: {msg}")

    df = df[df["liverLabel"] != "UNKNOWN"].copy()
    counts = df["liverLabel"].value_counts()
    valid = counts[counts >= 2].index.tolist()
    df = df[df["liverLabel"].isin(valid)]

    train_val_df, test_df = train_test_split(
        df, test_size=0.10, stratify=df["liverLabel"], random_state=42,
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=1/9,
        stratify=train_val_df["liverLabel"], random_state=42,
    )

    train_ds = LiverDataset(
        train_df, CROPPED, augment=True,
        target_size=HP["target_size"], apply_normalize=True,
        use_advanced_augment=True, return_path=True,
    )
    val_ds = LiverDataset(
        val_df, CROPPED, augment=False,
        target_size=HP["target_size"], apply_normalize=True,
        use_advanced_augment=False, return_path=True,
    )
    test_ds = LiverDataset(
        test_df, CROPPED, augment=False,
        target_size=HP["target_size"], apply_normalize=True,
        use_advanced_augment=False, return_path=True,
    )

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_sampler = create_weighted_sampler(train_ds, CSVAddress)
    train_loader = DataLoader(
        train_ds, batch_size=HP["batch_size"], sampler=train_sampler,
        num_workers=HP["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=HP["batch_size"], shuffle=False,
        num_workers=HP["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=HP["batch_size"], shuffle=False,
        num_workers=HP["num_workers"], pin_memory=True,
    )

    # ─── Teacher ───
    print("🏗️  Loading Teacher (ConvNext)...")
    teacher = load_teacher(TEACHER_CKPT, num_classes=len(CLASS_NAMES))

    # ─── Student ───
    print("🏗️  Building Student (Swin Transformer)...")
    student = build_swin_student(num_classes=len(CLASS_NAMES))
    n_params = sum(p.numel() for p in student.parameters())
    print(f"   ✅ Student built ({n_params:,} params)")

    # ─── Optimizer / Scheduler ───
    hard_criterion = LabelSmoothingCrossEntropy(smoothing=HP["label_smoothing"])
    val_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        student.parameters(),
        lr=HP["lr_student"],
        weight_decay=HP["weight_decay"],
        betas=(0.9, 0.999),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=HP["warmup_epochs"],
        total_epochs=HP["epochs"],
        min_lr=HP["min_lr"],
        max_lr=HP["lr_student"],
    )

    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    # ─── History ───
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "learning_rates": [], "epoch_times": [],
        "teacher_agreement": [],
    }

    best_val_acc = 0.0
    best_epoch = -1
    best_state = None
    patience_counter = 0

    # ─── Training loop ───
    print("\n🚀 Starting KD training...\n")

    for epoch in range(HP["epochs"]):
        epoch_start = time.time()
        student.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_agree = 0
        optimizer.zero_grad()

        loop = tqdm(
            train_loader,
            desc=f"[KD-Swin] Epoch {epoch+1}/{HP['epochs']}",
            leave=False,
        )

        for batch_idx, batch in enumerate(loop):
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(x)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    student_logits = student(x)
                    loss = kd_loss(
                        student_logits, teacher_logits, y,
                        HP["temperature"], HP["alpha"], hard_criterion,
                    ) / HP["accumulation_steps"]
                scaler.scale(loss).backward()

                if (batch_idx + 1) % HP["accumulation_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), HP["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                student_logits = student(x)
                loss = kd_loss(
                    student_logits, teacher_logits, y,
                    HP["temperature"], HP["alpha"], hard_criterion,
                ) / HP["accumulation_steps"]
                loss.backward()

                if (batch_idx + 1) % HP["accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), HP["grad_clip"])
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * HP["accumulation_steps"]
            pred = student_logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)

            teacher_pred = teacher_logits.argmax(dim=1)
            total_agree += (pred == teacher_pred).sum().item()

            loop.set_postfix({
                "loss": f"{loss.item()*HP['accumulation_steps']:.4f}",
                "acc": f"{total_correct/total_samples:.4f}",
            })

        # leftover accumulation
        if (batch_idx + 1) % HP["accumulation_steps"] != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), HP["grad_clip"])
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(student.parameters(), HP["grad_clip"])
                optimizer.step()
            optimizer.zero_grad()

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        agreement = total_agree / total_samples

        val_metrics = evaluate_model(student, val_loader, val_criterion)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        current_lr = scheduler.step()
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(epoch_time)
        history["teacher_agreement"].append(agreement)

        gap = val_loss - train_loss
        print(
            f"📊 Epoch {epoch+1}/{HP['epochs']} | "
            f"TrLoss={train_loss:.4f} TrAcc={train_acc:.4f} | "
            f"VaLoss={val_loss:.4f} VaAcc={val_acc:.4f} | "
            f"Gap={gap:+.4f} | Agree={agreement:.3f} | "
            f"LR={current_lr:.2e} | {epoch_time:.1f}s"
        )

        # ─── Save best ───
        if val_acc > best_val_acc + HP["min_delta"]:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
            print(f"   ✅ New best val_acc = {best_val_acc:.4f}")
            save_model_checkpoint(
                model=student, optimizer=optimizer, epoch=epoch,
                metrics={"val_acc": val_acc, "val_loss": val_loss},
                filename=os.path.basename(STUDENT_BEST),
                scheduler=scheduler,
                extra_info={"best_accuracy": best_val_acc},
                save_dir=KD_OUTPUT_DIR,
            )
        else:
            patience_counter += 1
            if patience_counter >= HP["patience"]:
                print(f"\n🛑 Early stopping at epoch {epoch+1} "
                      f"(best val_acc={best_val_acc:.4f} @ epoch {best_epoch+1})")
                break

        save_model_checkpoint(
            model=student, optimizer=optimizer, epoch=epoch,
            metrics={"val_acc": val_acc, "val_loss": val_loss},
            filename=os.path.basename(STUDENT_LAST),
            scheduler=scheduler, save_dir=KD_OUTPUT_DIR,
        )

    # ─── Restore best ───
    if best_state is not None:
        student.load_state_dict(best_state)
        print(f"\n✅ Restored best weights (val_acc={best_val_acc:.4f})")

    # ─── Baseline test ───
    print("\n📊 Baseline TEST evaluation...")
    test_metrics = evaluate_model(student, test_loader, val_criterion)
    print(f"   ✅ Test Acc={test_metrics['accuracy']:.4f} | "
          f"AUC={test_metrics['auc_macro']:.4f}")

    # ─── Advanced evaluation (CV5 + CI95 + TTA32) ───
    print("\n🔬 Running advanced evaluation (CV5 + CI95 + TTA32)...")
    from .advanced_eval import run_advanced_eval

    adv_results = run_advanced_eval(
        model=student,
        test_ds=test_ds,
        val_criterion=val_criterion,
        output_json=ADV_EVAL_JSON,
    )

    # ─── Save history + metrics ───
    with open(KD_HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

    meta = {
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "baseline_test_acc": float(test_metrics["accuracy"]),
        "baseline_test_auc": float(test_metrics["auc_macro"]),
        "epochs_trained": len(history["train_loss"]),
        "teacher_ckpt": TEACHER_CKPT,
        "student_ckpt": STUDENT_BEST,
        "hyperparameters": HP,
        "advanced_eval": adv_results,
    }
    with open(KD_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str, ensure_ascii=False)

    # ─── Save metrics JSON ───
    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    metrics_payload = {
        "model": "Swin-T-KD",
        "teacher": "ConvNext",
        "timestamp": datetime.now().isoformat(),
        "config": _clean({**HP, "test_accuracy": test_metrics["accuracy"]}),
        "metrics": _clean({
            "accuracy": test_metrics["accuracy"],
            "auc_macro": test_metrics["auc_macro"],
            "per_class": {
                CLASS_NAMES[i]: {
                    "precision": test_metrics["precision_per_class"][i],
                    "recall": test_metrics["recall_per_class"][i],
                    "f1": test_metrics["f1_per_class"][i],
                    "support": int(test_metrics["support_per_class"][i]),
                }
                for i in range(len(CLASS_NAMES))
            },
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": test_metrics["classification_report"],
        }),
        "advanced_eval": _clean(adv_results),
    }
    with open(KD_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4, ensure_ascii=False, default=str)

    # ─── Save predictions CSV ───
    pd.DataFrame({
        "true_idx": test_metrics["y_true"],
        "pred_idx": test_metrics["y_pred"],
        "true_label": [CLASS_NAMES[i] for i in test_metrics["y_true"]],
        "pred_label": [CLASS_NAMES[i] for i in test_metrics["y_pred"]],
        "correct": (test_metrics["y_true"] == test_metrics["y_pred"]).astype(int),
        **{f"prob_{c}": test_metrics["y_prob"][:, i]
           for i, c in enumerate(CLASS_NAMES)},
    }).to_csv(KD_PREDICTIONS, index=False)

    # ─── Gap plot ───
    _plot_gap(history, KD_GAP_PNG)

    # ─── Generate 24-page PDF report ───
    print("\n📄 Generating 24-page PDF report...")
    from .kd_reporter import generate_kd_report

    pdf_path = generate_kd_report(
        model=student,
        teacher=teacher,
        test_ds=test_ds,
        test_metrics=test_metrics,
        history=history,
        adv_results=adv_results,
        output_dir=KD_OUTPUT_DIR,
        student_name="SwinTransformer",
        teacher_name="ConvNext",
        config=HP,
    )

    print("\n" + "=" * 80)
    print("  ✅ KD COMPLETE")
    print("=" * 80)
    print(f"  Baseline Test Acc: {test_metrics['accuracy']:.4f}")
    print(f"  Baseline Test AUC: {test_metrics['auc_macro']:.4f}")
    print(f"  CV5 Mean Acc:      {adv_results.get('cv5', {}).get('mean_acc', 0):.4f}")
    print(f"  TTA-32 Acc:        {adv_results.get('tta32', {}).get('acc', 0):.4f}")
    print(f"  Student: {STUDENT_BEST}")
    print(f"  Report:  {pdf_path}")
    print("=" * 80 + "\n")

    return {
        "status": "success",
        "best_val_acc": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc_macro"],
        "student_path": STUDENT_BEST,
        "metrics_path": KD_METRICS,
        "advanced_eval_path": ADV_EVAL_JSON,
        "pdf_report": pdf_path,
        "advanced_eval": adv_results,
    }


def _plot_gap(history: Dict[str, Any], save_path: str):
    tr = history.get("train_loss", [])
    va = history.get("val_loss", [])
    if not tr or not va:
        return

    n = min(len(tr), len(va))
    tr, va = tr[:n], va[:n]
    ep = list(range(1, n + 1))
    gap = [v - t for t, v in zip(tr, va)]

    avg_gap = sum(gap) / len(gap)
    final_gap = gap[-1]

    if avg_gap > 0.3:
        status, color = "STRONG OVERFITTING", "#dc2626"
    elif avg_gap > 0.15:
        status, color = "MODERATE OVERFITTING", "#f59e0b"
    elif avg_gap > 0.05:
        status, color = "MILD OVERFITTING", "#fbbf24"
    elif avg_gap > -0.05:
        status, color = "GOOD FIT", "#22c55e"
    else:
        status, color = "UNDERFITTING", "#dc2626"

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=180,
                              gridspec_kw={"height_ratios": [2, 1.2]})
    axes[0].plot(ep, tr, "o-", color="#3498db", lw=2.2, ms=4, label="Train Loss")
    axes[0].plot(ep, va, "s-", color="#e74c3c", lw=2.2, ms=4, label="Val Loss")
    axes[0].fill_between(ep, tr, va, alpha=0.2, color="#f39c12", label="Gap")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Loss", fontweight="bold")
    axes[0].set_title("KD Training — Train vs Val Loss", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3, ls="--")
    axes[0].text(0.02, 0.98,
                 f"Gap Stats:\n  Max:   {max(gap):+.4f}\n  Min:   {min(gap):+.4f}\n"
                 f"  Avg:   {avg_gap:+.4f}\n  Final: {final_gap:+.4f}",
                 transform=axes[0].transAxes, va="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9e6",
                           edgecolor="#f39c12", alpha=0.95))
    colors = ["#e74c3c" if g > 0 else "#2ecc71" for g in gap]
    axes[1].bar(ep, gap, color=colors, alpha=0.75, edgecolor="black", lw=0.4)
    axes[1].axhline(0, color="black", lw=1.5)
    axes[1].axhline(avg_gap, color="#f39c12", lw=2, ls="--",
                    label=f"Avg Gap: {avg_gap:+.4f}")
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Gap (Val - Train)", fontweight="bold")
    axes[1].set_title("Overfitting / Underfitting Indicator", fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3, ls="--", axis="y")
    axes[1].text(0.98, 0.95, f"Status: {status}", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=11, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=color,
                           edgecolor="black", lw=1.2, alpha=0.9))
    plt.suptitle("KD Diagnostics — Loss & Gap Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
