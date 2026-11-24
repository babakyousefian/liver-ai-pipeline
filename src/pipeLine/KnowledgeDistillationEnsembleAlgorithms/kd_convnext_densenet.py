# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/kd_convnext_densenet.py
# -*- coding: utf-8 -*-

"""
Knowledge Distillation: ConvNeXt (Teacher) → DenseNet (Student)
================================================================
Full KD pipeline with:
  - Teacher: ConvNeXt (frozen, eval)
  - Student: DenseNet-121
  - Loss: α · T² · KLDiv(soft) + (1-α) · CE(hard)
  - Warmup + Cosine LR scheduler
  - EarlyStopping on val_acc
  - Gradient clipping
  - AMP (mixed precision)
  - Best-model tracking + Gap analysis

Outputs (in `bestModels/bestKDEnsemble/`):
  - kd_densenet_best.pth         ← best student checkpoint
  - kd_densenet_last.pth         ← last epoch
  - history_kd_densenet.json     ← train/val loss & acc
  - training_meta_kd_densenet.json
  - kd_metrics.json              ← final test metrics
  - summary.txt                  ← human-readable
  - train_val_gap.png            ← Gap analysis plot

Author: Babak Yousefian
"""

import os
import json
import time
import warnings
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import DenseNet121_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---- Project imports ----
from src.AI.ResNet.utils import (
    CLASS_NAMES,
    CROPPED,
    CSVAddress,
    LabelSmoothingCrossEntropy,
    WarmupCosineScheduler,
    create_weighted_sampler,
    get_class_weights,
    get_device,
    set_seed,
    validate_csv,
)
from src.AI.ResNet.utils import LiverDataset

from src.AI.ResNet.resnet18 import save_model_checkpoint


# =====================================================
# CONSTANTS
# =====================================================

FINAL_REPORTS_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"
BEST_MODELS_DIR = os.path.join(FINAL_REPORTS_DIR, "bestModels")

# ── Teacher folder (SINGULAR: bestKDEnsemble) ──
KD_TEACHER_DIR = os.path.join(BEST_MODELS_DIR, "bestKDEnsemble")
TEACHER_CKPT = os.path.join(KD_TEACHER_DIR, "bestKDE.pth")

# ── Students folder (PLURAL: bestKDEnsembles) ──
KD_STUDENTS_DIR = os.path.join(BEST_MODELS_DIR, "bestKDEnsembles")
STUDENT_NAME = "DenseNet121"  # ← name of student sub-folder
KD_OUTPUT_DIR = os.path.join(KD_STUDENTS_DIR, STUDENT_NAME)

# Student output paths
STUDENT_BEST = os.path.join(KD_OUTPUT_DIR, "kd_densenet_best.pth")
STUDENT_LAST = os.path.join(KD_OUTPUT_DIR, "kd_densenet_last.pth")
KD_HISTORY = os.path.join(KD_OUTPUT_DIR, "history_kd_densenet.json")
KD_META = os.path.join(KD_OUTPUT_DIR, "training_meta_kd_densenet.json")
KD_METRICS = os.path.join(KD_OUTPUT_DIR, "kd_metrics.json")
KD_SUMMARY = os.path.join(KD_OUTPUT_DIR, "summary.txt")
KD_GAP_PNG = os.path.join(KD_OUTPUT_DIR, "train_val_gap.png")

# =====================================================
# HYPERPARAMETERS
# =====================================================

HP = {
    # data
    "target_size": 224,
    "batch_size": 16,
    "accumulation_steps": 2,
    "num_workers": 0,

    # KD
    "temperature": 4.0,          # T — softmax temperature for KD
    "alpha": 0.7,                # weight for soft loss (teacher)
    "label_smoothing": 0.10,     # for hard loss

    # optimization
    "epochs": 60,
    "lr_student": 3e-4,          # student full backbone
    "lr_head": 1e-3,             # student head
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "grad_clip": 1.0,

    # regularization
    "dropout": 0.3,

    # early stopping
    "patience": 15,
    "min_delta": 1e-4,
}

# =====================================================
# DEVICE / SEED
# =====================================================

DEVICE = get_device()
set_seed(42)
torch.set_num_threads(4)


# =====================================================
# HELPERS
# =====================================================

def _detect_teacher_arch() -> str:
    meta_path = os.path.join(KD_TEACHER_DIR, "selection_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_name = meta.get("selected_model")
            if model_name:
                print(f"   📌 Teacher architecture detected: {model_name}")
                return model_name
        except Exception as e:
            print(f"   ⚠️ Could not read selection_meta.json: {e}")

    print(f"   ⚠️ selection_meta.json missing — assuming ConvNext")
    return "ConvNext"


def _build_teacher_by_name(name: str, num_classes: int, state_dict: dict = None) -> nn.Module:
    """
    Build the teacher model with head structure EXACTLY matching the checkpoint.
    Uses state_dict to auto-detect the structure (BatchNorm vs LayerNorm, etc).
    """
    # ═══════════════════════════════════════════════════════════════
    # CONVNEXT
    # ═══════════════════════════════════════════════════════════════
    if name == "ConvNext":
        teacher = models.convnext_tiny(weights=None)

        has_2_1 = state_dict is not None and any(
            k.startswith("classifier.2.1.") for k in state_dict.keys()
        )

        if has_2_1:
            # ── Detect: LayerNorm or BatchNorm for classifier.2.2?
            # BatchNorm has running_mean; LayerNorm doesn't.
            has_bn_2_2 = state_dict is not None and any(
                k.startswith("classifier.2.2.running_mean") for k in state_dict.keys()
            )

            in_features = 768

            try:
                from torchvision.models.convnext import LayerNorm2d
                ln_layer = LayerNorm2d(in_features)
            except ImportError:
                ln_layer = nn.LayerNorm((in_features, 1, 1))

            # Norm layer for classifier.2.2
            if has_bn_2_2:
                norm_2_2 = nn.BatchNorm1d(256)   # ✅ has running_mean
            else:
                norm_2_2 = nn.LayerNorm(256)     # ✅ only weight/bias

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

    # ═══════════════════════════════════════════════════════════════
    # DENSENET
    # ═══════════════════════════════════════════════════════════════
    elif name == "DenseNet":
        teacher = models.densenet121(weights=None)
        in_features = teacher.classifier.in_features  # 1024

        # ⚠️ DenseNet checkpoint: classifier.1, classifier.2, classifier.5
        # classifier.2 has weight+bias only (no running_mean) → LayerNorm, not BatchNorm
        has_bn_2 = state_dict is not None and any(
            k.startswith("classifier.2.running_mean") for k in state_dict.keys()
        )
        norm = nn.BatchNorm1d(512) if has_bn_2 else nn.LayerNorm(512)

        teacher.classifier = nn.Sequential(
            nn.Dropout(HP["dropout"]),
            nn.Linear(in_features, 512),
            norm,
            nn.ReLU(inplace=True),
            nn.Dropout(HP["dropout"] * 0.5),
            nn.Linear(512, num_classes),
        )

    # ═══════════════════════════════════════════════════════════════
    # VGG16
    # ═══════════════════════════════════════════════════════════════
    elif name == "VGG16":
        teacher = models.vgg16(weights=None)
        num_features = 512 * 7 * 7
        teacher.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(HP["dropout"]),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(HP["dropout"] * 0.4),
            nn.Linear(256, num_classes),
        )

    # ═══════════════════════════════════════════════════════════════
    # VISION TRANSFORMER
    # ═══════════════════════════════════════════════════════════════
    elif name == "VisionTransformer":
        teacher = models.vit_b_16(weights=None)
        num_features = teacher.heads.head.in_features
        teacher.heads = nn.Sequential(
            nn.Dropout(HP["dropout"]),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(HP["dropout"] * 0.5),
            nn.Linear(256, num_classes),
        )

    # ═══════════════════════════════════════════════════════════════
    # ALEXNET
    # ═══════════════════════════════════════════════════════════════
    elif name == "AlexNet":
        teacher = models.alexnet(weights=None)
        num_features = teacher.classifier[1].in_features
        teacher.classifier = nn.Sequential(
            nn.Dropout(HP["dropout"]),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(HP["dropout"] * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(HP["dropout"] * 0.3),
            nn.Linear(256, num_classes),
        )

    # ═══════════════════════════════════════════════════════════════
    # MOBILENETV2
    # ═══════════════════════════════════════════════════════════════
    elif name == "MobileNetV2":
        teacher = models.mobilenet_v2(weights=None)
        in_features = teacher.classifier[1].in_features
        teacher.classifier = nn.Sequential(
            nn.Dropout(HP["dropout"]),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(HP["dropout"] * 0.5),
            nn.Linear(256, num_classes),
        )

    else:
        raise ValueError(f"Unknown teacher arch: {name}")

    return teacher


def _load_teacher(checkpoint_path: str, num_classes: int) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    arch_name = _detect_teacher_arch()
    print(f"   🏗️  Building teacher: {arch_name}")

    # ── Load checkpoint FIRST (needed to detect head structure) ──
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # ── Build teacher with structure inferred from checkpoint ──
    teacher = _build_teacher_by_name(arch_name, num_classes=num_classes, state_dict=state)

    # ── Load weights ──
    try:
        teacher.load_state_dict(state, strict=True)
        print(f"   ✅ Teacher loaded (strict=True, all keys matched)")
    except RuntimeError as e:
        print(f"   ⚠️ strict load failed: {e}")
        print(f"   ↪ Retrying with strict=False...")
        missing, unexpected = teacher.load_state_dict(state, strict=False)
        if missing:
            print(f"   ⚠️ Missing {len(missing)} keys:")
            for k in list(missing)[:10]:
                print(f"      - {k}")
        if unexpected:
            print(f"   ⚠️ Unexpected {len(unexpected)} keys:")
            for k in list(unexpected)[:10]:
                print(f"      + {k}")

    teacher = teacher.to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    return teacher


def _build_student(num_classes: int) -> nn.Module:
    """
    DenseNet-121 with head EXACTLY matching the checkpoint structure:
      Sequential(
        Dropout,                # classifier.0 (no params)
        Linear(1024 → 512),     # classifier.1
        BatchNorm1d(512),       # classifier.2
        ReLU,                   # classifier.3 (no params)
        Dropout,                # classifier.4 (no params)
        Linear(512 → num_classes),  # classifier.5
      )
    """
    student = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    in_features = student.classifier.in_features  # 1024

    student.classifier = nn.Sequential(
        nn.Dropout(HP["dropout"]),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(HP["dropout"] * 0.5),
        nn.Linear(512, num_classes),
    )
    return student.to(DEVICE)


def _kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
    hard_criterion: nn.Module,
) -> torch.Tensor:
    """
    Knowledge Distillation loss:
        L = α · T² · KLDiv( softmax(student/T) || softmax(teacher/T) )
          + (1-α) · CE(student, targets)
    """
    # Soft loss
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(
        soft_student, soft_teacher,
        reduction="batchmean",
    ) * (temperature ** 2)

    # Hard loss
    hard_loss = hard_criterion(student_logits, targets)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def _evaluate(
    student: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, Any]:
    """Evaluate student on a loader (val or test)."""
    student.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = student(x)
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

    acc = total_correct / total_samples
    avg_loss = total_loss / len(loader)

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        average=None, zero_division=0,
    )
    wa = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )

    try:
        from sklearn.metrics import roc_auc_score
        auc_macro = float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro",
        ))
    except Exception:
        auc_macro = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "loss": float(avg_loss),
        "precision_weighted": float(wa[0]),
        "recall_weighted": float(wa[1]),
        "f1_weighted": float(wa[2]),
        "precision_per_class": p.tolist(),
        "recall_per_class": r.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": s.tolist(),
        "auc_macro": float(auc_macro),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# =====================================================
# MAIN KD PIPELINE
# =====================================================

def run_kd_convnext_densenet(
    epochs: Optional[int] = None,
    temperature: Optional[float] = None,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
    lr_student: Optional[float] = None,
    patience: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full KD pipeline: ConvNeXt (Teacher) → DenseNet (Student).
    Returns a dict with final metrics and paths.
    """
    # ---- Overrides ----
    if epochs is not None: HP["epochs"] = epochs
    if temperature is not None: HP["temperature"] = temperature
    if alpha is not None: HP["alpha"] = alpha
    if batch_size is not None: HP["batch_size"] = batch_size
    if lr_student is not None: HP["lr_student"] = lr_student
    if patience is not None: HP["patience"] = patience

    os.makedirs(KD_STUDENTS_DIR, exist_ok=True)
    os.makedirs(KD_OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 78)
    print("  🧠  KNOWLEDGE DISTILLATION: ConvNeXt (Teacher) → DenseNet (Student)")
    print("=" * 78)
    print(f"  Device         : {DEVICE}")
    print(f"  Teacher        : {TEACHER_CKPT}")
    print(f"  Output dir     : {KD_OUTPUT_DIR}")
    print(f"  Epochs         : {HP['epochs']}")
    print(f"  Temperature    : {HP['temperature']}")
    print(f"  Alpha          : {HP['alpha']}")
    print(f"  Batch size     : {HP['batch_size']}")
    print(f"  LR (student)   : {HP['lr_student']}")
    print(f"  Patience       : {HP['patience']}")
    print("=" * 78 + "\n")

    if not os.path.exists(TEACHER_CKPT):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {TEACHER_CKPT}\n"
            f"Run option 10/11 in main.py first to create bestModels/."
        )

    # ---- Data ----
    print("📊 Loading data...")
    is_valid, df, msg = validate_csv(CSVAddress)
    if not is_valid:
        raise ValueError(f"CSV validation failed: {msg}")

    df = df[df["liverLabel"] != "UNKNOWN"].copy()
    counts = df["liverLabel"].value_counts()
    valid_classes = counts[counts >= 2].index.tolist()
    df = df[df["liverLabel"].isin(valid_classes)]

    # Split: 80% train / 10% val / 10% test (stratified)
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
        num_workers=HP["num_workers"], pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=HP["batch_size"], shuffle=False,
        num_workers=HP["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=HP["batch_size"], shuffle=False,
        num_workers=HP["num_workers"], pin_memory=True,
    )

    # ---- Teacher ----
    print("🏗️  Loading Teacher (ConvNeXt)...")
    teacher = _load_teacher(TEACHER_CKPT, num_classes=len(CLASS_NAMES))
    print(f"   ✅ Teacher loaded (frozen)")

    # ---- Student ----
    print("🏗️  Building Student (DenseNet-121)...")
    student = _build_student(num_classes=len(CLASS_NAMES))
    n_params = sum(p.numel() for p in student.parameters())
    print(f"   ✅ Student built ({n_params:,} params)")

    # ---- Loss / Optimizer / Scheduler ----
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

    # AMP
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    # ---- History ----
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rates": [],
        "epoch_times": [],
        "teacher_agreement": [],
    }

    best_val_acc = 0.0
    best_epoch = -1
    best_state = None
    patience_counter = 0

    # ---- Training loop ----
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
            desc=f"[KD] Epoch {epoch+1}/{HP['epochs']}",
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
                    loss = _kd_loss(
                        student_logits, teacher_logits, y,
                        HP["temperature"], HP["alpha"], hard_criterion,
                    )
                    loss = loss / HP["accumulation_steps"]
                scaler.scale(loss).backward()

                if (batch_idx + 1) % HP["accumulation_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), HP["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                student_logits = student(x)
                loss = _kd_loss(
                    student_logits, teacher_logits, y,
                    HP["temperature"], HP["alpha"], hard_criterion,
                )
                loss = loss / HP["accumulation_steps"]
                loss.backward()

                if (batch_idx + 1) % HP["accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), HP["grad_clip"])
                    optimizer.step()
                    optimizer.zero_grad()

            # stats
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

        # finalize if leftover accumulation
        if (batch_idx + 1) % HP["accumulation_steps"] != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), HP["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(student.parameters(), HP["grad_clip"])
                optimizer.step()
            optimizer.zero_grad()

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        agreement = total_agree / total_samples

        # ---- Validation ----
        val_metrics = _evaluate(student, val_loader, val_criterion)
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

        # ---- Save best ----
        if val_acc > best_val_acc + HP["min_delta"]:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
            print(f"   ✅ New best val_acc = {best_val_acc:.4f} (epoch {epoch+1})")
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

        # Save last
        save_model_checkpoint(
            model=student, optimizer=optimizer, epoch=epoch,
            metrics={"val_acc": val_acc, "val_loss": val_loss},
            filename=os.path.basename(STUDENT_LAST),
            scheduler=scheduler,
            save_dir=KD_OUTPUT_DIR,
        )

    # ---- Restore best ----
    if best_state is not None:
        student.load_state_dict(best_state)
        print(f"\n✅ Restored best student weights (val_acc={best_val_acc:.4f})")

    # ---- Test evaluation ----
    print("\n📊 Evaluating best student on TEST set...")
    test_metrics = _evaluate(student, test_loader, val_criterion)

    print(f"✅ Test Acc  : {test_metrics['accuracy']:.4f}")
    print(f"✅ Test Loss : {test_metrics['loss']:.4f}")
    print(f"✅ Test AUC  : {test_metrics['auc_macro']:.4f}")

    # ---- Save history & meta ----
    with open(KD_HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

    meta = {
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "test_auc": float(test_metrics["auc_macro"]),
        "epochs_trained": len(history["train_loss"]),
        "teacher_ckpt": TEACHER_CKPT,
        "student_ckpt": STUDENT_BEST,
        "hyperparameters": HP,
    }
    with open(KD_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    # ---- Save metrics.json ----
    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    metrics_payload = {
        "model": "DenseNet-KD",
        "teacher": "ConvNeXt",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "temperature": HP["temperature"],
            "alpha": HP["alpha"],
            "label_smoothing": HP["label_smoothing"],
            "batch_size": HP["batch_size"],
            "epochs": HP["epochs"],
            "lr_student": HP["lr_student"],
            "weight_decay": HP["weight_decay"],
            "dropout": HP["dropout"],
            "test_accuracy": float(test_metrics["accuracy"]),
        },
        "metrics": _clean({
            "accuracy": test_metrics["accuracy"],
            "loss": test_metrics["loss"],
            "precision_weighted": test_metrics["precision_weighted"],
            "recall_weighted": test_metrics["recall_weighted"],
            "f1_weighted": test_metrics["f1_weighted"],
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
    }
    with open(KD_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4, ensure_ascii=False, default=str)

    # ---- Summary TXT ----
    with open(KD_SUMMARY, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("  KNOWLEDGE DISTILLATION — ConvNeXt (Teacher) → DenseNet (Student)\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"  Generated: {datetime.now().isoformat()}\n")
        f.write(f"  Teacher  : {TEACHER_CKPT}\n")
        f.write(f"  Student  : {STUDENT_BEST}\n\n")
        f.write("-" * 78 + "\n")
        f.write("  TEST METRICS\n")
        f.write("-" * 78 + "\n")
        f.write(f"  Accuracy           : {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision (weighted): {test_metrics['precision_weighted']:.4f}\n")
        f.write(f"  Recall (weighted)   : {test_metrics['recall_weighted']:.4f}\n")
        f.write(f"  F1 (weighted)       : {test_metrics['f1_weighted']:.4f}\n")
        f.write(f"  AUC (macro)         : {test_metrics['auc_macro']:.4f}\n\n")
        f.write("-" * 78 + "\n")
        f.write("  PER-CLASS METRICS\n")
        f.write("-" * 78 + "\n")
        for i, cname in enumerate(CLASS_NAMES):
            f.write(
                f"  {cname:<6s} P={test_metrics['precision_per_class'][i]:.4f} "
                f"R={test_metrics['recall_per_class'][i]:.4f} "
                f"F1={test_metrics['f1_per_class'][i]:.4f} "
                f"N={int(test_metrics['support_per_class'][i])}\n"
            )
        f.write("\n" + "=" * 78 + "\n")

    # ---- Gap plot ----
    _plot_gap(history, KD_GAP_PNG)

    print("\n" + "=" * 78)
    print("  ✅ KD TRAINING COMPLETE")
    print("=" * 78)
    print(f"  Best Val Acc: {best_val_acc:.4f}")
    print(f"  Test Acc    : {test_metrics['accuracy']:.4f}")
    print(f"  Test AUC    : {test_metrics['auc_macro']:.4f}")
    print(f"  Student     : {STUDENT_BEST}")
    print(f"  Metrics     : {KD_METRICS}")
    print(f"  Summary     : {KD_SUMMARY}")
    print(f"  Gap plot    : {KD_GAP_PNG}")
    print("=" * 78 + "\n")

    return {
        "status": "success",
        "best_val_acc": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc_macro"],
        "student_path": STUDENT_BEST,
        "metrics_path": KD_METRICS,
        "summary_path": KD_SUMMARY,
        "gap_plot": KD_GAP_PNG,
    }


# =====================================================
# GAP PLOT
# =====================================================

def _plot_gap(history: Dict[str, Any], save_path: str):
    """Plot train/val loss + gap analysis."""
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
    elif avg_gap > -0.15:
        status, color = "MILD UNDERFITTING", "#fbbf24"
    else:
        status, color = "STRONG UNDERFITTING", "#dc2626"

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=180,
                              gridspec_kw={"height_ratios": [2, 1.2]})

    axes[0].plot(ep, tr, "o-", color="#3498db", lw=2.2, ms=4, label="Train Loss")
    axes[0].plot(ep, va, "s-", color="#e74c3c", lw=2.2, ms=4, label="Val Loss")
    axes[0].fill_between(ep, tr, va, alpha=0.2, color="#f39c12", label="Gap")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Loss", fontweight="bold")
    axes[0].set_title("KD Training — Train vs Val Loss", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3, ls="--")

    text = (
        f"Gap Stats:\n"
        f"  Max:   {max(gap):+.4f}\n"
        f"  Min:   {min(gap):+.4f}\n"
        f"  Avg:   {avg_gap:+.4f}\n"
        f"  Final: {final_gap:+.4f}"
    )
    axes[0].text(0.02, 0.98, text, transform=axes[0].transAxes, va="top",
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.5",
                 facecolor="#fff9e6", edgecolor="#f39c12", alpha=0.95))

    colors = ["#e74c3c" if g > 0 else "#2ecc71" for g in gap]
    axes[1].bar(ep, gap, color=colors, alpha=0.75, edgecolor="black", lw=0.4)
    axes[1].axhline(0, color="black", lw=1.5)
    axes[1].axhline(avg_gap, color="#f39c12", lw=2, ls="--",
                    label=f"Avg Gap: {avg_gap:+.4f}")
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Gap (Val - Train)", fontweight="bold")
    axes[1].set_title("Overfitting / Underfitting Indicator", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3, ls="--", axis="y")
    axes[1].text(0.98, 0.95, f"Status: {status}", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=11, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=color,
                           edgecolor="black", lw=1.2, alpha=0.9))

    plt.suptitle("KD Diagnostics — Loss & Gap Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Gap plot saved: {save_path}")

