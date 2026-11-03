# src/pipeLine/findBestModels.py
# -*- coding: utf-8 -*-
"""
Find Best Model — Option 43
===========================
Evaluates 4 fine-tuned models (AlexNet, DenseNet, MobileNetV2, VGG16)
on the same test set, selects the best, and copies it to the destination.
"""

import os
import shutil
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CANDIDATES = {
    "AlexNet": os.path.join(
        PROJECT_ROOT, "finalReports", "selfSupervisedLearning", "SSL_Reporter",
        "alexnet_finetune", "alexnet_finetuned_final.pth"
    ),
    "DenseNet": os.path.join(
        PROJECT_ROOT, "finalReports", "selfSupervisedLearning", "SSL_Reporter",
        "densenet_finetune", "densenet_finetuned_final.pth"
    ),
    "MobileNetV2": os.path.join(
        PROJECT_ROOT, "finalReports", "selfSupervisedLearning", "SSL_Reporter",
        "mobilenetv2_finetune", "mobilenetv2_finetuned_final.pth"
    ),
    "VGG16": os.path.join(
        PROJECT_ROOT, "finalReports", "selfSupervisedLearning", "SSL_Reporter",
        "vgg16_finetune", "vgg16_finetuned_final.pth"
    ),
}

DEST_DIR = os.path.join(
    PROJECT_ROOT, "finalReports", "selfSupervisedLearning", "SSL_Reporter",
    "FindBestNewModels"
)
DEST_PATH = os.path.join(DEST_DIR, "best_models.pth")
REPORT_JSON = os.path.join(DEST_DIR, "best_models_report.json")

LABELS_CSV = os.path.join(PROJECT_ROOT, "finalReports", "labels.csv")
CROPPED_DIR = os.path.join(PROJECT_ROOT, "finalReports", "cropped")

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = 4
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ================================================================
# Dataset
# ================================================================

class EvalDataset(Dataset):
    def __init__(self, paths, labels, target_size=224):
        self.paths = paths
        self.labels = labels
        self.target_size = target_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i])
        if img is None:
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size))
        img = Image.fromarray(img)
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return tf(img), self.labels[i]


# ================================================================
# Model builders
# ================================================================

def _build_alexnet():
    from src.models.AlexNet import create_alexnet
    return create_alexnet(
        num_classes=NUM_CLASSES, width_mult=0.5, fc6_features=1024,
        fc7_features=512, dropout_fc=0.4, use_batch_norm=True,
        input_size=224, verbose=False,
    )


def _build_densenet():
    from src.models.DenseNet.DenseNet import DenseNet
    return DenseNet(
        num_classes=NUM_CLASSES, input_channels=3, input_size=224,
        layers=[6, 12, 24, 16], growth_rate=24, bn_size=4,
        compression=0.5, init_channels=32, dropout=0.2, verbose=False,
    )


def _build_mobilenetv2():
    from torchvision.models import mobilenet_v2
    return mobilenet_v2(num_classes=NUM_CLASSES, width_mult=1.0)


def _build_vgg16():
    from torchvision.models import vgg16
    return vgg16(num_classes=NUM_CLASSES, weights=None)


BUILDERS = {
    "AlexNet": _build_alexnet,
    "DenseNet": _build_densenet,
    "MobileNetV2": _build_mobilenetv2,
    "VGG16": _build_vgg16,
}


# ================================================================
# Load labeled data
# ================================================================

def _load_data():
    df = pd.read_csv(LABELS_CSV)
    df = df[df["liverLabel"].isin(CLASS_NAMES)]
    paths, labels = [], []
    for _, row in df.iterrows():
        name = row["liverName"]
        lbl = row["liverLabel"]
        p = os.path.join(CROPPED_DIR, name)
        if not os.path.exists(p):
            base = os.path.splitext(name)[0]
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                test = os.path.join(CROPPED_DIR, base + ext)
                if os.path.exists(test):
                    p = test
                    break
            else:
                continue
        paths.append(p)
        labels.append(LABEL_MAP[lbl])
    return paths, labels


# ================================================================
# Evaluate
# ================================================================

@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, lbls in tqdm(loader, desc="  Evaluating", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "num_samples": int(len(y_true)),
    }


# ================================================================
# Main
# ================================================================

def run_find_best_models():
    print("\n" + "=" * 80)
    print("🏆 Option 43 — Find Best Model Among 4 Architectures")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Device: {device}")

    # Check candidates exist
    print("\n🔍 Checking candidate checkpoints:")
    available = {}
    for name, path in CANDIDATES.items():
        ok = os.path.exists(path)
        icon = "✅" if ok else "❌"
        print(f"   {icon} {name:<12} {path}")
        if ok:
            available[name] = path

    if not available:
        print("\n❌ No checkpoints found. Run options 23/29/35/41 first.")
        return {"status": "failed"}

    # Load data — same split as fine-tune (10% test, seed=42)
    print("\n📥 Loading labeled data...")
    all_paths, all_labels = _load_data()
    _, test_paths, _, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.10, random_state=42,
        stratify=all_labels,
    )
    print(f"   Test set: {len(test_paths)} images")

    test_ds = EvalDataset(test_paths, test_labels, target_size=224)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    # Evaluate each candidate
    print("\n" + "=" * 80)
    print("🔬 Evaluating each candidate on the same test set")
    print("=" * 80)

    results = {}
    for name, path in available.items():
        print(f"\n📌 Loading {name}...")
        try:
            model = BUILDERS[name]()
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif "encoder_state_dict" in ckpt:
                state = ckpt["encoder_state_dict"]
            else:
                state = ckpt

            missing, unexpected = model.load_state_dict(state, strict=False)
            if len(missing) > 50:
                print(f"   ⚠️ {len(missing)} missing keys — skipping {name}")
                continue

            model = model.to(device)
            metrics = _evaluate(model, test_loader, device)
            metrics["checkpoint"] = path
            results[name] = metrics

            print(f"   ✅ Acc: {metrics['accuracy']:.4f} | "
                  f"Bal Acc: {metrics['balanced_accuracy']:.4f} | "
                  f"F1(w): {metrics['f1_weighted']:.4f} | "
                  f"F1(m): {metrics['f1_macro']:.4f}")

        except Exception as e:
            print(f"   ❌ Failed to evaluate {name}: {e}")
            continue

    if not results:
        print("\n❌ No model could be evaluated.")
        return {"status": "failed"}

    # Pick best by balanced accuracy (robust to imbalance)
    print("\n" + "=" * 80)
    print("🏆 Selection")
    print("=" * 80)

    best_name = max(results.keys(), key=lambda k: results[k]["balanced_accuracy"])
    best_metrics = results[best_name]
    best_path = best_metrics["checkpoint"]

    print(f"\n   Winner: {best_name}")
    print(f"   Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
    print(f"   Accuracy:          {best_metrics['accuracy']:.4f}")
    print(f"   F1 (weighted):     {best_metrics['f1_weighted']:.4f}")
    print(f"   F1 (macro):        {best_metrics['f1_macro']:.4f}")

    # Save best to destination
    os.makedirs(DEST_DIR, exist_ok=True)
    shutil.copy2(best_path, DEST_PATH)
    print(f"\n✅ Best model copied to: {DEST_PATH}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_name,
        "best_metrics": best_metrics,
        "all_results": results,
        "destination": DEST_PATH,
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=4, default=str)
    print(f"✅ Report saved: {REPORT_JSON}")

    print("\n" + "=" * 80)
    print("🎉 DONE")
    print("=" * 80)

    return {"status": "success", "best_model": best_name, "best_metrics": best_metrics}