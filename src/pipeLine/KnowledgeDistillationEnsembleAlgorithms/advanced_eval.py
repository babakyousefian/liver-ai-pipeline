# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/advanced_eval.py
# -*- coding: utf-8 -*-

"""
Advanced Evaluation: CV5 + CI95 (2000 bootstrap) + TTA32
=========================================================
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from src.AI.ResNet.utils import CLASS_NAMES


def _extract_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "samples"):
        return np.array([s[1] for s in dataset.samples])
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    if hasattr(dataset, "data") and hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "df"):
        df = dataset.df
        for col in ("label", "Label", "class", "Class", "target", "Target"):
            if col in df.columns:
                return df[col].values
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            labels.append(int(item[1]))
        else:
            raise ValueError(f"Cannot extract label from item {i}: {type(item)}")
    return np.array(labels)


def _extract_paths(dataset) -> list:
    if hasattr(dataset, "samples"):
        return [s[0] for s in dataset.samples]
    if hasattr(dataset, "imgs"):
        return [s[0] for s in dataset.imgs]
    if hasattr(dataset, "paths"):
        return list(dataset.paths)
    if hasattr(dataset, "image_paths"):
        return list(dataset.image_paths)
    if hasattr(dataset, "df"):
        df = dataset.df
        for col in ("path", "Path", "filepath", "file_path", "image", "filename"):
            if col in df.columns:
                return df[col].tolist()
    if hasattr(dataset, "data"):
        return list(dataset.data)
    raise ValueError(f"Cannot extract image paths from {type(dataset)}")


def _predict_probs(model, loader, device="cuda"):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1)
            pred = prob.argmax(dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def _compute_metrics(y_true, y_pred, y_prob):
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc_macro = float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro",
        ))
    except Exception:
        auc_macro = 0.0
    return {"acc": acc, "auc": auc_macro}


def run_cv5(
    model: nn.Module,
    test_ds,
    n_folds: int = 5,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    labels = _extract_labels(test_ds)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        val_loader = DataLoader(
            Subset(test_ds, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        y_true, y_pred, y_prob = _predict_probs(model, val_loader, device)
        m = _compute_metrics(y_true, y_pred, y_prob)
        m["fold"] = fold
        m["n_samples"] = len(y_true)
        fold_results.append(m)
        print(f"   Fold {fold}: acc={m['acc']:.4f} | auc={m['auc']:.4f}")

    accs = [f["acc"] for f in fold_results]
    aucs = [f["auc"] for f in fold_results]
    return {
        "n_folds": n_folds,
        "folds": fold_results,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }


def run_ci95(
    model: nn.Module,
    test_ds,
    n_bootstrap: int = 2000,
    ci: int = 95,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, y_pred, y_prob = _predict_probs(model, loader, device)

    n = len(y_true)
    rng = np.random.default_rng(42)
    boot_accs, boot_aucs = [], []

    print(f"   Running {n_bootstrap} bootstrap samples...")
    for _ in tqdm(range(n_bootstrap), leave=False):
        idx = rng.integers(0, n, size=n)
        try:
            m = _compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
            boot_accs.append(m["acc"])
            boot_aucs.append(m["auc"])
        except Exception:
            continue

    lo = (100 - ci) / 2
    hi = 100 - lo
    return {
        "n_bootstrap": n_bootstrap,
        "ci": ci,
        "acc": {
            "mean": float(np.mean(boot_accs)),
            "std": float(np.std(boot_accs)),
            "lower": float(np.percentile(boot_accs, lo)),
            "upper": float(np.percentile(boot_accs, hi)),
        },
        "auc": {
            "mean": float(np.mean(boot_aucs)),
            "std": float(np.std(boot_aucs)),
            "lower": float(np.percentile(boot_aucs, lo)),
            "upper": float(np.percentile(boot_aucs, hi)),
        },
    }


def run_tta32(
    model: nn.Module,
    test_ds,
    n_aug: int = 32,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    from torchvision import transforms
    from PIL import Image
    import torch.nn.functional as F

    tta_transforms = [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0 if i % 2 == 0 else 0.0),
            transforms.RandomRotation(degrees=5 if i % 3 == 0 else 0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        for i in range(n_aug)
    ]

    model.eval()
    all_probs = []
    all_labels = []

    paths = _extract_paths(test_ds)
    labels = _extract_labels(test_ds)

    print(f"   Running TTA-{n_aug}...")
    for aug_idx in tqdm(range(n_aug), leave=False):
        tfm = tta_transforms[aug_idx]
        probs_this_aug = []
        labels_this_aug = []

        for i in range(len(paths)):
            path = paths[i]
            label = int(labels[i])
            try:
                img = Image.open(path).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(x)
                    prob = F.softmax(out, dim=1).cpu().numpy().flatten()
                probs_this_aug.append(prob)
                labels_this_aug.append(label)
            except Exception:
                continue

        all_probs.append(np.array(probs_this_aug))
        if not all_labels:
            all_labels = labels_this_aug

    stacked = np.stack(all_probs, axis=0)
    avg_prob = stacked.mean(axis=0)
    y_pred = avg_prob.argmax(axis=1)
    y_true = np.array(all_labels)

    m = _compute_metrics(y_true, y_pred, avg_prob)
    print(f"   TTA-{n_aug}: acc={m['acc']:.4f} | auc={m['auc']:.4f}")
    return {
        "n_aug": n_aug,
        "acc": m["acc"],
        "auc": m["auc"],
        "n_samples": len(y_true),
    }


def run_advanced_eval(
    model: nn.Module,
    test_ds,
    val_criterion: nn.Module = None,
    output_json: str = None,
    device: str = "cuda",
) -> dict:
    results = {}

    print("\n   ═══ CV5 ═══")
    results["cv5"] = run_cv5(model, test_ds, n_folds=5, device=device)

    print("\n   ═══ CI95 (2000 bootstrap) ═══")
    results["ci95"] = run_ci95(model, test_ds, n_bootstrap=2000, device=device)

    print("\n   ═══ TTA-32 ═══")
    results["tta32"] = run_tta32(model, test_ds, n_aug=32, device=device)

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   ✅ Advanced eval saved: {output_json}")

    return results