# src/pipeLine/KnowledgeDistillationEnsembleAlgorithms/regen_swin_kd_report.py
# -*- coding: utf-8 -*-

"""
Regenerate Swin KD Report from existing checkpoint (NO retraining).
- Loads kd_swin_best.pth
- Rebuilds history from training log (hardcoded from user's log)
- Runs test eval + CV5 + CI95 + TTA32
- Saves all JSONs, plots, predictions.csv
- Generates 24-page PDF via kd_reporter.generate_kd_report
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.AI.ResNet.utils import (
    CLASS_NAMES, CROPPED, CSVAddress, validate_csv, LiverDataset, get_device, set_seed,
)
from .kd_convnext_swin import (
    KD_OUTPUT_DIR, STUDENT_BEST, KD_HISTORY, KD_META, KD_METRICS,
    KD_GAP_PNG, KD_PREDICTIONS, ADV_EVAL_JSON, HP,
    build_swin_student, evaluate_model, _compute_metrics, load_teacher,
    TEACHER_CKPT,
)
from .advanced_eval import run_cv5, run_ci95

DEVICE = get_device()
set_seed(42)


# ═════════════════════════════════════════════════════════════════════
# History reconstructed from the user's training log (76 epochs)
# ═════════════════════════════════════════════════════════════════════

HISTORY_FROM_LOG = {
    "train_loss": [0.7825,0.4567,0.4134,0.3913,0.3896,0.3689,0.3460,0.3642,0.3581,0.3162,0.3251,0.3184,0.3237,0.3020,0.2820,0.2887,0.2812,0.2892,0.2833,0.2777,0.2718,0.2660,0.2624,0.2610,0.2588,0.2472,0.2555,0.2531,0.2434,0.2512,0.2418,0.2428,0.2411,0.2428,0.2392,0.2373,0.2269,0.2299,0.2254,0.2269,0.2264,0.2224,0.2274,0.2211,0.2223,0.2186,0.2256,0.2177,0.2138,0.2224,0.2211,0.2170,0.2156,0.2166,0.2179,0.2146,0.2091,0.2079,0.2135,0.2122,0.2094,0.2108,0.2074,0.2076,0.2075,0.2075,0.2157,0.2118,0.2089,0.2058,0.2100,0.2089,0.2095,0.2081,0.2105,0.2081],
    "val_loss":   [0.5334,0.5360,0.4914,0.4724,0.4697,0.5063,0.4808,0.4523,0.4647,0.5268,0.4991,0.4578,0.5322,0.4721,0.5516,0.5248,0.5243,0.4659,0.4695,0.5347,0.5095,0.4694,0.4998,0.5034,0.4939,0.4606,0.5228,0.4791,0.4861,0.4559,0.5038,0.4679,0.4671,0.4688,0.4468,0.4680,0.4825,0.4571,0.4776,0.4522,0.4743,0.4573,0.4655,0.4624,0.4822,0.4774,0.4602,0.4736,0.4598,0.4765,0.4424,0.4908,0.4522,0.4495,0.4722,0.4785,0.4614,0.4420,0.4405,0.4619,0.4681,0.4447,0.4386,0.4558,0.4410,0.4651,0.4467,0.4439,0.4523,0.4440,0.4411,0.4409,0.4546,0.4502,0.4537,0.4452],
    "train_acc":  [0.5630,0.7334,0.7670,0.7765,0.7888,0.7865,0.8158,0.7917,0.8007,0.8305,0.8272,0.8300,0.8210,0.8404,0.8670,0.8575,0.8561,0.8594,0.8561,0.8518,0.8703,0.8788,0.8641,0.8797,0.8793,0.8944,0.8783,0.8821,0.9010,0.8925,0.8954,0.8982,0.9015,0.8958,0.9001,0.9067,0.9181,0.9105,0.9186,0.9148,0.9138,0.9129,0.9157,0.9205,0.9223,0.9233,0.9186,0.9228,0.9271,0.9115,0.9190,0.9205,0.9347,0.9233,0.9167,0.9223,0.9347,0.9318,0.9285,0.9356,0.9366,0.9384,0.9342,0.9299,0.9361,0.9356,0.9171,0.9295,0.9318,0.9366,0.9205,0.9366,0.9408,0.9337,0.9361,0.9304],
    "val_acc":    [0.7698,0.7811,0.7962,0.7887,0.8038,0.8075,0.8226,0.8038,0.8075,0.8000,0.8038,0.8000,0.8075,0.8226,0.7887,0.8038,0.8075,0.8302,0.8302,0.8000,0.8075,0.8340,0.8038,0.8113,0.8038,0.8226,0.8000,0.8151,0.8000,0.8264,0.8113,0.8264,0.8340,0.8226,0.8189,0.8189,0.8038,0.8377,0.8189,0.8302,0.8226,0.8302,0.8340,0.8340,0.8264,0.8113,0.8340,0.8264,0.8340,0.8302,0.8415,0.8226,0.8226,0.8302,0.8189,0.8302,0.8340,0.8302,0.8340,0.8302,0.8226,0.8377,0.8340,0.8302,0.8302,0.8264,0.8340,0.8340,0.8340,0.8377,0.8340,0.8340,0.8302,0.8377,0.8377,0.8340],
    "learning_rates": [1.34e-05,2.58e-05,3.81e-05,5.05e-05,6.29e-05,7.52e-05,8.76e-05,1.00e-04,1.00e-04,9.98e-05,9.96e-05,9.92e-05,9.88e-05,9.83e-05,9.77e-05,9.70e-05,9.62e-05,9.54e-05,9.44e-05,9.34e-05,9.22e-05,9.10e-05,8.98e-05,8.84e-05,8.70e-05,8.55e-05,8.39e-05,8.23e-05,8.06e-05,7.89e-05,7.71e-05,7.52e-05,7.34e-05,7.14e-05,6.94e-05,6.74e-05,6.54e-05,6.33e-05,6.12e-05,5.91e-05,5.70e-05,5.48e-05,5.27e-05,5.05e-05,4.83e-05,4.62e-05,4.40e-05,4.19e-05,3.98e-05,3.77e-05,3.56e-05,3.36e-05,3.16e-05,2.96e-05,2.76e-05,2.58e-05,2.39e-05,2.21e-05,2.04e-05,1.87e-05,1.71e-05,1.55e-05,1.40e-05,1.26e-05,1.12e-05,9.95e-06,8.75e-06,7.63e-06,6.59e-06,5.64e-06,4.77e-06,3.99e-06,3.29e-06,2.69e-06,2.17e-06,1.75e-06],
    "teacher_agreement": [0.593,0.786,0.826,0.827,0.816,0.856,0.866,0.827,0.844,0.861,0.856,0.865,0.858,0.869,0.891,0.885,0.880,0.878,0.882,0.874,0.887,0.894,0.886,0.891,0.894,0.896,0.895,0.896,0.915,0.896,0.907,0.908,0.896,0.887,0.891,0.901,0.919,0.901,0.910,0.911,0.911,0.900,0.909,0.911,0.911,0.917,0.903,0.915,0.914,0.903,0.911,0.915,0.902,0.910,0.910,0.917,0.914,0.915,0.905,0.919,0.913,0.913,0.924,0.911,0.916,0.918,0.908,0.918,0.912,0.923,0.915,0.903,0.897,0.909,0.910,0.912],
    "epoch_times": [69.2]*10 + [65.0]*(76-10),  # approximate
}


def _extract_paths(dataset) -> list:
    """Extract image paths from LiverDataset."""
    df = dataset.df
    return [os.path.join(dataset.img_dir, n) for n in df["liverName"].tolist()]


def run_tta32_liverdataset(
    model, test_ds, n_aug: int = 32, batch_size: int = 32, device: str = "cuda",
) -> dict:
    """TTA-32 using batched DataLoader (fast)."""
    import cv2
    from torch.utils.data import Dataset, DataLoader
    from src.AI.ResNet.utils import normalize_resnet, resize_with_padding, LABEL_MAP

    paths = _extract_paths(test_ds)
    labels = np.array([LABEL_MAP[l] for l in test_ds.df["liverLabel"].tolist()])

    class TTADataset(Dataset):
        def __init__(self, paths, labels, aug_idx, target_size):
            self.paths = paths
            self.labels = labels
            self.aug_idx = aug_idx
            self.target_size = target_size

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            img = resize_with_padding(img, self.target_size)
            if self.aug_idx % 2 == 0:
                img = cv2.flip(img, 1)
            if self.aug_idx % 3 == 0:
                h, w = img.shape
                M = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
            img = img.astype(np.float32) / 255.0
            img = np.stack([img, img, img], axis=0)
            t = torch.tensor(img, dtype=torch.float32)
            t = normalize_resnet(t)
            return t, int(self.labels[i])

    model.eval()
    all_probs = []
    print(f"   Running TTA-{n_aug} (batched, bs={batch_size})...")

    with torch.no_grad():
        for aug_idx in range(n_aug):
            ds = TTADataset(paths, labels, aug_idx, HP["target_size"])
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )
            probs_this_aug = []
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                out = model(x)
                probs_this_aug.append(F.softmax(out, dim=1).cpu().numpy())
            all_probs.append(np.concatenate(probs_this_aug, axis=0))
            print(f"     aug {aug_idx+1}/{n_aug} done")

    stacked = np.stack(all_probs, axis=0)
    avg_prob = stacked.mean(axis=0)
    y_pred = avg_prob.argmax(axis=1)
    y_true = labels

    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc_macro = float(roc_auc_score(y_true, avg_prob, multi_class="ovr", average="macro"))
    except Exception:
        auc_macro = 0.0
    print(f"   TTA-{n_aug}: acc={acc:.4f} | auc={auc_macro:.4f}")
    return {"n_aug": n_aug, "acc": acc, "auc": auc_macro, "n_samples": len(y_true)}


def regen_swin_kd_report():
    """Main entry: load checkpoint, rebuild history, eval, save PDF."""
    print("\n" + "=" * 80)
    print("  🔄 Regenerate Swin KD Report (no retraining)")
    print("=" * 80)

    if not os.path.exists(STUDENT_BEST):
        raise FileNotFoundError(f"Checkpoint not found: {STUDENT_BEST}")

    os.makedirs(KD_OUTPUT_DIR, exist_ok=True)

    # ─── Data ───
    print("📊 Loading data...")
    is_valid, df, msg = validate_csv(CSVAddress)
    if not is_valid:
        raise ValueError(f"CSV validation failed: {msg}")
    df = df[df["liverLabel"] != "UNKNOWN"].copy()
    counts = df["liverLabel"].value_counts()
    valid = counts[counts >= 2].index.tolist()
    df = df[df["liverLabel"].isin(valid)]

    from sklearn.model_selection import train_test_split
    train_val_df, test_df = train_test_split(
        df, test_size=0.10, stratify=df["liverLabel"], random_state=42,
    )
    # (we only need test_df)
    test_ds = LiverDataset(
        test_df, CROPPED, augment=False, target_size=HP["target_size"],
        apply_normalize=True, use_advanced_augment=False, return_path=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=HP["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print(f"   Test: {len(test_ds)}")

    # ─── Model ───
    print("🏗️  Loading student (Swin) from checkpoint...")
    student = build_swin_student(num_classes=len(CLASS_NAMES))
    ckpt = torch.load(STUDENT_BEST, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    student.load_state_dict(state)
    student = student.to(DEVICE).eval()
    print(f"   ✅ Loaded: {STUDENT_BEST}")

    # ─── Teacher ───
    print("🏗️  Loading teacher (ConvNext)...")
    teacher = load_teacher(TEACHER_CKPT, num_classes=len(CLASS_NAMES))

    # ─── Baseline test ───
    print("📊 Baseline TEST evaluation...")
    val_criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_model(student, test_loader, val_criterion)
    print(f"   ✅ Test Acc={test_metrics['accuracy']:.4f} | AUC={test_metrics['auc_macro']:.4f}")

    # ─── Advanced eval ───
    print("\n🔬 Advanced eval (CV5 + CI95 + TTA32)...")
    print("\n   ═══ CV5 ═══")
    cv5 = run_cv5(student, test_ds, n_folds=5, device=DEVICE)
    print("\n   ═══ CI95 ═══")
    ci95 = run_ci95(student, test_ds, n_bootstrap=2000, device=DEVICE)
    print("\n   ═══ TTA-32 ═══")
    tta32 = run_tta32_liverdataset(
        student, test_ds, n_aug=32, batch_size=32, device=DEVICE,
    )
    adv_results = {"cv5": cv5, "ci95": ci95, "tta32": tta32}

    # ─── Save JSONs ───
    with open(KD_HISTORY, "w", encoding="utf-8") as f:
        json.dump(HISTORY_FROM_LOG, f, indent=2, default=str)

    meta = {
        "best_epoch": 50,
        "best_val_acc": 0.8415,
        "baseline_test_acc": float(test_metrics["accuracy"]),
        "baseline_test_auc": float(test_metrics["auc_macro"]),
        "epochs_trained": 76,
        "teacher_ckpt": TEACHER_CKPT,
        "student_ckpt": STUDENT_BEST,
        "hyperparameters": HP,
        "advanced_eval": adv_results,
        "regenerated": True,
    }
    with open(KD_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str, ensure_ascii=False)

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
                } for i in range(len(CLASS_NAMES))
            },
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": test_metrics["classification_report"],
        }),
        "advanced_eval": _clean(adv_results),
    }
    with open(KD_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4, ensure_ascii=False, default=str)

    with open(ADV_EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(_clean(adv_results), f, indent=2, default=str)

    # ─── predictions.csv ───
    pd.DataFrame({
        "true_idx": test_metrics["y_true"],
        "pred_idx": test_metrics["y_pred"],
        "true_label": [CLASS_NAMES[i] for i in test_metrics["y_true"]],
        "pred_label": [CLASS_NAMES[i] for i in test_metrics["y_pred"]],
        "correct": (test_metrics["y_true"] == test_metrics["y_pred"]).astype(int),
        **{f"prob_{c}": test_metrics["y_prob"][:, i] for i, c in enumerate(CLASS_NAMES)},
    }).to_csv(KD_PREDICTIONS, index=False)

    # ─── Gap plot ───
    from .kd_convnext_swin import _plot_gap
    _plot_gap(HISTORY_FROM_LOG, KD_GAP_PNG)
    print(f"   ✅ Gap plot: {KD_GAP_PNG}")

    # ─── PDF ───
    print("\n📄 Generating 24-page PDF report...")
    from .kd_reporter import generate_kd_report
    pdf_path = generate_kd_report(
        model=student, teacher=teacher, test_ds=test_ds,
        test_metrics=test_metrics, history=HISTORY_FROM_LOG,
        adv_results=adv_results, output_dir=KD_OUTPUT_DIR,
        student_name="SwinTransformer", teacher_name="ConvNext", config=HP,
    )

    print("\n" + "=" * 80)
    print("  ✅ REGENERATION COMPLETE")
    print("=" * 80)
    print(f"  Test Acc:   {test_metrics['accuracy']:.4f}")
    print(f"  Test AUC:   {test_metrics['auc_macro']:.4f}")
    print(f"  CV5 Mean:   {cv5['mean_acc']:.4f}")
    print(f"  TTA-32 Acc: {tta32['acc']:.4f}")
    print(f"  PDF:        {pdf_path}")
    print(f"  Dir:        {KD_OUTPUT_DIR}")
    print("=" * 80)

    return {"status": "success", "pdf_path": pdf_path, "test_acc": test_metrics["accuracy"]}