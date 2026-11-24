# src/pipeLine/SelectBestModels.py
# -*- coding: utf-8 -*-

"""
Select Best Models — Pretrained vs Scratch (STRICTLY by test_accuracy)
=======================================================================
For each of 7 architectures:
  1) Read `test_accuracy` from the corresponding `reports/metrics.json`
     - Pretrained : finalReports/AI_Comparisons/<Model>/reports/metrics.json
     - Scratch    : finalReports/models/<Model>/reports/metrics.json
     - VGG16 scratch has a special path: `vgg16_task5_report.json`
  2) Pick the winner by HIGHEST test_accuracy.
  3) Copy the winner `.pth` checkpoint to:
        finalReports/bestModels/<ModelName>/<modelname>BEST.pth

Also writes `summary.json` and `summary.txt` in `bestModels/`.

Author: Babak Yousefian
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional

import torch


# =====================================================
# BASE PATHS
# =====================================================

FINAL_REPORTS_DIR = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports"

BEST_MODELS_DIR = os.path.join(FINAL_REPORTS_DIR, "bestModels")


# =====================================================
# MODEL CONFIG
# =====================================================
# For each model, we define:
#   - metrics_json_pretrained : path to reports/metrics.json for pretrained
#   - metrics_json_scratch    : path to reports/metrics.json for scratch
#   - ckpt_pretrained         : path to .pth for pretrained
#   - ckpt_scratch            : path to .pth for scratch
#   - output_name             : filename to save in bestModels/<Model>/

MODEL_PAIRS = {
    "ConvNext": {
        "metrics_json_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "ConvNext", "reports", "metrics.json"
        ),
        "metrics_json_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "ConvNext", "reports", "metrics.json"
        ),
        "ckpt_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "ConvNext", "convnext", "ConvNext_best.pth"
        ),
        "ckpt_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "ConvNext", "checkpoints", "ConvNext_best.pth"
        ),
        "output_name": "convnextBEST.pth",
    },

    "DenseNet": {
        "metrics_json_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "DensNet", "reports", "metrics.json"
        ),
        "metrics_json_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "DenseNet", "reports", "metrics.json"
        ),
        "ckpt_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "DensNet", "DensNet", "DensNet_best.pth"
        ),
        "ckpt_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "DenseNet", "checkpoints", "DenseNet_best.pth"
        ),
        "output_name": "densenetBEST.pth",
    },

    "VGG16": {
        "metrics_json_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "VGG16", "reports", "metrics.json"
        ),
        # ⚠️ VGG16 scratch uses a different filename
        "metrics_json_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "VGG16", "reports", "vgg16_task5_report.json"
        ),
        "ckpt_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "VGG16", "vgg16", "vgg16_best.pth"
        ),
        "ckpt_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "VGG16", "checkpoints", "vgg16_best.pth"
        ),
        "output_name": "vgg16BEST.pth",
    },

    "VisionTransformer": {
        "metrics_json_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "visioTransformer", "reports", "metrics.json"
        ),
        "metrics_json_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "visio_transformer", "reports", "metrics.json"
        ),
        "ckpt_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "visioTransformer", "visioTransformer",
            "visioTransformer_best.pth"
        ),
        "ckpt_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "visio_transformer", "checkpoints",
            "visio_transformer_best.pth"
        ),
        "output_name": "visiontransformerBEST.pth",
    },

    "AlexNet": {
        "metrics_json_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "AlexNet", "reports", "metrics.json"
        ),
        "metrics_json_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "AlexNet", "reports", "metrics.json"
        ),
        "ckpt_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "AlexNet", "AlexNet", "AlexNet_best.pth"
        ),
        "ckpt_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "AlexNet", "checkpoints", "alexnet_best.pth"
        ),
        "output_name": "alexnetBEST.pth",
    },

    "CNNLSTM": {
        "metrics_json_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "CNNLSTM", "reports", "metrics.json"
        ),
        "metrics_json_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "CNNLSTM", "reports", "metrics.json"
        ),
        "ckpt_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "CNNLSTM", "CNNLSTM", "CNNLSTM_best.pth"
        ),
        "ckpt_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "CNNLSTM", "checkpoints", "cnnlstm_best.pth"
        ),
        "output_name": "cnnlstmBEST.pth",
    },

    "MobileNetV2": {
        "metrics_json_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "MobileNetV2", "reports", "metrics.json"
        ),
        "metrics_json_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "MobileNetV2", "reports", "metrics.json"
        ),
        "ckpt_pretrained": os.path.join(
            FINAL_REPORTS_DIR, "AI_Comparisons", "MobileNetV2", "mobilenetv2",
            "mobilenetv2_best.pth"
        ),
        "ckpt_scratch": os.path.join(
            FINAL_REPORTS_DIR, "models", "MobileNetV2", "checkpoints", "mobilenetV2_best.pth"
        ),
        "output_name": "mobilenetv2BEST.pth",
    },
}


# =====================================================
# HELPERS
# =====================================================

def _try_load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file safely; return None if missing/corrupt."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"   ⚠️ Could not load JSON {os.path.basename(path)}: {e}")
        return None


def _extract_test_acc_from_metrics_json(data: Dict[str, Any]) -> Optional[float]:
    """
    Extract test_accuracy from a metrics.json.
    Handles multiple layouts:
      A) {"config": {"test_accuracy": X}, "metrics": {...}}     ← our Reporter format
      B) {"metrics": {"accuracy": X}, ...}                       ← sklearn format
      C) {"test_accuracy": X, ...}                                ← flat format
      D) {"metrics": {"test_accuracy": X}, ...}
      E) {"accuracy": X, ...}
    """
    if not isinstance(data, dict):
        return None

    # A) config.test_accuracy (our Reporter writes this)
    cfg = data.get("config")
    if isinstance(cfg, dict):
        for key in ("test_accuracy", "test_acc"):
            if key in cfg and cfg[key] is not None:
                try:
                    return float(cfg[key])
                except (TypeError, ValueError):
                    pass

    # B) metrics.accuracy  (sklearn)
    m = data.get("metrics")
    if isinstance(m, dict):
        for key in ("test_accuracy", "test_acc", "accuracy"):
            if key in m and m[key] is not None:
                try:
                    return float(m[key])
                except (TypeError, ValueError):
                    pass

    # C) Top-level test_accuracy / test_acc
    for key in ("test_accuracy", "test_acc"):
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                pass

    # D) Top-level accuracy
    if "accuracy" in data and data["accuracy"] is not None:
        try:
            return float(data["accuracy"])
        except (TypeError, ValueError):
            pass

    return None


# =====================================================
# MAIN FUNCTION
# =====================================================

def select_best_models() -> Dict[str, Any]:
    """
    Compare pretrained vs scratch for each architecture (STRICTLY by test_accuracy).
    test_accuracy is read from `reports/metrics.json`.
    Winner's `.pth` is copied to `bestModels/<ModelName>/<output_name>`.
    """
    print("\n" + "=" * 78)
    print("  🏆 SELECTING BEST MODELS (Pretrained vs Scratch) — by TEST ACC")
    print("=" * 78)

    os.makedirs(BEST_MODELS_DIR, exist_ok=True)

    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "best_models_dir": BEST_MODELS_DIR,
        "scoring_metric": "test_accuracy",
        "source_of_metric": "reports/metrics.json",
        "pairs": {},
    }

    for model_name, cfg in MODEL_PAIRS.items():
        print(f"\n📌 {model_name}")

        # ---- Paths ----
        pre_json = cfg["metrics_json_pretrained"]
        scr_json = cfg["metrics_json_scratch"]
        pre_ckpt = cfg["ckpt_pretrained"]
        scr_ckpt = cfg["ckpt_scratch"]
        output_name = cfg["output_name"]

        pre_json_exists = os.path.exists(pre_json)
        scr_json_exists = os.path.exists(scr_json)
        pre_ckpt_exists = os.path.exists(pre_ckpt)
        scr_ckpt_exists = os.path.exists(scr_ckpt)

        print(f"   Pretrained JSON : {'✅' if pre_json_exists else '❌'}  {pre_json}")
        print(f"   Scratch    JSON : {'✅' if scr_json_exists else '❌'}  {scr_json}")
        print(f"   Pretrained CKPT : {'✅' if pre_ckpt_exists else '❌'}  {pre_ckpt}")
        print(f"   Scratch    CKPT : {'✅' if scr_ckpt_exists else '❌'}  {scr_ckpt}")

        # ---- Read test_accuracy ----
        pre_data = _try_load_json(pre_json) if pre_json_exists else None
        scr_data = _try_load_json(scr_json) if scr_json_exists else None

        pre_score = _extract_test_acc_from_metrics_json(pre_data) if pre_data else None
        scr_score = _extract_test_acc_from_metrics_json(scr_data) if scr_data else None

        pre_str = f"{pre_score:.4f}" if pre_score is not None else "N/A"
        scr_str = f"{scr_score:.4f}" if scr_score is not None else "N/A"
        print(f"   Pretrained test_acc: {pre_str}")
        print(f"   Scratch    test_acc: {scr_str}")

        # ---- Decision ----
        winner: Optional[str] = None
        winner_ckpt: Optional[str] = None
        winner_score: Optional[float] = None

        has_pre = (pre_score is not None and pre_score > 0.0 and pre_ckpt_exists)
        has_scr = (scr_score is not None and scr_score > 0.0 and scr_ckpt_exists)

        if not has_pre and not has_scr:
            print(f"   ❌ No valid (test_acc > 0 + .pth exists) for either — skipping")
            results["pairs"][model_name] = {
                "pretrained_json": pre_json,
                "pretrained_json_exists": pre_json_exists,
                "pretrained_score": pre_score,
                "pretrained_ckpt": pre_ckpt,
                "pretrained_ckpt_exists": pre_ckpt_exists,
                "scratch_json": scr_json,
                "scratch_json_exists": scr_json_exists,
                "scratch_score": scr_score,
                "scratch_ckpt": scr_ckpt,
                "scratch_ckpt_exists": scr_ckpt_exists,
                "winner": None,
                "winner_score": None,
                "output_path": None,
                "status": "missing_or_invalid",
            }
            continue

        if has_pre and not has_scr:
            winner, winner_ckpt, winner_score = "pretrained", pre_ckpt, pre_score
            print(f"   ℹ️ Scratch has no valid test_acc → choosing PRETRAINED")
        elif has_scr and not has_pre:
            winner, winner_ckpt, winner_score = "scratch", scr_ckpt, scr_score
            print(f"   ℹ️ Pretrained has no valid test_acc → choosing SCRATCH")
        else:
            if pre_score >= scr_score:
                winner, winner_ckpt, winner_score = "pretrained", pre_ckpt, pre_score
            else:
                winner, winner_ckpt, winner_score = "scratch", scr_ckpt, scr_score

        # ---- Copy .pth ----
        out_dir = os.path.join(BEST_MODELS_DIR, model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, output_name)

        try:
            shutil.copy2(winner_ckpt, out_path)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"   ✅ Winner: {winner.upper()} (test_acc={winner_score:.4f})")
            print(f"      → {out_path}  ({size_mb:.2f} MB)")
            status = "copied"
        except Exception as e:
            print(f"   ❌ Copy failed: {e}")
            out_path = None
            status = "copy_failed"

        results["pairs"][model_name] = {
            "pretrained_json": pre_json,
            "pretrained_json_exists": pre_json_exists,
            "pretrained_score": pre_score,
            "pretrained_ckpt": pre_ckpt,
            "pretrained_ckpt_exists": pre_ckpt_exists,
            "scratch_json": scr_json,
            "scratch_json_exists": scr_json_exists,
            "scratch_score": scr_score,
            "scratch_ckpt": scr_ckpt,
            "scratch_ckpt_exists": scr_ckpt_exists,
            "winner": winner,
            "winner_score": winner_score,
            "output_path": out_path,
            "status": status,
        }

    # ---- Save JSON ----
    json_path = os.path.join(BEST_MODELS_DIR, "summary.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=str)
        print(f"\n✅ Summary JSON: {json_path}")
    except Exception as e:
        print(f"⚠️ JSON save failed: {e}")

    # ---- Save TXT ----
    txt_path = os.path.join(BEST_MODELS_DIR, "summary.txt")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 78 + "\n")
            f.write("  BEST MODELS SUMMARY (Pretrained vs Scratch) — by TEST ACC\n")
            f.write("=" * 78 + "\n\n")
            f.write(f"  Generated : {results['timestamp']}\n")
            f.write(f"  Output dir: {BEST_MODELS_DIR}\n")
            f.write(f"  Metric    : test_accuracy (from reports/metrics.json)\n\n")
            f.write("-" * 78 + "\n")
            f.write(f"  {'Model':<20s} {'Winner':<14s} {'TestAcc':>10s}\n")
            f.write("-" * 78 + "\n")
            for name, info in results["pairs"].items():
                w = info.get("winner") or "NONE"
                s = info.get("winner_score")
                s_str = f"{s:.4f}" if s is not None else "N/A"
                f.write(f"  {name:<20s} {w:<14s} {s_str:>10s}\n")
            f.write("-" * 78 + "\n\n")
            f.write("  Detailed:\n")
            for name, info in results["pairs"].items():
                f.write(f"\n  🔹 {name}\n")
                ps = info.get("pretrained_score")
                ss = info.get("scratch_score")
                f.write(f"     Pretrained: "
                        f"{f'{ps:.4f}' if ps is not None else 'N/A'}"
                        f"  json_exists={info.get('pretrained_json_exists')}"
                        f"  ckpt_exists={info.get('pretrained_ckpt_exists')}\n")
                f.write(f"     Scratch   : "
                        f"{f'{ss:.4f}' if ss is not None else 'N/A'}"
                        f"  json_exists={info.get('scratch_json_exists')}"
                        f"  ckpt_exists={info.get('scratch_ckpt_exists')}\n")
                f.write(f"     Winner    : {info.get('winner')}\n")
                f.write(f"     Copied to : {info.get('output_path')}\n")
            f.write("\n" + "=" * 78 + "\n")
            f.write("  Report generated by Liver AI Pipeline\n")
            f.write("  developed by : babak yousefian\n")
            f.write("=" * 78 + "\n")
        print(f"✅ Summary TXT: {txt_path}")
    except Exception as e:
        print(f"⚠️ TXT save failed: {e}")

    # ---- Final ----
    print("\n" + "=" * 78)
    print("  📊 FINAL SUMMARY")
    print("=" * 78)
    for name, info in results["pairs"].items():
        w = info.get("winner") or "NONE"
        s = info.get("winner_score")
        s_str = f"{s:.4f}" if s is not None else "N/A"
        print(f"   {name:<20s} → {w:<12s}  (test_acc={s_str})")
    print("=" * 78 + "\n")

    return results


if __name__ == "__main__":
    select_best_models()
