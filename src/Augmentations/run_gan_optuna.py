# run_gan_optuna.py
"""
Optuna-based hyperparameter optimization for Ultimate GAN.

Usage:
    python run_gan_optuna.py --class_name G3 --n_trials 20 --max_epochs 50

Author: Babak Yousefian
Version: 1.0.0
"""

import argparse
import json
import os
import random
import sys

import cv2
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_gan_ultimate import ClassDataset
from src.self_supervised.OptimizationsAlgorithms.Augmentations.fid_score import calculate_fid_from_paths
from src.self_supervised.OptimizationsAlgorithms.Augmentations.gan_trainer import create_ultimate_gan_trainer


# ================================================================
# ✅ OBJECTIVE FUNCTION
# ================================================================


def objective(trial, args, class_paths, device):
    """Optuna objective function."""

    # ✅ Suggest hyperparameters
    lr_g = trial.suggest_float("lr_g", 1e-5, 5e-4, log=True)
    lr_d = trial.suggest_float("lr_d", 1e-5, 5e-4, log=True)
    latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 192])
    feature_maps = trial.suggest_categorical("feature_maps", [32, 48, 64])
    num_blocks = trial.suggest_int("num_blocks", 4, 6)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    fm_weight = trial.suggest_float("fm_weight", 5.0, 30.0)
    r1_weight = trial.suggest_float("r1_weight", 1.0, 30.0)
    use_attention = trial.suggest_categorical("use_attention", [True, False])
    loss_type = trial.suggest_categorical("loss_type", ["hinge", "wgan"])

    # ✅ Create dataloader
    dataset = ClassDataset(class_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ✅ Create trainer
    trainer = create_ultimate_gan_trainer(
        latent_dim=latent_dim,
        img_shape=(3, 224, 224),
        device=device,
        lr_g=lr_g,
        lr_d=lr_d,
        feature_maps=feature_maps,
        num_blocks=num_blocks,
        use_feature_matching=True,
        feature_matching_weight=fm_weight,
        use_r1=True,
        r1_weight=r1_weight,
        use_ada=True,
        use_ema=True,
        use_attention=use_attention,
        loss_type=loss_type,
        n_critic=1,
        verbose=False,
    )

    # ✅ Train
    save_dir = os.path.join(args.output_dir, args.class_name, f"trial_{trial.number}")
    os.makedirs(save_dir, exist_ok=True)

    history = trainer.train(
        dataloader=dataloader,
        epochs=args.max_epochs,
        save_interval=1000,  # Don't save during Optuna
        save_dir=save_dir,
        generate_interval=1000,  # Don't generate during Optuna
    )

    # ✅ Compute FID
    fake_images = trainer.generate_images(200, use_ema=True)
    temp_dir = os.path.join(save_dir, "temp_fid")
    os.makedirs(temp_dir, exist_ok=True)
    for i, img in enumerate(fake_images):
        cv2.imwrite(
            os.path.join(temp_dir, f"temp_{i:04d}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

    real_sample = random.sample(class_paths, min(200, len(class_paths)))
    fake_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]

    fid = calculate_fid_from_paths(
        real_paths=real_sample,
        fake_paths=fake_paths[:200],
        device=device,
        verbose=False,
    )

    # Cleanup
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)
    del trainer
    torch.cuda.empty_cache()

    # ✅ Report intermediate value (for pruning)
    trial.report(fid, args.max_epochs)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return fid


# ================================================================
# ✅ MAIN
# ================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, default="G3")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=12 * 3600, help="Total timeout in seconds")
    parser.add_argument("--labels_csv", type=str, default="finalReports/labels.csv")
    parser.add_argument("--image_dir", type=str, default="finalReports/cropped")
    parser.add_argument("--output_dir", type=str, default="finalReports/selfSupervisedLearning/GAN_Optuna")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ✅ Load data
    df = pd.read_csv(args.labels_csv)
    class_paths = []
    for _, row in df.iterrows():
        if row["liverLabel"] == args.class_name:
            img_name = row["liverName"]
            for ext in ["", ".png", ".jpg"]:
                img_path = os.path.join(args.image_dir, img_name + ext)
                if os.path.exists(img_path):
                    class_paths.append(img_path)
                    break

    print(f"\n📊 Class: {args.class_name}")
    print(f"   Images: {len(class_paths)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ Create Optuna study
    study_name = f"gan_{args.class_name}_{optuna.datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///{args.output_dir}/{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        load_if_exists=True,
    )

    print(f"\n🚀 Starting Optuna optimization")
    print(f"   Trials: {args.n_trials}")
    print(f"   Max epochs per trial: {args.max_epochs}")
    print(f"   Timeout: {args.timeout / 3600:.1f} hours")
    print(f"   Study: {study_name}")

    # ✅ Optimize
    study.optimize(
        lambda trial: objective(trial, args, class_paths, device),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # ✅ Report best
    print("\n" + "=" * 70)
    print("🏆 OPTUNA OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"   Best FID: {study.best_value:.2f}")
    print(f"   Best Trial: {study.best_trial.number}")
    print(f"   Total trials: {len(study.trials)}")

    print("\n📋 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")

    # ✅ Save best params
    best_params_path = os.path.join(args.output_dir, f"{args.class_name}_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({
            "class_name": args.class_name,
            "best_fid": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }, f, indent=4)

    print(f"\n💾 Best params saved to: {best_params_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    