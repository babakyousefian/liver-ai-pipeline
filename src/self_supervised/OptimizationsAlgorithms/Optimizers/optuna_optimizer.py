# src/self_supervised/optuna_optimizer.py

"""
Optuna Hyperparameter Optimization for Self-Supervised Learning (ULTIMATE EDITION)

This module provides automated hyperparameter optimization for SSL methods:
- BYOL, SimCLR, MAE
- Supports ResNet18, ResNet50, EfficientNet-B0, EfficientNet-B1
- TPE Sampler with Median Pruner
- Automatic checkpoint saving
- Comprehensive logging
- Integration with existing pipeline

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 5.0.0 (ULTIMATE EDITION - FIXED)
"""

import copy
import gc
import json
import os
import time
import warnings
from datetime import datetime
from typing import Any, Callable, Dict

import optuna
import torch
import torch.nn as nn
from optuna.trial import Trial
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# ================================================================
# ✅ IMPORTS FROM PROJECT
# ================================================================

from src.self_supervised import (
    create_byol_trainer,
    create_mae_trainer,
    create_simclr_trainer,
    set_seed,
)

# ================================================================
# ✅ CONSTANTS
# ================================================================

CLASS_NAMES = ["NL", "G1", "G2", "G3"]
NUM_CLASSES = len(CLASS_NAMES)

# Default hyperparameter ranges
DEFAULT_SEARCH_SPACE = {
    "batch_size": [8, 16, 32, 64],
    "lr": (1e-5, 1e-2),
    "weight_decay": (1e-6, 1e-3),
    "momentum": (0.85, 0.999),
    "warmup_epochs": (2, 15),
    "temperature": (0.1, 1.0),
    "projection_dim": [64, 128, 256],
    "hidden_dim": [1024, 2048, 4096],
    "dropout_rate": (0.0, 0.3),
    "gradient_accumulation_steps": [1, 2, 4],
    "clip_grad_norm": (0.5, 2.0),
}


# ================================================================
# ✅ SSL TRAINER WRAPPER
# ================================================================


class SSLTrainerWrapper:
    """
    Wrapper for SSL trainers to work with Optuna.

    This class handles:
    - Creating trainer with trial parameters
    - Training with early stopping
    - Returning validation loss for optimization
    """

    def __init__(
        self,
        encoder: nn.Module,
        method: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = 4,
        input_size: int = 224,
        device: str = "cpu",
        verbose: bool = False,
        max_epochs: int = 30,
    ):
        """
        Initialize SSL Trainer Wrapper.

        Args:
            encoder: Base encoder model
            method: SSL method ('byol', 'simclr', 'mae')
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_classes: Number of classes
            input_size: Input image size
            device: Device to run on
            verbose: Print progress
            max_epochs: Maximum epochs for each trial
        """
        self.encoder = encoder
        self.method = method
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = device
        self.verbose = verbose
        self.max_epochs = max_epochs

        # ✅ FIX: Validate train_loader
        if train_loader is None or len(train_loader) == 0:
            raise ValueError("train_loader is None or empty!")

        try:
            batch = next(iter(train_loader))
            if isinstance(batch, (list, tuple)):
                if len(batch) != 2:
                    raise ValueError(f"Expected 2 views, got {len(batch)}")
        except StopIteration:
            raise ValueError("train_loader is empty!")

        # ✅ FIX: Check drop_last safely
        if hasattr(train_loader, "drop_last") and train_loader.drop_last is False:
            print(
                "⚠️ Warning: train_loader.drop_last is False. For SSL (BYOL/SimCLR), drop_last=True is recommended."
            )

        # Store best results
        self.best_loss = float("inf")
        self.best_params = {}
        self.results = []

    def create_trainer(self, trial_params: Dict[str, Any]):
        """
        Create SSL trainer with trial parameters.

        Args:
            trial_params: Dictionary of hyperparameters

        Returns:
            SSL trainer instance
        """
        # Extract parameters with defaults
        method = trial_params.get("method", self.method)
        batch_size = trial_params.get("batch_size", 32)
        lr = trial_params.get("lr", 0.001)
        weight_decay = trial_params.get("weight_decay", 1e-4)
        warmup_epochs = trial_params.get("warmup_epochs", 10)
        temperature = trial_params.get("temperature", 0.5)
        projection_dim = trial_params.get("projection_dim", 256)
        hidden_dim = trial_params.get("hidden_dim", 4096)
        momentum = trial_params.get("momentum", 0.996)
        dropout_rate = trial_params.get("dropout_rate", 0.1)
        gradient_accumulation_steps = trial_params.get("gradient_accumulation_steps", 1)
        clip_grad_norm = trial_params.get("clip_grad_norm", 1.0)

        # Create trainer based on method
        if method == "simclr":
            trainer = create_simclr_trainer(
                encoder=self.encoder,
                input_size=self.input_size,
                projection_dim=projection_dim,
                hidden_dim=hidden_dim,
                temperature=temperature,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                use_lars=False,
                use_mixed_precision=False,
                use_multi_gpu=False,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_grad_norm=clip_grad_norm,
                warmup_epochs=warmup_epochs,
                total_epochs=self.max_epochs,
                color_jitter_strength=1.0,
                blur_kernel_size=23,
                device=self.device,
                verbose=self.verbose,
                num_workers=2,
                save_dir=f"optuna_checkpoints/{method}/trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

        elif method == "byol":
            trainer = create_byol_trainer(
                model=self.encoder,
                model_name="encoder",
                input_size=self.input_size,
                projection_dim=projection_dim,
                hidden_dim=hidden_dim,
                momentum=momentum,
                momentum_schedule="cosine",
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                use_lars=False,
                use_mixed_precision=False,
                use_multi_gpu=False,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_grad_norm=clip_grad_norm,
                warmup_epochs=warmup_epochs,
                total_epochs=self.max_epochs,
                device=self.device,
                verbose=self.verbose,
                num_workers=2,
                augmentation_method="byol",
                save_dir=f"optuna_checkpoints/{method}/trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

        elif method == "mae":
            trainer = create_mae_trainer(
                encoder=self.encoder,
                img_size=self.input_size,
                patch_size=16,
                mask_ratio=0.75,
                encoder_embed_dim=768,
                encoder_depth=12,
                encoder_num_heads=12,
                decoder_embed_dim=512,
                decoder_depth=8,
                decoder_num_heads=8,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                use_mixed_precision=False,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_grad_norm=clip_grad_norm,
                warmup_epochs=warmup_epochs,
                total_epochs=self.max_epochs,
                device=self.device,
                verbose=self.verbose,
                num_workers=2,
                save_dir=f"optuna_checkpoints/{method}/trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

        else:
            raise ValueError(f"Unsupported SSL method: {method}")

        return trainer

    def train_and_evaluate(self, trial_params: Dict[str, Any]) -> float:
        """
        Train and evaluate a trial.

        Args:
            trial_params: Dictionary of hyperparameters

        Returns:
            Validation loss
        """
        # Create trainer
        trainer = self.create_trainer(trial_params)

        # Train with early stopping
        best_val_loss = float("inf")
        best_params = trial_params.copy()
        patience_counter = 0
        patience = trial_params.get("patience", 5)
        min_delta = trial_params.get("min_delta", 1e-4)

        for epoch in range(self.max_epochs):
            # Train one epoch
            train_metrics = trainer.train_epoch(self.train_loader)
            train_loss = train_metrics.get("loss", float("inf"))

            # Validate
            val_metrics = trainer.validate(self.val_loader)
            val_loss = val_metrics.get("loss", float("inf"))

            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{self.max_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
                )

            # Update best
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                if hasattr(trainer, "save_checkpoint"):
                    try:
                        trainer.save_checkpoint(
                            os.path.join(f"optuna_checkpoints/{self.method}/best_trial.pth")
                        )
                    except:
                        pass
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                if self.verbose:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

            # Prune if needed
            if val_loss > best_val_loss * 1.5 and epoch > 5:
                if self.verbose:
                    print(f"✂️ Pruning trial at epoch {epoch+1} (loss divergence)")
                break

        # Clean up memory
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ✅ Store best results
        self.best_loss = best_val_loss
        self.best_params = best_params

        return best_val_loss


# ================================================================
# ✅ OPTUNA OBJECTIVE FUNCTION
# ================================================================


def objective(
    trial: Trial,
    encoder: nn.Module,
    method: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 4,
    input_size: int = 224,
    device: str = "cpu",
    verbose: bool = False,
    max_epochs: int = 30,
    search_space: Dict[str, Any] = None,
) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        encoder: Base encoder model
        method: SSL method ('byol', 'simclr', 'mae')
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_classes: Number of classes
        input_size: Input image size
        device: Device to run on
        verbose: Print progress
        max_epochs: Maximum epochs per trial
        search_space: Custom search space

    Returns:
        Validation loss (lower is better)
    """

    # Use default search space if not provided
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    # ============================================================
    # SUGGEST HYPERPARAMETERS
    # ============================================================

    # Batch size (categorical)
    batch_size = trial.suggest_categorical(
        "batch_size", search_space.get("batch_size", [8, 16, 32, 64])
    )

    # Learning rate (log-uniform)
    lr_min, lr_max = search_space.get("lr", (1e-5, 1e-2))
    lr = trial.suggest_float("lr", lr_min, lr_max, log=True)

    # Weight decay (log-uniform)
    wd_min, wd_max = search_space.get("weight_decay", (1e-6, 1e-3))
    weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)

    # Momentum (for BYOL)
    mom_min, mom_max = search_space.get("momentum", (0.85, 0.999))
    momentum = trial.suggest_float("momentum", mom_min, mom_max)

    # Warmup epochs
    wu_min, wu_max = search_space.get("warmup_epochs", (2, 15))
    warmup_epochs = trial.suggest_int("warmup_epochs", wu_min, wu_max)

    # Temperature (for SimCLR)
    temp_min, temp_max = search_space.get("temperature", (0.1, 1.0))
    temperature = trial.suggest_float("temperature", temp_min, temp_max)

    # Projection dimension
    projection_dim = trial.suggest_categorical(
        "projection_dim", search_space.get("projection_dim", [64, 128, 256])
    )

    # Hidden dimension
    hidden_dim = trial.suggest_categorical(
        "hidden_dim", search_space.get("hidden_dim", [1024, 2048, 4096])
    )

    # Dropout rate
    dp_min, dp_max = search_space.get("dropout_rate", (0.0, 0.3))
    dropout_rate = trial.suggest_float("dropout_rate", dp_min, dp_max)

    # Gradient accumulation
    grad_acc = trial.suggest_categorical(
        "gradient_accumulation_steps",
        search_space.get("gradient_accumulation_steps", [1, 2, 4]),
    )

    # Clip grad norm
    cg_min, cg_max = search_space.get("clip_grad_norm", (0.5, 2.0))
    clip_grad_norm = trial.suggest_float("clip_grad_norm", cg_min, cg_max)

    # ============================================================
    # BUILD CONFIG
    # ============================================================

    trial_params = {
        "method": method,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "warmup_epochs": warmup_epochs,
        "temperature": temperature,
        "projection_dim": projection_dim,
        "hidden_dim": hidden_dim,
        "dropout_rate": dropout_rate,
        "gradient_accumulation_steps": grad_acc,
        "clip_grad_norm": clip_grad_norm,
        "patience": min(10, max(3, warmup_epochs // 2)),
        "min_delta": 1e-4,
    }

    # ============================================================
    # CREATE WRAPPER AND TRAIN
    # ============================================================

    # Create a fresh copy of encoder for each trial
    encoder_copy = copy.deepcopy(encoder)

    wrapper = SSLTrainerWrapper(
        encoder=encoder_copy,
        method=method,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        input_size=input_size,
        device=device,
        verbose=verbose,
        max_epochs=max_epochs,
    )

    # Train and get validation loss
    val_loss = wrapper.train_and_evaluate(trial_params)

    # Clean up
    del encoder_copy
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_loss


# ================================================================
# ✅ MAIN OPTIMIZATION FUNCTION
# ================================================================


def optimize_hyperparameters(
    encoder: nn.Module,
    method: str = "byol",
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    num_classes: int = 4,
    input_size: int = 224,
    device: str = "cpu",
    n_trials: int = 30,
    max_epochs: int = 30,
    verbose: bool = True,
    search_space: Dict[str, Any] = None,
    study_name: str = None,
    storage: str = None,
    load_if_exists: bool = False,
    save_results: bool = True,
    output_dir: str = "optuna_results",
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna.

    Args:
        encoder: Base encoder model
        method: SSL method ('byol', 'simclr', 'mae')
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_classes: Number of classes
        input_size: Input image size
        device: Device to run on
        n_trials: Number of trials
        max_epochs: Maximum epochs per trial
        verbose: Print progress
        search_space: Custom search space
        study_name: Name of the study
        storage: Storage URL for study
        load_if_exists: Load existing study if exists
        save_results: Save results to file
        output_dir: Directory to save results

    Returns:
        Dictionary with best parameters and study results
    """

    # ✅ FIX: Validation MUST be at the BEGINNING
    if train_loader is None or len(train_loader) == 0:
        raise ValueError("train_loader is None or empty!")

    try:
        batch = next(iter(train_loader))
        if isinstance(batch, (list, tuple)):
            if len(batch) != 2:
                raise ValueError(f"Expected 2 views (x1, x2), got {len(batch)}")
    except StopIteration:
        raise ValueError("train_loader is empty!")

    if verbose:
        print("\n" + "=" * 80)
        print("🚀 OPTUNA HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        print(f"   Method: {method.upper()}")
        print(f"   Trials: {n_trials}")
        print(f"   Max Epochs per Trial: {max_epochs}")
        print(f"   Device: {device}")
        print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)

    # Create study
    if study_name is None:
        study_name = f"{method}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if storage is None:
        storage = f"sqlite:///{output_dir}/{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
        load_if_exists=load_if_exists,
    )

    if verbose:
        print(f"\n📊 Study: {study_name}")
        print(f"💾 Storage: {storage}")
        print("\n🔬 Starting optimization...")

    # Optimize
    start_time = time.time()

    def objective_wrapper(trial):
        return objective(
            trial=trial,
            encoder=encoder,
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            input_size=input_size,
            device=device,
            verbose=verbose,
            max_epochs=max_epochs,
            search_space=search_space,
        )

    try:
        study.optimize(
            objective_wrapper,
            n_trials=n_trials,
            n_jobs=1,
            timeout=None,
            show_progress_bar=verbose,
        )
    except KeyboardInterrupt:
        if verbose:
            print("\n⚠️ Optimization interrupted by user")

    elapsed_time = time.time() - start_time

    # Get best results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    if verbose:
        print("\n" + "=" * 80)
        print("🏆 OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print(f"   Best Loss: {best_value:.6f}")
        print(f"   Best Trial: {best_trial.number}")
        print(f"   Total Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
        print("\n📋 Best Parameters:")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        print("=" * 80)

    # Prepare results
    results = {
        "study_name": study_name,
        "method": method,
        "best_params": best_params,
        "best_value": best_value,
        "best_trial_number": best_trial.number,
        "n_trials": len(study.trials),
        "elapsed_time": elapsed_time,
        "datetime": datetime.now().isoformat(),
        "study": study,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }

    # Save results
    if save_results:
        results_path = os.path.join(output_dir, f"{method}_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "best_params": best_params,
                    "best_value": best_value,
                    "n_trials": len(study.trials),
                    "elapsed_time": elapsed_time,
                    "datetime": results["datetime"],
                },
                f,
                indent=4,
            )

        if verbose:
            print(f"\n💾 Results saved to: {results_path}")

        # Save best params as config
        config_path = os.path.join(output_dir, f"{method}_best_config.json")
        with open(config_path, "w") as f:
            json.dump(best_params, f, indent=4)

        if verbose:
            print(f"💾 Best config saved to: {config_path}")

    return results


# ================================================================
# ✅ HELPER FUNCTIONS
# ================================================================


def get_best_params_from_study(study_name: str, storage: str = None) -> Dict[str, Any]:
    """
    Load best parameters from existing study.

    Args:
        study_name: Name of the study
        storage: Storage URL

    Returns:
        Dictionary with best parameters
    """
    if storage is None:
        storage = f"sqlite:///optuna_results/{study_name}.db"

    study = optuna.load_study(study_name=study_name, storage=storage)
    return study.best_params


def plot_optimization_history(study_name: str, storage: str = None, output_path: str = None):
    """
    Plot optimization history.

    Args:
        study_name: Name of the study
        storage: Storage URL
        output_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt

        if storage is None:
            storage = f"sqlite:///optuna_results/{study_name}.db"

        study = optuna.load_study(study_name=study_name, storage=storage)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Optimization history
        trials = study.trials
        values = [t.value for t in trials if t.value is not None]
        numbers = [t.number for t in trials if t.value is not None]

        axes[0, 0].plot(numbers, values, "b-", linewidth=2)
        axes[0, 0].set_xlabel("Trial")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Optimization History")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Best value progress
        best_values = []
        best = float("inf")
        for v in values:
            if v < best:
                best = v
            best_values.append(best)

        axes[0, 1].plot(numbers, best_values, "g-", linewidth=2)
        axes[0, 1].set_xlabel("Trial")
        axes[0, 1].set_ylabel("Best Loss")
        axes[0, 1].set_title("Best Value Progress")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Parameter importances
        try:
            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())
            imp_values = list(importances.values())

            axes[1, 0].barh(params, imp_values, color="skyblue")
            axes[1, 0].set_xlabel("Importance")
            axes[1, 0].set_title("Parameter Importances")
        except:
            pass

        # 4. Learning rate distribution
        lr_values = []
        for t in trials:
            if t.params and "lr" in t.params:
                lr_values.append(t.params["lr"])
        if lr_values:
            axes[1, 1].hist(lr_values, bins=20, color="orange", alpha=0.7)
            axes[1, 1].set_xlabel("Learning Rate")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Learning Rate Distribution")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✅ Plot saved to: {output_path}")

        plt.close()

    except Exception as e:
        print(f"⚠️ Could not plot optimization history: {e}")


# ================================================================
# ✅ FACTORY FUNCTION
# ================================================================


def create_optuna_optimizer(
    encoder: nn.Module,
    method: str = "byol",
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    num_classes: int = 4,
    input_size: int = 224,
    device: str = "cpu",
    n_trials: int = 30,
    max_epochs: int = 30,
    verbose: bool = True,
    search_space: Dict[str, Any] = None,
    output_dir: str = "optuna_results",
) -> Callable:
    """
    Factory function to create Optuna optimizer.

    Returns:
        Function that returns best parameters when called
    """

    def optimizer():
        return optimize_hyperparameters(
            encoder=encoder,
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            input_size=input_size,
            device=device,
            n_trials=n_trials,
            max_epochs=max_epochs,
            verbose=verbose,
            search_space=search_space,
            output_dir=output_dir,
        )

    return optimizer


# ================================================================
# ✅ TESTING
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Optuna Optimizer (ULTIMATE EDITION - FIXED)")
    print("=" * 80)

    # Create dummy encoder
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dim = 128
            self.conv = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 128),
            )

        def get_embedding(self, x):
            return self.conv(x)

        def forward(self, x):
            return self.conv(x)

    # Create dummy data
    from torch.utils.data import DataLoader, TensorDataset

    dummy_images = torch.randn(64, 3, 224, 224)
    dummy_labels = torch.zeros(64)
    dataset = TensorDataset(dummy_images, dummy_labels)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Test optimization
    print("\n📌 Testing optimization...")
    encoder = DummyEncoder()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = optimize_hyperparameters(
        encoder=encoder,
        method="byol",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_trials=3,
        max_epochs=2,
        verbose=True,
        output_dir="test_optuna_results",
    )

    print("\n📌 Testing get_best_params...")
    best = get_best_params_from_study(results["study_name"])
    print(f"   Best params: {best}")

    print("\n✅ All Optuna tests passed!")
    print("=" * 80)
