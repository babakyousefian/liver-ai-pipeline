# src/self_supervised/early_stopping.py

"""
Early Stopping for Self-Supervised Learning (ULTIMATE EDITION)

This module provides a flexible and intelligent early stopping mechanism
that works with all SSL methods (SimCLR, BYOL, MAE).

Features:
- Patience-based stopping with configurable patience
- Minimum delta for improvement
- Best model checkpoint saving
- Automatic restoration of best weights
- Detailed logging and tracking
- Works with all SSL trainers

Author: Babak Yousefian
Project: Liver AI Pipeline - Task 5
Version: 5.0.0
"""

import copy
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union

import torch


class EarlyStopping:
    """
    Intelligent Early Stopping for SSL training.

    This class monitors the validation loss and stops training when
    no improvement is seen for a specified number of epochs.

    Features:
    - Configurable patience (default: 20)
    - Minimum delta for improvement (default: 1e-4)
    - Saves the best model automatically
    - Restores best model weights when stopping
    - Tracks best loss and epoch
    - Logs improvement history
    - Works with SimCLR, BYOL, and MAE trainers

    Example:
        >>> early_stopping = EarlyStopping(patience=20, min_delta=1e-4)
        >>>
        >>> for epoch in range(epochs):
        ...     train_loss = trainer.train_epoch(train_loader)
        ...     val_loss = trainer.validate(val_loader)
        ...
        ...     if early_stopping(val_loss, trainer, epoch):
        ...         print(f"Early stopping triggered at epoch {epoch}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        verbose: bool = True,
        save_best: bool = True,
        restore_best: bool = True,
        checkpoint_dir: str = "early_stopping_checkpoints",
    ):
        """
        Initialize Early Stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print progress messages
            save_best: Whether to save the best model checkpoint
            restore_best: Whether to restore best weights on stop
            checkpoint_dir: Directory to save best checkpoints
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_best = save_best
        self.restore_best = restore_best
        self.checkpoint_dir = checkpoint_dir

        # Internal state
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state_dict = None
        self.best_config = None
        self.is_stopped = False
        self.history = {
            "epochs": [],
            "losses": [],
            "improvements": [],
            "counter_values": [],
        }

        # Create checkpoint directory
        if self.save_best:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.verbose:
            self._print_init_info()

    def _print_init_info(self):
        """Print initialization information."""
        print("\n" + "=" * 60)
        print("🧠 Initializing Early Stopping (ULTIMATE EDITION)")
        print("=" * 60)
        print(f"   Patience: {self.patience} epochs")
        print(f"   Min Delta: {self.min_delta}")
        print(f"   Save Best: {self.save_best}")
        print(f"   Restore Best: {self.restore_best}")
        print("=" * 60)

    def __call__(self, current_loss: float, trainer: Any, epoch: int, **kwargs) -> bool:
        """
        Check if training should stop.

        Args:
            current_loss: Current validation loss
            trainer: Trainer instance (SimCLRTrainer, BYOLTrainer, MAETrainer)
            epoch: Current epoch number
            **kwargs: Additional arguments (model_name, method, etc.)

        Returns:
            bool: True if training should stop, False otherwise
        """
        # Record history
        self.history["epochs"].append(epoch)
        self.history["losses"].append(current_loss)
        self.history["counter_values"].append(self.counter)

        # Check for improvement
        if current_loss < self.best_loss - self.min_delta:
            # Improvement detected
            improvement = self.best_loss - current_loss
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.counter = 0

            # Save best model state
            self._save_best_state(trainer, epoch)

            if self.verbose:
                print(
                    f"📈 Improvement detected at epoch {epoch}: "
                    f"{current_loss:.6f} (Δ = {improvement:.6f})"
                )

            self.history["improvements"].append(True)

        else:
            # No improvement
            self.counter += 1
            self.history["improvements"].append(False)

            if self.verbose:
                print(
                    f"⏳ No improvement at epoch {epoch}. "
                    f"Counter: {self.counter}/{self.patience}"
                )

        # Check if should stop
        if self.counter >= self.patience:
            self.is_stopped = True

            if self.verbose:
                print("\n" + "=" * 60)
                print(f"🛑 Early Stopping Triggered!")
                print(f"   Best Loss: {self.best_loss:.6f} (Epoch {self.best_epoch})")
                print(f"   Current Loss: {current_loss:.6f}")
                print("=" * 60)

            # Restore best model weights
            if self.restore_best and self.best_state_dict is not None:
                self._restore_best_state(trainer)

            return True

        return False

    def _save_best_state(self, trainer: Any, epoch: int):
        """
        Save the best model state.

        Args:
            trainer: Trainer instance
            epoch: Current epoch number
        """
        # Save state dict
        if hasattr(trainer, "get_encoder"):
            # For SSL trainers (SimCLR, BYOL, MAE)
            if hasattr(trainer, "online_encoder"):
                # BYOL
                encoder = trainer.online_encoder
            elif hasattr(trainer, "encoder"):
                # SimCLR, MAE
                encoder = trainer.encoder
            else:
                encoder = trainer.model if hasattr(trainer, "model") else None

            if encoder is not None:
                self.best_state_dict = copy.deepcopy(encoder.state_dict())

        # Save config
        if hasattr(trainer, "get_config"):
            self.best_config = trainer.get_config()

        # Save to disk
        if self.save_best:
            self._save_checkpoint(trainer, epoch)

    def _save_checkpoint(self, trainer: Any, epoch: int):
        """
        Save checkpoint to disk.

        Args:
            trainer: Trainer instance
            epoch: Current epoch number
        """
        # Determine method name
        method_name = "unknown"
        if hasattr(trainer, "method"):
            method_name = trainer.method
        elif hasattr(trainer, "online_encoder"):
            # BYOL
            method_name = "byol"
        elif hasattr(trainer, "encoder"):
            method_name = "simclr" if hasattr(trainer, "projector") else "mae"

        # Determine model name
        model_name = "model"
        if hasattr(trainer, "model_name"):
            model_name = trainer.model_name
        elif hasattr(trainer, "encoder") and hasattr(trainer.encoder, "__class__"):
            model_name = trainer.encoder.__class__.__name__.lower()

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{method_name}_best_epoch_{epoch}_{timestamp}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "state_dict": self.best_state_dict,
            "config": self.best_config,
            "history": {
                "best_epoch": self.best_epoch,
                "counter": self.counter,
                "patience": self.patience,
                "min_delta": self.min_delta,
            },
            "timestamp": timestamp,
        }

        torch.save(checkpoint, filepath)

        if self.verbose:
            print(f"💾 Best model saved: {filepath}")

    def _restore_best_state(self, trainer: Any):
        """
        Restore the best model weights.

        Args:
            trainer: Trainer instance
        """
        if self.best_state_dict is None:
            return

        # Determine which encoder to restore
        if hasattr(trainer, "online_encoder"):
            # BYOL
            trainer.online_encoder.load_state_dict(self.best_state_dict)
            if self.verbose:
                print(f"🔄 Restored best weights to online_encoder")

        elif hasattr(trainer, "encoder"):
            # SimCLR, MAE
            trainer.encoder.load_state_dict(self.best_state_dict)
            if self.verbose:
                print(f"🔄 Restored best weights to encoder")

        elif hasattr(trainer, "model"):
            trainer.model.load_state_dict(self.best_state_dict)
            if self.verbose:
                print(f"🔄 Restored best weights to model")

        else:
            if self.verbose:
                print(f"⚠️ Could not restore best weights: no encoder found")

    def reset(self):
        """
        Reset the early stopping state.

        Use this when starting a new training run.
        """
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state_dict = None
        self.best_config = None
        self.is_stopped = False
        self.history = {
            "epochs": [],
            "losses": [],
            "improvements": [],
            "counter_values": [],
        }

        if self.verbose:
            print("🔄 Early Stopping reset")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the early stopping state.

        Returns:
            Dictionary with summary information
        """
        return {
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "patience": self.patience,
            "is_stopped": self.is_stopped,
            "history": self.history,
            "total_epochs_monitored": len(self.history["epochs"]),
        }

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, "
            f"min_delta={self.min_delta}, "
            f"best_loss={self.best_loss:.6f}, "
            f"counter={self.counter}/{self.patience})"
        )


# ================================================================
# ✅ Factory Function
# ================================================================


def create_early_stopping(
    patience: int = 20,
    min_delta: float = 1e-4,
    verbose: bool = True,
    save_best: bool = True,
    restore_best: bool = True,
    checkpoint_dir: str = "early_stopping_checkpoints",
) -> EarlyStopping:
    """
    Factory function to create an EarlyStopping instance.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        verbose: Whether to print progress messages
        save_best: Whether to save the best model checkpoint
        restore_best: Whether to restore best weights on stop
        checkpoint_dir: Directory to save best checkpoints

    Returns:
        EarlyStopping instance

    Example:
        >>> early_stopping = create_early_stopping(patience=20, verbose=True)
    """
    return EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        verbose=verbose,
        save_best=save_best,
        restore_best=restore_best,
        checkpoint_dir=checkpoint_dir,
    )
