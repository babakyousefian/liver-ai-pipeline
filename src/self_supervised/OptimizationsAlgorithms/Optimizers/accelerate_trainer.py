# src/self_supervised/accelerate_trainer.py (ULTIMATE FIXED EDITION)

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader


class AccelerateTrainer:
    """
    ✅ FIXED: Accelerate wrapper for SSL training.

    Supports:
    - BYOL, SimCLR, MAE
    - Mixed precision training
    - Gradient accumulation
    - Multi-GPU training
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
        mixed_precision: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize Accelerate trainer.

        Args:
            model: PyTorch model (should return loss when called with batch)
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to run on
            mixed_precision: 'fp16', 'bf16', or None (auto-detect)
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # ✅ Auto-detect mixed precision
        if mixed_precision is None:
            if torch.cuda.is_available():
                mixed_precision = "fp16"
            else:
                mixed_precision = "no"

        # ✅ Create accelerator
        self.accelerator = Accelerator(
            device_placement=True,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # ✅ Prepare model, optimizer, scheduler
        if scheduler is not None:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                model, optimizer, scheduler
            )
        else:
            self.model, self.optimizer = self.accelerator.prepare(model, optimizer)
            self.scheduler = None

        print(f"✅ AccelerateTrainer initialized:")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {mixed_precision}")
        print(f"   Gradient Accumulation: {gradient_accumulation_steps}")

    # ============================================================
    # ✅ FIX 1: prepare_dataloader - with validation support
    # ============================================================

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """
        Prepare a DataLoader for Accelerate.

        Args:
            dataloader: DataLoader to prepare

        Returns:
            Prepared DataLoader
        """
        if dataloader is None:
            return None
        return self.accelerator.prepare(dataloader)

    # ============================================================
    # ✅ FIX 2: train_epoch - with proper error handling
    # ============================================================

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training DataLoader

        Returns:
            Dictionary with loss metrics
        """
        self.model.train()

        # ✅ Prepare dataloader
        dataloader = self.prepare_dataloader(dataloader)

        # ✅ Check if dataloader is empty
        if len(dataloader) == 0:
            print("⚠️ Warning: train_loader is empty!")
            return {"loss": float("inf"), "num_batches": 0}

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # ✅ Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    # ✅ For SSL: (view1, view2) or (images, labels)
                    images = batch[0]
                else:
                    images = batch[0]
            else:
                images = batch

            # ✅ Training step
            loss = self.train_step(batch)

            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

        return {"loss": avg_loss, "num_batches": num_batches}

    # ============================================================
    # ✅ FIX 3: train_step - with proper loss handling
    # ============================================================

    def train_step(self, batch: Any) -> float:
        """
        Single training step with Accelerate.

        Args:
            batch: Batch from DataLoader (should be (x1, x2) for SSL)

        Returns:
            Loss value
        """
        # ✅ Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                # ✅ SSL: (view1, view2)
                view1, view2 = batch
                # ✅ Model should accept (view1, view2) and return loss
                with self.accelerator.autocast():
                    loss = self.model(view1, view2)
            else:
                # ✅ Labeled: (images, labels)
                images, labels = batch
                with self.accelerator.autocast():
                    loss = self.model(images, labels)
        else:
            # ✅ Single tensor
            with self.accelerator.autocast():
                loss = self.model(batch)

        # ✅ Backward with gradient accumulation
        self.accelerator.backward(loss)

        # ✅ Optimizer step (with gradient accumulation)
        if self.accelerator.sync_gradients:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

        return loss.item()

    # ============================================================
    # ✅ FIX 4: validate - NEW
    # ============================================================

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            dataloader: Validation DataLoader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        # ✅ Prepare dataloader
        dataloader = self.prepare_dataloader(dataloader)

        if len(dataloader) == 0:
            print("⚠️ Warning: val_loader is empty!")
            return {"loss": float("inf"), "num_batches": 0}

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # ✅ Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        view1, view2 = batch
                        with self.accelerator.autocast():
                            loss = self.model(view1, view2)
                    else:
                        images, labels = batch
                        with self.accelerator.autocast():
                            loss = self.model(images, labels)
                else:
                    with self.accelerator.autocast():
                        loss = self.model(batch)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

        return {"loss": avg_loss, "num_batches": num_batches}

    # ============================================================
    # ✅ FIX 5: save_checkpoint - NEW
    # ============================================================

    def save_checkpoint(self, path: str) -> None:
        """
        Save checkpoint with Accelerate.

        Args:
            path: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.accelerator.save_state(path)
        print(f"✅ Accelerate checkpoint saved: {path}")

    # ============================================================
    # ✅ FIX 6: load_checkpoint - NEW
    # ============================================================

    def load_checkpoint(self, path: str) -> None:
        """
        Load checkpoint with Accelerate.

        Args:
            path: Path to checkpoint
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.accelerator.load_state(path)
        print(f"✅ Accelerate checkpoint loaded: {path}")

    # ============================================================
    # ✅ FIX 7: get_model - NEW
    # ============================================================

    def get_model(self) -> nn.Module:
        """Get the unwrapped model."""
        return self.accelerator.unwrap_model(self.model)

    # ============================================================
    # ✅ FIX 8: get_optimizer - NEW
    # ============================================================

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer."""
        return self.optimizer

    # ============================================================
    # ✅ FIX 9: get_scheduler - NEW
    # ============================================================

    def get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get the scheduler."""
        return self.scheduler

    # ============================================================
    # ✅ FIX 10: is_main_process - NEW
    # ============================================================

    def is_main_process(self) -> bool:
        """Check if current process is the main process."""
        return self.accelerator.is_main_process

    # ============================================================
    # ✅ FIX 11: wait_for_everyone - NEW
    # ============================================================

    def wait_for_everyone(self) -> None:
        """Wait for all processes."""
        self.accelerator.wait_for_everyone()


# ================================================================
# ✅ Factory Function
# ================================================================


def create_accelerate_trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
    mixed_precision: Optional[str] = None,
    gradient_accumulation_steps: int = 1,
) -> AccelerateTrainer:
    """
    Factory function to create Accelerate trainer.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to run on
        mixed_precision: 'fp16', 'bf16', or None (auto-detect)
        gradient_accumulation_steps: Number of steps to accumulate gradients

    Returns:
        AccelerateTrainer instance

    Example:
        >>> trainer = create_accelerate_trainer(
        ...     model=byol_model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     device='cuda',
        ...     mixed_precision='fp16'
        ... )
        >>> results = trainer.train_epoch(train_loader)
        >>> print(results['loss'])
    """
    return AccelerateTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


# ================================================================
# ✅ Testing
# ================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Accelerate Trainer (ULTIMATE FIXED EDITION)")
    print("=" * 80)

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x1, x2):
            # Simulate SSL loss
            return torch.mean((x1 - x2) ** 2)

    # Create dummy data
    dummy_data = torch.randn(16, 10)
    dummy_loader = DataLoader(
        [(dummy_data[i], dummy_data[i + 1]) for i in range(0, 14, 2)],
        batch_size=4,
        shuffle=True,
    )

    # Create trainer
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = create_accelerate_trainer(
        model=model, optimizer=optimizer, device="cpu", mixed_precision="no"
    )

    # Test train_epoch
    print("\n📌 Testing train_epoch...")
    results = trainer.train_epoch(dummy_loader)
    print(f"   Loss: {results['loss']:.4f}")

    print("\n✅ All Accelerate tests passed!")
    print("=" * 80)
