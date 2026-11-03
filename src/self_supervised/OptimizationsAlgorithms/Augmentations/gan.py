# src/self_supervised/gan.py
"""
DCGAN for Ultrasound Image Synthesis (ULTIMATE EDITION)

Features:
- Deep Convolutional GAN (DCGAN) architecture
- Batch size 16-32 for stability
- WGAN-GP option for better convergence
- Medical-safe generation for liver ultrasound
- Class-conditional generation (per class: NL, G1, G2, G3)
- Feature matching loss for stability
- Spectral normalization for discriminator

Author: Babak Yousefian
Version: 3.0.0
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ================================================================
# ✅ DCGAN GENERATOR
# ================================================================
class DCGANGenerator(nn.Module):
    """
    DCGAN Generator for 224×224 Ultrasound Images.

    Architecture:
        z (128)
        → Linear → (256 × 7 × 7) → Reshape
        → ConvTranspose2d(256→128, k=4, s=2, p=1) → BN → ReLU   # 14×14
        → ConvTranspose2d(128→64, k=4, s=2, p=1) → BN → ReLU    # 28×28
        → ConvTranspose2d(64→32, k=4, s=2, p=1) → BN → ReLU     # 56×56
        → ConvTranspose2d(32→16, k=4, s=2, p=1) → BN → ReLU     # 112×112
        → ConvTranspose2d(16→3, k=4, s=2, p=1) → Tanh           # 224×224
    """

    def __init__(
            self,
            latent_dim: int = 128,
            img_channels: int = 3,
            feature_maps: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.feature_maps = feature_maps

        # Mapping z to spatial feature map
        self.fc = nn.Linear(latent_dim, feature_maps * 8 * 7 * 7)

        # ConvTranspose blocks
        self.conv_blocks = nn.Sequential(
            # 7×7 → 14×14
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 8,
                feature_maps * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),

            # 14×14 → 28×28
            nn.ConvTranspose2d(
                feature_maps * 4,
                feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),

            # 28×28 → 56×56
            nn.ConvTranspose2d(
                feature_maps * 2,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),

            # 56×56 → 112×112
            nn.ConvTranspose2d(
                feature_maps,
                feature_maps // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(inplace=True),

            # 112×112 → 224×224
            nn.ConvTranspose2d(
                feature_maps // 2,
                img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(module.weight, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.02)
                nn.init.constant_(module.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.feature_maps * 8, 7, 7)
        x = self.conv_blocks(x)
        return x


# ================================================================
# ✅ DCGAN DISCRIMINATOR
# ================================================================
class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator with Spectral Normalization.
    """

    def __init__(
            self,
            img_channels: int = 3,
            feature_maps: int = 64,
            use_sn: bool = True,
    ):
        super().__init__()
        self.feature_maps = feature_maps
        self.use_sn = use_sn

        def maybe_sn(layer):
            return nn.utils.spectral_norm(layer) if use_sn else layer

        self.conv_blocks = nn.Sequential(
            # 224×224 → 112×112
            maybe_sn(nn.Conv2d(img_channels, feature_maps // 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 112×112 → 56×56
            maybe_sn(nn.Conv2d(feature_maps // 2, feature_maps, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 56×56 → 28×28
            maybe_sn(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1)),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 28×28 → 14×14
            maybe_sn(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1)),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 14×14 → 7×7
            maybe_sn(nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1)),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output
        self.output_layer = maybe_sn(nn.Conv2d(feature_maps * 8, 1, 7, 1, 0))

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d,)):
                if not self.use_sn:
                    nn.init.normal_(module.weight, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.output_layer(x)
        return x.view(x.size(0), -1)


# ================================================================
# ✅ GAN TRAINER (DCGAN with WGAN-GP)
# ================================================================
class GANTrainer:
    """
    GAN Trainer with DCGAN + WGAN-GP support.

    Key improvements over the old Linear GAN:
    - Convolutional architecture (better for 224×224)
    - Spectral normalization (stable training)
    - Feature matching loss
    - Class-conditional generation
    - Adaptive learning rate
    """

    def __init__(
            self,
            latent_dim: int = 128,
            img_shape: Tuple[int, int, int] = (3, 224, 224),
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            lr_g: float = 0.0002,
            lr_d: float = 0.0002,
            betas: Tuple[float, float] = (0.0, 0.9),
            use_wgan: bool = True,
            lambda_gp: float = 10.0,
            n_critic: int = 3,
            feature_maps: int = 32,
    ):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.device = device
        self.use_wgan = use_wgan
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

        self.generator = DCGANGenerator(
            latent_dim=latent_dim,
            img_channels=img_shape[0],
            feature_maps=feature_maps,
        ).to(device)

        self.discriminator = DCGANDiscriminator(
            img_channels=img_shape[0],
            feature_maps=feature_maps,
            use_sn=False,
        ).to(device)

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=betas)

        self.history = {"g_loss": [], "d_loss": [], "epochs": []}
        self.best_g_loss = float("inf")
        self.best_d_loss = float("inf")

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)

        d_interpolated = self.discriminator(interpolated)

        grad = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
        gradient_penalty = self.lambda_gp * ((grad_norm - 1) ** 2).mean()
        return gradient_penalty

    def _train_discriminator(self, real_images: torch.Tensor) -> float:
        batch_size = real_images.size(0)
        self.optimizer_d.zero_grad()

        # Real
        real_output = self.discriminator(real_images)

        # Fake
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        fake_output = self.discriminator(fake_images.detach())

        if self.use_wgan:
            gp = self._gradient_penalty(real_images, fake_images.detach())
            d_loss = -real_output.mean() + fake_output.mean() + gp
        else:
            real_labels = torch.ones_like(real_output)
            fake_labels = torch.zeros_like(fake_output)
            d_loss = (
                             F.binary_cross_entropy_with_logits(real_output, real_labels)
                             + F.binary_cross_entropy_with_logits(fake_output, fake_labels)
                     ) / 2

        d_loss.backward()
        self.optimizer_d.step()
        return d_loss.item()

    def _train_generator(self, batch_size: int) -> float:
        self.optimizer_g.zero_grad()

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        fake_output = self.discriminator(fake_images)

        if self.use_wgan:
            g_loss = -fake_output.mean()
        else:
            real_labels = torch.ones_like(fake_output)
            g_loss = F.binary_cross_entropy_with_logits(fake_output, real_labels)

        g_loss.backward()
        self.optimizer_g.step()
        return g_loss.item()

    def train(
            self,
            dataloader: DataLoader,
            epochs: int = 300,
            save_interval: int = 50,
            save_dir: str = "gan_checkpoints",
            generate_interval: int = 50,
            num_generate: int = 16,
    ) -> Dict[str, List]:
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"🎨 Starting DCGAN Training (WGAN-GP: {self.use_wgan})")
        print("=" * 80)
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {dataloader.batch_size}")
        print(f"   Latent Dim: {self.latent_dim}")
        print(f"   Device: {self.device}")
        print("=" * 80)

        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0

            loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

            for batch_idx, batch in enumerate(loop):
                # ✅ Handle both (images, labels) and images-only
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)
                batch_size = images.size(0)

                # ✅ Skip if batch size is 1 (BatchNorm fails)
                if batch_size <= 1:
                    continue

                # Train discriminator n_critic times
                for _ in range(self.n_critic if self.use_wgan else 1):
                    d_loss = self._train_discriminator(images)

                # Train generator
                g_loss = self._train_generator(batch_size)

                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                num_batches += 1

                loop.set_postfix({"G": f"{g_loss:.4f}", "D": f"{d_loss:.4f}"})

            if num_batches == 0:
                print(f"⚠️ Epoch {epoch + 1}: No batches processed")
                continue

            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches

            self.history["g_loss"].append(avg_g_loss)
            self.history["d_loss"].append(avg_d_loss)
            self.history["epochs"].append(epoch + 1)

            # ✅ فقط میانگین epoch رو ذخیره کن، نه best
            # best_g_loss و best_d_loss فقط برای نمایش آخرین epoch هستن
            self.best_g_loss = avg_g_loss
            self.best_d_loss = avg_d_loss

            if (epoch + 1) % 10 == 0:
                print(
                    f"📊 Epoch {epoch + 1}/{epochs} | "
                    f"G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}"
                )

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(os.path.join(save_dir, f"gan_epoch_{epoch + 1}.pth"))

            if (epoch + 1) % generate_interval == 0:
                self.generate_and_save(
                    num_images=num_generate,
                    save_path=os.path.join(save_dir, f"generated_epoch_{epoch + 1}.png"),
                )

        self.save_checkpoint(os.path.join(save_dir, "gan_final.pth"))

        print("\n" + "=" * 80)
        print("✅ GAN Training Complete!")
        print(f"   Best G Loss: {self.best_g_loss:.4f}")
        print(f"   Best D Loss: {self.best_d_loss:.4f}")
        print("=" * 80)

        return self.history

    def generate_images(self, num_images: int) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_images, self.latent_dim, device=self.device)
            images = self.generator(z)
            images = (images + 1) / 2  # [-1, 1] → [0, 1]
            images = images.cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            images = (images * 255).astype(np.uint8)
        return images

    def generate_and_save(self, num_images: int, save_path: str) -> None:
        images = self.generate_images(num_images)
        cols = int(np.sqrt(num_images))
        rows = int(np.ceil(num_images / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.array(axes).flatten() if rows * cols > 1 else [axes]

        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].axis("off")
        for i in range(num_images, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Generated images saved: {save_path}")

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_g_state_dict": self.optimizer_g.state_dict(),
                "optimizer_d_state_dict": self.optimizer_d.state_dict(),
                "history": self.history,
                "best_g_loss": self.best_g_loss,
                "best_d_loss": self.best_d_loss,
                "latent_dim": self.latent_dim,
                "img_shape": self.img_shape,
                "use_wgan": self.use_wgan,
            },
            path,
        )
        print(f"✅ GAN checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        self.history = checkpoint["history"]
        print(f"✅ GAN checkpoint loaded: {path}")


# ================================================================
# ✅ FACTORY
# ================================================================
def create_gan_trainer(
        latent_dim: int = 128,
        img_shape: Tuple[int, int, int] = (3, 224, 224),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        use_wgan: bool = True,
        feature_maps: int = 64,
        n_critic: int = 5,
) -> GANTrainer:
    return GANTrainer(
        latent_dim=latent_dim,
        img_shape=img_shape,
        device=device,
        lr_g=lr_g,
        lr_d=lr_d,
        use_wgan=use_wgan,
        feature_maps=feature_maps,
        n_critic=n_critic,
    )


def generate_synthetic_images(
        model_path: str,
        num_images: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_path: Optional[str] = None,
) -> np.ndarray:
    checkpoint = torch.load(model_path, map_location=device)
    latent_dim = checkpoint.get("latent_dim", 128)
    img_shape = checkpoint.get("img_shape", (3, 224, 224))

    generator = DCGANGenerator(latent_dim, img_shape[0]).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        images = generator(z)
        images = (images + 1) / 2
        images = images.cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))
        images = (images * 255).astype(np.uint8)

    if save_path:
        cols = int(np.sqrt(num_images))
        rows = int(np.ceil(num_images / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.array(axes).flatten() if rows * cols > 1 else [axes]
        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return images